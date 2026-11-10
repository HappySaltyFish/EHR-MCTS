"""Run MCTS on supplied history trajectories without ground-truth futures."""

import argparse
import functools
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from utils import (
    ATOMIC_CANDIDATE_POOL_SIZE,
    KERNEL_INITIAL_COUNT,
    KERNEL_PREFIX_LAMBDA,
    LANGEVIN_SAMPLES_PER_CANDIDATE,
    LANGEVIN_STEPS,
    MACRO_PRIOR_EPS,
    MACRO_PRIOR_ETA,
    MACRO_PRIOR_TEMPERATURE,
    MIMIC_PREDICTION_WINDOW_DAYS,
    TIME_UNCERTAINTY_BETA,
    TYPE_SAMPLING_TEMPERATURE,
    TYPE_SAMPLING_TOP_K,
    UKB_PREDICTION_END_AGE_DAYS,
)
from reason.env.base_env import DelphiEnv
from reason.guided_search.tree import SearchTree
from reason.inference.local_gpt_call import LocalGPTCallerSmurf
from reason.inference.local_rm_call import LocalMedRMCaller, LocalMedRewardModelConfig


def load_histories(path):
    """Accept JSONL rows with only an observed history and optional id."""
    with open(path, encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) - {"id", "history", "input_tokens", "input_ages"}:
                raise ValueError(f"Invalid history row {line_no}: only observed inputs are accepted")
            if "history" in row:
                if "input_tokens" in row or "input_ages" in row:
                    raise ValueError(f"Use only one history representation at row {line_no}")
                history = row["history"]
            else:
                tokens, ages = row.get("input_tokens"), row.get("input_ages")
                if tokens is None or ages is None or len(tokens) != len(ages):
                    raise ValueError(f"Invalid input token/age lengths at row {line_no}")
                history = list(zip(tokens, ages))
            if not history or any(not isinstance(event, (list, tuple)) or len(event) != 2 for event in history):
                raise ValueError(f"Invalid history at row {line_no}")
            parsed = [(int(token), float(age)) for token, age in history]
            if any(not math.isfinite(age) for _, age in parsed) or any(
                parsed[index][1] < parsed[index - 1][1] for index in range(1, len(parsed))
            ):
                raise ValueError(f"History ages must be finite and nondecreasing at row {line_no}")
            yield row.get("id", line_no - 1), parsed


def load_language_model(config, device):
    config_type = config["config_type"]
    checkpoint = str(Path(config["out_dir"]) / "ckpt.pt")
    if config_type not in {"EHRCS", "Mimic"}:
        raise ValueError(f"Unsupported model type: {config_type}")
    return LocalGPTCallerSmurf(checkpoint_path=checkpoint, device=device, torch_dtype=torch.float32)


def prediction_events(trajectory, max_age, termination_token):
    """Return only clinical events inside the inclusive prediction window."""
    events = []
    for token, age in trajectory:
        if age > max_age or token == termination_token:
            break
        events.append([int(token), float(age)])
    return events


def run_one(history, lm_call, rm_call, args, config_type, observer=None):
    max_age = (UKB_PREDICTION_END_AGE_DAYS if args.task_type == "ukb"
               else history[-1][1] + MIMIC_PREDICTION_WINDOW_DAYS)
    if history[-1][1] >= max_age:
        raise ValueError("Observed history ends at or after the prediction endpoint")
    generation_config = {
        "temperature": TYPE_SAMPLING_TEMPERATURE,
        "top_k": TYPE_SAMPLING_TOP_K,
        "config_type": config_type,
        "no_repeat": False,
        "max_age": max_age,
        "age_is_normalized": False,
        "time_uncertainty_beta": TIME_UNCERTAINTY_BETA,
        "atomic_candidate_pool_size": ATOMIC_CANDIDATE_POOL_SIZE,
        "calibration_dataset": args.task_type,
        "n_samples": LANGEVIN_SAMPLES_PER_CANDIDATE,
        "n_langevin_steps": LANGEVIN_STEPS,
        "max_new_tokens": args.generation_step or (4 if args.task_type == "ukb" else 1),
    }
    termination_token = args.termination_token
    env = DelphiEnv(
        config={
            "max_actions": args.tree_width,
            "max_length": args.tree_depth,
            "generation_config": generation_config,
            "prior_eta": MACRO_PRIOR_ETA,
            "prior_eps": MACRO_PRIOR_EPS,
            "prior_temperature": MACRO_PRIOR_TEMPERATURE,
            "use_termination_tokens": False,
            "mcts_termination_tokens": [termination_token] if termination_token is not None else [],
            "mcts_max_age_days": max_age,
        },
        ehr_records=[{"question": history}],
        llm_gen_fn=lm_call,
        reset=False,
        observer=observer,
    )
    tree = SearchTree(
        cfg={
            "pb_c_base": 19652,
            "pb_c_init": 1.25,
            "init_critic_value": True,
            "use_kernel_visit": True,
            "kernel_prefix_lambda": KERNEL_PREFIX_LAMBDA,
            "kernel_init_count": KERNEL_INITIAL_COUNT,
        },
        observer=observer,
    )
    paths = tree.vanila_mcts(
        simulate_env=env,
        num_path=args.num_path,
        reward_model_fn=functools.partial(rm_call, lm_step_tag=lm_call.lm_step_tag),
    )
    return [
        {"trajectory": prediction_events(path["text"], max_age, termination_token),
         "path_score": float(path["path_score"])}
        for path in paths
    ]


def main():
    parser = argparse.ArgumentParser(description="UKB/MIMIC MCTS inference from observed histories")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rm_config", required=True)
    parser.add_argument("--rm_model_path", required=True)
    parser.add_argument("--rm_prefix_mode", choices=("1prefix", "2prefix"), default="2prefix")
    parser.add_argument("--rm_label_path")
    parser.add_argument("--rm_label_jsonl")
    parser.add_argument("--rm_rag_jsonl")
    parser.add_argument("--no_rag", action="store_true")
    parser.add_argument("--histories", required=True, help="JSONL of observed event/time pairs")
    parser.add_argument("--output", required=True, help="JSONL prediction output")
    parser.add_argument("--task_type", choices=["ukb", "mimic"], required=True)
    parser.add_argument("--num_path", type=int, default=4)
    parser.add_argument("--tree_depth", type=int, default=50)
    parser.add_argument("--tree_width", type=int, default=4)
    parser.add_argument("--generation_step", type=int)
    parser.add_argument("--termination_token", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.num_path < 1 or args.tree_depth < 1 or args.tree_width < 1:
        parser.error("num_path, tree_depth and tree_width must be positive")
    if not args.no_rag and not args.rm_rag_jsonl:
        parser.error("--rm_rag_jsonl is required when RAG is enabled")
    if args.task_type == "ukb" and not args.rm_label_jsonl:
        parser.error("--rm_label_jsonl is required for UKB token-to-ICD mapping")
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    with open(args.rm_config, encoding="utf-8") as stream:
        rm_config = yaml.safe_load(stream)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm_call = load_language_model(config, device)
    if args.termination_token is None:
        args.termination_token = config.get("visit_token" if args.task_type == "mimic" else "death_token", 1145 if args.task_type == "ukb" else None)
    rm_call = LocalMedRMCaller(
        config=LocalMedRewardModelConfig(
            step_tag="ки", model_id=args.rm_model_path, device=device,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            label_path=args.rm_label_path or str(Path("data") / config["dataset"] / ("labels.json" if args.task_type == "mimic" else "labels.csv")),
            label_jsonl=args.rm_label_jsonl, rag_jsonl=args.rm_rag_jsonl,
            use_rag=not args.no_rag, medtok_emb_npy=rm_config["medtok_emb_npy"],
            max_token_len=rm_config["max_token_len"],
            train_label=rm_config["train_label"], formatter_type=args.task_type,
            medtok_prefix_mode=args.rm_prefix_mode,
        ),
        train_config=config,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for sample_id, history in load_histories(args.histories):
            paths = run_one(history, lm_call, rm_call, args, config["config_type"])
            stream.write(json.dumps({"id": sample_id, "paths": paths}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
