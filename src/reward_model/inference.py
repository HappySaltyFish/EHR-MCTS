from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.reward_model.prompts import ORM_SYSTEM_PROMPT, PRM_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT
from src.reward_model.review import Review, PROJECTOR_FILE, META_FILE
from src.reward_model.medtok_features import load_medtok_table, mean_pool_events, pair_features, solution_prefixes
from src.reward_model.rag_prompt import build_prm_prompt
from src.sample.formatters import EHRTextFormatter
import json

STEP_TAG = "ки"


@dataclass
class RewardModelBundle:
    model: torch.nn.Module
    tokenizer: Any
    device: torch.device
    dtype: torch.dtype
    max_token_len: int
    train_label: str
    use_rag: str
    use_medtok: bool
    medtok_emb_table: Optional[np.ndarray]
    step_tag: str = STEP_TAG
    medtok_prefix_mode: str = "2prefix"


def _resolve_torch_dtype(dtype: str) -> torch.dtype:
    dtype = str(dtype).lower()
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp16", "float16", "half"}:
        return torch.float16
    if dtype in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _build_system_prompt(train_label: str, use_rag: str) -> str:
    if use_rag == "yes":
        return RAG_SYSTEM_PROMPT
    if train_label == "orm_label":
        return ORM_SYSTEM_PROMPT
    return PRM_SYSTEM_PROMPT


def _maybe_resize_for_tokenizer(model, tokenizer) -> None:
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        return
    if input_embeddings.num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))


def _build_prompt_text(
    rm: RewardModelBundle,
    question_text: str,
    prm_processed_solution: str,
    related_docs: Optional[Sequence[str]] = None,
) -> str:
    system_prompt = _build_system_prompt(rm.train_label, rm.use_rag)
    return build_prm_prompt(
        rm.tokenizer, system_prompt, question_text, prm_processed_solution,
        rm.max_token_len,
        (related_docs or ()) if rm.use_rag == "yes" else (),
        step_tag=rm.step_tag,
    )

max_ids_len = -1

def _build_label_mask_and_values(
    raw_text: str,
    tokenizer,
    step_tag: str,
    max_token_len: int,
) -> Dict[str, Any]:
    global max_ids_len
    step_tag_id = tokenizer.encode(f" {step_tag}", add_special_tokens=False)[-1]

    enc = tokenizer(raw_text, add_special_tokens=True)
    ids = enc["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = enc["attention_mask"] + [0]

    raw_label = raw_text.replace(f" {step_tag}", " +")
    ref = tokenizer(raw_label, add_special_tokens=True)["input_ids"]
    ref.insert(0, tokenizer.pad_token_id)
    ref[0] = -100

    if len(ref) != len(ids):
        return {"valid": False, "reason": "token_alignment_mismatch"}

    step_positions = 0

    for j in range(len(ref)):
        if j == 0:
            continue

        if ids[j - 1] == step_tag_id:
            step_positions += 1
        else:
            ref[j] = -100

    max_ids_len = max(max_ids_len, len(ids))
    # print(f"[DEBUG.build_label_mask_and_values]: len(ids)={len(ids)}, current_max_len(ids)={max_ids_len}, max_token_len={max_token_len}")

    if len(ids) < max_token_len:
        pad = max_token_len - len(ids)
        ids = ids + [0] * pad
        attention_mask = attention_mask + [0] * pad
        ref = ref + [-100] * pad
    else:
        ids = ids[:max_token_len]
        attention_mask = attention_mask[:max_token_len]
        ref = ref[:max_token_len]

    if not any(x != -100 for x in ref):
        return {"valid": False, "reason": "no_supervised_positions_after_truncation"}

    return {
        "valid": True,
        "input_ids": ids,
        "attention_mask": attention_mask,
        "labels": ref,
        "num_steps_in_text": step_positions,
    }

def load_reward_model(
    checkpoint_path: str,
    *,
    device: str = "cuda",
    dtype: str = "bfloat16",
    max_token_len: int = 4096,
    train_label: str = "prm_target_label",
    use_rag: str = "no",
    medtok_emb_npy: Optional[str] = None,
    expected_vocab_size: Optional[int] = None,
    attn_type: Optional[str] = "flash_attention_2",
    medtok_prefix_mode: str = "2prefix",
) -> RewardModelBundle:
    if not medtok_emb_npy:
        raise ValueError("medtok_emb_npy is required because MedTok is always enabled.")
    if medtok_prefix_mode not in {"1prefix", "2prefix"}:
        raise ValueError("medtok_prefix_mode must be 1prefix or 2prefix")

    meta_path = os.path.join(checkpoint_path, META_FILE)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {META_FILE} under checkpoint: {checkpoint_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    checkpoint_prefix_count = int(meta.get("medtok_prefix_version", 1))
    requested_prefix_count = int(medtok_prefix_mode[0])
    if checkpoint_prefix_count != requested_prefix_count:
        raise ValueError(
            f"Checkpoint has {checkpoint_prefix_count} MedTok prefix(es); "
            f"requested {medtok_prefix_mode}"
        )

    torch_dtype = _resolve_torch_dtype(dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    step_token = f" {STEP_TAG}"
    if step_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [step_token]})

    base_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation=attn_type
    )
    _maybe_resize_for_tokenizer(base_model, tokenizer)

    projector_path = os.path.join(checkpoint_path, PROJECTOR_FILE)
    if not os.path.exists(projector_path):
        raise FileNotFoundError(f"Missing {PROJECTOR_FILE} under checkpoint: {checkpoint_path}")

    if not os.path.exists(medtok_emb_npy):
        raise FileNotFoundError(f"medtok_emb_npy not found: {medtok_emb_npy}")

    medtok_emb_table = load_medtok_table(medtok_emb_npy, expected_vocab_size)

    hidden_size = int(meta.get("model_hidden_size", getattr(base_model.config, "hidden_size")))
    concat_length = int(requested_prefix_count + max_token_len)
    embedding_dim = int(meta.get("embedding_dim", medtok_emb_table.shape[1]))
    filler_token_id = int(meta.get("filler_token_id", tokenizer.eos_token_id or tokenizer.pad_token_id))

    if embedding_dim != medtok_emb_table.shape[1]:
        raise ValueError(
            f"Embedding dim mismatch: checkpoint={embedding_dim}, npy={medtok_emb_table.shape[1]}"
        )

    model = Review(
        model=base_model,
        embedding_dim=embedding_dim,
        model_hidden_size=hidden_size,
        filler_token_id=filler_token_id,
        concat_length=concat_length,
        prefix_count=requested_prefix_count,
    )
    model.load_projector(checkpoint_path)

    model.to(device)
    model.eval()

    if hasattr(model, "config"):
        model.config.use_cache = True

    return RewardModelBundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=torch_dtype,
        max_token_len=max_token_len,
        train_label=train_label,
        use_rag=use_rag,
        use_medtok=True,
        medtok_emb_table=medtok_emb_table,
        medtok_prefix_mode=medtok_prefix_mode,
    )


def score_solution_from_text(
    rm: RewardModelBundle,
    *,
    question_text: str,
    prm_processed_solution: str,
    related_docs_by_step: Optional[Sequence[Sequence[str]]] = None,
    history_traj: Optional[Sequence[int]] = None,
    solution_tokens: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    prefixes = solution_prefixes(prm_processed_solution, rm.step_tag)
    if solution_tokens is None or len(prefixes) != len(solution_tokens):
        return {"valid": False, "reason": "marker_token_count_mismatch"}
    if rm.use_rag == "yes" and (
        related_docs_by_step is None or len(related_docs_by_step) != len(prefixes)
    ):
        return {"valid": False, "reason": "missing_per_prefix_rag_documents"}
    if history_traj is None:
        return {"valid": False, "reason": "missing_history_tokens"}

    history_tokens = [int(event[0]) for event in history_traj]
    plus_id = rm.tokenizer.encode(" +", add_special_tokens=False)[-1]
    minus_id = rm.tokenizer.encode(" -", add_special_tokens=False)[-1]
    plus_logits, minus_logits = [], []

    for step, prefix_text in enumerate(prefixes):
        raw_text = _build_prompt_text(
            rm, question_text=question_text,
            prm_processed_solution=prefix_text,
            related_docs=related_docs_by_step[step] if rm.use_rag == "yes" else None,
        )
        built = _build_label_mask_and_values(
            raw_text=raw_text, tokenizer=rm.tokenizer, step_tag=rm.step_tag,
            max_token_len=rm.max_token_len,
        )
        if not built["valid"]:
            return {**built, "step": step}
        marker_positions = [index for index, label in enumerate(built["labels"])
                            if label != -100]
        if len(marker_positions) != step + 1:
            return {"valid": False, "reason": "reward_marker_truncated", "step": step}

        model_inputs: Dict[str, Any] = {
            "input_ids": torch.tensor([built["input_ids"]], dtype=torch.long, device=rm.device),
            "attention_mask": torch.tensor([built["attention_mask"]], dtype=torch.long, device=rm.device),
        }
        if rm.medtok_prefix_mode == "1prefix":
            features = {"medtok_embeddings": mean_pool_events(
                solution_tokens[:step + 1], rm.medtok_emb_table).tolist()}
        else:
            features = pair_features(
                history_tokens, solution_tokens[:step + 1], rm.medtok_emb_table)
        for name, value in features.items():
            model_inputs[name] = torch.tensor([value], dtype=torch.float32, device=rm.device)

        with torch.no_grad():
            logits = rm.model(**model_inputs)["logits"]
        prefix_count = int(rm.medtok_prefix_mode[0])
        if logits.size(1) != len(built["labels"]) + prefix_count:
            return {"valid": False, "reason": "sequence_length_mismatch", "step": step}
        # The prefix vectors shift the reward-marker logit by their count.
        marker_logit = logits[0, marker_positions[-1] + prefix_count - 1]
        plus_logits.append(marker_logit[plus_id])
        minus_logits.append(marker_logit[minus_id])

    if not plus_logits:
        return {"valid": False, "reason": "no_supervised_positions"}
    pred_plus = torch.stack(plus_logits)
    pred_minus = torch.stack(minus_logits)
    pair_logits = torch.stack((pred_plus, pred_minus), dim=1)
    plus_probs = F.softmax(pair_logits, dim=1)[:, 0]
    pred_binary = (pred_plus > pred_minus).long()
    result: Dict[str, Any] = {
        "valid": True,
        "num_steps": len(prefixes),
        "plus_logits": pred_plus.detach().cpu().tolist(),
        "minus_logits": pred_minus.detach().cpu().tolist(),
        "plus_probs": plus_probs.detach().cpu().tolist(),
        "pred_binary": pred_binary.detach().cpu().tolist(),
        "margins": (pred_plus - pred_minus).detach().cpu().tolist(),
    }
    return result


def score_solution_from_trajectory(
    rm: RewardModelBundle,
    *,
    history_traj: Sequence[Tuple[int, float]],
    pred_traj: Sequence[Tuple[int, float]],
    formatter: Optional[EHRTextFormatter] = None,
    labels_jsonl: Optional[str] = None,
    rag_jsonl: Optional[str] = None,
) -> Dict[str, Any]:
    if formatter is None:
        if not labels_jsonl:
            raise ValueError("labels_jsonl is required when formatter is omitted")
        if rm.use_rag == "yes" and not rag_jsonl:
            raise ValueError("rag_jsonl is required when RAG is enabled")
        formatter = EHRTextFormatter(labels_jsonl=labels_jsonl, rag_jsonl=rag_jsonl)

    question_text = formatter.create_question_text(list(history_traj))
    prm_processed_solution = formatter.format_prediction_steps(list(pred_traj), list(history_traj))
    related_docs_by_step = (
        [formatter.key_related_docs(list(history_traj), list(pred_traj[:step + 1]))
         for step in range(len(pred_traj))]
        if rm.use_rag == "yes" else None
    )
    solution_tokens = [int(tok) for tok, _ in pred_traj]

    return score_solution_from_text(
        rm,
        question_text=question_text,
        prm_processed_solution=prm_processed_solution,
        related_docs_by_step=related_docs_by_step,
        history_traj=history_traj,
        solution_tokens=solution_tokens,
    )


""" Score Trajectory Easiest Example:
from src.reward_model.inference import load_reward_model, score_solution_from_trajectory

rm = load_reward_model(
    checkpoint_path="/path/to/rm_checkpoint",
    device="cuda:0",
    dtype="bfloat16",
    max_token_len=4096,
    train_label="prm_target_label",
    use_rag="yes",
    use_medtok=True,
    medtok_emb_npy="data/ukb_data/medtok_embeddings.npy",
)

out = score_solution_from_trajectory(
    rm,
    history_traj=[(12, 18000.0), (45, 18120.0)],
    pred_traj=[(77, 18200.0), (91, 18350.0)],
    label_path="data/ukb_data/labels.csv",
)
"""
