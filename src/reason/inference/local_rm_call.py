"""Local medical PRM scoring for EHR search."""

from dataclasses import dataclass
from typing import Dict, Optional
import torch
from reason.inference.rm_call import RewardModelBaseConfig, RewardModelCallingFunction
from reward_model.inference import load_reward_model, score_solution_from_trajectory
from sample.formatters import EHRTextFormatter, MimicTextFormatter

@dataclass
class LocalMedRewardModelConfig(RewardModelBaseConfig):
    model_id: str
    label_path: Optional[str] = None
    label_jsonl: Optional[str] = None
    rag_jsonl: Optional[str] = None
    use_rag: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype: Optional[torch.dtype] = torch.float16

    medtok_emb_npy: Optional[str] = None
    max_token_len: int = None
    train_label: str = None

    # TextFormatter selection
    formatter_type: str = "ukb"  # "ukb" for UKB, "mimic" for MIMIC-3
    medtok_prefix_mode: str = "2prefix"


def _torch_dtype_to_str(dtype: Optional[torch.dtype]) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    return "float32"


class LocalMedRMCaller(RewardModelCallingFunction):
    def __init__(
        self,
        config: LocalMedRewardModelConfig,
        train_config: Dict
    ):
        super().__init__(config)

        if not config.label_path:
            raise ValueError("LocalMedRMCaller requires config.label_path")
        if not config.medtok_emb_npy:
            raise ValueError("LocalMedRMCaller requires config.medtok_emb_npy")

        self.model_id = config.model_id
        self.device = config.device
        self.torch_dtype = config.torch_dtype

        self.label_path = config.label_path
        self.label_jsonl = config.label_jsonl
        self.rag_jsonl = config.rag_jsonl
        self.use_rag = bool(config.use_rag)
        if self.use_rag and not self.rag_jsonl:
            raise ValueError("RAG is enabled but rag_jsonl is not configured")
        self.train_label = config.train_label
        self.max_token_len = int(config.max_token_len)

        # Instantiate the appropriate formatter based on formatter_type
        # TODO: actually only need one label_jsonl or label_path
        formatter_type = config.formatter_type.lower()
        if formatter_type == "ukb":
            self.formatter = EHRTextFormatter(
                labels_jsonl=self.label_jsonl,
                rag_jsonl=self.rag_jsonl,
                train_config=train_config
            )
            print(f"Using EHRTextFormatter for UKB/EHR task")
        elif formatter_type == "mimic":
            self.formatter = MimicTextFormatter(
                label_path=self.label_path,
                rag_jsonl=self.rag_jsonl,
                train_config=train_config
            )
            print(f"Using MimicTextFormatter for MIMIC-3 task")
        else:
            raise ValueError(
                f"Unknown formatter_type: {config.formatter_type}. "
                f"Supported types: 'ukb', 'mimic'"
            )

        if self.use_rag:
            rag_docs = (self.formatter.doc_retriever.rag_docs
                        if formatter_type == "ukb" else self.formatter.rag_docs)
            if not rag_docs:
                raise ValueError("RAG is enabled but the knowledge base is empty")

        print(f"Loading local medical reward model via unified inference: {self.model_id}")

        self.rm_bundle = load_reward_model(
            checkpoint_path=self.model_id,
            device=self.device,
            dtype=_torch_dtype_to_str(self.torch_dtype),
            max_token_len=self.max_token_len,
            train_label=self.train_label,
            use_rag="yes" if self.use_rag else "no",
            medtok_emb_npy=config.medtok_emb_npy,
            expected_vocab_size=train_config.get("vocab_size") if formatter_type == "mimic" else None,
            medtok_prefix_mode=config.medtok_prefix_mode,
        )

        print("!!! Local medical reward model loaded successfully (unified inference path) !!!")
    
    def __call__(
        self,
        question_answer_pairs,
        lm_step_tag: str,
    ):
        if isinstance(question_answer_pairs, tuple):
            question_answer_pairs = [question_answer_pairs]

        all_values = []

        for question, answer in question_answer_pairs:
            history_traj = [(int(t), float(a)) for t, a in question]
            pred_traj = [(int(t), float(a)) for t, a in answer]

            out = score_solution_from_trajectory(
                self.rm_bundle,
                history_traj=history_traj,
                pred_traj=pred_traj,
                formatter=self.formatter,
            )
            if not out.get("valid"):
                raise ValueError(f"PRM could not score candidate: {out.get('reason', 'unknown reason')}")

            values = [float(x) for x in out["plus_probs"]]

            if len(values) != len(pred_traj):
                raise ValueError(
                    f"PRM returned {len(values)} scores for {len(pred_traj)} predicted steps"
                )

            all_values.append(values)

        return all_values
