from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Sequence
import json
import os
import pandas as pd
import yaml

from .doc_rag.doc_rag import DocRetriever

Trajectory = List[Tuple[int, float]]


def load_delphi_labels(label_path: str) -> List[str]:
    # try single-column tsv/csv first
    try:
        df = pd.read_csv(label_path, sep="\t", header=None)
        if df.shape[1] == 1:
            return df.iloc[:, 0].astype(str).tolist()
    except Exception:
        pass

    df = pd.read_csv(label_path)
    if {"index", "name"}.issubset(df.columns):
        max_idx = int(df["index"].max())
        labels = [""] * (max_idx + 1)
        for _, row in df.iterrows():
            labels[int(row["index"])] = str(row["name"])
        return labels

    return df.iloc[:, 0].astype(str).tolist()


def split_trajectory(traj: Optional[Trajectory], to_years: bool=True) -> tuple[List[int], List[float]]:
    if not traj:
        return [], []
    tokens, ages = zip(*traj)
    if to_years:
        return [int(t) for t in tokens], [float(a/365.25) for a in ages]
    else:
        return [int(t) for t in tokens], [float(a) for a in ages]

    
def build_user_content(doc_text, question_text, sol_text):
    # return f"{doc_text}History: {question_text}\n\nPrediction: {sol_text}"
    return f"{doc_text}\nHistory: {question_text}\n\nPrediction: {sol_text}"


def normalize_icd9(code: Any) -> str:
    return str(code).strip().upper().replace(".", "")


def load_mimic_label_table(label_path: str) -> Dict[int, Dict[str, Any]]:
    if not label_path:
        raise ValueError("label_path is required for MIMIC formatting.")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"MIMIC label file not found: {label_path}")

    with open(label_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict) or not raw:
        raise ValueError("MIMIC labels must be a non-empty mapping of zero-based type_event IDs")

    table: Dict[int, Dict[str, Any]] = {}
    for k, v in raw.items():
        idx = int(k)
        if not isinstance(v, dict) or v.get("type_event") != idx:
            raise ValueError(f"MIMIC label {k} must have a matching type_event")
        table[idx + 1] = v
    if set(table) != set(range(1, len(raw) + 1)):
        raise ValueError("MIMIC label IDs must be contiguous and start at zero")
    return table


def infer_segments_by_gap(
    traj: Trajectory,
    gap_days: float,
) -> List[List[Tuple[int, float]]]:
    """Split a trajectory into segments whenever adjacent events are separated by a large gap."""
    if not traj:
        return []

    segments: List[List[Tuple[int, float]]] = []
    cur_segment: List[Tuple[int, float]] = [traj[0]]

    for prev, cur in zip(traj[:-1], traj[1:]):
        prev_age = float(prev[1])
        cur_age = float(cur[1])
        if (cur_age - prev_age) > float(gap_days):
            segments.append(cur_segment)
            cur_segment = [cur]
        else:
            cur_segment.append(cur)

    if cur_segment:
        segments.append(cur_segment)

    return segments


def format_gap_days(gap_days: float) -> str:
    gap_days = float(gap_days)
    if gap_days < 1.0:
        gap_hours = gap_days * 24.0
        return f"{gap_hours:.1f} hours"
    return f"{gap_days:.1f} days"


class TextFormatter(ABC):

    @abstractmethod
    def format_history(self, history_traj: Trajectory) -> str:
        """Format the history trajectory into human-readable text."""
        pass

    @abstractmethod
    def create_question_text(self, history_traj: Trajectory) -> str:
        """Create the question text from history trajectory."""
        pass

    @abstractmethod
    def format_prediction_steps(self, pred_traj: Trajectory, history_traj: Trajectory) -> str:
        """Format the prediction trajectory into step-by-step text."""
        pass

    @abstractmethod
    def build_rm_user_content(
        self,
        history_traj: Trajectory,
        pred_traj: Trajectory,
        use_rag: bool = False,
    ) -> str:
        """Build the complete user content for reward model input."""
        pass

    @abstractmethod
    def ICD_code(self, token: int) -> str:
        """Get the ICD code or label for a token."""
        pass


@dataclass
class EHRTextFormatter(TextFormatter):
    labels: List[str]
    doc_retriever: Optional[DocRetriever] = None

    def __init__(self, labels_jsonl = None, rag_jsonl = None, train_config = None):
        self.doc_retriever=DocRetriever(jsonl_path=labels_jsonl, rag_jsonl_path=rag_jsonl)
        self.data_source = "ukb_data"
        if train_config is not None:
            if isinstance(train_config, str):
                with open(train_config, "r") as fp:
                    config = yaml.safe_load(fp)
                self.ignore_tokens = config.get("ignore_tokens", [])
            elif isinstance(train_config, Dict):
                self.ignore_tokens = train_config.get("ignore_tokens", [])
            else:
                raise ValueError(f"Unknown train_config type for EHRTextFormatter: {type(train_config)}")
        else:
            self.ignore_tokens = []

    def _format_baseline_label(self, token: int) -> str:
        if token==1: return "drink alcohol daily", 3
        elif token==2: return None, 999
        elif token==3: return "drink alcohol one to three times a month", 3
        elif token==4: return "never drink alcohol", 3
        elif token==5: return None, 999
        elif token==6: return "drink alcohol on special occasions only", 3
        elif token==7: return "drink alcohol once or twice a week", 3
        elif token==8: return "drink alcohol three or four times a week", 3
        elif token==9: return "low bmi", 2
        elif token==10: return "middle bmi", 2
        elif token==11: return "high bmi", 2
        elif token==12: return "female", 1
        elif token==13: return "male", 1
        elif token==14: return None, 999
        elif token==15: return "smoke all days", 4
        elif token==16: return "never smoke", 4
        elif token==17: return "smoke only occasionally", 4
        elif token==18: return None, 999
        else:
            raise ValueError(f"not baseline token: {token}")

    def _is_baseline_event(self, token: int) -> bool:
        # if self.ignore_tokens:
        #     return token in self.ignore_tokens
        # else:
        #     return None
        return token in self.ignore_tokens
        
    def _form_single_desc(self, token:int, age:float) -> str:
        doc = self.doc_retriever.get_doc(token)
        icd10 = self.doc_retriever.get_icd10(token)
        # return f"{doc} ({icd10}, age={age:.1f} years)"
        if icd10 is not None and icd10 != "INVALID":
            return f"{doc} ({icd10}) at age {age:.1f} years"
        else:
            return f"{doc} at age {age:.1f} years"

    def ICD_code(self, token: int) -> str:
        token = int(token)
        label = self.doc_retriever.get_label(token)
        if label is None:
            return f"UNKNOWN_TOKEN_{token}"
        else:
            return label

    def format_history(self, history_traj: Trajectory) -> str:
        tokens, age_years = split_trajectory(history_traj)
        if len(tokens) == 0:
            return "No events recorded."

        baseline_items = []
        medical_events = []

        for i, (token, age_year) in enumerate(zip(tokens, age_years)):
            icd_code = self.ICD_code(token)

            if self._is_baseline_event(token):
                baseline_name, priority = self._format_baseline_label(token)
                if baseline_name is not None:
                    baseline_items.append((priority, baseline_name))
                continue

            if i == 0:
                medical_events.append(f"At age {age_year:.1f} years: {icd_code}")
            else:
                # gap_days = age - ages[i - 1]
                medical_events.append(self._form_single_desc(token, age_year))

        baseline_items = [
            baseline_name
            for _, baseline_name in sorted(baseline_items, key=lambda x: x[0])
        ]

        sections = []

        if baseline_items:
            sections.append("Patient baseline:\n" + "; ".join(baseline_items))

        if medical_events:
            sections.append("Medical history:\n" + "; ".join(medical_events))
        elif not baseline_items:
            sections.append("Medical history:\nNo events recorded.")

        return "\n\n".join(sections)

    def create_question_text(self, history_traj: Trajectory) -> str:
        history = self.format_history(history_traj)
        return (
            f"Given the following patient information:\n{history}\n\n"
            f"What are the most likely future medical events for this patient?"
        )

    def key_related_docs(self, history_traj: Trajectory, action: Trajectory):
        if self.doc_retriever is None:
            raise ValueError("RAG retriever is not configured")
        _history_tokens, _ = split_trajectory(history_traj)
        history_tokens = []
        for t in _history_tokens:
            if t not in self.ignore_tokens: history_tokens.append(t)
        
        docs = []
        action_tokens, _ = split_trajectory(action)
        for action_token in action_tokens:
            if action_token in self.ignore_tokens:
                continue
            docs += self.doc_retriever.retrieve_by_key(action_token, history_tokens, top_k=1)
            history_tokens.append(action_token)
        return docs


    def format_prediction_steps(self, pred_traj: Trajectory, history_traj: Trajectory, step_marker: str="ки") -> str:
        pred_tokens, pred_ages = split_trajectory(pred_traj)
        _, history_ages = split_trajectory(history_traj)

        if len(pred_tokens) == 0:
            return "No solution steps."

        # base_age = history_ages[-1] if len(history_ages) > 0 else 0.0
        steps = []

        for i, (token, age) in enumerate(zip(pred_tokens, pred_ages)):
            # doc = self.doc_retriever.get_doc(token)
            step_text = (
                f"Predict {i+1}: {self._form_single_desc(token, age)}"
            )

            steps.append(step_text + f" {step_marker}")

        return " ".join(steps)

    def build_rm_user_content(
        self,
        history_traj: Trajectory,
        pred_traj: Trajectory,
        use_rag: bool = False,
    ) -> str:
        question_text = self.create_question_text(history_traj)
        solution_text = self.format_prediction_steps(pred_traj, history_traj)

        doc_block = ""
        if use_rag:
            # docs = self.related_docs(history_traj)
            docs = self.key_related_docs(history_traj, pred_traj)
            doc_block = "".join(
                f"Document {i + 1}: {doc}\n\n" for i, doc in enumerate(docs)
            )

        # return f"{doc_block}Question: {question_text}\n\nExplanation: {solution_text}"
        return build_user_content(doc_block, question_text, solution_text)


@dataclass
class MimicTextFormatter(TextFormatter):
    label_table: Dict[int, Dict[str, Any]]
    rag_docs: List[Dict[str, Any]]
    rag_by_code: Dict[str, List[Dict[str, Any]]]

    def __init__(
        self,
        label_path: Optional[str] = None,
        rag_jsonl: Optional[str] = None,
        train_config: Optional[Any] = None,
        same_visit_gap_days: float = 1.0,
        data_source: str = "mimic_3",
    ):
        self.data_source = data_source
        self.same_visit_gap_days = float(same_visit_gap_days) / 86400.0
        self.doc_retriever = None
        self.ignore_tokens = []

        if train_config is not None:
            if isinstance(train_config, str):
                with open(train_config, "r", encoding="utf-8") as fp:
                    config = yaml.safe_load(fp) or {}
            elif isinstance(train_config, Dict):
                config = dict(train_config)
            else:
                raise ValueError(f"Unknown train_config type for MimicTextFormatter: {type(train_config)}")

            self.ignore_tokens = [int(t) for t in (config.get("ignore_tokens", []) or [])]
            self.data_source = str(config.get("dataset", self.data_source))

            if label_path is None:
                label_path = config.get("label_path", None)

        if label_path is None:
            raise ValueError(
                "MimicTextFormatter requires label_path (labels.json) or a train_config that provides it."
            )

        self.label_path = label_path
        self.label_table = load_mimic_label_table(label_path)
        self.rag_docs = []
        self.rag_by_code = {}

        if rag_jsonl:
            self._load_rag_docs(rag_jsonl)

    def _load_rag_docs(self, rag_jsonl: str) -> None:
        if not os.path.exists(rag_jsonl):
            raise FileNotFoundError(f"RAG JSONL not found: {rag_jsonl}")

        with open(rag_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if not {"doc", "key_icd9", "rel_icd9"}.issubset(item.keys()):
                    continue
                doc = str(item["doc"]).strip()
                if not doc:
                    continue

                key_icd9 = normalize_icd9(item["key_icd9"])
                rel_icd9 = [normalize_icd9(x) for x in (item.get("rel_icd9") or [])]
                rel_icd9 = [x for x in dict.fromkeys(rel_icd9) if x]

                rec = {
                    "doc": doc,
                    "key_icd9": key_icd9,
                    "rel_icd9": rel_icd9,
                    "doc_id": len(self.rag_docs),
                }
                self.rag_docs.append(rec)
                self.rag_by_code.setdefault(key_icd9, []).append(rec)

    def _label_info(self, token: int) -> Dict[str, Any]:
        return self.label_table.get(int(token), {})

    def _token_name(self, token: int) -> str:
        info = self._label_info(token)
        if not info:
            return f"UNKNOWN_TOKEN_{int(token)}"
        code = normalize_icd9(info.get("code_norm") or info.get("representative_icd9_code") or "")
        short_title = str(info.get("short_title") or "").strip()
        long_title = str(info.get("long_title") or short_title).strip()
        title = short_title or long_title or f"Token {int(token)}"
        return f"{title} (ICD-9 {code})" if code else title

    def ICD_code(self, token: int) -> str:
        info = self._label_info(token)
        if not info:
            return f"UNKNOWN_TOKEN_{int(token)}"
        code = normalize_icd9(info.get("code_norm") or info.get("representative_icd9_code") or "")
        if code:
            return code
        return f"UNKNOWN_TOKEN_{int(token)}"

    def _normalize_traj(self, traj: Trajectory) -> List[Tuple[int, float]]:
        tokens, ages_days = split_trajectory(traj, to_years=False)
        return [
            (int(t), float(a))
            for t, a in zip(tokens, ages_days)
            if int(t) not in self.ignore_tokens
        ]

    def _format_event(self, token: int, age_days: float) -> str:
        return f"{self._token_name(token)} at day {age_days:.1f} from first visit"

    def _segment_text(self, segment: List[Tuple[int, float]], segment_idx: int) -> str:
        if not segment:
            return ""
        start_age_days = float(segment[0][1])
        lines = [f"Admission {segment_idx + 1} (starting at day {start_age_days:.1f}):"]
        for token, age_days in segment:
            lines.append(f"- {self._format_event(token, age_days)}")
        return "\n".join(lines)

    def format_history(self, history_traj: Trajectory) -> str:
        traj = self._normalize_traj(history_traj)
        if len(traj) == 0:
            return "No events recorded."

        segments = infer_segments_by_gap(traj, self.same_visit_gap_days)
        if len(segments) == 0:
            return "Medical history:\nNo events recorded."

        return "Medical history:\n" + "\n\n".join(
            self._segment_text(segment, idx) for idx, segment in enumerate(segments)
        )

    def create_question_text(self, history_traj: Trajectory) -> str:
        history = self.format_history(history_traj)
        return (
            "Given the following patient information:\n"
            f"{history}\n\n"
            "What are the most likely future medical events for this patient?"
        )

    def key_related_docs(self, history_traj: Trajectory, action: Trajectory) -> List[str]:
        if not self.rag_docs:
            raise ValueError("RAG knowledge base is not loaded")

        history_codes = set()
        for token, _ in self._normalize_traj(history_traj):
            info = self._label_info(token)
            code = normalize_icd9(info.get("code_norm") or info.get("representative_icd9_code") or "")
            if code:
                history_codes.add(code)

        docs = []
        for token, _ in self._normalize_traj(action):
            code = normalize_icd9(self.ICD_code(token))
            if not code:
                continue
            scored = []
            for doc in self.rag_by_code.get(code, []):
                rel_set = set(doc.get("rel_icd9", []))
                overlap = len(rel_set.intersection(history_codes))
                if overlap > 0:
                    specificity = 1.0 / max(len(rel_set), 1)
                    scored.append((overlap, specificity, -int(doc["doc_id"]), doc["doc"]))
            scored.sort(reverse=True)
            if scored:
                docs.append(scored[0][-1])
            history_codes.add(code)
        return docs

    def format_prediction_steps(self, pred_traj: Trajectory, history_traj: Trajectory) -> str:
        pred = self._normalize_traj(pred_traj)
        history = self._normalize_traj(history_traj)

        if len(pred) == 0:
            return "No solution steps."

        combined = history + pred
        segments = infer_segments_by_gap(combined, self.same_visit_gap_days)
        history_len = len(history)

        pred_segment_ids: List[int] = []
        cursor = 0
        for seg_idx, segment in enumerate(segments):
            for _ in segment:
                if cursor >= history_len:
                    pred_segment_ids.append(seg_idx)
                cursor += 1

        steps = []
        for i, (token, age) in enumerate(pred):
            seg_id = pred_segment_ids[i] if i < len(pred_segment_ids) else 0
            step_text = f"Predict {i + 1} [Admission {seg_id + 1}]: {self._format_event(token, age)}"
            steps.append(step_text + " ки")

        return " ".join(steps)

    def build_rm_user_content(
        self,
        history_traj: Trajectory,
        pred_traj: Trajectory,
        use_rag: bool = False,
    ) -> str:
        question_text = self.create_question_text(history_traj)
        solution_text = self.format_prediction_steps(pred_traj, history_traj)

        doc_block = ""
        if use_rag:
            docs = self.key_related_docs(history_traj, pred_traj)
            doc_block = "".join(
                f"Document {i + 1}: {doc}\n\n" for i, doc in enumerate(docs)
            )

        return build_user_content(doc_block, question_text, solution_text)


