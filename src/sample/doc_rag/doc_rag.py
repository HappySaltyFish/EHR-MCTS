import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple


def normalize_icd10(code: str) -> str:
    """
    Use the same canonical ICD format as build_medlineplus_rag.py:
    uppercase and remove dots.
    """
    return str(code).strip().upper().replace(".", "")


@dataclass
class RagDoc:
    doc_id: int
    doc: str
    key_icd10: str
    rel_icd10: List[str]


class DocRetriever:
    """Load ICD labels and retrieve key-matched RAG documents."""

    def __init__(
        self,
        jsonl_path: str,
        rag_jsonl_path: Optional[str] = None,
        top_k: int = 3,
        debug_timing: Optional[bool] = None,
    ):
        self.docs: Dict[int, str] = {}
        self.labels: Dict[int, str] = {}

        self.token_to_icd10: Dict[int, str] = {}
        self.icd10_to_tokens: Dict[str, List[int]] = defaultdict(list)

        self.rag_docs: List[RagDoc] = []
        self.rag_by_key_icd10: Dict[str, List[RagDoc]] = defaultdict(list)

        self.top_k = int(top_k)

        if debug_timing is None:
            debug_timing = os.getenv("EHRSC_RAG_DEBUG", "0") == "1"
        self.debug_timing = bool(debug_timing)

        self._load_legacy_docs(jsonl_path)

        if rag_jsonl_path:
            self._load_rag_docs(rag_jsonl_path)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_legacy_docs(self, jsonl_path: str) -> None:
        if not jsonl_path:
            return

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Doc JSONL not found: {jsonl_path}")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to parse {jsonl_path} at line {line_no}: {e}"
                    )

                # Legacy schema: {"idx": ..., "label": ..., "doc": ...}
                if "idx" in item and "label" in item and "doc" in item:
                    try:
                        idx = int(item["idx"])
                    except Exception as e:
                        raise RuntimeError(
                            f"Invalid idx in {jsonl_path} at line {line_no}: {e}"
                        )

                    label = str(item["label"]).strip()
                    doc = str(item["doc"]).strip()

                    self.docs[idx] = doc
                    self.labels[idx] = label

                    code = normalize_icd10(label)
                    self.token_to_icd10[idx] = code
                    self.icd10_to_tokens[code].append(idx)

                # If someone accidentally passes the RAG jsonl here, do not crash.
                # RAG records are loaded only if _load_rag_docs is called.
                elif {"doc", "key_icd10", "rel_icd10"}.issubset(item.keys()):
                    continue

                else:
                    raise RuntimeError(
                        f"Unknown JSONL schema in {jsonl_path} at line {line_no}: "
                        f"keys={list(item.keys())}"
                    )

    def _load_rag_docs(self, rag_jsonl_path: str) -> None:
        if not os.path.exists(rag_jsonl_path):
            raise FileNotFoundError(f"RAG JSONL not found: {rag_jsonl_path}")

        with open(rag_jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to parse {rag_jsonl_path} at line {line_no}: {e}"
                    )

                if not {"doc", "key_icd10", "rel_icd10"}.issubset(item.keys()):
                    # Allow mixed files but ignore non-RAG records here.
                    continue

                doc_text = str(item["doc"]).strip()
                key_icd10 = normalize_icd10(item["key_icd10"])
                rel_raw = item["rel_icd10"]

                if not doc_text:
                    continue

                if not isinstance(rel_raw, list):
                    rel_raw = []

                rel_icd10 = []
                seen = set()

                for code in rel_raw:
                    code = normalize_icd10(code)
                    if code and code not in seen:
                        seen.add(code)
                        rel_icd10.append(code)

                rag_doc = RagDoc(
                    doc_id=len(self.rag_docs),
                    doc=doc_text,
                    key_icd10=key_icd10,
                    rel_icd10=rel_icd10,
                )

                self.rag_docs.append(rag_doc)
                self.rag_by_key_icd10[key_icd10].append(rag_doc)

        print(
            f"[RAG] Loaded {len(self.rag_docs)} RAG docs from {rag_jsonl_path}; "
            f"{len(self.rag_by_key_icd10)} key ICD10 codes."
        )

    # ------------------------------------------------------------------
    # Label and code helpers
    # ------------------------------------------------------------------

    def get_label(self, input_token: int) -> Optional[str]:
        return self.labels.get(int(input_token), None)

    def get_doc(self, input_token: int) -> Optional[str]:
        return self.docs.get(int(input_token), None)

    def get_icd10(self, input_token: int) -> Optional[str]:
        return self.token_to_icd10.get(int(input_token), None)

    def _tokens_to_icd10_set(self, tokens: Sequence[int]) -> Set[str]:
        codes = set()
        for tok in tokens:
            code = self.get_icd10(int(tok))
            if code:
                codes.add(code)
        return codes

    # ------------------------------------------------------------------
    # RAG scoring
    # ------------------------------------------------------------------

    def _score_rag_doc(
        self,
        rag_doc: RagDoc,
        history_codes: Set[str],
    ) -> Tuple[int, float, int]:
        """
        Main score:
        number of history ground-truth ICD10 codes covered by rel_icd10.

        Tie breakers:
        - specificity: prefer shorter rel_icd10 list if overlap is the same
        - doc_id: stable deterministic order
        """
        rel_set = set(rag_doc.rel_icd10)
        overlap_count = len(rel_set & history_codes)

        specificity = 1.0 / max(len(rel_set), 1)

        return (
            overlap_count,
            specificity,
            -rag_doc.doc_id,
        )

    def _debug_time(
        self,
        name: str,
        start_time: float,
        **kwargs,
    ) -> None:
        if not self.debug_timing:
            return

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        info = ", ".join(f"{k}={v}" for k, v in kwargs.items())

        if info:
            print(f"[RAG DEBUG] {name}: {elapsed_ms:.3f} ms | {info}")
        else:
            print(f"[RAG DEBUG] {name}: {elapsed_ms:.3f} ms")

    # ------------------------------------------------------------------
    # Public RAG API
    # ------------------------------------------------------------------

    def retrieve_by_key(
        self,
        key_token: int,
        history_tokens: Sequence[int],
        top_k: Optional[int] = None,
    ) -> List[str]:
        """
        Required behavior:

        Given:
        - one primary key token
        - tokens in the observed and previously generated prefix

        Retrieve documents that satisfy:
        - document.key_icd10 == ICD10(key_token)

        Rank by:
        - number of prefix ICD10 codes covered by document.rel_icd10

        Return:
        - top_k document strings
        """
        start = time.perf_counter()
        if not self.rag_docs:
            raise ValueError("RAG knowledge base is not loaded")

        top_k = self.top_k if top_k is None else int(top_k)

        key_token = int(key_token)
        key_icd10 = self.get_icd10(key_token)

        if not key_icd10:
            self._debug_time(
                "retrieve_by_key",
                start,
                key_token=key_token,
                key_icd10=None,
                returned=0,
            )
            return []

        history_codes = self._tokens_to_icd10_set(history_tokens)
        candidates = self.rag_by_key_icd10.get(key_icd10, [])

        scored = []
        for rag_doc in candidates:
            score = self._score_rag_doc(rag_doc, history_codes)
            overlap_count = score[0]

            if overlap_count > 0:
                scored.append((score, rag_doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [rag_doc.doc for _, rag_doc in scored[:top_k]]

        self._debug_time(
            "retrieve_by_key",
            start,
            key_token=key_token,
            key_icd10=key_icd10,
            history_codes=len(history_codes),
            candidates=len(candidates),
            returned=len(selected),
        )

        return selected

    def __len__(self) -> int:
        if self.rag_docs:
            return len(self.rag_docs)
        return len(self.docs)
