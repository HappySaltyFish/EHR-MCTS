"""Causal event summaries shared by PRM training and inference."""

from typing import Sequence

import numpy as np


def load_medtok_table(path: str, expected_rows: int | None = None) -> np.ndarray:
    table = np.load(path)
    if table.ndim != 2 or (expected_rows is not None and table.shape[0] != expected_rows):
        raise ValueError(f"MedTok table shape {table.shape} does not match expected token rows {expected_rows}")
    return table


STEP_TAG = "ки"


def solution_prefixes(solution_text: str, step_tag: str = STEP_TAG) -> list[str]:
    """Return text ending at each reward marker, excluding later actions."""
    marker = f" {step_tag}"
    parts = solution_text.split(marker)
    return [marker.join(parts[:index]) + marker for index in range(1, len(parts))]


def mean_pool_events(tokens: Sequence[int], medtok_table: np.ndarray) -> np.ndarray:
    if medtok_table.ndim != 2:
        raise ValueError("MedTok table must be a rank-two array")
    rows = []
    for token in tokens:
        index = int(token)
        if index < 0 or index >= len(medtok_table):
            raise IndexError(f"Event token {index} is outside the embedding table")
        rows.append(medtok_table[index])
    if not rows:
        raise ValueError("Cannot mean-pool an empty event sequence")
    return np.mean(rows, axis=0).astype(np.float32)


def pair_features(
    history_tokens: Sequence[int], prefix_tokens: Sequence[int],
    medtok_table: np.ndarray,
) -> dict[str, list[float]]:
    """Keep observed history and generated prefix as separate summaries."""
    return {
        "history_embeddings": mean_pool_events(history_tokens, medtok_table).tolist(),
        "prefix_embeddings": mean_pool_events(prefix_tokens, medtok_table).tolist(),
    }
