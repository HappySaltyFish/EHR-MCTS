"""Candidate-level uncertainty-calibrated atomic prior, paper equation (2)."""

import math
import torch
from utils import normalize_time_uncertainty


def atomic_candidate_log_prior(
    type_logprobs: torch.Tensor,
    gap_samples: torch.Tensor,
    beta: float,
    calibration: dict,
    clip_max: float,
    eps: float,
    prefix_group_ids: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return log-softmax over sampled (type, gap) candidates and normalized u.

    type_logprobs has shape [batch, candidates]; gap_samples [batch,
    candidates, Langevin samples]. The calibration belongs to the training
    split and must be supplied by the caller.
    """
    if gap_samples.ndim != 3 or type_logprobs.shape != gap_samples.shape[:2]:
        raise ValueError("Expected type [B,M] and gap samples [B,M,S]")
    if gap_samples.shape[1] < 2 or gap_samples.shape[2] < 2:
        raise ValueError("At least two candidates and two gap samples are required")
    if not math.isfinite(beta) or beta < 0 or clip_max <= 0 or eps <= 0:
        raise ValueError("beta, clip_max, or eps is invalid")
    normalized = normalize_time_uncertainty(gap_samples, calibration, clip_max, eps)
    scores = type_logprobs.float() - float(beta) * normalized
    if prefix_group_ids is None:
        return torch.log_softmax(scores, dim=-1), normalized
    if len(prefix_group_ids) != scores.shape[0]:
        raise ValueError("One prefix group id is required per batch row")
    log_prior = torch.empty_like(scores)
    for group_id in set(prefix_group_ids):
        rows = [i for i, current_id in enumerate(prefix_group_ids) if current_id == group_id]
        group_scores = scores[rows].reshape(-1)
        log_prior[rows] = torch.log_softmax(group_scores, dim=0).reshape(len(rows), -1)
    return log_prior, normalized
