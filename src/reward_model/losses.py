"""Losses for the stepwise process reward model."""

import torch


PRM_LOSS_NAME = "positive_token_probability_squared_error"
PRM_LOSS_REDUCTION = "sum_markers_per_sequence_mean_nonempty_sequences"
PRM_OBJECTIVE_FILE = "prm_training_objective.json"


def reward_marker_mse(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    targets: torch.Tensor,
    marker_mask: torch.Tensor,
) -> torch.Tensor:
    """Paper equation (6): sum marker squared errors per sequence, then batch mean.

    Only rows with at least one supervised marker enter the batch mean. This
    accepts existing training records and their marker masks unchanged.
    """
    if positive_logits.ndim != 2 or not (
        positive_logits.shape == negative_logits.shape == targets.shape == marker_mask.shape
    ):
        raise ValueError("PRM logits, targets, and marker mask must have shape [batch, time]")

    marker_mask = marker_mask.bool()
    valid_rows = marker_mask.any(dim=1)
    if not valid_rows.any():
        raise ValueError("PRM batch has no supervised reward markers")

    targets = targets.float()
    valid_targets = targets[marker_mask]
    if not torch.isfinite(valid_targets).all() or ((valid_targets < 0) | (valid_targets > 1)).any():
        raise ValueError("PRM soft targets must be finite values in [0, 1]")

    reward = torch.sigmoid(positive_logits.float() - negative_logits.float())
    squared_error = (reward - targets).square().masked_fill(~marker_mask, 0.0)
    return squared_error.sum(dim=1)[valid_rows].mean()
