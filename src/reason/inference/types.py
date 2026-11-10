"""Inputs and transition-aligned outputs for EHR trajectory generation."""

from dataclasses import dataclass
from typing import Optional
import torch
from utils import (
    ATOMIC_CANDIDATE_POOL_SIZE, LANGEVIN_SAMPLES_PER_CANDIDATE,
    LANGEVIN_STEPS, TIME_UNCERTAINTY_BETA,
)


@dataclass
class TrajectoryGenerationConfig:
    n: int = 1
    temperature: float = 1.0
    top_k: Optional[int] = None
    max_new_tokens: int = 512
    max_age: float = 85 * 365.25
    no_repeat: bool = True
    termination_tokens: Optional[list[int]] = None
    config_type: str = "EHRCS"
    n_samples: int = LANGEVIN_SAMPLES_PER_CANDIDATE
    n_langevin_steps: int = LANGEVIN_STEPS
    age_is_normalized: bool = False
    time_uncertainty_beta: float = TIME_UNCERTAINTY_BETA
    atomic_candidate_pool_size: int = ATOMIC_CANDIDATE_POOL_SIZE
    calibration_dataset: Optional[str] = None


@dataclass
class TrajectoryGenerationResult:
    """Events/ages: [B, T], ages in raw days; probability fields: [B, T-1]."""

    events: torch.Tensor
    ages: torch.Tensor
    num_tokens: list[int]
    finish_reason: list[str]
    step_logprob: torch.Tensor
    type_logprob: torch.Tensor
    time_gap_logprob: torch.Tensor
    time_uncertainty: torch.Tensor

    @property
    def completion_tokens(self) -> int:
        return sum(self.num_tokens)
