"""Language-model interface used by EHR search."""

from abc import ABC, abstractmethod
import torch
from reason.inference.types import TrajectoryGenerationConfig, TrajectoryGenerationResult


class LanguageModelCallingFunction(ABC):
    def __init__(self, lm_step_tag: str = None):
        self.lm_step_tag = lm_step_tag

    @abstractmethod
    def forward(
        self, input_ehr: tuple[torch.Tensor, torch.Tensor],
        config: TrajectoryGenerationConfig,
    ) -> TrajectoryGenerationResult:
        raise NotImplementedError
