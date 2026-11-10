"""Reward-model interface used by EHR search."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

Trajectory = list[tuple[int, float]]


@dataclass
class RewardModelBaseConfig:
    step_tag: str


class RewardModelCallingFunction(ABC):
    def __init__(self, config: RewardModelBaseConfig):
        self.config = config
        self.step_tag = config.step_tag

    @abstractmethod
    def __call__(
        self, question_answer_pairs: list[tuple[Trajectory, Trajectory]],
        lm_step_tag: str = None,
    ) -> list[list[float]]:
        raise NotImplementedError
