"""Base class for Harvy skills.

A skill is a small controller that translates high-level intent into low-level
actions. The RL agent decides which skill to activate; the skill decides the
movement/camera/attack/use details for a few ticks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np


class BaseSkill(ABC):
    """Abstract skill controller."""

    def __init__(self, name: str):
        self.name = name
        self.active = False
        self.step_count = 0
        self.max_steps = 30

    def reset(self) -> None:
        self.active = False
        self.step_count = 0

    def activate(self) -> None:
        self.active = True
        self.step_count = 0

    @abstractmethod
    def can_use(self, obs: Dict[str, np.ndarray], raw_obs: Dict[str, Any]) -> bool:
        """Return True if this skill has a valid target/condition right now."""

    @abstractmethod
    def step(
        self, obs: Dict[str, np.ndarray], raw_obs: Dict[str, Any]
    ) -> Tuple[int, float, bool, Dict[str, Any]]:
        """Return (low_level_action_index, reward_modifier, skill_done, info)."""

    def get_state_features(self) -> np.ndarray:
        """Optional skill-state vector merged into the observation."""
        return np.zeros(0, dtype=np.float32)
