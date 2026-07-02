"""Skill executor: maps high-level skill actions to low-level commands."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base_skill import BaseSkill
from env.action_space import LOW_LEVEL_ACTION_COUNT


class SkillExecutor:
    """Manages active skill and routes steps to the right controller."""

    def __init__(self, skills: List[BaseSkill]):
        self.skills = skills
        self.active_skill: Optional[BaseSkill] = None
        self.active_index = -1

    def reset(self) -> None:
        if self.active_skill:
            self.active_skill.reset()
        self.active_skill = None
        self.active_index = -1

    def skill_count(self) -> int:
        return len(self.skills)

    def skill_action_indices(self) -> List[int]:
        """Indices the policy uses to select each skill."""
        return [LOW_LEVEL_ACTION_COUNT + i for i in range(len(self.skills))]

    def select_skill(self, action_idx: int, obs: Dict[str, np.ndarray], raw_obs: Dict[str, Any]) -> bool:
        """Activate a skill from a high-level action index."""
        local_idx = action_idx - LOW_LEVEL_ACTION_COUNT
        if local_idx < 0 or local_idx >= len(self.skills):
            return False
        skill = self.skills[local_idx]
        if not skill.can_use(obs, raw_obs):
            return False
        self.active_skill = skill
        self.active_index = local_idx
        skill.activate()
        return True

    def is_skill_action(self, action_idx: int) -> bool:
        return action_idx >= LOW_LEVEL_ACTION_COUNT

    def step(
        self, action_idx: int, obs: Dict[str, np.ndarray], raw_obs: Dict[str, Any]
    ) -> Tuple[int, float, bool, Dict[str, Any]]:
        """Resolve the actual low-level action for this step.

        If the action is a skill selection, activate it and take its first step.
        If a skill is already active, let it run (unless action_idx is a new
        skill selection, which interrupts).
        """
        info = {"skill": None, "skill_step": 0}

        # New skill selection interrupts any active skill.
        if self.is_skill_action(action_idx):
            self.select_skill(action_idx, obs, raw_obs)

        if self.active_skill is None:
            return action_idx, 0.0, False, info

        skill = self.active_skill
        info["skill"] = skill.name
        info["skill_step"] = skill.step_count

        low_action, reward_mod, done, skill_info = skill.step(obs, raw_obs)
        info.update(skill_info)

        if done or skill.step_count >= skill.max_steps:
            self.active_skill.reset()
            self.active_skill = None
            self.active_index = -1

        return low_action, reward_mod, done, info

    def get_state_features(self) -> np.ndarray:
        """One-hot active skill + step fraction."""
        n = len(self.skills)
        vec = np.zeros(n + 1, dtype=np.float32)
        if self.active_skill and self.active_index >= 0:
            vec[self.active_index] = 1.0
            vec[-1] = min(1.0, self.active_skill.step_count / max(1, self.active_skill.max_steps))
        return vec
