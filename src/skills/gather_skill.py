"""Gather skill controller.

Finds the nearest target block in the voxel grid, faces it, approaches, and
mines it. Targets are configurable (default: logs and coal).
"""

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from .base_skill import BaseSkill
from env.action_space import ACTION_NAMES

# Block state IDs that count as gatherable. These are vanilla state IDs for
# common blocks; in a real deployment you'd map from minecraft-data.
TARGET_STATE_IDS = {
    17,   # oak log (approximate; real state IDs vary by version)
    18,   # birch log approx
    19,   # spruce log approx
    20,   # jungle log approx
    21,   # acacia log approx
    22,   # dark oak log approx
    173,  # coal ore approx
}


class GatherSkill(BaseSkill):
    """Harvest target blocks from the environment."""

    def __init__(self, target_state_ids: List[int] = None):
        super().__init__("gather")
        self.target_state_ids = set(target_state_ids or TARGET_STATE_IDS)
        self.max_steps = 80
        self.target_voxel: Tuple[int, int, int] = (-1, -1, -1)

        self._noop = ACTION_NAMES.index("noop")
        self._forward = ACTION_NAMES.index("move_forward")
        self._mine = ACTION_NAMES.index("mine")
        self._turn_left_15 = ACTION_NAMES.index("turn_left_15")
        self._turn_right_15 = ACTION_NAMES.index("turn_right_15")

    def reset(self) -> None:
        super().reset()
        self.target_voxel = (-1, -1, -1)

    def activate(self) -> None:
        super().activate()
        self.target_voxel = (-1, -1, -1)

    def can_use(self, obs: Dict[str, np.ndarray], raw_obs: Dict[str, Any]) -> bool:
        return self._find_target(obs) is not None

    def step(
        self, obs: Dict[str, np.ndarray], raw_obs: Dict[str, Any]
    ) -> Tuple[int, float, bool, Dict[str, Any]]:
        self.step_count += 1
        info = {"target_voxel": None, "action": "noop"}

        if self.target_voxel == (-1, -1, -1):
            target = self._find_target(obs)
            if target is None:
                info["action"] = "no_target"
                return self._noop, 0.0, True, info
            self.target_voxel = target

        info["target_voxel"] = self.target_voxel
        x, z, y = self.target_voxel
        # Voxel grid center is the bot position; convert to relative world coords.
        dx = x - 5
        dy = y - 3
        dz = z - 5
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist > 3.5:
            info["action"] = "approach"
            return self._forward, -0.01, False, info

        yaw_error = self._yaw_error_to_point(obs, dx, dz)
        if abs(yaw_error) > 10:
            info["action"] = "face"
            return self._turn_left_15 if yaw_error < 0 else self._turn_right_15, -0.01, False, info

        # Mine and pick next target.
        self.target_voxel = (-1, -1, -1)
        info["action"] = "mine"
        return self._mine, 0.2, False, info

    def _find_target(self, obs: Dict[str, np.ndarray]) -> Tuple[int, int, int]:
        voxel = obs["voxel_grid"]
        best = None
        best_dist = float("inf")
        for x in range(voxel.shape[0]):
            for z in range(voxel.shape[1]):
                for y in range(voxel.shape[2]):
                    state_id = int(voxel[x, z, y])
                    if state_id in self.target_state_ids:
                        dx = x - 5
                        dz = z - 5
                        dy = y - 3
                        dist = math.sqrt(dx * dx + dz * dz + dy * dy)
                        if dist < best_dist:
                            best_dist = dist
                            best = (x, z, y)
        return best

    def _yaw_error_to_point(self, obs: Dict[str, np.ndarray], dx: float, dz: float) -> float:
        yaw_to_target = math.degrees(math.atan2(dx, dz))
        self_yaw = float(obs["self_yaw"][0])
        error = yaw_to_target - self_yaw
        while error > 180:
            error -= 360
        while error < -180:
            error += 360
        return error
