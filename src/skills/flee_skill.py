"""Flee skill controller.

Turns away from the nearest hostile and runs until safe or max steps.
"""

import math
from typing import Any, Dict, Tuple

import numpy as np

from .base_skill import BaseSkill
from env.action_space import ACTION_NAMES


class FleeSkill(BaseSkill):
    """Run away from nearby hostiles."""

    def __init__(self, safe_distance: float = 12.0):
        super().__init__("flee")
        self.safe_distance = safe_distance
        self.max_steps = 40

        self._noop = ACTION_NAMES.index("noop")
        self._forward = ACTION_NAMES.index("move_forward")
        self._sprint_forward = ACTION_NAMES.index("sprint_forward")
        self._turn_left_45 = ACTION_NAMES.index("turn_left_45")
        self._turn_right_45 = ACTION_NAMES.index("turn_right_45")

    def can_use(self, obs: Dict[str, np.ndarray], raw_obs: Dict[str, Any]) -> bool:
        return self._nearest_hostile_distance(obs) is not None

    def step(
        self, obs: Dict[str, np.ndarray], raw_obs: Dict[str, Any]
    ) -> Tuple[int, float, bool, Dict[str, Any]]:
        self.step_count += 1
        info = {"distance_to_threat": 0.0, "action": "noop"}

        dist = self._nearest_hostile_distance(obs)
        if dist is None or dist >= self.safe_distance:
            info["action"] = "safe"
            return self._noop, 0.1, True, info

        info["distance_to_threat"] = dist
        target = self._nearest_hostile(obs)
        if target is None:
            return self._noop, 0.0, True, info

        # Run directly away from the threat.
        dx = -target["dx"]
        dz = -target["dz"]
        yaw_error = self._yaw_error_to_vector(obs, dx, dz)

        if abs(yaw_error) > 15:
            info["action"] = "face_away"
            return self._turn_left_45 if yaw_error < 0 else self._turn_right_45, -0.02, False, info

        info["action"] = "sprint_away"
        return self._sprint_forward, -0.01, False, info

    def _nearest_hostile_distance(self, obs: Dict[str, np.ndarray]) -> float:
        target = self._nearest_hostile(obs)
        return target["distance"] if target else None

    def _nearest_hostile(self, obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        type_ids = obs["entity_type_ids"]
        distances = obs["entity_distances"].squeeze(-1)
        hostiles = obs["entity_hostiles"].squeeze(-1)
        positions = obs["entity_positions"]

        best_idx = -1
        best_dist = float("inf")
        for i in range(len(type_ids)):
            if hostiles[i] > 0 and distances[i] > 0:
                if distances[i] < best_dist:
                    best_dist = distances[i]
                    best_idx = i

        if best_idx < 0:
            return None
        return {
            "index": best_idx,
            "distance": float(distances[best_idx]),
            "dx": float(positions[best_idx][0]),
            "dz": float(positions[best_idx][2]),
        }

    def _yaw_error_to_vector(self, obs: Dict[str, np.ndarray], dx: float, dz: float) -> float:
        yaw_to_target = math.degrees(math.atan2(dx, dz))
        self_yaw = float(obs["self_yaw"][0])
        error = yaw_to_target - self_yaw
        while error > 180:
            error -= 360
        while error < -180:
            error += 360
        return error
