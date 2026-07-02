"""Building skill controller.

Reads a simple JSON schematic and executes placement steps. Each step either
moves toward the next target position, faces it, or places a block.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .base_skill import BaseSkill
from env.action_space import ACTION_NAMES


class BuildSkill(BaseSkill):
    """Build a small structure from a schematic."""

    def __init__(self, schematic_path: str = "schematics/wall_4x4.json"):
        super().__init__("build")
        self.schematic_path = Path(schematic_path)
        self.schematic: List[Dict[str, Any]] = []
        self.target_index = 0
        self.max_steps = 200
        self._load_schematic()

        self._noop = ACTION_NAMES.index("noop")
        self._forward = ACTION_NAMES.index("move_forward")
        self._back = ACTION_NAMES.index("move_back")
        self._left = ACTION_NAMES.index("strafe_left")
        self._right = ACTION_NAMES.index("strafe_right")
        self._place = ACTION_NAMES.index("place_block")
        self._turn_left_15 = ACTION_NAMES.index("turn_left_15")
        self._turn_right_15 = ACTION_NAMES.index("turn_right_15")

    def _load_schematic(self) -> None:
        if not self.schematic_path.exists():
            self.schematic = []
            return
        with open(self.schematic_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.schematic = data.get("blocks", [])

    def reset(self) -> None:
        super().reset()
        self.target_index = 0

    def activate(self) -> None:
        super().activate()
        self.target_index = 0

    def can_use(self, obs: Dict[str, np.ndarray], raw_obs: Dict[str, Any]) -> bool:
        return len(self.schematic) > 0 and self._has_materials(obs)

    def _has_materials(self, obs: Dict[str, np.ndarray]) -> bool:
        # Rough check: any building block in inventory.
        inv = obs["inventory"]
        return float(inv.sum()) > len(self.schematic) * 0.5

    def step(
        self, obs: Dict[str, np.ndarray], raw_obs: Dict[str, Any]
    ) -> Tuple[int, float, bool, Dict[str, Any]]:
        self.step_count += 1
        info = {"target_block": None, "action": "noop"}

        if self.target_index >= len(self.schematic):
            info["action"] = "done"
            return self._noop, 0.5, True, info

        block = self.schematic[self.target_index]
        info["target_block"] = block
        target_pos = np.array([block["dx"], block["dy"], block["dz"]], dtype=np.float32)
        dist = float(np.linalg.norm(target_pos))

        if dist > 3.5:
            info["action"] = "approach"
            return self._forward, -0.01, False, info

        # Face target horizontally.
        yaw_error = self._yaw_error_to_point(obs, target_pos[0], target_pos[2])
        if abs(yaw_error) > 10:
            info["action"] = "face"
            return self._turn_left_15 if yaw_error < 0 else self._turn_right_15, -0.01, False, info

        # Place block.
        self.target_index += 1
        info["action"] = "place"
        return self._place, 0.2, False, info

    def _yaw_error_to_point(self, obs: Dict[str, np.ndarray], dx: float, dz: float) -> float:
        yaw_to_target = math.degrees(math.atan2(dx, dz))
        self_yaw = float(obs["self_yaw"][0])
        error = yaw_to_target - self_yaw
        while error > 180:
            error -= 360
        while error < -180:
            error += 360
        return error
