"""Combat skill controller.

When active, the bot selects the nearest hostile entity, faces it, approaches,
attacks, and retreats if health is low. The skill reports damage dealt/taken
and ends when the target dies, escapes, or the bot is too wounded.
"""

from typing import Any, Dict, Tuple

import numpy as np

from .base_skill import BaseSkill
from env.action_space import ACTION_NAMES
from env.observation import ITEM_TO_ID

WEAPON_PRIORITY = [
    "diamond_sword",
    "iron_sword",
    "stone_sword",
    "wooden_sword",
    "diamond_axe",
    "iron_axe",
    "stone_axe",
    "wooden_axe",
]


class CombatSkill(BaseSkill):
    """Close-combat controller."""

    def __init__(
        self,
        retreat_health: float = 6.0,
        attack_range: float = 3.5,
        approach_range: float = 16.0,
    ):
        super().__init__("combat")
        self.retreat_health = retreat_health
        self.attack_range = attack_range
        self.approach_range = approach_range
        self.max_steps = 60
        self.last_health = 20.0

        # Cached action indices.
        self._noop = ACTION_NAMES.index("noop")
        self._forward = ACTION_NAMES.index("move_forward")
        self._back = ACTION_NAMES.index("move_back")
        self._left = ACTION_NAMES.index("strafe_left")
        self._right = ACTION_NAMES.index("strafe_right")
        self._attack = ACTION_NAMES.index("attack")
        self._turn_left_45 = ACTION_NAMES.index("turn_left_45")
        self._turn_right_45 = ACTION_NAMES.index("turn_right_45")
        self._turn_left_15 = ACTION_NAMES.index("turn_left_15")
        self._turn_right_15 = ACTION_NAMES.index("turn_right_15")
        self._hotbar_indices = [ACTION_NAMES.index(f"hotbar_{i}") for i in range(1, 10)]

    def reset(self) -> None:
        super().reset()
        self.target_index = -1
        self.last_health = 20.0
        self.damage_dealt = 0.0
        self.damage_taken = 0.0

    def activate(self) -> None:
        super().activate()
        self.target_index = -1
        self.damage_dealt = 0.0
        self.damage_taken = 0.0

    def can_use(self, obs: Dict[str, np.ndarray], raw_obs: Dict[str, Any]) -> bool:
        return self._find_target(obs) is not None

    def step(
        self, obs: Dict[str, np.ndarray], raw_obs: Dict[str, Any]
    ) -> Tuple[int, float, bool, Dict[str, Any]]:
        self.step_count += 1
        info = {"target": None, "distance": 0.0, "action": "noop"}
        reward = 0.0

        health = float(obs["self_health"][0])
        health_delta = health - self.last_health
        self.last_health = health
        if health_delta < 0:
            self.damage_taken += -health_delta
            reward -= 0.2  # shaped penalty while in combat

        # Retreat if critically wounded.
        if health <= self.retreat_health:
            info["action"] = "retreat"
            done = self.step_count > 10
            return self._back, reward - 0.1, done, info

        # Ensure we have a weapon equipped.
        weapon_action = self._select_weapon_action(obs)
        if weapon_action is not None:
            info["action"] = "equip_weapon"
            return weapon_action, reward, False, info

        target = self._find_target(obs)
        if target is None:
            info["action"] = "no_target"
            return self._noop, reward, True, info

        self.target_index = target["index"]
        info["target"] = target["type"]
        info["distance"] = target["distance"]

        # If target is far, approach.
        if target["distance"] > self.attack_range:
            info["action"] = "approach"
            return self._forward, reward, False, info

        # Face target horizontally before attacking.
        yaw_error = self._yaw_error_to_target(obs, target)
        if abs(yaw_error) > 10:
            info["action"] = "face_target"
            return self._turn_left_15 if yaw_error < 0 else self._turn_right_15, reward, False, info

        # In range and facing: attack with occasional strafe.
        if self.step_count % 5 == 0:
            info["action"] = "strafe_attack"
            return self._left if self.step_count % 10 == 0 else self._right, reward, False, info

        info["action"] = "attack"
        # Small reward for maintaining contact; actual kill reward comes from env.
        reward += 0.02
        return self._attack, reward, False, info

    def _select_weapon_action(self, obs: Dict[str, np.ndarray]) -> int:
        """Return hotbar action if current held item is not the best weapon."""
        held_id = int(obs["self_held_item_id"][0])
        held_name = None
        for name, idx in ITEM_TO_ID.items():
            if idx == held_id:
                held_name = name
                break

        # Find best weapon in inventory (slots 0-8 correspond to hotbar).
        inv = obs["inventory"]
        best_weapon_name = None
        best_weapon_slot = -1
        for slot in range(9):
            # We don't know which hotbar slot has which item from inventory alone.
            # Approximation: scan weapon list and pick first found in inventory.
            for weapon in WEAPON_PRIORITY:
                weapon_id = ITEM_TO_ID.get(weapon)
                if weapon_id is not None and inv[weapon_id] > 0:
                    best_weapon_name = weapon
                    best_weapon_slot = slot
                    break
            if best_weapon_name:
                break

        if best_weapon_name is None:
            return None
        if held_name == best_weapon_name:
            return None
        return self._hotbar_indices[best_weapon_slot]

    def _find_target(self, obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Return nearest hostile entity within approach_range."""
        type_ids = obs["entity_type_ids"]
        distances = obs["entity_distances"].squeeze(-1)
        hostiles = obs["entity_hostiles"].squeeze(-1)
        healths = obs["entity_healths"].squeeze(-1)

        best_idx = -1
        best_dist = float("inf")
        for i in range(len(type_ids)):
            if hostiles[i] > 0 and distances[i] > 0 and distances[i] <= self.approach_range:
                if distances[i] < best_dist:
                    best_dist = distances[i]
                    best_idx = i

        if best_idx < 0:
            return None

        return {
            "index": best_idx,
            "type": int(type_ids[best_idx]),
            "distance": float(distances[best_idx]),
            "health": float(healths[best_idx]),
        }

    def _yaw_error_to_target(
        self, obs: Dict[str, np.ndarray], target: Dict[str, Any]
    ) -> float:
        """Return signed yaw difference from bot to target (degrees)."""
        idx = target["index"]
        dx = float(obs["entity_positions"][idx][0])
        dz = float(obs["entity_positions"][idx][2])
        yaw_to_target = np.degrees(np.arctan2(dx, dz))
        self_yaw = float(obs["self_yaw"][0])
        error = yaw_to_target - self_yaw
        while error > 180:
            error -= 360
        while error < -180:
            error += 360
        return error
