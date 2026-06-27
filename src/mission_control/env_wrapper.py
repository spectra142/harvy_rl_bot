"""Wrapper that streams bot state and supports manual override."""

import logging
from typing import Any, Dict

import gymnasium as gym
import numpy as np

from env.actions import ACTION_NAMES
from env.observation import ITEM_ID_TO_NAME

logger = logging.getLogger(__name__)

ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTION_NAMES)}


def _item_name_from_id(item_id: int) -> str:
    return ITEM_ID_TO_NAME.get(int(item_id), f"item_{int(item_id)}")


def _first(arr, default=0.0):
    """Return the first scalar value from *arr*, falling back to *default*."""
    try:
        return np.asarray(arr).flatten()[0]
    except (IndexError, TypeError, ValueError):
        return default


class MissionControlEnvWrapper(gym.Wrapper):
    def __init__(self, env, manager):
        super().__init__(env)
        self.manager = manager

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._publish_state(obs, info)
        return obs, info

    def step(self, action):
        if self.manager.is_manual_override():
            action_name = self.manager.get_pending_action()
            if action_name:
                action = ACTION_TO_INDEX.get(action_name, 0)
                self.manager.clear_pending_action()

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._publish_state(obs, info)

        goal = None
        with self.manager._lock:
            if self.manager.pending_reset is not None:
                goal = self.manager.pending_reset
                self.manager.pending_reset = None

        if goal is not None:
            obs, info = self.env.reset(options={"goal": goal})
            self._publish_state(obs, info)
            return obs, 0.0, False, False, info

        return obs, reward, terminated, truncated, info

    def _publish_state(self, obs: Dict[str, Any], info: Dict[str, Any]):
        inventory = []
        inv_counts = info.get("inventory_counts", {})
        for slot_idx, (item_name, count) in enumerate(sorted(inv_counts.items())):
            count = int(count)
            if count > 0:
                inventory.append(
                    {
                        "index": slot_idx,
                        "item": item_name,
                        "count": count,
                    }
                )

        held_item = info.get("held_item")
        if held_item is None:
            held_item_id = int(_first(obs.get("self_held_item_id", 0.0), 0.0))
            held_item = _item_name_from_id(held_item_id)

        raw_position = info.get("position")
        if raw_position is not None:
            position_arr = np.asarray(raw_position)
            position = {
                "x": float(position_arr[0]),
                "y": float(position_arr[1]) if len(position_arr) > 1 else 64.0,
                "z": float(position_arr[2]) if len(position_arr) > 2 else 0.0,
            }
        else:
            position_arr = np.asarray(
                obs.get("self_position", [0.0, 64.0, 0.0])
            ).flatten()
            position = {
                "x": float(_first(position_arr, 0.0)),
                "y": float(_first(position_arr[1:], 64.0)),
                "z": float(_first(position_arr[2:], 0.0)),
            }

        health = info.get("health")
        if health is None:
            health = float(_first(obs.get("self_health", [20.0]), 20.0))
        else:
            health = float(health)

        hunger = info.get("food")
        if hunger is None:
            hunger = float(_first(obs.get("self_food", [20.0]), 20.0))
        else:
            hunger = float(hunger)

        self.manager.post_message(
            {
                "type": "bot_state",
                "health": health,
                "hunger": hunger,
                "position": position,
                "held_item": held_item,
                "inventory": inventory,
                "goal": str(info.get("goal", "")),
                "termination_reason": info.get("termination_reason"),
            }
        )
