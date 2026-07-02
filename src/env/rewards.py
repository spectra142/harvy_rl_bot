"""Reward shaping for Harvy v2.

Simple, dense, and fully clipped.
"""

import numpy as np
from typing import Dict, Any


class RewardCalculator:
    """Compute reward from one step of reward_signal + observation."""

    def __init__(
        self,
        alive_tick: float = 0.01,
        damage: float = -0.5,
        heal: float = 0.1,
        death: float = -10.0,
        item_pickup: float = 0.1,
        clip: float = 1.0,
    ):
        self.alive_tick = alive_tick
        self.damage = damage
        self.heal = heal
        self.death = death
        self.item_pickup = item_pickup
        self.clip = clip

        self.prev_health = 20.0
        self.prev_inventory_sum = 0.0

    def reset(self) -> None:
        self.prev_health = 20.0
        self.prev_inventory_sum = 0.0

    def compute(
        self,
        obs: Dict[str, np.ndarray],
        reward_signal: Dict[str, Any],
    ) -> float:
        reward = 0.0

        # Per-tick survival.
        reward += reward_signal.get("alive_tick", 0) * self.alive_tick

        # Damage / heal from observation health (more reliable than signal).
        health = float(obs["self_health"][0])
        health_delta = health - self.prev_health
        if health_delta < 0:
            reward += health_delta * abs(self.damage)
        elif health_delta > 0:
            reward += health_delta * self.heal
        self.prev_health = health

        # Item pickup growth.
        inventory = obs["inventory"]
        inv_sum = float(inventory.sum())
        inv_delta = inv_sum - self.prev_inventory_sum
        if inv_delta > 0:
            reward += inv_delta * self.item_pickup
        self.prev_inventory_sum = inv_sum

        # Food eaten / items mined signals.
        reward += reward_signal.get("food_eaten", 0) * 0.05
        reward += reward_signal.get("item_mined", 0) * 0.05

        # Clip the continuous/shaped components so they stay in range,
        # but keep the terminal death penalty unclipped so it dominates.
        reward = float(np.clip(reward, -self.clip, self.clip))

        # Death.
        if reward_signal.get("death", 0):
            reward += self.death

        return reward
