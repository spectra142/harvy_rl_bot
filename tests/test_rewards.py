"""Tests for reward shaping."""

import numpy as np

from env.rewards import RewardCalculator


def _obs(health: float = 20.0, inventory_sum: float = 0.0):
    inv = np.zeros(42, dtype=np.float32)
    inv[0] = inventory_sum
    return {
        "self_health": np.array([health], dtype=np.float32),
        "inventory": inv,
    }


def test_alive_tick():
    calc = RewardCalculator(alive_tick=0.01, clip=1.0)
    r = calc.compute(_obs(), {"alive_tick": 1})
    assert r == 0.01


def test_damage_penalty():
    calc = RewardCalculator(damage=-0.5, clip=1.0)
    calc.prev_health = 20.0
    r = calc.compute(_obs(health=17.0), {})
    assert r == -1.0  # clipped


def test_death_unclipped():
    calc = RewardCalculator(death=-10.0, clip=1.0)
    r = calc.compute(_obs(health=20.0), {"death": 1})
    assert r == -10.0


def test_death_dominates_other_rewards():
    calc = RewardCalculator(damage=-0.5, death=-10.0, clip=1.0)
    calc.prev_health = 20.0
    r = calc.compute(_obs(health=10.0), {"death": 1})
    assert r == -11.0  # shaped clipped to -1 + death -10


def test_inventory_growth():
    calc = RewardCalculator(item_pickup=0.1, clip=1.0)
    calc.prev_inventory_sum = 0.0
    obs = _obs(inventory_sum=5.0)
    r = calc.compute(obs, {})
    assert r == 0.5


def test_reward_clip_without_death():
    calc = RewardCalculator(damage=-100.0, clip=1.0)
    calc.prev_health = 20.0
    r = calc.compute(_obs(health=0.0), {})
    assert r == -1.0
