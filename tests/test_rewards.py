"""Tests for reward calculation and clipping."""

import numpy as np
import pytest

from env.rewards import RewardCalculator
from env.knowledge_base import MinecraftKnowledgeBase


def _make_obs(health=20.0, food=20.0, inventory_sum=0.0):
    return {
        "self_health": np.array([health], dtype=np.float32),
        "self_food": np.array([food], dtype=np.float32),
        "inventory": np.array([inventory_sum], dtype=np.float32),
        "self_held_item_id": np.array([0.0], dtype=np.float32),
    }


def test_reward_clipping():
    calc = RewardCalculator()
    # A large negative signal should be clipped to -10.
    reward = calc.compute(_make_obs(), {"death": -1000}, action_idx=0, danger_level=0.0)
    assert reward == pytest.approx(-10.0)


def test_alive_tick_reward():
    calc = RewardCalculator()
    reward = calc.compute(
        _make_obs(), {"alive_tick": 1}, action_idx=0, danger_level=0.0
    )
    assert reward == pytest.approx(0.05)


def test_noop_bias():
    calc = RewardCalculator(noop_bias=0.05)
    reward = calc.compute(
        _make_obs(), {"alive_tick": 1}, action_idx=1, danger_level=0.0
    )
    assert reward == pytest.approx(0.05 - 0.05)


def test_noop_bias_only_when_safe():
    calc = RewardCalculator(noop_bias=0.05)
    reward = calc.compute(
        _make_obs(), {"alive_tick": 1}, action_idx=1, danger_level=0.5
    )
    assert reward == pytest.approx(0.05)


def test_knowledge_base_bonus_for_correct_tool():
    kb = MinecraftKnowledgeBase()
    calc = RewardCalculator(knowledge_base=kb)
    obs = {
        "self_health": np.array([20.0], dtype=np.float32),
        "self_food": np.array([20.0], dtype=np.float32),
        "inventory": np.array([0.0], dtype=np.float32),
        "self_held_item_id": np.array([1.0], dtype=np.float32),  # wooden_pickaxe
    }
    reward = calc.compute(
        obs,
        {"item_mined": 1, "block_broken_name": "stone"},
        action_idx=0,
        danger_level=0.0,
    )
    assert reward > 0.0


def test_knowledge_base_penalty_for_wrong_tool():
    kb = MinecraftKnowledgeBase()
    calc = RewardCalculator(knowledge_base=kb)
    obs = {
        "self_health": np.array([20.0], dtype=np.float32),
        "self_food": np.array([20.0], dtype=np.float32),
        "inventory": np.array([0.0], dtype=np.float32),
        "self_held_item_id": np.array([3.0], dtype=np.float32),  # wooden_sword
    }
    reward = calc.compute(
        obs,
        {"item_mined": 1, "block_broken_name": "stone"},
        action_idx=0,
        danger_level=0.0,
    )
    assert reward < 0.0
