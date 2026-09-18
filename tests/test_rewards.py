"""Tests for reward calculation and clipping."""

import numpy as np
import pytest

from env.rewards import RewardCalculator
from env.knowledge_base import MinecraftKnowledgeBase
from env.observation import ITEM_NAME_TO_ID, INVENTORY_SIZE


def _make_obs(health=20.0, food=20.0, inventory_sum=0.0):
    return {
        "self_health": np.array([health], dtype=np.float32),
        "self_food": np.array([food], dtype=np.float32),
        "inventory": np.array([inventory_sum], dtype=np.float32),
        "self_held_item_id": np.array([0.0], dtype=np.float32),
    }


def test_reward_clipping():
    calc = RewardCalculator()
    # death is a 0/1 count signal; the -100 penalty must clip to -10.
    reward = calc.compute(_make_obs(), {"death": 1}, action_idx=1, danger_level=0.0)
    assert reward == pytest.approx(-10.0)


def test_alive_tick_reward():
    calc = RewardCalculator()
    reward = calc.compute(
        _make_obs(), {"alive_tick": 1}, action_idx=1, danger_level=0.5
    )
    assert reward == pytest.approx(0.01)


def test_noop_bias_penalizes_idling():
    """The noop bias punishes standing still when safe, not moving."""
    calc = RewardCalculator(noop_bias=0.05)
    reward = calc.compute(
        _make_obs(), {"alive_tick": 1}, action_idx=0, danger_level=0.0
    )
    assert reward == pytest.approx(0.01 - 0.05)


def test_noop_bias_not_when_moving():
    calc = RewardCalculator(noop_bias=0.05)
    reward = calc.compute(
        _make_obs(), {"alive_tick": 1}, action_idx=1, danger_level=0.0
    )
    assert reward == pytest.approx(0.01)


def test_noop_bias_only_when_safe():
    calc = RewardCalculator(noop_bias=0.05)
    reward = calc.compute(
        _make_obs(), {"alive_tick": 1}, action_idx=0, danger_level=0.5
    )
    assert reward == pytest.approx(0.01)


def test_damage_punished_once():
    """Damage comes only from the damage_taken signal, not health deltas."""
    calc = RewardCalculator()
    calc.prev_health = 20.0
    # 4 damage via signal, health drops 4 in the obs: must count ONCE (-4).
    reward = calc.compute(
        _make_obs(health=16.0),
        {"alive_tick": 1, "damage_taken": 4.0},
        action_idx=1,
        danger_level=0.5,
    )
    assert reward == pytest.approx(0.01 - 4.0)


def test_mob_kill_scales_from_count():
    calc = RewardCalculator()
    reward = calc.compute(
        _make_obs(), {"mob_killed": 2}, action_idx=1, danger_level=0.5
    )
    assert reward == pytest.approx(10.0)  # 2 kills x 5.0


def test_knowledge_base_bonus_for_correct_tool():
    kb = MinecraftKnowledgeBase()
    calc = RewardCalculator(knowledge_base=kb)
    obs = {
        "self_health": np.array([20.0], dtype=np.float32),
        "self_food": np.array([20.0], dtype=np.float32),
        "inventory": np.zeros((INVENTORY_SIZE,), dtype=np.float32),
        "self_held_item_id": np.array(
            [float(ITEM_NAME_TO_ID["wooden_pickaxe"])], dtype=np.float32
        ),
    }
    reward = calc.compute(
        obs,
        {"item_mined": 1, "block_broken_name": "stone"},
        action_idx=1,
        danger_level=0.0,
    )
    assert reward > 0.0


def test_knowledge_base_penalty_for_wrong_tool():
    kb = MinecraftKnowledgeBase()
    calc = RewardCalculator(knowledge_base=kb)
    obs = {
        "self_health": np.array([20.0], dtype=np.float32),
        "self_food": np.array([20.0], dtype=np.float32),
        "inventory": np.zeros((INVENTORY_SIZE,), dtype=np.float32),
        "self_held_item_id": np.array(
            [float(ITEM_NAME_TO_ID["wooden_sword"])], dtype=np.float32
        ),
    }
    reward = calc.compute(
        obs,
        {"item_mined": 1, "block_broken_name": "stone"},
        action_idx=1,
        danger_level=0.0,
    )
    assert reward < 0.0


def test_goal_reward_item_pickup():
    """gather_logs goal rewards actual oak_log inventory gains."""
    calc = RewardCalculator(goal="gather_logs")
    inv = np.zeros((INVENTORY_SIZE,), dtype=np.float32)
    obs = {
        "self_health": np.array([20.0], dtype=np.float32),
        "self_food": np.array([20.0], dtype=np.float32),
        "inventory": inv.copy(),
        "self_held_item_id": np.array([0.0], dtype=np.float32),
    }
    calc.reset(obs)  # prime baselines
    inv[ITEM_NAME_TO_ID["oak_log"]] = 3.0
    obs["inventory"] = inv
    reward = calc.compute(obs, {"alive_tick": 1}, action_idx=1, danger_level=0.5)
    # bonus 5.0 + 3 * per_item 1.0 + inventory growth 3 * 0.1 + alive 0.01
    assert reward == pytest.approx(5.0 + 3.0 + 0.3 + 0.01)


def test_explore_goal_rewards_distance():
    calc = RewardCalculator(goal="explore")
    calc.reset(_make_obs())
    reward = calc.compute(
        _make_obs(),
        {"alive_tick": 1, "distance_moved": 10.0},
        action_idx=1,
        danger_level=0.5,
    )
    assert reward == pytest.approx(0.01 + 10.0 * 0.1)


def test_reset_primes_inventory_baseline():
    """No spurious inventory jackpot on step 1 after a reset."""
    calc = RewardCalculator()
    inv = np.zeros((INVENTORY_SIZE,), dtype=np.float32)
    inv[ITEM_NAME_TO_ID["oak_log"]] = 10.0  # carried over from last episode
    obs = {
        "self_health": np.array([20.0], dtype=np.float32),
        "self_food": np.array([20.0], dtype=np.float32),
        "inventory": inv,
        "self_held_item_id": np.array([0.0], dtype=np.float32),
    }
    calc.reset(obs)
    reward = calc.compute(obs, {"alive_tick": 1}, action_idx=1, danger_level=0.5)
    assert reward == pytest.approx(0.01)  # no inventory_delta bonus
