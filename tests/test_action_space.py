"""Tests for the low-level action space."""

import json

from env.action_space import (
    ACTION_NAMES,
    ACTION_COUNT,
    action_index_to_command,
    action_index_to_json,
    get_action_space,
)


def test_action_count_matches_space():
    space = get_action_space()
    assert space.n == ACTION_COUNT
    assert len(ACTION_NAMES) == ACTION_COUNT


def test_noop_is_zero():
    cmd = action_index_to_command(0)
    assert all(v == 0 or v is None for v in cmd.values())


def test_move_forward():
    cmd = action_index_to_command(ACTION_NAMES.index("move_forward"))
    assert cmd["forward"] == 1
    assert cmd["back"] == 0


def test_turn_right_45():
    cmd = action_index_to_command(ACTION_NAMES.index("turn_right_45"))
    assert cmd["yaw_delta"] == 45


def test_hotbar_5():
    cmd = action_index_to_command(ACTION_NAMES.index("hotbar_5"))
    assert cmd["hotbar"] == 4


def test_json_roundtrip():
    s = action_index_to_json(ACTION_NAMES.index("jump_forward"))
    cmd = json.loads(s)
    assert cmd["forward"] == 1
    assert cmd["jump"] == 1
    assert cmd["place_block"] == 0
    assert cmd["eat"] == 0


def test_place_block_distinct_from_use():
    cmd = action_index_to_command(ACTION_NAMES.index("place_block"))
    assert cmd["place_block"] == 1
    assert cmd["use"] == 0


def test_eat_distinct_from_use():
    cmd = action_index_to_command(ACTION_NAMES.index("eat"))
    assert cmd["eat"] == 1
    assert cmd["use"] == 0


def test_invalid_index_defaults_to_noop():
    cmd = action_index_to_command(-1)
    assert cmd == action_index_to_command(0)
    cmd = action_index_to_command(999)
    assert cmd == action_index_to_command(0)
