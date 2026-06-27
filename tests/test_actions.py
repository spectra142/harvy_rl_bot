"""Tests for the action space mapping."""

import pytest
from env.actions import (
    ACTION_NAMES,
    ACTION_COUNT,
    action_name_to_json,
    get_action_space,
)

EXPECTED_LOW_LEVEL = [
    "noop",
    "move_forward",
    "move_back",
    "strafe_left",
    "strafe_right",
    "sprint",
    "sneak",
    "jump",
    "turn_left_15",
    "turn_right_15",
    "look_up_15",
    "look_down_15",
    "attack",
    "use_item",
    "place_block",
    "select_slot_0",
    "select_slot_1",
    "select_slot_2",
    "select_slot_3",
    "select_slot_4",
    "select_slot_5",
    "select_slot_6",
    "select_slot_7",
    "select_slot_8",
    "jump_forward",
]

EXPECTED_SKILLS = [
    "skill_gather_log",
    "skill_gather_stone",
    "skill_gather_coal",
    "skill_craft_planks",
    "skill_craft_sticks",
    "skill_craft_pickaxe",
    "skill_craft_sword",
    "skill_craft_crafting_table",
    "skill_eat_food",
    "skill_equip_best_sword",
    "skill_equip_best_pickaxe",
    "skill_build_shelter",
    "skill_flee",
    "skill_explore",
    "skill_attack_hostile",
]


def test_action_count():
    assert ACTION_COUNT == 40
    assert len(ACTION_NAMES) == 40


def test_action_order():
    assert ACTION_NAMES[:25] == EXPECTED_LOW_LEVEL
    assert ACTION_NAMES[25:] == EXPECTED_SKILLS


@pytest.mark.parametrize("idx, name", list(enumerate(ACTION_NAMES)))
def test_action_name_to_json(idx, name):
    assert action_name_to_json(idx) == {"action": name, "value": 1.0}


def test_action_space_size():
    space = get_action_space()
    assert space.n == 40
