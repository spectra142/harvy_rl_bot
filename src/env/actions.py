"""Action space definitions and mapping."""

import gymnasium as gym

# Discrete action space: 40 actions (25 low-level + 15 high-level skills)
ACTION_NAMES = [
    # Low-level actions (indices 0-24)
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
    # High-level skill actions (indices 25-39)
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

ACTION_COUNT = len(ACTION_NAMES)


def action_name_to_json(action_idx: int) -> dict:
    """Convert action index to JSON command for the bot."""
    name = ACTION_NAMES[action_idx]
    return {"action": name, "value": 1.0}


def get_action_space() -> gym.spaces.Discrete:
    """Return the Gymnasium action space."""
    return gym.spaces.Discrete(ACTION_COUNT)
