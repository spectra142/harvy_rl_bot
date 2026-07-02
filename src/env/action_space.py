"""Low-level discrete action space for Harvy v2.

Includes low-level actions plus skill-selection actions.
"""

import json
import gymnasium as gym
from typing import Dict, Any

# Low-level actions.
LOW_LEVEL_ACTION_NAMES = [
    # Movement (0-7)
    "noop",
    "move_forward",
    "move_back",
    "strafe_left",
    "strafe_right",
    "jump",
    "jump_forward",
    "sprint_forward",
    # Camera (8-13)
    "turn_left_15",
    "turn_right_15",
    "turn_left_45",
    "turn_right_45",
    "look_up_15",
    "look_down_15",
    # Interaction (14-17)
    "attack",
    "mine",
    "use",
    "place_block",
    "eat",
    # Hotbar (17-25)
    "hotbar_1",
    "hotbar_2",
    "hotbar_3",
    "hotbar_4",
    "hotbar_5",
    "hotbar_6",
    "hotbar_7",
    "hotbar_8",
    "hotbar_9",
]

LOW_LEVEL_ACTION_COUNT = len(LOW_LEVEL_ACTION_NAMES)

# Skill actions appended after low-level actions.
SKILL_ACTION_NAMES = [
    "skill_combat",
    "skill_build",
    "skill_gather",
    "skill_flee",
]

SKILL_ACTION_COUNT = len(SKILL_ACTION_NAMES)
ACTION_NAMES = LOW_LEVEL_ACTION_NAMES + SKILL_ACTION_NAMES
ACTION_COUNT = len(ACTION_NAMES)


def is_skill_action(action_idx: int) -> bool:
    return action_idx >= LOW_LEVEL_ACTION_COUNT


def skill_action_name(action_idx: int) -> str:
    if not is_skill_action(action_idx):
        return ""
    return SKILL_ACTION_NAMES[action_idx - LOW_LEVEL_ACTION_COUNT]


def action_index_to_command(action_idx: int) -> Dict[str, Any]:
    """Convert a discrete action index to the low-level JSON command.

    For skill actions, returns a noop command; the skill executor resolves the
    actual low-level action.
    """
    action_idx = int(action_idx)
    if action_idx < 0 or action_idx >= ACTION_COUNT:
        action_idx = 0

    # Skill selection: emit noop; executor will override.
    if is_skill_action(action_idx):
        return _noop_command()

    name = ACTION_NAMES[action_idx]
    cmd = _noop_command()

    if name == "noop":
        pass
    elif name == "move_forward":
        cmd["forward"] = 1
    elif name == "move_back":
        cmd["back"] = 1
    elif name == "strafe_left":
        cmd["left"] = 1
    elif name == "strafe_right":
        cmd["right"] = 1
    elif name == "jump":
        cmd["jump"] = 1
    elif name == "jump_forward":
        cmd["forward"] = 1
        cmd["jump"] = 1
    elif name == "sprint_forward":
        cmd["forward"] = 1
        cmd["sprint"] = 1
    elif name == "turn_left_15":
        cmd["yaw_delta"] = -15
    elif name == "turn_right_15":
        cmd["yaw_delta"] = 15
    elif name == "turn_left_45":
        cmd["yaw_delta"] = -45
    elif name == "turn_right_45":
        cmd["yaw_delta"] = 45
    elif name == "look_up_15":
        cmd["pitch_delta"] = 15
    elif name == "look_down_15":
        cmd["pitch_delta"] = -15
    elif name == "attack":
        cmd["attack"] = 1
    elif name == "mine":
        cmd["mine"] = 1
    elif name == "use":
        cmd["use"] = 1
    elif name == "place_block":
        cmd["place_block"] = 1
    elif name == "eat":
        cmd["eat"] = 1
    elif name.startswith("hotbar_"):
        slot = int(name.split("_")[1]) - 1
        cmd["hotbar"] = slot

    return cmd


def _noop_command() -> Dict[str, Any]:
    return {
        "forward": 0,
        "back": 0,
        "left": 0,
        "right": 0,
        "jump": 0,
        "sneak": 0,
        "sprint": 0,
        "attack": 0,
        "mine": 0,
        "use": 0,
        "place_block": 0,
        "eat": 0,
        "hotbar": None,
        "yaw_delta": 0,
        "pitch_delta": 0,
    }


def action_index_to_json(action_idx: int) -> str:
    """Return the JSON string to send to the JS bot."""
    return json.dumps(action_index_to_command(action_idx))


def get_action_space() -> gym.spaces.Discrete:
    return gym.spaces.Discrete(ACTION_COUNT)
