"""Tests for the Gymnasium environment and observation helpers."""

import numpy as np
import pytest

from env.minecraft_env import MinecraftEnv
from env.observation import build_observation, get_observation_space
from env.frame_stack import FrameStack
from env.observation_normalizer import ObservationNormalizer


def test_observation_space_contains_build_observation():
    """A mocked raw observation must produce arrays that fit the declared space."""
    raw_obs = {
        "self": {
            "health": 20.0,
            "food": 20.0,
            "armor": 0,
            "position": [0.0, 64.0, 0.0],
            "yaw": 0.0,
            "pitch": 0.0,
            "held_item": "wooden_pickaxe",
            "inventory_counts": {"oak_log": 5, "stick": 2},
        },
        "nearby_entities": [
            {"type": "zombie", "distance": 3.0, "health": 20, "hostile": True}
        ],
        "nearby_blocks": [{"type": "lava", "position": [1, -2, 3]}],
        "nearby_pois": [{"type": "tree", "distance": 5.0, "position": [2, 0, 1]}],
        "voxel_grid": "",
        "voxel_hardness": "",
        "voxel_tool_required": "",
        "visited_chunks": [{"cx": 0, "cz": 0, "timestamp": 1_700_000_000}],
        "recent_events": [
            {"type": "damage_taken", "timestamp": 1_700_000_000, "data": 2}
        ],
        "environment": {
            "time_of_day": 6000,
            "can_see_sky": True,
            "in_water": False,
            "on_ground": True,
            "danger_level": 0.2,
        },
        "goal": "survive_first_night",
        "reward_signal": {},
        "skill_status": {},
    }
    obs = build_observation(raw_obs)
    space = get_observation_space()
    assert space.contains(obs)


def test_frame_stack_only_stacks_voxel():
    space = get_observation_space()
    stacker = FrameStack(space, n_frames=4)
    obs = {
        "voxel_grid": np.zeros((11, 11, 7), dtype=np.float32),
        "self_health": np.array([20.0], dtype=np.float32),
        "inventory": np.zeros((32,), dtype=np.float32),
    }
    stacked = stacker.reset(obs)
    assert stacked["voxel_grid"].shape == (4, 11, 11, 7)
    assert stacked["self_health"].shape == (1,)
    assert stacked["inventory"].shape == (32,)

    obs2 = {
        "voxel_grid": np.ones((11, 11, 7), dtype=np.float32),
        "self_health": np.array([18.0], dtype=np.float32),
        "inventory": np.zeros((32,), dtype=np.float32),
    }
    stacked2 = stacker.step(obs2)
    assert stacked2["voxel_grid"].shape == (4, 11, 11, 7)
    assert np.allclose(stacked2["voxel_grid"][-1], 1.0)
    assert np.allclose(stacked2["self_health"], [18.0])


def test_normalizer_handles_stacked_voxel():
    space = get_observation_space()
    space.spaces["voxel_grid"] = type(space.spaces["voxel_grid"])(
        0.0, 255.0, (4, 11, 11, 7), dtype=np.float32
    )
    normalizer = ObservationNormalizer(space)
    obs = {
        "voxel_grid": np.ones((4, 11, 11, 7), dtype=np.float32) * 128,
        "self_health": np.array([20.0], dtype=np.float32),
    }
    normalizer.update(obs)
    norm = normalizer.normalize(obs)
    assert norm["voxel_grid"].shape == (4, 11, 11, 7)


def test_env_reset_and_step_without_server():
    """The environment must gracefully fall back to dummy observations."""
    env = MinecraftEnv(host="invalid", port=9876, frame_stack=4)
    obs, info = env.reset()
    assert obs["voxel_grid"].shape == (4, 11, 11, 7)
    assert info["goal"] == "survive_first_night"
    obs2, reward, terminated, truncated, info2 = env.step(0)
    assert obs2["voxel_grid"].shape == (4, 11, 11, 7)  # connection-lost dummy path
    assert terminated is True
    env.close()
