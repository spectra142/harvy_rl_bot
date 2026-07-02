"""Mocked environment tests."""

import base64
import json
import socket
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from env.minecraft_env import MinecraftEnv
from env.observation import VOXEL_SHAPE


def _make_voxel_b64():
    voxel = np.zeros(VOXEL_SHAPE, dtype=np.uint16)
    return base64.b64encode(voxel.tobytes()).decode("utf-8")


def _mock_obs_response():
    return {
        "obs": {
            "self": {
                "health": 20.0,
                "food": 20.0,
                "armor": 0,
                "position": [0.0, 64.0, 0.0],
                "yaw": 0.0,
                "pitch": 0.0,
                "held_item": "air",
                "inventory_counts": {},
            },
            "nearby_entities": [],
            "voxel_grid": _make_voxel_b64(),
            "environment": {
                "time_of_day": 6000,
                "can_see_sky": True,
                "in_water": False,
                "on_ground": True,
                "danger_level": 0.0,
            },
        },
        "reward_signal": {"alive_tick": 1},
        "terminated": False,
        "overridden": False,
    }


def test_reset_and_step_with_mock_socket():
    env = MinecraftEnv(host="localhost", port=9876)
    mock_socket = MagicMock()

    # First recv on reset returns mock obs; second recv on step returns mock obs then terminates.
    responses = [
        _mock_obs_response(),
        {**_mock_obs_response(), "terminated": True},
    ]

    def recv_side_effect(*args, **kwargs):
        payload = responses.pop(0)
        return (json.dumps(payload) + "\n").encode("utf-8")

    mock_socket.recv.side_effect = recv_side_effect

    with patch.object(env, "_connect"):
        env.sock = mock_socket
        obs, info = env.reset()
        assert obs["self_health"][0] == 20.0
        assert info["health"] == 20.0

        obs2, reward, terminated, truncated, info2 = env.step(
            env.action_space.sample()
        )
        assert terminated is True
        assert info2["termination_reason"] == "death"

    env.close()


def test_reset_raises_on_connection_failure():
    env = MinecraftEnv(host="invalid.invalid.invalid", port=9876)
    with pytest.raises((ConnectionError, OSError)):
        env.reset()


def test_recv_buffers_multiple_messages():
    env = MinecraftEnv(host="localhost", port=9876)
    mock_socket = MagicMock()

    # Two complete JSON lines arrive in a single recv() call.
    lines = [
        _mock_obs_response(),
        {**_mock_obs_response(), "terminated": True},
    ]
    payload = "\n".join(json.dumps(line) for line in lines) + "\n"
    mock_socket.recv.side_effect = [payload.encode("utf-8")]

    with patch.object(env, "_connect"):
        env.sock = mock_socket
        obs1, _ = env.reset()
        assert obs1["self_health"][0] == 20.0

        obs2, _reward, terminated, _trunc, _info2 = env.step(env.action_space.sample())
        assert terminated is True

    env.close()
