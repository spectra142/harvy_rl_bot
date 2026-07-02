"""Tests for observation encoding."""

import base64
import numpy as np

from env.observation import build_observation, get_observation_space, VOXEL_SHAPE


def _mock_raw_obs():
    voxel = np.zeros(VOXEL_SHAPE, dtype=np.uint16)
    voxel[5, 5, 3] = 1  # state id 1 at (x=5, z=5, y=3)
    voxel_b64 = base64.b64encode(voxel.tobytes()).decode("utf-8")
    return {
        "obs": {
            "self": {
                "health": 18.0,
                "food": 19.0,
                "armor": 0,
                "position": [10.5, 64.0, -3.2],
                "yaw": 45.0,
                "pitch": -10.0,
                "held_item": "wooden_sword",
                "inventory_counts": {"oak_log": 5, "cobblestone": 12},
            },
            "nearby_entities": [
                {"type": "zombie", "distance": 3.0, "health": 20.0, "hostile": True, "relative_position": [1.5, 0.0, 2.5]},
                {"type": "cow", "distance": 8.0, "health": 10.0, "hostile": False, "relative_position": [-5.0, 0.0, 6.0]},
            ],
            "voxel_grid": voxel_b64,
            "environment": {
                "time_of_day": 12000,
                "can_see_sky": True,
                "in_water": False,
                "on_ground": True,
                "danger_level": 0.5,
            },
        },
        "reward_signal": {},
        "terminated": False,
        "overridden": False,
    }


def test_build_observation_fits_space():
    raw = _mock_raw_obs()
    obs = build_observation(raw)
    space = get_observation_space()
    assert space.contains(obs)


def test_voxel_grid_decoded():
    raw = _mock_raw_obs()
    obs = build_observation(raw)
    assert obs["voxel_grid"].shape == VOXEL_SHAPE
    assert obs["voxel_grid"][5, 5, 3] == 1.0


def test_voxel_axis_ordering():
    """Ensure JS (X,Z,Y) flat order matches Python (X,Z,Y) reshape."""
    voxel = np.zeros(VOXEL_SHAPE, dtype=np.uint16)
    voxel[2, 3, 4] = 42
    voxel[7, 8, 1] = 99
    raw = {"obs": {"self": _mock_raw_obs()["obs"]["self"], "nearby_entities": [], "voxel_grid": base64.b64encode(voxel.tobytes()).decode("utf-8"), "environment": _mock_raw_obs()["obs"]["environment"]}, "reward_signal": {}, "terminated": False}
    obs = build_observation(raw)
    assert obs["voxel_grid"][2, 3, 4] == 42.0
    assert obs["voxel_grid"][7, 8, 1] == 99.0


def test_entities_padded():
    raw = _mock_raw_obs()
    obs = build_observation(raw)
    assert obs["entity_type_ids"].shape == (16,)
    assert obs["entity_distances"].shape == (16, 1)
    assert obs["entity_positions"].shape == (16, 3)
    assert obs["entity_type_ids"][0] == 2.0  # zombie
    assert obs["entity_type_ids"][1] == 20.0  # cow
    assert obs["entity_type_ids"][2] == 0.0  # padded
    assert obs["entity_positions"][0][0] == 1.5
    assert obs["entity_positions"][0][2] == 2.5


def test_inventory_encoded():
    raw = _mock_raw_obs()
    obs = build_observation(raw)
    from env.observation import ITEM_TO_ID
    assert obs["inventory"][ITEM_TO_ID["oak_log"]] == 5.0
    assert obs["inventory"][ITEM_TO_ID["cobblestone"]] == 12.0
