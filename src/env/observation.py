"""Observation builder: raw JS payload -> Gymnasium-compatible numpy dict."""

import base64
import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional

try:
    import minecraft_data
except ImportError:
    minecraft_data = None

# Capacities.
MAX_ENTITIES = 16
VOXEL_SHAPE = (11, 11, 7)
ENTITY_VOCAB_SIZE = 128

# Stable IDs for common items. Unknown items map to 0.
ITEM_TO_ID = {
    "air": 0,
    "wooden_pickaxe": 1,
    "stone_pickaxe": 2,
    "iron_pickaxe": 3,
    "diamond_pickaxe": 4,
    "golden_pickaxe": 5,
    "wooden_axe": 6,
    "stone_axe": 7,
    "iron_axe": 8,
    "diamond_axe": 9,
    "wooden_sword": 10,
    "stone_sword": 11,
    "iron_sword": 12,
    "diamond_sword": 13,
    "bow": 14,
    "shield": 15,
    "torch": 16,
    "oak_log": 17,
    "birch_log": 18,
    "spruce_log": 19,
    "jungle_log": 20,
    "acacia_log": 21,
    "dark_oak_log": 22,
    "cobblestone": 23,
    "stone": 24,
    "iron_ingot": 25,
    "diamond": 26,
    "coal": 27,
    "stick": 28,
    "crafting_table": 29,
    "furnace": 30,
    "bread": 31,
    "cooked_beef": 32,
    "cooked_porkchop": 33,
    "cooked_chicken": 34,
    "apple": 35,
    "arrow": 36,
    "oak_planks": 37,
    "dirt": 38,
    "sand": 39,
    "gravel": 40,
    "water_bucket": 41,
}

ENTITY_TO_ID = {
    "player": 1,
    "zombie": 2,
    "skeleton": 3,
    "creeper": 4,
    "spider": 5,
    "enderman": 6,
    "witch": 7,
    "slime": 8,
    "drowned": 9,
    "husk": 10,
    "pillager": 11,
    "vindicator": 12,
    "cow": 20,
    "pig": 21,
    "sheep": 22,
    "chicken": 23,
    "rabbit": 24,
    "horse": 25,
    "wolf": 26,
    "villager": 27,
    "item": 50,
    "arrow": 51,
    "experience_orb": 52,
}


def _encode_entities(entities: list) -> Dict[str, np.ndarray]:
    entities = entities[:MAX_ENTITIES]
    type_ids = [ENTITY_TO_ID.get(e.get("type", ""), 0) for e in entities]
    distances = [e.get("distance", 0.0) for e in entities]
    healths = [e.get("health", 0.0) for e in entities]
    hostiles = [float(e.get("hostile", False)) for e in entities]
    positions = [e.get("relative_position", [0.0, 0.0, 0.0]) for e in entities]

    while len(type_ids) < MAX_ENTITIES:
        type_ids.append(0)
        distances.append(0.0)
        healths.append(0.0)
        hostiles.append(0.0)
        positions.append([0.0, 0.0, 0.0])

    return {
        "entity_type_ids": np.array(type_ids, dtype=np.float32),
        "entity_distances": np.array(distances, dtype=np.float32).reshape(
            MAX_ENTITIES, 1
        ),
        "entity_healths": np.array(healths, dtype=np.float32).reshape(MAX_ENTITIES, 1),
        "entity_hostiles": np.array(hostiles, dtype=np.float32).reshape(
            MAX_ENTITIES, 1
        ),
        "entity_positions": np.array(positions, dtype=np.float32),
    }


def _encode_inventory(inventory_counts: Dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(ITEM_TO_ID), dtype=np.float32)
    if not inventory_counts:
        return vec
    for name, count in inventory_counts.items():
        idx = ITEM_TO_ID.get(name)
        if idx is not None:
            vec[idx] += float(count)
    return vec


def _decode_voxel_grid(b64: str) -> np.ndarray:
    if not b64:
        return np.zeros(VOXEL_SHAPE, dtype=np.float32)
    try:
        raw = base64.b64decode(b64)
        # JS sends Uint16 state IDs in (X, Z, Y) order.
        grid = np.frombuffer(raw, dtype=np.uint16).reshape(VOXEL_SHAPE)
        return grid.astype(np.float32)
    except Exception:
        return np.zeros(VOXEL_SHAPE, dtype=np.float32)


def build_observation(raw: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
    """Convert raw JS payload to numpy observation dict."""
    obs = raw.get("obs")
    if obs is None:
        return None

    self_data = obs.get("self", {})
    entities = obs.get("nearby_entities", [])
    voxel_b64 = obs.get("voxel_grid", "")
    env = obs.get("environment", {})

    entity_obs = _encode_entities(entities)
    inventory = _encode_inventory(self_data.get("inventory_counts", {}))
    voxel_grid = _decode_voxel_grid(voxel_b64)

    return {
        "self_health": np.array([self_data.get("health", 20.0)], dtype=np.float32),
        "self_food": np.array([self_data.get("food", 20.0)], dtype=np.float32),
        "self_armor": np.array([self_data.get("armor", 0.0)], dtype=np.float32),
        "self_position": np.array(
            self_data.get("position", [0.0, 64.0, 0.0]), dtype=np.float32
        ),
        "self_yaw": np.array([self_data.get("yaw", 0.0)], dtype=np.float32),
        "self_pitch": np.array([self_data.get("pitch", 0.0)], dtype=np.float32),
        "self_held_item_id": np.array(
            [ITEM_TO_ID.get(self_data.get("held_item", "air"), 0)], dtype=np.float32
        ),
        "inventory": inventory,
        **entity_obs,
        "voxel_grid": voxel_grid,
        "env_time_of_day": np.array([env.get("time_of_day", 6000.0)], dtype=np.float32),
        "env_can_see_sky": np.array(
            [float(env.get("can_see_sky", True))], dtype=np.float32
        ),
        "env_in_water": np.array(
            [float(env.get("in_water", False))], dtype=np.float32
        ),
        "env_on_ground": np.array(
            [float(env.get("on_ground", True))], dtype=np.float32
        ),
        "env_danger_level": np.array(
            [env.get("danger_level", 0.0)], dtype=np.float32
        ),
    }


def get_observation_space() -> gym.spaces.Dict:
    return gym.spaces.Dict(
        {
            "self_health": gym.spaces.Box(0.0, 20.0, (1,), dtype=np.float32),
            "self_food": gym.spaces.Box(0.0, 20.0, (1,), dtype=np.float32),
            "self_armor": gym.spaces.Box(0.0, 20.0, (1,), dtype=np.float32),
            "self_position": gym.spaces.Box(
                -30000000.0, 30000000.0, (3,), dtype=np.float32
            ),
            "self_yaw": gym.spaces.Box(-180.0, 180.0, (1,), dtype=np.float32),
            "self_pitch": gym.spaces.Box(-90.0, 90.0, (1,), dtype=np.float32),
            "self_held_item_id": gym.spaces.Box(
                0.0, float(len(ITEM_TO_ID) - 1), (1,), dtype=np.float32
            ),
            "inventory": gym.spaces.Box(
                0.0, 2304.0, (len(ITEM_TO_ID),), dtype=np.float32
            ),
            "entity_type_ids": gym.spaces.Box(
                0.0, float(ENTITY_VOCAB_SIZE - 1), (MAX_ENTITIES,), dtype=np.float32
            ),
            "entity_distances": gym.spaces.Box(
                0.0, 64.0, (MAX_ENTITIES, 1), dtype=np.float32
            ),
            "entity_healths": gym.spaces.Box(
                0.0, 100.0, (MAX_ENTITIES, 1), dtype=np.float32
            ),
            "entity_hostiles": gym.spaces.Box(
                0.0, 1.0, (MAX_ENTITIES, 1), dtype=np.float32
            ),
            "entity_positions": gym.spaces.Box(
                -64.0, 64.0, (MAX_ENTITIES, 3), dtype=np.float32
            ),
            "voxel_grid": gym.spaces.Box(
                0.0, 255.0, VOXEL_SHAPE, dtype=np.float32
            ),
            "env_time_of_day": gym.spaces.Box(0.0, 24000.0, (1,), dtype=np.float32),
            "env_can_see_sky": gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            "env_in_water": gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            "env_on_ground": gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            "env_danger_level": gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
        }
    )
