"""Observation builder: converts raw bot JSON to a flat Gymnasium-compatible dict."""

import base64
import numpy as np
import gymnasium as gym
from typing import Any

# Block type ID mapping (subset of common blocks)
BLOCK_TYPE_TO_ID = {
    "air": 0,
    "stone": 1,
    "granite": 2,
    "diorite": 3,
    "andesite": 4,
    "grass_block": 5,
    "dirt": 6,
    "coarse_dirt": 7,
    "podzol": 8,
    "cobblestone": 9,
    "oak_planks": 10,
    "oak_log": 11,
    "oak_leaves": 12,
    "glass": 13,
    "lapis_ore": 14,
    "sandstone": 15,
    "sand": 16,
    "gravel": 17,
    "coal_ore": 18,
    "iron_ore": 19,
    "gold_ore": 20,
    "diamond_ore": 21,
    "redstone_ore": 22,
    "emerald_ore": 23,
    "oak_sapling": 24,
    "bedrock": 25,
    "water": 26,
    "lava": 27,
    "flowing_water": 28,
    "flowing_lava": 29,
    "torch": 30,
    "crafting_table": 31,
    "furnace": 32,
    "chest": 33,
    "ladder": 34,
    "oak_door": 35,
    "wheat": 36,
    "farmland": 37,
    "furnace_lit": 38,
    "oak_sign": 39,
    "oak_stairs": 40,
    "cobblestone_stairs": 41,
}

ENTITY_TYPE_TO_ID = {
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

ITEM_NAME_TO_ID = {
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
    "wooden_shovel": 14,
    "stone_shovel": 15,
    "bow": 16,
    "shield": 17,
    "torch": 18,
    "oak_log": 19,
    "cobblestone": 20,
    "iron_ingot": 21,
    "diamond": 22,
    "coal": 23,
    "stick": 24,
    "crafting_table": 25,
    "furnace": 26,
    "bread": 27,
    "cooked_beef": 28,
    "apple": 29,
    "arrow": 30,
    "oak_planks": 31,
}

ITEM_ID_TO_NAME = {int(item_id): name for name, item_id in ITEM_NAME_TO_ID.items()}

GOAL_TO_ID = {
    "basic_movement": 0,
    "punch_wood": 1,
    "craft_pickaxe": 2,
    "mine_stone": 3,
    "fight_passive": 4,
    "fight_hostile": 5,
    "build_shelter": 6,
    "survive_first_night": 7,
    "mine_iron": 8,
    "full_survival": 9,
    "pvp_combat": 10,
}

MAX_ENTITIES = 16
MAX_DANGER_BLOCKS = 8
MAX_VISITED_CHUNKS = 64
MAX_POIS = 16
MAX_EVENTS = 10
INVENTORY_SIZE = len(ITEM_NAME_TO_ID)

POI_TYPE_TO_ID = {
    "tree": 1,
    "ore_coal": 2,
    "ore_iron": 3,
    "ore_diamond": 4,
    "chest": 5,
    "crafting_table": 6,
    "furnace": 7,
    "mob_hostile": 8,
    "mob_passive": 9,
    "water_source": 10,
}

EVENT_TYPE_TO_ID = {
    "damage_taken": 1,
    "mob_attacked": 2,
    "block_broken": 3,
    "item_collected": 4,
    "food_eaten": 5,
}


def _encode_entities(entities: list) -> dict:
    """Encode nearby entities into fixed-size float arrays."""
    entities = entities[:MAX_ENTITIES]
    type_ids = [ENTITY_TYPE_TO_ID.get(e.get("type", ""), 0) for e in entities]
    distances = [e.get("distance", 0.0) for e in entities]
    healths = [e.get("health", 0.0) for e in entities]
    hostiles = [float(e.get("hostile", False)) for e in entities]

    while len(type_ids) < MAX_ENTITIES:
        type_ids.append(0)
        distances.append(0.0)
        healths.append(0.0)
        hostiles.append(0.0)

    return {
        "entity_type_ids": np.array(type_ids, dtype=np.float32),
        "entity_distances": np.array(distances, dtype=np.float32).reshape(
            MAX_ENTITIES, 1
        ),
        "entity_healths": np.array(healths, dtype=np.float32).reshape(MAX_ENTITIES, 1),
        "entity_hostiles": np.array(hostiles, dtype=np.float32).reshape(
            MAX_ENTITIES, 1
        ),
    }


def _encode_inventory(inventory_counts: dict) -> np.ndarray:
    """Encode inventory into a fixed-size count vector."""
    vec = np.zeros(INVENTORY_SIZE, dtype=np.float32)
    if not inventory_counts:
        return vec
    for name, count in inventory_counts.items():
        item_id = ITEM_NAME_TO_ID.get(name)
        if item_id is not None:
            vec[item_id] += float(count)
    return vec


def _encode_danger_blocks(blocks: list) -> dict:
    """Encode nearby dangerous/important blocks into fixed-size arrays."""
    blocks = blocks[:MAX_DANGER_BLOCKS]
    type_ids = []
    positions = []
    distances = []
    for b in blocks:
        type_ids.append(BLOCK_TYPE_TO_ID.get(b.get("type", ""), 0))
        pos = b.get("position", [0, 0, 0])
        positions.append([float(pos[0]), float(pos[1]), float(pos[2])])
        distances.append(float(abs(pos[0]) + abs(pos[1]) + abs(pos[2])))

    while len(type_ids) < MAX_DANGER_BLOCKS:
        type_ids.append(0)
        positions.append([0.0, 0.0, 0.0])
        distances.append(0.0)

    return {
        "danger_block_types": np.array(type_ids, dtype=np.float32),
        "danger_block_positions": np.array(positions, dtype=np.float32),
        "danger_block_distances": np.array(distances, dtype=np.float32).reshape(
            MAX_DANGER_BLOCKS, 1
        ),
    }


def _encode_visited_chunks(chunks: list) -> np.ndarray:
    """Encode last visited chunks as a fixed-length vector."""
    chunks = chunks[-MAX_VISITED_CHUNKS:]
    vec = []
    for c in chunks:
        vec.append(float(c.get("cx", 0)))
        vec.append(float(c.get("cz", 0)))
        # Normalize timestamp to roughly [-1, 1] range (seconds since epoch / 1e9)
        vec.append(float(c.get("timestamp", 0)) / 1e9)
    while len(vec) < MAX_VISITED_CHUNKS * 3:
        vec.append(0.0)
    return np.array(vec, dtype=np.float32)


def _encode_pois(pois: list) -> dict:
    """Encode nearby points of interest into fixed-size arrays."""
    pois = pois[:MAX_POIS]
    type_ids = [POI_TYPE_TO_ID.get(p.get("type", ""), 0) for p in pois]
    distances = [p.get("distance", 0.0) for p in pois]
    positions = []
    for p in pois:
        pos = p.get("position", [0, 0, 0])
        positions.append([float(pos[0]), float(pos[1]), float(pos[2])])

    while len(type_ids) < MAX_POIS:
        type_ids.append(0)
        distances.append(0.0)
        positions.append([0.0, 0.0, 0.0])

    return {
        "poi_type_ids": np.array(type_ids, dtype=np.float32),
        "poi_distances": np.array(distances, dtype=np.float32).reshape(MAX_POIS, 1),
        "poi_positions": np.array(positions, dtype=np.float32),
    }


def _encode_recent_events(events: list) -> dict:
    """Encode recent event stream into fixed-size arrays."""
    events = events[-MAX_EVENTS:]
    type_ids = [EVENT_TYPE_TO_ID.get(e.get("type", ""), 0) for e in events]
    data = [float(e.get("data", 0)) for e in events]

    while len(type_ids) < MAX_EVENTS:
        type_ids.append(0)
        data.append(0.0)

    return {
        "event_type_ids": np.array(type_ids, dtype=np.float32),
        "event_data": np.array(data, dtype=np.float32),
    }


def _decode_voxel_base64(b64: str, dtype: np.dtype, default_shape: tuple) -> np.ndarray:
    """Decode a base64-encoded voxel array or return zeros."""
    if not b64:
        return np.zeros(default_shape, dtype=np.float32)
    try:
        arr = np.frombuffer(base64.b64decode(b64), dtype=dtype).reshape(default_shape)
        return arr.astype(np.float32)
    except Exception:
        return np.zeros(default_shape, dtype=np.float32)


def build_observation(raw_obs: dict) -> dict:
    """Convert raw bot observation to a flat Gymnasium observation dict."""
    self_data = raw_obs.get("self", {})
    held_item = self_data.get("held_item", "")

    entities = raw_obs.get("nearby_entities", [])
    entity_obs = _encode_entities(entities)

    inventory_obs = _encode_inventory(self_data.get("inventory_counts", {}))

    danger_blocks = raw_obs.get("nearby_blocks", [])
    danger_obs = _encode_danger_blocks(danger_blocks)

    # Voxel grid (base64 -> numpy)
    # Stored as float32 so Stable Baselines3 doesn't treat it as a 2D image
    # and auto-transpose the channels. The CNN normalizes by /255 internally.
    voxel_b64 = raw_obs.get("voxel_grid", "")
    if voxel_b64:
        voxel_bytes = base64.b64decode(voxel_b64)
        voxel_grid = (
            np.frombuffer(voxel_bytes, dtype=np.uint8)
            .reshape((11, 11, 7))
            .astype(np.float32)
        )
    else:
        voxel_grid = np.zeros((11, 11, 7), dtype=np.float32)

    voxel_hardness = _decode_voxel_base64(
        raw_obs.get("voxel_hardness", ""), np.float32, (11, 11, 7)
    )
    voxel_tool_required = _decode_voxel_base64(
        raw_obs.get("voxel_tool_required", ""), np.uint8, (11, 11, 7)
    )

    env_data = raw_obs.get("environment", {})
    goal_str = raw_obs.get("goal", "survive_first_night")

    visited_obs = _encode_visited_chunks(raw_obs.get("visited_chunks", []))
    poi_obs = _encode_pois(raw_obs.get("nearby_pois", []))
    event_obs = _encode_recent_events(raw_obs.get("recent_events", []))

    obs = {
        "self_health": np.array([self_data.get("health", 20.0)], dtype=np.float32),
        "self_food": np.array([self_data.get("food", 20.0)], dtype=np.float32),
        "self_armor": np.array([self_data.get("armor", 0.0)], dtype=np.float32),
        "self_position": np.array(
            self_data.get("position", [0.0, 0.0, 0.0]), dtype=np.float32
        ),
        "self_yaw": np.array([self_data.get("yaw", 0.0)], dtype=np.float32),
        "self_pitch": np.array([self_data.get("pitch", 0.0)], dtype=np.float32),
        "self_held_item_id": np.array(
            [ITEM_NAME_TO_ID.get(held_item, 0)], dtype=np.float32
        ),
        "inventory": inventory_obs,
        **entity_obs,
        **danger_obs,
        **poi_obs,
        "voxel_grid": voxel_grid,
        "voxel_hardness": voxel_hardness,
        "voxel_tool_required": voxel_tool_required,
        "visited_chunks": visited_obs,
        **event_obs,
        "env_time_of_day": np.array(
            [env_data.get("time_of_day", 6000.0)], dtype=np.float32
        ),
        "env_can_see_sky": np.array(
            [float(env_data.get("can_see_sky", True))], dtype=np.float32
        ),
        "env_in_water": np.array(
            [float(env_data.get("in_water", False))], dtype=np.float32
        ),
        "env_on_ground": np.array(
            [float(env_data.get("on_ground", True))], dtype=np.float32
        ),
        "env_danger_level": np.array(
            [env_data.get("danger_level", 0.0)], dtype=np.float32
        ),
        "goal_id": np.array([GOAL_TO_ID.get(goal_str, 7)], dtype=np.float32),
    }
    return obs


def get_observation_space() -> gym.spaces.Dict:
    """Return the flattened Gymnasium observation space.

    Stable Baselines3's MultiInputPolicy does not support nested Dict/Tuple
    spaces, so every modality is exposed as a top-level Box.
    """
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
            "self_held_item_id": gym.spaces.Box(0.0, 255.0, (1,), dtype=np.float32),
            "inventory": gym.spaces.Box(
                0.0, 2304.0, (INVENTORY_SIZE,), dtype=np.float32
            ),
            "entity_type_ids": gym.spaces.Box(
                0.0, 127.0, (MAX_ENTITIES,), dtype=np.float32
            ),
            "entity_distances": gym.spaces.Box(
                0.0, 100.0, (MAX_ENTITIES, 1), dtype=np.float32
            ),
            "entity_healths": gym.spaces.Box(
                0.0, 100.0, (MAX_ENTITIES, 1), dtype=np.float32
            ),
            "entity_hostiles": gym.spaces.Box(
                0.0, 1.0, (MAX_ENTITIES, 1), dtype=np.float32
            ),
            "danger_block_types": gym.spaces.Box(
                0.0, 127.0, (MAX_DANGER_BLOCKS,), dtype=np.float32
            ),
            "danger_block_positions": gym.spaces.Box(
                -10.0, 10.0, (MAX_DANGER_BLOCKS, 3), dtype=np.float32
            ),
            "danger_block_distances": gym.spaces.Box(
                0.0, 30.0, (MAX_DANGER_BLOCKS, 1), dtype=np.float32
            ),
            "poi_type_ids": gym.spaces.Box(0.0, 15.0, (MAX_POIS,), dtype=np.float32),
            "poi_distances": gym.spaces.Box(0.0, 64.0, (MAX_POIS, 1), dtype=np.float32),
            "poi_positions": gym.spaces.Box(
                -32.0, 32.0, (MAX_POIS, 3), dtype=np.float32
            ),
            "voxel_grid": gym.spaces.Box(0.0, 255.0, (11, 11, 7), dtype=np.float32),
            "voxel_hardness": gym.spaces.Box(0.0, 100.0, (11, 11, 7), dtype=np.float32),
            "voxel_tool_required": gym.spaces.Box(
                0.0, 7.0, (11, 11, 7), dtype=np.float32
            ),
            "visited_chunks": gym.spaces.Box(
                -3000000.0, 3000000.0, (MAX_VISITED_CHUNKS * 3,), dtype=np.float32
            ),
            "event_type_ids": gym.spaces.Box(
                0.0, 10.0, (MAX_EVENTS,), dtype=np.float32
            ),
            "event_data": gym.spaces.Box(0.0, 1000.0, (MAX_EVENTS,), dtype=np.float32),
            "env_time_of_day": gym.spaces.Box(0.0, 24000.0, (1,), dtype=np.float32),
            "env_can_see_sky": gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            "env_in_water": gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            "env_on_ground": gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            "env_danger_level": gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            "goal_id": gym.spaces.Box(0.0, 15.0, (1,), dtype=np.float32),
        }
    )
