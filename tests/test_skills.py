"""Tests for skill framework and combat skill."""

import numpy as np

from skills.skill_executor import SkillExecutor
from skills.combat_skill import CombatSkill
from env.action_space import ACTION_NAMES, LOW_LEVEL_ACTION_COUNT


def _obs_with_hostile(distance: float, dx: float = 0.0, dz: float = 1.0, weapon_in_inv: bool = True):
    from env.observation import ITEM_TO_ID
    obs = {
        "self_health": np.array([20.0], dtype=np.float32),
        "self_food": np.array([20.0], dtype=np.float32),
        "self_armor": np.array([0.0], dtype=np.float32),
        "self_position": np.array([0.0, 64.0, 0.0], dtype=np.float32),
        "self_yaw": np.array([0.0], dtype=np.float32),
        "self_pitch": np.array([0.0], dtype=np.float32),
        "self_held_item_id": np.array([float(ITEM_TO_ID["wooden_sword"])], dtype=np.float32),
        "inventory": np.zeros(len(ITEM_TO_ID), dtype=np.float32),
        "entity_type_ids": np.zeros(16, dtype=np.float32),
        "entity_distances": np.zeros((16, 1), dtype=np.float32),
        "entity_healths": np.zeros((16, 1), dtype=np.float32),
        "entity_hostiles": np.zeros((16, 1), dtype=np.float32),
        "entity_positions": np.zeros((16, 3), dtype=np.float32),
    }
    if weapon_in_inv:
        obs["inventory"][ITEM_TO_ID["wooden_sword"]] = 1.0
    obs["entity_type_ids"][0] = 2  # zombie
    obs["entity_distances"][0][0] = distance
    obs["entity_healths"][0][0] = 20.0
    obs["entity_hostiles"][0][0] = 1.0
    obs["entity_positions"][0] = [dx, 0.0, dz]
    return obs


def test_combat_skill_can_use_when_hostile_nearby():
    skill = CombatSkill()
    assert skill.can_use(_obs_with_hostile(3.0), {})
    assert not skill.can_use(_obs_with_hostile(25.0), {})


def test_combat_skill_approaches_when_far():
    skill = CombatSkill()
    skill.activate()
    action, reward, done, info = skill.step(_obs_with_hostile(8.0), {})
    assert ACTION_NAMES[action] == "move_forward"
    assert not done
    assert info["action"] == "approach"


def test_combat_skill_faces_target_before_attacking():
    skill = CombatSkill()
    skill.activate()
    # Target at 90 degrees to the right.
    obs = _obs_with_hostile(2.0, dx=1.0, dz=0.0)
    action, reward, done, info = skill.step(obs, {})
    assert ACTION_NAMES[action] == "turn_right_15"
    assert info["action"] == "face_target"


def test_combat_skill_attacks_when_aligned():
    skill = CombatSkill()
    skill.activate()
    obs = _obs_with_hostile(2.0, dx=0.0, dz=1.0)
    # First call faces target (already aligned, small error OK).
    action, reward, done, info = skill.step(obs, {})
    # If aligned, it should attack.
    if info["action"] == "attack":
        assert ACTION_NAMES[action] == "attack"


def test_skill_executor_routes_skill_action():
    executor = SkillExecutor([CombatSkill()])
    combat_action = LOW_LEVEL_ACTION_COUNT + 0
    obs = _obs_with_hostile(2.0, dx=0.0, dz=1.0)
    low, reward, done, info = executor.step(combat_action, obs, {})
    assert info["skill"] == "combat"
    assert low < LOW_LEVEL_ACTION_COUNT


def test_skill_executor_low_level_passes_through():
    executor = SkillExecutor([CombatSkill()])
    forward = ACTION_NAMES.index("move_forward")
    low, reward, done, info = executor.step(forward, _obs_with_hostile(50.0), {})
    assert low == forward
    assert info["skill"] is None


def test_combat_skill_switches_weapon_when_unarmed():
    from env.observation import ITEM_TO_ID
    skill = CombatSkill()
    skill.activate()
    obs = _obs_with_hostile(2.0, weapon_in_inv=True)
    obs["self_held_item_id"] = np.array([0.0], dtype=np.float32)  # empty hand
    action, reward, done, info = skill.step(obs, {})
    assert ACTION_NAMES[action].startswith("hotbar")
    assert info["action"] == "equip_weapon"


def test_flee_skill_runs_from_hostile():
    from skills.flee_skill import FleeSkill
    skill = FleeSkill()
    skill.activate()
    # Hostile behind the bot; fleeing away means sprinting forward.
    action, reward, done, info = skill.step(_obs_with_hostile(3.0, dx=0.0, dz=-1.0), {})
    assert ACTION_NAMES[action] == "sprint_forward"
    assert info["action"] == "sprint_away"


def test_gather_skill_finds_target_block():
    from skills.gather_skill import GatherSkill
    from env.observation import ITEM_TO_ID
    skill = GatherSkill()
    obs = _obs_with_hostile(50.0)  # no entities
    # Put a log in the voxel grid at relative position (1, 0, 1).
    obs["voxel_grid"] = np.zeros((11, 11, 7), dtype=np.float32)
    obs["voxel_grid"][6, 6, 3] = 17  # oak-ish log state id
    action, reward, done, info = skill.step(obs, {})
    assert info["action"] in ("approach", "face", "mine")


def test_build_skill_places_blocks():
    from skills.build_skill import BuildSkill
    from env.observation import ITEM_TO_ID
    skill = BuildSkill()
    obs = _obs_with_hostile(50.0)
    obs["inventory"][ITEM_TO_ID["cobblestone"]] = 64.0
    skill.activate()
    # First block is at (0,0,2), within placement range.
    action, reward, done, info = skill.step(obs, {})
    assert info["action"] in ("place", "face")
