"""Tests for the observation normalizer's categorical handling and goal IDs."""

import numpy as np
import pytest

from env.observation import (
    GOAL_TO_ID,
    NUM_GOALS,
    build_observation,
    get_observation_space,
    goal_to_id,
)
from env.observation_normalizer import CATEGORICAL_KEYS, ObservationNormalizer


def test_all_canonical_goals_have_unique_ids():
    assert len(set(GOAL_TO_ID.values())) == NUM_GOALS
    assert NUM_GOALS <= 16  # goal embedding has 16 slots
    assert min(GOAL_TO_ID.values()) >= 0
    assert max(GOAL_TO_ID.values()) < 16


def test_autonomous_catalog_goals_all_mapped():
    from skills.autonomous_curriculum import AutonomousCurriculum

    for goal in AutonomousCurriculum.GOAL_CATALOG:
        assert goal in GOAL_TO_ID, f"autonomous goal '{goal}' missing from GOAL_TO_ID"


def test_goal_rewards_keys_all_mapped():
    from env.rewards import RewardCalculator

    for goal in RewardCalculator.GOAL_REWARDS:
        assert goal in GOAL_TO_ID, f"reward goal '{goal}' missing from GOAL_TO_ID"


def test_legacy_goal_aliases():
    assert goal_to_id("punch_wood") == GOAL_TO_ID["gather_logs"]
    assert goal_to_id("mine_stone") == GOAL_TO_ID["gather_stone"]
    assert goal_to_id("full_survival") == GOAL_TO_ID["survive"]
    assert goal_to_id("nonexistent_goal") == GOAL_TO_ID["survive"]


def test_normalizer_passes_categorical_keys_through():
    """goal_id and other embedding indices must NOT be z-scored."""
    space = get_observation_space()
    normalizer = ObservationNormalizer(space)

    for _ in range(50):
        obs = {key: np.zeros(space.spaces[key].shape, dtype=np.float32)
               for key in space.spaces}
        obs["goal_id"] = np.array([15.0], dtype=np.float32)
        obs["entity_type_ids"] = np.full((16,), 52.0, dtype=np.float32)
        normalizer.update(obs)
        out = normalizer.normalize(obs)

    # After 50 constant updates the running std collapses toward 0; a z-score
    # would explode or go negative. Categorical keys must pass through as-is.
    assert out["goal_id"][0] == pytest.approx(15.0)
    assert np.all(out["entity_type_ids"] == pytest.approx(52.0))
    # And a continuous key IS normalized (mean ~0 after updates).
    assert abs(out["self_health"][0]) < 1.0


def test_normalizer_never_produces_negative_ids():
    space = get_observation_space()
    normalizer = ObservationNormalizer(space)
    obs = {key: np.zeros(space.spaces[key].shape, dtype=np.float32)
           for key in space.spaces}
    obs["goal_id"] = np.array([3.0], dtype=np.float32)
    normalizer.update(obs)
    obs["goal_id"] = np.array([8.0], dtype=np.float32)  # goal changes
    out = normalizer.normalize(obs)
    assert out["goal_id"][0] == pytest.approx(8.0)  # untouched, still valid


def test_normalizer_state_roundtrip(tmp_path):
    space = get_observation_space()
    normalizer = ObservationNormalizer(space)
    obs = {key: np.ones(space.spaces[key].shape, dtype=np.float32)
           for key in space.spaces}
    for _ in range(5):
        normalizer.update(obs)
    path = tmp_path / "norm.pkl"
    normalizer.save(str(path))

    fresh = ObservationNormalizer(space)
    fresh.load(str(path))
    assert fresh.count == normalizer.count
    assert np.allclose(fresh.mean["self_health"], normalizer.mean["self_health"])


def test_categorical_keys_cover_embedding_inputs():
    # Every obs key consumed by nn.Embedding in networks.py must be excluded
    # from normalization.
    for key in (
        "goal_id",
        "entity_type_ids",
        "danger_block_types",
        "self_held_item_id",
        "voxel_grid",
        "voxel_tool_required",
    ):
        assert key in CATEGORICAL_KEYS
