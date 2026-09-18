"""Running mean/std observation normalizer."""

import pickle

import numpy as np
import gymnasium as gym
from typing import Dict

# Keys that hold categorical / embedding indices or raw CNN inputs.
# These must NEVER be z-scored: the network feeds them straight into
# nn.Embedding (negative or shifted indices crash training) or divides
# them by 255 internally (voxel grids).
CATEGORICAL_KEYS = frozenset(
    {
        "goal_id",
        "entity_type_ids",
        "danger_block_types",
        "poi_type_ids",
        "event_type_ids",
        "self_held_item_id",
        "voxel_grid",
        "voxel_tool_required",
    }
)


class ObservationNormalizer:
    """Online observation normalization with running mean and std.

    Uses Welford's online algorithm to maintain running mean and variance
    statistics for each continuous observation key. Categorical keys
    (embedding indices, raw voxel IDs) are passed through untouched.

    Attributes:
        epsilon: Small constant to avoid division by zero.
        count: Number of observations seen so far.
        mean: Running mean for each observation key.
        var: Running variance for each observation key.
    """

    def __init__(self, obs_space: gym.spaces.Dict, epsilon: float = 1e-8):
        """Initialize the normalizer with observation space shape.

        Args:
            obs_space: The Dict observation space defining shapes.
            epsilon: Small constant to avoid division by zero.
        """
        self.epsilon = epsilon
        self.count = 0
        # Initialize running mean and var for each key
        self.mean: Dict[str, np.ndarray] = {}
        self.var: Dict[str, np.ndarray] = {}
        for key, space in obs_space.spaces.items():
            if key in CATEGORICAL_KEYS:
                continue
            self.mean[key] = np.zeros(space.shape, dtype=np.float64)
            self.var[key] = np.ones(space.shape, dtype=np.float64)

    def update(self, obs: Dict[str, np.ndarray]) -> None:
        """Update running statistics with a new observation.

        Uses Welford's online algorithm for numerically stable
        incremental variance computation. Categorical keys are skipped.

        Args:
            obs: Dictionary of observation arrays.
        """
        self.count += 1
        for key, value in obs.items():
            if key not in self.mean:
                continue
            value = np.asarray(value, dtype=np.float64)
            delta = value - self.mean[key]
            self.mean[key] += delta / self.count
            delta2 = value - self.mean[key]
            self.var[key] += delta * delta2

    def normalize(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Normalize observation using running statistics.

        Categorical keys pass through unmodified (as float32); continuous
        keys are z-scored and clipped to [-10, 10].

        Args:
            obs: Raw observation dictionary.

        Returns:
            Normalized observation dictionary with clipped values.
        """
        normalized: Dict[str, np.ndarray] = {}
        for key, value in obs.items():
            if key not in self.mean:
                # Categorical or unknown key -- pass through untouched.
                normalized[key] = np.asarray(value, dtype=np.float32)
                continue
            std = np.sqrt(self.var[key] / max(self.count, 1)) + self.epsilon
            normalized[key] = (value - self.mean[key]) / std
            # Clip to reasonable range to prevent outliers
            normalized[key] = np.clip(normalized[key], -10.0, 10.0).astype(np.float32)
        return normalized

    def get_state(self) -> dict:
        """Return serializable running statistics."""
        return {"count": self.count, "mean": self.mean, "var": self.var}

    def set_state(self, state: dict) -> None:
        """Restore running statistics from :meth:`get_state` output."""
        self.count = int(state.get("count", 0))
        for key, value in state.get("mean", {}).items():
            if key in self.mean:
                self.mean[key] = np.asarray(value, dtype=np.float64)
        for key, value in state.get("var", {}).items():
            if key in self.var:
                self.var[key] = np.asarray(value, dtype=np.float64)

    def save(self, path: str) -> None:
        """Persist running statistics to disk."""
        with open(path, "wb") as f:
            pickle.dump(self.get_state(), f)

    def load(self, path: str) -> None:
        """Restore running statistics from disk."""
        with open(path, "rb") as f:
            self.set_state(pickle.load(f))
