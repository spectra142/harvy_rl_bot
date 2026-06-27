"""Running mean/std observation normalizer."""

import numpy as np
import gymnasium as gym
from typing import Dict


class ObservationNormalizer:
    """Online observation normalization with running mean and std.

    Uses Welford's online algorithm to maintain running mean and variance
    statistics for each observation key. This stabilizes training by
    ensuring observations have zero mean and unit variance.

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
            self.mean[key] = np.zeros(space.shape, dtype=np.float64)
            self.var[key] = np.ones(space.shape, dtype=np.float64)

    def update(self, obs: Dict[str, np.ndarray]) -> None:
        """Update running statistics with a new observation.

        Uses Welford's online algorithm for numerically stable
        incremental variance computation.

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

        Args:
            obs: Raw observation dictionary.

        Returns:
            Normalized observation dictionary with clipped values.
        """
        normalized: Dict[str, np.ndarray] = {}
        for key, value in obs.items():
            if key not in self.mean:
                normalized[key] = value
                continue
            std = np.sqrt(self.var[key] / max(self.count, 1)) + self.epsilon
            normalized[key] = (value - self.mean[key]) / std
            # Clip to reasonable range to prevent outliers
            normalized[key] = np.clip(normalized[key], -10.0, 10.0).astype(np.float32)
        return normalized
