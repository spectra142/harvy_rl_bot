"""Random Network Distillation (RND) for intrinsic curiosity rewards.

RND uses two networks:
- Target network: fixed random network that processes observations
- Predictor network: trained to predict the target network's output

The prediction error serves as an intrinsic reward -- the agent is rewarded
for visiting states it cannot yet predict (i.e., novel states).

References:
    Burda et al., "Exploration by Random Network Distillation", ICLR 2019.
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class RNDTargetNetwork(nn.Module):
    """Fixed random target network.

    Processes voxel-grid observations through a small 3-D CNN and produces
    a fixed-dimension feature vector.  All parameters are frozen after
    initialization so the network acts as a stationary, random feature
    extractor.
    """

    def __init__(self, input_shape: tuple = (11, 11, 7), output_dim: int = 128):
        """Initialize the target network.

        Args:
            input_shape: Spatial shape of the voxel grid (X, Y, Z).
            output_dim: Dimensionality of the output feature vector.
        """
        super().__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim

        # 3-D CNN backbone for voxel grid
        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        # Dynamically compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            flat_size = int(self.conv(dummy).shape[1])

        self.fc = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim),
        )

        # Freeze all parameters -- target network never trains
        for param in self.parameters():
            param.requires_grad = False

        logger.debug(
            "RNDTargetNetwork initialized: shape=%s, flat_size=%d, output_dim=%d",
            input_shape,
            flat_size,
            output_dim,
        )

    def forward(self, voxel_grid: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            voxel_grid: Tensor of shape ``[batch, X, Y, Z]`` containing
                integer block type IDs.

        Returns:
            Feature tensor of shape ``[batch, output_dim]``.
        """
        # Normalize integer block IDs to [0, 1]
        x = voxel_grid.unsqueeze(1).float() / 255.0
        x = self.conv(x)
        return self.fc(x)


class RNDPredictorNetwork(nn.Module):
    """Trainable predictor network.

    Mirrors the architecture of :class:`RNDTargetNetwork` but with an
    additional hidden layer and *trainable* parameters.  The predictor
    is trained to minimise MSE against the target network's output.
    """

    def __init__(self, input_shape: tuple = (11, 11, 7), output_dim: int = 128):
        """Initialize the predictor network.

        Args:
            input_shape: Spatial shape of the voxel grid (X, Y, Z).
            output_dim: Dimensionality of the output feature vector.
        """
        super().__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim

        # 3-D CNN backbone (same as target)
        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        # Dynamically compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            flat_size = int(self.conv(dummy).shape[1])

        # Deeper FC head to give the predictor more capacity
        self.fc = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim),
        )

        logger.debug(
            "RNDPredictorNetwork initialized: shape=%s, flat_size=%d, output_dim=%d",
            input_shape,
            flat_size,
            output_dim,
        )

    def forward(self, voxel_grid: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            voxel_grid: Tensor of shape ``[batch, X, Y, Z]`` containing
                integer block type IDs.

        Returns:
            Feature tensor of shape ``[batch, output_dim]``.
        """
        x = voxel_grid.unsqueeze(1).float() / 255.0
        x = self.conv(x)
        return self.fc(x)


class RNDCuriosity:
    """RND-based intrinsic curiosity module.

    Computes intrinsic rewards by measuring the prediction error of the
    predictor network against a fixed random target network.  Running
    mean/variance statistics are maintained to normalize rewards.
    """

    def __init__(
        self,
        reward_scale: float = 1.0,
        learning_rate: float = 1e-4,
        update_interval: int = 1,
        device: str = "cpu",
        voxel_shape: tuple = (11, 11, 7),
        output_dim: int = 128,
    ):
        """Initialize the RND curiosity module.

        Args:
            reward_scale: Multiplier applied to the normalized intrinsic
                reward before returning it.
            learning_rate: Adam learning rate for the predictor network.
            update_interval: Update the predictor every N calls to
                :meth:`compute`.  Set to 1 to update every step.
            device: PyTorch device string (``"cpu"`` or ``"cuda"``).
            voxel_shape: Spatial shape of the incoming voxel grids.
            output_dim: Dimensionality of the RND feature vectors.
        """
        self.reward_scale = reward_scale
        self.update_interval = max(update_interval, 1)
        self.device = torch.device(device)
        self.step_count = 0
        self.voxel_shape = voxel_shape

        # Networks
        self.target_net = RNDTargetNetwork(voxel_shape, output_dim).to(self.device)
        self.predictor_net = RNDPredictorNetwork(voxel_shape, output_dim).to(
            self.device
        )
        self.optimizer = optim.Adam(self.predictor_net.parameters(), lr=learning_rate)

        # Running statistics for reward normalization (Welford's algorithm)
        self.reward_mean: float = 0.0
        self.reward_var: float = 1.0
        self.reward_count: int = 0

        # Move networks to eval mode (no dropout/BN, but we use MSE loss)
        self.target_net.eval()
        self.predictor_net.train()

        logger.info(
            "RNDCuriosity initialized: device=%s, scale=%.3f, lr=%.1e, "
            "update_interval=%d",
            self.device,
            reward_scale,
            learning_rate,
            self.update_interval,
        )

    @torch.no_grad()
    def compute(self, obs: Dict[str, np.ndarray]) -> float:
        """Compute RND intrinsic reward for an observation.

        Extracts the ``"voxel_grid"`` field from *obs*, runs it through
        both target and predictor networks, and returns the normalised
        prediction error as the intrinsic reward.

        Args:
            obs: Observation dict from the environment.

        Returns:
            Normalised intrinsic reward (float).
        """
        self.step_count += 1

        # Extract voxel grid from observation. If frame-stacked, use the latest frame.
        voxel_grid = obs.get("voxel_grid")
        if voxel_grid is None:
            logger.debug("No voxel_grid in observation -- returning 0.0 RND reward")
            return 0.0

        # Frame-stacked input: (n_frames, X, Y, Z) -> take the last frame.
        if voxel_grid.ndim == 4 and voxel_grid.shape[0] == self.voxel_shape[0]:
            # Unstacked input whose first dim happens to equal X; still use as-is.
            pass
        if voxel_grid.ndim == 4 and voxel_grid.shape != self.voxel_shape:
            voxel_grid = voxel_grid[-1]

        # Validate shape
        if voxel_grid.shape != self.voxel_shape:
            logger.warning(
                "Voxel grid shape mismatch: expected %s, got %s",
                self.voxel_shape,
                voxel_grid.shape,
            )
            return 0.0

        # Convert to torch tensor [1, X, Y, Z]
        voxel_tensor = torch.from_numpy(voxel_grid).unsqueeze(0).to(self.device)

        # Forward passes
        with torch.no_grad():
            target_features = self.target_net(voxel_tensor)
            pred_features = self.predictor_net(voxel_tensor)

        # Prediction error as intrinsic reward (L2 norm across feature dim)
        error = torch.norm(target_features - pred_features, dim=-1).item()

        # Update running statistics using Welford's online algorithm
        self.reward_count += 1
        delta = error - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = error - self.reward_mean
        self.reward_var += delta * delta2

        # Normalize reward
        std = np.sqrt(self.reward_var / max(self.reward_count, 1)) + 1e-8
        normalized_reward = error / std

        # Update predictor every N steps
        if self.step_count % self.update_interval == 0:
            self._update_predictor(voxel_tensor, target_features)

        return float(normalized_reward * self.reward_scale)

    def _update_predictor(
        self, voxel_tensor: torch.Tensor, target_features: torch.Tensor
    ) -> None:
        """Train the predictor to match the target.

        Args:
            voxel_tensor: Input voxel grid tensor.
            target_features: Target features from the frozen target network.
        """
        self.predictor_net.train()
        self.optimizer.zero_grad()
        pred_features = self.predictor_net(voxel_tensor)
        loss = nn.functional.mse_loss(pred_features, target_features)
        loss.backward()
        self.optimizer.step()

        logger.debug("RND predictor loss: %.6f", loss.item())

    def reset(self) -> None:
        """Reset episodic state.

        RND maintains learned state across episodes (the predictor
        parameters are global), so this is a no-op.
        """
        pass

    def save(self, path: str) -> None:
        """Save predictor network and running statistics to disk.

        Args:
            path: File path for the checkpoint.
        """
        torch.save(
            {
                "predictor_state_dict": self.predictor_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "reward_mean": self.reward_mean,
                "reward_var": self.reward_var,
                "reward_count": self.reward_count,
                "step_count": self.step_count,
            },
            path,
        )
        logger.info("RND curiosity saved to %s", path)

    def load(self, path: str) -> None:
        """Load predictor network and running statistics from disk.

        Args:
            path: File path of the checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.predictor_net.load_state_dict(checkpoint["predictor_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.reward_mean = checkpoint.get("reward_mean", 0.0)
        self.reward_var = checkpoint.get("reward_var", 1.0)
        self.reward_count = checkpoint.get("reward_count", 0)
        self.step_count = checkpoint.get("step_count", 0)
        logger.info("RND curiosity loaded from %s", path)
