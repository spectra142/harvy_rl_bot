"""Custom neural network architectures for the RL agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np

try:
    from env.observation import INVENTORY_SIZE, MAX_DANGER_BLOCKS, MAX_ENTITIES
except ImportError:
    from src.env.observation import INVENTORY_SIZE, MAX_DANGER_BLOCKS, MAX_ENTITIES


class SelfEmbedding(nn.Module):
    """Process self-status features (health, food, armor, position, yaw, pitch, held_item)."""

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.item_embedding = nn.Embedding(256, 16)
        # item_emb(16) + health(1) + food(1) + armor(1) + position(3) + yaw(1) + pitch(1) = 24
        self.mlp = nn.Sequential(
            nn.Linear(24, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        item_ids = observations["self_held_item_id"].squeeze(-1).long()
        item_ids = item_ids.clamp(0, self.item_embedding.num_embeddings - 1)
        item_emb = self.item_embedding(item_ids)
        stats = torch.cat(
            [
                observations["self_health"],
                observations["self_food"],
                observations["self_armor"],
                observations["self_position"],
                observations["self_yaw"],
                observations["self_pitch"],
            ],
            dim=-1,
        )
        x = torch.cat([item_emb, stats], dim=-1)
        return self.mlp(x)


class InventoryEncoder(nn.Module):
    """Encode the bot's inventory counts."""

    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(INVENTORY_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        return self.mlp(observations["inventory"])


class EntityEncoder(nn.Module):
    """Process nearby entities with attention-like weighting."""

    def __init__(self, embedding_dim: int = 64, max_entities: int = MAX_ENTITIES):
        super().__init__()
        self.type_embedding = nn.Embedding(128, 16)
        # type_emb(16) + distance(1) + health(1) + hostile(1) = 19
        self.mlp = nn.Sequential(
            nn.Linear(19, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
            nn.ReLU(),
        )
        self.max_entities = max_entities

    def forward(self, observations: dict) -> torch.Tensor:
        type_ids = observations["entity_type_ids"].long()
        type_ids = type_ids.clamp(0, self.type_embedding.num_embeddings - 1)
        distances = observations["entity_distances"]
        healths = observations["entity_healths"]
        hostiles = observations["entity_hostiles"]

        type_emb = self.type_embedding(type_ids)  # [batch, max_entities, 16]
        features = torch.cat([type_emb, distances, healths, hostiles], dim=-1)
        emb = self.mlp(features)  # [batch, max_entities, embedding_dim]
        return emb.max(dim=1)[0]  # [batch, embedding_dim]


class DangerBlockEncoder(nn.Module):
    """Encode nearby dangerous/important blocks."""

    def __init__(self, embedding_dim: int = 32, max_blocks: int = MAX_DANGER_BLOCKS):
        super().__init__()
        self.type_embedding = nn.Embedding(128, 16)
        # type_emb(16) + position(3) + distance(1) = 20
        self.mlp = nn.Sequential(
            nn.Linear(20, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
            nn.ReLU(),
        )
        self.max_blocks = max_blocks

    def forward(self, observations: dict) -> torch.Tensor:
        type_ids = observations["danger_block_types"].long()
        type_ids = type_ids.clamp(0, self.type_embedding.num_embeddings - 1)
        positions = observations["danger_block_positions"]
        distances = observations["danger_block_distances"]

        type_emb = self.type_embedding(type_ids)  # [batch, max_blocks, 16]
        features = torch.cat([type_emb, positions, distances], dim=-1)
        emb = self.mlp(features)  # [batch, max_blocks, embedding_dim]
        return emb.max(dim=1)[0]  # [batch, embedding_dim]


class VoxelEncoder(nn.Module):
    """3D CNN encoder for voxel grid observation.

    Supports both unstacked input ``(B, 11, 11, 7)`` and frame-stacked input
    ``(B, C, 11, 11, 7)``. The number of input channels is inferred from the
    observation space and passed in at construction time.
    """

    def __init__(self, in_channels: int = 1, embedding_dim: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # -> ~6x6x4
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(
                64, embedding_dim, kernel_size=3, stride=2, padding=1
            ),  # -> ~3x3x2
            nn.ReLU(),
        )
        self.fc = nn.Linear(embedding_dim * 3 * 3 * 2, embedding_dim)

    def forward(self, voxel_grid: torch.Tensor) -> torch.Tensor:
        # Add channel dimension for unstacked input; keep it for stacked input.
        if voxel_grid.dim() == 4:
            x = voxel_grid.unsqueeze(1).float() / 255.0
        else:
            x = voxel_grid.float() / 255.0
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return F.relu(self.fc(x))


class EnvEncoder(nn.Module):
    """Process environmental features."""

    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, 32),  # time + sky + water + ground + danger
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        features = torch.cat(
            [
                observations["env_time_of_day"] / 24000.0,
                observations["env_can_see_sky"],
                observations["env_in_water"],
                observations["env_on_ground"],
                observations["env_danger_level"],
            ],
            dim=-1,
        )
        return self.mlp(features)


class MinecraftFeatureExtractor(BaseFeaturesExtractor):
    """Combined feature extractor for all observation modalities."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Infer voxel input channels from the observation space.
        voxel_shape = observation_space["voxel_grid"].shape
        voxel_in_channels = int(voxel_shape[0]) if len(voxel_shape) == 4 else 1

        self.self_encoder = SelfEmbedding(64)
        self.inventory_encoder = InventoryEncoder(32)
        self.entity_encoder = EntityEncoder(64)
        self.danger_block_encoder = DangerBlockEncoder(32)
        self.voxel_encoder = VoxelEncoder(
            in_channels=voxel_in_channels, embedding_dim=64
        )
        self.env_encoder = EnvEncoder(32)
        self.goal_embedding = nn.Embedding(16, 16)

        total_dim = 64 + 32 + 64 + 32 + 64 + 32 + 16  # = 304
        self.projection = nn.Sequential(
            nn.Linear(total_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        self_emb = self.self_encoder(observations)
        inventory_emb = self.inventory_encoder(observations)
        entity_emb = self.entity_encoder(observations)
        danger_emb = self.danger_block_encoder(observations)
        voxel_emb = self.voxel_encoder(observations["voxel_grid"])
        env_emb = self.env_encoder(observations)
        goal_ids = observations["goal_id"].squeeze(-1).long()
        goal_ids = goal_ids.clamp(0, self.goal_embedding.num_embeddings - 1)
        goal_emb = self.goal_embedding(goal_ids)

        combined = torch.cat(
            [
                self_emb,
                inventory_emb,
                entity_emb,
                danger_emb,
                voxel_emb,
                env_emb,
                goal_emb,
            ],
            dim=-1,
        )
        return self.projection(combined)
