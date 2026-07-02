"""Custom feature extractor for Harvy v2.

SB3 only supports Dict observation spaces composed of Box spaces, so categorical
features (held item, entity types, block IDs) are stored as floats and cast to
long inside the embedding layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

from env.observation import ITEM_TO_ID, MAX_ENTITIES, VOXEL_SHAPE, ENTITY_VOCAB_SIZE


class SelfEncoder(nn.Module):
    """Encode self-state: health/food/position/yaw/pitch + held item embedding."""

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.item_embedding = nn.Embedding(len(ITEM_TO_ID), 16)
        # item_emb(16) + health(1) + food(1) + armor(1) + position(3) + yaw(1) + pitch(1) = 24
        self.mlp = nn.Sequential(
            nn.Linear(24, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        item_ids = observations["self_held_item_id"].squeeze(-1).long()
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
    """Encode inventory counts."""

    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(len(ITEM_TO_ID), 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        return self.mlp(observations["inventory"])


class EntityEncoder(nn.Module):
    """Encode nearby entities with sum pooling."""

    def __init__(self, embedding_dim: int = 64, max_entities: int = MAX_ENTITIES):
        super().__init__()
        self.type_embedding = nn.Embedding(ENTITY_VOCAB_SIZE, 16)
        # type_emb(16) + distance(1) + health(1) + hostile(1) + position(3) = 22
        self.mlp = nn.Sequential(
            nn.Linear(22, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
            nn.ReLU(),
        )
        self.max_entities = max_entities

    def forward(self, observations: dict) -> torch.Tensor:
        type_ids = observations["entity_type_ids"].long()
        type_emb = self.type_embedding(type_ids)  # [B, N, 16]
        features = torch.cat(
            [
                type_emb,
                observations["entity_distances"],
                observations["entity_healths"],
                observations["entity_hostiles"],
                observations["entity_positions"],
            ],
            dim=-1,
        )
        emb = self.mlp(features)  # [B, N, D]
        return emb.sum(dim=1)  # [B, D]


class VoxelEncoder(nn.Module):
    """3D CNN over the voxel grid.

    Block IDs are categorical but stored as floats. We keep them as a single
    channel and let the network learn; with only 847 voxels an embedding lookup
    table is also reasonable but adds complexity.
    """

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, embedding_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # 11x11x7 -> after stride2 -> 6x6x4 -> after stride2 -> 3x3x2
        self.fc = nn.Linear(embedding_dim * 3 * 3 * 2, embedding_dim)

    def forward(self, observations: dict) -> torch.Tensor:
        x = observations["voxel_grid"].unsqueeze(1).float() / 255.0
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return F.relu(self.fc(x))


class EnvEncoder(nn.Module):
    """Encode environmental features."""

    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, 32),
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


class HarvyFeatureExtractor(BaseFeaturesExtractor):
    """Combined feature extractor for all observation modalities."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        self.self_encoder = SelfEncoder(64)
        self.inventory_encoder = InventoryEncoder(32)
        self.entity_encoder = EntityEncoder(64)
        self.voxel_encoder = VoxelEncoder(64)
        self.env_encoder = EnvEncoder(32)

        total_dim = 64 + 32 + 64 + 64 + 32  # = 256
        self.projection = nn.Sequential(
            nn.Linear(total_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        self_emb = self.self_encoder(observations)
        inv_emb = self.inventory_encoder(observations)
        entity_emb = self.entity_encoder(observations)
        voxel_emb = self.voxel_encoder(observations)
        env_emb = self.env_encoder(observations)

        combined = torch.cat(
            [self_emb, inv_emb, entity_emb, voxel_emb, env_emb],
            dim=-1,
        )
        return self.projection(combined)
