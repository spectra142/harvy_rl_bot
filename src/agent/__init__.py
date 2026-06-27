"""PPO-based RL agent with custom feature extractor for Dict observation spaces."""

from .networks import (
    MinecraftFeatureExtractor,
    SelfEmbedding,
    EntityEncoder,
    VoxelEncoder,
    EnvEncoder,
)
from .ppo_agent import (
    create_ppo_agent,
    make_vec_env,
    linear_schedule,
    load_agent,
)
from .callbacks import (
    CurriculumCallback,
    MetricsLoggerCallback,
    SkillEvaluationCallback,
)

__all__ = [
    # Networks
    "MinecraftFeatureExtractor",
    "SelfEmbedding",
    "EntityEncoder",
    "VoxelEncoder",
    "EnvEncoder",
    # PPO Agent
    "create_ppo_agent",
    "make_vec_env",
    "linear_schedule",
    "load_agent",
    # Callbacks
    "CurriculumCallback",
    "MetricsLoggerCallback",
    "SkillEvaluationCallback",
]
