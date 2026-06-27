"""PPO agent setup and training utilities."""

import os
from typing import Optional, Dict, Any

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from .networks import MinecraftFeatureExtractor


def create_ppo_agent(
    env: gym.Env,
    tensorboard_log: str = "./logs/tensorboard/",
    device: str = "auto",
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    verbose: int = 1,
) -> PPO:
    """Create a PPO agent with custom feature extractor."""

    policy_kwargs = dict(
        features_extractor_class=MinecraftFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
        activation_fn=torch.nn.ReLU,
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        tensorboard_log=tensorboard_log,
        device=device,
        verbose=verbose,
    )

    return model


def make_vec_env(env_fn, n_envs: int = 1, start_port: int = 9876) -> DummyVecEnv:
    """Create vectorized environment."""
    if n_envs == 1:
        return DummyVecEnv([env_fn])

    def make_env(rank):
        def _init():
            env = env_fn()
            env = Monitor(env)
            return env

        return _init

    return SubprocVecEnv([make_env(i) for i in range(n_envs)])


def linear_schedule(initial_value: float):
    """Linear learning rate schedule."""

    def func(progress_remaining: float):
        return progress_remaining * initial_value

    return func


def load_agent(path: str, env: gym.Env, device: str = "auto") -> Optional[PPO]:
    """Load a trained PPO agent from checkpoint."""
    if not os.path.exists(path):
        print(f"Checkpoint not found: {path}")
        return None
    return PPO.load(path, env=env, device=device)
