#!/usr/bin/env python3
"""Training entry point for Harvy RL Bot v2."""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch

# Add src to path for direct execution.
sys.path.insert(0, str(Path(__file__).parent))

from common.config import load_config
from env.minecraft_env import MinecraftEnv
from agent.ppo_agent import create_ppo_agent, make_vec_env, load_agent
from agent.callbacks import HarvyTrainingCallback, find_latest_checkpoint
from memory.knowledge_base import KnowledgeBase
from memory.experience_replay import ExperienceReplay
from memory.trust_manager import TrustManager
from web.command_queue import CommandQueue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Harvy RL Bot v2")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Directory for checkpoints and logs",
    )
    parser.add_argument(
        "--memory_dir",
        type=str,
        default="./memory",
        help="Directory for SQLite memory database",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=None,
        help="Override total training timesteps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cpu/cuda/auto). Overrides config.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="auto",
        help="Path to checkpoint to resume from, or 'auto' for latest, or 'none'",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=1,
        help="Number of parallel environments",
    )
    return parser.parse_args()


def make_env(config: dict, rank: int = 0, kb=None, replay=None, tm=None, cq=None):
    """Factory for a single MinecraftEnv."""
    env_cfg = config["environment"]
    reward_cfg = config.get("rewards", {})

    def _init():
        env = MinecraftEnv(
            host=env_cfg.get("host", "localhost"),
            port=env_cfg.get("port", 9876) + rank,
            max_episode_steps=env_cfg.get("max_episode_steps", 6000),
            tcp_timeout=env_cfg.get("tcp_timeout", 30.0),
            reward_config=reward_cfg,
            render_mode=env_cfg.get("render_mode", None),
            knowledge_base=kb,
            replay_buffer=replay,
            trust_manager=tm,
            command_queue=cq,
        )
        return env

    return _init


def resolve_resume_path(resume_arg: str, ckpt_dir: Path) -> str | None:
    if resume_arg == "none":
        return None
    if resume_arg == "auto":
        latest = find_latest_checkpoint(str(ckpt_dir))
        return latest
    return resume_arg


def main():
    args = parse_args()
    config = load_config(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    tb_dir = log_dir / "tensorboard"
    tb_dir.mkdir(exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    memory_dir = Path(args.memory_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(memory_dir / "harvy_memory.db")

    kb = KnowledgeBase(db_path)
    replay = ExperienceReplay(db_path)
    tm = TrustManager(kb)
    cq = CommandQueue(db_path)

    owner_cfg = config.get("owner", {})
    owner_name = owner_cfg.get("name", "")
    owner_uuid = owner_cfg.get("uuid", "")
    if owner_name:
        tm.set_owner(owner_uuid or owner_name, owner_name)
        logger.info("Owner set to %s", owner_name)

    train_cfg = config["training"]
    total_timesteps = args.total_timesteps or train_cfg["total_timesteps"]
    device = args.device or train_cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Creating %d environment(s)", args.n_envs)
    if args.n_envs == 1:
        env = make_env(config, 0, kb=kb, replay=replay, tm=tm, cq=cq)()
    else:
        # Note: shared KB across SubprocVecEnv is not thread-safe; for now only
        # single-env training persists memory per run. Multi-env is future work.
        env = make_vec_env(
            [make_env(config, i) for i in range(args.n_envs)],
            n_envs=args.n_envs,
        )

    resume_path = resolve_resume_path(args.resume, ckpt_dir)
    model = None
    if resume_path:
        logger.info("Resuming from %s", resume_path)
        model = load_agent(resume_path, env, device=device)

    if model is None:
        logger.info("Creating new PPO agent")
        model = create_ppo_agent(
            env=env,
            tensorboard_log=str(tb_dir),
            device=device,
            learning_rate=train_cfg.get("learning_rate", 3e-4),
            n_steps=train_cfg.get("n_steps", 2048),
            batch_size=train_cfg.get("batch_size", 64),
            n_epochs=train_cfg.get("n_epochs", 4),
            gamma=train_cfg.get("gamma", 0.99),
            gae_lambda=train_cfg.get("gae_lambda", 0.95),
            clip_range=train_cfg.get("clip_range", 0.2),
            ent_coef=train_cfg.get("ent_coef", 0.01),
            vf_coef=train_cfg.get("vf_coef", 0.5),
            max_grad_norm=train_cfg.get("max_grad_norm", 0.5),
            verbose=train_cfg.get("verbose", 1),
        )

    checkpoint_freq = train_cfg.get("checkpoint_freq", 100_000)
    callback = HarvyTrainingCallback(
        checkpoint_dir=str(ckpt_dir),
        checkpoint_freq=checkpoint_freq,
        knowledge_base=kb,
        replay_buffer=replay if args.n_envs == 1 else None,
        verbose=1,
    )

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=False if resume_path else True,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        final_path = ckpt_dir / "final_model.zip"
        model.save(str(final_path))
        logger.info("Saved final model to %s", final_path)
        env.close()
        kb.close()
        replay.close()
        cq.close()
        # TrustManager shares the KB connection; no separate close needed.


if __name__ == "__main__":
    main()
