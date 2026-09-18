#!/usr/bin/env python3
"""
Main training script for the RL Minecraft Bot.

Usage:
    python src/train.py --config configs/default.yaml --output_dir ./checkpoints
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import yaml
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from env.minecraft_env import MinecraftEnv
from env.intrinsic_motivation import IntrinsicMotivation
from env.rnd_curiosity import RNDCuriosity
from agent.ppo_agent import create_ppo_agent, make_vec_env, linear_schedule, load_agent
from agent.callbacks import (
    CurriculumCallback,
    MetricsLoggerCallback,
    SkillEvaluationCallback,
    BenchmarkEvaluationCallback,
    AuxStateCallback,
)
from skills.curriculum import CurriculumManager
from skills.autonomous_curriculum import AutonomousCurriculum

# Benchmark imports (best-effort)
try:
    from eval_benchmarks import BenchmarkSuite
except ImportError:
    try:
        from src.eval_benchmarks import BenchmarkSuite
    except ImportError:
        BenchmarkSuite = None  # type: ignore[misc,assignment]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL Minecraft Bot")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=None,
        help="Override total training timesteps",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to train on (cpu/cuda/auto)"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--curriculum_only",
        action="store_true",
        help="Only train with curriculum learning",
    )
    parser.add_argument(
        "--autonomous",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let Harvy pick its own goals and explore (default: True; use --no-autonomous or --guided to disable)",
    )
    parser.add_argument(
        "--guided",
        dest="autonomous",
        action="store_false",
        help="Use the fixed human-defined curriculum instead",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def create_env(
    config: dict,
    goal: str = "survive_first_night",
    autonomous: bool = False,
    goal_generator: Optional[AutonomousCurriculum] = None,
    intrinsic_motivation: Optional[IntrinsicMotivation] = None,
    rnd_curiosity: Optional[RNDCuriosity] = None,
) -> MinecraftEnv:
    """Create a single Minecraft environment."""
    env_config = config.get("environment", {})
    training_config = config.get("training", {})
    env = MinecraftEnv(
        host=env_config.get("host", "localhost"),
        port=env_config.get("port", 9876),
        max_episode_steps=env_config.get("max_episode_steps", 6000),
        goal=goal,
        autonomous=autonomous,
        goal_generator=goal_generator,
        intrinsic_motivation=intrinsic_motivation,
        rnd_curiosity=rnd_curiosity,
        normalize_observations=training_config.get("normalize_observations", True),
        frame_stack=training_config.get("frame_stack", 4),
        noop_bias=training_config.get("noop_bias", 0.01),
    )
    env = Monitor(env)
    return env


def train(args: argparse.Namespace) -> None:
    """Main training function."""
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Override config with CLI args
    total_timesteps = args.total_timesteps or config["training"]["total_timesteps"]
    device = (
        args.device
        if args.device != "auto"
        else config["training"].get("device", "cpu")
    )

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    tb_dir = log_dir / "tensorboard"
    tb_dir.mkdir(exist_ok=True)

    # Autonomous mode: Harvy picks its own goals and gets curiosity rewards
    if args.autonomous:
        logger.info("AUTONOMOUS MODE: Harvy will pick its own goals and explore")
        goal_generator = AutonomousCurriculum()
        intrinsic_motivation = IntrinsicMotivation()

        # Initialize RND curiosity module if enabled in config
        rnd_config = config.get("rnd_curiosity", {})
        rnd: Optional[RNDCuriosity] = None
        if rnd_config.get("enabled", True):
            device = (
                args.device
                if args.device != "auto"
                else config["training"].get("device", "cpu")
            )
            rnd = RNDCuriosity(
                reward_scale=rnd_config.get("reward_scale", 1.0),
                learning_rate=rnd_config.get("learning_rate", 1e-4),
                update_interval=rnd_config.get("update_interval", 1),
                device=device,
            )
            logger.info(
                "RND curiosity enabled: scale=%.3f, lr=%.1e, update_interval=%d",
                rnd_config.get("reward_scale", 1.0),
                rnd_config.get("learning_rate", 1e-4),
                rnd_config.get("update_interval", 1),
            )
        else:
            logger.info("RND curiosity disabled by config")

        env = create_env(
            config,
            autonomous=True,
            goal_generator=goal_generator,
            intrinsic_motivation=intrinsic_motivation,
            rnd_curiosity=rnd,
        )
    else:
        # Fixed human-defined curriculum
        curriculum = CurriculumManager(
            auto_advance=config["curriculum"].get("auto_advance", True),
            advance_after_n_successes=config["curriculum"].get(
                "advance_after_n_successes", 5
            ),
        )
        initial_goal = curriculum.current_stage.goal
        logger.info(f"Starting with goal: {initial_goal}")
        env = create_env(config, goal=initial_goal)

    # Resume from checkpoint if specified
    model = None
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        model = load_agent(args.resume, env, device=device)
        # Restore auxiliary state saved alongside the checkpoint
        # (observation normalizer stats, RND predictor) so the observation
        # and intrinsic-reward distributions don't silently shift.
        prefix = args.resume[:-4] if args.resume.endswith(".zip") else args.resume
        raw_env = getattr(env, "unwrapped", env)
        if hasattr(raw_env, "load_aux_state"):
            raw_env.load_aux_state(prefix)
        if args.autonomous:
            auto_state_path = output_dir / "harvy_autonomous_state.json"
            if auto_state_path.exists():
                goal_generator.load_progress(str(auto_state_path))
                logger.info(f"Restored autonomous goal state from {auto_state_path}")

    if model is None:
        # Create new agent
        logger.info("Creating new PPO agent")
        lr_config = config["training"].get("learning_rate", {})
        lr = lr_config.get("initial", 3e-4)
        if lr_config.get("schedule", "constant") == "linear":
            lr = linear_schedule(lr)

        model = create_ppo_agent(
            env=env,
            tensorboard_log=str(tb_dir),
            device=device,
            learning_rate=lr,
            n_steps=config["training"].get("n_steps", 2048),
            batch_size=config["training"].get("batch_size", 64),
            n_epochs=config["training"].get("n_epochs", 10),
            gamma=config["training"].get("gamma", 0.99),
            gae_lambda=config["training"].get("gae_lambda", 0.95),
            clip_range=config["training"].get("clip_range", 0.2),
            ent_coef=config["training"].get("ent_coef", 0.01),
            verbose=config["training"].get("verbose", 1),
        )

    # Setup callbacks
    callbacks = []

    # Checkpoint callback -- save every N steps (PPO weights + aux state)
    checkpoint_freq = config["callbacks"].get("checkpoint_freq", 100000)
    callbacks.append(
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(output_dir / "checkpoints"),
            name_prefix="harvy",
        )
    )
    callbacks.append(
        AuxStateCallback(
            save_freq=checkpoint_freq,
            save_path_prefix=str(output_dir / "checkpoints" / "harvy"),
        )
    )

    # Metrics logger
    callbacks.append(
        MetricsLoggerCallback(
            log_file=str(log_dir / "metrics.json"),
            log_freq=config["callbacks"].get("log_freq", 1000),
        )
    )

    # Curriculum callback (only in guided mode) -- drives the shared manager
    if not args.autonomous and config["curriculum"].get("enabled", True):
        callbacks.append(
            CurriculumCallback(
                curriculum_manager=curriculum,
                switch_threshold=config["curriculum"].get("switch_threshold", 0.7),
            )
        )

    # Skill evaluation callback. NOTE: the eval env shares the bot's single
    # TCP socket -- with multiple envs per bot this steals the connection
    # from the training env. Keep eval_freq unset unless you run a second
    # bot instance on a separate TCP port.
    eval_env = None
    benchmark_eval_env = None
    if config["callbacks"].get("eval_freq", None):
        eval_env = create_env(config, goal="survive")
        callbacks.append(
            SkillEvaluationCallback(
                eval_env=eval_env,
                eval_freq=config["callbacks"]["eval_freq"],
                n_eval_episodes=config["callbacks"].get("n_eval_episodes", 10),
                skills=["survival", "combat", "gathering"],
            )
        )

    # Benchmark evaluation callback (same single-socket caveat as above)
    benchmark_cfg = config.get("benchmarks", {})
    if benchmark_cfg.get("enabled", False) and BenchmarkSuite is not None:
        logger.info("Benchmark suite enabled – will evaluate %d skill benchmarks", 7)
        benchmark_suite = BenchmarkSuite(output_dir=str(output_dir / "benchmarks"))
        # Create a dedicated eval env for benchmarks
        benchmark_eval_env = create_env(config, goal="survive_first_night")
        callbacks.append(
            BenchmarkEvaluationCallback(
                eval_env=benchmark_eval_env,
                benchmark_suite=benchmark_suite,
                eval_freq=benchmark_cfg.get("eval_freq", 50000),
                n_eval_episodes=benchmark_cfg.get("n_eval_episodes", 3),
                save_best=benchmark_cfg.get("save_best", True),
            )
        )
    elif benchmark_cfg.get("enabled", False):
        logger.warning(
            "Benchmarks enabled in config but eval_benchmarks module not found"
        )

    # Training loop
    logger.info(f"Starting training for {total_timesteps} timesteps")
    logger.info(f"Device: {device}")
    if args.autonomous:
        logger.info("Harvy is choosing its own goals each episode")
    else:
        logger.info(f"Curriculum stage: {curriculum.current_stage.name}")

    training_error: Optional[BaseException] = None
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        training_error = e
        logger.error(f"Training FAILED: {e}", exc_info=True)
    finally:
        # Save final model + auxiliary state (normalizer stats, RND
        # predictor) so a later --resume keeps identical distributions.
        final_path = output_dir / "final_model.zip"
        model.save(str(final_path))
        logger.info(f"Saved final model to {final_path}")
        raw_env = getattr(env, "unwrapped", env)
        if hasattr(raw_env, "save_aux_state"):
            raw_env.save_aux_state(str(output_dir / "final_model"))

        # Save curriculum / autonomous goal state
        if args.autonomous:
            progress_path = output_dir / "harvy_autonomous_state.json"
            goal_generator.save_progress(str(progress_path))
            logger.info(f"Saved autonomous goal state to {progress_path}")
        else:
            curriculum_path = output_dir / "curriculum_state.yaml"
            curriculum.to_yaml(str(curriculum_path))
            logger.info(f"Saved curriculum state to {curriculum_path}")

        env.close()
        # Eval envs hold their own sockets -- close them too.
        if eval_env is not None:
            eval_env.close()
        if benchmark_eval_env is not None:
            benchmark_eval_env.close()

    if training_error is not None:
        # Don't mask a failed run behind a clean exit code.
        sys.exit(1)


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
