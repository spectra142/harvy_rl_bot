#!/usr/bin/env python3
"""
Inference script -- load a trained model and control the Minecraft bot.

Usage:
    python src/infer.py --model checkpoints/final_model.zip --episodes 10
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from env.minecraft_env import MinecraftEnv
from agent.ppo_agent import load_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RL Minecraft Bot inference")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="Minecraft server host"
    )
    parser.add_argument("--port", type=int, default=9876, help="Bot communication port")
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes to run"
    )
    parser.add_argument(
        "--max_steps", type=int, default=6000, help="Max steps per episode"
    )
    parser.add_argument(
        "--goal", type=str, default="full_survival", help="Goal for the bot"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy (no exploration)",
    )
    parser.add_argument(
        "--render", action="store_true", help="Print state info each step"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay between steps (seconds) for visualization",
    )
    return parser.parse_args()


def run_episode(
    env, model, deterministic: bool = True, render: bool = False, delay: float = 0.0
) -> dict:
    """Run a single episode with the trained model."""
    obs, info = env.reset()
    done = False
    step = 0
    total_reward = 0.0

    episode_stats = {
        "steps": 0,
        "total_reward": 0.0,
        "max_health": 20.0,
        "min_health": 20.0,
        "mobs_killed": 0,
        "blocks_mined": 0,
        "survived": False,
    }

    while not done:
        # Get action from model
        action, _states = model.predict(obs, deterministic=deterministic)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        total_reward += reward

        # Update stats
        health = info.get("health", 20.0)
        episode_stats["max_health"] = max(episode_stats["max_health"], health)
        episode_stats["min_health"] = min(episode_stats["min_health"], health)
        episode_stats["mobs_killed"] = info.get("mobs_killed", 0)

        if render:
            logger.info(
                f"Step {step} | Action: {action} | Reward: {reward:.2f} | "
                f"Health: {health:.1f} | Total: {total_reward:.1f}"
            )

        if delay > 0:
            time.sleep(delay)

    episode_stats["steps"] = step
    episode_stats["total_reward"] = total_reward
    episode_stats["survived"] = info.get("health", 0) > 0

    return episode_stats


def main():
    args = parse_args()

    # Create environment
    env = MinecraftEnv(
        host=args.host,
        port=args.port,
        max_episode_steps=args.max_steps,
        goal=args.goal,
    )

    # Load model
    logger.info(f"Loading model from {args.model}")
    model = load_agent(args.model, env)
    if model is None:
        logger.error("Failed to load model")
        return

    # Run episodes
    all_stats = []
    for episode in range(args.episodes):
        logger.info(f"\n{'='*50}")
        logger.info(f"Episode {episode + 1}/{args.episodes}")
        logger.info(f"{'='*50}")

        stats = run_episode(
            env,
            model,
            deterministic=args.deterministic,
            render=args.render,
            delay=args.delay,
        )
        all_stats.append(stats)

        logger.info(
            f"Episode complete: {stats['steps']} steps | "
            f"Reward: {stats['total_reward']:.1f} | "
            f"Survived: {stats['survived']} | "
            f"Min Health: {stats['min_health']:.1f}"
        )

    # Summary statistics
    logger.info(f"\n{'='*50}")
    logger.info("EVALUATION SUMMARY")
    logger.info(f"{'='*50}")

    survival_rate = sum(1 for s in all_stats if s["survived"]) / len(all_stats)
    mean_reward = np.mean([s["total_reward"] for s in all_stats])
    mean_steps = np.mean([s["steps"] for s in all_stats])
    mean_mobs = np.mean([s["mobs_killed"] for s in all_stats])

    logger.info(f"Episodes: {len(all_stats)}")
    logger.info(f"Survival Rate: {survival_rate*100:.1f}%")
    logger.info(f"Mean Reward: {mean_reward:.1f}")
    logger.info(f"Mean Steps: {mean_steps:.0f}")
    logger.info(f"Mean Mobs Killed: {mean_mobs:.1f}")

    env.close()


if __name__ == "__main__":
    main()
