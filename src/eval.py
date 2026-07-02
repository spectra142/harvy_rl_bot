#!/usr/bin/env python3
"""Evaluation / benchmark runner for Harvy v2."""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from common.config import load_config
from env.minecraft_env import MinecraftEnv
from agent.ppo_agent import load_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


BENCHMARKS = {
    "survive_5min": {
        "description": "Stay alive for 5 minutes (6000 steps)",
        "max_steps": 6000,
        "score": lambda steps, info: min(steps / 6000.0, 1.0),
        "success": lambda steps, info: steps >= 6000,
    },
    "gather_8_logs": {
        "description": "Collect at least 8 oak logs",
        "max_steps": 4000,
        "score": lambda steps, info: min(
            info["inventory_counts"].get("oak_log", 0) / 8.0, 1.0
        ),
        "success": lambda steps, info: info["inventory_counts"].get("oak_log", 0) >= 8,
    },
    "break_16_stone": {
        "description": "Mine at least 16 cobblestone",
        "max_steps": 8000,
        "score": lambda steps, info: min(
            info["inventory_counts"].get("cobblestone", 0) / 16.0, 1.0
        ),
        "success": lambda steps, info: info["inventory_counts"].get("cobblestone", 0) >= 16,
    },
    "kill_1_hostile": {
        "description": "Kill at least 1 hostile mob",
        "max_steps": 2000,
        "score": lambda steps, info: min(info.get("hostiles_killed", 0) / 1.0, 1.0),
        "success": lambda steps, info: info.get("hostiles_killed", 0) >= 1,
    },
    "survive_combat_1min": {
        "description": "Survive 60 seconds while danger_level >= 0.5",
        "max_steps": 1200,
        "score": lambda steps, info: min(steps / 1200.0, 1.0),
        "success": lambda steps, info: steps >= 1200,
    },
}


def make_env(config: dict):
    env_cfg = config["environment"]
    reward_cfg = config.get("rewards", {})
    return MinecraftEnv(
        host=env_cfg.get("host", "localhost"),
        port=env_cfg.get("port", 9876),
        max_episode_steps=env_cfg.get("max_episode_steps", 6000),
        tcp_timeout=env_cfg.get("tcp_timeout", 30.0),
        reward_config=reward_cfg,
    )


def run_benchmark(
    env: MinecraftEnv,
    model,
    benchmark_name: str,
    n_episodes: int,
) -> Dict[str, Any]:
    cfg = BENCHMARKS[benchmark_name]
    logger.info("Running benchmark '%s': %s", benchmark_name, cfg["description"])

    scores = []
    successes = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        max_steps = cfg["max_steps"]
        final_info = info

        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            final_info = info
            steps += 1

        score = cfg["score"](steps, final_info)
        success = cfg["success"](steps, final_info)
        scores.append(score)
        successes.append(success)
        logger.info("  Episode %d/%d: steps=%d score=%.3f success=%s", ep + 1, n_episodes, steps, score, success)

    return {
        "benchmark": benchmark_name,
        "mean_score": float(np.mean(scores)),
        "success_rate": float(np.mean(successes)),
        "n_episodes": n_episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Harvy v2")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, required=True, help="Path to model.zip")
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument("--output", type=str, default="./eval_results.json")
    args = parser.parse_args()

    config = load_config(args.config)
    env = make_env(config)
    model = load_agent(args.model, env)
    if model is None:
        logger.error("Failed to load model: %s", args.model)
        sys.exit(1)

    results = {}
    for name in BENCHMARKS:
        try:
            results[name] = run_benchmark(env, model, name, args.n_episodes)
        except Exception as e:
            logger.error("Benchmark %s failed: %s", name, e, exc_info=True)
            results[name] = {"error": str(e)}

    env.close()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", args.output)
    logger.info("Overall: %s", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
