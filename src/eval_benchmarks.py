#!/usr/bin/env python3
"""Evaluation benchmark suite for Harvy RL Minecraft Bot.

Runs deterministic evaluation episodes to measure skill mastery.
Saves best checkpoint per benchmark and logs results to JSON.

Benchmarks:
- survive_10min: Stay alive for 10 minutes (12000 ticks)
- gather_16_logs: Collect at least 16 oak logs
- craft_pickaxe: Craft any pickaxe
- mine_16_stone: Collect at least 16 cobblestone
- kill_hostile: Kill any hostile mob
- build_shelter: Build an enclosed shelter (place >= 8 blocks)
- survive_first_night: Survive from spawn through the first night
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    benchmark_name: str
    success: bool
    score: float  # 0-1 normalized score
    steps: int
    raw_value: float  # Raw measured value
    details: Dict[str, Any]  # Extra info (health, items collected, etc.)
    timestamp: float
    model_checkpoint: str


class BenchmarkSuite:
    """Collection of all benchmarks for evaluating Harvy's skill mastery."""

    BENCHMARKS: Dict[str, Dict[str, Any]] = {
        "survive_10min": {
            "description": "Survive for 10 minutes (12000 ticks)",
            "max_steps": 12000,
            "goal": "survive",
            "target": 12000,  # target steps for scoring
        },
        "gather_16_logs": {
            "description": "Collect at least 16 oak logs",
            "max_steps": 6000,
            "goal": "gather_logs",
            "target": 16,  # target log count for scoring
        },
        "craft_pickaxe": {
            "description": "Craft any pickaxe",
            "max_steps": 4000,
            "goal": "craft_pickaxe",
            "target": 1,
        },
        "mine_16_stone": {
            "description": "Collect at least 16 cobblestone",
            "max_steps": 8000,
            "goal": "gather_stone",
            "target": 16,
        },
        "kill_hostile": {
            "description": "Kill any hostile mob",
            "max_steps": 6000,
            "goal": "fight_hostile",
            "target": 1,
        },
        "build_shelter": {
            "description": "Build enclosed shelter (8+ blocks)",
            "max_steps": 4000,
            "goal": "build_shelter",
            "target": 8,
        },
        "survive_first_night": {
            "description": "Survive from spawn through first night",
            "max_steps": 14000,
            "goal": "survive_first_night",
            "target": 12000,  # night ends at ~12000 ticks
        },
    }

    def __init__(self, output_dir: str = "./benchmarks") -> None:
        """Initialize the benchmark suite.

        Args:
            output_dir: Directory to save benchmark results and history.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.output_dir / "benchmark_results.json"
        self.results_history: List[Dict[str, Any]] = self._load_history()
        self.best_scores: Dict[str, float] = {name: 0.0 for name in self.BENCHMARKS}
        self._load_best_scores()
        # Set by run_all() when any benchmark beat its previous best; read by
        # is_best_checkpoint() (run_all updates best_scores itself, so a naive
        # "score > best" check afterwards would always be False).
        self._last_run_had_new_best: bool = False

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load historical benchmark results from disk."""
        if self.results_file.exists():
            try:
                with open(self.results_file, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load benchmark history: {e}")
                return []
        return []

    def _load_best_scores(self) -> None:
        """Populate best_scores from loaded history."""
        for entry in self.results_history:
            name = entry.get("benchmark_name")
            score = entry.get("score", 0)
            if name and score > self.best_scores.get(name, 0):
                self.best_scores[name] = score

    def run_benchmark(
        self,
        benchmark_name: str,
        env: Any,
        model: Any,
        n_episodes: int = 3,
    ) -> BenchmarkResult:
        """Run a single benchmark across multiple episodes.

        Args:
            benchmark_name: Name of benchmark to run.
            env: MinecraftEnv instance.
            model: Trained PPO model with a ``predict`` method.
            n_episodes: Number of episodes to average over.

        Returns:
            BenchmarkResult with aggregated score across episodes.

        Raises:
            ValueError: If the benchmark name is not recognised.
        """
        if benchmark_name not in self.BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        cfg = self.BENCHMARKS[benchmark_name]
        logger.info(
            "Running benchmark '%s': %s (%d episodes)",
            benchmark_name,
            cfg["description"],
            n_episodes,
        )

        episode_results: List[Dict[str, Any]] = []
        for ep in range(n_episodes):
            result = self._run_episode(benchmark_name, env, model, cfg)
            episode_results.append(result)
            logger.info(
                "  Episode %d/%d: score=%.3f, success=%s, steps=%d",
                ep + 1,
                n_episodes,
                result["score"],
                result["success"],
                result["steps"],
            )

        # Aggregate across episodes
        avg_score = float(np.mean([r["score"] for r in episode_results]))
        avg_steps = int(np.mean([r["steps"] for r in episode_results]))
        success_rate = float(np.mean([r["success"] for r in episode_results]))

        # Merge details – average numeric fields
        merged_details: Dict[str, Any] = {}
        for r in episode_results:
            for key, value in r["details"].items():
                if key not in merged_details:
                    merged_details[key] = []
                merged_details[key].append(value)
        for key, values in merged_details.items():
            if values and isinstance(values[0], (int, float)):
                merged_details[key] = float(np.mean(values))

        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            success=success_rate >= 0.5,  # Success if >= 50% episodes succeed
            score=avg_score,
            steps=avg_steps,
            raw_value=merged_details.get("raw_value", avg_score),
            details=merged_details,
            timestamp=time.time(),
            model_checkpoint="",  # Filled by caller
        )

        return result

    def _run_episode(
        self,
        benchmark_name: str,
        env: Any,
        model: Any,
        cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a single evaluation episode.

        Args:
            benchmark_name: Name of the benchmark (for goal setting).
            env: MinecraftEnv instance.
            model: Trained PPO model.
            cfg: Benchmark configuration dict.

        Returns:
            Episode result dictionary with score, success, steps, and details.
        """
        obs: Any
        info: Dict[str, Any]
        # Benchmarks may need longer episodes than training (e.g.
        # survive_10min needs 12000 steps > the usual 6000-step cap), so
        # raise the episode cap on this dedicated eval env.
        raw_env = getattr(env, "unwrapped", env)
        if hasattr(raw_env, "max_episode_steps"):
            raw_env.max_episode_steps = max(raw_env.max_episode_steps, cfg["max_steps"])
        obs, info = env.reset(options={"goal": cfg["goal"]})
        done = False
        steps = 0
        max_steps: int = cfg["max_steps"]

        # Track benchmark-specific metrics
        metrics: Dict[str, Any] = {
            "min_health": 20.0,
            "max_inventory_logs": 0,
            "max_inventory_stone": 0,
            "mobs_killed": 0,
            "blocks_placed": 0,
            "pickaxe_crafted": False,
        }

        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

            # Update metrics from info dict
            health: float = info.get("health", 20.0)
            metrics["min_health"] = min(metrics["min_health"], health)
            metrics["mobs_killed"] = info.get("mobs_killed", metrics["mobs_killed"])
            metrics["blocks_placed"] = info.get(
                "blocks_placed", metrics["blocks_placed"]
            )

            # Check for pickaxe crafting from info
            if info.get("pickaxe_crafted", False):
                metrics["pickaxe_crafted"] = True

            # Item counts come from the info dict (a name->count dict), NOT
            # from obs["inventory"] (which is a fixed-size numpy vector).
            inv: Dict[str, Any] = info.get("inventory_counts", {}) or {}
            log_count = sum(
                float(v) for k, v in inv.items() if k.endswith("_log")
            )
            metrics["max_inventory_logs"] = max(
                metrics["max_inventory_logs"], log_count
            )
            metrics["max_inventory_stone"] = max(
                metrics["max_inventory_stone"],
                float(inv.get("cobblestone", 0)),
            )

        # Calculate score based on benchmark
        score, success, raw_value = self._calculate_score(
            benchmark_name, steps, metrics, info
        )

        return {
            "score": score,
            "success": success,
            "steps": steps,
            "raw_value": raw_value,
            "details": {
                **metrics,
                "final_health": info.get("health", 0),
                "raw_value": raw_value,
            },
        }

    def _calculate_score(
        self,
        benchmark_name: str,
        steps: int,
        metrics: Dict[str, Any],
        info: Dict[str, Any],
    ) -> tuple:
        """Calculate a normalised score (0-1) for a benchmark.

        Args:
            benchmark_name: Which benchmark to score.
            steps: Number of steps taken in the episode.
            metrics: Collected metrics dictionary.
            info: Final info dict from the environment.

        Returns:
            Tuple of (score: float, success: bool, raw_value: float).
        """
        if benchmark_name == "survive_10min":
            target_steps = 12000
            score = min(steps / target_steps, 1.0)
            return score, steps >= target_steps, float(steps)

        if benchmark_name == "gather_16_logs":
            log_count = metrics["max_inventory_logs"]
            score = min(log_count / 16.0, 1.0)
            return score, log_count >= 16, float(log_count)

        if benchmark_name == "craft_pickaxe":
            has_pickaxe = bool(metrics.get("pickaxe_crafted", False))
            score = 1.0 if has_pickaxe else 0.0
            return score, has_pickaxe, float(has_pickaxe)

        if benchmark_name == "mine_16_stone":
            stone_count = metrics["max_inventory_stone"]
            score = min(stone_count / 16.0, 1.0)
            return score, stone_count >= 16, float(stone_count)

        if benchmark_name == "kill_hostile":
            kills = metrics["mobs_killed"]
            score = min(kills / 1.0, 1.0)
            return score, kills >= 1, float(kills)

        if benchmark_name == "build_shelter":
            blocks = metrics["blocks_placed"]
            score = min(blocks / 8.0, 1.0)
            return score, blocks >= 8, float(blocks)

        if benchmark_name == "survive_first_night":
            survived = info.get("health", 0) > 0 and steps >= 12000
            score = 1.0 if survived else min(steps / 12000.0, 1.0)
            return score, survived, float(steps)

        return 0.0, False, 0.0

    def run_all(
        self,
        env: Any,
        model: Any,
        checkpoint_path: str = "",
        n_episodes: int = 3,
    ) -> Dict[str, Optional[BenchmarkResult]]:
        """Run all configured benchmarks.

        Args:
            env: MinecraftEnv instance.
            model: Trained PPO model.
            checkpoint_path: Identifier for the current model checkpoint.
            n_episodes: Number of episodes per benchmark.

        Returns:
            Dictionary mapping benchmark name to its result (or None if it
            failed).
        """
        results: Dict[str, Optional[BenchmarkResult]] = {}
        self._last_run_had_new_best = False
        for name in self.BENCHMARKS:
            try:
                result = self.run_benchmark(name, env, model, n_episodes)
                result.model_checkpoint = checkpoint_path
                results[name] = result

                # Save if best
                if result.score > self.best_scores[name]:
                    self.best_scores[name] = result.score
                    self._last_run_had_new_best = True
                    result.details["is_new_best"] = True
                    logger.info("  NEW BEST for %s: %.3f", name, result.score)

                # Record result
                self.results_history.append(asdict(result))
                self._save_history()

            except Exception as exc:  # noqa: BLE001
                logger.error("Benchmark %s failed: %s", name, exc, exc_info=True)
                results[name] = None

        return results

    def _save_history(self) -> None:
        """Persist benchmark history to disk."""
        try:
            with open(self.results_file, "w", encoding="utf-8") as f:
                json.dump(self.results_history, f, indent=2, default=str)
        except OSError as exc:
            logger.error("Failed to save benchmark history: %s", exc)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all benchmark scores.

        Returns:
            Dictionary with best_scores, overall_score, and n_evaluations.
        """
        n_benchmarks = len(self.BENCHMARKS) if self.BENCHMARKS else 0
        n_evaluations = len(self.results_history) // n_benchmarks if n_benchmarks else 0
        return {
            "best_scores": self.best_scores,
            "overall_score": float(np.mean(list(self.best_scores.values()))),
            "n_evaluations": n_evaluations,
        }

    def is_best_checkpoint(self, results: Dict[str, Optional[BenchmarkResult]]) -> bool:
        """Check if the most recent run_all() beat any previous best.

        ``run_all`` updates ``best_scores`` itself, so this must rely on the
        flag recorded during that run rather than re-comparing scores (which
        would always be False).
        """
        return self._last_run_had_new_best

    def get_benchmark_descriptions(self) -> Dict[str, str]:
        """Return a mapping of benchmark names to their descriptions."""
        return {name: cfg["description"] for name, cfg in self.BENCHMARKS.items()}
