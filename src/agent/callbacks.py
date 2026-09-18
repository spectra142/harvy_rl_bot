"""Custom training callbacks for logging, checkpointing, and curriculum management."""

import os
import json
import time
import logging
from typing import Dict, Any, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class CurriculumCallback(BaseCallback):
    """Drives the shared CurriculumManager during guided training.

    This is the ONLY curriculum driver in guided mode: it records episode
    outcomes into the manager and pushes the manager's current stage goal
    into the environment, so the two never drift out of sync.
    """

    def __init__(
        self,
        curriculum_manager,
        switch_threshold: float = 0.7,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.manager = curriculum_manager
        self.switch_threshold = switch_threshold
        self._episodes_recorded = 0

    def _on_training_start(self) -> None:
        stage = self.manager.current_stage
        self.logger.record("curriculum/stage", self.manager.current_stage_idx)
        self.logger.record("curriculum/stage_name", stage.name)
        self.training_env.env_method("set_goal", stage.goal)

    def _on_rollout_end(self) -> None:
        # Record only NEW episodes into the manager (it decides advancement).
        # Monitor's get_episode_rewards() is append-only, so a simple offset
        # into it is a reliable cursor (unlike the rolling ep_info_buffer).
        try:
            per_env = self.training_env.env_method("get_episode_rewards")
            all_rewards = [r for env_rewards in per_env for r in env_rewards]
        except Exception:
            all_rewards = []
        fresh = all_rewards[self._episodes_recorded:]
        self._episodes_recorded = len(all_rewards)
        for r in fresh:
            self.manager.record_episode(r > self.switch_threshold, r)

        stage = self.manager.current_stage
        self.training_env.env_method("set_goal", stage.goal)
        self.logger.record("curriculum/stage", self.manager.current_stage_idx)
        self.logger.record("curriculum/stage_name", stage.name)


class AuxStateCallback(BaseCallback):
    """Saves auxiliary training state (obs normalizer stats, RND predictor)
    alongside the PPO checkpoints so resuming keeps identical observation
    and intrinsic-reward distributions."""

    def __init__(self, save_freq: int, save_path_prefix: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path_prefix = save_path_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            prefix = f"{self.save_path_prefix}_{self.num_timesteps}_steps"
            self.training_env.env_method("save_aux_state", prefix)
        return True


class MetricsLoggerCallback(BaseCallback):
    """Logs detailed training metrics to JSON file."""

    def __init__(self, log_file: str = "logs/metrics.json", log_freq: int = 1000):
        super().__init__()
        self.log_file = log_file
        self.log_freq = log_freq
        self.metrics_history = []
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            metrics = {
                "timesteps": self.num_timesteps,
                "time": time.time(),
                "mean_reward": (
                    np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                    if len(self.model.ep_info_buffer) > 0
                    else 0
                ),
                "mean_ep_length": (
                    np.mean([ep["l"] for ep in self.model.ep_info_buffer])
                    if len(self.model.ep_info_buffer) > 0
                    else 0
                ),
                "mean_value_loss": np.mean(
                    self.model.logger.name_to_value.get("train/value_loss", 0)
                ),
            }
            self.metrics_history.append(metrics)
            # Write to file
            with open(self.log_file, "w") as f:
                json.dump(self.metrics_history, f, indent=2)
        return True


class SkillEvaluationCallback(BaseCallback):
    """Evaluates specific skills during training."""

    def __init__(
        self,
        eval_env,
        eval_freq: int = 50000,
        n_eval_episodes: int = 10,
        skills: Optional[list] = None,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.skills = skills or []
        self.eval_results = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            results = self._evaluate_skills()
            self.eval_results.append(results)
            for skill, score in results.items():
                self.logger.record(f"eval/{skill}", score)
        return True

    def _evaluate_skills(self) -> Dict[str, float]:
        """Run evaluation episodes and score each skill."""
        results = {skill: 0.0 for skill in self.skills}

        for episode in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            steps = 0
            max_steps = 2000

            while not done and steps < max_steps:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                steps += 1

            # Score skills based on episode outcome
            health = info.get("health", 0)
            results["survival"] += min(health / 20.0, 1.0)
            results["combat"] += info.get("mobs_killed", 0) / 5.0
            results["gathering"] += info.get("resources_collected", 0) / 20.0

        # Average across episodes
        return {k: v / self.n_eval_episodes for k, v in results.items()}


class BenchmarkEvaluationCallback(BaseCallback):
    """Runs evaluation benchmarks during training.

    Uses the :class:`BenchmarkSuite` to run deterministic evaluation
    episodes at regular intervals and log the results to TensorBoard.
    Saves the best-performing model checkpoint automatically.
    """

    def __init__(
        self,
        eval_env: Any,
        benchmark_suite: "BenchmarkSuite",
        eval_freq: int = 50000,
        n_eval_episodes: int = 3,
        save_best: bool = True,
        verbose: int = 0,
    ):
        """Initialize the callback.

        Args:
            eval_env: Environment instance used for evaluation.
            benchmark_suite: Configured :class:`BenchmarkSuite` instance.
            eval_freq: Run benchmarks every ``eval_freq`` training steps.
            n_eval_episodes: Number of episodes per benchmark.
            save_best: Whether to save the model when a new best score is
                achieved on any benchmark.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.benchmark_suite = benchmark_suite
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_best = save_best

    def _on_step(self) -> bool:
        """Check if it is time to run benchmarks."""
        if self.n_calls % self.eval_freq == 0:
            logger.info("Running benchmarks at step %d", self.num_timesteps)
            try:
                results = self.benchmark_suite.run_all(
                    self.eval_env,
                    self.model,
                    checkpoint_path=f"step_{self.num_timesteps}",
                    n_episodes=self.n_eval_episodes,
                )

                # Log individual benchmark scores
                for name, result in results.items():
                    if result is not None:
                        self.logger.record(f"benchmark/{name}", result.score)
                        self.logger.record(
                            f"benchmark/{name}_success",
                            1.0 if result.success else 0.0,
                        )

                # Log summary
                summary = self.benchmark_suite.get_summary()
                self.logger.record("benchmark/overall", summary["overall_score"])
                logger.info("Benchmark overall score: %.3f", summary["overall_score"])

                # Save best checkpoint
                if self.save_best and self.benchmark_suite.is_best_checkpoint(results):
                    log_dir = (
                        self.model.tensorboard_log
                        if hasattr(self.model, "tensorboard_log")
                        and self.model.tensorboard_log
                        else "./logs"
                    )
                    path = os.path.join(log_dir, "..", "best_benchmark_model")
                    os.makedirs(path, exist_ok=True)
                    self.model.save(os.path.join(path, "model"))
                    logger.info("Saved new best benchmark model to %s", path)
            except Exception as exc:  # noqa: BLE001
                logger.error("Benchmark evaluation failed: %s", exc, exc_info=True)

        return True
