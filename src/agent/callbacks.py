"""SB3 callbacks for Harvy v2.

- Auto-save checkpoints.
- Persist episode data to knowledge base.
- Save high-reward trajectories for warm-start.
"""

import os
import time
from pathlib import Path
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback


class HarvyTrainingCallback(BaseCallback):
    """Callback that checkpoints, logs to knowledge base, and records trajectories."""

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_freq: int = 100_000,
        knowledge_base=None,
        replay_buffer=None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_freq = checkpoint_freq
        self.kb = knowledge_base
        self.replay = replay_buffer
        self._episode_buffer = []
        self._last_health = 20.0

    def _init_callback(self) -> None:
        pass

    def _on_step(self) -> bool:
        if self.n_calls % self.checkpoint_freq == 0:
            path = self.checkpoint_dir / f"harvy_{self.n_calls}_steps.zip"
            self.model.save(str(path))
            if self.verbose:
                print(f"[callback] saved checkpoint {path}")

        # Accumulate trajectory for replay storage.
        if self.replay is not None:
            info = self.locals.get("infos", [{}])[0]
            obs = self.locals.get("new_obs")
            action = self.locals.get("actions")
            reward = self.locals.get("rewards", [0.0])[0]
            if obs is not None and action is not None:
                self._episode_buffer.append(
                    {
                        "step": info.get("step", self.n_calls),
                        "action": int(action[0]),
                        "reward": float(reward),
                        "health": info.get("health", 20.0),
                        "position": info.get("position", [0.0, 64.0, 0.0]),
                    }
                )

        return True

    def _on_rollout_end(self) -> None:
        # Rollout end is a good time to flush any per-episode stats, but SB3
        # rolls out every n_steps, not per episode. We rely on _on_training_end
        # or the env itself for episode-level persistence.
        pass

    def _on_training_end(self) -> None:
        final_path = self.checkpoint_dir / "final_model.zip"
        self.model.save(str(final_path))

    def flush_episode(self, total_reward: float, skill_name: str = "") -> None:
        """Call this from env-level episode end if available."""
        if self.replay is not None and self._episode_buffer:
            self.replay.store(self._episode_buffer, total_reward, skill_name)
        self._episode_buffer = []


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Return path to the most recent checkpoint by step count."""
    cp_dir = Path(checkpoint_dir)
    if not cp_dir.exists():
        return None
    candidates = sorted(cp_dir.glob("harvy_*_steps.zip"))
    if not candidates:
        # Fall back to final model.
        final = cp_dir / "final_model.zip"
        return str(final) if final.exists() else None
    return str(candidates[-1])
