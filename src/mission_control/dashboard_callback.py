"""SB3 callback that forwards metrics to the mission control manager."""

import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class DashboardCallback(BaseCallback):
    def __init__(self, manager, log_freq: int = 1000):
        super().__init__()
        self.manager = manager
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.manager.stop_event.is_set():
            self.manager.post_message(
                {"type": "terminal", "text": "Stop requested", "kind": "info"}
            )
            return False

        if self.manager.pause_event.is_set():
            self.manager.post_message(
                {"type": "terminal", "text": "Training paused", "kind": "info"}
            )
            while self.manager.pause_event.is_set():
                if self.manager.stop_event.is_set():
                    return False
                time.sleep(0.2)

        # Apply pending live hyperparameter changes.
        hyperparams = self.manager.hyperparams
        if hyperparams:
            lr_key = None
            for key in ("learning_rate", "learningRate"):
                if key in hyperparams and isinstance(hyperparams[key], (int, float)):
                    self.model.learning_rate = hyperparams[key]
                    lr_key = key
                    break
            if lr_key is not None:
                del hyperparams[lr_key]

            for key in list(hyperparams.keys()):
                self.manager.post_message(
                    {
                        "type": "terminal",
                        "text": f"Hyperparameter '{key}' stored but not applied live",
                        "kind": "info",
                    }
                )
                del hyperparams[key]

        if self.n_calls % self.log_freq == 0:
            ep_buffer = self.model.ep_info_buffer
            mean_reward = (
                float(np.mean([ep["r"] for ep in ep_buffer])) if ep_buffer else 0.0
            )
            mean_length = (
                float(np.mean([ep["l"] for ep in ep_buffer])) if ep_buffer else 0.0
            )
            loss = float(self.model.logger.name_to_value.get("train/value_loss", 0.0))
            with self.manager._lock:
                self.manager._status["stepCount"] = self.n_calls
                self.manager._status["episodeCount"] = len(ep_buffer)
            self.manager.post_message(
                {
                    "type": "metrics",
                    "timesteps": self.num_timesteps,
                    "episode": len(ep_buffer),
                    "step": self.n_calls,
                    "mean_reward": mean_reward,
                    "mean_ep_length": mean_length,
                    "loss": loss,
                }
            )
        return True
