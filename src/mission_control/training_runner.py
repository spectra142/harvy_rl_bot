"""Training thread manager with pause/stop/manual-override support."""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from mission_control.dashboard_callback import DashboardCallback
from mission_control.env_wrapper import MissionControlEnvWrapper

logger = logging.getLogger(__name__)


class TrainingManager:
    def __init__(self):
        self.training_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self._lock = threading.Lock()
        self.manual_override = False
        self.pending_action: Optional[str] = None
        self.hyperparams: Dict[str, float] = {}
        self._message_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self._status = {
            "isTraining": False,
            "isPaused": False,
            "episodeCount": 0,
            "stepCount": 0,
        }
        self.env: Optional["MinecraftEnv"] = None
        self.pending_reset: Optional[str] = None

    def post_message(self, msg: Dict[str, Any]):
        self._message_queue.put(msg)

    def pop_message(self, timeout: float = 0.05) -> Optional[Dict[str, Any]]:
        try:
            return self._message_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"type": "status", **self._status}

    def _update_status(self, **kwargs):
        with self._lock:
            self._status.update(kwargs)
            status_msg = {"type": "status", **self._status}
        self.post_message(status_msg)

    def set_manual_action(self, action: Optional[str]):
        with self._lock:
            self.manual_override = action is not None
            self.pending_action = action
        self.post_message(
            {"type": "terminal", "text": f"Manual action: {action}", "kind": "info"}
        )

    def set_hyperparam(self, key: str, value: float):
        with self._lock:
            self.hyperparams[key] = value
        self.post_message(
            {
                "type": "terminal",
                "text": f"Hyperparam {key}={value}",
                "kind": "info",
            }
        )

    def is_manual_override(self) -> bool:
        with self._lock:
            return self.manual_override

    def get_pending_action(self) -> Optional[str]:
        with self._lock:
            return self.pending_action

    def clear_pending_action(self):
        with self._lock:
            self.pending_action = None

    def execute_command(self, cmd: str):
        """Forward a server command to the live bot via the env's socket."""
        self.post_message({"type": "terminal", "text": f"> {cmd}", "kind": "command"})
        env = self.env
        target = getattr(env, "env", env)  # unwrap MissionControlEnvWrapper
        if target is not None and hasattr(target, "send_text_command"):
            if target.send_text_command(cmd):
                self.post_message(
                    {
                        "type": "terminal",
                        "text": f"Sent to bot: {cmd}",
                        "kind": "success",
                    }
                )
            else:
                self.post_message(
                    {
                        "type": "terminal",
                        "text": "Command failed: bot socket not connected",
                        "kind": "error",
                    }
                )
        else:
            self.post_message(
                {
                    "type": "terminal",
                    "text": "Command NOT executed: no live bot connection "
                    "(start training first)",
                    "kind": "error",
                }
            )

    def inventory_action(self, action: str, data: Dict[str, Any]):
        self.post_message(
            {"type": "terminal", "text": f"Inventory {action} {data}", "kind": "info"}
        )

    def start_training(
        self,
        config: str,
        device: str = "cpu",
        total_timesteps: Optional[int] = None,
    ):
        if self.training_thread and self.training_thread.is_alive():
            self.post_message({"type": "error", "message": "Training already running"})
            return
        self.stop_event.clear()
        self.pause_event.clear()
        self._update_status(isTraining=True, isPaused=False)
        self.training_thread = threading.Thread(
            target=_run_training,
            args=(self, config, device, total_timesteps),
            daemon=True,
        )
        self.training_thread.start()

    def pause_training(self):
        self.pause_event.set()
        self._update_status(isPaused=True)

    def resume_training(self):
        self.pause_event.clear()
        self._update_status(isPaused=False)

    def stop_training(self):
        self.stop_event.set()
        self._update_status(isTraining=False, isPaused=False)

    def reset_environment(self, goal: Optional[str] = None):
        with self._lock:
            self.pending_reset = goal
        self.post_message({"type": "reset", "goal": goal})
        logger.info("Environment reset requested with goal: %s", goal)


def _run_training(
    manager: TrainingManager,
    config_path: str,
    device: str,
    total_timesteps: Optional[int],
):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import yaml
    from env.minecraft_env import MinecraftEnv
    from env.intrinsic_motivation import IntrinsicMotivation
    from env.rnd_curiosity import RNDCuriosity
    from agent.ppo_agent import create_ppo_agent, linear_schedule
    from skills.autonomous_curriculum import AutonomousCurriculum

    with open(config_path) as f:
        config = yaml.safe_load(f)

    env_config = config.get("environment", {})
    training_config = config.get("training", {})
    callbacks_config = config.get("callbacks", {})

    goal_generator = AutonomousCurriculum()
    intrinsic = IntrinsicMotivation()

    # Only enable RND curiosity when configured, matching src/train.py logic.
    rnd_config = config.get("rnd_curiosity", {})
    rnd: Optional[RNDCuriosity] = None
    if rnd_config.get("enabled", True):
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

    env = None
    try:
        env = MissionControlEnvWrapper(
            MinecraftEnv(
                host=env_config.get("host", "localhost"),
                port=env_config.get("port", 9876),
                max_episode_steps=env_config.get("max_episode_steps", 6000),
                goal="survive_first_night",
                autonomous=True,
                goal_generator=goal_generator,
                intrinsic_motivation=intrinsic,
                rnd_curiosity=rnd,
                normalize_observations=training_config.get(
                    "normalize_observations", True
                ),
                frame_stack=training_config.get("frame_stack", 4),
                noop_bias=training_config.get("noop_bias", 0.01),
            ),
            manager,
        )
        manager.env = env

        lr_config = training_config.get("learning_rate", {})
        lr = lr_config.get("initial", 3e-4)
        if lr_config.get("schedule", "constant") == "linear":
            lr = linear_schedule(lr)

        ts = total_timesteps or training_config["total_timesteps"]
        model = create_ppo_agent(
            env=env,
            tensorboard_log="./checkpoints/logs/tensorboard",
            device=device,
            learning_rate=lr,
            n_steps=training_config.get("n_steps", 2048),
            batch_size=training_config.get("batch_size", 64),
            n_epochs=training_config.get("n_epochs", 10),
            gamma=training_config.get("gamma", 0.99),
            gae_lambda=training_config.get("gae_lambda", 0.95),
            clip_range=training_config.get("clip_range", 0.2),
            ent_coef=training_config.get("ent_coef", 0.01),
            verbose=0,
        )

        callback = DashboardCallback(
            manager, log_freq=callbacks_config.get("log_freq", 1000)
        )
        model.learn(total_timesteps=ts, callback=callback, progress_bar=False)
    except Exception as exc:
        logger.exception("Training failed")
        manager.post_message({"type": "error", "message": str(exc)})
    finally:
        manager._update_status(isTraining=False, isPaused=False)
        if env is not None:
            env.close()
