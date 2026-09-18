"""Minecraft Gymnasium Environment -- bridges Python RL to Mineflayer bot."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import socket
import json
import base64
import threading
import time
import logging
from typing import Optional, Tuple, Dict, Any

from .observation import (
    build_observation,
    get_observation_space,
    goal_to_id,
    INVENTORY_SIZE,
)
from .rewards import RewardCalculator
from .intrinsic_motivation import IntrinsicMotivation
from .rnd_curiosity import RNDCuriosity
from .actions import action_name_to_json, get_action_space, ACTION_COUNT
from .observation_normalizer import ObservationNormalizer
from .frame_stack import FrameStack
from .knowledge_base import MinecraftKnowledgeBase

logger = logging.getLogger(__name__)


class MinecraftEnv(gym.Env):
    """Gymnasium environment for Minecraft RL training."""

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9876,
        max_episode_steps: int = 6000,
        skill_module: str = "survival",
        goal: str = "survive_first_night",
        render_mode: Optional[str] = None,
        autonomous: bool = False,
        goal_generator: Optional[Any] = None,
        intrinsic_motivation: Optional[IntrinsicMotivation] = None,
        rnd_curiosity: Optional[RNDCuriosity] = None,
        normalize_observations: bool = True,
        frame_stack: int = 4,
        noop_bias: float = 0.01,
    ):
        super().__init__()
        self.host = host
        self.port = port
        self.max_episode_steps = max_episode_steps
        self.skill_module = skill_module
        self.goal = goal
        self.render_mode = render_mode
        self.autonomous = autonomous
        self.goal_generator = goal_generator
        self.intrinsic_motivation = intrinsic_motivation or IntrinsicMotivation()
        self.rnd_curiosity = rnd_curiosity
        self.normalize_observations = normalize_observations
        self.frame_stack_n = frame_stack

        self.action_space = get_action_space()
        self.observation_space = get_observation_space()

        # Initialize frame stack first so the observation space reflects it.
        self.frame_stacker: Optional[FrameStack] = None
        if self.frame_stack_n > 1:
            self.frame_stacker = FrameStack(
                self.observation_space, n_frames=self.frame_stack_n
            )
            # Update the voxel observation space to include the temporal stack.
            self.observation_space.spaces["voxel_grid"] = spaces.Box(
                0.0, 255.0, (self.frame_stack_n, 11, 11, 7), dtype=np.float32
            )

        # Knowledge base for reward shaping and skill planning.
        self.knowledge_base = MinecraftKnowledgeBase()

        self.reward_calculator = RewardCalculator(
            skill_module,
            noop_bias=noop_bias,
            knowledge_base=self.knowledge_base,
            goal=goal,
        )
        self.sock: Optional[socket.socket] = None
        self._recv_buffer = b""
        self._send_lock = threading.Lock()
        self.current_step = 0
        self.episode_reward = 0.0
        self._last_raw_obs: Optional[dict] = None

        # Initialize observation normalizer after the space has been finalised.
        self.normalizer: Optional[ObservationNormalizer] = None
        if self.normalize_observations:
            self.normalizer = ObservationNormalizer(self.observation_space)

    def _connect(self) -> None:
        """Establish TCP connection to Mineflayer bot with retry."""
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        max_retries = 3
        backoff = 1.0
        for attempt in range(max_retries):
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(30.0)
            try:
                self.sock.connect((self.host, self.port))
                self._recv_buffer = b""
                logger.info(
                    f"Connected to bot at {self.host}:{self.port} (attempt {attempt + 1})"
                )
                return
            except (ConnectionRefusedError, socket.timeout) as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Connection attempt {attempt + 1} failed, retrying in {backoff}s..."
                    )
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    logger.error(
                        f"Failed to connect to bot after {max_retries} attempts: {e}"
                    )
                    raise

    def _mark_socket_dead(self) -> None:
        """Tear down the current socket so the next reset() reconnects.

        Called whenever a send/receive fails. We never try to resurrect a
        half-open TCP connection -- a clean reconnect is the only recovery
        that keeps the strict request/response protocol aligned.
        """
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        self._recv_buffer = b""

    def _send_action(self, action_idx: int) -> None:
        """Send action command to bot."""
        if not self.sock:
            return
        try:
            cmd = action_name_to_json(action_idx)
            msg = json.dumps(cmd) + "\n"
            with self._send_lock:
                self.sock.sendall(msg.encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logger.error(f"Failed to send action: {e}")
            self._mark_socket_dead()

    def _receive_observation(self) -> Optional[dict]:
        """Receive one observation from the bot.

        The bot speaks a strict request/response protocol: exactly one
        observation per action/reset we send. Any extra lines in the buffer
        (e.g. left over from a reconnect) are drained and the freshest one
        wins, so the agent never acts on stale state.
        """
        if not self.sock:
            return None
        try:
            while True:
                data = self.sock.recv(65536)
                if not data:
                    # Orderly shutdown from the bot side.
                    self._mark_socket_dead()
                    return None
                self._recv_buffer += data
                if b"\n" in self._recv_buffer:
                    lines = self._recv_buffer.split(b"\n")
                    self._recv_buffer = lines.pop()
                    latest = None
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            latest = json.loads(line.decode("utf-8", errors="replace"))
                        except json.JSONDecodeError:
                            continue
                    if latest is not None:
                        return latest
        except (socket.timeout, ConnectionResetError, OSError) as e:
            logger.error(f"Failed to receive observation: {e}")
            self._mark_socket_dead()
            return None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        """Reset environment -- starts new episode."""
        super().reset(seed=seed)

        self.current_step = 0
        self.episode_reward = 0.0

        # Reuse the socket across episodes. The bot accepts exactly one
        # Python client and rejects extras, so a close+reconnect per episode
        # can race the server's disconnect handling and get our fresh
        # connection destroyed. Only reconnect when the socket is actually
        # dead (first reset, or after a connection loss).
        if self.sock is None:
            try:
                self._connect()
            except Exception:
                # Return dummy observation if bot not available
                obs = self._dummy_observation()
                # Prime reward baselines even on the dummy path so the first
                # real episode after a bot outage doesn't see a phantom delta.
                self.reward_calculator.reset(obs)
                if self.frame_stacker is not None:
                    obs = self.frame_stacker.reset(obs)
                if self.normalizer is not None:
                    self.normalizer.update(obs)
                    obs = self.normalizer.normalize(obs)
                info = {
                    "goal": self.goal,
                    "health": 20.0,
                    "food": 20.0,
                    "position": [0.0, 64.0, 0.0],
                    "held_item": "",
                    "inventory_counts": {},
                }
                return obs, info

        # Pick goal: autonomous generator, explicit option, or fixed default
        if self.autonomous and self.goal_generator is not None:
            # Initial selection (will be refined after we receive state)
            goal = self.goal_generator.select_goal()
        elif options and "goal" in options:
            goal = options["goal"]
        else:
            goal = self.goal
        self.goal = goal
        self.reward_calculator.set_goal(goal)

        reset_cmd = {"action": "reset", "goal": goal}
        try:
            with self._send_lock:
                self.sock.sendall((json.dumps(reset_cmd) + "\n").encode("utf-8"))
        except OSError:
            self._mark_socket_dead()

        # Wait for first observation
        raw_obs = self._receive_observation()

        # If autonomous, re-select goal using actual game state
        if self.autonomous and self.goal_generator is not None and raw_obs:
            state = {
                "inventory": raw_obs.get("self", {}).get("inventory_counts", {}),
                "health": raw_obs.get("self", {}).get("health", 20.0),
                "food": raw_obs.get("self", {}).get("food", 20.0),
                "time_of_day": raw_obs.get("environment", {}).get("time_of_day", 6000),
                "danger_level": raw_obs.get("environment", {}).get("danger_level", 0.0),
                "nearby_hostiles": sum(
                    1 for e in raw_obs.get("nearby_entities", []) if e.get("hostile")
                ),
                "position": raw_obs.get("self", {}).get("position", [0, 64, 0]),
                "held_item": raw_obs.get("self", {}).get("held_item", ""),
            }
            new_goal = self.goal_generator.select_goal(state)
            if new_goal != self.goal:
                self.goal = new_goal
                self.reward_calculator.set_goal(new_goal)
                # Update the goal WITHOUT a second world reset (avoids
                # killing/respawning the bot twice per episode).
                goal_cmd = {"action": "set_goal", "goal": new_goal}
                try:
                    with self._send_lock:
                        self.sock.sendall((json.dumps(goal_cmd) + "\n").encode("utf-8"))
                except OSError:
                    self._mark_socket_dead()
                # Receive fresh observation for the new goal
                raw_obs = self._receive_observation()

        if raw_obs:
            obs = build_observation(raw_obs)
            self._last_raw_obs = raw_obs
        else:
            obs = self._dummy_observation()

        # Prime reward baselines from the actual starting state so items the
        # bot carries across episodes don't grant a spurious inventory jackpot.
        self.reward_calculator.reset(obs)

        # Apply frame stacking first, then normalization (normalizer stats match stacked shapes).
        if self.frame_stacker is not None:
            obs = self.frame_stacker.reset(obs)
        if self.normalizer is not None:
            self.normalizer.update(obs)
            obs = self.normalizer.normalize(obs)

        self_data = raw_obs.get("self", {}) if raw_obs else {}
        info = {
            "goal": self.goal,
            "health": self_data.get("health", 20.0),
            "food": self_data.get("food", 20.0),
            "position": list(self_data.get("position", [0.0, 64.0, 0.0])),
            "held_item": self_data.get("held_item", ""),
            "inventory_counts": dict(self_data.get("inventory_counts", {})),
        }
        return obs, info

    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        """Execute one step in the environment."""
        self.current_step += 1

        # Send action to bot
        action = int(np.clip(action, 0, ACTION_COUNT - 1))
        self._send_action(action)

        # Receive next observation
        raw_obs = self._receive_observation()

        if raw_obs is None:
            # Connection lost -- end episode. The socket is already marked
            # dead, so the next reset() reconnects transparently.
            obs = self._dummy_observation()
            if self.frame_stacker is not None:
                obs = self.frame_stacker.step(obs)
            if self.normalizer is not None:
                self.normalizer.update(obs)
                obs = self.normalizer.normalize(obs)
            reward = -100.0
            terminated = True
            truncated = False
            info = {"connection_lost": True, "termination_reason": "connection_lost"}
            return obs, reward, terminated, truncated, info

        self._last_raw_obs = raw_obs
        obs = build_observation(raw_obs)

        # Compute reward / termination using the raw (pre-normalization) observation.
        reward_signal = raw_obs.get("reward_signal", {})
        danger_level = float(obs.get("env_danger_level", [0.0])[0])
        extrinsic_reward = self.reward_calculator.compute(
            obs, reward_signal, action, danger_level
        )

        # Add intrinsic curiosity reward (count-based exploration)
        intrinsic_reward = self.intrinsic_motivation.compute(obs)

        # Add RND intrinsic reward (novelty-based exploration)
        rnd_reward = 0.0
        if self.rnd_curiosity is not None:
            rnd_reward = self.rnd_curiosity.compute(obs)

        reward = extrinsic_reward + intrinsic_reward + rnd_reward
        self.episode_reward += reward

        # Check termination using raw health from the bot. The death penalty
        # itself is applied once inside RewardCalculator via the death signal.
        terminated = False
        termination_reason = None
        health = raw_obs.get("self", {}).get("health", 20.0)
        if health <= 0:
            terminated = True
            termination_reason = "death"

        truncated = self.current_step >= self.max_episode_steps
        if truncated:
            termination_reason = "timeout"

        # If the episode ended, tell the autonomous goal generator how it went
        if (
            self.autonomous
            and self.goal_generator is not None
            and (terminated or truncated)
        ):
            success = not terminated  # Surviving the full episode = success
            self.goal_generator.record_episode(
                self.goal, success, reward=self.episode_reward
            )

        self_data = raw_obs.get("self", {})
        episode_stats = raw_obs.get("episode_stats", {})
        info = {
            "step": self.current_step,
            "episode_reward": self.episode_reward,
            "health": health,
            "food": self_data.get("food", 20.0),
            "goal": self.goal,
            "intrinsic_reward": intrinsic_reward,
            "extrinsic_reward": extrinsic_reward,
            "rnd_reward": rnd_reward,
            "termination_reason": termination_reason,
            "position": list(self_data.get("position", [0.0, 64.0, 0.0])),
            "held_item": self_data.get("held_item", ""),
            "inventory_counts": dict(self_data.get("inventory_counts", {})),
            # Episode-level stats reported by the bot (benchmarks/eval use these)
            "mobs_killed": episode_stats.get("mobsKilled", 0),
            "players_killed": episode_stats.get("playersKilled", 0),
            "blocks_mined": episode_stats.get("blocksMined", 0),
            "blocks_placed": episode_stats.get("blocksPlaced", 0),
            "items_crafted": episode_stats.get("itemsCrafted", 0),
            "resources_collected": episode_stats.get("resourcesCollected", 0),
            "pickaxe_crafted": bool(episode_stats.get("pickaxeCrafted", False)),
            "deaths": episode_stats.get("deaths", 0),
            # Honest action accounting (survival layer may have overridden)
            "executed_action": raw_obs.get("executed_action"),
            "action_overridden": bool(raw_obs.get("action_overridden", False)),
        }

        # Apply frame stacking and normalization AFTER reward computation.
        if self.frame_stacker is not None:
            obs = self.frame_stacker.step(obs)
        if self.normalizer is not None:
            self.normalizer.update(obs)
            obs = self.normalizer.normalize(obs)

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Render environment state (no-op for headless)."""
        if self._last_raw_obs is None:
            return
        if self.render_mode == "human":
            logger.info(
                f"Step {self.current_step} | "
                f"Health: {self._last_raw_obs.get('self', {}).get('health', 0):.1f} | "
                f"Reward: {self.episode_reward:.1f}"
            )

    def set_goal(self, goal: str) -> None:
        """Update the environment's active goal (used by curriculum callbacks)."""
        self.goal = goal
        self.reward_calculator.set_goal(goal)

    def set_reward_shaping(self, shaping: Optional[Dict[str, float]]) -> None:
        """Install additive per-signal reward shaping.

        Driven by CurriculumCallback in guided mode so each curriculum
        stage's shaping actually reaches the reward function.
        """
        self.reward_calculator.set_reward_shaping(shaping)

    def send_text_command(self, command: str) -> bool:
        """Send an out-of-band server command to the bot (no obs response).

        Used by Mission Control's execute_command. Returns True if the
        command was written to the socket.
        """
        if not self.sock:
            return False
        try:
            msg = json.dumps({"action": "server_command", "command": command}) + "\n"
            with self._send_lock:
                self.sock.sendall(msg.encode("utf-8"))
            return True
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logger.error(f"Failed to send server command: {e}")
            self._mark_socket_dead()
            return False

    def save_aux_state(self, prefix: str) -> None:
        """Persist auxiliary training state (normalizer stats, RND) to disk."""
        if self.normalizer is not None:
            self.normalizer.save(f"{prefix}.normalizer.pkl")
        if self.rnd_curiosity is not None:
            self.rnd_curiosity.save(f"{prefix}.rnd.pt")

    def load_aux_state(self, prefix: str) -> None:
        """Restore auxiliary training state saved by :meth:`save_aux_state`."""
        import os

        norm_path = f"{prefix}.normalizer.pkl"
        if self.normalizer is not None and os.path.exists(norm_path):
            self.normalizer.load(norm_path)
            logger.info(f"Loaded observation normalizer stats from {norm_path}")
        rnd_path = f"{prefix}.rnd.pt"
        if self.rnd_curiosity is not None and os.path.exists(rnd_path):
            self.rnd_curiosity.load(rnd_path)

    def close(self) -> None:
        """Close environment and socket connection."""
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def _dummy_observation(self) -> dict:
        """Return a dummy observation when bot is unavailable."""
        return {
            "self_health": np.array([20.0], dtype=np.float32),
            "self_food": np.array([20.0], dtype=np.float32),
            "self_armor": np.array([0.0], dtype=np.float32),
            "self_position": np.array([0.0, 64.0, 0.0], dtype=np.float32),
            "self_yaw": np.array([0.0], dtype=np.float32),
            "self_pitch": np.array([0.0], dtype=np.float32),
            "self_held_item_id": np.array([0.0], dtype=np.float32),
            "inventory": np.zeros((INVENTORY_SIZE,), dtype=np.float32),
            "entity_type_ids": np.zeros((16,), dtype=np.float32),
            "entity_distances": np.zeros((16, 1), dtype=np.float32),
            "entity_healths": np.zeros((16, 1), dtype=np.float32),
            "entity_hostiles": np.zeros((16, 1), dtype=np.float32),
            "danger_block_types": np.zeros((8,), dtype=np.float32),
            "danger_block_positions": np.zeros((8, 3), dtype=np.float32),
            "danger_block_distances": np.zeros((8, 1), dtype=np.float32),
            "poi_type_ids": np.zeros((16,), dtype=np.float32),
            "poi_distances": np.zeros((16, 1), dtype=np.float32),
            "poi_positions": np.zeros((16, 3), dtype=np.float32),
            "voxel_grid": np.zeros((11, 11, 7), dtype=np.float32),
            "voxel_hardness": np.zeros((11, 11, 7), dtype=np.float32),
            "voxel_tool_required": np.zeros((11, 11, 7), dtype=np.float32),
            "visited_chunks": np.zeros((64 * 3,), dtype=np.float32),
            "event_type_ids": np.zeros((10,), dtype=np.float32),
            "event_data": np.zeros((10,), dtype=np.float32),
            "env_time_of_day": np.array([6000.0], dtype=np.float32),
            "env_can_see_sky": np.array([1.0], dtype=np.float32),
            "env_in_water": np.array([0.0], dtype=np.float32),
            "env_on_ground": np.array([1.0], dtype=np.float32),
            "env_danger_level": np.array([0.0], dtype=np.float32),
            "goal_id": np.array([goal_to_id("survive_first_night")], dtype=np.float32),
        }
