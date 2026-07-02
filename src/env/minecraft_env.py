"""Gymnasium environment bridging Python RL to the Mineflayer bot."""

import gymnasium as gym
import socket
import json
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any

from .observation import build_observation, get_observation_space, MAX_ENTITIES, VOXEL_SHAPE
from .action_space import action_index_to_command, get_action_space, is_skill_action, LOW_LEVEL_ACTION_COUNT, ACTION_NAMES
from .rewards import RewardCalculator
from skills.skill_executor import SkillExecutor
from skills.combat_skill import CombatSkill
from skills.build_skill import BuildSkill
from skills.gather_skill import GatherSkill
from skills.flee_skill import FleeSkill

logger = logging.getLogger(__name__)


class MinecraftEnv(gym.Env):
    """Low-level Minecraft RL environment."""

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9876,
        max_episode_steps: int = 6000,
        tcp_timeout: float = 30.0,
        reward_config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
        knowledge_base: Any = None,
        replay_buffer: Any = None,
        trust_manager: Any = None,
        command_queue: Any = None,
    ):
        super().__init__()
        self.host = host
        self.port = port
        self.max_episode_steps = max_episode_steps
        self.tcp_timeout = tcp_timeout
        self.render_mode = render_mode
        self.knowledge_base = knowledge_base
        self.replay_buffer = replay_buffer
        self.trust_manager = trust_manager
        self.command_queue = command_queue

        self.action_space = get_action_space()
        self.observation_space = get_observation_space()

        reward_config = reward_config or {}
        self.reward_calculator = RewardCalculator(**reward_config)

        self.sock: Optional[socket.socket] = None
        self.current_step = 0
        self.episode_reward = 0.0
        self._last_obs: Optional[Dict[str, np.ndarray]] = None
        self._last_raw_obs: Optional[Dict[str, Any]] = None
        self._last_info: Optional[Dict[str, Any]] = None
        self._recv_buffer = b""
        self._recv_queue: list = []
        self._episode_buffer: list = []

        # Skill framework.
        self.skill_executor = SkillExecutor([CombatSkill(), BuildSkill(), GatherSkill(), FleeSkill()])

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------
    def _connect(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.tcp_timeout)
        try:
            self.sock.connect((self.host, self.port))
            logger.info("Connected to bot at %s:%s", self.host, self.port)
        except (ConnectionRefusedError, socket.timeout) as e:
            logger.error("Failed to connect to bot: %s", e)
            raise

    def _send(self, payload: Dict[str, Any]) -> None:
        if not self.sock:
            return
        try:
            msg = json.dumps(payload) + "\n"
            self.sock.sendall(msg.encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logger.error("Failed to send: %s", e)

    def _recv(self) -> Optional[Dict[str, Any]]:
        if not self.sock:
            return None

        # Drain any complete messages already queued.
        while self._recv_queue:
            payload = self._recv_queue.pop(0)
            if payload is not None:
                return payload

        try:
            while True:
                data = self.sock.recv(8192)
                if not data:
                    return None
                self._recv_buffer += data
                while b"\n" in self._recv_buffer:
                    line, self._recv_buffer = self._recv_buffer.split(b"\n", 1)
                    try:
                        payload = json.loads(line.decode("utf-8", errors="replace"))
                        self._recv_queue.append(payload)
                    except json.JSONDecodeError:
                        continue
                if self._recv_queue:
                    return self._recv_queue.pop(0)
        except (socket.timeout, ConnectionResetError, OSError) as e:
            logger.error("Failed to receive: %s", e)
            return None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        self.reward_calculator.reset()
        self.current_step = 0
        self.episode_reward = 0.0
        self._episode_buffer = []
        self.skill_executor.reset()

        self._connect()

        # Tell the bot to reset control state.
        self._send({"reset": True, "seq": 0})
        raw = self._recv()

        if raw is None:
            raise ConnectionError(f"No response from bot at {self.host}:{self.port}")

        obs = build_observation(raw)
        if obs is None:
            obs = self._dummy_observation()
        self._last_obs = obs
        self._last_raw_obs = raw
        info = self._build_info(raw)
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        action = int(np.clip(action, 0, self.action_space.n - 1))

        # Allow web/owner commands to override the RL action for one step.
        action = self._poll_command(action)

        # Resolve high-level skill actions to low-level commands.
        low_action = action
        skill_reward = 0.0
        skill_done = False
        skill_info: Dict[str, Any] = {}
        if self.skill_executor.is_skill_action(action) or self.skill_executor.active_skill:
            if self._last_obs is not None and self._last_raw_obs is not None:
                low_action, skill_reward, skill_done, skill_info = self.skill_executor.step(
                    action, self._last_obs, self._last_raw_obs
                )

        if self.sock:
            self._send({"action": action_index_to_command(low_action), "seq": self.current_step})
            raw = self._recv()
        else:
            raw = None

        if raw is None:
            obs = self._dummy_observation()
            reward = -1.0
            terminated = True
            truncated = False
            info = {"connection_lost": True, "termination_reason": "connection_lost"}
            return obs, reward, terminated, truncated, info

        obs = build_observation(raw)
        if obs is None:
            obs = self._dummy_observation()
        self._last_obs = obs
        self._last_raw_obs = raw

        reward_signal = raw.get("reward_signal", {})
        reward = self.reward_calculator.compute(obs, reward_signal) + skill_reward
        self.episode_reward += reward

        terminated = raw.get("terminated", False)
        truncated = self.current_step >= self.max_episode_steps

        info = self._build_info(raw)
        info["step"] = self.current_step
        info["episode_reward"] = self.episode_reward
        info["reward_signal"] = reward_signal
        info["termination_reason"] = (
            "death" if terminated else ("timeout" if truncated else None)
        )
        info["skill_info"] = skill_info
        info["hostiles_killed"] = reward_signal.get("hostiles_killed", 0)
        if skill_done:
            info["skill_done"] = True

        # Persist memory.
        self._record_chat(raw)
        if terminated:
            self._record_death(raw)
            self._flush_episode()
        elif truncated:
            self._flush_episode()

        if self.replay_buffer is not None:
            self._episode_buffer.append(
                {
                    "step": self.current_step,
                    "action": action,
                    "reward": reward,
                    "health": float(obs["self_health"][0]),
                    "position": [float(x) for x in obs["self_position"]],
                }
            )

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode != "human" or self._last_raw_obs is None:
            return
        obs = self._last_raw_obs.get("obs", {}).get("self", {})
        logger.info(
            "Step %s | Health %.1f | Reward %.2f",
            self.current_step,
            obs.get("health", 0),
            self.episode_reward,
        )

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_info(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        obs = raw.get("obs", {})
        self_data = obs.get("self", {})
        return {
            "health": self_data.get("health", 20.0),
            "food": self_data.get("food", 20.0),
            "position": list(self_data.get("position", [0.0, 64.0, 0.0])),
            "held_item": self_data.get("held_item", "air"),
            "inventory_counts": dict(self_data.get("inventory_counts", {})),
            "overridden": raw.get("overridden", False),
        }

    def _record_chat(self, raw: Dict[str, Any]) -> None:
        chat = raw.get("chat_event")
        if not chat:
            return
        if self.trust_manager is not None:
            response = self.trust_manager.handle_chat(chat)
            if response and self.command_queue is not None:
                self.command_queue.push("chat_response", response, issued_by="bot")
        elif self.knowledge_base is not None:
            self.knowledge_base.upsert_player(
                uuid=chat.get("uuid", chat.get("username", "unknown")),
                name=chat.get("username", "unknown"),
                position=chat.get("position"),
            )

    def _poll_command(self, action: int) -> int:
        if self.command_queue is None:
            return action
        cmd = self.command_queue.pop()
        if not cmd:
            return action
        name = cmd.get("command", "").lower()
        logger.info("Executing queued command: %s", name)
        if name == "attack_nearest":
            return ACTION_NAMES.index("skill_combat")
        if name == "build_wall":
            return ACTION_NAMES.index("skill_build")
        if name in ("stop", "noop"):
            return 0
        return action

    def _record_death(self, raw: Dict[str, Any]) -> None:
        if self.knowledge_base is None:
            return
        death_context = raw.get("death_context") or {}
        self.knowledge_base.record_death(death_context)

    def _flush_episode(self) -> None:
        if self.replay_buffer is not None and self._episode_buffer:
            self.replay_buffer.store(
                self._episode_buffer,
                self.episode_reward,
                skill_name="",
            )
        self._episode_buffer = []

    def _dummy_observation(self) -> Dict[str, np.ndarray]:
        space = self.observation_space
        return {
            "self_health": np.array([20.0], dtype=np.float32),
            "self_food": np.array([20.0], dtype=np.float32),
            "self_armor": np.array([0.0], dtype=np.float32),
            "self_position": np.array([0.0, 64.0, 0.0], dtype=np.float32),
            "self_yaw": np.array([0.0], dtype=np.float32),
            "self_pitch": np.array([0.0], dtype=np.float32),
            "self_held_item_id": np.array([0.0], dtype=np.float32),
            "inventory": np.zeros(space.spaces["inventory"].shape[0], dtype=np.float32),
            "entity_type_ids": np.zeros(MAX_ENTITIES, dtype=np.float32),
            "entity_distances": np.zeros((MAX_ENTITIES, 1), dtype=np.float32),
            "entity_healths": np.zeros((MAX_ENTITIES, 1), dtype=np.float32),
            "entity_hostiles": np.zeros((MAX_ENTITIES, 1), dtype=np.float32),
            "entity_positions": np.zeros((MAX_ENTITIES, 3), dtype=np.float32),
            "voxel_grid": np.zeros(VOXEL_SHAPE, dtype=np.float32),
            "env_time_of_day": np.array([6000.0], dtype=np.float32),
            "env_can_see_sky": np.array([1.0], dtype=np.float32),
            "env_in_water": np.array([0.0], dtype=np.float32),
            "env_on_ground": np.array([1.0], dtype=np.float32),
            "env_danger_level": np.array([0.0], dtype=np.float32),
        }

    def _dummy_info(self) -> Dict[str, Any]:
        return {
            "health": 20.0,
            "food": 20.0,
            "position": [0.0, 64.0, 0.0],
            "held_item": "air",
            "inventory_counts": {},
            "overridden": False,
            "connection_lost": True,
            "termination_reason": "connection_lost",
        }
