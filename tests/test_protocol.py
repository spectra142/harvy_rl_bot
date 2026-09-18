"""Protocol-level tests: a fake bot server modelling the real bot's
strict request/response wire protocol (one observation per action/reset),
which is where the worst production bugs lived."""

import json
import socket
import threading

import numpy as np
import pytest

from env.minecraft_env import MinecraftEnv


def _fake_observation(goal="survive", step=0):
    return {
        "self": {
            "health": 20.0,
            "food": 20.0,
            "armor": 0,
            "position": [float(step), 64.0, 0.0],
            "yaw": 0.0,
            "pitch": 0.0,
            "held_item": None,
            "inventory_counts": {"oak_log": step},
        },
        "nearby_entities": [],
        "nearby_blocks": [],
        "nearby_pois": [],
        "voxel_grid": "",
        "voxel_hardness": "",
        "voxel_tool_required": "",
        "visited_chunks": [],
        "recent_events": [],
        "environment": {
            "time_of_day": 6000,
            "can_see_sky": True,
            "in_water": False,
            "on_ground": True,
            "danger_level": 0.0,
        },
        "goal": goal,
        "reward_signal": {
            "alive_tick": 1,
            "damage_taken": 0.0,
            "mob_killed": 0,
            "player_killed": 0,
            "item_mined": 0,
            "item_lost": 0,
            "item_crafted": 0,
            "block_placed": 0,
            "shelter_complete": 0,
            "distance_moved": 1.0,
            "block_broken_name": None,
            "food_eaten": 0,
            "death": 0,
        },
        "episode_stats": {
            "mobsKilled": 0,
            "playersKilled": 0,
            "blocksMined": 0,
            "blocksPlaced": 0,
            "itemsCrafted": 0,
            "resourcesCollected": step,
            "pickaxeCrafted": False,
            "deaths": 0,
        },
        "executed_action": "noop",
        "action_overridden": False,
        "skill_status": {"active_skill": None, "skill_state": "idle", "skill_progress": 0},
    }


class FakeBotServer:
    """Minimal fake of mineflayer_bot.js: replies with exactly one
    observation per action/reset line received, and ignores
    server_command lines (no reply)."""

    def __init__(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(1)
        self.port = self.server.getsockname()[1]
        self.received = []
        self._conn = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        conn, _ = self.server.accept()
        self._conn = conn
        conn.settimeout(5.0)
        buffer = b""
        step = 0
        while not self._stop.is_set():
            try:
                data = conn.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                break
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue
                msg = json.loads(line.decode())
                self.received.append(msg)
                if msg.get("action") == "server_command":
                    continue  # no response, like the real bot
                if msg.get("action") == "reset":
                    step = 0  # episode stats reset, like the real bot
                else:
                    step += 1
                goal = msg.get("goal", "survive")
                obs = _fake_observation(goal=goal, step=step)
                conn.sendall((json.dumps(obs) + "\n").encode())
        try:
            conn.close()
        except OSError:
            pass

    def close(self):
        self._stop.set()
        if self._conn is not None:
            try:
                self._conn.shutdown(socket.SHUT_RDWR)
                self._conn.close()
            except OSError:
                pass
        try:
            self.server.close()
        except OSError:
            pass
        self._thread.join(timeout=7.0)


@pytest.fixture
def fake_bot():
    server = FakeBotServer()
    yield server
    server.close()


def test_request_response_protocol(fake_bot):
    """reset() and each step() must yield exactly one fresh observation."""
    env = MinecraftEnv(
        host="127.0.0.1",
        port=fake_bot.port,
        max_episode_steps=10,
        normalize_observations=False,
        frame_stack=1,
    )
    obs, info = env.reset(options={"goal": "explore"})
    assert info["goal"] == "explore"

    for i in range(5):
        obs, reward, terminated, truncated, info = env.step(1)
        assert not terminated
        assert info["health"] == 20.0
        # info carries episode stats forwarded from the bot
        assert info["resources_collected"] == i + 1
        assert info["executed_action"] == "noop"

    env.close()
    # 1 reset + 5 actions = 6 messages; every one got exactly one reply
    # (no stale-queue buildup, no dropped rewards).
    assert len(fake_bot.received) == 6


def test_server_command_does_not_desync(fake_bot):
    """A server_command between steps must not shift the obs stream."""
    env = MinecraftEnv(
        host="127.0.0.1",
        port=fake_bot.port,
        max_episode_steps=10,
        normalize_observations=False,
        frame_stack=1,
    )
    env.reset(options={"goal": "survive"})
    assert env.send_text_command("time set day") is True
    obs, reward, terminated, truncated, info = env.step(1)
    assert info["resources_collected"] == 1  # still 1:1 aligned
    env.close()


def test_connection_loss_terminates(fake_bot):
    env = MinecraftEnv(
        host="127.0.0.1",
        port=fake_bot.port,
        max_episode_steps=10,
        normalize_observations=False,
        frame_stack=1,
    )
    env.reset(options={"goal": "survive"})
    fake_bot.close()  # kill the "bot"
    obs, reward, terminated, truncated, info = env.step(1)
    assert terminated is True
    assert info["termination_reason"] == "connection_lost"
    env.close()
