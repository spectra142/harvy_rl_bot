#!/usr/bin/env python3
"""Skill-level live test: sustain skill_gather_log and verify gathering rewards."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from env.minecraft_env import MinecraftEnv
from env.actions import ACTION_NAMES

SKILL_GATHER_LOG = ACTION_NAMES.index("skill_gather_log")


def main():
    env = MinecraftEnv(
        host="127.0.0.1", port=9876, max_episode_steps=500,
        goal="gather_logs", normalize_observations=False, frame_stack=1,
    )
    obs, info = env.reset(options={"goal": "gather_logs"})
    print(f"reset | pos={info['position']} inv={info['inventory_counts']}")

    total = 0.0
    for i in range(120):
        obs, reward, terminated, truncated, info = env.step(SKILL_GATHER_LOG)
        total += reward
        inv = info["inventory_counts"]
        logs = sum(v for k, v in inv.items() if k.endswith("_log"))
        if i % 10 == 0 or reward > 0.5:
            print(f"step {i+1:3d} | r={reward:+.3f} | logs={logs} "
                  f"mined={info['blocks_mined']} collected={info['resources_collected']} "
                  f"| exec={info.get('executed_action')} override={info.get('action_overridden')}")
        if terminated or truncated:
            print(f"ended: {info.get('termination_reason')}")
            break

    inv = info["inventory_counts"]
    print(f"=== final inventory: {inv} | total_reward={total:.2f} ===")

    # Test mid-episode reset: /kill + respawn should fire, no penalty artifact
    print("=== mid-episode reset ===")
    obs, info = env.reset(options={"goal": "survive"})
    print(f"reset ok | pos={info['position']} inv={info['inventory_counts']}")
    obs, reward, _, _, info = env.step(0)
    print(f"step after reset | r={reward:+.3f} (should be ~0, no /kill artifact)")
    env.close()


if __name__ == "__main__":
    main()
