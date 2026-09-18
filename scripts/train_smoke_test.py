#!/usr/bin/env python3
"""Tiny end-to-end training smoke test against the live bot (not a real run)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from env.minecraft_env import MinecraftEnv
from env.intrinsic_motivation import IntrinsicMotivation
from skills.autonomous_curriculum import AutonomousCurriculum
from agent.ppo_agent import create_ppo_agent


def main():
    goal_generator = AutonomousCurriculum()
    env = MinecraftEnv(
        host="127.0.0.1", port=9876, max_episode_steps=100,
        autonomous=True, goal_generator=goal_generator,
        intrinsic_motivation=IntrinsicMotivation(),
        normalize_observations=True, frame_stack=4,
    )
    model = create_ppo_agent(
        env, device="cpu", verbose=0, n_steps=32, batch_size=32, n_epochs=2,
    )
    print("learning 64 steps (2 tiny PPO updates)...", flush=True)
    model.learn(total_timesteps=64)
    print("learn OK")
    # Save + aux state roundtrip
    out = Path("/tmp/harvy_smoke")
    out.mkdir(exist_ok=True)
    model.save(str(out / "model.zip"))
    env.save_aux_state(str(out / "model"))
    print("saved:", sorted(p.name for p in out.iterdir()))
    env.close()
    print("AUTONOMOUS SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
