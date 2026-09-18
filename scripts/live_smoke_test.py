#!/usr/bin/env python3
"""Live smoke test: bot + env + network forward pass against the real server."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from env.minecraft_env import MinecraftEnv
from env.actions import ACTION_NAMES


def main():
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    env = MinecraftEnv(
        host="127.0.0.1",
        port=9876,
        max_episode_steps=200,
        goal="gather_logs",
        normalize_observations=True,
        frame_stack=4,
    )

    print("=== reset ===")
    t0 = time.time()
    obs, info = env.reset(options={"goal": "gather_logs"})
    print(f"reset ok in {time.time()-t0:.2f}s | goal={info['goal']} "
          f"health={info['health']} pos={info['position']}")
    print(f"voxel_grid shape={obs['voxel_grid'].shape} "
          f"range=[{obs['voxel_grid'].min():.0f},{obs['voxel_grid'].max():.0f}] "
          f"(255=unknown block is fine)")
    print(f"goal_id={obs['goal_id']}")

    # Verify obs fits the declared space
    assert env.observation_space.contains(
        {k: np.asarray(v) for k, v in obs.items()}
    ), "observation does not fit space!"

    # Network forward pass: the old normalizer bug crashed here
    print("=== PPO forward pass ===")
    from agent.ppo_agent import create_ppo_agent
    model = create_ppo_agent(env, device="cpu", verbose=0)
    action, _ = model.predict(obs, deterministic=False)
    print(f"predict ok -> action {action} ({ACTION_NAMES[int(action)]})")

    print(f"=== {n_steps} live steps ===")
    total_reward = 0.0
    actions_to_try = [1, 7, 25, 1, 1]  # forward, jump, skill_gather_log, ...
    for i in range(n_steps):
        if i < len(actions_to_try):
            action = actions_to_try[i]
        else:
            action = int(np.random.randint(0, 8))
        t0 = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"step {i+1:3d} | {ACTION_NAMES[action]:<22s} -> "
              f"{str(info.get('executed_action')):<22s} "
              f"r={reward:+.3f} health={info['health']:.0f} "
              f"food={info['food']:.0f} pos={[round(p,1) for p in info['position']]} "
              f"({(time.time()-t0)*1000:.0f}ms)")
        if terminated or truncated:
            print(f"episode ended: {info.get('termination_reason')}")
            break

    print(f"=== done: total_reward={total_reward:.2f} ===")
    print(f"episode stats: mobs_killed={info.get('mobs_killed')} "
          f"blocks_mined={info.get('blocks_mined')} "
          f"resources_collected={info.get('resources_collected')}")
    env.close()


if __name__ == "__main__":
    main()
