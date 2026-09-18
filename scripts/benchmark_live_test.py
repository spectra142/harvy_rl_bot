#!/usr/bin/env python3
"""Live benchmark-harness test: runs two benchmarks against the real server
with a scripted policy (not PPO) to verify the scoring pipeline end-to-end.

  - gather_16_logs: policy always picks skill_gather_log -> verifies the
    inventory_counts-based scoring (was permanently 0 before the fix).
  - kill_hostile: summons a zombie, policy picks skill_attack_hostile ->
    verifies info['mobs_killed'] scoring.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from env.minecraft_env import MinecraftEnv
from env.actions import ACTION_NAMES
from eval_benchmarks import BenchmarkSuite


class ScriptedModel:
    """Predicts one fixed skill action forever (per benchmark)."""

    def __init__(self, skill_name):
        self.action = ACTION_NAMES.index(skill_name)

    def predict(self, obs, deterministic=True):
        return self.action, None


class PostResetCommandEnv:
    """Wraps MinecraftEnv: after every reset, issues server commands (teleport
    to fresh terrain / summon a mob). Benchmark episodes always reset the bot
    to world spawn, so setup must happen post-reset, not before."""

    def __init__(self, env, commands):
        self._env = env
        self._commands = commands

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        for cmd in self._commands:
            self._env.send_text_command(cmd)
        time.sleep(2.5)
        return obs, info


def main():
    suite = BenchmarkSuite(output_dir="./benchmarks_live_test")
    # Speed: cap the scripted episodes (scoring path is identical, just shorter)
    for name in ("gather_16_logs", "kill_hostile"):
        suite.BENCHMARKS[name] = {**suite.BENCHMARKS[name], "max_steps": 1000}

    env = MinecraftEnv(
        host="127.0.0.1", port=9876, max_episode_steps=20000,
        goal="gather_logs", normalize_observations=False, frame_stack=1,
    )

    print("== gather_16_logs (scripted gather policy, 1 episode) ==", flush=True)
    gather_env = PostResetCommandEnv(env, ["tp harvy 5500 80 5500"])
    result = suite.run_benchmark(
        "gather_16_logs", gather_env, ScriptedModel("skill_gather_log"), n_episodes=1
    )
    print(f"score={result.score:.3f} success={result.success} "
          f"raw_value={result.raw_value} details={result.details}")

    print("\n== kill_hostile (scripted attack policy, 1 episode) ==", flush=True)
    fight_env = PostResetCommandEnv(env, ["summon minecraft:zombie ~5 ~ ~5"])
    result = suite.run_benchmark(
        "kill_hostile", fight_env, ScriptedModel("skill_attack_hostile"), n_episodes=1
    )
    print(f"score={result.score:.3f} success={result.success} "
          f"raw_value={result.raw_value} details={result.details}")

    # is_best_checkpoint sanity: best_scores were updated by run_benchmark;
    # a WORSE result must not be flagged best, a BETTER one must be.
    import copy
    worse = copy.deepcopy(result)
    worse.score = -1.0
    better = copy.deepcopy(result)
    better.score = result.score + 1.0
    results_map = {"kill_hostile": worse}
    print(f"\nis_best_checkpoint(worse) = {suite.is_best_checkpoint(results_map)} (expect False)")
    results_map = {"kill_hostile": better}
    print(f"is_best_checkpoint(better) = {suite.is_best_checkpoint(results_map)} (expect True)")

    env.close()


if __name__ == "__main__":
    main()
