#!/usr/bin/env python3
"""Round-2 focused live test on fresh terrain: full craft chain, hostile kill
attribution, shelter building. Requires a DEBUG=1 bot for log visibility."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from env.minecraft_env import MinecraftEnv
from env.actions import ACTION_NAMES


def act(name):
    return ACTION_NAMES.index(name)


def invc(info, *suffixes):
    inv = info.get("inventory_counts", {})
    return sum(v for k, v in inv.items() if any(k.endswith(s) for s in suffixes))


def step_n(env, action_name, n, print_every=10, watch=None):
    last = None
    for i in range(n):
        obs, r, term, trunc, info = env.step(act(action_name))
        last = info
        extra = watch(info) if watch else ""
        if (i + 1) % print_every == 0 or r > 0.5:
            print(f"  {action_name} {i+1:3d} r={r:+.2f} inv={info['inventory_counts']} {extra}",
                  flush=True)
        if term or trunc:
            print(f"  episode ended: {info.get('termination_reason')}")
            break
    return last


def main():
    env = MinecraftEnv(
        host="127.0.0.1", port=9876, max_episode_steps=20000,
        goal="craft_pickaxe", normalize_observations=False, frame_stack=1,
    )
    obs, info = env.reset(options={"goal": "craft_pickaxe"})
    print(f"reset | pos={info['position']} inv={info['inventory_counts']}", flush=True)

    # Fresh terrain: teleport far out so trees are untouched
    env.send_text_command("tp harvy 5000 80 5000")
    time.sleep(4)
    obs, r, t, tr, info = env.step(0)
    print(f"teleported | pos={info['position']} health={info['health']}", flush=True)

    print("\n== craft chain ==", flush=True)
    info = step_n(env, "skill_gather_log", 200)
    logs = invc(info, "_log")
    print(f"gather_log: {'PASS' if logs >= 3 else 'FAIL'} ({logs} logs)")
    info = step_n(env, "skill_craft_planks", 30)
    planks = invc(info, "_planks")
    print(f"craft_planks: {'PASS' if planks >= 4 else 'FAIL'} ({planks} planks)")
    info = step_n(env, "skill_craft_crafting_table", 40)
    table = invc(info, "crafting_table")
    print(f"craft_table: {'PASS' if table >= 1 else 'FAIL'} ({table})")
    info = step_n(env, "skill_craft_sticks", 30)
    sticks = invc(info, "stick")
    print(f"craft_sticks: {'PASS' if sticks >= 2 else 'FAIL'} ({sticks})")
    info = step_n(env, "skill_craft_pickaxe", 60)
    pick = invc(info, "_pickaxe")
    print(f"craft_pickaxe: {'PASS' if pick >= 1 else 'FAIL'} ({pick})")

    print("\n== attack_hostile ==", flush=True)
    env.send_text_command("summon minecraft:zombie ~3 ~ ~3")
    time.sleep(1)
    info = step_n(env, "skill_attack_hostile", 150, print_every=25,
                  watch=lambda i: f"kills={i.get('mobs_killed')}")
    print(f"attack_hostile: {'PASS' if info.get('mobs_killed', 0) >= 1 else 'FAIL'} "
          f"(mobs_killed={info.get('mobs_killed')})")

    print("\n== build_shelter ==", flush=True)
    env.send_text_command("give harvy minecraft:cobblestone 32")
    time.sleep(1)
    info = step_n(env, "skill_build_shelter", 250, print_every=25,
                  watch=lambda i: f"placed={i.get('blocks_placed')}")
    print(f"build_shelter: {'PASS' if info.get('blocks_placed', 0) >= 8 else 'FAIL'} "
          f"(placed={info.get('blocks_placed')})")

    env.close()


if __name__ == "__main__":
    main()
