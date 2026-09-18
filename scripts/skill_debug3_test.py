#!/usr/bin/env python3
"""Round-5: full craft chain with early-stop on success + balanced materials.

sticks -> table -> pickaxe -> equip -> gather_stone -> sword
"""

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


def step_until(env, action_name, n, pred, print_every=10):
    last = None
    for i in range(n):
        obs, r, term, trunc, info = env.step(act(action_name))
        last = info
        if (i + 1) % print_every == 0 or r > 0.5:
            print(f"  {action_name} {i+1:3d} r={r:+.2f} inv={info['inventory_counts']}",
                  flush=True)
        if pred(info):
            return last, True, i + 1
        if term or trunc:
            print(f"  episode ended: {info.get('termination_reason')}")
            break
    return last, False, n


def main():
    env = MinecraftEnv(
        host="127.0.0.1", port=9876, max_episode_steps=20000,
        goal="craft_pickaxe", normalize_observations=False, frame_stack=1,
    )
    obs, info = env.reset(options={"goal": "craft_pickaxe"})
    print(f"reset | inv={info['inventory_counts']}", flush=True)

    env.send_text_command("give harvy minecraft:oak_planks 12")
    time.sleep(1)
    env.step(0)

    info, ok, n = step_until(env, "skill_craft_sticks", 40,
                             lambda i: invc(i, "stick") >= 4)
    print(f"craft_sticks: {'PASS' if ok else 'FAIL'} ({invc(info, 'stick')} sticks, {n} steps)")

    info, ok, n = step_until(env, "skill_craft_crafting_table", 40,
                             lambda i: invc(i, "crafting_table") >= 1)
    print(f"craft_table: {'PASS' if ok else 'FAIL'} ({invc(info, 'crafting_table')}, {n} steps)")

    info, ok, n = step_until(env, "skill_craft_pickaxe", 100,
                             lambda i: invc(i, "_pickaxe") >= 1)
    print(f"craft_pickaxe: {'PASS' if ok else 'FAIL'} ({invc(info, '_pickaxe')}, {n} steps)")

    info, ok, n = step_until(env, "skill_equip_best_pickaxe", 20,
                             lambda i: "pickaxe" in str(i.get("held_item") or ""))
    print(f"equip_pickaxe: {'PASS' if ok else 'FAIL'} (held={info.get('held_item')})")

    info, ok, n = step_until(env, "skill_gather_stone", 150,
                             lambda i: invc(i, "cobblestone") >= 2)
    print(f"gather_stone: {'PASS' if ok else 'FAIL'} ({invc(info, 'cobblestone')} cobble, {n} steps)")

    info, ok, n = step_until(env, "skill_craft_sword", 100,
                             lambda i: invc(i, "_sword") >= 1)
    print(f"craft_sword: {'PASS' if ok else 'FAIL'} ({invc(info, '_sword')}, {n} steps)")

    env.close()


if __name__ == "__main__":
    main()
