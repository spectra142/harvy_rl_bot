#!/usr/bin/env python3
"""Focused debug for the failing skills: craft_table, attack_hostile, eat_food, build_shelter.

Uses server commands (bot is opped) to set up preconditions so each skill
can be tested in isolation with per-step inventory/state prints.
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


def step_n(env, action_name, n, print_every=1, watch=None):
    last = None
    for i in range(n):
        obs, r, term, trunc, info = env.step(act(action_name))
        last = info
        extra = watch(info) if watch else ""
        if i % print_every == 0:
            print(f"  {action_name} step {i+1:3d} r={r:+.2f} inv={info['inventory_counts']} {extra}",
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
    print(f"reset | inv={info['inventory_counts']}", flush=True)

    # --- 1. craft_table chain with full visibility ---
    print("\n== PHASE 1: craft_table chain ==", flush=True)
    step_n(env, "skill_gather_log", 80, print_every=20)
    step_n(env, "skill_craft_planks", 20)
    info = step_n(env, "skill_craft_crafting_table", 40)
    table_ok = invc(info, "crafting_table") >= 1
    print(f"craft_table: {'PASS' if table_ok else 'FAIL'}")

    # continue chain if table worked
    if table_ok:
        step_n(env, "skill_craft_sticks", 30)
        info = step_n(env, "skill_craft_pickaxe", 60)
        print(f"craft_pickaxe: {'PASS' if invc(info, '_pickaxe') >= 1 else 'FAIL'}")

    # --- 2. attack_hostile with summoned zombie ---
    print("\n== PHASE 2: attack_hostile (summon zombie) ==", flush=True)
    env.send_text_command("summon minecraft:zombie ~3 ~ ~3")
    time.sleep(1)
    info = step_n(env, "skill_attack_hostile", 150, print_every=25,
                  watch=lambda i: f"kills={i.get('mobs_killed')}")
    print(f"attack_hostile: {'PASS' if info.get('mobs_killed', 0) >= 1 else 'FAIL'}")

    # --- 3. eat_food with given bread + hunger effect ---
    print("\n== PHASE 3: eat_food (give bread + hunger) ==", flush=True)
    env.send_text_command("give harvy minecraft:bread 3")
    env.send_text_command("effect give harvy minecraft:hunger 10 40")
    time.sleep(3)
    obs, r, t, tr, info = env.step(0)  # noop to refresh obs
    food_before = info["food"]
    info = step_n(env, "skill_eat_food", 60, print_every=10,
                  watch=lambda i: f"food={i.get('food')}")
    print(f"eat_food: {'PASS' if info['food'] > food_before else 'FAIL'} "
          f"(food {food_before} -> {info['food']})")

    # --- 4. build_shelter with given cobblestone ---
    print("\n== PHASE 4: build_shelter (give cobblestone) ==", flush=True)
    env.send_text_command("give harvy minecraft:cobblestone 32")
    time.sleep(1)
    info = step_n(env, "skill_build_shelter", 250, print_every=50,
                  watch=lambda i: f"placed={i.get('blocks_placed')}")
    print(f"build_shelter: {'PASS' if info.get('blocks_placed', 0) >= 8 else 'FAIL'}")

    env.close()


if __name__ == "__main__":
    main()
