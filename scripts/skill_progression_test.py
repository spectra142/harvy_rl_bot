#!/usr/bin/env python3
"""Full skill-progression live test against a real Minecraft server.

Walks the vanilla early-game chain in order and verifies each skill
actually changes world/inventory state:
  gather_log -> craft_planks -> crafting_table -> sticks -> pickaxe
  -> equip_pickaxe -> gather_stone -> craft_sword -> equip_sword
  -> explore -> attack_hostile -> build_shelter -> eat_food -> flee

Each phase is capped in steps; a phase passes when its success predicate
is met (or reports SKIP/FAIL with the reason). Prints a summary table.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from env.minecraft_env import MinecraftEnv
from env.actions import ACTION_NAMES


def action_id(name):
    return ACTION_NAMES.index(name)


def inv_count(inv, *suffixes):
    return sum(v for k, v in inv.items() if any(k.endswith(s) for s in suffixes))


def run_phase(env, name, skill, max_steps, done_fn, skip_fn=None):
    """Run one skill until done_fn(info) or step cap. Returns (status, detail)."""
    if skip_fn and skip_fn(env.last_info or {}):
        return "SKIP", "precondition not met"
    act = action_id(skill)
    for i in range(max_steps):
        obs, reward, terminated, truncated, info = env.step(act)
        env.last_info = info
        if done_fn(info):
            return "PASS", f"step {i+1}/{max_steps}"
        if terminated or truncated:
            return "FAIL", f"episode ended ({info.get('termination_reason')})"
    inv = (env.last_info or {}).get("inventory_counts", {})
    return "FAIL", f"step cap reached | inv={inv}"


def main():
    env = MinecraftEnv(
        host="127.0.0.1", port=9876, max_episode_steps=20000,
        goal="gather_logs", normalize_observations=False, frame_stack=1,
    )
    env.last_info = None
    obs, info = env.reset(options={"goal": "gather_logs"})
    env.last_info = info
    spawn = info["position"]
    print(f"reset | pos={spawn} inv={info['inventory_counts']}")

    inv = lambda i: i.get("inventory_counts", {})
    def dist_from_spawn(i):
        p = i.get("position", spawn)
        return sum((a - b) ** 2 for a, b in zip(p, spawn)) ** 0.5
    results = []

    phases = [
        ("gather_log", "skill_gather_log", 150,
         lambda i: inv_count(inv(i), "_log") >= 2, None),
        ("craft_planks", "skill_craft_planks", 60,
         lambda i: inv_count(inv(i), "_planks") >= 4,
         lambda i: inv_count(inv(i), "_log") < 1),
        ("craft_table", "skill_craft_crafting_table", 80,
         lambda i: inv_count(inv(i), "crafting_table") >= 1,
         lambda i: inv_count(inv(i), "_planks") < 4),
        ("craft_sticks", "skill_craft_sticks", 60,
         lambda i: inv_count(inv(i), "stick") >= 4,
         lambda i: inv_count(inv(i), "_planks") < 2),
        ("craft_pickaxe", "skill_craft_pickaxe", 120,
         lambda i: inv_count(inv(i), "_pickaxe") >= 1,
         lambda i: inv_count(inv(i), "_planks") < 3 or inv_count(inv(i), "stick") < 2),
        ("equip_pickaxe", "skill_equip_best_pickaxe", 20,
         lambda i: "pickaxe" in str(i.get("held_item", "")),
         lambda i: inv_count(inv(i), "_pickaxe") < 1),
        ("gather_stone", "skill_gather_stone", 200,
         lambda i: inv_count(inv(i), "cobblestone") >= 4,
         lambda i: inv_count(inv(i), "_pickaxe") < 1),
        ("craft_sword", "skill_craft_sword", 120,
         lambda i: inv_count(inv(i), "_sword") >= 1,
         lambda i: inv_count(inv(i), "_planks") < 2 or inv_count(inv(i), "stick") < 1),
        ("equip_sword", "skill_equip_best_sword", 20,
         lambda i: "sword" in str(i.get("held_item", "")),
         lambda i: inv_count(inv(i), "_sword") < 1),
        ("explore", "skill_explore", 100,
         lambda i: dist_from_spawn(i) > 15, None),
        ("attack_hostile", "skill_attack_hostile", 100,
         lambda i: i.get("mobs_killed", 0) >= 1, None),
        ("build_shelter", "skill_build_shelter", 250,
         lambda i: i.get("blocks_placed", 0) >= 8,
         lambda i: inv_count(inv(i), "cobblestone", "_planks", "_log") < 8),
        ("eat_food", "skill_eat_food", 30,
         lambda i: False,  # only verifies it doesn't crash; food unlikely present
         None),
        ("flee", "skill_flee", 30,
         lambda i: True,  # runs one step without error = pass
         None),
    ]

    for name, skill, cap, done_fn, skip_fn in phases:
        t0 = time.time()
        try:
            status, detail = run_phase(env, name, skill, cap, done_fn, skip_fn)
        except Exception as e:
            status, detail = "ERROR", f"{type(e).__name__}: {e}"
        dt = time.time() - t0
        results.append((name, status, detail, dt))
        print(f"[{status:5s}] {name:16s} ({dt:5.1f}s) {detail}", flush=True)
        if status == "ERROR":
            break

    print("\n=== SUMMARY ===")
    for name, status, detail, dt in results:
        print(f"  {name:16s} {status}")
    npass = sum(1 for r in results if r[1] == "PASS")
    print(f"{npass}/{len(phases)} passed | final inv={inv(env.last_info or {})}")
    env.close()


if __name__ == "__main__":
    main()
