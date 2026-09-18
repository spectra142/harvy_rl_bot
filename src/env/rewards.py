"""Reward calculation per skill module.

The bot sends raw event *counts* (mobs killed, blocks placed, damage points,
...); all scaling/clipping lives here so there is exactly one place that
turns events into reward magnitudes.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional

from .knowledge_base import MinecraftKnowledgeBase
from .observation import ITEM_NAME_TO_ID, ITEM_ID_TO_NAME

logger = logging.getLogger(__name__)


class RewardCalculator:
    """Computes shaped rewards based on bot state transitions."""

    # Goal-specific reward configuration. Keys are the canonical goal names
    # from env.observation.GOAL_TO_ID.
    GOAL_REWARDS: Dict[str, Dict[str, Any]] = {
        "gather_logs": {"item": "oak_log", "bonus": 5.0, "per_item": 1.0},
        "gather_stone": {"item": "cobblestone", "bonus": 3.0, "per_item": 0.5},
        "gather_coal": {"item": "coal", "bonus": 4.0, "per_item": 1.0},
        "gather_iron": {"item": "raw_iron", "bonus": 6.0, "per_item": 1.5},
        "craft_pickaxe": {"item_contains": "pickaxe", "bonus": 10.0},
        "craft_sword": {"item_contains": "sword", "bonus": 8.0},
        "craft_planks": {"item": "oak_planks", "bonus": 2.0, "per_item": 0.2},
        "craft_sticks": {"item": "stick", "bonus": 2.0, "per_item": 0.2},
        "craft_crafting_table": {"item": "crafting_table", "bonus": 5.0},
        "craft_torch": {"item": "torch", "bonus": 3.0, "per_item": 0.5},
        "build_shelter": {
            "block_placed": True,
            "per_block": 5.0,
            "completion_bonus": 50.0,
        },
        "fight_hostile": {"mob_killed": True, "bonus": 20.0},
        "eat_food": {"food_eaten": True, "bonus": 3.0},
        "explore": {"distance_moved": True, "per_block": 0.1},
        "survive": {"per_tick": 0.1},
        "survive_first_night": {"per_tick": 0.1},
    }

    def __init__(
        self,
        skill_module: str = "survival",
        noop_bias: float = 0.01,
        knowledge_base: Optional[MinecraftKnowledgeBase] = None,
        goal: Optional[str] = None,
    ):
        self.skill_module = skill_module
        self.noop_bias = noop_bias
        self.knowledge_base = knowledge_base
        self.goal = goal
        self.prev_health = 20.0
        self.prev_food = 20.0
        self.prev_inventory_sum = 0.0
        self.prev_alive = True
        self._prev_held_item: Optional[str] = None
        # Track per-item counts for goal-specific reward shaping
        self._prev_inventory_counts: Dict[str, float] = {}

    def reset(self, obs: Optional[dict] = None):
        """Reset per-episode state.

        Args:
            obs: If given, the first observation of the new episode. The
                inventory/health baselines are primed from it so items the
                bot carries across episodes do not produce a spurious
                inventory-delta jackpot on step 1.
        """
        self.prev_health = 20.0
        self.prev_food = 20.0
        self.prev_inventory_sum = 0.0
        self.prev_alive = True
        self._prev_held_item = None
        self._prev_inventory_counts = {}
        if obs is not None:
            inventory = np.asarray(
                obs.get("inventory", np.zeros(1, dtype=np.float32))
            )
            self.prev_inventory_sum = float(inventory.sum())
            self._prev_inventory_counts = self._inventory_counts_from_obs(obs)
            # NOTE: health/food baselines are deliberately NOT primed from
            # the reset observation -- it can race the respawn (health=0
            # mid-/kill), which would fabricate a huge heal reward on step 1.
            # A fresh episode always starts at full health after respawn.

    @staticmethod
    def _inventory_counts_from_obs(obs: dict) -> Dict[str, float]:
        """Decode the fixed-size inventory vector into name -> count."""
        inventory = np.asarray(obs.get("inventory", np.zeros(0, dtype=np.float32)))
        counts: Dict[str, float] = {}
        for name, idx in ITEM_NAME_TO_ID.items():
            if idx < inventory.shape[0] and inventory[idx] > 0:
                counts[name] = float(inventory[idx])
        return counts

    def compute(
        self,
        obs: dict,
        reward_signal: dict,
        action_idx: int = 0,
        danger_level: float = 0.0,
    ) -> float:
        """Compute total reward from observation and reward signal."""
        reward = 0.0

        # ----- Survival / per-tick signals (sent by the Mineflayer bot) -----
        # Small alive bonus; intentionally tiny so "stand still forever" is
        # never the optimal policy.
        reward += reward_signal.get("alive_tick", 0) * 0.01
        # Death penalty lives ONLY here (the env does not add its own).
        reward += reward_signal.get("death", 0) * -100.0
        # Damage is punished ONLY here (no health_delta double-count below).
        reward += reward_signal.get("damage_taken", 0) * -1.0

        # ----- Combat -----
        reward += reward_signal.get("mob_killed", 0) * 5.0
        reward += reward_signal.get("player_killed", 0) * 10.0

        # ----- Gathering / crafting / building -----
        items_mined = reward_signal.get("item_mined", 0)
        reward += items_mined * 0.5
        reward += reward_signal.get("item_crafted", 0) * 2.0
        reward += reward_signal.get("block_placed", 0) * 0.2
        reward += reward_signal.get("item_lost", 0) * -5.0

        # ----- Knowledge-base mining efficiency bonus -----
        if self.knowledge_base is not None and items_mined > 0:
            held_item_id = float(obs.get("self_held_item_id", [0.0])[0])
            held_item_name = ITEM_ID_TO_NAME.get(int(held_item_id))
            self._prev_held_item = held_item_name

            block_broken = reward_signal.get("block_broken_name", None)
            if block_broken is not None and held_item_name is not None:
                efficiency = self.knowledge_base.get_mining_efficiency(
                    block_broken, held_item_name
                )
                required_tool = self.knowledge_base.get_required_tool(block_broken)

                if required_tool is not None:
                    # Determine actual tool type
                    tool_type = None
                    for suffix in ("pickaxe", "axe", "shovel", "sword", "hoe"):
                        if held_item_name.endswith(suffix):
                            tool_type = suffix
                            break

                    if tool_type == required_tool:
                        # Correct tool type bonus
                        reward += 0.5
                        # Higher tier than minimum bonus
                        tier = self.knowledge_base.get_tool_tier(held_item_name)
                        if tier >= 2:
                            reward += 0.2 * (tier - 1)
                    else:
                        # Wrong tool penalty
                        reward -= 1.0
                else:
                    # Hand-minable block -- small reward for any tool use
                    if efficiency > 0.15:
                        reward += 0.1

        # Reward for any inventory growth (picking up items, crafting, etc.)
        inventory = obs.get("inventory", np.zeros(1, dtype=np.float32))
        inventory_sum = float(np.asarray(inventory).sum())
        inventory_delta = inventory_sum - self.prev_inventory_sum
        if inventory_delta > 0:
            reward += inventory_delta * 0.1
        self.prev_inventory_sum = inventory_sum

        # ----- Eating -----
        reward += reward_signal.get("food_eaten", 0) * 2.0

        # ----- Health regen (positive only; damage is via damage_taken) -----
        health = float(np.asarray(obs["self_health"]).flat[0])
        health_delta = health - self.prev_health
        if health_delta > 0:
            reward += health_delta * 2.0
        self.prev_health = health

        # ----- Food changes -----
        food = float(np.asarray(obs["self_food"]).flat[0])
        food_delta = food - self.prev_food
        if food_delta > 0:
            reward += food_delta * 2.0
        self.prev_food = food

        # ----- Goal-specific reward shaping -----
        if self.goal and self.goal in self.GOAL_REWARDS:
            reward += self._compute_goal_reward(self.goal, obs, reward_signal)

        # Update per-item inventory tracking (after goal shaping used it)
        self._prev_inventory_counts = self._inventory_counts_from_obs(obs)

        # ----- Noop bias: small penalty for idling when safe -----
        if action_idx == 0 and danger_level < 0.1:
            reward -= self.noop_bias

        return float(np.clip(reward, -10.0, 10.0))

    def _compute_goal_reward(
        self,
        goal: str,
        obs: dict,
        reward_signal: dict,
    ) -> float:
        """Compute goal-specific reward bonus.

        Detects progress toward the active goal and returns a shaped bonus.

        Args:
            goal: Active goal ID.
            obs: Current observation dict.
            reward_signal: Raw reward signals from the bot.

        Returns:
            Additional reward float (to be added to the base reward).
        """
        goal_cfg = self.GOAL_REWARDS[goal]
        bonus = 0.0
        inventory = np.asarray(obs.get("inventory", np.zeros(0, dtype=np.float32)))

        # --- Item acquisition goals: per-item inventory delta ---
        if "item" in goal_cfg:
            item_name: str = goal_cfg["item"]
            idx = ITEM_NAME_TO_ID.get(item_name)
            if idx is not None and idx < inventory.shape[0]:
                delta = float(inventory[idx]) - self._prev_inventory_counts.get(
                    item_name, 0.0
                )
                if delta > 0:
                    bonus += goal_cfg.get("bonus", 0.0)
                    bonus += delta * goal_cfg.get("per_item", 0.0)

        # --- Item-containment goals (e.g. any pickaxe, any sword) ---
        if "item_contains" in goal_cfg:
            contain_str: str = goal_cfg["item_contains"]
            gained = 0.0
            for name, idx in ITEM_NAME_TO_ID.items():
                if contain_str not in name or idx >= inventory.shape[0]:
                    continue
                delta = float(inventory[idx]) - self._prev_inventory_counts.get(
                    name, 0.0
                )
                if delta > 0:
                    gained += delta
            if gained > 0:
                bonus += goal_cfg.get("bonus", 0.0)

        # --- Block placement goals ---
        if "block_placed" in goal_cfg and goal_cfg["block_placed"]:
            blocks_placed = reward_signal.get("block_placed", 0)
            if blocks_placed > 0:
                bonus += blocks_placed * goal_cfg.get("per_block", 0.0)
            # Completion check: shelter has enough blocks
            if reward_signal.get("shelter_complete", 0) > 0:
                bonus += goal_cfg.get("completion_bonus", 0.0)

        # --- Combat goals ---
        if "mob_killed" in goal_cfg and goal_cfg["mob_killed"]:
            mobs_killed = reward_signal.get("mob_killed", 0)
            if mobs_killed > 0:
                bonus += goal_cfg.get("bonus", 0.0)

        # --- Eating goals ---
        if "food_eaten" in goal_cfg and goal_cfg["food_eaten"]:
            food_eaten = reward_signal.get("food_eaten", 0)
            if food_eaten > 0:
                bonus += goal_cfg.get("bonus", 0.0)

        # --- Per-tick survival ---
        if "per_tick" in goal_cfg:
            bonus += goal_cfg["per_tick"]

        # --- Exploration: distance moved ---
        if "distance_moved" in goal_cfg and goal_cfg["distance_moved"]:
            distance = reward_signal.get("distance_moved", 0.0)
            if distance > 0:
                bonus += distance * goal_cfg.get("per_block", 0.0)

        return bonus
