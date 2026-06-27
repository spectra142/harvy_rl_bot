"""Autonomous goal generation with subgoal chaining for Harvy.

Instead of following a fixed curriculum, Harvy:
1. Analyzes current state (inventory, health, time, threats)
2. Proposes candidate goals that are relevant to the situation
3. Generates prerequisite subgoal chains using crafting recipes
4. Selects goals in the 'zone of proximal development' (30-70% estimated success)
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque

# Import knowledge base
try:
    from env.knowledge_base import MinecraftKnowledgeBase
except ImportError:
    from src.env.knowledge_base import MinecraftKnowledgeBase

logger = logging.getLogger(__name__)


class GoalProposal:
    """A proposed goal with metadata."""

    def __init__(
        self,
        goal_id: str,
        description: str,
        priority: float,
        prerequisites: List[str],
        estimated_success: float,
    ):
        self.goal_id = goal_id
        self.description = description
        self.priority = priority
        self.prerequisites = prerequisites
        self.estimated_success = estimated_success
        self.subgoals: List[str] = []  # Filled by chaining

    def __repr__(self) -> str:
        return (
            f"GoalProposal({self.goal_id}, prio={self.priority:.2f}, "
            f"est_success={self.estimated_success:.2f}, subgoals={self.subgoals})"
        )


class AutonomousCurriculum:
    """Self-directed goal generator with subgoal chaining.

    Backward-compatible public API
    ------------------------------
    * ``select_goal(state=None) -> str``
    * ``record_episode(goal, success, reward=0.0) -> None``
    * ``get_progress() -> Dict``
    """

    # Goal definitions: goal_id -> {description, base_priority}
    GOAL_CATALOG: Dict[str, Dict[str, object]] = {
        "gather_logs": {
            "description": "Collect oak logs from trees",
            "priority": 1.0,
        },
        "gather_stone": {
            "description": "Mine cobblestone underground",
            "priority": 1.0,
        },
        "gather_coal": {
            "description": "Mine coal ore for torches",
            "priority": 0.8,
        },
        "gather_iron": {
            "description": "Mine iron ore for tools",
            "priority": 0.7,
        },
        "craft_planks": {
            "description": "Convert logs to planks",
            "priority": 0.9,
        },
        "craft_sticks": {
            "description": "Craft sticks from planks",
            "priority": 0.9,
        },
        "craft_crafting_table": {
            "description": "Craft a crafting table",
            "priority": 0.85,
        },
        "craft_pickaxe": {
            "description": "Craft a pickaxe",
            "priority": 0.85,
        },
        "craft_sword": {
            "description": "Craft a sword for combat",
            "priority": 0.8,
        },
        "craft_torch": {
            "description": "Craft torches for lighting",
            "priority": 0.7,
        },
        "eat_food": {
            "description": "Eat food to survive",
            "priority": 1.5,  # High priority when hungry
        },
        "fight_hostile": {
            "description": "Kill a hostile mob",
            "priority": 0.6,
        },
        "build_shelter": {
            "description": "Build a shelter for the night",
            "priority": 1.2,  # High at night
        },
        "explore": {
            "description": "Explore new chunks",
            "priority": 0.5,
        },
        "survive": {
            "description": "Just survive as long as possible",
            "priority": 0.3,
        },
    }

    # ------------------------------------------------------------------ #
    #  Zone of proximal development bounds
    # ------------------------------------------------------------------ #
    ZPD_MIN = 0.30
    ZPD_MAX = 0.70

    def __init__(self, temperature: float = 2.0, mc_version: str = "1.19.2"):
        """
        Args:
            temperature: Higher = more random exploration. Lower = stick to best goals.
            mc_version: Minecraft version string for the knowledge base.
        """
        self.temperature = temperature
        self.kb = MinecraftKnowledgeBase(mc_version)

        # Episode history for each goal
        self.episode_counts: Dict[str, int] = {g: 0 for g in self.GOAL_CATALOG}
        self.success_counts: Dict[str, int] = {g: 0 for g in self.GOAL_CATALOG}
        self.recent_rewards: Dict[str, deque] = {
            g: deque(maxlen=20) for g in self.GOAL_CATALOG
        }

        self.last_goal: str = "survive"
        self.last_inventory: Dict[str, int] = {}
        self.last_health: float = 20.0
        self.last_food: float = 20.0
        self.last_time: float = 6000.0

    # ================================================================ #
    #  Public API (backward-compatible)
    # ================================================================ #

    def select_goal(self, state: Optional[Dict] = None) -> str:
        """Select a goal based on current state and history.

        Args:
            state: Current bot state dict with keys:
                - inventory: dict of item_name -> count
                - health: float (0-20)
                - food: float (0-20)
                - time_of_day: float (0-24000)
                - danger_level: float (0-1)
                - nearby_hostiles: int
                - position: [x, y, z]
                - held_item: str

        Returns:
            Selected goal ID string.
        """
        state = state or {}

        # 1. Propose candidate goals based on state
        proposals = self._propose_goals(state)

        # 2. Generate subgoal chains for each proposal
        for proposal in proposals:
            proposal.subgoals = self._build_subgoal_chain(proposal.goal_id, state)

        # 3. Estimate success probability for each
        for proposal in proposals:
            proposal.estimated_success = self._estimate_success(proposal, state)

        # 4. Select from zone of proximal development (30-70% success)
        zpd_goals = [
            p for p in proposals if self.ZPD_MIN <= p.estimated_success <= self.ZPD_MAX
        ]
        if not zpd_goals:
            # Fall back to all proposals if none in ZPD
            zpd_goals = proposals
            logger.debug(
                "No goals in ZPD (%.2f-%.2f); falling back to all %d proposals",
                self.ZPD_MIN,
                self.ZPD_MAX,
                len(proposals),
            )

        # 5. Sample weighted by priority and exploration bonus
        scores: List[float] = []
        for p in zpd_goals:
            # Historical success rate
            hist_success = self.success_counts[p.goal_id] / max(
                self.episode_counts[p.goal_id], 1
            )
            # Exploration bonus (prefer less-attempted goals)
            exploration_bonus = 1.0 / (1.0 + np.log1p(self.episode_counts[p.goal_id]))
            score = (
                p.priority * max(p.estimated_success, 0.1) * (1.0 + exploration_bonus)
            ) ** self.temperature
            scores.append(max(score, 0.01))

        probs = np.array(scores, dtype=np.float64)
        probs /= probs.sum()

        goal_ids = [p.goal_id for p in zpd_goals]
        selected: str = np.random.choice(goal_ids, p=probs)

        # Update cached state for next call
        self.last_goal = selected
        self.last_inventory = dict(state.get("inventory", {}))
        self.last_health = state.get("health", 20.0)
        self.last_food = state.get("food", 20.0)
        self.last_time = state.get("time_of_day", 6000.0)

        logger.debug(
            "Selected goal '%s' from %d candidates (ZPD: %s). " "Subgoals: %s",
            selected,
            len(zpd_goals),
            any(self.ZPD_MIN <= p.estimated_success <= self.ZPD_MAX for p in proposals),
            [p.subgoals for p in zpd_goals if p.goal_id == selected],
        )
        return selected

    def record_episode(self, goal: str, success: bool, reward: float = 0.0) -> None:
        """Record episode outcome for a goal.

        Args:
            goal: The goal ID that was pursued.
            success: Whether the episode was successful.
            reward: Total episode reward (optional, for tracking).
        """
        self.episode_counts[goal] = self.episode_counts.get(goal, 0) + 1
        if success:
            self.success_counts[goal] = self.success_counts.get(goal, 0) + 1
        self.recent_rewards[goal].append(reward)

    def get_progress(self) -> Dict:
        """Return progress snapshot.

        Returns:
            Dict with current goal, per-goal episode counts, success rates,
            and mean recent rewards.
        """
        return {
            "current_goal": self.last_goal,
            "goals": {
                g: {
                    "episodes": self.episode_counts[g],
                    "successes": self.success_counts[g],
                    "success_rate": (
                        self.success_counts[g] / max(self.episode_counts[g], 1)
                    ),
                    "mean_reward": (
                        float(np.mean(self.recent_rewards[g]))
                        if self.recent_rewards[g]
                        else 0.0
                    ),
                }
                for g in self.GOAL_CATALOG
            },
        }

    # ================================================================ #
    #  Internal: Goal proposal
    # ================================================================ #

    def _propose_goals(self, state: Dict) -> List[GoalProposal]:
        """Propose relevant goals based on current game state."""
        proposals: List[GoalProposal] = []
        inventory: Dict[str, int] = state.get("inventory", {})
        health: float = state.get("health", 20.0)
        food: float = state.get("food", 20.0)
        time_of_day: float = state.get("time_of_day", 6000.0)
        danger_level: float = state.get("danger_level", 0.0)
        nearby_hostiles: int = state.get("nearby_hostiles", 0)
        held_item: str = state.get("held_item", "")

        # ---- Always propose survival as a fallback ----
        proposals.append(GoalProposal("survive", "Survive", 0.3, [], 0.5))

        # ---- Hunger-critical: eat food ----
        if food < 14:
            has_food = any(
                k in inventory
                for k in [
                    "apple",
                    "bread",
                    "cooked_beef",
                    "cooked_porkchop",
                    "cooked_chicken",
                    "cooked_mutton",
                    "cooked_rabbit",
                ]
            )
            if has_food:
                proposals.append(GoalProposal("eat_food", "Eat food", 2.0, [], 0.8))

        # ---- Health-critical: boost survival priority ----
        if health < 8:
            proposals[0].priority = 3.0  # Boost survival

        # ---- Low danger: can work on non-urgent goals ----
        if danger_level < 0.3:
            # Log count across all wood types
            log_count = sum(
                inventory.get(k, 0)
                for k in [
                    "oak_log",
                    "birch_log",
                    "spruce_log",
                    "jungle_log",
                    "acacia_log",
                    "dark_oak_log",
                ]
            )

            # No logs -> get logs
            if log_count < 8:
                proposals.append(GoalProposal("gather_logs", "Get logs", 1.2, [], 0.6))

            # Have logs but few planks -> craft planks
            plank_count = inventory.get("oak_planks", 0)
            if log_count >= 1 and plank_count < 12:
                proposals.append(
                    GoalProposal(
                        "craft_planks",
                        "Craft planks",
                        1.0,
                        ["gather_logs"],
                        0.7,
                    )
                )

            # Need sticks
            if plank_count >= 2 and inventory.get("stick", 0) < 4:
                proposals.append(
                    GoalProposal(
                        "craft_sticks",
                        "Craft sticks",
                        0.9,
                        ["craft_planks"],
                        0.7,
                    )
                )

            # Need crafting table
            if plank_count >= 4 and "crafting_table" not in inventory:
                proposals.append(
                    GoalProposal(
                        "craft_crafting_table",
                        "Craft table",
                        1.0,
                        ["craft_planks"],
                        0.7,
                    )
                )

            # Need pickaxe
            has_pickaxe = (
                any("pickaxe" in k for k in inventory) or "pickaxe" in held_item
            )
            if not has_pickaxe and inventory.get("stick", 0) >= 2 and plank_count >= 3:
                proposals.append(
                    GoalProposal(
                        "craft_pickaxe",
                        "Craft pickaxe",
                        1.1,
                        ["craft_sticks"],
                        0.6,
                    )
                )

            # Need sword
            has_sword = any("sword" in k for k in inventory) or "sword" in held_item
            if not has_sword and inventory.get("stick", 0) >= 1 and plank_count >= 2:
                proposals.append(
                    GoalProposal(
                        "craft_sword",
                        "Craft sword",
                        0.8,
                        ["craft_sticks"],
                        0.6,
                    )
                )

            # Have pickaxe -> get stone
            stone_count = inventory.get("cobblestone", 0)
            if has_pickaxe and stone_count < 16:
                proposals.append(
                    GoalProposal(
                        "gather_stone",
                        "Mine stone",
                        0.9,
                        ["craft_pickaxe"],
                        0.5,
                    )
                )

            # Night approaching -> build shelter
            if time_of_day > 11000 or time_of_day < 1000:
                proposals.append(
                    GoalProposal(
                        "build_shelter",
                        "Build shelter",
                        1.5,
                        [],
                        0.4,
                    )
                )

            # Hostiles nearby + have sword -> fight
            if nearby_hostiles > 0 and has_sword:
                proposals.append(
                    GoalProposal(
                        "fight_hostile",
                        "Fight mob",
                        1.0,
                        ["craft_sword"],
                        0.4,
                    )
                )

            # Have stone/iron pickaxe -> get coal
            has_stone_pickaxe = any(
                k in inventory
                for k in ["stone_pickaxe", "iron_pickaxe", "diamond_pickaxe"]
            )
            if has_stone_pickaxe:
                proposals.append(
                    GoalProposal(
                        "gather_coal",
                        "Mine coal",
                        0.7,
                        ["gather_stone"],
                        0.5,
                    )
                )

            # Have iron pickaxe -> get iron
            has_iron_pickaxe = any(
                k in inventory for k in ["iron_pickaxe", "diamond_pickaxe"]
            )
            if has_iron_pickaxe:
                proposals.append(
                    GoalProposal(
                        "gather_iron",
                        "Mine iron",
                        0.6,
                        ["craft_pickaxe"],
                        0.4,
                    )
                )

            # Explore if nothing else urgent
            if len(proposals) <= 2:
                proposals.append(GoalProposal("explore", "Explore", 0.5, [], 0.6))

        return proposals

    # ================================================================ #
    #  Internal: Subgoal chaining
    # ================================================================ #

    def _build_subgoal_chain(self, goal_id: str, state: Dict) -> List[str]:
        """Build a chain of subgoals needed to achieve the main goal.

        Uses the knowledge base for recipe-aware planning.  The chain is
        returned in *dependency* order (first subgoal must be done first).

        Args:
            goal_id: The main goal to achieve.
            state: Current bot state (used for inventory snapshot).

        Returns:
            Ordered list of subgoal IDs.
        """
        chain: List[str] = []
        inventory: Dict[str, int] = dict(state.get("inventory", {}))

        # ---- Crafting goals ----
        if goal_id == "craft_pickaxe":
            if inventory.get("stick", 0) < 2:
                chain.append("craft_sticks")
            if inventory.get("oak_planks", 0) < 3:
                chain.append("craft_planks")
            if inventory.get("oak_log", 0) < 1:
                chain.append("gather_logs")

        elif goal_id == "craft_sword":
            if inventory.get("stick", 0) < 1:
                chain.append("craft_sticks")
            if inventory.get("oak_planks", 0) < 2:
                chain.append("craft_planks")
            if inventory.get("oak_log", 0) < 1:
                chain.append("gather_logs")

        elif goal_id == "craft_sticks":
            if inventory.get("oak_planks", 0) < 2:
                chain.append("craft_planks")
            if inventory.get("oak_log", 0) < 1:
                chain.append("gather_logs")

        elif goal_id == "craft_planks":
            if inventory.get("oak_log", 0) < 1:
                chain.append("gather_logs")

        elif goal_id == "craft_crafting_table":
            if inventory.get("oak_planks", 0) < 4:
                chain.append("craft_planks")
            if inventory.get("oak_log", 0) < 1:
                chain.append("gather_logs")

        # ---- Gathering goals ----
        elif goal_id == "gather_stone":
            has_pickaxe = any("pickaxe" in k for k in inventory)
            if not has_pickaxe:
                chain.append("craft_pickaxe")

        elif goal_id == "gather_coal":
            has_pickaxe = any("pickaxe" in k for k in inventory)
            if not has_pickaxe:
                chain.append("craft_pickaxe")

        elif goal_id == "gather_iron":
            has_pickaxe = any(
                k in inventory for k in ["iron_pickaxe", "diamond_pickaxe"]
            )
            if not has_pickaxe:
                chain.append("craft_pickaxe")

        # ---- Combat goals ----
        elif goal_id == "fight_hostile":
            has_sword = any("sword" in k for k in inventory)
            if not has_sword:
                chain.append("craft_sword")

        # ---- Building goals ----
        elif goal_id == "build_shelter":
            if (
                inventory.get("oak_planks", 0) < 20
                and inventory.get("cobblestone", 0) < 20
            ):
                if inventory.get("oak_log", 0) < 5:
                    chain.append("gather_logs")
                else:
                    chain.append("craft_planks")

        return chain

    # ================================================================ #
    #  Internal: Success estimation
    # ================================================================ #

    def _estimate_success(self, proposal: GoalProposal, state: Dict) -> float:
        """Estimate success probability (0-1) for a goal given current state.

        Blends historical success rate with state-based heuristics.  As more
        episodes are recorded, historical data gains weight.

        Args:
            proposal: The proposed goal with metadata.
            state: Current bot state.

        Returns:
            Estimated success probability in [0.05, 0.95].
        """
        # Base from historical success rate
        n_episodes = self.episode_counts[proposal.goal_id]
        hist_rate = self.success_counts[proposal.goal_id] / max(n_episodes, 1)

        # State-based adjustments
        health: float = state.get("health", 20.0)
        food: float = state.get("food", 20.0)
        danger: float = state.get("danger_level", 0.0)

        health_factor = min(health / 20.0, 1.0)
        food_factor = min(food / 20.0, 1.0)
        danger_factor = max(0.0, 1.0 - danger)

        state_factor = (health_factor + food_factor + danger_factor) / 3.0

        # Blend history and state (more weight to history as we get data)
        hist_weight = min(n_episodes / 20.0, 0.8)  # Max 80% weight to history

        estimated = hist_weight * hist_rate + (1.0 - hist_weight) * state_factor * 0.5
        return float(np.clip(estimated, 0.05, 0.95))

    # ================================================================ #
    #  Utility
    # ================================================================ #

    def get_subgoal_chain(self, goal_id: str, state: Dict) -> List[str]:
        """Public accessor for subgoal chain of a given goal.

        Args:
            goal_id: Goal to build chain for.
            state: Current bot state.

        Returns:
            Ordered list of subgoal IDs.
        """
        return self._build_subgoal_chain(goal_id, state)

    def get_goal_proposals(self, state: Dict) -> List[GoalProposal]:
        """Public accessor for goal proposals (useful for debugging/UIs).

        Args:
            state: Current bot state.

        Returns:
            List of :class:`GoalProposal` objects with subgoals and
            estimated success filled in.
        """
        proposals = self._propose_goals(state)
        for p in proposals:
            p.subgoals = self._build_subgoal_chain(p.goal_id, state)
            p.estimated_success = self._estimate_success(p, state)
        return proposals
