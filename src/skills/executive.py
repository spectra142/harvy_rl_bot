"""Executive policy that decides which skill module to invoke."""

import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass

from .skill_policies import (
    SkillModule,
    MovementSkill,
    CombatSkill,
    MiningSkill,
    CraftingSkill,
    BuildingSkill,
    SkillType,
)


@dataclass
class ExecutiveDecision:
    """Output of the executive policy."""

    skill_type: SkillType
    action: int  # Action index 0-24
    confidence: float  # 0-1 confidence in decision
    reasoning: str


class ExecutivePolicy:
    """High-level policy that selects and coordinates skill modules."""

    def __init__(self):
        self.skills = {
            SkillType.MOVEMENT: MovementSkill(),
            SkillType.COMBAT: CombatSkill(),
            SkillType.MINING: MiningSkill(),
            SkillType.CRAFTING: CraftingSkill(),
            SkillType.BUILDING: BuildingSkill(),
        }
        self.active_skill: Optional[SkillType] = None
        self.skill_switch_cooldown = 0
        self.cooldown_duration = 10  # Steps between skill switches

    def decide(self, observation: Dict) -> ExecutiveDecision:
        """Decide which skill to activate and what action to take."""
        self_obs = observation.get("self", {})
        env = observation.get("environment", {})
        entities = observation.get("nearby_entities", [])

        # Priority 1: Survival -- flee if critical health
        health = self_obs.get("health", 20.0)
        if health < 4.0:
            return ExecutiveDecision(
                skill_type=SkillType.MOVEMENT,
                action=2,  # move_back (flee)
                confidence=0.95,
                reasoning="Critical health -- fleeing",
            )

        # Priority 2: Combat -- fight hostiles
        hostiles = [e for e in entities if e.get("hostile", False)]
        if hostiles:
            combat_skill = self.skills[SkillType.COMBAT]
            if combat_skill.should_activate(observation):
                action = combat_skill.get_preferred_action(observation)
                return ExecutiveDecision(
                    skill_type=SkillType.COMBAT,
                    action=action,
                    confidence=0.85,
                    reasoning=f"Fighting {len(hostiles)} hostile(s)",
                )

        # Priority 3: Crafting -- if we can craft something useful
        crafting_skill = self.skills[SkillType.CRAFTING]
        if crafting_skill.should_activate(observation):
            next_craft = crafting_skill.get_next_craft(observation)
            if next_craft:
                return ExecutiveDecision(
                    skill_type=SkillType.CRAFTING,
                    action=13,  # use_item
                    confidence=0.7,
                    reasoning=f"Crafting {next_craft}",
                )

        # Priority 4: Building -- if night is approaching
        time_of_day = env.get("time_of_day", 6000)
        building_skill = self.skills[SkillType.BUILDING]
        if building_skill.should_activate(observation):
            action = building_skill.get_preferred_action(observation)
            return ExecutiveDecision(
                skill_type=SkillType.BUILDING,
                action=action,
                confidence=0.6,
                reasoning="Building shelter before night",
            )

        # Priority 5: Mining -- if we have pickaxe and need resources
        mining_skill = self.skills[SkillType.MINING]
        if mining_skill.should_activate(observation):
            action = mining_skill.get_preferred_action(observation)
            return ExecutiveDecision(
                skill_type=SkillType.MINING,
                action=action,
                confidence=0.5,
                reasoning=f"Mining for {mining_skill.target_resource}",
            )

        # Default: Movement
        movement_skill = self.skills[SkillType.MOVEMENT]
        action = movement_skill.get_preferred_action(observation)
        return ExecutiveDecision(
            skill_type=SkillType.MOVEMENT,
            action=action,
            confidence=0.3,
            reasoning="Default movement",
        )

    def get_skill_weights(self, observation: Dict) -> Dict[SkillType, float]:
        """Get confidence weights for each skill given current state."""
        weights = {}
        for skill_type, skill in self.skills.items():
            if skill.should_activate(observation):
                weights[skill_type] = skill.config.priority / 10.0
            else:
                weights[skill_type] = 0.0
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return weights

    def should_switch_skill(self, new_skill: SkillType) -> bool:
        """Determine if we should switch to a new skill."""
        if self.skill_switch_cooldown > 0:
            return False
        if self.active_skill == new_skill:
            return False
        return True

    def on_step(self):
        """Update cooldowns and skill states."""
        if self.skill_switch_cooldown > 0:
            self.skill_switch_cooldown -= 1
        for skill in self.skills.values():
            if skill.active:
                skill.step()
