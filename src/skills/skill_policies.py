"""Skill module definitions -- each skill is a trained sub-policy."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class SkillType(Enum):
    """Types of skills the bot can execute."""

    MOVEMENT = "movement"
    COMBAT = "combat"
    MINING = "mining"
    CRAFTING = "crafting"
    BUILDING = "building"
    EXPLORATION = "exploration"


@dataclass
class SkillConfig:
    """Configuration for a skill module."""

    skill_type: SkillType
    priority: int  # Higher = more important
    activation_conditions: List[str]  # When to activate
    required_items: List[str]
    max_duration: int  # Max steps before forced switch


class SkillModule:
    """Base class for skill modules."""

    def __init__(self, config: SkillConfig):
        self.config = config
        self.active = False
        self.step_count = 0

    def should_activate(self, observation: Dict) -> bool:
        """Determine if this skill should activate based on observation."""
        raise NotImplementedError

    def get_preferred_action(self, observation: Dict) -> int:
        """Get the preferred action for this skill (action index 0-24)."""
        raise NotImplementedError

    def on_activate(self):
        """Called when skill is activated."""
        self.active = True
        self.step_count = 0

    def on_deactivate(self):
        """Called when skill is deactivated."""
        self.active = False
        self.step_count = 0

    def step(self):
        """Increment step counter."""
        self.step_count += 1

    def is_timed_out(self) -> bool:
        """Check if skill has exceeded max duration."""
        return self.step_count >= self.config.max_duration


class MovementSkill(SkillModule):
    """Handles basic movement: walking, jumping, swimming, parkour."""

    CONFIG = SkillConfig(
        skill_type=SkillType.MOVEMENT,
        priority=1,
        activation_conditions=["always_active"],
        required_items=[],
        max_duration=10000,
    )

    def __init__(self):
        super().__init__(self.CONFIG)
        self.target_position: Optional[Tuple[float, float, float]] = None
        self.path: List[Tuple[float, float, float]] = []

    def should_activate(self, observation: Dict) -> bool:
        return True  # Movement is always active as fallback

    def get_preferred_action(self, observation: Dict) -> int:
        env = observation.get("environment", {})
        self_obs = observation.get("self", {})

        # If in water, move forward to get out
        if env.get("in_water", False):
            return 1  # move_forward

        # If on ground and not blocked, move forward
        if env.get("on_ground", True):
            # Simple: move forward most of the time
            return 1  # move_forward

        return 0  # noop

    def navigate_to(self, target: Tuple[float, float, float]) -> None:
        """Set navigation target."""
        self.target_position = target

    def avoid_obstacle(self, observation: Dict) -> int:
        """Get action to avoid nearby obstacles."""
        # Look at voxel grid for obstacles
        voxel_grid = observation.get(
            "voxel_grid", np.zeros((11, 11, 7), dtype=np.uint8)
        )
        # If block directly ahead, jump or turn
        forward_block = voxel_grid[5, 6, 3]  # Center + forward
        if forward_block != 0:  # Not air
            return 7  # jump
        return 1  # move_forward


class CombatSkill(SkillModule):
    """Handles fighting: target selection, attacking, retreating, weapon switching."""

    CONFIG = SkillConfig(
        skill_type=SkillType.COMBAT,
        priority=10,
        activation_conditions=["hostile_nearby", "target_acquired"],
        required_items=["sword", "axe", "bow"],
        max_duration=500,
    )

    # Health thresholds for combat decisions
    RETREAT_HEALTH = 8.0
    AGGRESSIVE_HEALTH = 15.0
    CRITICAL_HEALTH = 4.0

    def __init__(self):
        super().__init__(self.CONFIG)
        self.target: Optional[Dict] = None
        self.combat_state = "idle"  # idle, approaching, attacking, retreating, healing
        self.last_health = 20.0

    def should_activate(self, observation: Dict) -> bool:
        entities = observation.get("nearby_entities", [])
        hostiles = [e for e in entities if e.get("hostile", False)]
        return len(hostiles) > 0

    def get_preferred_action(self, observation: Dict) -> int:
        self_obs = observation.get("self", {})
        health = self_obs.get("health", 20.0)
        entities = observation.get("nearby_entities", [])
        hostiles = [e for e in entities if e.get("hostile", False)]

        if not hostiles:
            self.combat_state = "idle"
            return 0  # noop

        # Select nearest hostile
        nearest = min(hostiles, key=lambda e: e.get("distance", 100))
        self.target = nearest

        # Critical health -- flee and eat
        if health < self.CRITICAL_HEALTH:
            self.combat_state = "retreating"
            return 6  # sneak away (move_back + sneak)

        # Low health -- retreat
        if health < self.RETREAT_HEALTH:
            self.combat_state = "retreating"
            return 2  # move_back

        # Health recovering -- approach
        if health < self.AGGRESSIVE_HEALTH:
            self.combat_state = "approaching"
            if nearest.get("distance", 10) > 3:
                return 1  # move_forward
            else:
                return 2  # move_back (keep distance)

        # Healthy -- attack aggressively
        self.combat_state = "attacking"
        if nearest.get("distance", 10) < 4:
            return 12  # attack
        else:
            return 1  # move_forward to close distance

    def should_retreat(self, observation: Dict) -> bool:
        """Determine if we should retreat from combat."""
        health = observation.get("self", {}).get("health", 20.0)
        entities = observation.get("nearby_entities", [])
        hostile_count = sum(1 for e in entities if e.get("hostile", False))

        # Retreat if health is low or outnumbered
        if health < self.RETREAT_HEALTH:
            return True
        if hostile_count > 2 and health < self.AGGRESSIVE_HEALTH:
            return True
        return False

    def should_switch_weapon(self, observation: Dict) -> Optional[int]:
        """Suggest weapon slot based on situation."""
        health = observation.get("self", {}).get("health", 20.0)
        held_item = observation.get("self", {}).get("held_item_id", 0)

        # Low health -- switch to food (slot 8 usually)
        if health < 10:
            return 8

        # Close combat -- sword/axe
        target_dist = self.target.get("distance", 10) if self.target else 10
        if target_dist < 3:
            return 0 if "sword" in str(held_item) else 1  # sword slot
        else:
            return 3 if "bow" in str(held_item) else 1  # bow slot for ranged

        return None


class MiningSkill(SkillModule):
    """Handles resource gathering: finding ores, strip mining, avoiding lava."""

    CONFIG = SkillConfig(
        skill_type=SkillType.MINING,
        priority=5,
        activation_conditions=["need_resources", "pickaxe_available"],
        required_items=["pickaxe"],
        max_duration=2000,
    )

    OPTIMAL_Y_LEVELS = {
        "diamond": -59,
        "iron": 16,
        "coal": 0,
        "gold": -16,
        "redstone": -59,
        "lapis": 0,
    }

    def __init__(self):
        super().__init__(self.CONFIG)
        self.target_resource = "iron"
        self.mining_level = -1  # Target Y level

    def should_activate(self, observation: Dict) -> bool:
        held_item = observation.get("self", {}).get("held_item_id", 0)
        # Activate if we have a pickaxe
        has_pickaxe = held_item in [1, 2, 3, 4, 5]  # pickaxe IDs
        return has_pickaxe

    def get_preferred_action(self, observation: Dict) -> int:
        self_obs = observation.get("self", {})
        position = self_obs.get("position", [0, 64, 0])
        current_y = position[1]

        # Navigate to optimal Y level
        target_y = self.OPTIMAL_Y_LEVELS.get(self.target_resource, 16)

        if current_y > target_y + 2:
            # Need to dig down -- look down and mine
            return 11  # look_down_15
        elif current_y < target_y - 2:
            # Need to go up
            return 9  # look_up_15
        else:
            # At right level -- strip mine (forward, looking at wall)
            return 12  # attack (mine forward)

    def set_target_resource(self, resource: str) -> None:
        """Set the target resource to mine for."""
        self.target_resource = resource
        self.mining_level = self.OPTIMAL_Y_LEVELS.get(resource, 16)


class CraftingSkill(SkillModule):
    """Handles crafting: deciding what to craft and when."""

    CONFIG = SkillConfig(
        skill_type=SkillType.CRAFTING,
        priority=7,
        activation_conditions=["crafting_table_nearby", "materials_available"],
        required_items=["crafting_table"],
        max_duration=100,
    )

    CRAFTING_PRIORITY = [
        ("diamond_pickaxe", {"diamond": 3, "stick": 2}),
        ("iron_pickaxe", {"iron_ingot": 3, "stick": 2}),
        ("stone_pickaxe", {"cobblestone": 3, "stick": 2}),
        ("wooden_pickaxe", {"oak_planks": 3, "stick": 2}),
        ("iron_sword", {"iron_ingot": 2, "stick": 1}),
        ("stone_sword", {"cobblestone": 2, "stick": 1}),
        ("wooden_sword", {"oak_planks": 2, "stick": 1}),
        ("shield", {"iron_ingot": 1, "oak_planks": 6}),
        ("torch", {"coal": 1, "stick": 1}),
    ]

    def __init__(self):
        super().__init__(self.CONFIG)
        self.crafting_queue: List[str] = []

    def should_activate(self, observation: Dict) -> bool:
        # Check if we should craft something
        return self._should_craft(observation)

    def get_preferred_action(self, observation: Dict) -> int:
        # Use item to open crafting table/furnace
        return 13  # use_item

    def _should_craft(self, observation: Dict) -> bool:
        """Check if we have materials to craft something useful."""
        inventory = observation.get("self", {}).get("inventory_counts", {})

        for item_name, materials in self.CRAFTING_PRIORITY:
            can_craft = all(
                inventory.get(mat, 0) >= qty for mat, qty in materials.items()
            )
            if can_craft:
                return True
        return False

    def get_next_craft(self, observation: Dict) -> Optional[str]:
        """Determine what to craft next based on priority and materials."""
        inventory = observation.get("self", {}).get("inventory_counts", {})

        for item_name, materials in self.CRAFTING_PRIORITY:
            can_craft = all(
                inventory.get(mat, 0) >= qty for mat, qty in materials.items()
            )
            if can_craft:
                return item_name
        return None


class BuildingSkill(SkillModule):
    """Handles construction: shelters, walls, defensive structures."""

    CONFIG = SkillConfig(
        skill_type=SkillType.BUILDING,
        priority=6,
        activation_conditions=["night_approaching", "building_materials"],
        required_items=["blocks"],
        max_duration=1000,
    )

    SHELTER_TEMPLATE = [
        # 5x5 floor/wall/roof pattern
        # Each tuple: (dx, dy, dz, block_type)
    ]

    def __init__(self):
        super().__init__(self.CONFIG)
        self.build_plan: List = []
        self.build_index = 0

    def should_activate(self, observation: Dict) -> bool:
        time = observation.get("environment", {}).get("time_of_day", 6000)
        # Activate near sunset (12000) or when dangerous
        danger = observation.get("environment", {}).get("danger_level", 0)
        return time > 11000 or danger > 0.5

    def get_preferred_action(self, observation: Dict) -> int:
        if self.build_index < len(self.build_plan):
            return 14  # place_block
        return 0  # noop -- building complete

    def generate_shelter_plan(self, center: Tuple[float, float, float]) -> List:
        """Generate a 5x5 shelter build plan."""
        cx, cy, cz = int(center[0]), int(center[1]), int(center[2])
        plan = []

        # Floor (5x5)
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                plan.append((cx + dx, cy - 1, cz + dz, "oak_planks"))

        # Walls (4 sides, 3 high)
        for y in range(3):
            for dx in range(-2, 3):
                plan.append((cx + dx, cy + y, cz - 2, "oak_planks"))  # Back wall
                plan.append((cx + dx, cy + y, cz + 2, "oak_planks"))  # Front wall
            for dz in range(-1, 2):
                plan.append((cx - 2, cy + y, cz + dz, "oak_planks"))  # Left wall
                plan.append((cx + 2, cy + y, cz + dz, "oak_planks"))  # Right wall

        # Roof
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                plan.append((cx + dx, cy + 3, cz + dz, "oak_planks"))

        # Doorway (remove one block from front wall)
        plan = [p for p in plan if not (p[0] == cx and p[2] == cz + 2 and p[1] == cy)]

        return plan
