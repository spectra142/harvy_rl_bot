"""Minecraft knowledge base for RL reward shaping and goal generation.

Uses minecraft-data to provide crafting recipes, block drops, tool tiers,
mob behaviors, and other game knowledge.
"""

import json
from typing import Dict, List, Optional, Tuple


class MinecraftKnowledgeBase:
    """Static and cached Minecraft game knowledge.

    Provides crafting recipes, tool tiers, mob behavior data, block hardness,
    mining efficiency calculations, and optimal Y-levels for resources.  The
    data is stored as plain Python dicts so the module works even when the
    ``minecraft-data`` package is not installed.
    """

    def __init__(self, mc_version: str = "1.19.2"):
        """Initialize knowledge base for a specific Minecraft version.

        Args:
            mc_version: Minecraft version string (e.g. ``"1.19.2"``).
        """
        try:
            import minecraft_data  # type: ignore[import-untyped]

            self.mc_data = minecraft_data(mc_version)
        except (ImportError, KeyError):
            self.mc_data = None

        self._load_recipes()
        self._load_tool_tiers()
        self._load_mob_info()
        self._load_block_info()

    # ------------------------------------------------------------------ #
    # Internal loaders
    # ------------------------------------------------------------------ #

    def _load_recipes(self) -> None:
        """Parse crafting recipes from minecraft-data (fallback to hardcoded)."""
        self.recipes: Dict[str, Dict] = {
            # ---- Planks ----
            "oak_planks": {
                "ingredients": {"oak_log": 1},
                "result_count": 4,
            },
            "birch_planks": {
                "ingredients": {"birch_log": 1},
                "result_count": 4,
            },
            "spruce_planks": {
                "ingredients": {"spruce_log": 1},
                "result_count": 4,
            },
            "jungle_planks": {
                "ingredients": {"jungle_log": 1},
                "result_count": 4,
            },
            "acacia_planks": {
                "ingredients": {"acacia_log": 1},
                "result_count": 4,
            },
            "dark_oak_planks": {
                "ingredients": {"dark_oak_log": 1},
                "result_count": 4,
            },
            # ---- Sticks ----
            "stick": {
                "ingredients": {"oak_planks": 2},
                "result_count": 4,
            },
            # ---- Crafting table ----
            "crafting_table": {
                "ingredients": {"oak_planks": 4},
                "result_count": 1,
            },
            # ---- Chest ----
            "chest": {
                "ingredients": {"oak_planks": 8},
                "result_count": 1,
            },
            # ---- Furnace ----
            "furnace": {
                "ingredients": {"cobblestone": 8},
                "result_count": 1,
            },
            # ---- Torches ----
            "torch": {
                "ingredients": {"coal": 1, "stick": 1},
                "result_count": 4,
            },
            "torch_from_charcoal": {
                "ingredients": {"charcoal": 1, "stick": 1},
                "result_count": 4,
            },
            # ---- Wooden tools ----
            "wooden_pickaxe": {
                "ingredients": {"oak_planks": 3, "stick": 2},
                "result_count": 1,
            },
            "wooden_axe": {
                "ingredients": {"oak_planks": 3, "stick": 2},
                "result_count": 1,
            },
            "wooden_sword": {
                "ingredients": {"oak_planks": 2, "stick": 1},
                "result_count": 1,
            },
            "wooden_shovel": {
                "ingredients": {"oak_planks": 1, "stick": 2},
                "result_count": 1,
            },
            "wooden_hoe": {
                "ingredients": {"oak_planks": 2, "stick": 2},
                "result_count": 1,
            },
            # ---- Stone tools ----
            "stone_pickaxe": {
                "ingredients": {"cobblestone": 3, "stick": 2},
                "result_count": 1,
            },
            "stone_axe": {
                "ingredients": {"cobblestone": 3, "stick": 2},
                "result_count": 1,
            },
            "stone_sword": {
                "ingredients": {"cobblestone": 2, "stick": 1},
                "result_count": 1,
            },
            "stone_shovel": {
                "ingredients": {"cobblestone": 1, "stick": 2},
                "result_count": 1,
            },
            "stone_hoe": {
                "ingredients": {"cobblestone": 2, "stick": 2},
                "result_count": 1,
            },
            # ---- Iron tools ----
            "iron_pickaxe": {
                "ingredients": {"iron_ingot": 3, "stick": 2},
                "result_count": 1,
            },
            "iron_axe": {
                "ingredients": {"iron_ingot": 3, "stick": 2},
                "result_count": 1,
            },
            "iron_sword": {
                "ingredients": {"iron_ingot": 2, "stick": 1},
                "result_count": 1,
            },
            "iron_shovel": {
                "ingredients": {"iron_ingot": 1, "stick": 2},
                "result_count": 1,
            },
            "iron_hoe": {
                "ingredients": {"iron_ingot": 2, "stick": 2},
                "result_count": 1,
            },
            # ---- Diamond tools ----
            "diamond_pickaxe": {
                "ingredients": {"diamond": 3, "stick": 2},
                "result_count": 1,
            },
            "diamond_axe": {
                "ingredients": {"diamond": 3, "stick": 2},
                "result_count": 1,
            },
            "diamond_sword": {
                "ingredients": {"diamond": 2, "stick": 1},
                "result_count": 1,
            },
            "diamond_shovel": {
                "ingredients": {"diamond": 1, "stick": 2},
                "result_count": 1,
            },
            "diamond_hoe": {
                "ingredients": {"diamond": 2, "stick": 2},
                "result_count": 1,
            },
            # ---- Golden tools ----
            "golden_pickaxe": {
                "ingredients": {"gold_ingot": 3, "stick": 2},
                "result_count": 1,
            },
            "golden_axe": {
                "ingredients": {"gold_ingot": 3, "stick": 2},
                "result_count": 1,
            },
            "golden_sword": {
                "ingredients": {"gold_ingot": 2, "stick": 1},
                "result_count": 1,
            },
            # ---- Armor (iron) ----
            "iron_helmet": {
                "ingredients": {"iron_ingot": 5},
                "result_count": 1,
            },
            "iron_chestplate": {
                "ingredients": {"iron_ingot": 8},
                "result_count": 1,
            },
            "iron_leggings": {
                "ingredients": {"iron_ingot": 7},
                "result_count": 1,
            },
            "iron_boots": {
                "ingredients": {"iron_ingot": 4},
                "result_count": 1,
            },
            # ---- Armor (diamond) ----
            "diamond_helmet": {
                "ingredients": {"diamond": 5},
                "result_count": 1,
            },
            "diamond_chestplate": {
                "ingredients": {"diamond": 8},
                "result_count": 1,
            },
            "diamond_leggings": {
                "ingredients": {"diamond": 7},
                "result_count": 1,
            },
            "diamond_boots": {
                "ingredients": {"diamond": 4},
                "result_count": 1,
            },
            # ---- Miscellaneous ----
            "bucket": {
                "ingredients": {"iron_ingot": 3},
                "result_count": 1,
            },
            "shears": {
                "ingredients": {"iron_ingot": 2},
                "result_count": 1,
            },
            "flint_and_steel": {
                "ingredients": {"iron_ingot": 1, "flint": 1},
                "result_count": 1,
            },
            "bow": {
                "ingredients": {"stick": 3, "string": 3},
                "result_count": 1,
            },
            "arrow": {
                "ingredients": {"flint": 1, "stick": 1, "feather": 1},
                "result_count": 4,
            },
            "shield": {
                "ingredients": {"oak_planks": 6, "iron_ingot": 1},
                "result_count": 1,
            },
            "bed": {
                "ingredients": {"oak_planks": 3, "white_wool": 3},
                "result_count": 1,
            },
            "door": {
                "ingredients": {"oak_planks": 6},
                "result_count": 3,
            },
            "ladder": {
                "ingredients": {"stick": 7},
                "result_count": 3,
            },
        }

    def _load_tool_tiers(self) -> None:
        """Define tool tier mapping.

        Tier 1 = wood/gold, Tier 2 = stone, Tier 3 = iron, Tier 4 = diamond.
        """
        self.tool_tiers: Dict[str, int] = {
            "wooden_pickaxe": 1,
            "wooden_axe": 1,
            "wooden_sword": 1,
            "wooden_shovel": 1,
            "wooden_hoe": 1,
            "stone_pickaxe": 2,
            "stone_axe": 2,
            "stone_sword": 2,
            "stone_shovel": 2,
            "stone_hoe": 2,
            "iron_pickaxe": 3,
            "iron_axe": 3,
            "iron_sword": 3,
            "iron_shovel": 3,
            "iron_hoe": 3,
            "diamond_pickaxe": 4,
            "diamond_axe": 4,
            "diamond_sword": 4,
            "diamond_shovel": 4,
            "diamond_hoe": 4,
            "golden_pickaxe": 1,
            "golden_axe": 1,
            "golden_sword": 1,
            "golden_shovel": 1,
            "golden_hoe": 1,
        }
        self.tool_speeds: Dict[str, float] = {
            "wooden_pickaxe": 0.35,
            "wooden_axe": 0.35,
            "wooden_shovel": 0.35,
            "wooden_sword": 0.35,
            "stone_pickaxe": 0.5,
            "stone_axe": 0.5,
            "stone_shovel": 0.5,
            "stone_sword": 0.5,
            "iron_pickaxe": 0.65,
            "iron_axe": 0.65,
            "iron_shovel": 0.65,
            "iron_sword": 0.65,
            "diamond_pickaxe": 0.85,
            "diamond_axe": 0.85,
            "diamond_shovel": 0.85,
            "diamond_sword": 0.85,
            "golden_pickaxe": 1.0,
            "golden_axe": 1.0,
            "golden_shovel": 1.0,
            "golden_sword": 1.0,
        }

    def _load_mob_info(self) -> None:
        """Define mob behavior knowledge."""
        self.mob_info: Dict[str, Dict] = {
            # --- Hostile mobs ---
            "zombie": {
                "hostile": True,
                "ranged": False,
                "health": 20,
                "damage": 3,
                "sunlight_sensitive": True,
            },
            "skeleton": {
                "hostile": True,
                "ranged": True,
                "health": 20,
                "damage": 4,
                "sunlight_sensitive": True,
            },
            "creeper": {
                "hostile": True,
                "ranged": False,
                "health": 20,
                "damage": 49,
                "sunlight_sensitive": False,
            },
            "spider": {
                "hostile": True,
                "ranged": False,
                "health": 16,
                "damage": 2,
                "sunlight_sensitive": False,
            },
            "enderman": {
                "hostile": True,
                "ranged": False,
                "health": 40,
                "damage": 7,
                "sunlight_sensitive": False,
            },
            "witch": {
                "hostile": True,
                "ranged": True,
                "health": 26,
                "damage": 0,
                "sunlight_sensitive": False,
            },
            "drowned": {
                "hostile": True,
                "ranged": False,
                "health": 20,
                "damage": 3,
                "sunlight_sensitive": False,
            },
            "husk": {
                "hostile": True,
                "ranged": False,
                "health": 20,
                "damage": 3,
                "sunlight_sensitive": False,
            },
            "stray": {
                "hostile": True,
                "ranged": True,
                "health": 20,
                "damage": 4,
                "sunlight_sensitive": False,
            },
            "phantom": {
                "hostile": True,
                "ranged": False,
                "health": 20,
                "damage": 6,
                "sunlight_sensitive": True,
            },
            "slime": {
                "hostile": True,
                "ranged": False,
                "health": 16,
                "damage": 4,
                "sunlight_sensitive": False,
            },
            # --- Passive mobs ---
            "cow": {
                "hostile": False,
                "ranged": False,
                "health": 10,
                "damage": 0,
                "sunlight_sensitive": False,
            },
            "pig": {
                "hostile": False,
                "ranged": False,
                "health": 10,
                "damage": 0,
                "sunlight_sensitive": False,
            },
            "sheep": {
                "hostile": False,
                "ranged": False,
                "health": 8,
                "damage": 0,
                "sunlight_sensitive": False,
            },
            "chicken": {
                "hostile": False,
                "ranged": False,
                "health": 4,
                "damage": 0,
                "sunlight_sensitive": False,
            },
            "rabbit": {
                "hostile": False,
                "ranged": False,
                "health": 3,
                "damage": 0,
                "sunlight_sensitive": False,
            },
            "horse": {
                "hostile": False,
                "ranged": False,
                "health": 30,
                "damage": 0,
                "sunlight_sensitive": False,
            },
            "villager": {
                "hostile": False,
                "ranged": False,
                "health": 20,
                "damage": 0,
                "sunlight_sensitive": False,
            },
        }

    def _load_block_info(self) -> None:
        """Define block hardness, required tools, and optimal mining levels."""
        self.block_hardness: Dict[str, float] = {
            "stone": 1.5,
            "cobblestone": 2.0,
            "dirt": 0.5,
            "grass_block": 0.6,
            "sand": 0.5,
            "gravel": 0.6,
            "sandstone": 0.8,
            "oak_log": 2.0,
            "birch_log": 2.0,
            "spruce_log": 2.0,
            "jungle_log": 2.0,
            "acacia_log": 2.0,
            "dark_oak_log": 2.0,
            "oak_planks": 2.0,
            "bedrock": -1.0,
            "obsidian": 50.0,
            "coal_ore": 3.0,
            "iron_ore": 3.0,
            "gold_ore": 3.0,
            "diamond_ore": 3.0,
            "redstone_ore": 3.0,
            "lapis_ore": 3.0,
            "emerald_ore": 3.0,
            "copper_ore": 3.0,
            "deepslate": 3.0,
            "deepslate_iron_ore": 4.5,
            "deepslate_gold_ore": 4.5,
            "deepslate_diamond_ore": 4.5,
            "deepslate_redstone_ore": 4.5,
            "deepslate_lapis_ore": 4.5,
            "deepslate_emerald_ore": 4.5,
            "deepslate_coal_ore": 4.5,
            "deepslate_copper_ore": 4.5,
            "nether_quartz_ore": 3.0,
            "ancient_debris": 30.0,
        }
        self.block_required_tools: Dict[str, str] = {
            "stone": "pickaxe",
            "cobblestone": "pickaxe",
            "coal_ore": "pickaxe",
            "iron_ore": "pickaxe",
            "gold_ore": "pickaxe",
            "diamond_ore": "pickaxe",
            "redstone_ore": "pickaxe",
            "lapis_ore": "pickaxe",
            "emerald_ore": "pickaxe",
            "copper_ore": "pickaxe",
            "deepslate": "pickaxe",
            "deepslate_iron_ore": "pickaxe",
            "deepslate_gold_ore": "pickaxe",
            "deepslate_diamond_ore": "pickaxe",
            "deepslate_redstone_ore": "pickaxe",
            "deepslate_lapis_ore": "pickaxe",
            "deepslate_emerald_ore": "pickaxe",
            "deepslate_coal_ore": "pickaxe",
            "deepslate_copper_ore": "pickaxe",
            "nether_quartz_ore": "pickaxe",
            "ancient_debris": "pickaxe",
            "obsidian": "pickaxe",
            "oak_log": "axe",
            "birch_log": "axe",
            "spruce_log": "axe",
            "jungle_log": "axe",
            "acacia_log": "axe",
            "dark_oak_log": "axe",
            "oak_planks": "axe",
            "dirt": "shovel",
            "sand": "shovel",
            "gravel": "shovel",
            "grass_block": "shovel",
        }
        self.optimal_y_levels: Dict[str, int] = {
            "diamond": -59,
            "iron": 16,
            "coal": 0,
            "gold": -16,
            "redstone": -59,
            "lapis": 0,
            "emerald": -16,
            "copper": 48,
            "ancient_debris": 15,
            "nether_quartz": 15,
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_recipe(self, item_name: str) -> Optional[Dict]:
        """Get crafting recipe for an item.

        Returns a dict with ``ingredients`` (item -> count mapping) and
        ``result_count``, or ``None`` if no recipe is known.
        """
        return self.recipes.get(item_name)

    def get_tool_tier(self, tool_name: str) -> int:
        """Get tool tier.

        Returns 1 (wood/gold), 2 (stone), 3 (iron), 4 (diamond), or 0
        for non-tools.
        """
        return self.tool_tiers.get(tool_name, 0)

    def get_mob_info(self, mob_name: str) -> Optional[Dict]:
        """Get mob behavior info.

        Returns a dict with keys ``hostile``, ``ranged``, ``health``,
        ``damage``, ``sunlight_sensitive``, or ``None`` if unknown.
        """
        return self.mob_info.get(mob_name)

    def get_block_hardness(self, block_name: str) -> float:
        """Get block hardness.

        Returns a positive float where higher = harder to break, or ``-1``
        for unbreakable blocks (e.g. bedrock).
        """
        return self.block_hardness.get(block_name, 1.0)

    def get_required_tool(self, block_name: str) -> Optional[str]:
        """Get required tool type for a block.

        Returns ``"pickaxe"``, ``"axe"``, ``"shovel"``, or ``None`` if
        the block can be mined by hand.
        """
        return self.block_required_tools.get(block_name)

    def get_optimal_y(self, resource: str) -> Optional[int]:
        """Get optimal Y level for mining a resource.

        Returns the recommended altitude (can be negative for deep ores),
        or ``None`` if unknown.
        """
        return self.optimal_y_levels.get(resource)

    def can_craft(self, item_name: str, inventory: Dict[str, int]) -> bool:
        """Check whether *item_name* can be crafted from *inventory*.

        Args:
            item_name: The desired crafted item.
            inventory: Mapping from item name to available count.

        Returns:
            ``True`` if the recipe exists and every ingredient is present
            in sufficient quantity.
        """
        recipe = self.get_recipe(item_name)
        if recipe is None:
            return False
        ingredients = recipe["ingredients"]
        for needed_item, needed_count in ingredients.items():
            if inventory.get(needed_item, 0) < needed_count:
                return False
        return True

    def get_craftable_items(self, inventory: Dict[str, int]) -> List[str]:
        """Get list of all items that can be crafted from *inventory*.

        Args:
            inventory: Mapping from item name to available count.

        Returns:
            Alphabetically sorted list of craftable item names.
        """
        craftable = [
            item_name
            for item_name in self.recipes.keys()
            if self.can_craft(item_name, inventory)
        ]
        return sorted(craftable)

    def get_mining_efficiency(self, block_name: str, tool_name: str) -> float:
        """Calculate a mining efficiency score in the range ``[0, 1]``.

        Higher values indicate a faster / more appropriate tool for the
        block.  Uses tool speed, required tool type matching, and tool
        tier to compute the score.

        Args:
            block_name: The block being mined.
            tool_name: The tool used (e.g. ``"iron_pickaxe"``).

        Returns:
            Efficiency score between 0 and 1.
        """
        required = self.get_required_tool(block_name)
        if required is None:
            # Hand-minable block -- any tool is fine, but correct type helps
            speed = self.tool_speeds.get(tool_name, 0.15)
            return float(np.clip(speed, 0.0, 1.0))

        # Determine tool type from name suffix
        tool_type: Optional[str] = None
        for suffix in ("pickaxe", "axe", "shovel", "sword", "hoe"):
            if tool_name.endswith(suffix):
                tool_type = suffix
                break

        if tool_type is None:
            # Unknown tool -- bare-hand efficiency
            return 0.1

        if tool_type != required:
            # Wrong tool type -- major penalty
            return 0.0

        # Right tool type -- base on speed and tier
        speed = self.tool_speeds.get(tool_name, 0.15)
        tier = self.get_tool_tier(tool_name)
        tier_bonus = (tier - 1) * 0.05  # +0% / +5% / +10% / +15%
        score = min(speed + tier_bonus, 1.0)
        return float(score)


# Convenience singleton for global access
_default_kb: Optional[MinecraftKnowledgeBase] = None


def get_default_knowledge_base() -> MinecraftKnowledgeBase:
    """Return the default (singleton) knowledge base."""
    global _default_kb
    if _default_kb is None:
        _default_kb = MinecraftKnowledgeBase()
    return _default_kb
