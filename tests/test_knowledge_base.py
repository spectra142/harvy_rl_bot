"""Tests for the Minecraft knowledge base."""

import pytest

from env.knowledge_base import MinecraftKnowledgeBase


@pytest.fixture
def kb():
    return MinecraftKnowledgeBase()


def test_get_recipe(kb):
    recipe = kb.get_recipe("wooden_pickaxe")
    assert recipe is not None
    assert recipe["ingredients"] == {"oak_planks": 3, "stick": 2}


def test_can_craft(kb):
    assert kb.can_craft("wooden_pickaxe", {"oak_planks": 3, "stick": 2})
    assert not kb.can_craft("wooden_pickaxe", {"oak_planks": 1, "stick": 2})


def test_get_tool_tier(kb):
    assert kb.get_tool_tier("wooden_pickaxe") == 1
    assert kb.get_tool_tier("stone_pickaxe") == 2
    assert kb.get_tool_tier("iron_pickaxe") == 3
    assert kb.get_tool_tier("diamond_pickaxe") == 4


def test_get_required_tool(kb):
    assert kb.get_required_tool("stone") == "pickaxe"
    assert kb.get_required_tool("oak_log") == "axe"
    assert kb.get_required_tool("dirt") == "shovel"


def test_get_mining_efficiency_correct_tool(kb):
    assert kb.get_mining_efficiency("stone", "iron_pickaxe") > 0.5


def test_get_mining_efficiency_wrong_tool(kb):
    assert kb.get_mining_efficiency("stone", "wooden_axe") == 0.0


def test_get_optimal_y(kb):
    assert kb.get_optimal_y("diamond") == -59
    assert kb.get_optimal_y("iron") == 16
