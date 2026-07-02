"""Tests for persistent knowledge base."""

import tempfile
from pathlib import Path

from memory.knowledge_base import KnowledgeBase


def test_player_upsert_and_owner():
    with tempfile.TemporaryDirectory() as tmp:
        kb = KnowledgeBase(Path(tmp) / "test.db")
        kb.set_owner("uuid-1", "owner_name")
        assert kb.is_owner("uuid-1")
        assert kb.is_owner("owner_name")
        kb.set_threat("uuid-2", "griefer", "hostile")
        assert kb.is_hostile("griefer")
        assert not kb.is_hostile("owner_name")
        kb.close()


def test_death_recorded():
    with tempfile.TemporaryDirectory() as tmp:
        kb = KnowledgeBase(Path(tmp) / "test.db")
        kb.record_death(
            {
                "cause": "zombie",
                "killer_uuid": "z-1",
                "killer_name": "Zombie",
                "position": [10.0, 64.0, -3.0],
                "health": 0,
                "inventory": {"oak_log": 5},
            }
        )
        deaths = kb.recent_deaths(5)
        assert len(deaths) == 1
        assert deaths[0]["killer_name"] == "Zombie"
        kb.close()


def test_discovery_recorded():
    with tempfile.TemporaryDirectory() as tmp:
        kb = KnowledgeBase(Path(tmp) / "test.db")
        kb.record_discovery(5, -3, "diamond_ore", value=10.0, note="cave")
        discoveries = kb.get_discoveries(5, -3)
        assert len(discoveries) == 1
        assert discoveries[0]["block_type"] == "diamond_ore"
        kb.close()
