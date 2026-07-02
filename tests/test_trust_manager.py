"""Tests for trust / owner authorization."""

import tempfile
from pathlib import Path

from memory.knowledge_base import KnowledgeBase
from memory.trust_manager import TrustManager


def test_owner_can_issue_commands():
    with tempfile.TemporaryDirectory() as tmp:
        kb = KnowledgeBase(Path(tmp) / "test.db")
        tm = TrustManager(kb)
        tm.set_owner("uuid-owner", "Alice")
        assert tm.is_owner("Alice")
        assert tm.is_authorized("Alice")

        response = tm.handle_chat({"username": "Alice", "message": "harvy trust Bob"})
        assert "bob is now trusted" in response.lower()
        assert tm.is_authorized("Bob")
        kb.close()


def test_unauthorized_chat_ignored():
    with tempfile.TemporaryDirectory() as tmp:
        kb = KnowledgeBase(Path(tmp) / "test.db")
        tm = TrustManager(kb)
        tm.set_owner("uuid-owner", "Alice")
        response = tm.handle_chat({"username": "Eve", "message": "harvy trust Bob"})
        assert response is None
        kb.close()


def test_hostile_marking():
    with tempfile.TemporaryDirectory() as tmp:
        kb = KnowledgeBase(Path(tmp) / "test.db")
        tm = TrustManager(kb)
        tm.set_owner("uuid-owner", "Alice")
        tm.handle_chat({"username": "Alice", "message": "harvy hostile Griefer"})
        assert tm.is_hostile("Griefer")
        kb.close()
