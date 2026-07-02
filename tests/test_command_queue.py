"""Tests for web command queue."""

import tempfile
from pathlib import Path

from web.command_queue import CommandQueue


def test_push_and_pop():
    with tempfile.TemporaryDirectory() as tmp:
        cq = CommandQueue(Path(tmp) / "test.db")
        cq.push("attack_nearest", "", "web")
        cmd = cq.pop()
        assert cmd["command"] == "attack_nearest"
        assert cq.pop() is None
        cq.close()


def test_pending_list():
    with tempfile.TemporaryDirectory() as tmp:
        cq = CommandQueue(Path(tmp) / "test.db")
        cq.push("build_wall", "", "web")
        cq.push("stop", "", "web")
        pending = cq.list_pending()
        assert len(pending) == 2
        cq.pop()
        assert len(cq.list_pending()) == 1
        cq.close()
