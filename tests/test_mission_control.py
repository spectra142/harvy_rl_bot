"""Tests for the Mission Control bridge server."""

import httpx
import pytest
from fastapi.testclient import TestClient

from mission_control.server import app, manager


@pytest.fixture(autouse=True)
def reset_manager(monkeypatch):
    """Reset the module-level manager and avoid real training threads."""
    manager.stop_training()
    manager.pending_action = None
    manager.pending_reset = None
    manager.manual_override = False
    manager.hyperparams.clear()
    manager._status.update(
        {"isTraining": False, "isPaused": False, "episodeCount": 0, "stepCount": 0}
    )
    # Drain any stale queued messages.
    while manager.pop_message(timeout=0.0):
        pass

    def _fake_start_training(config=None, device=None, total_timesteps=None):
        manager._update_status(isTraining=True, isPaused=False)

    monkeypatch.setattr(manager, "start_training", _fake_start_training)

    yield

    manager.stop_training()
    manager.pending_action = None
    manager.pending_reset = None
    manager.manual_override = False
    manager.hyperparams.clear()
    while manager.pop_message(timeout=0.0):
        pass


@pytest.mark.asyncio
async def test_websocket_status():
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            msg = ws.receive_json()
    assert msg["type"] == "status"
    assert "isTraining" in msg


@pytest.mark.asyncio
async def test_start_training_message():
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # initial status
            ws.send_json({"type": "start_training"})
            # Give the server coroutine a chance to process the message.
            import asyncio

            await asyncio.sleep(0.05)
    assert manager._status["isTraining"] is True


@pytest.mark.asyncio
async def test_manual_action_message():
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # initial status
            ws.send_json({"type": "manual_action", "action": "jump"})
    assert manager.pending_action == "jump"
    assert manager.manual_override is True


@pytest.mark.asyncio
async def test_set_hyperparam_message():
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # initial status
            ws.send_json(
                {"type": "set_hyperparam", "key": "learningRate", "value": 0.001}
            )
    assert manager.hyperparams["learningRate"] == 0.001


@pytest.mark.asyncio
async def test_reset_environment_message():
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # initial status
            ws.send_json({"type": "reset_environment", "goal": "explore"})
    assert manager.pending_reset == "explore"


@pytest.mark.asyncio
async def test_dashboard_serves_static():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
