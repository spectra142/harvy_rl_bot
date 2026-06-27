"""FastAPI + WebSocket bridge for the mission control dashboard."""

import asyncio
import json
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mission_control.training_runner import TrainingManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

manager = TrainingManager()

DASHBOARD_DIST = (
    Path(__file__).resolve().parents[2]
    / "mission_control"
    / "dashboard"
    / "app"
    / "dist"
)


class ConnectionManager:
    def __init__(self):
        self.connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.connections:
            self.connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        dead = []
        for conn in self.connections:
            try:
                await conn.send_json(message)
            except Exception:
                dead.append(conn)
        for conn in dead:
            self.disconnect(conn)


conn_manager = ConnectionManager()


async def broadcaster():
    """Forward queued messages from the training manager to all dashboards."""
    while True:
        msg = manager.pop_message(timeout=0.05)
        if msg:
            await conn_manager.broadcast(msg)
        else:
            await asyncio.sleep(0.05)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background broadcaster on startup."""
    broadcaster_task = asyncio.create_task(broadcaster())
    yield
    broadcaster_task.cancel()
    try:
        await broadcaster_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Harvy Mission Control", lifespan=lifespan)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await conn_manager.connect(websocket)
    await websocket.send_json(manager.status())
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = data.get("type")
            if msg_type == "start_training":
                manager.start_training(
                    config=data.get("config", "configs/default.yaml"),
                    device=data.get("device", "cpu"),
                    total_timesteps=data.get("total_timesteps"),
                )
            elif msg_type == "pause_training":
                manager.pause_training()
            elif msg_type == "resume_training":
                manager.resume_training()
            elif msg_type == "stop_training":
                manager.stop_training()
            elif msg_type == "reset_environment":
                manager.reset_environment(goal=data.get("goal"))
            elif msg_type == "manual_action":
                manager.set_manual_action(data.get("action"))
            elif msg_type == "set_hyperparam":
                manager.set_hyperparam(data.get("key"), data.get("value"))
            elif msg_type == "execute_command":
                manager.execute_command(data.get("command", ""))
            elif msg_type == "inventory_action":
                manager.inventory_action(data.get("action"), data)
            else:
                await websocket.send_json(
                    {"type": "error", "message": f"Unknown type: {msg_type}"}
                )
    except WebSocketDisconnect:
        conn_manager.disconnect(websocket)


if DASHBOARD_DIST.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(DASHBOARD_DIST / "assets")),
        name="assets",
    )

    @app.get("/{full_path:path}")
    async def serve_dashboard(full_path: str):
        index = DASHBOARD_DIST / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return {"detail": "Dashboard not built"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9877)
