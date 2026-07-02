"""FastAPI web control panel for Harvy v2."""

import os
import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.knowledge_base import KnowledgeBase
from web.command_queue import CommandQueue

MEMORY_DIR = os.environ.get("HARVY_MEMORY_DIR", "./memory")
DB_PATH = Path(MEMORY_DIR) / "harvy_memory.db"
SECRET_TOKEN = os.environ.get("HARVY_WEB_TOKEN", "changeme")

app = FastAPI(title="Harvy Control Panel")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "static"))


def get_kb() -> KnowledgeBase:
    return KnowledgeBase(str(DB_PATH))


def get_queue() -> CommandQueue:
    return CommandQueue(str(DB_PATH))


def require_token(request: Request):
    token = request.headers.get("x-harvy-token")
    if token != SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status")
def status(_=Depends(require_token)):
    kb = get_kb()
    try:
        players = kb.list_players()
        recent_deaths = kb.recent_deaths(5)
        recent_combat = kb.combat_summary(n=5)
        return {
            "players": players,
            "recent_deaths": recent_deaths,
            "recent_combat": recent_combat,
        }
    finally:
        kb.close()


@app.post("/api/command/{cmd}")
def issue_command(cmd: str, args: str = "", _=Depends(require_token)):
    queue = get_queue()
    try:
        queue.push(cmd, args, issued_by="web")
        return {"ok": True, "command": cmd, "args": args}
    finally:
        queue.close()


@app.get("/api/commands")
def list_commands(_=Depends(require_token)):
    queue = get_queue()
    try:
        return {
            "pending": queue.list_pending(),
            "history": queue.list_history(),
        }
    finally:
        queue.close()


@app.get("/api/players")
def list_players(_=Depends(require_token)):
    kb = get_kb()
    try:
        return {"players": kb.list_players()}
    finally:
        kb.close()


@app.post("/api/players/{name}/threat")
def set_threat(name: str, threat: str, _=Depends(require_token)):
    kb = get_kb()
    try:
        kb.set_threat("", name, threat)
        return {"ok": True, "name": name, "threat": threat}
    finally:
        kb.close()
