"""SQLite-backed command queue for the web control panel.

The web API writes commands here. The MinecraftEnv polls for pending commands
and executes them.
"""

import sqlite3
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

DEFAULT_DB_PATH = "./memory/harvy_memory.db"


class CommandQueue:
    """Thread-safe-ish SQLite command queue."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_table()

    def _init_table(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_commands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time REAL NOT NULL,
                command TEXT NOT NULL,
                args TEXT,
                issued_by TEXT,
                executed INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.commit()

    def push(self, command: str, args: str = "", issued_by: str = "web") -> None:
        self._conn.execute(
            "INSERT INTO pending_commands (time, command, args, issued_by) VALUES (?, ?, ?, ?)",
            (time.time(), command, args, issued_by),
        )
        self._conn.commit()

    def pop(self) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM pending_commands WHERE executed = 0 ORDER BY time ASC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        self._conn.execute(
            "UPDATE pending_commands SET executed = 1 WHERE id = ?", (row["id"],)
        )
        self._conn.commit()
        return dict(row)

    def list_pending(self, limit: int = 20) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM pending_commands WHERE executed = 0 ORDER BY time DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM pending_commands ORDER BY time DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
