"""Persistent SQLite knowledge base for Harvy v2.

Stores player reputations, deaths, discoveries, stashes, and combat outcomes
so the bot "remembers" across runs.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

DEFAULT_DB_PATH = "./memory/harvy_memory.db"

THREAT_LEVELS = ["owner", "ally", "neutral", "hostile"]


class KnowledgeBase:
    """Thread-unsafe SQLite knowledge base. Use one instance per process."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS players (
            uuid TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            threat_level TEXT NOT NULL DEFAULT 'neutral',
            last_seen_x REAL,
            last_seen_y REAL,
            last_seen_z REAL,
            last_seen_time REAL,
            owner INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS deaths (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time REAL NOT NULL,
            cause TEXT,
            killer_uuid TEXT,
            killer_name TEXT,
            pos_x REAL,
            pos_y REAL,
            pos_z REAL,
            health REAL,
            inventory_json TEXT
        );

        CREATE TABLE IF NOT EXISTS discoveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time REAL NOT NULL,
            chunk_x INTEGER,
            chunk_z INTEGER,
            block_type TEXT,
            value REAL DEFAULT 0.0,
            note TEXT
        );

        CREATE TABLE IF NOT EXISTS stashes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time REAL NOT NULL,
            item_name TEXT NOT NULL,
            pos_x REAL,
            pos_y REAL,
            pos_z REAL,
            count INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS combat_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time REAL NOT NULL,
            target_uuid TEXT,
            target_name TEXT,
            outcome TEXT,  -- 'killed', 'died', 'fled', 'timeout'
            damage_dealt REAL DEFAULT 0.0,
            damage_taken REAL DEFAULT 0.0,
            duration_steps INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_players_name ON players(name);
        CREATE INDEX IF NOT EXISTS idx_deaths_time ON deaths(time);
        CREATE INDEX IF NOT EXISTS idx_discoveries_chunk ON discoveries(chunk_x, chunk_z);
        """
        self._conn.executescript(schema)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Players / trust
    # ------------------------------------------------------------------
    def upsert_player(
        self,
        uuid: str,
        name: str,
        threat_level: Optional[str] = None,
        position: Optional[List[float]] = None,
        is_owner: bool = False,
    ) -> None:
        now = time.time()
        x, y, z = position if position else (None, None, None)
        if threat_level and threat_level not in THREAT_LEVELS:
            threat_level = "neutral"

        existing = self.get_player(uuid)
        if existing:
            # Don't downgrade owner status accidentally.
            final_owner = 1 if is_owner or existing.get("owner") else 0
            final_threat = threat_level or existing.get("threat_level", "neutral")
            self._conn.execute(
                """
                UPDATE players
                SET name = ?, threat_level = ?, last_seen_x = ?, last_seen_y = ?,
                    last_seen_z = ?, last_seen_time = ?, owner = ?
                WHERE uuid = ?
                """,
                (name, final_threat, x, y, z, now, final_owner, uuid),
            )
        else:
            self._conn.execute(
                """
                INSERT INTO players (uuid, name, threat_level, last_seen_x, last_seen_y,
                                     last_seen_z, last_seen_time, owner)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (uuid, name, threat_level or "neutral", x, y, z, now, 1 if is_owner else 0),
            )
        self._conn.commit()

    def get_player(self, uuid: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute("SELECT * FROM players WHERE uuid = ?", (uuid,)).fetchone()
        return dict(row) if row else None

    def get_player_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute("SELECT * FROM players WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None

    def set_owner(self, uuid: str, name: str) -> None:
        self.upsert_player(uuid, name, threat_level="owner", is_owner=True)

    def set_threat(self, uuid: str, name: str, threat_level: str) -> None:
        self.upsert_player(uuid, name, threat_level=threat_level)

    def is_owner(self, uuid_or_name: str) -> bool:
        row = self._conn.execute(
            "SELECT owner FROM players WHERE uuid = ? OR name = ?",
            (uuid_or_name, uuid_or_name),
        ).fetchone()
        return bool(row and row["owner"])

    def is_hostile(self, uuid_or_name: str) -> bool:
        row = self._conn.execute(
            "SELECT threat_level FROM players WHERE uuid = ? OR name = ?",
            (uuid_or_name, uuid_or_name),
        ).fetchone()
        return bool(row and row["threat_level"] == "hostile")

    def list_players(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM players ORDER BY last_seen_time DESC").fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Deaths
    # ------------------------------------------------------------------
    def record_death(self, context: Dict[str, Any]) -> None:
        pos = context.get("position", [None, None, None])
        self._conn.execute(
            """
            INSERT INTO deaths (time, cause, killer_uuid, killer_name, pos_x, pos_y, pos_z,
                                health, inventory_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                context.get("cause"),
                context.get("killer_uuid"),
                context.get("killer_name"),
                pos[0] if len(pos) > 0 else None,
                pos[1] if len(pos) > 1 else None,
                pos[2] if len(pos) > 2 else None,
                context.get("health"),
                json.dumps(context.get("inventory", {})),
            ),
        )
        self._conn.commit()

    def recent_deaths(self, n: int = 10) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM deaths ORDER BY time DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Discoveries
    # ------------------------------------------------------------------
    def record_discovery(
        self,
        chunk_x: int,
        chunk_z: int,
        block_type: str,
        value: float = 0.0,
        note: str = "",
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO discoveries (time, chunk_x, chunk_z, block_type, value, note)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (time.time(), chunk_x, chunk_z, block_type, value, note),
        )
        self._conn.commit()

    def get_discoveries(self, chunk_x: int, chunk_z: int) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM discoveries WHERE chunk_x = ? AND chunk_z = ? ORDER BY time DESC",
            (chunk_x, chunk_z),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Stashes
    # ------------------------------------------------------------------
    def record_stash(
        self,
        item_name: str,
        position: List[float],
        count: int,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO stashes (time, item_name, pos_x, pos_y, pos_z, count)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (time.time(), item_name, position[0], position[1], position[2], count),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Combat
    # ------------------------------------------------------------------
    def record_combat_outcome(self, outcome: Dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO combat_outcomes (time, target_uuid, target_name, outcome,
                                         damage_dealt, damage_taken, duration_steps)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                outcome.get("target_uuid"),
                outcome.get("target_name"),
                outcome.get("outcome"),
                outcome.get("damage_dealt", 0.0),
                outcome.get("damage_taken", 0.0),
                outcome.get("duration_steps", 0),
            ),
        )
        self._conn.commit()

    def combat_summary(self, target_name: Optional[str] = None, n: int = 20) -> List[Dict[str, Any]]:
        if target_name:
            rows = self._conn.execute(
                "SELECT * FROM combat_outcomes WHERE target_name = ? ORDER BY time DESC LIMIT ?",
                (target_name, n),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM combat_outcomes ORDER BY time DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
