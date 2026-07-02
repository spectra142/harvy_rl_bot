"""Simple trajectory storage for warm-starting or offline analysis.

Stores (observation, action, reward) tuples from completed episodes.
Not a full replay buffer — just a way to remember what worked.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DB_PATH = "./memory/harvy_memory.db"


class ExperienceReplay:
    """Stores selected high-reward trajectories."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH, min_episode_reward: float = 5.0):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self.min_episode_reward = min_episode_reward
        self._init_tables()

    def _init_tables(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trajectories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time REAL NOT NULL,
                total_reward REAL NOT NULL,
                length INTEGER NOT NULL,
                skill_name TEXT,
                trajectory_json TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_traj_reward ON trajectories(total_reward)"
        )
        self._conn.commit()

    def store(
        self,
        trajectory: List[Dict[str, Any]],
        total_reward: float,
        skill_name: str = "",
    ) -> bool:
        """Store trajectory if it clears the reward threshold."""
        if total_reward < self.min_episode_reward:
            return False
        self._conn.execute(
            """
            INSERT INTO trajectories (time, total_reward, length, skill_name, trajectory_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                total_reward,
                len(trajectory),
                skill_name,
                json.dumps(trajectory),
            ),
        )
        self._conn.commit()
        return True

    def top_trajectories(
        self,
        skill_name: Optional[str] = None,
        n: int = 10,
    ) -> List[Dict[str, Any]]:
        if skill_name:
            rows = self._conn.execute(
                """
                SELECT * FROM trajectories
                WHERE skill_name = ?
                ORDER BY total_reward DESC
                LIMIT ?
                """,
                (skill_name, n),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT * FROM trajectories
                ORDER BY total_reward DESC
                LIMIT ?
                """,
                (n,),
            ).fetchall()

        results = []
        for row in rows:
            d = dict(row)
            d["trajectory"] = json.loads(d.pop("trajectory_json"))
            results.append(d)
        return results

    def close(self) -> None:
        self._conn.close()
