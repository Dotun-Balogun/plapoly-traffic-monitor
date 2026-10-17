"""Tiny SQLite store for past analysis runs (powers the history chart)."""
import json
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path

# Set DATA_DIR to a persistent disk path (e.g. /var/data on Render) to keep
# history across restarts. Defaults to ./data next to this file.
DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent / "data"))
DB_PATH = DATA_DIR / "history.db"


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with closing(_connect()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                filename TEXT,
                total INTEGER,
                by_type TEXT,
                congestion TEXT,
                avg_concurrent REAL,
                avg_speed_kmh REAL,
                zone_counts TEXT
            )
            """
        )
        conn.commit()


def save_run(result: dict, filename: str) -> None:
    with closing(_connect()) as conn:
        conn.execute(
            """INSERT INTO runs (timestamp, filename, total, by_type, congestion,
                                 avg_concurrent, avg_speed_kmh, zone_counts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(),
                filename,
                result["total"],
                json.dumps(result["by_type"]),
                result["congestion"],
                result["avg_concurrent"],
                result["avg_speed_kmh"],
                json.dumps(result["zone_counts"]) if result["zone_counts"] is not None else None,
            ),
        )
        conn.commit()


def load_history(limit: int = 50) -> list[dict]:
    """Return the most recent runs, oldest first (left-to-right on a chart)."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [
        {
            "id": r["id"],
            "timestamp": r["timestamp"],
            "filename": r["filename"],
            "total": r["total"],
            "by_type": json.loads(r["by_type"]) if r["by_type"] else {},
            "congestion": r["congestion"],
            "avg_concurrent": r["avg_concurrent"],
            "avg_speed_kmh": r["avg_speed_kmh"],
            "zone_counts": json.loads(r["zone_counts"]) if r["zone_counts"] else None,
        }
        for r in reversed(rows)
    ]


def clear_history() -> None:
    with closing(_connect()) as conn:
        conn.execute("DELETE FROM runs")
        conn.commit()
