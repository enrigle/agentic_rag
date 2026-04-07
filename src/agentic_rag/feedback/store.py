"""SQLite-backed feedback store."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("feedback.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS feedback (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT    NOT NULL,
    query     TEXT    NOT NULL,
    answer    TEXT    NOT NULL,
    sources   TEXT    NOT NULL,
    top_score REAL    NOT NULL,
    rating    INTEGER NOT NULL,
    note      TEXT    NOT NULL DEFAULT '',
    category  TEXT    NOT NULL DEFAULT ''
)
"""


@dataclass
class FeedbackEntry:
    query: str
    answer: str
    sources: list[dict]  # [{title, content, score}]
    top_score: float
    rating: int  # 1 = thumbs up, -1 = thumbs down
    note: str = ""
    category: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    id: int | None = None


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(_CREATE_TABLE)
    conn.commit()


def save(entry: FeedbackEntry, db_path: Path = DB_PATH) -> int:
    """Persist a FeedbackEntry. Returns the new row id."""
    with sqlite3.connect(db_path) as conn:
        _ensure_table(conn)
        cursor = conn.execute(
            "INSERT INTO feedback (timestamp, query, answer, sources, top_score, rating, note, category) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.timestamp,
                entry.query,
                entry.answer,
                json.dumps(entry.sources),
                entry.top_score,
                entry.rating,
                entry.note,
                entry.category,
            ),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]


def get_all(db_path: Path = DB_PATH) -> list[FeedbackEntry]:
    """Return all entries ordered oldest first."""
    with sqlite3.connect(db_path) as conn:
        _ensure_table(conn)
        rows = conn.execute(
            "SELECT id, timestamp, query, answer, sources, top_score, rating, note, category "
            "FROM feedback ORDER BY id ASC"
        ).fetchall()
    return [
        FeedbackEntry(
            id=row[0],
            timestamp=row[1],
            query=row[2],
            answer=row[3],
            sources=json.loads(row[4]),
            top_score=row[5],
            rating=row[6],
            note=row[7],
            category=row[8],
        )
        for row in rows
    ]


def update_category(entry_id: int, category: str, db_path: Path = DB_PATH) -> None:
    """Write the judge's classification back to the row."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE feedback SET category = ? WHERE id = ?",
            (category, entry_id),
        )
        conn.commit()
