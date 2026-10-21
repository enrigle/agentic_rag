"""SQLite-backed feedback store."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, get_args

logger = logging.getLogger(__name__)

Rating = Literal[1, -1]  # thumbs up / thumbs down
FailureCategory = Literal[
    "retrieval_miss", "synthesis_failure", "missing_content", "unknown"
]

DB_PATH = Path("data/feedback.db")

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
    category  TEXT    NOT NULL DEFAULT '',
    top_similarity REAL
)
"""


@dataclass(frozen=True)
class FeedbackSource:
    """A source shown with a rated answer, as stored in the row's JSON column."""

    title: str
    content: str = ""
    score: float = 0.0


@dataclass
class FeedbackEntry:
    query: str
    answer: str
    sources: list[FeedbackSource]
    top_score: float  # best final ranking score (cross-encoder logit, or web score)
    rating: Rating
    note: str = ""
    category: FailureCategory | Literal[""] = ""  # "" = not classified (thumbs up)
    # Best query-chunk cosine; None for web answers and rows saved before this column.
    top_similarity: float | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    id: int | None = None


def _parse_sources(raw_json: str) -> list[FeedbackSource]:
    """Stored JSON → FeedbackSource; tolerates legacy rows with extra/missing keys."""
    return [
        FeedbackSource(
            title=str(d.get("title", "")),
            content=str(d.get("content", "")),
            score=float(d.get("score", 0.0)),
        )
        for d in json.loads(raw_json)
    ]


def _parse_category(raw: str) -> FailureCategory | Literal[""]:
    if raw == "" or raw in get_args(FailureCategory):
        return raw  # type: ignore[return-value]  # narrowed by the membership check
    logger.warning("feedback: unexpected category %r in DB; reading as 'unknown'", raw)
    return "unknown"


def _parse_rating(raw: int) -> Rating:
    if raw not in (1, -1):
        raise ValueError(f"feedback DB holds invalid rating {raw!r}")
    return 1 if raw == 1 else -1


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(_CREATE_TABLE)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(feedback)")}
    if "top_similarity" not in columns:
        # Legacy DB: old rows keep NULL, so the tuner ignores their non-cosine top_score.
        conn.execute("ALTER TABLE feedback ADD COLUMN top_similarity REAL")
    conn.commit()


def save(entry: FeedbackEntry, db_path: Path = DB_PATH) -> int:
    """Persist a FeedbackEntry. Returns the new row id."""
    if entry.rating not in (1, -1):
        raise ValueError(f"rating must be 1 or -1, got {entry.rating!r}")
    with sqlite3.connect(db_path) as conn:
        _ensure_table(conn)
        cursor = conn.execute(
            "INSERT INTO feedback (timestamp, query, answer, sources, top_score, rating, note, "
            "category, top_similarity) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.timestamp,
                entry.query,
                entry.answer,
                json.dumps([asdict(s) for s in entry.sources]),
                entry.top_score,
                entry.rating,
                entry.note,
                entry.category,
                entry.top_similarity,
            ),
        )
        conn.commit()
        row_id = cursor.lastrowid
        if row_id is None:
            raise RuntimeError("INSERT succeeded but returned no lastrowid")
        return row_id


def get_all(db_path: Path = DB_PATH) -> list[FeedbackEntry]:
    """Return all entries ordered oldest first."""
    with sqlite3.connect(db_path) as conn:
        _ensure_table(conn)
        rows = conn.execute(
            "SELECT id, timestamp, query, answer, sources, top_score, rating, note, category, "
            "top_similarity FROM feedback ORDER BY id ASC"
        ).fetchall()
    return [
        FeedbackEntry(
            id=row[0],
            timestamp=row[1],
            query=row[2],
            answer=row[3],
            sources=_parse_sources(row[4]),
            top_score=row[5],
            rating=_parse_rating(row[6]),
            note=row[7],
            category=_parse_category(row[8]),
            top_similarity=row[9],
        )
        for row in rows
    ]


def update_category(
    entry_id: int, category: FailureCategory, db_path: Path = DB_PATH
) -> None:
    """Write the judge's classification back to the row."""
    with sqlite3.connect(db_path) as conn:
        _ensure_table(conn)
        conn.execute(
            "UPDATE feedback SET category = ? WHERE id = ?",
            (category, entry_id),
        )
        conn.commit()
