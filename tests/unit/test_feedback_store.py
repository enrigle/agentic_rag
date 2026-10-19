from pathlib import Path
from agentic_rag.feedback.store import FeedbackEntry, save, get_all, update_category


def test_save_and_get_all(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    entry = FeedbackEntry(
        query="What is ML?",
        answer="Machine learning is...",
        sources=[{"title": "ML Basics", "content": "ML intro.", "score": 0.05}],
        top_score=0.05,
        rating=1,
    )
    entry_id = save(entry, db_path=db)
    assert entry_id == 1
    all_entries = get_all(db_path=db)
    assert len(all_entries) == 1
    assert all_entries[0].query == "What is ML?"
    assert all_entries[0].rating == 1
    assert all_entries[0].sources == [
        {"title": "ML Basics", "content": "ML intro.", "score": 0.05}
    ]


def test_update_category(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    entry = FeedbackEntry(query="q", answer="a", sources=[], top_score=0.0, rating=-1)
    entry_id = save(entry, db_path=db)
    update_category(entry_id, "retrieval_miss", db_path=db)
    entries = get_all(db_path=db)
    assert entries[0].category == "retrieval_miss"


def test_save_empty_sources(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    entry = FeedbackEntry(query="q", answer="a", sources=[], top_score=0.0, rating=1)
    save(entry, db_path=db)
    entries = get_all(db_path=db)
    assert entries[0].sources == []


def test_multiple_entries_oldest_first(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    for i in range(3):
        save(
            FeedbackEntry(
                query=f"q{i}", answer="a", sources=[], top_score=0.0, rating=1
            ),
            db_path=db,
        )
    entries = get_all(db_path=db)
    assert [e.query for e in entries] == ["q0", "q1", "q2"]


def test_update_category_nonexistent_id_is_noop(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    # No error raised, no rows affected — silent no-op
    update_category(999, "retrieval_miss", db_path=db)
    assert get_all(db_path=db) == []


def test_top_similarity_round_trips(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    save(
        FeedbackEntry(
            query="q",
            answer="a",
            sources=[],
            top_score=-3.2,
            rating=1,
            top_similarity=0.71,
        ),
        db_path=db,
    )
    save(
        FeedbackEntry(query="q", answer="a", sources=[], top_score=0.9, rating=-1),
        db_path=db,
    )
    entries = get_all(db_path=db)
    assert entries[0].top_similarity == 0.71
    assert entries[1].top_similarity is None


def test_legacy_table_gets_top_similarity_column(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "legacy.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "timestamp TEXT NOT NULL, query TEXT NOT NULL, answer TEXT NOT NULL, "
            "sources TEXT NOT NULL, top_score REAL NOT NULL, rating INTEGER NOT NULL, "
            "note TEXT NOT NULL DEFAULT '', category TEXT NOT NULL DEFAULT '')"
        )
        conn.execute(
            "INSERT INTO feedback (timestamp, query, answer, sources, top_score, rating) "
            "VALUES ('t', 'old q', 'old a', '[]', -4.0, 1)"
        )
    entries = get_all(db_path=db)
    assert entries[0].query == "old q"
    assert entries[0].top_similarity is None
