import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterable, Tuple, List

DB_PATH = "sentiments.db"


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """
    Create the sentiments table if it does not exist.
    Columns: id (PK), text, sentiment, timestamp (ISO string).
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sentiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.commit()


def insert_sentiment(text: str, sentiment: str) -> None:
    """
    Insert a new sentiment record using a parameterized query
    to avoid SQL injection issues.
    """
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?)",
            (text, sentiment, ts),
        )
        conn.commit()


def fetch_recent(limit: int = 50) -> List[Tuple[int, str, str, str]]:
    """
    Fetch up to `limit` most recent sentiment records, ordered by timestamp desc.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, text, sentiment, timestamp FROM sentiments "
            "ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows: Iterable[Tuple[int, str, str, str]] = cursor.fetchall()
    return list(rows)



