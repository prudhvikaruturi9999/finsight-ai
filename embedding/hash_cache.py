import hashlib
import sqlite3
from pathlib import Path


class HashCache:
    """SQLite-backed SHA-256 content hash cache.

    Tracks which chunk texts have already been embedded so re-runs skip
    redundant API calls for unchanged content.
    """

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunk_hashes (
                    content_hash TEXT PRIMARY KEY,
                    chunk_id     TEXT NOT NULL,
                    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)

    def has(self, content_hash: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM chunk_hashes WHERE content_hash = ?", (content_hash,)
            ).fetchone()
            return row is not None

    def add(self, content_hash: str, chunk_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO chunk_hashes (content_hash, chunk_id) VALUES (?, ?)",
                (content_hash, chunk_id),
            )

    def size(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM chunk_hashes").fetchone()[0]

    @staticmethod
    def compute(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
