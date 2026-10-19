"""
SQLite database module for the Intelligent Face Tracker.

Manages visitor records, facial embeddings, and entry/exit events with
parameterized queries, transactions, foreign keys, and idempotency keys.
"""

import sqlite3
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# --- Schema ---

_CREATE_VISITORS_TABLE = """
CREATE TABLE IF NOT EXISTS visitors (
    id TEXT PRIMARY KEY,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    embedding BLOB NOT NULL,
    visit_count INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""

_CREATE_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    visitor_id TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    event_type TEXT NOT NULL CHECK(event_type IN ('entry', 'exit')),
    timestamp TEXT NOT NULL,
    image_path TEXT,
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (visitor_id) REFERENCES visitors(id) ON DELETE CASCADE
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_events_visitor_id ON events(visitor_id);",
    "CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event_type);",
    "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_events_idempotency ON events(idempotency_key);",
    "CREATE INDEX IF NOT EXISTS idx_visitors_last_seen ON visitors(last_seen);",
]


# --- Embedding serialization ---

def serialize_embedding(embedding: np.ndarray) -> bytes:
    """Serialize a numpy embedding to bytes for SQLite BLOB storage."""
    return embedding.astype(np.float32).tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    """Deserialize a BLOB back to a numpy float32 array."""
    return np.frombuffer(blob, dtype=np.float32).copy()


# --- Database class ---

class Database:
    """SQLite database manager for visitor and event persistence."""

    def __init__(self, db_path: str) -> None:
        """
        Initialize the database connection.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Open a database connection and initialize tables."""
        logger.info("Connecting to database: %s", self.db_path)
        self._conn = sqlite3.connect(self.db_path, timeout=10)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._init_tables()
        logger.info("Database initialized successfully")

    def _init_tables(self) -> None:
        """Create tables and indexes if they don't exist."""
        assert self._conn is not None, "Database not connected"
        cursor = self._conn.cursor()
        cursor.execute(_CREATE_VISITORS_TABLE)
        cursor.execute(_CREATE_EVENTS_TABLE)
        for index_sql in _CREATE_INDEXES:
            cursor.execute(index_sql)
        self._conn.commit()
        logger.debug("Database tables and indexes verified")

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("Database connection closed")

    @property
    def conn(self) -> sqlite3.Connection:
        """Get the active connection, raising if not connected."""
        if self._conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._conn

    # --- Visitor operations ---

    def insert_visitor(
        self,
        embedding: np.ndarray,
        timestamp: Optional[str] = None,
    ) -> str:
        """
        Register a new visitor with their face embedding.

        Args:
            embedding: Face embedding vector (numpy array).
            timestamp: UTC ISO timestamp. Defaults to current time.

        Returns:
            The generated visitor ID.
        """
        visitor_id = f"V-{uuid.uuid4().hex[:8].upper()}"
        ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        blob = serialize_embedding(embedding)

        self.conn.execute(
            "INSERT INTO visitors (id, first_seen, last_seen, embedding, visit_count) "
            "VALUES (?, ?, ?, ?, 1)",
            (visitor_id, ts, ts, blob),
        )
        self.conn.commit()
        logger.info("Registered new visitor: %s", visitor_id)
        return visitor_id

    def get_visitor(self, visitor_id: str) -> Optional[dict]:
        """
        Retrieve a visitor record by ID.

        Args:
            visitor_id: The visitor's unique ID.

        Returns:
            Dict with visitor data, or None if not found.
        """
        row = self.conn.execute(
            "SELECT id, first_seen, last_seen, embedding, visit_count "
            "FROM visitors WHERE id = ?",
            (visitor_id,),
        ).fetchone()

        if row is None:
            return None

        return {
            "id": row["id"],
            "first_seen": row["first_seen"],
            "last_seen": row["last_seen"],
            "embedding": deserialize_embedding(row["embedding"]),
            "visit_count": row["visit_count"],
        }

    def get_all_visitors(self) -> list[dict]:
        """
        Retrieve all visitor records with their embeddings.

        Returns:
            List of visitor dictionaries.
        """
        rows = self.conn.execute(
            "SELECT id, first_seen, last_seen, embedding, visit_count FROM visitors"
        ).fetchall()

        return [
            {
                "id": row["id"],
                "first_seen": row["first_seen"],
                "last_seen": row["last_seen"],
                "embedding": deserialize_embedding(row["embedding"]),
                "visit_count": row["visit_count"],
            }
            for row in rows
        ]

    def update_visitor_last_seen(
        self, visitor_id: str, timestamp: Optional[str] = None
    ) -> None:
        """Update the last_seen timestamp and increment visit_count."""
        ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        self.conn.execute(
            "UPDATE visitors SET last_seen = ?, visit_count = visit_count + 1 "
            "WHERE id = ?",
            (ts, visitor_id),
        )
        self.conn.commit()

    # --- Event operations ---

    def insert_event(
        self,
        visitor_id: str,
        track_id: int,
        event_type: str,
        image_path: Optional[str] = None,
        timestamp: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Optional[str]:
        """
        Insert an entry or exit event.

        Uses an idempotency key to prevent duplicate events. If a duplicate
        key is detected, the insertion is silently skipped.

        Args:
            visitor_id: Permanent visitor ID.
            track_id: Temporary tracker ID.
            event_type: 'entry' or 'exit'.
            image_path: Path to the saved face crop image.
            timestamp: UTC ISO timestamp. Defaults to current time.
            idempotency_key: Unique key for deduplication.

        Returns:
            The event ID if inserted, None if duplicate.
        """
        if event_type not in ("entry", "exit"):
            raise ValueError(f"event_type must be 'entry' or 'exit', got '{event_type}'")

        event_id = f"E-{uuid.uuid4().hex[:12].upper()}"
        ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        idem_key = idempotency_key or f"{event_type}_{track_id}_{ts}"

        try:
            self.conn.execute(
                "INSERT INTO events (id, visitor_id, track_id, event_type, "
                "timestamp, image_path, idempotency_key) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (event_id, visitor_id, track_id, event_type, ts, image_path, idem_key),
            )
            self.conn.commit()
            logger.info(
                "Event recorded: %s | type=%s | visitor=%s | track=%d",
                event_id, event_type, visitor_id, track_id,
            )
            return event_id
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e) and "idempotency_key" in str(e):
                logger.warning(
                    "Duplicate event skipped (idempotency_key=%s)", idem_key
                )
                return None
            raise

    def get_events(
        self,
        event_type: Optional[str] = None,
        visitor_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Retrieve events with optional filtering.

        Args:
            event_type: Filter by 'entry' or 'exit'.
            visitor_id: Filter by visitor ID.
            limit: Maximum number of records.

        Returns:
            List of event dictionaries.
        """
        query = "SELECT * FROM events WHERE 1=1"
        params: list = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if visitor_id:
            query += " AND visitor_id = ?"
            params.append(visitor_id)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_active_visitor_ids(self) -> set[str]:
        """
        Return visitors whose latest recorded event is an entry.

        This derives runtime recovery state from existing events without
        changing the database schema or modifying historical records.
        """
        rows = self.conn.execute(
            """
            SELECT visitor_id
            FROM events AS latest
            WHERE latest.id = (
                SELECT previous.id
                FROM events AS previous
                WHERE previous.visitor_id = latest.visitor_id
                ORDER BY previous.timestamp DESC,
                         previous.created_at DESC,
                         previous.id DESC
                LIMIT 1
            )
            AND latest.event_type = 'entry'
            """
        ).fetchall()
        return {row["visitor_id"] for row in rows}

    # --- Counting operations ---

    def get_unique_visitor_count(self) -> int:
        """Return the total number of unique registered visitors."""
        row = self.conn.execute("SELECT COUNT(*) AS cnt FROM visitors").fetchone()
        return row["cnt"]

    def get_entry_count(self) -> int:
        """Return the total number of entry events."""
        row = self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM events WHERE event_type = 'entry'"
        ).fetchone()
        return row["cnt"]

    def get_exit_count(self) -> int:
        """Return the total number of exit events."""
        row = self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM events WHERE event_type = 'exit'"
        ).fetchone()
        return row["cnt"]

    def get_stats(self) -> dict:
        """
        Get a summary of all visitor and event statistics.

        Returns:
            Dictionary with unique_visitors, total_entries, total_exits.
        """
        return {
            "unique_visitors": self.get_unique_visitor_count(),
            "total_entries": self.get_entry_count(),
            "total_exits": self.get_exit_count(),
        }
