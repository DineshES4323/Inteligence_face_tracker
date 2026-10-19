"""
Tests for event idempotency and deduplication.

Verifies that duplicate events are prevented via idempotency keys,
and that the system handles repeated event insertion gracefully.
"""

import os
import tempfile

import numpy as np
import pytest

from app.database import Database
from app.event_logger import generate_idempotency_key


@pytest.fixture
def db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        database = Database(db_path)
        database.connect()
        yield database
        database.close()


@pytest.fixture
def visitor_id(db: Database) -> str:
    """Create a test visitor and return the ID."""
    embedding = np.random.randn(512).astype(np.float32)
    return db.insert_visitor(embedding)


class TestIdempotencyKeyGeneration:
    """Test idempotency key generation."""

    def test_generates_consistent_key(self):
        """Same inputs should produce the same idempotency key."""
        key1 = generate_idempotency_key("entry", 1, "V-TEST0001")
        key2 = generate_idempotency_key("entry", 1, "V-TEST0001")
        assert key1 == key2

    def test_different_event_types_produce_different_keys(self):
        """Entry and exit for the same track should have different keys."""
        entry_key = generate_idempotency_key("entry", 1, "V-TEST0001")
        exit_key = generate_idempotency_key("exit", 1, "V-TEST0001")
        assert entry_key != exit_key

    def test_different_tracks_produce_different_keys(self):
        """Same event type but different tracks should differ."""
        key1 = generate_idempotency_key("entry", 1, "V-TEST0001")
        key2 = generate_idempotency_key("entry", 2, "V-TEST0001")
        assert key1 != key2


class TestEventDuplication:
    """Test that duplicate events are prevented."""

    def test_duplicate_idempotency_key_returns_none(
        self, db: Database, visitor_id: str
    ):
        """Second insert with same idempotency key should return None."""
        idem_key = "entry_track1_test"

        # First insert should succeed
        event_id1 = db.insert_event(
            visitor_id=visitor_id,
            track_id=1,
            event_type="entry",
            idempotency_key=idem_key,
        )
        assert event_id1 is not None

        # Second insert with same key should be silently skipped
        event_id2 = db.insert_event(
            visitor_id=visitor_id,
            track_id=1,
            event_type="entry",
            idempotency_key=idem_key,
        )
        assert event_id2 is None

    def test_duplicate_does_not_increase_count(
        self, db: Database, visitor_id: str
    ):
        """Duplicate events should not increase the event count."""
        idem_key = "entry_track1_count_test"

        db.insert_event(visitor_id, 1, "entry", idempotency_key=idem_key)
        assert db.get_entry_count() == 1

        # Try duplicate
        db.insert_event(visitor_id, 1, "entry", idempotency_key=idem_key)
        assert db.get_entry_count() == 1  # Should still be 1

    def test_same_track_entry_and_exit_both_succeed(
        self, db: Database, visitor_id: str
    ):
        """Entry and exit for the same track should both succeed."""
        entry_key = generate_idempotency_key("entry", 1, visitor_id)
        exit_key = generate_idempotency_key("exit", 1, visitor_id)

        entry_id = db.insert_event(
            visitor_id, 1, "entry", idempotency_key=entry_key
        )
        exit_id = db.insert_event(
            visitor_id, 1, "exit", idempotency_key=exit_key
        )

        assert entry_id is not None
        assert exit_id is not None
        assert db.get_entry_count() == 1
        assert db.get_exit_count() == 1

    def test_multiple_tracks_same_visitor(
        self, db: Database, visitor_id: str
    ):
        """Different tracks for the same visitor should produce distinct events."""
        key1 = generate_idempotency_key("entry", 1, visitor_id)
        key2 = generate_idempotency_key("entry", 2, visitor_id)

        id1 = db.insert_event(visitor_id, 1, "entry", idempotency_key=key1)
        id2 = db.insert_event(visitor_id, 2, "entry", idempotency_key=key2)

        assert id1 is not None
        assert id2 is not None
        assert db.get_entry_count() == 2

    def test_rapid_duplicate_attempts(
        self, db: Database, visitor_id: str
    ):
        """Rapid repeated inserts should all be deduplicated."""
        idem_key = "rapid_test_key"

        results = []
        for _ in range(10):
            event_id = db.insert_event(
                visitor_id, 1, "entry", idempotency_key=idem_key
            )
            results.append(event_id)

        # Only the first should have succeeded
        assert results[0] is not None
        assert all(r is None for r in results[1:])
        assert db.get_entry_count() == 1
