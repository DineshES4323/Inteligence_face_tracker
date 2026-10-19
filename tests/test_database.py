"""
Tests for the SQLite database module.

Tests cover: initialization, visitor CRUD, embedding serialization/
deserialization round-trip, event insertion, and counting queries.
"""

import os
import tempfile

import numpy as np
import pytest

from app.database import Database, serialize_embedding, deserialize_embedding


@pytest.fixture
def db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        database = Database(db_path)
        database.connect()
        yield database
        database.close()


class TestEmbeddingSerialization:
    """Test embedding serialization and deserialization."""

    def test_round_trip_preserves_values(self):
        """Serialize then deserialize should return identical values."""
        original = np.random.randn(512).astype(np.float32)
        blob = serialize_embedding(original)
        recovered = deserialize_embedding(blob)
        np.testing.assert_array_almost_equal(original, recovered)

    def test_round_trip_preserves_shape(self):
        """Shape should be preserved through serialization."""
        original = np.random.randn(512).astype(np.float32)
        blob = serialize_embedding(original)
        recovered = deserialize_embedding(blob)
        assert recovered.shape == original.shape

    def test_different_embeddings_differ(self):
        """Two different embeddings should produce different blobs."""
        emb1 = np.ones(512, dtype=np.float32)
        emb2 = np.zeros(512, dtype=np.float32)
        assert serialize_embedding(emb1) != serialize_embedding(emb2)


class TestDatabaseInit:
    """Test database initialization."""

    def test_creates_database_file(self):
        """Database file should be created on connect."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "test.db")
            database = Database(db_path)
            database.connect()
            assert os.path.exists(db_path)
            database.close()

    def test_tables_exist(self, db: Database):
        """Visitors and events tables should exist after init."""
        tables = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('visitors', 'events')"
        ).fetchall()
        table_names = {row["name"] for row in tables}
        assert "visitors" in table_names
        assert "events" in table_names

    def test_foreign_keys_enabled(self, db: Database):
        """Foreign keys pragma should be enabled."""
        row = db.conn.execute("PRAGMA foreign_keys").fetchone()
        assert row[0] == 1


class TestVisitorOperations:
    """Test visitor CRUD operations."""

    def test_insert_visitor_returns_id(self, db: Database):
        """Inserting a visitor should return a string ID starting with V-."""
        embedding = np.random.randn(512).astype(np.float32)
        visitor_id = db.insert_visitor(embedding)
        assert visitor_id.startswith("V-")

    def test_get_visitor_returns_data(self, db: Database):
        """Getting a visitor should return all fields."""
        embedding = np.random.randn(512).astype(np.float32)
        visitor_id = db.insert_visitor(embedding)

        visitor = db.get_visitor(visitor_id)
        assert visitor is not None
        assert visitor["id"] == visitor_id
        assert visitor["visit_count"] == 1
        np.testing.assert_array_almost_equal(visitor["embedding"], embedding)

    def test_get_nonexistent_visitor_returns_none(self, db: Database):
        """Getting a non-existent visitor should return None."""
        assert db.get_visitor("V-NONEXISTENT") is None

    def test_get_all_visitors(self, db: Database):
        """Should retrieve all inserted visitors."""
        emb1 = np.random.randn(512).astype(np.float32)
        emb2 = np.random.randn(512).astype(np.float32)
        db.insert_visitor(emb1)
        db.insert_visitor(emb2)

        visitors = db.get_all_visitors()
        assert len(visitors) == 2

    def test_update_last_seen(self, db: Database):
        """Updating last_seen should change timestamp and increment count."""
        embedding = np.random.randn(512).astype(np.float32)
        visitor_id = db.insert_visitor(embedding, timestamp="2024-01-01T00:00:00Z")

        db.update_visitor_last_seen(visitor_id, timestamp="2024-01-02T00:00:00Z")

        visitor = db.get_visitor(visitor_id)
        assert visitor["last_seen"] == "2024-01-02T00:00:00Z"
        assert visitor["visit_count"] == 2

    def test_unique_visitor_count(self, db: Database):
        """Unique visitor count should reflect actual number of visitors."""
        assert db.get_unique_visitor_count() == 0

        emb1 = np.random.randn(512).astype(np.float32)
        db.insert_visitor(emb1)
        assert db.get_unique_visitor_count() == 1

        emb2 = np.random.randn(512).astype(np.float32)
        db.insert_visitor(emb2)
        assert db.get_unique_visitor_count() == 2


class TestEventOperations:
    """Test event insertion and retrieval."""

    def _create_visitor(self, db: Database) -> str:
        embedding = np.random.randn(512).astype(np.float32)
        return db.insert_visitor(embedding)

    def test_insert_entry_event(self, db: Database):
        """Should successfully insert an entry event."""
        visitor_id = self._create_visitor(db)
        event_id = db.insert_event(
            visitor_id=visitor_id,
            track_id=1,
            event_type="entry",
            idempotency_key="entry_track1_v1",
        )
        assert event_id is not None
        assert event_id.startswith("E-")

    def test_insert_exit_event(self, db: Database):
        """Should successfully insert an exit event."""
        visitor_id = self._create_visitor(db)
        event_id = db.insert_event(
            visitor_id=visitor_id,
            track_id=1,
            event_type="exit",
            idempotency_key="exit_track1_v1",
        )
        assert event_id is not None

    def test_invalid_event_type_raises(self, db: Database):
        """Should raise ValueError for invalid event types."""
        visitor_id = self._create_visitor(db)
        with pytest.raises(ValueError, match="event_type must be"):
            db.insert_event(
                visitor_id=visitor_id,
                track_id=1,
                event_type="invalid",
                idempotency_key="test",
            )

    def test_entry_count(self, db: Database):
        """Entry count should match inserted entry events."""
        visitor_id = self._create_visitor(db)
        assert db.get_entry_count() == 0

        db.insert_event(visitor_id, 1, "entry", idempotency_key="e1")
        assert db.get_entry_count() == 1

        db.insert_event(visitor_id, 2, "entry", idempotency_key="e2")
        assert db.get_entry_count() == 2

    def test_exit_count(self, db: Database):
        """Exit count should match inserted exit events."""
        visitor_id = self._create_visitor(db)
        db.insert_event(visitor_id, 1, "exit", idempotency_key="x1")
        assert db.get_exit_count() == 1

    def test_get_events_filtered(self, db: Database):
        """Should filter events by type and visitor."""
        v1 = self._create_visitor(db)
        v2 = self._create_visitor(db)

        db.insert_event(v1, 1, "entry", idempotency_key="e1")
        db.insert_event(v1, 1, "exit", idempotency_key="x1")
        db.insert_event(v2, 2, "entry", idempotency_key="e2")

        entries = db.get_events(event_type="entry")
        assert len(entries) == 2

        v1_events = db.get_events(visitor_id=v1)
        assert len(v1_events) == 2

    def test_stats(self, db: Database):
        """Stats should aggregate all counts correctly."""
        v1 = self._create_visitor(db)
        v2 = self._create_visitor(db)

        db.insert_event(v1, 1, "entry", idempotency_key="e1")
        db.insert_event(v1, 1, "exit", idempotency_key="x1")
        db.insert_event(v2, 2, "entry", idempotency_key="e2")

        stats = db.get_stats()
        assert stats["unique_visitors"] == 2
        assert stats["total_entries"] == 2
        assert stats["total_exits"] == 1
