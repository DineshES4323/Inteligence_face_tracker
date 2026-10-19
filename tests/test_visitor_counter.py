"""
Tests for visitor counting logic.

Verifies that unique visitor counts are accurate, re-identification
does not inflate counts, and metrics are derived from the database.
"""

import os
import tempfile

import numpy as np
import pytest

from app.database import Database
from app.visitor_counter import VisitorCounter, VisitorMetrics


@pytest.fixture
def db():
    """Create a temporary database."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        database = Database(db_path)
        database.connect()
        yield database
        database.close()


@pytest.fixture
def counter(db: Database) -> VisitorCounter:
    """Create a VisitorCounter with the test database."""
    return VisitorCounter(db)


class TestEmptyDatabase:
    """Test metrics with no data."""

    def test_initial_counts_are_zero(self, counter: VisitorCounter):
        """All counts should be zero with no data."""
        metrics = counter.get_metrics()
        assert metrics.unique_visitors == 0
        assert metrics.total_entries == 0
        assert metrics.total_exits == 0
        assert metrics.active_tracks == 0
        assert metrics.recognized_visible == 0


class TestUniqueVisitorCounting:
    """Test unique visitor counting accuracy."""

    def test_single_visitor_counted_once(
        self, db: Database, counter: VisitorCounter
    ):
        """One registered visitor should give count of 1."""
        emb = np.random.randn(512).astype(np.float32)
        db.insert_visitor(emb)

        metrics = counter.get_metrics()
        assert metrics.unique_visitors == 1

    def test_multiple_visitors(
        self, db: Database, counter: VisitorCounter
    ):
        """Multiple unique visitors should each be counted."""
        for _ in range(5):
            emb = np.random.randn(512).astype(np.float32)
            db.insert_visitor(emb)

        metrics = counter.get_metrics()
        assert metrics.unique_visitors == 5

    def test_reidentification_does_not_increase_count(
        self, db: Database, counter: VisitorCounter
    ):
        """
        Updating a visitor's last_seen (re-identification) should
        NOT increase the unique visitor count.
        """
        emb = np.random.randn(512).astype(np.float32)
        visitor_id = db.insert_visitor(emb)

        assert counter.get_metrics().unique_visitors == 1

        # Simulate re-identification (update last_seen multiple times)
        for _ in range(10):
            db.update_visitor_last_seen(visitor_id)

        # Unique count should still be 1
        assert counter.get_metrics().unique_visitors == 1

    def test_visit_count_increments_on_reidentification(
        self, db: Database, counter: VisitorCounter
    ):
        """Visit count should increase but unique count should not."""
        emb = np.random.randn(512).astype(np.float32)
        visitor_id = db.insert_visitor(emb)

        db.update_visitor_last_seen(visitor_id)
        db.update_visitor_last_seen(visitor_id)

        visitor = db.get_visitor(visitor_id)
        assert visitor["visit_count"] == 3  # 1 initial + 2 updates
        assert counter.get_metrics().unique_visitors == 1


class TestEventCounting:
    """Test entry and exit event counting."""

    def test_entry_count(
        self, db: Database, counter: VisitorCounter
    ):
        """Entry count should match inserted events."""
        emb = np.random.randn(512).astype(np.float32)
        vid = db.insert_visitor(emb)

        db.insert_event(vid, 1, "entry", idempotency_key="e1")
        db.insert_event(vid, 2, "entry", idempotency_key="e2")

        metrics = counter.get_metrics()
        assert metrics.total_entries == 2
        assert metrics.total_exits == 0

    def test_exit_count(
        self, db: Database, counter: VisitorCounter
    ):
        """Exit count should match inserted events."""
        emb = np.random.randn(512).astype(np.float32)
        vid = db.insert_visitor(emb)

        db.insert_event(vid, 1, "exit", idempotency_key="x1")

        metrics = counter.get_metrics()
        assert metrics.total_exits == 1

    def test_mixed_events(
        self, db: Database, counter: VisitorCounter
    ):
        """Mixed entry and exit events should be counted separately."""
        emb1 = np.random.randn(512).astype(np.float32)
        emb2 = np.random.randn(512).astype(np.float32)
        v1 = db.insert_visitor(emb1)
        v2 = db.insert_visitor(emb2)

        db.insert_event(v1, 1, "entry", idempotency_key="e1")
        db.insert_event(v1, 1, "exit", idempotency_key="x1")
        db.insert_event(v2, 2, "entry", idempotency_key="e2")
        db.insert_event(v1, 3, "entry", idempotency_key="e3")

        metrics = counter.get_metrics()
        assert metrics.unique_visitors == 2
        assert metrics.total_entries == 3
        assert metrics.total_exits == 1


class TestSummaryString:
    """Test the formatted summary string."""

    def test_summary_contains_all_metrics(
        self, db: Database, counter: VisitorCounter
    ):
        """Summary string should contain all metric values."""
        emb = np.random.randn(512).astype(np.float32)
        vid = db.insert_visitor(emb)
        db.insert_event(vid, 1, "entry", idempotency_key="e1")

        summary = counter.get_summary_string()
        assert "Unique Visitors: 1" in summary
        assert "Entries: 1" in summary
        assert "Exits: 0" in summary
