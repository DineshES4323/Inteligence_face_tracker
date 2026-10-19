"""
Visitor counting module for the Intelligent Face Tracker.

Provides real-time and historical visitor metrics by querying the
persistent database. All counts are derived from database records
rather than volatile in-memory counters.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from app.database import Database
from app.tracker import FaceTracker

logger = logging.getLogger(__name__)


@dataclass
class VisitorMetrics:
    """Current visitor counting metrics."""

    unique_visitors: int
    total_entries: int
    total_exits: int
    active_tracks: int
    recognized_visible: int  # Currently visible recognized visitors


class VisitorCounter:
    """
    Visitor counting and metrics aggregation.

    All persistent counts come from the SQLite database.
    Active track counts come from the current tracker state.
    """

    def __init__(self, database: Database) -> None:
        """
        Args:
            database: Database instance for querying persistent counts.
        """
        self.database = database

    def get_metrics(
        self,
        tracker: Optional[FaceTracker] = None,
    ) -> VisitorMetrics:
        """
        Get current visitor metrics.

        Args:
            tracker: Optional tracker for real-time track counts.

        Returns:
            VisitorMetrics with all current counts.
        """
        stats = self.database.get_stats()

        active_tracks = 0
        recognized_visible = 0

        if tracker:
            confirmed = tracker.confirmed_tracks
            active_tracks = len(confirmed)
            recognized_visible = sum(
                1 for t in confirmed if t.visitor_id is not None
            )

        return VisitorMetrics(
            unique_visitors=stats["unique_visitors"],
            total_entries=stats["total_entries"],
            total_exits=stats["total_exits"],
            active_tracks=active_tracks,
            recognized_visible=recognized_visible,
        )

    def get_summary_string(
        self, tracker: Optional[FaceTracker] = None
    ) -> str:
        """
        Get a formatted summary string for display overlay.

        Args:
            tracker: Optional tracker for real-time counts.

        Returns:
            Multi-line summary string.
        """
        metrics = self.get_metrics(tracker)
        return (
            f"Unique Visitors: {metrics.unique_visitors} | "
            f"Entries: {metrics.total_entries} | "
            f"Exits: {metrics.total_exits} | "
            f"Active: {metrics.active_tracks} | "
            f"Recognized: {metrics.recognized_visible}"
        )

    def print_report(
        self, tracker: Optional[FaceTracker] = None
    ) -> None:
        """Print a formatted metrics report to the logger."""
        metrics = self.get_metrics(tracker)
        logger.info("=" * 60)
        logger.info("VISITOR METRICS REPORT")
        logger.info("-" * 60)
        logger.info("  Total Unique Visitors : %d", metrics.unique_visitors)
        logger.info("  Total Entry Events    : %d", metrics.total_entries)
        logger.info("  Total Exit Events     : %d", metrics.total_exits)
        logger.info("  Active Tracks         : %d", metrics.active_tracks)
        logger.info("  Recognized Visible    : %d", metrics.recognized_visible)
        logger.info("=" * 60)
