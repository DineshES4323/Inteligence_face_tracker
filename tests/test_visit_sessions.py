import os
import tempfile
import time
from types import SimpleNamespace

import numpy as np

from app.database import Database
from app.pipeline import Pipeline


def _pipeline(database: Database) -> Pipeline:
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.database = database
    pipeline.recognizer = None
    pipeline.image_saver = None
    pipeline._active_visitor_tracks = {}
    pipeline._visit_tokens = {}
    pipeline._recently_exited = {}
    pipeline._suppress_recent_exit = set()
    pipeline._visit_grace_seconds = 5.0
    return pipeline


def _track(track_id: int, visitor_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        track_id=track_id,
        visitor_id=visitor_id,
        face_crop=None,
        embedding=None,
        entry_event_logged=False,
    )


def test_track_id_change_does_not_create_duplicate_visit() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        database = Database(os.path.join(tmp_dir, "events.db"))
        database.connect()
        visitor_id = database.insert_visitor(np.ones(512, dtype=np.float32))
        pipeline = _pipeline(database)

        first = _track(1, visitor_id)
        replacement = _track(2, visitor_id)
        pipeline._handle_entry(first, None)
        pipeline._handle_entry(replacement, None)

        assert database.get_entry_count() == 1

        pipeline._handle_exit(first)
        pipeline._handle_exit(replacement)
        assert database.get_exit_count() == 1
        database.close()


def test_legitimate_visit_after_grace_period_creates_new_entry() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        database = Database(os.path.join(tmp_dir, "events.db"))
        database.connect()
        visitor_id = database.insert_visitor(np.ones(512, dtype=np.float32))
        pipeline = _pipeline(database)

        first = _track(1, visitor_id)
        pipeline._handle_entry(first, None)
        pipeline._handle_exit(first)
        pipeline._recently_exited[visitor_id] = time.monotonic() - 6

        second = _track(2, visitor_id)
        pipeline._handle_entry(second, None)

        assert database.get_entry_count() == 2
        database.close()
