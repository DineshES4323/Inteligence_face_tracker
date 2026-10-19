import numpy as np

from app.detector import FaceDetection
from app.tracker import FaceTracker, TrackState


def _detection(
    bbox: tuple[int, int, int, int],
    embedding: np.ndarray | None = None,
) -> FaceDetection:
    return FaceDetection(
        bbox=bbox,
        confidence=0.9,
        face_crop=np.ones((40, 40, 3), dtype=np.uint8),
        embedding=embedding,
    )


def test_far_same_embedding_does_not_match_existing_track() -> None:
    embedding = np.ones(512, dtype=np.float32)
    tracker = FaceTracker()

    first = tracker.update([_detection((10, 10, 50, 50), embedding)])
    first_id = next(iter(first))

    second = tracker.update([_detection((700, 400, 740, 440), embedding)])

    assert first_id in tracker.all_tracks
    assert len(second) == 1
    assert next(iter(second)) != first_id


def test_one_to_one_assignment_keeps_two_people_separate() -> None:
    embedding_a = np.ones(512, dtype=np.float32)
    embedding_b = -np.ones(512, dtype=np.float32)
    tracker = FaceTracker()

    first = tracker.update(
        [
            _detection((10, 10, 50, 50), embedding_a),
            _detection((200, 10, 240, 50), embedding_b),
        ]
    )
    track_ids = set(first)

    second = tracker.update(
        [
            _detection((12, 10, 52, 50), embedding_a),
            _detection((202, 10, 242, 50), embedding_b),
        ]
    )

    assert set(second) == track_ids
    assert len(second) == 2


def test_temporary_detection_miss_preserves_track_id() -> None:
    detection = _detection((10, 10, 50, 50), np.ones(512, dtype=np.float32))
    tracker = FaceTracker()

    first = tracker.update([detection])
    track_id = next(iter(first))
    tracker.update([detection])
    tracker.update([detection])
    assert tracker.all_tracks[track_id].state == TrackState.CONFIRMED

    tracker.update([])
    tracker.update([])
    matched = tracker.update([detection])

    assert set(matched) == {track_id}
