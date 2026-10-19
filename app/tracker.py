"""
Face tracking module for the Intelligent Face Tracker.

Implements a lightweight IoU-based multi-object tracker that associates
face detections across frames, maintains track states (tentative,
confirmed, lost), and triggers entry/exit events.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.detector import FaceDetection

logger = logging.getLogger(__name__)


class TrackState(Enum):
    """Track lifecycle states."""

    TENTATIVE = "tentative"  # Newly created, not yet confirmed
    CONFIRMED = "confirmed"  # Confirmed after N consecutive detections
    LOST = "lost"  # Lost, waiting for re-association


@dataclass
class Track:
    """Represents a tracked face across frames."""

    track_id: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    state: TrackState = TrackState.TENTATIVE
    visitor_id: Optional[str] = None  # Permanent visitor ID from recognition
    face_crop: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    hits: int = 1  # Number of consecutive detections
    misses: int = 0  # Number of consecutive missed frames
    age: int = 0  # Total frames since creation
    confidence: float = 0.0
    entry_event_processed: bool = False
    entry_event_logged: bool = False

    # Predicted position for frame-skip updates
    velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    _prev_bbox: Optional[tuple[int, int, int, int]] = field(
        default=None, repr=False
    )


class FaceTracker:
    """
    IoU-based multi-face tracker with Hungarian assignment.

    Features:
    - IoU-based detection-to-track association
    - Linear assignment via Hungarian algorithm
    - Track states: tentative → confirmed → lost
    - Configurable confirmation and loss thresholds
    - Velocity-based prediction for skipped frames

    Association remains heuristic and can fragment during crowded scenes,
    severe occlusion, or rapid movement.
    """

    # Number of consecutive hits to confirm a track
    CONFIRM_HITS: int = 3
    # Minimum IoU for association
    IOU_THRESHOLD: float = 0.3
    # Embeddings can recover a detection with low overlap only when it is
    # still spatially close to the predicted track position.
    EMBEDDING_THRESHOLD: float = 0.60
    MAX_CENTER_DISTANCE_DIAGONALS: float = 1.5
    INVALID_MATCH_COST: float = 1_000_000.0

    def __init__(self, max_missing_frames: int = 30) -> None:
        """
        Args:
            max_missing_frames: Frames before a lost track is removed.
        """
        self.max_missing_frames = max_missing_frames
        self._tracks: dict[int, Track] = {}
        self._next_id: int = 1
        self._removed_tracks: list[Track] = []  # Tracks removed this update

    @property
    def active_tracks(self) -> list[Track]:
        """Return all active (tentative + confirmed) tracks."""
        return [
            t for t in self._tracks.values()
            if t.state != TrackState.LOST
        ]

    @property
    def confirmed_tracks(self) -> list[Track]:
        """Return only confirmed tracks."""
        return [
            t for t in self._tracks.values()
            if t.state == TrackState.CONFIRMED
        ]

    @property
    def all_tracks(self) -> dict[int, Track]:
        """Return all tracks including lost ones."""
        return self._tracks.copy()

    def get_removed_tracks(self) -> list[Track]:
        """
        Get tracks that were removed in the last update cycle.

        Returns:
            List of recently removed tracks (for exit event processing).
        """
        removed = self._removed_tracks.copy()
        self._removed_tracks.clear()
        return removed

    def get_newly_confirmed_tracks(self) -> list[Track]:
        """Return confirmed tracks that have not processed entry yet."""
        newly_confirmed = []

        for track in self._tracks.values():
            if (
                track.state == TrackState.CONFIRMED
                and not track.entry_event_processed
                and track.hits >= self.CONFIRM_HITS
        ):
                track.entry_event_processed = True
                newly_confirmed.append(track)

        return newly_confirmed

    def update(
        self, detections: list[FaceDetection]
    ) -> dict[int, FaceDetection]:
        """
        Update tracks with new detections.

        Associates detections to existing tracks using IoU and Hungarian
        assignment, creates new tracks for unmatched detections, and
        handles track aging/removal.

        Args:
            detections: List of face detections from the current frame.

        Returns:
            Dict mapping track_id → matched detection.
        """
        self._removed_tracks.clear()
        matched_pairs: dict[int, FaceDetection] = {}

        if not detections and not self._tracks:
            return matched_pairs

        # Build IoU cost matrix
        track_ids = list(self._tracks.keys())
        tracks = [self._tracks[tid] for tid in track_ids]

        if tracks and detections:
            cost_matrix = self._build_iou_cost_matrix(tracks, detections)

            # Hungarian assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            matched_track_indices: set[int] = set()
            matched_det_indices: set[int] = set()

            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < self.INVALID_MATCH_COST:
                    tid = track_ids[row]
                    det = detections[col]

                    self._update_track(tid, det)
                    matched_pairs[tid] = det
                    matched_track_indices.add(row)
                    matched_det_indices.add(col)

            # Handle unmatched tracks (increment miss count)
            for i, tid in enumerate(track_ids):
                if i not in matched_track_indices:
                    self._miss_track(tid)

            # Handle unmatched detections (create new tracks)
            for j, det in enumerate(detections):
                if j not in matched_det_indices:
                    new_tid = self._create_track(det)
                    matched_pairs[new_tid] = det

        elif detections:
            # No existing tracks — create all new
            for det in detections:
                new_tid = self._create_track(det)
                matched_pairs[new_tid] = det

        else:
            # No detections — all tracks miss
            for tid in track_ids:
                self._miss_track(tid)

        # Remove tracks that have been lost too long
        self._prune_lost_tracks()

        # Age all tracks
        for track in self._tracks.values():
            track.age += 1

        return matched_pairs

    def predict(self) -> None:
        """
        Predict track positions for skipped frames using linear velocity.

        Called on frames where detection is not run (frame_skip).
        """
        for track in self._tracks.values():
            if track.state == TrackState.LOST:
                track.misses += 1
                continue

            vx1, vy1, vx2, vy2 = track.velocity
            x1, y1, x2, y2 = track.bbox
            track.bbox = (
                int(x1 + vx1),
                int(y1 + vy1),
                int(x2 + vx2),
                int(y2 + vy2),
            )
            track.age += 1

        # Prune after prediction too
        self._prune_lost_tracks()

    def set_visitor_id(self, track_id: int, visitor_id: str) -> None:
        """Associate a permanent visitor ID with a track."""
        if track_id in self._tracks:
            self._tracks[track_id].visitor_id = visitor_id

    
    def force_remove_all(self) -> list[Track]:
        """
        Remove all tracks at end-of-stream.

        Return confirmed or previously confirmed/lost tracks so the
        pipeline can finalize their exit events.
        """
        removed = []

        for track in self._tracks.values():
            if (
                track.state in (TrackState.CONFIRMED, TrackState.LOST)
                and track.entry_event_logged
                and track.visitor_id is not None
            ):
                removed.append(track)

        self._tracks.clear()
        self._removed_tracks.clear()

        return removed


    # --- Private methods ---

    
    def _create_track(self, detection: FaceDetection) -> int:
        """Create a new tentative track from a detection."""
        tid = self._next_id
        self._next_id += 1

        track = Track(
            track_id=tid,
            bbox=detection.bbox,
            state=TrackState.TENTATIVE,
            face_crop=detection.face_crop,
            embedding=detection.embedding,
            hits=1,
            misses=0,
            confidence=detection.confidence,
        )

        self._tracks[tid] = track
        logger.debug("Created new track: %d", tid)
        return tid


    def _update_track(self, track_id: int, detection: FaceDetection) -> None:
        """Update an existing track with a matched detection."""
        track = self._tracks[track_id]

        # Compute velocity from bbox change
        old_bbox = track.bbox
        new_bbox = detection.bbox
        track.velocity = (
            new_bbox[0] - old_bbox[0],
            new_bbox[1] - old_bbox[1],
            new_bbox[2] - old_bbox[2],
            new_bbox[3] - old_bbox[3],
        )
        track._prev_bbox = old_bbox

        track.bbox = new_bbox
        track.face_crop = detection.face_crop
        if detection.embedding is not None:
            track.embedding = detection.embedding
        track.confidence = detection.confidence
        track.hits += 1
        track.misses = 0

        # Promote to confirmed if enough hits
        if (
            track.state == TrackState.TENTATIVE
            and track.hits >= self.CONFIRM_HITS
        ):
            track.state = TrackState.CONFIRMED
            logger.info("Track %d confirmed", track_id)

        # Re-activate lost tracks
        if track.state == TrackState.LOST:
            track.state = TrackState.CONFIRMED
            logger.info("Track %d re-activated", track_id)

    def _miss_track(self, track_id: int) -> None:
        """Handle a track that wasn't matched in this frame."""
        track = self._tracks[track_id]
        track.misses += 1

        if track.state == TrackState.CONFIRMED:
            if track.misses > 5:
                track.state = TrackState.LOST
                logger.debug("Track %d lost", track_id)

    def _prune_lost_tracks(self) -> None:
        """Remove tracks that have been lost for too many frames."""
        to_remove: list[int] = []

        for tid, track in self._tracks.items():
            should_remove = False

            if track.state == TrackState.LOST:
                if track.misses >= self.max_missing_frames:
                    should_remove = True
            elif track.state == TrackState.TENTATIVE:
                # Remove tentative tracks that miss early
                if track.misses > 5:
                    should_remove = True

            if should_remove:
                to_remove.append(tid)
                if track.state in (TrackState.CONFIRMED, TrackState.LOST):
                    self._removed_tracks.append(track)
                    logger.info(
                        "Track %d removed (visitor=%s, misses=%d)",
                        tid, track.visitor_id, track.misses,
                    )

        for tid in to_remove:
            del self._tracks[tid]

    def _build_iou_cost_matrix(
        self,
        tracks: list[Track],
        detections: list[FaceDetection],
    ) -> np.ndarray:
        """
        Build gated association costs using box overlap and face embeddings.

        A face embedding can help recover the same face when its bounding box
        moves between detection frames, but it cannot override a spatial gate.
        """
        n_tracks = len(tracks)
        n_dets = len(detections)
        cost_matrix = np.full(
            (n_tracks, n_dets),
            FaceTracker.INVALID_MATCH_COST,
            dtype=np.float32,
        )

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou = _compute_iou(track.bbox, det.bbox)
                center_distance = _center_distance(track.bbox, det.bbox)
                max_diagonal = max(
                    _box_diagonal(track.bbox),
                    _box_diagonal(det.bbox),
                )
                spatially_close = (
                    center_distance
                    <= self.MAX_CENTER_DISTANCE_DIAGONALS * max_diagonal
                )

                similarity = _embedding_similarity(
                    track.embedding,
                    det.embedding,
                )

                if iou >= self.IOU_THRESHOLD:
                    score = iou
                    if similarity is not None:
                        score = 0.70 * iou + 0.30 * similarity
                elif (
                    spatially_close
                    and similarity is not None
                    and similarity >= self.EMBEDDING_THRESHOLD
                ):
                    score = 0.35 * iou + 0.65 * similarity
                else:
                    continue

                cost_matrix[i, j] = 1.0 - score

        return cost_matrix


def _embedding_similarity(
    embedding_a: Optional[np.ndarray],
    embedding_b: Optional[np.ndarray],
) -> Optional[float]:
    """Return cosine similarity for compatible, finite, non-zero vectors."""
    if embedding_a is None or embedding_b is None:
        return None

    a = np.asarray(embedding_a, dtype=np.float32).reshape(-1)
    b = np.asarray(embedding_b, dtype=np.float32).reshape(-1)

    if (
        a.shape != b.shape
        or a.size == 0
        or not np.all(np.isfinite(a))
        or not np.all(np.isfinite(b))
    ):
        return None

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        return None

    similarity = float(np.dot(a, b) / (norm_a * norm_b))
    return similarity if np.isfinite(similarity) else None


def _center_distance(
    box1: tuple[int, int, int, int],
    box2: tuple[int, int, int, int],
) -> float:
    """Return Euclidean distance between the centers of two boxes."""
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    return float(np.hypot(center1[0] - center2[0], center1[1] - center2[1]))


def _box_diagonal(box: tuple[int, int, int, int]) -> float:
    """Return the diagonal length of a bounding box."""
    width = max(0, box[2] - box[0])
    height = max(0, box[3] - box[1])
    return float(np.hypot(width, height))


def _compute_iou(
    box1: tuple[int, int, int, int],
    box2: tuple[int, int, int, int],
) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: (x1, y1, x2, y2) of first box.
        box2: (x1, y1, x2, y2) of second box.

    Returns:
        IoU value in [0, 1].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union
