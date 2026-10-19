
"""
InsightFace SCRFD face detector for Intelligent Face Tracker.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """Represents a single face detection result."""

    bbox: tuple[int, int, int, int]
    confidence: float
    face_crop: np.ndarray
    embedding: Optional[np.ndarray] = None


class YOLOFaceDetector:
    """Face detector using InsightFace SCRFD and face embeddings."""

    MIN_FACE_SIZE = 20

    def __init__(
        self,
        model_path: str = "yolov8n-face.pt",
        confidence_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model_path = model_path
        self.model = None
        self.face_app = None

    def initialize(self) -> None:
        """Initialize InsightFace detection and recognition models."""

        model_root = r"D:\AIModels\insightface"

        model_dir = os.path.join(
            model_root, "models", "buffalo_l"
        )

        detector_path = os.path.join(
            model_dir, "det_10g.onnx"
        )

        recognizer_path = os.path.join(
            model_dir, "w600k_r50.onnx"
        )

        if not os.path.isfile(detector_path):
            raise FileNotFoundError(
                f"Detector model not found: {detector_path}"
            )

        if not os.path.isfile(recognizer_path):
            raise FileNotFoundError(
                f"Recognition model not found: {recognizer_path}"
            )

        self.face_app = FaceAnalysis(
            name="buffalo_l",
            root=model_root,
            providers=["CPUExecutionProvider"],
        )

        self.face_app.prepare(
            ctx_id=-1,
            det_size=(320, 320),
        )

        logger.info(
            "InsightFace initialized using models in %s",
            model_dir,
        )

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        """Detect faces and return their embeddings from the full frame."""

        if self.face_app is None:
            raise RuntimeError(
                "Detector not initialized. Call initialize() first."
            )

        if frame is None or frame.size == 0:
            logger.warning("Empty frame received.")
            return []

        height, width = frame.shape[:2]
        detections: list[FaceDetection] = []

        faces = self.face_app.get(frame)

        for face in faces:
            x1, y1, x2, y2 = map(
                int, face.bbox
            )

            x1, y1, x2, y2 = self._clamp_bbox(
                x1, y1, x2, y2, width, height
            )

            confidence = float(face.det_score)

            if confidence < self.confidence_threshold:
                continue

            if (
                x2 - x1 < self.MIN_FACE_SIZE
                or y2 - y1 < self.MIN_FACE_SIZE
            ):
                continue

            face_crop = frame[y1:y2, x1:x2].copy()

            if face_crop.size == 0:
                continue

            embedding = getattr(
                face, "normed_embedding", None
            )

            if embedding is not None:
                embedding = np.asarray(
                    embedding, dtype=np.float32
                ).copy()

            detections.append(
                FaceDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    face_crop=face_crop,
                    embedding=embedding,
                )
            )

        logger.debug("Detected %d faces.", len(detections))
        return detections

    @staticmethod
    def _clamp_bbox(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        w: int,
        h: int,
    ) -> tuple[int, int, int, int]:
        return (
            max(0, min(x1, w - 1)),
            max(0, min(y1, h - 1)),
            max(0, min(x2, w)),
            max(0, min(y2, h)),
        )

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list[FaceDetection],
        track_ids: Optional[dict[int, str]] = None,
    ) -> np.ndarray:
        """Draw face bounding boxes and labels."""

        annotated = frame.copy()

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            label = f"Face {det.confidence:.2f}"

            if track_ids and i in track_ids:
                label = (
                    f"{track_ids[i]} ({det.confidence:.2f})"
                )

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            text_size = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1,
            )[0]

            text_y = max(y1, text_size[1] + 8)

            cv2.rectangle(
                annotated,
                (x1, text_y - text_size[1] - 8),
                (x1 + text_size[0] + 4, text_y),
                (0, 255, 0),
                -1,
            )

            cv2.putText(
                annotated,
                label,
                (x1 + 2, text_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        return annotated
