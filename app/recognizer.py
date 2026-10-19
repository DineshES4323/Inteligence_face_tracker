
"""
Face recognition module using InsightFace / ArcFace embeddings.

Generates face embeddings, matches against known visitors using cosine
similarity, and handles auto-registration of new visitors.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.spatial.distance import cosine

from app.database import Database

logger = logging.getLogger(__name__)


@dataclass
class RecognitionResult:
    """Result of face recognition attempt."""

    visitor_id: Optional[str]
    similarity: float
    is_new: bool
    embedding: np.ndarray


class FaceRecognizer:
    """Face recognition using InsightFace ArcFace embeddings."""

    MIN_FACE_SIZE: int = 30

    def __init__(
        self,
        database: Database,
        recognition_threshold: float = 0.45,
        model_name: str = "buffalo_l",
    ) -> None:
        self.database = database
        self.recognition_threshold = recognition_threshold
        self._model_name = model_name
        self._app = None
        self._embedding_cache: dict[str, np.ndarray] = {}

    
    def initialize(self, face_app=None) -> None:
        """Reuse the detector's model when supplied; otherwise load one."""

        if face_app is not None:
            self._app = face_app
            logger.info("Reusing shared InsightFace model from detector")
        else:
            try:
                from insightface.app import FaceAnalysis
            except ImportError as exc:
                raise ImportError(
                    "insightface is required. Install with: "
                    "pip install insightface onnxruntime"
                ) from exc

            logger.info(
                "Initializing standalone InsightFace model: %s",
                self._model_name,
            )

            self._app = FaceAnalysis(
                name=self._model_name,
                root="D:/AIModels/insightface",
                providers=["CPUExecutionProvider"],
            )

            self._app.prepare(
                ctx_id=-1,
                det_size=(640, 640),
            )

        logger.info("InsightFace model initialized successfully")
        self._refresh_embedding_cache()


    def _refresh_embedding_cache(self) -> None:
        """Load visitor embeddings from the database."""

        visitors = self.database.get_all_visitors()

        self._embedding_cache = {
            v["id"]: np.asarray(v["embedding"], dtype=np.float32)
            for v in visitors
            if v.get("embedding") is not None
        }

        logger.info(
            "Loaded %d visitor embeddings into cache",
            len(self._embedding_cache),
        )

    def get_embedding(
        self,
        face_image: Optional[np.ndarray],
        embedding: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Use a precomputed full-frame embedding when available.
        Otherwise, attempt to extract one from the supplied face image.
        """

        if self._app is None:
            raise RuntimeError(
                "Recognizer not initialized. Call initialize() first."
            )

        # Preferred path: use the embedding produced by the detector.
        if embedding is not None:
            vector = np.asarray(
                embedding, dtype=np.float32
            ).reshape(-1)

            if vector.size != 512:
                logger.warning(
                    "Invalid embedding size: %d",
                    vector.size,
                )
                return None

            norm = np.linalg.norm(vector)

            if not np.isfinite(norm) or norm == 0:
                return None

            return (vector / norm).astype(np.float32)

        # Fallback path: extract an embedding from a face crop.
        if face_image is None or face_image.size == 0:
            return None

        h, w = face_image.shape[:2]

        if h < self.MIN_FACE_SIZE or w < self.MIN_FACE_SIZE:
            logger.debug(
                "Face too small for embedding: %dx%d",
                w,
                h,
            )
            return None

        try:
            faces = self._app.get(face_image)

            if not faces:
                # Retry with a resized crop.
                scale = max(112 / h, 112 / w, 1.0)

                if scale > 1.0:
                    resized = cv2.resize(
                        face_image,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_LINEAR,
                    )
                    faces = self._app.get(resized)

            if not faces:
                logger.debug(
                    "InsightFace could not detect a face in the crop"
                )
                return None

            vector = np.asarray(
                faces[0].normed_embedding,
                dtype=np.float32,
            ).reshape(-1)

            if vector.size != 512:
                return None

            norm = np.linalg.norm(vector)

            if not np.isfinite(norm) or norm == 0:
                return None

            return (vector / norm).astype(np.float32)

        except Exception:
            logger.exception("Embedding extraction failed")
            return None

    def recognize(
        self,
        face_image: Optional[np.ndarray],
        auto_register: bool = True,
        embedding: Optional[np.ndarray] = None,
    ) -> Optional[RecognitionResult]:
        """
        Recognize a face using a supplied embedding or face image.

        If auto_register is True, unmatched faces are registered.
        """

        vector = self.get_embedding(
            face_image,
            embedding=embedding,
        )

        if vector is None:
            return None

        best_match_id, best_similarity = self._find_best_match(vector)

        if (
            best_match_id is not None
            and best_similarity >= self.recognition_threshold
        ):
            logger.info(
                "Recognized visitor %s (similarity: %.3f)",
                best_match_id,
                best_similarity,
            )

            self.database.update_visitor_last_seen(best_match_id)

            return RecognitionResult(
                visitor_id=best_match_id,
                similarity=best_similarity,
                is_new=False,
                embedding=vector,
            )

        if auto_register:
            visitor_id = self.database.insert_visitor(vector)
            self._embedding_cache[visitor_id] = vector

            logger.info(
                "Registered new visitor %s (best match similarity: %.3f)",
                visitor_id,
                best_similarity if best_similarity > 0 else 0.0,
            )

            return RecognitionResult(
                visitor_id=visitor_id,
                similarity=0.0,
                is_new=True,
                embedding=vector,
            )

        return RecognitionResult(
            visitor_id=None,
            similarity=best_similarity,
            is_new=False,
            embedding=vector,
        )

    def _find_best_match(
        self,
        embedding: np.ndarray,
    ) -> tuple[Optional[str], float]:
        """Find the best visitor match using cosine similarity."""

        if not self._embedding_cache:
            return None, 0.0

        best_id: Optional[str] = None
        best_similarity = 0.0

        for visitor_id, stored_embedding in self._embedding_cache.items():
            try:
                stored = np.asarray(
                    stored_embedding, dtype=np.float32
                ).reshape(-1)

                if stored.shape != embedding.shape:
                    continue

                if not np.all(np.isfinite(stored)):
                    continue

                similarity = 1.0 - cosine(embedding, stored)

                if not np.isfinite(similarity):
                    continue

                if similarity > best_similarity:
                    best_similarity = float(similarity)
                    best_id = visitor_id

            except (ValueError, TypeError, FloatingPointError):
                logger.debug(
                    "Skipping invalid embedding for visitor %s",
                    visitor_id,
                )

        return best_id, best_similarity

    def match_embedding(
        self,
        embedding: np.ndarray,
    ) -> tuple[Optional[str], float]:
        """Return a visitor ID only when the threshold is met."""

        vector = np.asarray(
            embedding, dtype=np.float32
        ).reshape(-1)

        if vector.size != 512:
            return None, 0.0

        norm = np.linalg.norm(vector)

        if not np.isfinite(norm) or norm == 0:
            return None, 0.0

        vector = vector / norm
        best_id, best_similarity = self._find_best_match(vector)

        if (
            best_id is not None
            and best_similarity >= self.recognition_threshold
        ):
            return best_id, best_similarity

        return None, best_similarity
