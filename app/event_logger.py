"""
Event logging and image storage for the Intelligent Face Tracker.

Manages structured logging output, face crop image saving for entry/exit
events, and the log directory structure.
"""

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np


logger = logging.getLogger(__name__)


def setup_logging(log_directory: str, log_level: int = logging.INFO) -> None:
    """
    Configure the application-wide logging system.

    Sets up both console and file handlers with consistent formatting.

    Args:
        log_directory: Directory for the events.log file.
        log_level: Logging level (default INFO).
    """
    os.makedirs(log_directory, exist_ok=True)
    log_file = os.path.join(log_directory, "events.log")

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove any existing handlers to avoid duplication on re-init
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logger.info("Logging initialized. Log file: %s", log_file)


class EventImageSaver:
    """
    Saves cropped face images for entry and exit events.

    Directory structure:
        logs/
            entries/YYYY-MM-DD/
            exits/YYYY-MM-DD/
    """

    def __init__(self, log_directory: str) -> None:
        """
        Args:
            log_directory: Base directory for event images.
        """
        self.log_directory = log_directory

    def save_event_image(
        self,
        face_crop: np.ndarray,
        visitor_id: str,
        event_type: str,
        event_id: str,
        timestamp: Optional[str] = None,
    ) -> Optional[str]:
        """
        Save a cropped face image for an event.

        Args:
            face_crop: The face image as a numpy array (BGR).
            visitor_id: Permanent visitor ID.
            event_type: 'entry' or 'exit'.
            event_id: Unique event ID.
            timestamp: ISO timestamp for directory organization.

        Returns:
            The path to the saved image, or None if saving failed.
        """
        try:
            ts = timestamp or datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            )

            # Parse date for directory organization
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                dt = datetime.now(timezone.utc)

            date_str = dt.strftime("%Y-%m-%d")
            time_str = dt.strftime("%H%M%S")

            # Build directory path
            if event_type == "entry":
                event_dir = os.path.join(self.log_directory, "entries", date_str)
            elif event_type == "exit":
                event_dir = os.path.join(self.log_directory, "exits", date_str)
            else:
                logger.error("Invalid event_type for image save: %s", event_type)
                return None

            os.makedirs(event_dir, exist_ok=True)

            # Build filename: visitor_id_eventtype_time_eventid.jpg
            short_event_id = event_id.replace("E-", "")[:8]
            filename = f"{visitor_id}_{event_type}_{time_str}_{short_event_id}.jpg"
            filepath = os.path.join(event_dir, filename)

            # Validate the face crop
            if face_crop is None or face_crop.size == 0:
                logger.warning(
                    "Empty face crop for event %s, skipping image save", event_id
                )
                return None

            success = cv2.imwrite(filepath, face_crop)
            if not success:
                logger.error("Failed to write image: %s", filepath)
                return None

            logger.debug("Saved event image: %s", filepath)
            return filepath

        except Exception as e:
            logger.error(
                "Error saving event image for %s: %s", event_id, str(e),
                exc_info=True,
            )
            return None


def generate_idempotency_key(
    event_type: str,
    track_id: int,
    visitor_id: str,
    visit_token: Optional[str] = None,
) -> str:
    """
    Generate a unique idempotency key for an event.

    The default key remains track-based for backwards compatibility. A
    visitor-level visit token can be supplied when deduplication must span
    multiple tracker IDs belonging to one visit.

    Args:
        event_type: 'entry' or 'exit'.
        track_id: Temporary tracker ID.
        visitor_id: Permanent visitor ID.
        visit_token: Optional token identifying a visitor visit.

    Returns:
        A unique idempotency key string.
    """
    if visit_token is not None:
        return f"{event_type}_visitor_{visitor_id}_visit_{visit_token}"
    return f"{event_type}_track{track_id}_{visitor_id}"
