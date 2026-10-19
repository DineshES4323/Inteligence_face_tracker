
"""
Video source abstraction for the Intelligent Face Tracker.

Provides a unified interface for reading from local video files and
RTSP camera streams, with reconnection logic and graceful shutdown.
"""

import logging
import time
from typing import Iterator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoSource:
    """Read video files and RTSP streams."""

    MAX_RECONNECT_ATTEMPTS: int = 10
    RECONNECT_DELAY_SECONDS: float = 3.0

    # Resize large frames to reduce memory usage.
    PROCESSING_WIDTH: int = 960
    PROCESSING_HEIGHT: int = 540

    def __init__(
        self,
        source: str,
        camera_mode: str = "video",
    ) -> None:
        self.source = source
        self.camera_mode = camera_mode
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_open: bool = False
        self._should_stop: bool = False
        self._frame_count: int = 0

    @property
    def is_open(self) -> bool:
        return (
            self._is_open
            and self._cap is not None
            and self._cap.isOpened()
        )

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def fps(self) -> float:
        if self._cap and self._cap.isOpened():
            value = self._cap.get(cv2.CAP_PROP_FPS)
            return value if value > 0 else 30.0
        return 30.0

    @property
    def frame_size(self) -> tuple[int, int]:
        if self._cap and self._cap.isOpened():
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        return 0, 0

    @property
    def total_frames(self) -> int:
        if self._cap and self.camera_mode == "video":
            return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return 0

    def open(self) -> bool:
        """Open the configured video source."""
        try:
            self._should_stop = False

            if self.camera_mode == "rtsp":
                logger.info("Opening RTSP stream: %s", self.source)
                self._cap = cv2.VideoCapture(
                    self.source, cv2.CAP_FFMPEG
                )
            else:
                logger.info("Opening video file: %s", self.source)
                self._cap = cv2.VideoCapture(self.source)

            if self._cap is None or not self._cap.isOpened():
                logger.error(
                    "Failed to open video source: %s", self.source
                )
                self._is_open = False
                return False

            self._is_open = True
            width, height = self.frame_size

            logger.info(
                "Video source opened: %dx%d @ %.1f FPS "
                "(total_frames=%d, mode=%s)",
                width,
                height,
                self.fps,
                self.total_frames,
                self.camera_mode,
            )
            return True

        except Exception:
            logger.exception("Error opening video source")
            self._is_open = False
            return False

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize a frame to reduce processing memory."""
        height, width = frame.shape[:2]

        if (
            width <= self.PROCESSING_WIDTH
            and height <= self.PROCESSING_HEIGHT
        ):
            return frame

        return cv2.resize(
            frame,
            (self.PROCESSING_WIDTH, self.PROCESSING_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read one frame and resize it for processing."""
        if not self.is_open or self._should_stop:
            return False, None

        ret, frame = self._cap.read()  # type: ignore[union-attr]

        if ret and frame is not None and frame.size > 0:
            self._frame_count += 1
            return True, self._resize_frame(frame)

        if self.camera_mode == "rtsp" and not self._should_stop:
            logger.warning(
                "RTSP frame read failed. Attempting reconnection..."
            )

            if self._reconnect():
                ret, frame = self._cap.read()  # type: ignore[union-attr]

                if ret and frame is not None and frame.size > 0:
                    self._frame_count += 1
                    return True, self._resize_frame(frame)

        logger.info(
            "End of video stream (frames read: %d)",
            self._frame_count,
        )
        return False, None

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames until the source ends or is stopped."""
        while not self._should_stop:
            ret, frame = self.read()

            if not ret or frame is None:
                break

            yield frame

    def stop(self) -> None:
        """Signal the frame generator to stop."""
        self._should_stop = True

    def release(self) -> None:
        """Release the video capture resource safely."""
        self._should_stop = True

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        self._is_open = False
        logger.info(
            "Video source released (total frames: %d)",
            self._frame_count,
        )

    def _reconnect(self) -> bool:
        """Attempt to reconnect to an RTSP stream."""
        for attempt in range(1, self.MAX_RECONNECT_ATTEMPTS + 1):
            if self._should_stop:
                return False

            logger.info(
                "RTSP reconnection attempt %d/%d...",
                attempt,
                self.MAX_RECONNECT_ATTEMPTS,
            )

            if self._cap is not None:
                self._cap.release()

            time.sleep(self.RECONNECT_DELAY_SECONDS)

            self._cap = cv2.VideoCapture(
                self.source, cv2.CAP_FFMPEG
            )

            if self._cap.isOpened():
                ret, _ = self._cap.read()

                if ret:
                    self._is_open = True
                    logger.info(
                        "RTSP reconnection successful on attempt %d",
                        attempt,
                    )
                    return True

        self._is_open = False
        logger.error(
            "RTSP reconnection failed after %d attempts",
            self.MAX_RECONNECT_ATTEMPTS,
        )
        return False

    def __enter__(self) -> "VideoSource":
        if not self.open():
            raise RuntimeError(
                f"Could not open video source: {self.source}"
            )
        return self

    def __exit__(self, *args: object) -> None:
        self.release()
