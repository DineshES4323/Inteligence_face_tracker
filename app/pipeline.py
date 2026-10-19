"""
Pipeline orchestrator for the Intelligent Face Tracker.

Coordinates video capture, face detection, recognition, tracking,
event logging, and visitor counting into a cohesive processing loop.
"""

import logging
import os
import signal
import time
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

from app.config import AppConfig
from app.database import Database
from app.detector import YOLOFaceDetector, FaceDetection
from app.event_logger import EventImageSaver, generate_idempotency_key, setup_logging
from app.recognizer import FaceRecognizer
from app.tracker import FaceTracker, TrackState
from app.video_source import VideoSource
from app.visitor_counter import VisitorCounter

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Main processing pipeline for the Intelligent Face Tracker.

    Orchestrates the full flow:
    video → detect → recognize → track → log events → count visitors
    """

    def __init__(self, config: AppConfig) -> None:
        """
        Args:
            config: Validated application configuration.
        """
        self.config = config
        self._should_stop = False

        # Initialize components (constructed here, initialized in .start())
        self.database = Database(config.database_path)
        self.video_source = VideoSource(config.video_source, config.camera_mode)
        self.detector = YOLOFaceDetector(
            confidence_threshold=config.detection_confidence,
            device="cpu",
        )
        self.recognizer = FaceRecognizer(
            database=self.database,
            recognition_threshold=config.recognition_threshold,
        )
        self.tracker = FaceTracker(max_missing_frames=config.max_missing_frames)
        self.image_saver = EventImageSaver(config.log_directory)
        self.visitor_counter = VisitorCounter(self.database)

        # Frame processing state
        self._frame_number: int = 0
        self._fps_start_time: float = 0.0
        self._fps_frame_count: int = 0
        self._current_fps: float = 0.0

        # Visitor-level visit state prevents track fragmentation from
        # creating another entry or exit for the same visit.
        self._active_visitor_tracks: dict[str, set[int]] = {}
        self._visit_tokens: dict[str, str] = {}
        self._recently_exited: dict[str, float] = {}
        self._suppress_recent_exit: set[str] = set()
        self._visit_grace_seconds = 5.0

    def start(self) -> None:
        """
        Initialize all components and start the processing loop.

        Sets up signal handlers for graceful shutdown via Ctrl+C.
        """
        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)

        logger.info("=" * 60)
        logger.info("INTELLIGENT FACE TRACKER - Starting Pipeline")
        logger.info("=" * 60)

        try:
            self._initialize_components()
            self._run_loop()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error("Pipeline error: %s", str(e), exc_info=True)
        finally:
            self._shutdown()

    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.config.database_path), exist_ok=True)
        os.makedirs(self.config.log_directory, exist_ok=True)
        os.makedirs(self.config.output_directory, exist_ok=True)

        # Database
        self.database.connect()
        self._active_visitor_tracks = {
            visitor_id: set()
            for visitor_id in self.database.get_active_visitor_ids()
        }
        logger.info("Database connected")

        # Video source
        if not self.video_source.open():
            raise RuntimeError(
                f"Failed to open video source: {self.config.video_source}"
            )

        # Detector
        logger.info("Initializing face detector...")
        self.detector.initialize()

        # Recognizer
        # Recognizer: reuse the detector's already-loaded model
        logger.info("Initializing face recognizer with shared model...")
        self.recognizer.initialize(face_app=self.detector.face_app)

        logger.info("All components initialized successfully")

    def _run_loop(self) -> None:
        """Main processing loop."""
        self._fps_start_time = time.time()
        self._fps_frame_count = 0

        logger.info("Starting video processing loop...")

        for frame in self.video_source.frames():
            if self._should_stop:
                break

            self._frame_number += 1
            self._process_frame(frame)

            # FPS calculation
            self._fps_frame_count += 1
            elapsed = time.time() - self._fps_start_time
            if elapsed >= 2.0:
                self._current_fps = self._fps_frame_count / elapsed
                self._fps_start_time = time.time()
                self._fps_frame_count = 0

        # End of stream — generate exit events for remaining tracks
        logger.info("Video stream ended. Processing remaining tracks...")
        self._handle_end_of_stream()

    def _process_frame(self, frame: np.ndarray) -> None:
        """
        Process a single video frame.

        On detection frames: detect → recognize → track → log events
        On skipped frames: predict track positions only

        Args:
            frame: BGR video frame.
        """
        is_detection_frame = (self._frame_number % (self.config.frame_skip + 1)) == 0

        if is_detection_frame:
            # Run full detection pipeline
            detections = self.detector.detect(frame)

            # Update tracker with detections
            matched = self.tracker.update(detections)

            # Process newly confirmed tracks (entry events)
            for track in self.tracker.get_newly_confirmed_tracks():
                self._handle_entry(track, frame)

            # Run recognition on matched detections
            for track_id, detection in matched.items():
                track = self.tracker.all_tracks.get(track_id)
                if track and track.state == TrackState.CONFIRMED and track.visitor_id is None:
                    self._recognize_face(track, detection)

            # Process removed tracks (exit events)
            for track in self.tracker.get_removed_tracks():
                self._handle_exit(track)

        else:
            # Skipped frame — predict track positions
            self.tracker.predict()

        # Display if enabled
        if self.config.display_window:
            self._display_frame(frame)

    def _recognize_face(
        self, track, detection: FaceDetection
    ) -> None:
        """Attempt to recognize a face and associate with visitor ID."""
        result = self.recognizer.recognize(
        detection.face_crop,
        embedding=detection.embedding,
        )
        if result and result.visitor_id:
            self.tracker.set_visitor_id(track.track_id, result.visitor_id)
            if not track.entry_event_logged:
                self._handle_entry(track, None)
            logger.debug(
                "Track %d associated with visitor %s (sim=%.3f, new=%s)",
                track.track_id, result.visitor_id,
                result.similarity, result.is_new,
            )

    def _handle_entry(self, track, frame: Optional[np.ndarray]) -> None:
        """Handle a newly confirmed track — generate entry event."""
        # Try to recognize first
        if track.face_crop is not None and track.visitor_id is None:
            result = self.recognizer.recognize(
            track.face_crop,
            embedding=track.embedding,
            )
            if result and result.visitor_id:
                track.visitor_id = result.visitor_id

        # Must have a visitor ID for event logging
        if track.visitor_id is None:
            logger.debug(
                "Track %d confirmed but no visitor ID yet", track.track_id
            )
            return

        if track.entry_event_logged:
            return

        visitor_id = track.visitor_id
        existing_visit = visitor_id in self._active_visitor_tracks
        active_tracks = self._active_visitor_tracks.get(visitor_id)
        now = time.monotonic()

        if active_tracks is None:
            recent_exit = self._recently_exited.get(visitor_id)
            active_tracks = set()
            self._active_visitor_tracks[visitor_id] = active_tracks

            if (
                recent_exit is not None
                and now - recent_exit <= self._visit_grace_seconds
            ):
                self._suppress_recent_exit.add(visitor_id)
            else:
                self._visit_tokens[visitor_id] = str(time.time_ns())
        elif visitor_id not in self._visit_tokens:
            self._visit_tokens[visitor_id] = f"recovered_{time.time_ns()}"

        active_tracks.add(track.track_id)
        track.entry_event_logged = True

        if existing_visit or visitor_id in self._suppress_recent_exit:
            logger.debug(
                "Suppressing fragmented entry for visitor=%s, track=%d",
                visitor_id,
                track.track_id,
            )
            return

        # Generate an idempotency key for the visitor-level visit, not the
        # temporary tracker ID.
        idem_key = generate_idempotency_key(
            "entry",
            track.track_id,
            visitor_id,
            visit_token=self._visit_tokens[visitor_id],
        )

        # Save face crop
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        image_path = None
        if track.face_crop is not None:
            event_id_temp = f"E-TEMP{track.track_id:08d}"
            image_path = self.image_saver.save_event_image(
                face_crop=track.face_crop,
                visitor_id=track.visitor_id,
                event_type="entry",
                event_id=event_id_temp,
                timestamp=ts,
            )

        # Insert event
        event_id = self.database.insert_event(
            visitor_id=visitor_id,
            track_id=track.track_id,
            event_type="entry",
            image_path=image_path,
            timestamp=ts,
            idempotency_key=idem_key,
        )

        if event_id:
            logger.info(
                "ENTRY EVENT: visitor=%s, track=%d, event=%s",
                track.visitor_id, track.track_id, event_id,
            )

    def _handle_exit(self, track) -> None:
        """Handle a removed track — generate exit event."""
        if track.visitor_id is None:
            logger.debug(
                "Track %d removed without visitor ID, no exit event",
                track.track_id,
            )
            return

        visitor_id = track.visitor_id
        active_tracks = self._active_visitor_tracks.get(visitor_id)
        if active_tracks is not None:
            active_tracks.discard(track.track_id)
            if active_tracks:
                logger.debug(
                    "Suppressing exit while visitor=%s has another active track",
                    visitor_id,
                )
                return

        if visitor_id in self._suppress_recent_exit:
            self._suppress_recent_exit.remove(visitor_id)
            self._active_visitor_tracks.pop(visitor_id, None)
            self._visit_tokens.pop(visitor_id, None)
            return

        visit_token = self._visit_tokens.pop(
            visitor_id,
            f"recovered_{time.time_ns()}",
        )
        self._active_visitor_tracks.pop(visitor_id, None)
        self._recently_exited[visitor_id] = time.monotonic()

        idem_key = generate_idempotency_key(
            "exit",
            track.track_id,
            visitor_id,
            visit_token=visit_token,
        )

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        # Save face crop (last known face)
        image_path = None
        if track.face_crop is not None:
            event_id_temp = f"E-EXIT{track.track_id:08d}"
            image_path = self.image_saver.save_event_image(
                face_crop=track.face_crop,
                visitor_id=visitor_id,
                event_type="exit",
                event_id=event_id_temp,
                timestamp=ts,
            )

        # Insert event
        event_id = self.database.insert_event(
            visitor_id=visitor_id,
            track_id=track.track_id,
            event_type="exit",
            image_path=image_path,
            timestamp=ts,
            idempotency_key=idem_key,
        )

        if event_id:
            logger.info(
                "EXIT EVENT: visitor=%s, track=%d, event=%s",
                track.visitor_id, track.track_id, event_id,
            )

    def _handle_end_of_stream(self) -> None:
        """Generate exit events for all remaining confirmed tracks."""
        remaining = self.tracker.force_remove_all()

        for track in remaining:
            self._handle_exit(track)

        logger.info("Processed %d remaining tracks at end-of-stream", len(remaining))

    def _display_frame(self, frame: np.ndarray) -> None:
        """Display the annotated frame in a window."""
        # Draw detections from active tracks
        annotated = frame.copy()

        for track in self.tracker.confirmed_tracks:
            x1, y1, x2, y2 = track.bbox
            label = f"T{track.track_id}"
            if track.visitor_id:
                label = f"{track.visitor_id}"

            # Color: green for recognized, yellow for unrecognized
            color = (0, 255, 0) if track.visitor_id else (0, 255, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label background
            text_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )[0]
            cv2.rectangle(
                annotated,
                (x1, y1 - text_size[1] - 8),
                (x1 + text_size[0] + 4, y1),
                color,
                -1,
            )
            cv2.putText(
                annotated, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
            )

        # Draw metrics overlay
        summary = self.visitor_counter.get_summary_string(self.tracker)
        cv2.putText(
            annotated, summary, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        # Draw FPS
        fps_text = f"FPS: {self._current_fps:.1f} | Frame: {self._frame_number}"
        cv2.putText(
            annotated, fps_text, (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        cv2.imshow("Intelligent Face Tracker", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # q or ESC
            logger.info("User pressed quit key")
            self._should_stop = True

    def _shutdown(self) -> None:
        """Clean shutdown of all components."""
        logger.info("Shutting down pipeline...")

        # Print final metrics
        self.visitor_counter.print_report(self.tracker)

        # Release resources
        self.video_source.release()
        cv2.destroyAllWindows()
        self.database.close()

        logger.info("Pipeline shutdown complete")

    def _signal_handler(self, signum, frame) -> None:
        """Handle interrupt signal for graceful shutdown."""
        logger.info("Received signal %d, initiating graceful shutdown...", signum)
        self._should_stop = True
        self.video_source.stop()
