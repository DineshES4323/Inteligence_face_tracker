"""
Intelligent Face Tracker - Main Entry Point

AI-driven face detection, recognition, tracking, auto-registration,
and entry/exit logging system.

Usage:
    python main.py                     # Use config.json defaults
    python main.py --config my.json    # Use custom config file
"""

import argparse
import logging
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import load_config
from app.event_logger import setup_logging
from app.pipeline import Pipeline


def main() -> int:
    """
    Main entry point for the Intelligent Face Tracker.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        description="Intelligent Face Tracker - AI-driven face detection, "
        "recognition, tracking, and visitor counting.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to configuration JSON file (default: config.json)",
    )
    args = parser.parse_args()

    try:
        # Load and validate configuration
        config = load_config(args.config)
        print(f"Configuration loaded from: {args.config}")

        # Setup logging
        setup_logging(config.log_directory)
        logger = logging.getLogger(__name__)

        logger.info("=" * 60)
        logger.info("Intelligent Face Tracker v1.0")
        logger.info("=" * 60)
        logger.info("Config: video_source=%s, mode=%s", config.video_source, config.camera_mode)
        logger.info("Config: frame_skip=%d, detection_conf=%.2f, recognition_thresh=%.2f",
                     config.frame_skip, config.detection_confidence, config.recognition_threshold)
        logger.info("Config: max_missing_frames=%d, display=%s",
                     config.max_missing_frames, config.display_window)

        # Create and run pipeline
        pipeline = Pipeline(config)
        pipeline.start()

        return 0

    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"CONFIG ERROR: {e}", file=sys.stderr)
        return 1
    except ImportError as e:
        print(f"DEPENDENCY ERROR: {e}", file=sys.stderr)
        print("Run: pip install -r requirements.txt", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        logging.exception("Fatal error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
