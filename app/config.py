"""
Configuration loader and validator for the Intelligent Face Tracker.

Loads application settings from a JSON configuration file, validates types
and value ranges, and provides sensible defaults.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any


# Required fields and their expected types
_FIELD_SPECS: dict[str, type] = {
    "video_source": str,
    "camera_mode": str,
    "frame_skip": int,
    "detection_confidence": (int, float),  # type: ignore[assignment]
    "recognition_threshold": (int, float),  # type: ignore[assignment]
    "max_missing_frames": int,
    "database_path": str,
    "log_directory": str,
    "output_directory": str,
    "display_window": bool,
}

# Valid camera modes
_VALID_CAMERA_MODES = {"video", "rtsp"}


@dataclass
class AppConfig:
    """Application configuration with validated fields."""

    video_source: str = "data/sample.mp4"
    camera_mode: str = "video"
    frame_skip: int = 3
    detection_confidence: float = 0.5
    recognition_threshold: float = 0.45
    max_missing_frames: int = 30
    database_path: str = "data/visitors.db"
    log_directory: str = "logs"
    output_directory: str = "output"
    display_window: bool = True

    def validate(self) -> list[str]:
        """
        Validate all configuration values.

        Returns:
            List of validation error messages. Empty list means valid.
        """
        errors: list[str] = []

        # camera_mode must be one of the valid modes
        if self.camera_mode not in _VALID_CAMERA_MODES:
            errors.append(
                f"'camera_mode' must be one of {_VALID_CAMERA_MODES}, "
                f"got '{self.camera_mode}'"
            )

        # frame_skip must be >= 0
        if self.frame_skip < 0:
            errors.append(
                f"'frame_skip' must be >= 0, got {self.frame_skip}"
            )

        # detection_confidence must be in (0, 1]
        if not (0 < self.detection_confidence <= 1.0):
            errors.append(
                f"'detection_confidence' must be in (0, 1], "
                f"got {self.detection_confidence}"
            )

        # recognition_threshold must be in (0, 1]
        if not (0 < self.recognition_threshold <= 1.0):
            errors.append(
                f"'recognition_threshold' must be in (0, 1], "
                f"got {self.recognition_threshold}"
            )

        # max_missing_frames must be > 0
        if self.max_missing_frames <= 0:
            errors.append(
                f"'max_missing_frames' must be > 0, "
                f"got {self.max_missing_frames}"
            )

        # Paths should not be empty
        for path_field in ("video_source", "database_path", "log_directory", "output_directory"):
            value = getattr(self, path_field)
            if not value or not value.strip():
                errors.append(f"'{path_field}' must not be empty")

        return errors


def load_config(config_path: str = "config.json") -> AppConfig:
    """
    Load and validate configuration from a JSON file.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        Validated AppConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config contains invalid values.
        json.JSONDecodeError: If the config file is not valid JSON.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config: dict[str, Any] = json.load(f)

    # Check for unexpected fields (warn but don't fail)
    known_fields = set(_FIELD_SPECS.keys())
    unknown_fields = set(raw_config.keys()) - known_fields
    if unknown_fields:
        import logging
        logging.warning(f"Unknown config fields will be ignored: {unknown_fields}")

    # Type checking for known required fields
    type_errors: list[str] = []
    for field_name, expected_type in _FIELD_SPECS.items():
        if field_name not in raw_config:
            type_errors.append(f"Missing required field: '{field_name}'")
            continue

        value = raw_config[field_name]
        if not isinstance(value, expected_type):
            type_errors.append(
                f"Field '{field_name}' must be {expected_type.__name__ if isinstance(expected_type, type) else expected_type}, "
                f"got {type(value).__name__}: {value!r}"
            )

    if type_errors:
        raise ValueError(
            "Configuration type errors:\n  " + "\n  ".join(type_errors)
        )

    # Build the config object
    config = AppConfig(
        video_source=raw_config["video_source"],
        camera_mode=raw_config["camera_mode"],
        frame_skip=raw_config["frame_skip"],
        detection_confidence=float(raw_config["detection_confidence"]),
        recognition_threshold=float(raw_config["recognition_threshold"]),
        max_missing_frames=raw_config["max_missing_frames"],
        database_path=raw_config["database_path"],
        log_directory=raw_config["log_directory"],
        output_directory=raw_config["output_directory"],
        display_window=raw_config["display_window"],
    )

    # Run value validation
    validation_errors = config.validate()
    if validation_errors:
        raise ValueError(
            "Configuration validation errors:\n  " + "\n  ".join(validation_errors)
        )

    return config
