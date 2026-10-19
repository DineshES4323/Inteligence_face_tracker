"""
Tests for configuration loading and validation.

Tests cover: valid config, missing fields, invalid types, out-of-range
values, and missing config file.
"""

import json
import os
import tempfile

import pytest

from app.config import AppConfig, load_config


def _write_config(tmp_dir: str, config_data: dict) -> str:
    """Write a config dict to a temp JSON file and return the path."""
    config_path = os.path.join(tmp_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f)
    return config_path


def _valid_config() -> dict:
    """Return a valid config dictionary."""
    return {
        "video_source": "data/sample.mp4",
        "camera_mode": "video",
        "frame_skip": 3,
        "detection_confidence": 0.5,
        "recognition_threshold": 0.45,
        "max_missing_frames": 30,
        "database_path": "data/visitors.db",
        "log_directory": "logs",
        "output_directory": "output",
        "display_window": True,
    }


class TestValidConfig:
    """Test loading a valid configuration."""

    def test_loads_valid_config(self):
        """Should successfully load a valid config file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, _valid_config())
            config = load_config(path)
            assert isinstance(config, AppConfig)

    def test_all_fields_populated(self):
        """All fields should have the correct values."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, _valid_config())
            config = load_config(path)
            assert config.video_source == "data/sample.mp4"
            assert config.camera_mode == "video"
            assert config.frame_skip == 3
            assert config.detection_confidence == 0.5
            assert config.recognition_threshold == 0.45
            assert config.max_missing_frames == 30
            assert config.database_path == "data/visitors.db"
            assert config.log_directory == "logs"
            assert config.output_directory == "output"
            assert config.display_window is True

    def test_rtsp_mode_valid(self):
        """RTSP mode should be accepted."""
        cfg = _valid_config()
        cfg["camera_mode"] = "rtsp"
        cfg["video_source"] = "rtsp://example.com/stream"
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, cfg)
            config = load_config(path)
            assert config.camera_mode == "rtsp"


class TestMissingFields:
    """Test handling of missing config fields."""

    def test_missing_video_source(self):
        """Should raise ValueError for missing video_source."""
        cfg = _valid_config()
        del cfg["video_source"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, cfg)
            with pytest.raises(ValueError, match="Missing required field"):
                load_config(path)

    def test_missing_multiple_fields(self):
        """Should report all missing fields."""
        cfg = {"video_source": "test.mp4"}
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, cfg)
            with pytest.raises(ValueError, match="Missing required field"):
                load_config(path)


class TestInvalidTypes:
    """Test handling of wrong types in config."""

    def test_frame_skip_string(self):
        """Should reject string value for frame_skip."""
        cfg = _valid_config()
        cfg["frame_skip"] = "three"
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, cfg)
            with pytest.raises(ValueError):
                load_config(path)

    def test_display_window_string(self):
        """Should reject string value for display_window."""
        cfg = _valid_config()
        cfg["display_window"] = "yes"
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, cfg)
            with pytest.raises(ValueError):
                load_config(path)


class TestInvalidValues:
    """Test handling of out-of-range values."""

    def test_negative_frame_skip(self):
        """Should reject negative frame_skip."""
        cfg = _valid_config()
        cfg["frame_skip"] = -1
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, cfg)
            with pytest.raises(ValueError, match="frame_skip"):
                load_config(path)

    def test_zero_detection_confidence(self):
        """Should reject zero detection confidence."""
        cfg = _valid_config()
        cfg["detection_confidence"] = 0
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, cfg)
            with pytest.raises(ValueError, match="detection_confidence"):
                load_config(path)

    def test_confidence_above_one(self):
        """Should reject detection confidence above 1."""
        cfg = _valid_config()
        cfg["detection_confidence"] = 1.5
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, cfg)
            with pytest.raises(ValueError, match="detection_confidence"):
                load_config(path)

    def test_zero_max_missing_frames(self):
        """Should reject zero max_missing_frames."""
        cfg = _valid_config()
        cfg["max_missing_frames"] = 0
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, cfg)
            with pytest.raises(ValueError, match="max_missing_frames"):
                load_config(path)

    def test_invalid_camera_mode(self):
        """Should reject invalid camera modes."""
        cfg = _valid_config()
        cfg["camera_mode"] = "webcam"
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, cfg)
            with pytest.raises(ValueError, match="camera_mode"):
                load_config(path)

    def test_empty_video_source(self):
        """Should reject empty video source."""
        cfg = _valid_config()
        cfg["video_source"] = ""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_config(tmp_dir, cfg)
            with pytest.raises(ValueError, match="video_source"):
                load_config(path)


class TestMissingFile:
    """Test handling of missing config file."""

    def test_nonexistent_file(self):
        """Should raise FileNotFoundError for missing config file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.json")

    def test_invalid_json(self):
        """Should raise JSONDecodeError for invalid JSON."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "bad.json")
            with open(path, "w") as f:
                f.write("not valid json {{{")
            with pytest.raises(json.JSONDecodeError):
                load_config(path)


class TestConfigValidation:
    """Test the AppConfig.validate() method directly."""

    def test_valid_config_has_no_errors(self):
        """Valid config should return empty errors list."""
        config = AppConfig()
        errors = config.validate()
        assert len(errors) == 0

    def test_edge_case_frame_skip_zero(self):
        """frame_skip=0 should be valid (process every frame)."""
        config = AppConfig(frame_skip=0)
        errors = config.validate()
        assert len(errors) == 0
