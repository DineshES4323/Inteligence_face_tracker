# Testing Documentation

## Overview

The project uses **pytest** for automated testing. All tests use synthetic data (random embeddings, mock detections) and temporary databases — no real model inference is required to run the test suite.

## Test Suite Summary

| Test File | Tests | Description |
|-----------|-------|-------------|
| `tests/test_config.py` | 16 | Configuration loading and validation |
| `tests/test_database.py` | 16 | SQLite database operations |
| `tests/test_event_idempotency.py` | 8 | Event deduplication via idempotency keys |
| `tests/test_visitor_counter.py` | 13 | Visitor counting accuracy |
| **Total** | **53** | |

## Running Tests

```powershell
# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_database.py -v

# Run a specific test class
python -m pytest tests/test_database.py::TestVisitorOperations -v

# Run with output capture disabled (see print statements)
python -m pytest tests/ -v -s
```

## Test Details

### test_config.py

Tests the `app/config.py` module:

- **TestValidConfig**: Loading valid config, all fields populated, RTSP mode accepted
- **TestMissingFields**: Missing required fields raise `ValueError`
- **TestInvalidTypes**: Wrong types (string for int, etc.) raise `ValueError`
- **TestInvalidValues**: Out-of-range values (negative frame_skip, confidence > 1, etc.)
- **TestMissingFile**: Non-existent file, invalid JSON
- **TestConfigValidation**: Direct `AppConfig.validate()` tests, edge cases

### test_database.py

Tests the `app/database.py` module:

- **TestEmbeddingSerialization**: Round-trip preservation of values and shape
- **TestDatabaseInit**: File creation, table existence, foreign keys enabled
- **TestVisitorOperations**: Insert, get, get_all, update_last_seen, unique count
- **TestEventOperations**: Insert entry/exit, invalid type rejection, counts, filtering, stats

### test_event_idempotency.py

Tests duplicate event prevention:

- **TestIdempotencyKeyGeneration**: Consistent keys, different event types differ, different tracks differ
- **TestEventDuplication**: Duplicate key returns None, count stays same, entry+exit both succeed, multi-track, rapid duplicates

### test_visitor_counter.py

Tests the `app/visitor_counter.py` module:

- **TestEmptyDatabase**: All counts zero initially
- **TestUniqueVisitorCounting**: Single visitor, multiple visitors, re-identification does NOT increase count, visit_count increments separately
- **TestEventCounting**: Entry count, exit count, mixed events
- **TestSummaryString**: Formatted output contains all metrics

## Test Categories

### Unit Tests (Mocked/Synthetic)
All current tests use:
- Temporary SQLite databases (cleaned up after each test)
- Random numpy embeddings (synthetic, not from real faces)
- Direct function calls (no model inference)

### Integration / E2E Tests (Require Models)
To test with actual model inference:
```powershell
# Requires: ultralytics, insightface, onnxruntime, and a sample video
python main.py --config config.json
```
This is a manual integration test that verifies the full pipeline.

## Test Results (Last Run)

```
53 passed in ~3 seconds
Platform: Windows 11, Python 3.11.4
```
