# AI Planning Document

## Problem Statement

Build an AI-driven face detection, recognition, tracking, automatic registration, and entry/exit logging system that processes local video files and live RTSP camera streams. The system must accurately detect faces, generate unique embeddings, track individuals across frames, register new visitors, re-identify returning ones, and log entry/exit events — all without duplicates.

## Requirements and Acceptance Criteria

1. **Face Detection**: YOLO-based detector with configurable confidence threshold
2. **Face Recognition**: InsightFace/ArcFace 512-d embeddings with cosine similarity matching
3. **Tracking**: IoU-based multi-object tracking with Hungarian assignment
4. **Auto-Registration**: New visitor ID created when face doesn't match existing visitors
5. **Re-Identification**: Returning visitors matched via embedding similarity
6. **Entry/Exit Events**: Exactly one entry per confirmed track, one exit when lost
7. **No Duplicates**: Idempotency keys prevent duplicate events
8. **Visitor Counting**: Unique visitors from DB records, not volatile counters
9. **Image Logging**: Cropped face saved for every entry/exit event
10. **RTSP Support**: Live camera streams with reconnection logic

## Chosen Models and Rationale

### Face Detection: YOLO (Ultralytics)
- **Model**: `yolov8n-face.pt` (face-specific) with fallback to `yolov8n.pt` (person detection + face region estimation)
- **Rationale**: Ultralytics provides a well-maintained, production-ready YOLO implementation with simple Python API. The nano variant (~6MB) offers good speed-accuracy tradeoff for CPU inference.
- **Fallback**: If face-specific weights are unavailable, the generic model detects persons (class 0) and estimates the face region from the upper 25% of the body bounding box.

### Face Recognition: InsightFace (ArcFace)
- **Model Pack**: `buffalo_l` with ArcFace recognition model
- **Embedding**: 512-dimensional L2-normalized vector
- **Rationale**: InsightFace is the most widely-used open-source face recognition library with state-of-the-art accuracy. ArcFace loss produces highly discriminative embeddings. The `buffalo_l` pack includes both detection and recognition models.

### Tracking: Custom IoU-based Tracker
- **Algorithm**: IoU-based detection-to-track association with Hungarian (linear sum) assignment
- **Rationale**: ByteTrack requires C++ compilation dependencies that are problematic on Windows. Our custom tracker provides equivalent core functionality (IoU matching, track states, velocity prediction) without the build complexity.

## Detection, Recognition, and Tracking Workflow

```
Video Frame
    │
    ├── [Detection Frame] ──► YOLO Face Detection
    │                              │
    │                              ▼
    │                         Bounding Boxes + Confidence
    │                              │
    │                              ▼
    │                         IoU Tracker Update
    │                              │
    │                    ┌─────────┼─────────┐
    │                    ▼         ▼         ▼
    │              New Tracks  Matched   Lost Tracks
    │                    │     Tracks         │
    │                    ▼         │         ▼
    │             (Tentative)      │    Exit Event
    │                    │         ▼
    │                    ▼    InsightFace
    │             After 3 hits: Recognition
    │             Confirmed        │
    │                    │         ▼
    │                    ▼    Match/Register
    │              Entry Event     │
    │                              ▼
    │                         Set Visitor ID
    │
    └── [Skipped Frame] ──► Velocity Prediction Only
```

## Database Design

### `visitors` Table
| Column | Type | Description |
|--------|------|-------------|
| id | TEXT PK | `V-XXXXXXXX` format |
| first_seen | TEXT | UTC ISO timestamp |
| last_seen | TEXT | UTC ISO timestamp |
| embedding | BLOB | 512×float32 = 2048 bytes |
| visit_count | INTEGER | Incremented on re-identification |
| created_at | TEXT | Record creation timestamp |

### `events` Table
| Column | Type | Description |
|--------|------|-------------|
| id | TEXT PK | `E-XXXXXXXXXXXX` format |
| visitor_id | TEXT FK | References visitors(id) |
| track_id | INTEGER | Temporary tracker ID |
| event_type | TEXT | 'entry' or 'exit' |
| timestamp | TEXT | UTC ISO timestamp |
| image_path | TEXT | Path to saved face crop |
| idempotency_key | TEXT UNIQUE | Deduplication key |
| created_at | TEXT | Record creation timestamp |

## Entry/Exit Event Semantics

- **Entry Event**: Generated exactly once when a track transitions from `tentative` to `confirmed` (after 3 consecutive detections). The track must have an associated visitor ID.
- **Exit Event**: Generated exactly once when a confirmed track's `misses` exceed `max_missing_frames` and is removed.
- **Idempotency Key Format**: `{event_type}_track{track_id}_{visitor_id}` — database UNIQUE constraint prevents duplicates.
- **Limitation**: With a single camera, "entry" and "exit" represent face appearance and disappearance in the camera view, not physical entry/exit through a doorway. True entry/exit detection would require a two-camera setup or a line-crossing algorithm.

## Re-Identification Approach

1. When a face is detected and tracked, its face crop is passed to InsightFace
2. A 512-d embedding is extracted and L2-normalized
3. Cosine similarity is computed against all stored visitor embeddings
4. If `similarity >= recognition_threshold` (default 0.45): existing visitor is matched
5. If no match: a new visitor is registered with the embedding stored in SQLite
6. The visitor ID is associated with the temporary track ID
7. If a returning visitor gets a new track ID, face recognition re-associates them with their existing permanent visitor ID

## Frame-Skipping Strategy

- `frame_skip = N` means N frames are skipped between detection cycles
- Example: `frame_skip = 3` → detect on frames 0, 4, 8, 12, ...
- On skipped frames: tracker uses linear velocity prediction to update positions
- This reduces CPU load by ~75% with `frame_skip=3`
- Trade-off: fast-moving faces may not be tracked well with high frame_skip values

## Failure Handling

| Failure | Handling |
|---------|----------|
| Invalid video path | Clear error message at startup |
| RTSP disconnect | Auto-reconnect with 10 attempts, 3s delay |
| Empty frame | Skip and continue |
| Model load failure | Fallback model or clear error |
| Embedding extraction failure | Skip face, continue processing |
| Database write failure | Log error, continue (consistency note in docs) |
| Image write failure | Log error, event still recorded in DB |

## Testing Plan

| Test File | Tests | Description |
|-----------|-------|-------------|
| `test_config.py` | 16 | Config loading, validation, missing/invalid fields |
| `test_database.py` | 16 | DB init, visitor CRUD, embedding roundtrip, events |
| `test_event_idempotency.py` | 8 | Duplicate prevention via idempotency keys |
| `test_visitor_counter.py` | 13 | Unique counts, re-ID non-inflation, metrics |

## Performance Considerations

- **CPU-only**: ~2-5 FPS expected on ~7.5GB RAM system without GPU
- **frame_skip=3** recommended to achieve reasonable throughput
- Model init: ~5-15s for YOLO + InsightFace on first run
- Embedding cache kept in memory to avoid repeated DB reads

## Known Limitations

1. Single camera cannot determine physical entry/exit direction
2. Partial face occlusions reduce recognition accuracy
3. Rapid face changes (masks on/off) may create duplicate visitor records
4. CPU inference significantly slower than GPU
5. Small faces (<20px) are filtered out, limiting detection range
6. Cosine similarity threshold requires tuning per deployment environment
7. No face anti-spoofing (photos/screens may be detected as real faces)
8. SQLite single-writer limitation (adequate for single-camera use)

## AI Tools Used

- **Antigravity AI** (Google DeepMind): Architecture design, code generation, test creation, and documentation
- **Key prompts**: Complete project specification with 12-section requirements, phased development workflow, and definition of done criteria
