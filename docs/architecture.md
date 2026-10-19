# System Architecture

## Overview

The Intelligent Face Tracker is a modular pipeline-based application that processes video streams through face detection, recognition, tracking, and event logging stages.

## Architecture Diagram

```mermaid
graph TD
    subgraph Input
        A[Video File / RTSP Stream]
    end

    subgraph VideoCapture["Video Capture (video_source.py)"]
        B[OpenCV VideoCapture]
        B1[RTSP Reconnection]
        B2[Frame Iterator]
    end

    subgraph Pipeline["Pipeline Orchestrator (pipeline.py)"]
        C[Frame Skip Logic]
        D[Processing Loop]
    end

    subgraph Detection["Face Detection (detector.py)"]
        E[YOLO Model]
        E1[BBox Validation]
        E2[Face Cropping]
    end

    subgraph Recognition["Face Recognition (recognizer.py)"]
        F[InsightFace / ArcFace]
        F1[Embedding Extraction]
        F2[Cosine Similarity Matching]
        F3[Auto-Registration]
    end

    subgraph Tracking["Face Tracking (tracker.py)"]
        G[IoU-based Tracker]
        G1[Hungarian Assignment]
        G2["Track States (Tentative → Confirmed → Lost)"]
        G3[Velocity Prediction]
    end

    subgraph Storage["Persistent Storage"]
        H[(SQLite Database)]
        H1[visitors Table]
        H2[events Table]
        I[Face Crop Images]
        I1["logs/entries/YYYY-MM-DD/"]
        I2["logs/exits/YYYY-MM-DD/"]
    end

    subgraph Logging["Event Logging (event_logger.py)"]
        J[Python Logging Module]
        J1["logs/events.log"]
    end

    subgraph Metrics["Visitor Counting (visitor_counter.py)"]
        K[Unique Visitor Count]
        K1[Entry/Exit Counts]
        K2[Active Tracks]
    end

    subgraph Output
        L[Display Window]
        L1[Annotated Video Frames]
        L2[Metrics Overlay]
    end

    A --> B
    B --> B1
    B --> B2
    B2 --> C
    C -->|Detection Frame| E
    C -->|Skipped Frame| G3
    E --> E1 --> E2
    E2 --> G
    G --> G1 --> G2
    G2 -->|Confirmed Track| F
    G2 -->|Lost Track| J
    F --> F1 --> F2
    F2 -->|Match Found| H1
    F2 -->|No Match| F3 --> H1
    G2 -->|Entry Event| H2
    G2 -->|Exit Event| H2
    H2 --> I
    I --> I1
    I --> I2
    J --> J1
    H --> K
    K --> K1 --> K2
    K2 --> L2
    D --> L --> L1
```

## Module Responsibilities

| Module | File | Responsibility |
|--------|------|----------------|
| Configuration | `app/config.py` | Load, validate, and provide config settings |
| Video Source | `app/video_source.py` | Video capture, RTSP reconnection, frame iteration |
| Detector | `app/detector.py` | YOLO face detection, bbox validation, face cropping |
| Recognizer | `app/recognizer.py` | Embedding extraction, cosine matching, registration |
| Tracker | `app/tracker.py` | IoU tracking, Hungarian assignment, track lifecycle |
| Database | `app/database.py` | SQLite CRUD, embedding serialization, event storage |
| Event Logger | `app/event_logger.py` | Structured logging, face image saving |
| Visitor Counter | `app/visitor_counter.py` | DB-backed metrics aggregation |
| Pipeline | `app/pipeline.py` | Orchestration of all components |
| Main | `main.py` | Entry point, CLI argument parsing |

## Data Flow

1. **Video Input** → OpenCV captures frames from file or RTSP stream
2. **Frame Skip** → Process every (N+1)th frame; predict on skipped frames
3. **Detection** → YOLO extracts face bounding boxes with confidence scores
4. **Tracking** → IoU matcher associates detections to existing tracks
5. **Recognition** → InsightFace generates embeddings for confirmed tracks
6. **Matching** → Cosine similarity against DB embeddings (threshold: 0.45)
7. **Registration** → Unmatched faces get new visitor IDs in SQLite
8. **Entry Event** → Generated when track transitions to CONFIRMED state
9. **Exit Event** → Generated when track is LOST for max_missing_frames
10. **Logging** → Events saved to DB + face crops saved to disk + events.log

## Database Schema

```sql
CREATE TABLE visitors (
    id TEXT PRIMARY KEY,          -- V-XXXXXXXX
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    embedding BLOB NOT NULL,      -- 512 x float32
    visit_count INTEGER DEFAULT 1,
    created_at TEXT
);

CREATE TABLE events (
    id TEXT PRIMARY KEY,          -- E-XXXXXXXXXXXX
    visitor_id TEXT NOT NULL REFERENCES visitors(id),
    track_id INTEGER NOT NULL,
    event_type TEXT CHECK(event_type IN ('entry', 'exit')),
    timestamp TEXT NOT NULL,
    image_path TEXT,
    idempotency_key TEXT UNIQUE,  -- Prevent duplicates
    created_at TEXT
);
```
