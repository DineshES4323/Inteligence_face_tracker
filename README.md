# Intelligent Face Tracker

An AI-driven face detection, recognition, tracking, automatic registration, and entry/exit logging system. Processes local video files and live RTSP camera streams to detect faces, generate unique embeddings, track individuals across frames, register new visitors automatically, re-identify returning visitors, and log entry/exit events with face crop images.

## Features

- **YOLO Face Detection**: Real-time face detection using Ultralytics YOLO with configurable confidence
- **ArcFace Recognition**: 512-dimensional face embeddings via InsightFace for visitor identification
- **IoU Multi-Object Tracking**: Hungarian assignment-based tracker with track lifecycle management
- **Auto-Registration**: Automatically register new visitors when no embedding match is found
- **Re-Identification**: Recognize returning visitors via cosine similarity matching
- **Entry/Exit Events**: Exactly one entry and one exit event per track lifecycle with idempotency protection
- **Face Image Logging**: Cropped face saved for every entry/exit event, organized by date
- **SQLite Database**: Persistent visitor records and event history with foreign keys and indexes
- **RTSP Support**: Live camera streams with automatic reconnection
- **Visitor Metrics**: Unique visitor count, entry/exit totals, and active track statistics â€” all DB-backed
- **Configurable Pipeline**: JSON-based configuration for all parameters
- **Graceful Shutdown**: Clean exit on Ctrl+C with remaining exit events logged

## Architecture

```mermaid
graph LR
    A[Video/RTSP] --> B[YOLO Detection]
    B --> C[IoU Tracker]
    C --> D[InsightFace Recognition]
    D --> E[(SQLite DB)]
    C --> F[Event Logger]
    F --> G[Face Crops]
    F --> H[events.log]
    E --> I[Visitor Metrics]
```

See [docs/architecture.md](docs/architecture.md) for the full architecture diagram.

## Prerequisites

- **Python 3.11+**
- **Windows 10/11** (tested), Linux/macOS (should work)
- **Internet connection** for first-run model downloads (~370 MB)
- ~1 GB free disk space (models + data)

## Setup Instructions (Windows PowerShell)

### 1. Clone or Navigate to the Project

```powershell
cd "c:\Users\ashik\OneDrive\Documents\Intelligent Face Tracker"
```

### 2. Create Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Download Sample Video

Download the sample video from:
https://drive.google.com/drive/folders/15YCN3CYb97GyIoNUV6NJxGNIf-rUBFUJ?usp=sharing

1. Open the link in your browser
2. Download the video file
3. Place it at: `data/sample.mp4`

### 5. Model Setup

Models are **downloaded automatically** on first run:
- **YOLO face model** (~6 MB): Downloaded by Ultralytics
- **InsightFace buffalo_l** (~360 MB): Downloaded by InsightFace

No manual model download is required. Ensure a stable internet connection for the first run.

## Configuration

Edit `config.json` to customize the application:

```json
{
  "video_source": "data/sample.mp4",
  "camera_mode": "video",
  "frame_skip": 3,
  "detection_confidence": 0.5,
  "recognition_threshold": 0.45,
  "max_missing_frames": 30,
  "database_path": "data/visitors.db",
  "log_directory": "logs",
  "output_directory": "output",
  "display_window": true
}
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `video_source` | Path to video file or RTSP URL | `data/sample.mp4` |
| `camera_mode` | `video` for files, `rtsp` for streams | `video` |
| `frame_skip` | Frames skipped between detections (3 = process every 4th) | `3` |
| `detection_confidence` | YOLO detection confidence threshold (0-1) | `0.5` |
| `recognition_threshold` | Cosine similarity threshold for face matching (0-1) | `0.45` |
| `max_missing_frames` | Frames before a lost track generates an exit event | `30` |
| `database_path` | SQLite database file path | `data/visitors.db` |
| `log_directory` | Directory for logs and face crop images | `logs` |
| `output_directory` | Directory for output files | `output` |
| `display_window` | Show annotated video window | `true` |

## Running

### Process a Local Video

```powershell
python main.py
```

### Use a Custom Config

```powershell
python main.py --config my_config.json
```

### Configure an RTSP Stream

Edit `config.json`:
```json
{
  "video_source": "rtsp://username:password@camera-ip:554/stream",
  "camera_mode": "rtsp"
}
```

Then run:
```powershell
python main.py
```

**Controls during playback:**
- Press `q` or `ESC` to quit gracefully
- Press `Ctrl+C` in the terminal for graceful shutdown

## Running Tests

```powershell
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_database.py -v

# Run with print output
python -m pytest tests/ -v -s
```

**Test results:** 58 automated tests passed in the latest local test run.

See [docs/TESTING.md](docs/TESTING.md) for detailed test documentation.

## Database Schema

### Visitors Table
```sql
SELECT id, first_seen, last_seen, visit_count FROM visitors;
```

### Events Table
```sql
SELECT id, visitor_id, track_id, event_type, timestamp, image_path FROM events;
```

### Sample Queries

```sql
-- Unique visitor count
SELECT COUNT(*) AS unique_visitors FROM visitors;

-- Entry/exit event counts
SELECT event_type, COUNT(*) AS count FROM events GROUP BY event_type;

-- Events for a specific visitor
SELECT * FROM events WHERE visitor_id = 'V-XXXXXXXX' ORDER BY timestamp;

-- Most recent events
SELECT * FROM events ORDER BY timestamp DESC LIMIT 10;

-- Visitors seen today
SELECT * FROM visitors WHERE last_seen >= date('now');
```

## Output Structure

```
logs/
    events.log              # Structured application log
    entries/
        2024-01-15/         # Entry face crops by date
            V-ABC12345_entry_143022_E1234567.jpg
    exits/
        2024-01-15/         # Exit face crops by date
            V-ABC12345_exit_143500_E7654321.jpg
data/
    visitors.db             # SQLite database
```

## Example Event Log Output

```
2024-01-15T14:30:22 | INFO     | app.database              | Registered new visitor: V-A1B2C3D4
2024-01-15T14:30:22 | INFO     | app.tracker               | Track 1 confirmed
2024-01-15T14:30:22 | INFO     | app.pipeline              | ENTRY EVENT: visitor=V-A1B2C3D4, track=1, event=E-123456789ABC
2024-01-15T14:35:00 | INFO     | app.tracker               | Track 1 removed (visitor=V-A1B2C3D4, misses=30)
2024-01-15T14:35:00 | INFO     | app.pipeline              | EXIT EVENT: visitor=V-A1B2C3D4, track=1, event=E-DEF456789012
```

## Compute Requirements

| Component | CPU-Only | With GPU |
|-----------|----------|----------|
| Expected FPS (720p) | 2-5 FPS | 15-30 FPS |
| RAM Usage | ~300-600 MB | ~300-600 MB |
| VRAM | N/A | ~1-2 GB |
| Model Init Time | 10-25 seconds | 5-10 seconds |

See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for detailed benchmarks and optimization tips.

## Assumptions and Limitations

1. **Single Camera View**: "Entry" and "exit" refer to face appearance/disappearance in the camera view, not physical doorway transitions
2. **CPU-Only Default**: No NVIDIA GPU required, but inference is slower (~2-5 FPS vs 15-30 FPS)
3. **Lighting**: Performance degrades in poor lighting or extreme angles
4. **Occlusion**: Partially occluded faces may not be detected or may create duplicate registrations
5. **Threshold Tuning**: Recognition threshold (0.45) is a default â€” tune per environment
6. **No Anti-Spoofing**: Photos or screens showing faces may be detected as real faces
7. **SQLite**: Single-writer limitation (adequate for single-camera deployments)
8. **Frame Skip**: High values may miss fast-moving faces

## Documentation

- [AI Planning Document](docs/AI_PLANNING.md) â€” Problem statement, model rationale, design decisions
- [Architecture Diagram](docs/architecture.md) â€” Mermaid diagrams and module responsibilities
- [Testing Guide](docs/TESTING.md) â€” Test suite details and run commands
- [Performance Guide](docs/PERFORMANCE.md) â€” Benchmarks, bottlenecks, optimization

## Demo Video

> **Placeholder**: A demo video will be recorded and linked here once the system has been tested end-to-end with real video data. No demo video exists yet.
>
> *Record a Loom or YouTube video showing: video processing, face detection bounding boxes, visitor IDs, entry/exit events in the log, and the SQLite database contents.*

## Troubleshooting

### "ModuleNotFoundError: No module named 'ultralytics'"
```powershell
pip install ultralytics
```

### "ModuleNotFoundError: No module named 'insightface'"
```powershell
pip install insightface onnxruntime
```

### "Failed to open video source"
- Check that the video file exists at the path specified in `config.json`
- For RTSP: verify the URL, credentials, and that the camera is accessible

### "InsightFace could not detect face in crop"
- The face crop may be too small or blurry
- Try lowering `detection_confidence` to get more detections
- Increase `MIN_FACE_SIZE` in `detector.py` if you're getting too many false positives

### RTSP Stream Disconnects
- The system automatically attempts 10 reconnections with 3-second delays
- Check network stability and camera availability
- Increase `MAX_RECONNECT_ATTEMPTS` in `video_source.py` for unstable networks

### Slow Performance
- Set `frame_skip` to 3 or higher for CPU-only systems
- Lower the video resolution if possible
- Consider GPU acceleration (see [PERFORMANCE.md](docs/PERFORMANCE.md))

### Database Locked
- Ensure only one instance of the application is running
- SQLite uses WAL journal mode for better concurrency
- Delete `data/visitors.db-wal` and `data/visitors.db-shm` if stale locks remain

## Sample Output Workflow

To process the sample video and verify output:

```powershell
# 1. Run the application
python main.py

# 2. After processing, check the event log
Get-Content logs\events.log | Select-Object -Last 20

# 3. Query the database
sqlite3 data\visitors.db "SELECT COUNT(*) AS unique_visitors FROM visitors;"
sqlite3 data\visitors.db "SELECT event_type, COUNT(*) AS count FROM events GROUP BY event_type;"

# 4. Check saved face images
Get-ChildItem logs\entries -Recurse
Get-ChildItem logs\exits -Recurse
```

This project is a part of a hackathon run by https://katomaran.com

