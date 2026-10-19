# Models Directory

This directory stores model weight files. These files are **not** committed to version control due to their large size.

## Required Models

### 1. YOLO Face Detection Model
- **File**: Auto-downloaded by `ultralytics` on first run
- **Source**: Ultralytics model hub
- **Size**: ~6 MB (nano variant)

### 2. InsightFace / ArcFace Model
- **File**: Auto-downloaded by `insightface` on first run  
- **Model Pack**: `buffalo_l`
- **Source**: InsightFace model zoo
- **Size**: ~360 MB

## Notes

- Models are automatically downloaded on first application run
- Ensure a stable internet connection for the initial download
- Downloaded models are cached in `~/.insightface/models/` (InsightFace) and local directory (YOLO)
