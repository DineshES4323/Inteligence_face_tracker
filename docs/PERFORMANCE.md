# Performance Documentation

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Quad-core x64 | 8+ cores |
| RAM | 4 GB | 8+ GB |
| GPU | Not required | NVIDIA with CUDA (optional) |
| Storage | 500 MB (models) | 1+ GB (models + data) |
| Python | 3.11+ | 3.11+ |

## Model Initialization Overhead

| Model | Download Size | Load Time (CPU) | Notes |
|-------|--------------|-----------------|-------|
| YOLOv8n-face | ~6 MB | 2-5 seconds | Auto-downloads on first run |
| InsightFace buffalo_l | ~360 MB | 5-15 seconds | Auto-downloads on first run |
| **Total first-run** | ~370 MB | 10-25 seconds | Subsequent runs use cached models |

## Expected Performance (CPU-Only)

> [!IMPORTANT]
> These are estimates based on typical hardware. Actual performance will vary. Benchmarks should be measured on the target deployment machine.

### Estimated FPS by Resolution

| Resolution | frame_skip=0 | frame_skip=3 | frame_skip=5 |
|------------|-------------|-------------|-------------|
| 640×480 | ~3-5 FPS | ~8-12 FPS | ~12-18 FPS |
| 1280×720 | ~1-3 FPS | ~4-8 FPS | ~8-12 FPS |
| 1920×1080 | ~0.5-1.5 FPS | ~2-5 FPS | ~5-8 FPS |

*FPS includes detection + recognition + tracking. Recognition runs only on newly confirmed tracks.*

### RAM Usage (Estimated)

| Component | Memory |
|-----------|--------|
| YOLO model | ~50-100 MB |
| InsightFace models | ~200-400 MB |
| Video frame buffer | ~5-15 MB per frame |
| Embedding cache (100 visitors) | ~0.2 MB |
| **Total** | **~300-600 MB** |

## Frame-Skip Trade-offs

| frame_skip | Detection Rate | FPS Gain | Trade-off |
|------------|---------------|----------|-----------|
| 0 | Every frame | 1× (baseline) | Maximum accuracy, slowest |
| 1 | Every 2nd frame | ~1.8× | Minimal accuracy loss |
| 3 | Every 4th frame | ~3.0× | **Recommended for CPU** |
| 5 | Every 6th frame | ~4.5× | May miss fast-moving faces |
| 10 | Every 11th frame | ~7× | Significant tracking gaps |

On skipped frames, the tracker uses linear velocity prediction to estimate positions.

## Bottleneck Analysis

| Stage | CPU Time (per frame) | Bottleneck Level |
|-------|---------------------|-----------------|
| Video decode | ~1-3 ms | Low |
| YOLO detection | ~50-200 ms | **High** |
| InsightFace embedding | ~30-100 ms | **High** (per face) |
| IoU tracking | ~0.1-1 ms | Low |
| Database queries | ~0.5-2 ms | Low |
| Display rendering | ~1-5 ms | Low |

**Primary bottleneck**: Neural network inference (YOLO + InsightFace) on CPU.

## GPU Acceleration

If an NVIDIA GPU with CUDA is available:

1. Install `onnxruntime-gpu` instead of `onnxruntime`:
   ```powershell
   pip install onnxruntime-gpu
   ```

2. Update the detector device in `app/detector.py`:
   ```python
   self.detector = YOLOFaceDetector(device="cuda")
   ```

3. Expected GPU speedup: 5-20× for detection and recognition

## Optimization Recommendations

1. **Use frame_skip=3+** on CPU systems for real-time operation
2. **Lower detection resolution** via YOLO's `imgsz` parameter
3. **Limit concurrent faces** — recognition is per-face, so crowded scenes are slower
4. **Pre-filter by size** — MIN_FACE_SIZE filters out distant, low-quality faces
5. **Batch recognition** if processing stored video (not implemented yet)
6. **Use GPU inference** when NVIDIA hardware is available

## Benchmark Procedure

To measure actual performance on your hardware:

```powershell
# Run with a sample video and check the FPS overlay
python main.py

# The FPS is displayed on the video window and logged to events.log
# Look for lines like: "FPS: X.X | Frame: NNNN"
```

For automated benchmarking:
```python
import time
# Read FPS from pipeline._current_fps after processing
# Log start/end time and total frames for average FPS
```

> [!NOTE]
> All performance figures in this document are estimates. They have NOT been measured on the target deployment hardware. Run the benchmark procedure above for accurate numbers.
