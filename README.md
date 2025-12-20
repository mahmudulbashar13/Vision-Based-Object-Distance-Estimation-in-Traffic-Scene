# Vision-Based Object Distance Estimation in Traffic Scenes

A Python project for real-time object detection, tracking, and distance estimation in traffic video sequences.

## Pipeline

```
Video Input → YOLO 11 Detection → IoU Tracker → Depth Estimation (v2) → Distance Calculation → Annotated Output
```

## Features

- **Object Detection**: Uses YOLO 11 (via Ultralytics) for robust vehicle/pedestrian detection
- **Object Tracking**: Simple IoU-based tracker for persistent object IDs across frames
- **Depth Estimation**: Supports Depth Anything v2 (trained for meters) or MiDaS fallback
- **Distance Measurement**: Computes per-object distances in meters using depth maps
- **Video Annotation**: Outputs annotated video with bounding boxes, track IDs, and distances
- **CSV Export**: Exports frame-by-frame distance data for analysis

## Requirements

- Python 3.8+
- PyTorch (with CUDA support optional)
- Ultralytics (YOLO)
- OpenCV
- NumPy

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Basic Example (MiDaS fallback — relative depth)

```bash
python run_distance_estimation.py \
  --yolo-weights models/yolov11.pt \
  --input traffic_video.mp4 \
  --out-video output_annotated.mp4 \
  --out-csv distances.csv \
  --device cpu
```

### With Depth Anything v2 (absolute meters)

```bash
python run_distance_estimation.py \
  --yolo-weights models/yolov11.pt \
  --depth-type custom \
  --depth-checkpoint models/depth_anything_v2.pth \
  --input traffic_video.mp4 \
  --out-video output_annotated.mp4 \
  --out-csv distances.csv \
  --device cuda
```

## Options

- `--yolo-weights`: Path to YOLO model weights (required)
- `--depth-type`: Depth model type: `midas` (default) or `custom`
- `--depth-checkpoint`: Path to Depth Anything v2 checkpoint (required if `--depth-type custom`)
- `--input`: Input video file path (required)
- `--out-video`: Output annotated video path (default: `output_annotated.mp4`)
- `--out-csv`: Output CSV path (default: `distances.csv`)
- `--device`: PyTorch device (`cpu` or `cuda`, default: `cpu`)
- `--conf`: Detection confidence threshold (default: `0.25`)

## Output

### Annotated Video
- Bounding boxes around detected objects
- Track IDs (persistent across frames)
- Class labels (vehicle, pedestrian, etc.)
- Estimated distances in meters

### CSV File
Columns:
- `frame`: Frame index
- `track_id`: Persistent object ID
- `class_id`: YOLO class ID
- `confidence`: Detection confidence score
- `x1, y1, x2, y2`: Bounding box coordinates
- `distance_m`: Estimated distance in meters

## Model Files

Place your model weights in a `models/` directory:

```
models/
├── yolov11.pt          # YOLO 11 weights
└── depth_anything_v2.pth  # Depth Anything v2 checkpoint (optional)
```

## Notes

- For production use, consider replacing `SimpleTracker` with SORT, DeepSORT, or ByteTrack
- MiDaS outputs relative depth (normalized); for absolute meters, use Depth Anything v2
- GPU acceleration (CUDA) significantly speeds up processing
- Distance estimates depend on depth model accuracy and camera calibration

## Architecture

- `src/distance_estimation/detector.py` — YOLO wrapper
- `src/distance_estimation/tracker.py` — Simple IoU-based tracker
- `src/distance_estimation/depth.py` — Depth model adapter
- `src/distance_estimation/pipeline.py` — Main processing pipeline
- `run_distance_estimation.py` — CLI entry point
