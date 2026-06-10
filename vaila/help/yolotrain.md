# yolotrain

## 📋 Module Information

- **Category:** Ml
- **File:** `vaila/yolotrain.py`
- **Lines:** 1170+
- **Version:** 0.3.51
- **Author:** Paulo Roberto Pereira Santiago
- **Email:** paulosantiago@usp.br
- **GitHub:** https://github.com/vaila-multimodaltoolbox/vaila
- **GUI Interface:** ✅ Yes

## 📖 Description

YOLO training interface for vailá/getpixelvideo and YOLO-format datasets. It can create a tracking/detection dataset from `sam_points_georeid.csv` or any `getpixelvideo.py` CSV with `frame,pN_x,pN_y,...` columns, then train Ultralytics YOLO from the generated `data.yaml`.

### Key Features
- Support for **YOLO26** (latest), YOLO11, YOLOv9, and YOLOv8 models
- Create YOLO tracking/detection datasets from `getpixelvideo.py` / `sam_points_georeid.csv` pixel data
- Fixed-size box labels centered on marked points for YOLO tracker retraining
- Support for existing YOLO format exports
- Automatic YAML file generation with comprehensive options
- GPU auto-detection (CUDA, MPS, CPU)
- Model download management to `vaila/models/` directory

### Supported Models

#### YOLO26 (Recommended - 2026)
- `yolo26n.pt` - Nano (fastest)
- `yolo26s.pt` - Small
- `yolo26m.pt` - Medium (default)
- `yolo26l.pt` - Large
- `yolo26x.pt` - XLarge (most accurate)

#### YOLO11
- `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt`

#### YOLOv9
- `yolov9c.pt`, `yolov9s.pt`, `yolov9m.pt`, `yolov9l.pt`, `yolov9e.pt`

#### YOLOv8
- `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`

## Usage

### GUI (recommended)

- In the main window: **Frame B → "Video AI tools" → "Train YOLO"**

### CLI (open GUI)

Run with `uv` (do not activate venv manually):

```bash
uv run python -m vaila.yolotrain
```

### getpixelvideo CSV to YOLO tracking dataset

Use the GUI button **Create Dataset from getpixelvideo CSV**. Select the CSV, the source video, enter the class name, and choose a fixed box size. The generated dataset is selected automatically for **Start Training**.

Expected CSV columns include `frame` and point pairs such as `p1_x,p1_y,p2_x,p2_y,...`; `pN_mx/pN_my` and `pN_cx/pN_cy` are used as fallbacks when present.

### CLI (Ultralytics train / val / test / predict)

This module is a Tk GUI wrapper. If you want fully headless runs (train/val/test),
use the Ultralytics CLI directly:

```bash
# Detect/tracking training (vaila getpixelvideo CSV export or YOLO dataset)
uv run yolo detect train model=yolo11n.pt data=/ABS/path/to/data.yaml epochs=100 imgsz=640 device=0

# Validate on val split
uv run yolo detect val model=/ABS/path/to/runs/detect/train/weights/best.pt data=/ABS/path/to/data.yaml split=val

# Evaluate on test split (if your dataset has test/)
uv run yolo detect val model=/ABS/path/to/runs/detect/train/weights/best.pt data=/ABS/path/to/data.yaml split=test

# Segmentation (mask) training
uv run yolo segment train model=yolo11n-seg.pt data=/ABS/path/to/data.yaml epochs=100 imgsz=640 device=0

# Pose training (keypoints)
uv run yolo pose train model=yolo26s-pose.pt data=/ABS/path/to/data.yaml epochs=200 imgsz=1280 device=0

# Predict (inference) on a video
uv run yolo predict model=/ABS/path/to/weights/best.pt source=/ABS/path/to/video.mp4 conf=0.25 device=0
```

Notes:

- Use absolute paths for `data=` and `model=` to avoid saving under nested `runs/pose/runs/pose/...`.
- For FIFA soccer-pitch keypoints (32 kp), see `soccerfield_keypoints_ai` + `docs/fifa_workflow.md` §4.5.

## Requirements
- Python 3.x
- Ultralytics YOLO
- Tkinter (for GUI operations)
- PyTorch with CUDA support (optional, for GPU training)

## 🔧 Main Functions

- `run_yolotrain_gui()` - Entry point for training GUI
- `YOLOTrainApp` - Main application class
- `create_widgets()` - Create GUI elements
- `_update_model_list()` - Filter models by category (YOLO26, YOLO11, YOLOv8, YOLOv9)
- `show_model_help()` - Show model selection guide
- `show_anylabeling_help()` - Show dataset guide
- `create_dataset_from_getpixelvideo_csv()` - Build YOLO dataset from pixel CSV + video
- `browse_dataset()` - Browse for dataset folder
- `browse_yaml()` / `create_new_yaml()` - YAML configuration
- `start_training_thread()` - Start training in background thread
- `_run_training()` - Execute YOLO training
- `_show_model_characteristics()` - Display model info

## 📁 Model Storage

All models are downloaded to: `vaila/models/`

This includes:
- Pre-trained detection models
- Pose estimation models
- Segmentation models
- OBB (Oriented Bounding Box) models

## 🔬 Dataset Structure

Expected YOLO export structure:
```
your_dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/ (optional)
│   ├── images/
│   └── labels/
└── classes.txt
```

---

📅 **Last Updated:** 2026-06-10 (v0.3.51 - getpixelvideo CSV tracking dataset builder)  
🔗 **Part of vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)

