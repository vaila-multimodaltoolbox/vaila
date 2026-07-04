# yolov26track

## 📋 Module Information

- **Category:** Ml
- **File:** `vaila/yolov26track.py`
- **Lines:** 3980+
- **Version:** 0.3.68
- **Author:** Paulo Roberto Pereira Santiago
- **Email:** paulosantiago@usp.br
- **GitHub:** https://github.com/vaila-multimodaltoolbox/vaila
- **GUI Interface:** ✅ Yes

## 📖 Description

This script performs object detection and tracking on video files using the **YOLO model v26** (latest version), with integrated pose estimation capabilities. It provides a comprehensive solution for tracking objects across video frames and performing pose estimation within tracked bounding boxes.

### Key Features
- Object detection and tracking using the Ultralytics YOLO26 library
- A graphical interface (Tkinter) for dynamic parameter configuration
- Video processing with OpenCV, including drawing bounding boxes and overlaying tracking data
- Generation of CSV files containing frame-by-frame tracking information per tracker ID
- Video conversion to more compatible formats using FFmpeg
- **Pose estimation within tracked bounding boxes**
- **Merge multiple tracking CSVs**
- **Multiple ID selection for pose estimation**
- **Configurable pose detection parameters** (conf, iou)
- **Automatic GPU detection** - Uses CUDA if available
- **Run modes**: `track`, `track+pose` (**default**), `track+seg`, `run_all (track+seg+pose)`
- **Single-pass track+pose (v0.3.66)**: geometric ID stabilize (SAM3-style IoU+centroid linker), upscaled bbox ROI for YOLO pose, global keypoint remap in the same tracking loop
- **Unified geometric Re-ID (v0.3.68)**: shared `vaila/geometric_reid.py` module — Hungarian assignment + velocity-direction penalty + optional homography gate; replaces old greedy matching in YOLO, SAM, and markers. CLI exposes `--reid-max-gap`, `--reid-max-dist`, `--reid-min-iou`, `--reid-direction-weight`, `--reid-homography`, `--appearance-reid`
- **New outputs (pose mode)**: `all_id_pose.csv`, `yolo_reid_links.csv`, `<stem>_track_pose_overlay.mp4`, `{label}_id_NN_pose.csv`
- **Segmentation exports** (when model provides masks): `yolo_masks_manifest.csv` (`frame,id,area,mask_png`), `yolo_contours.json` (schema `vaila_yolo_contours_v1` with top-level `video`, `width`, `height`, `fps`, `n_frames`, `object_ids` aligned with SAM-style consumers), `yolo_masks/` PNGs

### YOLO26 Models Available
- Detection: `yolo26n.pt`, `yolo26s.pt`, `yolo26m.pt`, `yolo26l.pt`, `yolo26x.pt`
- Pose: `yolo26n-pose.pt`, `yolo26s-pose.pt`, `yolo26m-pose.pt`, `yolo26l-pose.pt`, `yolo26x-pose.pt`
- Segmentation: `yolo26n-seg.pt`, `yolo26s-seg.pt`, `yolo26m-seg.pt`, `yolo26l-seg.pt`, `yolo26x-seg.pt`
- OBB (Oriented Bounding Box): `yolo26n-obb.pt`, `yolo26s-obb.pt`, `yolo26m-obb.pt`, `yolo26l-obb.pt`, `yolo26x-obb.pt`

Usage:
    Run the script from the command line:
            uv run python -m vaila.yolov26track

Requirements:
    - Python 3.x
    - OpenCV
    - PyTorch
    - Ultralytics (YOLO)
    - Tkinter

## 🔧 Main Functions

**Total functions found:** 30+

### Tracking Functions
- `run_yolov26track()` - Main entry point for tracking workflow
- `initialize_csv()` - Initialize CSV file for a specific tracker ID
- `update_csv()` - Update CSV file with detection data for a specific frame
- `create_combined_detection_csv()` - Create combined CSV with detection data organized by ID columns
- `create_merged_detection_csv()` - Create wide per-label CSV(s) with one row per frame

### Pose Estimation Functions
- `select_id_and_run_pose()` - GUI to select tracking directory, view video with bboxes/IDs, and select ID(s) for pose estimation
- `process_pose_for_single_id()` - Process pose estimation for a single tracked ID
- `process_pose_for_merged_csv()` - Process pose estimation using a merged CSV file (from multiple IDs)
- `_process_pose_from_csv()` - Shared function to process pose estimation from a CSV file
- `merge_tracking_csvs()` - Merge multiple tracking CSV files into a single CSV
- `pick_video_for_pose()` - Find video file for pose estimation (prioritizes `processed_*.mp4`, accepts any video)

### Utility Functions
- `get_hardware_info()` - Get detailed hardware information for GPU/CPU detection
- `detect_optimal_device()` - Detect and return the optimal device for processing (CUDA/MPS/CPU)
- `validate_device_choice()` - Validate user device choice and provide feedback
- `get_color_for_id()` - Generate distinct color for each tracker ID using pre-calculated palette

### ROI Functions
- `select_free_polygon_roi()` - Let user draw a free polygon ROI on the first frame of the video
- `save_roi_to_toml()` - Save ROI polygon to a TOML file
- `load_roi_from_toml()` - Load ROI polygon from a TOML file

### GUI Dialogs
- `TrackerConfigDialog` - Configuration dialog for tracker parameters (device, conf, iou, stride, ROI)
- `ModelSelectorDialog` - Dialog for selecting YOLO26 model (pre-trained or custom)
- `TrackerSelectorDialog` - Dialog for selecting tracking method (bytetrack, botsort)
- `ClassSelectorDialog` - Dialog for selecting classes to track

## 🖥️ Headless CLI (`track`) — export biomechanics CSVs without the GUI

Run detection + tracking on one video and write the CSVs vailá's kinematics
pipeline needs. Ultralytics' own `yolo track` only saves a video; this
subcommand also exports the **REC2D/REC3D markers CSV** and the per-ID bbox
tracks:

```bash
uv run python -m vaila.yolov26track track \
  --model /ABS/runs/<run>/weights/best.pt \
  --source /ABS/video.mp4 \
  --class-name person --conf 0.25 --imgsz 1280 \
  --tracker botsort.yaml --max-ids 26 --anchor bottom \
  --pose --pose-model yolo26n-pose.pt --stabilize-ids --pose-min-roi 256
```

Outputs (in `<video_dir>/processed_yolotrack_<stem>_<timestamp>/` by default,
or `--output DIR`):

- `<stem>_markers.csv` — **getpixelvideo point format** `frame,p1_x,p1_y,...,pN_x,pN_y` (one anchor point per player). **This is the file you feed into REC2D (`rec2d.py`) / REC3D (`rec3d.py`)** with your DLT parameters.
- `{label}_id_NN.csv` — one per tracked ID (`Frame, Tracker ID, Label, X_min, Y_min, X_max, Y_max, Confidence, Color_R/G/B`)
- `all_id_detection.csv` — wide bbox table, **load this in getpixelvideo** (`Load Tracking CSV`) to edit/inspect tracks
- `<stem>_track_overlay.mp4` — annotated **H.264 MP4** (bbox only; skip with `--no-save-video`)
- `<stem>_track_pose_overlay.mp4` — bbox + skeleton overlay when `--pose` (default on)
- `all_id_pose.csv` — long-format pose keypoints for all IDs
- `yolo_reid_links.csv` — audit trail `frame,raw_id,stable_id` when geometric stabilize is on
- `{label}_id_NN_pose.csv` — per-ID COCO keypoints (17 joints × x,y,conf)

Key flags:

- `--pose` / `--no-pose` — inline upscaled pose in tracked bboxes (default **on**)
- `--pose-model`, `--pose-conf`, `--pose-iou`, `--pose-min-roi`, `--pose-pad-pct`
- `--stabilize-ids` / `--no-stabilize-ids` — Hungarian geometric ID linker after BoT-SORT (default on)
- `--reid-max-gap N` — max frame gap for Re-ID (default 12)
- `--reid-max-dist PX` — max centroid distance in pixels (default 180)
- `--reid-min-iou F` — minimum IoU gate (default 0.05)
- `--reid-direction-weight F` — velocity-direction penalty (default 0.5, 0=off)
- `--reid-homography FILE` — optional 3×3 homography (.npy/.npz/.csv) for pitch-plane distances
- `--appearance-reid` / `--no-appearance-reid` — optional OSNet post-pass via reid_yolotrack (default off)
- `--appearance-reid-threshold F` — cosine similarity threshold (default 0.6)

- `--anchor center|bottom|top|left|right|corners` — which point of the bbox becomes the marker for REC2D/REC3D. Default `bottom` (foot/ground contact), best for planar gait/field kinematics; use `center` for centroid trajectories.
- `--max-ids N` — keep only the N most persistent IDs, re-ranked 1..N (recommended to clean up fragmented tracklets; also gives stable `p1..pN` columns).
- `--classes 0 32` — restrict class indices; `--vid-stride N` — process every Nth frame; `--device auto|cuda|cpu`; `--conf/--iou/--imgsz`.

### Biomechanics / kinematics flow (REC2D / REC3D)

1. Train a detector (see `yolotrain` / `sam_to_yolo`).
2. `track` each camera video → `<stem>_markers.csv` (use the **same `--anchor`** and **`--max-ids`** for every camera so `p1..pN` columns line up across views for REC3D).
3. Run **REC2D** (single camera, planar) or **REC3D** (multi-camera) with your DLT2D/DLT3D parameters; the markers CSV is the direct pixel-coordinate input (`frame,p1_x,p1_y,...`).
4. Need manual correction? Open the video in **getpixelvideo** and `Load Tracking CSV` (the `*_markers.csv` or `all_id_detection.csv`).

## 🎮 Usage: Tracking Workflow

1. **Start Tracking**:
   - GUI: **Frame B → "Video AI tools" → "Tracker (v26)"**
   - CLI (open GUI): `uv run python -m vaila.yolov26track`
   - CLI (headless tracking → CSVs): `uv run python -m vaila.yolov26track track --model best.pt --source video.mp4`
2. **Select Directories**: Choose input directory (videos) and output directory  
3. **Ultralytics paths**: Weights and cache live under `vaila/models/` (including BoT-SORT classifier downloads); stray `yolo*.pt` in the repo root are moved into `vaila/models/` on startup.
4. **TensorRT**: If `trtexec` fails (e.g. TensorRT 10 flag changes), processing falls back to the `.pt` model automatically. Check the log for full `trtexec` stderr.
5. **Select Model**: Choose YOLO26 model (detection, pose, segmentation, OBB) - pre-trained or custom
6. **Select Tracker**: Choose tracking method (ByteTrack or BoTSORT)
7. **Configure Parameters**: Set device (CPU/CUDA), confidence threshold, IoU threshold, video stride, **Run Mode**, and optionally ROI
8. **Select Classes**: Choose which object classes to track (default: person and sports ball)
9. **Process Videos**: Script processes all videos in the input directory and generates:
   - Individual CSV files per tracked ID: `{label}_id_{tracker_id:02d}.csv`
   - Combined CSV: `all_id_detection.csv`
   - Merged CSV(s): `all_id_merge_{label}.csv` or `all_id_merge.csv`
   - Processed videos: `processed_{video_name}.mp4`
   - If **Run Mode** includes **seg** and model outputs masks:
     - `yolo_masks_manifest.csv` with header `frame,id,area,mask_png`
     - `yolo_contours.json` (`vaila_yolo_contours_v1`) with SAM-compatible top-level metadata
     - `processed_<stem>_seg.mp4` and `processed_<stem>_all.mp4` (H.264 via FFmpeg, same path as bbox video)
     - `yolo_masks/` (PNG masks per frame/object)
   - If **Run Mode** includes **pose** (default `track+pose`):
     - Inline pose in each tracked bbox (ROI padded + upscaled to `--pose-min-roi`)
     - `all_id_pose.csv`, `yolo_reid_links.csv`, `<stem>_track_pose_overlay.mp4`
     - `{label}_id_NN_pose.csv` per tracked person
     - Optional **Geometric ID stabilize** checkbox (SAM3-style linker)

## 🎯 Usage: Pose Estimation Workflow

1. **Start Pose Estimation**: GUI: **Frame B → "Video AI tools" → "Pose (video)"** or **"Pose (tracking)"**
2. **Select Tracking Directory** (only for **Pose (tracking)**): Choose results root or subfolder; nested `*_id_*.csv` trees under e.g. a `vailatracker_*` root are discovered automatically (bounded depth), with a picker if several leaves exist. For **Pose (video)** you only select a video file next.
3. **Select Video** (**Pose (tracking)**): Prioritizes `processed_*.mp4`, accepts any video in the resolved tracking folder
4. **Select ID(s)**: 
   - **Single ID**: Click on one ID from the list
   - **Multiple IDs**: Hold Ctrl/Cmd and click multiple IDs to merge
5. **Configure Pose Parameters**:
   - Select pose model: `yolo26n-pose.pt` to `yolo26x-pose.pt`
   - Set confidence threshold (default: 0.10)
   - Set IoU threshold (default: 0.70)
6. **Process**: Click OK to start pose estimation
7. **Outputs**:
   - **Directory**: `{video_basename}_pose_{timestamp}/`
   - **CSV File**: Keypoint data for all 17 COCO keypoints
   - **Video File**: With skeleton overlay

## 📁 Output Directory Structure

Models are downloaded to: `vaila/models/`
- Tracker configs: `vaila/models/trackers/`
- Ultralytics runs/cache redirected under: `vaila/models/ultralytics/`
- Pose outputs: `{tracking_dir}/{video_name}_pose_{timestamp}/`

## 🔬 GPU Auto-Detection

- Script automatically detects and uses optimal device
- CUDA (NVIDIA GPU) is preferred if available
- MPS (Apple Silicon) as secondary option
- Falls back to CPU if no GPU available
- Device selection logged: `Using device: cuda`
- Optional TensorRT engines via `trtexec`: on failure (e.g. TensorRT 10 CLI changes), vailá logs full stderr and continues with the `.pt` weights

### Memory (CUDA + host RAM)

Long broadcast clips with **Max tracked IDs** > 0 use a two-phase pass:

1. **Phase 1** — YOLO+BoT-SORT inference; only detection metadata is buffered (not full frames). Periodic `gc` + `torch.cuda.empty_cache` runs on CUDA (same pattern as SAM3).
2. **Phase 2** — annotated video and CSVs are written by **re-reading the source video** from disk.

Terminal lines such as `VRAM 18.2/24.0 GiB free | RAM 42.1 GiB avail` help correlate with `nvidia-smi`. If host RAM is exhausted, prefer **Max tracked IDs = 0** (single streaming pass) or a higher **vid_stride**.

## 🐛 Troubleshooting

- **No tracking CSV files found**: Run tracking first, or pick a higher-level output folder — nested folders with `*_id_*.csv` are scanned up to a bounded depth; if several are found, a folder picker appears.
- **GPU not being used**: Check CUDA availability with `torch.cuda.is_available()`
- **Model download fails**: Check internet connection, models are downloaded from Ultralytics servers
- **YOLO26 model not found**: Models are automatically downloaded on first use to `vaila/models/`

---

📅 **Last Updated:** 04 July 2026 (v0.3.68)  
🔗 **Part of vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
