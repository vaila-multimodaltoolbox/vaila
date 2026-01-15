# yolov26track

## 📋 Module Information

- **Category:** Ml
- **File:** `vaila/yolov26track.py`
- **Lines:** 3980+
- **Version:** 0.0.1
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

### YOLO26 Models Available
- Detection: `yolo26n.pt`, `yolo26s.pt`, `yolo26m.pt`, `yolo26l.pt`, `yolo26x.pt`
- Pose: `yolo26n-pose.pt`, `yolo26s-pose.pt`, `yolo26m-pose.pt`, `yolo26l-pose.pt`, `yolo26x-pose.pt`
- Segmentation: `yolo26n-seg.pt`, `yolo26s-seg.pt`, `yolo26m-seg.pt`, `yolo26l-seg.pt`, `yolo26x-seg.pt`
- OBB (Oriented Bounding Box): `yolo26n-obb.pt`, `yolo26s-obb.pt`, `yolo26m-obb.pt`, `yolo26l-obb.pt`, `yolo26x-obb.pt`

Usage:
    Run the script from the command line:
            python yolov26track.py

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

## 🎮 Usage: Tracking Workflow

1. **Start Tracking**: Run `python yolov26track.py` or call `run_yolov26track()` from the main GUI
2. **Select Directories**: Choose input directory (containing videos) and output directory
3. **Select Model**: Choose YOLO26 model (detection, pose, segmentation, OBB) - pre-trained or custom
4. **Select Tracker**: Choose tracking method (ByteTrack or BoTSORT)
5. **Configure Parameters**: Set device (CPU/CUDA), confidence threshold, IoU threshold, video stride, and optionally ROI
6. **Select Classes**: Choose which object classes to track (default: person and sports ball)
7. **Process Videos**: Script processes all videos in the input directory and generates:
   - Individual CSV files per tracked ID: `{label}_id_{tracker_id:02d}.csv`
   - Combined CSV: `all_id_detection.csv`
   - Merged CSV(s): `all_id_merge_{label}.csv` or `all_id_merge.csv`
   - Processed videos: `processed_{video_name}.mp4`

## 🎯 Usage: Pose Estimation Workflow

1. **Start Pose Estimation**: Click "YOLOv26 Pose" button in the main GUI
2. **Select Tracking Directory**: Choose the directory containing tracking results (CSV files and video)
3. **Select Video**: Prioritizes `processed_*.mp4` files, accepts any video file
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
- Pose outputs: `{tracking_dir}/{video_name}_pose_{timestamp}/`

## 🔬 GPU Auto-Detection

- Script automatically detects and uses optimal device
- CUDA (NVIDIA GPU) is preferred if available
- MPS (Apple Silicon) as secondary option
- Falls back to CPU if no GPU available
- Device selection logged: `Using device: cuda`

## 🐛 Troubleshooting

- **No tracking CSV files found**: Ensure you've run the tracking workflow first
- **GPU not being used**: Check CUDA availability with `torch.cuda.is_available()`
- **Model download fails**: Check internet connection, models are downloaded from Ultralytics servers
- **YOLO26 model not found**: Models are automatically downloaded on first use to `vaila/models/`

---

📅 **Last Updated:** January 2026 (v0.0.1 - Initial YOLOv26 support)  
🔗 **Part of vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
