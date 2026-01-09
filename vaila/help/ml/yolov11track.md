# yolov11track

## 📋 Module Information

- **Category:** Ml
- **File:** `vaila\yolov11track.py`
- **Lines:** 3325
- **Version:** 0.0.5
- **Author:** Paulo Roberto Pereira Santiago
- **Email:** paulosantiago@usp.br
- **GitHub:** https://github.com/vaila-multimodaltoolbox/vaila
- **GUI Interface:** ✅ Yes

## 📖 Description

This script performs object detection and tracking on video files using the YOLO model v11, with integrated pose estimation capabilities. It provides a comprehensive solution for tracking objects across video frames and performing pose estimation within tracked bounding boxes.

### Key Features
- Object detection and tracking using the Ultralytics YOLO library
- A graphical interface (Tkinter) for dynamic parameter configuration
- Video processing with OpenCV, including drawing bounding boxes and overlaying tracking data
- Generation of CSV files containing frame-by-frame tracking information per tracker ID
- Video conversion to more compatible formats using FFmpeg
- **Pose estimation within tracked bounding boxes** - New!
- **Merge multiple tracking CSVs** - New!
- **Multiple ID selection for pose estimation** - New!
- **Configurable pose detection parameters** (conf, iou) - New!

### Recent Updates (January 2026)
- **Pose Estimation**: Added YOLO pose estimation functionality within tracked bounding boxes
- **CSV Merge**: Merge multiple tracking CSVs (e.g., `person_id_01.csv`, `person_id_03.csv`, `person_id_06.csv`) into a single CSV for pose estimation
- **Multiple ID Selection**: GUI now supports selecting multiple IDs simultaneously (Ctrl/Cmd + click) to merge their tracking data
- **Flexible Video Loading**: Pose estimation can use any video file in the tracking directory (not just `processed_*.mp4`)
- **Pose Configuration Dialog**: Configure confidence threshold (conf) and IoU threshold (iou) for pose detection with validation and tooltips
- **Directory Structure**: Pose outputs organized by video basename (`{basename}_pose_outputs`)
- **Pose Video Output**: Generates video with skeleton overlay alongside CSV keypoint data

Usage:
    Run the script from the command line by passing the path to a video file as an argument:
            python yolov11track.py

Requirements:
    - Python 3.x
    - OpenCV
    - PyTo...

## 🔧 Main Functions

**Total functions found:** 30+

### Tracking Functions
- `run_yolov11track()` - Main entry point for tracking workflow
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
- `_draw_keypoints_and_skeleton()` - Draw keypoints and skeleton on frame
- `_get_pose_skeleton()` - Return COCO keypoint skeleton edges (17-keypoint layout)

### Utility Functions
- `get_hardware_info()` - Get detailed hardware information for GPU/CPU detection
- `detect_optimal_device()` - Detect and return the optimal device for processing
- `validate_device_choice()` - Validate user device choice and provide feedback
- `get_color_for_id()` - Generate distinct color for each tracker ID using pre-calculated palette
- `get_color_palette()` - Generate maximally distinct color palette using HSV color space
- `standardize_filename()` - Remove unwanted characters and replace spaces with underscores

### ROI Functions
- `select_free_polygon_roi()` - Let user draw a free polygon ROI on the first frame of the video
- `save_roi_to_toml()` - Save ROI polygon to a TOML file
- `load_roi_from_toml()` - Load ROI polygon from a TOML file

### GUI Dialogs
- `TrackerConfigDialog` - Configuration dialog for tracker parameters (device, conf, iou, stride, ROI)
- `ModelSelectorDialog` - Dialog for selecting YOLO model (pre-trained or custom)
- `TrackerSelectorDialog` - Dialog for selecting tracking method (bytetrack, botsort)
- `ClassSelectorDialog` - Dialog for selecting classes to track

## 🎮 Usage: Tracking Workflow

1. **Start Tracking**: Run `python yolov11track.py` or call `run_yolov11track()` from the main GUI
2. **Select Directories**: Choose input directory (containing videos) and output directory
3. **Select Model**: Choose YOLO model (detection, pose, segmentation, OBB) - pre-trained or custom
4. **Select Tracker**: Choose tracking method (ByteTrack or BoTSORT)
5. **Configure Parameters**: Set device (CPU/CUDA), confidence threshold, IoU threshold, video stride, and optionally ROI
6. **Select Classes**: Choose which object classes to track (default: person and sports ball)
7. **Process Videos**: Script processes all videos in the input directory and generates:
   - Individual CSV files per tracked ID: `{label}_id_{tracker_id:02d}.csv`
   - Combined CSV: `all_id_detection.csv`
   - Merged CSV(s): `all_id_merge_{label}.csv` or `all_id_merge.csv`
   - Processed videos: `processed_{video_name}.mp4`

## 🎯 Usage: Pose Estimation Workflow

1. **Start Pose Estimation**: Click "YOLO Pose" button in the main GUI (or call `select_id_and_run_pose()`)
2. **Select Tracking Directory**: Choose the directory containing tracking results (CSV files and video)
3. **Select Video**: 
   - If multiple videos found, a dialog appears to select one
   - Prioritizes `processed_*.mp4` files
   - Accepts any video file (`.mp4`, `.avi`, `.mov`, `.mkv`)
   - Allows manual file selection if no videos found
4. **Select ID(s)**: 
   - **Single ID**: Click on one ID from the list
   - **Multiple IDs**: Hold Ctrl/Cmd and click multiple IDs to merge (e.g., `person_id_01`, `person_id_03`, `person_id_06`)
   - Tip displayed: "Hold Ctrl/Cmd to select multiple IDs for merging"
5. **Configure Pose Parameters**:
   - Select pose model: Nano (fastest), Small, Medium, Large, or XLarge (most accurate)
   - Set confidence threshold (default: 0.10, range: 0.0-1.0) - tooltip available
   - Set IoU threshold (default: 0.70, range: 0.0-1.0) - tooltip available
6. **Process**: Click OK to start pose estimation
7. **Outputs**:
   - **Directory**: `{video_basename}_pose_outputs/`
   - **CSV File**: `{video_basename}_id_{tracker_id:02d}_pose.csv` (or `{video_basename}_{label}_merged_pose.csv` for merged)
   - **Video File**: `{video_basename}_id_{tracker_id:02d}_pose.mp4` (with skeleton overlay)

## 📁 File Formats

### Tracking Outputs
- **Individual CSVs**: `{label}_id_{tracker_id:02d}.csv`
  - Columns: `Frame`, `Tracker ID`, `Label`, `X_min`, `Y_min`, `X_max`, `Y_max`, `Confidence`, `Color_R`, `Color_G`, `Color_B`
- **Combined CSV**: `all_id_detection.csv`
  - Wide format with all IDs as columns
- **Merged CSV**: `all_id_merge_{label}.csv` or `all_id_merge.csv`
  - One row per frame, merging all IDs of the same label

### Pose Estimation Outputs
- **Directory Structure**: `{video_basename}_pose_outputs/`
- **CSV Format**: `{video_basename}_id_{tracker_id:02d}_pose.csv`
  - Columns: `Frame`, `Tracker_ID`, `Label`, then for each keypoint: `{keypoint_name}_x`, `{keypoint_name}_y`, `{keypoint_name}_conf`
  - 17 keypoints: nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle
- **Merged CSV**: `{video_basename}_{label}_merged_pose.csv` (when multiple IDs are selected)
- **Video Output**: `{video_basename}_id_{tracker_id:02d}_pose.mp4` (with skeleton and keypoints drawn)

### ROI Configuration
- **ROI Files**: `roi_configs/{video_name}_roi_{timestamp}.toml`
  - Contains ROI polygon points and video metadata

## 🔬 Advanced Features

### CSV Merging for Pose Estimation
When multiple IDs track the same person over time, you can merge their CSVs:
- Select multiple IDs in the GUI (Ctrl/Cmd + click)
- System automatically merges CSVs using the first valid bbox found for each frame
- Result: Single CSV with continuous tracking across all selected IDs
- Output: `{label}_id_merged.csv` in the tracking directory

### Pose Detection Parameters
- **Confidence Threshold (conf)**: Controls how confident the model must be to detect a pose
  - Lower values (e.g., 0.1): More detections but may include false positives
  - Higher values (e.g., 0.5): Fewer but more accurate detections
  - Recommended: 0.1-0.3 for pose estimation
- **IoU Threshold (iou)**: Controls how much overlap is needed to merge multiple detections
  - Higher values (e.g., 0.9): Very strict matching
  - Lower values (e.g., 0.5): More lenient matching
  - Recommended: 0.7 for most cases

### ROI (Region of Interest)
- Define a polygon area to limit detection and tracking
- Create new ROI interactively or load existing ROI from TOML file
- ROI is saved and can be reused for batch processing
- Applied to all videos in the batch




## 🐛 Troubleshooting

- **No tracking CSV files found**: Ensure you've run the tracking workflow first. CSV files should be named `{label}_id_{tracker_id:02d}.csv`
- **No video found for pose estimation**: The script looks for `processed_*.mp4` files first, then any video file (`.mp4`, `.avi`, `.mov`, `.mkv`). If none found, you can manually select a video file
- **Multiple IDs for same person**: Use the multiple selection feature (Ctrl/Cmd + click) to select all IDs that track the same person. The system will automatically merge their CSVs
- **Pose estimation fails**: Check that:
  - The tracking CSV file exists and contains valid bounding boxes
  - The video file matches the tracking data (same frame count)
  - The pose model is downloaded (first run will download automatically)
- **ROI not working**: Ensure ROI file exists and contains valid polygon points (at least 3 points). Check the `roi_configs` directory
- **CUDA/GPU issues**: If CUDA is selected but not available, the script will fallback to CPU automatically

## 📝 Notes

- Tracking CSVs use 0-based frame indexing internally but display 1-based in filenames
- Pose estimation processes frame-by-frame, so it may take time for long videos
- Merged CSVs use the first valid bbox found for each frame across all selected IDs
- Video basename is used for organizing pose outputs to avoid conflicts when processing multiple videos
- Confidence and IoU thresholds can be adjusted in the pose configuration dialog before processing

---

📅 **Last Updated:** January 2026 (v0.0.5 - Pose estimation, CSV merging, multiple ID selection)  
🔗 **Part of vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
