# Face Mesh Analysis - Help

Welcome to the **Face Mesh Analysis** module. This tool performs batch processing of videos for 2D face mesh detection using MediaPipe's FaceMesh model. It processes videos from a specified input directory, overlays face landmarks on each video frame, and exports both normalized and pixel-based landmark coordinates to CSV files.

## 🎯 Key Feature - Iris Landmarks

This module collects **478 face landmarks** including **10 iris landmarks (468-477)** that are critical for gaze/attention analysis in projects like UPro_Soccer. The iris data enables precise tracking of eye movement and attention patterns.

## 🚀 Workflow

1. **Launch**: Open `vaila.py`, verify your settings, and click the **Face Mesh** button (B5_r6_c2, next to Sprint).
2. **Select Device**: Choose between CPU or NVIDIA GPU processing
   - **CPU**: Standard MediaPipe FaceMesh processing (Recommended - GPU support is limited)
   - **NVIDIA GPU**: GPU-accelerated processing (Note: FaceMesh currently uses CPU, GPU support may be available in future versions)
3. **Select Input Directory**: Choose the directory containing your video files (.mp4, .avi, .mov)
4. **Select Output Directory**: Choose where to save processed results
5. **Configure Parameters**: Set MediaPipe parameters via GUI or load TOML configuration
6. **Optional ROI Selection**: 
   - **Bounding Box ROI**: Drag to select rectangular region
   - **Polygon ROI**: Left click to add points, right click to undo, Enter to confirm
7. **Processing**: The script processes all videos in the input directory
8. **Output**: Results are saved in timestamped folders

---

## ⚠️ GPU Support and Limitations

**Important Note:** MediaPipe FaceLandmarker currently has **limited GPU delegate support**. Even when GPU is selected and configured, the processing typically occurs on CPU (XNNPACK delegate). This is a known limitation of MediaPipe FaceLandmarker, not a bug in our implementation.

**Current Status:**
- ✅ GPU detection works correctly
- ✅ CUDA_VISIBLE_DEVICES is configured properly
- ✅ GPU delegate can be created
- ⚠️ FaceLandmarker uses CPU internally (MediaPipe limitation)
- ✅ Code is prepared for when MediaPipe adds full GPU support

**Recommendation:** Use CPU mode for now. GPU acceleration may be available in future MediaPipe versions.

### GPU Requirements (For Future Support)

When MediaPipe implements full GPU support for FaceLandmarker, you will need:

- **NVIDIA GPU**: Any CUDA-capable NVIDIA GPU
- **NVIDIA Drivers**: Latest drivers installed (`nvidia-smi` to verify)
- **CUDA Toolkit**: Required for MediaPipe GPU delegate
  - Linux: Install via NVIDIA website or `conda install -c nvidia cuda-toolkit`
  - Windows: Download from NVIDIA Developer website
  - macOS: Not supported (no CUDA NVIDIA support)
- **cuDNN**: CUDA Deep Neural Network Library
  - Usually included with CUDA Toolkit installation
  - Or install via: `conda install -c nvidia cudnn`
- **MediaPipe**: Version with GPU delegate support (`pip install --upgrade mediapipe`)

### Verifying GPU Setup

```bash
# Check NVIDIA GPU
nvidia-smi

# Check CUDA installation
nvcc --version

# Check cuDNN (if using TensorFlow)
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## 📂 Output Structure

For each processed video, the following files are generated:

### 1. Processed Video (`{video_name}_facemesh.mp4`)
Video with face landmarks overlaid on original frames. Uses H.264 codec (avc1) for better compatibility with QuickTime and web browsers.

### 2. Normalized CSV (`{video_name}_facemesh_norm.csv`)
Landmark coordinates normalized to 0-1 scale. Columns include:
- `frame_index`: Frame number
- `face_idx`: Face index (for multi-face detection)
- `{landmark_name}_x, {landmark_name}_y, {landmark_name}_z`: For each of 478 landmarks

### 3. Pixel CSV (`{video_name}_facemesh_pixel.csv`)
Landmark coordinates in pixel values. Same structure as normalized CSV.

### 4. Configuration File (`configuration_used.toml`)
TOML file containing all parameters used for processing. Can be reused for batch processing.

### 5. Log File (`log_info.txt`)
Processing metadata including:
- Video path and resolution
- FPS and total frames
- Execution time
- FaceMesh configuration
- Frames with missing data

---

## 📈 Understanding the Landmarks

### Total Landmarks: 478

The module collects:
- **468 standard face landmarks**: Complete face mesh including eyes, nose, mouth, eyebrows, face contour
- **10 iris landmarks (468-477)**: Essential for gaze analysis
  - **Right Iris**: 468 (center), 469-472 (peripheral points)
  - **Left Iris**: 473 (center), 474-477 (peripheral points)

### Key Landmark Regions

- **Eyes**: Outer/inner corners, top/bottom points
- **Mouth**: Corners, center, upper/lower inner points
- **Nose**: Tip, bottom, bridge, top
- **Eyebrows**: Inner, middle, outer points
- **Face Contour**: Complete oval outline
- **Iris**: Center and peripheral points for each eye

---

## ⚙️ Configuration Parameters

### MediaPipe FaceMesh Settings

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `min_detection_confidence` | 0.0 - 1.0 | 0.25 | Minimum confidence for face detection |
| `min_tracking_confidence` | 0.0 - 1.0 | 0.25 | Minimum confidence for face tracking |
| `max_num_faces` | 1, 2, ... | 1 | Maximum number of faces to detect |
| `refine_landmarks` | True/False | True | Enable refined landmark detection (includes iris) |
| `apply_filtering` | True/False | True | Apply basic filtering to landmarks |

### Video Processing Settings

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `enable_resize` | True/False | False | Resize video for better detection |
| `resize_scale` | 2, 3, ... | 2 | Scale factor for video resize |
| `enable_padding` | True/False | True | Enable initial frame padding for stabilization |
| `pad_start_frames` | 1, 2, ... | 30 | Number of padding frames at start |

### ROI (Region of Interest) Settings

| Parameter | Type | Description |
|-----------|------|-------------|
| `enable_crop` | Boolean | Enable ROI cropping |
| `bbox_x_min, bbox_y_min` | Integer (pixels) | Top-left corner of bounding box |
| `bbox_x_max, bbox_y_max` | Integer (pixels) | Bottom-right corner of bounding box |
| `roi_polygon_points` | Array of [x, y] | Polygon ROI points (minimum 3) |
| `enable_resize_crop` | Boolean | Upscale cropped region |
| `resize_crop_scale` | Integer | Scale factor for cropped region |

---

## 🔧 Advanced Filtering

The module supports multiple filtering and smoothing methods:

### Interpolation Methods
- **Linear**: Simple linear interpolation
- **Cubic**: Cubic spline interpolation
- **Nearest**: Nearest neighbor interpolation

### Smoothing Methods
- **Savitzky-Golay**: Preserves features while smoothing
- **LOWESS**: Locally weighted scatterplot smoothing
- **Kalman**: State-space filtering
- **Butterworth**: Frequency-domain filtering
- **Splines**: Univariate spline smoothing
- **ARIMA**: Time series smoothing

Configure these via the TOML file or GUI advanced settings.

---

## 🎥 ROI Selection

### Bounding Box ROI
1. Click "Select BBox ROI" button
2. Drag to select rectangular region
3. Press SPACE/ENTER to confirm, ESC to cancel

### Polygon ROI
1. Click "Select Polygon ROI" button
2. Left click to add points
3. Right click to remove last point
4. Press Enter to confirm (minimum 3 points required)
5. Press Esc to cancel, 'R' to reset

---

## 🛠 Troubleshooting

### No faces detected
- Lower `min_detection_confidence` (try 0.1-0.2)
- Enable video resize to improve detection
- Check video quality and lighting conditions
- Ensure face is clearly visible in frame

### Poor landmark accuracy
- Enable `refine_landmarks` for better accuracy
- Use ROI to focus on face region
- Increase video resolution or enable resize
- Ensure good lighting and minimal occlusion

### Video codec issues
- The module attempts H.264 (avc1) first, falls back to mp4v
- If video won't play, try converting with ffmpeg:
  ```bash
  ffmpeg -i input.mp4 -c:v libx264 -preset medium -crf 23 output.mp4
  ```

### GPU not being used
- This is expected - MediaPipe FaceLandmarker currently uses CPU processing
- GPU acceleration may be available in future MediaPipe versions
- Use CPU mode for now - it's optimized and efficient
- If you want to prepare for future GPU support:
  1. Install CUDA Toolkit and cuDNN
  2. Monitor MediaPipe releases for GPU support
  3. Update MediaPipe when GPU support is added

---

## 📦 Dependencies

**Required:**
- opencv-python (cv2)
- mediapipe
- pandas
- numpy
- tkinter
- toml

**Optional (for advanced filtering):**
- scipy
- pykalman
- statsmodels

---

## 💡 Use Cases

- **Sports Performance**: Gaze and attention analysis in team sports (e.g., UPro_Soccer project)
- **Biomechanics**: Facial expression and movement analysis
- **Research**: Behavioral studies requiring precise face tracking
- **Training**: Visual attention and focus assessment
- **Medical**: Facial movement analysis for rehabilitation

---

## 📚 Additional Resources

- [MediaPipe FaceMesh Documentation](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
- [vailá Multimodal Toolbox GitHub](https://github.com/vaila-multimodaltoolbox/vaila)
- [Project Documentation](../../docs/index.md)
