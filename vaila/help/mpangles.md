# MP Angles - MediaPipe Angle Calculation

## 📋 Module Information

- **Category:** Utils
- **File:** `vaila/mpangles.py`
- **Lines:** 2272
- **Version:** 0.0.2
- **Author:** Paulo R. P. Santiago
- **Email:** paulosantiago@usp.br
- **Creation Date:** 31 March 2025
- **Update Date:** 10 November 2025
- **Python Version:** 3.12.12
- **GUI Interface:** ✅ Yes
- **License:** AGPL v3.0

## 📖 Description

The **MP Angles** module calculates absolute and relative joint angles from landmark coordinates obtained from MediaPipe pose estimation. This tool processes CSV files containing MediaPipe landmark data (33 landmarks) and generates comprehensive angle calculations for biomechanical analysis.

### Key Features

1. **Absolute Angles:**
   - Calculates angles between body segments and the horizontal axis
   - Uses `arctan2` for robust angle calculation
   - Supports two formats: 0-360° or -180° to +180°

2. **Relative Angles:**
   - Computes joint angles between connected body segments
   - Uses dot product and `arccos` for angle calculation
   - Provides biomechanically meaningful joint angles

3. **Video Visualization:**
   - Creates annotated videos with skeleton overlay
   - Displays both relative and absolute angles in real-time
   - Color-coded visualization (red for right side, blue for left side)

## 🔧 Main Functions

### Core Functions

- **`run_mp_angles()`**: Main entry point - handles user interaction and processing workflow
- **`process_angles(input_csv, output_csv, format_360=False)`**: Processes CSV files and calculates all angles
- **`process_video_with_visualization(video_path, csv_path, output_dir, format_360=False)`**: Processes video with angle visualization
- **`select_directory()`**: Opens dialog to select directory with CSV files
- **`select_csv_file()`**: Opens dialog to select a CSV file
- **`select_video_file()`**: Opens dialog to select a video file

### Angle Calculation Functions

- **`compute_absolute_angle(p_proximal, p_distal, format_360=False)`**: Calculates absolute angle between two points
- **`compute_relative_angle(a, b, c)`**: Calculates angle at point b between vectors ba and bc
- **`compute_knee_angle(hip, knee, ankle)`**: Calculates knee joint angle
- **`compute_hip_angle(hip, knee, trunk_vector)`**: Calculates hip joint angle
- **`compute_ankle_angle(knee, ankle, foot_index, heel)`**: Calculates ankle joint angle
- **`compute_shoulder_angle(shoulder, elbow, trunk_vector)`**: Calculates shoulder joint angle
- **`compute_elbow_angle(shoulder, elbow, wrist)`**: Calculates elbow joint angle
- **`compute_wrist_angle(elbow, wrist, pinky, index)`**: Calculates wrist joint angle
- **`compute_neck_angle(mid_ear, mid_shoulder, trunk_vector)`**: Calculates neck angle
- **`get_vector_landmark(data, landmark)`**: Extracts x,y coordinates for a specific MediaPipe landmark

## 📊 Supported Angles

### Relative Angles (Joint Angles)

The module calculates the following relative angles for both left and right sides:

1. **Neck Angle**: Angle between mid_ear and mid_shoulder relative to trunk
2. **Trunk Angle**: Angle of trunk segment
3. **Shoulder Angle**: Angle between trunk and upper arm
4. **Elbow Angle**: Angle between upper arm and forearm
5. **Wrist Angle**: Angle between forearm and hand
6. **Hip Angle**: Angle between trunk and thigh
7. **Knee Angle**: Angle between thigh and shank
8. **Ankle Angle**: Angle between shank and foot

### Absolute Angles (Segment Angles)

The module calculates absolute angles for the following segments:

**Upper Body:**
- Upper arm (shoulder to elbow)
- Forearm (elbow to wrist)
- Hand (wrist to mid_hand)

**Lower Body:**
- Thigh (hip to knee)
- Shank (knee to ankle)
- Foot (heel to foot_index)

**Central Segments:**
- Trunk (mid_shoulder to mid_hip)
- Neck (mid_ear to mid_shoulder)

## 📥 Input Requirements

### CSV File Format

The input CSV files must contain MediaPipe landmark coordinates in the following format:

- **First column**: Frame index (0, 1, 2, ...)
- **Subsequent columns**: Landmark coordinates in pairs (x, y)
  - Format: `p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, ...`
  - Total: 33 landmarks × 2 coordinates = 66 columns + 1 frame column = 67 columns

### MediaPipe Landmark Order

The CSV must follow MediaPipe's 33-landmark format:
- 0: nose
- 1-10: face landmarks (eyes, ears, mouth)
- 11-12: shoulders
- 13-14: elbows
- 15-16: wrists
- 17-22: hands (pinky, index, thumb)
- 23-24: hips
- 25-26: knees
- 27-28: ankles
- 29-30: heels
- 31-32: foot_index

### Video File Requirements (for visualization)

- Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`
- Video must correspond to the CSV file (same number of frames)
- Video resolution should match the coordinate system used in CSV

## 📤 Output Files

### CSV Processing Mode

When processing CSV files only, the module generates:

1. **`processed_{filename}_rel.csv`**: Relative angles (joint angles)
   - Columns: `frame_index`, `neck`, `trunk`, `right_shoulder`, `right_elbow`, `right_wrist`, `right_hip`, `right_knee`, `right_ankle`, `left_shoulder`, `left_elbow`, `left_wrist`, `left_hip`, `left_knee`, `left_ankle`
   - Units: Degrees (°)
   - Format: 2 decimal places

2. **`processed_{filename}_abs.csv`**: Absolute angles (segment angles)
   - Columns: `frame_index`, `neck_abs`, `trunk_abs`, `right_upperarm_abs`, `right_forearm_abs`, `right_hand_abs`, `right_thigh_abs`, `right_shank_abs`, `right_foot_abs`, `left_upperarm_abs`, `left_forearm_abs`, `left_hand_abs`, `left_thigh_abs`, `left_shank_abs`, `left_foot_abs`
   - Units: Degrees (°)
   - Format: 0-360° or -180° to +180° (user selectable)

### Video Processing Mode

When processing video with visualization, the module generates:

1. **`angles_{videoname}.mp4`**: Annotated video with:
   - Skeleton overlay (red for right side, blue for left side, white for trunk/neck)
   - Real-time display of all relative angles
   - Real-time display of all absolute angles
   - Color-coded text (red/blue/white/yellow)

2. **`{videoname}_rel.csv`**: Relative angles CSV (same format as above)

3. **`{videoname}_abs.csv`**: Absolute angles CSV (same format as above)

### Output Directory Structure

```
input_directory/
├── processed_angles_YYYYMMDD_HHMMSS/
│   ├── processed_file1_rel.csv
│   ├── processed_file1_abs.csv
│   ├── processed_file2_rel.csv
│   └── processed_file2_abs.csv
```

Or for video processing:

```
video_directory/
└── angles_video_YYYYMMDD_HHMMSS/
    ├── angles_video.mp4
    ├── video_rel.csv
    └── video_abs.csv
```

## 🚀 Usage

### Method 1: GUI Mode (Recommended)

1. **Launch from vailá GUI:**
   - Click the **"MP Angles"** button (B4_r4_c4) in the Multimodal Analysis section
   - Or run: `python -m vaila.mpangles`

2. **Select Angle Format:**
   - Choose **"Yes"** for 0-360° format
   - Choose **"No"** for -180° to +180° format

3. **Choose Processing Type:**
   - **"Yes"**: Process video with visualization
     - Select CSV file with coordinates
     - Select corresponding video file
   - **"No"**: Process CSV files only
     - Select directory containing CSV files

4. **Wait for Processing:**
   - Progress is shown in the terminal
   - Output files are saved automatically

### Method 2: Command Line

```bash
# Activate vailá environment
conda activate vaila

# Run the module
python -m vaila.mpangles
```

### Method 3: Programmatic Usage

```python
from vaila import mpangles

# Process CSV files in a directory
mpangles.run_mp_angles()
```

## ⚙️ Configuration Parameters

### Angle Format Selection

- **0-360° Format**: Absolute angles range from 0° to 360°
  - Useful for continuous angle tracking
  - No negative values

- **-180° to +180° Format**: Absolute angles range from -180° to +180°
  - Useful for biomechanical analysis
  - Negative values indicate direction

### Processing Options

1. **CSV Processing Only:**
   - Faster processing
   - No video output
   - Batch processing of multiple CSV files

2. **Video Processing:**
   - Slower but provides visualization
   - Requires both CSV and video files
   - Single video processing at a time

## 📐 Angle Definitions

### Relative Angles (Joint Angles)

- **Elbow Angle**: Angle between upper arm and forearm segments
  - 180° = fully extended
  - 0° = fully flexed

- **Shoulder Angle**: Angle between trunk and upper arm
  - Measured relative to trunk orientation

- **Hip Angle**: Angle between trunk and thigh
  - Measured relative to trunk orientation

- **Knee Angle**: Angle between thigh and shank
  - 180° = fully extended
  - 0° = fully flexed

- **Ankle Angle**: Angle between shank and foot
  - Measured using heel and foot_index landmarks

- **Wrist Angle**: Angle between forearm and hand
  - Measured using pinky and index finger landmarks

### Absolute Angles (Segment Angles)

- **Upper Arm**: Angle of upper arm segment relative to horizontal
- **Forearm**: Angle of forearm segment relative to horizontal
- **Hand**: Angle of hand segment relative to horizontal
- **Thigh**: Angle of thigh segment relative to horizontal
- **Shank**: Angle of shank segment relative to horizontal
- **Foot**: Angle of foot segment relative to horizontal
- **Trunk**: Angle of trunk segment relative to horizontal
- **Neck**: Angle of neck segment relative to horizontal

## 🔗 Integration with Other Modules

### Input from Other Modules

- **Markerless 2D Analysis**: Output CSV files can be directly used as input
- **Markerless 3D Analysis**: 2D projections can be used
- **Any MediaPipe-based analysis**: Compatible with standard MediaPipe landmark format

### Output to Other Modules

- **Data Analysis**: Angle CSV files can be imported into analysis tools
- **Visualization**: Video output can be used for presentations
- **Statistical Analysis**: CSV files compatible with pandas, numpy, etc.

## ⚠️ Requirements

### Python Dependencies

- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `opencv-python` (cv2) - Video processing and visualization
- `tkinter` - GUI dialogs (usually included with Python)
- `rich` - Enhanced terminal output

### System Requirements

- Python 3.12.12 or compatible
- Sufficient disk space for output files
- For video processing: FFmpeg (for video codec support)

## 🐛 Troubleshooting

### Common Issues

1. **"No CSV files found"**
   - **Solution**: Ensure CSV files are in the selected directory
   - **Check**: File extensions must be `.csv` (lowercase)

2. **"Error reading CSV file"**
   - **Solution**: Verify CSV format matches MediaPipe 33-landmark structure
   - **Check**: First column should be frame index, followed by 66 coordinate columns

3. **"Video and CSV frame count mismatch"**
   - **Solution**: Ensure video and CSV have the same number of frames
   - **Check**: Use corresponding video and CSV files from the same analysis

4. **"Landmark not found"**
   - **Solution**: Verify CSV contains all 33 MediaPipe landmarks
   - **Check**: Missing landmarks may cause calculation errors

5. **"Permission denied" when saving**
   - **Solution**: Check write permissions for output directory
   - **Check**: Ensure sufficient disk space

### Performance Tips

- **For large datasets**: Process CSV files only (faster than video processing)
- **For visualization**: Use video processing only when needed
- **Batch processing**: Process multiple CSV files in one directory

## 📚 Example Workflow

### Complete Analysis Workflow

1. **Record Video**: Capture movement with camera
2. **Run Markerless 2D Analysis**: Process video to get landmark coordinates
   - Use `markerless_2d_analysis.py` or `markerless2d_analysis_v2.py`
   - Output: CSV files with landmark coordinates
3. **Run MP Angles**: Calculate joint and segment angles
   - Input: CSV files from step 2
   - Output: Angle CSV files and optional annotated video
4. **Analyze Results**: Import angle CSV files into analysis software

### Example: Gait Analysis

1. Record walking video
2. Extract landmarks using Markerless 2D Analysis
3. Calculate angles using MP Angles
4. Analyze knee and hip angles during gait cycle
5. Visualize results using annotated video

## 📝 Notes

- All angles are calculated in degrees
- Missing landmarks are handled gracefully (NaN values)
- Progress is displayed every 30 frames during processing
- Output files use timestamps to avoid overwriting

## 🔗 Related Documentation

- **[MP Angles Button Documentation](../../docs/vaila_buttons/mp-angles-calculation.md)** - GUI button documentation
- **[Markerless 2D Analysis Help](markerless_2d_analysis.md)** - Input data source
- **[vailá Main Documentation](../../docs/index.md)** - Complete toolbox documentation

## 📄 License

This program is licensed under the **GNU Affero General Public License v3.0 (AGPL v3.0)**.

---

**Last Updated:** November 2025  
**Part of vailá - Multimodal Toolbox**
