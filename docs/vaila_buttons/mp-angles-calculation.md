# MP Angles - Button B4_r4_c4

## Overview

The **MP Angles** button (B4_r4_c4) in the vailá GUI provides access to the MediaPipe Angle Calculation module. This tool calculates absolute and relative joint angles from MediaPipe pose estimation landmark coordinates, enabling comprehensive biomechanical analysis of human movement.

**Button Position:** B4_r4_c4 (Button 4, Row 4, Column 4)  
**Method Name:** `mp_angles_calculation`  
**Button Text:** MP Angles  
**GUI Category:** Multimodal Analysis → Row 4

## Description

The MP Angles module processes CSV files containing MediaPipe landmark coordinates (33 landmarks) and calculates:

1. **Relative Angles (Joint Angles)**: Angles between connected body segments
   - Neck, Trunk, Shoulder, Elbow, Wrist, Hip, Knee, Ankle (left and right)

2. **Absolute Angles (Segment Angles)**: Angles of body segments relative to horizontal
   - Upper arm, Forearm, Hand, Thigh, Shank, Foot (left and right)
   - Trunk and Neck

3. **Video Visualization** (optional): Creates annotated videos with skeleton overlay and real-time angle display

## Usage Workflow

### Step 1: Launch the Module

1. Click the **"MP Angles"** button in the vailá GUI
2. The module will start and display dialogs

### Step 2: Select Angle Format

A dialog will ask: **"Choose the angle format for absolute angles"**

- **Click "Yes"**: Use 0-360° format (no negative values)
- **Click "No"**: Use -180° to +180° format (biomechanical standard)

**Recommendation:** Use -180° to +180° for biomechanical analysis

### Step 3: Choose Processing Type

A dialog will ask: **"Do you want to process a video file?"**

#### Option A: Process Video with Visualization (Yes)

1. **Select CSV File**: Choose the CSV file containing MediaPipe landmark coordinates
   - File must be from Markerless 2D Analysis or compatible format
   - Format: 33 landmarks × 2 coordinates = 66 columns + frame index

2. **Select Video File**: Choose the corresponding video file
   - Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`
   - Video must have the same number of frames as the CSV

3. **Processing**: The module will:
   - Calculate all angles for each frame
   - Create an annotated video with skeleton overlay
   - Display angles in real-time on the video
   - Save angle CSV files

4. **Output Location**: 
   - Directory: `angles_video_YYYYMMDD_HHMMSS/` (in video directory)
   - Files:
     - `angles_{videoname}.mp4` - Annotated video
     - `{videoname}_rel.csv` - Relative angles
     - `{videoname}_abs.csv` - Absolute angles

#### Option B: Process CSV Files Only (No)

1. **Select Directory**: Choose directory containing CSV files
   - All `.csv` files in the directory will be processed
   - Files must follow MediaPipe 33-landmark format

2. **Processing**: The module will:
   - Process each CSV file in the directory
   - Calculate all angles for each frame
   - Save angle CSV files

3. **Output Location**:
   - Directory: `processed_angles_YYYYMMDD_HHMMSS/` (in input directory)
   - Files (per input CSV):
     - `processed_{filename}_rel.csv` - Relative angles
     - `processed_{filename}_abs.csv` - Absolute angles

### Step 4: Review Results

- Check the output directory for generated files
- Open CSV files in spreadsheet software or analysis tools
- For video processing: Play the annotated video to visualize angles

## Input Requirements

### CSV File Format

The input CSV files must contain MediaPipe landmark coordinates:

- **Structure**: 
  - Column 1: Frame index (0, 1, 2, ...)
  - Columns 2-67: Landmark coordinates (p0_x, p0_y, p1_x, p1_y, ..., p32_x, p32_y)
  - Total: 67 columns

- **Landmark Order**: Must follow MediaPipe's 33-landmark format:
  - 0: nose
  - 1-10: face (eyes, ears, mouth)
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

- **Formats**: `.mp4`, `.avi`, `.mov`, `.mkv`
- **Correspondence**: Video must match CSV file (same number of frames)
- **Resolution**: Should match coordinate system in CSV

## Output Files

### Relative Angles CSV (`*_rel.csv`)

Contains joint angles in degrees:

| Column | Description |
|--------|-------------|
| `frame_index` | Frame number |
| `neck` | Neck angle relative to trunk |
| `trunk` | Trunk angle |
| `right_shoulder` | Right shoulder joint angle |
| `right_elbow` | Right elbow joint angle |
| `right_wrist` | Right wrist joint angle |
| `right_hip` | Right hip joint angle |
| `right_knee` | Right knee joint angle |
| `right_ankle` | Right ankle joint angle |
| `left_shoulder` | Left shoulder joint angle |
| `left_elbow` | Left elbow joint angle |
| `left_wrist` | Left wrist joint angle |
| `left_hip` | Left hip joint angle |
| `left_knee` | Left knee joint angle |
| `left_ankle` | Left ankle joint angle |

### Absolute Angles CSV (`*_abs.csv`)

Contains segment angles in degrees (format depends on user selection):

| Column | Description |
|--------|-------------|
| `frame_index` | Frame number |
| `neck_abs` | Neck segment angle |
| `trunk_abs` | Trunk segment angle |
| `right_upperarm_abs` | Right upper arm angle |
| `right_forearm_abs` | Right forearm angle |
| `right_hand_abs` | Right hand angle |
| `right_thigh_abs` | Right thigh angle |
| `right_shank_abs` | Right shank angle |
| `right_foot_abs` | Right foot angle |
| `left_upperarm_abs` | Left upper arm angle |
| `left_forearm_abs` | Left forearm angle |
| `left_hand_abs` | Left hand angle |
| `left_thigh_abs` | Left thigh angle |
| `left_shank_abs` | Left shank angle |
| `left_foot_abs` | Left foot angle |

### Annotated Video (if video processing selected)

- **File**: `angles_{videoname}.mp4`
- **Features**:
  - Skeleton overlay (red: right side, blue: left side, white: trunk/neck)
  - Real-time display of relative angles (left side of screen)
  - Real-time display of absolute angles (right side of screen)
  - Color-coded text for easy identification

## Angle Definitions

### Relative Angles (Joint Angles)

- **Elbow Angle**: Angle between upper arm and forearm
  - 180° = fully extended
  - 0° = fully flexed

- **Knee Angle**: Angle between thigh and shank
  - 180° = fully extended
  - 0° = fully flexed

- **Shoulder/Hip Angles**: Measured relative to trunk orientation

- **Ankle Angle**: Measured using heel and foot_index landmarks

### Absolute Angles (Segment Angles)

- Angles of body segments relative to horizontal axis
- Format: 0-360° or -180° to +180° (user selectable)
- Useful for analyzing segment orientation in space

## Integration with Other Modules

### Input Sources

- **Markerless 2D Analysis**: Direct compatibility with output CSV files
- **Markerless 3D Analysis**: 2D projections can be used
- **Any MediaPipe-based analysis**: Compatible with standard format

### Output Usage

- **Data Analysis**: Import CSV files into analysis software (pandas, numpy, MATLAB, etc.)
- **Visualization**: Use annotated videos for presentations
- **Statistical Analysis**: Angle data ready for statistical processing
- **Biomechanical Analysis**: Joint angles for gait analysis, movement studies, etc.

## Configuration Options

### Angle Format

- **0-360° Format**: 
  - All angles positive
  - Continuous angle tracking
  - Useful for some analysis types

- **-180° to +180° Format** (Recommended):
  - Standard biomechanical format
  - Negative values indicate direction
  - Easier interpretation for movement analysis

### Processing Mode

- **CSV Only**: 
  - Faster processing
  - Batch processing of multiple files
  - No video output

- **Video + CSV**:
  - Slower but provides visualization
  - Single video at a time
  - Useful for quality control and presentations

## Requirements

### Python Dependencies

- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `opencv-python` - Video processing
- `tkinter` - GUI dialogs
- `rich` - Enhanced terminal output

### System Requirements

- Python 3.12.12 or compatible
- Sufficient disk space for output files
- FFmpeg (for video codec support, if processing videos)

## Troubleshooting

### Common Issues

1. **"No CSV files found"**
   - Ensure CSV files are in the selected directory
   - Check file extensions are `.csv` (lowercase)

2. **"Error reading CSV file"**
   - Verify CSV format matches MediaPipe 33-landmark structure
   - Check that first column is frame index
   - Ensure 66 coordinate columns follow (33 landmarks × 2)

3. **"Video and CSV frame count mismatch"**
   - Use corresponding video and CSV files from the same analysis
   - Ensure same number of frames in both files

4. **"Landmark not found"**
   - Verify CSV contains all 33 MediaPipe landmarks
   - Check landmark order matches MediaPipe format

5. **"Permission denied" when saving**
   - Check write permissions for output directory
   - Ensure sufficient disk space

### Performance Tips

- Use CSV-only processing for large datasets (faster)
- Process videos only when visualization is needed
- Batch process multiple CSV files in one directory

## Example Use Cases

### Gait Analysis

1. Record walking video
2. Extract landmarks using Markerless 2D Analysis
3. Calculate angles using MP Angles
4. Analyze knee and hip angles during gait cycle
5. Visualize results using annotated video

### Sports Performance

1. Record sports movement (jump, throw, etc.)
2. Extract landmarks
3. Calculate joint angles
4. Analyze movement patterns
5. Compare with normative data

### Rehabilitation

1. Record patient movement
2. Extract landmarks
3. Calculate angles
4. Track progress over time
5. Generate reports with annotated videos

## Related Documentation

- **[Script Help](../../vaila/help/utils/mpangles.md)** - Detailed script documentation
- **[Markerless 2D Analysis Button](markerless-2d-button.md)** - Input data source
- **[vailá Main Documentation](../index.md)** - Complete toolbox documentation

## Notes

- All angles are calculated in degrees
- Missing landmarks are handled gracefully (NaN values)
- Progress is displayed every 30 frames during processing
- Output files use timestamps to avoid overwriting
- Video processing is slower but provides valuable visualization

---

**Last Updated:** November 2025  
**Part of vailá - Multimodal Toolbox**  
**License:** AGPLv3.0
