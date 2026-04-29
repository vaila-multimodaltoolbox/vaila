# User Guide - Pixel Coordinate Tool (getpixelvideo.py)

## Introduction

The Pixel Coordinate Tool (`getpixelvideo.py`) is a comprehensive video annotation tool that allows you to mark and save pixel coordinates in video frames. Developed by Prof. Dr. Paulo R. P. Santiago, this tool offers advanced features including zoom for precise annotations, dynamic window resizing, frame navigation, multi-format CSV support, and advanced data visualization capabilities.

**Version:** 0.6.0  
**Date:** April 2026  
**Authors:** Prof. Dr. Paulo R. P. Santiago, Rafael L. M. Monteiro
**Project:** vailá - Multimodal Toolbox

## Key Features

- **Multi-format Support:** Load and visualize MediaPipe, YOLO tracking, and vailá standard formats
- **Advanced Visualization:** Stick figures for MediaPipe, bounding boxes for YOLO tracking
- **Flexible Marking:** Multiple marker modes for different annotation needs
- **Labeling Mode:** Create bounding box annotations for Machine Learning datasets
- **Dataset Export:** Export structured datasets (train/val/test) with images and JSON annotations
- **YOLO-pose dataset (F9):** Export clicked markers as an Ultralytics pose dataset (`data.yaml` with `kpt_shape`, train/val/test splits); append across videos with F7 + F8
- **Save ML (button / Ctrl+E):** Export a PNG pose dataset with user-selected `train/val/test` split and create `all_labels/` with split-prefixed label copies for didactic review
- **FIFA Labeling Mode (NEW v0.6.0):** Dedicated workflow to grow the FIFA dataset at `/home/preto/data/FIFA/dataset_vaila_fifa` from new videos. Uses the FIFA 32-kp pitch reference (`idx 0 = top_left_corner`) and writes images + labels in the FIFA `images/{train,val,test}` layout.
- **FIFA KP button (NEW):** Configure fixed CSV matrix export in FIFA mode (fixed `N`, start index skip, header base `0/1`) while keeping unmarked keypoints empty.
- **Zoom & Navigation:** Full zoom capabilities with frame-by-frame navigation
- **Persistence Mode:** View marker trails across multiple frames
- **Auto-detection:** Automatically detect CSV format or manual selection
- **Comprehensive Documentation:** HTML help file with detailed instructions

## Requirements

- **Python 3.12+**
- **OpenCV (cv2):** Video processing
- **Pygame:** GUI and visualization
- **Pandas:** CSV data handling
- **NumPy:** Numerical operations
- **Tkinter:** File dialogs (usually included with Python)

### Installation

From the vailá repository (recommended):

```bash
uv sync
uv run vaila/getpixelvideo.py
```

Standalone dependencies:

```bash
pip install opencv-python pygame pandas numpy
```

## Getting Started

1. **Run the script:** `uv run vaila/getpixelvideo.py` (or `python vaila/getpixelvideo.py` if your environment is already set up)
2. **Select video file:** Choose the video to process
3. **Load existing data:** Use 'Load' button in interface (optional)
4. **Select format:** If loading data, choose CSV format:
   - **Auto-detect (recommended):** Automatically detects the format
   - **MediaPipe format:** For landmark data with stick figure visualization
   - **YOLO tracking format:** For tracking data with bounding box visualization
   - **vailá standard format:** For standard coordinate data
5. **Navigate & annotate:** Use the interface to navigate, zoom, and edit markers
6. **Save results:** Save the annotated data in CSV format

## Interface

The tool interface consists of:

- **Video viewing area** (upper section) with zoom and pan capabilities
- **Control panel** (bottom section) with:
  - Current frame information
  - Slider for navigating between frames
  - Buttons for main functions (Load, Save, Help, 1 Line, Persistence, Sequential)
  - Format-specific visualization controls

## Supported File Formats

### MediaPipe Format

Used for pose estimation and landmark detection data.

**Format:** `frame, landmark_0_x, landmark_0_y, landmark_0_z, landmark_1_x, landmark_1_y, landmark_1_z, ...`

**Example:**

```csv
frame,landmark_0_x,landmark_0_y,landmark_0_z,landmark_1_x,landmark_1_y,landmark_1_z
0,100.5,200.3,0.0,150.2,250.1,0.0
1,105.2,205.1,0.0,155.3,255.2,0.0
```

**Visualization:** Stick figures with landmarks connected by lines

### YOLO Tracking Format

Used for object tracking and detection data.

**Format:** `Frame, Tracker ID, Label, X_min, Y_min, X_max, Y_max, Confidence, Color_R, Color_G, Color_B`

**Example:**

```csv
Frame,Tracker ID,Label,X_min,Y_min,X_max,Y_max,Confidence,Color_R,Color_G,Color_B
0,1,person,100,200,200,300,0.9,255,0,0
0,2,person,300,400,400,500,0.8,0,255,0
```

**Visualization:** Bounding boxes with labels and tracker IDs

### vailá Standard Format

Standard coordinate format used by vailá toolbox.

**Format:** `frame, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, ...`

**Example:**

```csv
frame,p1_x,p1_y,p2_x,p2_y
0,100.5,200.3,150.2,250.1
1,105.2,205.1,155.3,255.2
```

**Visualization:** Point markers with IDs

## Marking Modes

### Normal Mode (default)

- Each click selects and updates the currently selected marker
- Navigate between markers using TAB
- Each marker maintains its ID across all frames
- **Use case:** Tracking specific points across frames
- **Activation:** Default mode at startup

### 1 Line Mode (C key)

- Creates points in sequence in one frame
- Each click adds a new marker in sequential order
- **Use case:** Tracing contours, paths, or outlines
- **Activation:** Press C key to toggle
- **Behavior:** Each click creates a new sequential marker

### Sequential Mode (S key)

- Each click creates a new marker with incrementing IDs
- No need to select markers first
- Only available in Normal mode
- **Use case:** Quick annotation of multiple points
- **Activation:** Press S key to toggle
- **Behavior:** Automatic ID increment for each new marker

### Labeling Mode (L key)

- Draws bounding boxes on video frames
- Creates datasets for object detection training (YOLO/COCO format)
- **Use case:** Creating Machine Learning datasets
- **Activation:** Press L key or click "Labeling" button
- **Behavior:**
  - Click and drag to draw bounding boxes
  - Press Z or Right Click to remove last box in current frame
  - Press N to rename the current object label
  - F5: Save project & export dataset (creates `dataset_YYYYMMDD_HHMMSS/` or appends if dataset loaded via F7)
  - F6: Load labeling project (JSON)
  - F7: Load dataset folder (next F5 appends; multi-video)
  - F8: Open another video without closing the app (keeps dataset)
  - Save button or F5 exports structured dataset (train/val/test)
- **Export:** Generates folder structure with images and JSON annotations

### Pose dataset export (YOLO-pose) — **F9**

Use this when you want a **keypoint / pose** training set (e.g. football pitch with 32 points), not only bounding-box detection.

- **When:** Any time you have markers on at least one frame (normal marker workflow; Labeling mode does **not** need to be on).
- **F9:** Writes `pose_dataset_YYYYMMDD_HHMMSS/` next to the video with `train|val|test/{images,labels}/`, `classes.txt`, and `data.yaml` including `kpt_shape: [Nkp, 3]` and `flip_idx`.
- **Multi-video / append:** Press **F7** (Dataset) and select an existing dataset’s `data.yaml`, then **F8** to open the next video from another folder; annotate and press **F9** again — new frames are prefixed with `<video_stem>_` so nothing collides. If the existing `data.yaml` already defines `kpt_shape` (e.g. 32), that **Nkp** is preserved.
- **CLI:** `uv run vaila/getpixelvideo.py -f VIDEO.mp4 --dataset /path/to/existing_pose_dataset` sets the append target before launch (same as F7).
- **Class name:** The pose instance uses the same class string as bbox labeling (`current_label`; rename with **N** while in Labeling mode), default `object`.
- **BBox per frame:** If the frame has at least one bbox drawn in Labeling mode, the **first** box is used; otherwise a tight box around visible keypoints (with padding) is computed.
- **Project JSON:** `<video_stem>_pose_project.json` in the dataset folder stores raw marker coordinates for later editing.

### FIFA Labeling Mode — Expand `/home/preto/data/FIFA/dataset_vaila_fifa`

A guided workflow to **append new videos to the existing FIFA dataset** built by `vaila.fifa_dataset_builder` and used by the YOLO pitch keypoints retrain (`docs/fifa_workflow.md` §4.5).

**What you get:**

- The on-screen reference shows the **FIFA 32-keypoint pitch** with **`idx 0 = top_left_corner`** through `idx 31 = center_circle_right`. Numbers match `unified/keypoints_reference_drawsportsfields.csv`.
- The Pitch Guide walks you through every keypoint in order: click each one, press `A` to skip an unseen point, `B`/`Backspace` to step back.
- F9 exports in the **FIFA dataset layout** so files drop straight into the `unified/` tree:

  ```
  <dataset_dir>/
      data.yaml         # path: <abs>, train: images/train, val: images/val, test: images/test
                        # kpt_shape: [32, 3]
                        # flip_idx: [24, 25, ..., 31, 30]
                        # names: { 0: football_pitch }
      images/{train,val,test}/<videoname>_frame_NNNNNN.jpg  # F9 quick export
      images/{train,val,test}/<videoname>_frame_NNNNNN.png  # Save ML export
      labels/{train,val,test}/<videoname>_frame_NNNNNN.txt
  ```

- The class is automatically set to `football_pitch` (no need to rename with `N`).
- `flip_idx` is loaded from `vaila/models/soccerfield_ref3d_fifa_dataset.csv`; when appending, the existing `flip_idx` from `data.yaml` is reused.

**Activation — three equivalent ways:**

1. **CLI (one-shot, recommended):**

   ```bash
   uv run vaila/getpixelvideo.py \
       -f /path/to/new_video.mp4 \
       --fifa-dataset /home/preto/data/FIFA/dataset_vaila_fifa/unified
   ```

   Equivalent short alias `--fifa` enables FIFA Mode without an existing dataset (F9 will then create a fresh `pose_dataset_YYYYMMDD_HHMMSS/` in the FIFA layout).

2. **GUI button:** click **`FIFA`** in the bottom button cluster (turns blue). It loads the 32-kp guide, auto-enables PitchGuide and sets the class to `football_pitch`. Click **`Dataset`** afterwards to point at `unified/` for append.

3. **Mid-session:** press **`G`** for PitchGuide (legacy 37-kp scheme) **or** the **`FIFA`** button (32-kp scheme).

**New: `FIFA KP` button (inside FIFA mode)**

- **Left click:** cycle fixed keypoint matrix size `N` (`8, 16, 24, 32, 40, 50, 64`).
- **Right click:** toggle CSV header base between `0` and `1`.
- **Middle click:** increment `start` (skip initial keypoints in output header/matrix).
- Works only when **FIFA mode is ON**.
- Save (`S`/Save button) writes a fixed matrix with empty fields for unmarked points.

**Workflow (single new video):**

1. Run the CLI command above (or open the GUI then click **FIFA** + **Dataset**).
2. The FIFA reference panel shows on the right of the video, with the next keypoint highlighted in yellow.
3. Click the corresponding pixel in the video. The point is stored at the FIFA index (idx 0 = top_left_corner, idx 1 = left_pen_box_top_outer, …).
4. Skip occluded points with `A`, step back with `B`/`Backspace`, delete a placed point with right-click or `DELETE`.
5. Navigate frames (`Space`, `←/→`, slider) and repeat for as many frames as you want. Tip: pick frames with **clear pitch geometry** for best label quality.
6. Press **F9**. New images and labels are appended to `unified/{images,labels}/{train,val,test}/` with filename prefix `<video_stem>_`.

**Workflow (multiple new videos in a batch):**

```bash
for v in /path/to/new_videos/*.mp4; do
    uv run vaila/getpixelvideo.py -f "$v" \
        --fifa-dataset /home/preto/data/FIFA/dataset_vaila_fifa/unified
done
```

Or, in the same session: after F9, press **F8** to open another video while keeping the dataset target.

**Train YOLO with the augmented dataset:**

```bash
uv run yolo pose train \
    data=/home/preto/data/FIFA/dataset_vaila_fifa/unified/data.yaml \
    model=yolo11x-pose.pt imgsz=1280 mosaic=0 erasing=0 epochs=80
```

**Notes / safeguards:**

- The reference CSV `vaila/models/soccerfield_ref3d_fifa_dataset.csv` ships with the repo; missing it disables FIFA Mode (a status message is shown).
- Both layouts (`<dir>/{split}/{images,labels}` and `<dir>/{images,labels}/{split}`) are auto-detected when appending — vailá won't create a wrong subdirectory inside an existing FIFA dataset.
- The legacy 37-kp guide (`G` key) is still available for non-dataset projects.

## Keyboard Commands

### Video Navigation

| Key             | Action                       |
| --------------- | ---------------------------- |
| **Space**       | Play/Pause                   |
| **→**           | Next frame (when paused)     |
| **←**           | Previous frame (when paused) |
| **↑**           | Fast forward (when paused)   |
| **↓**           | Rewind (when paused)         |
| **Drag Slider** | Jump to specific frame       |

### Zoom & Pan

| Key              | Action          |
| ---------------- | --------------- |
| **+**            | Zoom in         |
| **-**            | Zoom out        |
| **Mouse Wheel**  | Zoom in/out     |
| **Middle Click** | Enable pan/move |

### Marker Management

| Key             | Action                           |
| --------------- | -------------------------------- |
| **Left Click**  | Add/update marker                |
| **Right Click** | Remove last marker               |
| **TAB**         | Next marker in current frame     |
| **SHIFT+TAB**   | Previous marker in current frame |
| **DELETE**      | Delete selected marker           |
| **A**           | Add new empty marker to file     |
| **R** or **D**  | Remove last marker from file     |

### Mode Controls

| Key             | Action                                            |
| --------------- | ------------------------------------------------- |
| **C**           | Toggle "1 Line" mode                              |
| **O** or **S**  | Toggle Sequential mode (Normal mode only)        |
| **P**           | Toggle Persistence mode                           |
| **L**           | Toggle Labeling mode (Bounding Boxes)             |
| **G**           | Toggle PitchGuide (legacy 37-kp soccer field)     |
| **FIFA button** | Toggle FIFA Labeling Mode (32 kp, idx 0 = top_left_corner) |
| **FIFA KP button (L/R/M click)** | FIFA CSV matrix setup: left=`N`, right=`base 0/1`, middle=`start` skip |
| **Z / R-Click** | Remove last bounding box (Labeling mode)          |
| **N**           | Rename object label (Labeling Mode Only)          |
| **F5**          | Save Labeling Project & export dataset (Labeling Mode Only) |
| **F6**          | Load Labeling Project (JSON) (Labeling Mode Only) |
| **F7**          | Load dataset folder – next Save appends (Labeling Mode Only) |
| **F8**          | Open another video (keeps dataset; no need to close app) |
| **F9**          | Export YOLO-pose dataset from clicked markers (see Pose dataset section) |
| **Ctrl+E**      | Save ML: choose split, export PNG dataset + `all_labels/` didactic view |
| **1**           | Decrease persistence frames                       |
| **2**           | Increase persistence frames                      |
| **3**           | Toggle full persistence                           |

### File Operations (buttons and keys)

| Key   | Action                                            |
| ----- | ------------------------------------------------- |
| **H** | Show help dialog                                  |
| **R** or **D** | Remove last marker from file                 |
| **Save/Load**  | Use bottom panel buttons (no dedicated key)  |

### Other

| Key     | Action        |
| ------- | ------------- |
| **ESC** | Save and exit |

## Data Visualization

### MediaPipe Visualization

When loading MediaPipe data, the tool displays stick figures:

- **Landmarks:** Red dots for each detected point
- **Connections:** Green lines connecting related landmarks
- **Pose Structure:** Full body pose with head, arms, torso, and legs

### YOLO Tracking Visualization

When loading YOLO tracking data, the tool displays bounding boxes:

- **Bounding Boxes:** Colored rectangles around detected objects
- **Labels:** Object class and tracker ID displayed
- **Colors:** Unique colors for different tracker IDs
- **Confidence:** Detection confidence values shown

### vailá Standard Visualization

Standard point marker visualization:

- **Markers:** Green circles for each point
- **Numbers:** Marker IDs displayed next to points
- **Selection:** Orange highlight for selected marker

## Persistence Mode

Persistence mode shows markers from previous frames, creating a visual "trail":

- **P:** Enables/disables persistence
- **1:** Decreases the number of displayed frames
- **2:** Increases the number of displayed frames
- **3:** Toggles between persistence modes (disabled → full → limited)

**Features:**

- Fading trails show marker movement
- Configurable number of frames to display
- Visual feedback for marker trajectories

## Saving and Loading

### Saving Options

#### Standard Save (S key)

- **Format:** `frame, p1_x, p1_y, p2_x, p2_y, ...`
- **File:** `{video_name}_markers.csv`
- **Location:** Same directory as video file

In **FIFA mode** with **FIFA KP** configured, Save uses a fixed matrix:

- Header follows configured `N`, `start`, and `base` (e.g., `frame,p0_x,p0_y,...` or `frame,p1_x,p1_y,...`).
- You can skip initial keypoints (e.g., start at 5).
- Unmarked keypoints remain empty (`""`) in CSV.

**Example (FIFA mode, `N=8`, `start=5`, `base=0`):**

```csv
frame,p5_x,p5_y,p6_x,p6_y,p7_x,p7_y,p8_x,p8_y,p9_x,p9_y,p10_x,p10_y,p11_x,p11_y,p12_x,p12_y
120,,,,1020.4,410.1,1005.8,398.7,,,,,,,,,
121,,,,1018.6,409.2,1003.5,397.2,,,,,,,,,
```

**Example (same frame idea, `base=1`):**

```csv
frame,p6_x,p6_y,p7_x,p7_y,p8_x,p8_y,p9_x,p9_y,p10_x,p10_y,p11_x,p11_y,p12_x,p12_y,p13_x,p13_y
120,,,,1020.4,410.1,1005.8,398.7,,,,,,,,,
```

#### 1 Line Save

- **Format:** `frame, p1_x, p1_y, p2_x, p2_y, ...`
- **File:** `{video_name}_markers_sequential.csv`
- **Use:** For path tracing and contour data

#### Sequential Save

- **Format:** `frame, p1_x, p1_y, p2_x, p2_y, ...`
- **File:** `{video_name}_markers_sequential.csv`
- **Use:** For multiple point annotations

#### Labeling Save (Bounding Box Mode)

- **Format:** Structured dataset with images and JSON annotations
- **Directory:** New dataset: `dataset_YYYYMMDD_HHMMSS/` (same folder as video). To reuse one dataset for multiple videos: F7 → select existing `data.yaml` → next F5 appends.
- **Structure:** train/val/test with images/ and labels/
- **Use:** For creating Machine Learning datasets (object detection)
- **Activation:** Save when Labeling mode is active (F5). Use F8 to open another video without closing the app.

#### Pose dataset save (YOLO-pose) — **F9**

- **Format:** Ultralytics YOLO-pose labels (`.txt` per image: `cls cx cy w h` + `kp_x kp_y v` × Nkp)
- **Directory:** New: `pose_dataset_YYYYMMDD_HHMMSS/` beside the video; append: set target with **F7** or `--dataset` before **F9**
- **Structure:** Same split layout as detection export (`train/val/test`, `images/`, `labels/`), plus `data.yaml` with `kpt_shape` and `names`
- **Use:** Retraining pose networks (e.g. soccer field keypoints)
- **Activation:** Press **F9** when at least one frame has markers

#### Save ML dataset (PNG split + all_labels) — **Button or Ctrl+E**

- Runs pose export with `train/val/test` split (same base flow as F9) and saves frame images as `.png`.
- Before export, choose one preset or type custom percentages:
  - **1 / Clássica:** `70% train`, `15% val`, `15% test`
  - **2 / Foco em treino:** `80% train`, `10% val`, `10% test`
  - **3 / Big Data:** `98% train`, `1% val`, `1% test`
  - **Custom:** type `train,val,test`, for example `75,15,10`.
- Also writes `all_labels/` inside the dataset with split-prefixed `.txt` copies (`train_...`, `val_...`, `test_...`) for quick didactic inspection.
- Works in normal and FIFA workflows.

### Loading Coordinates

- Click the **Load** button at any time
- **Auto-detection:** Automatically detects CSV format
- **Manual selection:** Choose format manually if needed

## Advanced Features

### Auto-detection

The tool automatically detects CSV format based on column structure:

- **MediaPipe:** Detects 'landmark' in column names
- **YOLO:** Detects 'Frame', 'Tracker ID', 'X_min' columns
- **vailá:** Detects 'frame' and 'p' pattern in columns

### Backup and Recovery

Built-in backup system for data safety:

- **Backup:** Press B to create backup
- **Reload:** Press L to reload from file
- **Auto-backup:** Automatic backups before major operations
- **Hidden history:** Backups are stored under `.vaila_markers_history/<video_stem>/`
- **Retention policy:** Keeps the latest 100 backups per video

### Documentation Access

- **Quick Help:** Press H for in-app help
- **Full Documentation:** Press D in help dialog to open HTML documentation
- **HTML Documentation:** Located at `vaila/help/getpixelvideo.html`

## Tips and Best Practices

1. **Use Sequential mode** when you want to create multiple markers without worrying about selection
2. **Use 1 Line mode** to trace contours or paths in a single frame
3. **Automatic backups** are created with timestamps to prevent data loss
4. **Use zoom** for greater precision when marking coordinates
5. **The A key** is useful for adding empty markers that can be filled in later
6. **Persistence mode** is great for visualizing movement patterns
7. **Auto-detection** works best with properly formatted CSV files
8. **Backup regularly** using the B key to prevent data loss

## Troubleshooting

### Common Issues

#### Video Not Loading

- Check if video file is corrupted
- Ensure video format is supported (MP4, AVI, MOV, MKV)
- Verify file path doesn't contain special characters

#### CSV Format Not Detected

- Check CSV file structure matches expected format
- Use manual format selection if auto-detection fails
- Verify CSV file is not corrupted

#### Performance Issues

- Reduce video resolution for better performance
- Close other applications to free memory
- Use smaller zoom levels for large videos

#### Visualization Problems

- Ensure CSV data is in the correct format
- Check if coordinate values are within video dimensions
- Verify landmark connections for MediaPipe data

### Error Messages

| Error                       | Solution                               |
| --------------------------- | -------------------------------------- |
| "Error opening video file"  | Check video format and file integrity  |
| "No keypoint file selected" | Select a valid CSV file or start fresh |
| "Unknown format"            | Use manual format selection            |
| "Error loading coordinates" | Check CSV file format and structure    |

## Support and Documentation

- **In-app Help:** Press H for quick help
- **Full Documentation:** Press D in help dialog for complete HTML documentation
- **HTML Documentation:** `vaila/help/getpixelvideo.html`
- **Project Repository:** https://github.com/paulopreto/vaila-multimodaltoolbox

## Version History

### Version 0.6.0 (April 2026)

- **FIFA Labeling Mode**: dedicated workflow to append new videos to `/home/preto/data/FIFA/dataset_vaila_fifa`
- New CLI flags `--fifa-dataset DIR` and `--fifa`
- New **`FIFA`** GUI button (bottom button cluster)
- New reference CSV `vaila/models/soccerfield_ref3d_fifa_dataset.csv` (32 kp, `idx 0 = top_left_corner`, with `flip_idx`)
- `export_pose_dataset()` now auto-detects the FIFA dataset layout (`<dir>/{images,labels}/{split}/`) and writes `data.yaml` with FIFA `flip_idx` and `names: { 0: football_pitch }`
- F7 (`Dataset`) accepts both layouts: `<dir>/train/images/` (vailá) **and** `<dir>/images/train/` (FIFA)
- New on-screen FIFA pitch reference (32-kp variant) drawn on top of the existing legacy guide

### Version 0.5.0 (April 2026)

- **F9 — YOLO-pose dataset export** from clicked markers (train/val/test, `data.yaml` with `kpt_shape` and `flip_idx`)
- Multi-video append via existing **F7** dataset folder + **F8** new video; optional **`--dataset`** on CLI
- Writes `<video_stem>_pose_project.json` alongside exports for reproducibility
- In-app help (H) updated with Pose dataset + F9

### Version 0.3.0 (January 2026)

- Added Labeling Mode (Bounding Boxes) for creating Machine Learning datasets
- Implemented structured dataset export (train/val/test)
- Dataset naming: new exports use `dataset_YYYYMMDD_HHMMSS/`; load existing via F7 to append (multi-video)
- Added F8: Open another video without closing the app (keeps dataset; auto-loads project when present)
- Added "Labeling" button to interface
- Enhanced help dialog with detailed labeling mode instructions
- **Improvements:**
  - F5/F6: Save/Load Labeling Project (JSON). F7: Load dataset folder (next Save appends). F8: Open another video.
  - Added ability to Rename object labels (N key)
  - Added Scrolling to Help Dialog
  - Improved Linux compatibility (pygame file browser for Dataset/Open video; no Tkinter for dialogs on Linux)

### Version 0.0.8 (27 July 2025)

- Added support for multiple CSV formats (MediaPipe, YOLO tracking, vailá standard)
- Implemented auto-detection of CSV format
- Added visualization for MediaPipe stick figures and YOLO bounding boxes
- Created comprehensive HTML documentation
- Added quick access to full documentation via 'D' key
- Improved help dialog with format information
- Enhanced error handling and user feedback

### Previous Versions

- Version 0.0.7: Basic functionality with zoom and marker modes
- Version 0.0.6: Initial implementation with video navigation
