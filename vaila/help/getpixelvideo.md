# User Guide - Pixel Coordinate Tool (getpixelvideo.py)

## Introduction

The Pixel Coordinate Tool (`getpixelvideo.py`) is a comprehensive video annotation tool that allows you to mark and save pixel coordinates in video frames. Developed by Prof. Dr. Paulo R. P. Santiago, this tool offers advanced features including zoom for precise annotations, dynamic window resizing, frame navigation, multi-format CSV support, and advanced data visualization capabilities.

**Version:** 0.3.46  
**Date:** 28 May 2026  
**Authors:** Prof. Dr. Paulo R. P. Santiago, Rafael L. M. Monteiro
**Project:** vailá - Multimodal Toolbox

## Key Features

- **Template Marker Mode:** Choose fixed keypoint templates in the toolbar:
  - **FIFA Soccer-Field:** 32 pitch keypoints (`idx 0 = top_left_corner`) + TOML config (right-click or `K`)
  - **MediaPipe Pose:** 33 pose landmarks
  - **YOLO Pose:** COCO-17 keypoints
- **Multi-format Support:** Load and visualize MediaPipe, YOLO tracking, vailá standard formats, and markerless 2D named-landmark CSVs (`frame_index,nose_x,nose_y,nose_z,...`)
- **Advanced Visualization:** Stick figures for MediaPipe, bounding boxes for YOLO tracking
- **Flexible Marking:** Multiple marker modes for different annotation needs
- **Del Range:** Button to delete one or more marker/keypoint numbers across an inclusive frame range; use commas (`0,3,7`) and sequential ranges (`1:10`)
- **Swap Range:** Button to swap marker/keypoint pairs over a frame range; first marker line maps pairwise to the second (`26,28` with `27,29`, or `1:10` with `11:20`)
- **Labeling Mode:** Create bounding box annotations for Machine Learning datasets
- **Dataset Export:** Export structured datasets (train/val/test) with images and JSON annotations
- **YOLO-pose dataset (F9):** Export clicked markers as an Ultralytics pose dataset (`data.yaml` with `kpt_shape`, train/val/test splits); append across videos with F7 + F8; may write `keypoints.json` when keypoint names are known
- **Save ML (button / Ctrl+E):** Export a PNG pose dataset with user-selected `train/val/test` split and create `all_labels/` with split-prefixed label copies for didactic review
- **FIFA Labeling Mode:** Configure via **TOML** (FIFA button or `K` key — no separate Tk config dialog). Default **31** FIFA pitch keypoints (`idx 0 = top_left_corner`); fixed `N`, optional `start` skip, header base `0/1`; sparse CSV with **integer** pixels and empty cells for unmarked KPs. Optional `--fifa-dataset DIR` to append into an existing unified / pose tree.
- **Guide (`G`):** **Visual only** overlay for active template:
  - **FIFA:** soccer-field guide + optional reference map (`V`)
  - **MediaPipe:** pose skeleton guide (33)
  - **YOLO:** pose skeleton guide (COCO-17)
  Marking behaviour is the same as with the guide off (TAB, **Ctrl+G** / Go KP, left/right click).
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

Optional one-off run without syncing the whole repo: `uv run --with opencv-python --with pygame --with pandas --with numpy vaila/getpixelvideo.py` (prefer `uv sync` from the vailá clone).

## Getting Started

1. **Run the script:** `uv run vaila/getpixelvideo.py` (or `python vaila/getpixelvideo.py` if your environment is already set up)
2. **Select video file:** Choose the video to process
3. **Load existing data:** Use 'Load' button in interface (optional)
4. **Select format:** If loading data, choose CSV format:
   - **Auto-detect (recommended):** Automatically detects the format
   - **MediaPipe format:** For landmark data with stick figure visualization
   - **YOLO tracking format:** For tracking data with bounding box visualization
   - **vailá standard format:** For standard coordinate data
   - **markerless_2d_analysis named landmarks:** CSVs like `frame_index,nose_x,nose_y,nose_z,...`; x/y columns are loaded in file order and z columns are ignored
5. **Navigate & annotate:** Use the interface to navigate, zoom, and edit markers (**TAB** / **SHIFT+TAB**, **Ctrl+G** to jump to a keypoint index; **Del Range** deletes one marker/keypoint, comma-separated list, or `A:B` range; **Swap Range** swaps paired marker lists/ranges across a frame interval)
6. **Save results:** Save the annotated data in CSV format

## CLI quick reference

```bash
uv run vaila/getpixelvideo.py --help
uv run vaila/getpixelvideo.py -f VIDEO.mp4
uv run vaila/getpixelvideo.py -f /path/to/png_folder      # or single .png
uv run vaila/getpixelvideo.py -d /path/to/png_folder      # same as --sequence; subfolder with PNGs is auto-picked
uv run vaila/getpixelvideo.py --sequence /path/to/frames
uv run vaila/getpixelvideo.py -f VIDEO.mp4 --fifa-dataset /path/to/unified
uv run vaila/getpixelvideo.py -f VIDEO.mp4 --dataset /path/to/vaila_dataset
```

Flags include `--fifa`, `--fifa-dataset [DIR]`, `--dataset DIR` (see module docstring in `vaila/getpixelvideo.py` for the full list).

## Interface

The tool interface consists of:

- **Video viewing area** (upper section) with zoom and pan capabilities
- **Control panel** (bottom section) with:
  - Current frame information
  - Slider for navigating between frames
  - Buttons for main functions (Load, Save, Help, Del Range, Swap Range, 1 Line, Persistence, Sequential)
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
- Use **Del Range** to hide/delete one marker ID, multiple comma-separated IDs, or a sequential `A:B` marker range over an inclusive frame range without touching other markers
- Use **Swap Range** to exchange paired marker IDs over an inclusive frame range. The first marker line maps pairwise to the second: `26,28` with `27,29` swaps `26<->27` and `28<->29`; `1:10` with `11:20` swaps each expanded position.
- **Use case:** Tracking specific points across frames
- **Activation:** Default mode at startup

### 1 Line Mode (C key)

- Creates points in sequence in one frame
- Each click adds a new marker in sequential order
- **Use case:** Tracing contours, paths, or outlines
- **Activation:** Press C key to toggle
- **Behavior:** Each click creates a new sequential marker

### Sequential Mode (S or O key)

- Only in **Normal** mode (not 1 Line / Labeling / etc.)
- **First click** fills the **currently selected** keypoint slot (choose with **TAB** / **SHIFT+TAB** or **Ctrl+G** / Go KP) if that slot is still empty — including **keypoint 0**
- **Next clicks** advance to `selected + 1`, etc.
- **Use case:** Fast sparse FIFA-style labeling while jumping between indices
- **Activation:** Press **S** or **O** to toggle

### Labeling Mode (L key)

- Draws bounding boxes on video frames
- Creates datasets for object detection training (YOLO/COCO format)
- **Use case:** Creating Machine Learning datasets
- **Activation:** Press L key or click "Labeling" button
- **Behavior:**
  - Click and drag to draw bounding boxes
  - Press **Z** to remove the last box in the current frame
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
- **F9:** Writes `vaila_dataset_YYYYMMDD_HHMMSS/` next to the video with `train|val|test/{images,labels}/`, `classes.txt`, `data.yaml` including `kpt_shape: [Nkp, 3]` and `flip_idx`, and also `<video_stem>_markers.csv` (readable vailá format) for easy loading.
- **Multi-video / append:** Press **F7** (Dataset) and select an existing dataset’s `data.yaml`, then **F8** to open the next video from another folder; annotate and press **F9** again — new frames are prefixed with `<video_stem>_` so nothing collides. If the existing `data.yaml` already defines `kpt_shape` (e.g. 32), that **Nkp** is preserved.
- **CLI:** `uv run vaila/getpixelvideo.py -f VIDEO.mp4 --dataset /path/to/existing_vaila_dataset` sets the append target before launch (same as F7).
- **Class name:** The pose instance uses the same class string as bbox labeling (`current_label`; rename with **N** while in Labeling mode), default `object`.
- **BBox per frame:** If the frame has at least one bbox drawn in Labeling mode, the **first** box is used; otherwise a tight box around visible keypoints (with padding) is computed.
- **Project JSON:** `<video_stem>_pose_project.json` in the dataset folder stores raw marker coordinates for later editing.

### FIFA Labeling Mode — append to a unified / FIFA pose tree

Use this when you want **FIFA pitch keypoints** in the sparse CSV workflow and **F9** exports that match the `unified/` layout from `vaila.fifa_dataset_builder` (see `docs/fifa_workflow.md` for the full YOLO retrain story).

**What you get:**

- **Configuration in TOML** next to the video (template fields `n_keypoints`, `start_keypoint`, `base_index`). Press **`K`** or click **`FIFA`** to load or create that file — there is no separate Tk “FIFA config” dialog.
- **Default in vailá:** **31** FIFA field keypoints (`idx 0 = top_left_corner` …). If you append to an older tree whose `data.yaml` still has `kpt_shape: [32, 3]`, that **N** is preserved on export; new standalone FIFA exports use the TOML / default **31** unless you change `n_keypoints`.
- **Guide (`G`)** does **not** drive clicking: it is **overlay** only (FIFA field or pose skeleton). `V` toggles the upper-right guide map. Marking is the same as with the guide off (**TAB**, **Ctrl+G** / Go KP, left click to place on the selected slot).
- **F9** writes into the dataset layout (when the loaded `data.yaml` matches a FIFA-style tree), for example:

  ```
  <dataset_dir>/
      data.yaml         # path: <abs>, train/val/test paths, kpt_shape, flip_idx, names
      images/{train,val,test}/<videoname>_frame_NNNNNN.jpg   # F9 export
      images/{train,val,test}/<videoname>_frame_NNNNNN.png   # Save ML (Ctrl+E)
      labels/{train,val,test}/<videoname>_frame_NNNNNN.txt
  ```

- Class **`football_pitch`** is used for FIFA-style exports (no need to rename with `N` in bbox labeling).
- `flip_idx` comes from `vaila/models/soccerfield_ref3d_fifa_dataset.csv` when applicable; when appending, existing `data.yaml` metadata is reused.

**Activation:**

1. **CLI (recommended for batch):**

   ```bash
   uv run vaila/getpixelvideo.py \
       -f /path/to/new_video.mp4 \
       --fifa-dataset /path/to/your_dataset/unified
   ```

   Use **`--fifa`** to enable FIFA mode without a pre-existing dataset (first **F9** creates a fresh `vaila_dataset_YYYYMMDD_HHMMSS/` in the FIFA layout beside the video).

2. **GUI:** click **`FIFA`** (or **`K`**) and edit/save the TOML when prompted. Optionally use **`F7` (Dataset)** to point at an existing `unified/` folder so the next **F9** appends there.

3. **`G`** toggles **Guide** (visual reference only). It does **not** replace FIFA mode or TOML setup.

**Workflow (single new video):**

1. Launch with `--fifa-dataset …/unified` or enable FIFA in-session and set the dataset with **F7** if needed.
2. Select the keypoint slot with **TAB** / **SHIFT+TAB** or **Ctrl+G** (Go KP). Place points with **left click**; **right click** clears the **selected** slot (same idea as **R**). **D** clears **all** markers on the **current frame**.
3. Navigate frames (`Space`, arrows, bottom slider, **marker timeline strip**, **SHIFT+←/→**). Prefer frames with **clear pitch geometry**.
4. **F9** appends images/labels under `unified/` with prefix `<video_stem>_`. **Ctrl+E** / Save ML runs the PNG split export + `all_labels/`.

**Batch example:**

```bash
for v in /path/to/new_videos/*.mp4; do
    uv run vaila/getpixelvideo.py -f "$v" --fifa-dataset /path/to/your_dataset/unified
done
```

Or stay in one session: after **F9**, press **F8** to open the next video while keeping the dataset target.

**Train YOLO (example):**

```bash
uv run yolo pose train \
    data=/ABS/PATH/to/unified/data.yaml \
    model=yolo11x-pose.pt imgsz=1280 mosaic=0 erasing=0 epochs=80
```

**Notes:**

- Reference data ships as `vaila/models/soccerfield_ref3d_fifa_dataset.csv` (and related files); if pitch geometry cannot load, a status message explains it.
- Both dataset layouts (`<dir>/{split}/{images,labels}` and `<dir>/{images,labels}/{split}`) are auto-detected when appending.
- A small **example** TOML for tests lives under `tests/sport_fields/` (e.g. `fifa_template.toml`); copy the idea beside your own videos.

## Keyboard Commands

### Video Navigation

| Key             | Action                       |
| --------------- | ---------------------------- |
| **Space**       | Play/Pause                   |
| **→**           | Next frame (when paused)     |
| **←**           | Previous frame (when paused) |
| **↑**           | Fast forward (when paused)   |
| **↓**           | Rewind (when paused)         |
| **Drag bottom scrub slider** | Jump to specific frame |
| **SHIFT+→** / **SHIFT+←** | Jump to next/previous frame that has visible markers (wraps) |
| **Marker timeline strip** (above scrub bar) | **Click** or **drag**: jump to that position; **green** = spans containing marked frames; **gold** line = current frame; inside a column with markers, snaps to one of those frames (middle of group), otherwise same proportional mapping as the slider |

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
| **Left Click**  | Add/update marker at **selected** slot |
| **Right Click** | Remove **selected** marker in current frame (then select previous) |
| **TAB**         | Next marker in current frame     |
| **SHIFT+TAB**   | Previous marker in current frame |
| **Ctrl+G**      | Go KP — jump to a keypoint index by number |
| **DELETE**      | Delete selected marker           |
| **A**           | Add new empty marker to file     |
| **R**           | Remove **selected** marker in current frame (same idea as right click) |
| **D**           | Mark **all** keypoint slots in the **current frame** as deleted (bulk clear) |

### Mode Controls

| Key             | Action                                            |
| --------------- | ------------------------------------------------- |
| **C**           | Toggle "1 Line" mode                              |
| **O** or **S**  | Toggle Sequential mode (Normal mode only)        |
| **P**           | Toggle Persistence mode                           |
| **L**           | Toggle Labeling mode (Bounding Boxes)             |
| **G**           | Toggle Guide (**visual** overlay for template; `V` toggles map) |
| **FIFA** / **K** | Load or create **FIFA TOML** beside the video (`n_keypoints`, `start`, `base`) |
| **Z**           | Remove last bounding box (Labeling mode)          |
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
| **?** | Open **HTML** help in the default browser (`vaila/help/getpixelvideo.html`) |
| **Save / Load** | Bottom panel buttons (no dedicated save hotkey) |

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

#### Standard Save (Save button)

- **Format:** `frame, p1_x, p1_y, p2_x, p2_y, ...` (or sparse FIFA header when FIFA mode is on)
- **File:** `{video_name}_markers.csv`
- **Location:** Same directory as video file

In **FIFA mode**, the CSV header follows the **TOML** (`n_keypoints`, `start_keypoint`, `base_index`). Coordinates are written as **integers**; unmarked slots stay empty.

**Example (FIFA, `n_keypoints=8`, `start_keypoint=5`, `base_index=0`):**

```csv
frame,p5_x,p5_y,p6_x,p6_y,p7_x,p7_y,p8_x,p8_y,p9_x,p9_y,p10_x,p10_y,p11_x,p11_y,p12_x,p12_y
120,,,,1020,410,1005,398,,,,,,,,,
121,,,,1018,409,1003,397,,,,,,,,,
```

**Example (same idea, `base_index=1`):**

```csv
frame,p6_x,p6_y,p7_x,p7_y,p8_x,p8_y,p9_x,p9_y,p10_x,p10_y,p11_x,p11_y,p12_x,p12_y,p13_x,p13_y
120,,,,1020,410,1005,398,,,,,,,,,
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
- **Directory:** New: `vaila_dataset_YYYYMMDD_HHMMSS/` beside the video; append: set target with **F7** or `--dataset` before **F9**
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

- **Auto-backup:** Timestamped copies are written automatically before major edits (see `make_backup()` in the module)
- **Hidden history:** `.vaila_markers_history/<video_stem>/`
- **Retention:** Keeps the latest **100** backups per video stem
- **Reload coordinates:** Use the **Load** button and pick your CSV again (there is no dedicated “reload” hotkey — **L** toggles **Labeling** mode)

### Documentation Access

- **Quick Help:** Press **H** for the in-app overlay
- **HTML guide:** Press **?** (Help Web) to open `vaila/help/getpixelvideo.html` in your browser
- **Canonical docs:** `vaila/help/getpixelvideo.md` (Markdown) and `vaila/help/getpixelvideo.html` (browser)

## Tips and Best Practices

1. **Sequential mode (S/O):** first click fills the **selected** slot if it is empty (including kp **0**), then advances — pair with **TAB** / **Ctrl+G**
2. **1 Line mode** traces contours or paths in a single frame
3. **Automatic backups** run before marker removals and similar operations
4. **Use zoom** for precise placement on broadcast footage
5. **The A key** adds trailing empty slots when you need more indices than the video suggests
6. **Persistence mode** helps visualize motion across frames
7. **Auto-detection** works best with CSV files that match the documented column patterns
8. **FIFA:** keep the TOML beside each video under version control; treat `.vaila_markers_history/` as a local safety net, not the primary archive

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

- **In-app Help:** Press **H** for the overlay; **?** opens the HTML guide in a browser
- **Canonical docs:** `vaila/help/getpixelvideo.md` · `vaila/help/getpixelvideo.html`
- **Project repository:** https://github.com/vaila-multimodaltoolbox/vaila

## Version History

### Version 0.6.0 (April 2026)

- **FIFA labeling:** TOML-driven `n_keypoints` / `start_keypoint` / `base_index` (**K** or **FIFA** button); default **31** FIFA field KPs in vailá; sparse integer CSV; `--fifa-dataset` / `--fifa` CLI
- **`FIFA`** button in the control cluster; **F7** accepts vailá and FIFA dataset layouts
- Reference geometry: `vaila/models/soccerfield_ref3d_fifa_dataset.csv` (and related dataset metadata)
- **Guide (`G`):** visual overlay (FIFA field or pose skeleton); **`V`** toggles the reference map
- **Sequential mode:** fills the **selected** empty slot first (including index **0**), then advances
- **Ctrl+G** (Go KP), **Ctrl+E** / Save ML, `.vaila_markers_history/` backups (100 newest per video)

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
