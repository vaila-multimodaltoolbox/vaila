# mkvis3d (OpenBiomech 3D Motion Viewer) — User & Developer Guide

*Version: 0.4.3 | Updated: 17 September 2026*
*Platform: Linux, macOS, Windows | Python 3.12+*

---

## 1. Overview

**mkvis3d** (OpenBiomech) is an interactive 3D motion viewer, biomechanical visualization environment, and reproducible kinematic/kinetic analysis engine developed in tandem with **vailá** (Multimodal Toolbox).

It provides:
- **Interactive WebGL/Three.js 3D Viewer**: Hardware-accelerated 3D marker rendering, skeleton connectivity, floor grids, force plate vectors, and multi-camera synchronized video overlays.
- **Seamless Integration with vailá**: Direct loading of 3D reconstruction outputs from `rec3d.py`, `rec3d_one_dlt3d.py`, `monocular_dlt_align.py`, `viewc3d.py`, and SAM 3D Body (DINOv3) MHR-70 markerless tracking.
- **Comprehensive Skeleton Template Catalog**: Ready-to-use models matching all trackers (MediaPipe 33, YOLO COCO-17, OpenPose 25, Halpe 26, FIFA Body-15, SAM3+DINOv3 MHR-70, Sapiens2 Goliath-308, Hand 21, Hands 42, Holistic 75, COCO WholeBody 133, Soccer Field Pitch 32, Soccer Field Calib 29, Soccer Field Kiki 49).
- **Dual Indexing Compatibility**: Transparent support for both **0-based** (`p0..p(N-1)`) and **1-based** (`p1..pN`) marker tokens.
- **Biomechanical Math Engine**: Visual3D-compliant Laboratory Coordinate System (LCS) transforms, zero-lag Butterworth filtering, gap interpolation, segment kinematics, and inverse dynamics.
- **Blender & BVH Export**: Automated companion scripts for Blender animation matching exact framerates and bone hierarchies.

---

## 2. Launching mkvis3d

The application can be started directly via Python, shell script, batch file, or CLI command:

### Linux & macOS
```bash
# Launch interactive GUI in default browser
python mkvis3d.py

# Open a specific motion file directly (C3D, CSV, or .3d)
python mkvis3d.py data/rec3d_sample_m.c3d

# Launch using convenience launcher
./mkvis3d_launcher.sh

# Using the installed openbiomech CLI package
openbiomech gui
openbiomech gui path/to/trial.c3d --video camera1.mp4 camera2.mp4
```

### Windows
```powershell
# Double-click or run from PowerShell / Command Prompt
python mkvis3d.py
.\mkvis3d.bat

# Open with specific file
python mkvis3d.py data\rec3d_sample_m.c3d
```

---

## 3. CLI Command Reference

`mkvis3d.py` delegates to `openbiomech.cli` and supports a rich set of subcommands:

| Subcommand | Description | Example Usage |
|---|---|---|
| `gui` | Launch interactive local web server with 3D viewer | `python mkvis3d.py gui trial.c3d --port 8080` |
| `view` | Export standalone, self-contained HTML viewer file | `python mkvis3d.py view trial.c3d --output viewer.html` |
| `info` | Print trial summary: frames, markers, rate, force plates | `python mkvis3d.py info trial.c3d` |
| `lcs` | Transform trial into target Laboratory Coordinate System | `python mkvis3d.py lcs trial.c3d --ap +Z --axial +Y -o aligned.c3d` |
| `filter` | Gap interpolation and Butterworth/Savitzky-Golay smoothing | `python mkvis3d.py filter trial.c3d --cutoff 6.0 -o filtered.c3d` |
| `segment` | Compute 3D segment length, range, and longitudinal axis | `python mkvis3d.py segment trial.c3d LASI RASI` |
| `dynamics` | Run inverse dynamics and export joint kinetics | `python mkvis3d.py dynamics trial.c3d -o dynamics.csv` |
| `blender` | Generate companion Python script for Blender 3D | `python mkvis3d.py blender trial.c3d -s sam3dinov3_mhr70 -o viz.py` |
| `bvh` | Export trial markers to Biovision Hierarchy (.bvh) | `python mkvis3d.py bvh trial.c3d -o motion.bvh` |
| `demo` | Generate synthetic gait/squat trial for testing | `python mkvis3d.py demo -o demo.c3d` |
| `install` | Install desktop shortcuts and .c3d file associations | `python mkvis3d.py install` |

---

## 4. Integration with vailá Workflows

`mkvis3d` is designed as the visual inspection and kinematic analysis companion for `vaila`:

```
               ┌───────────────────────────────┐
               │    vailá Multimodal Toolbox    │
               └───────────────┬───────────────┘
                               │
       ┌───────────────────────┼────────────────────────┐
       ▼                       ▼                        ▼
[rec3d / DLT3D]       [sam3dinov3.py]           [Soccer Calibration]
- rec3d_m.c3d         - MHR70 keypoints (m)    - soccerfield_kiki.csv
- rec3d_mm.c3d        - sam_tracks.csv          - soccerfield_kiki_custom.c3d
- wide CSV (p0..pN)   - meshes_obj/ (.obj)      - 32-pt / 49-pt pitch models
       │                       │                        │
       └───────────────────────┼────────────────────────┘
                               ▼
               ┌───────────────────────────────┐
               │  mkvis3d (OpenBiomech Viewer) │
               │  - Interactive 3D Orbit       │
               │  - Skeleton Templates (14+)   │
               │  - Video Synchronization      │
               │  - LCS Alignment (AP/Axial)   │
               │  - Force Plate Vectors (COP)  │
               └───────────────────────────────┘
```

### 1. Loading 3D Reconstructions (`rec3d.py` / `rec3d_one_dlt3d.py`)
- Run 3D reconstruction in `vaila`. The output folder contains:
  - `rec3d_YYYYMMDD_HHMMSS_m.c3d` (coordinates in meters)
  - `rec3d_YYYYMMDD_HHMMSS.csv` (wide CSV: `frame, p0_x, p0_y, p0_z, ...`)
- Open in `mkvis3d`:
  ```bash
  python /home/preto/data/mkvis3d/mkvis3d.py path/to/rec3d_YYYYMMDD_HHMMSS_m.c3d
  ```

### 2. Loading SAM 3D Body DINOv3 Trajectories (`sam3dinov3.py`)
- SAM 3D Body outputs 70 3D body keypoints (MHR-70) aligned with world or camera frame.
- Load the resulting C3D or wide CSV into `mkvis3d`.
- In the viewer header dropdown **Skeleton Template**, select `SAM3+DINOv3 MHR-70 (70)`.
- Left side connections render in **Green** (`#00ff00`), right side in **Orange** (`#ff8000`), and midline in **Blue** (`#3399ff`).

### 3. Soccer Field Pitch & Goal Visualization
- Load `soccerfield_kiki_custom.c3d` or wide soccer calibration CSV:
  ```bash
  python /home/preto/data/mkvis3d/mkvis3d.py data/soccerfield_kiki_custom.c3d
  ```
- Select template `Soccer Field Kiki (49)`:
  - Draws touchlines, goal lines, penalty boxes, and midfield line.
  - Draws 3D goal posts, crossbar, net depth ground points, and corner flags ($z = 1.5$ m).
  - Center circle ($R = 9.15$ m) and penalty arcs are drawn with precision curves.

---

## 5. Skeleton Templates & Dual-Index Support

Skeleton presets define wireframe connectivity (`connections: [["pA", "pB"], ...]`) and lateralization.

All presets are maintained in sync across:
1. `vaila/skeletons/`
2. `vaila/models/skeleton_templates/`
3. `tests/skeleton_templates/`
4. `/home/preto/data/mkvis3d/skeleton_templates/`

### Supported Presets

| Template Preset | Keypoints | Tracker / Standard | Indexing Invariant |
|---|---|---|---|
| `mediapipe_pose33.json` | 33 | MediaPipe BlazePose | 0-based `p0..p32` / 1-based `p1..p33` |
| `yolo_coco17.json` | 17 | YOLO / COCO Pose | 0-based `p0..p16` / 1-based `p1..p17` |
| `openpose_body25.json` | 25 | OpenPose Body-25 | 0-based `p0..p24` / 1-based `p1..p25` |
| `halpe26.json` | 26 | Halpe Body + Feet | 0-based `p0..p25` / 1-based `p1..p26` |
| `fifa_body15.json` | 15 | FIFA Skeletal Challenge | 0-based `p0..p14` / 1-based `p1..p15` |
| `sam3dinov3_mhr70.json` | 70 | SAM 3D Body DINOv3 | 0-based `p0..p69` / 1-based `p1..p70` |
| `sapiens2_goliath308.json` | 308 | Sapiens2 Sociopticon | 0-based `p0..p307` / 1-based `p1..p308` |
| `mediapipe_hand21.json` | 21 | MediaPipe Hand | 0-based `p0..p20` / 1-based `p1..p21` |
| `mediapipe_hands42.json` | 42 | MediaPipe Both Hands | 0-based `p0..p41` / 1-based `p1..p42` |
| `mediapipe_holistic75.json` | 75 | MediaPipe Holistic | 0-based `p0..p74` / 1-based `p1..p75` |
| `coco_wholebody133.json` | 133 | COCO WholeBody / Sapiens | 0-based `p0..p132` / 1-based `p1..p133` |
| `soccerfield_pitch32.json` | 32 | FIFA 32 Pitch Points | 0-based `p0..p31` / 1-based `p1..p32` |
| `soccerfield_calib29.json` | 29 | FIFA 29 Calib Landmarks | 0-based `p0..p28` / 1-based `p1..p29` |
| `soccerfield_kiki49.json` | 49 | Pitch + 3D Goals + Flags | 0-based `p0..p48` / 1-based `p1..p49` |

### Dual Indexing Engine
`mkvis3d`'s viewer (`openbiomech/viewer.js`) and classification engine (`openbiomech/skeleton.py`) automatically detect whether an active template or trial uses 0-based or 1-based indexing by inspecting the template note or connection tokens:
- If `p0` is detected or the note indicates 0-based, keypoint indices map directly (`p0 → index 0`).
- If legacy 1-based tokens are present, tokens map via 1-offset (`p1 → index 0`).
- Semantic labels in C3D files (e.g. `Nose`, `Left_Shoulder`) are resolved case-insensitively with hyphen/underscore normalization.

---

## 6. GUI Keyboard Shortcuts & Navigation

| Key / Action | Action in 3D Viewer |
|---|---|
| `Space` | Play / Pause animation playback |
| `←` / `→` | Step one frame backward / forward |
| `Home` / `End` | Jump to start / end frame |
| `+` / `-` | Increase / decrease marker sphere radius |
| `C` | Toggle trajectory trail visibility |
| `F` | Focus / center camera on active subject |
| `Left Click + Drag` | Orbit 3D perspective |
| `Right Click + Drag` | Pan camera laterally / vertically |
| `Scroll Wheel` | Zoom in / out |
| Double-click Splitter | Reset bottom plot panel height to default (170 px) |

---

## 7. Troubleshooting & FAQ

### 1. "No markers visible in viewer"
- Check units: `rec3d` outputs both `_m.c3d` (meters) and `_mm.c3d` (millimeters). `mkvis3d` defaults to meters. If using millimeters, pass `--units mm`.
- Check bounding box: Press `F` to focus on the cloud center.

### 2. "Skeleton bones connect to wrong markers"
- Ensure the template selected in the **Skeleton Template** menu matches your tracker model (e.g. `SAM3+DINOv3 MHR-70` vs `MediaPipe Pose 33`).
- If using custom labels, ensure the label names or column order match the template specification.

### 3. "Port already in use"
- If port 8000 is occupied, specify another port:
  ```bash
  python mkvis3d.py gui trial.c3d --port 8085
  ```

---

## 8. Verification & Test Suite

To run automated unit and integration tests:

```bash
# In mkvis3d repository:
cd /home/preto/data/mkvis3d
.venv/bin/pytest tests/test_skeleton_templates.py -v
.venv/bin/pytest tests/test_application.py tests/test_cli.py tests/test_blender_io.py -v

# In vaila repository:
cd /home/preto/data/vaila
uv run pytest tests/test_skeleton_templates.py -v
uv run pytest tests/test_skeleton_catalog.py -v
uv run pytest tests/test_soccerfield_kiki_and_calib.py -v
```
