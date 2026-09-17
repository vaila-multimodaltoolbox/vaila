# mkvis3d — OpenBiomech 3D Motion Viewer

- **Version:** 0.4.3
- **Updated:** 17 September 2026
- **Module:** `openbiomech` / `mkvis3d.py`
- **Location:** `/home/preto/data/mkvis3d/mkvis3d.py`
- **License:** AGPLv3

---

## What It Is

**mkvis3d** is the dedicated WebGL/Three.js 3D motion viewer and reproducible biomechanics visualization suite developed alongside **vailá**. It allows researchers and practitioners to inspect, filter, calibrate, and analyze marker trajectories, force plate ground reaction forces, and skeleton postures produced by vailá's multimodal tracking modules (`rec3d`, `sam3dinov3`, `sapiens2`, `mediapipe`, `yolo`, and `soccerfield_calib`).

---

## Quick Start & Launch Commands

Run from the terminal or launcher scripts:

```bash
# Launch interactive web GUI
python /home/preto/data/mkvis3d/mkvis3d.py

# Open a 3D reconstruction file directly
python /home/preto/data/mkvis3d/mkvis3d.py path/to/rec3d_m.c3d

# Open with synchronized video panels
openbiomech gui path/to/trial.c3d --video cam1.mp4 cam2.mp4

# Export standalone HTML viewer (no Python needed to open)
python /home/preto/data/mkvis3d/mkvis3d.py view trial.c3d --output viewer.html
```

---

## Integration with vailá

### 1. 3D Reconstructions (`rec3d.py` / `rec3d_one_dlt3d.py`)
`vaila` exports standard C3D files in meters (`*_m.c3d`) and millimeters (`*_mm.c3d`), as well as wide CSV coordinate files (`frame, p0_x, p0_y, p0_z, ...`). Both formats open natively in `mkvis3d`.

### 2. SAM 3D Body (DINOv3) MHR-70 Keypoints
Reconstructions from `vaila/sam3dinov3.py` load directly into `mkvis3d`. Select the preset `SAM3+DINOv3 MHR-70 (70)` in the viewer dropdown to render anatomically lateralized bones:
- **Left Side**: Green (`#00ff00`)
- **Right Side**: Orange (`#ff8000`)
- **Midline / Center**: Light Blue (`#3399ff`)

### 3. Soccer Field Pitch & 3D Goal Models
For broadcast soccer calibrations, load `data/soccerfield_kiki_custom.c3d` or calibrated CSVs and choose `Soccer Field Kiki (49)` to display pitch lines, 3D goal posts, crossbars, net depth ground points, and corner flags.

---

## Supported Skeleton Presets

All 14 standard skeleton presets are synchronized across `vaila/skeletons/`, `vaila/models/skeleton_templates/`, `tests/skeleton_templates/`, and `mkvis3d/skeleton_templates/`:

1. **MediaPipe Pose (33)**: `mediapipe_pose33.json`
2. **YOLO / COCO-17 (17)**: `yolo_coco17.json`
3. **OpenPose Body-25 (25)**: `openpose_body25.json`
4. **Halpe 26 Body+Feet (26)**: `halpe26.json`
5. **FIFA Body-15 (15)**: `fifa_body15.json`
6. **SAM3+DINOv3 MHR-70 (70)**: `sam3dinov3_mhr70.json`
7. **Sapiens2 Goliath-308 (308)**: `sapiens2_goliath308.json`
8. **MediaPipe Hand (21)**: `mediapipe_hand21.json`
9. **MediaPipe Hands (42)**: `mediapipe_hands42.json`
10. **MediaPipe Holistic (75)**: `mediapipe_holistic75.json`
11. **COCO WholeBody (133)**: `coco_wholebody133.json`
12. **Soccer Field Pitch 32 (32)**: `soccerfield_pitch32.json`
13. **Soccer Field Calib 29 (29)**: `soccerfield_calib29.json`
14. **Soccer Field Kiki 49 (49)**: `soccerfield_kiki49.json`

Both 0-based (`p0..p(N-1)`) and legacy 1-based (`p1..pN`) conventions are automatically detected and resolved.

---

## Command Reference

| Command | Purpose |
|---|---|
| `gui` | Launch interactive local web server with 3D viewer |
| `view` | Export standalone, self-contained HTML viewer file |
| `info` | Print trial summary: frames, markers, rate, force plates |
| `lcs` | Transform trial into target Laboratory Coordinate System |
| `filter` | Gap interpolation and Butterworth/Savitzky-Golay smoothing |
| `segment` | Compute 3D segment length, range, and longitudinal axis |
| `dynamics` | Run inverse dynamics and export joint kinetics |
| `blender` | Generate companion Python script for Blender 3D |
| `bvh` | Export trial markers to Biovision Hierarchy (.bvh) |
| `demo` | Generate synthetic gait/squat trial for testing |
