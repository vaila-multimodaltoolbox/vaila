# Skeleton connection presets

Shipped `--skeleton` presets for `rec3d.py` and `rec3d_one_dlt3d.py`'s optional
Blender skeleton-visualization companion script
(`generate_blender_companion_script()`). Each file's `"connections"` list is
`[["pA","pB"], ...]` pairs referencing the **1-based marker index** a
reconstructed `rec3d_*.csv`/`.bvh` uses for that column (`p1_x,p1_y,p1_z`,
`p2_x,...`, always renumbered 1..N in column order regardless of the
original tracker's own labels — see `rec3d.load_pixel_csv_positional`).

| File | Keypoint set | Source | Points | Edges |
|------|-------------|--------|--------|-------|
| `mediapipe_pose33.json` | MediaPipe BlazePose | MediaPipe pose specification | 33 | 32 |
| `yolo_coco17.json` | YOLO / COCO-17 | Ultralytics / COCO standard | 17 | 16 |
| `sam3dinov3_mhr70.json` | SAM3+DINOv3 (SAM 3D Body) MHR70 | `vaila/sam3dinov3.py`'s `MHR70_NAMES`/`SKELETON_EDGE_NAMES` | 70 | 30 |
| `sapiens2_goliath308.json` | Sapiens2 Sociopticon/Goliath | vendored `sapiens2/sapiens/pose/configs/_base_/keypoints308.py` | 308 | 71 |
| `fifa_body15.json` | FIFA Skeletal Challenge 2026 | `vaila/fifa_skeletal_pipeline.py` / Body-15 | 15 | 16 |
| `openpose_body25.json` | OpenPose Body-25 | OpenPose standard / SAM3D body25 | 25 | 24 |
| `mediapipe_hand21.json` | MediaPipe Hand | MediaPipe hand landmarks | 21 | 21 |
| `mediapipe_hands42.json` | MediaPipe Both Hands (Left+Right) | MediaPipe hands 2x21 | 42 | 42 |
| `mediapipe_holistic75.json` | MediaPipe Holistic Body+Hands | MediaPipe Body (33) + Hands (42) | 75 | 76 |
| `halpe26.json` | Halpe 26 (AlphaPose / YOLO-Pose) | Halpe Body+Feet dataset | 26 | 29 |
| `coco_wholebody133.json` | COCO WholeBody / Sapiens-133 | `coco_wholebody_info` in `keypoints308.py` | 133 | 71 |
| `soccerfield_pitch32.json` | Soccer Field 32 Keypoints | `vaila/fifa_dataset_builder.py` pitch layout | 32 | 37 |
| `soccerfield_calib29.json` | Soccer Field 29 Keypoints | `vaila/soccerfield_calib.py` FIFA ref | 29 | 23 |

## Usage

Pick the preset matching **the tracker that produced your pixel CSVs** (the
same 2D keypoint order the reconstruction re-numbers as `p1..pN`), then pass
it as `--skeleton`:

```bash
uv run python -m vaila.rec3d_one_dlt3d \
  --dlt3d c1.dlt3d c2.dlt3d c3.dlt3d \
  --pixels c1_id/markers.csv c2_id/markers.csv c3_id/markers.csv \
  --fps 119.88012001 -o ./out \
  --swap-yz --skeleton vaila/skeletons/sam3dinov3_mhr70.json
```

Or in the GUI: the "(Optional) Skeleton Pose JSON" step's file dialog opens
anywhere — point it at one of these files directly.

## Regenerating the presets

To regenerate all presets across `vaila/skeletons/` and `tests/skeleton_templates/`:

```bash
uv run python vaila/skeletons/generate_skeleton_jsons.py
```
