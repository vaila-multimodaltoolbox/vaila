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
| `mediapipe_pose33.json` | MediaPipe BlazePose | hand-authored | 33 | 30 |
| `yolo_coco17.json` | YOLO / COCO-17 | hand-authored | 17 | 16 |
| `sam3dinov3_mhr70.json` | SAM3+DINOv3 (SAM 3D Body) MHR70 | `vaila/sam3dinov3.py`'s `MHR70_NAMES`/`SKELETON_EDGE_NAMES` | 70 | 30 |
| `sapiens2_goliath308.json` | Sapiens2 Sociopticon/Goliath | vendored `sapiens2/sapiens/pose/configs/_base_/keypoints308.py` | 308 | 71 |

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

## Regenerating the model-derived presets

`mediapipe_pose33.json` and `yolo_coco17.json` are hand-authored (no
canonical machine-readable source in this repo) — edit them directly if
needed. `sam3dinov3_mhr70.json` and `sapiens2_goliath308.json` are derived
programmatically from each model's own keypoint/skeleton definition — **do
not hand-edit them**; if `vaila/sam3dinov3.py`'s `MHR70_NAMES`/
`SKELETON_EDGE_NAMES` change, or the vendored Sapiens2 checkout is updated,
regenerate both from source instead:

```bash
uv run python vaila/skeletons/generate_skeleton_jsons.py
```

(The Sapiens2 preset additionally requires `bash bin/setup_sapiens2.sh` to
have been run at least once, so `.local/third_party/sapiens2/` exists — the
generator reads `keypoints308.py` directly from that checkout, no CUDA or
`sapiens` package import required.)
