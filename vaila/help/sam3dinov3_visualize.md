# SAM3+DINOv3 3D — Visualize selected ID(s)

**Version:** 0.4.10
**Updated:** 2026-10-19

This CPU-only tool rerenders an existing `processed_sam3dinov3_*` (SAM3+DINOv3 3D / SAM 3D Body) result. It does not load SAM3 or SAM 3D Body weights, so it is safe to run right after a GPU inference run to isolate one or more people, and does not repeat GPU allocation.

It draws each selected person's SAM contour, bounding box, ID, and reprojected MHR70 skeleton on the original video (one composite overlay), then writes a new directory filtered to that ID set. The root includes filtered keypoint/camera/contour/predictions/overlay outputs (and filtered per-frame mesh, when the source run has `meshes/`); `source_artifacts/` preserves the original run for provenance.

## Overlay style (v0.3.96)

The selected-ID overlay matches the SAM3+DINOv3 look:

1. **SAM3 contour** — semi-transparent silhouette fill (anti-aliased edge, alpha 0.45) + outline, bbox
2. **MHR70 skeleton** — 2D reprojection (`keypoints_2d_px`) drawn with `sam3dinov3.skeleton_edges`, colored by joint side using the `left-`/`right-` name prefix (see table below). An edge only takes a side color when both its endpoints share that side; mixed edges (e.g. neck→shoulder) fall back to the center color.
3. **`ID nn  z=… m`** label — depth taken from `cam_t_m[2]`

Identity is never reassigned: `person_id == sam_obj_id`, exactly as in the source run.

### Left / right colors and C3D `pN` numbers

Palette (RGB) from `sam3dinov3.COLOR_*_RGB` — same live overlay and this rerenderer:

| Side | Color | RGB |
| --- | --- | --- |
| **Left** (`left-*`) | green | `(0, 255, 0)` |
| **Right** (`right-*`) | orange | `(255, 128, 0)` |
| Center / spine (`nose`, `neck`, …) | blue | `(51, 153, 255)` |

C3D / `*_mhr70_rec3d.csv` and `*_markers.csv` markers are **`p0`…`p69`** (0-based index into `MHR70_NAMES`). A legacy file that still starts at `p1` is shifted down to `p0` when this tool copies it. C3D `POINT:LABELS` usually stay as `pN` (no `left_`/`right_` string); use this map (or load `vaila/skeletons/sam3dinov3_mhr70.json`):

| Side | Key `pN` (body) |
| --- | --- |
| Left | `p1` eye, `p3` ear, `p5` shoulder, `p7` elbow, `p9` hip, `p11` knee, `p13` ankle, `p15–p17` foot, `p42–p62` hand/wrist, `p63` olecranon, `p65` cubital fossa, `p67` acromion |
| Right | `p2` eye, `p4` ear, `p6` shoulder, `p8` elbow, `p10` hip, `p12` knee, `p14` ankle, `p18–p20` foot, `p21–p41` hand/wrist, `p64` olecranon, `p66` cubital fossa, `p68` acromion |
| Center | `p0` nose, `p69` neck |

Full order: `vaila/sam3dinov3.py` → `MHR70_NAMES` (`p0=nose` … `p69=neck`).

### Monocular 3D without DLT3D (not a left/right swap)

`*_mhr70_rec3d.csv` / mesh export without `--dlt3d` stay in the **camera frame** (OpenCV: +X right, **+Y down**, +Z forward). Shape and anatomical labels match the 2D overlay (green = anatomical left). Opening that camera-frame CSV/mesh in a Y-up / Z-up viewer without `monocular_dlt_align` looks upside-down; a naive axis fix (bare Y/Z column swap, or 180° about Z) is a **reflection** and can *look* left/right mirrored even though labels are correct. Prefer `monocular_dlt_align` (or `sapiens2_3d --dlt3d`) for lab-frame C3D/BVH, and the BVH path that uses `(x, y, z) → (x, z, −y)` (det +1) for Blender.

## Mesh export for Blender (v0.3.96)

When the *source* `sam3dinov3.py` run was executed with `--save-mesh`, this tool can also export the selected person's per-frame body mesh (MHR vertices + shared faces, with `cam_t` translation applied so the body keeps moving instead of resetting to the origin every frame) as a sequence Blender can play back as an animation:

```bash
uv run python -u vaila/sam3dinov3_visualize.py \
  --sam3d-results /path/to/processed_sam3dinov3_.../video_stem \
  --video /path/to/video.mp4 --id 2 --output /path/to/selected_id_02 \
  --export-mesh obj   # or: --export-mesh ply (smaller, binary)
```

This writes `meshes_obj/frame_NNNNNN.obj` (or `meshes_ply/frame_NNNNNN.ply`) — one complete mesh file per frame, sharing the MHR topology recorded in `mesh_faces.npy`.

**To view it in Blender:**

1. `Edit > Preferences > Add-ons`, enable the built-in **"Mesh: Stop Motion OBJ"** add-on (ships with Blender, no download needed).
2. `File > Import > Mesh Sequence`, then point it at the `meshes_obj/` (or `meshes_ply/`) folder.
3. Blender creates one object that swaps its mesh data every frame — press play to see the body move.

> **Note:** `sam3sapiens2_visualize.py` (SAM3+Sapiens2, 2D/3D **keypoints only**) has no surface mesh — Sapiens2 does not regress a body mesh, only joints — so mesh export only applies to the SAM3+DINOv3 3D pipeline. If you only need a skeleton in Blender, drive an Empty/Armature per joint from the `*_id_NN_mhr70_rec3d.csv` (or the Sapiens2 equivalent) instead.

If the source run has no `meshes/` (the common case — mesh export is off by default in `sam3dinov3.py` because it is large, ~220 KB/person/frame), `--export-mesh` logs a warning and skips silently; rerun `sam3dinov3.py` with `--save-mesh` first.

## CLI

```bash
uv run python -u vaila/sam3dinov3_visualize.py \
  --sam3d-results /path/to/processed_sam3dinov3_.../video_stem \
  --video /path/to/video.mp4 --id 2 --output /path/to/selected_id_02
```

### Multiple IDs (composite overlay)

Pass several IDs to draw them all on **one** video and filter artifacts to that set:

```bash
uv run python -u vaila/sam3dinov3_visualize.py \
  --sam3d-results /path/to/processed_sam3dinov3_.../video_stem \
  --id 0 1 2 -o /path/to/selected_ids_00_01_02

# Equivalent comma form:
uv run python -u vaila/sam3dinov3_visualize.py \
  --sam3d-results /path/to/.../video_stem --id 0,1,2 -o /path/to/out
```

Output folder / overlay use `visualized_ids_00_01_02` / `*_sam3dinov3_ids_00_01_02_overlay.mp4` when more than one ID is selected; a single ID keeps the previous `visualized_id_NN` naming. `--markers-csv` applies only when exactly one ID is selected.

`--id` selects the SAM/person identity (or identities). If you omit `--id` while supplying `--sam3d-results`, `--video`, and `--output`, the CLI prints the available IDs and **prompts interactively** (comma/space-separated list OK):

```bash
uv run python -u vaila/sam3dinov3_visualize.py \
  --sam3d-results /path/to/processed_sam3dinov3_.../video_stem \
  --video /path/to/video.mp4 --output /path/to/selected_id_out
# >> Available person IDs: [0, 1, 2, 4]
# >> Enter SAM/person ID(s) to visualize (comma/space-separated): 0,2,4
```

Use `--list-ids` to discover IDs and exit without rendering. `--dry-run` validates paths and the video's frame count/dimensions against the run. `--overwrite` allows a non-empty output directory. `--export-mesh {obj,ply}` writes the Blender mesh sequence described above (default `none`).

### Recursive batch across subdirectories (CLI-only)

```bash
uv run python -u vaila/sam3dinov3_visualize.py \
  --sam3d-results /path/to/root -r -d -1
```

`--recursive`/`-r` walks `--sam3d-results` for every completed
`processed_sam3dinov3_*` run directory found below it, instead of a single
run — `--video` is not used in this mode, each run's source video is
auto-discovered. `--depth`/`-d N` bounds the walk (`-1` unlimited — default,
`0` root only, `1-99` levels). If `--id` is omitted, **every** available ID is
rendered **separately** per video (legacy overnight behaviour); with
`--id 1 3` one **composite** multi-ID output is written per video. An ID/set
already rendered (`visualized_id_NN` / `visualized_ids_…`) is never
re-rendered. This mode is CLI-only.

### Manual Marker Correction Override (`--markers-csv`) (v0.4.5)

When keypoint tracking coordinates are manually edited frame-by-frame in `getpixelvideo.py`, the corrected coordinates can be fed back into `sam3dinov3_visualize.py` without re-running SAM3 or SAM 3D Body model inference:

```bash
uv run python -u vaila/sam3dinov3_visualize.py \
  --input-dir /path/to/video_sam3dinov3_visualized_id_00 \
  --markers-csv /path/to/video_sam3dinov3_id_00_overlay_markers.csv
```

**Resolution Priority:**
1. Explicit `--markers-csv <path>` (highest priority)
2. GUI-selected corrected CSV
3. Auto-detected unambiguous `*overlay_markers.csv` (saved by `getpixelvideo.py`) or `*_markers_corrected.csv` in the run directory
4. Original automatically generated marker CSV / model predictions

**Regenerated vs. Preserved Artifacts:**
- **Regenerated with corrected coordinates:** `*_sam3dinov3_id_NN_overlay.mp4` (skeleton reprojection), `<video>_sam3dinov3_keypoints2d.csv`, `<video>_id_NN_markers.csv`, and `<video>_sam3dinov3_predictions.json.gz`.
- **Preserved without modification:** Upstream SAM3 segmentation contours (`sam_contours.json`), bounding box tracks, camera translation (`cam_t`), 3D body mesh sequence, and tracking metadata.
- **Output safety:** When `--markers-csv` is supplied and `--output` is omitted, the tool automatically defaults to `<run_dir>_corrected/` to avoid overwriting original automatic results.
- **Provenance tracking:** The manifest (`sam3dinov3_selected_id_manifest.json`) records `marker_source: "manual_correction"`, `marker_csv: "<path>"`, and `corrected_markers_applied: true`.

## GUI

Frame B → **Markerless 3D** → **SAM3+DINOv3 Visualize ID** (or run with no arguments). Select the processed run directory first. The matching source video is filled automatically from `sam3dinov3_summary.json`; if the run was moved, choose the synchronized/cropped source video manually. If a corrected marker file (such as `*overlay_markers.csv`) is present, it is auto-detected into **Markers CSV (optional)**, or you can browse for a corrected CSV. Select an **output parent** and type one or more IDs (`0` or `0,1,2`) in the ID field (available IDs are listed below it). Check **"Export mesh sequence (.obj, for Blender)"** to also write the OBJ sequence described above (needs `--save-mesh` in the source run). The GUI creates a new ID-specific (or multi-ID) child directory, so an existing non-empty parent is safe and repeated runs receive a numeric suffix.

Before rendering, the CLI `--dry-run` path verifies the source frame count and image dimensions against `width`/`height`/`n_frames` recorded in the gzipped predictions — this prevents accidentally applying a synchronized/cropped result to the wrong source video.

## Outputs

| File | Content |
| --- | --- |
| `<video>_sam3dinov3_id_NN_overlay.mp4` | Single-ID overlay (SAM contour/bbox/ID + MHR70 skeleton) |
| `<video>_sam3dinov3_ids_AA_BB_…_overlay.mp4` | Multi-ID composite overlay (all selected people) |
| `<video>_sam3dinov3_keypoints3d.csv` | Long table filtered to selected `person_id`(s) |
| `<video>_sam3dinov3_keypoints2d.csv` | Long table filtered to selected `person_id`(s) |
| `<video>_sam3dinov3_camera.csv` | Per-frame focal/`cam_t`/bbox filtered to selected ID(s) |
| `<video>_sam3dinov3_joint_angles.csv` | Local joint-angle table filtered to selected ID(s) — v0.3.99, only when the source run has one (see [joint_kinematics](joint_kinematics.md)) |
| `<video>_id_NN_mhr70_3d.csv` | Copied as-is for each selected ID (already per-ID in the source run) |
| `<video>_id_NN_mhr70_rec3d.csv` | Copied as-is (vailá `rec3d` convention) |
| `<video>_id_NN_markers.csv` | Copied as-is (wide 2D for REC2D / `getpixelvideo.py`) |
| `<video>_sam3dinov3_predictions.json.gz` | Filtered provenance for the selected ID set |
| `sam_tracks.csv`, `sam_bbox_tracks.csv`, `sam_contours.json` | Filtered to selected `obj_id`(s) |
| `meshes/frame_NNNNNN.npz` | Only when the source run has `meshes/`: selected-person `vertices`/`cam_t` slice(s) |
| `mesh_faces.npy` | Copied when meshes were filtered |
| `meshes_obj/frame_NNNNNN.obj` or `meshes_ply/frame_NNNNNN.ply` | Only with `--export-mesh`: Blender-importable mesh sequence |
| `sam3dinov3_selected_id_manifest.json` | Machine-readable summary of this selection |
| `README_sam3dinov3_selected_id.txt` | Human-readable provenance note |
| `source_artifacts/` | Full original run, preserved for provenance |

`NN` is the SAM `obj_id`, zero-padded — the same identity used throughout `sam3dinov3.py`.

Frames are zero-based; 2D coordinates are full-frame pixels; 3D coordinates are metres. SAM3 remains the identity authority; SAM 3D Body never reassigns IDs.

## Related modules

- [sam3dinov3](sam3dinov3.md) — the GPU inference run this tool rerenders
- [sam3sapiens2_visualize](sam3sapiens2_visualize.md) — the 2D-pose counterpart this tool mirrors
- [joint_kinematics](joint_kinematics.md) — shared math for the filtered joint-angle CSV
