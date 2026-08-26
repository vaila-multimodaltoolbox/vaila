# rec3d

## Module information

| Field | Value |
|-------|--------|
| **Category** | Processing |
| **File** | `vaila/rec3d.py` |
| **Version** | 0.3.117 |
| **Author** | Paulo Santiago |
| **GUI** | Yes |
| **CLI** | Yes |

---

## Description

Batch 3D reconstruction using the **Direct Linear Transformation (DLT)** method with multiple cameras and DLT3D parameters that **vary per frame** — a DLT "matrix" (one row per frame, per camera), as opposed to **rec3d_one_dlt3d**, which uses one fixed set of DLT3D parameters per camera for the whole clip. Use `rec3d` when your cameras move or get re-calibrated during the recording (e.g. broadcast/pan-tilt-zoom cameras).

For each camera you provide:

- One **DLT3D parameter file** with **one row per frame** (frame, then 11 coefficients).
- One **pixel-coordinate CSV**, placed together with the other cameras' pixel files in a single input directory.

All camera pixel files must live in the **same input directory** (one CSV per camera) so they can be correlated frame by frame; they are paired with `--dlt-files` by **sorted filename order** — the same convention `rec3d_one_dlt3d` uses to pair `--dlt3d`/`--pixels`, just via a directory instead of an explicit file list.

> **v0.3.93 fix:** earlier versions read each camera's pixel file completely independently and treated the `input_directory`'s CSVs as unrelated batch trials, silently reusing a single file's row as if it were every camera's observation. It has been rewritten to correlate all camera pixel files by frame and reconstruct one 3D result — the CSV file count in `input_directory` must now match `--dlt-files` exactly, or the run stops with a clear error.

---

## Input file formats

### DLT3D file (per camera)

- CSV with **one row per frame**: `frame`, then 11 DLT coefficients.
- **One file per camera**, passed to `--dlt-files` in the same order the pixel CSVs will sort alphabetically.
- If a frame is missing from a camera's DLT3D file (or its coefficients contain `NaN`), that frame is skipped for reconstruction.

### Pixel CSV (per camera)

- **Column labels are not inspected — only column order matters.** Column 0 is the frame identifier and every pair of columns after that is one marker's (x, y), regardless of header text (vailá `p1_x`/`p1_y`, SAM3, YOLO, MediaPipe named joints, etc.).
- Exactly **one CSV per camera** must be placed in `--input-dir`, matching `--dlt-files` in count. Files are paired by **sorted filename**, so name them so that alphabetical order matches camera/DLT-file order (e.g. `cam1_pixels.csv`, `cam2_pixels.csv`, ...).
- If camera pixel files have different marker counts, the smallest common count is used (with a warning).

Only frames present in **every** camera's pixel file, with valid (non-`NaN`) DLT parameters for that frame in every camera, are reconstructed.

---

## Output files

A single reconstruction result is written inside a new subfolder: `vaila_rec3d_YYYYMMDD_HHMMSS/`.

| File | Description |
|------|-------------|
| `rec3d_*.csv` | 3D points: `frame`, `p1_x`, `p1_y`, `p1_z`, `p2_x`, ... |
| `rec3d_*.3d` | Same data as CSV (duplicate format). |
| `rec3d_*.bvh` | Mocap format for Blender (each marker as an independent ROOT node). |
| `rec3d_*_blender_skeleton_viz.py` | Companion script (see "Skeleton visualization" below). |

Unlike `rec3d_one_dlt3d`, this module does not export C3D; use `rec3d_one_dlt3d` or `readcsv_export` if you need C3D from the resulting CSV. **v0.3.99** added the same BVH + skeleton-visualization export `rec3d_one_dlt3d` already had.

---

## Skeleton visualization (`--swap-yz` / `--skeleton`)

Every run also writes a `.bvh` file (each marker as an independent ROOT node — there is no rigid skeleton model, since marker sets vary by tracker) and a `_blender_skeleton_viz.py` companion script. The BVH imports natively into Blender; running the companion script inside Blender's Text Editor afterward draws bone connections between the markers using the `--skeleton` JSON's `"connections"` list (`[["pA","pB"], ...]`, referencing the **1-based** `pN` column index — always renumbered positionally, regardless of the original tracker's own column labels).

Ready-made presets for every tracker vailá supports ship in `vaila/skeletons/` (and test templates in `tests/skeleton_templates/`):

| File | Keypoint set | Points |
|------|-------------|--------|
| `fifa_body15.json` | FIFA Skeletal Challenge 2026 | 15 |
| `yolo_coco17.json` | YOLO / COCO-17 | 17 |
| `mediapipe_hand21.json` | MediaPipe Hand (Single) | 21 |
| `openpose_body25.json` | OpenPose Body-25 | 25 |
| `halpe26.json` | Halpe 26 Body+Feet | 26 |
| `soccerfield_calib29.json` | Soccer Field 29 Keypoints | 29 |
| `soccerfield_pitch32.json` | Soccer Field 32 Keypoints | 32 |
| `mediapipe_pose33.json` | MediaPipe BlazePose | 33 |
| `mediapipe_hands42.json` | MediaPipe Both Hands | 42 |
| `sam3dinov3_mhr70.json` | SAM3+DINOv3 (SAM 3D Body) MHR70 | 70 |
| `mediapipe_holistic75.json` | MediaPipe Holistic (Body+Hands) | 75 |
| `coco_wholebody133.json` | COCO WholeBody / Sapiens-133 | 133 |
| `sapiens2_goliath308.json` | Sapiens2 Sociopticon/Goliath | 308 |

Pick the preset matching the tracker that produced your pixel CSVs (without a `--skeleton` JSON, a hardcoded MediaPipe-33 default is used). `--swap-yz` swaps Y/Z axes so height ends up vertical (Z-up) in Blender — **this is the default since v0.3.99**; pass `--no-swap-yz` to keep the raw DLT axes.

### The companion script imports everything already aligned (v0.3.99)

Run `rec3d_*_blender_skeleton_viz.py` inside Blender (Text Editor > Run Script) on an empty scene and it will, in order: set the scene rate and frame range, import the BVH, and draw the skeleton bones. It is also safe to run *after* importing the BVH/C3D by hand — it will not duplicate an existing armature and still fixes the scene settings.

**Why the script sets the scene rate itself:** Blender's BVH importer defaults to `update_scene_fps=False` and `update_scene_duration=False`. Importing a 631-frame / 120 Hz capture with File > Import > BVH therefore leaves the scene at **24 fps with `frame_end=250`** — the animation plays in slow motion (26 s instead of 5.3 s) and stops a third of the way through, while an imported C3D (whose importer *does* read `POINT:RATE`) plays correctly. The exported data was never wrong; only the scene settings were. Fractional capture rates are preserved exactly through Blender's `fps`/`fps_base` pair (e.g. 119.88012001 Hz → `fps=120`, `fps_base=1.001`).

### Saving the mesh animation so anyone can open it (v0.3.117)

Once the companion script has run (via the **Animation Blender** button/CLI or manually), the whole scene — BVH skeleton, bone armature *and* the per-frame body mesh — is now made of real Blender data-blocks. A plain **File > Save As** produces a single self-contained `.blend` that plays back correctly in a *fresh* Blender install, on any machine, with no *vailá*, no Python script, and no `meshes_obj/`/`.bvh` folder anywhere near it.

Before v0.3.117, the mesh sequence was swapped frame-by-frame from a Python dict via a live `frame_change_post` handler — fast to build, but neither the dict nor the handler is a Blender data-block, so neither one survives a save/reload; reopening a saved file showed the skeleton fine but the mesh frozen on whatever frame it was saved at. The mesh sequence is now baked into the mesh's own **Shape Keys** instead (one key per frame, keyframed to `value=1` only on its own frame with `CONSTANT` interpolation so frames swap discretely) — the same mechanism Blender uses to save any other shape-keyed animation, so it is exactly as portable as the BVH action already was.

This does make the mesh object itself heavier to build (one shape key per frame, so a 631-frame / ~18k-vertex sequence takes on the order of 30 s to bake and saves to a `.blend` on the order of 100+ MB) — that cost is paid once, at save time, in exchange for the result being shareable as a single file.

---

## GUI mode

Run with no arguments:

1. **DLT3D files** — select one file per camera (each with one row per frame).
2. **Input directory** — a single folder containing exactly one pixel CSV per camera.
3. **Output directory** — where to create the timestamped result folder.
4. **Data rate (Hz)** — recorded in the console/summary (not used for interpolation).
5. **Swap Y/Z for BVH** — defaults to Yes (height vertical in Blender); answer No only to keep raw DLT axes.
6. **(Optional) Skeleton Pose JSON** — pick one of the `vaila/skeletons/` presets, or your own.

---

## CLI mode

**Required (headless):** `--dlt-files`, `--input-dir`, `--output-dir`, `--rate`.
**Optional:** `--no-swap-yz`, `--skeleton`.

| Argument | Description |
|----------|-------------|
| `--dlt-files` *FILE* [*FILE* ...] | DLT3D parameter files, one per camera (each with one row per frame). |
| `--input-dir` *DIR* | Directory containing exactly one pixel CSV per camera, matching `--dlt-files` in count. |
| `--output-dir` *DIR* | Output directory; a timestamped `vaila_rec3d_*` subfolder is created here. |
| `--rate` *HZ* | Data frequency in Hz (recorded in the console summary). Accepts fractional rates, e.g. `119.88012001`. |
| `--swap-yz` | Swap Y and Z axes in BVH output so height is vertical (Z-up) in Blender. **Default since v0.3.99**; kept as an explicit opt-in. |
| `--no-swap-yz` | Keep the raw DLT axes in the BVH output (no Y/Z swap). |
| `--skeleton` *FILE* | Path to a skeleton connections JSON; see `vaila/skeletons/` for presets. |

### Examples

```bash
# Two moving/re-calibrated cameras, one pixel CSV each in ./cams
python vaila/rec3d.py \
  --dlt-files cam1_matrix.dlt3d cam2_matrix.dlt3d \
  --input-dir ./cams \
  --output-dir ./out \
  --rate 100

# Same, with BVH Y/Z swap and a SAM3+DINOv3 skeleton for Blender
python vaila/rec3d.py \
  --dlt-files cam1_matrix.dlt3d cam2_matrix.dlt3d \
  --input-dir ./cams --output-dir ./out --rate 119.88012001 \
  --skeleton vaila/skeletons/sam3dinov3_mhr70.json   # --swap-yz is the default
```

---

## Main functions

| Function | Description |
|----------|-------------|
| `rec3d_multicam` | Reconstruct one 3D point from multiple camera observations (DLT least squares). Shared with `rec3d_one_dlt3d`. |
| `load_pixel_csv_positional` | Column-order-based pixel CSV reader (frame + N marker x,y pairs), ignoring header labels. Shared with `rec3d_one_dlt3d`. |
| `find_common_frames` | Sorted intersection of frame numbers across camera pixel files. Shared with `rec3d_one_dlt3d`. |
| `process_files_in_directory` | Core logic: correlate N camera pixel files by frame, look up each camera's per-frame DLT3D parameters, reconstruct, save CSV/.3d/BVH + skeleton script. |
| `save_rec3d_as_bvh` | BVH export (owns this function; `rec3d_one_dlt3d` imports it). |
| `generate_blender_companion_script` | Skeleton-visualization companion script (owns this function; `rec3d_one_dlt3d` imports it). |
| `run_rec3d` | GUI/CLI entry point. |

---

## Related modules

| Module | Role |
|--------|------|
| **dlt3d** | Compute DLT3D coefficients from calibration (pixel + 3D reference), one row per frame when the camera moves. |
| **rec3d_one_dlt3d** | Same DLT3D method with one fixed set of parameters per camera; also imports `save_rec3d_as_bvh`/`generate_blender_companion_script` from this module. |
| **readcsv_export** | CSV → C3D (used internally by `rec3d_one_dlt3d`); batch convert. |

---

Part of *vailá* - Multimodal Toolbox
[GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
