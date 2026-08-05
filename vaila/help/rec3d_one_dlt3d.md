# rec3d_one_dlt3d

## Module information

| Field | Value |
|-------|--------|
| **Category** | Processing |
| **File** | `vaila/rec3d_one_dlt3d.py` |
| **Version** | 0.3.99 |
| **Author** | Paulo Santiago |
| **GUI** | Yes |
| **CLI** | Yes |

---

## Description

Batch 3D reconstruction using the **Direct Linear Transformation (DLT)** method with multiple cameras and **one fixed set of DLT3D parameters per camera** (the whole clip uses the same 11 coefficients). For DLT3D parameters that vary frame by frame (a DLT "matrix"), use **rec3d** instead. For each camera you provide:

- One **DLT3D parameter file** (11 coefficients per camera, e.g. from the `dlt3d` module).
- One **pixel-coordinate CSV**: column 0 is the frame identifier and every pair of columns after that is one marker's (x, y).

Frames common to all pixel files are reconstructed; results are written to a **timestamped subfolder** in the chosen output directory.

---

## Input file formats

### DLT3D file

- CSV with one row of 11 DLT coefficients (e.g. produced by the *vailá* **dlt3d** module).
- **One file per camera**; order must match the pixel file order.

### Pixel CSV

- **Column labels are not inspected — only column order matters.** Column 0 is the frame identifier and every pair of columns after that is one marker's (x, y), regardless of what the header text says. This makes the file compatible with *vailá*'s own `frame,p1_x,p1_y,...` convention as well as CSVs coming from SAM3, YOLO, MediaPipe (named joints), or any other tracker — as long as the same markers appear in the same order in every camera's file.
- **One file per camera**; same number of markers and overlapping frame sets recommended.
- In GUI mode, each file can be chosen from a **different directory** (one dialog per camera).

---

## Output files

All outputs share the same base name and are written inside a new subfolder: `rec3d_YYYYMMDD_HHMMSS/`.

| File | Description |
|------|-------------|
| `rec3d_*.csv` | 3D points: `frame`, `p1_x`, `p1_y`, `p1_z`, `p2_x`, ... |
| `rec3d_*.3d` | Same data as CSV (duplicate format). |
| `rec3d_*_m.c3d` | C3D in **meters** (`POINT:UNITS=m`, `POINT:FRAMES` set). |
| `rec3d_*_mm.c3d` | C3D in **millimeters** (`POINT:UNITS=mm`). |

C3D files are compatible with **viewc3d**, **viewc3d_pyvista**, **readc3d_export** (inspect/convert), and standard C3D tools. They are generated via `readcsv_export.auto_create_c3d_from_csv`.

---

## Optional: mesh-for-Blender export

If your pixel files are the MHR70-ordered `p1_x,p1_y,...,p70_x,p70_y` markers CSVs written by **sam3dinov3_visualize.py**'s "Visualize ID" output, you can also export a per-frame **body mesh** aligned into this same DLT world space.

For each camera, pass a `--mesh-source-dir` (same order as `--dlt3d`) pointing at that camera's Visualize-ID output directory. Each must contain:

- `<stem>_markers.csv` — that camera's 2D pixel markers (used for triangulation).
- `<stem>_mhr70_rec3d.csv` — that camera's own **monocular** 3D MHR70 estimate.
- `meshes_obj/` or `meshes_ply/` — per-frame meshes (from `sam3dinov3_visualize.py --export-mesh obj|ply`).
- `mesh_faces.npy` — shared face topology.

**`--pixels` can be omitted** when `--mesh-source-dir` is given: each run directory already contains its own `*_markers.csv`, so one path per camera (the run directory itself) drives both triangulation and mesh alignment — no need to point at the same directory twice. Explicit `--pixels` is still supported (required if you are not using mesh export, or your pixel CSVs live elsewhere).

At each frame, a **similarity transform** (rotation + uniform scale + translation, via Umeyama 1991) is fit per camera from its monocular MHR70 estimate (torso/hip/knee subset — see `vaila/mesh_alignment.py`) onto the DLT-triangulated skeleton; the camera with the lowest fit residual is used as that frame's mesh source, and the same transform is applied to its mesh vertices. This is a **coordinate-frame reconciliation, not a re-triangulation** — mesh shape/proportion accuracy is inherited entirely from the monocular estimate; the alignment only fixes position/scale/orientation against the metrically-calibrated world frame.

The mesh is always exported in the **raw** `(x, y, z)` DLT/world frame, the same convention as the reconstruction CSV, **regardless of `--swap-yz`** (which only affects the BVH). **Corrected 2026-08-05** — a same-convention swap was tried first (mesh swapped whenever the BVH was), on the assumption that any OBJ consumer applies the same Y-up→Z-up conversion Blender's BVH importer applies by default; that holds for Blender's own `File > Import > Wavefront (.obj)` dialog but not for the "Stop Motion OBJ"/OBJSequence family of mesh-*sequence* add-ons most people actually use to play this back, which applies no axis conversion at all (confirmed from its source) — so a swapped mesh looked right one static frame at a time but showed height on the wrong axis relative to the BVH once played as a sequence. See `README_mesh_import.txt` (written next to every `meshes_<fmt>/`) for the exact manual-import axis override this implies.

| Argument | Description |
|----------|-------------|
| `--mesh-source-dir` *DIR* [*DIR* ...] | Per-camera Visualize-ID output directory (order must match `--dlt3d`). Also supplies `--pixels` automatically when `--pixels` is omitted. |
| `--export-mesh` `{none,obj,ply}` | Export format (default: `none`). Requires `--mesh-source-dir`. |

Output (in the same timestamped subfolder):

| File | Description |
|------|-------------|
| `meshes_<fmt>/frame_NNNNNN.<fmt>` | Aligned per-frame mesh, Blender-importable as a mesh-cache sequence ("Stop Motion OBJ" / OBJSequence add-on). |
| `rec3d_*_mesh_alignment.csv` | Manifest: `frame`, `camera_index` (which camera's mesh was used), `mean_residual_m`/`rms_residual_m`/`max_residual_m` (Umeyama fit residual), `n_fit_points`. |

```bash
# Simplified: one path per camera (the Visualize-ID run directory) is enough
python -m vaila.rec3d_one_dlt3d \
  --dlt3d c1.dlt3d c2.dlt3d c3.dlt3d \
  --mesh-source-dir c1_id/ c2_id/ c3_id/ \
  --export-mesh obj \
  --fps 119.88012001 -o ./out

# Equivalent, with --pixels given explicitly (redundant here, but still supported)
python -m vaila.rec3d_one_dlt3d \
  --dlt3d c1.dlt3d c2.dlt3d c3.dlt3d \
  --pixels c1_id/markers.csv c2_id/markers.csv c3_id/markers.csv \
  --mesh-source-dir c1_id/ c2_id/ c3_id/ \
  --export-mesh obj \
  --fps 119.88012001 -o ./out
```

---

## Blender alignment (v0.3.99, mesh axis convention corrected 2026-08-05)

`--swap-yz` is the **default** (pass `--no-swap-yz` for raw DLT axes) and applies to the **BVH only**. The mesh is always raw, unconditionally — see above.

The generated `rec3d_*_blender_skeleton_viz.py` imports everything **already aligned**: it sets the scene rate and frame range, imports the BVH with `update_scene_fps`/`update_scene_duration` enabled, imports the OBJ mesh sequence starting on the same frame (reading the raw floats itself, no axis conversion needed), then builds the skeleton bones. Run it on an empty scene (it imports everything itself) or after importing the BVH/C3D by hand (it will not duplicate an existing armature and still fixes the scene settings, since `setup_scene()` always runs last).

**Why the scene setup was needed:** Blender's BVH importer defaults to `update_scene_fps=False` and `update_scene_duration=False`. A File > Import > BVH of a 631-frame / 120 Hz capture therefore leaves the scene at **24 fps with `frame_end=250`** — the BVH and OBJ mesh sequence play in slow motion (26 s instead of 5.3 s) and stop a third of the way through. Fractional capture rates survive exactly through Blender's `fps`/`fps_base` pair (119.88012001 Hz → `fps=120`, `fps_base=1.001`), and the BVH `Frame Time` is written with 9 decimals so the file alone also reads back at the right rate. Blender's bundled C3D importer is not exempt either: even though this exporter's C3D correctly states `POINT:RATE` (verified: both `POINT:RATE` and the file header's `frame_rate` read back as `120.0` for a 120 Hz run), `bpy.ops.import_anim.c3d`'s own `adapt_frame_rate=True` default does not reliably apply it to the scene — re-running this companion script after a manual C3D import fixes that too, since it sets the scene rate last regardless of import order.

A **GUI run prints the equivalent CLI command both before and after processing**, so the last thing on screen is a copy-pasteable headless re-run.

---

## GUI mode

Run with **no arguments** or with `--gui`:

1. **Number of cameras** — e.g. 2.
2. **DLT3D files** — One file dialog per camera; each file can be in a different directory.
3. **Pixel source** — Yes/No: use per-camera **SAM3+DINOv3 "Visualize ID" run directories** (one dialog per camera; each run's own `*_markers.csv` is used automatically, and you can optionally also export the aligned mesh from the same directories) — or No for a plain pixel-coordinate CSV per camera (original flow, no mesh export).
4. **Output directory** — Where to create the timestamped result folder.
5. **Data rate (Hz)** — Point data rate for C3D/CSV (e.g. 60, 100).
6. **Swap Y/Z for BVH** — defaults to Yes (height vertical in Blender, BVH and mesh in the same convention); answer No only to keep raw DLT axes.
7. **(Optional) Skeleton Pose JSON** — Pick one of the `vaila/skeletons/` presets, or your own, for the Blender skeleton-visualization companion script.
8. **(Optional) Mesh export** — If you chose plain pixel CSVs in step 3, one directory dialog per camera (Visualize-ID output) plus OBJ/PLY format choice; if you already picked run directories in step 3, just the OBJ/PLY format choice. The equivalent CLI command is printed (`>>` prefix) before running.

---

## CLI mode

**Required:** `--dlt3d`, `--output`, and either `--pixels` or `--mesh-source-dir`.
**Optional:** `--fps`, `--gui`, `--export-mesh`.

| Argument | Description |
|----------|-------------|
| `--dlt3d` *FILE* [*FILE* ...] | DLT3D parameter files (one per camera); order must match `--pixels`/`--mesh-source-dir`. |
| `--pixels` *FILE* [*FILE* ...] | Pixel coordinate CSV files (one per camera). Optional if `--mesh-source-dir` is given. |
| `--fps` *HZ* | Point data rate in Hz (default: 100). Accepts fractional rates, e.g. `119.88012001` for NTSC-derived 120000/1001 capture. |
| `-o`, `--output` *DIR* | Output directory; a timestamped subfolder will be created here. |
| `--gui` | Launch GUI instead of CLI. |
| `--swap-yz` | Swap Y/Z in BVH **and** mesh so height is vertical (Z-up) in Blender. **Default since v0.3.99**; kept as an explicit opt-in. |
| `--no-swap-yz` | Keep the raw DLT axes (no Y/Z swap). |

### Examples — every supported run type

```bash
# 1) Minimal: two cameras, 60 Hz, output under ./out
python -m vaila.rec3d_one_dlt3d --dlt3d cam1.dlt3d cam2.dlt3d --pixels cam1.csv cam2.csv --fps 60 --output ./out

# 2) Short form for output (-o instead of --output)
python -m vaila.rec3d_one_dlt3d -o ./results --dlt3d a.dlt3d b.dlt3d --pixels a.csv b.csv

# 3) Three cameras, real NTSC-derived fractional rate (120000/1001 Hz)
python -m vaila.rec3d_one_dlt3d \
  --dlt3d cam1.dlt3d cam2.dlt3d cam3.dlt3d \
  --pixels cam1.csv cam2.csv cam3.csv \
  --fps 119.88012001 -o ./out

# 4) BVH for Blender (Y/Z swapped) + a skeleton JSON for the companion
#    visualization script (presets in vaila/skeletons/)
python -m vaila.rec3d_one_dlt3d \
  --dlt3d cam1.dlt3d cam2.dlt3d \
  --pixels cam1.csv cam2.csv \
  --fps 100 -o ./out \
  --skeleton vaila/skeletons/mediapipe_pose33.json   # --swap-yz is the default

# 5) Simplified mesh-for-Blender export: one path per camera (the
#    Visualize-ID run dir) is enough, --pixels is derived automatically
python -m vaila.rec3d_one_dlt3d \
  --dlt3d c1.dlt3d c2.dlt3d c3.dlt3d \
  --mesh-source-dir c1_id/ c2_id/ c3_id/ \
  --export-mesh obj \
  --fps 119.88012001 -o ./out

# 6) Everything combined: 3 cameras, fractional fps, BVH swap, a
#    SAM3+DINOv3 skeleton, and mesh export in one run
python -m vaila.rec3d_one_dlt3d \
  --dlt3d c1.dlt3d c2.dlt3d c3.dlt3d \
  --mesh-source-dir c1_id/ c2_id/ c3_id/ --export-mesh ply \
  --fps 119.88012001 -o ./out \
  --skeleton vaila/skeletons/sam3dinov3_mhr70.json   # --swap-yz is the default

# 7) Show full CLI help (lists every flag with its default)
python -m vaila.rec3d_one_dlt3d --help

# 8) Launch the GUI instead (same 8 steps as "GUI mode" above)
python -m vaila.rec3d_one_dlt3d --gui
```

---

## Main functions

| Function | Description |
|----------|-------------|
| `rec3d_multicam` | Reconstruct one 3D point from multiple camera observations (DLT least squares). |
| `run_reconstruction` | Core logic: load DLT3D and pixel data, reconstruct, save CSV/.3d/C3D/BVH, optionally align+export mesh (used by GUI and CLI). |
| `reconstruct_mesh_sequence` | Per-frame Umeyama camera selection + mesh alignment + OBJ/PLY export + manifest. |
| `save_rec3d_as_c3d` | Save current reconstruction as C3D via file dialog (uses same C3D structure as batch output). |
| `run_rec3d_one_dlt3d` | GUI entry: dialogs then `run_reconstruction`. |

---

## Related modules

| Module | Role |
|--------|------|
| **dlt3d** | Compute DLT3D coefficients from calibration (pixel + 3D reference). |
| **mesh_alignment** | Umeyama similarity fit + OBJ/PLY I/O used by the mesh-export feature. |
| **sam3dinov3_visualize** | Produces the per-camera "Visualize ID" mesh-source directories consumed by `--mesh-source-dir`. |
| **vaila/skeletons/** | Ready-made `--skeleton` presets (MediaPipe, YOLO/COCO-17, SAM3+DINOv3 MHR70, Sapiens2 Goliath-308). |
| **readcsv_export** | CSV → C3D (used internally); batch convert. |
| **readc3d_export** | C3D → CSV; inspect C3D. |
| **viewc3d** / **viewc3d_pyvista** | Visualize C3D files. |

---

Part of *vailá* - Multimodal Toolbox  
[GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
