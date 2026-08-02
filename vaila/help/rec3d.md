# rec3d

## Module information

| Field | Value |
|-------|--------|
| **Category** | Processing |
| **File** | `vaila/rec3d.py` |
| **Version** | 0.3.94 |
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

Unlike `rec3d_one_dlt3d`, this module does not export C3D/BVH; use `rec3d_one_dlt3d` or `readcsv_export` if you need those formats from the resulting CSV.

---

## GUI mode

Run with no arguments:

1. **DLT3D files** — select one file per camera (each with one row per frame).
2. **Input directory** — a single folder containing exactly one pixel CSV per camera.
3. **Output directory** — where to create the timestamped result folder.
4. **Data rate (Hz)** — recorded in the console/summary (not used for interpolation).

---

## CLI mode

**Required (headless):** `--dlt-files`, `--input-dir`, `--output-dir`, `--rate`.

| Argument | Description |
|----------|-------------|
| `--dlt-files` *FILE* [*FILE* ...] | DLT3D parameter files, one per camera (each with one row per frame). |
| `--input-dir` *DIR* | Directory containing exactly one pixel CSV per camera, matching `--dlt-files` in count. |
| `--output-dir` *DIR* | Output directory; a timestamped `vaila_rec3d_*` subfolder is created here. |
| `--rate` *HZ* | Data frequency in Hz (recorded in the console summary). Accepts fractional rates, e.g. `119.88012001`. |

### Example

```bash
# Two moving/re-calibrated cameras, one pixel CSV each in ./cams
python vaila/rec3d.py \
  --dlt-files cam1_matrix.dlt3d cam2_matrix.dlt3d \
  --input-dir ./cams \
  --output-dir ./out \
  --rate 100
```

---

## Main functions

| Function | Description |
|----------|-------------|
| `rec3d_multicam` | Reconstruct one 3D point from multiple camera observations (DLT least squares). Shared with `rec3d_one_dlt3d`. |
| `load_pixel_csv_positional` | Column-order-based pixel CSV reader (frame + N marker x,y pairs), ignoring header labels. Shared with `rec3d_one_dlt3d`. |
| `find_common_frames` | Sorted intersection of frame numbers across camera pixel files. Shared with `rec3d_one_dlt3d`. |
| `process_files_in_directory` | Core logic: correlate N camera pixel files by frame, look up each camera's per-frame DLT3D parameters, reconstruct, save CSV/.3d. |
| `run_rec3d` | GUI/CLI entry point. |

---

## Related modules

| Module | Role |
|--------|------|
| **dlt3d** | Compute DLT3D coefficients from calibration (pixel + 3D reference), one row per frame when the camera moves. |
| **rec3d_one_dlt3d** | Same DLT3D method with one fixed set of parameters per camera; also exports C3D/BVH. |
| **readcsv_export** | CSV → C3D (used internally by `rec3d_one_dlt3d`); batch convert. |

---

Part of **vailá** - Multimodal Toolbox
[GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
