# dlt3d

## Module information

| Field | Value |
|-------|--------|
| **Category** | Processing |
| **File** | `vaila/dlt3d.py` |
| **Version** | 0.3.120 |
| **Author** | Paulo Roberto Pereira Santiago |
| **GUI** | Yes |
| **CLI** | Yes |

---

## Description

Calculates **Direct Linear Transformation (DLT3D)** parameters (11 coefficients) from a pixel-coordinate calibration file and a corresponding real-world 3D reference (REF3D) file, one set of parameters per frame.

The script can also generate a **REF3D template** (`_x`, `_y`, `_z` columns) from a pixel file for you to fill in with real-world 3D coordinates.

### REF3D file formats (auto-detected)

`read_ref3d_file` accepts three layouts and normalizes them internally to **format 1** (wide CSV):

| Format | Layout | Example |
|--------|--------|---------|
| **1** (default) | Header `frame,p1_x,p1_y,p1_z,…`; one or more rows | `tests/DLT3D_and_Rec3d/ref3d_realworld/ref3d_realworld_format1.ref3d` |
| **2** | No header; one `x,y,z` row per point (row order → p1…pN) | `…/ref3d_realworld_format2.ref3d` |
| **3** | No header; one `index,x,y,z` row per point (`index` → pN) | `…/ref3d_realworld_format3.ref3d` |

Formats 2 and 3 are converted to format 1 before DLT3D; downstream `.dlt3d` output is identical for the same physical points.

### Point matching (pixel ↔ REF3D)

Points are correlated by **label** — the point prefix before the underscore (e.g. `p3` in `p3_x`/`p3_y`/`p3_z`), not by column order. This lets the REF3D file define **more** calibration points than a given pixel/video file actually tracks; only the points common to both files are used.

> **v0.3.93 fix:** earlier versions derived the point range from the pixel file only and assumed the REF3D file had the same range, raising a raw `KeyError` as soon as they diverged. Point matching is now based on the actual intersection of point labels in both files, and DLT3D requires **at least 6 common points** (11 unknowns, 2 equations per point) — with fewer, a clear message is printed and no `.dlt3d` file is written instead of crashing.

### Frame matching (pixel ↔ REF3D)

- **Single reference mode**: the REF3D file has exactly 1 row — those 3D coordinates are used for every frame in the pixel file.
- **Per-frame mode**: the REF3D file has multiple rows — matched by the `frame` column.

---

## Output files

| File | Description |
|------|-------------|
| `<pixel_basename>.ref3d` | REF3D template (only when `--create-ref` is used); `p{i}_x`, `p{i}_y`, `p{i}_z` columns, values cleared. |
| `<pixel_basename>.dlt3d` | CSV: `frame`, then 11 DLT3D coefficients, one row per frame. |

---

## GUI mode

Run with no arguments:

1. Select the **pixel coordinate CSV** file.
2. Choose whether to **create a REF3D template**, or select an **existing REF3D** file with real-world 3D coordinates.
3. If a template was created, edit it with real 3D coordinates and re-run to compute the `.dlt3d` file.

---

## CLI mode

| Argument | Description |
|----------|-------------|
| `--pixel` *FILE* | Pixel coordinate CSV file. |
| `--real` *FILE* | Real-world coordinate REF3D file (required unless `--create-ref`). |
| `--create-ref` | Create a REF3D template from `--pixel` instead of computing DLT3D parameters. |

### Examples

```bash
# Create a REF3D template from a pixel file
python vaila/dlt3d.py --pixel calib.csv --create-ref

# Compute DLT3D parameters (one row per frame)
python vaila/dlt3d.py --pixel calib.csv --real calib.ref3d
```

---

## Main functions

| Function | Description |
|----------|-------------|
| `calculate_dlt3d_params` | Compute 11 DLT3D parameters from real-world 3D and pixel coordinate pairs (least squares). |
| `read_ref3d_file` | Read REF3D formats 1–3; normalize to wide format 1; validate `_x`/`_y`/`_z` columns. |
| `detect_ref3d_format` | Return 1, 2, or 3 for a REF3D path (or `None` if unrecognized). |
| `normalize_ref3d_to_format1` | Load any supported REF3D variant as a format-1 DataFrame. |
| `process_files` | Core logic: match points by label (≥6 common points required), match frames, compute DLT3D per frame. |
| `save_dlt_parameters` | Write the `.dlt3d` CSV. |
| `main` | GUI/CLI entry point. |

---

## Related modules

| Module | Role |
|--------|------|
| **rec3d** | Reconstruct 3D coordinates using per-frame DLT3D parameters (this module's output) across multiple cameras. |
| **rec3d_one_dlt3d** | Reconstruct 3D coordinates using one fixed DLT3D row per camera. |
| **dlt2d** | Same idea for 2D (8 parameters, REF2D file). |

---

Part of **vailá** - Multimodal Toolbox
[GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
