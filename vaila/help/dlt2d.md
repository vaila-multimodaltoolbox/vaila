# dlt2d

## Module information

| Field | Value |
|-------|--------|
| **Category** | Processing |
| **File** | `vaila/dlt2d.py` |
| **Version** | 0.3.93 |
| **Author** | Paulo Santiago |
| **GUI** | Yes |
| **CLI** | Yes |

---

## Description

Calculates **Direct Linear Transformation (DLT2D)** parameters (8 coefficients) from a pixel-coordinate calibration file and a corresponding real-world 2D reference (REF2D) file, one set of parameters per frame.

The script can also generate a **REF2D template** from a pixel file (same frames/points, coordinates cleared) for you to fill in with real-world values.

### Point matching (pixel ↔ REF2D)

Points are correlated by **label** — the point prefix before the underscore (e.g. `p3` in `p3_x`/`p3_y`), not by column order. This lets the REF2D file define **more** calibration points than a given pixel/video file actually tracks; only the points common to both files are used for each frame's DLT2D solve (minimum 4 non-collinear points).

### Frame matching (pixel ↔ REF2D)

Two modes, detected automatically:

- **Single reference mode**: the REF2D file has exactly 1 row — those coordinates are used for every frame in the pixel file.
- **Per-frame mode**: the REF2D file has the same number of rows as the pixel file — matched row by row.

If the REF2D file has more than 1 row **and** a different row count than the pixel file, no DLT2D parameters are computed; a clear warning is printed and no `.dlt2d` file is written (instead of raising an exception).

---

## Output files

| File | Description |
|------|-------------|
| `<pixel_basename>.ref2d` | REF2D template (only when `--create-ref` is used); same frames as the pixel file, coordinates cleared. |
| `<pixel_basename>.dlt2d` | CSV: `frame`, then 8 DLT2D coefficients, one row per frame. |

---

## GUI mode

Run with no arguments:

1. Select the **pixel coordinate CSV** file.
2. Choose whether to **create a REF2D template** from it, or select an **existing REF2D** file with real-world coordinates.
3. If a template was created, edit it with real coordinates and re-run to compute the `.dlt2d` file.

---

## CLI mode

| Argument | Description |
|----------|-------------|
| `--pixel` *FILE* | Pixel coordinate CSV file. |
| `--real` *FILE* | Real-world coordinate REF2D file (required unless `--create-ref`). |
| `--create-ref` | Create a REF2D template from `--pixel` instead of computing DLT2D parameters. |

### Examples

```bash
# Create a REF2D template from a pixel file
python vaila/dlt2d.py --pixel calib.csv --create-ref

# Compute DLT2D parameters (one row per frame)
python vaila/dlt2d.py --pixel calib.csv --real calib.ref2d
```

---

## Main functions

| Function | Description |
|----------|-------------|
| `dlt2d` | Compute 8 DLT2D parameters from real-world and pixel coordinate pairs. |
| `create_ref2d_template` | Generate a `.ref2d` template from a pixel file. |
| `process_files` | Core logic: match points by label, match frames (single-ref or per-frame), compute DLT2D per frame. |
| `save_dlt_parameters` | Write the `.dlt2d` CSV. |
| `run_dlt2d` | GUI/CLI entry point. |

---

## Related modules

| Module | Role |
|--------|------|
| **rec2d** | Reconstruct 2D coordinates using per-frame DLT2D parameters (this module's output). |
| **rec2d_one_dlt2d** | Reconstruct 2D coordinates using one fixed DLT2D row. |
| **dlt3d** | Same idea for 3D (11 parameters, REF3D file). |

---

Part of **vailá** - Multimodal Toolbox
[GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
