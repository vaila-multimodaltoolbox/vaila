# rec3d_one_dlt3d

## Module information

| Field | Value |
|-------|--------|
| **Category** | Processing |
| **File** | `vaila/rec3d_one_dlt3d.py` |
| **Version** | 0.0.5 |
| **Author** | Paulo Santiago |
| **GUI** | Yes |
| **CLI** | Yes |

---

## Description

Batch 3D reconstruction using the **Direct Linear Transformation (DLT)** method with multiple cameras. For each camera you provide:

- One **DLT3D parameter file** (11 coefficients per camera, e.g. from the `dlt3d` module).
- One **pixel-coordinate CSV** with columns: `frame`, `p1_x`, `p1_y`, `p2_x`, `p2_y`, ..., `pN_x`, `pN_y` (vailá standard).

Frames common to all pixel files are reconstructed; results are written to a **timestamped subfolder** in the chosen output directory.

---

## Input file formats

### DLT3D file

- CSV with one row of 11 DLT coefficients (e.g. produced by the vailá **dlt3d** module).
- **One file per camera**; order must match the pixel file order.

### Pixel CSV

- Header: `frame,p1_x,p1_y,p2_x,p2_y,...,pN_x,pN_y`.
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

## GUI mode

Run with **no arguments** or with `--gui`:

1. **Number of cameras** — e.g. 2.
2. **DLT3D files** — One file dialog per camera; each file can be in a different directory.
3. **Pixel CSV files** — One file dialog per camera; again, each can be from a different directory.
4. **Output directory** — Where to create the timestamped result folder.
5. **Data rate (Hz)** — Point data rate for C3D/CSV (e.g. 60, 100).

---

## CLI mode

**Required:** `--dlt3d`, `--pixels`, `--output`.  
**Optional:** `--fps`, `--gui`.

| Argument | Description |
|----------|-------------|
| `--dlt3d` *FILE* [*FILE* ...] | DLT3D parameter files (one per camera); order must match `--pixels`. |
| `--pixels` *FILE* [*FILE* ...] | Pixel coordinate CSV files (one per camera); order must match `--dlt3d`. |
| `--fps` *HZ* | Point data rate in Hz (default: 100). |
| `-o`, `--output` *DIR* | Output directory; a timestamped subfolder will be created here. |
| `--gui` | Launch GUI instead of CLI. |

### Examples

```bash
# Two cameras, 60 Hz, output under ./out
python -m vaila.rec3d_one_dlt3d --dlt3d cam1.dlt3d cam2.dlt3d --pixels cam1.csv cam2.csv --fps 60 --output ./out

# Short form for output
python -m vaila.rec3d_one_dlt3d -o ./results --dlt3d a.dlt3d b.dlt3d --pixels a.csv b.csv

# Show full CLI help
python -m vaila.rec3d_one_dlt3d --help

# Launch GUI
python -m vaila.rec3d_one_dlt3d --gui
```

---

## Main functions

| Function | Description |
|----------|-------------|
| `rec3d_multicam` | Reconstruct one 3D point from multiple camera observations (DLT least squares). |
| `run_reconstruction` | Core logic: load DLT3D and pixel data, reconstruct, save CSV/.3d/C3D (used by GUI and CLI). |
| `save_rec3d_as_c3d` | Save current reconstruction as C3D via file dialog (uses same C3D structure as batch output). |
| `run_rec3d_one_dlt3d` | GUI entry: dialogs then `run_reconstruction`. |

---

## Related modules

| Module | Role |
|--------|------|
| **dlt3d** | Compute DLT3D coefficients from calibration (pixel + 3D reference). |
| **readcsv_export** | CSV → C3D (used internally); batch convert. |
| **readc3d_export** | C3D → CSV; inspect C3D. |
| **viewc3d** / **viewc3d_pyvista** | Visualize C3D files. |

---

Part of **vailá** - Multimodal Toolbox  
[GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
