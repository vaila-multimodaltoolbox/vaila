# rec3d_one_dlt3d

## Module Information

- **Category:** Processing
- **File:** `vaila/rec3d_one_dlt3d.py`
- **Version:** 0.0.4
- **Author:** Paulo Santiago
- **GUI:** Yes | **CLI:** Yes

## Description

Optimized batch 3D reconstruction using the Direct Linear Transformation (DLT) method with multiple cameras. Each camera has a corresponding DLT3D parameter file (one set of 11 parameters per file) and a pixel coordinate CSV file. The pixel files use vailá's standard header: `frame,p1_x,p1_y,p2_x,p2_y,...,pN_x,pN_y`. Output includes CSV, .3d, and C3D files (meters and millimeters).

## GUI mode

Run with no arguments or with `--gui`. Steps:

1. **Number of cameras** — Enter how many cameras (e.g. 2).
2. **DLT3D files** — One file dialog per camera; you can choose each file from a **different directory** (camera 1 from folder A, camera 2 from folder B).
3. **Pixel CSV files** — One file dialog per camera; again each can be from a different directory.
4. **Output directory** — Where to write the results (a timestamped subdir is created).
5. **Data rate (Hz)** — Point data rate (e.g. 60, 100).

## CLI mode

Pass `--dlt3d`, `--pixels`, and `--output`; optionally `--fps`. Order of files must match (first DLT3D with first pixel CSV, etc.).

| Argument | Description |
|----------|-------------|
| `--dlt3d` | DLT3D parameter files (one per camera), space-separated |
| `--pixels` | Pixel coordinate CSV files (one per camera), space-separated |
| `--fps` | Point data rate in Hz (default: 100) |
| `-o`, `--output` | Output directory for reconstruction results |
| `--gui` | Launch GUI instead of CLI |

**Example:**

```bash
python -m vaila.rec3d_one_dlt3d --dlt3d cam1.dlt3d cam2.dlt3d --pixels cam1.csv cam2.csv --fps 60 --output ./out
```

## Main functions

- `rec3d_multicam` — Reconstruct one 3D point from multiple camera observations.
- `run_reconstruction` — Core logic: load DLT3D and pixel data, reconstruct, save CSV/.3d/C3D (used by GUI and CLI).
- `save_rec3d_as_c3d` — Legacy: save reconstruction as C3D via file dialog.
- `run_rec3d_one_dlt3d` — GUI entry (dialogs then run_reconstruction).

---

Part of vailá - Multimodal Toolbox  
[GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
