# rec2d

## Module information

| Field | Value |
|-------|--------|
| **Category** | Processing |
| **File** | `vaila/rec2d.py` |
| **Version** | 0.3.94 |
| **Author** | Paulo Santiago |
| **GUI** | Yes |
| **CLI** | Yes |

---

## Description

Batch 2D reconstruction using the **Direct Linear Transformation (DLT2D)** method, with DLT2D parameters that **vary per frame** (a DLT "matrix" — one row per frame, single camera). For a single fixed set of DLT2D parameters applied to every frame, use **rec2d_one_dlt2d** instead.

Every CSV file in the input directory is processed **independently** (each is a separate trial/subject using the same per-frame DLT2D matrix) — unlike the 3D family, 2D reconstruction from a single camera does not need to correlate multiple files.

---

## Input file formats

### DLT2D file

- CSV with **one row per frame**: `frame`, then 8 DLT2D coefficients.

### Pixel CSV (per file in the input directory)

- **Column labels are not inspected — only column order matters.** Column 0 is the frame identifier and every pair of columns after that is one marker's (x, y), regardless of header text (vailá `p1_x`/`p1_y`, SAM3, YOLO, MediaPipe named joints, etc.).
- Output always uses vailá's standard `Frame` label for column 0 (renamed regardless of the input's original label); coordinate column labels are preserved as-is from the input file.

---

## Output files

One reconstructed result **per input CSV**, all inside a shared timestamped subfolder: `vaila_rec2d_YYYYMMDD_HHMMSS/`.

| File | Description |
|------|-------------|
| `<name>_<timestamp>.2d` | 2D points: `Frame`, then the original coordinate column labels. |
| `<name>_<timestamp>.csv` | Same data as the `.2d` file. |

---

## GUI mode

Run with no arguments:

1. **DLT2D file** — one file with one row per frame.
2. **Input directory** — folder with one or more pixel CSVs to reconstruct.
3. **Output directory** — where to create the timestamped result folder.
4. **Data rate (Hz)** — recorded in the console/summary.

---

## CLI mode

**Required (headless):** `--dlt-file`, `--input-dir`, `--output-dir`, `--rate`.

| Argument | Description |
|----------|-------------|
| `--dlt-file` *FILE* | DLT2D parameter file (one row per frame). |
| `--input-dir` *DIR* | Directory containing one or more pixel CSVs to reconstruct. |
| `--output-dir` *DIR* | Output directory; a timestamped `vaila_rec2d_*` subfolder is created here. |
| `--rate` *HZ* | Data frequency in Hz (recorded in the console summary). Accepts fractional rates, e.g. `119.88012001`. |

### Example

```bash
python vaila/rec2d.py \
  --dlt-file calib.dlt2d \
  --input-dir ./trials \
  --output-dir ./out \
  --rate 100
```

---

## Main functions

| Function | Description |
|----------|-------------|
| `rec2d` | Reconstruct 2D real-world coordinates from pixel coordinates using DLT2D parameters. |
| `process_files_in_directory` | Core logic: per-frame DLT2D lookup, column-order-based pixel reading, save `.2d`/`.csv`. |
| `run_rec2d` | GUI/CLI entry point. |

---

## Related modules

| Module | Role |
|--------|------|
| **dlt2d** | Compute DLT2D coefficients from calibration (pixel + 2D reference). |
| **rec2d_one_dlt2d** | Same DLT2D method with one fixed set of parameters for all frames. |

---

Part of **vailá** - Multimodal Toolbox
[GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
