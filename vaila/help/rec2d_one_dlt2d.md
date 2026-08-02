# rec2d_one_dlt2d

## Module information

| Field | Value |
|-------|--------|
| **Category** | Processing |
| **File** | `vaila/rec2d_one_dlt2d.py` |
| **Version** | 0.3.93 |
| **Author** | Paulo Santiago |
| **GUI** | Yes |
| **CLI** | Yes |

---

## Description

Batch 2D reconstruction using the **Direct Linear Transformation (DLT2D)** method with **one fixed set of DLT2D parameters** (the file's first row) applied to every frame of every pixel CSV in the input directory. For DLT2D parameters that vary per frame (a DLT "matrix"), use **rec2d** instead.

Every CSV file in the input directory is processed independently (each is a separate trial/subject reconstructed with the same fixed DLT2D parameters).

---

## Input file formats

### DLT2D file

- CSV with at least one row of 8 DLT2D coefficients. Only the **first row** is used, regardless of how many rows the file has.

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

1. **DLT2D file** — a single file; only its first row is used.
2. **Input directory** — folder with one or more pixel CSVs to reconstruct.
3. **Output directory** — where to create the timestamped result folder.

---

## CLI mode

**Required (headless):** `--dlt-file`, `--input-dir`, `--output-dir`.

| Argument | Description |
|----------|-------------|
| `--dlt-file` *FILE* | DLT2D parameter file (`*.dlt2d`); only the first row is used. |
| `--input-dir` *DIR* | Directory containing one or more pixel CSVs to reconstruct. |
| `--output-dir` *DIR* | Output directory; a timestamped `vaila_rec2d_*` subfolder is created here. |

### Example

```bash
python vaila/rec2d_one_dlt2d.py \
  --dlt-file calib.dlt2d \
  --input-dir ./trials \
  --output-dir ./out
```

---

## Main functions

| Function | Description |
|----------|-------------|
| `rec2d` | Reconstruct 2D real-world coordinates from pixel coordinates using DLT2D parameters. |
| `process_files_in_directory` | Core logic: fixed DLT2D lookup, column-order-based pixel reading, save `.2d`/`.csv`. |
| `run_rec2d_one_dlt2d` | GUI/CLI entry point. |

---

## Related modules

| Module | Role |
|--------|------|
| **dlt2d** | Compute DLT2D coefficients from calibration (pixel + 2D reference). |
| **rec2d** | Same DLT2D method with parameters that vary per frame. |

---

Part of **vailá** - Multimodal Toolbox
[GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
