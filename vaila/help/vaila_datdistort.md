# vaila_datdistort

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila/vaila_datdistort.py`
- **Version:** 0.3.114
- **Updated:** 25 August 2026
- **Author:** Prof. Dr. Paulo R. P. Santiago
- **GUI Interface:** ✅ Yes
- **CLI Support:** ✅ Yes

## 📖 Description

This tool applies lens distortion correction to 2D coordinates from DAT/CSV files using the same camera calibration parameters as vaila_lensdistortvideo.py. The parameters file (TOML with fx, fy, cx, cy, k1, k2, k3, p1, p2) is never processed; the output CSV "frame" column is written as integer.

## How to run

### GUI mode (default)

Run without arguments; dialogs will ask for the parameters file and input directory.

```bash
# From project root (recommended)
uv run vaila/vaila_datdistort.py
python vaila/vaila_datdistort.py
```

### CLI mode

**From project root:**

```bash
# Using uv (recommended)
uv run vaila/vaila_datdistort.py --params_file /path/to/distortionparameters.toml --input /path/to/file_or_dir

# Optional: specify output directory (otherwise creates input_dir/distorted_TIMESTAMP)
uv run vaila/vaila_datdistort.py --params_file /path/to/params.toml --input /path/to/data --output_dir /path/to/output
```

**As Python module:**

```bash
uv run python -m vaila.vaila_datdistort --params_file /path/to/params.toml --input /path/to/file_or_dir [--output_dir /path/to/output]
python -m vaila.vaila_datdistort --params_file /path/to/params.toml --input /path/to/file_or_dir [--output_dir /path/to/output]
```

**CLI help:**

```bash
uv run vaila/vaila_datdistort.py --help
python -m vaila.vaila_datdistort --help
```

### Arguments

| Argument       | Description |
|----------------|-------------|
| `--params_file` | Path to the camera calibration parameters TOML file (required in CLI). |
| `--input`       | Single CSV/DAT file or directory containing CSV/DAT files to process. |
| `--output_dir`  | (Optional) Output directory. If omitted, a new subdirectory `distorted_TIMESTAMP` is created under the input path; if given, files are written directly there. |

- The parameters file is **excluded from the batch** (never distortion-corrected).
- Output CSV **"frame"** column is written as **integer**.

## 🔧 Main Functions

- `load_distortion_parameters` – Load calibration from TOML
- `undistort_points` – Apply OpenCV undistort to 2D points
- `process_dat_file` – Process one DAT/CSV file
- `select_file` / `select_directory` – GUI file/dir selection
- `run_datdistort` – Main entry (GUI or CLI)

---

🔗 **Part of vailá - Multimodal Toolbox**  
🌐 [GitHub](https://github.com/paulopreto/vaila-multimodaltoolbox)
