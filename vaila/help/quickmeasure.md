# quickmeasure

## Module information

| Field | Value |
|-------|--------|
| **Category** | Processing |
| **File** | `vaila/quickmeasure.py` |
| **Version** | 0.3.127 |
| **Updated** | 07 September 2026 |
| **Author** | Paulo Santiago |
| **GUI** | Yes (embedded in `getpixelvideo.py`) |
| **CLI** | Yes (`python -m vaila.quickmeasure --points-csv ...`) |

---

## Description

Kinovea-style quick on-image measurement engine for `getpixelvideo.py`: **calibrate first**, then click points on the video and classify the current point set as **Distance**, **Area**, **Angle**, **Velocity**, or **Acceleration**. This module owns all quick-measurement *state*, *math*, and (for the pygame-based pieces) the *modal submenu + overlay rendering* — it does **not** reimplement calibration math, and `getpixelvideo.py` only owns the integration glue (a mode-toggle key, one click handler, one draw call, one help-text block), so the already-large host file stays a thin integration layer.

## Calibration first (v0.3.127)

Pressing **Q** or the **QMeas** button on an uncalibrated session starts with calibration:

| Mode | Input | Model |
|------|--------|-------|
| **Line** (`1`) | 2 clicks + typed length | Isotropic scale; origin at first click. |
| **Plane** (`2`) | 4 clicks + typed width/height | 8-parameter DLT2D homography. |
| **REF3D** (`3`) | `.ref3d` (mode1 wide / mode2 `point,x,y,z` / mode3 bare `x,y,z`) + drop axis (`z`→XY, `y`→XZ, `x`→YZ) + pixel CSV **or** guided clicks with scheme overlay | Planar DLT2D via `rec2d` (duplicate collapsed points are deduped). |
| **Skip** (`0`) | — | Stay in pixels (`px`). |

### Live measure modes (digit keys after Q)

Once calibrated (or skipped), press a digit on the **video** window — not only in the Enter menu:

| Key | Mode | Clicks | On-image |
|-----|------|--------|----------|
| **1** | distance | 2 (auto) | Line + length for each pair |
| **2** | area | ≥3, then **Enter** | Polygon + area |
| **3** | angle | 3 (auto, vertex in the middle) | Rays + degrees |
| **4** | velocity | 2 on **different frames** | FPS auto-detected; override with `I` or **FPS … Hz** |
| **5** | acceleration | 3 on distinct frames | Uses the same automatic/manual FPS |
| **6–0** | reserved | — | — |

Each completed set is stored with `result_frame` and pixel geometry. The first calibration/result save creates one `processed_quickmeasure_<timestamp>/` next to the video. Every later **S** save in that video session updates the same directory, so all measurements remain together. It contains the combined results CSV, per-type files (`…_results_distance.csv`, …), and `quickmeasure_report.html`.

While calibrating from REF3D without a CSV, a scheme panel highlights the next world point to click on the image. Menu **R** / **C** can also load REF3D / REF2D / `.dlt2d` later.

Calibration reuses:

- **`dlt2d.py`** / **`dlt3d.normalize_ref3d_to_format1`** for REF3D modes
- **`rec2d_one_dlt2d.py`** `rec2d()` for pixel→real

Single-video sessions stay **DLT2D**. Stereo **DLT3D** remains a batch/CLI workflow via `rec3d_one_dlt3d.py`.

---

## Point-set convention

Fixed and documented, not user-configurable, so a result is reproducible from the clicks alone:

| Measurement | Uses |
|-------------|------|
| Distance | Last 2 clicked points |
| Area | All clicked points, in click order (shoelace polygon) |
| Angle | 3 points (vertex in the middle) or 4 points (two lines) |
| Velocity | Last 2 clicked points (needs fps + 2 distinct frames) |
| Acceleration | Last 3 clicked points (needs fps + 3 distinct frames) |

## Units

- **Uncalibrated session:** pixel units (`px`, `px/s`, `px/s^2`).
- **Calibrated session** (a DLT2D loaded): whatever unit the `.ref2d` file used (assumed metres — `m`, `m/s`, `m/s^2` — matching `dlt2d.py`/`rec2d.py`'s own convention); the caller may override `unit_label`.

---

## Usage from getpixelvideo.py

| Key | Action |
|-----|--------|
| **Q** / **QMeas** button | Toggle Quick Measure mode — starts with the calibration prompt when the session is not calibrated yet |
| Left-click | Add a calibration point (while calibrating) or a measure point (after) |
| Right-click | Undo the last point |
| Middle-click | Pan (unchanged) |
| **Enter** | While calibrating: (re)open the real-measurement prompt. After: open the classification submenu (`1` Distance, `2` Area, `3` Velocity, `4` Acceleration, `S` save CSVs, `C` load calibration from file, `X` clear points, `Esc` close menu) |
| **Backspace** | Clear all quick-measure points |

`Esc` in the main tool always means save-and-quit the whole application; it is never repurposed to exit Quick Measure mode — press `Q` again instead.

---

## Saved CSVs and recomputation

`S` in the submenu updates the session's single timestamped `processed_quickmeasure_YYYYMMDD_HHMMSS/` folder next to the video. Open `quickmeasure_report.html` in any browser for a didactic summary of the current results, measurement definitions and formulas, units, file purposes, and CSV column meanings.

| File | Columns |
|------|---------|
| `<stem>_quickmeasure_points.csv` | `point_id, frame, x_px, y_px, x_real, y_real, unit, calibration_kind` |
| `<stem>_quickmeasure_calibration.csv` | calibration mode, unit, typed measures, calibration clicks (pixel + real), DLT2D parameters |
| `<stem>_quickmeasure_results.csv` | Matrix layout: one result per row; scalar metadata plus repeated `point_N_id, point_N_frame, point_N_x_px, point_N_y_px, point_N_x_real, point_N_y_real` columns. |
| `<stem>_quickmeasure_results_<type>.csv` | Same matrix columns and width as the combined file, filtered to one measurement type. |
| `quickmeasure_report.html` | Browser-readable live report explaining all results, measurements, metrics, files, units, and columns. |
| `README_quickmeasure.txt` | Compact file list and recomputation command. |

The result CSVs never pack several frames, IDs, or coordinates into one cell. There are no space-delimited lists or `x,y` strings: each scalar occupies its own matrix cell. The number of `point_N_*` groups equals the largest point set in that saved session; shorter measurements leave the extra numeric cells empty.

Because each point carries its calibrated real-world coordinates, every measurement can be recomputed from the saved file alone — without the video:

```bash
uv run python -m vaila.quickmeasure --points-csv processed_quickmeasure_*/clip_quickmeasure_points.csv --measure distance
uv run python -m vaila.quickmeasure --points-csv POINTS.csv --measure velocity --fps 240
uv run python -m vaila.quickmeasure --points-csv POINTS.csv --measure area --point-ids 1,2,3,4 --out results.csv
```

---

## Main functions / classes

| Name | Description |
|------|-------------|
| `QuickMeasureCalibration` | Scale or DLT2D calibration; `pixel_to_real()`, `describe()`; `from_line_clicks()`, `from_plane_clicks()`, `from_point_correspondences()`, `from_dlt2d_file()`, `from_calibration_points()`. |
| `CalibrationDraft` | Calibration-first state machine: `required_points`, `required_measures`, `add_point`, `undo_last`, `instructions()`, `build(measures)`. |
| `QuickMeasureSession` | Click-session state (`fps`, `calibration`, `points`, `results`); `add_point`, `undo_last`, `clear`, `measure(kind)`, `points_dataframe`, `calibration_dataframe`, `results_dataframe`, `save_session`. |
| `finish_calibration_draft` | Prompts for the real measurement(s) through an injected `ask_text` callback and builds the calibration. |
| `session_from_points_csv` / `measure_from_points_csv` | Rebuild a session and recompute a measurement from a saved points CSV. |
| `main` | CLI entry point (`python -m vaila.quickmeasure`). |
| `format_result` | One-line human-readable string for a measurement result dict. |
| `draw_quickmeasure_overlay` / `draw_calibration_overlay` | Draw measure points (magenta) and calibration points + instruction banner (cyan). |
| `show_quickmeasure_menu` | Blocking modal submenu (own pygame event loop, mirrors `show_help_dialog`'s pattern). |

---

## Related modules

| Module | Role |
|--------|------|
| **dlt2d** | Compute DLT2D coefficients from calibration (pixel + `.ref2d` reference). |
| **rec2d_one_dlt2d** | Apply one fixed set of DLT2D parameters to pixel coordinates (reused directly for `pixel_to_real`). |
| **rec3d_one_dlt3d** | Batch stereo DLT3D reconstruction — the intended path for two-video 3D quick measurements (not covered by this module). |
| **getpixelvideo** | Host tool; owns only the hotkeys, click handler, overlay draw call, and help text for this feature. |

---

Part of **vailá** - Multimodal Toolbox
[GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
