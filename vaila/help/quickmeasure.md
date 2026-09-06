# quickmeasure

## Module information

| Field | Value |
|-------|--------|
| **Category** | Processing |
| **File** | `vaila/quickmeasure.py` |
| **Version** | 0.3.124 |
| **Author** | Paulo Santiago |
| **GUI** | Yes (embedded in `getpixelvideo.py`) |
| **CLI** | Yes (`python -m vaila.quickmeasure --points-csv ...`) |

---

## Description

Kinovea-style quick on-image measurement engine for `getpixelvideo.py`: **calibrate first**, then click points on the video and classify the current point set as **Distance**, **Area**, **Velocity**, or **Acceleration**. This module owns all quick-measurement *state*, *math*, and (for the pygame-based pieces) the *modal submenu + overlay rendering* — it does **not** reimplement calibration math, and `getpixelvideo.py` only owns the integration glue (a mode-toggle key, one click handler, one draw call, one help-text block), so the already-large host file stays a thin integration layer.

## Calibration first (v0.3.124)

Pressing **Q** or the **QMeas** button on an uncalibrated session starts with calibration, built from clicks alone — no file picking:

| Mode | Clicks | You then type | Model |
|------|--------|---------------|-------|
| **Line** (`1`) | 2 clicks on a segment of known length | the real length | Isotropic scale `real / pixel`; origin at the first click, `+x` right, `+y` up. Valid for a plane parallel to the sensor. |
| **Plane** (`2`) | 4 clicks around a known rectangle (click 1 = origin, click 2 = width direction) | width and height | 8-parameter DLT2D homography — perspective corrected. |
| **Skip** (`0`) | — | — | Session stays in pixels (`px`). |

The unit label (`m`, `cm`, …) is typed in the same prompt sequence and is written into every CSV row. While calibrating, the clicked calibration points are drawn in cyan with a banner telling you what to click next; right-click undoes the last calibration click.

Loading an existing calibration from files still works (`C` in the submenu). Calibration reuses the existing DLT2D pipeline exactly the way it already works elsewhere in vailá:

- **`dlt2d.py`** (`dlt2d()`, `process_files()`) computes the 8 DLT2D coefficients from a pixel-calibration CSV + a `.ref2d` real-world reference file.
- **`rec2d_one_dlt2d.py`**'s `rec2d()` applies those fixed coefficients to convert a clicked pixel to real-world coordinates.

Single-video sessions can only ever be calibrated with **DLT2D** — one image plane is 2D by construction. Stereo **DLT3D** triangulation (two synchronized cameras/videos, `rec3d_multicam()` from `rec3d.py`) is intentionally out of scope for this module's live single-video click session; it remains a batch/CLI workflow via `rec3d_one_dlt3d.py`.

---

## Point-set convention

Fixed and documented, not user-configurable, so a result is reproducible from the clicks alone:

| Measurement | Uses |
|-------------|------|
| Distance | Last 2 clicked points |
| Area | All clicked points, in click order (shoelace polygon) |
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

`S` in the submenu writes a timestamped `processed_quickmeasure_YYYYMMDD_HHMMSS/` folder next to the video:

| File | Columns |
|------|---------|
| `<stem>_quickmeasure_points.csv` | `point_id, frame, x_px, y_px, x_real, y_real, unit, calibration_kind` |
| `<stem>_quickmeasure_calibration.csv` | calibration mode, unit, typed measures, calibration clicks (pixel + real), DLT2D parameters |
| `<stem>_quickmeasure_results.csv` | `result_id, type, value, unit, n_points, frames, point_ids, fps` |

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
