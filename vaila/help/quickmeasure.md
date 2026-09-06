# quickmeasure

## Module information

| Field | Value |
|-------|--------|
| **Category** | Processing |
| **File** | `vaila/quickmeasure.py` |
| **Version** | 0.3.122 |
| **Author** | Paulo Santiago |
| **GUI** | Yes (embedded in `getpixelvideo.py`) |
| **CLI** | No |

---

## Description

Kinovea-style quick on-image measurement engine for `getpixelvideo.py`: click points on the video, then classify the current point set as **Distance**, **Area**, **Velocity**, or **Acceleration**. This module owns all quick-measurement *state*, *math*, and (for the pygame-based pieces) the *modal submenu + overlay rendering* — it does **not** reimplement calibration math, and `getpixelvideo.py` only owns the integration glue (a mode-toggle key, one click handler, one draw call, one help-text block), so the already-large host file stays a thin integration layer.

Calibration reuses the existing DLT2D pipeline exactly the way it already works elsewhere in vailá:

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
| **Q** | Toggle Quick Measure mode (disables Labeling/1 Line/Sequential mode while active) |
| Left-click | Add a point |
| Right-click | Undo the last point |
| Middle-click | Pan (unchanged) |
| **Enter** | Open the classification submenu (`1` Distance, `2` Area, `3` Velocity, `4` Acceleration, `C` load calibration, `X` clear points, `Esc` close menu) |
| **Backspace** | Clear all quick-measure points |

`Esc` in the main tool always means save-and-quit the whole application; it is never repurposed to exit Quick Measure mode — press `Q` again instead.

---

## Main functions / classes

| Name | Description |
|------|-------------|
| `QuickMeasureCalibration` | Holds 8 DLT2D coefficients; `pixel_to_real()`; `from_dlt2d_file()`; `from_calibration_points()`. |
| `QuickMeasureSession` | Click-session state (`fps`, `calibration`, `points`); `add_point`, `undo_last`, `clear`, `measure_distance`, `measure_area`, `measure_velocity`, `measure_acceleration`, `measure(kind)`. |
| `format_result` | One-line human-readable string for a measurement result dict. |
| `draw_quickmeasure_overlay` | Draws clicked points + connecting lines on the pygame `screen`. |
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
