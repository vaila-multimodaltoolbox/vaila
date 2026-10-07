# quickmeasure

## Module information

| Field | Value |
|-------|--------|
| **Category** | Processing |
| **File** | `vaila/quickmeasure.py` |
| **Version** | 0.4.9 |
| **Updated** | 07 October 2026 |
| **Author** | Paulo Santiago |
| **GUI** | Yes (embedded in `getpixelvideo.py`) |
| **CLI** | Yes (`python -m vaila.quickmeasure --points-csv ...`) |

---

## Description

Kinovea-style on-image calibration and measurement engine for `getpixelvideo.py`. The host UI splits into two toolbar modes:

- **CALIB** (`Shift+Q` / **CALIB** button) — build or load planar calibrations.
- **MEASURE** (`Q` / **MEASURE** button) — click points and classify as **Distance**, **Area**, **Angle**, **Velocity**, or **Acceleration**. A checkbox beside **MEASURE** toggles didactic on-image badges (`#N · type · value unit`).

This module owns measurement *state*, *math*, and pygame overlays. `getpixelvideo.py` only owns hotkeys, click routing, and toolbar buttons.

---

## CALIB modes (now)

The CALIB chooser dialog lists each mode on its own line (column layout) so the
options fit the pygame window.

| Mode | Input | Model |
|------|-------|-------|
| **Line** (`1`) | 2 clicks + typed length | Isotropic scale; origin at first click. |
| **Plane** (`2`) | 4 clicks + width/height | 8-parameter DLT2D homography. |
| **REF3D** (`3`) | `.ref3d` + drop axis (`z`→XY, `y`→XZ, `x`→YZ) + pixel CSV or guided clicks | Planar DLT2D via `rec2d`. |
| **DLT3D** (`4`) | — | Reserved (coming soon): extract 11 DLT3D params; if one world axis is held at 0, usable as planar 2D. |
| **Load** (`L`) | `.dlt2d` / REF2D+CSV / REF3D | Reuse existing files. |
| **Clear** (`0`) | — | Stay in pixels (`px`). |

Line and Plane use a persistent on-image guide. Points are named in click order,
the next edge follows the cursor, and completed edges show their dimension:

- **Line:** `1 START → 2 END`, with the segment marked `LENGTH`.
- **Plane:** `1 ORIGIN → 2 WIDTH → 3 OPPOSITE → 4 CLOSE`; edges `1–2` and
  `4–3` are `WIDTH`, while `2–3` and `1–4` are `HEIGHT`.

After the final click, inspect the complete geometry. Right-click removes the
last point; **Enter** confirms the drawing and opens the real measurement fields.
Cancelling or entering an invalid value keeps the points for correction or retry.

### Scope (default vs this frame)

After choosing a mode, pick (dialog shows one option per line):

1. **Default (whole video)** — used on every frame unless overridden.
2. **This frame only** — stored under the current video frame index.

MEASURE on frame N uses the per-frame calibration if present, otherwise the default.

---

## MEASURE

Digit keys after MEASURE is on:

| Key | Measurement |
|-----|-------------|
| `1` | Distance (2 clicks) |
| `2` | Area (≥3 clicks, `Enter` closes polygon) |
| `3` | Angle (3 or 4 points) |
| `4` | Velocity (2 frames, needs FPS) |
| `5` | Acceleration (3 frames, needs FPS) |

Each completed set is stored with `result_frame`, pixel geometry, real coordinates, and `calibration_frame` (`-1` = default).

---

## Math reuse

- **`dlt2d.py`** / **`dlt3d.normalize_ref3d_to_format1`** for REF3D modes
- **`rec2d_one_dlt2d.py`** `rec2d()` for pixel→real

Single-video sessions stay **DLT2D**. Stereo **DLT3D** reconstruction remains a batch/CLI workflow via `rec3d_one_dlt3d.py`. Future monocular DLT3D (11 params) is reserved in CALIB.

---

## Units

- **Uncalibrated session:** pixel units (`px`, `px/s`, `px/s^2`).
- **Calibrated session:** unit chosen at CALIB time (`m`, `mm`, `cm`, …).
- **Hover:** after calibration, the control-bar status shows `Pix: (x, y)` and `Real: (X, Y) <unit>` for the mouse position (`format_hover_coords`, frame-aware).

---

## Usage from getpixelvideo.py

| Key | Action |
|-----|--------|
| **Shift+Q** / **CALIB** | Toggle CALIB mode — Line / Plane / REF3D / Load / Clear + scope |
| **Q** / **MEASURE** | Toggle MEASURE mode (no calibration-first gate) |
| Left-click | Add a calibration point (CALIB) or a measure point (MEASURE) |
| Right-click | Undo the last point |
| **Enter** | In CALIB after all points: confirm the visible geometry and enter real measures. In MEASURE: close area or open save menu |
| **Backspace** | Clear all measure points/results |
| **I** / **FPS** | Set video FPS for velocity/accel |

`Esc` in the main tool always means save-and-quit the whole application; press `Q` / `Shift+Q` again to exit MEASURE / CALIB.

---

## Export layout

The first calibration or result save creates one `processed_quickmeasure_<timestamp>/` next to the video. Later saves in the same session update that directory.

| File | Contents |
|------|----------|
| `*_quickmeasure_points.csv` | Clicks: frame, px, real, `calibration_frame` |
| `*_quickmeasure_calibration.csv` | All calibs: `frame` (`-1` = default), kind, refs, DLT/scale |
| `*_quickmeasure.dlt2d` | One row per DLT calib; `frame` 0 = default, else 1-based video frame |
| `*_quickmeasure_results*.csv` | Measures + `calibration_frame` |
| `quickmeasure_report.html` | Didactic report |
| `README_quickmeasure.txt` | File glossary |

---

## API highlights

| Symbol | Role |
|--------|------|
| `QuickMeasureCalibration` | Line / plane / DLT2D / REF3D calib; `pixel_to_real()`, `source_frame` |
| `QuickMeasureSession` | `set_calibration(calib, frame=None)`, `calibration_for_frame(n)`, live modes, `save_session` |
| `format_hover_coords(session, x, y, frame=None)` | Status-bar pixel + real text |
| `CalibrationDraft` / `Ref3dCalibrationDraft` | Click collectors for CALIB |

---

## Related

| Module | Role |
|--------|------|
| **dlt2d** | DLT2D coefficients from calibration |
| **rec2d_one_dlt2d** | Apply fixed DLT2D to pixels |
| **getpixelvideo** | Host: CALIB + MEASURE buttons and hotkeys |
