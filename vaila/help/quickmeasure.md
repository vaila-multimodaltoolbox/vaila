# quickmeasure

## Module information

| Field | Value |
|-------|--------|
| **Category** | Processing |
| **File** | `vaila/quickmeasure.py` |
| **Version** | 0.4.19 |
| **Updated** | 24 November 2026 |
| **Author** | Paulo Santiago |
| **GUI** | Yes (embedded in `getpixelvideo.py`) |
| **CLI** | Yes (`python -m vaila.quickmeasure --points-csv ...`) |

---

## Description

Kinovea-style on-image calibration and measurement engine for `getpixelvideo.py`. The host UI splits into two toolbar modes:

- **CALIB** (`Shift+Q` / **CALIB** button) — build or load planar calibrations.
- **MEASURE** (`Q` / **MEASURE** button) — click points and classify as **Distance**, **Area**, **Angle**, **Velocity**, **Acceleration**, or **Level** (H+V laser guide + axis-snapped tape). A checkbox beside **MEASURE** toggles didactic on-image badges; **Clear** / Backspace wipe drawn feedback (keeps calib); `Ctrl+Z` / `Ctrl+Y` undo/redo while MEASURE is on.

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
| `6` | Level / tape (origin → H+V green crosshair; 2nd click snaps to nearer image axis) |
| `7`–`0` | Reserved |

**Clear** (toolbar) / Backspace clears draft, points, and drawn results (calibration stays). `Ctrl+Z` / `Ctrl+Y` undo/redo measure steps while MEASURE is active.

Each completed set is stored with `result_frame`, pixel geometry, real coordinates, and `calibration_frame` (`-1` = default). Level results also store `axis` (`H` or `V`).

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
| `QuickMeasureSession` | `set_calibration(calib, frame=None)`, `calibration_for_frame(n)`, live modes, `save_session`, `world_anchor(frame, x, y)` and `world_to_pixel_at(frame, X, Y, z)` (camera3d: pixel ↔ pitch point) |
| `format_hover_coords(session, x, y, frame=None)` | Status-bar pixel + real text |
| `CalibrationDraft` / `Ref3dCalibrationDraft` | Click collectors for CALIB |

---

## Related

| Module | Role |
|--------|------|
| **dlt2d** | DLT2D coefficients from calibration |
| **rec2d_one_dlt2d** | Apply fixed DLT2D to pixels |
| **getpixelvideo** | Host: CALIB + MEASURE buttons and hotkeys |

### Moving-camera DLT2D calibration

In **CALIB → L**, a DLT2D file with one data row per video frame automatically
loads per-frame calibration. Rows map by position: the first data row calibrates
the first video frame, regardless of the identifiers in the `frame` column.
The file must contain a frame column and eight finite numeric coefficients
(`L1…L8` or the usual vailá parameter headers). A single row keeps the fixed
calibration scope dialog. Other row counts are rejected without replacing the
current calibration. Cursor coordinates and measurements use each frame's
calibration; the sequence is saved once after loading. Exported DLT2D frame
identifiers retain the existing convention (1-based per-frame, 0 for default).
Video and calibration must already be synchronized and describe the same plane.

### Per-frame calibration (`*_frames.dlt3d` / `*_frames.dlt2d`)

A `kick_analysis_*` folder of *vailá* holds one camera per video frame, written next to the
SAM 3 outputs: `work/<stem>/sam3dinov3/processed_sam3dinov3_*/<stem>/<stem>_frames.dlt3d`
(`frame,L1..L11`, one row per video frame) and the matching `<stem>_frames.dlt2d`
(`frame,L1..L8`, the ground homography). In **CALIB → L**:

- `<stem>_frames.dlt3d` becomes one `camera3d` calibration: every click uses the camera of its
  own frame, so modes 1-8 give metres on the pitch in any frame.
- `<stem>_frames.dlt2d` becomes one DLT2D calibration per frame.

The FreeKiki detect run also writes `<stem>.dlt3d` / `<stem>.dlt2d` (in
`work/<stem>/freekiki_predict_*/`), but those are **sparse**: one row per frame, with numbers
only on the frames FreeKiki calibrated and empty (NaN) rows elsewhere. A calibration built from
them answers `(nan, nan)` on every empty frame. When you pick such a file, *vailá* looks in the
same kick folder for the per-frame file of the same stem; if it has a camera on every frame it
is loaded instead (the terminal prints `has no camera on N of M frames (sparse FreeKiki file);
loaded the per-frame calibration <path> instead`, and the calibration, exports and HUD name the
real file). If there is none, the load is refused with the number of frames without camera and
how to make the file: `uv run --no-sync vaila/freekiki.py kick-cameras <kick folder>` (or the
FreeKiki button **Calibration per frame**), or open the video with `--kick-run`.

Frames whose camera was not measured but copied from the nearest good frame are marked
**approximate** in `<stem>_frames_cameras.csv`; the status line counts them
(`N approx. frames`): measures there are less reliable, check them before using the numbers.

### Measures anchored on the pitch (moving camera)

With a `camera3d` calibration (one camera per video frame: `--kick-run`, or CALIB → L with a
`*_frames.dlt3d`) the camera moves from frame to frame, so a spot clicked on frame 0 is not at the
same screen pixel on frame 92. The overlay now draws the points of a measure where that spot of the
pitch is **in the frame you are looking at**: the click becomes a point of the grass with the camera
of its own frame (`QuickMeasureSession.world_anchor(frame, x, y)`) and is projected back with the
camera of the current frame (`world_to_pixel_at(frame, X, Y, z=0)`). On the frame of the click the
point is exactly where you clicked.

- Lines and polygons are drawn on the pitch and clipped to the video image (not the black bars, not
  under the open Kick panel); the value badge stays near the middle of the visible part.
- A spot outside the image gets a marker on the border with a small arrow towards it and the label
  `P<n> · q<frame>`: the point number and the frame of the click, counted from 0 as in the CSV
  (the HUD `Frame: 1/93` counts from 1, so `q0` is `Frame: 1`). A spot behind the camera gets the
  marker with `behind camera` (Portuguese window: `atrás da câmera`).
- Anchored types: distance, area, angle, velocity, acceleration, arc, and height (the base on the
  grass and the top above it, at the measured height). `level`, plane of the line, height line,
  event and spin stay at the clicked pixel: they belong to the image or to a vertical plane.
- Only `camera3d` is anchored. Line, plane, REF3D and DLT2D (fixed camera) draw exactly as before.
- Only the drawing changes: the values, `points.csv` (`x_px`, `y_px` of the click frame) and every
  export are unchanged; each point was already converted with the camera of its own frame.
- The anchor is the grass under the pixel (`z = 0`), the same assumption the values make: a click on
  the ball in the air is anchored on the grass under that pixel. Close to the horizon one pixel is many
  metres, so a point there can jump from frame to frame.
