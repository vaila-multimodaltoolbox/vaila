---
name: soccerfield-3d-visualization-loop
category: Vailá
trigger: manual
verification-level: 1
theory-base: arXiv:2607.00038
---

# Soccer Field 3D Visualization and Calibration Loop

## Description
Provide full 3D spatial visualization and verification of soccer field calibration keypoints (including the `soccerfield_kiki.csv` 48-keypoint model and user-created custom calibration models) across vailá. Enables saving models to both `.c3d` and `.csv`, adapts spatial environments (regulation pitch outlines, boundary lines, center circles, turf planes, and 3D labels) in `vaila/showc3d.py`, `vaila/viewc3d_pyvista.py`, and `vaila/readcsv.py`, and provides direct 3D previewing from the field editor.

## Use When
- Inspecting soccer field calibration keypoints (corners, penalty spots, crossbar heights, flag tops) in a 3D coordinate system.
- Exporting calibration models to C3D (`ezc3d`) for multi-camera calibration and MoCap pipeline integration.
- Reading calibration model CSV files in 3D viewers (`showc3d`, `readcsv`, `viewc3d_pyvista`).
- Verifying height differences between ground markers ($Z=0$), flag tops ($Z=1.5\,\text{m}$), and crossbars ($Z=2.44\,\text{m}$) in a 3D scene.

## Inputs
1. `soccerfield_kiki.csv` / custom calibration CSV — `vaila/models/soccerfield_kiki.csv` (48-keypoint reference model).
2. Calibration C3D files — exported via `save_calibration_model_c3d`.
3. 3D Viewers — Matplotlib (`showc3d.py`), PyVista (`viewc3d_pyvista.py`), Open3D (`viewc3d.py`), and CSV reader (`readcsv.py`).

## Goal
1. In `vaila/drawsportsfields.py`, users can export calibration models to both CSV (`save_calibration_model_csv`) and C3D (`save_calibration_model_c3d`), and click **View in 3D...** to immediately preview points in PyVista or Matplotlib.
2. In `vaila/showc3d.py`, the 3D viewer automatically detects soccer field scale ($>20\,\text{m}$), adjusts Cartesian axes to match field dimensions, renders regulation pitch boundary lines, halfway line, center circle, ground plane, and adds 3D text labels for all keypoints with an interactive toggle button (`Labels: ON/OFF`). Single-frame calibration models are rendered cleanly without unnecessary multi-frame playback sliders.
3. In `vaila/viewc3d_pyvista.py`, field-scale models render green turf ground plane, regulation boundary lines, halfway line, center circle at $Z=0$, and auto-enable 3D keypoint labels for single-frame calibration models.
4. In `vaila/readcsv.py`, `read_csv_generic` automatically identifies calibration model CSV files (rows with `point_name`, `x`, `y`, `z`), parsing all keypoint names and coordinates into 3D structures compatible with PyVista, Open3D, and Matplotlib viewers.
5. In `vaila.py`, the **Choose C3D viewer** dialog offers Open3D, PyVista, and Matplotlib (`showc3d`) viewers.

## Verification (Governing Check)
- **True level:** 1 (deterministic unit and regression tests).
- **Check:**
  ```bash
  uv run pytest tests/test_soccerfield_3d_viewers.py tests/test_soccerfield_kiki_and_calib.py -v
  uv run ruff check vaila/drawsportsfields.py vaila/showc3d.py vaila/readcsv.py vaila/viewc3d_pyvista.py vaila.py
  uv run ty check vaila/showc3d.py vaila/readcsv.py vaila/viewc3d_pyvista.py
  ```
- **Evidence:** Pytest output proving:
  - C3D files are created with correct homogeneous shapes `(4, num_points, 1)`, rate, and unit labels (`m` and `mm`).
  - `showc3d.load_c3d_file` and `viewc3d_pyvista._load_c3d_arrays` correctly read calibration C3D files in meters.
  - `readcsv.read_csv_generic` detects calibration model CSVs, parses keypoint names, and yields `(1, num_points, 3)` coordinate arrays.
  - Soccer field features (perimeter, halfway line, center circle) render without GUI errors.
- **Completion criterion:** All unit and integration tests pass; Ruff linter and Ty type check pass; help docs and metadata synced.
- **Verifier protection:** Frozen schema in `tests/test_soccerfield_3d_viewers.py` verifies both meter and millimeter round-trip persistence.
- **Scientific validity:** Calibration coordinates are strictly preserved in metric units (metres), coordinate axes correspond to length (X), width (Y), and vertical height (Z). Non-coplanar points retain true 3D spatial elevations.

## Trigger
Manual or goal-driven invocation via `/preto-loop` or `/goal`.

## Iteration
0. Validate baseline imports and tests.
1. Implement `save_calibration_model_c3d` and `preview_calibration_in_3d` in `vaila/drawsportsfields.py`.
2. Add "Save Model C3D..." and "View in 3D..." buttons to the keypoint editor dialog in `drawsportsfields.py`.
3. Upgrade `vaila/showc3d.py` to auto-detect units (`m` vs `mm`), adapt spatial scaling for soccer fields, render pitch lines/center circle, and support 3D text labels with toggle.
4. Enhance `vaila/readcsv.py` to identify calibration model CSV files and support Matplotlib viewer choice alongside PyVista and Open3D.
5. Enhance `vaila/viewc3d_pyvista.py` to render field turf plane, regulation lines, and auto-enable labels for single-frame calibration models.
6. Wire Matplotlib viewer option into `vaila.py` C3D viewer chooser dialog.
7. Write and pass automated test suite `tests/test_soccerfield_3d_viewers.py`.
8. Update help documentation and synchronization metadata across scripts and documentation.

## Terminal States
- **success:** C3D export, CSV detection, and 3D visualization operational across all viewers; all tests pass; linter/type checks clean.
- **no-progress/stalled:** Two consecutive failures in C3D encoding or coordinate parsing without resolution.
- **blocked:** Missing required visualization backend libraries (`ezc3d`, `pyvista`, `matplotlib`).
- **exhausted:** Maximum allocated turns reached without satisfying completion criteria.

## Guardrails
- Maintain backward compatibility for standard multi-frame MoCap C3D and tracking CSV files.
- Never alter raw coordinate data or overwrite model files silently.
- Preserve single Tkinter root convention (`tk.Toplevel` for secondary dialogs).
- Ensure headless/automated testability without blocking on GUI windows.

## State Memory
- **Path:** `loops/state/soccerfield-3d-visualization-loop-state.json`.
- **Persist:** Baseline status, attempts, test outcomes, accepted changes.

## Skills
- `$safe-refactor` — Update 3D viewers and readers while maintaining existing MoCap workflows.
- `$surgical-patch` — Add C3D export and 3D preview options with minimal invasive changes.

## Why It Works
By standardizing calibration keypoint representations across both C3D (`ezc3d`) and CSV formats, and making `readcsv.py` and `showc3d.py` aware of soccer field dimensions, users can seamlessly transition between 2D pitch planning and full 3D spatial verification with complete geometric fidelity.

## How to Trigger
```bash
uv run pytest tests/test_soccerfield_3d_viewers.py -v
```

## Health Metrics
- **Cost per accepted change:** Total tokens / verified changes retained.
- **Test coverage:** 100% pass rate on C3D/CSV 3D visualization tests.
