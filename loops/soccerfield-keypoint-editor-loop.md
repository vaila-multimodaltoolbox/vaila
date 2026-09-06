---
name: soccerfield-keypoint-editor-loop
category: Vailá
trigger: manual
verification-level: 1
theory-base: arXiv:2607.00038
---

# Soccer Field Keypoint Calibration Editor & Kiki Model Loop

## Description
Provide interactive calibration keypoint editing in `vaila/drawsportsfields.py` (add keypoints by clicking on the pitch, define 3D coordinates, build new models or start from scratch), introduce the `soccerfield_kiki.csv` 48-keypoint reference model, and visually differentiate coincident keypoints (such as corner points and corner flag tops) in orientation and position.

## Use When
- Calibrating camera setups or building customized pitch reference models for DLT2D/DLT3D reconstruction.
- Needing to add arbitrary 3D control points (flags, poles, benches, goal tops) by clicking directly on the 2D pitch view.
- Visualizing soccer field models with multi-height coincident points where corner flag tops or goal posts share horizontal (x, y) coordinates with ground line intersections.
- Using the `soccerfield_kiki.csv` 48-keypoint model.

## Inputs
1. `soccerfield_kiki.csv` — `vaila/models/soccerfield_kiki.csv` (48-keypoint FIFA pitch reference, including 32 pitch points and 16 3D features).
2. `soccerfield_ref3d_fifa_dataset.csv` — `vaila/models/soccerfield_ref3d_fifa_dataset.csv` (source dataset reference).
3. Field view axes — Matplotlib TkAgg axes rendered in `drawsportsfields.py`.

## Goal
1. The new model `soccerfield_kiki.csv` is created and registered in `SPORT_REGISTRY` under `"kiki"`, with auto-detection in `_detect_sport`.
2. When drawing the soccer field with 48 keypoints, overlapping points (corner ground points vs corner flags, goal post bases vs tops, and net ground points) are rendered with differentiated label positions, rotation angles, and dotted leader lines. Goal posts are positioned inside the field while net points are positioned behind the net (>= 6m separation), completely eliminating visual clutter around the goal area.
3. An interactive GUI mechanism (`Calib Keypoints`) provides two creation modes:
   - **Click-to-Add**: click directly on the pitch to capture coordinates and immediately define the label name and height Z in a focused dialog.
   - **Direct Coordinates Entry**: write coordinates (X, Y, Z, label name, point #) directly into the form in the Keypoint Editor window with presets (Center, Left Goal, Right Goal, Ground, Flag, Bar) and instant addition/updating.
   - Supports augmenting existing models or building a new model from scratch, with direct CSV export.

## Verification (Governing Check)
- **True level:** 1 (deterministic schema, coordinate accuracy, and unit tests).
- **Check:**
  ```bash
  uv run pytest tests/test_soccerfield_kiki_and_calib.py -v
  uv run pytest tests/test_drawsportsfields_dataset_loading.py tests/test_drawsportsfields_ref3d_export.py -v
  uv run ruff check vaila/drawsportsfields.py
  uv run ty check vaila/drawsportsfields.py
  ```
- **Evidence:** Pytest output proving:
  - `soccerfield_kiki.csv` has 48 rows matching canonical keypoint names and coordinate values.
  - `_detect_sport` resolves `soccerfield_kiki.csv` to `"kiki"`.
  - `SPORT_REGISTRY["kiki"]` is registered and functional.
  - De-overlapping logic assigns different coordinates and rotation angles to coincident points (corner 0 vs flag 38, corner 5 vs flag 39, corner 24 vs flag 46, corner 29 vs flag 47, goal posts 32/34, 33/35, 40/42, 41/43).
  - Calibration keypoint operations (add, clear/scratch, save to CSV) operate accurately.
- **Completion criterion:** All unit and integration tests pass; `drawsportsfields.py` passes Ruff and Ty checks; metadata synced.
- **Verifier protection:** Frozen reference coordinates in `tests/test_soccerfield_fifa_expansion.py` and canonical keypoint names cannot be modified.
- **Scientific validity:** Coordinate frame matches FIFA pitch standards (center origin or corner origin in metres, Z represents vertical height); non-coplanar points retain true 3D spatial definitions.

## Trigger
Manual or goal-driven invocation via `/preto-loop` or `/goal`.

## Iteration
0. Confirm baseline tests pass.
1. Create `vaila/models/soccerfield_kiki.csv` from the 48-keypoint FIFA dataset reference.
2. Register `"kiki"` in `SPORT_REGISTRY` and update `_detect_sport` in `vaila/drawsportsfields.py`.
3. Implement coincident point detection and orientation/position offset handling in `_draw_fifa32_dataset_keypoints_overlay` and `_ref_label`.
4. Add GUI controls and interactive click-to-add keypoints mechanism (`toggle_calib_keypoint_mode`, dialog for x, y, z, clear/scratch, save model).
5. Write and execute comprehensive pytest suite in `tests/test_soccerfield_kiki_and_calib.py`.
6. Run linter and type checker, sync metadata across scripts and documentation.

## Terminal States
- **success:** `soccerfield_kiki.csv` verified, coincident points rendered with distinct position/orientation, keypoint calibration creation and export working in GUI/tests, all checks green.
- **no-progress/stalled:** Two consecutive test failures without coordinate or syntax resolution.
- **blocked:** Missing Tkinter or Matplotlib GUI backend dependency preventing canvas event binding.
- **exhausted:** Maximum turns reached without passing the governing check.

## Guardrails
- Preserve existing model files and backward compatibility for `fifa_dataset`, `soccerfield_ref3d_fifa.csv`, and `soccerfield_ref3d.csv`.
- Calibration coordinates must remain in metres within valid pitch boundaries.
- No second `tk.Tk()` root windows; use `tk.Toplevel(root)`.

## State Memory
- **Path:** `loops/state/soccerfield-keypoint-editor-loop-state.json`.
- **Persist:** Baseline status, attempts, test outcomes, accepted changes.

## Skills
- `$safe-refactor` — Modify `drawsportsfields.py` while preserving existing plotting workflows.
- `$surgical-patch` — Add button and canvas click handler without disrupting manual frame marker mode.

## Why It Works
Separating label offsets and rotation angles prevents visual collision when 3D features project to the same 2D ground coordinates. The click-to-add calibration workflow translates Matplotlib data-space coordinates directly into metric pitch units, allowing immediate inspection and export for DLT workflows.

## How to Trigger
```bash
uv run pytest tests/test_soccerfield_kiki_and_calib.py -v
```

## Health Metrics
- **Cost per accepted change:** Total tokens / verified changes retained.
- **Test coverage:** 100% pass rate across sports field detection, plotting, and keypoint editing tests.
