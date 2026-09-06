---
name: getpixelvideo-quickmeasure-calibration-first
category: Vailá
trigger: manual
verification-level: 1
theory-base: arXiv:2607.00038
---

# Quick Measure calibration-first (Kinovea-style) loop

## Description
Make **calibration the first step** of Quick Measure in `vaila/getpixelvideo.py`: pressing `Q` or clicking the **QMeas** button asks the user to click calibration points on the image and type the real-world measurement(s), and only then allows free measuring. Every clicked point is persisted to CSV with both pixel and calibrated real-world coordinates, and distance/area/velocity/acceleration can be recomputed from that saved points CSV (module API + CLI), mirroring [Kinovea](https://github.com/Kinovea/Kinovea) line/plane calibration.

## Use When
- Changing the Quick Measure entry flow, its calibration model, its CSV schema, or the "measure from a saved calibrated points file" API/CLI.
- **Exclusions:** stereo DLT3D triangulation from two synchronized videos stays a batch workflow (`rec3d_one_dlt3d.py`); marker tracking CSVs (`load_tracking_csv`) are a different point store and must not be merged with measure sessions.

## Inputs
1. `calibration_mode` — `line` (2 clicks + 1 known length) or `plane` (4 clicks around a known rectangle + width/height). Default proposed in the UI: `line`.
2. `real_measures` — typed by the user in the calibration prompt: one length for `line`; width and height for `plane`. Must be finite and > 0.
3. `unit_label` — real-world unit string typed by the user; default `m`.
4. `fps` — from `get_precise_video_metadata()`; required only for velocity/acceleration.
5. `output_dir` — defaults to the video's directory; CSVs are timestamped.

## Goal
1. Toggling Quick Measure (`Q` **or** QMeas button) with no calibration in the session enters **calibration mode first**: on-screen instructions, clicks collected as calibration points, then a text prompt for the real measurement(s) and unit.
2. After the calibration is accepted, the session switches to free measuring; clicks become measure points in the calibrated frame.
3. Points are saved to a CSV containing `point_id, frame, x_px, y_px, x_real, y_real, unit` plus a calibration sidecar CSV recording mode, unit, typed measures, calibration clicks and resulting parameters; measurement results are saved to their own CSV.
4. `vaila/quickmeasure.py` exposes `measure_from_points_csv()` (and a `python -m vaila.quickmeasure` CLI) that recomputes distance/area/velocity/acceleration from a saved calibrated points CSV without needing the video.
5. Existing file-based calibration (`.dlt2d`, pixel CSV + `.ref2d`) still works; pixel-only (uncalibrated) measuring remains reachable by explicitly skipping calibration.
6. Metadata synced: module headers, `vaila/help/quickmeasure.{md,html}`, `vaila/help/getpixelvideo.{md,html}`, `vaila/help/index.{md,html}`, root `README.md`.

## Verification (Governing Check)
- **True level:** 1 — deterministic assertions against hand-computable calibration fixtures; the pygame click flow itself is level 5 (manual smoke) and is explicitly reported as such.
- **Check:**
  ```bash
  uv run pytest tests/test_quickmeasure.py -v
  uv run pytest tests/test_dlt_rec.py tests/test_dlt_rec_integration.py -v
  uv run ruff check vaila/quickmeasure.py vaila/getpixelvideo.py --fix
  uv run ruff format --check vaila/quickmeasure.py vaila/getpixelvideo.py
  uv run ty check vaila/quickmeasure.py vaila/getpixelvideo.py
  uv run python -m vaila.quickmeasure --points-csv tests/fixtures/quickmeasure/points_calibrated.csv --measure distance
  ```
- **Evidence:** raw pytest output showing line-calibration scale, plane-calibration DLT2D parameters, CSV round-trip, and CLI stdout printing the expected distance; ruff/ty output with zero findings on both files.
- **Completion criterion:** every assertion above passes, the CLI prints the fixture's hand-computed value, and a manual smoke run confirms that pressing `Q`/QMeas with an uncalibrated session shows the calibration prompt before any free measuring.
- **Verifier protection:** `tests/fixtures/quickmeasure/*` expected numbers (including the existing `sample.dlt2d` affine solution `[100,0,100,0,100,100,0,0]`) are frozen; a failing assertion must be fixed in code, never by editing the fixture, unless a red-before/green-after note is recorded in the state file.
- **Scientific validity:**
  - **Line calibration** assumes an isotropic scale on a plane parallel to the sensor: `scale = real_length / pixel_distance`, origin at the first calibration click, `+x` right, `+y` up (image `y` inverted). Out-of-plane points are *not* corrected — the UI and CSV must state the mode.
  - **Plane calibration** solves the 8-parameter DLT2D homography with `dlt2d.dlt2d(F=real, L=pixel)` and applies it with `rec2d_one_dlt2d.rec2d()`; no reimplementation of that math.
  - Units come from the user's typed `unit_label` and are written in every CSV row; pixel-only sessions must say `px`.
  - `fps` used by velocity/acceleration is written into the results CSV; missing frames are refused, never interpolated.
  - Degenerate input (zero-length calibration line, non-positive width/height, fewer than 4 plane points, NaN parameters) raises `QuickMeasureError` with a user-visible message.

## Trigger
Manual (`/goal`, `/preto-loop`, or a developer). Duplicate-run guard: read `loops/state/getpixelvideo-quickmeasure-calibration-first-loop-state.json` and do not redo a milestone already marked `done` without an explicit request.

## Iteration
0. First iteration: freeze the calibration contract (modes, click order, origin, axis directions, unit handling) in this file before touching code.
1. Load this spec and the state file; confirm the remaining budget.
2. Snapshot `git status --porcelain`; run the governing check to establish the baseline.
3. Rank unresolved goal items worst-first; select exactly one.
4. Invoke `$surgical-patch` (narrow behavior change), `$gui-developer` (pygame calibration prompt/overlay), or `$test-writer` (fixtures) to make that one change.
5. Re-run the governing check; capture raw stdout/stderr.
6. Keep the change only if its own assertion passes and nothing previously green broke; otherwise `git checkout --` the files touched in this iteration.
7. Sync metadata (module headers, help pages, index, README) as part of acceptance, not cleanup.
8. Persist state atomically; evaluate terminal states.

## Terminal States
- **success:** all six goal items evidenced by the governing check, plus a recorded manual smoke confirming calibration-first behavior in the running GUI.
- **no-op:** the selected item is already implemented and its assertion already passes with an empty diff.
- **no-progress/stalled:** two consecutive iterations fail the same assertion with the same expected-vs-actual values and no new hypothesis.
- **blocked:** the pygame modal cannot collect clicks without restructuring `getpixelvideo.py`'s main event loop (escalate instead of rewriting that loop), or a calibration fixture cannot be derived by hand.
- **exhausted:** 40 turns or 6 attempts reached.

Errors, missing evidence, and budget exhaustion are never success.

## Guardrails
- **Maximum allocation:** 40 turns, 6 attempts.
- **Human approval required:** remapping any existing hotkey; `git commit` / `git push`; deleting or overwriting an existing fixture under `tests/fixtures/quickmeasure/`; changing the meaning of an already-released CSV column.
- **Isolation and credentials:** local working tree only; no network, no credentials.
- **Protected verifier:** `tests/fixtures/quickmeasure/*`, `tests/test_dlt_rec.py`, `tests/test_dlt_rec_integration.py`.
- **Rollback:** `git checkout -- <files listed in this iteration's step-2 snapshot>`.

## State Memory
- **Path:** `loops/state/getpixelvideo-quickmeasure-calibration-first-loop-state.json`.
- **Persist:** baseline commit, current goal item, attempts, accepted/rejected change summaries, raw evidence excerpts, frozen calibration contract, curated lessons, turns spent.
- **Recovery:** a fresh context re-reads this file plus the state file and re-runs the governing check before editing; a state file without a terminal status means the previous write was interrupted and the check must be re-run from scratch.

## Skills
- `$surgical-patch` — narrow, reversible behavior changes to the toggle/calibration path.
- `$gui-developer` — pygame calibration overlay, text prompt, and submenu wiring.
- `$test-writer` — hand-computable calibration and CSV round-trip fixtures.
- Reused library modules (direct imports): `vaila/dlt2d.py` (`dlt2d()`), `vaila/rec2d_one_dlt2d.py` (`rec2d()`).

## Sub-Loops
- `getpixelvideo-quickmeasure-loop.md` — the original measurement-math loop; call it only if a Distance/Area/Velocity/Acceleration formula itself is wrong. Contract: it must terminate on its own named states before this loop resumes. Parent 40 × child 40 = at most 1600 child iterations worst case; circular calls are prohibited.

## Why It Works
Calibration-first is a *flow* requirement, which is easy to fake with a plausible-looking UI; binding it to hand-computable fixtures (a 100 px line declared as 1 m must yield `scale = 0.01 m/px`; a clicked rectangle must reproduce the frozen affine DLT2D solution) converts it into level-1 assertions. Persisting every click with both pixel and real coordinates, then recomputing the same measurements from that CSV through a separate CLI entry point, gives an independent check that the saved file — not hidden session state — is what the numbers come from. Freezing the fixtures and the DLT math (reused from `dlt2d.py`/`rec2d_one_dlt2d.py`) blocks the cheapest failure mode: adjusting expectations instead of the code.

## How to Trigger
### Context-bound
Read this file, then work goal items one at a time, running the governing check after each.

### Fresh-context / Ralph
An external runner re-reads this file and the state file every turn, resumes at the recorded goal item, and stops only on a named terminal state — never on "looks done".

## Health Metrics
- **Cost per accepted change:** turns spent / verified non-regressive changes retained.
- **Goal items completed:** 0–6 (persisted in state).
- **Regression count:** previously-passing tests broken by an iteration (must stay 0).
- **Manual smoke recorded:** boolean; false means `success` cannot be claimed.
