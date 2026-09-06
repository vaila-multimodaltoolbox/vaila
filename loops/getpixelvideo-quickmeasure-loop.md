---
name: getpixelvideo-quickmeasure
category: Vailá
trigger: manual
verification-level: 1
theory-base: arXiv:2607.00038
---

# getpixelvideo.py Quick Measurement Tool (Kinovea-style) Loop

## Description
Add an interactive quick-measurement capability to `vaila/getpixelvideo.py`: click points directly on the displayed frame, then open a submenu (new hotkey) to classify the click set as a **Distance**, **Area**, or — once points exist on 2+ frames — **Velocity**/**Acceleration** measurement. Values are reported in pixels by default and in real-world units when a DLT2D (single video) or DLT3D (two synchronized videos, stereo) calibration is supplied, built the same way `dlt2d.py`/`dlt3d.py` already build `.dlt2d`/`.dlt3d` files from a calibration pixel CSV + `.ref2d`/`.ref3d` file. All new math/IO lives in a new standalone module, `vaila/quickmeasure.py`, imported by `getpixelvideo.py` rather than grown inline, to keep the 12k-line host file from growing further. This is vailá's own answer to Kinovea's quick on-image measurement tools — not a port, a native reuse of vailá's existing DLT stack.

## Use When
- Extending or fixing quick on-image measurement in `getpixelvideo.py` (click-to-measure, submenu, DLT overlay, stereo mode).
- Regression-testing after touching `dlt2d.py`, `dlt3d.py`, `rec2d_one_dlt2d.py`, `rec3d_one_dlt3d.py`/`rec3d.py` math that `quickmeasure.py` reuses.
- **Exclusions:** this loop does not touch marker *tracking* (YOLO/MediaPipe/manual tracking, `load_tracking_csv`, dataset export) — quick measurements are a separate, ephemeral click session, not persisted tracking data. Do not conflate the two point stores.

## Inputs
1. `video_path` — primary video (or PNG sequence) already open in the running `getpixelvideo.py` session.
2. `video_path2` — optional second, time-synchronized video; required only for Milestone 3 (stereo DLT3D mode), otherwise omitted.
3. `calib_pixel_file` + `ref_file` (`.ref2d`/`.ref3d`) **or** a precomputed `.dlt2d`/`.dlt3d` file — optional per camera. Absent → measurements stay in pixel units and the UI must say so explicitly.
4. `fps` — frame rate for velocity/acceleration. Default: read from `get_precise_video_metadata()`; user override via the submenu.
5. `output_dir` — where the measurement session CSV is written. Default: alongside the video, timestamped (`vaila` convention).

## Goal
1. **Milestone 1 (pixel-only):** in `getpixelvideo.py`, a hotkey toggles "measure mode"; clicks place measurement points (visually distinct from tracking markers); a submenu (opened by another hotkey) lets the user tag the current point set as Distance (2 pts) or Area (≥3 pts, polygon) and shows the pixel-unit result on screen and in a saved session CSV.
2. **Milestone 2 (DLT2D real-world):** the submenu gains a "Calibrate" action that loads/builds a `.dlt2d` (via `vaila/quickmeasure.py` calling into `dlt2d.py`'s functions, not reimplementing them) and re-expresses Distance/Area in real-world units (metres) using `rec2d_one_dlt2d.py`'s `rec2d()`; when the same point is placed on 2+ frames at a known `fps`, Velocity and Acceleration become available (finite difference on real-world coordinates).
3. **Milestone 3 (DLT3D stereo):** with `video_path2` and one `.dlt3d` per camera (via `dlt3d.py`), clicking the same physical point in both video frames (frame-synchronized) triangulates it with `rec3d_multicam()` (from `rec3d.py`, already used by `rec3d_one_dlt3d.py`); Distance/Area/Velocity/Acceleration are then reported in 3D real-world units.
4. `getpixelvideo.py` gains no more than a thin integration layer (hotkey, submenu wiring, delegation calls); all measurement math/IO/session-state lives in `vaila/quickmeasure.py`.
5. GUI and CLI stay behaviorally aligned: entering measure mode from the GUI prints a copy-pasteable `>>` CLI-equivalent line for headless reproduction where one is meaningful (calibration + a fixed point list).

## Verification (Governing Check)
- **True level:** 1 (deterministic synthetic-fixture assertions).
- **Check** (run the subset matching the milestone being worked, all three before declaring overall success):
  ```bash
  uv run pytest tests/test_quickmeasure.py -v
  uv run pytest tests/test_dlt_rec.py tests/test_dlt_rec_integration.py -v
  uv run ruff check vaila/quickmeasure.py vaila/getpixelvideo.py --fix
  uv run ruff format --check vaila/quickmeasure.py vaila/getpixelvideo.py
  uv run ty check vaila/quickmeasure.py vaila/getpixelvideo.py
  ```
- **Evidence:** raw pytest output showing, per milestone, computed Distance/Area/(Velocity/Acceleration where applicable) matching the synthetic fixture's known expected value within `atol=1e-3` (metres) / exact integer pixels for M1; ruff/ty output showing zero findings on the two changed files.
- **Completion criterion:** all three milestone fixture files pass, the DLT regression suite still passes unchanged, ruff/ty are clean on both files, and the metadata checklist below is satisfied.
- **Verifier protection:** synthetic calibration + expected-value fixtures live under `tests/fixtures/quickmeasure/` (a known-size calibration square/cube + hand-computed expected distance/area/velocity/acceleration). The maker may *add* new fixture files for a milestone it is implementing but may not edit an existing fixture's expected numeric values without recording a red-before/green-after justification in the state file — a failing test must be fixed by fixing the code, not the expectation.
- **Scientific validity:**
  - Units: pixels (M1) or metres (M2/M3); the UI and CSV column headers always state which.
  - Coordinate frame/convention: matches `dlt2d.py`/`dlt3d.py` (world coordinates as calibrated by the user's `.ref2d`/`.ref3d`), not re-derived.
  - Sampling rate: `fps` used for Velocity/Acceleration is explicit in the session CSV, never assumed silently.
  - Missing/occluded data: a Velocity/Acceleration request over a frame range with a missing click is refused with a clear message, never silently interpolated.
  - Degenerate calibration (fewer than 4 pts for DLT2D / 6 for DLT3D) is refused with the same message `dlt2d.py`/`dlt3d.py` already use, not a raw exception.

## Trigger
Manual, via `/preto-loop`, `/goal`, or direct developer invocation. Duplicate-run guard: read `loops/state/getpixelvideo-quickmeasure-loop-state.json`'s `current_milestone` and `attempts` before starting; do not restart a milestone already marked `done` without an explicit user request to redo it.

## Iteration
0. On the first iteration: confirm `uv run pytest tests/ -v` passes on the current baseline (or record pre-existing failures as a known baseline, not something this loop must fix); freeze which milestone is targeted this run.
1. Load this spec and `loops/state/getpixelvideo-quickmeasure-loop-state.json`; confirm the per-milestone attempt budget below is not exhausted.
2. Snapshot `git status --porcelain` as the rollback baseline; run the governing check subset for the current milestone to confirm the starting point.
3. Rank the milestone's unresolved requirement (e.g. "submenu missing Area option") worst-first; pick exactly one.
4. Invoke `$gui-developer` (hotkey/submenu wiring in `getpixelvideo.py`), `$biomechanics-analyst` (DLT unit/coordinate correctness in `quickmeasure.py`), or `$test-writer` (fixture + assertions) to make exactly that one change.
5. Run the governing check for the current milestone and the DLT regression suite; capture raw stdout/stderr.
6. Retain the change only if its own milestone's fixture now passes AND no previously-passing milestone/regression test broke; otherwise `git checkout -- <files touched this iteration>` (from the step-2 snapshot) and record why.
7. Update Python header date/version (global vailá version, matching `vaila.py`), `vaila/help/index.md`/`.html`, and root `README.md` "Last updated" per `CLAUDE.md`'s metadata checklist — treat this as part of the check, not cleanup.
8. Persist state (milestone, attempt count, accepted/rejected diff summary, evidence excerpts, cost) atomically to the state file.
9. Evaluate terminal states; otherwise begin the next iteration.

## Terminal States
- **success:** Milestones 1–3 fixtures pass, `tests/test_dlt_rec*.py` still pass unchanged, ruff/ty clean on both files, metadata synced, and a manual smoke run of `getpixelvideo.py` shows the hotkey/submenu working without breaking existing tracking-mode hotkeys.
- **no-op:** the targeted milestone's requirement is already implemented and its fixture already passes with zero diff needed.
- **no-progress/stalled:** two consecutive iterations on the same milestone fail the same assertion (same fixture, same expected-vs-actual mismatch) without a new hypothesis.
- **blocked:** no synthetic calibration fixture can be built for the milestone (e.g. Milestone 3 needs a second synchronized test video and none is available/fixturable), or a pygame event-loop conflict with the existing tracking-mode key handling cannot be resolved without redesigning that loop (escalate, don't silently rework the host file's event loop).
- **exhausted:** the turn budget in Guardrails is reached for the current milestone.

Errors, missing evidence, and budget exhaustion are never success.

## Guardrails
- **Maximum allocation:** 40 turns per milestone attempt, max 6 attempts per milestone (240-turn hard ceiling across all 3 milestones, tracked per-milestone in the state file). No external/paid API budget applies (local dev only).
- **Human approval required:** before changing any *existing* hotkey binding or `draw_controls()`/`show_help_dialog()` text (must not silently remap muscle-memory keys); before any `git commit`/`git push`; before deleting or overwriting a fixture under `tests/fixtures/quickmeasure/`.
- **Isolation and credentials:** local working tree, no network access or credentials needed; no worktree required since changes are additive (new module) plus a bounded, reviewable diff to `getpixelvideo.py`.
- **Protected verifier:** `tests/fixtures/quickmeasure/*` expected values, and `tests/test_dlt_rec.py`/`tests/test_dlt_rec_integration.py` — the maker must not edit these to make a check pass.
- **Rollback:** `git checkout -- <files touched this iteration>`, scoped to the exact file list captured in step 2's `git status --porcelain` snapshot for that iteration only.

## State Memory
- **Path:** `loops/state/getpixelvideo-quickmeasure-loop-state.json`.
- **Persist:** baseline commit hash, `current_milestone` (1/2/3), per-milestone `attempts`, accepted/rejected change summaries (files + one-line reason), raw check evidence excerpts, curated lessons (kept only while still true), cumulative turns spent.
- **Recovery:** a fresh context reads `current_milestone` and the last accepted change, then re-runs the governing check for that milestone before making any new change — it never trusts a stale "passed last time" note without rerunning it.

## Skills
- `$gui-developer` — pygame/Tkinter UI wiring in `getpixelvideo.py` (measure-mode hotkey, submenu, click handling, on-screen readout).
- `$biomechanics-analyst` — DLT2D/DLT3D unit, coordinate-frame, and finite-difference correctness in `vaila/quickmeasure.py`.
- `$test-writer` — synthetic calibration fixtures and pytest assertions per milestone under `tests/fixtures/quickmeasure/` and `tests/test_quickmeasure.py`.
- Reused library modules (direct imports, not agent skills): `vaila/dlt2d.py`, `vaila/dlt3d.py`, `vaila/rec2d_one_dlt2d.py` (`rec2d()`), `vaila/rec3d_one_dlt3d.py`/`vaila/rec3d.py` (`rec3d_multicam()`).

## Why It Works
A frozen synthetic fixture with a hand-computed expected distance/area/velocity/acceleration turns "does the DLT-calibrated measurement look right" (level 4/5, Kinovea-style eyeballing) into a level-1 numeric assertion, closing the main scientific failure mode named in the domain reference: mistaking a plausible on-screen number for a verified one. Splitting the work into three milestones (pixel → DLT2D → DLT3D-stereo), each gated by its own fixture before the next starts, keeps the DLT/rec2d/rec3d math correct at each layer instead of compounding an early error into the stereo case. Keeping the new logic in `vaila/quickmeasure.py` rather than growing the 12k-line `getpixelvideo.py` further preserves ownership: the host file only wires a hotkey and a submenu, the new module owns click-session state and DLT math, and the existing `dlt2d.py`/`dlt3d.py`/`rec2d_one_dlt2d.py`/`rec3d_one_dlt3d.py` modules are reused, not reimplemented — one architecturally-scoped module instead of a inline pile of new code hard to maintain.

## How to Trigger
### Context-bound
Within a single Claude Code session: read this file, then execute one milestone's iteration steps turn-by-turn (a milestone's worth of work fits one context window); confirm the governing check for that milestone before declaring it done.

### Fresh-context / Ralph
For the full 3-milestone run (larger than one context window): an external runner re-reads this loop-doc and `loops/state/getpixelvideo-quickmeasure-loop-state.json` every turn, resumes at `current_milestone`, and stops only on a named terminal state above — never on "looks done."

## Health Metrics
- **Cost per accepted change:** total turns spent / verified non-regressive changes retained.
- **Milestones completed:** 0–3 (persisted in state).
- **Fixture pass rate:** passing fixtures / total fixtures defined so far.
- **Regression count:** previously-passing tests broken by an iteration (must stay 0 for a change to be retained).
