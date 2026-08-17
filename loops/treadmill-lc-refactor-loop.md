---
name: treadmill-lc-refactor-loop
category: Biomechanics
trigger: manual
verification-level: 1
theory-base: arXiv:2607.00038
---

# Treadmill Load-Cell Refactoring Loop

## Description
Refactor `vaila/treadmill_lc.py` and its test suite (`tests/test_treadmill_lc.py`, `tests/treadmill_lc`)
to enforce a strict 5-stage logical calibration pipeline order (Zero Offset → Calibration Matrix →
Coordinate Transformation → Signal Filtering → Event Detection), adopt standardized side encoding
(Right = 0, Left = 1), use standardized English biomechanics keys (`medial_lateral`, `anterior_posterior`,
`vertical`), add unified `--config` TOML CLI/GUI support with single-window file selection, and integrate
Rich console panel/table execution reporting matching vailá standards.

## Use When
- Refactoring `vaila/treadmill_lc.py` into a modular, headless-first, TOML-configured pipeline.
- Updating treadmill test fixtures and verification tests in `tests/test_treadmill_lc.py` and `tests/treadmill_lc`.
- Standardizing force plate and instrumented treadmill calibration routines and coordinate definitions.
- Not for: video markerless tracking (SAM3/Sapiens2), FIFA soccer field homography, or unrelated MoCap modules.

## Inputs
1. `source_script` — `vaila/treadmill_lc.py` (target module).
2. `test_suite` — `tests/test_treadmill_lc.py` and `tests/treadmill_lc/`.
3. `sample_config` — `tests/treadmill_lc/processing_configuration_used.toml`.
4. `session_context` — `docs/sessions/2026-08-17-treadmill-lc-english.md`, `.agent/rules/treadmill-lc-continue.md`, and `.claude/skills/treadmill-lc-continuation/SKILL.md`.

## Goal
`vaila/treadmill_lc.py` and its accompanying tests satisfy all four architecture requirements without regressing existing functionality:
1. **Logical Calibration Pipeline**: Execution follows the strict chronological order:
   1. *Zero Offset / Tare*: baseline voltage subtraction per cell.
   2. *Calibration Matrix*: polynomial/linear gain matrix conversion ($V \rightarrow \text{kg} \rightarrow \text{BW}$).
   3. *Coordinate Transformation*: 4-cell load deck geometry ($58\text{ cm ML} \times 113\text{ cm AP}$) to COP (`medial_lateral`, `anterior_posterior`, `vertical`).
   4. *Signal Filtering*: optional low-pass Butterworth filtering on transformed components.
   5. *Event Detection*: strike and step detection with strict side encoding (`0` for Right, `1` for Left).
2. **CLI & TOML Support**:
   - `argparse` accepts `--config <path.toml>`, `--input-dir <dir>`, `--output-dir <dir>`, `--step <all|filter|adjust|interpolate|process>`, and `--gui`.
   - Supplying `--config` runs headlessly in batch mode without prompting or opening GUI dialogs.
   - If no `--config` is provided (or `--gui` is passed), a single streamlined Tkinter window opens to select the `.toml` configuration file (retiring multiple cascading popups).
3. **Rich Console Display**:
   - Uses `rich.console.Console`, `rich.panel.Panel`, and `rich.table.Table` to output a stylized configuration card at startup and step-by-step pipeline execution milestones.
4. **Tests & Verification**:
   - `tests/test_treadmill_lc.py` and tests under `tests/treadmill_lc` pass 100% with `.toml` fixtures and CLI verification.
   - Preserves English UI labels (`COP X - Mediolateral`, `COP Y - Anteroposterior`) and backwards-compatible Portuguese input aliases (`tara`, `peso`, `LIMPO`).

## Verification (Governing Check)
- **True level:** 1 (Deterministic exit codes & assertions) and 2 (Ruff/Ty static validation).
- **Check (per iteration):**
  ```bash
  uv run ruff check vaila/treadmill_lc.py tests/test_treadmill_lc.py --fix
  uv run ruff format vaila/treadmill_lc.py tests/test_treadmill_lc.py
  uv run ty check vaila/treadmill_lc.py tests/test_treadmill_lc.py
  uv run pytest tests/test_treadmill_lc.py -v
  ```
- **Check (CLI headless run verification):**
  ```bash
  uv run python -m vaila.treadmill_lc --config tests/treadmill_lc/processing_configuration_used.toml --input-dir tests/treadmill_lc --step filter
  ```
- **Evidence:** Raw stdout/stderr of `pytest`, `ruff`, `ty`, and CLI run output recorded directly in transcript and persisted to state.
- **Completion criterion:**
  - 100% tests in `tests/test_treadmill_lc.py` pass.
  - Zero Ruff lint/format errors.
  - Zero new Ty type violations.
  - Headless CLI execution succeeds with Rich formatted output and produces valid timestamped output directories.
  - Mandatory script metadata updated in `vaila/treadmill_lc.py`, `README.md`, and `vaila/help/treadmill_lc.md`.
- **Verifier protection:**
  - Canonical test data fixtures in `tests/treadmill_lc/*.csv` and `info_s01_d01.txt` must remain intact and unmodified.
  - Matplotlib non-interactive `Agg` backend must remain locked during test execution.
- **Scientific validity:**
  - Treadmill deck dimensions: $58.0\text{ cm}$ mediolateral width, $113.0\text{ cm}$ anteroposterior length.
  - Cell positions: Cell 1 (anterior-left), Cell 2 (posterior-left), Cell 3 (anterior-right), Cell 4 (posterior-right).
  - Side encoding: `0 = Right`, `1 = Left`.
  - Coordinate axes: `medial_lateral` (X), `anterior_posterior` (Y), `vertical` (Z / Total GRF).

## Trigger
Manual. Invoked by operator or test harness. Mid-loop re-entry loads `loops/state/treadmill-lc-refactor-loop-state.json` and resumes from the current active target.

## Iteration
0. **Initialization:**
   - Verify environment, load `loops/state/treadmill-lc-refactor-loop-state.json`.
   - Snapshot pre-refactor baseline: run `uv run pytest tests/test_treadmill_lc.py -v` (assert 39 passed baseline).
1. **Target 1: Pipeline Reordering & Data Structures:**
   - Restructure `load_data()` and pipeline functions into the explicit 5-stage order: Zero Offset → Calibration Matrix → Coordinate Transformation → Signal Filtering → Event Detection.
   - Enforce side encoding: `0 = Right`, `1 = Left` in step dictionaries (`"side": 0` / `"side": 1`, with `"foot": "R"` / `"foot": "L"` preserved for legacy compatibility).
   - Standardize English biomechanics dictionary keys: `medial_lateral`, `anterior_posterior`, `vertical`.
   - Run Governing Check (unit tests + ruff).
2. **Target 2: Unified CLI & TOML Configuration:**
   - Implement `load_process_config_from_toml(path)` and CLI `argparse` with `--config`, `--input-dir`, `--output-dir`, `--step`, `--gui`.
   - Implement single streamlined Tkinter dialog `select_config_file_dialog()` when `--gui` or no `--config` is passed.
   - Run Governing Check (CLI headless check).
3. **Target 3: Rich Console Dashboard:**
   - Add `rich.console.Console`, `rich.panel.Panel`, and `rich.table.Table` logging for configuration display and step-by-step progress.
   - Verify terminal output matches vailá design patterns (`sam3sapiens2.py` / `gputest.py`).
4. **Target 4: Test Suite & Fixture Hardening:**
   - Update `tests/test_treadmill_lc.py` to add tests for TOML CLI loading, 0/1 side encoding assertions, 5-stage pipeline order validation, and Rich logging smoke test.
   - Run full regression test suite across `tests/`.
5. **Target 5: Documentation & Metadata:**
   - Update script header in `vaila/treadmill_lc.py` (Version, Update Date).
   - Update `vaila/help/treadmill_lc.md`, `vaila/help/treadmill_lc.html`, `vaila/help/index.md`, and `README.md`.
6. Retain changes only if all tests pass; otherwise roll back to the previous iteration commit/stash.
7. Record decisions and metrics into `loops/state/treadmill-lc-refactor-loop-state.json`.
8. Evaluate terminal states.

## Terminal States
- **success:** All 5 targets accepted, 100% pytest pass rate, ruff/ty clean, headless CLI works, Rich output validated, metadata updated.
- **no-op:** Code already conforms to the 5-stage order, TOML CLI, 0/1 side encoding, and Rich logging; zero changes required.
- **no-progress/stalled:** 2 consecutive iterations without resolving failing tests or type errors.
- **blocked:** Missing dependencies or unresolvable environment issues.
- **exhausted:** Maximum allocated turns or iterations reached before achieving green state.

## Guardrails
- **Maximum allocation:** 10 iterations, 30 tool turns.
- **Human approval required:** Git commit on `iatm` branch, deleting existing test fixtures.
- **Isolation:** Headless execution must not launch blocking GUI loops or spawn duplicate Tk root windows.
- **Protected verifier:** Test data in `tests/treadmill_lc/*.csv` must remain immutable.
- **Rollback:** `git checkout -- vaila/treadmill_lc.py tests/test_treadmill_lc.py` per failed iteration.

## State Memory
- **Path:** `loops/state/treadmill-lc-refactor-loop-state.json`
- **Persist:** Baseline status, active target, accepted changes, raw test output, diagnostic counts, and lessons learned.
- **Recovery:** Re-reading the JSON state file reconstructs the active stage, verified targets, and pending goals.

## Skills
- `$investigate-first` — Diagnose test failures and coordinate mapping mismatches before applying edits.
- `$surgical-patch` — Apply precision modifications to `vaila/treadmill_lc.py` and `tests/test_treadmill_lc.py`.
- `$verify-and-stop` — Validate test suite completion and enforce stop conditions.
- `$treadmill-lc-continuation` — Maintain repository conventions, English labels, and Portuguese alias compatibility.

## Why It Works
- **Explicit Stage Separation:** Prevents coupling between sensor voltage offsets, physical gain calibration, and anatomical coordinate frames.
- **Strict Integer Side Encoding (`0/1`):** Eliminates ambiguity across internationalized codebases and machine learning downstream processors while retaining string aliases for human inspection.
- **Headless-First TOML Architecture:** Enables fully automated batch processing in clusters/CI without human interaction, while providing a single, clean file-dialog GUI when needed.
- **Rich Status Reporting:** Offers immediate visibility into pipeline progress and parameter confirmation in terminal and logs.

## How to Trigger
### Context-bound
Execute the loop iteratively in the active assistant conversation until reaching a named terminal state.

### Fresh-context / Ralph
```bash
# External loop runner pattern:
# 1. Read loops/treadmill-lc-refactor-loop.md and loops/state/treadmill-lc-refactor-loop-state.json
# 2. Execute next iteration target
# 3. Update state file
# 4. Stop when status is 'success', 'no-op', 'blocked', or 'exhausted'
```

## Health Metrics
- **Cost per accepted change:** `total iterations / accepted targets`.
- **Test pass rate:** `100%` across `tests/test_treadmill_lc.py`.
- **Ruff violations:** `0`.
- **Ty diagnostic count:** $\le$ baseline.
