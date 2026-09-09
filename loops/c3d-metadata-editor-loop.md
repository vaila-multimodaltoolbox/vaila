---
name: c3d-metadata-editor-loop
category: Vailá
trigger: manual
verification-level: 1
theory-base: arXiv:2607.00038
---

# C3D Metadata Editor & Environment Inspector Loop

## Description
Provide an end-to-end, reproducible pipeline to inspect, edit, and create C3D metadata
(frame rate, analog channels/rates, units, manufacturer, software tags) with GUI and CLI parity,
eliminate terminal flooding during CSV/C3D staging, prevent unintentional auto-saving in
`rearrange_data.py`, and provide an environment/imagination inspector for `.venv` and IPython workflows.

## Use When
- Inspecting or modifying header and parameter groups of `.c3d` files (`POINT:RATE`, `POINT:UNITS`,
  `ANALOG:RATE`, `ANALOG:USED`, `MANUFACTURER:COMPANY`, `MANUFACTURER:SOFTWARE`).
- Creating template `.c3d` files with specified markers, analog channels, and sample rates.
- Suppressing noisy terminal logging of marker arrays and paths during file staging/saving.
- Ensuring column reordering (`rearrange_data.py`) saves only on explicit user request (`Ctrl+S`, menu, or Save button).
- Checking installed scientific library versions (e.g. `ezc3d 1.7.2`, PyTorch, NumPy, OpenCV) and
  guiding users on `.venv` activation and interactive IPython sessions.
- Not for: writing raw DLT reconstruction coefficients (use `rec3d.py` / `rec3d_one_dlt3d.py`);
  not for video keypoint inference (use `vaila/soccerfield_keypoints_ai.py`).

## Inputs
1. `c3d_metadata` — `vaila/c3d_metadata.py` (CLI + Tkinter GUI dialog for metadata read/write/create).
2. `vaila_env` — `vaila/vaila_env.py` (CLI + Tkinter GUI dialog for environment, package versions, and IPython guide).
3. `rearrange_data` — `vaila/rearrange_data.py` (`ColumnReorderGUI` with menu bar, `Ctrl+S`, discard on close).
4. `readc3d_export` / `readcsv_export` — `vaila/readc3d_export.py` and `vaila/readcsv_export.py` (`verbose=False` by default).
5. `edit_csv_c3d` — `vaila/edit_csv_c3d.py` (quiet staging and save-only output emission).
6. `vaila_main` — `vaila.py` (button `C_A_r2_c2`, "imagination!" button, `--env-info` and `--metadata` CLI flags).
7. `vaila_cli_menu` — `vaila/vaila_cli_menu.py` (registered code `C_A_r2_c2` -> `edit_c3d_metadata`).
8. `tests` — `tests/test_c3d_metadata.py`, `tests/test_vaila_env.py`, `tests/test_rearrange_data_gui.py`,
   `tests/test_readc3d_export_gui.py`, `tests/test_edit_csv_c3d.py`, `tests/test_vaila_cli_menu.py`.
9. `help` — `vaila/help/c3d_metadata.md`, `vaila/help/vaila_env.md`, and updated `vaila/help/index.md`.

## Goal
Objectively verifiable end state:
1. Terminal output remains clean during data file selection, staging, and conversions (no hundreds of marker printouts).
2. `rearrange_data.py` does not save automatically upon closing; saves via `Ctrl+S`, File menu, or explicit Save button; prompts to confirm discard if changes exist.
3. C3D metadata can be inspected, updated, and created from both CLI and Tkinter GUI dialog without ezc3d C++ SWIG encoding failures.
4. Analog rate stays strictly proportional to point rate according to the subframe ratio invariant: `nb_analog_frames * POINT:RATE == nb_point_frames * ANALOG:RATE`.
5. Button `C_A_r2_c2` in Frame C opens the C3D Metadata GUI; CLI `vaila.py --metadata` delegates to CLI/GUI; CLI `vaila.py --env-info` prints system and dependency report.
6. The "imagination!" button opens the Environment & Imagination window with copyable `.venv` activation commands and IPython biomechanical cheatsheets.
7. All unit tests pass, ruff and ty type checking are clean, and user-facing documentation is updated.

**Targets (worst-first):**
1. **Sanitize C3D parameters for ezc3d:** Strip Latin-1 / surrogate escapes (`\udcb2`, `mm/s²` -> `mm/s^2`) so `c.write()` never throws SWIG `TypeError` or `ValueError`.
2. **Synchronize analog subframe ratio:** Automatically maintain `ANALOG:RATE = POINT:RATE * subframe_ratio` on point rate modifications to avoid ezc3d frame count mismatch exceptions.
3. **Silence terminal output:** Ensure `importc3d`, `c3d_markers_to_dataframe`, and `auto_create_c3d_from_csv` respect `verbose=False` by default; print only reproduction commands (`>> Equivalent CLI:`).
4. **Non-destructive GUI close:** Implement `WM_DELETE_WINDOW` with unsaved change detection in `rearrange_data.py`; bind `<Control-s>` / `<Control-S>` to save.
5. **Environment & Dependency Inspector:** Query `importlib.metadata` for 16 core packages; detect OS and virtual environment path; generate OS-accurate activation commands and IPython snippets.
6. **GUI and CLI parity:** Implement full CLI flags in `vaila/c3d_metadata.py` (`--show`, `-i`, `-o`, `--fps`, `--point-units`, `--scale-coords`, `--manufacturer`, `--software`, `--create`, `--markers`, `--analogs`) and register `C_A_r2_c2` in `vaila.py` and `vaila_cli_menu.py`.
7. **Regression tests & Quality gates:** Write unit tests for C3D metadata, environment inspector, and GUI menus; ensure all 29+ test cases pass without warnings.

## Verification (Governing Check)
- **True level:** 1 (deterministic pytest assertions, CLI exit codes, and file existence) with level-2 ruff/ty static checks.
- **Check (every iteration):**
  ```bash
  uv run ruff check vaila/ tests/ --fix
  uv run ruff format vaila/ tests/
  uv run ty check vaila/
  uv run pytest tests/test_c3d_metadata.py tests/test_vaila_env.py tests/test_rearrange_data_gui.py tests/test_readc3d_export_gui.py tests/test_edit_csv_c3d.py tests/test_vaila_cli_menu.py -v
  ```
- **Evidence:** Raw pytest output, exit codes, and test counts recorded in `loops/state/c3d-metadata-editor-loop-state.json`.
- **Completion criterion:**
  - `test_c3d_metadata.py` (4 tests) passing.
  - `test_vaila_env.py` (6 tests) passing.
  - `test_edit_csv_c3d.py` (7 tests) passing.
  - `test_rearrange_data_gui.py` and `test_readc3d_export_gui.py` passing.
  - `test_vaila_cli_menu.py` (20 tests) passing.
  - Ruff and Ty clean on all touched modules.

## Trigger
Manual developer trigger or CI suite run after modifications to C3D, CSV, or GUI menu modules.

## Iteration
0. On first iteration, validate fixture `tests/C3D_to_CSV/C3D_to_CSV_01.c3d`.
1. Run baseline verification and record passing count.
2. Address highest-priority target from Targets list.
3. Make one targeted change in `vaila/` or `tests/`.
4. Run governing check and capture stdout/stderr.
5. Retain edit only if governing check passes without regressions; otherwise revert change with `git checkout`.
6. Record result in state memory.
7. Evaluate terminal states; stop on success.

## Terminal States
- **success:** All 7 targets met, 39+ tests pass, ruff/ty clean, docs synchronized.
- **no-op:** No changes requested or codebase already satisfies all criteria.
- **no-progress/stalled:** Two consecutive iterations fail to resolve a target without introducing regressions.
- **blocked:** Missing required Python 3.12 dependencies (`ezc3d`, `numpy`, `pandas`) or corrupt C3D binary fixture.
- **exhausted:** Maximum allocated turn limit (10 iterations) reached.

## Guardrails
- **Maximum allocation:** 10 iterations, 30 tool calls.
- **Human approval required:** Deleting existing files outside timestamped output or staging folders.
- **Protected verifier:** Test files and fixtures cannot be relaxed or hard-coded to produce false passes.
- **Rollback:** Single-iteration `git checkout <file>` or `replace_file_content`.

## State Memory
- **Path:** `loops/state/c3d-metadata-editor-loop-state.json`.
- **Persist:** Current target, iteration count, test results, accepted changes, and error logs.
- **Recovery:** Reads state file on startup; continues from highest unaccepted target.

## Skills
- `$safe-refactor` — Restructure `ColumnReorderGUI` menu and handlers without breaking existing button bindings.
- `$surgical-patch` — Patch `c3d_metadata.py` parameter sanitization and CLI delegation.
- `$verify-and-stop` — Validate test suite and documentation completeness before concluding.

## Why It Works
- **SWIG string sanitization:** Prevents low-level C++ crashes caused by non-ASCII unit characters in historical C3D files.
- **Analog rate invariant preservation:** Prevents frame dimension mismatch errors in `ezc3d.c3d.write()`.
- **Non-destructive GUI workflow:** Eliminates user frustration from accidental file overwrites on window close.
- **Quiet staging:** Maintains clean CLI logs so users can easily see and copy the `>> Equivalent CLI:` execution commands.
- **GUI/CLI parity:** Allows all C3D metadata operations to run headlessly in scripts and HPC clusters or interactively in Tkinter.
