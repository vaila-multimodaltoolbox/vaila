---
name: interp-smooth-split-c3d-loop
category: Data Science
trigger: manual
verification-level: 1
theory-base: arXiv:2607.00038
---

# Interp / Smooth / Split C3D Roundtrip Loop

## Description
Extend `vaila/interp_smooth_split.py` so directories of `.c3d` files are processed
the same way as CSV: stage markers to CSV via existing *vailá* helpers, run the
shared numerical pipeline, then write results back to `.c3d` — without changing
the proven CSV path and without inventing a second filter implementation.

## Use When
- Adding or repairing C3D input/output for Smooth & Filter (`interp_smooth_split`).
- Aligning C3D I/O with the Frame C pattern already used by `edit_csv_c3d.py`.
- Not for: rewriting filter math (that is `interp-smooth-split-parity-loop.md`);
  not for DLT/rec3d C3D writers; not for video/markerless pipelines.

## Inputs
1. `module` — `vaila/interp_smooth_split.py` (GUI + CLI + `process_file` / batch).
2. `core` — `vaila/interp_smooth_core.py` (unchanged numerical contract unless a
   C3D staging bug forces a shared fix).
3. `c3d_read` — `vaila/readc3d_export.c3d_markers_to_dataframe` (markers DataFrame
   + meta: rate, units, analogs).
4. `c3d_write` — `vaila/readcsv_export.auto_create_c3d_from_csv`.
5. `precedent` — `vaila/edit_csv_c3d.py` staging/write pattern
   (`_stage_inputs` / `_write_output`) and `tests/test_edit_csv_c3d.py`.
6. `tests` — extend `tests/test_interp_smooth_split.py` and/or add
   `tests/test_interp_smooth_split_c3d.py`; synthetic C3D only via
   `auto_create_c3d_from_csv` under `tmp_path` (no tracked `.c3d` required).
7. `help` — `vaila/help/interp_smooth_split.md` + `.html` (+ index) when CLI/GUI
   surface documents C3D.

## Goal
Objectively verifiable end state: Smooth & Filter accepts mixed or C3D-only
input directories; each `.c3d` is converted to marker CSV, processed by the
existing pipeline, and saved again as `.c3d` in a timestamped output directory;
source files stay untouched; CSV-only runs behave as today.

**Targets (worst-first):**

1. **Stage + write plumbing:** For each input `.c3d`, call
   `c3d_markers_to_dataframe` → process DataFrame/CSV with current
   `process_file` (or equivalent shared batch step) →
   `auto_create_c3d_from_csv` with `meta` (`point_rate`, `point_units`,
   `analog_df` / `analog_rate` when present). Prefer extracting a small shared
   staging helper or mirroring `edit_csv_c3d` rather than duplicating ezc3d
   logic inside the GUI.
2. **Identity roundtrip (no smooth):** With `interp_method=none` /
   `smooth_method=none` (or equivalent no-op config), output C3D matches
   input on POINT LABELS, RATE, UNITS, and XYZ (`atol` ≤ 1e-5), matching
   `test_c3d_identity_round_trip` in `test_edit_csv_c3d.py`.
3. **Occlusion / residuals:** Known NaN marker samples survive the C3D→CSV→C3D
   bridge as invalid points (negative residual), not as origin (0,0,0) valid
   samples — same assertions as `test_occlusion_residuals_preserved_through_round_trip`.
4. **Analogs when present:** Analog channels from meta are preserved through
   write-back when the smooth path does not edit them (points-only processing).
5. **Smooth affects points:** With a deterministic smooth (e.g. savgol or
   butterworth on a synthetic sinusoid in C3D), output XYZ differs from input
   where expected, finite, same labels/rate; CSV twin under the same config
   matches C3D-derived numeric columns within tolerance.
6. **Split on C3D:** When `do_split=True`, emit two `.c3d` halves (row partition
   consistent with CSV split semantics), not CSV-only leftovers.
7. **CLI / GUI surface:** `-i` directory may contain `.c3d`; GUI file filters /
   batch listing include `.c3d`; GUI Run prints `>>` equivalent CLI; headless
   path never creates `tk.Tk()`.
8. **Metadata:** Header date/version on touched `*.py`; help + index updated
   for C3D support; CSV regression suite still green.

**Logged interview defaults (2026-08-26; no Q2 reply):** fidelity bar **A**
(same as `edit_csv_c3d`); `do_split` emits two `.c3d`; destination `./loops/`.

## Verification (Governing Check)
- **True level:** 1 (pytest + CLI exit codes) with level-2 ruff/ty. Plot
  inspection is not acceptance. Scientific claims (mm vs m, clinical meaning
  of smoothed trajectories) stay out of “tests green.”
- **Check (every iteration):**
  ```bash
  uv run ruff check vaila/interp_smooth_split.py vaila/interp_smooth_core.py tests/test_interp_smooth_split.py tests/test_interp_smooth_split_c3d.py --fix
  uv run ruff format vaila/interp_smooth_split.py vaila/interp_smooth_core.py tests/test_interp_smooth_split.py tests/test_interp_smooth_split_c3d.py
  uv run ty check vaila/interp_smooth_split.py vaila/interp_smooth_core.py tests/test_interp_smooth_split.py tests/test_interp_smooth_split_c3d.py
  uv run pytest tests/test_interp_smooth_split.py tests/test_interp_smooth_split_c3d.py -v
  ```
  If `tests/test_interp_smooth_split_c3d.py` does not exist yet, create it in
  the first accepted iteration that adds C3D tests; until then run the CSV
  suite alone and treat missing C3D file as open target, not as success.
  Optionally (risk-proportional): `uv run pytest tests/test_edit_csv_c3d.py -v`
  when shared helpers are touched.
- **Evidence:** raw stdout/stderr of the check commands, stored under the
  current attempt in
  `loops/state/interp-smooth-split-c3d-loop-state.json`.
- **Completion criterion:**
  - Targets 1–8 accepted with recorded evidence.
  - New C3D tests pass; existing `test_interp_smooth_split` count stays ≥
    baseline 35 and all green.
  - Ruff/ty clean on touched files.
  - Source `.c3d`/`.csv` bytes unchanged after a run (asserted in tests).
  - Help documents C3D; metadata checklist satisfied.
- **Verifier protection:**
  - Do not weaken or delete `tests/test_edit_csv_c3d.py` or existing CSV
    interp tests to pass.
  - Do not fork a second Butterworth/savgol implementation for C3D; C3D must
    call the shared pipeline.
  - Prefer red-before/green-after for each new C3D test.
  - Governing check command block is frozen here.
- **Scientific validity:**
  - POINT UNITS and RATE come from C3D meta (or writer args); do not silently
    convert mm↔m.
  - Marker LABELS order must match the processed columns used for write-back.
  - Occluded samples must remain invalid in C3D meta_points, not valid zeros.
  - Analogs: preserve when present; do not invent analog data.
  - Sampling: Butterworth `fs` must equal POINT RATE (or explicit config)
    when filtering C3D-derived markers; document the chosen rule in help.
  - Hardware: CPU-only; `ezc3d` required for C3D tests (`pytest.importorskip`).

## Trigger
Manual. Load state JSON on re-entry; skip accepted targets.

## Iteration
0. Freeze inputs; confirm `ezc3d` importable; snapshot baseline pytest counts.
1. Load this spec + state; confirm budget/approvals.
2. Run governing check; record raw evidence.
3. Rank open targets worst-first (plumbing before polish).
4. One attributable change via `$surgical-patch` (or `$investigate-first`).
5. Re-run governing check; record evidence.
6. Retain only if non-regressive; else rollback that change set.
7. Curate evidence-backed lessons; atomically persist state.
8. Evaluate terminal states; else continue.

## Terminal States
- **success:** Targets 1–8 accepted; checks green; metadata current.
- **no-op:** Module already stages/writes C3D with tests covering identity,
  residuals, analogs, smooth, split, CLI/GUI — verified by evidence, not chat.
- **no-progress/stalled:** Two consecutive iterations with no accepted target
  and no new failing test that defines the next fix.
- **blocked:** `ezc3d` unavailable in the environment and cannot be installed
  via project deps; or `auto_create_c3d_from_csv` / `c3d_markers_to_dataframe`
  API breaks without a separate approved fix loop.
- **exhausted:** Hard budget reached.

Errors, missing evidence, and budget exhaustion are never success.

## Guardrails
- **Maximum allocation:** 12 iterations; 40 agent tool-turns; local `uv` only.
- **Human approval required:** git commit/push; deleting fixtures; changing
  shared C3D writer semantics used by `edit_csv_c3d`; adding network/paid deps.
- **Isolation and credentials:** headless tests; no second `tk.Tk()`; no
  network for governing check.
- **Protected verifier:** existing CSV tests and `test_edit_csv_c3d.py`;
  frozen check commands; no silent clamp of Nyquist/cutoff rules.
- **Rollback:** `git checkout -- <touched paths>` for that iteration only.

## State Memory
- **Path:** `loops/state/interp-smooth-split-c3d-loop-state.json`
- **Persist:** baseline counts, open/accepted targets, attempt evidence,
  lessons, decisions, cost counters.
- **Recovery:** re-read this markdown + state JSON; on corrupt JSON, restore
  `*.bak` or rebuild open targets from Goal (do not trust chat).

## Skills
- `$investigate-first` — diagnose C3D roundtrip or ezc3d failures.
- `$surgical-patch` — one staging/write/test change per iteration.
- `$verify-and-stop` — enforce completion; refuse filter-math scope creep
  (defer to `interp-smooth-split-parity-loop.md`).
- `$preto-loop` — maintain this specification; do not execute unless the
  operator explicitly starts the loop.

## Sub-Loops
- Optional handoff only: `interp-smooth-split-parity-loop.md` if a C3D failure
  reveals a core numerical bug — call out by filename; no circular calls.
  Parent×child budget ≤ `12 × 12` worst case if both run; prefer fixing
  shared core in the parity loop, then resume this one.

## Why It Works
- Reuses battle-tested C3D↔CSV helpers instead of a third ezc3d path.
- Deterministic synthetic roundtrip tests catch label/rate/occlusion bugs
  that plots miss.
- Separates “C3D I/O works” from “filter is biomechanically optimal.”
- One change + rollback keeps the CSV path safe.
- Explicit split→two `.c3d` prevents half-finished format support.

## How to Trigger
### Context-bound
Instruct the agent to execute `interp-smooth-split-c3d-loop` until a named
terminal state, re-reading the state JSON each turn.

### Fresh-context / Ralph
```text
1. Read loops/interp-smooth-split-c3d-loop.md
2. Read loops/state/interp-smooth-split-c3d-loop-state.json
3. One Iteration (exactly one change)
4. Update state atomically
5. Stop only on success | no-op | no-progress/stalled | blocked | exhausted
```
Illustrative — do not bypass approvals.

## Health Metrics
- **Cost per accepted change:** `iterations_or_turns / accepted_targets`.
- **Coverage progress:** `accepted_targets / 8`.
- **CSV regression:** `test_interp_smooth_split` pass count ≥ 35.
- **Source immutability:** failing “source bytes unchanged” → retain = false.
