---
name: dlt3d-ref3d-multi-format-loop
category: Vailá
trigger: manual
verification-level: 1
theory-base: arXiv:2607.00038
---

# DLT3D REF3D Multi-Format Parser

## Description
Extend `vaila/dlt3d.py` so REF3D calibration files in three real-world layouts
auto-normalize to the canonical format-1 wide CSV (`frame,p1_x,p1_y,p1_z,…`)
before DLT3D coefficient computation and REC3D reconstruction.

## Use When
- Loading legacy or external REF3D exports that store one point per row (xyz
  only, or index + xyz) instead of the vailá wide header row.
- Verifying that format 2/3 fixtures under
  `tests/DLT3D_and_Rec3d/ref3d_realworld/` produce identical DLT outputs to
  format 1.
- Not for lens-distortion modeling (`dlt-distortion-loop.md`) or REF3D export
  from `drawsportsfields.py` (already emits format 1).

## Inputs
1. `format1` — `tests/DLT3D_and_Rec3d/ref3d_realworld/ref3d_realworld_format1.ref3d`
   (wide CSV, header `frame,p1_x,…`, single row, 25 points, metres).
2. `format2` — `ref3d_realworld_format2.ref3d` (25 rows × 3 cols `x,y,z`, no header).
3. `format3` — `ref3d_realworld_format3.ref3d` (25 rows × 4 cols `index,x,y,z`, no header).
4. `pixel_fixture` — `tests/DLT3D_and_Rec3d/pixelcorrds/c01_markers_1_line.csv`
   (25 labelled 2D points, frame column).

## Goal
`read_ref3d_file` / `normalize_ref3d_to_format1` accept formats 1–3, always
return an internal format-1 `DataFrame`, and `process_files` yields identical
11-parameter DLT3D coefficients for all three fixtures paired with the pixel
file. Backward compatibility with existing format-1 consumers (`rec3d_one_dlt3d`,
`drawsportsfields` export tests) is preserved.

## Verification (Governing Check)
- **True level:** 1 (deterministic numeric + schema equality).
- **Check:**
  ```bash
  uv run pytest tests/test_dlt3d_ref3d_formats.py -v
  uv run pytest tests/test_dlt_rec.py tests/test_drawsportsfields_ref3d_export.py -v
  uv run ruff check vaila/dlt3d.py --fix && uv run ruff format vaila/dlt3d.py
  uv run ty check vaila/dlt3d.py
  ```
- **Evidence:** full pytest stdout showing:
  - `detect_ref3d_format` → 1/2/3 for each fixture;
  - `normalize_ref3d_to_format1` frames equal across formats;
  - `process_files` DLT vectors `allclose` per frame across formats;
  - format-3 index column respected when rows are shuffled.
- **Completion criterion:** all tests above pass; `dlt3d.py` header
  Version/Last Updated synced to global vailá version; `vaila/help/dlt3d.md` +
  `.html` document the three REF3D layouts.
- **Verifier protection:** fixture files under `tests/DLT3D_and_Rec3d/ref3d_realworld/`
  are read-only inputs; tests assert against them, not generated in-module values.
- **Scientific validity:** coordinates stay in the same world frame (metres, origin
  at format-1 reference); point labels `p1..p25` preserved; minimum 6 points for
  DLT3D unchanged.

## Trigger
Manual — developer or agent resumes after `/preto-loop` or explicit task. One
iteration may complete the work if red/green tests pass on first implementation.

## Iteration
0. Validate fixtures exist and record baseline pytest output (format 1 only before change).
1. Load spec + `loops/state/dlt3d-ref3d-multi-format-loop-state.json` if present.
2. Run governing check; capture failures.
3. Select worst failure (detection vs normalization vs DLT parity).
4. Invoke `$surgical-patch` — exactly one attributable change in `vaila/dlt3d.py`
   or `tests/test_dlt3d_ref3d_formats.py`.
5. Re-run governing check; append raw stdout to state.
6. Retain change only if non-regressive; else `git checkout --` scoped files.
7. Update state counters and lessons.
8. Exit on terminal state.

## Terminal States
- **success:** governing pytest + ruff + ty pass; three formats normalize identically;
  DLT outputs match; help docs updated.
- **no-op:** parser already supports all formats and tests green (baseline only).
- **no-progress/stalled:** two consecutive iterations with identical failing test
  and no new evidence.
- **blocked:** fixture files missing or pixel/reference point counts diverge.
- **exhausted:** 8 iterations or 200k maker tokens without success.

## Guardrails
- **Maximum allocation:** 8 iterations, 200k maker tokens.
- **Human approval required:** git commit, changing fixture coordinates, deleting
  `ref3d_realworld_format*.ref3d`.
- **Isolation and credentials:** local pytest only; no network.
- **Protected verifier:** `tests/test_dlt3d_ref3d_formats.py` and fixture paths
  require explicit user approval to weaken assertions.
- **Rollback:** `git checkout -- vaila/dlt3d.py tests/test_dlt3d_ref3d_formats.py`
  per failed iteration.

## State Memory
- **Path:** `loops/state/dlt3d-ref3d-multi-format-loop-state.json`.
- **Persist:** baseline exit codes, per-format detection results, accepted diffs,
  pytest stdout tails, iteration count, token estimate.
- **Recovery:** fresh context reads this file + spec; resumes at next iteration
  if status not `success`.

## Skills
- `$surgical-patch` — minimal parser change per iteration.
- `$verify-and-stop` — final gate before marking success.
- `$investigate-first` — classify unknown REF3D layout before editing.

## Why It Works
Auto-detection + normalization to format 1 keeps downstream REC3D code unchanged;
deterministic cross-format DLT parity proves label mapping is correct; frozen
fixtures prevent spec gaming; one-change-per-iteration isolates regressions.

## How to Trigger
### Context-bound
User message: `/goal Implement dlt3d REF3D formats 2/3 per loops/dlt3d-ref3d-multi-format-loop.md`

### Fresh-context / Ralph
Each turn: read this file + state JSON → run governing check → one patch → update state →
stop only on named terminal state.

## Health Metrics
- **Cost per accepted change:** total tokens / retained non-regressive iterations.
- **Format parity:** max abs diff of DLT L coefficients across format 1/2/3 per frame.
- **Detection accuracy:** 3/3 fixtures classified correctly.
