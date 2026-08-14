---
name: sam3-batch-auto-resume-loop
category: Vailá
trigger: manual
verification-level: 1
theory-base: arXiv:2607.00038
---

# SAM3 Batch Auto-Resume Loop

## Description
Make `vaila/sam3sapiens2.py` and `vaila/sam3dinov3.py` auto-detect and resume an
interrupted/previous batch run by default — no `--resume` flag required — while
keeping `--resume` as an explicit pin and adding an explicit opt-out for a clean
run. Reuses the existing per-video `sam3sapiens2_summary.json` /
`sam3dinov3_summary.json` completion marker already read by
`load_completed_summary()`; does not introduce a second bookkeeping file.

## Use When
- Extending the batch `main()` loop in either module with auto-resume detection,
  the opt-out flag, and the start-of-run completed/remaining summary line.
- Adding/():running the regression tests that prove auto-resume, opt-out, and
  interrupted-run recovery, without requiring GPU hardware for the per-iteration
  check.
- Not for: SAM3/SAM 3D Body model changes, output CSV schema changes, or any
  edit to `load_completed_summary()`'s own completion criteria — those are out
  of scope and must not be touched by this loop.

## Inputs
1. `target_files` — fixed: `vaila/sam3sapiens2.py`, `vaila/sam3dinov3.py`. No other
   source file is in scope unless investigation proves a shared helper module is
   the correct home (see Iteration step 0).
2. `real_test_path` — `/media/preto/Expansion/Anna/All_videos_subjects_Sync/S4/s4_pre_R_ah`
   (or its current video files). Used only for the final human-checkpoint GPU
   run, never for per-iteration checks.

## Goal
On a second invocation of `sam3sapiens2.py`/`sam3dinov3.py` with the same
`--input` and an `--output` whose parent already holds a matching
`processed_sam3sapiens2_*`/`processed_sam3dinov3_*` batch directory, the script
auto-resumes that directory by default: videos with a valid completed summary
JSON are skipped with `[SKIP] Already processed: <name>`, a startup line
reports `Resume: N/M videos already completed, M-N remaining`, and only the
remaining videos run. `--resume <dir>` still pins an explicit directory.
`--fresh` (new flag) forces a new timestamped output directory even when a
match exists. Both scripts behave identically and share one detection helper.

## Verification (Governing Check)
- **True level:** 1 (deterministic) per iteration; level 5 (human checkpoint)
  once at the end for the real GPU run.
- **Check (per iteration):**
  ```
  uv run ruff check vaila/sam3sapiens2.py vaila/sam3dinov3.py --fix
  uv run ruff format vaila/sam3sapiens2.py vaila/sam3dinov3.py
  uv run ty check vaila/sam3sapiens2.py vaila/sam3dinov3.py
  uv run pytest tests/test_sam3sapiens2.py tests/test_sam3dinov3.py -v
  ```
- **Check (final human checkpoint only):** run `sam3sapiens2.py` (or
  `sam3dinov3.py`) twice against `real_test_path` with the same `-o`: kill/let
  the first run finish 1 video, confirm the second run's stdout shows the
  `Resume:` line and skips the completed video without `--resume`.
- **Evidence:** raw pytest output (pass/fail counts + names), ruff/ty diagnostic
  counts before vs. after (must not increase on files outside the two targets —
  scope discipline per the 2026-08-11 ruff-directory incident noted in project
  memory), and for the checkpoint the exact two stdout transcripts.
- **Completion criterion:** ruff clean, ty diagnostic count on the two target
  files unchanged-or-improved (never worse than the pre-loop baseline), all new
  + existing tests in both test files pass, and the human checkpoint transcript
  shows the `Resume:` skip line with zero `--resume` flag passed.
- **Verifier protection:** `load_completed_summary()` and the per-video summary
  JSON schema are frozen — the loop may call them, never edit their completion
  logic to make a test pass. New tests use synthetic fixture directories
  (fabricated `sam3sapiens2_summary.json`/`sam3dinov3_summary.json` +
  `video.stem` subdirs) so no GPU/model weight is needed to prove skip logic.
- **Scientific validity:** not applicable — this is a batch-orchestration
  change, not a numeric/biomechanical one. No output CSV schema, unit,
  coordinate frame, or keypoint content is touched. The one thing to protect is
  **idempotency**: re-running must never re-write or truncate a completed
  video's existing output files.

## Trigger
Manual. Operator (paulopreto) starts the loop by invoking it in this session or
a fresh one; re-entry mid-loop reads `state/sam3-batch-auto-resume-loop-state.json`
and continues rather than restarting step 0.

## Iteration
0. First iteration only: read both `main()` functions in full (already located
   at `sam3sapiens2.py:1909` and `sam3dinov3.py:1944`), confirm exactly where
   `batch_summary` (schema `vaila_sam3sapiens2_batch_v1` /
   `vaila_sam3dinov3_batch_v1`) is written relative to the per-video loop —
   if it is written only after the loop finishes (not progressively), a bare
   `output_base` scan cannot tell "this dir belongs to this input" until the
   very end. In that case the auto-detect helper must write a lightweight
   `BATCH_INPUT.json` (`{"input": str(input_path), "schema": "..."}`) into
   `output_base` immediately after `output_base.mkdir(...)`, before the loop
   starts, so an interrupted run is still matchable. Record the finding in
   state before writing any code.
1. Load this spec + `state/sam3-batch-auto-resume-loop-state.json`; confirm
   turn budget and human-approval gates below are still intact.
2. Snapshot baseline: `git status vaila/sam3sapiens2.py vaila/sam3dinov3.py`
   (must be clean or only this loop's prior commits), run the governing check
   once to record the pre-change diagnostic/test baseline.
3. Rank remaining targets worst-first from this fixed list (skip any already
   accepted per state file):
   1. `sam3sapiens2.py`: add `_resolve_auto_resume_output_base()` (or shared
      module placement per step 0's finding) + `--fresh` flag + startup
      completed/remaining summary line + wire into `main()`'s existing
      `args.resume is not None` branch so it also fires when `args.resume is
      None` and a match exists.
   2. `sam3dinov3.py`: import the same helper via the existing dual-import
      block (`from .sam3sapiens2 import ...` / `from sam3sapiens2 import ...`)
      rather than duplicating it; wire into its `main()` the same way.
   3. Tests: red-before/green-after pair per script — a test that fails
      against the pre-change `main()` (proves it exercises the new behavior,
      not a tautology) then passes after, plus: opt-out `--fresh` creates a
      new dir even with a match; a video with a completed summary is skipped
      and printed; a video without one still runs; interrupted-run match via
      `BATCH_INPUT.json` if step 0 found progressive-write is false.
   4. Metadata: bump both files' header Update Date/Version to the current
      global vailá version per repo convention; update
      `vaila/help/sam3sapiens2.md`/`.html` and `vaila/help/sam3dinov3.md`/`.html`
      if the new `--fresh` flag or default behavior needs documenting; update
      root `README.md` `Last updated:` line.
4. Invoke `caveman:surgical-patch` for each of steps 3.1–3.2 (narrowest
   responsible layer, existing tests as regression guard) and the
   `test-writer` agent (`.claude/agents/test-writer.md`) for step 3.3, one
   target per iteration — never both scripts' behavior changes and the tests
   in the same iteration.
5. Run the governing check; record raw stdout/stderr for ruff, ty, pytest.
6. Retain the change only if: ruff clean, ty diagnostic count on the two
   target files not increased, and pytest shows 0 regressions among
   previously-passing tests. Otherwise revert only that iteration's file via
   `git checkout -- <file>` (each target touches at most 2 files, always
   individually stageable) and record the failure reason.
7. Curate lessons (e.g. "progressive vs. end-of-run batch_summary write",
   "shared-helper import direction") only when the governing check confirms
   them; persist state, evidence, counters, cost atomically (write to a `.tmp`
   file then rename).
8. Evaluate terminal states; otherwise continue to the next unresolved target.

## Terminal States
- **success:** all 4 ranked targets accepted, governing check green on both
  files, AND the level-5 human checkpoint transcript (real path, two runs, no
  `--resume` on the second, `Resume:` line present, completed video skipped)
  is recorded in state.
- **no-op:** not expected — investigation already confirmed the automatic
  default path is genuinely missing (`args.resume is not None` gates the
  existing skip logic in both scripts). Only reachable if step 0 finds the
  feature was added since this spec was written; must be confirmed by reading
  current `main()`, not assumed.
- **no-progress/stalled:** 2 consecutive iterations where the governing check's
  failing-test count and ty diagnostic count are both unchanged after an
  attempted fix.
- **blocked:** CUDA/GPU unavailable for the final checkpoint, or
  `real_test_path` not mounted/accessible (external USB drive) — loop pauses
  at step 4 (of 4) with everything else complete; resumes when the path is
  available again, does not require restarting from step 0.
- **exhausted:** 20 iterations elapsed (see Guardrails) without reaching
  success.

Errors, missing evidence, and budget exhaustion are never success.

## Guardrails
- **Maximum allocation:** 20 iterations, no external API/token spend beyond
  normal Claude Code usage; the level-5 GPU checkpoint run is bounded to the
  videos already present in `real_test_path` (no new data fetch).
- **Human approval required:** running the final GPU checkpoint against
  `real_test_path` (multi-minute, real hardware, real external drive — confirm
  the drive is mounted and the run is intentional before starting); any `git
  checkout --`/revert affecting more than the current iteration's file; any
  edit outside the fixed `target_files` list (surfaced, not auto-expanded).
- **Isolation and credentials:** no network access needed; no credentials
  touched; runs against local `.venv` only.
- **Protected verifier:** `load_completed_summary()`, the summary JSON schema
  (`sam3sapiens2_summary.json`/`sam3dinov3_summary.json` field names), and the
  existing `--resume`/`plan_video_processing` semantics are frozen — the loop
  may read and call them, never redefine "complete."
- **Rollback:** `git checkout -- vaila/sam3sapiens2.py` /
  `vaila/sam3dinov3.py` / the specific new test file, scoped to the file(s)
  touched in the failed iteration only.

## State Memory
- **Path:** `loops/state/sam3-batch-auto-resume-loop-state.json`.
- **Persist:** baseline ruff/ty/pytest counts, step-0 finding (progressive vs.
  end-of-run batch_summary write and the resulting design — helper-only vs.
  `BATCH_INPUT.json` marker), per-target accept/reject with raw check output,
  curated lessons, iteration count, terminal status, human-checkpoint
  transcript once run.
- **Recovery:** a fresh context reads this file first; if the last write has no
  matching "iteration complete" marker, treat that iteration's file changes as
  unverified and re-run the governing check before trusting them (never trust
  an uncommitted diff without re-running the check).

## Skills
- `caveman:surgical-patch` — narrow behavior change to `main()` in each script,
  one file per invocation, existing tests as the regression guard.
- `test-writer` (agent, `.claude/agents/test-writer.md`) — red-before/green-after
  tests for auto-resume, `--fresh` opt-out, and interrupted-run recovery.
- `yolo-fb-gui-cli` — reference for this repo's GUI→CLI mirror convention if
  the new `--fresh` flag needs a GUI-side checkbox/exposure later (out of
  scope for this loop unless the human explicitly extends it).

## Sub-Loops
None.

## Why It Works
One shared helper (imported by `sam3dinov3.py` from `sam3sapiens2.py`, matching
the file's existing dual-import pattern) prevents the two near-identical
`main()` loops from drifting into two different resume behaviors. Worst-first,
one-file-per-iteration changes keep each accepted diff small enough to revert
cleanly. Red-before/green-after tests prove the new tests actually exercise
the added behavior rather than passing vacuously. Freezing
`load_completed_summary()`/the summary schema stops the loop from "fixing" a
failing test by redefining what counts as complete. The level-1 per-iteration
check (no GPU) keeps iteration cost low; reserving the real GPU run for a
single human-approved checkpoint avoids burning hardware time on a
batch-orchestration change that pytest can already prove deterministically.

## How to Trigger
### Context-bound
In this Claude Code session: "resume sam3-batch-auto-resume-loop" or "continue
loops/sam3-batch-auto-resume-loop.md" — reads state file, continues from the
last recorded iteration.

### Fresh-context / Ralph
External runner re-reads `loops/sam3-batch-auto-resume-loop.md` and
`loops/state/sam3-batch-auto-resume-loop-state.json` every turn, re-runs the
governing check before trusting any prior iteration's uncommitted diff, and
stops only on a state file `"status"` of `success`, `blocked`, or `exhausted`
— never on its own judgment that the diff "looks done."

## Health Metrics
- **Cost per accepted change:** iterations spent / targets accepted (4 max).
- **Ty diagnostic delta:** post-loop count minus pre-loop baseline on the two
  target files (must be ≤ 0).
- **Test count delta:** new passing tests added to
  `tests/test_sam3sapiens2.py` + `tests/test_sam3dinov3.py`.
- **Checkpoint latency:** wall-clock time of the final real-data GPU run
  (informational, not a gate).
