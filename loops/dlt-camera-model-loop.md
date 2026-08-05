---
name: dlt-camera-model-loop
category: Biomechanics
trigger: manual
verification-level: 1
theory-base: arXiv:2607.00038
---

# DLT Camera Model Upgrade (parent)

## Description
Orchestrates two independent sub-loops that upgrade vailá's DLT/reconstruction
family: (A) add camera-lens-distortion terms to `dlt2d.py`/`dlt3d.py` and prove
they improve reconstruction accuracy against real calibration-volume reference
points, and (B) extend `rec3d.py`/`rec3d_one_dlt3d.py` to also emit a
Blender-importable 3D mesh sequence (SAM3+DINOv3 body mesh, alongside the
existing skeleton keypoints). Each sub-loop has its own governing check and
its own terminal states; this parent only sequences them and checks the
combined regression suite at the end.

## Use When
- You are ready to run sub-loop A, then sub-loop B, on the fixture data at
  `/home/preto/data/sep_runcod_01072026/REC3D_COD/rec3d_todo/`.
- Do not use this parent loop to write the planned Kwon3D-style didactic
  documentation of DLT theory — that is a one-shot writing task (human
  checkpoint, level 5), not a loop. Write it separately once both sub-loops
  have working, tested code to document accurately (a docs task written
  against unfinished math would need a rewrite).

## Inputs
1. `fixture_dir` — `/home/preto/data/sep_runcod_01072026/REC3D_COD/rec3d_todo/`.
   Contains `c1c2c3_cod.ref3d` (12 known 3D reference points, 1 row),
   `c{1,2,3}_cod_markers_1_line.csv` (pixel coords of those same 12 points,
   1 row each), `c{1,2,3}_cod_markers_1_line.dlt3d` (existing linear-DLT
   parameters), and `c{1,2,3}_cod.mp4` (source video for later markerless
   2D→3D + mesh work). **Treat every file in this directory as read-only
   ground truth** — the loop never overwrites it; all outputs go to new
   timestamped subfolders exactly as vailá convention requires.
2. `changed_files` — implicit: `vaila/dlt2d.py`, `vaila/dlt3d.py`,
   `vaila/rec3d.py`, `vaila/rec3d_one_dlt3d.py`, plus any new test files.

## Goal
Both sub-loops reach their own `success` terminal state without regressing
the existing DLT/rec3d test suite, and the combined regression suite below
passes on the real fixture data (not just synthetic fixtures).

## Verification (Governing Check)
- **True level:** 1 (deterministic) for the combined regression gate; each
  sub-loop's own governing check is documented in its file and is the
  authority for that sub-loop's terminal states.
- **Check:**
  ```bash
  uv run pytest tests/test_dlt_rec.py tests/test_dlt_rec_integration.py \
    tests/test_rec_dlt_header_independence.py tests/test_sam3dinov3.py \
    tests/test_sam3dinov3_visualize.py -v
  uv run ruff check vaila/dlt2d.py vaila/dlt3d.py vaila/rec3d.py \
    vaila/rec3d_one_dlt3d.py
  uv run ty check vaila/dlt2d.py vaila/dlt3d.py vaila/rec3d.py \
    vaila/rec3d_one_dlt3d.py
  ```
- **Evidence:** full pytest/ruff/ty stdout+stderr, appended to the parent
  state file after each sub-loop finishes.
- **Completion criterion:** all listed pytest files pass (exit 0), ruff and
  ty report zero errors on the four changed modules, AND both sub-loop state
  files report `status: success`.
- **Verifier protection:** this parent never edits the pre-existing test
  files listed above (only sub-loops may add *new* test files); the parent
  cannot mark itself successful — it only aggregates the two sub-loop
  verdicts plus the frozen regression command above.
- **Scientific validity:** the fixture is a single 12-point calibration frame
  per camera (no repeated frames), so any accuracy claim from sub-loop A is a
  calibration-residual claim, not a generalization claim, unless sub-loop A's
  own leave-one-out design (see its file) is used — the parent forwards that
  caveat verbatim into any summary it produces.

## Trigger
Manual. You ask the agent to "run the next iteration of dlt-camera-model-loop"
(or run a named sub-loop directly). No cron/event trigger; duplicate-run
protection is the state file's `status` field — do not start a sub-loop whose
state file already says `success` or `exhausted` without first archiving or
resetting it.

## Iteration
0. On the first iteration, validate `fixture_dir` exists and its five file
   groups are present and unmodified (compare a stored checksum in the state
   file on subsequent runs); confirm budget and the four approval gates below
   are understood.
1. Load this file plus both sub-loop files and the parent state file.
2. If sub-loop A's state file is not `success`, hand control to
   `dlt-distortion-loop.md` and stop this iteration once it returns a
   terminal state (`success`, `blocked`, `no-progress`, or `exhausted`).
3. If sub-loop A returned anything but `success`, the parent's status becomes
   that same terminal state and the loop stops (no point starting B on top of
   an unresolved A, since B's mesh work is independent of A's math but the
   parent's "done" claim would otherwise be misleading).
4. If sub-loop A is `success` and sub-loop B's state file is not `success`,
   hand control to `rec3d-mesh-blender-loop.md` and stop this iteration once
   it returns a terminal state.
5. Once both sub-loops report `success`, run the combined regression Check
   above and record raw evidence in the parent state file.
6. Evaluate terminal states; otherwise begin the next iteration.

## Terminal States
- **success:** both sub-loop state files report `success` AND the combined
  regression Check (pytest + ruff + ty) exits clean on real, current file
  content (re-verified, not read from a stale sub-loop cache).
- **no-op:** both sub-loops already `success` and the combined regression
  check already passed on the current file content in a prior run with no
  intervening edits — nothing to do.
- **no-progress/stalled:** either sub-loop reports `no-progress`.
- **blocked:** either sub-loop reports `blocked`, or `fixture_dir` is
  missing/modified from its recorded checksum.
- **exhausted:** the parent has sequenced sub-loops 3 times total (its own
  budget, see Guardrails) without reaching `success`.

Errors, missing evidence, and budget exhaustion are never success.

## Guardrails
- **Maximum allocation:** 3 parent cycles. Parent cycles × sub-loop budgets
  (8 + 8 each) give a worst-case of 3 × (8 + 8) = 48 child iterations total.
- **Human approval required:** (1) any `git commit`/`git push`; (2) any
  change to the `.dlt2d`/`.dlt3d` file *format* that would break
  compatibility with `.dlt3d` files already produced by other users (e.g.
  adding distortion columns to the header) — propose the format change and
  wait for explicit sign-off before writing it into `dlt3d.py`'s output path;
  (3) running SAM3+DINOv3 (GPU/CUDA, can be slow/costly) on anything outside
  `fixture_dir`; (4) writing to anything under `fixture_dir` itself (it stays
  read-only — all loop outputs go to new timestamped subfolders).
- **Isolation and credentials:** runs in the existing dev checkout, normal
  `uv` venv; sub-loop B additionally requires a CUDA GPU (SAM 3D Body has no
  CPU/MPS path) — treat missing CUDA as `blocked`, not a reason to skip the
  mesh work silently.
- **Protected verifier:** neither sub-loop may edit
  `tests/test_dlt_rec.py`, `tests/test_dlt_rec_integration.py`, or
  `tests/test_rec_dlt_header_independence.py` — those are the pre-existing
  regression floor. New tests may only be *added*, never used to replace or
  weaken these.
- **Rollback:** since commits require approval (gate 1), all in-progress
  changes stay uncommitted; revert a single iteration with
  `git checkout -- <file>` scoped to only the files that iteration touched
  (never a bare `git checkout .`).

## State Memory
- **Path:** `loops/state/dlt-camera-model-state.json`.
- **Persist:** fixture checksum, current phase (`A` | `B` | `regression` |
  terminal), pointer to each sub-loop's own state file, combined regression
  evidence (raw pytest/ruff/ty output) from the most recent run, cumulative
  cost, and parent cycle count.
- **Recovery:** a fresh context reads `status`; if not terminal, re-checks
  the fixture checksum, re-reads both sub-loop state files to see which
  phase is incomplete, and resumes at that phase. A `status: "in_progress"`
  with a `updated_at` older than the current session's start is treated as
  an interrupted write — re-verify the last recorded evidence before trusting
  it.

## Skills
- No named vailá skill directly covers DLT-distortion math or rec3d mesh
  export today; both sub-loops rely on direct code editing plus the
  repository's standard `uv run pytest` / `ruff` / `ty` checks per
  `.claude/skills/preto-loop/references/vaila-biomechanics.md`.
- `$sam3-video` and `$yolo-fb-gui-cli` — relevant if sub-loop B needs to
  regenerate SAM3 tracking data on the fixture videos as an intermediate
  step.

## Sub-Loops
- `dlt-distortion-loop.md` — distortion-aware DLT2D/DLT3D, validated by
  reconstructing the fixture's 12 known reference points. Terminal states:
  `success | no-progress | blocked | exhausted`.
- `rec3d-mesh-blender-loop.md` — mesh-for-Blender output from
  `rec3d.py`/`rec3d_one_dlt3d.py` via SAM3+DINOv3. Terminal states:
  `success | no-progress | blocked | exhausted`.

Parent × child limits: 3 × (8 + 8) = 48 child iterations worst case.
Neither sub-loop calls the other or the parent — no circular references.

## Why It Works
The two mathematically/technically unrelated improvements (lens-distortion
modeling vs. mesh export) get independent worst-first iteration and
independent rollback instead of being conflated into one change per
iteration, which would make regressions unattributable. The parent's own
check never trusts a sub-loop's self-report alone — it re-runs the full
regression suite once both report success, catching any interaction between
A's and B's edits to the same files (`rec3d.py`, `rec3d_one_dlt3d.py`).
Read-only fixture data plus a commit-approval gate keeps the loop from
silently corrupting the one real dataset available for validation.

## How to Trigger
### Context-bound
Ask the agent: "Run the next iteration of dlt-camera-model-loop" — it reads
this file plus the two sub-loop files and the state files under
`loops/state/`, then proceeds per the Iteration section above.

### Fresh-context / Ralph
An external runner re-reads this file, both sub-loop files, and
`loops/state/dlt-camera-model-state.json` every turn; it stops only on one of
the four named terminal states above and never treats a missing/errored
check as success.

## Health Metrics
- **Cost per accepted change:** `total tokens spent across both sub-loops /
  (accepted distortion-model change [0 or 1] + accepted mesh-export change
  [0 or 1])`.
- Combined regression pass/fail per cycle; wall-clock GPU time consumed by
  sub-loop B's CUDA runs (cost driver worth tracking separately since it
  dominates spend).
