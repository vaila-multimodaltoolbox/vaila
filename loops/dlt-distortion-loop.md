---
name: dlt-distortion-loop
category: Biomechanics
trigger: manual
verification-level: 1
theory-base: arXiv:2607.00038
---

# DLT2D/DLT3D Lens-Distortion Model

## Description
Add camera/lens-distortion terms to `vaila/dlt2d.py` and `vaila/dlt3d.py`
(currently a pure 8-/11-parameter linear DLT with no distortion modeling) and
prove, on real calibration data, that the distortion-aware model reduces
reconstruction error of known reference points versus the current linear
baseline — without weakening the model into a curve-fit that only works
in-sample.

## Use When
- Implementing the radial/tangential distortion extension described in Rossi
  et al. 2013 (`rec3d_todo/rossi2013.pdf`, DOI
  10.1080/10255842.2013.866231), the Kwon3D DLT/camera-model write-ups
  (`kwon3d.com/theory/dlt/dlt.html`,
  `biomech.web.unc.edu/dlt-to-from-intrinsic-extrinsic/`), and any additional
  distortion-DLT literature the user names (e.g. the ScienceDirect and IEEE
  papers referenced alongside Rossi 2013).
- Not for markerless/mesh work — that is `rec3d-mesh-blender-loop.md`.
- Not for writing the didactic explainer doc — do that once this loop reaches
  `success`, referencing the final, tested parameter set.

## Inputs
1. `fixture_dir` — `/home/preto/data/sep_runcod_01072026/REC3D_COD/rec3d_todo/`
   (read-only; see parent loop). Specifically:
   `c1c2c3_cod.ref3d` (12 ground-truth 3D points, 1 row),
   `c{1,2,3}_cod_markers_1_line.csv` (pixel coords of the same 12 points),
   `c{1,2,3}_cod_markers_1_line.dlt3d` (existing linear-DLT baseline).
2. `distortion_model_candidates` — ordered worst-first list to try, default:
   `["none (baseline)", "radial-1param (k1)", "radial-2param (k1,k2)",
   "radial-tangential (k1,k2,p1,p2)"]`, matching the common DLT-distortion
   extensions in the cited literature. Stop trying further candidates once
   one both improves on baseline (per the Check) and is confirmed via the
   human approval gate for format changes.

## Goal
`dlt3d.py` (and `dlt2d.py` for the 2D case) can optionally fit and apply
lens-distortion coefficients alongside the existing linear DLT coefficients,
and the resulting 3D reconstruction of the fixture's 12 known reference
points has strictly lower leave-one-out cross-validated error than the
current linear-only model, with no individual point's error more than 2x
worse than its linear-only error. Objectively verifiable via the Check below.

## Verification (Governing Check)
- **True level:** 1 (deterministic numeric comparison).
- **Check:** `uv run pytest tests/test_dlt_distortion_reference_recovery.py -v`
  — a new test file this loop creates, which:
  1. Loads `c1c2c3_cod.ref3d` (12 points) and the three
     `c{1,2,3}_cod_markers_1_line.csv` pixel files.
  2. For **leave-one-out cross-validation** (fit on 11 points, predict the
     12th, repeat for all 12): computes DLT3D parameters with (a) the
     existing linear model and (b) each candidate distortion model, using
     only the 11 in-fold points, then reconstructs the held-out 12th point
     in 3D from all 3 cameras and records its error (mm) against
     `c1c2c3_cod.ref3d`.
  3. Reports per-model LOOCV RMSE (mm) over all 12 held-out points, plus the
     max single-point error, plus (secondary evidence) mean 2D reprojection
     error in pixels for context against Rossi 2013's reported numbers.
  4. Asserts the best distortion model's LOOCV RMSE is strictly less than
     the linear baseline's LOOCV RMSE, and no single held-out point regresses
     by more than 2x its baseline error.
  Also run the pre-existing regression floor:
  `uv run pytest tests/test_dlt_rec.py tests/test_dlt_rec_integration.py tests/test_rec_dlt_header_independence.py -v`
  and `uv run ruff check vaila/dlt2d.py vaila/dlt3d.py --fix && uv run ty check vaila/dlt2d.py vaila/dlt3d.py`.
- **Evidence:** full pytest stdout (per-model LOOCV RMSE table + 2D
  reprojection numbers), ruff/ty output, appended verbatim to the state file
  each iteration.
- **Completion criterion:** the new LOOCV test passes (best distortion model
  beats baseline per the assertion above) AND the three pre-existing test
  files still pass AND ruff/ty are clean on the two changed files AND
  `dlt2d.py`/`dlt3d.py` header Version/Last Updated fields, `vaila/help/dlt2d.md`
  + `.html`, `vaila/help/dlt3d.md` + `.html`, and `vaila/help/index.md`/`.html`
  are updated per `CLAUDE.md`'s mandatory-metadata checklist — treat a
  passing test suite with stale metadata as incomplete, not as `success`.
- **Verifier protection:** LOOCV means every reported number is on a point
  the model never saw while fitting — with only 12 total calibration points
  this is the only credible holdout available, and it directly prevents a
  distortion model with more free parameters from winning purely by
  in-sample overfitting. The three pre-existing DLT/rec test files are
  frozen — this loop may add `test_dlt_distortion_reference_recovery.py` but
  must never edit or delete the existing ones to make them pass.
- **Scientific validity:** units are meters in `.ref3d`/reconstruction output
  and pixels in the marker CSVs (matches existing vailá convention); the
  fixture is a single static frame per camera (no repeated calibration
  trials), so report LOOCV RMSE with its 12-sample size explicitly — do not
  claim it generalizes to a different calibration volume, camera, or lens
  setting without new data. Record whether a candidate model is
  under-determined (more free parameters than the ~7 effective LOOCV
  degrees of freedom would responsibly support) and treat that as a `blocked`
  candidate, not a data point to report.

## Trigger
Manual, invoked directly or via `dlt-camera-model-loop.md`. Duplicate-run
guard: do not start if this loop's own state file already says `success` or
`exhausted`.

## Iteration
0. On the first iteration, verify the fixture files exist and record their
   checksums; confirm the candidate list and budget.
1. Load this file and the state file; if the state file lists a candidate
   already tried with recorded evidence, skip it.
2. Snapshot current `dlt2d.py`/`dlt3d.py` (git diff is empty at start of a
   clean iteration) and run the Check to get the current baseline evidence.
3. Pick the next untried candidate from `distortion_model_candidates`
   (worst-first = simplest-first, since each adds exactly one more
   distortion term over the last — this *is* the worst-first ordering here,
   from "no fix" to "most parameters").
4. Implement exactly that one candidate's fitting + application code in
   `dlt3d.py` (and mirror the 2D-only radial terms into `dlt2d.py` if the
   candidate is expressible in 2D, per Kwon3D's DLT distortion formulation).
5. Run the Check and record raw evidence.
6. If the candidate passes the completion criterion, retain it and stop
   (mark `success`) — do not keep trying further candidates once one wins,
   per the Inputs section's stopping rule. If it fails or regresses relative
   to the previous best, revert only this iteration's code changes
   (`git checkout -- vaila/dlt2d.py vaila/dlt3d.py`) and record it as a
   rejected candidate with its evidence (a real result worth keeping in
   state, not just a discard).
7. Curate lessons (e.g. "radial-tangential is under-determined at 12
   points — do not retry without more calibration points") and persist
   state atomically.
8. Evaluate terminal states; otherwise begin the next iteration with the
   next candidate.

## Terminal States
- **success:** a candidate distortion model passes the Check's completion
  criterion and the pre-existing regression floor stays green.
- **no-op:** the Check already passes with a previously accepted model and
  no candidates remain untried — nothing left to do.
- **no-progress/stalled:** two consecutive candidates both fail to beat the
  current best LOOCV RMSE (including the linear baseline as "current best"
  before any candidate has won).
- **blocked:** the fixture is missing/modified from its recorded checksum;
  or every remaining candidate is flagged under-determined per the
  Scientific validity note; or the human approval gate for a `.dlt3d` format
  change (adding distortion columns) is not yet granted and no
  format-compatible alternative (e.g. a sidecar `.dlt3d.distortion.json`)
  has been agreed.
- **exhausted:** 8 iterations reached without a passing candidate.

Errors, missing evidence, and budget exhaustion are never success.

## Guardrails
- **Maximum allocation:** 8 iterations, one candidate model per iteration.
- **Human approval required:** any change to the on-disk `.dlt3d`/`.dlt2d`
  file format (new columns/header for distortion coefficients) needs
  explicit sign-off before being written by `dlt3d.py`'s save path, since
  existing `.dlt3d` files from other users/projects must remain readable;
  propose a sidecar-file or backward-compatible-append design first if
  approval is pending.
- **Isolation and credentials:** pure CPU numerical fitting (least squares /
  `scipy.optimize`), no GPU, no network access needed.
- **Protected verifier:** `tests/test_dlt_rec.py`,
  `tests/test_dlt_rec_integration.py`, `tests/test_rec_dlt_header_independence.py`
  are never edited by this loop.
- **Rollback:** `git checkout -- vaila/dlt2d.py vaila/dlt3d.py` after a
  rejected candidate, scoped to only those two files.

## State Memory
- **Path:** `loops/state/dlt-distortion-state.json`.
- **Persist:** fixture checksum, candidate list with per-candidate LOOCV
  RMSE/max-error/2D-reprojection evidence (accepted or rejected), current
  best model, iteration count, format-change approval status, cost.
- **Recovery:** a fresh context reads which candidates already have evidence
  recorded and resumes at the next untried one; an `in_progress` entry with
  no evidence attached is treated as an interrupted write and is redone.

## Skills
- No dedicated vailá skill for DLT math; follows
  `.claude/skills/preto-loop/references/vaila-biomechanics.md` §"Data and AI
  checks" (DLT calibration geometry, degeneracy, reprojection error).

## Sub-Loops
None.

## Why It Works
LOOCV on the only 12 real calibration points available is the strongest
honest check this fixture supports — it directly blocks the failure mode
where a model with more free parameters "wins" only because it was scored on
the same points it was fit to. One-candidate-per-iteration with immediate
revert-on-regression keeps every accepted change attributable and keeps the
existing linear-DLT users' file format safe behind an explicit approval gate
instead of a silent breaking change.

## How to Trigger
### Context-bound
Ask the agent: "Run the next iteration of dlt-distortion-loop" (or let
`dlt-camera-model-loop.md` invoke it).

### Fresh-context / Ralph
An external runner re-reads this file and
`loops/state/dlt-distortion-state.json` every turn, resumes at the next
untried candidate, and stops only on a named terminal state.

## Health Metrics
- **Cost per accepted change:** `total tokens spent / 1 if a candidate was
  accepted else 0` (this loop accepts at most one model).
- LOOCV RMSE improvement (mm) of the accepted model over the linear
  baseline; number of candidates flagged under-determined vs. actually
  tried.
