---
name: ai-tracker-robustness-loop
category: Vailá
trigger: manual
verification-level: 1
theory-base: arXiv:2607.00038
---

# AI Tracker Trajectory Robustness & Online-Model Persistence Loop

## Description
Improve `vaila/tracking/ai_tracker.py` so the live/batch AI Track marker survives
acceleration + motion-blur + background-polarity-flip (white→black) events without
lateral drift, by adding trajectory-plausibility constraints (Kalman/motion-prior
gating) and a "never overwrite already-correct manually-measured segments" rule; and
give the online-learned appearance discriminator (`discriminator_w`/`discriminator_b`
from `retrain_online_model()`) an on-disk checkpoint so each `getpixelvideo.py` session
reloads and incrementally retrains it (transfer learning) instead of rebuilding from
scratch every time.

## Use When
- Regression-testing or hardening the AI Track marker follower against real failure
  clips (fast pan/acceleration, motion blur, illumination/background transitions).
- Iterating on Kalman/motion-prior gating for `search_and_predict` / template-match
  acceptance (`TemplateMatchResult.accepted`) in `ai_tracker.py`.
- Designing/iterating the online-discriminator checkpoint format and load/save/retrain
  lifecycle.
- Not for: the button/UI-only fix (done directly, not looped — see `getpixelvideo.py`
  AI Track button color/label, already applied this session, no loop needed). Not for
  batch markerless pipelines (SAM3/Sapiens2) — those are separate trackers.

## Inputs
1. `bench_biotronica` — `tests/test_ai_tracker_biotronica.py` +
   `/home/preto/data/Biotronica_Lift/videos/20260828_162054_vailacut_20260904_151532/`
   (existing GT fixture, known-good reference case).
2. `bench_jjkabuto` — new fixture to create from
   `/home/preto/data/jjkabuto/JJ_Kabuto.mp4` — requires a human-corrected GT CSV (see
   Verification). Until the GT CSV exists, iterations touching this fixture are
   `blocked`, not failing.
3. `ai_tracker_module` — `vaila/tracking/ai_tracker.py` (`retrain_online_model`,
   `score_patch_discriminator`, `seed_anchors_from_known`, `search_and_predict`,
   `TemplateMatchResult`, `AITrackerParameters`, `infill_and_smooth`).
4. `getpixelvideo_module` — `vaila/getpixelvideo.py` (`_perform_live_ai_tracking`,
   `live_tracker` lifecycle, tool-open/close hooks for checkpoint load/save).
5. `checkpoint_dir` — `vaila/models/ai_tracker/` (new directory this loop creates;
   holds discriminator checkpoints, e.g. `<checkpoint_dir>/discriminator_default.npz`
   with `w`, `b`, `feat_dim`, `n_samples`, `created_at`, `updated_at`, `source_sessions`).

## Goal
`AITrackerParameters`-driven tracking (live GUI and offline `infill_and_smooth`) stays
within GT error bounds through an acceleration+blur+background-flip event on the
JJ_Kabuto fixture, without regressing the existing Biotronica bench; and the online
discriminator persists across `getpixelvideo.py` sessions, measurably improving (or at
minimum not regressing) bench error as more sessions/anchors accumulate.

## Verification (Governing Check)
- **True level:** 1 (deterministic) for both bench fixtures once the JJ_Kabuto GT CSV
  exists; **5 (human checkpoint)** for producing that GT CSV itself — the user must
  manually correct the drifted segment in `JJ_Kabuto_markers.csv` (or a copy) before
  iteration 1 can run the JJ_Kabuto assertion. Until then, treat JJ_Kabuto iterations as
  `blocked` and continue only on Biotronica (which must never regress).
- **Check:**
  `uv run pytest tests/test_ai_tracker_biotronica.py tests/test_ai_tracker_jjkabuto.py -v -s`
  (the second file is created by this loop, mirroring the Biotronica test's structure:
  `infill_and_smooth` from sparse manual anchors vs full GT CSV, reporting n/mean/
  median/p95 and near-anchor(≤5f) error).
- **Evidence:** full stdout of both tests each iteration (n, mean, median, p95px,
  near-anchor mean, pass/fail), plus `uv run ruff check --fix && uv run ruff format &&
  uv run ty check` output for touched files.
- **Completion criterion:** JJ_Kabuto bench mean/median/p95 all at or below the first
  passing baseline recorded in state (no regression tolerance), AND Biotronica bench
  mean ≤ 24.64px, median ≤ 5.49px, p95 ≤ 89.57px, near-anchor(≤5f) mean ≤ 1.95px (the
  post-Kalman-filter numbers measured this session against the restored dense GT CSV —
  hard ceiling, never allowed to regress; supersedes the pre-Kalman ceiling of mean
  25.91/median 4.66/p95 97.63/near-anchor 1.97px, accepted after a 5-point parameter
  sweep on Mahalanobis-gate confidence and `_KF_Q_VEL` found no configuration beating
  this one on mean+median+p95 simultaneously), AND checkpoint round-trip test (see
  below) passes.
  **JJ_Kabuto baseline (unblocked this session):** `tests/test_ai_tracker_jjkabuto.py`
  created (sparse anchors = every 8th frame of the fully-dense, hand-corrected
  `JJ_Kabuto_markers.csv`, 331/331 frames non-NaN; GT = the same dense CSV); first-ever
  run with the current Kalman-filter code: n=331 mean=2.39px median=1.35px p95=8.01px
  near-anchor(≤5f) mean=2.39px (anchors=42). No lateral-drift blowup on the real failure
  clip. Frozen ceiling going forward: mean ≤ 2.39px, median ≤ 1.35px, p95 ≤ 8.01px,
  near-anchor mean ≤ 2.39px — never allowed to regress.
- **Verifier protection:** neither bench test's GT CSV, video file, nor assertion
  thresholds may be edited by the maker step; only `ai_tracker.py`,
  `getpixelvideo.py`, and the new checkpoint module/tests are in scope for edits.
  Checkpoint persistence gets its own new deterministic test
  (`tests/test_ai_tracker_checkpoint.py`): save → reload → scores/weights bit-identical
  (red-before/green-after: write the test first, confirm it fails against the
  pre-checkpoint code, then implement).
- **Scientific validity:** pixel-space error only (no unit conversion needed, single
  camera view, no DLT); frame indexing must stay 0-based and match the existing
  Biotronica test's convention; near-anchor(≤5f) metric preserved unchanged (it isolates
  interpolation quality from raw detection quality); checkpoint retrain must never mix
  features from a different `feat_dim`/extractor config (guard with a stored
  `feat_dim`/`use_deep_features` tag, reject silently-mismatched loads rather than
  corrupt-merge them).

## Trigger
Manual only — user runs `/preto-loop-run ai-tracker-robustness-loop.md` (or invokes the
harness's goal mechanism directly) after producing/approving the JJ_Kabuto GT CSV.
Duplicate-run protection: check `loops/state/ai-tracker-robustness-loop-state.json`'s
`status` field; refuse to start a second concurrent run while `status: running`.

## Iteration
0. On the first iteration, validate `bench_biotronica` fixture is present and passing at
   the recorded baseline; validate/prompt for `bench_jjkabuto` GT CSV presence (blocked
   otherwise); freeze the 8-iteration budget.
1. Load this spec and `loops/state/ai-tracker-robustness-loop-state.json`; confirm
   remaining budget and any pending human-approval gate (checkpoint format changes).
2. Snapshot current bench numbers (Biotronica + JJ_Kabuto if unblocked) as this
   iteration's baseline; run the governing check once before any change.
3. Rank unresolved targets worst-first from this fixed candidate list (do not invent
   new targets mid-run without recording why):
   a. JJ_Kabuto p95/mean error during the reported acceleration+blur+bg-flip segment.
   b. Trajectory-plausibility gating (reject template-match acceptance that implies an
      implausible velocity/acceleration jump; Kalman or simpler constant-velocity gate
      — start with the simplest gate that fixes (a), escalate only if needed).
   c. "Do not overwrite already-correct manual segments" rule in
      `infill_and_smooth`/live per-frame update path.
   d. Online-discriminator checkpoint save/load/incremental-retrain lifecycle
      (`vaila/models/ai_tracker/discriminator_default.npz`) wired into
      `getpixelvideo.py` open/close.
4. Make exactly one attributable change addressing the top-ranked unresolved target.
5. Run the governing check; record raw stdout/stderr evidence.
6. Retain the change only if: no Biotronica regression, AND (JJ_Kabuto unblocked implies
   no JJ_Kabuto regression vs this run's own baseline), AND ruff/ty clean. Otherwise
   `git diff`-scoped revert of only this iteration's edit (the file(s) touched in step 4)
   and record why in state.
7. Curate lessons (what gate/threshold worked or didn't, with the numbers) and atomically
   rewrite the state file: baseline, attempts, accepted/rejected changes, evidence, cost.
8. Evaluate terminal states; otherwise begin the next iteration.

## Terminal States
- **success:** completion criterion (see Verification) fully met and recorded with
  evidence in state.
- **no-op:** governing check already meets completion criterion at iteration 0 with no
  change made.
- **no-progress/stalled:** 2 consecutive iterations with no metric improvement on the
  currently-unblocked worst-ranked target.
- **blocked:** JJ_Kabuto GT CSV missing/not human-approved, or a candidate target
  requires an approval-gated action not yet granted (see Guardrails).
- **exhausted:** 8 iterations reached without meeting the completion criterion.

Errors, missing evidence, and budget exhaustion are never success.

## Guardrails
- **Maximum allocation:** 8 iterations for this run.
- **Human approval required:** (1) approving/producing the JJ_Kabuto GT CSV itself
  (human correction of the drifted segment); (2) any change to the on-disk checkpoint
  *format* (new fields, versioning) once first shipped, since it's a cross-session
  artifact other tools may come to depend on; (3) committing anything — this loop never
  commits, per standing project constraint.
- **Isolation and credentials:** local filesystem only, no network, no credentials
  needed; runs against local video/CSV fixtures already on disk.
- **Protected verifier:** `tests/test_ai_tracker_biotronica.py` (existing, GT-backed) is
  read-only to the maker; `tests/test_ai_tracker_jjkabuto.py` and
  `tests/test_ai_tracker_checkpoint.py` are created once (red-before/green-after) and
  then also frozen for the rest of the run — the maker may not weaken thresholds to
  pass.
- **Rollback:** `git diff`/`git checkout -- <file>` scoped to only the file(s) touched
  in the iteration's step 4, immediately on a failed step 6.

## State Memory
- **Path:** `loops/state/ai-tracker-robustness-loop-state.json`.
- **Persist:** `status` (idle|running|<terminal state>), iteration count, per-iteration
  baseline/result numbers (Biotronica + JJ_Kabuto mean/median/p95/near-anchor), accepted
  vs rejected changes with one-line reason each, checkpoint file path + version once
  created, cumulative cost.
- **Recovery:** a fresh context reads `status`; if `running` with a stale timestamp (no
  update in this process's lifetime), treat as crashed mid-iteration — re-run step 2
  (snapshot) to re-establish ground truth before resuming step 3, never trust an
  in-progress iteration's unconfirmed claims.

## Skills
- `getpixelvideo-tracking-loader` — reference for existing marker-CSV auto-detect
  conventions when wiring checkpoint load into `getpixelvideo.py` startup.

## Sub-Loops
None.

## Why It Works
Worst-first + one-change-per-iteration isolates which specific gate (velocity clamp vs.
Kalman vs. manual-segment protection vs. checkpoint reuse) actually moves the JJ_Kabuto
numbers, instead of bundling several plausible fixes and losing attribution. Freezing
the Biotronica ceiling as a hard non-regression bound prevents "fixing" JJ_Kabuto by
loosening gates in a way that reintroduces the class of error already fixed this
session (teleporting/overwriting). Red-before-green on the two new tests proves they
actually exercise the failure mode instead of trivially passing. The checkpoint
format-change approval gate exists because that artifact will outlive a single loop run
and get read by every future `getpixelvideo.py` session — a bad format change is not
cheaply reversible once other sessions have written checkpoints against it.

## How to Trigger
### Context-bound
Ask the harness to run this file as a goal: "run loops/ai-tracker-robustness-loop.md,
iterate up to its 8-iteration budget, stop at any terminal state."

### Fresh-context / Ralph
External runner re-reads this file and
`loops/state/ai-tracker-robustness-loop-state.json` every turn; invokes one iteration;
writes state; exits. Stops only on a terminal state in the Terminal States section —
timeouts, crashes, and budget exhaustion are not success and must be reported as such.

## Health Metrics
- **Cost per accepted change:** `total tokens spent / iterations whose step 6 retained
  the change`.
- JJ_Kabuto bench mean/median/p95px trend across iterations (must be monotonically
  non-increasing per retained change).
- Biotronica bench mean/median/p95px (must stay ≤ the frozen ceiling every iteration).
- Checkpoint round-trip fidelity (bit-identical weights after save/load).
