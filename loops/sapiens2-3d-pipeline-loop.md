---
name: sapiens2-3d-pipeline-loop
category: Markerless/video/AI
trigger: manual
verification-level: 3
theory-base: arXiv:2607.00038
---

# Sapiens2-Guided 3D Pose (SAM3+Sapiens2 → SAM 3D Body)

## Description
Add a new monocular 3D pipeline, `vaila/sapiens2_3d.py`, that complements
`vaila/sam3dinov3.py` by using Meta Sapiens2's 308-keypoint 2D pose (already
integrated as the 2D-only `vaila/vaila_sapiens.py` / combined
`vaila/sam3sapiens2.py`) as extra guidance into the **same** SAM 3D Body
(DINOv3 backbone) MHR mesh regressor `sam3dinov3.py` already uses — not a
second, independent 3D lifter. This is a scoping correction versus the
original request: Sapiens2 as integrated in this repo has **no depth, normal,
or mesh head** — it is 2D keypoints (x, y, score) only — so it cannot produce
3D on its own. The chosen design keeps the one proven 3D lifter and swaps its
front-end guidance.

## Use When
- Building/iterating this exact feature: SAM3 (identity/mask authority) +
  Sapiens2 (2D keypoint guidance) → SAM 3D Body (mesh) → vailá-standard
  MHR70 CSV/camera/mesh outputs, wired into the Markerless 3D GUI chooser and
  CLI.
- Not for a genuinely independent Sapiens-depth-head 3D pipeline (rejected
  during scoping — see Why It Works) — if that direction is wanted later, it
  is a new loop, not a resumption of this one.
- Not for `dlt-camera-model-loop.md`/`dlt-distortion-loop.md`'s DLT math, nor
  for `rec3d-mesh-blender-loop.md`'s multi-camera triangulation/export work —
  this loop only produces one more monocular per-camera source directory in
  the same shape `sam3dinov3.py` already produces; wiring it into rec3d as a
  mesh source is an optional follow-on, not part of this loop's goal.

## Inputs
1. `fixture_dir` — `/home/preto/data/sep_runcod_01072026/REC3D_COD/rec3d_todo/`
   (read-only; user-confirmed for testing). Contains `c1_cod.mp4`,
   `c2_cod.mp4`, `c3_cod.mp4` (1920x1080) and their `.dlt3d`/`.ref3d`/marker
   CSVs, but **no pre-generated `processed_sam3sapiens2_*` or
   `processed_sam3dinov3_*` directories exist here yet** (verified at
   authoring time) — unlike the fixture state `rec3d-mesh-blender-loop.md`
   assumes. This loop's GPU stage must therefore run the SAM3+Sapiens2 front
   end itself (see Guardrails on GPU-run approval) rather than reuse existing
   output.
2. `smoke_clip` — a short trimmed window (recommend first ~150–300 frames of
   `c1_cod.mp4`, ~1.25–2.5 s at 119.88 fps) for the first real-data pass, to
   keep GPU-time and approval scope small before running full videos.
3. Hardware — NVIDIA CUDA GPU (both SAM 3D Body's estimator and Sapiens2's
   pose model hardcode `cuda`; no CPU/MPS path, matching `sam3dinov3.py` and
   `vaila_sapiens.py`).

## Goal
`vaila/sapiens2_3d.py` exists with a `run_*()`/CLI entry point that: (1)
reuses `sam3sapiens2.py`'s SAM3+Sapiens2 front end (imports, does not
duplicate, its bbox/contour/ID/keypoint machinery — same reuse pattern
`sam3dinov3.py`'s own header documents against the plain SAM3 front end); (2)
derives a keypoint-tightened bbox (and, if the upstream API allows it — to be
confirmed in iteration 1, see Iteration step 3 — a pose hint) per person per
frame from the Sapiens2 308 keypoints; (3) feeds that guidance into
`SAM3DBodyEstimator.process_one_image(bboxes=..., masks=...)`, the exact
function `sam3dinov3.py` already calls; (4) writes the same MHR70 long/wide
CSV + camera CSV + optional mesh family `sam3dinov3.py` writes, plus one
extra column/flag recording that a frame's guidance came from Sapiens2
keypoints (so downstream consumers and the evaluation in this loop can tell
guided frames from unguided ones). Wired into the GUI's **Markerless 3D**
chooser (`vaila.py`'s `markerless_3d_analysis()`), not a new standalone
top-level button — the repo's own v0.3.97/98 history explicitly moved away
from flat per-tool buttons into choosers; a literal new top-level "Sapiens2 3D
Pose" button would fight that established layout. Success is only claimed
with real-video evidence that the guidance changes something measurable
(tighter bbox / lower reprojection error) or an honest, evidenced no-op
finding that it doesn't — never asserted from code review alone.

## Verification (Governing Check)
- **True level:** mixed, reported per tier, never blended into one pass/fail:
  level 2 (lint/type), level 1 (synthetic unit tests), level 3 (real GPU run
  on real video — delayed field truth), level 5 (final human sign-off before
  any commit).
- **Check, tier 1 (level 2, every iteration):**
  `uv run ruff check vaila/sapiens2_3d.py --fix && uv run ruff format vaila/sapiens2_3d.py && uv run ty check vaila/sapiens2_3d.py`.
- **Check, tier 2 (level 1, every iteration, CPU-only, no GPU/network):**
  `uv run pytest tests/test_sapiens2_3d.py -v` — a new file this loop
  creates, mirroring `tests/test_sam3dinov3.py`'s pattern (synthetic
  keypoints/bboxes, a stub/mock `SAM3DBodyEstimator` so the estimator itself
  is never loaded): keypoint→bbox tightening math against known synthetic
  points, the guidance-flag column round-trips through the CSV writer,
  NaN/occlusion handling for missing Sapiens2 keypoints falls back to the
  SAM-mask-only bbox instead of crashing or silently guiding from garbage.
  Also run the frozen regression floor:
  `uv run pytest tests/test_sam3dinov3.py tests/test_sam3sapiens2.py tests/test_vaila_sapiens.py -v`.
- **Check, tier 3 (level 3, gated by human approval — see Guardrails):** real
  GPU run of `vaila/sapiens2_3d.py` on `smoke_clip`, then full `c1_cod.mp4`
  once tier 3 first passes on the clip. Evidence to record: (a) segment
  lengths (thigh, shank, shoulder width) compared against the
  independently-obtained, already-frozen bounds in project memory — thigh
  0.387±0.017 m, shank 0.371±0.012 m, shoulder width 0.360 m — as a
  regression floor, not a new claim, since the underlying mesh regressor is
  unchanged; (b) mean 2D reprojection error of the guided run versus a
  plain `sam3dinov3.py` run on the identical clip (same frames, same
  person) — the whole point of adding Sapiens2 guidance is that this number
  should not get worse, and ideally improves on frames where SAM3's mask-only
  bbox is loose (fast motion, partial occlusion); (c) fraction of frames
  where guidance was actually available vs. fell back to mask-only.
- **Check (human checkpoint, level 5):** before any `git commit`, a human
  reviews the tier-3 evidence and either (i) confirms guided reprojection
  error is not worse and, on some measurable subset of frames, better —
  proceed to `success` — or (ii) confirms no measurable difference — proceed
  to the `no-op` terminal state instead of merging a placebo feature.
- **Evidence:** raw ruff/ty/pytest stdout every iteration; the tier-3
  segment-length and reprojection-error numbers (guided vs. baseline) logged
  even before any threshold is frozen; the level-5 note with its decision.
- **Completion criterion:** tiers 1–2 pass AND the regression floor stays
  green AND tier 3 has been run at least once on the full `c1_cod.mp4` with
  recorded evidence AND the level-5 human checkpoint recorded a `success`
  decision AND `vaila/sapiens2_3d.py` header Version/Update Date, `vaila.py`
  banner (if touched), `vaila/help/sapiens2_3d.md`+`.html`, and
  `vaila/help/index.md`+`.html` are updated per `CLAUDE.md`'s mandatory
  metadata checklist.
- **Verifier protection:** tiers 1–2 are the maker's own checks and cannot
  substitute for the tier-3/level-5 real-video evidence — this loop must
  never report `success` from synthetic tests alone. The frozen
  segment-length bounds are copied from a prior, independently-obtained
  measurement (not recomputed by this loop). `tests/test_sam3dinov3.py`,
  `tests/test_sam3sapiens2.py`, `tests/test_vaila_sapiens.py` are frozen
  (read, never edited, by this loop).
- **Scientific validity:** units meters, camera frame (+y down, OpenCV
  convention, matching `sam3dinov3.py`'s documented convention), root-relative
  keypoints + `pred_cam_t` for camera-frame position — identical to
  `sam3dinov3.py`, since the mesh regressor is unchanged. Sapiens2's 308
  keypoints are pixel-space (x, y, score); confirm they share the same pixel
  origin/orientation as SAM3's bbox/mask pixel space before combining (both
  are read from the same decoded video frame, so they should already agree —
  verify, do not assume). `person_id` must stay `== sam_obj_id` exactly as in
  `sam3dinov3.py`, since Sapiens2 guidance is keyed off SAM3's existing
  per-person bbox, not a second identity assignment. Record which of the
  22–70 relevant Sapiens2 keypoints (COCO-body subset, per
  `vaila_sapiens.py`'s own `_sapiens_log`/COCO-index comment) are used for
  bbox tightening, and why.

## Trigger
Manual, this session. Duplicate-run guard: do not start if this loop's state
file already says `success`, `no-op`, or `exhausted`.

## Iteration
0. On the first iteration, verify `fixture_dir` exists with the three `.mp4`
   files; verify CUDA is visible (`nvidia-smi` or `torch.cuda.is_available()`);
   verify `sam3dinov3.py`, `sam3sapiens2.py`, `vaila_sapiens.py` are importable
   (dependencies already installed per `CLAUDE.md`'s `--extra sam`/`--extra
   sapiens`/`bin/setup_fifa_sam3d.sh` setup). Record findings.
1. Load this file and the state file.
2. Snapshot current repo state; run tiers 1–2 to get current evidence.
3. Rank unresolved targets worst-first: (a) **investigate**
   `SAM3DBodyEstimator.process_one_image`'s real signature/upstream source to
   confirm whether it accepts any keypoint/pose hint beyond `bboxes`/`masks`
   — if not, the guidance mechanism is scoped down to bbox-tightening only,
   document this explicitly rather than implying a richer integration →
   (b) keypoint→bbox tightening function + missing-keypoint fallback →
   (c) module skeleton reusing `sam3sapiens2.py`'s front end (dual-import
   pattern, lazy imports) → (d) MHR70 CSV/camera/mesh writers reusing
   `sam3dinov3.py`'s existing writer functions (do not duplicate) plus the new
   guidance-flag column → (e) GUI wiring inside `markerless_3d_analysis()` +
   `_print_chooser_launch` CLI mirror → (f) synthetic unit tests → (g) tier-3
   real-GPU smoke clip run → (h) full-video tier-3 run + baseline comparison
   → (i) help docs + version metadata. Pick the single worst unresolved item.
4. Make exactly one attributable code change addressing that item.
5. Run tiers 1–2 and record raw evidence. For items (g)/(h), request human
   approval first (see Guardrails), then run tier 3 and record its evidence.
6. If this iteration's change is believed to complete the feature, perform
   the level-5 human checkpoint now and record its `success`/`no-op`
   decision; otherwise skip the checkpoint this iteration.
7. Retain the change only if tiers 1–2 pass without regressing the frozen
   floor; otherwise revert via
   `git checkout -- vaila/sapiens2_3d.py vaila.py tests/test_sapiens2_3d.py`
   (or `git clean -f` for a same-iteration untracked new file) scoped to this
   iteration.
8. Curate lessons (e.g. "upstream estimator has no pose-hint parameter —
   guidance is bbox-only" or "Sapiens2 keypoints agree with SAM mask bbox on
   >95% of frames — guidance rarely changes anything") and persist state
   atomically.
9. Evaluate terminal states; otherwise begin the next iteration.

## Terminal States
- **success:** tiers 1–2 pass, regression floor stays green, tier 3 has run
  on the full `c1_cod.mp4` with recorded evidence showing guided reprojection
  error is not worse than the `sam3dinov3.py`-only baseline (and better on a
  measurable subset of frames), GUI/CLI wired, and the level-5 checkpoint
  recorded a `success` decision, AND metadata/help docs updated.
- **no-op:** investigation in step 3(a) finds the upstream estimator accepts
  no pose hint AND the resulting bbox-only guidance produces no measurable
  reprojection-error difference from the existing `sam3dinov3.py` baseline on
  the smoke clip — documented finding with tier-3 evidence, no pipeline
  merged (avoids shipping a placebo feature), level-5 checkpoint records this
  explicitly.
- **no-progress/stalled:** two consecutive iterations with no previously
  failing tier-1/tier-2 assertion turning green.
- **blocked:** CUDA unavailable; SAM 3D Body / Sapiens2 weights not
  downloaded or license not accepted; `fixture_dir` videos missing/modified
  from their recorded state; or `SAM3DBodyEstimator.process_one_image` cannot
  be called safely with the intended guidance without upstream changes
  (escalate, do not patch vendored `sam_3d_body` code without explicit
  approval).
- **exhausted:** 10 iterations reached without reaching `success` or `no-op`.

Errors, missing evidence, and budget exhaustion are never success.

## Guardrails
- **Maximum allocation:** 10 iterations.
- **Human approval required:** any GPU run beyond the first short
  `smoke_clip` pass (i.e., approval before running the full `c1_cod.mp4`, and
  again before running `c2_cod.mp4`/`c3_cod.mp4` if needed); any new model
  weight download; any `git commit`/`push`; overwriting anything under
  `fixture_dir` (never — outputs go to new timestamped subfolders only,
  matching `sam3dinov3.py`/`sam3sapiens2.py` convention); the final
  `success`-vs-`no-op` judgment in step 6 (a product/scientific call, not
  purely mechanical).
- **Isolation and credentials:** GPU stages run in the existing vailá CUDA
  venv with already-provisioned `--extra sam`/`--extra sapiens` weights (per
  `bin/setup_fifa_sam3d.sh`/`bin/setup_sapiens2.sh`); no new network access
  beyond what those setup scripts already require; no credentials handled by
  this loop.
- **Protected verifier:** `tests/test_sam3dinov3.py`, `tests/test_sam3sapiens2.py`,
  `tests/test_vaila_sapiens.py` are never edited by this loop. The frozen
  segment-length bounds are copied from prior independent measurement, not
  recomputed here.
- **Rollback:** `git checkout -- vaila/sapiens2_3d.py vaila.py tests/test_sapiens2_3d.py`
  (or `git clean -f` if the file is new and untracked) scoped to this
  iteration's files only.

## State Memory
- **Path:** `loops/state/sapiens2-3d-pipeline-state.json`.
- **Persist:** CUDA/dependency check results, the step-3(a) API-surface
  finding (does `process_one_image` accept pose hints — yes/no + source
  evidence), ranked target list with per-item status, tier-1/2 evidence per
  iteration, tier-3 evidence (segment lengths, guided-vs-baseline
  reprojection error, guidance-availability fraction) once reached, the
  level-5 checkpoint decision (date + note), which Sapiens2 keypoints are
  used for bbox tightening, iteration count, cost.
- **Recovery:** a fresh context re-checks CUDA/dependency availability, reads
  which ranked targets are already resolved, and resumes at the next worst
  unresolved one; an `in_progress` entry with no evidence is treated as
  interrupted and redone.

## Skills
- `$yolo-fb-gui-cli` — GUI→CLI parity conventions for wiring the new tool
  into the `markerless_3d_analysis()` chooser with a `>>` CLI-mirror print.
- `.claude/skills/create-analysis-module.md` — module scaffold, dual-import
  pattern, `run_*()` entry point convention for the new file.
- `.claude/skills/preto-loop/references/vaila-biomechanics.md` — markerless
  keypoint-ordering, bbox/mask coordinate conversion, and ID-persistence
  checks to apply while implementing.

## Sub-Loops
None. Optional future follow-on (not wired, no circular reference): once
`success`, `rec3d-mesh-blender-loop.md`'s `--mesh-dir` input could be extended
to accept this pipeline's output as an alternative per-camera mesh source
alongside plain `sam3dinov3.py` — a separate loop invocation, not a call from
here.

## Why It Works
The interview surfaced a real scoping error before any code was written:
Sapiens2 as vendored here (`vaila_sapiens.py`) is 2D-keypoints-only, so
"Sapiens2 3D" cannot mean an independent 3D lifter without inventing a
capability the repo doesn't have. Anchoring the design on the *existing*,
already-validated 3D lifter (SAM 3D Body via `sam3dinov3.py`) and treating
Sapiens2 purely as extra 2D guidance keeps the loop's claims falsifiable:
either the guidance measurably tightens bboxes/lowers reprojection error on
real video, or it doesn't, and either outcome is a valid, evidenced terminal
state (`success` or `no-op`) rather than a feature shipped on faith. Gating
GPU time behind approval (short clip first, full video second) bounds cost
before the loop knows whether the idea works at all. Freezing the
segment-length floor from a prior independent measurement, and never letting
this loop edit `sam3dinov3.py`'s or `sam3sapiens2.py`'s own test files,
prevents it from grading its own homework on the one axis (mesh plausibility)
the underlying regressor was already validated against.

## How to Trigger
### Context-bound
Ask the agent: "Run the next iteration of sapiens2-3d-pipeline-loop."

### Fresh-context / Ralph
An external runner re-reads this file and
`loops/state/sapiens2-3d-pipeline-state.json` every turn, re-verifies
CUDA/fixture availability, resumes at the next unresolved ranked target, and
stops only on a named terminal state.

## Health Metrics
- **Cost per accepted change:** `total tokens spent / 1 if the guided
  pipeline reached success else 0`.
- Tier-1/2 assertions passing per iteration (progress signal pre-GPU);
  tier-3 guided-vs-baseline reprojection-error delta trend across iterations
  once reached; fraction of frames where Sapiens2 guidance was actually
  available (context for the level-5 review).
