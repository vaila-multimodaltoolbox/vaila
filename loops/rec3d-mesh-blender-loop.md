---
name: rec3d-mesh-blender-loop
category: Markerless/video/AI
trigger: manual
verification-level: 2
theory-base: arXiv:2607.00038
---

# rec3d Mesh-for-Blender Export

## Description
Extend `vaila/rec3d_one_dlt3d.py` (primary target — see Use When) and
`vaila/rec3d.py` (secondary, only if a per-frame-varying-DLT dataset shows up
later) so that, in addition to the skeleton keypoints they already
reconstruct via multi-camera DLT triangulation, a run can also carry through
a per-frame body **mesh** sourced from `vaila/sam3dinov3.py`'s SAM3+DINOv3
output, aligned into the same DLT-calibrated metric world space as the
triangulated skeleton via a per-frame Umeyama similarity fit, and exported as
a Blender-importable OBJ/PLY sequence.

## Use When
- The real fixture at `rec3d_todo/` already has everything needed to build
  and test this feature *without* any new GPU run: `c{1,2,3}_cod_markers_1_line.dlt3d`
  are **single-row (fixed) DLT3D files** — one calibration per camera for the
  whole clip — which is exactly `rec3d_one_dlt3d.py`'s input contract, not
  `rec3d.py`'s (which requires one DLT row *per frame* and would silently
  reconstruct only frame 0 against a single-row file — do not target
  `rec3d.py` first against this fixture).
- Not for the distortion-DLT math — that is `dlt-distortion-loop.md`. If that
  loop reaches `success` first, prefer its distortion-aware `.dlt3d` output
  here instead of the linear-only files, once approved (see its own
  format-change gate).
- Not for writing the didactic Kwon3D-style documentation — separate one-shot,
  after this loop is `success`.

## Inputs
1. `fixture_dir` — `/home/preto/data/sep_runcod_01072026/REC3D_COD/rec3d_todo/`
   (read-only, already contains everything below — nothing needs to be
   regenerated to iterate on this loop):
   - `c{1,2,3}_cod_markers_1_line.dlt3d` — fixed per-camera DLT3D calibration.
   - `sam3dinov3_one_person_visualized/c1_cod_sam3dinov3_visualized_id_04/`,
     `c2_cod_sam3dinov3_visualized_id_08/`, `c3_cod_sam3dinov3_visualized_id_03/`
     — already produced by `sam3dinov3_visualize.py`, one selected person per
     camera. Each contains: `c{N}_cod_id_{ID}_markers.csv` (2D pixel, 70 MHR
     keypoints, `frame,p1_x,p1_y,...,p70_x,p70_y` — directly compatible with
     `rec3d_one_dlt3d.py`'s pixel-file convention), `c{N}_cod_id_{ID}_mhr70_rec3d.csv`
     (that camera's own **monocular** 3D estimate of the same 70 keypoints,
     `p*_x,p*_y,p*_z` in meters, root-relative + `cam_t`, NOT yet in the DLT
     world frame), `c{N}_cod_sam3dinov3_camera.csv` (per-frame `focal_length_px`,
     `cam_t_x/y/z_m`, bbox), `meshes_obj/frame_NNNNNN.obj` (631 per-frame
     meshes) + `mesh_faces.npy` (shared topology).
   - `sam3sapiens2_one_person_visualized/c{1,2,3}_cod_sam3sapiens2_visualized_id_{04,08,03}/`
     — same 3 people, Sapiens2 keypoints only (no mesh; 308 kp) — usable as an
     independent cross-check of triangulated-skeleton plausibility, not as a
     mesh source.
2. `reference_camera_selection` — **auto per frame**: at each frame, compute
   the Umeyama fit (see Verification) for all 3 cameras' monocular MHR70
   keypoints against that frame's DLT-triangulated skeleton, and use whichever
   camera has the lowest fit residual as the mesh source for that frame (this
   handles occlusion/bad-view frames per camera without a fixed manual
   choice).

## Goal
`rec3d_one_dlt3d.py` gains a `--mesh-dir` (one per camera, matching
`--dlt3d`/`--pixels` order) + `--export-mesh {obj,ply}` path that: (1)
triangulates the 70 MHR keypoints exactly as it already triangulates markers
today; (2) per frame, fits a similarity transform (Umeyama: rotation +
uniform scale + translation, no shear/distortion) from each camera's
monocular MHR70 3D estimate onto the triangulated skeleton, picks the
lowest-residual camera, and applies that same transform to that camera's mesh
vertices for that frame; (3) writes the transformed mesh as an OBJ/PLY
sequence alongside the existing CSV/BVH/C3D keypoint outputs. Objectively
verifiable per the Check below; a human confirms the result imports
correctly and looks anatomically plausible in Blender.

## Verification (Governing Check)
- **True level:** 2 for the automated gate (numeric/schema/constraint check),
  5 for the final Blender review — both required; report separately, never
  blended into one pass/fail.
- **Check (deterministic, level 2), two tiers:**
  1. **Synthetic unit test** (fast, CPU-only, no real data):
     `uv run pytest tests/test_rec3d_mesh_alignment.py -v` — a new test file
     this loop creates. Construct a known ground-truth skeleton (≥8
     non-coplanar synthetic points, so the Umeyama fit is well-conditioned),
     apply a known rotation+scale+translation to build a fake "monocular"
     skeleton, verify the fitted transform recovers the known
     rotation/scale/translation to float tolerance, and that a synthetic mesh
     (e.g. a small tetrahedron) transformed by the same fit lands at the
     expected world coordinates. Also assert the fit **rejects/flags** a
     degenerate/near-planar input (condition-number or minimum-eigenvalue
     check on the point covariance) instead of silently returning an unstable
     transform — this is the numerical failure mode Umeyama is prone to with
     sparse or near-planar keypoint sets.
  2. **Real-data regression test** (marked slow/optional if `fixture_dir` is
     absent, e.g. on CI — must run locally against the real fixture before
     accepting any iteration as final):
     `uv run pytest tests/test_rec3d_mesh_export.py -v` — asserts, over all
     631 fixture frames: (a) triangulated MHR70 skeleton segment lengths
     (thigh, shank, shoulder width) stay within the ranges already validated
     in this exact dataset per project memory —
     thigh 0.387±0.017 m, shank 0.371±0.012 m, shoulder width 0.360 m — as a
     regression floor, not a new claim; (b) per-frame chosen-camera Umeyama
     residual (mean 3D distance between transformed monocular joints and
     triangulated joints, mm) stays below a threshold you set once real
     numbers are in hand (record the observed distribution in state before
     freezing a number — do not invent a threshold blind); (c) exported mesh
     vertex/face counts match the source OBJ for that frame and camera
     (no silent decimation); (d) frame-to-frame mesh centroid movement is
     continuous (no periodic reset to origin — regression guard for the bug
     class already fixed once in `sam3dinov3_visualize.py`, see its CLAUDE.md
     history).
  Also run the pre-existing regression floor:
  `uv run pytest tests/test_dlt_rec.py tests/test_dlt_rec_integration.py tests/test_sam3dinov3.py tests/test_sam3dinov3_visualize.py -v`
  and `uv run ruff check vaila/rec3d.py vaila/rec3d_one_dlt3d.py --fix && uv run ty check vaila/rec3d.py vaila/rec3d_one_dlt3d.py`.
- **Check (human checkpoint, level 5):** import the exported OBJ/PLY sequence
  into Blender (OBJSequence/"Stop Motion OBJ" extension, per the method
  already documented for `sam3dinov3_visualize.py`) on the fixture data, and
  confirm: no exploding/degenerate geometry at camera-switch frames (the
  moment the auto-selected reference camera changes is the likeliest place
  for a visible jump — look there specifically), plausible body proportions,
  motion continuity matching the source video.
- **Evidence:** pytest stdout for both test tiers, recorded every iteration;
  the real-data test's residual distribution (min/median/max mm) logged even
  while the threshold is still being set; a dated note + camera-switch-frame
  timestamps reviewed for the level-5 checkpoint, recorded only once the
  deterministic checks already pass.
- **Completion criterion:** both test tiers pass AND the pre-existing
  regression floor stays green AND the human checkpoint has been performed at
  least once on the final accepted implementation with a recorded "looks
  correct" note (explicitly including the camera-switch-frame review) AND
  `rec3d_one_dlt3d.py`/`rec3d.py` header Version/Last Updated fields, their
  `vaila/help/*.md`+`.html` pages, and `vaila/help/index.md`/`.html` are
  updated per `CLAUDE.md`'s mandatory-metadata checklist.
- **Verifier protection:** the level-2 tests are the maker's own check and
  cannot substitute for the level-5 human checkpoint — this loop must not
  report `success` from automated evidence alone. The synthetic unit test's
  ground-truth transform is fixed at authoring time and never tuned to make
  the implementation pass. Pre-existing test files listed above are frozen
  (add, never edit). The real-data segment-length ranges are copied from a
  prior, independently-obtained validation (not computed by this loop's own
  code), so this loop cannot move the goalposts by silently recomputing them.
- **Scientific validity:** the per-camera monocular MHR70 3D estimate and the
  DLT-triangulated skeleton describe the *same physical body at the same
  instant* in two different (mono-metric-guessed vs. true-metric) frames —
  Umeyama similarity alignment is the correct tool for reconciling two point
  sets known to be the same shape at different scale/pose/origin; it is not
  triangulation and it introduces no new depth information beyond what the
  monocular mesh already encoded, so document explicitly in code comments
  that "aligned" does not mean "independently verified" — the mesh's absolute
  shape/proportions are only as good as the monocular SAM 3D Body estimate,
  the loop only fixes its position/scale/orientation. Units: meters
  throughout (world frame matches `.ref3d`/triangulated-skeleton convention);
  record which of the 70 MHR keypoints are used for the fit (prefer torso +
  hip + shoulder joints — high real-world confidence, low soft-tissue
  artifact — over hand/foot tips) and log that choice in state so it is
  reviewable, not silently hardcoded.

## Trigger
Manual, invoked directly or via `dlt-camera-model-loop.md`. Duplicate-run
guard: do not start if this loop's state file already says `success` or
`exhausted`.

## Iteration
0. On the first iteration, verify `fixture_dir` and the three
   `sam3dinov3_one_person_visualized/*` subdirectories exist with their
   expected file sets (per Inputs); record checksums. No CUDA check needed
   here — every file this loop's iterations consume already exists on disk;
   CUDA is only a prerequisite for someone *regenerating* the source SAM3+DINOv3
   run, which is out of this loop's iteration budget (see Guardrails).
1. Load this file and the state file.
2. Snapshot current `rec3d_one_dlt3d.py`/`rec3d.py`; run both level-2 test
   tiers to get current evidence.
3. Rank unresolved targets worst-first: (a) load per-camera monocular MHR70
   3D + mesh + shared faces → (b) per-frame Umeyama fit per camera + the
   degenerate-input guard → (c) per-frame lowest-residual camera selection →
   (d) apply the winning transform to that frame's mesh vertices →
   (e) OBJ/PLY sequence writer → (f) `--mesh-dir`/`--export-mesh` CLI (and GUI,
   with the `>>` CLI-mirror print) wiring on `rec3d_one_dlt3d.py`. Pick the
   single worst unresolved item.
4. Make exactly one attributable code change addressing that item.
5. Run both level-2 test tiers and record raw evidence, including the
   real-data residual distribution once that tier is reachable.
6. If this iteration's change is believed to complete the feature (all
   level-2 assertions pass, including a now-frozen residual threshold),
   perform the level-5 Blender human checkpoint now, specifically reviewing
   camera-switch frames, and record its outcome; otherwise skip the
   checkpoint this iteration.
7. Retain the change only if the level-2 checks pass without regressing the
   frozen floor; otherwise revert via
   `git checkout -- vaila/rec3d.py vaila/rec3d_one_dlt3d.py` scoped to this
   iteration.
8. Curate lessons (e.g. "hand/foot MHR keypoints destabilize the Umeyama fit
   — restrict to torso/hip/shoulder") and persist state atomically.
9. Evaluate terminal states; otherwise begin the next iteration.

## Terminal States
- **success:** both level-2 test tiers pass, regression floor stays green,
  AND the level-5 Blender checkpoint (including camera-switch-frame review)
  has been performed with a recorded confirming note.
- **no-op:** both test tiers and the level-5 checkpoint already passed in a
  prior run with no intervening edits.
- **no-progress/stalled:** two consecutive iterations with no new level-2
  assertion turning from failing to passing.
- **blocked:** `fixture_dir` or any of the three `*_visualized_id_*`
  subdirectories missing/modified from their recorded checksum; the
  synthetic unit test cannot be made to pass with a mathematically correct
  Umeyama implementation (indicates a design error, escalate rather than
  loosen the test); or the level-5 checkpoint reveals a structural
  camera-switch artifact the automated check did not catch (treat as blocked
  until a continuity check across camera switches is added, not as a reason
  to lower the bar).
- **exhausted:** 8 iterations reached without both level-2 tiers passing.

Errors, missing evidence, and budget exhaustion are never success.

## Guardrails
- **Maximum allocation:** 8 iterations.
- **Human approval required:** any *new* SAM3+DINOv3 GPU run (regenerating or
  extending the fixture — not needed for this loop's normal iterations, since
  the mesh/keypoint data already exists); any `git commit`/`push`; overwriting
  anything under `fixture_dir` (never — outputs go to new timestamped
  subfolders only, matching existing `rec3d_one_dlt3d.py`/`sam3dinov3.py`
  convention); freezing the real-data residual threshold in the completion
  criterion (a scientific judgment call, not purely mechanical — present the
  observed distribution and get sign-off before hardcoding it into the test).
- **Isolation and credentials:** this loop's own iterations are CPU-only
  (linear algebra + file I/O against already-generated fixture data); no GPU
  or network access needed. A CUDA GPU is only required outside this loop's
  scope, to regenerate the source `sam3dinov3_one_person_visualized/*` data.
- **Protected verifier:** `tests/test_dlt_rec.py`,
  `tests/test_dlt_rec_integration.py`, `tests/test_sam3dinov3.py`,
  `tests/test_sam3dinov3_visualize.py` are never edited by this loop; the
  synthetic ground-truth transform in `test_rec3d_mesh_alignment.py` is fixed
  at authoring time.
- **Rollback:** `git checkout -- vaila/rec3d.py vaila/rec3d_one_dlt3d.py`
  scoped to this iteration's files only.

## State Memory
- **Path:** `loops/state/rec3d-mesh-blender-state.json`.
- **Persist:** fixture/subdirectory checksums, ranked target list with
  per-item status, level-2 evidence per iteration (both tiers), observed
  real-data residual distribution and the frozen threshold once approved,
  which MHR70 keypoints are used for the Umeyama fit, level-5 checkpoint
  outcome (date + note, including camera-switch-frame review), iteration
  count, cost.
- **Recovery:** a fresh context re-checks fixture checksums, reads which
  ranked targets are already resolved, and resumes at the next worst
  unresolved one; an `in_progress` entry with no evidence is treated as
  interrupted and redone.

## Skills
- No dedicated vailá skill for Umeyama/similarity-transform alignment; follows
  `.claude/skills/preto-loop/references/vaila-biomechanics.md` for DLT/rec3d
  conventions and metadata requirements.
- `$yolo-fb-gui-cli` — GUI→CLI parity conventions to follow when wiring the
  new `--mesh-dir`/`--export-mesh` flags into `rec3d_one_dlt3d.py`'s GUI
  dialog (must print the equivalent `>>` CLI command on Run).

## Sub-Loops
None.

## Why It Works
Splitting the deterministic checks (level 2 — a fast synthetic unit test that
proves the alignment math itself is correct, independent of the messy real
data, plus a real-data regression test anchored to segment lengths already
validated by an earlier, independent measurement) from the Blender human
checkpoint (level 5, expensive, run only once automated checks pass) means
most iterations get fast feedback while the loop still never claims "done" on
automated evidence alone. Auto-selecting the lowest-residual camera per frame
instead of a fixed reference camera directly addresses the real failure mode
in this data (each camera occasionally loses clean sight of the tracked
person), and reviewing camera-switch frames specifically in the human
checkpoint targets exactly where a bad transform would be visible. Freezing
the segment-length validation ranges from a prior independent measurement
(not recomputed by this loop) prevents the loop from grading its own homework
on the one number that would otherwise be easiest to quietly loosen.

## How to Trigger
### Context-bound
Ask the agent: "Run the next iteration of rec3d-mesh-blender-loop" (or let
`dlt-camera-model-loop.md` invoke it).

### Fresh-context / Ralph
An external runner re-reads this file and
`loops/state/rec3d-mesh-blender-state.json` every turn, re-checks fixture
checksums, resumes at the next unresolved ranked target, and stops only on a
named terminal state.

## Health Metrics
- **Cost per accepted change:** `total tokens spent / 1 if the export
  feature was accepted else 0`.
- Count of level-2 assertions passing per iteration (progress signal before
  the feature is complete); real-data Umeyama residual distribution trend
  across iterations (should tighten, not just pass/fail); count of frames
  where the auto-selected reference camera switches (context for the
  level-5 review, not itself a pass/fail gate).
