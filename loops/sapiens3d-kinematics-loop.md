---
name: sapiens3d-kinematics
category: Biomechanics
trigger: manual
verification-level: 1
theory-base: arXiv:2607.00038
---

# Sapiens2 3D Lower-Limb Joint Kinematics (Hip/Knee/Ankle from REC3D C3D)

## Description
Add `vaila/sapiens3d_kinematics.py`: a new vailá module that reads a Sapiens2
REC3D C3D file, builds functional pelvis/thigh/shank/foot coordinate frames
from Sapiens2 keypoint positions, computes Hip/Knee/Ankle relative rotation
matrices, converts them to quaternions and Euler/Cardan angles (functional
and Vicon-compatible sequences), and exposes the result through a new GUI
button, a standalone Tkinter dialog, a CLI, QC plots, help docs, and tests.

## Use When
- Building/iterating this exact feature end to end: geometry math, GUI
  button + dialog, CLI, outputs, docs, tests — as scoped by the 43-section
  `/goal` this loop was generated from (full text preserved in this
  session's transcript; key normative points are restated below so the loop
  is self-contained).
- **Exclusions:** this loop does not implement the future CUBE/Plug-in Gait
  comparison pipeline (§38 of the source goal) — only keeps outputs shaped
  so that comparison stays possible later. Does not touch
  `vaila/joint_kinematics.py`'s SAM-3D-Body MHR-rig pipeline, `rotation.py`'s
  generic 3-point/4-point toolkit, or `mpangles.py`'s 2D angle pipeline —
  those stay as-is; this is a fourth, explicitly-scoped convention (see
  Frozen design decisions), not a rewrite of any of them.

## Inputs
1. `c3d_path` — a Sapiens2 REC3D C3D file (single file) or a folder of them
   (batch mode, GUI "Folder" browse / CLI accepts a directory for `-i`).
   Confirmed real fixtures for testing: `tests/viewc3d/rec3d_240hz.c3d`
   (240 Hz, 255 points, `p1`..`p255` labels, units `m`, 8298 frames) and
   `tests/viewc3d/rec3d_200hz.c3d` (200 Hz, verify point count/labels in
   iteration 0 the same way). Both are read-only inputs for this loop.
2. `output_dir` — defaults to a timestamped
   `processed_sapiens3d_kinematics_YYYYMMDD_HHMMSS/` next to the input,
   overridable via `-o`/GUI browse, per repo convention
   (`edit_csv_c3d.py`/`interp_smooth_split.py` timestamp their own outputs
   the same way).

## Goal
`vaila/sapiens3d_kinematics.py` exists with a `run_sapiens3d_kinematics()`
GUI entry point and a CLI (`python -m vaila.sapiens3d_kinematics` /
`uv run vaila/sapiens3d_kinematics.py`) that together satisfy every
numbered requirement below. This restates the source `/goal`'s 43 sections
in loop-actionable form; where the source goal offered two candidate names
or asked the loop to verify an assumption, the resolved choice is recorded
here as a **frozen design decision** so iterations don't re-litigate it.

1. **Read C3D** — `ezc3d.c3d(path)` directly (matches every other vailá C3D
   consumer — `readc3d_export.py`, `viewc3d.py`, `rec3d_one_dlt3d.py`; no
   shared `load_c3d()` wrapper exists in the repo to call instead, so this
   is not a missed-reuse case). Read `POINT.RATE`, `POINT.UNITS`, point
   labels, and the `(4, n_points, n_frames)` data array; treat 4th row
   (residual) as validity signal if present and non-degenerate.
2. **Identify Sapiens2 landmarks** — see "Sapiens2 keypoint mapping" below.
   Must be resolved from real, verified data, not the source goal's
   unverified index guess.
3. **Optional Butterworth filtering** — reuse `vaila/filter_utils.py`'s
   `butter_filter()` (SOS, `sosfiltfilt`, zero-phase; exact function the
   source goal's §24 pseudocode independently re-derives) — do not
   reimplement. Apply to 3D coordinates before frame construction. Default
   cutoff 10 Hz, order 4, `fs` read from `POINT.RATE`. Do not filter across
   gaps larger than a documented threshold (reuse `butter_filter`'s own gap
   handling if present; otherwise segment-and-filter, documented).
4. **Segment frames** — pelvis, left/right thigh, left/right shank,
   left/right foot, built per the exact vector recipes in the source
   goal §8–§12 (functional pelvis from hip-mid/shoulder-mid, thigh
   longitudinal hip→knee with pelvis-Y projected secondary axis,
   foot-constrained shank, foot from heel→toe-mid with a left/right-mirrored
   `foot_left` convention). These recipes are a distinct geometric method
   from `rotation.py`'s generic `createortbase`/`createortbase_4points`
   (3-point/4-point triangle configurations) — not a case of skipped reuse;
   log this distinction in the module docstring the way
   `joint_kinematics.py` already logs its own divergence from `rotation.py`
   and `mpangles.py` (see Frozen design decisions).
5. **`project_to_so3(M)`** — SVD/polar projection to the nearest proper
   rotation (source goal §7's exact recipe), used after every frame build
   as numerical-noise cleanup; reflection matrices (`det < 0`) are corrected,
   never silently accepted.
6. **Relative joint rotations** — `R_hip = R_pelvis.T @ R_thigh`,
   `R_knee = R_thigh.T @ R_shank`, `R_ankle = R_shank.T @ R_foot`, both sides.
7. **Quaternions** — `scipy.spatial.transform.Rotation.from_matrix(...).as_quat()`,
   scalar-last `[x, y, z, w]`, documented explicitly as such in module
   docstring, help doc, and metadata JSON. This is a **deliberate divergence**
   from `vaila/joint_kinematics.py`'s scalar-first `[w, x, y, z]` convention
   (`rotmat_to_quat_wxyz`) and from `rotation.py`'s `rotmat2quat` (which
   *claims* `(w, x, y, z)` in its docstring but actually returns scipy's
   `[x, y, z, w]` — a known, already-flagged repo inconsistency). This
   module picks scipy's native convention and documents it correctly rather
   than repeating either existing mismatch or inventing a fourth
   undocumented one; source goal §18 explicitly requires the scipy
   scalar-last convention.
   Enforce temporal sign continuity (`if dot(q[t], q[t-1]) < 0: q[t] *= -1`)
   before any quaternion plot or CSV write.
8. **Euler/Cardan** — two conventions: `functional_xyz` (repo's existing
   default sequence, matching `rotation.py.rotmat2euler`'s `"xyz"`) and
   `vicon_compatible` (Hip Y-X-Z, Knee Y-X-Z, Ankle Y-Z-X, intrinsic —
   verify current scipy's intrinsic/extrinsic letter-case behavior with a
   synthetic test before trusting it, per source goal §20).
9. **Neutral calibration** — none / first-valid-frame / mean-of-first-N /
   frame-range / time-range, computed as `R_zeroed(t) = R0.T @ R_joint(t)`
   in SO(3) (never Euler-angle subtraction). A proper rotation/quaternion
   mean (not independent-component averaging) when `R0` comes from multiple
   frames.
10. **Missing data / degeneracy** — NaN/zero-filled points, near-zero
    vectors, near-collinear planes → NaN output + QC flag, never an
    arbitrary orientation. Report per-keypoint valid-data percentage for
    all 14 required landmarks (source goal §26 list).
11. **Outputs** — the five files named in source goal §30–31, with the
    exact column sets specified there (segment matrices, joint matrices,
    quaternions, Euler/Cardan raw + anatomical-alias columns, metadata
    JSON with the explicit `model_type`/`anatomical_equivalence`/
    `shank_orientation` limitation strings from §31).
12. **QC plots** — the five figures in source goal §32 (Hip/Knee/Ankle
    angle panels, quaternion components, `det(R)`/orthogonality-error/
    invalid-frame QC), matplotlib, no misleading smoothing.
13. **New GUI button** — **Frame C-A** (Data Files), a new flat button
    (frozen design decision — see below), label **"Sapiens2 3D Kinematics"**,
    opening a dedicated Tkinter dialog matching source goal §27's sections
    (Input, Coordinate system, Output convention, Filtering, Neutral
    calibration, Output options, Side, Buttons), never blocking the main
    Tk root (lazy import inside the button handler, matching every other
    Frame C-A button; new `Toplevel`, not a second `tk.Tk()`).
14. **CLI** — `python -m vaila.sapiens3d_kinematics` with the flags listed
    in source goal §29, argparse, matching this repo's existing CLI-flag
    naming (compare `rec3d_one_dlt3d.py`/`interp_smooth_split.py` flag
    style during iteration 0).
15. **Help doc** — `vaila/help/sapiens3d_kinematics.md` + `.html`, the 17
    sections in source goal §39, `vailá` styled per `CLAUDE.md`'s italic-
    lowercase convention; central `vaila/help/index.md`/`.html` updated.
16. **Tests** — synthetic rotation validation (source goal §22, mandatory:
    identity, isolated flexion/frontal/axial rotations, combined rotations,
    both Euler sequences, both sides, determinant, quaternion unit-norm,
    sign-continuity, neutral zeroing) plus a real-fixture smoke test against
    `tests/viewc3d/rec3d_240hz.c3d`.
17. **Metadata bump** — `vaila.py` version (`0.3.111` → `0.3.112` at time of
    writing; re-check current value at iteration-0 time, it may have moved),
    new module header, root `README.md` "Last updated", `vaila/help/index.md`
    + `.html` "Generated on".

### Sapiens2 keypoint mapping — unresolved, iteration-0-critical
The source goal's index table (`5 left_shoulder`, `9 left_hip`, ... assuming
a 308-keypoint Sociopticon order) is **explicitly unverified** and the real
fixture contradicts its headline assumption: `tests/viewc3d/rec3d_240hz.c3d`
has **255** points (`p1`..`p255`), not 308. The repo's actual name source is
`vaila/vaila_sapiens.py`'s `_load_sapiens308_keypoint_names_cached()` /
`_resolve_sapiens_keypoint_names()` (Sociopticon 308 canonical order, cached
from `sapiens.pose.datasets.parse_pose_metainfo`) plus
`_resolve_keypoint_labels()`/`_sanitize_keypoint_label()` which sanitize
those names into CSV column prefixes. Downstream, `rec3d.py`'s
`load_pixel_csv_positional()` and `rec3d_one_dlt3d.py`'s `save_rec3d_as_c3d()`
both re-label points generically as `p1..pN` by **column position**, not by
carrying the semantic name through — so the semantic identity of `p{i}` is
only recoverable by cross-referencing the *pre-REC3D* 2D pixel CSV's column
order (`pN_x`/`pN_y`) against whichever keypoint subset/order that specific
Sapiens2 run used, not by assuming a fixed universal 308-index table.
Iteration 0 must: (a) confirm whether `sapiens.pose.datasets` is importable
in this environment at all (it may not be — that package is part of the
heavy GPU inference stack, and this new module must run without it, since
it's pure C3D post-processing); (b) if not importable, locate or regenerate
the 2D pixel/pose CSV that fed the `rec3d_240hz.c3d` fixture (or an
equivalent sidecar) to recover the real `p{i}` → keypoint-name order for
that fixture; (c) freeze a small local constant table
(`SAPIENS3D_REQUIRED_LANDMARKS: dict[str, int]`, index only for the 14
landmarks this module needs) inside the new module, derived and documented
from (a)/(b) — not a live dependency on the `sapiens` package; (d) if no
such sidecar can be found and the mapping cannot be verified from data,
downgrade to a GUI/CLI-exposed, user-confirmed manual index override with a
loud warning, and record that as an accepted limitation rather than a
silent guess. This resolution is Goal item 2 and gates almost everything
else — rank it first in every iteration's target list until closed.

### Frozen design decisions (from the interview — do not re-ask)
- **GUI placement:** new flat button in **Frame C-A**, not folded into an
  existing chooser. (User-selected; the repo's general trend toward
  choosers — e.g. MP Angles moving into the Markerless 2D chooser — was
  raised and explicitly overridden for this module, since it's C3D
  post-processing, not live detection.)
- **Module filename:** `vaila/sapiens3d_kinematics.py` (first of the two
  source-goal candidates).
- **Quaternion convention:** scipy scalar-last `[x, y, z, w]`, diverging
  deliberately from `joint_kinematics.py` and `rotation.py` (see Goal
  item 7). Document, don't silently pick.
- **Reused, not reimplemented:** `vaila/filter_utils.py.butter_filter`
  (filtering — do not write a second Butterworth implementation).
  `scipy.spatial.transform.Rotation` for matrix↔quat↔Euler conversions
  (already a project dependency via `rotation.py`/`joint_kinematics.py`).
- **Not reused (by design, documented divergence):** `rotation.py`'s
  `createortbase*`/`calcmatrot`/`rotmat2euler`/`rotmat2quat` — different
  geometric recipe and a known docstring/behavior mismatch in
  `rotmat2quat`; `joint_kinematics.py`'s `local_rotations_from_global`/
  `rotmat_to_euler_xyz_deg`/`rotmat_to_quat_wxyz` — built for SAM-3D-Body's
  regressed full-body rig rotations (127-joint MHR tree), not a 2/3-point
  plane heuristic from bare keypoint positions; wrong shape of input.

## Verification (Governing Check)
- **True level:** primarily **1 (deterministic)** — the synthetic rotation
  test suite (source goal §22) against hand-constructed known rotation
  matrices, with exact expected Euler/quaternion outputs computed
  independently of the implementation. Layered with **2 (rule/constraint)**
  — ruff/ty. Real-fixture numeric QC (`det(R)`, orthogonality error,
  valid-frame %, plausible angle ranges on `rec3d_240hz.c3d`) is **3
  (delayed field truth)** — informative, not itself a pass/fail gate, since
  there's no independent ground-truth angle for that fixture. GUI dialog
  open/click and plot rendering are **5 (human checkpoint)** evidence.
- **Check (run every iteration, in order):**
  ```bash
  uv run ruff check vaila/sapiens3d_kinematics.py --fix
  uv run ruff format vaila/sapiens3d_kinematics.py
  uv run ty check vaila/sapiens3d_kinematics.py
  uv run pytest tests/test_sapiens3d_kinematics.py -v
  uv run pytest tests/ -v -k "sapiens3d_kinematics or not sapiens3d_kinematics" --co -q
  ```
- **Evidence:** full stdout/stderr of each command; synthetic-test expected
  values shown alongside actual output; a real-fixture run transcript
  (`uv run vaila/sapiens3d_kinematics.py -i tests/viewc3d/rec3d_240hz.c3d -o
  /tmp/.../out --side both --convention vicon --plots`) with the resulting
  CSV/JSON/plot file list and a QC summary (mean `det(R)`, max orthogonality
  error, % valid frames per joint) recorded as level-3 evidence.
- **Completion criterion:** the five commands exit 0 with no regression vs.
  the last accepted iteration; the synthetic suite asserts (independently
  computed, not derived from the implementation under test):
  - Identity segment matrices → `0°, 0°, 0°` on all three axes, both
    Euler conventions.
  - A pure `+30°` flexion-axis rotation (matching whichever Euler-sequence
    slot the goal's §21 anatomical alias maps to flexion — verified, not
    assumed) round-trips to `30.0°` (±1e-6) on that slot and ~0° on the
    other two.
  - Isolated `+10°`/`-10°` frontal-plane and `+15°`/`-15°` axial rotations
    round-trip on their respective slots for both `functional_xyz` and
    `vicon_compatible` (Hip/Knee Y-X-Z, Ankle Y-Z-X).
  - A combined `Y=+30°, X=+10°, Z=+15°` intrinsic rotation (constructed via
    `Rotation.from_euler(seq, [...], degrees=True)` in the *intended*
    intrinsic sequence) decomposes back to the same three values.
  - `project_to_so3` on a matrix with injected noise/reflection returns a
    matrix with `det ≈ +1` and `‖RᵀR − I‖_F` below the documented tolerance.
  - Quaternion `‖q‖ ≈ 1` for every synthetic case; a constructed sign-flip
    sequence (`q, -q, q, -q, ...`) is corrected to continuous sign.
  - Neutral zeroing: `R_zeroed(t) = R0.T @ R_joint(t)` reproduces identity
    at the reference frame(s) and the expected relative rotation elsewhere.
  - Left and right sides both tested, not just one mirrored assumption.
  AND the real-fixture run produces all five documented output files with
  the documented columns, non-empty QC plots, and a metadata JSON containing
  the required limitation strings verbatim (§31).
- **Verifier protection:** the five governing commands and
  `tests/test_sapiens3d_kinematics.py`'s hand-computed synthetic rotation
  constants are never edited in the same iteration that touches
  `vaila/sapiens3d_kinematics.py`'s implementation; a wrong hand-computed
  constant is its own attributable iteration with the corrected math shown.
- **Scientific validity:** units/frame stay exactly as read from the C3D
  (already meters per the confirmed fixture; no silent unit conversion).
  Every anatomical-alias output column name is only assigned to a specific
  Euler-sequence slot after that slot's meaning is confirmed by the
  synthetic isolated-rotation tests in this section — never assigned
  because "the numbers look plausible" (source goal §21's explicit ban).
  The functional/surrogate limitation (axial rotation of thigh/shank is not
  anatomically observable from joint centers alone, per Wu et al. 2002 ISB
  recommendations and Grood & Suntay JCS) is stated in module docstring,
  GUI, help doc, and metadata JSON — not just one of the four.

## Trigger
Manual — invoked directly in this dev session, current branch (`multiview`
at time of writing; confirm at run time). Not scheduled or event-driven.
Duplicate-run protection: check
`loops/state/sapiens3d-kinematics-loop-state.json` for an `in_progress`
status with a fresh-enough timestamp before starting a second run.

## Iteration
0. On the first iteration: confirm `git status --porcelain` on the current
   branch; confirm `tests/viewc3d/rec3d_240hz.c3d` and `rec3d_200hz.c3d` are
   readable (`ezc3d.c3d(...)`); resolve the Sapiens2 keypoint-mapping
   question above and freeze the resulting constant table with its
   verification method recorded in state; re-check the live `vaila.py`
   version line (may have moved since this doc was written) before
   committing to a version-bump target; skim `rec3d_one_dlt3d.py` and
   `interp_smooth_split.py` for the CLI flag/timestamped-output-dir
   conventions to mirror exactly.
1. Load this spec and
   `loops/state/sapiens3d-kinematics-loop-state.json`; confirm the
   40-iteration budget and approval gates below are intact.
2. Snapshot `git diff` as baseline; run the governing check; record
   evidence.
3. Rank unresolved Goal items worst-first by blocking order: (1) keypoint
   mapping resolution (blocks everything); (2) pure geometry functions —
   `project_to_so3`, frame builders, relative rotation, quaternion/Euler
   conversion with continuity — each independently unit-testable before
   any I/O exists; (3) synthetic rotation test suite (red before the
   geometry functions are correct, green after — prove it, don't just
   write green tests); (4) C3D read + landmark resolution + filtering
   glue; (5) neutral calibration; (6) CSV/JSON writers; (7) QC plots;
   (8) CLI; (9) GUI dialog + Frame C-A button wiring; (10) help doc +
   index update; (11) metadata bump; (12) real-fixture smoke run + QC
   summary recorded as evidence.
4. Make exactly one attributable change addressing the current top-ranked
   item.
5. Run the governing check; record raw evidence verbatim.
6. Keep the change only if it passes with no regression vs. the last
   accepted state; otherwise `git checkout -- vaila/sapiens3d_kinematics.py
   tests/test_sapiens3d_kinematics.py` (and any touched GUI/help/doc files)
   to roll back just this iteration, then record why.
7. Curate lessons (only ones borne out by evidence); atomically persist
   state, evidence, counters, and cost.
8. Evaluate terminal states; otherwise begin the next iteration.

## Terminal States
- **success:** every completion-criterion clause is met, evidence for all
  of them is present in the state file, the main vailá GUI opens, the new
  button opens the dialog without freezing the root, and a real-fixture
  CLI run transcript with the resulting output-file list is recorded.
- **no-op:** re-running the loop finds every Goal item already satisfied by
  evidence already recorded in state.
- **no-progress/stalled:** two consecutive iterations produce an identical
  pass/fail signature with no Goal item advanced — stop, surface the exact
  failing evidence for human review.
- **blocked:** the keypoint-mapping question (see above) cannot be resolved
  from any available data source and no human-in-the-loop override is
  approved; the `ruff`/`ty`/`pytest` toolchain is unavailable; `ezc3d` or
  `scipy.spatial.transform` cannot be imported in this environment.
- **exhausted:** 40 iterations reached without full success.

Errors, timeouts, missing evidence, and budget exhaustion are never success.

## Guardrails
- **Maximum allocation:** 40 iterations (one attributable change +
  governing check each). No dollar/token ceiling requested.
- **Human approval required:** any `git commit`/`git push`; deleting any
  file; modifying either fixture `.c3d` file in `tests/viewc3d/` (read-only
  inputs); adopting the manual-index-override fallback for keypoint mapping
  (a scientific-validity-affecting decision, not a routine implementation
  step).
- **Isolation and credentials:** runs in the existing repo working tree; no
  network access needed; no credentials involved. If keypoint-mapping
  resolution turns out to require the `sapiens` GPU package, that import
  attempt is sandboxed/optional — the shipped module must not hard-require
  it at runtime (see Frozen design decisions).
- **Protected verifier:** the five governing commands and the hand-computed
  synthetic-rotation constants in `tests/test_sapiens3d_kinematics.py` are
  not edited by the same iteration that changes
  `vaila/sapiens3d_kinematics.py`.
- **Rollback:** `git checkout -- <touched files>` scoped to the current
  iteration's changes only.

## State Memory
- **Path:** `loops/state/sapiens3d-kinematics-loop-state.json`.
- **Persist:** baseline `git diff` hash, terminal status, resolved
  keypoint-mapping table + verification method, per-iteration attempts
  (what changed, which Goal item it targeted), accepted/rejected changes
  with reasons, raw governing-check evidence, curated lessons, human-
  approval decisions, iteration/cost counters.
- **Recovery:** a fresh context re-reads this loop document and the state
  file before acting; an `in_progress` status with no matching
  iteration-completion record for the last logged attempt means that write
  was interrupted — treat the last *fully recorded* iteration as the
  resume point and re-run its governing check before proceeding.

## Skills
- `$check` — repo's ruff+ty+pytest pipeline convention; the governing
  check's first three commands each iteration.
- `$gui-developer` — Frame C-A button wiring, the new Tkinter dialog,
  lazy-import + single-Tk-root discipline (matching every neighboring
  Frame C-A button: `edit_csv_c3d`, `rearrange_data`, `interp_smooth_split`).
- `$biomechanics-analyst` — ISB/Grood-Suntay joint-coordinate-system
  correctness, the functional-vs-anatomical limitation language, sign/QC
  conventions for pelvis/thigh/shank/foot frames.
- `$test-writer` — the mandatory synthetic rotation validation suite
  (source goal §22), red-before/green-after on each geometry function.

## Why It Works
- Resolving the keypoint-mapping question first (rather than assuming the
  source goal's unverified 308-index table) prevents building an entire
  pipeline on a wrong landmark identity — the real fixture already disagrees
  with the assumption (255 points, not 308), so this would have surfaced as
  a silent correctness bug rather than a loud iteration-0 finding otherwise.
- Documenting the quaternion-convention divergence from
  `joint_kinematics.py`/`rotation.py` up front (instead of picking one
  ad hoc mid-implementation) stops this module from becoming a *fourth*
  undocumented convention in a repo that already has three
  (`joint_kinematics.py`'s own docstring names the first three) — the
  divergence is real (different source data shape demands it) but now it's
  a logged decision, not an accident.
- Pure, independently-testable geometry functions (`project_to_so3`, frame
  builders, relative rotation, quaternion continuity, Euler decomposition)
  make the mandatory synthetic-rotation suite possible at level 1 —
  without that extraction, correctness of the Euler-sequence-to-anatomical-
  name mapping would only be checkable by eyeballing plots, which the
  source goal's own §21/§22 explicitly forbid ("do not just assign labels
  because the numbers look plausible").
- SO(3) projection after every frame build, plus explicit NaN/QC flagging
  on degenerate vectors, keeps numerical noise and missing markers from
  producing silently wrong (but plausible-looking) rotation matrices.
- Per-iteration git-scoped rollback keeps this large, multi-file addition
  cheaply reversible without a second branch.

## How to Trigger
### Context-bound
Within this Claude Code session with the repo open: read this file, then
execute Iteration steps 0–8 directly, keeping evidence in this turn's
context and in the state file.

### Fresh-context / Ralph
For a run spanning multiple sessions: each fresh context must re-read this
loop document and `loops/state/sapiens3d-kinematics-loop-state.json` in
full before acting, resume from the last fully recorded iteration, and stop
only on a named terminal state.

## Health Metrics
- **Cost per accepted change:** total tokens spent / count of iterations
  whose change was retained.
- **Goal items closed / 17** — from the Goal enumeration, tracked per
  iteration.
- **Regression count** — iterations rolled back because the governing check
  regressed vs. the last accepted state.
</content>
