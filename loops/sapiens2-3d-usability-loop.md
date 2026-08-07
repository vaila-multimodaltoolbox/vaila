---
name: sapiens2-3d-usability-loop
category: Markerless/video/AI
trigger: manual
verification-level: 3
theory-base: arXiv:2607.00038
---

# Sapiens2 3D Pose — Usability Fix + DLT3D/ref3d Auto-Chaining

## Description
Fix a real, diagnosed bug in shared derived-video filtering that caused
`vaila/sapiens2_3d.py` to queue a rendered overlay video as if it were raw
footage; add smart results-directory resolution so pointing the GUI/CLI at
the wrong (but related) folder either auto-resolves or fails with an
actionable suggestion instead of a bare `FileNotFoundError`; and add an
optional one-run chain from `sapiens2_3d.py`'s monocular camera-frame
mesh/keypoints into `monocular_dlt_align.py`'s DLT3D-calibrated lab frame, so
producing calibrated mesh + keypoints no longer needs a manual second step.

## Use When
- Following on from `sapiens2-3d-pipeline-loop.md` (status: `success`,
  2026-08-06) — that loop built `sapiens2_3d.py` itself; this loop fixes real
  friction reported from actually using it, plus adds the DLT3D/ref3d
  integration explicitly requested but out of that loop's original scope.
  Not a sub-loop call — a sibling loop, run standalone.
- Not for revisiting the bbox-tightening accuracy question (already closed,
  `success`, on smoke-clip evidence) — that is closed unless new contrary
  evidence shows up on a harder clip.

## Inputs
1. `fixture_dir` — `/home/preto/data/sep_runcod_01072026/REC3D_COD/rec3d_todo/`
   (read-only). At authoring time already contains a **complete, real**
   `processed_sam3sapiens2_20260806_233956/` run (from the user's own session,
   not this loop) with `{c1_cod,c2_cod,c3_cod}/` combined SAM3+Sapiens2
   results (each has its own `*_sam3sapiens2_predictions.json`) **and** three
   `*_sam3sapiens2_visualized_id_NN/` rerender directories — i.e. the exact
   pair of "right directory" / "wrong directory" this loop needs to
   distinguish already exists on disk, needing **no fresh SAM3/Sapiens2 GPU
   work** to reproduce the reported failure or verify the fix. Also has
   `c{1,2,3}_cod_markers_1_line.dlt3d` (fixed per-camera DLT3D) and
   `c1c2c3_cod.ref3d` (control points) for the DLT-chaining half.
2. `reported_failure_transcript` — the user's pasted terminal log (2026-08-07):
   GUI run with Input video/folder = `.../c1_cod_sam3sapiens2_visualized_id_04`,
   both `--sapiens2-results` and `--sam-results` set to that same folder,
   queued video = `c1_cod_sam3sapiens2_id_04_overlay.mp4` (the rerender's own
   output, not raw footage), failed with
   `No *_sam3sapiens2_predictions.json found under ...`.

## Goal
Three concrete, independently-verifiable deliverables:
1. `_is_derived_video()` (in `sam3sapiens2.py`, shared by `sam3sapiens2.py`,
   `sam3dinov3.py`, and `sapiens2_3d.py` via `_find_videos()`) recognizes
   every known derived-overlay naming pattern via a regex, not a growing
   substring tuple — confirmed root cause: it matched `_sam3sapiens2_overlay`
   but not `_sam3sapiens2_id_04_overlay` (the actual `sam3sapiens2_visualize.py`
   output pattern), nor `_sam3dinov3_id_NN_overlay`, nor `sapiens2_3d.py`'s own
   `_sapiens2_3d_overlay`.
2. `sapiens2_3d.py` (CLI + GUI) resolves a plausible-but-wrong results
   directory (e.g. a `*_visualized_id_NN` folder) to its sibling combined-run
   directory automatically when unambiguous, and otherwise fails with an
   error naming the specific directory it suggests instead of a bare
   "not found" — reusing the `"video"` field already stored in
   `*_sam3sapiens2_predictions.json` to cross-check/pre-fill the correct raw
   video path.
3. `sapiens2_3d.py` gains optional `--dlt3d`/`--ref3d`/`--export-mesh`/
   `--smooth-hz`/`--origin-markers` flags (CLI + GUI, new "Calibrated lab
   frame" GUI section matching `markerless_3d_analysis()`'s existing section
   pattern). When `--dlt3d` is given, after each person's
   `*_id_NN_mhr70_rec3d.csv`/`*_id_NN_markers.csv` are written, the run calls
   `monocular_dlt_align.align_monocular_to_world()` for that person
   automatically (same function the standalone tool already exposes — no
   duplication), writing calibrated-world CSV/BVH/mesh into a
   `dlt_world/id_NN/` subfolder. Absent `--dlt3d`, behavior is byte-identical
   to today (pure additive, backward compatible).

## Verification (Governing Check)
- **True level:** 2 (lint/type), 1 (unit tests), 3 (real reproduction against
  the user's own existing fixture data — no fresh GPU stage needed for SAM3/
  Sapiens2, only SAM3D Body + DLT placement), 5 (human re-tries the GUI).
- **Check, tier 1 (level 2, every iteration):**
  `uv run ruff check vaila/sapiens2_3d.py vaila/sam3sapiens2.py vaila/monocular_dlt_align.py --fix && uv run ty check vaila/sapiens2_3d.py vaila/sam3sapiens2.py vaila/monocular_dlt_align.py`.
- **Check, tier 2 (level 1, every iteration, CPU-only):**
  `uv run pytest tests/test_sapiens2_3d.py tests/test_sam3sapiens2.py -v` plus
  new tests this loop adds: (a) `_is_derived_video()` regex matches all five
  known patterns (`_sam_overlay`, `_sapiens_overlay`, `_sam3sapiens2_overlay`,
  `_sam3sapiens2_id_NN_overlay`, `_sam3dinov3_overlay`,
  `_sam3dinov3_id_NN_overlay`, `_sapiens2_3d_overlay`) **and does not**
  match real raw-video names already used in tests/fixtures (regression guard
  — this is exactly the kind of fix that can overreach and start excluding
  legitimate input); (b) sibling-directory resolution finds the combined-run
  dir from a `*_visualized_id_NN` dir and produces a specific, actionable
  error (not a bare "not found") when no sibling exists; (c) the DLT-chaining
  call site: with a mocked `align_monocular_to_world`, `--dlt3d` given calls
  it once per person with the right `mono3d`/`pixels`/`dlt3d`/`ref3d` paths;
  omitted, it is never called (backward-compatibility guard). Also run the
  frozen regression floor: `uv run pytest tests/test_sam3dinov3.py tests/test_vaila_sapiens.py tests/test_monocular_dlt_align.py tests/test_vaila_cli_menu.py -v`.
- **Check, tier 3 (level 3, real data, no GPU-run approval needed for the
  reused SAM3+Sapiens2 stage — approval still required before any *new*
  SAM3D Body / DLT placement GPU-adjacent run, per Guardrails):**
  (a) rerun the **exact reported scenario** —
  `sapiens2_3d.py -i .../c1_cod_sam3sapiens2_visualized_id_04 --sapiens2-results .../c1_cod_sam3sapiens2_visualized_id_04` —
  and confirm it now either succeeds (auto-resolved to `.../c1_cod`) or fails
  with a message naming that exact sibling path; (b) run
  `sapiens2_3d.py --sapiens2-results .../c1_cod --dlt3d .../c1_cod_markers_1_line.dlt3d --ref3d .../c1c2c3_cod.ref3d --save-mesh --export-mesh obj`
  end to end and confirm `dlt_world/id_NN/` outputs exist with a plausible
  reprojection error (compare against the already-documented
  `monocular_dlt_align` validation on this exact fixture family — **1.18 px**
  mean reprojection — as a regression floor, not a new claim; a wildly worse
  number here would indicate the auto-chain wired something up incorrectly,
  not that the placement math regressed, since that math is untouched).
- **Check (human checkpoint, level 5):** the user re-runs the GUI end to end
  once (their own original scenario, or a fresh one) and confirms the
  reported friction is gone.
- **Evidence:** raw ruff/ty/pytest stdout every iteration; tier-3 stdout +
  the resolved directory/error message text; the tier-3(b) reprojection-error
  number; the human's confirmation note.
- **Completion criterion:** tiers 1–2 pass, regression floor stays green,
  tier 3(a) and 3(b) both produce the evidence described above, AND the
  level-5 human checkpoint confirms the friction is resolved, AND
  `sapiens2_3d.py`/`sam3sapiens2.py` header Version/Update Date,
  `vaila/help/sapiens2_3d.md`+`.html`, and `vaila/help/index.md`+`.html` are
  updated per `CLAUDE.md`'s metadata checklist.
- **Verifier protection:** the derived-video regex test's pattern list is
  fixed from the *actual* naming conventions read out of
  `sam3sapiens2_visualize.py`/`sam3dinov3_visualize.py`/`sapiens2_3d.py`
  source at authoring time (see Goal item 1) — not invented, not loosened to
  make a failing case pass. `tests/test_sam3dinov3.py`,
  `tests/test_vaila_sapiens.py`, `tests/test_monocular_dlt_align.py`,
  `tests/test_vaila_cli_menu.py` are frozen (read, never edited). The tier-3(b)
  reprojection floor (1.18 px) is copied from a prior, independently-obtained
  measurement on this same fixture, not recomputed by this loop's own code.
- **Scientific validity:** the DLT-chain reuses `align_monocular_to_world()`
  unmodified — units, coordinate frame (OpenCV camera frame → DLT lab frame),
  and the "scale is not free, only placement is solved" caveat are exactly as
  documented in that module's own docstring and CLAUDE.md's 2026-08-05 entry;
  this loop does not re-derive or restate that math, only wires the call site.
  Never overwrite anything under `fixture_dir` — DLT-chained outputs go into
  the run's own output directory, never back into `rec3d_todo`.

## Trigger
Manual. Duplicate-run guard: do not start if this loop's state file already
says `success` or `exhausted`.

## Iteration
0. On the first iteration, verify `fixture_dir`'s
   `processed_sam3sapiens2_20260806_233956/{c1_cod,c2_cod,c3_cod}` combined
   runs and their sibling `*_visualized_id_NN` dirs still exist and match a
   recorded checksum; verify the three `.dlt3d` files and `c1c2c3_cod.ref3d`
   are present. CUDA check only needed before tier-3(b) (SAM 3D Body still
   runs); tier-3(a) and the DLT placement math are CPU-only.
1. Load this file and the state file.
2. Snapshot current repo state; run tiers 1–2.
3. Rank unresolved targets worst-first: (a) fix `_is_derived_video()` to a
   regex covering all real patterns + regression test against real raw-video
   names → (b) sibling-directory resolution + actionable error message in
   `sapiens2_3d.py`'s front-end resolver → (c) GUI pre-fill of the raw video
   path from the predictions JSON's own `"video"` field → (d) `--dlt3d`/
   `--ref3d`/`--export-mesh`/`--smooth-hz`/`--origin-markers` CLI flags +
   call-site wiring to `align_monocular_to_world()` → (e) GUI "Calibrated lab
   frame" section, mirroring `markerless_3d_analysis()`'s existing pattern →
   (f) unit tests for (a)–(d) → (g) tier-3(a) real reproduction of the
   reported scenario → (h) tier-3(b) real DLT-chain run → (i) docs/version
   metadata. Pick the single worst unresolved item.
4. Make exactly one attributable change addressing that item.
5. Run tiers 1–2 and record evidence. For (g)/(h), request human approval
   first only if a genuinely new GPU stage becomes necessary (expected: no,
   since SAM3+Sapiens2 results already exist on disk); otherwise proceed
   directly since SAM 3D Body inference on an already-guided run is the same
   class of cost already approved in the prior loop.
6. If the change is believed to complete the feature set, perform the
   level-5 human checkpoint now; otherwise skip it this iteration.
7. Retain the change only if tiers 1–2 pass without regressing the frozen
   floor; otherwise revert via
   `git checkout -- vaila/sapiens2_3d.py vaila/sam3sapiens2.py vaila.py`
   scoped to this iteration.
8. Curate lessons and persist state atomically.
9. Evaluate terminal states; otherwise begin the next iteration.

## Terminal States
- **success:** tiers 1–2 pass, regression floor stays green, tier 3(a) and
  3(b) both produce their described evidence, and the level-5 checkpoint
  confirms the friction is resolved, AND metadata/help docs updated.
- **no-op:** not expected for items (a)/(b)/(c) — a real bug and a real
  missing feature are already diagnosed. Could apply narrowly to item (d) if
  investigation surfaces a reason full auto-chaining is unsafe (e.g. a path
  ambiguity `align_monocular_to_world()` cannot resolve automatically); in
  that case fall back to the "fix discoverability only" alternative from the
  original interview and document why, rather than forcing an unsafe chain.
- **no-progress/stalled:** two consecutive iterations with no previously
  failing tier-1/tier-2 assertion turning green.
- **blocked:** the recorded fixture checksums no longer match (files
  moved/deleted); CUDA unavailable for tier-3(b); `align_monocular_to_world()`
  cannot be called without modifying its own signature (escalate — that
  function is meant to stay a stable reusable entry point).
- **exhausted:** 10 iterations reached without `success` or `no-op`.

Errors, missing evidence, and budget exhaustion are never success.

## Guardrails
- **Maximum allocation:** 10 iterations.
- **Human approval required:** any genuinely *new* SAM3/Sapiens2 GPU run
  (not expected — existing results are reused); any `git commit`/push;
  overwriting anything under `fixture_dir` (never); the final level-5
  checkpoint confirmation itself is the human approval for `success`.
- **Isolation and credentials:** SAM 3D Body + DLT placement stages run in
  the existing vailá CUDA venv; no new network access; no credentials
  handled.
- **Protected verifier:** `tests/test_sam3dinov3.py`,
  `tests/test_vaila_sapiens.py`, `tests/test_monocular_dlt_align.py`,
  `tests/test_vaila_cli_menu.py` never edited by this loop. The tier-3(b)
  reprojection floor (1.18 px) is copied from prior independent measurement,
  not recomputed here. `align_monocular_to_world()`'s own signature/behavior
  is never modified — only called.
- **Rollback:**
  `git checkout -- vaila/sapiens2_3d.py vaila/sam3sapiens2.py vaila.py`
  scoped to this iteration's files only.

## State Memory
- **Path:** `loops/state/sapiens2-3d-usability-loop-state.json`.
- **Persist:** fixture checksums, ranked target list with per-item status,
  tier-1/2 evidence per iteration, tier-3(a) resolved-path or suggested-path
  text, tier-3(b) reprojection-error number, the level-5 checkpoint outcome
  (date + note), iteration count, cost.
- **Recovery:** a fresh context re-checks fixture checksums, reads which
  ranked targets are already resolved, and resumes at the next worst
  unresolved one; an `in_progress` entry with no evidence is treated as
  interrupted and redone.

## Skills
- `$yolo-fb-gui-cli` — GUI→CLI parity conventions for the new "Calibrated lab
  frame" GUI section and its `>>` CLI mirror.
- `$getpixelvideo-tracking-loader` — precedent for "smart" input-resolution
  UX (auto-detect format/anchor from a loosely-specified path) that item (b)/
  (c) should follow in spirit, even though the concrete formats differ.
- `.claude/skills/preto-loop/references/vaila-biomechanics.md` — markerless
  ID-persistence and coordinate-frame checks to apply while wiring the DLT
  chain.

## Sub-Loops
None. Sibling to (not called by, not calling) `sapiens2-3d-pipeline-loop.md`
(status: `success`).

## Why It Works
The failure was reproducible and root-caused from the transcript alone —
`_is_derived_video()`'s substring list simply predates the
`_id_NN_overlay` naming convention `sam3sapiens2_visualize.py` and
`sam3dinov3_visualize.py` actually use, and `sapiens2_3d.py` inherited the
same shared helper. Fixing it as a regex closes the exact class of bug, not
just the one instance, and the regression test (real raw-video names must
still pass) prevents the fix from overreaching. Reusing the real
already-computed `processed_sam3sapiens2_20260806_233956/` run for tier-3
evidence means the loop can validate against genuine data without spending
GPU time re-deriving SAM3/Sapiens2 results that already exist — the same
efficiency reasoning as `rec3d-mesh-blender-loop.md`'s fixture reuse. Wiring
`align_monocular_to_world()` as an unmodified call site (never editing its
math) means the DLT-chain feature cannot silently alter a placement algorithm
that was already independently validated (1.18 px on this exact fixture) —
any regression there would point at the *wiring*, not the *math*, which
keeps debugging targeted. The level-5 checkpoint is unavoidable here because
"mais fácil" is a human-experience claim no automated check can certify on
its own — the automated tiers can only prove correctness and non-regression.

## How to Trigger
### Context-bound
Ask the agent: "Run the next iteration of sapiens2-3d-usability-loop."

### Fresh-context / Ralph
An external runner re-reads this file and
`loops/state/sapiens2-3d-usability-loop-state.json` every turn, re-verifies
fixture checksums, resumes at the next unresolved ranked target, and stops
only on a named terminal state.

## Health Metrics
- **Cost per accepted change:** `total tokens spent / count of {derived-video
  fix, sibling-resolution fix, DLT auto-chain} actually shipped`.
- Tier-1/2 assertions passing per iteration; tier-3(a) resolution outcome
  (auto-resolved vs. actionable-error, either acceptable) as a discrete
  signal; tier-3(b) reprojection-error trend (should stay ≈1.18 px, not
  drift); human checkpoint pass/fail.
