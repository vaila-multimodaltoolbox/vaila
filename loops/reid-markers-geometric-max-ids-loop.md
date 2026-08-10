---
name: reid-markers-geometric-max-ids
category: Refactoring
trigger: manual
verification-level: 1
theory-base: arXiv:2607.00038
---

# ReID Markers — Geometric (2D + Velocity) Engine, Schema Auto-Detection, and `max_ids`

## Description
Upgrade `vaila/reid_markers.py` from a GUI-only, point-only ID-merge tool into a
headless-capable, schema-agnostic (bbox or point) Geometric ReID engine bounded
by an optional `max_ids` slot pool, and give `vaila/yolov26track.py` an optional,
non-invasive offline hook to invoke it after writing `all_id_detection.csv`.

## Use When
- A YOLO/SAM/markerless tracking CSV has fragmented tracker IDs (ID switches from
  occlusion, re-entry, or detector noise) that need merging into a bounded,
  physically-plausible set of trajectories, offline, after tracking already ran.
- The input file's schema (bbox vs point, wide-per-slot vs long) is not known in
  advance and must be auto-detected rather than passed as a manual flag.
- **Exclusions:** this loop does not touch `yolov26track.py`'s live tracking loop,
  its existing `--stabilize-ids` `GeometricFrameLinker` pass, or its existing
  drop-based `--max-ids` (deliberately frozen — see Goal). It does not add
  real-time/streaming ReID, and it does not require CUDA (CPU-only pandas/NumPy/
  SciPy work).

## Inputs
1. `input_csv` — path to a tracking CSV. Validation: must exist, be readable, and
   match one of the detected schemas (bbox-wide-per-slot, bbox xyxy/xywh-per-row,
   point `pN_x/pN_y`-per-row, or SAM `sam_tracks.csv` long format — the last two
   already partially handled and must keep working). Default test value:
   `/home/preto/data/ComercialFC/vailatracker_20260807_162350/yoyo_comercial_01052026_dbox/all_id_detection.csv`
   (read-only fixture — see Guardrails).
2. `max_ids` — optional positive int. When given, bounds the number of
   concurrently-active identity slots per frame (see Goal for the frozen
   slot-pool semantics). When omitted or `0`, auto-estimate from the observed
   peak count of simultaneously-present detections in the input. Default test
   value: `18`.
3. `output_dir` — optional. Default: a new timestamped
   `processed_reid_maxids_YYYYMMDD_HHMMSS/` next to the input, per repo
   convention. Never the input file's own path.

## Goal
End state, objectively verifiable:

1. `vaila/reid_markers.py` gains a real `argparse` CLI reachable via
   `uv run python -u -m vaila.reid_markers --input <csv> [--max-ids N] [--output-dir DIR] ...`
   that runs to completion **headlessly** — no `tk.Tk()` root, no blocking
   `messagebox`/dialog call on the CLI path (same class of bug fixed in
   `rec3d.py`/`rec2d.py`: thread a `gui: bool` flag from `__main__` through every
   function that currently calls `messagebox.*`/`simpledialog.*` unconditionally,
   defaulting `gui=False` when `--input` is supplied). The existing
   `create_gui_menu()` path stays the default when invoked with no CLI args.
2. Schema auto-detection (new, e.g. `detect_input_schema(df) -> Literal["bbox_wide_slot","bbox_row","point_row","sam_tracks"]`)
   classifies the input from column headers alone — no manual format flag — and
   a schema-specific centroid extractor produces a uniform internal
   `(frame, slot_id, cx, cy, bbox_or_none)` representation before ReID runs.
   Must correctly classify: the real `all_id_detection.csv` (bbox-wide-per-slot,
   `X_min_person_id_NN` etc.), a `x1,y1,x2,y2`-per-row bbox CSV, an `x,y,w,h`-per-row
   bbox CSV, a `pN_x,pN_y` point CSV, and `sam_tracks.csv` (already handled by
   `is_sam_tracks_file`/`sam_tracks_to_marker_points` — reuse, don't duplicate).
3. The geometric engine is **extended**, not rewritten: `geometric_reid_align_markers`
   (2D + velocity-direction, point-only today) gains bbox-aware cost terms by
   reusing `geometric_reid.assignment_min_cost`, `bbox_iou_xyxy`, and the
   `GeometricLinkerConfig`/`pairwise_link_cost` machinery already shared with
   `yolov26track.py`/SAM — do not hand-roll a second Hungarian/IoU implementation.
   Temporal gaps are bridged by velocity extrapolation (already partially present
   via `max_gap`); confirm/extend gap handling so a slot can re-acquire its
   detection after occlusion within `max_gap` frames.
4. **Frozen `max_ids` design decision** (resolves the fixed-vs-peak ambiguity in
   the original request as one mechanism, not two code paths): `max_ids` bounds
   the number of **concurrently active identity slots at any single frame**
   (a fixed-size slot pool, Hungarian-assigned per frame with velocity/IoU
   gating). A slot that has been unmatched longer than `max_gap` frames becomes
   free and may be re-acquired by a *different* physical subject entering later
   — this is what lets `max_ids=18` correctly model "peak concurrency, subjects
   enter/exit over time" for the real Yo-Yo file. It also correctly serves the
   small-N "fixed subject count" case: if the true peak concurrency is ≤ N, the
   same mechanism naturally never needs more than N slots and gap-reconnection
   keeps a returning subject on its own slot when velocity is consistent. No
   separate `--max-ids-mode` flag. This assumption is explicit and reviewable —
   flagged for your override in the final report.
5. **No detections are dropped.** Unlike `yolov26track.build_id_rerank_map`
   (drop-based, deliberately untouched), every input detection row survives;
   only its slot/ID label changes. Verified by an exact preserved count of
   non-null detection cells (bbox or point) before vs. after.
6. Output schema mirrors the input convention: for `bbox_wide_slot` input, a new
   CSV in the same wide per-slot layout but with at most `max_ids` slot-groups
   (`X_min_person_id_01..NN` etc.); analogous slot-bounded convention for the
   row-per-detection schemas. Written to `output_dir`, source file untouched.
7. `--max-ids` (and the new CLI as a whole) is exposed with GUI parity: a control
   in `reid_markers.py`'s existing Tkinter dialogs, not CLI-only. GUI action
   prints the copy-pasteable `>>` CLI-mirror command on Run, per repo convention
   (`.claude/skills/yolo-fb-gui-cli`).
8. `vaila/yolov26track.py` gains an **optional, additive** post-process hook
   (e.g. `--reid-postprocess` / a GUI checkbox) that, after writing
   `all_id_detection.csv`, can invoke the new `reid_markers` CLI with a
   `max_ids` value sourced from `yolov26track`'s own args/GUI. Its existing
   live `--max-ids` (drop-based, `build_id_rerank_map`) and `--stabilize-ids`
   (`GeometricFrameLinker`) are byte-for-byte unchanged — this is additive, not
   a replacement (frozen per your explicit answer).
9. New pytest coverage: synthetic fixtures with a **known correct merge answer**
   (constructed fragmented-ID trajectories where the true merge is computable by
   hand) for the algorithm; schema-detector unit tests for all 4+ formats; a
   headless CLI smoke test (subprocess, `timeout`-guarded, asserts exit 0 and no
   TclError/hang); a mocked unit test proving `yolov26track.py`'s new hook calls
   `reid_markers` with the right arguments (no live GPU tracking required).
10. Required vailá metadata: header `Update Date`/`Version` bumped to the current
    global version (read `vaila.py`'s `Version:` at execution time, bump one
    patch) in both `reid_markers.py` and `yolov26track.py`; root `README.md`
    "Last updated"; `vaila/help/index.md`/`.html` "Generated on"; the changed
    modules' own help pages (`vaila/help/reid_markers.md`/`.html`,
    `vaila/help/reid_yolotrack.md`/`.html`, `vaila/help/yolov26track.md`/`.html`
    — all pre-existing, confirmed present).

Objectively verifiable: yes, for items 1–3, 5–6, 9–10 (deterministic/constraint
checks). Item 4's *semantic correctness on the real file* (are the merged IDs
truly the right 18 people) has **no ground truth** and is explicitly evidence-only,
not a pass/fail gate — see Verification.

## Verification (Governing Check)
- **True level:** primarily **1 (deterministic)** — pytest against synthetic
  fixtures with a hand-computable correct merge, schema-detector exact-match
  assertions, and row/cell-count-preservation assertions; layered with
  **2 (rule/constraint)** — ruff/ty, headless-CLI-no-hang guard, GUI-control
  presence checked via source inspection (a live GUI cannot run unattended).
  The real-file run is **5 (human checkpoint)** evidence only — flagged as such,
  never presented as proof the 18-slot merge is semantically correct.
- **Check (run every iteration, in order):**
  ```bash
  uv run ruff check vaila/reid_markers.py vaila/yolov26track.py --fix
  uv run ruff format vaila/reid_markers.py vaila/yolov26track.py
  uv run ty check vaila/reid_markers.py vaila/yolov26track.py
  uv run pytest tests/test_reid_markers_geometric.py tests/test_reid_markers_schema.py \
                 tests/test_reid_markers_cli.py tests/test_reid_yolotrack.py -v
  ```
  Plus, once the CLI exists (not before), the real-file smoke run as **evidence**,
  not a pass/fail gate on its own:
  ```bash
  timeout 600 uv run python -u -m vaila.reid_markers \
    --input /home/preto/data/ComercialFC/vailatracker_20260807_162350/yoyo_comercial_01052026_dbox/all_id_detection.csv \
    --max-ids 18 \
    --output-dir /home/preto/data/vaila/loops/state/reid_maxids_smoketest
  ```
- **Evidence:** full stdout/stderr of each command, verbatim, written into the
  state file per iteration; the smoke run's own printed summary (rows in/out,
  raw unique-ID count before, slot count after, peak per-frame concurrency
  observed, detected schema name, runtime, exit code); a small checker script's
  output confirming detection-cell-count preservation and the ≤18-per-frame
  slot bound on the real output file.
- **Completion criterion:** all four `ruff`/`ty`/`pytest` commands exit 0 with
  no new failures and no regressions vs. the last accepted iteration, AND the
  real-file smoke run exits 0 within the 600 s timeout with the checker script
  confirming (a) detection-cell count in == out, (b) max concurrent slots per
  frame in the output ≤ 18, (c) schema logged as `bbox_wide_slot`, AND the GUI
  `max_ids` control and `>>` CLI-mirror print are present (source-level check),
  AND `yolov26track.py`'s hook is wired and covered by its mocked unit test,
  AND metadata items in Goal §10 are updated.
- **Verifier protection:** the four governing commands and the checker script
  are never edited by the same iteration that touches `reid_markers.py`/
  `yolov26track.py` — if a test needs changing, that is itself one attributable
  iteration with red-before/green-after evidence recorded, not a silent edit
  bundled with a feature change. The synthetic fixtures' "known correct merge"
  values are computed independently (by hand / a separate script) before the
  engine change, not derived from the engine's own output.
- **Scientific validity:** input modality is 2D image-plane pixel tracking data
  (OpenCV/YOLO convention: origin top-left, +x right, +y down); no physical
  units or lab-frame conversion involved at this stage (pure pixel-space ID
  merge). Frame indexing must be preserved exactly (no off-by-one against the
  `Frame` column). No sampling-rate/fps dependency for the merge itself, but if
  present in the source it must pass through unchanged. Hardware: CPU-only;
  no CUDA required or invoked by this loop.

## Trigger
Manual — invoked by you (or an agent you direct) in a single dev session; not
scheduled or event-driven. Duplicate-run protection: check
`loops/state/reid-markers-geometric-max-ids-loop-state.json` for an
`in_progress` status with a fresh-enough timestamp before starting a second run.

## Iteration
0. On the first iteration: confirm the repo is clean on `main`
   (`git status --porcelain`), create and switch to a feature branch
   (`reid-markers-geometric-max-ids`), validate `input_csv` exists and is
   readable, freeze `max_ids=18` and the output path convention.
1. Load this spec and `loops/state/reid-markers-geometric-max-ids-loop-state.json`;
   confirm the 30-iteration budget and approval gates below are intact.
2. Snapshot `git diff` as the baseline; run the governing check; record evidence.
3. Rank unresolved Goal items worst-first by blocking order (this ordering *is*
   the priority ranking, not a suggestion): (1) headless CLI existence/no-hang
   guard — nothing else is testable without it; (2) schema auto-detection;
   (3) bbox-aware velocity/IoU engine reusing `geometric_reid.py`; (4) `max_ids`
   slot-pool; (5) no-dropped-detections + output schema; (6) GUI parity + `>>`
   mirror; (7) `yolov26track.py` additive hook; (8) new tests for 1–7;
   (9) metadata/help-doc updates; (10) real-file smoke evidence.
4. Invoke `$test-writer` (for the synthetic fixtures ahead of each algorithmic
   change, red-before) or `$gui-developer` (for CLI/GUI plumbing) or
   `$video-processor` (for the `yolov26track.py` hook) to make exactly one
   attributable change addressing the current top-ranked item.
5. Run the governing check; record raw evidence verbatim.
6. Keep the change only if the check passes with no regression vs. the last
   accepted state; otherwise `git checkout -- vaila/reid_markers.py
   vaila/yolov26track.py` (and any touched test/help files) to roll back just
   this iteration, then record why.
7. Curate lessons (only ones borne out by evidence — e.g., "the real file's
   schema needed X because Y", not speculation); atomically persist state,
   evidence, counters, and cost.
8. Evaluate terminal states; otherwise begin the next iteration.

## Terminal States
- **success:** every completion-criterion clause above is met, evidence for all
  of them is present in the state file, and `$code-review` has been run once
  against the full diff with zero unresolved CONFIRMED findings.
- **no-op:** re-running the loop finds every Goal item already satisfied by
  evidence already recorded in state (e.g., resuming a completed run) — report
  the existing evidence, make no changes.
- **no-progress/stalled:** two consecutive iterations produce an identical
  pass/fail signature across the four governing commands with no Goal item
  advanced and no regression newly fixed — stop and surface the exact failing
  evidence for human review rather than iterating blindly.
- **blocked:** the fixed real-file path becomes unreadable/moved; `ruff`/`ty`/
  `pytest` toolchain is unavailable; or an architecture question arises that
  the frozen slot-pool design (Goal §4) does not resolve — surface it rather
  than guessing.
- **exhausted:** 30 iterations reached without full success.

Errors, timeouts, missing evidence, and budget exhaustion are never success.

## Guardrails
- **Maximum allocation:** 30 iterations (one attributable change + governing
  check each). No dollar/token ceiling was requested; escalate to a human
  checkpoint if projected cost materially exceeds a normal single-feature
  session before continuing.
- **Human approval required:** any `git commit`/`git push`; deleting any file;
  writing to anywhere under `/home/preto/data/ComercialFC/...` (the source
  fixture is read-only for this loop — all writes go to `output_dir` or the
  repo's own `loops/state/` scratch area); running `yolov26track.py`'s live
  GPU tracking path to smoke-test the new hook end-to-end (not required for
  this loop's completion criterion, since the hook's unit test mocks the call;
  if a contributor later wants a live run, that needs explicit approval plus
  hardware confirmation per CLAUDE.md's CUDA-work requirement).
- **Isolation and credentials:** runs in the existing repo working tree on the
  dedicated feature branch; no network access needed; no credentials involved.
- **Protected verifier:** the four governing commands, the checker script, and
  the synthetic fixtures' hand-computed expected merge are not edited by the
  same iteration that changes `reid_markers.py`/`yolov26track.py` (see
  Verifier protection above).
- **Rollback:** `git checkout -- <touched files>` scoped to the current
  iteration's changes only; the feature branch itself is never force-pushed or
  rebased destructively without explicit approval.

## State Memory
- **Path:** `loops/state/reid-markers-geometric-max-ids-loop-state.json`.
- **Persist:** baseline `git diff` hash, terminal status, per-iteration attempts
  (what changed, which Goal item it targeted), accepted/rejected changes with
  reasons, raw governing-check evidence (ruff/ty/pytest/CLI-smoke stdout+stderr),
  curated lessons, human-approval decisions, and iteration/cost counters.
- **Recovery:** a fresh context re-reads this loop document and the state file
  before acting; an `in_progress` status with no matching iteration-completion
  record for the last logged attempt means that write was interrupted — treat
  the last *fully recorded* iteration as the resume point and re-run its
  governing check before proceeding (never trust an unconfirmed "accepted"
  entry).

## Skills
- `$check` — the repo's ruff+ty+pytest pipeline convention; run as the governing
  check's first three commands each iteration.
- `$test-writer` — writes the synthetic fixtures with hand-computed correct
  merges (red-before), schema-detector unit tests, and the CLI smoke test.
- `$gui-developer` — the `max_ids` Tkinter control, `>>` CLI-mirror print, and
  keeping the single-root Tk model intact (no second `tk.Tk()`).
- `$video-processor` — the `yolov26track.py` additive hook and its bbox-schema
  interop with `all_id_detection.csv`.
- `$getpixelvideo-tracking-loader` — existing precedent/pattern for this
  repo's "smart" format auto-detection (SAM3 vs YOLO); mirror its structure for
  the new bbox-vs-point schema detector rather than inventing a new pattern.
- `$yolo-fb-gui-cli` — the GUI→CLI mirror print convention (`>>` prefix, never
  `[bracketed]` — absl eats brackets) applied to both the new `reid_markers`
  CLI output and `yolov26track.py`'s new hook.
- `$code-review` — final correctness pass against the full diff before
  declaring success (medium/high effort; this repo's history shows this step
  reliably finds real bugs, not just style issues).

## Why It Works
- The blocking-order ranking (CLI existence → schema detection → engine →
  max_ids → no-drop/output → GUI parity → yolov26track hook → tests → metadata
  → real-file evidence) prevents wasted iterations on downstream work that
  can't even be tested yet (e.g., writing max_ids tests before a CLI exists to
  run them against).
- Reusing `geometric_reid.py`'s Hungarian/IoU/velocity primitives instead of a
  second implementation prevents exactly the kind of duplicate-linker drift
  this repo's own history (`geometric_reid.py`'s consolidation of 3 duplicate
  `_assignment_min_cost` implementations) already had to fix once.
- Freezing the slot-pool `max_ids` semantics up front (Goal §4) prevents the
  loop from oscillating between two incompatible interpretations mid-run —
  the single largest risk in the original request.
- Hand-computed synthetic ground truth (not the engine's own output) as the
  Level-1 check prevents the loop from grading its own homework; the real
  file's 18-slot result is explicitly evidence, not a pass condition, so the
  loop cannot claim scientific correctness it hasn't earned.
- Per-iteration git-scoped rollback plus a dedicated feature branch keeps every
  iteration cheaply reversible and keeps `main` untouched until a human merges.
- The headless-CLI-no-hang guard directly targets a failure mode this repo has
  hit before verbatim (`rec3d.py`/`rec2d.py` hanging test suites on unguarded
  `messagebox` calls) — checked explicitly rather than assumed absent.

## How to Trigger
### Context-bound
Within a single Claude Code session with this repo open: read this file, then
execute Iteration steps 0–8 directly, keeping all evidence in this turn's
context and in the state file.

### Fresh-context / Ralph
For a longer run spanning multiple sessions: each fresh context must re-read
this loop document and `loops/state/reid-markers-geometric-max-ids-loop-state.json`
in full before acting, resume from the last fully recorded iteration (see
Recovery), and stop only on a named terminal state — never on "looks done"
without the governing check's evidence in hand.

## Health Metrics
- **Cost per accepted change:** total tokens spent / count of iterations whose
  change was retained (not rolled back).
- **Goal items closed / 10** — from the Goal enumeration, tracked per iteration.
- **Regression count** — iterations rolled back because the governing check
  regressed vs. the last accepted state.
- **Real-file smoke evidence:** cell-count-preservation delta (must be exactly
  0), max per-frame slot count in output (must be ≤ `max_ids`), CLI runtime
  (seconds), schema detected (must equal `bbox_wide_slot` for this fixture).
