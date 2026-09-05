---
name: markerless-2d-mp-world-seg
category: Vailá
trigger: manual
verification-level: 1
theory-base: arXiv:2607.00038
---

# Markerless 2D MediaPipe World Landmarks + Segmentation Loop

## Description
Extend `vaila/markerless_2d_analysis.py` so MediaPipe Tasks `PoseLandmarkerResult.world_landmarks`
and `segmentation_masks` are configured, executed, and exported consistently across GUI
(`ConfidenceInputDialog`), TOML, CLI `batch`, and the video processing path — with
deterministic tests for config/schema and mocked result extraction.

## Use When
- Adding or repairing `export_world_landmarks` → `*_mp_world.csv`.
- Wiring `enable_segmentation` / `save_segmentation_mask` / `smooth_segmentation` all the
  way from config into `PoseLandmarkerOptions` and annotated-frame / disk outputs.
- Not for: YOLO/SAM/Sapiens pipelines; 3D world-frame biomechanical inverse dynamics;
  changing MediaPipe model packs unrelated to pose landmarker options.

## Inputs
1. `module` — `vaila/markerless_2d_analysis.py`.
2. `help` — `vaila/help/markerless_2d_analysis.md` + `.html` (and nvidia help siblings if
   they document the same outputs).
3. `test_fixture` — synthetic/mocked pose results under pytest `tmp_path` (mandatory CI):
   fake `world_landmarks` (33 × x,y,z,visibility) and optional mask arrays; no GPU video
   required for the governing check.
4. `optional_smoke_video` — external short `.mp4` only if present on disk; skip if absent;
   never commit large media.
5. `hardware` — CPU for unit/config tests; full MediaPipe video smoke optional when
   weights + GPU/CPU delegate available.

**Logged interview defaults (2026-09-05):**
- Loop name `markerless-2d-mp-world-seg`; save under `./loops/`.
- `export_world_landmarks` default **true**; `enable_segmentation` default **false**;
  `save_segmentation_mask` default **false**; `smooth_segmentation` remains existing default.
- World CSV schema mirrors `*_mp_norm.csv` column naming but adds visibility:
  `frame_index` + per landmark `{name}_x`, `{name}_y`, `{name}_z`, `{name}_visibility`
  (33 MediaPipe body landmarks; units: **meters** for x,y,z; visibility in \[0,1\]).
- MediaPipe world frame is the Tasks API hip-relative metric frame — document in help;
  do not invent a clinical anatomical frame.
- When `enable_segmentation=true`: set Tasks `output_segmentation_masks=True`, blend mask
  onto annotated `*_mp.mp4` frames.
- When `save_segmentation_mask=true`: write **both** a per-frame PNG sequence under
  `<video_out>/segmentation_masks/` (`frame_%06d.png`) **and** optional mask-overlay MP4
  sibling if cheap to encode alongside the annotated pipe; if encoding cost is high,
  PNG sequence is mandatory and overlay MP4 is best-effort (document which shipped).
- CLI: TOML remains source of truth; also add explicit overrides
  `--export-world-landmarks` / `--no-export-world-landmarks`,
  `--enable-segmentation` / `--no-enable-segmentation`,
  `--save-segmentation-mask` / `--no-save-segmentation-mask`.
- No launcher artifact unless requested later.
- Existing `enable_segmentation` GUI/TOML keys are incomplete until Options + result path
  consume them — treat that gap as in-scope.

## Goal
Objectively verifiable end state:

1. **Config keys** present in `get_default_config()`, `get_flat_default_pose_config()`,
   `merge_pose_config_with_defaults()`, `save_config_to_toml()`, `load_config_from_toml()`:
   - `export_world_landmarks` (bool, default `true`)
   - `enable_segmentation` (bool, default `false` — already exists; must round-trip)
   - `smooth_segmentation` (bool — already exists; must round-trip)
   - `save_segmentation_mask` (bool, default `false`)
2. **GUI** (`ConfidenceInputDialog`): toggles for `export_world_landmarks` and
   `save_segmentation_mask`; existing segmentation fields remain; load/save TOML and
   apply-to-dialog populate all four.
3. **CLI** `uv run python vaila/markerless_2d_analysis.py batch ...` honors TOML values and
   the override flags above; GUI Run prints `>>` equivalent CLI including these options
   when mirroring is already used for this module (or extend mirror if present).
4. **World landmarks path:** when enabled, extract `result.world_landmarks[0]` → x,y,z
   (meters) + visibility for 33 landmarks; write `{stem}_mp_world.csv` next to
   `*_mp_norm.csv` / `*_mp_pixel.csv`. When disabled, do not write the file.
5. **Segmentation path:** when `enable_segmentation`, landmarker options request masks;
   blend onto annotated frames. When `save_segmentation_mask`, persist mask PNGs (and
   overlay MP4 if implemented). Missing masks on a frame → transparent/no-op blend,
   no crash; NaN policy for missing world landmarks matches existing missing-pose
   (NaN rows / NaN landmark tuples).
6. **Docs/metadata:** module header Version/Update Date; help lists new outputs and
   units; README / help index “Generated on” / Last updated when repo rules require.
7. **Regression:** existing norm/pixel CSV schemas and batch device/nvenc flags keep
   working; focused tests green.

## Verification (Governing Check)
- **True level:** 1 (pytest on config round-trip, CSV schema builders, mocked
  `PoseLandmarkerResult` extraction/blend helpers) with level-2 ruff/ty. Optional
  real-video smoke is level 3/5 QA — not the autonomous gate.
- **Check (every iteration):**
  ```bash
  uv run ruff check vaila/markerless_2d_analysis.py tests/test_markerless_2d_world_seg.py --fix
  uv run ruff format vaila/markerless_2d_analysis.py tests/test_markerless_2d_world_seg.py
  uv run ty check vaila/markerless_2d_analysis.py tests/test_markerless_2d_world_seg.py
  uv run pytest tests/test_markerless_2d_world_seg.py -v
  ```
  Create `tests/test_markerless_2d_world_seg.py` on the first test iteration (red before
  green). Prefer extracting pure helpers (e.g. `world_landmarks_to_row`,
  `build_world_csv_columns`, `blend_segmentation_mask`, TOML key presence) so tests do
  not require Tk or a live MediaPipe GPU session.
- **Evidence:** raw stdout/stderr of the check commands recorded in
  `loops/state/markerless-2d-mp-world-seg-loop-state.json`.
- **Completion criterion:**
  - All Goal items 1–7 accepted with evidence in state.
  - Defaults: `export_world_landmarks=true`, `save_segmentation_mask=false`.
  - Round-trip: save TOML → load → flat dict contains new keys with correct types.
  - Synthetic world row: known x,y,z,visibility recover in CSV headers and values.
  - Mocked options builder: `output_segmentation_masks` True iff `enable_segmentation`.
  - Ruff/ty clean on touched files; header + help metadata updated (stale metadata with
    green tests = incomplete).
- **Verifier protection:**
  - Do not weaken existing markerless2d tests or delete them to pass.
  - Do not hard-code live video paths into runtime code.
  - Do not silently drop visibility columns or change units to pixels for world CSV.
  - Maker must not edit the frozen check command list in this file to skip failures.
- **Scientific validity:**
  - World: meters; MediaPipe Tasks world landmark frame; visibility \[0,1\].
  - Image/normalized landmarks remain unitless \[0,1\] in `*_mp_norm.csv`; pixel in
    `*_mp_pixel.csv` — do not mix frames in one file.
  - Mask: same spatial resolution as the frame passed to MediaPipe (document crop/resize
    mapping if ROI crop is enabled; if ambiguous, save mask in processing-frame space and
    state that in help).
  - Frame index alignment with existing CSV `frame_index` convention.
  - Missing pose: NaNs, not zeros pretending to be origin.

## Trigger
Manual. Operator starts the loop; re-entry loads state JSON and skips accepted targets.
Duplicate concurrent runs forbidden (one writer of the state file).

## Iteration
0. On first iteration: freeze Inputs defaults; snapshot baseline (`git status` + current
   defaults/TOML keys); add failing tests for missing keys/schema if not present.
1. Load this specification and durable state; confirm budget and approvals.
2. Snapshot baseline and run the governing check; record raw evidence.
3. Rank unresolved Goal targets worst-first (config keys → extraction helpers → video
   path wiring → GUI → CLI flags → help/metadata).
4. Invoke `$surgical-patch` (or `$investigate-first` on failures) for **exactly one**
   attributable change.
5. Run the governing check; append raw evidence to state.
6. Retain only if non-regressive; otherwise rollback that iteration’s paths.
7. Curate lessons; atomically persist state, counters, cost.
8. Evaluate terminal states; otherwise next iteration.

## Terminal States
- **success:** Goal items 1–7 each have accepted evidence; governing check green.
- **no-op:** Feature already fully present (keys, CSV writer, Options masks, GUI/CLI,
  tests, help) with green governing check on first baseline.
- **no-progress/stalled:** Two consecutive iterations with no accepted target and no
  metric improvement on open Goal items.
- **blocked:** MediaPipe Tasks API shape incompatible with assumed attributes and cannot
  be mocked without a separate dependency upgrade; or Tk single-root conflict requires
  human architecture decision beyond this scope.
- **exhausted:** Hard budget reached.

Errors, missing evidence, and budget exhaustion are never success.

## Guardrails
- **Maximum allocation:** 12 iterations; 40 agent tool-turns; local `uv` only; no paid APIs.
- **Human approval required:** git commit/push; committing videos/weights; changing
  existing `*_mp_norm.csv` / `*_mp_pixel.csv` column schemas; raising default
  `enable_segmentation` to true (VRAM/CPU cost).
- **Isolation and credentials:** no network except existing model download paths already
  used by the module; tests offline with mocks; no second `tk.Tk()` in helpers; headless
  batch must not open a hidden Tk root.
- **Protected verifier:** frozen check commands above; synthetic fixtures owned by tests.
- **Rollback:** `git checkout -- <touched paths>` for that iteration only.

## State Memory
- **Path:** `loops/state/markerless-2d-mp-world-seg-loop-state.json`
- **Persist:** baseline, terminal status, attempts, accepted/rejected changes, evidence
  (command exit codes + truncated stdout/stderr), curated lessons, decisions, cost
  (iterations + tool-turns), open Goal checklist.
- **Recovery:** fresh context re-reads this markdown + state JSON; if JSON corrupt or
  partial write, rebuild open targets from Goal and mark last attempt `interrupted`.

## Skills
- `$investigate-first` — locate where Options / result parsing omit world/masks.
- `$surgical-patch` — one config, helper, GUI, CLI, or test change per iteration.
- `$verify-and-stop` — refuse scope creep into YOLO/SAM or biomechanical reinterpretation
  of the MediaPipe world frame.
- `$lean-build` — keep helpers thin; reuse existing CSV write patterns.
- `$preto-loop` — maintain this document; **do not execute** this loop unless the
  operator explicitly starts it.

## Sub-Loops
None. Circular nesting prohibited. If MediaPipe package upgrade is required, stop
**blocked** and open a separate dependency loop — do not nest it here.

Parent × child worst case: N/A (no children).

## Why It Works
- External pytest + ruff/ty prevent “looks blended” self-approval (anti self-scoring).
- Mocked result extraction separates software acceptance from full-video field truth.
- Atomic one-change iterations attribute regressions to a single edit.
- Explicit NaN/units/frame docs stop silent scientific misuse of world meters vs norm \[0,1\].
- Terminal states distinguish success from stalled/blocked/exhausted.
- State on disk enables Ralph-style fresh contexts without chat memory.

## How to Trigger
### Context-bound
Instruct the agent to execute `markerless-2d-mp-world-seg` until a named terminal state,
re-reading this file and the state JSON each turn. Do **not** treat writing this spec as
starting the implementation loop.

### Fresh-context / Ralph
```text
1. Read loops/markerless-2d-mp-world-seg-loop.md
2. Read loops/state/markerless-2d-mp-world-seg-loop-state.json (create empty skeleton if missing)
3. One Iteration (exactly one attributable change)
4. Run governing check; update state atomically
5. Stop only on success | no-op | no-progress/stalled | blocked | exhausted
```

Illustrative only — not an unattended runner; approvals and budgets still apply.

## Health Metrics
- **Cost per accepted change:** `total_tool_turns_or_iterations / verified_changes_retained`.
- **Coverage progress:** accepted Goal items / 7.
- **Regression:** existing `tests/test_markerless2d_v2_tracker_reset.py` and
  `tests/test_markerless_2d_merge_warmup.py` remain green when run at risk milestones
  (at least once before **success**).
