---
name: viewc3d-multifile
category: Vailá
trigger: manual
verification-level: 1
theory-base: arXiv:2607.00038
---

# viewc3d Multi-File C3D Loading, Color-Coding, and FPS Sync

## Description
Extend `vaila/viewc3d.py`'s Open3D viewer to load and display multiple `.c3d`
files at once in the same running window, each with a distinct marker color,
on a single synchronized timeline when source FPS rates differ.

## Use When
- A user has two or more related `.c3d` recordings (e.g. two force-plate-synced
  cameras, a before/after trial pair, two subjects) and wants to overlay them
  in one Open3D session instead of opening separate viewer instances.
- **Exclusions:** the matplotlib fallback path (`run_viewc3d_fallback`,
  triggered when OpenGL/Open3D is unavailable) stays single-file — it has no
  per-marker geometry model to extend and this loop does not add one.
  `run_viewc3d_from_array` (CSV-sourced data) is untouched. Trails, dynamic
  skeleton overlay, measurement-line, and marker-swap editing stay scoped to
  the first-loaded file only — extending those to every loaded file is future
  work, not this loop's goal.

## Goal
End state, objectively verifiable:

1. A new `Ctrl+L` key action (via `register_key_action_callback` on
   `ord("L")`, checking `mods & 2`, mirroring the existing `ctrl_s_key_action`
   pattern) opens a Tk multi-select file dialog
   (`_create_centered_tk_root()` + `filedialog.askopenfilenames`, filter
   `*.c3d`) and loads every chosen file into the running viewer without
   closing it. Plain `L` (already bound to "Set custom view limits") is
   unchanged — this is a frozen key-mapping decision (`L` was unavailable;
   every other letter is already single-bound in this file).
2. A new lightweight `LoadedC3D` container (next to `VailaModel`, not a
   rewrite of it) holds one `VailaModel`, an assigned RGB color, its Open3D
   sphere geometries/base-vertex arrays, and a per-frame index-mapping
   callable. `loaded_files: list[LoadedC3D]` starts with the originally
   opened file at index 0; single-file behavior and existing tests are
   unchanged when nothing is loaded via `Ctrl+L`.
3. Each newly loaded file is assigned
   `available_colors[len(loaded_files) % len(available_colors)]` (the
   existing 11-entry palette) at load time. This is a **frozen** decision:
   colors for `Ctrl+L`-loaded files are fixed at load time, not manually
   re-cyclable via the existing `C` key in this iteration (`C` continues to
   cycle only file index 0's color, preserving current single-file
   behavior). Re-color-per-file is out of scope here.
4. Marker selection (`select_markers`) runs for each newly loaded file
   exactly as it already does for the first file — no special-casing needed
   at the data layer.
5. Pure, independently testable helper functions (new, colocated near
   `VailaModel`):
   - `resolve_master_fps(fps_list, downsample_to_lowest) -> float`:
     `max(fps_list)` by default, `min(fps_list)` when the user opts to
     downsample.
   - `master_frame_count(durations_seconds, master_fps) -> int`:
     `round(max(durations_seconds) * master_fps)`.
   - `map_master_frame_to_local(master_idx, master_fps, file_fps, file_num_frames) -> int`:
     `round(master_idx * file_fps / master_fps)` clamped to
     `[0, file_num_frames - 1]` (a shorter recording holds its last frame
     once a longer one continues — explicit, documented behavior, not a bug).
6. On loading a file whose FPS differs from the current master FPS (tolerance
   `1e-6`), a single `messagebox.askyesno` prompt ("Downsample higher-FPS
   file(s) to match the lowest FPS instead of the highest?") is shown once
   per load action (not once per file); the boolean answer recomputes
   `resolve_master_fps`/`master_frame_count`/`map_master_frame_to_local` for
   **every** currently-loaded file, matching Goal item 2's requirement that
   the timeline defaults to the highest-FPS dataset but can be downsampled to
   the lowest on request.
7. `update_spheres` (the existing per-frame redraw routine) is extended, not
   duplicated, to iterate `loaded_files` and redraw each file's own
   spheres/base-vertices/color at its mapped local frame index; the master
   `current_frame` nonlocal remains the single driver of navigation/playback
   (arrows, Space, `S`/`E`, `[`/`]`), now interpreted as a master-timeline
   index instead of a raw file frame index.
8. Window title, the `H` help overlay, module docstring, `vaila/help/viewc3d.md`
   + `.html`, and CLI `--help` text all mention `Ctrl+L` (load additional
   file), the per-file color legend, and the master-FPS/sync behavior.
9. New `tests/test_viewc3d_multifile.py` covers the pure helpers in item 5
   against the real fixtures' known FPS (`tests/viewc3d/rec3d_200hz.c3d` =
   200 Hz/8074 frames, `tests/viewc3d/rec3d_240hz.c3d` = 240 Hz/8298 frames)
   with hand-computed expected values (see Verification), plus color-cycling
   wraparound at the 12th load.
10. Required vailá metadata: `vaila.py` `Version: 0.3.110` → `0.3.111`;
    `vaila/viewc3d.py` header `Version`/`Last Updated` bumped to match;
    root `README.md` "Last updated"; `vaila/help/index.md`/`.html`
    "Generated on"; `vaila/help/viewc3d.md`/`.html` Version/Updated + new
    capability section.

Objectively verifiable: yes, for items 1–3, 5, 7, 9–10 (deterministic/
constraint checks against pure functions and source inspection). Item 6's
prompt and item 4/7/8's live Open3D rendering require a human to actually
run the viewer — flagged as level-5 evidence, not a pass/fail gate.

## Verification (Governing Check)
- **True level:** primarily **1 (deterministic)** — pytest against the pure
  timeline/color helpers with hand-computed values from the two real
  fixtures; layered with **2 (rule/constraint)** — ruff/ty, and source-level
  checks that `Ctrl+L`, the help text, and the docstring were actually
  updated. Live two-file rendering in the Open3D window is **5 (human
  checkpoint)** evidence only (no headless Open3D harness exists in this
  repo for the GLFW window itself).
- **Check (run every iteration, in order):**
  ```bash
  uv run ruff check vaila/viewc3d.py --fix
  uv run ruff format vaila/viewc3d.py
  uv run ty check vaila/viewc3d.py
  uv run pytest tests/test_viewc3d_multifile.py -v
  uv run pytest tests/ -v -k "viewc3d or not viewc3d" --co -q  # confirm no collection breakage repo-wide
  ```
- **Evidence:** full stdout/stderr of each command; the hand-computed
  expected values below compared against the helper functions' actual
  output; a manual run transcript (`uv run vaila/viewc3d.py
  tests/viewc3d/rec3d_200hz.c3d`, press `Ctrl+L`, load
  `rec3d_240hz.c3d`) confirming two colors on screen and no crash — recorded
  as evidence, not as the pass/fail gate.
- **Completion criterion:** the four `ruff`/`ty`/`pytest` commands exit 0
  with no new failures and no regressions vs. the last accepted iteration,
  AND `tests/test_viewc3d_multifile.py` asserts (computed by hand, not
  derived from the implementation):
  - `resolve_master_fps([200.0, 240.0], False) == 240.0`;
    `resolve_master_fps([200.0, 240.0], True) == 200.0`.
  - `master_frame_count([8074/200.0, 8298/240.0], 240.0) == 9689`
    (durations 40.37 s / 34.575 s, max 40.37 s × 240 Hz = 9688.8 → 9689).
  - `master_frame_count([8074/200.0, 8298/240.0], 200.0) == 8074`.
  - `map_master_frame_to_local(9688, 240.0, 200.0, 8074) == 8073` (last
    master frame maps to the 200 Hz file's own last frame, no clamping
    needed).
  - `map_master_frame_to_local(8073, 200.0, 240.0, 8298) == 8297` (clamped:
    raw `round(8073*240/200) = 9688` exceeds `8297`, so the shorter 240 Hz
    recording holds its final frame).
  - 12 sequential color assignments cycle `available_colors` exactly once
    and repeat identically starting at the 12th.
  AND `Ctrl+L`, the color legend, and the sync behavior are present in the
  docstring, `H`-key help text, and `vaila/help/viewc3d.md`, AND metadata
  items in Goal §10 are updated.
- **Verifier protection:** the governing commands and
  `tests/test_viewc3d_multifile.py`'s hand-computed constants are not edited
  by the same iteration that touches `vaila/viewc3d.py`'s implementation —
  if a hand-computed value needs correcting, that is its own attributable
  iteration with the arithmetic shown, not a silent edit bundled with a
  feature change.
- **Scientific validity:** frame data stays in the C3D file's own coordinate
  frame and units (already normalized to meters by `detect_c3d_units` before
  this loop runs) — no coordinate transform between files is introduced;
  multi-file overlay is purely a visualization convenience, not a spatial
  registration/alignment claim between the two recordings. FPS/frame-count
  values are read from `GetPointFrequency()`/`GetFrameNumber()` per file,
  never assumed equal. The frame-mapping formula is a nearest-index playback
  lookup only — it does not resample, interpolate, or mutate the underlying
  marker position arrays, so no analysis derived from the raw per-file data
  is affected by this feature.

## Trigger
Manual — invoked directly in this dev session (already on the `multiview`
branch). Not scheduled or event-driven. Duplicate-run protection: check
`loops/state/viewc3d-multifile-loop-state.json` for an `in_progress` status
with a fresh-enough timestamp before starting a second run.

## Iteration
0. On the first iteration: confirm `git status --porcelain` is clean on the
   current `multiview` branch (already checked out — no new branch needed);
   confirm `tests/viewc3d/rec3d_200hz.c3d` and `rec3d_240hz.c3d` exist and
   are readable; freeze the design decisions in Goal §1, §3, §5–6.
1. Load this spec and `loops/state/viewc3d-multifile-loop-state.json`;
   confirm the 15-iteration budget and approval gates below are intact.
2. Snapshot `git diff` as the baseline; run the governing check; record
   evidence.
3. Rank unresolved Goal items worst-first by blocking order: (1) pure
   timeline/color helpers + their hand-computed tests — nothing else is
   testable without them; (2) `LoadedC3D` container + `loaded_files` list;
   (3) `Ctrl+L` load action wired to the multi-select dialog; (4)
   `update_spheres` extended to iterate `loaded_files`; (5) FPS-mismatch
   downsample prompt wired to recompute the master timeline; (6) window
   title/help/docstring/help-doc updates; (7) metadata bump.
4. Make exactly one attributable change addressing the current top-ranked
   item, guided by the existing in-file conventions (`ctrl_s_key_action` for
   the key-action pattern, `_create_centered_tk_root`/`select_markers` for
   dialogs, `update_spheres`/`available_colors` for rendering).
5. Run the governing check; record raw evidence verbatim.
6. Keep the change only if the check passes with no regression vs. the last
   accepted state; otherwise `git checkout -- vaila/viewc3d.py
   tests/test_viewc3d_multifile.py` (and any touched help/doc files) to roll
   back just this iteration, then record why.
7. Curate lessons (only ones borne out by evidence); atomically persist
   state, evidence, counters, and cost.
8. Evaluate terminal states; otherwise begin the next iteration.

## Terminal States
- **success:** every completion-criterion clause above is met, evidence for
  all of them is present in the state file, and a manual two-file run
  transcript is recorded as level-5 evidence.
- **no-op:** re-running the loop finds every Goal item already satisfied by
  evidence already recorded in state (e.g., resuming a completed run).
- **no-progress/stalled:** two consecutive iterations produce an identical
  pass/fail signature with no Goal item advanced — stop and surface the
  exact failing evidence for human review.
- **blocked:** the fixture files become unreadable/moved, the `ruff`/`ty`/
  `pytest` toolchain is unavailable, or `open3d`/`ezc3d` cannot be imported
  in this environment (viewer cannot be exercised at all, even for the
  helper-function tests that don't need a window but do need the module to
  import).
- **exhausted:** 15 iterations reached without full success.

Errors, timeouts, missing evidence, and budget exhaustion are never success.

## Guardrails
- **Maximum allocation:** 15 iterations (one attributable change + governing
  check each). No dollar/token ceiling requested.
- **Human approval required:** any `git commit`/`git push`; deleting any
  file; modifying either fixture `.c3d` file in `tests/viewc3d/` (read-only
  inputs for this loop).
- **Isolation and credentials:** runs in the existing repo working tree on
  the already-checked-out `multiview` branch; no network access needed; no
  credentials involved.
- **Protected verifier:** the four governing commands and the hand-computed
  constants in `tests/test_viewc3d_multifile.py` are not edited by the same
  iteration that changes `vaila/viewc3d.py` (see Verifier protection above).
- **Rollback:** `git checkout -- <touched files>` scoped to the current
  iteration's changes only; the `multiview` branch itself is never
  force-pushed or rebased destructively without explicit approval.

## State Memory
- **Path:** `loops/state/viewc3d-multifile-loop-state.json`.
- **Persist:** baseline `git diff` hash, terminal status, per-iteration
  attempts (what changed, which Goal item it targeted), accepted/rejected
  changes with reasons, raw governing-check evidence, curated lessons,
  human-approval decisions, and iteration/cost counters.
- **Recovery:** a fresh context re-reads this loop document and the state
  file before acting; an `in_progress` status with no matching
  iteration-completion record for the last logged attempt means that write
  was interrupted — treat the last *fully recorded* iteration as the resume
  point and re-run its governing check before proceeding.

## Skills
- `$check` — the repo's ruff+ty+pytest pipeline convention; run as the
  governing check's first three commands each iteration.
- `$gui-developer` — the `Ctrl+L` key-action wiring, Tk multi-select dialog,
  and keeping the single-root Tk model intact (reusing
  `_create_centered_tk_root`, no second blocking root).
- `$test-writer` — `tests/test_viewc3d_multifile.py`'s hand-computed
  fixture-driven assertions (red-before on the new helper functions).

## Why It Works
- Extending `update_spheres`/`VailaModel` instead of forking a parallel
  multi-file code path keeps the existing single-file behavior and its
  implicit test coverage (manual smoke use) byte-for-byte unchanged when
  `Ctrl+L` is never pressed.
- Isolating the FPS/master-timeline/color-assignment logic into pure
  functions is what makes item 5–6 testable at all in a repo with no
  headless-Open3D harness — without that extraction, the only evidence
  available would be level-5 (a human watching the window), which the
  skill's ladder disfavors for unattended acceptance.
- Freezing the `Ctrl+L` key choice up front (plain `L` was already taken —
  every other letter is too) prevents rediscovering the same collision
  mid-implementation.
- Nearest-index frame mapping (no interpolation of marker positions) keeps
  the feature a pure visualization convenience and avoids silently
  fabricating biomechanical data for the lower/higher-FPS file — a
  correctness line the skill's scientific-validity gate specifically
  guards.
- Per-iteration git-scoped rollback keeps this large, monolithic-file change
  cheaply reversible without needing a second branch (already isolated on
  `multiview`).

## How to Trigger
### Context-bound
Within this Claude Code session with the repo open: read this file, then
execute Iteration steps 0–8 directly, keeping evidence in this turn's
context and in the state file.

### Fresh-context / Ralph
For a longer run spanning multiple sessions: each fresh context must
re-read this loop document and
`loops/state/viewc3d-multifile-loop-state.json` in full before acting,
resume from the last fully recorded iteration, and stop only on a named
terminal state.

## Health Metrics
- **Cost per accepted change:** total tokens spent / count of iterations
  whose change was retained.
- **Goal items closed / 10** — from the Goal enumeration, tracked per
  iteration.
- **Regression count** — iterations rolled back because the governing check
  regressed vs. the last accepted state.
</content>
