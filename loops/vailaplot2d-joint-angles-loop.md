---
name: vailaplot2d-joint-angles-loop
category: Biomechanics
trigger: manual
verification-level: 1
theory-base: arXiv:2607.00038
---

# VailáPlot2D Joint-Angles Time Series Loop

## Description
Add a **Joint Angles** plot type and GUI button inside `vaila/vailaplot2d.py` that
loads a long-format `*_joint_angles.csv`, lets the user pick `person_id` and
`joint_name` (e.g. `left-knee`), and plots time on X with the three Euler
components (`euler_x_deg`, `euler_y_deg`, `euler_z_deg`) on Y — displayed as
flexion/extension, abduction/adduction, and internal/external rotation aliases.

## Use When
- Adding or repairing time-series joint-angle visualization for REC3D /
  SAM3D / Sapiens3D `*_joint_angles.csv` exports inside Plot 2D.
- Not for: computing new angles (that is `joint_kinematics` / rec3d exporters);
  not for 3D mesh viewers; not for wide-format angle CSVs without this schema.

## Inputs
1. `module` — `vaila/vailaplot2d.py` (GUI plot-type buttons + plot dispatch).
2. `schema` — long table columns (required):
   `frame,person_id,joint_idx,joint_name,parent_idx,euler_x_deg,euler_y_deg,euler_z_deg`
   (quaternion columns optional/ignored for this plot).
3. `reference_csv` (external, read-only QA) —
   `/home/preto/data/sep_runcod_01072026/REC3D_COD/jessica/vaila_rec3d_out/vaila_rec3d_20260826_121305/rec3d_20260826_121305_joint_angles.csv`
   — 631 frames × multi-person; named joints include `left-knee`, `right-knee`,
   hips, ankles, etc. **Do not commit** this file; use only for manual/CLI smoke
   when present.
4. `test_fixture` — synthetic long-format CSV under pytest `tmp_path` (mandatory
   for CI).
5. `help` — `vaila/help/vailaplot2d.md` + `.html` (+ index) when the button ships.
6. `hardware` — CPU only (matplotlib + pandas).

## Goal
Objectively verifiable end state:

1. Plot 2D GUI has a **Joint Angles** button (new plot type id
   `joint_angles_time`).
2. User selects one or more `*_joint_angles.csv` files (or the button opens a
   file dialog if the existing multi-file flow does not fit).
3. UI prompts (or listboxes) for **person_id** and **joint_name** present in the
   loaded table; default joint preference order includes `left-knee` /
   `right-knee` when available.
4. Plot: X = time (`frame` index by default; if a positive sample rate `fs` is
   provided, X = `frame / fs` in seconds), Y = three series:
   - `euler_x_deg` legend **Flexion/Extension**
   - `euler_y_deg` legend **Abduction/Adduction**
   - `euler_z_deg` legend **Internal/External Rotation**
5. Units on Y: **degrees**. Axis labels and title name the joint and person.
6. Headless helper (e.g. `plot_joint_angles_time(df, person_id, joint_name, fs=None)`)
   callable from tests / CLI without creating `tk.Tk()`.
7. Existing plot types still work; mandatory metadata/help updated.

**Logged interview defaults (2026-08-26):**
- Loop name `vailaplot2d-joint-angles-loop`; save under `./loops/`.
- Anatomical legend aliases map 1:1 to XYZ Cardan columns from the exporter;
  help must state that true clinical axis meaning follows
  `vaila/joint_kinematics.py` / the producing pipeline — do not invent a new
  rotation convention in the plotter.
- Multi-person: filter by selected `person_id` (example file has ids 3, 4, 8).
- No launcher artifact unless requested later.

## Verification (Governing Check)
- **True level:** 1 (pytest on pure plot-data helpers + optional Agg smoke) with
  level-2 ruff/ty. “Looks like a knee” from a screenshot is **not** acceptance
  (level 5 only as optional human QA on the external reference CSV).
- **Check (every iteration):**
  ```bash
  uv run ruff check vaila/vailaplot2d.py tests/test_vailaplot2d_joint_angles.py --fix
  uv run ruff format vaila/vailaplot2d.py tests/test_vailaplot2d_joint_angles.py
  uv run ty check vaila/vailaplot2d.py tests/test_vailaplot2d_joint_angles.py
  uv run pytest tests/test_vailaplot2d_joint_angles.py -v
  ```
  Create `tests/test_vailaplot2d_joint_angles.py` on the first test iteration.
  Optional when the external file exists:
  ```bash
  uv run python -c "from pathlib import Path; import pandas as pd; from vaila.vailaplot2d import load_joint_angles_series; p=Path('/home/preto/data/sep_runcod_01072026/REC3D_COD/jessica/vaila_rec3d_out/vaila_rec3d_20260826_121305/rec3d_20260826_121305_joint_angles.csv'); df=pd.read_csv(p); t,y=load_joint_angles_series(df, person_id=8, joint_name='left-knee'); assert len(t)==len(y['euler_x_deg'])>0"
  ```
- **Evidence:** raw stdout/stderr of the check commands in
  `loops/state/vailaplot2d-joint-angles-loop-state.json`.
- **Completion criterion:**
  - Button + dispatch wired; headless loader/plot helper covered by tests.
  - Synthetic fixture: known sine on `euler_x_deg` recovers length and values.
  - Missing joint/person raises a clear error (no silent empty plot).
  - Ruff/ty clean on touched files; header Version/Update Date + help/index
    updated; stale metadata with green tests = incomplete.
- **Verifier protection:**
  - Do not weaken other vailaplot2d behaviors or delete unrelated tests.
  - Do not hard-code the external jessica path into runtime code (tests may
    skip if absent).
  - Prefer red-before/green-after for the new test file.
- **Scientific validity:**
  - Y units: degrees (`*_deg` columns).
  - X: frames (dimensionless count) or seconds when `fs` given; label axis
    accordingly (`Frame` vs `Time (s)`).
  - Filter `person_id` + `joint_name` before plotting; sort by `frame`.
  - Legend aliases Flex/Ext, Abd/Add, Int/Ext are **display labels** for
    `euler_x/y/z_deg`; document Cardan XYZ provenance — clinical sign/axis
    meaning is owned by the exporter, not re-derived in the plotter.
  - NaN frames: leave gaps (matplotlib default) or drop — pick one and test it;
    never replace with zero without marking invalid.

## Trigger
Manual. Re-entry loads state JSON and skips accepted targets.

## Iteration
0. Freeze schema against the reference CSV header; snapshot baseline.
1. Load this spec + state; confirm budget/approvals.
2. Run governing check; record evidence.
3. Rank open targets worst-first (data helper → plot → button → help).
4. One attributable change via `$surgical-patch` (or `$investigate-first`).
5. Re-run check; record evidence.
6. Retain only if non-regressive; else rollback.
7. Persist state; evaluate terminal states.

## Terminal States
- **success:** All Goal items 1–7 accepted with evidence.
- **no-op:** Feature already present with tests covering schema + button id.
- **no-progress/stalled:** Two consecutive iterations without an accepted target.
- **blocked:** matplotlib/Tk environment cannot run Agg tests; or schema of
  producer CSVs diverges and needs a separate exporter fix first.
- **exhausted:** Hard budget reached.

Errors, missing evidence, and budget exhaustion are never success.

## Guardrails
- **Maximum allocation:** 10 iterations; 35 agent tool-turns; local `uv` only.
- **Human approval required:** git commit/push; committing large external CSVs;
  changing `joint_kinematics` conventions.
- **Isolation:** headless tests use `matplotlib` Agg; no second `tk.Tk()` in
  helpers; no network.
- **Protected verifier:** frozen check commands; synthetic fixture ownership in
  tests only.
- **Rollback:** `git checkout -- <touched paths>` for that iteration.

## State Memory
- **Path:** `loops/state/vailaplot2d-joint-angles-loop-state.json`
- **Persist:** baseline, targets, attempt evidence, lessons, cost.
- **Recovery:** re-read this markdown + state JSON; rebuild targets from Goal
  if JSON corrupt.

## Skills
- `$investigate-first` — schema/GUI wiring failures.
- `$surgical-patch` — one button/helper/test change per iteration.
- `$verify-and-stop` — stop at Goal; refuse angle-math scope creep.
- `$preto-loop` — maintain this document; do not execute unless the operator
  explicitly starts the loop.

## Sub-Loops
None. If exporter Euler convention is wrong, open a separate kinematics loop —
do not nest it here.

## Why It Works
- Deterministic series extraction tests catch wrong joint/person filters.
- Separates visualization acceptance from biomechanical correctness of angles.
- Button lives in the existing Plot 2D surface users already open.
- External jessica CSV is optional QA, not a CI dependency.

## How to Trigger
### Context-bound
Instruct the agent to execute `vailaplot2d-joint-angles-loop` until a named
terminal state, re-reading the state JSON each turn.

### Fresh-context / Ralph
```text
1. Read loops/vailaplot2d-joint-angles-loop.md
2. Read loops/state/vailaplot2d-joint-angles-loop-state.json
3. One Iteration (exactly one change)
4. Update state atomically
5. Stop only on success | no-op | no-progress/stalled | blocked | exhausted
```

## Health Metrics
- **Cost per accepted change:** `iterations_or_turns / accepted_targets`.
- **Coverage progress:** accepted Goal items / 7.
- **Regression:** existing Plot 2D types still dispatch without error on smoke.
