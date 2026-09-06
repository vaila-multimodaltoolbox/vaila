---
name: getpixelvideo-qm-guide-hotkeys
category: Vailá
trigger: manual
verification-level: 1
theory-base: arXiv:2607.00038
---

# getpixelvideo Quick Measure button, Kiki guide, hotkeys help

## Description
Add a visible toolbar button equivalent to hotkey `Q` (Quick Measure), make soccerfield Guide load `vaila/models/soccerfield_kiki.csv` for non-FIFA-dataset guide paths, and add a clickable hotkeys submenu in `vaila/help/getpixelvideo.html` while keeping the existing long-form controls content.

## Use When
- Finishing or regressing the Measure button, Guide→kiki wiring, or hotkeys cheat-sheet in getpixelvideo help.
- **Exclusions:** does not re-implement `quickmeasure.py` math (see `getpixelvideo-quickmeasure-loop.md`); does not replace FIFA dataset 32/48 labeling CSV when `prefer_fifa_dataset=True`.

## Inputs
1. `vaila/models/soccerfield_kiki.csv` — must exist (49 points); Guide non-FIFA-dataset path prefers it.
2. Running `getpixelvideo` session (manual smoke) — optional for UI click proof.

## Goal
1. Toolbar **QMeas** (or equivalent) button toggles the same state as `pygame.K_q`.
2. `load_pitch_guide_points(prefer_fifa_dataset=False)` resolves to `soccerfield_kiki.csv` when present.
3. Guide (G / Guide button) for FIFA template uses the kiki path (not legacy/FIFA-dataset) for field hints.
4. HTML help keeps current controls tables; adds a click-to-expand organized hotkeys listing.
5. Metadata (header date/version, help Version/Updated, index Generated on) synced to global vailá version.

## Verification (Governing Check)
- **True level:** 1 (deterministic tests + static HTML presence) with level-5 optional pygame smoke.
- **Check:**
  ```bash
  uv run pytest tests/test_soccerfield_fifa_expansion.py tests/test_soccerfield_kiki_and_calib.py tests/test_getpixelvideo_media_classify.py -v
  uv run ruff check vaila/getpixelvideo.py --fix
  uv run ruff format --check vaila/getpixelvideo.py
  uv run ty check vaila/getpixelvideo.py
  rg -n "soccerfield_kiki|QMeas|hotkeys-quick|All short keys" vaila/getpixelvideo.py vaila/help/getpixelvideo.html
  ```
- **Evidence:** raw pytest/ruff/ty stdout; `rg` hits proving kiki path, button label, and HTML `<details id="hotkeys-quick">` (or equivalent).
- **Completion criterion:** tests pass; ruff/ty clean on touched py; HTML submenu present and existing `#controls` content retained; Guide load source contains `soccerfield_kiki.csv` when `prefer_fifa_dataset=False`.
- **Verifier protection:** do not edit expected point counts in `tests/test_soccerfield_kiki_and_calib.py` or FIFA-dataset assertions for `prefer_fifa_dataset=True` without red/green justification.
- **Scientific validity:** kiki units metres (FIFA-style centre origin); guide remains visual hint — marking still uses template slots; FIFA dataset export path unchanged when `prefer_fifa_dataset=True`.

## Trigger
Manual (`/goal`, `/preto-loop`, or developer). Duplicate-run: read `loops/state/getpixelvideo-qm-guide-hotkeys-loop-state.json` if present.

## Iteration
0. Freeze defaults: Guide non-FIFA-dataset → kiki; FIFA labeling `prefer_fifa_dataset=True` → `soccerfield_ref3d_fifa_dataset.csv`.
1. Load this spec + state; confirm budget.
2. Snapshot `git status --porcelain`; run governing check.
3. Pick worst missing requirement; one change.
4. Skills: `$gui-developer` (button), `$surgical-patch` (load path), help HTML edit.
5. Re-run check; record evidence.
6. Retain only if non-regressive; else rollback that iteration’s files.
7. Persist state; evaluate terminals.

## Terminal States
- **success:** all goal items evidenced by check output.
- **no-op:** already implemented and check green with empty diff.
- **no-progress/stalled:** two consecutive iterations fail the same assertion.
- **blocked:** `soccerfield_kiki.csv` missing from tree.
- **exhausted:** 24 turns / 4 attempts hard ceiling.

Errors and missing evidence are never success.

## Guardrails
- **Maximum allocation:** 24 turns, 4 attempts.
- **Human approval required:** remapping existing hotkeys; git commit/push; deleting fixtures.
- **Rollback:** `git checkout -- <files from this iteration>`.
- **Protected verifier:** FIFA-dataset `prefer_fifa_dataset=True` test expectations; kiki 49-point schema tests.

## State Memory
- **Path:** `loops/state/getpixelvideo-qm-guide-hotkeys-loop-state.json`
- **Persist:** attempts, evidence excerpts, accepted/rejected changes, cost.
- **Recovery:** re-run governing check before new edits.

## Skills
- `$gui-developer` — pygame toolbar button + click handler.
- `$surgical-patch` — narrow `load_pitch_guide_points` / Guide wiring.
- `$verify-and-stop` — stop when check proves completion.

## Why It Works
Deterministic CSV-source assertions and HTML string presence turn UI/docs work into level-1 checks; FIFA-dataset path stays gated so labeling export does not silently switch to 49-point kiki.

## How to Trigger
### Context-bound
Read this file; execute iterations until a terminal state.
### Fresh-context / Ralph
Re-read this file + state each turn; stop only on named terminal states.

## Health Metrics
- **Cost per accepted change:** turns / retained non-regressive changes.
- **Check pass:** boolean from governing command.
