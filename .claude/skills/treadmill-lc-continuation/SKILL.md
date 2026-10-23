---
name: treadmill-lc-continuation
description: Resume Treadmill LC (load-cell GRF) work after the 2026-08-17 English translation. Use when the user reopens treadmill_lc, asks where that work stopped, wants Antigravity/Cursor to continue coding, or mentions COP mediolateral/anteroposterior labels, tare/weight calibration files, or tests/treadmill_lc.
---

# Treadmill LC Continuation

Read first:

1. `docs/sessions/2026-08-17-treadmill-lc-english.md` — what changed and why
2. `docs/treadmill_lc_integration_handoff.md` — pipeline contract
3. `vaila/help/treadmill_lc.md` — user-facing behavior
4. `AGENTS.md` — metadata, `uv`/`ruff`/`ty`, Tkinter-only

Antigravity also loads `.agent/rules/treadmill-lc-continue.md`.

## Current baseline (validated 2026-08-17)

- Branch: `iatm`
- Module: `vaila/treadmill_lc.py` v0.3.107, Update Date 17 August 2026
- GUI: Frame B → **Treadmill LC** (`vaila.py` `B6_r7_c3`)
- Tests: `uv run pytest tests/test_treadmill_lc.py -v` (39 passed)
- Full suite that day: `910 passed, 11 skipped`

User-facing language is **English**. Canonical calibration names are `tare` and `weight`. COP labels are **Mediolateral** (X, horizontal) and **Anteroposterior** (Y, vertical).

## Hard rules

- Keep Tkinter. Interactive matplotlib = `TkAgg`; tests/CLI savefig = `Agg`. Never Qt.
- Keep Portuguese **input** aliases: `tara`, `peso`, `limpo`/`LIMPO`, Borg `Peso`, modes `nulo`/`média`/`cortar`.
- Do not rename COP geometry: 58 cm ML × 113 cm AP, cells 1 AL / 2 PL / 3 AR / 4 PR.
- Do not commit `tests/treadmill_lc/{clean,filtered,filter_analysis,results,figures}_*` unless the user asks.
- Do not commit unless the user asks.
- On any `*.py` edit: Update Date = today, Version = global from `vaila.py`, plus help/README per `AGENTS.md`.

## Verify after edits

```bash
uv run ruff check vaila/treadmill_lc.py tests/test_treadmill_lc.py --fix
uv run ruff format vaila/treadmill_lc.py tests/test_treadmill_lc.py
uv run pytest tests/test_treadmill_lc.py -v
```

CLI smoke (headless process needs a display for adjust/window clicks):

```bash
uv run python -m vaila.treadmill_lc --input-dir tests/treadmill_lc --step filter
```

## Stop

Stop when the requested treadmill LC change is done, tests above pass, and help matches the code. Do not start FIFA or SAM work from this skill.
