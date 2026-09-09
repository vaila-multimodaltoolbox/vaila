---
name: treadmill-lc-continuation
description: Resume Treadmill LC (load-cell GRF) work after the 2026-08-17 English translation. Use when the user reopens treadmill_lc, asks where that work stopped, wants Antigravity/Cursor to continue coding, or mentions COP mediolateral/anteroposterior labels, tare/weight calibration files, or tests/treadmill_lc.
---
# Treadmill LC Continuation

Read first:

1. `docs/sessions/2026-08-17-treadmill-lc-english.md` — what change, why
2. `docs/treadmill_lc_integration_handoff.md` — pipeline contract
3. `vaila/help/treadmill_lc.md` — user-facing behavior
4. `AGENTS.md` — metadata, `uv`/`ruff`/`ty`, Tkinter-only

Antigravity also load `.agent/rules/treadmill-lc-continue.md`.

## Baseline (validated 2026-08-17)

- Branch: `iatm`
- Module: `vaila/treadmill_lc.py` v0.3.107, Update Date 17 August 2026
- GUI: Frame B → **Treadmill LC** (`vaila.py` `B6_r7_c3`)
- Tests: `uv run pytest tests/test_treadmill_lc.py -v` (39 passed)
- Full suite that day: `910 passed, 11 skipped`

User-facing lang **English**. Canonical calibration names: `tare`, `weight`. COP labels: **Mediolateral** (X, horizontal), **Anteroposterior** (Y, vertical).

## Hard rules

- Keep Tkinter. Interactive matplotlib = `TkAgg`; tests/CLI savefig = `Agg`. Never Qt.
- Keep Portuguese **input** aliases: `tara`, `peso`, `limpo`/`LIMPO`, Borg `Peso`, modes `nulo`/`média`/`cortar`.
- Don't rename COP geometry: 58 cm ML × 113 cm AP, cells 1 AL / 2 PL / 3 AR / 4 PR.
- Don't commit `tests/treadmill_lc/{clean,filtered,filter_analysis,results,figures}_*` unless user ask.
- Don't commit unless user ask.
- Any `*.py` edit: Update Date = today, Version = global from `vaila.py`, plus help/README per `AGENTS.md`.

## Verify after edits

```bash
uv run ruff check vaila/treadmill_lc.py tests/test_treadmill_lc.py --fix
uv run ruff format vaila/treadmill_lc.py tests/test_treadmill_lc.py
uv run pytest tests/test_treadmill_lc.py -v
```

CLI smoke (headless process need display for adjust/window clicks):

```bash
uv run python -m vaila.treadmill_lc --input-dir tests/treadmill_lc --step filter
```

## Stop

Stop when requested treadmill LC change done, tests above pass, help match code. Don't start FIFA, SAM, caveman work from this skill.