# Continue Treadmill LC in Antigravity

Load this when the user asks to continue treadmill load-cell work, edit `treadmill_lc`, fix Treadmill LC tests, or pick up the 17 August 2026 English-translation session.

## Read before editing

1. `docs/sessions/2026-08-17-treadmill-lc-english.md`
2. `.claude/skills/treadmill-lc-continuation/SKILL.md`
3. `docs/treadmill_lc_integration_handoff.md`
4. `AGENTS.md` (uv, ruff, ty, Tkinter-only, metadata checklist)

Do not re-translate `vaila/treadmill_lc.py`. User-facing strings are already English.

## State to preserve

- Branch `iatm`. Work was uncommitted when the Cursor session ended.
- Canonical files: `s*_d*_tare.csv`, `s*_d*_weight.csv`, `s*_d*_t*.csv`.
- COP X label: `COP X - Mediolateral (cm)` (horizontal).
- COP Y label: `COP Y - Anteroposterior (cm)` (vertical).
- Cells: 1 anterior-left, 2 posterior-left, 3 anterior-right, 4 posterior-right.
- Legacy Portuguese inputs still accepted: `tara`, `peso`, `LIMPO`, header `Peso`.
- Tests lock matplotlib `Agg` before importing the module.

## Do next (only if the user asks)

- Further treadmill LC bugs, GUI, or docs on this English baseline.
- Commit on `iatm` only when requested. Stage treadmill LC sources/tests/help; exclude generated `tests/treadmill_lc/clean_*` (and sibling timestamp folders).

## Verify

```bash
uv run pytest tests/test_treadmill_lc.py -v
```

Full suite last green: 910 passed, 11 skipped (2026-08-17).
