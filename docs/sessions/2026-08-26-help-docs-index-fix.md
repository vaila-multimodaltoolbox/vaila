# Session — Fix main help/docs indexes (2026-08-26)

**Version:** v0.3.115  
**Date:** 26 August 2026  
**Status:** Done — generator verified with `uv run python bin/generate_help_index.py`

## Problem

Main help/docs entry pages look wrong on GitHub and local:

- [`vaila/help/index.md`](../../vaila/help/index.md) start with **duplicate** Markerless 2D/3D chooser blocks, look like "only YOLO/SAM scripts," even though ~153 help pages exist.
- [`vaila/help/index.html`](../../vaila/help/index.html) was **GUI Frame A/B/C button map**, not same full script catalog as Markdown index.
- [`docs/index.md`](../index.md) / [`docs/help.html`](../help.html) / [`docs/help.md`](../help.md) over-feature FIFA / VEK / DLT instead of "project idea + path to every script."
- Stale counts (`140` modules), junk `__init__` entry, missing topics (`syncvid`, `viewc3d_pyvista`, `codec_benchmark`, `gpu_guide`, …).

## Goal (accepted)

1. Short *vailá* project idea (Frames A/B/C).
2. Full categorized list, every help topic, HTML + Markdown links.
3. Docs hub slim: idea → help index CTA → short Guides list.

## What shipped

### Generator

[`bin/generate_help_index.py`](../../bin/generate_help_index.py)

```bash
uv run python bin/generate_help_index.py
```

- Scan `vaila/help/*.md` (skip `index.md`, `README.md`, `__*`).
- Normalize Category metadata → Analysis / ML / Processing / Tools / Utils / Visualization / Guides.
- Pull one-liners from Description/Overview (skip `====` banners, Author/Date noise, "Generated automatically").
- Write both `vaila/help/index.md` and `index.html` (search box kept on HTML).

### Last successful run

```text
Topics: 153 | version 0.3.115 | 26/08/2026
  Analysis: 27
  ML: 19
  Processing: 21
  Tools: 54
  Utils: 19
  Visualization: 7
  Guides: 6
```

Checks: 0× "Markerless 2D chooser" dup, 0× `__init__`, 153 HTML `.tool-card`s, search input present. Spot links OK: `imu_analysis`, `vaila_sam`, `syncvid`, `gpu_guide`, `viewc3d_pyvista`.

### Docs hub

- [`docs/index.md`](../index.md), [`docs/help.md`](../help.md), [`docs/help.html`](../help.html) — project idea + primary link to `vaila/help/index.html` + demoted Guides (FIFA, VEK, DLT, buttons, PDF, GPU).
- [`vaila/help/README.md`](../../vaila/help/README.md) — points at generator; hand-curated partial script list gone; notes flat `vaila/help/` layout.

## Resume instructions (Claude / Codex / Cursor / CLI)

1. **No hand-edit** `vaila/help/index.md` or `index.html` — regen instead:

```bash
uv run python bin/generate_help_index.py
```

2. After adding new `vaila/help/<module>.md` (+ `.html`), set `**Category:**` in Module Information, write real Description paragraph, then run generator.
3. Project overview for users: `docs/index.md` / `docs/help.html` → catalog: `vaila/help/index.html`.
4. Global version still **0.3.115** / **26.Aug.2026** in `vaila.py` (this change docs/tooling only; no bump needed unless you touch analysis `*.py`).

## Out of scope (still open, next if wanted)

- Rewrite individual module help bodies.
- Make missing HTML for MD-only pages (e.g. `viewc3d_pyvista`, `BRAINSTORM_GUIDE`, `help_reid_markers`).
- Wire generator into CI / pre-commit.

## Files touched

| Path | Role |
|------|------|
| `bin/generate_help_index.py` | New generator |
| `vaila/help/index.md` | Regenerated catalog |
| `vaila/help/index.html` | Regenerated catalog + search |
| `vaila/help/README.md` | Generator docs |
| `docs/index.md` | Slim hub |
| `docs/help.md` | Slim hub |
| `docs/help.html` | Slim hub |

## How to cite this handoff in another IDE

```text
Read docs/sessions/2026-08-26-help-docs-index-fix.md then continue from Resume instructions.
```