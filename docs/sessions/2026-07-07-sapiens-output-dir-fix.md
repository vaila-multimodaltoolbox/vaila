# Session — Sapiens2 duplicate output directory fix (2026-07-07)

**Version:** v0.3.76  
**Module:** `vaila/vaila_sapiens.py`  
**Reporter:** user (CLI + GUI)

## Symptom

Every Sapiens2 run (CLI or GUI) produced **two** folders under the output parent:

```text
out/processed_sapiens_20260707_120001/   ← empty
out/processed_sapiens_20260707_120003/   ← real CSVs + overlay under <stem>/
```

## Root cause

Default **subprocess-per-video isolation** (same pattern as SAM 3 batch) spawns one Python child per video via `_build_isolated_sapiens_cmd()`.

Each child invoked `main()` with `--video-output-dir` pointing at the parent's `<timestamp>/<stem>/`, but **without** `--output-base`. `main()` always created a fresh `processed_sapiens_<new_timestamp>/` before the early return on `--video-output-dir` — hence the empty sibling directory.

GUI batch had the same leak: parent passed `--output-base` to the batch subprocess, but isolated per-video workers did not.

## Fix

1. **`main()`** — when `--video-output-dir` is set, process the single video and `return` **before** any `processed_sapiens_*` mkdir.
2. **`_build_isolated_sapiens_cmd()`** — add `output_base` parameter; pass `--output-base` to each isolated worker so parent and child agree on the batch folder.

## Files changed

| File | Change |
|------|--------|
| `vaila/vaila_sapiens.py` | Reorder `main()`; `--output-base` in isolated cmd |
| `tests/test_vaila_sapiens.py` | `test_build_isolated_sapiens_cmd_passes_output_base` |
| `vaila/help/vaila_sapiens.{md,html}` | § Output directory (v0.3.76) |
| `vaila.py` | Global version/date bump |
| `AGENTS.md`, `CLAUDE.md`, skills | History + resume notes |

## Expected layout after fix

```text
<output_parent>/
  processed_sapiens_YYYYMMDD_HHMMSS/
    my_video/
      my_video_markers.csv
      my_video_sapiens_overlay.mp4
      sapiens_bbox_tracks.csv
      README_sapiens.txt
      ...
```

## QA

```bash
uv run pytest tests/test_vaila_sapiens.py -v
uv run vaila/vaila_sapiens.py -i tests/markerless_2d_analysis/ -o /tmp/sapiens_qa --model 1b --dry-run
# Full GPU run: confirm only one processed_sapiens_* under -o
```

## Resume pointers

- Skill: `.claude/skills/yolo-fb-gui-cli/SKILL.md` § Sapiens2 Pose
- Help: `vaila/help/vaila_sapiens.md`
- Related SAM pattern: `.claude/skills/sam3-video/SKILL.md` § subprocess-per-video
