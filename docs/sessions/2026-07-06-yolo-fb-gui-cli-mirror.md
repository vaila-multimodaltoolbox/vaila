# Session: YOLO + FB GUI→CLI mirror (2026-07-06, v0.3.72)

## Goal

When users click GUI buttons under **Frame B → YOLO + FB**, the terminal must print
copy-paste CLI commands so they can replay the same work headless later (Cursor CLI,
SSH, batch scripts).

## What shipped

### Chooser (`vaila.py`)

- `_print_yolo_fb_launch()` on all 7 chooser buttons
- Banners in `sam_video()` / `sapiens_video()` include `>> Equivalent launch CLI`

### Run mirrors

| Module | Change |
|--------|--------|
| `vaila_sapiens.py` | `_format_sapiens_cli_command` + print on GUI Run |
| `vaila_sam.py` | `_build_sam_cli_argv` + `_print_sam_equivalent_cli` on batch Run |
| `yolov26track.py` | `_format_track_cli_command` — one `track` cmd per video before loop |
| `yolov26track.py` | Pose (video) / Pose (tracking) workflow hints |
| `yolotrain.py` | Launch line on GUI open + training cmd in `_run_training` thread |

### Docs

- Helps rebranded: **Video AI tools** → **YOLO + FB**
- New `docs/vaila_buttons/yolo-fb.md`
- `AGENTS.md` / `CLAUDE.md` convention + history bullets

### Tests

- `test_format_sapiens_cli_command_quotes_paths`
- `test_build_sam_cli_argv_includes_prompt`
- `test_format_track_cli_command_maps_config`

## Pattern for future agents

- Prefix `>>` (absl eats `[bracketed]` stdout)
- Chooser = launcher CLI only; Run = full args
- Reference: `.claude/skills/yolo-fb-gui-cli/SKILL.md`

## Related prior work

- v0.3.71: YOLO + SAM → **YOLO + FB**, Sapiens2 Pose module + `bin/setup_sapiens2.sh`
- v0.3.69: SAM default postprocess `all` + VAILA anchor CSVs
- getpixelvideo / yolotrain already had partial CLI mirrors
