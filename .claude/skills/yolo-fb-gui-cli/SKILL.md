---
name: yolo-fb-gui-cli
description: YOLO + FB chooser (Frame B), Sapiens2 Pose, and GUI→CLI terminal mirror for vailá video-AI tools. Use when reopening Cursor CLI, wiring GUI buttons, printing copy-paste CLI from Tkinter runs, or continuing work on vaila.py yolotrackerpose / vaila_sam / vaila_sapiens / yolov26track / yolotrain.
---

# YOLO + FB Chooser & GUI→CLI Mirror (v0.3.91)

Use when the user works on **Frame B → YOLO + FB**, wants **terminal commands** equivalent to GUI clicks, or resumes after closing the terminal in **Cursor CLI**.

Companion docs:

- `docs/vaila_buttons/yolo-fb.md` — chooser table + launcher CLI
- `AGENTS.md` § Conventions (**GUI→CLI mirror**) + § History (v0.3.72, v0.3.76)
- `docs/sessions/2026-07-06-yolo-fb-gui-cli-mirror.md` — implementation log
- `docs/sessions/2026-07-07-sapiens-output-dir-fix.md` — one `processed_sapiens_*` per run (v0.3.76)
- `.claude/skills/sam3-video/SKILL.md` — SAM 3 specifics
- `vaila/help/vaila_sapiens.md` — Sapiens2 Pose

---

## Chooser (vaila.py `yolotrackerpose`)

**Button:** Frame B **B4_r4_c1** — **YOLO + FB** (was “Video AI tools” / “YOLO + SAM”).

Each chooser button calls `_print_yolo_fb_launch()` before launching the tool:

| Chooser button | Launcher CLI |
|----------------|--------------|
| Tracker (v26) | `uv run python -u -m vaila.yolov26track` |
| Pose (video) | in-process `yolov26track.run_yolov26pose_video` from main GUI |
| Pose (tracking) | `uv run python -u -m vaila.yolov26track` + GUI ID picker |
| Seg (v26) | same as Tracker; pick `-seg.pt` + seg run mode |
| SAM 3 video | `uv run python -u vaila/vaila_sam.py` |
| Sapiens2 Pose | `uv run python -u vaila/vaila_sapiens.py` |
| SAM3+Sapiens2 | `uv run python -u vaila/sam3sapiens2.py` |
| SAM3+Sapiens2 Visualize ID | `uv run python -u vaila/sam3sapiens2_visualize.py` |
| Train YOLOv26 | `uv run python -u -m vaila.yolotrain` |

---

## GUI→CLI mirror convention

**Rule:** any module with a CLI must print copy-paste commands on GUI **Run**.

- **Prefix:** `>>` (not `[bracketed]` — absl from mediapipe/opencv eats bracketed stdout)
- **Chooser:** launcher only (`_print_yolo_fb_launch` in `vaila.py`)
- **Run:** full args after user confirms dialogs
- **Highlight (v0.3.103+):** the printed banner is bold-yellow ANSI on an interactive
  TTY (plain text when redirected/piped or `NO_COLOR` is set), via the shared
  `vaila/cli_highlight.py` module — `print_gui_cli_mirror(module_label, cli)` for a
  single reproducible command (banner + `>>` header + command line), or `highlight(text)`
  to wrap an individual line when a module prints a multi-line hint block instead of
  one clean command (e.g. `yolov26track.py`'s Pose workflow hints, `getpixelvideo.py`'s
  STOP/READ warnings). Every module below imports one or both of these instead of
  hand-rolling its own ANSI/print formatting — don't reintroduce a local duplicate.

| Module | Helper | When printed |
|--------|--------|--------------|
| `vaila_sam.py` | `_build_sam_cli_argv`, `_print_sam_equivalent_cli` (→ `print_gui_cli_mirror`) | SAM GUI **Run** |
| `vaila_sapiens.py` | `_format_sapiens_cli_command`, `_print_sapiens_equivalent_cli` (→ `print_gui_cli_mirror`) | Sapiens2 GUI **Run** |
| `sam3sapiens2.py` | `_format_gui_cli` → `print_gui_cli_mirror` | SAM3+Sapiens2 GUI **Run** |
| `sam3dinov3.py` | `_format_gui_cli` → `print_gui_cli_mirror` | SAM3+DINOv3 3D GUI **Run** |
| `sapiens2_3d.py` | `_format_gui_cli` → `print_gui_cli_mirror` | Sapiens2 3D Pose GUI **Run** |
| `reid_markers.py` | `run_geometric_merge` callers → `print_gui_cli_mirror` | CLI `main()` and GUI merge dialog |
| `markerless_2d_analysis.py` | inline → `print_gui_cli_mirror` | Batch GUI **Run** |
| `markerless2d_analysis_v2.py` | inline → `print_gui_cli_mirror` | `main()` when `-i`/`-o` given |
| `rec3d.py` | inline → `print_gui_cli_mirror` | Preview + repeated after processing |
| `rec3d_one_dlt3d.py` | inline → `print_gui_cli_mirror` | Preview + repeated after processing |
| `blender_viz.py` | `format_blender_viz_cli` → `print_gui_cli_mirror` | Animation Blender button |
| `vaila_deadlift_imu.py` | `_format_cli_command` → `highlight()` per line | Run-configuration summary |
| `yolov26track.py` | `_format_track_cli_command` → `highlight()` per line | Tracker GUI before video loop (one `track` per file) |
| `yolov26track.py` | `_print_pose_video_equivalent_cli` → `highlight()` per line | Pose (video) after config dialog |
| `yolov26track.py` | `_print_pose_from_tracking_workflow_hint` → `highlight()` per line | Pose (tracking) after dir pick |
| `yolotrain.py` | `_format_training_cli_command`, `_print_gui_state` | Start Training + GUI open |
| `getpixelvideo.py` | `>> Equivalent CLI` blocks → `highlight()` per block | Load Tracking CSV / save hints |

### Adding mirror to a new GUI module

```python
try:
    from .cli_highlight import print_gui_cli_mirror
except ImportError:
    from cli_highlight import print_gui_cli_mirror  # ty: ignore[unresolved-import]

import shlex

def _format_my_cli_command(input_path: str, output: str, *, flag: int) -> list[str]:
    return ["uv", "run", "vaila/my_module.py", "-i", input_path, "-o", output, "--flag", str(flag)]

def _print_equivalent_cli(...) -> None:
    print_gui_cli_mirror("vaila/my_module", _format_my_cli_command(...))
```

`print_gui_cli_mirror` accepts either an argv `list[str]` (joined via `shlex.join`) or an
already-formatted command string. Call it **after** dialogs OK, **before** long work.

---

## Sapiens2 Pose (v0.3.71+)

```bash
uv sync --extra sapiens
bash bin/setup_sapiens2.sh   # clones .local/third_party/sapiens2/ + HF weights
uv run vaila/vaila_sapiens.py -i VIDEO_OR_DIR -o OUT_PARENT --model 1b
```

- Default model **1b** (RTX 4090 24 GiB)
- CUDA only; Meta Sapiens2 License
- **One** `processed_sapiens_<timestamp>/` per run (v0.3.76); per-video subdirs `<timestamp>/<stem>/`
- Isolated workers receive `--output-base` from parent — no empty duplicate folder
- GUI→CLI mirror prints only `-o` (not `--output-base`); CLI creates `processed_sapiens_<ts>/` under `-o`
- Tests: `uv run pytest tests/test_vaila_sapiens.py -v`

---

## Quick QA after changes

```bash
uv run ruff check vaila/vaila_sapiens.py vaila/vaila_sam.py vaila/yolov26track.py vaila.py --fix
uv run ruff format vaila/
uv run pytest tests/test_vaila_sapiens.py tests/test_vaila_sam.py::test_build_sam_cli_argv_includes_prompt tests/test_yolov26track_pose_reid.py::test_format_track_cli_command_maps_config -v
```

---

## Cursor CLI resume checklist

1. `cd ~/data/vaila && uv sync` (add `--extra sam` / `--extra sapiens` / `--extra gpu` as needed)
2. Read `AGENTS.md` History § v0.3.76 (Sapiens2 output dir) and this skill
3. Global version: **0.3.91** (`vaila.py` header)
4. Never rename chooser back to “Video AI tools” in docs — use **YOLO + FB**
5. Visualize ID: CLI `--id N` or omit `--id` for interactive prompt; GUI uses combobox after browsing the run dir. Overlay = SAM3 contour fill + Sapiens2 left/right skeleton colors.
