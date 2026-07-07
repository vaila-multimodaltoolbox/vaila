---
name: yolo-fb-gui-cli
description: YOLO + FB chooser (Frame B), Sapiens2 Pose, and GUI→CLI terminal mirror for vailá video-AI tools. Use when reopening Cursor CLI, wiring GUI buttons, printing copy-paste CLI from Tkinter runs, or continuing work on vaila.py yolotrackerpose / vaila_sam / vaila_sapiens / yolov26track / yolotrain.
---

# YOLO + FB Chooser & GUI→CLI Mirror (v0.3.76)

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
| Train YOLOv26 | `uv run python -u -m vaila.yolotrain` |

---

## GUI→CLI mirror convention

**Rule:** any module with a CLI must print copy-paste commands on GUI **Run**.

- **Prefix:** `>>` (not `[bracketed]` — absl from mediapipe/opencv eats bracketed stdout)
- **Chooser:** launcher only (`_print_yolo_fb_launch` in `vaila.py`)
- **Run:** full args after user confirms dialogs

| Module | Helper | When printed |
|--------|--------|--------------|
| `vaila_sam.py` | `_build_sam_cli_argv`, `_print_sam_equivalent_cli` | SAM GUI **Run** |
| `vaila_sapiens.py` | `_format_sapiens_cli_command`, `_print_sapiens_equivalent_cli` | Sapiens2 GUI **Run** |
| `yolov26track.py` | `_format_track_cli_command` | Tracker GUI before video loop (one `track` per file) |
| `yolov26track.py` | `_print_pose_video_equivalent_cli` | Pose (video) after config dialog |
| `yolov26track.py` | `_print_pose_from_tracking_workflow_hint` | Pose (tracking) after dir pick |
| `yolotrain.py` | `_format_training_cli_command`, `_print_gui_state` | Start Training + GUI open |
| `getpixelvideo.py` | `>> Equivalent CLI` blocks | Load Tracking CSV / save hints |

### Adding mirror to a new GUI module

```python
import shlex

def _format_my_cli_command(input_path: str, output: str, *, flag: int) -> str:
    parts = ["uv", "run", "vaila/my_module.py", "-i", input_path, "-o", output, "--flag", str(flag)]
    return " ".join(shlex.quote(p) for p in parts)

def _print_equivalent_cli(...) -> None:
    print("\n>> vaila/my_module: Equivalent CLI (copy/paste):", flush=True)
    print(f">>   {_format_my_cli_command(...)}", flush=True)
    print("", flush=True)
```

Call `_print_equivalent_cli` **after** dialogs OK, **before** long work.

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
3. Global version: **0.3.76** (`vaila.py` header)
4. Never rename chooser back to “Video AI tools” in docs — use **YOLO + FB**
