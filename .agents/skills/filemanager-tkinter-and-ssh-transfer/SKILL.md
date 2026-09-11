---
name: filemanager-tkinter-and-ssh-transfer
description: Use when maintaining vailá File Manager Frame A buttons, previewed local operations, CLI parity, or interactive SSH transfer with status-file completion.
---

# File Manager: previewed ops, CLI, and SSH transfer

Architecture since **v0.3.137** (11 September 2026). Parameter collection is separate from execution. GUI and CLI share the same plan/execute path.

## Modules

| File | Role |
|------|------|
| `vaila/filemanager.py` | `build_plan` / `execute_plan`, CLI (`python -m vaila.filemanager`), thin wrappers |
| `vaila/filemanager_gui.py` | Tkinter: fields → Preview → Apply; Transfer opens a terminal |
| `vaila/task_feedback.py` | `>> vaila/filemanager:` messages, `WorkerTask` queue, redaction |
| `vaila.py` Frame A | Lazy wrappers → `copy_file`, `move_file`, …, `transfer_file` → `show_action` |

Help: `vaila/help/filemanager.md` (+ `.html`). Button notes: `docs/vaila_buttons/*-file.md`. Diagnose failures with `/debug` → `.agents/skills/debug/SKILL.md`.

## vailá maintenance rule (version/date)

When editing these modules (or any `*.py`), also update:

- Script header **Update Date** + **Version** (global from `vaila.py` banner)
- Root `README.md` `Last updated: YYYY-MM-DD`
- Help: `vaila/help/filemanager.{md,html}`, `vaila/help/index.{md,html}`
- Installers / `vaila.py` if install/run UX changes

See `AGENTS.md` checklist.

## Flow

```
Frame A / CLI args
    → FileManagerGUI or argparse
    → build_plan(action, source, …)   # fixed Target list + identity stamps
    → preview (dry-run / GUI Preview) # no silent changes
    → execute_plan(plan)              # worker thread in GUI; refuse stale/changed targets
    → Feedback summary + exit code
```

Invariants:

- Preview locks the target list; files added after preview are not selected.
- No silent overwrite; collisions are partial failures.
- Destination inside source skips previous `vaila_*` output trees and the active log file.
- Removal is permanent; CLI needs `--yes` (GUI confirms the fixed list).
- Workers must not touch Tk widgets; GUI drains `WorkerTask` via `after()`.
- One busy task per window; Cancel cooperates between files.

## GUI vs CLI

- **Rename button** → `normalize_names()` → action `normalize` (accents/spaces cleanup).
- **Literal rename** → CLI/action `rename` (`--text` / `--replacement`).
- **Import** → `import-vicon` only; other formats marked unavailable (no fake success).
- GUI prints a safely quoted equivalent CLI (`command_text`); removal prints default to `--dry-run`.
- Subcommands: `copy`, `move`, `remove`, `rename`, `normalize`, `find`, `tree`, `export`, `import-vicon`, `transfer`.
- Local mutating ops support `--dry-run`. Optional `--debug` and `--log-file` (keep log outside selected removal targets).

```bash
uv run python -m vaila.filemanager copy --help
uv run python -m vaila.filemanager remove --source /data/tmp --pattern .tmp --removal-type ext --dry-run
uv run python -m vaila.filemanager transfer --local "/data/my files" --host server --user analyst --remote /data
```

## Tkinter window ownership

Still use hybrid roots: if `tk._default_root` exists, open `Toplevel` + `wait_window()`; else `Tk()` + `mainloop()`. Implemented in `filemanager_gui.show_action`. Do not create a second `Tk()` under the main vailá process.

Keep StringVars / widgets alive for the window lifetime (bind to the window or class attrs). Returning from a non-blocking dialog while Entries still reference GC’d `StringVar`s caused segfault 139 historically.

## SSH Transfer

rsync/scp need a real TTY for passwords. Pattern:

1. GUI collects local/host/user/port/remote/mode.
2. Writes a temp script that runs `python -m vaila.filemanager transfer … --status-file PATH`.
3. Launches an external terminal (`gnome-terminal`, etc.).
4. Status line: **Terminal opened - waiting for transfer result** ≠ completed.
5. When `--status-file` appears with an exit code, GUI shows success or `failed (exit N)`.

Do not treat “terminal launched” as transfer success. Never auto-run real transfers from `/debug`.

## Tests

```bash
uv run pytest tests/test_filemanager_operations.py tests/test_filetools_downloader.py -v
```

Coverage includes collisions, nested destination, stale targets after preview, cancel, VICON import, headless CLI, transfer argv safety + status-file, GUI/CLI equivalence, Frame A dispatch to `show_action`, and simulated Transfer success/failure (no real SSH).

GUI smoke needs a display; set `VAILA_GUI_ARTIFACTS=/tmp/…` for window PNGs.
