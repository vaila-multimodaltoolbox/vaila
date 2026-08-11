"""
Project: vailá
Script: cli_highlight.py
Authors: Paulo Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 11 August 2026
Update Date: 11 August 2026
Version: 0.3.103

Description:
    Shared terminal-highlight helper for the "GUI->CLI mirror" convention
    (see AGENTS.md / CLAUDE.md and `.claude/skills/yolo-fb-gui-cli/SKILL.md`):
    any vailá module with both a GUI and a CLI prints, on GUI Run, the exact
    command line that reproduces the just-completed run headlessly. Every
    module that prints one of these mirrors reuses `highlight()` /
    `print_gui_cli_mirror()` from here instead of re-implementing its own
    ANSI wrapping, so the banner looks and behaves identically everywhere.

Usage:
    from vaila.cli_highlight import print_gui_cli_mirror
    print_gui_cli_mirror("vaila/my_module", cli_argv_list_or_string)
"""

from __future__ import annotations

import os
import shlex
import sys


def highlight(text: str) -> str:
    """Wrap `text` in bold-yellow ANSI codes when stdout is an interactive TTY.

    Redirected/piped output (logs, `tee`, CI) and `NO_COLOR` keep plain text,
    so the escape codes never corrupt a saved run log while still standing
    out on a live terminal.
    """
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return text
    return f"\033[1;33m{text}\033[0m"


def print_gui_cli_mirror(
    module_label: str,
    cli: list[str] | str,
    *,
    note: str = "Equivalent CLI (copy/paste to repeat this run):",
) -> None:
    """Print the GUI-equivalent CLI command inside a highlighted banner.

    `cli` may be a pre-built argv list (joined with `shlex.join`, so any
    argument containing spaces is quoted correctly) or an already-formatted
    command string. The banner is easy to spot while scrolling back the
    terminal, so the command can be copy/pasted to repeat this exact run
    later without reopening the GUI.
    """
    cli_str = cli if isinstance(cli, str) else shlex.join(cli)
    header = f">> {module_label}: {note}"
    body = f">>   {cli_str}"
    banner = "=" * min(max(len(body), 40), 100)
    print()
    print(highlight(banner))
    print(highlight(header))
    print(highlight(body))
    print(highlight(banner))
    print()
