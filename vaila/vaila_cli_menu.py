"""Terminal menu for vailá — compact grid + ``/`` search + CLI run hints."""

from __future__ import annotations

import re
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from vaila.vaila_cli_hints import CliRunHint, get_cli_hint

if TYPE_CHECKING:
    from vaila import Vaila

__version__ = "0.3.82"
__updated__ = "10 July 2026"

_ROW_RE = re.compile(r"_r(\d+)_")
_COL_RE = re.compile(r"_c(\d+)$")


@dataclass(frozen=True, slots=True)
class VailaMenuEntry:
    code: str
    section: str
    subsection: str
    label: str
    handler: str  # Vaila method name or external:* key


_EXTERNAL_HANDLERS: dict[str, Callable[[], None]] = {
    "external:mne_overview": lambda: webbrowser.open(
        "https://mne.tools/dev/auto_tutorials/intro/10_overview.html"
    ),
    "external:heartrate_py": lambda: webbrowser.open(
        "https://github.com/paulvangentcom/heartrate_analysis_python"
    ),
}

VAILA_MENU_ENTRIES: tuple[VailaMenuEntry, ...] = (
    # Frame A — File Manager
    VailaMenuEntry("A_r1_c1", "A", "File Manager", "Rename", "rename_files"),
    VailaMenuEntry("A_r1_c2", "A", "File Manager", "Import", "import_file"),
    VailaMenuEntry("A_r1_c3", "A", "File Manager", "Export", "export_file"),
    VailaMenuEntry("A_r1_c4", "A", "File Manager", "Copy", "copy_file"),
    VailaMenuEntry("A_r1_c5", "A", "File Manager", "Move", "move_file"),
    VailaMenuEntry("A_r1_c6", "A", "File Manager", "Remove", "remove_file"),
    VailaMenuEntry("A_r1_c7", "A", "File Manager", "Tree", "tree_file"),
    VailaMenuEntry("A_r1_c8", "A", "File Manager", "Find", "find_file"),
    VailaMenuEntry("A_r1_c9", "A", "File Manager", "Transfer", "transfer_file"),
    # Frame B — Multimodal Analysis
    VailaMenuEntry("B1_r1_c1", "B", "Multimodal Analysis", "IMU", "imu_analysis"),
    VailaMenuEntry(
        "B1_r1_c2", "B", "Multimodal Analysis", "Motion Capture Cluster", "cluster_analysis"
    ),
    VailaMenuEntry(
        "B1_r1_c3", "B", "Multimodal Analysis", "Motion Capture Full Body", "mocap_analysis"
    ),
    VailaMenuEntry(
        "B1_r1_c4", "B", "Multimodal Analysis", "Markerless 2D", "markerless_2d_analysis"
    ),
    VailaMenuEntry(
        "B1_r1_c5", "B", "Multimodal Analysis", "Markerless 3D", "markerless_3d_analysis"
    ),
    VailaMenuEntry("B2_r2_c1", "B", "Multimodal Analysis", "Vector Coding", "vector_coding"),
    VailaMenuEntry("B2_r2_c2", "B", "Multimodal Analysis", "EMG", "emg_analysis"),
    VailaMenuEntry("B2_r2_c3", "B", "Multimodal Analysis", "Force Plate", "force_analysis"),
    VailaMenuEntry("B2_r2_c4", "B", "Multimodal Analysis", "GNSS/GPS", "gnss_analysis"),
    VailaMenuEntry("B2_r2_c5", "B", "Multimodal Analysis", "MEG/EEG", "external:mne_overview"),
    VailaMenuEntry("B3_r3_c1", "B", "Multimodal Analysis", "HR/ECG", "external:heartrate_py"),
    VailaMenuEntry(
        "B3_r3_c2", "B", "Multimodal Analysis", "Yolo + Markerless_MP", "markerless2d_mpyolo"
    ),
    VailaMenuEntry("B3_r3_c3", "B", "Multimodal Analysis", "Vertical Jump", "vailajump"),
    VailaMenuEntry("B3_r3_c4", "B", "Multimodal Analysis", "Cube2D", "cube2d_kinematics"),
    VailaMenuEntry(
        "B3_r3_c5", "B", "Multimodal Analysis", "Animal Open Field", "animal_open_field"
    ),
    VailaMenuEntry("B4_r4_c1", "B", "Multimodal Analysis", "YOLO + FB", "yolo_and_sam"),
    VailaMenuEntry("B4_r4_c2", "B", "Multimodal Analysis", "ML Walkway", "ml_walkway"),
    VailaMenuEntry("B4_r4_c3", "B", "Multimodal Analysis", "Markerless Hands", "markerless_hands"),
    VailaMenuEntry("B4_r4_c4", "B", "Multimodal Analysis", "MP Angles", "mp_angles_calculation"),
    VailaMenuEntry("B4_r4_c5", "B", "Multimodal Analysis", "Markerless Live", "markerless_live"),
    VailaMenuEntry("B5_r5_c1", "B", "Multimodal Analysis", "Ultrasound", "ultrasound"),
    VailaMenuEntry("B5_r5_c2", "B", "Multimodal Analysis", "Brainstorm", "brainstorm"),
    VailaMenuEntry("B5_r5_c3", "B", "Multimodal Analysis", "Scout", "scout"),
    VailaMenuEntry("B5_r5_c4", "B", "Multimodal Analysis", "Start Block", "startblock"),
    VailaMenuEntry("B5_r5_c5", "B", "Multimodal Analysis", "Pynalty", "pynalty"),
    VailaMenuEntry("B5_r6_c1", "B", "Multimodal Analysis", "Sprint", "sprint"),
    VailaMenuEntry("B5_r6_c2", "B", "Multimodal Analysis", "Face Mesh", "face_mesh_analysis"),
    VailaMenuEntry("B5_r6_c3", "B", "Multimodal Analysis", "tugturn", "tugturn"),
    VailaMenuEntry("B5_r6_c4", "B", "Multimodal Analysis", "Soccer Tools", "soccer_tools"),
    VailaMenuEntry("B5_r6_c5", "B", "Multimodal Analysis", "Deadlift", "deadlift_analysis"),
    VailaMenuEntry("B6_r7_c1", "B", "Multimodal Analysis", "vailá", "show_vaila_message"),
    VailaMenuEntry("B6_r7_c2", "B", "Multimodal Analysis", "vailá", "show_vaila_message"),
    VailaMenuEntry("B6_r7_c3", "B", "Multimodal Analysis", "Treadmill LC", "treadmill_lc"),
    VailaMenuEntry("B6_r7_c4", "B", "Multimodal Analysis", "vailá", "show_vaila_message"),
    VailaMenuEntry("B6_r7_c5", "B", "Multimodal Analysis", "vailá", "show_vaila_message"),
    # Frame C_A — Data Files
    VailaMenuEntry("C_A_r1_c1", "C_A", "Data Files", "Edit CSV", "reorder_csv_data"),
    VailaMenuEntry("C_A_r1_c2", "C_A", "Data Files", "C3D <--> CSV", "convert_c3d_csv"),
    VailaMenuEntry("C_A_r1_c3", "C_A", "Data Files", "Smooth & Filter", "gapfill_split"),
    VailaMenuEntry("C_A_r2_c1", "C_A", "Data Files", "Make DLT2D", "dlt2d"),
    VailaMenuEntry("C_A_r2_c2", "C_A", "Data Files", "Rec2D 1DLT", "rec2d_one_dlt2d"),
    VailaMenuEntry("C_A_r2_c3", "C_A", "Data Files", "Rec2D MultiDLT", "rec2d"),
    VailaMenuEntry("C_A_r3_c1", "C_A", "Data Files", "Make DLT3D", "run_dlt3d"),
    VailaMenuEntry("C_A_r3_c2", "C_A", "Data Files", "Rec3D 1DLT", "rec3d_one_dlt3d"),
    VailaMenuEntry("C_A_r3_c3", "C_A", "Data Files", "Rec3D MultiDLT", "rec3d"),
    VailaMenuEntry("C_A_r4_c1", "C_A", "Data Files", "ReID Marker", "reid_marker"),
    VailaMenuEntry("C_A_r4_c2", "C_A", "Data Files", "vailá", "show_vaila_message"),
    VailaMenuEntry("C_A_r4_c3", "C_A", "Data Files", "vailá", "show_vaila_message"),
    VailaMenuEntry("C_A_r5_c1", "C_A", "Data Files", "vailá", "show_vaila_message"),
    VailaMenuEntry("C_A_r5_c2", "C_A", "Data Files", "vailá", "show_vaila_message"),
    VailaMenuEntry("C_A_r5_c3", "C_A", "Data Files", "vailá", "show_vaila_message"),
    # Frame C_B — Video and Image
    VailaMenuEntry(
        "C_B_r1_c1", "C_B", "Video and Image", "Video<-->PNG", "extract_png_from_videos"
    ),
    VailaMenuEntry("C_B_r1_c2", "C_B", "Video and Image", "Crop Face", "crop_faces_atletas"),
    VailaMenuEntry("C_B_r1_c3", "C_B", "Video and Image", "Draw Box", "draw_box"),
    VailaMenuEntry("C_B_r2_c1", "C_B", "Video and Image", "Compress Video", "compress_videos_gui"),
    VailaMenuEntry("C_B_r2_c2", "C_B", "Video and Image", "vailá", "show_vaila_message"),
    VailaMenuEntry("C_B_r2_c3", "C_B", "Video and Image", "Make Sync file", "sync_videos"),
    VailaMenuEntry("C_B_r3_c1", "C_B", "Video and Image", "GetPixelCoord", "getpixelvideo"),
    VailaMenuEntry(
        "C_B_r3_c2", "C_B", "Video and Image", "Metadata info", "count_frames_in_videos"
    ),
    VailaMenuEntry(
        "C_B_r3_c3", "C_B", "Video and Image", "Merge|Split Video", "process_videos_gui"
    ),
    VailaMenuEntry("C_B_r4_c1", "C_B", "Video and Image", "Distort Video/data", "run_distortvideo"),
    VailaMenuEntry("C_B_r4_c2", "C_B", "Video and Image", "Cut Video", "cut_video"),
    VailaMenuEntry("C_B_r4_c3", "C_B", "Video and Image", "Resize Video", "resize_video"),
    VailaMenuEntry("C_B_r5_c1", "C_B", "Video and Image", "YT Downloader", "ytdownloader"),
    VailaMenuEntry("C_B_r5_c2", "C_B", "Video and Image", "Insert Audio", "run_iaudiovid"),
    VailaMenuEntry("C_B_r5_c3", "C_B", "Video and Image", "rm Dup PNG", "remove_duplicate_frames"),
    # Frame C_C — Visualization
    VailaMenuEntry("C_C_r1_c1", "C_C", "Visualization", "Show C3D", "show_c3d_data"),
    VailaMenuEntry("C_C_r1_c2", "C_C", "Visualization", "Show CSV 3D", "show_csv_file"),
    VailaMenuEntry("C_C_r2_c1", "C_C", "Visualization", "Plot 2D", "plot_2d_data"),
    VailaMenuEntry("C_C_r2_c2", "C_C", "Visualization", "Plot 3D", "plot_3d_data"),
    VailaMenuEntry("C_C_r3_c1", "C_C", "Visualization", "Draw Sports", "draw_sports_fields_courts"),
    VailaMenuEntry("C_C_r3_c2", "C_C", "Visualization", "Stroboscopic", "run_stroboscopic"),
    VailaMenuEntry("C_C_r4_c1", "C_C", "Visualization", "vailá", "show_vaila_message"),
    VailaMenuEntry("C_C_r4_c2", "C_C", "Visualization", "vailá", "show_vaila_message"),
    VailaMenuEntry("C_C_r5_c1", "C_C", "Visualization", "vailá", "show_vaila_message"),
    VailaMenuEntry("C_C_r5_c2", "C_C", "Visualization", "vailá", "show_vaila_message"),
    # Global
    VailaMenuEntry("HELP", "_", "Global", "Help", "display_help"),
    VailaMenuEntry("EXIT", "_", "Global", "Exit", "quit_app"),
    VailaMenuEntry("SHELL", "_", "Global", "imagination! (xonsh shell)", "open_terminal_shell"),
)

_SECTION_TITLES: dict[str, str] = {
    "A": "File Manager (Frame A)",
    "B": "Multimodal Analysis (Frame B)",
    "C_A": "Data Files (C_A)",
    "C_B": "Video and Image (C_B)",
    "C_C": "Visualization (C_C)",
}

_TOOL_SECTIONS = ("C_A", "C_B", "C_C")


def _row_num(code: str) -> int:
    match = _ROW_RE.search(code)
    return int(match.group(1)) if match else 0


def _col_num(code: str) -> int:
    match = _COL_RE.search(code)
    return int(match.group(1)) if match else 0


def _entries_by_section(section: str) -> list[VailaMenuEntry]:
    return [e for e in VAILA_MENU_ENTRIES if e.section == section]


def _entry_by_code(code: str) -> VailaMenuEntry | None:
    key = code.strip().upper()
    for entry in VAILA_MENU_ENTRIES:
        if entry.code.upper() == key:
            return entry
    return None


def _search_entries(query: str) -> list[VailaMenuEntry]:
    needle = query.strip().lower()
    if not needle:
        return []
    matches: list[VailaMenuEntry] = []
    for entry in VAILA_MENU_ENTRIES:
        hay = f"{entry.code} {entry.label} {entry.subsection} {entry.handler}".lower()
        if needle in hay:
            matches.append(entry)
    return matches


_NUMBERED_ORDER: tuple[str, ...] = ("A", "B", "C_A", "C_B", "C_C")


def _ordered_menu_entries() -> list[VailaMenuEntry]:
    ordered: list[VailaMenuEntry] = []
    for section in _NUMBERED_ORDER:
        section_entries = sorted(
            _entries_by_section(section),
            key=lambda e: (_row_num(e.code), _col_num(e.code)),
        )
        ordered.extend(section_entries)
    for code in ("HELP", "SHELL"):
        entry = _entry_by_code(code)
        if entry is not None:
            ordered.append(entry)
    return ordered


def _build_number_maps() -> tuple[dict[int, VailaMenuEntry], dict[str, int]]:
    by_number: dict[int, VailaMenuEntry] = {}
    by_code: dict[str, int] = {}
    for index, entry in enumerate(_ordered_menu_entries(), start=1):
        by_number[index] = entry
        by_code[entry.code.upper()] = index
    return by_number, by_code


_NUMBERED_BY_INDEX, _NUMBER_BY_CODE = _build_number_maps()


def _entry_by_number(value: str | int) -> VailaMenuEntry | None:
    try:
        num = int(value)
    except (TypeError, ValueError):
        return None
    return _NUMBERED_BY_INDEX.get(num)


def resolve_handler(app: Vaila, entry: VailaMenuEntry) -> Callable[[], None]:
    if entry.handler.startswith("external:"):
        ext = _EXTERNAL_HANDLERS.get(entry.handler)
        if ext is None:
            raise KeyError(f"Unknown external handler: {entry.handler}")
        return ext
    method = getattr(app, entry.handler, None)
    if method is None or not callable(method):
        raise AttributeError(f"Vaila has no callable handler '{entry.handler}'")
    return method


def _cell_width(console: Console) -> int:
    width = console.size.width or 100
    if width >= 140:
        return 30
    if width >= 120:
        return 28
    if width >= 100:
        return 26
    return 24


def _max_cols(console: Console) -> int:
    width = console.size.width or 100
    cell = _cell_width(console) + 1
    return max(2, min(5, width // cell))


def _format_cell(entry: VailaMenuEntry, width: int, number: int) -> str:
    label = entry.label.replace("|", "/")
    if len(label) > 12:
        label = label[:11] + "…"
    text = f"{number:>2}.{entry.code} {label}"
    return text[:width].ljust(width)


def _render_section_rows(entries: list[VailaMenuEntry], console: Console) -> list[str]:
    if not entries:
        return []
    cell_w = _cell_width(console)
    max_cols = _max_cols(console)
    by_row: dict[int, list[VailaMenuEntry]] = {}
    for entry in entries:
        by_row.setdefault(_row_num(entry.code), []).append(entry)
    lines: list[str] = []
    for row_id in sorted(by_row):
        row_entries = sorted(by_row[row_id], key=lambda e: _col_num(e.code))
        chunks = [row_entries[i : i + max_cols] for i in range(0, len(row_entries), max_cols)]
        for chunk in chunks:
            cells = []
            for entry in chunk:
                number = _NUMBER_BY_CODE.get(entry.code.upper(), 0)
                cells.append(_format_cell(entry, cell_w, number))
            lines.append("".join(cells).rstrip())
    return lines


def _tools_column_width(console: Console) -> int:
    term_w = max(96, console.size.width or 100)
    # Three columns + panel borders/gaps inside the outer Frame C panel
    return max(28, (term_w - 14) // 3)


_TOOLS_SECTION_HEADERS: dict[str, str] = {
    "C_A": "C_A Data Files",
    "C_B": "C_B Video/Image",
    "C_C": "C_C Visualiz.",
}


def _format_tools_cell(entry: VailaMenuEntry, width: int, number: int) -> str:
    code = entry.code
    for prefix in ("C_A_", "C_B_", "C_C_"):
        if code.startswith(prefix):
            code = code[len(prefix) :]
            break
    label = entry.label.replace("|", "/")
    if len(label) > 12:
        label = label[:11] + "…"
    text = f"{number:>2}.{code} {label}"
    return text[:width].ljust(width)


def _render_tools_grid(console: Console) -> Panel:
    col_w = _tools_column_width(console)
    columns: list[list[str]] = []
    for section in _TOOL_SECTIONS:
        entries = sorted(
            _entries_by_section(section),
            key=lambda e: (_row_num(e.code), _col_num(e.code)),
        )
        col_lines = [_TOOLS_SECTION_HEADERS[section][:col_w].ljust(col_w)]
        for entry in entries:
            number = _NUMBER_BY_CODE.get(entry.code.upper(), 0)
            col_lines.append(_format_tools_cell(entry, col_w, number))
        columns.append(col_lines)

    max_rows = max(len(col) for col in columns)
    for col in columns:
        while len(col) < max_rows:
            col.append(" " * col_w)

    body_lines = [
        "".join(col[row_idx][:col_w].ljust(col_w) for col in columns).rstrip()
        for row_idx in range(max_rows)
    ]
    body = Text("\n".join(body_lines), no_wrap=True, overflow="crop")
    return Panel(body, title="Tools Available (Frame C)", border_style="magenta")


def _print_compact_menu(console: Console) -> None:
    parts: list[Panel | Text] = []
    banner = Text()
    banner.append("vailá", style="bold italic blue")
    banner.append(f" CLI  {__updated__}  v{__version__}", style="dim")
    parts.append(
        Panel(
            banner,
            subtitle="Versatile Anarcho Integrated Liberation Ánalysis",
            border_style="blue",
        )
    )
    for section in ("A", "B"):
        lines = _render_section_rows(_entries_by_section(section), console)
        parts.append(
            Panel(
                "\n".join(lines),
                title=_SECTION_TITLES[section],
                border_style="cyan" if section == "A" else "green",
            )
        )
    parts.append(_render_tools_grid(console))
    # Global shortcuts with numbers
    global_lines: list[str] = []
    for code in ("HELP", "SHELL"):
        entry = _entry_by_code(code)
        if entry is None:
            continue
        number = _NUMBER_BY_CODE.get(entry.code.upper(), 0)
        global_lines.append(_format_cell(entry, _cell_width(console), number).strip())
    if global_lines:
        parts.append(
            Panel("  ".join(global_lines), title="Global", border_style="dim"),
        )
    footer = Text()
    footer.append("#", style="bold magenta")
    footer.append(" number  ", style="dim")
    footer.append("CODE", style="bold green")
    footer.append(" run  ", style="dim")
    footer.append("/term", style="bold yellow")
    footer.append(" search  ", style="dim")
    footer.append("h", style="bold")
    footer.append(" help  ", style="dim")
    footer.append("q", style="bold")
    footer.append(" quit", style="dim")
    parts.append(Panel(footer, border_style="dim"))
    console.print(Group(*parts))


def _print_search_results(console: Console, matches: list[VailaMenuEntry], query: str) -> None:
    if not matches:
        console.print(f"[yellow]No matches for[/] [bold]/{query}[/]")
        return
    cell_w = _cell_width(console)
    max_cols = max(2, min(3, (console.size.width or 100) // (cell_w + 1)))
    lines: list[str] = []
    for idx in range(0, len(matches), max_cols):
        chunk = matches[idx : idx + max_cols]
        numbered = []
        for offset, entry in enumerate(chunk, start=idx + 1):
            menu_num = _NUMBER_BY_CODE.get(entry.code.upper(), offset)
            numbered.append(
                f"{menu_num:>2}.{entry.code} {entry.label[:12]}".ljust(cell_w),
            )
        lines.append("".join(numbered).rstrip())
    console.print(
        Panel(
            "\n".join(lines),
            title=f"Search: /{query}  ({len(matches)} hits)",
            border_style="yellow",
        )
    )


def _print_cli_run_hint(console: Console, entry: VailaMenuEntry, hint: CliRunHint) -> None:
    body = Text()
    body.append(f"{entry.code}", style="bold green")
    body.append(f" — {entry.label}\n", style="white")
    if hint.note:
        body.append(f"{hint.note}\n\n", style="dim italic")
    if hint.commands:
        body.append("Equivalent CLI (copy/paste):\n", style="bold")
        for cmd in hint.commands:
            body.append("  >> ", style="cyan")
            body.append(f"{cmd}\n", style="bold white")
    else:
        body.append("No standalone CLI — use the desktop GUI.\n", style="yellow")
    console.print(Panel(body, border_style="green", title="Run via CLI"))


def _read_line(console: Console, prompt: str = "vailá ›") -> str:
    try:
        return console.input(f"[bold cyan]{prompt}[/] ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Interrupted — goodbye.[/]")
        raise SystemExit(0) from None


def _pick_from_matches(console: Console, matches: list[VailaMenuEntry]) -> VailaMenuEntry | None:
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    _print_search_results(console, matches, "(pick)")
    choice = _read_line(console, "pick # or code ›")
    if not choice:
        return None
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(matches):
            return matches[idx - 1]
    return _entry_by_code(choice)


def _run_entry(app: Vaila, entry: VailaMenuEntry, console: Console, *, headless: bool) -> bool:
    """Run or print CLI hint. Returns False when the CLI session should exit."""
    if entry.code.upper() == "EXIT":
        if headless:
            console.print("[dim]Goodbye.[/]")
        else:
            resolve_handler(app, entry)()
        return False

    hint = get_cli_hint(handler=entry.handler, code=entry.code, label=entry.label)

    if headless and not hint.invoke_handler:
        _print_cli_run_hint(console, entry, hint)
        return True

    if hint.invoke_handler and entry.handler.startswith("external:"):
        resolve_handler(app, entry)()
        console.print("[dim]Opened external link in browser.[/]\n")
        return True

    if entry.code.upper() == "HELP":
        resolve_handler(app, entry)()
        console.print("[dim]Help opened in browser.[/]\n")
        return True

    if headless:
        _print_cli_run_hint(console, entry, hint)
        return True

    console.print(f"\n[bold]>>[/] {entry.code} — {entry.label}")
    resolve_handler(app, entry)()
    console.print("[dim]Action finished.[/]\n")
    return True


def run_cli_menu(app: Vaila, *, initial_code: str | None = None, headless: bool = True) -> None:
    """Interactive terminal menu or one-shot action code."""
    console = Console(highlight=False)

    if initial_code:
        query = initial_code.strip()
        if query.startswith("/"):
            matches = _search_entries(query[1:])
            if not matches:
                console.print(f"[red]No matches for[/] [bold]/{query[1:]}[/]")
                raise SystemExit(1)
            if len(matches) == 1:
                _run_entry(app, matches[0], console, headless=headless)
            else:
                _print_search_results(console, matches, query[1:])
                console.print(
                    "[dim]Multiple matches — run one explicitly, e.g. "
                    "[bold]uv run vaila.py --cli B4_r4_c1[/][/]"
                )
            return

        entry = _entry_by_code(query)
        if entry is None:
            entry = _entry_by_number(query)
        if entry is None:
            console.print(f"[red]Unknown action:[/] {initial_code}")
            raise SystemExit(1)
        _run_entry(app, entry, console, headless=headless)
        return

    while True:
        console.clear()
        _print_compact_menu(console)
        choice = _read_line(console)

        if not choice:
            continue
        lowered = choice.lower()

        if lowered in {"q", "quit", "exit"}:
            if not _run_entry(
                app,
                _entry_by_code("EXIT") or VAILA_MENU_ENTRIES[-2],
                console,
                headless=headless,
            ):
                break
            continue
        if lowered in {"h", "help"}:
            _run_entry(
                app,
                _entry_by_code("HELP") or VAILA_MENU_ENTRIES[-3],
                console,
                headless=headless,
            )
            _read_line(console, "Enter ›")
            continue
        if lowered in {"s", "shell"}:
            _run_entry(
                app,
                _entry_by_code("SHELL") or VAILA_MENU_ENTRIES[-1],
                console,
                headless=headless,
            )
            _read_line(console, "Enter ›")
            continue

        if choice.startswith("/"):
            query = choice[1:].strip()
            matches = _search_entries(query)
            if not matches:
                console.print(f"[yellow]No matches for[/] [bold]/{query}[/]")
                _read_line(console, "Enter ›")
                continue
            if len(matches) == 1:
                _run_entry(app, matches[0], console, headless=headless)
            else:
                _print_search_results(console, matches, query)
                picked = _pick_from_matches(console, matches)
                if picked is not None:
                    _run_entry(app, picked, console, headless=headless)
            _read_line(console, "Enter ›")
            continue

        if choice.isdigit():
            numbered = _entry_by_number(choice)
            if numbered is not None:
                if not _run_entry(app, numbered, console, headless=headless):
                    break
                _read_line(console, "Enter ›")
                continue

        direct = _entry_by_code(choice)
        if direct is not None:
            if not _run_entry(app, direct, console, headless=headless):
                break
            _read_line(console, "Enter ›")
            continue

        matches = _search_entries(choice)
        if matches:
            picked = matches[0] if len(matches) == 1 else _pick_from_matches(console, matches)
            if picked is not None:
                _run_entry(app, picked, console, headless=headless)
            _read_line(console, "Enter ›")
            continue

        console.print("[yellow]Unknown input.[/] Use #, CODE, /search, h, q.")
        _read_line(console, "Enter ›")


if __name__ == "__main__":
    import importlib.util
    import sys
    from pathlib import Path

    root_vaila = Path(__file__).resolve().parents[1] / "vaila.py"
    spec = importlib.util.spec_from_file_location("vaila_main", root_vaila)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["vaila_main"] = module
    spec.loader.exec_module(module)
    Vaila = module.Vaila

    _app = Vaila(gui=False)
    run_cli_menu(_app)
