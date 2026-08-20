"""
===============================================================================
treadmill_lc.py
===============================================================================
Project: vailá Multimodal Toolbox
Script: treadmill_lc.py

Author: Abel Gonçalves Chinaglia
Email: abel.chinaglia@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 09 June 2026
Update Date: 20 August 2026
Version: 0.3.108

Description:
------------
Load-cell processing for instrumented treadmill data. The module provides
artifact adjustment with interpolation review, signal filtering, calibration,
body-weight normalization, COP calculation, step detection, and running metrics.

Usage:
------
GUI:
    uv run vaila/treadmill_lc.py

CLI:
    uv run vaila/treadmill_lc.py -i data -s all

License:
--------
This program is licensed under the GNU Affero General Public License v3.0.
===============================================================================
"""

import argparse
import contextlib
import gc
import json
import os
import re
import shutil
import tkinter as tk
import webbrowser
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

import matplotlib

# Default Agg: savefig/tests/CLI must never auto-pick QtAgg (PySide6).
# Interactive GUI paths switch to TkAgg via _ensure_interactive_backend().
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml
from matplotlib.patches import Rectangle
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy.fft import fft, fftfreq
from scipy.interpolate import Rbf
from scipy.ndimage import median_filter
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

try:
    from .cli_highlight import print_gui_cli_mirror
except ImportError:  # Standalone execution: python vaila/treadmill_lc.py
    from cli_highlight import print_gui_cli_mirror

FS = 1000
VERSION = "0.3.108"
CLI_STEPS = ("all", "adjust", "interpolate", "filter", "process")
CLI_SCRIPT_PATH = "vaila/treadmill_lc.py"
RUN_HISTORY_PREFIX = "treadmill_lc_run_history"
RUN_HISTORY_FILENAME = f"{RUN_HISTORY_PREFIX}.toml"
DEFAULT_TOML_PATH = RUN_HISTORY_FILENAME
AUTO_START_PEAK_RATIO = 0.5
SIDEFOOT_RIGHT = 0
SIDEFOOT_LEFT = 1
_SIDEFOOT_PLOT_COLORS = {SIDEFOOT_RIGHT: "tab:red", SIDEFOOT_LEFT: "tab:green"}


def _sidefoot(step: dict[str, Any]) -> int:
    """Return 0 (right) or 1 (left) from a step dict.

    Canonical key is ``sidefoot``. Legacy ``side`` (0/1) and ``foot`` (R/L)
    are accepted so older in-memory dicts still plot.
    """
    if "sidefoot" in step:
        return int(step["sidefoot"])
    if "side" in step:
        return int(step["side"])
    if step.get("foot") == "L":
        return SIDEFOOT_LEFT
    return SIDEFOOT_RIGHT


console = Console()


def _matplotlib_backend_name() -> str:
    return str(matplotlib.get_backend()).lower()


def _ensure_noninteractive_backend() -> None:
    """Force Agg for headless figure writes (tests, CLI, batch savefig)."""
    if _matplotlib_backend_name() == "agg":
        return
    plt.close("all")
    plt.switch_backend("Agg")


def _ensure_interactive_backend() -> None:
    """Force TkAgg for interactive windows — never Qt (PySide6/PyQt)."""
    if _matplotlib_backend_name() == "tkagg":
        return
    plt.close("all")
    plt.switch_backend("TkAgg")


def print_treadmill_config_summary(
    config: dict[str, Any],
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    mode: str = "Headless CLI",
    step: str | None = None,
) -> None:
    """Print every effective CLI parameter in a Rich table."""
    with contextlib.suppress(OSError, BrokenPipeError):
        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Category", style="dim")

        table.add_row("Execution Mode", mode, "System")
        effective_step = step or str(config.get("execution", {}).get("step", "all"))
        table.add_row("Pipeline Step", effective_step, "Execution")
        timestamp_output = bool(config.get("execution", {}).get("timestamp_output", False))
        table.add_row(
            "Output Folder Mode",
            "timestamped" if timestamp_output else "stable/reused",
            "Execution",
        )
        if input_dir:
            table.add_row("Input Directory", str(input_dir), "Paths")
        if output_dir:
            table.add_row("Output Directory", str(output_dir), "Paths")

        for section in ("filters", "interpolation", "processing"):
            values = config.get(section, {})
            if not isinstance(values, dict):
                continue
            for key, value in values.items():
                label = key.replace("_", " ").title()
                display_value = str(value).lower() if isinstance(value, bool) else str(value)
                table.add_row(label, display_value, section.title())

        adjustments = config.get("adjustments", {})
        if isinstance(adjustments, dict):
            for trial, metadata in adjustments.items():
                interpolation = metadata.get("interpolation", {})
                method = interpolation.get("final_method", interpolation.get("status", "none"))
                interval_count = len(metadata.get("intervals", []))
                table.add_row(
                    f"{trial} Adjustment",
                    f"method={method}, intervals={interval_count}",
                    "GUI Replay",
                )
        windows = config.get("analysis_windows", {})
        if isinstance(windows, dict):
            for trial, window in windows.items():
                table.add_row(
                    f"{trial} Analysis Window",
                    f"{window.get('start_index', 0)}:{window.get('end_index_exclusive', 'end')}",
                    "GUI Replay",
                )

        panel = Panel(
            table,
            title=f"[bold blue]vailá Multimodal Toolbox — Treadmill LC (v{VERSION})[/bold blue]",
            subtitle="[dim]Zero Offset → Calibration Matrix → Coordinate Transformation → Signal Filtering → Event Detection[/dim]",
            border_style="blue",
        )
        console.print(panel)


def print_run_history_paths(paths: list[Path]) -> None:
    """Print replay instructions for the generated execution histories."""
    with contextlib.suppress(OSError, BrokenPipeError):
        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column("Saved Run History", style="green")
        table.add_column("Replay Command", style="cyan")
        for path in paths:
            table.add_row(str(path), f"uv run {CLI_SCRIPT_PATH} --config {path}")
        console.print(
            Panel(
                table,
                title="[bold blue]Reproducible TOML History[/bold blue]",
                border_style="blue",
            )
        )


def print_pipeline_milestone(
    stage_num: int,
    total_stages: int,
    name: str,
    details: str = "",
    status: str = "success",
) -> None:
    """Print pipeline progress and status lines via rich console."""
    with contextlib.suppress(OSError, BrokenPipeError):
        if status == "success":
            icon = "[bold green]✓[/bold green]"
            tag = f"[bold cyan][Stage {stage_num}/{total_stages}][/bold cyan]"
        elif status == "running":
            icon = "[bold yellow]⚙[/bold yellow]"
            tag = f"[bold yellow][Stage {stage_num}/{total_stages}][/bold yellow]"
        else:
            icon = "[bold red]✗[/bold red]"
            tag = f"[bold red][Stage {stage_num}/{total_stages}][/bold red]"

        msg = f"{icon} {tag} [bold]{name}[/bold]"
        if details:
            msg += f" — [dim]{details}[/dim]"
        console.print(msg)


# =============================================================================
# STAGE 1: ARTIFACT REMOVAL
# =============================================================================


def merge_intervals(intervals):
    """Merges overlapping or adjacent intervals into a single interval."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:  # overlaps or touches
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    return merged


def capture_clicks_with_undo(fig, ax, t, signal, color, parent=None):
    """Captures mouse clicks on a matplotlib figure for marking intervals."""
    # Caller must already be on TkAgg (_ensure_interactive_backend) before creating fig.
    points = []

    def redraw_markers():
        ax.clear()
        ax.plot(t, signal, color=color, lw=1.2)
        ax.set_title(
            "CELL - Left click START/END; Right: undo last; ENTER finishes\n"
            "Markers follow the processing-window style: vertical dashed START/END lines",
            fontsize=14,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (V)")
        ax.grid(True, alpha=0.4)

        for i, point in enumerate(points):
            interval_number = (i // 2) + 1
            label = f"START {interval_number}" if i % 2 == 0 else f"END {interval_number}"
            line_color = "green" if i % 2 == 0 else "red"
            ax.axvline(point, color=line_color, linestyle="--", linewidth=1.2, label=label)
        if points:
            ax.legend(loc="upper right")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.button == 1:  # Left: add
            if event.xdata is not None:
                points.append(event.xdata)
                redraw_markers()
        elif event.button == 3 and points:  # Right: undo last
            points.pop()
            redraw_markers()

    def on_key(event):
        if event.key == "enter":
            if len(points) % 2 == 1:
                messagebox.showwarning(
                    "Incomplete interval",
                    "Each interpolation segment needs START and END. "
                    "Add the missing END point or use right click to undo the last START.",
                    parent=parent,
                )
                return
            plt.close(fig)

    redraw_markers()
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show(block=True)
    plt.close(fig)
    return points


def plot_segments(ax, t, y, color, label, dt_tol=0.0011):
    """Plots a signal in segments, avoiding connecting lines across gaps."""
    diffs = np.diff(t)
    gaps = np.where(diffs > dt_tol)[0] + 1
    starts = np.insert(gaps, 0, 0)
    ends = np.append(gaps, len(t))
    for s, e in zip(starts, ends, strict=False):
        ax.plot(t[s:e], y[s:e], color=color, lw=1.5, label=label if s == 0 else None)


def reset_times(t_clean):
    """Resets timestamps to create a continuous time array."""
    if len(t_clean) == 0:
        return t_clean
    dt = np.median(np.diff(t_clean))
    t_reset = np.arange(len(t_clean)) * dt
    return t_reset


ADJUSTMENT_MODE_ALIASES = {
    "remove": "remove",
    "remover": "remove",
    "cut": "remove",
    "cortar": "remove",
    "nan": "nan",
    "null": "nan",
    "nulo": "nan",
    "nulos": "nan",
    "zero": "zero",
    "zeros": "zero",
    "neutral": "neutral_mean",
    "neutro": "neutral_mean",
    "neutral_mean": "neutral_mean",
    "media": "neutral_mean",
    "média": "neutral_mean",
    "mean": "neutral_mean",
    "linear": "linear",
    "linha": "linear",
}

ADJUSTMENT_MODE_LABELS = {
    "remove": "Remove segment and shorten signal",
    "nan": "Keep samples and set marked segment to NaN",
    "zero": "Keep samples and set marked segment to zero",
    "neutral_mean": "Keep samples and fill with neutral mean from boundary line",
    "linear": "Keep samples and fill with linear bridge between boundaries",
    "excluded": "Signal completely excluded from processing",
}


def normalize_adjustment_mode(mode):
    """Normalize a user/TOML adjustment mode name."""
    key = str(mode or "nan").strip().lower()
    if key not in ADJUSTMENT_MODE_ALIASES:
        valid = ", ".join(sorted(set(ADJUSTMENT_MODE_ALIASES.values())))
        raise ValueError(f"Invalid adjustment mode '{mode}'. Use one of: {valid}")
    return ADJUSTMENT_MODE_ALIASES[key]


def _finite_or_none(value):
    """Return a finite float or None."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if np.isfinite(value):
        return value
    return None


def _linear_bridge_values(cell_data, start, end, column):
    """Return replacement values between the closest valid boundary samples."""
    count = max(0, int(end) - int(start))
    if count == 0:
        return np.array([], dtype=float)

    before = _finite_or_none(cell_data[start - 1, column]) if start > 0 else None
    after = _finite_or_none(cell_data[end, column]) if end < len(cell_data) else None

    if before is not None and after is not None:
        return np.linspace(before, after, count + 2, dtype=float)[1:-1]
    if before is not None:
        return np.full(count, before, dtype=float)
    if after is not None:
        return np.full(count, after, dtype=float)

    channel_mean = np.nanmean(cell_data[:, column])
    if not np.isfinite(channel_mean):
        channel_mean = 0.0
    return np.full(count, float(channel_mean), dtype=float)


def _normalize_cells(cells, n_cols):
    """Normalize selected cell indices to zero-based column indices."""
    if cells is None:
        return list(range(n_cols))
    if isinstance(cells, (int, np.integer)):
        cells = [int(cells)]
    normalized = []
    for cell in cells:
        cell = int(cell)
        if 0 <= cell < n_cols:
            normalized.append(cell)
    return sorted(set(normalized))


def _parse_interval_spec(interval, n_cols):
    """Return start, end, and selected cells from tuple/list/dict interval specs."""
    if isinstance(interval, dict):
        start = interval.get("start_index", interval.get("start"))
        end = interval.get("end_index_exclusive", interval.get("end"))
        cells = interval.get("cells_0based", interval.get("cells", interval.get("cell")))
    else:
        start = interval[0]
        end = interval[1]
        cells = interval[2] if len(interval) > 2 else None
    return int(start), int(end), _normalize_cells(cells, n_cols)


def _normalize_adjustment_intervals(intervals, n_samples, n_cols):
    """Clip, merge per selected cell, and group intervals sharing start/end."""
    per_cell = {cell: [] for cell in range(n_cols)}
    for interval in intervals:
        start, end, cells = _parse_interval_spec(interval, n_cols)
        start = max(0, start)
        end = min(n_samples, end)
        if end <= start or not cells:
            continue
        for cell in cells:
            per_cell[cell].append((start, end))

    grouped = {}
    for cell, cell_intervals in per_cell.items():
        for start, end in merge_intervals(cell_intervals):
            grouped.setdefault((start, end), []).append(cell)

    return [(start, end, sorted(cells)) for (start, end), cells in sorted(grouped.items())]


def _replacement_values_for_cells(cell_data, start, end, mode, cells):
    """Build replacement values for selected cell columns only."""
    count = max(0, int(end) - int(start))
    cells = list(cells)
    if mode == "nan":
        return np.full((count, len(cells)), np.nan, dtype=float)
    if mode == "zero":
        return np.zeros((count, len(cells)), dtype=float)

    values = np.zeros((count, len(cells)), dtype=float)
    for out_col, cell in enumerate(cells):
        bridge = _linear_bridge_values(cell_data, start, end, cell)
        if mode == "linear":
            values[:, out_col] = bridge
        elif mode == "neutral_mean":
            neutral = np.nanmean(bridge)
            if not np.isfinite(neutral):
                neutral = 0.0
            values[:, out_col] = neutral
        else:
            raise ValueError(f"Unsupported replacement mode: {mode}")
    return values


def _interval_records(intervals, t, mode):
    """Create serializable interval records for audit/reuse."""
    records = []
    for start, end, cells in intervals:
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        end_inclusive = min(end - 1, len(t) - 1)
        records.append(
            {
                "start_index": start,
                "end_index_exclusive": end,
                "end_index_inclusive": end_inclusive,
                "start_time_s": float(t[start]),
                "end_time_s": float(t[end_inclusive]),
                "samples": int(end - start),
                "mode": mode,
                "cells_0based": list(cells),
                "cells_1based": [int(cell) + 1 for cell in cells],
                "cell_labels": [f"Cell {int(cell) + 1}" for cell in cells],
            }
        )
    return records


def apply_adjustment_intervals(t, cell_data, intervals, mode="nan"):
    """Apply marked artifact intervals using remove, NaN, zero, neutral_mean, or linear mode."""
    mode = normalize_adjustment_mode(mode)
    n_samples = len(t)
    n_cols = cell_data.shape[1]
    clipped = _normalize_adjustment_intervals(intervals, n_samples, n_cols)

    records = _interval_records(clipped, t, mode)
    if not clipped:
        return t.copy(), cell_data.copy(), records

    if mode == "remove":
        # Removing rows changes the shared time base, so this mode remains global even when
        # intervals were marked in only one cell. Use nan/zero/neutral/linear for cell-only edits.
        row_intervals = merge_intervals([(start, end) for start, end, _cells in clipped])
        good_mask = np.ones(n_samples, dtype=bool)
        for start, end in row_intervals:
            good_mask[start:end] = False
        return t[good_mask], cell_data[good_mask, :], records

    adjusted = cell_data.astype(float, copy=True)
    for start, end, cells in clipped:
        adjusted[start:end, cells] = _replacement_values_for_cells(
            adjusted, start, end, mode, cells
        )
    return t.copy(), adjusted, records


def save_adjustment_metadata(file_path, interval_records, mode, interpolation_metadata=None):
    """Save marked adjustment/interpolation intervals as JSON, TOML, and CSV sidecars."""
    if not interval_records:
        return []

    source = Path(file_path)
    interpolation_metadata = interpolation_metadata or {}
    processed = interpolation_metadata.get("processed", True)
    payload = {
        "source_file": source.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "adjustment_mode": mode,
        "mode_label": ADJUSTMENT_MODE_LABELS.get(mode, mode),
        "processed": processed,
        "interpolation": interpolation_metadata,
        "intervals": interval_records,
    }

    json_path = source.with_name(f"{source.stem}_adjust_intervals.json")
    toml_path = source.with_name(f"{source.stem}_adjust_intervals.toml")
    csv_path = source.with_name(f"{source.stem}_adjust_intervals.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(toml_path, "w", encoding="utf-8") as f:
        toml.dump(
            {
                "adjustment": {
                    k: v for k, v in payload.items() if k not in {"intervals", "interpolation"}
                },
                "interpolation": interpolation_metadata,
                "intervals": interval_records,
            },
            f,
        )
    pd.DataFrame(interval_records).to_csv(csv_path, index=False)
    return [str(json_path), str(toml_path), str(csv_path)]


def record_adjustment_in_config(
    config: dict[str, Any] | None,
    file_path: str | Path,
    interval_records: list[dict[str, Any]],
    mode: str,
    interpolation_metadata: dict[str, Any] | None = None,
) -> None:
    """Record GUI artifact choices in the unified config for headless replay."""
    if config is None:
        return
    interpolation = interpolation_metadata or {}
    key = Path(file_path).stem.lower()
    config.setdefault("adjustments", {})[key] = {
        "source_file": Path(file_path).name,
        "adjustment_mode": mode,
        "processed": bool(interpolation.get("processed", True)),
        "interpolation": interpolation,
        "intervals": interval_records,
    }


def clean_signal_with_clicks(file_path, parent=None, config: dict[str, Any] | None = None):
    """Mark artifacts and immediately choose the best interpolation method."""
    _ensure_interactive_backend()
    df = pd.read_csv(file_path, header=None)
    t = df[0].values
    cell_data = df.iloc[:, 1:5].values.astype(float)
    cell_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    try:
        while True:
            fig_overview, ax = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
            for i in range(4):
                plot_segments(ax[i], t, cell_data[:, i], cell_colors[i], None)
                ax[i].set_ylabel(f"Cell {i + 1}")
                ax[i].grid(True)
            plot_segments(ax[4], t, cell_data.sum(axis=1), "black", None)
            ax[4].set_ylabel("Sum")
            ax[4].set_xlabel("Time (s)")
            ax[4].grid(True)
            plt.suptitle(f"Overview - {os.path.basename(file_path)}", fontsize=14)
            plt.tight_layout()
            plt.show(block=False)

            if not ask_yes_no_english(
                "Artifact Adjustment + Interpolation",
                "Do you want to adjust or exclude signal from any of the load cells in this file?",
                parent=parent,
            ):
                plt.close(fig_overview)
                record_adjustment_in_config(
                    config,
                    file_path,
                    [],
                    "none",
                    {"status": "no_adjustment", "processed": True},
                )
                return None, cell_data, []

            is_interpolation = ask_yes_no_english(
                "Select Action Type",
                "Do you want to interpolate bad segments? (Click 'Yes' to interpolate segments, or 'No' to exclude/discard the load-cell signal entirely)",
                parent=parent,
            )

            selection_parent = parent or tk._default_root or tk.Tk()
            if parent is None and selection_parent is not tk._default_root:
                selection_parent.withdraw()
            cell_dialog = LoadCellSelectionDialog(selection_parent)
            selection_parent.wait_window(cell_dialog)
            selected_cells = cell_dialog.selected_cells if cell_dialog.finished else []
            plt.close(fig_overview)

            if not selected_cells:
                messagebox.showinfo(
                    "Info", "No load cells were selected. No adjustments applied.", parent=parent
                )
                record_adjustment_in_config(
                    config,
                    file_path,
                    [],
                    "none",
                    {"status": "no_adjustment", "processed": True},
                )
                return None, cell_data, []

            if not is_interpolation:
                messagebox.showwarning(
                    "Trial Excluded",
                    f"Trial {os.path.basename(file_path)} will not be processed because it needs more filtering adjustments.",
                    parent=parent,
                )
                interval_records = [
                    {
                        "start_index": 0,
                        "end_index_exclusive": len(t),
                        "cells_0based": selected_cells,
                        "mode": "excluded",
                    }
                ]
                metadata_paths = save_adjustment_metadata(
                    file_path,
                    interval_records,
                    "excluded",
                    interpolation_metadata={"status": "excluded", "processed": False},
                )
                record_adjustment_in_config(
                    config,
                    file_path,
                    interval_records,
                    "excluded",
                    {"status": "excluded", "processed": False},
                )
                return None, cell_data, metadata_paths

            marked_intervals = []

            for cell in selected_cells:
                fig_cell, ax_cell = plt.subplots(figsize=(16, 8))
                ax_cell.plot(t, cell_data[:, cell], color=cell_colors[cell], lw=1.2)
                ax_cell.set_title(
                    f"CELL {cell + 1} - Left click START/END; Right: undo last\n"
                    "Press ENTER to confirm only after every START has an END",
                    fontsize=14,
                )
                ax_cell.set_xlabel("Time (s)")
                ax_cell.set_ylabel("Voltage (V)")
                ax_cell.grid(True, alpha=0.4)
                plt.tight_layout()

                points = capture_clicks_with_undo(
                    fig_cell, ax_cell, t, cell_data[:, cell], cell_colors[cell], parent=parent
                )

                if len(points) == 0:
                    continue

                for i in range(0, len(points), 2):
                    t_start = min(points[i], points[i + 1])
                    t_end = max(points[i], points[i + 1])
                    idx_start = np.searchsorted(t, t_start, side="left")
                    idx_end = np.searchsorted(t, t_end, side="right")
                    if idx_end > idx_start:
                        marked_intervals.append(
                            {"start": idx_start, "end": idx_end, "cells": [cell]}
                        )

            plt.close("all")

            if not marked_intervals:
                messagebox.showinfo(
                    "Info", "No intervals were marked. Nothing will be adjusted.", parent=parent
                )
                record_adjustment_in_config(
                    config,
                    file_path,
                    [],
                    "none",
                    {"status": "no_adjustment", "processed": True},
                )
                return None, cell_data, []

            interp_config = get_default_interp_config()["interpolation"].copy()
            configured_interpolation = (config or {}).get("interpolation", {})
            if isinstance(configured_interpolation, dict):
                interp_config.update(configured_interpolation)
            _, gap_data, interval_records = apply_adjustment_intervals(
                t, cell_data, marked_intervals, mode="nan"
            )
            df_gaps = pd.DataFrame(gap_data)

            interp_names = {
                "linear": "Linear",
                "quadratic": "Quadratic",
                "cubic": "Cubic",
                "pchip": "PCHIP",
                "akima": "Akima Spline",
                "spline": f"Spline (order {interp_config['spline_order']})",
                "barycentric": "Global Poly (Lagrange)",
                "rbf": "RBF (Gaussian)",
                "nearest": "Nearest",
                "pad": "Previous (Pad)",
                "backfill": "Next (Backfill)",
            }

            while True:
                dialog_parent = parent or tk._default_root or tk.Tk()
                if parent is None and dialog_parent is not tk._default_root:
                    dialog_parent.withdraw()
                dialog = InterpDialog(
                    dialog_parent, max_selection=interp_config["max_comparison_methods"]
                )
                dialog_parent.wait_window(dialog)
                selected_methods = dialog.selected_methods or ["linear"]

                results_comparison = interpolate_dataframe_methods(
                    df_gaps,
                    selected_methods,
                    interp_config["spline_order"],
                    interp_config["rbf_window_size"],
                )

                # Part 1: Plotting individual selected cells before vs. after interpolation
                fig_cells, axes = plt.subplots(
                    len(selected_cells), 1, figsize=(12, 2.5 * len(selected_cells) + 2), sharex=True
                )
                if len(selected_cells) == 1:
                    axes = [axes]

                fig_cells.suptitle(
                    "Part 1: Selected Cells - Before vs. After Interpolation", fontsize=14
                )
                colors = ["red", "blue", "green", "orange", "purple", "brown"]

                for cell_idx, cell in enumerate(selected_cells):
                    ax = axes[cell_idx]
                    ax.plot(t, cell_data[:, cell], color="gray", alpha=0.3, label="Original")
                    ax.plot(
                        t, df_gaps.values[:, cell], color="black", alpha=0.5, label="Marked Gap"
                    )
                    for idx, (method, res) in enumerate(results_comparison.items()):
                        ax.plot(
                            t,
                            res[:, cell],
                            color=colors[idx % len(colors)],
                            label=f"Method: {interp_names.get(method, method)}",
                            alpha=0.8,
                            lw=1.5,
                        )
                    ax.set_ylabel(f"Cell {cell + 1} (V)")
                    ax.legend(loc="upper right")
                    ax.grid(True)

                axes[-1].set_xlabel("Time (s)")
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.show(block=True)
                plt.close(fig_cells)

                # Part 2: Plotting summed signals before vs. after interpolation
                fig_sum, ax_sum = plt.subplots(figsize=(12, 8))
                fig_sum.suptitle(
                    "Part 2: Summed Signals - Before vs. After Interpolation", fontsize=14
                )

                ax_sum.plot(
                    t, np.sum(cell_data, axis=1), color="gray", alpha=0.25, label="Original Sum"
                )
                ax_sum.plot(
                    t,
                    np.nansum(df_gaps.values, axis=1),
                    color="black",
                    alpha=0.45,
                    label="Marked Gap Sum",
                )

                for idx, (method, res) in enumerate(results_comparison.items()):
                    ax_sum.plot(
                        t,
                        np.sum(res, axis=1),
                        color=colors[idx % len(colors)],
                        label=f"Method: {interp_names.get(method, method)}",
                        alpha=0.8,
                        lw=1.5,
                    )
                ax_sum.set_xlabel("Time (s)")
                ax_sum.set_ylabel("Sum of Load Cells (V)")
                ax_sum.legend(loc="upper right")
                ax_sum.grid(True)
                plt.tight_layout()
                plt.show(block=True)
                plt.close(fig_sum)

                # Part 3: Ask for approval
                approved = ask_yes_no_english(
                    "Approve Interpolation Changes",
                    "Do you approve the interpolation changes shown in the graphs?\n\n"
                    "Yes = choose a final method to save.\n"
                    "No = retry selection and interval marking.",
                    parent=parent,
                )

                if not approved:
                    del results_comparison
                    gc.collect()
                    plt.close("all")
                    # Break the inner loop to redo cell and interval selection
                    break

                # User approved, let them choose the final method to save
                choice_parent = parent or tk._default_root or tk.Tk()
                if parent is None and choice_parent is not tk._default_root:
                    choice_parent.withdraw()
                choice_diag = FinalChoiceDialog(choice_parent, selected_methods, interp_names)
                choice_parent.wait_window(choice_diag)
                final_method = choice_diag.choice if choice_diag.choice else selected_methods[0]

                if final_method == "retry":
                    del results_comparison
                    gc.collect()
                    continue

                adjusted_cell_data = results_comparison[final_method].copy()
                interpolation_metadata = {
                    "status": "adjusted_and_interpolated",
                    "selected_methods": selected_methods,
                    "final_method": final_method,
                    "final_method_label": interp_names.get(final_method, final_method),
                    "max_comparison_methods": interp_config["max_comparison_methods"],
                    "spline_order": interp_config["spline_order"],
                    "rbf_window_size": interp_config["rbf_window_size"],
                }
                del results_comparison
                gc.collect()

                for record in interval_records:
                    record["mode"] = "adjusted_and_interpolated"
                metadata_paths = save_adjustment_metadata(
                    file_path,
                    interval_records,
                    "nan",
                    interpolation_metadata=interpolation_metadata,
                )
                record_adjustment_in_config(
                    config,
                    file_path,
                    interval_records,
                    "nan",
                    interpolation_metadata,
                )

                output_data = np.column_stack((t, adjusted_cell_data))
                output_name = file_path.replace(".csv", "_clean.csv")
                np.savetxt(
                    output_name, output_data, delimiter=",", fmt="%.8f", header="", comments=""
                )
                messagebox.showinfo(
                    "Completed",
                    f"Adjusted + interpolated signal saved:\n{output_name}\n\n"
                    "Interval and interpolation metadata saved as TOML/JSON/CSV.",
                    parent=parent,
                )
                return output_name, adjusted_cell_data, metadata_paths
    finally:
        plt.close("all")
        gc.collect()


def ask_yes_no_english(title, message, parent=None) -> bool:
    """Standard messagebox Yes/No dialog to match the rest of the project (e.g. tugturn)."""
    return bool(messagebox.askyesno(title, message, parent=parent))


def get_output_base_folder(folder):
    """
    Returns the base directory where output subdirectories should be placed.
    If the current folder's name matches a pipeline stage output pattern, we place
    outputs in its parent folder (the 'dxx' level) to avoid nested folders.
    Otherwise, we place outputs inside the folder itself.
    """
    folder_abs = os.path.abspath(folder)
    name = os.path.basename(folder_abs).lower()
    # Support both English and Portuguese prefixes to handle legacy and new folders
    prefixes = [
        "limpos",
        "clean",
        "cleaned",
        "ajustado",
        "adjusted",
        "filtrado",
        "filtered",
        "filter_analysis",
        "results",
        "figures",
    ]
    # Check if the folder name starts with one of the prefixes, possibly followed by a timestamp
    pattern = r"^(" + "|".join(prefixes) + r")(?:_\d{8}_\d{6})?(?:_\d+)?$"
    if re.match(pattern, name):
        return os.path.dirname(folder_abs)
    return folder_abs


def is_trial_file(filename: str) -> bool:
    """Return True only for load-cell running signal CSVs.

    Valid trial signal names are ``sXX_dXX_tXX.csv`` and the adjusted/interpolated
    ``sXX_dXX_tXX_LIMPO.csv`` legacy variant. Sidecar CSVs produced by adjustment,
    filtering, or processing must not enter downstream signal stages.
    """
    name_lower = Path(filename).name.lower()
    if not name_lower.endswith(".csv"):
        return False
    return re.fullmatch(r"s\d+_d\d+_t\d+(?:_limpo|_clean)?\.csv", name_lower) is not None


def canonical_trial_filename(filename: str) -> str:
    """Return the standard trial filename used between pipeline stages."""
    name = Path(filename).name
    match = re.fullmatch(r"(s\d+_d\d+_t\d+)(?:_limpo|_clean)?\.csv", name, flags=re.IGNORECASE)
    if not match:
        return name
    return f"{match.group(1).lower()}.csv"


def deduplicate_trial_files(files: list[str]) -> list[str]:
    """Keep one source file per canonical trial, preferring the standard filename."""
    selected: dict[str, str] = {}
    for file_name in sorted(files):
        canonical = canonical_trial_filename(file_name)
        current = selected.get(canonical)
        if current is None:
            selected[canonical] = file_name
            continue
        if (
            current.lower().endswith("_limpo.csv") or current.lower().endswith("_clean.csv")
        ) and file_name.lower() == canonical:
            print(f"Skipping duplicate legacy adjusted trial during filtering: {current}")
            selected[canonical] = file_name
        else:
            print(f"Skipping duplicate trial during filtering: {file_name}")
    return [selected[key] for key in sorted(selected)]


def is_tare_calibration_file(filename: str) -> bool:
    """Return True for tare/zero calibration CSVs (English and legacy Portuguese names)."""
    stem = Path(filename).stem.lower()
    return bool(re.fullmatch(r"s\d+_d\d+_(tare|tara)", stem))


def is_bodyweight_calibration_file(filename: str) -> bool:
    """Return True for participant-weight calibration CSVs (English and legacy names)."""
    stem = Path(filename).stem.lower()
    return bool(re.fullmatch(r"s\d+_d\d+_(weight|peso)", stem))


def is_calibration_file(filename: str) -> bool:
    """Return True for load-cell calibration CSV files only."""
    name_lower = Path(filename).name.lower()
    if not name_lower.endswith(".csv"):
        return False
    return (
        is_tare_calibration_file(name_lower)
        or is_bodyweight_calibration_file(name_lower)
        or re.search(r"s\d+_d\d+_\d+kg\.csv$", name_lower) is not None
    )


def is_subject_metadata_file(filename: str) -> bool:
    """Return True for Borg/info metadata files used to read body weight."""
    name_lower = Path(filename).name.lower()
    if not name_lower.endswith(".txt"):
        return False
    return "borg" in name_lower or name_lower.startswith("info_")


def discover_calibration_and_borg(folder, subject_str, day_str):
    """Search for calibration and Borg files in the current, parent, or grandparent directories."""
    base_dirs = [folder, os.path.dirname(folder), os.path.dirname(os.path.dirname(folder))]

    search_dirs = []
    for d in base_dirs:
        if not d or not os.path.exists(d):
            continue
        if d not in search_dirs:
            search_dirs.append(d)
        try:
            for item in os.listdir(d):
                item_path = os.path.join(d, item)
                if (
                    os.path.isdir(item_path)
                    and item.lower() in ["calib", "calibration", "calibracao"]
                    and item_path not in search_dirs
                ):
                    search_dirs.append(item_path)
        except Exception:
            continue

    tare_file = None
    weight_file = None
    plate_files = []
    borg_file = None

    subj_num = int(subject_str)
    day_num = int(day_str)

    # Robust regex patterns for subject/day
    subj_pat = f"s0*{subj_num}"
    day_pat = f"d0*{day_num}"
    prefix_regex = re.compile(rf"{subj_pat}_+{day_pat}", re.IGNORECASE)

    for d in search_dirs:
        if not d or not os.path.exists(d):
            continue
        try:
            for name in os.listdir(d):
                name_lower = name.lower()

                if is_subject_metadata_file(name) and prefix_regex.search(name_lower):
                    candidate = os.path.join(d, name)
                    if borg_file is None or (
                        "borg" in name_lower and "borg" not in Path(borg_file).name.lower()
                    ):
                        borg_file = candidate

                # Check if matches subject_day prefix
                if prefix_regex.search(name_lower) and name_lower.endswith(".csv"):
                    if is_tare_calibration_file(name) and not tare_file:
                        tare_file = os.path.join(d, name)
                    elif is_bodyweight_calibration_file(name) and not weight_file:
                        weight_file = os.path.join(d, name)
                    elif "kg" in name_lower:
                        full_path = os.path.join(d, name)
                        if full_path not in plate_files:
                            plate_files.append(full_path)
        except Exception as e:
            print(f"Error listing directory {d}: {e}")

    return tare_file, weight_file, plate_files, borg_file


METADATA_WEIGHT_COLUMNS = ("Weight", "Peso", "peso", "weight", "BodyWeight", "body_weight")


def _first_matching_column(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    """Return the first matching column name, ignoring surrounding whitespace and case."""
    stripped = {str(col).strip(): col for col in df.columns}
    lowered = {str(col).strip().lower(): col for col in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name in stripped:
            return stripped[name]
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def get_group_weight_from_borg(borg_path):
    """Parse a Borg/info metadata file and return body weight from the first row."""
    try:
        if not os.path.exists(borg_path):
            return None

        df = None
        weight_col = None
        for sep in [",", ";", "\t"]:
            try:
                temp_df = pd.read_csv(borg_path, sep=sep)
                temp_df.columns = [c.strip() for c in temp_df.columns]
                weight_col = _first_matching_column(temp_df, METADATA_WEIGHT_COLUMNS)
                if weight_col is not None:
                    df = temp_df
                    break
            except Exception:
                continue

        if df is None or weight_col is None:
            print(f"Metadata file {borg_path} missing Weight column.")
            return None

        if not df.empty:
            return float(df[weight_col].iloc[0])
    except Exception as e:
        print(f"Error parsing metadata file {borg_path}: {e}")
    return None


def infer_subject_weight_kg(folder: str | Path) -> float | None:
    """Infer the first subject/day body weight available in an input folder."""
    folder_path = Path(folder)
    trial_files = sorted(path.name for path in folder_path.iterdir() if is_trial_file(path.name))
    subject_day = None
    if trial_files:
        match = re.search(r"s(\d+)_d(\d+)", trial_files[0], re.IGNORECASE)
        if match:
            subject_day = (match.group(1), match.group(2))

    borg_path = None
    if subject_day:
        _, _, _, borg_path = discover_calibration_and_borg(
            str(folder_path), subject_day[0], subject_day[1]
        )
    if borg_path:
        weight = get_group_weight_from_borg(borg_path)
        if weight is not None:
            return float(weight)

    for path in sorted(folder_path.iterdir()):
        if is_subject_metadata_file(path.name):
            weight = get_group_weight_from_borg(str(path))
            if weight is not None:
                return float(weight)
    return None


def calibration_center_slice(df, window_seconds=5.0, fs=FS):
    """Return the central calibration rows, avoiding start/end transients."""
    if df.empty:
        return df

    n_rows = len(df)
    if n_rows <= 1 or window_seconds <= 0:
        return df

    time_values = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    finite_time = np.isfinite(time_values)
    if finite_time.all() and n_rows > 2:
        diffs = np.diff(time_values)
        if np.all(diffs > 0):
            duration = time_values[-1] - time_values[0]
            if duration > window_seconds:
                center = (time_values[0] + time_values[-1]) / 2.0
                start_t = center - window_seconds / 2.0
                end_t = center + window_seconds / 2.0
                mask = (time_values >= start_t) & (time_values <= end_t)
                if np.any(mask):
                    return df.loc[mask].reset_index(drop=True)

    window_samples = int(round(window_seconds * fs))
    if window_samples <= 0 or n_rows <= window_samples:
        return df

    start = (n_rows - window_samples) // 2
    end = start + window_samples
    return df.iloc[start:end].reset_index(drop=True)


def read_calibration_cells(csv_path, window_seconds=5.0, fs=FS):
    """Read calibration cells using only the central stable window."""
    df = pd.read_csv(csv_path, sep=",", header=None)
    df_center = calibration_center_slice(df, window_seconds=window_seconds, fs=fs)
    cells = -1 * df_center[[1, 2, 3, 4]].to_numpy(dtype=float)
    if len(df_center) != len(df):
        print(
            f"   Calibration window: {os.path.basename(csv_path)} -> "
            f"{len(df_center)}/{len(df)} samples ({window_seconds:g}s center)"
        )
    return cells


def make_timestamped_output_dir(parent_folder, prefix):
    """Create a timestamped output folder without overwriting previous runs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(parent_folder, f"{prefix}_{timestamp}")
    suffix = 1
    while os.path.exists(path):
        path = os.path.join(parent_folder, f"{prefix}_{timestamp}_{suffix:02d}")
        suffix += 1
    os.makedirs(path, exist_ok=False)
    return path


def make_output_dir(parent_folder, prefix, timestamp_output: bool = False):
    """Create a stage directory, reusing it unless timestamp mode is requested."""
    if timestamp_output:
        return make_timestamped_output_dir(parent_folder, prefix)

    path = os.path.join(parent_folder, prefix)
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)
    os.makedirs(path, exist_ok=True)
    return path


def run_adjust_stage(
    parent=None,
    initial_dir=None,
    config: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    headless: bool = False,
) -> str | None:
    """Executes the artifact adjustment stage, ignoring calibration files."""
    folder = initial_dir or (
        filedialog.askdirectory(title="Select folder with CSV files", parent=parent)
        if not headless
        else None
    )
    if not folder:
        return None

    base_folder = output_dir or get_output_base_folder(folder)
    timestamp_output = bool((config or {}).get("execution", {}).get("timestamp_output", False))
    output_folder = make_output_dir(base_folder, "clean", timestamp_output)

    trial_files = []
    calibration_files = []

    for f in os.listdir(folder):
        name_lower = f.lower()
        if not name_lower.endswith(".csv") or "limpo" in name_lower or "clean" in name_lower:
            continue
        if is_calibration_file(f):
            calibration_files.append(f)
        elif is_trial_file(f):
            trial_files.append(f)
        else:
            print(f"Skipping non-trial/non-calibration CSV during adjustment: {f}")

    trial_files.sort()
    calibration_files.sort()

    if not trial_files and not calibration_files:
        if not headless:
            messagebox.showerror("Error", "No CSV files found in the folder.", parent=parent)
        return None
    # Process trial files interactively. The downstream filter stage expects one
    # homogeneous batch, so every trial is written to the clean folder with its original name.
    for file in trial_files:
        file_path = os.path.join(folder, file)
        output_path = os.path.join(output_folder, file)
        print(f"\nProcessing trial: {file}")
        try:
            if headless:
                # Prefer GUI-recorded choices embedded in the run-history TOML;
                # fall back to legacy sidecars beside the input trial.
                meta = get_recorded_adjustment(config, file_path) or load_adjustment_metadata(
                    file_path
                )
                if meta:
                    replayed = replay_adjustment_from_metadata(file_path, meta)
                    np.savetxt(output_path, replayed, delimiter=",", fmt="%.8f")
                    if meta.get("intervals"):
                        save_adjustment_metadata(
                            output_path,
                            meta["intervals"],
                            str(meta.get("adjustment_mode", "nan")),
                            interpolation_metadata=meta.get("interpolation", {}),
                        )
                    print(f"Replayed GUI adjustment metadata for: {file}")
                else:
                    shutil.copy2(file_path, output_path)
                    print(f"Headless mode: copied trial without interactive clicking: {file}")
            else:
                saved, _, metadata_paths = clean_signal_with_clicks(
                    file_path, parent=parent, config=config
                )
                if saved:
                    shutil.move(saved, output_path)
                    print(f"Saved adjusted/interpolated trial for filtering: {file}")
                else:
                    shutil.copy2(file_path, output_path)
                    print(f"No adjustment applied. Copied unchanged trial for filtering: {file}")
                for metadata_path in metadata_paths:
                    if metadata_path and os.path.exists(metadata_path):
                        shutil.copy2(
                            metadata_path,
                            os.path.join(output_folder, os.path.basename(metadata_path)),
                        )
        except Exception as e:
            if not headless:
                messagebox.showerror("Error", f"Failed on {file}:\n{e}", parent=parent)
            print(e)

    # Copy calibration files directly
    for file in calibration_files:
        src_path = os.path.join(folder, file)
        dest_path = os.path.join(output_folder, file)
        print(f"Copying calibration file directly: {file}")
        shutil.copy2(src_path, dest_path)

    if not headless:
        messagebox.showinfo(
            "Finished",
            f"All files processed!\nCleaned files are in:\n{output_folder}",
            parent=parent,
        )
    return output_folder


# =============================================================================
# STAGE 2: INTERPOLATION & ADJUSTMENT
# =============================================================================


def get_default_interp_config():
    """Get default configuration dictionary for interpolation settings."""
    return {
        "interpolation": {
            "max_comparison_methods": 4,
            "spline_order": 3,
            "rbf_window_size": 200,
        }
    }


def save_interp_config(config, filepath):
    """Save configuration settings to a TOML file."""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            toml.dump(config, f)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False


def load_interp_config(filepath):
    """Load configuration settings from a TOML file."""
    try:
        if not os.path.exists(filepath):
            return None
        with open(filepath, encoding="utf-8") as f:
            toml_config = toml.load(f)
        config = get_default_interp_config()
        if "interpolation" in toml_config:
            interp = toml_config["interpolation"]
            config["interpolation"].update(
                {
                    "max_comparison_methods": int(interp.get("max_comparison_methods", 4)),
                    "spline_order": int(interp.get("spline_order", 3)),
                    "rbf_window_size": int(interp.get("rbf_window_size", 200)),
                }
            )
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return None


class InterpConfigDialog(simpledialog.Dialog):
    """Dialog to configure interpolation parameters, with TOML support."""

    def __init__(self, parent):
        self.loaded_config = None
        self.use_toml = False
        self.toml_path = None
        super().__init__(parent, title="Interpolation Configuration")

    def body(self, master):
        params_frame = tk.LabelFrame(master, text="Interpolation Parameters", padx=10, pady=10)
        params_frame.pack(fill="both", expand=True, padx=10, pady=5)

        tk.Label(params_frame, text="Max Comparison Methods:").grid(
            row=0, column=0, sticky="e", pady=2
        )
        tk.Label(params_frame, text="Spline Order:").grid(row=1, column=0, sticky="e", pady=2)
        tk.Label(params_frame, text="RBF Window Size (points):").grid(
            row=2, column=0, sticky="e", pady=2
        )

        self.max_methods_entry = tk.Entry(params_frame)
        self.max_methods_entry.insert(0, "4")
        self.spline_order_entry = tk.Entry(params_frame)
        self.spline_order_entry.insert(0, "3")
        self.rbf_window_entry = tk.Entry(params_frame)
        self.rbf_window_entry.insert(0, "200")

        self.max_methods_entry.grid(row=0, column=1, pady=2, padx=5)
        self.spline_order_entry.grid(row=1, column=1, pady=2, padx=5)
        self.rbf_window_entry.grid(row=2, column=1, pady=2, padx=5)

        toml_frame = tk.LabelFrame(master, text="Advanced Configuration (TOML)", padx=10, pady=10)
        toml_frame.pack(fill="both", expand=True, padx=10, pady=5)

        btns_frame = tk.Frame(toml_frame)
        btns_frame.pack()
        tk.Button(btns_frame, text="Load Configuration TOML", command=self.load_config_file).pack(
            side="left", padx=5
        )
        tk.Button(
            btns_frame,
            text="Create Default TOML Template",
            command=self.create_default_toml_template,
        ).pack(side="left", padx=5)

        self.toml_label = tk.Label(toml_frame, text="No TOML loaded", fg="gray")
        self.toml_label.pack(pady=5)

        return self.max_methods_entry

    def create_default_toml_template(self):
        file_path = filedialog.asksaveasfilename(
            parent=self.master,
            title="Create Default TOML Configuration Template",
            defaultextension=".toml",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
            initialfile="interp_config_template.toml",
        )
        if file_path:
            default_config = get_default_interp_config()
            if save_interp_config(default_config, file_path):
                messagebox.showinfo(
                    "Template Created",
                    f"Default TOML template created successfully:\n{file_path}",
                    parent=self,
                )
            else:
                messagebox.showerror("Error", "Failed to create template file.", parent=self)

    def load_config_file(self):
        file_path = filedialog.askopenfilename(
            parent=self.master,
            title="Select TOML file",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
        )
        if file_path:
            config = load_interp_config(file_path)
            if config:
                self.loaded_config = config
                self.use_toml = True
                self.toml_path = file_path
                self.toml_label.config(
                    text=f"TOML loaded: {os.path.basename(file_path)}", fg="green"
                )
                self.populate_fields_from_config(config)
                messagebox.showinfo(
                    "TOML Parameters Loaded", "Configuration loaded successfully!", parent=self
                )
            else:
                self.toml_label.config(text="Error loading TOML", fg="red")
                messagebox.showerror("Error", "Failed to load TOML.", parent=self)

    def populate_fields_from_config(self, config):
        interp = config.get("interpolation", {})
        self.max_methods_entry.delete(0, tk.END)
        self.max_methods_entry.insert(0, str(interp.get("max_comparison_methods", 4)))
        self.spline_order_entry.delete(0, tk.END)
        self.spline_order_entry.insert(0, str(interp.get("spline_order", 3)))
        self.rbf_window_entry.delete(0, tk.END)
        self.rbf_window_entry.insert(0, str(interp.get("rbf_window_size", 200)))

    def apply(self):
        if self.use_toml and self.loaded_config:
            self.result = self.loaded_config
            self.result["interpolation"]["max_comparison_methods"] = int(
                self.max_methods_entry.get()
            )
            self.result["interpolation"]["spline_order"] = int(self.spline_order_entry.get())
            self.result["interpolation"]["rbf_window_size"] = int(self.rbf_window_entry.get())
        else:
            self.result = {
                "interpolation": {
                    "max_comparison_methods": int(self.max_methods_entry.get()),
                    "spline_order": int(self.spline_order_entry.get()),
                    "rbf_window_size": int(self.rbf_window_entry.get()),
                }
            }


class InterpDialog(tk.Toplevel):
    """Custom dialog for selecting interpolation methods via buttons."""

    def __init__(self, parent, max_selection=4):
        super().__init__(parent)
        self.title("Select Interpolation Methods")
        self.max_selection = max_selection
        self.selected_methods = []
        self.finished = False

        self.methods = [
            ("Quadratic", "quadratic"),
            ("PCHIP", "pchip"),
            ("Akima Spline", "akima"),
            ("Global Poly", "barycentric"),
            ("RBF (Gaussian)", "rbf"),
            ("Nearest", "nearest"),
            ("Previous", "pad"),
            ("Next", "backfill"),
            ("Linear", "linear"),
            ("Cubic", "cubic"),
            ("Spline", "spline"),
        ]

        tk.Label(
            self,
            text=f"Select up to {max_selection} methods for comparison:",
            font=("Arial", 12, "bold"),
        ).pack(pady=10)

        self.buttons = {}
        grid_frame = tk.Frame(self)
        grid_frame.pack(padx=20, pady=10)

        for i, (name, key) in enumerate(self.methods):
            btn = tk.Button(
                grid_frame,
                text=name,
                width=20,
                height=2,
                command=lambda k=key: self.toggle_method(k),
            )
            btn.grid(row=i // 2, column=i % 2, padx=5, pady=5)
            self.buttons[key] = btn

        self.default_bg = self.buttons["linear"].cget("bg")
        self.default_fg = self.buttons["linear"].cget("fg")

        self.confirm_btn = tk.Button(
            self,
            text=f"Confirm Selection (0/{max_selection})",
            state=tk.DISABLED,
            command=self.confirm,
            width=30,
            height=2,
            bg="#2c3e50",
            fg="white",
        )
        self.confirm_btn.pack(pady=20)

        self.geometry(f"+{parent.winfo_rootx() + 50}+{parent.winfo_rooty() + 50}")
        self.grab_set()

    def toggle_method(self, key):
        if key in self.selected_methods:
            self.selected_methods.remove(key)
            self.buttons[key].config(bg=self.default_bg, fg=self.default_fg)
        elif len(self.selected_methods) < self.max_selection:
            self.selected_methods.append(key)
            self.buttons[key].config(bg="#3498db", fg="white")

        self.confirm_btn.config(
            text=f"Confirm Selection ({len(self.selected_methods)}/{self.max_selection})"
        )
        if len(self.selected_methods) > 0:
            self.confirm_btn.config(state=tk.NORMAL, bg="#27ae60")
        else:
            self.confirm_btn.config(state=tk.DISABLED, bg="#2c3e50")

    def confirm(self):
        self.finished = True
        self.destroy()


class LoadCellSelectionDialog(tk.Toplevel):
    """Dialog for selecting which load cells need artifact adjustment."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Select Load Cells for Adjustment")
        self.selected_cells = []
        self.finished = False
        self.cell_options = [
            ("Cell 1", 0),
            ("Cell 2", 1),
            ("Cell 3", 2),
            ("Cell 4", 3),
        ]

        tk.Label(
            self,
            text="Select the load cells that need artifact adjustment:",
            font=("Arial", 12, "bold"),
        ).pack(pady=10)

        self.buttons = {}
        grid_frame = tk.Frame(self)
        grid_frame.pack(padx=20, pady=10)

        for i, (name, key) in enumerate(self.cell_options):
            btn = tk.Button(
                grid_frame,
                text=name,
                width=20,
                height=2,
                command=lambda k=key: self.toggle_cell(k),
            )
            btn.grid(row=i // 2, column=i % 2, padx=5, pady=5)
            self.buttons[key] = btn

        self.default_bg = self.buttons[0].cget("bg")
        self.default_fg = self.buttons[0].cget("fg")

        self.confirm_btn = tk.Button(
            self,
            text="Confirm Selection (0/4)",
            state=tk.DISABLED,
            command=self.confirm,
            width=30,
            height=2,
            bg="#2c3e50",
            fg="white",
        )
        self.confirm_btn.pack(pady=(15, 5))

        tk.Button(
            self,
            text="Cancel",
            command=self.destroy,
            width=30,
            height=2,
        ).pack(pady=(0, 20))

        self.geometry(f"+{parent.winfo_rootx() + 50}+{parent.winfo_rooty() + 50}")
        self.grab_set()

    def toggle_cell(self, key):
        if key in self.selected_cells:
            self.selected_cells.remove(key)
            self.buttons[key].config(bg=self.default_bg, fg=self.default_fg)
        else:
            self.selected_cells.append(key)
            self.buttons[key].config(bg="#3498db", fg="white")

        self.selected_cells.sort()
        self.confirm_btn.config(text=f"Confirm Selection ({len(self.selected_cells)}/4)")
        if self.selected_cells:
            self.confirm_btn.config(state=tk.NORMAL, bg="#27ae60")
        else:
            self.confirm_btn.config(state=tk.DISABLED, bg="#2c3e50")

    def confirm(self):
        self.finished = True
        self.destroy()


class FinalChoiceDialog(tk.Toplevel):
    """Custom dialog for choosing the final interpolation method."""

    def __init__(self, parent, choices, method_names):
        super().__init__(parent)
        self.title("Select Final Method")
        self.choice = None

        tk.Label(
            self,
            text="Choose the final interpolation method to apply:",
            font=("Arial", 12, "bold"),
        ).pack(pady=10)

        for key in choices:
            name = method_names.get(key, key)
            tk.Button(
                self,
                text=name,
                width=30,
                height=2,
                command=lambda k=key: self.select(k),
                bg="#34495e",
                fg="white",
            ).pack(pady=5, padx=20)

        tk.Button(
            self,
            text="None of these / Try again",
            width=30,
            height=2,
            command=lambda: self.select("retry"),
            bg="#c0392b",
            fg="white",
        ).pack(pady=15, padx=20)

        self.geometry(f"+{parent.winfo_rootx() + 50}+{parent.winfo_rooty() + 50}")
        self.grab_set()

    def select(self, key):
        self.choice = key
        self.destroy()


def apply_rbf_interp(df, column_idx, window_size=200):
    """Applies Radial Basis Function interpolation to a single column."""
    y = df[column_idx].copy()
    if hasattr(y, "values"):
        y = y.values

    nan_mask = np.isnan(y)
    if not np.any(nan_mask):
        return y

    x = np.arange(len(y))
    nan_indices = np.where(nan_mask)[0]
    if len(nan_indices) == 0:
        return y

    blocks = np.split(nan_indices, np.where(np.diff(nan_indices) != 1)[0] + 1)

    for block in blocks:
        start_idx = block[0]
        end_idx = block[-1]

        before_mask = ~nan_mask[:start_idx]
        before_valid = np.where(before_mask)[0]
        if len(before_valid) > window_size:
            before_valid = before_valid[-window_size:]

        after_mask = ~nan_mask[end_idx + 1 :]
        after_valid = np.where(after_mask)[0] + end_idx + 1
        if len(after_valid) > window_size:
            after_valid = after_valid[:window_size]

        local_valid_idx = np.concatenate([before_valid, after_valid])
        if len(local_valid_idx) == 0:
            continue

        x_valid = x[local_valid_idx]
        y_valid = y[local_valid_idx]

        try:
            rbf = Rbf(x_valid, y_valid, function="gaussian")
            y[block] = rbf(x[block])
        except Exception as e:
            print(f"RBF failed for gap {start_idx}-{end_idx}: {e}")
            y[block] = np.interp(x[block], x_valid, y_valid)

    return y


def _base_trial_stem_from_adjusted(file_path):
    """Return trial stem without the English or legacy Portuguese adjustment suffix."""
    stem = Path(file_path).stem
    lower = stem.lower()
    if lower.endswith("_limpo"):
        return stem[: -len("_limpo")]
    if lower.endswith("_clean"):
        return stem[: -len("_clean")]
    return stem


def find_adjustment_metadata_file(file_path):
    """Find adjustment interval sidecar for a clean/legacy trial file."""
    folder = Path(file_path).parent
    base = _base_trial_stem_from_adjusted(file_path)
    for suffix in ["json", "toml", "csv"]:
        candidate = folder / f"{base}_adjust_intervals.{suffix}"
        if candidate.exists():
            return candidate
    return None


def load_adjustment_metadata(file_path):
    """Load JSON/TOML/CSV adjustment metadata for a trial file, if present."""
    sidecar = find_adjustment_metadata_file(file_path)
    if sidecar is None:
        return None

    if sidecar.suffix.lower() == ".json":
        with open(sidecar, encoding="utf-8") as f:
            payload = json.load(f)
        payload["metadata_file"] = str(sidecar)
        if "processed" not in payload:
            payload["processed"] = True
        return payload

    if sidecar.suffix.lower() == ".toml":
        payload = toml.load(sidecar)
        adjustment = payload.get("adjustment", {})
        return {
            "source_file": adjustment.get("source_file", ""),
            "adjustment_mode": adjustment.get("adjustment_mode", ""),
            "processed": adjustment.get("processed", True),
            "interpolation": payload.get("interpolation", {}),
            "intervals": payload.get("intervals", []),
            "metadata_file": str(sidecar),
        }

    df = pd.read_csv(sidecar)
    intervals = []
    for record in df.to_dict("records"):
        cells = record.get("cells_0based")
        if isinstance(cells, str):
            cells = [int(x) for x in re.findall(r"\d+", cells)]
        elif pd.isna(cells):
            cells = None
        intervals.append(
            {
                "start_index": int(record["start_index"]),
                "end_index_exclusive": int(record["end_index_exclusive"]),
                "mode": record.get("mode", ""),
                "cells_0based": cells,
            }
        )
    return {
        "source_file": "",
        "adjustment_mode": "",
        "intervals": intervals,
        "metadata_file": str(sidecar),
    }


def _source_file_for_removed_adjustment(file_path, metadata):
    """Find original source CSV when the adjusted file had removed rows."""
    source_name = metadata.get("source_file") or f"{_base_trial_stem_from_adjusted(file_path)}.csv"
    folder = Path(file_path).parent
    candidates = [folder.parent / source_name, folder / source_name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _records_require_removed_source(records):
    """Return True if any sidecar interval came from remove mode."""
    return any(str(record.get("mode", "")).lower() == "remove" for record in records)


def adjustment_metadata_to_interval_specs(metadata):
    """Convert approved adjustment metadata to interval specs accepted by apply_adjustment_intervals."""
    interval_specs = []
    for record in metadata.get("intervals", []):
        start = record.get("start_index", record.get("start"))
        end = record.get("end_index_exclusive", record.get("end"))
        if start is None or end is None:
            continue
        interval_specs.append(
            {
                "start": int(start),
                "end": int(end),
                "cells": record.get("cells_0based", record.get("cells", record.get("cell"))),
            }
        )
    return interval_specs


def apply_adjustment_metadata_as_nan(df_filtered, metadata):
    """Set approved adjustment intervals to NaN using the shared adjustment engine."""
    interval_specs = adjustment_metadata_to_interval_specs(metadata)
    if not interval_specs:
        return []

    t_dummy = np.arange(len(df_filtered), dtype=float)
    _, adjusted, records = apply_adjustment_intervals(
        t_dummy,
        df_filtered.to_numpy(dtype=float, copy=True),
        interval_specs,
        mode="nan",
    )
    df_filtered.iloc[:, :] = adjusted
    return [
        (record["start_index"], record["end_index_exclusive"], record["cells_0based"])
        for record in records
    ]


def interpolate_dataframe_methods(df_filtered, selected_methods, spline_order, rbf_window):
    """Apply selected interpolation methods to a dataframe copy for comparison."""
    results_comparison = {}
    for method in selected_methods:
        df_temp = df_filtered.copy()
        try:
            if method == "barycentric" and len(df_temp) > 1000:
                print(
                    f"Warning: {method} interpolation memory-intensive for >1000 points. Using cubic fallback."
                )
                df_temp.interpolate(method="cubic", axis=0, inplace=True, limit_direction="both")
            elif method == "spline":
                df_temp.interpolate(
                    method="spline",
                    order=spline_order,
                    axis=0,
                    inplace=True,
                    limit_direction="both",
                )
            elif method == "rbf":
                for idx in range(df_temp.shape[1]):
                    df_temp[idx] = apply_rbf_interp(df_temp, idx, window_size=rbf_window)
            else:
                df_temp.interpolate(method=method, axis=0, inplace=True, limit_direction="both")
            results_comparison[method] = df_temp.values
        except Exception as e:
            print(f"Error in {method}: {e}")
            df_temp.interpolate(method="linear", axis=0, inplace=True, limit_direction="both")
            results_comparison[method] = df_temp.values
        del df_temp
        gc.collect()
    return results_comparison


def get_recorded_adjustment(
    config: dict[str, Any] | None, file_path: str | Path
) -> dict[str, Any] | None:
    """Return GUI-recorded adjustment metadata for a trial, if available."""
    adjustments = (config or {}).get("adjustments", {})
    if not isinstance(adjustments, dict):
        return None
    path = Path(file_path)
    keys = [path.stem.lower(), Path(canonical_trial_filename(path.name)).stem.lower()]
    for key in keys:
        value = adjustments.get(key)
        if isinstance(value, dict):
            return value
    return None


def replay_adjustment_from_metadata(file_path: str | Path, metadata: dict[str, Any]) -> np.ndarray:
    """Reproduce GUI artifact/interpolation choices without opening dialogs."""
    df = pd.read_csv(file_path, sep=",", header=None)
    t = df.iloc[:, 0].to_numpy(dtype=float)
    cells = df.iloc[:, 1:5].to_numpy(dtype=float)
    intervals = adjustment_metadata_to_interval_specs(metadata)
    interpolation = metadata.get("interpolation", {})

    if metadata.get("processed") is False or metadata.get("adjustment_mode") == "excluded":
        return df.to_numpy()
    if not intervals or metadata.get("adjustment_mode") == "none":
        return df.to_numpy()

    if interpolation.get("status") == "adjusted_and_interpolated":
        final_method = str(interpolation.get("final_method", "")).strip()
        if not final_method:
            raise ValueError(f"Recorded interpolation method is missing for {Path(file_path).name}")
        _, gap_cells, _ = apply_adjustment_intervals(t, cells, intervals, mode="nan")
        results = interpolate_dataframe_methods(
            pd.DataFrame(gap_cells),
            [final_method],
            int(interpolation.get("spline_order", 3)),
            int(interpolation.get("rbf_window_size", 200)),
        )
        return np.column_stack((t, results[final_method]))

    mode = normalize_adjustment_mode(str(metadata.get("adjustment_mode", "nan")))
    adjusted_t, adjusted_cells, _ = apply_adjustment_intervals(t, cells, intervals, mode=mode)
    return np.column_stack((adjusted_t, adjusted_cells))


def preprocess_file_interp(file_path, config, fs=1000, root=None, headless=False):
    """Compatibility pass for sidecars that still need interpolation."""
    max_selection = config["interpolation"]["max_comparison_methods"]
    spline_order = config["interpolation"]["spline_order"]
    rbf_window = config["interpolation"]["rbf_window_size"]

    metadata = load_adjustment_metadata(file_path)
    if metadata is None:
        print(f"No adjustment metadata for {os.path.basename(file_path)}. Copying unchanged.")
        df = pd.read_csv(file_path, sep=",", header=None)
        return df.values, df.iloc[:, 1:5].values, df[0].values, False

    if metadata.get("processed") is False or metadata.get("adjustment_mode") == "excluded":
        print(f"{os.path.basename(file_path)} marked as excluded/not processed. Copying unchanged.")
        df = pd.read_csv(file_path, sep=",", header=None)
        return df.values, df.iloc[:, 1:5].values, df[0].values, False

    if metadata.get("interpolation", {}).get("status") == "adjusted_and_interpolated":
        print(f"{os.path.basename(file_path)} already adjusted+interpolated. Copying unchanged.")
        df = pd.read_csv(file_path, sep=",", header=None)
        return df.values, df.iloc[:, 1:5].values, df[0].values, False

    records = metadata.get("intervals", [])
    source_path = Path(file_path)
    if _records_require_removed_source(records):
        original_source = _source_file_for_removed_adjustment(file_path, metadata)
        if original_source is None:
            warning = (
                f"{source_path.name} used remove mode, but the original source file was not "
                "found. The shortened file will be copied unchanged."
            )
            if headless:
                print(f"Interpolation skipped: {warning}")
            else:
                messagebox.showwarning("Interpolation skipped", warning, parent=root)
            df = pd.read_csv(file_path, sep=",", header=None)
            return df.values, df.iloc[:, 1:5].values, df[0].values, False
        source_path = original_source
        print(f"Using original source for removed intervals: {source_path}")

    df = pd.read_csv(source_path, sep=",", header=None)
    t = df[0].values
    raw = df[[1, 2, 3, 4]].values.astype(float)
    df_filtered = pd.DataFrame(raw.copy())
    applied_intervals = apply_adjustment_metadata_as_nan(df_filtered, metadata)

    if not applied_intervals or not df_filtered.isna().any().any():
        print(
            f"No valid interpolation intervals for {os.path.basename(file_path)}. Copying unchanged."
        )
        return df.values, raw, t, False

    if headless:
        print(
            f"No recorded final interpolation for {os.path.basename(file_path)}. "
            "Headless compatibility pass copied the adjusted signal unchanged."
        )
        current_df = pd.read_csv(file_path, sep=",", header=None)
        return (
            current_df.to_numpy(),
            current_df.iloc[:, 1:5].to_numpy(),
            current_df.iloc[:, 0].to_numpy(),
            False,
        )

    try:
        _ensure_interactive_backend()
        fig_raw, ax_raw = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
        for i, ch in enumerate(["Cell 1", "Cell 2", "Cell 3", "Cell 4"]):
            ax_raw[i].plot(t, raw[:, i], label="Input")
            ax_raw[i].plot(t, df_filtered[i], label="Marked gaps", alpha=0.7)
            ax_raw[i].set_ylabel(f"{ch} (V)")
            ax_raw[i].grid(True)
            ax_raw[i].legend()
        ax_raw[4].plot(t, np.sum(raw, axis=1), label="Input Sum")
        ax_raw[4].plot(t, np.nansum(df_filtered.values, axis=1), label="Marked Sum", alpha=0.7)
        ax_raw[4].set_xlabel("Time (s)")
        ax_raw[4].legend()
        ax_raw[4].grid(True)
        plt.suptitle(
            f"Interpolation gaps from adjustment metadata - {os.path.basename(file_path)}",
            fontsize=14,
        )
        plt.tight_layout()
        plt.show(block=False)

        if root is None:
            root = tk.Tk()
            root.withdraw()

        interp_names = {
            "linear": "Linear",
            "quadratic": "Quadratic",
            "cubic": "Cubic",
            "pchip": "PCHIP",
            "akima": "Akima Spline",
            "spline": f"Spline (order {spline_order})",
            "barycentric": "Global Poly (Lagrange)",
            "rbf": "RBF (Gaussian)",
            "nearest": "Nearest",
            "pad": "Previous (Pad)",
            "backfill": "Next (Backfill)",
        }

        while True:
            dialog = InterpDialog(root, max_selection=max_selection)
            root.wait_window(dialog)
            if not dialog.selected_methods:
                dialog.selected_methods = ["linear"]

            selected_methods = dialog.selected_methods
            results_comparison = interpolate_dataframe_methods(
                df_filtered, selected_methods, spline_order, rbf_window
            )

            fig_comp, ax_comp = plt.subplots(figsize=(12, 8))
            fig_comp.suptitle("Comparison of Summed Signals (Post-Interpolation)", fontsize=16)
            raw_sum = np.sum(raw, axis=1)
            marked_sum = np.nansum(df_filtered.values, axis=1)
            ax_comp.plot(t, raw_sum, color="gray", alpha=0.25, label="Input Sum", linewidth=1)
            ax_comp.plot(
                t,
                marked_sum,
                color="black",
                alpha=0.45,
                label="Marked gaps Sum",
                linewidth=1,
            )
            colors = ["red", "blue", "green", "orange", "purple", "brown"]
            for idx, (method, res) in enumerate(results_comparison.items()):
                res_sum = np.sum(res, axis=1)
                ax_comp.plot(
                    t,
                    res_sum,
                    color=colors[idx % len(colors)],
                    label=f"Method: {interp_names.get(method, method)}",
                    alpha=0.8,
                )
            ax_comp.set_xlabel("Time (s)")
            ax_comp.set_ylabel("Sum of Load Cells (V)")
            ax_comp.legend()
            ax_comp.grid(True)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show(block=True)
            plt.close(fig_comp)

            choice_diag = FinalChoiceDialog(root, selected_methods, interp_names)
            root.wait_window(choice_diag)
            final_method = choice_diag.choice if choice_diag.choice else selected_methods[0]

            if final_method == "retry":
                del results_comparison
                gc.collect()
                continue

            filtered = results_comparison[final_method].copy()
            messagebox.showinfo(
                "Final Selection",
                f"Applied: {interp_names.get(final_method, final_method)}",
                parent=root,
            )
            del results_comparison
            gc.collect()
            break

        plt.close(fig_raw)

        fig, ax = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
        fig.suptitle("Final Result: Input (gray) vs. Interpolated (blue)", fontsize=16)
        for i in range(4):
            ax[i].plot(t, raw[:, i], "gray", alpha=0.6, label="Input")
            ax[i].plot(t, filtered[:, i], "blue", label="Interpolated")
            ax[i].set_ylabel(f"Cell {i + 1}")
            ax[i].legend()
            ax[i].grid(True)
        ax[4].plot(t, np.sum(raw, axis=1), "gray", alpha=0.6, label="Input Sum")
        ax[4].plot(t, np.sum(filtered, axis=1), "blue", label="Interpolated Sum")
        ax[4].set_xlabel("Time (s)")
        ax[4].legend()
        ax[4].grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        plt.close(fig)

        save = ask_yes_no_english("Save", "Save final interpolated signal?", parent=root)
        if save:
            res = np.column_stack((t, filtered))
            return res, filtered, t, True

        return None, filtered, t, True
    finally:
        plt.close("all")
        gc.collect()


def run_interpolate_stage(
    parent=None,
    initial_dir=None,
    config: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    headless: bool = False,
) -> str | None:
    """Executes sidecar-driven multi-method interpolation for adjusted files."""
    folder = initial_dir or (
        filedialog.askdirectory(title="Folder with CSVs", parent=parent) if not headless else None
    )
    if not folder:
        return None

    base_folder = output_dir or get_output_base_folder(folder)
    timestamp_output = bool((config or {}).get("execution", {}).get("timestamp_output", False))
    path_interp = make_output_dir(base_folder, "adjusted", timestamp_output)

    if config and ("interpolation" in config or "spline_order" in config):
        interp_config = config if "interpolation" in config else {"interpolation": config}
    elif not headless:
        dialog = InterpConfigDialog(parent)
        if not dialog.result:
            return None
        interp_config = dialog.result
    else:
        interp_config = get_default_interp_config()

    config_save_path = os.path.join(path_interp, "interpolation_configuration_used.toml")
    save_interp_config(interp_config, config_save_path)

    try:
        trial_files = []
        calibration_files = []
        for f in os.listdir(folder):
            if not f.lower().endswith(".csv"):
                continue
            if is_calibration_file(f):
                calibration_files.append(f)
            elif is_trial_file(f):
                trial_files.append(f)
        trial_files.sort()
        calibration_files.sort()

        # Process trials
        for f in trial_files:
            print(f"Processing trial file: {f}...")
            fp = os.path.join(folder, f)
            saved, _, _, did_interpolate = preprocess_file_interp(
                fp,
                interp_config,
                root=parent,
                headless=headless,
            )
            if saved is not None:
                if did_interpolate:
                    print(f"Saving interpolated: {f}...")
                else:
                    print(f"Copying unchanged trial without interpolation: {f}...")
                np.savetxt(os.path.join(path_interp, f), saved, delimiter=",", fmt="%.8f")
            else:
                print(f"Interpolation of {f} skipped or cancelled.")
            plt.close("all")
            gc.collect()

        # Copy calibration files directly
        for f in calibration_files:
            src_path = os.path.join(folder, f)
            dest_path = os.path.join(path_interp, f)
            print(f"Copying calibration file directly: {f}")
            shutil.copy2(src_path, dest_path)
    except Exception as e:
        if not headless:
            messagebox.showerror(
                "Error", f"An error occurred: {e}\n\nCheck the console for details.", parent=parent
            )
        print(e)
        return None
    finally:
        plt.close("all")
        if not headless:
            messagebox.showinfo(
                "Completed", "Processing completed. Check 'adjusted' folder.", parent=parent
            )

    return path_interp


# =============================================================================
# STAGE 3: ADVANCED FILTERING
# =============================================================================


def get_default_filter_config():
    """Get default configuration dictionary for advanced filtering."""
    return {
        "filters": {
            "median_window": 5,
            "filter_type": "lowpass",
            "lowpass_cutoff": 40.0,
            "bandpass_lowcut": 0.0,
            "bandpass_highcut": 40.0,
            "filter_order": 4,
            "edge_mode": "nearest",
        }
    }


def normalize_filter_config(config):
    """Return a complete filter config, preserving compatibility with old TOMLs."""
    defaults = get_default_filter_config()
    filters_in = (config or {}).get("filters", {})
    filters = defaults["filters"].copy()

    filters["median_window"] = int(filters_in.get("median_window", filters["median_window"]))
    if filters["median_window"] < 1:
        filters["median_window"] = 1
    if filters["median_window"] > 1 and filters["median_window"] % 2 == 0:
        filters["median_window"] += 1

    filter_type = str(filters_in.get("filter_type", "lowpass")).lower().strip()
    aliases = {"band": "bandpass", "low": "lowpass", "high": "highpass", "off": "none"}
    filter_type = aliases.get(filter_type, filter_type)
    if filter_type not in {"lowpass", "bandpass", "highpass", "median", "none"}:
        filter_type = "lowpass"
    filters["filter_type"] = filter_type

    filters["bandpass_lowcut"] = float(
        filters_in.get("bandpass_lowcut", filters["bandpass_lowcut"])
    )
    filters["bandpass_highcut"] = float(
        filters_in.get("bandpass_highcut", filters["bandpass_highcut"])
    )
    filters["lowpass_cutoff"] = float(filters_in.get("lowpass_cutoff", filters["bandpass_highcut"]))
    filters["filter_order"] = int(filters_in.get("filter_order", filters["filter_order"]))

    edge_mode = str(filters_in.get("edge_mode", filters["edge_mode"])).lower().strip()
    if edge_mode not in {"reflect", "constant", "nearest", "mirror", "wrap"}:
        edge_mode = "nearest"
    filters["edge_mode"] = edge_mode

    return {"filters": filters}


def save_filter_config(config, filepath):
    """Save configuration settings to a TOML file."""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            toml.dump(config, f)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False


def load_filter_config(filepath):
    """Load configuration settings from a TOML file."""
    try:
        if not os.path.exists(filepath):
            return None
        with open(filepath, encoding="utf-8") as f:
            toml_config = toml.load(f)
        return normalize_filter_config(toml_config)
    except Exception as e:
        print(f"Error loading config: {e}")
        return None


class FilterConfigDialog(simpledialog.Dialog):
    """Dialog to configure load-cell filters, with TOML support."""

    def __init__(self, parent):
        self.loaded_config = None
        self.use_toml = False
        self.toml_path = None
        super().__init__(parent, title="Filter Configuration")

    def body(self, master):
        params_frame = tk.LabelFrame(master, text="Filter Parameters", padx=10, pady=10)
        params_frame.pack(fill="both", expand=True, padx=10, pady=5)

        tk.Label(params_frame, text="Filter Type:").grid(row=0, column=0, sticky="e", pady=2)
        tk.Label(params_frame, text="Median Window (odd number):").grid(
            row=1, column=0, sticky="e", pady=2
        )
        tk.Label(params_frame, text="Edge Mode:").grid(row=2, column=0, sticky="e", pady=2)
        tk.Label(params_frame, text="Low-Pass Cutoff (Hz):").grid(
            row=3, column=0, sticky="e", pady=2
        )
        tk.Label(params_frame, text="Band-Pass Lowcut (Hz):").grid(
            row=4, column=0, sticky="e", pady=2
        )
        tk.Label(params_frame, text="Band-Pass Highcut (Hz):").grid(
            row=5, column=0, sticky="e", pady=2
        )
        tk.Label(params_frame, text="Filter Order:").grid(row=6, column=0, sticky="e", pady=2)

        defaults = get_default_filter_config()["filters"]
        self.filter_type_var = tk.StringVar(value=defaults["filter_type"])
        self.edge_mode_var = tk.StringVar(value=defaults["edge_mode"])
        self.filter_type_menu = ttk.Combobox(
            params_frame,
            textvariable=self.filter_type_var,
            values=("lowpass", "bandpass", "highpass", "median", "none"),
            state="readonly",
            width=16,
        )
        self.edge_mode_menu = ttk.Combobox(
            params_frame,
            textvariable=self.edge_mode_var,
            values=("nearest", "reflect", "mirror", "constant", "wrap"),
            state="readonly",
            width=16,
        )
        self.median_win_entry = tk.Entry(params_frame)
        self.median_win_entry.insert(0, str(defaults["median_window"]))
        self.lowpass_entry = tk.Entry(params_frame)
        self.lowpass_entry.insert(0, str(defaults["lowpass_cutoff"]))
        self.lowcut_entry = tk.Entry(params_frame)
        self.lowcut_entry.insert(0, str(defaults["bandpass_lowcut"]))
        self.highcut_entry = tk.Entry(params_frame)
        self.highcut_entry.insert(0, str(defaults["bandpass_highcut"]))
        self.order_entry = tk.Entry(params_frame)
        self.order_entry.insert(0, str(defaults["filter_order"]))

        self.filter_type_menu.grid(row=0, column=1, pady=2, padx=5, sticky="w")
        self.median_win_entry.grid(row=1, column=1, pady=2, padx=5)
        self.edge_mode_menu.grid(row=2, column=1, pady=2, padx=5, sticky="w")
        self.lowpass_entry.grid(row=3, column=1, pady=2, padx=5)
        self.lowcut_entry.grid(row=4, column=1, pady=2, padx=5)
        self.highcut_entry.grid(row=5, column=1, pady=2, padx=5)
        self.order_entry.grid(row=6, column=1, pady=2, padx=5)

        toml_frame = tk.LabelFrame(master, text="Advanced Configuration (TOML)", padx=10, pady=10)
        toml_frame.pack(fill="both", expand=True, padx=10, pady=5)

        btns_frame = tk.Frame(toml_frame)
        btns_frame.pack()
        tk.Button(btns_frame, text="Load Configuration TOML", command=self.load_config_file).pack(
            side="left", padx=5
        )
        tk.Button(
            btns_frame,
            text="Create Default TOML Template",
            command=self.create_default_toml_template,
        ).pack(side="left", padx=5)

        self.toml_label = tk.Label(toml_frame, text="No TOML loaded", fg="gray")
        self.toml_label.pack(pady=5)

        return self.median_win_entry

    def create_default_toml_template(self):
        file_path = filedialog.asksaveasfilename(
            parent=self.master,
            title="Create Default TOML Configuration Template",
            defaultextension=".toml",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
            initialfile="filter_config_template.toml",
        )
        if file_path:
            default_config = get_default_filter_config()
            if save_filter_config(default_config, file_path):
                messagebox.showinfo(
                    "Template Created",
                    f"Default TOML template created successfully:\n{file_path}",
                    parent=self,
                )
            else:
                messagebox.showerror("Error", "Failed to create template file.", parent=self)

    def load_config_file(self):
        file_path = filedialog.askopenfilename(
            parent=self.master,
            title="Select TOML file",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
        )
        if file_path:
            config = load_filter_config(file_path)
            if config:
                self.loaded_config = config
                self.use_toml = True
                self.toml_path = file_path
                self.toml_label.config(
                    text=f"TOML loaded: {os.path.basename(file_path)}", fg="green"
                )
                self.populate_fields_from_config(config)
                messagebox.showinfo(
                    "TOML Parameters Loaded", "Configuration loaded successfully!", parent=self
                )
            else:
                self.toml_label.config(text="Error loading TOML", fg="red")
                messagebox.showerror("Error", "Failed to load TOML.", parent=self)

    def populate_fields_from_config(self, config):
        filters = normalize_filter_config(config).get("filters", {})
        self.filter_type_var.set(filters["filter_type"])
        self.edge_mode_var.set(filters["edge_mode"])
        self.median_win_entry.delete(0, tk.END)
        self.median_win_entry.insert(0, str(filters["median_window"]))
        self.lowpass_entry.delete(0, tk.END)
        self.lowpass_entry.insert(0, str(filters["lowpass_cutoff"]))
        self.lowcut_entry.delete(0, tk.END)
        self.lowcut_entry.insert(0, str(filters["bandpass_lowcut"]))
        self.highcut_entry.delete(0, tk.END)
        self.highcut_entry.insert(0, str(filters["bandpass_highcut"]))
        self.order_entry.delete(0, tk.END)
        self.order_entry.insert(0, str(filters["filter_order"]))

    def apply(self):
        med_win = int(self.median_win_entry.get())
        if med_win < 1:
            med_win = 1
        if med_win > 1 and med_win % 2 == 0:
            med_win += 1
            messagebox.showwarning(
                "Warning", f"Median window must be odd. Adjusted to {med_win}.", parent=self
            )

        self.result = normalize_filter_config(
            {
                "filters": {
                    "median_window": med_win,
                    "filter_type": self.filter_type_var.get(),
                    "lowpass_cutoff": float(self.lowpass_entry.get()),
                    "bandpass_lowcut": float(self.lowcut_entry.get()),
                    "bandpass_highcut": float(self.highcut_entry.get()),
                    "filter_order": int(self.order_entry.get()),
                    "edge_mode": self.edge_mode_var.get(),
                }
            }
        )


def _median_filter_1d(signal, window=5, edge_mode="nearest"):
    """Apply edge-safe median filtering to a 1D signal."""
    window = int(window)
    if window <= 1:
        return np.asarray(signal, dtype=float).copy()
    if window % 2 == 0:
        window += 1
    return median_filter(np.asarray(signal, dtype=float), size=window, mode=edge_mode)


def _butter_sos(filter_type, fs, order=4, cutoff=None, lowcut=None, highcut=None):
    """Create a Butterworth SOS filter for stable zero-phase filtering."""
    nyq = 0.5 * fs
    if filter_type == "lowpass":
        normalized = float(cutoff) / nyq
        if not 0 < normalized < 1:
            raise ValueError(f"lowpass cutoff must be between 0 and {nyq:g} Hz")
        return butter(order, normalized, btype="lowpass", output="sos")
    if filter_type == "highpass":
        normalized = float(cutoff) / nyq
        if not 0 < normalized < 1:
            raise ValueError(f"highpass cutoff must be between 0 and {nyq:g} Hz")
        return butter(order, normalized, btype="highpass", output="sos")
    if filter_type == "bandpass":
        low = float(lowcut) / nyq
        high = float(highcut) / nyq
        if not 0 < low < high < 1:
            raise ValueError(f"bandpass cutoffs must satisfy 0 < low < high < {nyq:g} Hz")
        return butter(order, [low, high], btype="bandpass", output="sos")
    raise ValueError(f"Unsupported Butterworth filter type: {filter_type}")


def apply_filter(signal, filter_type="lowpass", fs=1000, **kwargs):
    """Apply an edge-safe selected filter type to a single load-cell signal."""
    filter_type = str(filter_type or "lowpass").lower().strip()
    aliases = {"band": "bandpass", "low": "lowpass", "high": "highpass", "off": "none"}
    filter_type = aliases.get(filter_type, filter_type)
    if filter_type not in {"lowpass", "bandpass", "highpass", "median", "none"}:
        filter_type = "lowpass"

    sig = np.asarray(signal, dtype=float)
    if filter_type == "none":
        return sig.copy()

    median_window = kwargs.get("median_window", kwargs.get("window", 5))
    edge_mode = kwargs.get("edge_mode", "nearest")
    sig = _median_filter_1d(sig, window=median_window, edge_mode=edge_mode)
    if filter_type == "median":
        return sig

    order = int(kwargs.get("order", 4))
    if filter_type == "lowpass":
        sos = _butter_sos(
            "lowpass",
            fs,
            order=order,
            cutoff=kwargs.get("lowpass_cutoff", kwargs.get("highcut", 40.0)),
        )
    elif filter_type == "highpass":
        sos = _butter_sos(
            "highpass",
            fs,
            order=order,
            cutoff=kwargs.get("highpass_cutoff", kwargs.get("lowcut", 0.5)),
        )
    else:
        sos = _butter_sos(
            "bandpass",
            fs,
            order=order,
            lowcut=kwargs.get("lowcut", 0.5),
            highcut=kwargs.get("highcut", 40.0),
        )
    return sosfiltfilt(sos, sig)


def analyze_spectrum_filt(cells, timestamps, file_name, analysis_path, fs=1000):
    """Performs frequency domain analysis on load cell signals."""
    _ensure_noninteractive_backend()
    try:
        channels = ["Cell 1", "Cell 2", "Cell 3", "Cell 4"]
        metrics = {}

        for i in range(4):
            signal = cells[:, i]
            N = len(signal)
            yf = fft(signal)
            xf = fftfreq(N, 1 / fs)[: N // 2]
            amp = 2.0 / N * np.abs(yf[0 : N // 2])
            f_psd, psd = welch(signal, fs=fs, nperseg=min(1024, N))

            height = 0.001 * np.max(amp) if np.max(amp) > 0 else 1e-6
            peaks, _ = find_peaks(amp, height=height)
            freqs = xf[peaks][:10]
            amps = amp[peaks][:10]

            metrics[channels[i]] = {
                "dominant_freqs": freqs.tolist(),
                "dominant_amps": amps.tolist(),
                "total_energy": float(np.sum(psd)),
                "variance": float(np.var(signal)),
                "snr_db": float(
                    10 * np.log10(np.mean(signal**2) / np.var(signal))
                    if np.var(signal) > 0
                    else np.inf
                ),
            }

            fig, ax = plt.subplots(2, 1, figsize=(12, 8))
            ax[0].plot(xf, amp)
            ax[0].set_title(f"Spectrum - {channels[i]}")
            ax[0].set_xlabel("Frequency (Hz)")
            ax[0].set_ylabel("Amplitude")
            ax[0].grid(True)

            ax[1].plot(f_psd, 10 * np.log10(psd))
            ax[1].set_title(f"PSD - {channels[i]}")
            ax[1].set_xlabel("Frequency (Hz)")
            ax[1].set_ylabel("PSD (dB/Hz)")
            ax[1].grid(True)

            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    analysis_path,
                    f"{Path(file_name).stem}_filter_{channels[i].replace(' ', '_')}_spectrum.png",
                )
            )
            plt.close(fig)

        # Sum of cells
        sum_sig = np.sum(cells, axis=1)
        N = len(sum_sig)
        yf_sum = fft(sum_sig)
        amp_sum = 2.0 / N * np.abs(yf_sum[0 : N // 2])
        f_psd_sum, psd_sum = welch(sum_sig, fs=fs, nperseg=min(1024, N))
        height_sum = 0.001 * np.max(amp_sum) if np.max(amp_sum) > 0 else 1e-6
        peaks_sum, _ = find_peaks(amp_sum, height=height_sum)

        metrics["Sum"] = {
            "dominant_freqs": xf[peaks_sum][:10].tolist(),
            "dominant_amps": amp_sum[peaks_sum][:10].tolist(),
            "total_energy": float(np.sum(psd_sum)),
            "variance": float(np.var(sum_sig)),
            "snr_db": float(
                10 * np.log10(np.mean(sum_sig**2) / np.var(sum_sig))
                if np.var(sum_sig) > 0
                else np.inf
            ),
        }

        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        ax[0].plot(xf, amp_sum)
        ax[0].set_title("Spectrum - Sum")
        ax[0].set_xlabel("Frequency (Hz)")
        ax[0].set_ylabel("Amplitude")
        ax[0].grid(True)

        ax[1].plot(f_psd_sum, 10 * np.log10(psd_sum))
        ax[1].set_title("PSD - Sum")
        ax[1].set_xlabel("Frequency (Hz)")
        ax[1].set_ylabel("PSD (dB/Hz)")
        ax[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(analysis_path, f"{Path(file_name).stem}_filter_sum_spectrum.png"))
        plt.close(fig)

        pd.DataFrame.from_dict(metrics, orient="index").to_csv(
            os.path.join(analysis_path, f"{Path(file_name).stem}_filter_spectrum_metrics.csv")
        )
    finally:
        plt.close("all")
        gc.collect()


def preprocess_file_filt(file_path, config, fs=1000, root=None, preview=True, confirm_message=None):
    """Main preprocessing function for a single load cell data file (Filtering)."""
    config = normalize_filter_config(config)
    filters = config["filters"]
    median_win = filters["median_window"]
    filter_type = filters["filter_type"]
    lowpass_cutoff = filters["lowpass_cutoff"]
    lowcut = filters["bandpass_lowcut"]
    highcut = filters["bandpass_highcut"]
    order = filters["filter_order"]
    edge_mode = filters["edge_mode"]

    print(
        f"Filtering {os.path.basename(file_path)} with "
        f"type={filter_type}, median_window={median_win}, edge_mode={edge_mode}, "
        f"lowpass_cutoff={lowpass_cutoff}, bandpass_lowcut={lowcut}, "
        f"bandpass_highcut={highcut}, order={order}"
    )

    df = pd.read_csv(file_path, sep=",", header=None)
    t = df[0].values
    raw = df[[1, 2, 3, 4]].values
    filtered = raw.copy()

    for i in range(4):
        filtered_sig = apply_filter(
            filtered[:, i],
            filter_type=filter_type,
            fs=fs,
            median_window=median_win,
            edge_mode=edge_mode,
            lowpass_cutoff=lowpass_cutoff,
            lowcut=lowcut,
            highcut=highcut,
            order=order,
        )
        filtered[:, i] = filtered_sig

    try:
        if preview:
            _ensure_interactive_backend()
            fig, ax = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
            fig.suptitle("Final Result: Raw (gray) vs. Processed (blue)", fontsize=16)
            for i in range(4):
                ax[i].plot(t, raw[:, i], "gray", alpha=0.6, label="Raw")
                ax[i].plot(t, filtered[:, i], "blue", label="Processed (Filt)")
                ax[i].set_ylabel(f"Cell {i + 1}")
                ax[i].legend()
                ax[i].grid(True)
            ax[4].plot(t, np.sum(raw, axis=1), "gray", alpha=0.6, label="Raw Sum")
            ax[4].plot(t, np.sum(filtered, axis=1), "blue", label="Processed Sum")
            ax[4].set_xlabel("Time (s)")
            ax[4].legend()
            ax[4].grid(True)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
            plt.close(fig)

            message = confirm_message or "Save final processed signal?"
            save = ask_yes_no_english("Batch Filter Approval", message, parent=root)
            if not save:
                return None, filtered, t

        return np.column_stack((t, filtered)), filtered, t
    finally:
        plt.close("all")
        gc.collect()


def run_filter_stage(
    parent=None,
    initial_dir=None,
    config: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    headless: bool = False,
) -> str | None:
    """Executes edge-safe filtering and spectral FFT/PSD analysis."""
    folder = initial_dir or (
        filedialog.askdirectory(title="Folder with CSVs", parent=parent) if not headless else None
    )
    if not folder:
        return None

    base_folder = output_dir or get_output_base_folder(folder)
    timestamp_output = bool((config or {}).get("execution", {}).get("timestamp_output", False))
    path_filt = make_output_dir(base_folder, "filtered", timestamp_output)
    analysis_path = make_output_dir(base_folder, "filter_analysis", timestamp_output)

    if config and ("filters" in config or "filter_type" in config):
        filter_config = config if "filters" in config else {"filters": config}
    elif not headless:
        dialog = FilterConfigDialog(parent)
        if not dialog.result:
            return None
        filter_config = dialog.result
    else:
        filter_config = get_default_filter_config()

    config_save_path = os.path.join(analysis_path, "filtering_configuration_used.toml")
    save_filter_config(filter_config, config_save_path)

    try:
        csv_files = sorted([x for x in os.listdir(folder) if x.lower().endswith(".csv")])
        calibration_files = [f for f in csv_files if is_calibration_file(f)]
        trial_candidates = [f for f in csv_files if is_trial_file(f)]
        trial_files = deduplicate_trial_files(trial_candidates)
        skipped_files = [
            f for f in csv_files if f not in calibration_files and f not in trial_candidates
        ]
        for f in skipped_files:
            print(f"Skipping non-trial/non-calibration CSV during filtering: {f}")

        batches = [
            ("calibration", "calibration files", calibration_files),
            ("running", "running trial files", trial_files),
        ]
        for category_key, category_label, files in batches:
            if not files:
                continue
            print(f"Filtering {len(files)} {category_label}...")
            category_approved = False
            for idx, f in enumerate(files):
                preview = (idx == 0) and not headless and (_matplotlib_backend_name() != "agg")
                print(f"Processing {category_key} file: {f}...")
                fp = os.path.join(folder, f)
                confirm_message = None
                if preview:
                    confirm_message = (
                        f"Approve this filter result for the {category_label}?\n\n"
                        f"Preview file: {f}\n"
                        f"If approved, the same filter settings will be applied automatically "
                        f"to all {len(files)} {category_label} without more plot windows."
                    )
                saved, processed, t = preprocess_file_filt(
                    fp,
                    filter_config,
                    root=parent,
                    preview=preview,
                    confirm_message=confirm_message,
                )
                if preview and saved is None:
                    print(f"Batch filtering cancelled for {category_label} after preview: {f}")
                    break
                if preview:
                    category_approved = True
                if not preview and not category_approved and not headless:
                    pass

                if processed is not None and t is not None:
                    output_name = canonical_trial_filename(f) if category_key == "running" else f
                    print(f"Analyzing spectrum of {f}...")
                    analyze_spectrum_filt(processed, t, output_name, analysis_path)
                    if saved is not None:
                        print(f"Saving filtered: {output_name}...")
                        np.savetxt(
                            os.path.join(path_filt, output_name),
                            saved,
                            delimiter=",",
                            fmt="%.8f",
                        )
                else:
                    print(f"Processing of {f} skipped or cancelled.")
                plt.close("all")
                gc.collect()
    except Exception as e:
        if not headless:
            messagebox.showerror(
                "Error", f"An error occurred: {e}\n\nCheck the console for details.", parent=parent
            )
        print(e)
        return None
    finally:
        plt.close("all")
        if not headless:
            messagebox.showinfo(
                "Completed",
                f"Filtering completed.\n\nFiltered data: {path_filt}\nFilter analysis: {analysis_path}",
                parent=parent,
            )

    return path_filt


# =============================================================================
# STAGE 4: BIOMECHANICAL ANALYSIS & CONFIGURATION
# =============================================================================


def get_default_process_config():
    """Get default configuration dictionary for processing settings."""
    return {
        "processing": {
            "participant_weight_kg": 70.0,
            "participant_weight_source": "fallback",
            "use_advanced_calibration": False,
            "filter_cutoff_hz": 50.0,
            "apply_processing_filter": False,
            "detection_threshold_bw": 0.1,
            "generate_figures": True,
            "generate_interactive_report": True,
        }
    }


def get_default_unified_config() -> dict[str, Any]:
    """Get complete unified configuration dictionary for all treadmill stages."""
    return {
        "paths": {
            "input_dir": "",
            "output_dir": "",
        },
        "execution": {
            "step": "all",
            "timestamp_output": False,
        },
        "adjustments": {},
        "analysis_windows": {},
        "filters": get_default_filter_config()["filters"],
        "interpolation": get_default_interp_config()["interpolation"],
        "processing": get_default_process_config()["processing"],
    }


def save_process_config(config, filepath):
    """Save configuration settings to a TOML file."""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            toml.dump(config, f)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False


def load_process_config(filepath):
    """Load configuration settings from a TOML file."""
    try:
        if not os.path.exists(filepath):
            return None
        with open(filepath, encoding="utf-8") as f:
            toml_config = toml.load(f)
        config = get_default_process_config()
        if "processing" in toml_config:
            proc = toml_config["processing"]
            config["processing"].update(
                {
                    "participant_weight_kg": float(proc.get("participant_weight_kg", 70.0)),
                    "use_advanced_calibration": bool(proc.get("use_advanced_calibration", False)),
                    "filter_cutoff_hz": float(proc.get("filter_cutoff_hz", 50.0)),
                    "apply_processing_filter": bool(proc.get("apply_processing_filter", False)),
                    "detection_threshold_bw": float(proc.get("detection_threshold_bw", 0.1)),
                    "generate_figures": bool(proc.get("generate_figures", True)),
                    "generate_interactive_report": bool(
                        proc.get("generate_interactive_report", True)
                    ),
                }
            )
            if "participant_weight_kg" in proc and "participant_weight_source" not in proc:
                config["processing"]["participant_weight_source"] = "toml"
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return None


def load_unified_config(filepath: str | Path) -> dict[str, Any]:
    """Load configuration settings from a TOML file (supports full or partial configs)."""
    config = get_default_unified_config()
    try:
        p = Path(filepath)
        if not p.exists():
            return config
        with open(p, encoding="utf-8") as f:
            toml_data = toml.load(f)

        if "paths" in toml_data and isinstance(toml_data["paths"], dict):
            config["paths"].update(toml_data["paths"])
        if "execution" in toml_data and isinstance(toml_data["execution"], dict):
            config["execution"].update(toml_data["execution"])
        if "adjustments" in toml_data and isinstance(toml_data["adjustments"], dict):
            config["adjustments"].update(toml_data["adjustments"])
        if "analysis_windows" in toml_data and isinstance(toml_data["analysis_windows"], dict):
            config["analysis_windows"].update(toml_data["analysis_windows"])
        if "filters" in toml_data and isinstance(toml_data["filters"], dict):
            config["filters"].update(toml_data["filters"])
        if "interpolation" in toml_data and isinstance(toml_data["interpolation"], dict):
            config["interpolation"].update(toml_data["interpolation"])
        if "processing" in toml_data and isinstance(toml_data["processing"], dict):
            processing = dict(toml_data["processing"])
            if "weight" in processing:
                processing["participant_weight_kg"] = float(processing.pop("weight"))
                processing.setdefault("participant_weight_source", "toml")
            config["processing"].update(processing)
            if (
                "participant_weight_kg" in processing
                and "participant_weight_source" not in processing
            ):
                config["processing"]["participant_weight_source"] = "toml"

        # Map root-level properties if defined without table headers
        for k in [
            "participant_weight_kg",
            "filter_cutoff_hz",
            "detection_threshold_bw",
            "use_advanced_calibration",
            "apply_processing_filter",
            "generate_figures",
            "generate_interactive_report",
        ]:
            if k in toml_data:
                config["processing"][k] = toml_data[k]
        if "weight" in toml_data:
            config["processing"]["participant_weight_kg"] = float(toml_data["weight"])
            config["processing"]["participant_weight_source"] = "toml"
        elif "participant_weight_kg" in toml_data:
            config["processing"]["participant_weight_source"] = "toml"
        return config
    except Exception as e:
        print(f"Error loading unified config from {filepath}: {e}")
        return config


def prepare_cli_execution(
    config: dict[str, Any],
    input_dir: str | Path,
    output_dir: str | Path | None,
    step: str | None,
) -> tuple[dict[str, Any], Path, Path, str]:
    """Resolve effective CLI paths/step and return a complete replayable config."""
    input_path = Path(input_dir).expanduser().resolve()
    configured_output = str(output_dir).strip() if output_dir else ""
    output_path = (
        Path(configured_output).expanduser().resolve()
        if configured_output
        else input_path / "output"
    )

    configured_step = str(step or config.get("execution", {}).get("step", "all")).lower()
    effective_step = configured_step if configured_step in CLI_STEPS else "all"

    effective_config = get_default_unified_config()
    for section in (
        "filters",
        "interpolation",
        "processing",
        "adjustments",
        "analysis_windows",
    ):
        section_values = config.get(section, {})
        if isinstance(section_values, dict):
            effective_config[section].update(section_values)
    effective_config["paths"] = {
        "input_dir": str(input_path),
        "output_dir": str(output_path),
    }
    timestamp_output = bool(config.get("execution", {}).get("timestamp_output", False))
    effective_config["execution"] = {
        "step": effective_step,
        "timestamp_output": timestamp_output,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "vaila_version": VERSION,
    }

    output_path.mkdir(parents=True, exist_ok=True)
    return effective_config, input_path, output_path, effective_step


def save_cli_run_history(
    config: dict[str, Any], input_dir: str | Path, output_dir: str | Path
) -> list[Path]:
    """Write one run-history TOML at the input root and one at the output root.

    The filename is stable (no timestamp). Each run overwrites the same files so
    results_* subfolders share a single replay config instead of accumulating copies.
    """
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    destinations = list(dict.fromkeys((input_path, output_path)))
    paths = []
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / RUN_HISTORY_FILENAME
        with open(path, "w", encoding="utf-8") as history_file:
            toml.dump(config, history_file)
        paths.append(path)
    return paths


def update_cli_run_history(config: dict[str, Any], paths: list[Path]) -> None:
    """Rewrite all history copies after GUI-only choices have been recorded."""
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as history_file:
            toml.dump(config, history_file)


def format_treadmill_gui_cli_command(
    history_path: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    step: str,
) -> list[str]:
    """Build the exact copy/paste CLI mirror for a GUI execution."""
    return [
        "uv",
        "run",
        CLI_SCRIPT_PATH,
        "--config",
        str(Path(history_path).resolve()),
        "--input-dir",
        str(Path(input_dir).resolve()),
        "--output-dir",
        str(Path(output_dir).resolve()),
        "--step",
        step,
    ]


def prepare_gui_cli_run(
    config: dict[str, Any],
    input_dir: str | Path,
    output_dir: str | Path | None,
    step: str,
) -> tuple[dict[str, Any], Path, Path, list[Path], list[str]]:
    """Create GUI run history and print its highlighted, reproducible CLI mirror."""
    effective_config, input_path, output_path, effective_step = prepare_cli_execution(
        config,
        input_dir=input_dir,
        output_dir=output_dir,
        step=step,
    )
    history_paths = save_cli_run_history(effective_config, input_path, output_path)
    cli = format_treadmill_gui_cli_command(
        history_paths[0], input_path, output_path, effective_step
    )
    print_treadmill_config_summary(
        effective_config,
        input_dir=input_path,
        output_dir=output_path,
        mode="GUI Run → reproducible CLI",
        step=effective_step,
    )
    print_run_history_paths(history_paths)
    print_gui_cli_mirror("vaila/treadmill_lc", cli)
    return effective_config, input_path, output_path, history_paths, cli


def finish_gui_cli_run(
    config: dict[str, Any], history_paths: list[Path], cli: list[str], *, succeeded: bool
) -> None:
    """Persist GUI-only selections and repeat the mirror at the terminal tail."""
    update_cli_run_history(config, history_paths)
    paths = config.get("paths", {})
    print_treadmill_config_summary(
        config,
        input_dir=paths.get("input_dir"),
        output_dir=paths.get("output_dir"),
        mode="Completed GUI Run → reproducible CLI" if succeeded else "GUI Run Request",
        step=config.get("execution", {}).get("step", "all"),
    )
    print_run_history_paths(history_paths)
    note = (
        "Equivalent CLI for the completed GUI run (copy/paste):"
        if succeeded
        else "Equivalent CLI/history for this GUI request (copy/paste after reviewing):"
    )
    print_gui_cli_mirror("vaila/treadmill_lc", cli, note=note)


class ProcessConfigDialog(simpledialog.Dialog):
    """Dialog to configure processing parameters, with TOML support."""

    def __init__(self, parent):
        self.loaded_config = None
        self.use_toml = False
        self.toml_path = None
        super().__init__(parent, title="Processing Configuration")

    def body(self, master):
        params_frame = tk.LabelFrame(master, text="Analysis Parameters", padx=10, pady=10)
        params_frame.pack(fill="both", expand=True, padx=10, pady=5)

        tk.Label(params_frame, text="Participant Weight (kg):").grid(
            row=0, column=0, sticky="e", pady=2
        )
        tk.Label(params_frame, text="Filter Cutoff Freq (Hz):").grid(
            row=1, column=0, sticky="e", pady=2
        )
        tk.Label(params_frame, text="Detection Threshold (BW):").grid(
            row=2, column=0, sticky="e", pady=2
        )

        self.weight_entry = tk.Entry(params_frame)
        self.weight_entry.insert(0, "70.0")
        self.fc_entry = tk.Entry(params_frame)
        self.fc_entry.insert(0, "50.0")
        self.threshold_entry = tk.Entry(params_frame)
        self.threshold_entry.insert(0, "0.1")

        self.weight_entry.grid(row=0, column=1, pady=2, padx=5)
        self.fc_entry.grid(row=1, column=1, pady=2, padx=5)
        self.threshold_entry.grid(row=2, column=1, pady=2, padx=5)

        self.adv_calib_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            params_frame,
            text="Use Advanced Calibration (Multiple Weights)",
            variable=self.adv_calib_var,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=5)

        self.apply_filter_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            params_frame,
            text="Apply processing Butterworth filter (use only for raw data)",
            variable=self.apply_filter_var,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=5)

        self.gen_fig_var = tk.BooleanVar(value=True)
        tk.Checkbutton(params_frame, text="Generate Trial Figures", variable=self.gen_fig_var).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=5
        )

        self.interactive_report_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            params_frame,
            text="Generate Interactive COP Report (HTML)",
            variable=self.interactive_report_var,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=5)

        toml_frame = tk.LabelFrame(master, text="Advanced Configuration (TOML)", padx=10, pady=10)
        toml_frame.pack(fill="both", expand=True, padx=10, pady=5)

        btns_frame = tk.Frame(toml_frame)
        btns_frame.pack()
        tk.Button(btns_frame, text="Load Configuration TOML", command=self.load_config_file).pack(
            side="left", padx=5
        )
        tk.Button(
            btns_frame,
            text="Create Default TOML Template",
            command=self.create_default_toml_template,
        ).pack(side="left", padx=5)

        self.toml_label = tk.Label(toml_frame, text="No TOML loaded", fg="gray")
        self.toml_label.pack(pady=5)

        return self.weight_entry

    def create_default_toml_template(self):
        file_path = filedialog.asksaveasfilename(
            parent=self.master,
            title="Create Default TOML Configuration Template",
            defaultextension=".toml",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
            initialfile="processing_config_template.toml",
        )
        if file_path:
            default_config = get_default_process_config()
            if save_process_config(default_config, file_path):
                messagebox.showinfo(
                    "Template Created",
                    f"Default TOML template created successfully:\n{file_path}",
                    parent=self,
                )
            else:
                messagebox.showerror("Error", "Failed to create template file.", parent=self)

    def load_config_file(self):
        file_path = filedialog.askopenfilename(
            parent=self.master,
            title="Select TOML file",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
        )
        if file_path:
            config = load_process_config(file_path)
            if config:
                self.loaded_config = config
                self.use_toml = True
                self.toml_path = file_path
                self.toml_label.config(
                    text=f"TOML loaded: {os.path.basename(file_path)}", fg="green"
                )
                self.populate_fields_from_config(config)
                messagebox.showinfo(
                    "TOML Parameters Loaded", "Configuration loaded successfully!", parent=self
                )
            else:
                self.toml_label.config(text="Error loading TOML", fg="red")
                messagebox.showerror("Error", "Failed to load TOML.", parent=self)

    def populate_fields_from_config(self, config):
        proc = config.get("processing", {})
        self.weight_entry.delete(0, tk.END)
        self.weight_entry.insert(0, str(proc.get("participant_weight_kg", 70.0)))
        self.fc_entry.delete(0, tk.END)
        self.fc_entry.insert(0, str(proc.get("filter_cutoff_hz", 50.0)))
        self.threshold_entry.delete(0, tk.END)
        self.threshold_entry.insert(0, str(proc.get("detection_threshold_bw", 0.1)))
        self.adv_calib_var.set(proc.get("use_advanced_calibration", False))
        self.apply_filter_var.set(proc.get("apply_processing_filter", False))
        self.gen_fig_var.set(proc.get("generate_figures", True))
        self.interactive_report_var.set(proc.get("generate_interactive_report", True))

    def apply(self):
        if self.use_toml and self.loaded_config:
            self.result = self.loaded_config
            self.result["processing"]["participant_weight_kg"] = float(self.weight_entry.get())
            self.result["processing"]["participant_weight_source"] = "gui"
            self.result["processing"]["filter_cutoff_hz"] = float(self.fc_entry.get())
            self.result["processing"]["detection_threshold_bw"] = float(self.threshold_entry.get())
            self.result["processing"]["use_advanced_calibration"] = self.adv_calib_var.get()
            self.result["processing"]["apply_processing_filter"] = self.apply_filter_var.get()
            self.result["processing"]["generate_figures"] = self.gen_fig_var.get()
            self.result["processing"]["generate_interactive_report"] = (
                self.interactive_report_var.get()
            )
        else:
            self.result = {
                "processing": {
                    "participant_weight_kg": float(self.weight_entry.get()),
                    "participant_weight_source": "gui",
                    "use_advanced_calibration": self.adv_calib_var.get(),
                    "filter_cutoff_hz": float(self.fc_entry.get()),
                    "apply_processing_filter": self.apply_filter_var.get(),
                    "detection_threshold_bw": float(self.threshold_entry.get()),
                    "generate_figures": self.gen_fig_var.get(),
                    "generate_interactive_report": self.interactive_report_var.get(),
                }
            }


def butterworth_filter(dat, fc=50, fs=FS, order=4, btype="low"):
    """Applies a stable zero-phase Butterworth filter to the data."""
    w = fc / (fs / 2)
    sos = butter(order, w, btype, output="sos")
    return sosfiltfilt(sos, dat, axis=0)


# =============================================================================
# 5-STAGE LOGICAL CALIBRATION & BIOMECHANICS PIPELINE
# 1. Zero Offset -> 2. Calibration Matrix -> 3. Coordinate Transformation ->
# 4. Signal Filtering -> 5. Event Detection
# =============================================================================


def apply_zero_offset(cells: np.ndarray, tare_mean: np.ndarray) -> np.ndarray:
    """Stage 1: Zero Offset / Tare baseline subtraction.

    V_offset_corrected = V_raw - V_tare
    """
    return np.asarray(cells, dtype=float) - np.asarray(tare_mean, dtype=float)


def apply_calibration_matrix(
    offset_corrected_cells: np.ndarray,
    weight_kg: float = 70.0,
    m: float = 1.0,
    b: float = 0.0,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Stage 2: Calibration Matrix / Gain transformation.

    Converts offset-corrected voltages to force in Body Weight (BW) units.
    F_cells_kg = m * V_cells + b
    F_cells_bw = F_cells_kg / weight_kg
    F_vertical = sum(F_cells_bw)
    """
    if "calib_slope" in kwargs:
        m = float(kwargs["calib_slope"])
    if "calib_intercept" in kwargs:
        b = float(kwargs["calib_intercept"])
    offset_corrected_cells = np.asarray(offset_corrected_cells, dtype=float)
    grf_kg = m * offset_corrected_cells + b
    grf_bw = grf_kg / float(weight_kg)
    grf_vertical = np.sum(grf_bw, axis=1) if offset_corrected_cells.ndim > 1 else np.sum(grf_bw)
    return grf_bw, grf_vertical


def apply_coordinate_transformation(grf_bw: np.ndarray) -> dict[str, np.ndarray]:
    """Stage 3: Coordinate Transformation to physical COP & force components.

    Deck Geometry: 58.0 cm ML width (X: -29 to +29 cm), 113.0 cm AP length (Y: -56.5 to +56.5 cm).
    Cell 1: Anterior-Left  (-29.0, +56.5)
    Cell 2: Posterior-Left (-29.0, -56.5)
    Cell 3: Anterior-Right (+29.0, +56.5)
    Cell 4: Posterior-Right (+29.0, -56.5)

    Returns standard English biomechanics dictionary:
      - 'medial_lateral': COP X (cm)
      - 'anterior_posterior': COP Y (cm)
      - 'vertical': Total GRF in Body Weight (BW)
      - 'grf_bw_cells': Per-cell GRF in BW
    """
    cop_ml, cop_ap = calculate_cop_system(grf_bw)
    grf_vertical = np.sum(grf_bw, axis=1)
    return {
        "medial_lateral": cop_ml,
        "anterior_posterior": cop_ap,
        "vertical": grf_vertical,
        "cop_x": cop_ml,
        "cop_y": cop_ap,
        "grf_total": grf_vertical,
        "grf_bw_cells": grf_bw,
    }


def apply_signal_filtering(
    biomech_channels: dict[str, np.ndarray] | np.ndarray,
    fc: float = 50.0,
    fs: int = FS,
    order: int = 4,
    **kwargs: Any,
) -> dict[str, np.ndarray] | np.ndarray:
    """Stage 4: Signal Filtering (zero-phase Butterworth low-pass filter)."""
    if "cutoff_hz" in kwargs:
        fc = float(kwargs["cutoff_hz"])
    if "filter_order" in kwargs:
        order = int(kwargs["filter_order"])

    if isinstance(biomech_channels, np.ndarray):
        return butterworth_filter(biomech_channels, fc=fc, fs=fs, order=order)

    filtered = dict(biomech_channels)
    if "vertical" in biomech_channels:
        filtered["vertical"] = butterworth_filter(
            biomech_channels["vertical"], fc=fc, fs=fs, order=order
        )
        filtered["grf_total"] = filtered["vertical"]
    if "medial_lateral" in biomech_channels:
        filtered["medial_lateral"] = butterworth_filter(
            biomech_channels["medial_lateral"], fc=fc, fs=fs, order=order
        )
        filtered["cop_x"] = filtered["medial_lateral"]
    if "anterior_posterior" in biomech_channels:
        filtered["anterior_posterior"] = butterworth_filter(
            biomech_channels["anterior_posterior"], fc=fc, fs=fs, order=order
        )
        filtered["cop_y"] = filtered["anterior_posterior"]
    if "grf_bw_cells" in biomech_channels:
        filtered["grf_bw_cells"] = butterworth_filter(
            biomech_channels["grf_bw_cells"], fc=fc, fs=fs, order=order
        )
    return filtered


def detect_events(
    grf_vertical: np.ndarray,
    start_idx: int = 0,
    fs: int = FS,
    threshold: float = 0.1,
    mode: str = "legacy_valley",
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Stage 5: Event Detection (strikes and steps).

    Enforces a single ``sidefoot`` code on each step:
      0 = Right
      1 = Left
    """
    if "start" in kwargs:
        start_idx = int(kwargs["start"])
    steps, peaks = detect_steps(
        grf_vertical, start_idx=start_idx, fs=fs, threshold=threshold, mode=mode
    )
    for step in steps:
        step["sidefoot"] = _sidefoot(step)
        step.pop("side", None)
        step.pop("foot", None)
    return steps, peaks


def load_data(
    csv_path,
    tare_path,
    weight_path,
    weight_kg,
    filter_cutoff=50,
    m=1.0,
    b=0.0,
    apply_processing_filter=False,
):
    """Loads and processes running data following the 5-stage calibration pipeline:
    1. Zero Offset -> 2. Calibration Matrix -> 3. Coordinate Transformation ->
    4. Signal Filtering -> 5. Event Detection.
    """
    df = pd.read_csv(csv_path, sep=",", header=None)
    cells = -1 * df[[1, 2, 3, 4]].to_numpy()

    # Stage 1: Zero Offset
    tare_vals = read_calibration_cells(tare_path)
    tare_mean = np.mean(tare_vals, axis=0)
    offset_corrected = apply_zero_offset(cells, tare_mean)

    # Compute calibration polynomial parameters if needed
    weight_vals = read_calibration_cells(weight_path)
    corrected_weight = weight_vals - tare_mean
    weight_sum = np.sum(np.mean(corrected_weight, axis=0))
    if m == 1.0 and b == 0.0:
        m, b = np.polyfit([0, weight_sum], [0, weight_kg], 1)

    # Stage 2: Calibration Matrix
    grf_bw, grf_total = apply_calibration_matrix(offset_corrected, m=m, b=b, weight_kg=weight_kg)

    # Stage 4: Signal Filtering (if enabled)
    if apply_processing_filter:
        grf_bw = butterworth_filter(grf_bw, fc=filter_cutoff)
        grf_total = np.sum(grf_bw, axis=1)

    return grf_bw, grf_total, m, b


def calculate_cop_system(grf_bw):
    """Calculates COP in centimeters: 1 anterior-left, 2 posterior-left, 3 anterior-right, 4 posterior-right."""
    half_width_cm = 58.0 / 2.0
    half_length_cm = 113.0 / 2.0
    positions = np.array(
        [
            [-half_width_cm, half_length_cm],  # Cell 1 - anterior left
            [-half_width_cm, -half_length_cm],  # Cell 2 - posterior left
            [half_width_cm, half_length_cm],  # Cell 3 - anterior right
            [half_width_cm, -half_length_cm],  # Cell 4 - posterior right
        ]
    )

    force_sum = np.sum(grf_bw, axis=1)
    force_sum[force_sum < 0.01] = 0.01

    cop_x = np.sum(grf_bw * positions[:, 0], axis=1) / force_sum
    cop_y = np.sum(grf_bw * positions[:, 1], axis=1) / force_sum

    return cop_x, cop_y


def strikeattr(strike, fs=FS):
    """Extracts biomechanical attributes from a single foot strike."""
    strike = np.array(strike)
    strike_peak_indices, _ = find_peaks(strike)
    pos_peakmax = np.argmax(strike)
    val_peakmax = max(strike)

    strike_derivative = np.diff(strike)
    pos_maxdiff = np.argmax(strike_derivative)
    val_maxdiff = max(strike_derivative)

    neg_derivative_peaks, _ = find_peaks(-1 * strike_derivative[pos_maxdiff:pos_peakmax])
    if len(neg_derivative_peaks) == 0:
        pos_itransient = np.nan
        val_itransient = np.nan
    else:
        pos_itransient = neg_derivative_peaks[0] + 1 + pos_maxdiff
        val_itransient = strike[pos_itransient]

    der_post = np.diff(strike[pos_peakmax:])
    if len(der_post) > 0:
        min_der_post = min(der_post)
        max_unloading_rate = -min_der_post * fs
    else:
        max_unloading_rate = np.nan

    attr1 = val_peakmax
    attr2 = pos_peakmax
    attr3 = len(strike_peak_indices)
    attr4 = len(strike)
    attr5 = val_itransient
    attr6 = pos_itransient
    attr7 = strike[pos_maxdiff + 1] if pos_maxdiff + 1 < len(strike) else np.nan
    attr8 = pos_maxdiff + 1
    attr9 = np.trapezoid(strike)
    attr10 = np.trapezoid(strike[: pos_peakmax + 1]) if pos_peakmax > 0 else np.nan
    attr15 = np.trapezoid(strike[pos_peakmax:]) if pos_peakmax < len(strike) else np.nan
    attr12 = attr8 * attr7 if not np.isnan(attr7) else np.nan

    if np.isnan(attr6):
        attr11 = np.nan
        attr13 = np.nan
        attr14 = np.nan
    else:
        attr11 = np.trapezoid(strike[: int(attr6) + 1]) if attr6 > 0 else np.nan
        attr13 = (
            np.trapezoid(strike[int(attr6) : pos_peakmax + 1]) if attr6 < pos_peakmax else np.nan
        )
        attr14 = np.trapezoid(strike[int(attr8) : int(attr6) + 1]) if attr8 < attr6 else np.nan

    attr16 = attr1 * fs / attr2 if attr2 != 0 else np.nan
    attr17 = val_maxdiff * fs
    attr18 = attr5 * fs / attr6 if not np.isnan(attr6) and attr6 != 0 else np.nan
    attr19 = attr7 * fs / attr8 if attr8 != 0 else np.nan
    attr20 = max_unloading_rate

    t_loading_s = (pos_maxdiff + 1) / fs
    t_stance_s = attr4 / fs
    t_to_peak_s = attr2 / fs

    return {
        "peak_GRF_BW": attr1,
        "t_to_peak_s": t_to_peak_s,
        "n_peaks": attr3,
        "t_stance_s": t_stance_s,
        "t_loading_s": t_loading_s,
        "t_unloading_s": t_stance_s - t_to_peak_s,
        "itransient1_BW": attr5,
        "t_itransient1_s": attr6 / fs if not np.isnan(attr6) else np.nan,
        "itransient2_BW": attr7,
        "t_itransient2_s": attr8 / fs if not np.isnan(attr8) else np.nan,
        "imp_total_BW_s": attr9,
        "imp_to_peak_BW_s": attr10,
        "imp_to_trans1_BW_s": attr11,
        "imp_trans2_prod": attr12,
        "imp_trans1_to_peak_BW_s": attr13,
        "imp_trans2_to_trans1_BW_s": attr14,
        "imp_post_peak_BW_s": attr15,
        "avg_lr_to_peak_BW_s": attr16,
        "max_inst_lr_BW_s": attr17,
        "avg_lr_to_trans1_BW_s": attr18,
        "avg_lr_to_trans2_BW_s": attr19,
        "max_unloading_rate_BW_s": attr20,
        "max_loading_rate_BW_s": attr17,
    }


def calculate_kinetic_metrics_strike(grf_total, start, end, cop_x, cop_y, fs=FS):
    """Calculates spatial (COP) metrics for an individual strike using standard English biomechanics keys."""
    strike = grf_total[start:end]
    cop_x_s = cop_x[start:end] if cop_x is not None else np.zeros_like(strike)
    cop_y_s = cop_y[start:end] if cop_y is not None else np.zeros_like(strike)

    if len(strike) == 0:
        return {}

    contact_mask = strike > 0.01
    cop_x_contact = cop_x_s[contact_mask]
    cop_y_contact = cop_y_s[contact_mask]

    mean_ml = float(np.mean(cop_x_contact)) if len(cop_x_contact) > 0 else 0.0
    mean_ap = float(np.mean(cop_y_contact)) if len(cop_y_contact) > 0 else 0.0
    std_ml = float(np.std(cop_x_contact)) if len(cop_x_contact) > 1 else 0.0
    std_ap = float(np.std(cop_y_contact)) if len(cop_y_contact) > 1 else 0.0
    range_ml = float(np.ptp(cop_x_contact)) if len(cop_x_contact) > 0 else 0.0
    range_ap = float(np.ptp(cop_y_contact)) if len(cop_y_contact) > 0 else 0.0
    init_ap = float(cop_y_contact[0]) if len(cop_y_contact) > 0 else 0.0
    final_ap = float(cop_y_contact[-1]) if len(cop_y_contact) > 0 else 0.0

    return {
        "cop_x_mean": mean_ml,
        "cop_y_mean": mean_ap,
        "cop_x_std": std_ml,
        "cop_y_std": std_ap,
        "cop_x_range": range_ml,
        "cop_y_range": range_ap,
        "cop_y_initial": init_ap,
        "cop_y_final": final_ap,
        "cop_medial_lateral_mean": mean_ml,
        "cop_anterior_posterior_mean": mean_ap,
        "cop_medial_lateral_std": std_ml,
        "cop_anterior_posterior_std": std_ap,
        "cop_medial_lateral_range": range_ml,
        "cop_anterior_posterior_range": range_ap,
        "cop_anterior_posterior_initial": init_ap,
        "cop_anterior_posterior_final": final_ap,
    }


def normalize_analysis_window_points(points, n_samples):
    """Convert one or two clicked points into start/end sample indices."""
    if n_samples <= 0:
        return 0, 0

    clean_points = []
    for point in points:
        x_value = point[0] if isinstance(point, (list, tuple, np.ndarray)) else point
        if np.isfinite(x_value):
            clean_points.append(int(round(float(x_value))))

    if len(clean_points) not in {1, 2}:
        return None

    start_idx = min(max(clean_points[0], 0), n_samples - 1)
    if len(clean_points) == 1:
        return start_idx, n_samples

    end_idx = min(max(clean_points[1], 0), n_samples)
    if end_idx <= start_idx:
        return None
    return start_idx, end_idx


def record_analysis_window_in_config(
    config: dict[str, Any] | None,
    file_name: str,
    start_idx: int,
    end_idx: int,
    n_samples: int,
    *,
    start_source: str | None = None,
    end_source: str | None = None,
) -> None:
    """Persist a resolved process window in the replayable run config."""
    if config is None:
        return
    key = Path(canonical_trial_filename(file_name)).stem.lower()
    window = {
        "start_index": int(start_idx),
        "end_index_exclusive": int(end_idx),
        "source_samples": int(n_samples),
    }
    if start_source:
        window["start_source"] = start_source
    if end_source:
        window["end_source"] = end_source
    config.setdefault("analysis_windows", {})[key] = window


def get_recorded_analysis_window(
    config: dict[str, Any] | None, file_name: str, n_samples: int
) -> tuple[int, int] | None:
    """Load and validate a GUI-selected process window for headless replay."""
    windows = (config or {}).get("analysis_windows", {})
    if not isinstance(windows, dict):
        return None
    key = Path(canonical_trial_filename(file_name)).stem.lower()
    recorded = windows.get(key)
    if not isinstance(recorded, dict):
        return None
    start = recorded.get("start_index")
    end = recorded.get("end_index_exclusive")
    if start is None or end is None:
        return None
    return normalize_analysis_window_points([start, end], n_samples)


def detect_signal_start_index(
    signal: np.ndarray | list[float],
    peak_ratio: float = AUTO_START_PEAK_RATIO,
    min_peak_distance_samples: int = FS // 10,
) -> int:
    """Return the first large impact peak in a vertical-force signal.

    A peak is considered large when its height is at least ``peak_ratio`` of
    the finite signal maximum. The default 100 ms separation rejects tiny local
    oscillations inside the same contact. Signals without a positive detectable
    peak safely start at sample zero.
    """
    values = np.asarray(signal, dtype=float).reshape(-1)
    if values.size == 0:
        return 0
    if not 0 < peak_ratio <= 1:
        raise ValueError("peak_ratio must be greater than 0 and at most 1")
    if min_peak_distance_samples < 1:
        raise ValueError("min_peak_distance_samples must be at least 1")

    finite = np.isfinite(values)
    if not finite.any():
        return 0
    finite_max = float(np.max(values[finite]))
    if finite_max <= 0:
        return 0

    clean = values.copy()
    clean[~finite] = -np.inf
    peaks, _ = find_peaks(
        clean,
        height=finite_max * peak_ratio,
        distance=min_peak_distance_samples,
    )
    return int(peaks[0]) if len(peaks) else 0


def resolve_analysis_window(
    config: dict[str, Any] | None,
    file_name: str,
    signal: np.ndarray | list[float],
) -> tuple[int, int, str, str]:
    """Resolve one trial window using CLI, TOML, then automatic detection."""
    values = np.asarray(signal).reshape(-1)
    n_samples = len(values)
    if n_samples == 0:
        return 0, 0, "automatic", "automatic"

    execution = (config or {}).get("execution", {})
    if not isinstance(execution, dict):
        execution = {}
    force_auto = bool(execution.get("force_auto", False))
    cli_start = execution.get("start_index")
    cli_end = execution.get("end_index_exclusive")

    recorded = None if force_auto else get_recorded_analysis_window(config, file_name, n_samples)
    automatic_start = detect_signal_start_index(values)

    if cli_start is not None:
        start_idx = int(cli_start)
        start_source = "cli"
    elif recorded is not None:
        start_idx = recorded[0]
        start_source = "toml"
    else:
        start_idx = automatic_start
        start_source = "automatic"

    if cli_end is not None:
        end_idx = int(cli_end)
        end_source = "cli"
    elif recorded is not None:
        end_idx = recorded[1]
        end_source = "toml"
    else:
        end_idx = n_samples
        end_source = "automatic"

    normalized = normalize_analysis_window_points([start_idx, end_idx], n_samples)
    if normalized is None:
        raise ValueError(
            f"Invalid analysis window for {file_name}: start={start_idx}, end={end_idx}"
        )
    return normalized[0], normalized[1], start_source, end_source


def select_analysis_window(grf_total_raw, file_name, parent=None):
    """Select analysis window; right click clears points and Enter finalizes."""
    _ensure_interactive_backend()
    while True:
        clicked_points = []
        marker_artists = []
        finalized = False
        fig, ax = plt.subplots(figsize=(14, 5))
        time_s = np.arange(len(grf_total_raw)) / FS
        ax.plot(time_s, grf_total_raw, "b-", linewidth=0.5)
        ax.set_title(
            f"{file_name}: Left click START and optional END; right click clears; ENTER finishes\n"
            "One point = START to signal end. Two points = START to END. First peak = Right foot"
        )
        ax.set_ylabel("Total GRF (BW)")
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)

        def redraw_markers(
            clicked_points=clicked_points, marker_artists=marker_artists, ax=ax, fig=fig
        ):
            for artist in marker_artists:
                artist.remove()
            marker_artists.clear()
            for idx, point in enumerate(clicked_points):
                label = "START" if idx == 0 else "END" if idx == 1 else f"EXTRA {idx + 1}"
                color = "green" if idx == 0 else "red" if idx == 1 else "orange"
                marker_artists.append(
                    ax.axvline(
                        point[0] / FS, color=color, linestyle="--", linewidth=1.2, label=label
                    )
                )
            if marker_artists:
                ax.legend(loc="upper right")
            legend = ax.get_legend()
            if legend and not marker_artists:
                legend.remove()
            fig.canvas.draw_idle()

        def finish_selection(fig=fig):
            plt.close(fig)

        def on_click(event, ax=ax, clicked_points=clicked_points):
            if event.inaxes != ax or event.xdata is None:
                return
            if event.button == 3:
                clicked_points.clear()
                redraw_markers()
                print("   Analysis window points cleared by right click.")
                return
            if event.button != 1:
                return
            clicked_points.append(
                (event.xdata * FS, event.ydata if event.ydata is not None else 0.0)
            )
            redraw_markers()

        def on_key(event):
            nonlocal finalized
            if event.key in {"enter", "return"}:
                finalized = True
                finish_selection()

        click_cid = fig.canvas.mpl_connect("button_press_event", on_click)
        key_cid = fig.canvas.mpl_connect("key_press_event", on_key)

        try:
            plt.show(block=True)
        finally:
            fig.canvas.mpl_disconnect(click_cid)
            fig.canvas.mpl_disconnect(key_cid)
            plt.close(fig)
            gc.collect()

        if not finalized:
            retry = messagebox.askretrycancel(
                "Selection not finished",
                "Press ENTER after marking the analysis window. Retry this file?",
                parent=parent,
            )
            if retry:
                continue
            print("   Analysis window selection canceled. Using full signal.")
            return 0, len(grf_total_raw)

        window = normalize_analysis_window_points(clicked_points, len(grf_total_raw))
        if window is None:
            retry = messagebox.askretrycancel(
                "Invalid selection",
                "Select exactly one START point, or START and END in this order. "
                "Use right click to clear wrong marks, then press ENTER. Retry this file?",
                parent=parent,
            )
            if retry:
                continue
            print("   Invalid analysis window selection canceled. Using full signal.")
            return 0, len(grf_total_raw)

        start_idx, end_idx = window
        print(f"   Analysis window selected: start={start_idx}, end={end_idx}")
        return start_idx, end_idx


def _legacy_threshold_window(offset_signal, min_index, window_samples):
    """Return the threshold window used by the original selectstrikes routine."""
    start = max(int(min_index) - int(window_samples), 0)
    end = min(int(min_index) + int(window_samples), len(offset_signal))
    if end <= start:
        return offset_signal
    return offset_signal[start:end]


def detect_steps_peak_to_valley(grf_total, start_idx=0, fs=FS, threshold=0.1):
    """Previous peak-to-valley detector kept as fallback for difficult signals."""
    signal = np.asarray(grf_total[start_idx:], dtype=float)
    t = np.arange(len(signal)) / fs

    peaks, _ = find_peaks(signal, height=threshold, distance=int(fs * 0.3))
    if len(peaks) < 2:
        return [], []

    valleys, _ = find_peaks(-signal)
    valleys_between_peaks = []

    for i in range(len(peaks) - 1):
        mask = (valleys > peaks[i]) & (valleys < peaks[i + 1])
        if np.any(mask):
            valleys_between_peaks.append(valleys[mask][0])
        else:
            valleys_between_peaks.append(peaks[i + 1])

    steps = []
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = valleys_between_peaks[i] if i < len(valleys_between_peaks) else peaks[i + 1]
        if end <= start:
            continue
        sidefoot = SIDEFOOT_RIGHT if i % 2 == 0 else SIDEFOOT_LEFT
        steps.append(
            {
                "idx_start": int(start),
                "idx_end": int(end),
                "t_start": float(t[start]),
                "t_end": float(t[end]),
                "sidefoot": sidefoot,
                "detection_mode": "peak_to_valley_fallback",
            }
        )

    return steps, peaks


def detect_steps_legacy_valley(grf_total, start_idx=0, fs=FS, threshold=0.1, window_samples=18):
    """Detect foot strikes using the valley/cut logic from ia_treadmill.selectstrikes.

    The legacy algorithm thresholds a minimum-shifted signal and finds valleys in the
    compressed above-threshold signal. We keep the compressed strike for legacy
    attributes and map cut indices back to the original signal for COP/figures.
    """
    signal = np.asarray(grf_total[start_idx:], dtype=float)
    if len(signal) < 3:
        return [], []

    window_samples = int(window_samples)
    if window_samples % 2 != 0:
        window_samples += 1
    window_samples = max(window_samples, 2)

    signal_min = float(np.nanmin(signal))
    if not np.isfinite(signal_min):
        return [], []
    min_index = int(np.nanargmin(signal))
    offset_signal = signal - signal_min
    threshold_window = _legacy_threshold_window(offset_signal, min_index, window_samples)
    if len(threshold_window) == 0:
        return [], []

    auto_threshold = float(np.nanmax(threshold_window) + 2 * np.nanstd(threshold_window))
    effective_threshold = max(auto_threshold, float(threshold))
    active_mask = offset_signal > effective_threshold
    active_indices = np.flatnonzero(active_mask)
    active_signal = offset_signal[active_mask]
    if len(active_signal) < 3:
        return [], []

    inverted_active = -1 * active_signal
    peak_height = float(np.nanmean(inverted_active) + np.nanstd(inverted_active))
    distance = max(int(round(fs / window_samples)), 1)
    cuts, _ = find_peaks(inverted_active, height=peak_height, distance=distance)
    if len(cuts) < 2:
        cuts, _ = find_peaks(inverted_active, distance=distance)
    if len(cuts) % 2 != 0:
        cuts = cuts[:-1]
    if len(cuts) < 2:
        return [], []

    peaks_original = []
    steps = []
    for i in range(len(cuts) - 1):
        cut_start = int(cuts[i])
        cut_end = int(cuts[i + 1])
        if cut_end <= cut_start:
            continue

        legacy_signal = active_signal[cut_start:cut_end]
        if len(legacy_signal) < 3:
            continue

        original_start = int(active_indices[cut_start])
        original_end = int(active_indices[cut_end - 1]) + 1
        if original_end <= original_start:
            continue

        peak_local = int(np.nanargmax(legacy_signal))
        peak_original = int(active_indices[cut_start + peak_local])
        peaks_original.append(peak_original)

        sidefoot = SIDEFOOT_RIGHT if i % 2 == 0 else SIDEFOOT_LEFT
        steps.append(
            {
                "idx_start": original_start,
                "idx_end": original_end,
                "t_start": original_start / fs,
                "t_end": original_end / fs,
                "sidefoot": sidefoot,
                "detection_mode": "legacy_valley",
                "legacy_signal": legacy_signal,
                "legacy_cut_start_index": cut_start,
                "legacy_cut_end_index": cut_end,
                "legacy_original_start_index": original_start,
                "legacy_original_end_index": original_end,
                "legacy_peak_index": peak_original,
                "legacy_threshold_bw": effective_threshold,
                "legacy_auto_threshold_bw": auto_threshold,
                "legacy_signal_offset_min_bw": signal_min,
            }
        )

    return steps, np.asarray(peaks_original, dtype=int)


def detect_steps(grf_total, start_idx=0, fs=FS, threshold=0.1, mode="legacy_valley"):
    """Detect alternating steps, defaulting to the original valley/cut logic."""
    if mode == "legacy_valley":
        steps, peaks = detect_steps_legacy_valley(grf_total, start_idx, fs, threshold)
        if steps:
            return steps, peaks
        print("   Legacy valley step detection found no steps; using peak-to-valley fallback.")
    return detect_steps_peak_to_valley(grf_total, start_idx, fs, threshold)


def calculate_general_metrics(steps, grf_total, fs=FS):
    """Calculates general summary metrics of all steps in a running session."""
    if len(steps) < 2:
        return {}

    t_stance_r, t_stance_l = [], []
    t_double = []
    total_time = steps[-1]["t_end"]

    for i in range(len(steps)):
        p = steps[i]
        t_stance = p["t_end"] - p["t_start"]
        if _sidefoot(p) == SIDEFOOT_RIGHT:
            t_stance_r.append(t_stance)
        else:
            t_stance_l.append(t_stance)

        if i < len(steps) - 1:
            p_next = steps[i + 1]
            t_double_i = p_next["t_start"] - p["t_end"]
            if t_double_i > 0:
                t_double.append(t_double_i)

    n_r = sum(1 for p in steps if _sidefoot(p) == SIDEFOOT_RIGHT)
    n_l = sum(1 for p in steps if _sidefoot(p) == SIDEFOOT_LEFT)

    return {
        "n_steps_total": len(steps),
        "n_steps_R": n_r,
        "n_steps_L": n_l,
        "analysis_time_s": total_time,
        "cadence_steps_min": len(steps) * 60 / total_time if total_time > 0 else 0,
        "t_stance_mean_R_s": np.mean(t_stance_r) if t_stance_r else np.nan,
        "t_stance_mean_L_s": np.mean(t_stance_l) if t_stance_l else np.nan,
        "t_stance_std_R_s": np.std(t_stance_r) if t_stance_r else np.nan,
        "t_stance_std_L_s": np.std(t_stance_l) if t_stance_l else np.nan,
        "t_double_mean_s": np.mean(t_double) if t_double else np.nan,
        "t_double_std_s": np.std(t_double) if t_double else np.nan,
        "t_double_percent": (np.mean(t_double) / np.mean(t_stance_r + t_stance_l) * 100)
        if t_double
        else np.nan,
    }


def calculate_asymmetry(values_r, values_l):
    """Calculates the Asymmetry Index (ASI) between right and left foot."""
    mean_r = np.mean(values_r) if values_r else 0
    mean_l = np.mean(values_l) if values_l else 0
    std_r = np.std(values_r) if values_r else 0
    std_l = np.std(values_l) if values_l else 0

    if mean_r + mean_l == 0:
        return {"ASI": 0, "mean_R": 0, "mean_L": 0, "std_R": 0, "std_L": 0}

    asi = abs(mean_r - mean_l) / ((mean_r + mean_l) / 2) * 100
    return {
        "ASI": asi,
        "mean_R": mean_r,
        "mean_L": mean_l,
        "std_R": std_r,
        "std_L": std_l,
    }


def _representative_steps(steps, max_steps=4):
    """Return a small set of representative steps for lightweight diagnostic figures."""
    if not steps:
        return []
    selected = []
    for sidefoot in (SIDEFOOT_RIGHT, SIDEFOOT_LEFT):
        foot_steps = [step for step in steps if _sidefoot(step) == sidefoot]
        if foot_steps:
            selected.append(foot_steps[len(foot_steps) // 2])
    if len(selected) < max_steps:
        for step in steps:
            if step not in selected:
                selected.append(step)
            if len(selected) >= max_steps:
                break
    return selected[:max_steps]


def _step_signal(step, grf_total):
    """Return the legacy strike signal when available, otherwise slice the full GRF."""
    legacy_signal = step.get("legacy_signal")
    if legacy_signal is not None:
        return np.asarray(legacy_signal, dtype=float)
    return np.asarray(grf_total[step["idx_start"] : step["idx_end"]], dtype=float)


def _plot_strike_attribute_panel(ax_force, ax_derivative, signal, title, fs=FS):
    """Plot one strike with original-style peak/transient/slope/area annotations."""
    if len(signal) < 3:
        ax_force.set_title(f"{title} - insufficient samples")
        ax_force.plot(signal, color="0.4")
        ax_derivative.axis("off")
        return

    derivative = np.diff(signal)
    pos_peak = int(np.nanargmax(signal))
    val_peak = float(signal[pos_peak])
    pos_maxdiff = int(np.nanargmax(derivative)) if len(derivative) else 0
    val_maxdiff = float(derivative[pos_maxdiff]) if len(derivative) else np.nan
    transient_peaks, _ = find_peaks(-1 * derivative[pos_maxdiff:pos_peak])
    if len(transient_peaks) > 0:
        pos_transient = int(transient_peaks[0] + 1 + pos_maxdiff)
        val_transient = float(signal[pos_transient])
    else:
        pos_transient = None
        val_transient = np.nan

    x = np.arange(len(signal)) / fs
    baseline = float(signal[0])
    ax_force.plot(x, signal, color="black", linestyle="--", linewidth=1.2)
    ax_force.fill_between(x, baseline, signal, color="tab:blue", alpha=0.12, label="total area")
    ax_force.fill_between(
        x[: pos_maxdiff + 2],
        baseline,
        signal[: pos_maxdiff + 2],
        color="tab:green",
        alpha=0.35,
        label="initial slope area",
    )
    if pos_transient is not None:
        ax_force.fill_between(
            x[: pos_transient + 1],
            baseline,
            signal[: pos_transient + 1],
            color="tab:red",
            alpha=0.25,
            label="to transient",
        )
        ax_force.fill_between(
            x[pos_transient : pos_peak + 1],
            baseline,
            signal[pos_transient : pos_peak + 1],
            color="gold",
            alpha=0.35,
            label="transient to peak",
        )
        ax_force.plot(x[pos_transient], val_transient, "rv", markersize=7, label="transient 1")
    ax_force.plot(x[pos_peak], val_peak, "yv", markeredgecolor="black", markersize=8, label="peak")
    if pos_maxdiff + 1 < len(signal):
        ax_force.plot(
            x[pos_maxdiff + 1],
            signal[pos_maxdiff + 1],
            "gv",
            markersize=7,
            label="max loading slope",
        )
    ax_force.set_title(title)
    ax_force.set_ylabel("Vertical GRF (BW)")
    ax_force.grid(True, alpha=0.25)

    xd = np.arange(len(derivative)) / fs
    ax_derivative.plot(xd, derivative, color="black", linestyle="--", linewidth=1.0)
    if len(derivative):
        ax_derivative.plot(pos_maxdiff / fs, val_maxdiff, "gv", markersize=7)
    if pos_transient is not None and pos_transient > 0 and pos_transient - 1 < len(derivative):
        ax_derivative.plot(
            (pos_transient - 1) / fs, derivative[pos_transient - 1], "rv", markersize=7
        )
    if pos_peak > 0 and pos_peak - 1 < len(derivative):
        ax_derivative.plot((pos_peak - 1) / fs, derivative[pos_peak - 1], "yv", markersize=7)
    ax_derivative.axhline(0, color="0.5", linewidth=0.8)
    ax_derivative.set_ylabel("dGRF/dt (BW/sample)")
    ax_derivative.set_xlabel("Strike time (s)")
    ax_derivative.grid(True, alpha=0.25)


def _plot_strike_diagnostics(grf_total, steps, file_name, output_dir, fs=FS):
    """Save original-inspired strike attribute and stride-map diagnostic figures."""
    representative = _representative_steps(steps, max_steps=4)
    if representative:
        fig, axes = plt.subplots(len(representative), 2, figsize=(14, 3.4 * len(representative)))
        axes = np.atleast_2d(axes)
        try:
            for row, step in enumerate(representative):
                signal = _step_signal(step, grf_total)
                title = f"Step {steps.index(step) + 1} (sidefoot={_sidefoot(step)}) - attributes"
                _plot_strike_attribute_panel(axes[row, 0], axes[row, 1], signal, title, fs)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
            base_name = _base_trial_stem_from_adjusted(file_name)
            fig.suptitle(f"Original-style Strike Attributes: {file_name}", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.94])
            fig.savefig(
                os.path.join(output_dir, f"{base_name}_processing_strike_attributes.png"), dpi=150
            )
        finally:
            plt.close(fig)

    if steps:
        fig_map, (ax_full, ax_norm) = plt.subplots(2, 1, figsize=(14, 8))
        try:
            time_s = np.arange(len(grf_total)) / fs
            ax_full.plot(time_s, grf_total, color="black", linewidth=0.55, alpha=0.75)
            for step in steps:
                ax_full.axvspan(
                    step["idx_start"] / fs,
                    step["idx_end"] / fs,
                    color=_SIDEFOOT_PLOT_COLORS.get(_sidefoot(step), "gray"),
                    alpha=0.16,
                )
                peak = step.get("legacy_peak_index")
                if peak is not None and np.isfinite(peak):
                    ax_full.axvline(
                        float(peak) / fs, color="tab:orange", linewidth=0.45, alpha=0.55
                    )
            ax_full.set_title(f"Detected support regions and internal peaks: {file_name}")
            ax_full.set_ylabel("Total GRF (BW)")
            ax_full.grid(True, alpha=0.25)

            norm_x = np.linspace(0, 100, 101)
            for step in steps:
                signal = _step_signal(step, grf_total)
                if len(signal) < 2:
                    continue
                source_x = np.linspace(0, 100, len(signal))
                norm_y = np.interp(norm_x, source_x, signal)
                ax_norm.plot(
                    norm_x,
                    norm_y,
                    color=_SIDEFOOT_PLOT_COLORS.get(_sidefoot(step), "gray"),
                    alpha=0.18,
                    linewidth=0.8,
                )
            ax_norm.set_title("All detected strikes normalized to 0-100% support")
            ax_norm.set_xlabel("Support phase (%)")
            ax_norm.set_ylabel("Vertical GRF (BW)")
            ax_norm.grid(True, alpha=0.25)
            base_name = _base_trial_stem_from_adjusted(file_name)
            plt.tight_layout()
            fig_map.savefig(
                os.path.join(output_dir, f"{base_name}_processing_stride_map.png"), dpi=150
            )
        finally:
            plt.close(fig_map)


def _downsample_indices(n_samples, max_points=5000):
    """Return regularly spaced indices for lightweight plots/reports."""
    if n_samples <= max_points:
        return np.arange(n_samples)
    step = int(np.ceil(n_samples / max_points))
    return np.arange(0, n_samples, step)


def _write_interactive_cop_report(
    grf_total, derivative, cop_x, cop_y, steps, peaks, file_name, output_dir
):
    """Write a lightweight Plotly HTML report with GRF, derivative, and COP."""
    indices = _downsample_indices(len(grf_total))
    time_s = (indices / FS).astype(float).tolist()
    grf_ds = np.asarray(grf_total)[indices].astype(float).tolist()
    derivative_ds = np.asarray(derivative)[indices].astype(float).tolist()
    # Treadmill view: mediolateral displacement is horizontal, anteroposterior is vertical.
    cop_horizontal = np.asarray(cop_x)[indices].astype(float).tolist()
    cop_vertical = np.asarray(cop_y)[indices].astype(float).tolist()

    deck_width_cm = 58.0
    deck_length_cm = 113.0
    half_width_cm = deck_width_cm / 2.0
    half_length_cm = deck_length_cm / 2.0
    cell_x = [-half_width_cm, -half_width_cm, half_width_cm, half_width_cm]
    cell_y = [half_length_cm, -half_length_cm, half_length_cm, -half_length_cm]
    cell_labels = ["Cell 1", "Cell 2", "Cell 3", "Cell 4"]

    step_shapes = []
    for step in steps:
        step_shapes.append(
            {
                "type": "rect",
                "xref": "x",
                "yref": "paper",
                "x0": float(step["idx_start"] / FS),
                "x1": float(step["idx_end"] / FS),
                "y0": 0,
                "y1": 1,
                "fillcolor": "rgba(220, 38, 38, 0.10)"
                if _sidefoot(step) == SIDEFOOT_RIGHT
                else "rgba(22, 163, 74, 0.10)",
                "line": {"width": 0},
            }
        )
    for idx, peak in enumerate(peaks):
        step_shapes.append(
            {
                "type": "line",
                "xref": "x",
                "yref": "paper",
                "x0": float(peak / FS),
                "x1": float(peak / FS),
                "y0": 0,
                "y1": 1,
                "line": {
                    "color": "rgba(220, 38, 38, 0.45)"
                    if idx % 2 == 0
                    else "rgba(22, 163, 74, 0.45)",
                    "dash": "dot",
                    "width": 1,
                },
            }
        )

    payload = {
        "file": file_name,
        "time_s": time_s,
        "grf": grf_ds,
        "derivative": derivative_ds,
        "cop_horizontal": cop_horizontal,
        "cop_vertical": cop_vertical,
        "cop_color": time_s,
        "cell_x": cell_x,
        "cell_y": cell_y,
        "cell_labels": cell_labels,
        "step_shapes": step_shapes,
        "deck": {
            "width_cm": deck_width_cm,
            "length_cm": deck_length_cm,
            "x_min": -half_width_cm,
            "x_max": half_width_cm,
            "y_min": -half_length_cm,
            "y_max": half_length_cm,
            "x_range": [-50.0, 50.0],
            "y_range": [-100.0, 100.0],
        },
    }
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Load Cells COP Report - {file_name}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f9f9f9; color: #333; }}
.container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
h1 {{ text-align: center; color: #2c3e50; margin-bottom: 8px; }}
h2 {{ color: #34495e; font-size: 18px; font-weight: 600; margin-top: 0; }}
.section-title {{ border-bottom: 2px solid #3498db; padding-bottom: 5px; color: #2c3e50; margin-top: 34px; }}
.plot-container {{ margin-bottom: 34px; border: 1px solid #ddd; padding: 12px; border-radius: 6px; background: #fff; }}
.plot {{ width: 100%; height: 440px; }}
.note {{ color: #4b5563; line-height: 1.45; background: #f8f9fa; border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px; }}
.metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 18px 0; }}
.metric-card {{ background: #f8f9fa; border: 1px solid #ddd; border-radius: 6px; padding: 12px; }}
.metric-card strong {{ display: block; color: #2c3e50; margin-bottom: 4px; }}
</style>
</head>
<body>
<div class="container">
<h1><i>vailá</i> - Load Cells COP Interactive Report</h1>
<h2>Trial: {file_name}</h2>
<div class="note">COP is plotted over the fixed load-cell geometry: 58 cm mediolateral by 113 cm anteroposterior. The trajectory represents the contact-load center on the instrumented treadmill deck, not belt displacement and not stride length along the treadmill.</div>
<div class="metric-grid">
  <div class="metric-card"><strong>Horizontal axis</strong>COP X, mediolateral position in centimeters.</div>
  <div class="metric-card"><strong>Vertical axis</strong>COP Y, anteroposterior position in centimeters.</div>
  <div class="metric-card"><strong>Cell layout</strong>1 = anterior-left, 2 = posterior-left, 3 = anterior-right, 4 = posterior-right.</div>
</div>
<h3 class="section-title">Ground Reaction Force</h3>
<div class="plot-container"><div id="grf" class="plot"></div></div>
<h3 class="section-title">Derivative</h3>
<div class="plot-container"><div id="derivative" class="plot"></div></div>
<h3 class="section-title">COP Contact-Load Location</h3>
<div class="plot-container"><div id="cop" class="plot"></div></div>
</div>
<script>
const data = {json.dumps(payload)};
const deckShape = {{
  type: 'rect', xref: 'x', yref: 'y',
  x0: data.deck.x_min, x1: data.deck.x_max, y0: data.deck.y_min, y1: data.deck.y_max,
  fillcolor: 'rgba(148, 163, 184, 0.08)',
  line: {{color: '#111827', width: 2}}
}};
const centerLines = [
  {{type: 'line', xref: 'x', yref: 'y', x0: 0, x1: 0, y0: data.deck.y_min, y1: data.deck.y_max, line: {{color: 'rgba(17,24,39,0.35)', width: 1, dash: 'dot'}}}},
  {{type: 'line', xref: 'x', yref: 'y', x0: data.deck.x_min, x1: data.deck.x_max, y0: 0, y1: 0, line: {{color: 'rgba(17,24,39,0.35)', width: 1, dash: 'dot'}}}}
];
Plotly.newPlot('grf', [{{
  x: data.time_s, y: data.grf, type: 'scatter', mode: 'lines', name: 'Total GRF', line: {{color: '#2563eb', width: 1}}
}}], {{title: 'Total GRF (BW)', xaxis: {{title: 'Time (s)'}}, yaxis: {{title: 'BW'}}, shapes: data.step_shapes, template: 'plotly_white'}}, {{responsive: true}});
Plotly.newPlot('derivative', [{{
  x: data.time_s, y: data.derivative, type: 'scatter', mode: 'lines', name: 'dGRF/dt', line: {{color: '#7c3aed', width: 1}}
}}], {{title: 'Total GRF First Derivative', xaxis: {{title: 'Time (s)'}}, yaxis: {{title: 'BW/s'}}, template: 'plotly_white'}}, {{responsive: true}});
Plotly.newPlot('cop', [{{
  x: data.cop_horizontal, y: data.cop_vertical, type: 'scattergl', mode: 'lines+markers', name: 'COP contact-load center',
  marker: {{size: 4, color: data.cop_color, colorscale: 'Viridis', colorbar: {{title: 'Time (s)'}}}},
  line: {{color: 'rgba(75,85,99,0.45)', width: 1}}
}}, {{
  x: data.cell_x, y: data.cell_y, type: 'scatter', mode: 'markers+text', name: 'Load cells',
  text: data.cell_labels, textposition: 'top center', marker: {{size: 10, color: '#111827', symbol: 'square'}}
}}], {{
  title: 'COP Contact-Load Location on 58 x 113 cm Treadmill Deck',
  xaxis: {{title: 'COP X - Mediolateral (cm)', range: data.deck.x_range, zeroline: false}},
  yaxis: {{title: 'COP Y - Anteroposterior (cm)', range: data.deck.y_range, scaleanchor: 'x', scaleratio: 1, zeroline: false}},
  shapes: [deckShape, ...centerLines], template: 'plotly_white'
}}, {{responsive: true}});
</script>
</body>
</html>
"""
    base_name = _base_trial_stem_from_adjusted(file_name)
    report_path = os.path.join(output_dir, f"{base_name}_processing_cop_report_interactive.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path


def plot_trial_figures(
    grf_total,
    cop_x,
    cop_y,
    steps,
    peaks,
    file_name,
    output_dir,
    generate_interactive_report=True,
):
    """Save lightweight per-trial overview, full COP trajectory, and optional HTML report."""
    _ensure_noninteractive_backend()
    os.makedirs(output_dir, exist_ok=True)
    derivative = np.gradient(grf_total) * FS if len(grf_total) > 1 else np.zeros_like(grf_total)
    time_s = np.arange(len(grf_total)) / FS

    fig_all, (ax_grf, ax_der) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    try:
        ax_grf.plot(time_s, grf_total, "b-", linewidth=0.5, alpha=0.7)
        for step in steps:
            ax_grf.axvspan(
                step["idx_start"] / FS,
                step["idx_end"] / FS,
                alpha=0.2,
                color=_SIDEFOOT_PLOT_COLORS.get(_sidefoot(step), "gray"),
            )
        for idx, peak in enumerate(peaks):
            color = "red" if idx % 2 == 0 else "green"
            ax_grf.axvline(peak / FS, color=color, linestyle="--", alpha=0.5, linewidth=0.5)
        ax_grf.set_ylabel("Total GRF (BW)")
        ax_grf.set_title(f"Overview: {file_name}")
        ax_grf.grid(True, alpha=0.3)

        ax_der.plot(time_s, derivative, color="tab:purple", linewidth=0.5, alpha=0.85)
        ax_der.axhline(0, color="0.4", linewidth=0.8, alpha=0.6)
        ax_der.set_ylabel("dGRF/dt (BW/s)")
        ax_der.set_xlabel("Time (s)")
        ax_der.set_title("First Derivative")
        ax_der.grid(True, alpha=0.3)
        base_name = _base_trial_stem_from_adjusted(file_name)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_processing_overview.png"), dpi=150)
    finally:
        plt.close(fig_all)

    _plot_strike_diagnostics(grf_total, steps, file_name, output_dir, FS)

    fig_cop, ax_cop = plt.subplots(figsize=(7.5, 8.5))
    try:
        time_color = np.arange(len(cop_x)) / FS
        deck_width_cm = 58.0
        deck_length_cm = 113.0
        half_width_cm = deck_width_cm / 2.0
        half_length_cm = deck_length_cm / 2.0
        cell_positions = [
            (-half_width_cm, half_length_cm, "Cell 1"),
            (-half_width_cm, -half_length_cm, "Cell 2"),
            (half_width_cm, half_length_cm, "Cell 3"),
            (half_width_cm, -half_length_cm, "Cell 4"),
        ]

        deck = Rectangle(
            (-half_width_cm, -half_length_cm),
            deck_width_cm,
            deck_length_cm,
            facecolor="0.92",
            edgecolor="0.15",
            linewidth=1.4,
            alpha=0.35,
            zorder=0,
        )
        ax_cop.add_patch(deck)
        ax_cop.axhline(0, color="0.35", linestyle=":", linewidth=1.0, alpha=0.6, zorder=1)
        ax_cop.axvline(0, color="0.35", linestyle=":", linewidth=1.0, alpha=0.6, zorder=1)
        for x_cell, y_cell, label in cell_positions:
            ax_cop.scatter(x_cell, y_cell, c="0.1", s=45, marker="s", zorder=4)
            ax_cop.annotate(
                label,
                (x_cell, y_cell),
                xytext=(5, 6),
                textcoords="offset points",
                fontsize=8,
                color="0.1",
            )

        # Treadmill view: horizontal = ML (COP X), vertical = AP (COP Y).
        scatter = ax_cop.scatter(cop_x, cop_y, c=time_color, s=4, cmap="viridis", alpha=0.75)
        ax_cop.plot(cop_x, cop_y, color="0.55", linewidth=0.35, alpha=0.6)
        if len(cop_x) > 0:
            ax_cop.scatter(cop_x[0], cop_y[0], c="green", s=50, marker="o", label="Start")
            ax_cop.scatter(cop_x[-1], cop_y[-1], c="red", s=50, marker="x", label="End")
        ax_cop.set_xlim(-50, 50)
        ax_cop.set_ylim(-100, 100)
        ax_cop.set_xlabel("COP X - Mediolateral (cm)")
        ax_cop.set_ylabel("COP Y - Anteroposterior (cm)")
        ax_cop.set_title(
            f"COP Contact-Load Location: {file_name}\n"
            "Fixed load-cell deck: 58 cm ML x 113 cm AP; not belt displacement"
        )
        ax_cop.set_aspect("equal", adjustable="box")
        ax_cop.grid(True, alpha=0.3)
        ax_cop.legend(loc="best")
        fig_cop.colorbar(scatter, ax=ax_cop, label="Time (s)")
        base_name = _base_trial_stem_from_adjusted(file_name)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_processing_cop_trajectory.png"), dpi=150)
    finally:
        plt.close(fig_cop)

    if generate_interactive_report:
        _write_interactive_cop_report(
            grf_total, derivative, cop_x, cop_y, steps, peaks, file_name, output_dir
        )
    gc.collect()


def run_process_stage(
    parent=None,
    initial_dir=None,
    config: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    headless: bool = False,
) -> str | None:
    """Executes the biomechanical analysis and calibration stage."""
    folder = initial_dir or (
        filedialog.askdirectory(title="Select directory with CSV files", parent=parent)
        if not headless
        else None
    )
    if not folder:
        return None

    if config and ("processing" in config or "participant_weight_kg" in config):
        proc_config = config.get("processing", config)
        full_config = config if "processing" in config else {"processing": proc_config}
    elif not headless:
        dialog = ProcessConfigDialog(parent)
        if not dialog.result:
            return None
        proc_config = dialog.result["processing"]
        full_config = dialog.result
    else:
        full_config = get_default_process_config()
        proc_config = full_config["processing"]

    fallback_weight_kg = float(proc_config.get("participant_weight_kg", 70.0))
    configured_weight_source = str(proc_config.get("participant_weight_source", "")).lower()
    use_advanced_calibration = bool(proc_config.get("use_advanced_calibration", False))
    fc_filter = float(proc_config.get("filter_cutoff_hz", 50.0))
    apply_processing_filter = bool(proc_config.get("apply_processing_filter", False))
    threshold = float(proc_config.get("detection_threshold_bw", 0.1))
    generate_figures = bool(proc_config.get("generate_figures", True))
    generate_interactive_report = bool(proc_config.get("generate_interactive_report", True))

    # Identify running CSV files (trials)
    running_files = sorted([f for f in os.listdir(folder) if is_trial_file(f)])
    if not running_files:
        if not headless:
            messagebox.showerror("Error", "No running CSV files found.", parent=parent)
        return None

    # Group trials by subject and day (sXX_dXX)
    groups = {}
    for f in running_files:
        match = re.search(r"s(\d+)_d(\d+)", f, re.IGNORECASE)
        key = f"s{match.group(1)}_d{match.group(2)}".lower() if match else "unknown"
        if key not in groups:
            groups[key] = []
        groups[key].append(f)

    base_folder = output_dir or get_output_base_folder(folder)
    timestamp_output = bool(full_config.get("execution", {}).get("timestamp_output", False))
    path_figures = make_output_dir(base_folder, "figures", timestamp_output)
    path_results = make_output_dir(base_folder, "results", timestamp_output)

    config_save_path = os.path.join(path_results, "processing_configuration_used.toml")
    save_process_config(full_config, config_save_path)

    original_columns = [
        "peak_GRF_BW",
        "t_to_peak_s",
        "n_peaks",
        "t_stance_s",
        "t_loading_s",
        "t_unloading_s",
        "itransient1_BW",
        "t_itransient1_s",
        "itransient2_BW",
        "t_itransient2_s",
        "imp_total_BW_s",
        "imp_to_peak_BW_s",
        "imp_to_trans1_BW_s",
        "imp_trans2_prod",
        "imp_trans1_to_peak_BW_s",
        "imp_trans2_to_trans1_BW_s",
        "imp_post_peak_BW_s",
        "avg_lr_to_peak_BW_s",
        "max_inst_lr_BW_s",
        "avg_lr_to_trans1_BW_s",
        "avg_lr_to_trans2_BW_s",
        "max_unloading_rate_BW_s",
        "max_loading_rate_BW_s",
    ]

    kinetic_columns = [
        "cop_x_mean",
        "cop_y_mean",
        "cop_x_std",
        "cop_y_std",
        "cop_x_range",
        "cop_y_range",
        "cop_y_initial",
        "cop_y_final",
    ]

    try:
        for key, group_files in groups.items():
            print(f"\nProcessing Group: {key.upper()}")

            # Resolve subject/day string
            match = re.match(r"s(\d+)_d(\d+)", key)
            if match:
                subj_str, day_str = match.group(1), match.group(2)
            else:
                subj_str, day_str = "01", "01"

            # Auto-discover calibration files
            tare_file, weight_file, plate_files, borg_file = discover_calibration_and_borg(
                folder, subj_str, day_str
            )

            # Manual prompt fallback if auto-discovery fails
            if not tare_file and not headless:
                print(f"   Tare file not found for {key.upper()}. Please select it manually.")
                tare_file = filedialog.askopenfilename(
                    title=f"Select Tare File for {key.upper()}",
                    filetypes=[("CSV", "*.csv")],
                    parent=parent,
                )
            if not weight_file and not headless:
                print(
                    f"   Weight calibration file not found for {key.upper()}. Please select it manually."
                )
                weight_file = filedialog.askopenfilename(
                    title=f"Select Weight File for {key.upper()}",
                    filetypes=[("CSV", "*.csv")],
                    parent=parent,
                )

            if not tare_file or not weight_file:
                print(f"   Skipping group {key.upper()} due to missing tare/weight file.")
                continue

            print(f"   Using Calibration Files for {key.upper()}:")
            print(f"     Tare: {os.path.basename(tare_file)}")
            print(f"     Weight: {os.path.basename(weight_file)}")
            if borg_file:
                print(f"     Borg: {os.path.basename(borg_file)}")
            if plate_files:
                print(
                    f"     Plates ({len(plate_files)} files): {[os.path.basename(pf) for pf in plate_files]}"
                )

            # Basic Pre-computation for the group (Stage 1: Zero offset)
            tare_vals = read_calibration_cells(tare_file)
            tare_mean = np.mean(tare_vals, axis=0)

            weight_vals = read_calibration_cells(weight_file)
            corrected_weight = weight_vals - tare_mean
            weight_sum = np.sum(np.mean(corrected_weight, axis=0))

            # Parse body weight from Borg file (once per subject-day group)
            group_weight = None
            if borg_file:
                group_weight = get_group_weight_from_borg(borg_file)

            if configured_weight_source in {"cli", "toml", "gui"}:
                weight_kg = fallback_weight_kg
                print(
                    f"   Using {configured_weight_source.upper()} body weight for group: "
                    f"{weight_kg} kg"
                )
            elif group_weight is not None:
                weight_kg = group_weight
                print(f"   Using body weight from Borg for group: {weight_kg} kg")
            else:
                weight_kg = fallback_weight_kg
                print(f"   Using fallback body weight for group: {weight_kg} kg")

            # Stage 2: Calibration fit
            m, b = 1.0, 0.0
            if use_advanced_calibration and len(plate_files) > 0:
                summed_readings = [0.0]
                known_weights = [0.0]

                for pf in sorted(plate_files):
                    pf_name = os.path.basename(pf).lower()
                    w_match = re.search(r"(\d+)kg\.csv", pf_name)
                    if w_match:
                        w = int(w_match.group(1))
                        if w > 0:
                            cell_p = read_calibration_cells(pf)
                            corrected_p = np.mean(cell_p, axis=0) - tare_mean
                            summed_readings.append(np.sum(corrected_p))
                            known_weights.append(float(w))

                summed_readings.append(weight_sum)
                known_weights.append(weight_kg)

                if len(summed_readings) > 2:
                    m, b = np.polyfit(summed_readings, known_weights, 1)
                    print(f"   Advanced calibration for group: m={m:.6f}, b={b:.6f}")
                else:
                    m, b = np.polyfit([0, weight_sum], [0, weight_kg], 1)
                    print(f"   Simple calibration for group: m={m:.6f}, b={b:.6f}")
            else:
                m, b = np.polyfit([0, weight_sum], [0, weight_kg], 1)
                print(f"   Simple calibration for group: m={m:.6f}, b={b:.6f}")

            group_metrics_rows = []

            for file in group_files:
                print(f"\n>>> Processing: {file}")
                base_name = file[:-4]

                metadata = load_adjustment_metadata(os.path.join(folder, file))
                if metadata and metadata.get("processed") is False:
                    msg = f"Trial {file} skipped from processing (marked as excluded/not processed because it needs more filtering adjustments)."
                    print(msg)
                    if parent and hasattr(parent, "_write_log"):
                        parent._write_log(msg)
                    continue

                grf_bw, grf_total_raw, _, _ = load_data(
                    os.path.join(folder, file),
                    tare_file,
                    weight_file,
                    weight_kg,
                    int(fc_filter),
                    m,
                    b,
                    apply_processing_filter=apply_processing_filter,
                )

                if headless:
                    start_idx, end_idx, start_source, end_source = resolve_analysis_window(
                        full_config, file, grf_total_raw
                    )
                    record_analysis_window_in_config(
                        full_config,
                        file,
                        start_idx,
                        end_idx,
                        len(grf_total_raw),
                        start_source=start_source,
                        end_source=end_source,
                    )
                    print(
                        f"   Analysis window: [{start_idx}:{end_idx}] "
                        f"(start={start_source}, end={end_source})"
                    )
                else:
                    start_idx, end_idx = select_analysis_window(grf_total_raw, file, parent=parent)
                    record_analysis_window_in_config(
                        full_config,
                        file,
                        start_idx,
                        end_idx,
                        len(grf_total_raw),
                        start_source="gui",
                        end_source="gui",
                    )
                grf_total = grf_total_raw[start_idx:end_idx]

                # Stage 3: Coordinate Transformation
                cop_channels = apply_coordinate_transformation(grf_bw)
                cop_x_full = cop_channels["medial_lateral"]
                cop_y_full = cop_channels["anterior_posterior"]
                cop_x = cop_x_full[start_idx:end_idx]
                cop_y = cop_y_full[start_idx:end_idx]

                # Stage 5: Event Detection
                steps, peaks = detect_events(grf_total, 0, FS, threshold, mode="legacy_valley")
                general_metrics = calculate_general_metrics(steps, grf_total, FS)
                file_rows = []

                if len(steps) == 0:
                    print("   No steps detected.")
                    general_metrics = {
                        "n_steps_total": 0,
                        "n_steps_R": 0,
                        "n_steps_L": 0,
                        "analysis_time_s": len(grf_total) / FS if len(grf_total) else 0,
                        "cadence_steps_min": 0,
                        "t_stance_mean_R_s": 0,
                        "t_stance_mean_L_s": 0,
                        "t_double_mean_s": 0,
                    }
                    pd.DataFrame(
                        columns=["step_number", "sidefoot", "t_start_s", "t_end_s"]
                    ).to_csv(
                        os.path.join(path_results, f"{base_name}_processing_steps.csv"),
                        index=False,
                        float_format="%.8f",
                    )
                    group_metrics_rows.append(
                        {
                            "file": file,
                            "trial": base_name,
                            "analysis_start_index": start_idx,
                            "analysis_end_index": end_idx,
                            **general_metrics,
                        }
                    )
                    if generate_figures:
                        plot_trial_figures(
                            grf_total,
                            cop_x,
                            cop_y,
                            steps,
                            peaks,
                            file,
                            path_figures,
                            generate_interactive_report=generate_interactive_report,
                        )
                    plt.close("all")
                    gc.collect()
                    continue

                for i, p in enumerate(steps):
                    strike_grf = p.get("legacy_signal")
                    if strike_grf is None:
                        strike_grf = grf_total[p["idx_start"] : p["idx_end"]]
                    attrs_orig = strikeattr(strike_grf, FS)
                    idx_start_global = p["idx_start"]
                    idx_end_global = p["idx_end"]
                    kinetic_attrs = calculate_kinetic_metrics_strike(
                        grf_total, idx_start_global, idx_end_global, cop_x, cop_y, FS
                    )
                    file_rows.append(
                        {
                            "step_number": i + 1,
                            "sidefoot": _sidefoot(p),
                            "t_start_s": p["t_start"],
                            "t_end_s": p["t_end"],
                            "detection_mode": p.get("detection_mode", "legacy_valley"),
                            "legacy_cut_start_index": p.get("legacy_cut_start_index", np.nan),
                            "legacy_cut_end_index": p.get("legacy_cut_end_index", np.nan),
                            "legacy_original_start_index": p.get(
                                "legacy_original_start_index", idx_start_global
                            ),
                            "legacy_original_end_index": p.get(
                                "legacy_original_end_index", idx_end_global
                            ),
                            "legacy_peak_index": p.get("legacy_peak_index", np.nan),
                            "legacy_threshold_bw": p.get("legacy_threshold_bw", np.nan),
                            "legacy_auto_threshold_bw": p.get("legacy_auto_threshold_bw", np.nan),
                            "legacy_signal_offset_min_bw": p.get(
                                "legacy_signal_offset_min_bw", np.nan
                            ),
                            **dict(attrs_orig.items()),
                            **dict(kinetic_attrs.items()),
                        }
                    )

                df_file = pd.DataFrame(file_rows)
                df_r = (
                    df_file[df_file["sidefoot"] == SIDEFOOT_RIGHT].copy()
                    if len(df_file) > 0
                    else pd.DataFrame()
                )
                df_l = (
                    df_file[df_file["sidefoot"] == SIDEFOOT_LEFT].copy()
                    if len(df_file) > 0
                    else pd.DataFrame()
                )

                asymmetry_metrics = {}
                for col in original_columns + kinetic_columns:
                    vals_r = (
                        df_r[col].dropna().tolist() if col in df_r.columns and len(df_r) > 0 else []
                    )
                    vals_l = (
                        df_l[col].dropna().tolist() if col in df_l.columns and len(df_l) > 0 else []
                    )
                    asi = calculate_asymmetry(vals_r, vals_l)
                    asymmetry_metrics[f"{col}_ASI"] = asi["ASI"]
                    asymmetry_metrics[f"{col}_mean_R"] = asi["mean_R"]
                    asymmetry_metrics[f"{col}_mean_L"] = asi["mean_L"]
                    asymmetry_metrics[f"{col}_std_R"] = asi["std_R"]
                    asymmetry_metrics[f"{col}_std_L"] = asi["std_L"]

                final_metrics = {
                    "file": file,
                    "trial": base_name,
                    "analysis_start_index": start_idx,
                    "analysis_end_index": end_idx,
                    **general_metrics,
                    **asymmetry_metrics,
                }
                group_metrics_rows.append(final_metrics)

                df_file.to_csv(
                    os.path.join(path_results, f"{base_name}_processing_steps.csv"),
                    index=False,
                    float_format="%.8f",
                )
                if generate_figures:
                    plot_trial_figures(
                        grf_total,
                        cop_x,
                        cop_y,
                        steps,
                        peaks,
                        file,
                        path_figures,
                        generate_interactive_report=generate_interactive_report,
                    )
                print(f"   Saved: {base_name}_processing_steps.csv")

                plt.close("all")
                gc.collect()

            if group_metrics_rows:
                group_metrics_path = os.path.join(path_results, f"{key}_processing_metrics.csv")
                pd.DataFrame(group_metrics_rows).to_csv(
                    group_metrics_path, index=False, float_format="%.8f"
                )
                print(f"   Saved daily metrics: {os.path.basename(group_metrics_path)}")

        save_process_config(full_config, config_save_path)
        print("\n>>> Processing complete!")
        if not headless:
            messagebox.showinfo("Success", "Analysis complete!", parent=parent)
    except Exception as e:
        print(f"An error occurred: {e}")
        if not headless:
            messagebox.showerror("Error", f"An error occurred: {e}", parent=parent)
    finally:
        plt.close("all")

    return path_results


# =============================================================================
# LAUNCHER GUI (SINGLE STREAMLINED WINDOW)
# =============================================================================


class LoadCellTreadmillDialog:
    """Single streamlined launcher dialog with TOML configuration and workflow execution."""

    def __init__(self, parent: tk.Misc | None = None) -> None:
        self.parent = parent
        if parent is None:
            self.window = tk.Tk()
            self._owns_root = True
        elif isinstance(parent, (tk.Tk, tk.Toplevel)):
            self.window = tk.Toplevel(parent)
            self.window.transient(parent)
            self._owns_root = False
        else:
            self.window = tk.Toplevel(parent)
            self._owns_root = False

        self.window.title("vailá - Treadmill LC Biomechanics")
        self.window.geometry("680x560")
        self.window.minsize(620, 500)

        self.pipeline_config: dict[str, Any] = get_default_unified_config()
        self.config_path_var = tk.StringVar(master=self.window, value="")
        self.input_dir_var = tk.StringVar(master=self.window, value="")
        self.output_dir_var = tk.StringVar(master=self.window, value="")
        self.body_weight_var = tk.StringVar(master=self.window, value="Not detected")
        self.timestamp_output_var = tk.BooleanVar(master=self.window, value=False)
        self.summary_text_var = tk.StringVar(
            master=self.window, value="Default configuration loaded."
        )

        # Apply Premium Aesthetics
        self.window.configure(bg="#f8fafc")
        style = ttk.Style(self.window)
        style.theme_use("clam")
        style.configure("TFrame", background="#f8fafc")
        style.configure("TLabel", background="#f8fafc", font=("Helvetica", 10))
        style.configure(
            "Primary.TButton",
            font=("Helvetica", 10, "bold"),
            background="#2563eb",
            foreground="white",
            padding=6,
        )
        style.map("Primary.TButton", background=[("active", "#1d4ed8")])
        style.configure(
            "Secondary.TButton",
            font=("Helvetica", 10),
            background="#e2e8f0",
            foreground="#1e293b",
            padding=6,
        )
        style.map("Secondary.TButton", background=[("active", "#cbd5e1")])

        main_frame = ttk.Frame(self.window, padding=16)
        main_frame.pack(fill="both", expand=True)

        # Header
        ttk.Label(
            main_frame,
            text="Treadmill Load Cells Biomechanics Analysis",
            font=("Helvetica", 14, "bold"),
            foreground="#0f172a",
        ).pack(pady=(0, 4))

        ttk.Label(
            main_frame,
            text="5-Stage Logical Pipeline: Zero Offset → Calibration Matrix → Coordinate Transformation → Signal Filtering → Event Detection",
            wraplength=640,
            justify="center",
            foreground="#64748b",
            font=("Helvetica", 9),
        ).pack(pady=(0, 12))

        # Config & Directories Card
        config_frame = ttk.LabelFrame(main_frame, text="Configuration & Directories", padding=10)
        config_frame.pack(fill="x", pady=(0, 10))

        # TOML Config File
        r0 = ttk.Frame(config_frame)
        r0.pack(fill="x", pady=2)
        ttk.Label(r0, text="Config TOML:", width=14).pack(side="left")
        ttk.Entry(r0, textvariable=self.config_path_var).pack(
            side="left", fill="x", expand=True, padx=4
        )
        ttk.Button(r0, text="Browse...", style="Secondary.TButton", command=self._browse_toml).pack(
            side="left"
        )

        # Input Directory
        r1 = ttk.Frame(config_frame)
        r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="Input CSV Dir:", width=14).pack(side="left")
        ttk.Entry(r1, textvariable=self.input_dir_var).pack(
            side="left", fill="x", expand=True, padx=4
        )
        ttk.Button(
            r1, text="Browse...", style="Secondary.TButton", command=self._browse_input_dir
        ).pack(side="left")

        # Output Directory (Optional)
        r2 = ttk.Frame(config_frame)
        r2.pack(fill="x", pady=2)
        ttk.Label(r2, text="Output Dir (opt):", width=14).pack(side="left")
        ttk.Entry(r2, textvariable=self.output_dir_var).pack(
            side="left", fill="x", expand=True, padx=4
        )
        ttk.Button(
            r2, text="Browse...", style="Secondary.TButton", command=self._browse_output_dir
        ).pack(side="left")

        r3 = ttk.Frame(config_frame)
        r3.pack(fill="x", pady=2)
        ttk.Label(r3, text="Body Weight (kg):", width=14).pack(side="left")
        ttk.Label(r3, textvariable=self.body_weight_var, foreground="#0f766e").pack(
            side="left", padx=4
        )
        ttk.Checkbutton(
            r3,
            text="Timestamped output folders",
            variable=self.timestamp_output_var,
        ).pack(side="right")

        # Config Summary Label
        ttk.Label(
            config_frame,
            textvariable=self.summary_text_var,
            font=("Helvetica", 9, "italic"),
            foreground="#3b82f6",
        ).pack(anchor="w", pady=(4, 0))

        # Actions Card
        actions_card = ttk.LabelFrame(main_frame, text="Execution Pipeline", padding=10)
        actions_card.pack(fill="x", pady=(0, 10))

        # Primary Action: Run Full Pipeline
        ttk.Button(
            actions_card,
            text="🚀 Run Full Pipeline (Filter → Adjust/Interpolate → Process)",
            style="Primary.TButton",
            command=self._run_full_pipeline,
        ).pack(fill="x", pady=3)

        # Stage Buttons Grid
        stage_btn_row = ttk.Frame(actions_card)
        stage_btn_row.pack(fill="x", pady=3)
        ttk.Button(
            stage_btn_row,
            text="1. Filter Only",
            style="Secondary.TButton",
            command=lambda: self._execute_stage("filter"),
        ).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(
            stage_btn_row,
            text="2. Adjust + Interp",
            style="Secondary.TButton",
            command=lambda: self._execute_stage("adjust"),
        ).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(
            stage_btn_row,
            text="3. Process Only",
            style="Secondary.TButton",
            command=lambda: self._execute_stage("process"),
        ).pack(side="left", expand=True, fill="x", padx=2)

        # Log Card
        log_frame = ttk.LabelFrame(main_frame, text="Execution Log", padding=6)
        log_frame.pack(fill="both", expand=True, pady=(0, 10))
        self.log_text = tk.Text(
            log_frame, height=6, bg="#ffffff", fg="#1e293b", font=("Courier", 9)
        )
        self.log_text.pack(fill="both", expand=True)

        # Footer Actions
        footer = ttk.Frame(main_frame)
        footer.pack(fill="x")
        ttk.Button(
            footer, text="📖 Help Docs", style="Secondary.TButton", command=self._open_help
        ).pack(side="left", padx=4)
        ttk.Button(footer, text="✖ Close", style="Secondary.TButton", command=self.destroy).pack(
            side="right", padx=4
        )

        self._update_summary()
        self._write_log("Ready. Select input directory or TOML configuration to begin.")
        self._write_log(
            "Run prints a highlighted Equivalent CLI command and saves replay TOMLs. "
            "Output folders are reused unless timestamped output is enabled."
        )

    def destroy(self) -> None:
        """Close and destroy the window."""
        win = getattr(self, "window", None)
        if win is not None and hasattr(win, "destroy"):
            win.destroy()

    def _write_log(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        if hasattr(self, "log_text"):
            self.log_text.insert("end", f"[{ts}] {message}\n")
            self.log_text.see("end")
        win = getattr(self, "window", None)
        if win is not None and hasattr(win, "update_idletasks"):
            win.update_idletasks()

    def _browse_toml(self) -> None:
        win = getattr(self, "window", getattr(self, "parent", None))
        fp = filedialog.askopenfilename(
            title="Select Treadmill Processing TOML Configuration",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
            parent=win,
        )
        if fp:
            self.config_path_var.set(fp)
            self.pipeline_config = load_unified_config(fp)
            if self.pipeline_config.get("paths", {}).get("input_dir"):
                self.input_dir_var.set(self.pipeline_config["paths"]["input_dir"])
            if self.pipeline_config.get("paths", {}).get("output_dir"):
                self.output_dir_var.set(self.pipeline_config["paths"]["output_dir"])
            self.timestamp_output_var.set(
                bool(self.pipeline_config.get("execution", {}).get("timestamp_output", False))
            )
            if self.pipeline_config.get("paths", {}).get("input_dir"):
                self._sync_subject_weight(self.pipeline_config["paths"]["input_dir"])
            self._update_summary()
            self._write_log(f"Loaded TOML configuration from: {fp}")

    def _browse_input_dir(self) -> None:
        win = getattr(self, "window", getattr(self, "parent", None))
        d = filedialog.askdirectory(title="Select Input CSV Directory", parent=win)
        if d:
            self.input_dir_var.set(d)
            self._sync_subject_weight(d)
            self._update_summary()
            self._write_log(f"Selected input directory: {d}")

    def _browse_output_dir(self) -> None:
        win = getattr(self, "window", getattr(self, "parent", None))
        d = filedialog.askdirectory(title="Select Output Directory (Optional)", parent=win)
        if d:
            self.output_dir_var.set(d)
            self._write_log(f"Selected output directory: {d}")

    def _update_summary(self) -> None:
        proc = self.pipeline_config.get("processing", {})
        filt = self.pipeline_config.get("filters", {})
        w = proc.get("participant_weight_kg", 70.0)
        fc = proc.get("filter_cutoff_hz", filt.get("lowpass_cutoff", 50.0))
        th = proc.get("detection_threshold_bw", 0.1)
        adv = proc.get("use_advanced_calibration", False)
        source = str(proc.get("participant_weight_source", "")).lower()
        if self.input_dir_var.get().strip() or source in {"cli", "toml", "gui", "metadata"}:
            if source == "metadata":
                self.body_weight_var.set(f"{w:g} (info/Borg)")
            elif source in {"cli", "toml", "gui"}:
                self.body_weight_var.set(f"{w:g}")
            else:
                self.body_weight_var.set(f"{w:g} (fallback)")
        else:
            self.body_weight_var.set("Not detected — select input")
        self.summary_text_var.set(
            f"Body Weight: {self.body_weight_var.get()} | Filter Cutoff: {fc} Hz | "
            f"Event Threshold: {th} BW | Multi-Plate Calib: {'Yes' if adv else 'No'}"
        )

    def _sync_subject_weight(self, input_dir: str | Path) -> None:
        """Populate the GUI weight display from the selected subject metadata."""
        try:
            detected = infer_subject_weight_kg(input_dir)
        except (OSError, ValueError, TypeError) as exc:
            self._write_log(f"Could not infer subject body weight: {exc}")
            return
        if detected is None:
            return
        proc = self.pipeline_config.setdefault("processing", {})
        source = str(proc.get("participant_weight_source", "")).lower()
        if source not in {"cli", "toml", "gui"}:
            proc["participant_weight_kg"] = detected
            proc["participant_weight_source"] = "metadata"
        self.body_weight_var.set(f"{detected:g} (info/Borg)")

    def _sync_gui_runtime_options(self) -> None:
        execution = self.pipeline_config.setdefault("execution", {})
        execution["timestamp_output"] = bool(self.timestamp_output_var.get())

    def _get_target_dirs(self) -> tuple[str | None, str | None]:
        input_dir = self.input_dir_var.get().strip() or None
        output_dir = self.output_dir_var.get().strip() or None
        self._sync_gui_runtime_options()
        if input_dir:
            self._sync_subject_weight(input_dir)
            self._update_summary()
        return input_dir, output_dir

    def _execute_stage(self, stage: str) -> None:
        win = getattr(self, "window", getattr(self, "parent", None))
        input_dir, output_dir = self._get_target_dirs()
        if not input_dir:
            input_dir = filedialog.askdirectory(title="Select Folder with CSV Files", parent=win)
            if not input_dir:
                return
            self.input_dir_var.set(input_dir)

        try:
            (
                self.pipeline_config,
                input_path,
                output_path,
                history_paths,
                cli,
            ) = prepare_gui_cli_run(self.pipeline_config, input_dir, output_dir, stage)
        except (OSError, ValueError) as exc:
            self._write_log(f"ERROR preparing reproducible CLI: {exc}")
            messagebox.showerror("CLI Replay Error", str(exc), parent=win)
            return

        self.input_dir_var.set(str(input_path))
        self.output_dir_var.set(str(output_path))
        self._write_log(f"Run history saved to: {history_paths[0]}")
        self._write_log(f"Starting stage: {stage} on '{input_path}'...")
        res = None
        try:
            if stage == "adjust":
                res = run_adjust_stage(
                    parent=None,
                    initial_dir=str(input_path),
                    config=self.pipeline_config,
                    output_dir=output_path,
                    headless=True,
                )
            elif stage == "filter":
                res = run_filter_stage(
                    parent=None,
                    initial_dir=str(input_path),
                    config=self.pipeline_config,
                    output_dir=output_path,
                    headless=True,
                )
            elif stage == "process":
                res = run_process_stage(
                    parent=None,
                    initial_dir=str(input_path),
                    config=self.pipeline_config,
                    output_dir=output_path,
                    headless=True,
                )
            if res:
                self._write_log(f"Stage '{stage}' completed successfully! Output: {res}")
            else:
                self._write_log(f"Stage '{stage}' finished or was cancelled.")
        except Exception as exc:
            self._write_log(f"ERROR in stage {stage}: {exc}")
            messagebox.showerror(
                "Stage Error", f"Error executing stage {stage}:\n{exc}", parent=win
            )
        finally:
            finish_gui_cli_run(self.pipeline_config, history_paths, cli, succeeded=res is not None)

    def _run_full_pipeline(self) -> None:
        win = getattr(self, "window", getattr(self, "parent", None))
        input_dir, output_dir = self._get_target_dirs()
        if not input_dir:
            input_dir = filedialog.askdirectory(
                title="Select Folder with Raw CSV Files", parent=win
            )
            if not input_dir:
                return
            self.input_dir_var.set(input_dir)

        try:
            (
                self.pipeline_config,
                input_path,
                output_path,
                history_paths,
                cli,
            ) = prepare_gui_cli_run(self.pipeline_config, input_dir, output_dir, "all")
        except (OSError, ValueError) as exc:
            self._write_log(f"ERROR preparing reproducible CLI: {exc}")
            messagebox.showerror("CLI Replay Error", str(exc), parent=win)
            return

        self.input_dir_var.set(str(input_path))
        self.output_dir_var.set(str(output_path))
        self._write_log(f"Run history saved to: {history_paths[0]}")
        self._write_log(
            "Starting Full Sequential Pipeline (Filter → Adjust+Interpolate → Process)..."
        )
        succeeded = False
        try:
            # 1. Filtering
            self._write_log("Executing Stage 1: Signal Filtering...")
            filtered_folder = run_filter_stage(
                parent=None,
                initial_dir=str(input_path),
                config=self.pipeline_config,
                output_dir=output_path,
                headless=True,
            )
            if not filtered_folder:
                self._write_log("Pipeline stopped after Stage 1 (Filtering).")
                return

            # 2. Adjustment & Interpolation
            self._write_log(
                f"Executing Stage 2: Artifact Adjustment on folder '{filtered_folder}'..."
            )
            clean_folder = run_adjust_stage(
                parent=None,
                initial_dir=filtered_folder,
                config=self.pipeline_config,
                output_dir=output_path,
                headless=True,
            )
            if not clean_folder:
                self._write_log("Pipeline stopped after Stage 2 (Adjustment).")
                return

            adjusted_folder = run_interpolate_stage(
                parent=None,
                initial_dir=clean_folder,
                config=self.pipeline_config,
                output_dir=output_path,
                headless=True,
            )
            interp_target = adjusted_folder or clean_folder

            # 3. Processing Metrics
            self._write_log(
                f"Executing Stage 3: Biomechanical Metrics on folder '{interp_target}'..."
            )
            results_folder = run_process_stage(
                parent=None,
                initial_dir=interp_target,
                config=self.pipeline_config,
                output_dir=output_path,
                headless=True,
            )
            if not results_folder:
                self._write_log("Pipeline stopped after Stage 3 (Processing).")
                return

            succeeded = True
            self._write_log("Pipeline completed successfully!")
        finally:
            finish_gui_cli_run(self.pipeline_config, history_paths, cli, succeeded=succeeded)

    def _open_help(self) -> None:
        win = getattr(self, "window", getattr(self, "parent", None))
        help_path = Path(__file__).resolve().parent / "help" / "treadmill_lc.html"
        if help_path.exists():
            webbrowser.open_new_tab(help_path.as_uri())
        else:
            messagebox.showinfo("Help", f"Help file not found:\n{help_path}", parent=win)


def run_treadmill_lc_gui(parent: tk.Misc | None = None) -> None:
    """Entry point used by vaila.py and standalone execution."""
    app = LoadCellTreadmillDialog(parent)
    if app._owns_root:
        app.window.mainloop()
    else:
        app.window.grab_set()
        app.window.wait_window()


# =============================================================================
# CLI ENTRY POINT & HEADLESS TOML SUPPORT
# =============================================================================


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="treadmill_lc.py",
        description="vailá load-cell instrumented treadmill processing and biomechanics analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch single-window GUI:
  uv run vaila/treadmill_lc.py
  uv run vaila/treadmill_lc.py --gui

  # Automatic headless processing without TOML:
  uv run vaila/treadmill_lc.py -i tests/treadmill_lc -s process -a -w 61.6

  # Override only the analysis window; other settings still come from TOML:
  uv run vaila/treadmill_lc.py -i tests/treadmill_lc -s process -b 526 -e 60000

  # Run full pipeline headlessly into an explicit output root:
  uv run vaila/treadmill_lc.py -c config.toml -i /path/to/raw_csvs -o /path/to/results -s all

  # Re-run from the stable history written at the input or output root:
  uv run vaila/treadmill_lc.py -c /path/to/treadmill_lc_run_history.toml
        """,
    )
    parser.add_argument("-c", "--config", metavar="TOML", help="Path to configuration TOML file")
    parser.add_argument(
        "-t",
        "--toml-path",
        metavar="TOML",
        default=DEFAULT_TOML_PATH,
        help=f"Fallback TOML path (default: {DEFAULT_TOML_PATH})",
    )
    parser.add_argument(
        "-i", "--input-dir", metavar="DIR", help="Folder containing load-cell CSV files"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        metavar="DIR",
        help="Output root (default: INPUT_DIR/output)",
    )
    parser.add_argument(
        "-s",
        "--step",
        choices=CLI_STEPS,
        default=None,
        help="Pipeline step (default: value from TOML, otherwise all)",
    )
    parser.add_argument(
        "-g", "--gui", action="store_true", help="Force GUI mode (opens single-window launcher)"
    )
    parser.add_argument("-w", "--weight", type=float, help="Participant body weight in kg")
    parser.add_argument(
        "-b",
        "--start",
        "--start-index",
        dest="start_index",
        type=int,
        help="First analysis sample (inclusive)",
    )
    parser.add_argument(
        "-e",
        "--end",
        "--end-index",
        dest="end_index",
        type=int,
        help="Last analysis sample (exclusive)",
    )
    parser.add_argument(
        "-a",
        "--auto",
        "--force-auto",
        dest="force_auto",
        action="store_true",
        help="Ignore TOML settings and automatically detect each signal start",
    )
    parser.add_argument(
        "-T",
        "--timestamp",
        "--timestamp-output",
        dest="timestamp_output",
        action="store_true",
        help="Create timestamped stage folders instead of reusing output folders",
    )
    return parser.parse_args(argv)


def _resolve_optional_toml_path(toml_path: str, input_dir: str | None) -> Path:
    """Resolve the fallback TOML relative to the input directory when possible."""
    candidate = Path(toml_path).expanduser()
    if not candidate.is_absolute() and input_dir:
        input_candidate = Path(input_dir).expanduser() / candidate
        if input_candidate.is_file() or toml_path == DEFAULT_TOML_PATH:
            return input_candidate
    return candidate


def _validate_cli_overrides(args: argparse.Namespace) -> str | None:
    """Return a user-facing validation error for invalid CLI overrides."""
    if args.weight is not None and args.weight <= 0:
        return "--weight must be greater than zero"
    if args.start_index is not None and args.start_index < 0:
        return "--start-index must be zero or greater"
    if args.end_index is not None and args.end_index <= 0:
        return "--end-index must be greater than zero"
    if (
        args.start_index is not None
        and args.end_index is not None
        and args.end_index <= args.start_index
    ):
        return "--end-index must be greater than --start-index"
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    cli_requested = any(
        (
            args.config,
            args.input_dir,
            args.force_auto,
            args.weight is not None,
            args.start_index is not None,
            args.end_index is not None,
            args.toml_path != DEFAULT_TOML_PATH,
            args.timestamp_output,
        )
    )

    # CLI Headless mode triggered by any CLI processing option unless --gui is explicit.
    if cli_requested and not args.gui:
        _ensure_noninteractive_backend()

        validation_error = _validate_cli_overrides(args)
        if validation_error:
            console.print(f"[bold red]Error:[/bold red] {validation_error}")
            return 2

        # Load configuration
        config_path = None
        if args.force_auto:
            config = get_default_unified_config()
            console.print("[yellow]Automatic mode:[/yellow] TOML settings were ignored.")
        elif args.config:
            config_path = Path(args.config).expanduser()
            if not config_path.is_file():
                console.print(
                    f"[bold red]Error:[/bold red] Configuration file not found: {config_path}"
                )
                return 1
            config = load_unified_config(config_path)
        else:
            config_path = _resolve_optional_toml_path(args.toml_path, args.input_dir)
            if config_path.is_file():
                config = load_unified_config(config_path)
            else:
                legacy_path = (
                    Path(args.input_dir).expanduser() / "processing_configuration_used.toml"
                    if args.input_dir and args.toml_path == DEFAULT_TOML_PATH
                    else None
                )
                if legacy_path is not None and legacy_path.is_file():
                    config_path = legacy_path
                    config = load_unified_config(config_path)
                else:
                    console.print(
                        f"[yellow]Warning:[/yellow] TOML not found: {config_path}. "
                        "Using built-in defaults and automatic analysis windows."
                    )
                    config = get_default_unified_config()

        if args.weight is not None:
            config.setdefault("processing", {})["participant_weight_kg"] = float(args.weight)
            config["processing"]["participant_weight_source"] = "cli"

        configured_paths = config.get("paths", {})
        input_dir = args.input_dir or configured_paths.get("input_dir")
        if not input_dir and config_path is not None:
            candidate_dir = config_path.resolve().parent
            if os.path.isdir(candidate_dir):
                input_dir = str(candidate_dir)

        if not input_dir or not os.path.isdir(input_dir):
            console.print(
                f"[bold red]Error:[/bold red] Input directory not found or not specified: {input_dir}"
            )
            return 1

        processing = config.setdefault("processing", {})
        weight_source = str(processing.get("participant_weight_source", "fallback")).lower()
        if weight_source not in {"cli", "toml", "gui"}:
            detected_weight = infer_subject_weight_kg(input_dir)
            if detected_weight is not None:
                processing["participant_weight_kg"] = detected_weight
                processing["participant_weight_source"] = "metadata"

        requested_output_dir = args.output_dir or configured_paths.get("output_dir")
        try:
            config, input_path, output_path, step = prepare_cli_execution(
                config,
                input_dir=input_dir,
                output_dir=requested_output_dir,
                step=args.step,
            )
            execution = config.setdefault("execution", {})
            execution["force_auto"] = bool(args.force_auto)
            execution["timestamp_output"] = bool(
                args.timestamp_output or execution.get("timestamp_output", False)
            )
            if args.start_index is not None:
                execution["start_index"] = int(args.start_index)
            if args.end_index is not None:
                execution["end_index_exclusive"] = int(args.end_index)
            history_paths = save_cli_run_history(config, input_path, output_path)
        except OSError as exc:
            console.print(f"[bold red]Error preparing output/history:[/bold red] {exc}")
            return 1

        print_treadmill_config_summary(
            config,
            input_dir=input_path,
            output_dir=output_path,
            mode="Headless CLI",
            step=step,
        )
        print_run_history_paths(history_paths)

        if step == "all":
            print_pipeline_milestone(1, 3, "Signal Filtering", f"Input: {input_path}", "running")
            filtered_dir = run_filter_stage(
                parent=None,
                initial_dir=str(input_path),
                config=config,
                output_dir=output_path,
                headless=True,
            )
            if not filtered_dir:
                console.print("[bold red]Pipeline failed at filtering stage.[/bold red]")
                return 1
            print_pipeline_milestone(
                1, 3, "Signal Filtering Complete", f"Saved: {filtered_dir}", "success"
            )

            print_pipeline_milestone(
                2, 3, "Artifact Adjustment & Interpolation", f"Input: {filtered_dir}", "running"
            )
            clean_dir = run_adjust_stage(
                parent=None,
                initial_dir=filtered_dir,
                config=config,
                output_dir=output_path,
                headless=True,
            )
            if not clean_dir:
                console.print("[bold red]Pipeline failed at adjustment stage.[/bold red]")
                return 1

            adjusted_dir = run_interpolate_stage(
                parent=None,
                initial_dir=clean_dir,
                config=config,
                output_dir=output_path,
                headless=True,
            )
            interp_result_dir = adjusted_dir or clean_dir
            print_pipeline_milestone(
                2,
                3,
                "Adjustment & Interpolation Complete",
                f"Saved: {interp_result_dir}",
                "success",
            )

            print_pipeline_milestone(
                3,
                3,
                "Biomechanical Metrics Processing",
                f"Input: {interp_result_dir}",
                "running",
            )
            results_dir = run_process_stage(
                parent=None,
                initial_dir=interp_result_dir,
                config=config,
                output_dir=output_path,
                headless=True,
            )
            if not results_dir:
                console.print("[bold red]Pipeline failed at processing stage.[/bold red]")
                return 1
            print_pipeline_milestone(
                3, 3, "Full Pipeline Complete", f"Results saved to: {results_dir}", "success"
            )

        elif step == "filter":
            print_pipeline_milestone(1, 1, "Signal Filtering", f"Input: {input_path}", "running")
            res = run_filter_stage(
                parent=None,
                initial_dir=str(input_path),
                config=config,
                output_dir=output_path,
                headless=True,
            )
            if not res:
                return 1
            print_pipeline_milestone(1, 1, "Signal Filtering Complete", f"Saved: {res}", "success")

        elif step == "adjust":
            print_pipeline_milestone(1, 1, "Artifact Adjustment", f"Input: {input_path}", "running")
            res = run_adjust_stage(
                parent=None,
                initial_dir=str(input_path),
                config=config,
                output_dir=output_path,
                headless=True,
            )
            if not res:
                return 1
            print_pipeline_milestone(
                1, 1, "Artifact Adjustment Complete", f"Saved: {res}", "success"
            )

        elif step == "interpolate":
            print_pipeline_milestone(1, 1, "Interpolation", f"Input: {input_path}", "running")
            res = run_interpolate_stage(
                parent=None,
                initial_dir=str(input_path),
                config=config,
                output_dir=output_path,
                headless=True,
            )
            if not res:
                return 1
            print_pipeline_milestone(1, 1, "Interpolation Complete", f"Saved: {res}", "success")

        elif step == "process":
            print_pipeline_milestone(
                1,
                1,
                "Biomechanical Metrics Processing",
                f"Input: {input_path}",
                "running",
            )
            res = run_process_stage(
                parent=None,
                initial_dir=str(input_path),
                config=config,
                output_dir=output_path,
                headless=True,
            )
            if not res:
                return 1
            print_pipeline_milestone(1, 1, "Processing Complete", f"Saved: {res}", "success")
        update_cli_run_history(config, history_paths)
        return 0

    run_treadmill_lc_gui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
