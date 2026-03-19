"""
================================================================================
vailaplot3d.py
================================================================================
Author: Prof. Paulo Santiago
Created: 19 March 2026
Version: 0.0.2

Description:
------------
3D plotting Swiss-army-knife for vailá: Versatile Anarcho Integrated Liberation
Ánalysis in Multimodal Toolbox. Provides a Tkinter GUI with three plot types:

Plot Types Supported:
---------------------
1. 3D Trajectory: scatter / line plot of X, Y, Z coordinates with equal aspect.
   Auto-detects _X/_Y/_Z column triplets or accepts manual 3-column selection.
2. 3D Time Series: stacked subplots (X, Y, Z over time/frame) sharing the
   x-axis for each detected marker triplet.
3. Stick Figure 3D: frame-by-frame skeleton viewer with matplotlib slider and
   play/pause. Markers connected by user-defined or default body-model links.

Functionalities:
----------------
- GUI for plot type selection with status bar
- Clear All Plots / Clear Data / New Figure / Save Figure controls
- Cached file reading (CSV, C3D, XLSX, ODS)
- Auto-detection of coordinate triplets (_X, _Y, _Z suffixes)
================================================================================
"""

from __future__ import annotations

import contextlib
import gc
import os
from pathlib import Path
from tkinter import (
    END,
    MULTIPLE,
    Button,
    Frame,
    Label,
    LabelFrame,
    Listbox,
    Scrollbar,
    StringVar,
    Tk,
    Toplevel,
    filedialog,
    messagebox,
)

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button as MplButton
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from rich import print

try:
    import openpyxl  # noqa: F401

    EXCEL_SUPPORT = True
except ImportError:
    EXCEL_SUPPORT = False

ODS_SUPPORT = False
with contextlib.suppress(ImportError):
    from odf import opendocument  # type: ignore[import]  # noqa: F401

    ODS_SUPPORT = True

try:
    from ezc3d import c3d  # noqa: F401

    C3D_SUPPORT = True
except ImportError:
    C3D_SUPPORT = False

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
selected_files: list[str] = []
selected_headers: list[str] = []
plot_type: str | None = None
loaded_data_cache: dict[str, pd.DataFrame] = {}
current_figures: list[plt.Figure] = []

base_colors = ["r", "g", "b"]
additional_colors = list(mcolors.TABLEAU_COLORS.keys())
predefined_colors = base_colors + additional_colors

# Default body-model connections (MediaPipe-style landmark indices as names).
# Each tuple is (marker_A_stem, marker_B_stem).
DEFAULT_BODY_CONNECTIONS: list[tuple[str, str]] = [
    ("NOSE", "LEFT_EYE"),
    ("NOSE", "RIGHT_EYE"),
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),
]


# ---------------------------------------------------------------------------
# Utility helpers (self-contained so vailaplot3d works standalone)
# ---------------------------------------------------------------------------
def clear_plots() -> str:
    plt.close("all")
    global current_figures
    current_figures = []
    print("All 3D plots cleared!")
    return "All 3D plots cleared!"


def clear_data() -> str:
    global selected_files, selected_headers, loaded_data_cache, plot_type
    selected_files = []
    selected_headers = []
    loaded_data_cache = {}
    plot_type = None
    gc.collect()
    print("All data cleared!")
    return "All data cleared!"


def new_figure() -> str:
    plt.figure()
    current_figures.append(plt.gcf())
    print("New 3D figure created!")
    return "New 3D figure created!"


def save_figure() -> str:
    if not plt.get_fignums():
        return "No figure to save!"
    root = Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(
        title="Save Figure",
        defaultextension=".png",
        filetypes=[
            ("PNG files", "*.png"),
            ("PDF files", "*.pdf"),
            ("SVG files", "*.svg"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    if file_path:
        plt.savefig(file_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved to: {file_path}")
        return f"Figure saved to: {file_path}"
    return "Save cancelled."


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------
def read_csv_with_encoding(file_path: str, skipfooter: int = 0) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]
    read_args: dict = {"encoding": None}
    engine = None
    if skipfooter > 0:
        read_args["skipfooter"] = skipfooter
        engine = "python"
    if engine is not None:
        read_args["engine"] = engine

    for enc in encodings:
        try:
            read_args["encoding"] = enc
            return pd.read_csv(file_path, **read_args)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            if "codec" in str(e).lower() and "decode" in str(e).lower():
                continue
            raise

    read_args["encoding"] = "utf-8"
    read_args["errors"] = "replace"
    return pd.read_csv(file_path, **read_args)


def read_c3d_file(file_path: str) -> pd.DataFrame | None:
    if not C3D_SUPPORT:
        print("C3D support not available — ezc3d not installed")
        return None
    try:
        c3d_data = c3d(file_path, extract_forceplat_data=True)
        point_data = c3d_data["data"]["points"]
        marker_labels = c3d_data["parameters"]["POINT"]["LABELS"]["value"]
        if isinstance(marker_labels[0], list):
            marker_labels = marker_labels[0]

        marker_freq = c3d_data["header"]["points"]["frame_rate"]
        pts = point_data[:3, :, :]
        pts = np.transpose(pts, (2, 1, 0))

        valid = pts[~np.isnan(pts)]
        if len(valid) > 0 and np.mean(np.abs(valid)) > 100:
            pts = pts * 0.001

        data_dict: dict[str, np.ndarray] = {}
        for i, label in enumerate(marker_labels):
            if pts.shape[2] >= 3:
                data_dict[f"{label}_X"] = pts[:, i, 0]
                data_dict[f"{label}_Y"] = pts[:, i, 1]
                data_dict[f"{label}_Z"] = pts[:, i, 2]
        data_dict["Time"] = np.arange(pts.shape[0]) / marker_freq
        return pd.DataFrame(data_dict)
    except Exception as e:
        print(f"Error reading C3D file: {e}")
        return None


def _load_file(file_path: str) -> pd.DataFrame | None:
    """Load a data file into a DataFrame, using cache when available."""
    if file_path in loaded_data_cache:
        return loaded_data_cache[file_path]

    file_ext = file_path.lower().rsplit(".", 1)[-1]
    data: pd.DataFrame | None = None
    try:
        if file_ext == "csv":
            data = read_csv_with_encoding(file_path, skipfooter=0)
        elif file_ext == "xlsx":
            data = pd.read_excel(file_path)
        elif file_ext == "ods":
            data = pd.read_excel(file_path, engine="odf")
        elif file_ext == "c3d":
            data = read_c3d_file(file_path)
        else:
            data = read_csv_with_encoding(file_path, skipfooter=0)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    if data is not None:
        loaded_data_cache[file_path] = data
    return data


def get_file_headers(file_path: str) -> list[str]:
    df = _load_file(file_path)
    if df is None:
        return []
    return list(df.columns)


# ---------------------------------------------------------------------------
# Coordinate triplet detection
# ---------------------------------------------------------------------------
def detect_xyz_triplets(headers: list[str]) -> list[tuple[str, str, str]]:
    """Find _X/_Y/_Z (or .X/.Y/.Z) column triplets, case-insensitive suffix."""
    lower_map: dict[str, str] = {h.lower(): h for h in headers}
    triplets: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    for h in headers:
        hl = h.lower()
        if hl in seen:
            continue
        for sx, sy, sz in [("_x", "_y", "_z"), (".x", ".y", ".z")]:
            if hl.endswith(sx):
                stem = hl[: -len(sx)]
                py, pz = stem + sy, stem + sz
                if (
                    py in lower_map
                    and pz in lower_map
                    and lower_map[py].lower() not in seen
                    and lower_map[pz].lower() not in seen
                ):
                    triplets.append((lower_map[hl], lower_map[py], lower_map[pz]))
                    seen.update([hl, py, pz])
                    break
    return triplets


# ---------------------------------------------------------------------------
# GUI classes
# ---------------------------------------------------------------------------
class PlotGUI3D:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("vailá Plot 3D")
        self.root.geometry("900x500")

        self.status_var = StringVar()
        self.status_var.set("Ready")
        self._create_widgets()

    def _create_widgets(self) -> None:
        main_frame = Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill="both", expand=True)

        Label(main_frame, text="vailá 3D Plotting Tool", font=("Arial", 16, "bold")).pack(
            pady=(0, 20)
        )

        # --- Plot type buttons ---
        plot_frame = LabelFrame(main_frame, text="Select Plot Type", padx=15, pady=15)
        plot_frame.pack(fill="x", pady=10)
        self.plot_buttons: list[Button] = []

        for col, (label, ptype) in enumerate(
            [
                ("3D Trajectory", "trajectory_3d"),
                ("3D Time Series", "timeseries_3d"),
                ("Stick Figure", "stickfigure_3d"),
            ]
        ):
            btn = Button(
                plot_frame,
                text=label,
                command=lambda pt=ptype: self._set_plot_type(pt),
                width=20,
                height=2,
                font=("Arial", 10),
            )
            btn.grid(row=0, column=col, padx=8, pady=8, sticky="ew")
            self.plot_buttons.append(btn)

        # --- Control buttons ---
        ctrl_frame = LabelFrame(main_frame, text="Plot Controls", padx=15, pady=15)
        ctrl_frame.pack(fill="x", pady=10)

        for col, (label, cmd) in enumerate(
            [
                ("Clear All Plots", self._clear_plots),
                ("Clear All Data", self._clear_data),
                ("New Figure", self._new_figure),
                ("Save Figure", self._save_figure),
            ]
        ):
            Button(
                ctrl_frame, text=label, command=cmd, width=16, height=2, font=("Arial", 10)
            ).grid(row=0, column=col, padx=8, pady=8, sticky="ew")

        # --- Status bar ---
        status_frame = Frame(main_frame)
        status_frame.pack(fill="x", pady=(10, 0))
        Label(status_frame, textvariable=self.status_var, font=("Arial", 10), anchor="w").pack(
            fill="x"
        )

    # -- callbacks --
    def _set_plot_type(self, ptype: str) -> None:
        global plot_type
        plot_type = ptype
        self.status_var.set(f"Selected: {ptype}")
        FileSelectionWindow3D(self.root, self, ptype)

    def _clear_plots(self) -> None:
        self.status_var.set(clear_plots())

    def _clear_data(self) -> None:
        self.status_var.set(clear_data())

    def _new_figure(self) -> None:
        self.status_var.set(new_figure())

    def _save_figure(self) -> None:
        self.status_var.set(save_figure())


class FileSelectionWindow3D:
    """File & header picker — mirrors vailaplot2d.FileSelectionWindow."""

    def __init__(self, parent_root: Tk, parent: PlotGUI3D, ptype: str):
        self.parent = parent
        self.plot_type = ptype
        self.selected_files: list[str] = []
        self.selected_headers: list[str] = []

        self.window = Toplevel(parent_root)
        self.window.title(f"Select Files & Headers — {ptype}")
        self.window.geometry("850x600")
        self._create_widgets()

    def _create_widgets(self) -> None:
        main = Frame(self.window, padx=10, pady=10)
        main.pack(fill="both", expand=True)

        # File selection
        file_frame = LabelFrame(main, text="Selected Files", padx=10, pady=10)
        file_frame.pack(fill="x", pady=5)

        self.file_listbox = Listbox(file_frame, height=4, width=90, font=("Arial", 9))
        self.file_listbox.pack(side="left", fill="x", expand=True)

        file_btn_frame = Frame(file_frame)
        file_btn_frame.pack(side="right", padx=5)
        Button(file_btn_frame, text="Add Files", command=self._add_files, width=12).pack(pady=2)
        Button(file_btn_frame, text="Clear Files", command=self._clear_files, width=12).pack(pady=2)

        # Header selection
        header_frame = LabelFrame(
            main, text="Available Headers (select multiple)", padx=10, pady=10
        )
        header_frame.pack(fill="both", expand=True, pady=5)

        sb = Scrollbar(header_frame)
        sb.pack(side="right", fill="y")
        self.header_listbox = Listbox(
            header_frame,
            selectmode=MULTIPLE,
            width=90,
            height=15,
            yscrollcommand=sb.set,
            font=("Arial", 9),
        )
        self.header_listbox.pack(side="left", fill="both", expand=True)
        sb.config(command=self.header_listbox.yview)

        # Action buttons
        btn_frame = Frame(main)
        btn_frame.pack(fill="x", pady=10)
        Button(btn_frame, text="Select All", command=self._select_all, width=12).pack(
            side="left", padx=5
        )
        Button(btn_frame, text="Clear Selection", command=self._clear_selection, width=14).pack(
            side="left", padx=5
        )
        Button(
            btn_frame,
            text="Plot",
            command=self._on_plot,
            width=16,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold"),
        ).pack(side="right", padx=5)

    # -- file ops --
    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select data files",
            filetypes=[
                ("All supported", "*.csv;*.c3d;*.xlsx;*.ods"),
                ("CSV", "*.csv"),
                ("C3D", "*.c3d"),
                ("Excel", "*.xlsx"),
                ("ODS", "*.ods"),
                ("All", "*.*"),
            ],
        )
        for p in paths:
            if p not in self.selected_files:
                self.selected_files.append(p)
                self.file_listbox.insert(END, os.path.basename(p))
        if self.selected_files:
            self._refresh_headers()

    def _clear_files(self) -> None:
        self.selected_files.clear()
        self.file_listbox.delete(0, END)
        self.header_listbox.delete(0, END)

    def _refresh_headers(self) -> None:
        all_headers: list[str] = []
        for fp in self.selected_files:
            for h in get_file_headers(fp):
                if h not in all_headers:
                    all_headers.append(h)
        self.header_listbox.delete(0, END)
        for i, h in enumerate(all_headers):
            self.header_listbox.insert(END, f"{i + 1:3d}: {h}")
        self._all_headers = all_headers

    def _select_all(self) -> None:
        self.header_listbox.select_set(0, END)

    def _clear_selection(self) -> None:
        self.header_listbox.selection_clear(0, END)

    def _on_plot(self) -> None:
        sel_indices = self.header_listbox.curselection()
        self.selected_headers = [self._all_headers[i] for i in sel_indices]

        if not self.selected_files or not self.selected_headers:
            messagebox.showwarning("Warning", "Select at least one file and one header.")
            return

        global selected_files, selected_headers
        selected_files = self.selected_files.copy()
        selected_headers = self.selected_headers.copy()

        new_figure()

        if self.plot_type == "trajectory_3d":
            plot_3d_trajectory()
        elif self.plot_type == "timeseries_3d":
            plot_3d_timeseries()
        elif self.plot_type == "stickfigure_3d":
            plot_3d_stickfigure()

        if hasattr(self.parent, "status_var"):
            self.parent.status_var.set(f"Plot created: {self.plot_type}")
        self.window.destroy()


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------
def plot_3d_trajectory() -> None:
    """Plot 3D scatter/trajectory lines with equal aspect box."""
    fig = plt.gcf()
    ax = fig.add_subplot(111, projection="3d")
    plot_count = 0

    for file_idx, fp in enumerate(selected_files):
        data = _load_file(fp)
        if data is None:
            continue

        available = [h for h in selected_headers if h in data.columns]
        triplets = detect_xyz_triplets(available)

        if not triplets and len(available) >= 3:
            triplets = [
                (available[i], available[i + 1], available[i + 2])
                for i in range(0, len(available) - 2, 3)
            ]

        if not triplets:
            print(f"[WARNING] No valid XYZ triplets for {os.path.basename(fp)}")
            continue

        for t_idx, (cx, cy, cz) in enumerate(triplets):
            color = predefined_colors[
                (file_idx * max(len(triplets), 1) + t_idx) % len(predefined_colors)
            ]
            x = data[cx].to_numpy()
            y = data[cy].to_numpy()
            z = data[cz].to_numpy()
            stem = cx.rsplit("_", 1)[0] if "_" in cx else cx
            label = f"{os.path.basename(fp)} — {stem}"

            ax.plot(x, y, z, color=color, linewidth=1.2, label=label)
            ax.scatter(x[0], y[0], z[0], color=color, marker="o", s=40, zorder=5)
            ax.scatter(x[-1], y[-1], z[-1], color=color, marker="s", s=40, zorder=5)
            plot_count += 1

    if plot_count == 0:
        messagebox.showwarning("Warning", "No valid 3D data found. Need _X/_Y/_Z triplets.")
        return

    _set_equal_aspect_3d(ax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Trajectory (equal aspect)")
    ax.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.show()


def _set_equal_aspect_3d(ax) -> None:
    """Force equal aspect ratio on a 3D axis by adjusting limits."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = limits.mean(axis=1)
    half_range = (limits[:, 1] - limits[:, 0]).max() / 2.0
    ax.set_xlim3d(center[0] - half_range, center[0] + half_range)
    ax.set_ylim3d(center[1] - half_range, center[1] + half_range)
    ax.set_zlim3d(center[2] - half_range, center[2] + half_range)
    with contextlib.suppress(AttributeError):
        ax.set_box_aspect([1, 1, 1])


def plot_3d_timeseries() -> None:
    """Stacked X, Y, Z subplots over time for each marker triplet."""
    all_triplets_per_file: list[tuple[str, pd.DataFrame, list[tuple[str, str, str]]]] = []

    for fp in selected_files:
        data = _load_file(fp)
        if data is None:
            continue
        available = [h for h in selected_headers if h in data.columns]
        triplets = detect_xyz_triplets(available)
        if not triplets and len(available) >= 3:
            triplets = [
                (available[i], available[i + 1], available[i + 2])
                for i in range(0, len(available) - 2, 3)
            ]
        if triplets:
            all_triplets_per_file.append((fp, data, triplets))

    if not all_triplets_per_file:
        messagebox.showwarning("Warning", "No XYZ triplets found for time-series plot.")
        return

    total_triplets = sum(len(t) for _, _, t in all_triplets_per_file)
    fig, axes = plt.subplots(3, total_triplets, squeeze=False, sharex="col")
    fig.suptitle("3D Time Series", fontsize=14)

    col_idx = 0
    for fp, data, triplets in all_triplets_per_file:
        time_col = None
        for candidate in ["Time", "time", "Frame", "frame"]:
            if candidate in data.columns:
                time_col = candidate
                break
        x_vals = data[time_col].to_numpy() if time_col else np.arange(len(data))

        for cx, cy, cz in triplets:
            stem = cx.rsplit("_", 1)[0] if "_" in cx else cx
            for row, (col, axis_label) in enumerate([(cx, "X"), (cy, "Y"), (cz, "Z")]):
                ax = axes[row, col_idx]
                color = predefined_colors[col_idx % len(predefined_colors)]
                ax.plot(x_vals, data[col].to_numpy(), color=color, linewidth=1)
                ax.set_ylabel(axis_label)
                ax.grid(True, alpha=0.3)
                if row == 0:
                    ax.set_title(f"{os.path.basename(fp)}\n{stem}", fontsize=9)
                if row == 2:
                    ax.set_xlabel(time_col or "Frame")
            col_idx += 1

    plt.tight_layout()
    plt.show()


def plot_3d_stickfigure() -> None:
    """Interactive 3D stick-figure viewer with slider and play/pause."""
    if not selected_files:
        messagebox.showwarning("Warning", "No files selected.")
        return

    fp = selected_files[0]
    data = _load_file(fp)
    if data is None:
        messagebox.showwarning("Warning", f"Cannot read {fp}")
        return

    available = [h for h in selected_headers if h in data.columns]
    triplets = detect_xyz_triplets(available)

    if not triplets:
        messagebox.showwarning(
            "Warning",
            "No _X/_Y/_Z triplets detected. Select columns with coordinate suffixes.",
        )
        return

    stems = [cx.rsplit("_", 1)[0] if "_" in cx else cx for cx, _, _ in triplets]
    connections = _resolve_connections(stems)

    n_frames = len(data)
    coords = np.zeros((n_frames, len(triplets), 3))
    for i, (cx, cy, cz) in enumerate(triplets):
        coords[:, i, 0] = data[cx].to_numpy()
        coords[:, i, 1] = data[cy].to_numpy()
        coords[:, i, 2] = data[cz].to_numpy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.20)

    scatter = ax.scatter([], [], [], s=30, c="blue", depthshade=True)
    lines: list = []
    for _ in connections:
        (line,) = ax.plot([], [], [], "gray", linewidth=1.5)
        lines.append(line)

    labels: list = []
    for stem in stems:
        labels.append(ax.text(0, 0, 0, stem, fontsize=6, alpha=0.7))

    _set_global_limits(ax, coords)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Stick Figure — {os.path.basename(fp)}")

    ax_slider = plt.axes((0.15, 0.08, 0.65, 0.03))
    slider = Slider(ax_slider, "Frame", 0, n_frames - 1, valinit=0, valstep=1)

    ax_play = plt.axes((0.82, 0.08, 0.08, 0.03))
    btn_play = MplButton(ax_play, "Play")

    animation_state = {"playing": False, "timer": None}

    def update_frame(frame_idx: int | float) -> None:
        frame_idx = int(frame_idx)
        pts = coords[frame_idx]
        scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])

        for li, (ia, ib) in enumerate(connections):
            lines[li].set_data_3d(
                [pts[ia, 0], pts[ib, 0]],
                [pts[ia, 1], pts[ib, 1]],
                [pts[ia, 2], pts[ib, 2]],
            )

        for li, txt in enumerate(labels):
            txt.set_position_3d((pts[li, 0], pts[li, 1], pts[li, 2]))

        fig.canvas.draw_idle()

    def on_slider_change(val: float) -> None:
        update_frame(val)

    slider.on_changed(on_slider_change)

    def _animate() -> None:
        if not animation_state["playing"]:
            return
        cur = int(slider.val)
        nxt = (cur + 1) % n_frames
        slider.set_val(nxt)
        animation_state["timer"] = fig.canvas.new_timer(interval=33)
        animation_state["timer"].add_callback(_animate)
        animation_state["timer"].start()

    def on_play(event) -> None:
        animation_state["playing"] = not animation_state["playing"]
        btn_play.label.set_text("Pause" if animation_state["playing"] else "Play")
        if animation_state["playing"]:
            _animate()
        elif animation_state["timer"] is not None:
            animation_state["timer"].stop()

    btn_play.on_clicked(on_play)

    update_frame(0)
    plt.show()


def _resolve_connections(
    stems: list[str],
) -> list[tuple[int, int]]:
    """Build index-based connection list from stem names and default body model."""
    stem_upper = [s.upper() for s in stems]
    stem_index = {s: i for i, s in enumerate(stem_upper)}
    connections: list[tuple[int, int]] = []

    for a, b in DEFAULT_BODY_CONNECTIONS:
        if a in stem_index and b in stem_index:
            connections.append((stem_index[a], stem_index[b]))

    if not connections:
        for i in range(len(stems) - 1):
            connections.append((i, i + 1))

    return connections


def _set_global_limits(ax, coords: np.ndarray) -> None:
    """Set axis limits to encompass all frames."""
    valid = coords[~np.isnan(coords).any(axis=-1)]
    if len(valid) == 0:
        return
    mins = valid.min(axis=0)
    maxs = valid.max(axis=0)
    center = (mins + maxs) / 2
    half_range = (maxs - mins).max() / 2 * 1.1
    ax.set_xlim3d(center[0] - half_range, center[0] + half_range)
    ax.set_ylim3d(center[1] - half_range, center[1] + half_range)
    ax.set_zlim3d(center[2] - half_range, center[2] + half_range)
    with contextlib.suppress(AttributeError):
        ax.set_box_aspect([1, 1, 1])


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def select_plot_type_3d() -> None:
    root = Tk()
    PlotGUI3D(root)
    root.mainloop()


def run_plot_3d() -> None:
    """Main entry point — called from vaila.py."""
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent.resolve()}")
    select_plot_type_3d()
    if current_figures:
        messagebox.showinfo("Plotting Completed", "All 3D plots have been generated.")


if __name__ == "__main__":
    run_plot_3d()
