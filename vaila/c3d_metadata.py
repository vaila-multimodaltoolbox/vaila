"""
===============================================================================
c3d_metadata.py
===============================================================================
Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 09 September 2026
Update Date: 09 September 2026
Version: 0.3.131

Description:
Module and CLI tool for inspecting, modifying, and creating C3D file metadata
in the vailá toolbox.

Capabilities:
- Inspect metadata: Manufacturer (company/software), Point Rate (FPS in Hz),
  Point Units ("mm", "m"), Number of Markers and marker labels, Number of Frames,
  Analog Rate (Hz), Number of Analog Channels, channel labels and units,
  Force platform counts, and Trial parameters.
- Edit metadata of existing C3D files:
  - Manufacturer / Company / Software tags
  - Marker frequency (FPS in Hz) with automatic proportional synchronization
    of analog frequency (subframe ratio preservation required by C3D/ezc3d)
  - Point units ("mm" vs "m") with optional 3D coordinate scaling
  - Analog frequency and channel units
  - Automatic string sanitization (fixes non-UTF8 surrogate escape characters
    such as 'mm/s²' from legacy systems that crash SWIG bindings on write)
  - Safe, non-destructive write (never overwrites without explicit request)
- Create new C3D files from metadata templates:
  - Generate valid, compliant C3D files with specified FPS, frames, marker counts,
    and analog channels from scratch.
- Dual interface:
  - Headless CLI with copy-pasteable reproduction commands
  - Tkinter / ttk graphical editor with inspection tabs and interactive editing

Usage:
    CLI inspection:
        uv run vaila/c3d_metadata.py --show path/to/file.c3d

    CLI modification:
        uv run vaila/c3d_metadata.py -i input.c3d -o output.c3d --fps 120 --manufacturer "Qualisys"

    CLI creation:
        uv run vaila/c3d_metadata.py --create -o new.c3d --fps 100 --frames 200 --markers 12 --analogs 4

    GUI:
        uv run vaila/c3d_metadata.py
"""

from __future__ import annotations

import argparse
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np

try:
    import ezc3d
except ImportError:
    ezc3d = None

try:
    from .cli_highlight import print_gui_cli_mirror
except ImportError:
    try:
        from cli_highlight import print_gui_cli_mirror  # ty: ignore[unresolved-import]
    except ImportError:

        def print_gui_cli_mirror(name: str, args: list[str]) -> None:
            cmd = " ".join(args)
            print(f">> {name}: Equivalent CLI:\n>>   {cmd}")


def sanitize_str(s: Any) -> str:
    """Sanitize strings for ezc3d SWIG C++ compatibility.

    C3D files created on Windows/Latin-1 systems often contain characters
    like '²' (e.g. in 'mm/s²') which Python's surrogateescape decoder turns
    into lone surrogates. Passing these back to SWIG std::string raises
    TypeError/ValueError. This helper recovers standard UTF-8 text safely.
    """
    if not isinstance(s, str):
        return str(s) if s is not None else ""
    try:
        # Check if it encodes cleanly to utf-8
        s.encode("utf-8")
        return s
    except UnicodeEncodeError:
        try:
            return (
                s.encode("utf-8", errors="surrogateescape")
                .decode("latin1", errors="replace")
                .replace("²", "^2")
            )
        except Exception:
            return s.encode("ascii", errors="replace").decode("ascii")


def sanitize_c3d_parameters(c: Any) -> None:
    """In-place sanitize all string parameter values inside an ezc3d object."""
    if not hasattr(c, "get") or "parameters" not in c:
        return
    for group in c["parameters"]:
        for param in c["parameters"][group]:
            entry = c["parameters"][group][param]
            if isinstance(entry, dict) and "value" in entry:
                val = entry["value"]
                if isinstance(val, list) and val and isinstance(val[0], str):
                    entry["value"] = [sanitize_str(x) for x in val]


def get_all_point_labels(point_param: dict[str, Any]) -> list[str]:
    """Extract marker labels from POINT group, merging LABELS, LABELS2, etc."""
    labels: list[str] = []
    if not point_param:
        return labels
    base_labels = point_param.get("LABELS", {}).get("value", [])
    labels.extend([sanitize_str(x).strip() for x in base_labels])
    suffix = 2
    while f"LABELS{suffix}" in point_param:
        extra = point_param[f"LABELS{suffix}"].get("value", [])
        labels.extend([sanitize_str(x).strip() for x in extra])
        suffix += 1
    return labels


def read_c3d_metadata(c3d_path: str | Path) -> dict[str, Any]:
    """Read metadata from a C3D file without loading or modifying raw data.

    Returns a dict with:
      - file_path, file_name, file_size_kb
      - manufacturer_company, manufacturer_software, manufacturer_version
      - point_rate, point_units, num_markers, marker_labels, num_frames, duration_sec
      - analog_rate, num_analogs, analog_labels, analog_units, subframe_ratio
      - num_force_platforms, num_events, event_labels, camera_rate
    """
    if ezc3d is None:
        raise RuntimeError("ezc3d library is not installed. Install with `uv sync`.")

    path_str = str(c3d_path)
    if not os.path.isfile(path_str):
        raise FileNotFoundError(f"C3D file not found: {path_str}")

    c = ezc3d.c3d(path_str)
    params = c.get("parameters", {})
    header = c.get("header", {})

    # Manufacturer
    mfg = params.get("MANUFACTURER", {})
    mfg_company = sanitize_str(mfg.get("COMPANY", {}).get("value", ["Unknown"])[0])
    mfg_software = sanitize_str(mfg.get("SOFTWARE", {}).get("value", [""])[0])
    mfg_version = sanitize_str(mfg.get("VERSION_LABEL", {}).get("value", [""])[0])

    # Point info
    point_grp = params.get("POINT", {})
    point_rate = float(
        point_grp.get("RATE", {}).get("value", [header.get("points", {}).get("frame_rate", 100.0)])[
            0
        ]
    )
    point_units = sanitize_str(point_grp.get("UNITS", {}).get("value", ["mm"])[0])
    marker_labels = get_all_point_labels(point_grp)
    num_markers = int(point_grp.get("USED", {}).get("value", [len(marker_labels)])[0])

    # Point data shape / frames
    data_points = c.get("data", {}).get("points")
    if data_points is not None and hasattr(data_points, "shape") and len(data_points.shape) == 3:
        num_frames = int(data_points.shape[2])
    else:
        num_frames = int(point_grp.get("FRAMES", {}).get("value", [0])[0])

    duration_sec = (num_frames / point_rate) if point_rate > 0 else 0.0

    # Analog info
    analog_grp = params.get("ANALOG", {})
    analog_rate = float(
        analog_grp.get("RATE", {}).get(
            "value", [header.get("analogs", {}).get("frame_rate", 1000.0)]
        )[0]
    )
    analog_labels_raw = analog_grp.get("LABELS", {}).get("value", [])
    analog_labels = [sanitize_str(x).strip() for x in analog_labels_raw]
    analog_units_raw = analog_grp.get("UNITS", {}).get("value", [])
    analog_units = [sanitize_str(x).strip() for x in analog_units_raw]
    num_analogs = int(analog_grp.get("USED", {}).get("value", [len(analog_labels)])[0])
    subframe_ratio = (analog_rate / point_rate) if point_rate > 0 else 1.0

    # Force platforms
    has_platforms = "platform" in c.get("data", {})
    num_platforms = len(c["data"]["platform"]) if has_platforms else 0

    # Events
    events_grp = params.get("EVENT", {})
    event_labels_raw = events_grp.get("LABELS", {}).get("value", [])
    event_labels = [sanitize_str(x).strip() for x in event_labels_raw]
    num_events = len(event_labels)

    # Trial
    trial_grp = params.get("TRIAL", {})
    cam_rate_val = trial_grp.get("CAMERA_RATE", {}).get("value", [point_rate])
    camera_rate = float(cam_rate_val[0]) if cam_rate_val else point_rate

    return {
        "file_path": path_str,
        "file_name": os.path.basename(path_str),
        "file_size_kb": os.path.getsize(path_str) / 1024.0,
        "manufacturer_company": mfg_company,
        "manufacturer_software": mfg_software,
        "manufacturer_version": mfg_version,
        "manufacturer": {
            "company": mfg_company,
            "software": mfg_software,
            "version": mfg_version,
        },
        "point_rate": point_rate,
        "point_units": point_units,
        "num_markers": num_markers,
        "marker_labels": marker_labels,
        "num_frames": num_frames,
        "duration_sec": duration_sec,
        "analog_rate": analog_rate,
        "num_analogs": num_analogs,
        "analog_labels": analog_labels,
        "analog_units": analog_units,
        "subframe_ratio": subframe_ratio,
        "num_force_platforms": num_platforms,
        "num_events": num_events,
        "event_labels": event_labels,
        "camera_rate": camera_rate,
    }


def format_c3d_metadata_summary(meta: dict[str, Any], max_labels: int = 10) -> str:
    """Format metadata dictionary into human-readable text."""
    lines = [
        "================================================================================",
        f"C3D File: {meta['file_name']} ({meta['file_size_kb']:.1f} KB)",
        f"Path:     {meta['file_path']}",
        "================================================================================",
        f"Manufacturer:       {meta['manufacturer_company']} {meta['manufacturer_software']} {meta['manufacturer_version']}".strip(),
        f"Point Frequency:    {meta['point_rate']:.2f} Hz (FPS)",
        f"Point Units:        {meta['point_units']}",
        f"Number of Markers:  {meta['num_markers']}",
        f"Number of Frames:   {meta['num_frames']} ({meta['duration_sec']:.2f} seconds)",
        f"Analog Frequency:   {meta['analog_rate']:.2f} Hz (ratio: {meta['subframe_ratio']:.1f}x)",
        f"Analog Channels:    {meta['num_analogs']}",
        f"Force Platforms:    {meta['num_force_platforms']}",
        f"Events:             {meta['num_events']}",
    ]

    # Markers preview
    labels = meta.get("marker_labels", [])
    if labels:
        if len(labels) <= max_labels:
            lines.append(f"Marker Labels:      {', '.join(labels)}")
        else:
            preview = ", ".join(labels[:max_labels])
            lines.append(f"Marker Labels:      {preview} ... (+{len(labels) - max_labels} more)")

    # Analog labels preview
    a_labels = meta.get("analog_labels", [])
    if a_labels:
        if len(a_labels) <= max_labels:
            lines.append(f"Analog Channels:    {', '.join(a_labels)}")
        else:
            preview = ", ".join(a_labels[:max_labels])
            lines.append(f"Analog Channels:    {preview} ... (+{len(a_labels) - max_labels} more)")

    lines.append("================================================================================")
    return "\n".join(lines)


def update_c3d_metadata(
    input_path: str | Path,
    output_path: str | Path,
    *,
    manufacturer: str | None = None,
    software: str | None = None,
    point_rate: float | None = None,
    point_units: str | None = None,
    analog_rate: float | None = None,
    scale_coordinates: bool = False,
    analog_units: str | None = None,
) -> str:
    """Modify metadata of an existing C3D file and save to output_path.

    Preserves 3D points, residuals, camera masks, analog samples, platforms,
    and event contexts. Maintains analog subframe ratio synchronization.
    """
    if ezc3d is None:
        raise RuntimeError("ezc3d library is not installed.")

    in_str = str(input_path)
    out_str = str(output_path)

    if not os.path.isfile(in_str):
        raise FileNotFoundError(f"Input C3D not found: {in_str}")

    c = ezc3d.c3d(in_str, extract_forceplat_data=False)
    sanitize_c3d_parameters(c)

    # Manufacturer updates
    if manufacturer is not None or software is not None:
        if "MANUFACTURER" not in c["parameters"]:
            c.add_parameter("MANUFACTURER", "COMPANY", [manufacturer or "vailá"])
        else:
            if manufacturer is not None:
                c["parameters"]["MANUFACTURER"]["COMPANY"]["value"] = [sanitize_str(manufacturer)]
            if software is not None:
                c["parameters"]["MANUFACTURER"]["SOFTWARE"]["value"] = [sanitize_str(software)]

    # Point frequency & Analog frequency synchronization
    old_p_rate = float(c["parameters"]["POINT"]["RATE"]["value"][0])
    old_a_rate = (
        float(c["parameters"]["ANALOG"]["RATE"]["value"][0])
        if "ANALOG" in c["parameters"] and "RATE" in c["parameters"]["ANALOG"]
        else 0.0
    )
    subframes = (old_a_rate / old_p_rate) if old_p_rate > 0 else 1.0

    if point_rate is not None and point_rate > 0:
        c["parameters"]["POINT"]["RATE"]["value"] = [float(point_rate)]
        c["header"]["points"]["frame_rate"] = float(point_rate)
        if "TRIAL" in c["parameters"] and "CAMERA_RATE" in c["parameters"]["TRIAL"]:
            c["parameters"]["TRIAL"]["CAMERA_RATE"]["value"] = [float(point_rate)]

        # Maintain exact subframe ratio required by ezc3d data arrays
        if old_a_rate > 0:
            synced_a_rate = float(point_rate * subframes)
            c["parameters"]["ANALOG"]["RATE"]["value"] = [synced_a_rate]
            c["header"]["analogs"]["frame_rate"] = synced_a_rate

    if analog_rate is not None and analog_rate > 0:
        c["parameters"]["ANALOG"]["RATE"]["value"] = [float(analog_rate)]
        c["header"]["analogs"]["frame_rate"] = float(analog_rate)

    # Point units & Coordinate scaling
    if point_units is not None:
        target_units = point_units.strip().lower()
        curr_units = sanitize_str(c["parameters"]["POINT"]["UNITS"]["value"][0]).strip().lower()

        if scale_coordinates and target_units != curr_units:
            scale_factor = 1.0
            if curr_units == "mm" and target_units == "m":
                scale_factor = 0.001
            elif curr_units == "m" and target_units == "mm":
                scale_factor = 1000.0

            if scale_factor != 1.0 and "points" in c["data"]:
                # Scale XYZ (channels 0, 1, 2); leave channel 3 (homogeneous 1.0)
                c["data"]["points"][0:3, :, :] *= scale_factor

        c["parameters"]["POINT"]["UNITS"]["value"] = [sanitize_str(point_units)]

    # Analog units
    if (
        analog_units is not None
        and "ANALOG" in c["parameters"]
        and "UNITS" in c["parameters"]["ANALOG"]
    ):
        num_a = len(c["parameters"]["ANALOG"]["UNITS"]["value"])
        c["parameters"]["ANALOG"]["UNITS"]["value"] = [sanitize_str(analog_units)] * num_a

    # Ensure output dir exists
    out_dir = os.path.dirname(os.path.abspath(out_str))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    c.write(out_str)
    return out_str


def create_c3d(
    output_path: str | Path,
    *,
    num_frames: int = 100,
    point_rate: float = 100.0,
    point_units: str = "mm",
    marker_labels: list[str] | None = None,
    num_markers: int = 4,
    analog_rate: float = 1000.0,
    analog_labels: list[str] | None = None,
    num_analogs: int = 0,
    analog_units: str = "V",
    manufacturer: str = "vailá",
    software: str = "vailá multimodal toolbox",
) -> str:
    """Create a new valid C3D template file with specified metadata."""
    if ezc3d is None:
        raise RuntimeError("ezc3d library is not installed.")

    out_str = str(output_path)
    c = ezc3d.c3d()

    # Determine labels
    if marker_labels is not None:
        m_labels = [sanitize_str(lbl) for lbl in marker_labels]
        n_markers = len(m_labels)
    else:
        n_markers = max(1, num_markers)
        m_labels = [f"M{i + 1}" for i in range(n_markers)]

    if analog_labels is not None:
        a_labels = [sanitize_str(lbl) for lbl in analog_labels]
        n_analogs = len(a_labels)
    else:
        n_analogs = max(0, num_analogs)
        a_labels = [f"Channel_{i + 1}" for i in range(n_analogs)] if n_analogs > 0 else []

    # Configure POINT parameters
    c["parameters"]["POINT"]["USED"]["value"] = [n_markers]
    c["parameters"]["POINT"]["FRAMES"]["value"] = [num_frames]
    c["parameters"]["POINT"]["RATE"]["value"] = [float(point_rate)]
    c["parameters"]["POINT"]["UNITS"]["value"] = [sanitize_str(point_units)]
    c["parameters"]["POINT"]["LABELS"]["value"] = m_labels
    c["header"]["points"]["frame_rate"] = float(point_rate)

    # Configure ANALOG parameters
    if n_analogs > 0:
        subframes = max(1, int(round(analog_rate / point_rate)))
        synced_analog_rate = float(point_rate * subframes)
        total_analog_frames = num_frames * subframes
        c["parameters"]["ANALOG"]["USED"]["value"] = [n_analogs]
        c["parameters"]["ANALOG"]["RATE"]["value"] = [synced_analog_rate]
        c["parameters"]["ANALOG"]["LABELS"]["value"] = a_labels
        c["parameters"]["ANALOG"]["UNITS"]["value"] = [sanitize_str(analog_units)] * n_analogs
        c["header"]["analogs"]["frame_rate"] = synced_analog_rate
        c["data"]["analogs"] = np.zeros((1, n_analogs, total_analog_frames), dtype=np.float64)

    # Add Manufacturer group
    c.add_parameter("MANUFACTURER", "COMPANY", [sanitize_str(manufacturer)])
    c.add_parameter("MANUFACTURER", "SOFTWARE", [sanitize_str(software)])

    # Initialize data arrays
    c["data"]["points"] = np.zeros((4, n_markers, num_frames), dtype=np.float64)
    c["data"]["points"][3, :, :] = 1.0  # Homogeneous coordinate
    c["data"]["meta_points"]["residuals"] = np.zeros((1, n_markers, num_frames), dtype=np.float64)

    out_dir = os.path.dirname(os.path.abspath(out_str))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    c.write(out_str)
    return out_str


class C3DMetadataGUI(tk.Toplevel):
    """Interactive GUI window for inspecting and editing C3D metadata."""

    def __init__(self, parent: tk.Tk | tk.Toplevel | None = None, initial_file: str | None = None):
        super().__init__(parent)
        self.title("vailá - C3D Metadata Editor & Creator")
        self.geometry("780x680")
        self.minsize(640, 500)

        self.current_c3d_path: str | None = None
        self.current_meta: dict[str, Any] | None = None

        self._create_widgets()

        if initial_file and os.path.isfile(initial_file):
            self.load_file(initial_file)

    def _create_widgets(self) -> None:
        style = ttk.Style()
        style.configure("Header.TLabel", font=("Arial", 12, "bold"))
        style.configure("Bold.TLabel", font=("Arial", 9, "bold"))

        # Top frame: File selection
        top_frame = ttk.LabelFrame(self, text="File Selection", padding=10)
        top_frame.pack(fill=tk.X, padx=10, pady=5)

        self.path_var = tk.StringVar()
        entry = ttk.Entry(top_frame, textvariable=self.path_var, width=60)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        btn_browse = ttk.Button(top_frame, text="Browse C3D...", command=self.browse_file)
        btn_browse.pack(side=tk.LEFT, padx=5)

        btn_create_new = ttk.Button(
            top_frame, text="Create Blank C3D...", command=self.open_create_dialog
        )
        btn_create_new.pack(side=tk.LEFT, padx=5)

        # Middle Notebook: Tabs for Inspector & Editor
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Tab 1: Edit Metadata
        self.tab_edit = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_edit, text="Edit Metadata")
        self._build_edit_tab(self.tab_edit)

        # Tab 2: Full Inspection Report
        self.tab_inspect = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_inspect, text="Inspector View")
        self._build_inspect_tab(self.tab_inspect)

        # Bottom Frame: Actions & Status
        bot_frame = ttk.Frame(self, padding=10)
        bot_frame.pack(fill=tk.X, padx=10, pady=5)

        self.status_var = tk.StringVar(value="Select or create a C3D file to begin.")
        lbl_status = ttk.Label(bot_frame, textvariable=self.status_var, font=("Arial", 9, "italic"))
        lbl_status.pack(side=tk.LEFT, fill=tk.X, expand=True)

        btn_save_as = ttk.Button(
            bot_frame, text="Save Metadata As...", command=self.save_metadata_as
        )
        btn_save_as.pack(side=tk.RIGHT, padx=5)

        btn_close = ttk.Button(bot_frame, text="Close", command=self.destroy)
        btn_close.pack(side=tk.RIGHT, padx=5)

    def _build_edit_tab(self, parent: ttk.Frame) -> None:
        grid_frame = ttk.Frame(parent)
        grid_frame.pack(fill=tk.BOTH, expand=True)

        # Manufacturer
        ttk.Label(grid_frame, text="Manufacturer / Company:", style="Bold.TLabel").grid(
            row=0, column=0, sticky="w", pady=5
        )
        self.mfg_var = tk.StringVar()
        self.mfg_combo = ttk.Combobox(
            grid_frame,
            textvariable=self.mfg_var,
            values=[
                "Vicon",
                "Qualisys",
                "Motion Analysis",
                "BTS Bioengineering",
                "OptiTrack",
                "vailá",
                "Custom",
            ],
            width=30,
        )
        self.mfg_combo.grid(row=0, column=1, sticky="w", pady=5, padx=5)

        # Software
        ttk.Label(grid_frame, text="Software Name:", style="Bold.TLabel").grid(
            row=1, column=0, sticky="w", pady=5
        )
        self.software_var = tk.StringVar()
        ttk.Entry(grid_frame, textvariable=self.software_var, width=32).grid(
            row=1, column=1, sticky="w", pady=5, padx=5
        )

        # Point Rate (FPS)
        ttk.Label(grid_frame, text="Point FPS (Hz):", style="Bold.TLabel").grid(
            row=2, column=0, sticky="w", pady=5
        )
        self.point_fps_var = tk.StringVar()
        ttk.Entry(grid_frame, textvariable=self.point_fps_var, width=15).grid(
            row=2, column=1, sticky="w", pady=5, padx=5
        )

        # Point Units
        ttk.Label(grid_frame, text="Point Units:", style="Bold.TLabel").grid(
            row=3, column=0, sticky="w", pady=5
        )
        self.point_units_var = tk.StringVar()
        self.units_combo = ttk.Combobox(
            grid_frame, textvariable=self.point_units_var, values=["mm", "m"], width=10
        )
        self.units_combo.grid(row=3, column=1, sticky="w", pady=5, padx=5)

        # Scale coordinates checkbox
        self.scale_coords_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            grid_frame,
            text="Rescale 3D coordinates when changing units (e.g. mm ↔ m)",
            variable=self.scale_coords_var,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=5)

        # Analog Rate (Hz)
        ttk.Label(grid_frame, text="Analog Rate (Hz):", style="Bold.TLabel").grid(
            row=5, column=0, sticky="w", pady=5
        )
        self.analog_rate_var = tk.StringVar()
        ttk.Entry(grid_frame, textvariable=self.analog_rate_var, width=15).grid(
            row=5, column=1, sticky="w", pady=5, padx=5
        )

        # Default Analog Unit
        ttk.Label(grid_frame, text="Analog Units (Default):", style="Bold.TLabel").grid(
            row=6, column=0, sticky="w", pady=5
        )
        self.analog_units_var = tk.StringVar()
        ttk.Entry(grid_frame, textvariable=self.analog_units_var, width=15).grid(
            row=6, column=1, sticky="w", pady=5, padx=5
        )

        # Information note
        info_text = (
            "Note: Changing Point FPS will automatically scale Analog FPS proportionally\n"
            "to preserve exact subframe synchrony required by ezc3d and biomechanical standards."
        )
        lbl_info = ttk.Label(grid_frame, text=info_text, foreground="gray")
        lbl_info.grid(row=7, column=0, columnspan=2, sticky="w", pady=15)

    def _build_inspect_tab(self, parent: ttk.Frame) -> None:
        self.report_text = tk.Text(parent, wrap="word", font=("Courier", 10))
        scroll = ttk.Scrollbar(parent, orient="vertical", command=self.report_text.yview)
        self.report_text.configure(yscrollcommand=scroll.set)
        self.report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def browse_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select C3D File",
            filetypes=[("C3D Files", "*.c3d"), ("All Files", "*.*")],
            parent=self,
        )
        if path:
            self.load_file(path)

    def load_file(self, path: str) -> None:
        try:
            self.status_var.set(f"Loading {os.path.basename(path)}...")
            meta = read_c3d_metadata(path)
            self.current_c3d_path = path
            self.current_meta = meta
            self.path_var.set(path)

            # Populate edit fields
            self.mfg_var.set(meta["manufacturer_company"])
            self.software_var.set(meta["manufacturer_software"])
            self.point_fps_var.set(str(meta["point_rate"]))
            self.point_units_var.set(meta["point_units"])
            self.analog_rate_var.set(str(meta["analog_rate"]))
            default_a_unit = meta["analog_units"][0] if meta["analog_units"] else "V"
            self.analog_units_var.set(default_a_unit)

            # Populate report text
            self.report_text.delete("1.0", tk.END)
            self.report_text.insert(tk.END, format_c3d_metadata_summary(meta, max_labels=30))

            self.status_var.set(
                f"Loaded: {meta['file_name']} ({meta['num_markers']} markers, {meta['point_rate']} Hz)"
            )
        except Exception as e:
            messagebox.showerror("Error Loading C3D", f"Could not read metadata from {path}:\n{e}")
            self.status_var.set("Error loading file.")

    def save_metadata_as(self) -> None:
        if not self.current_c3d_path:
            messagebox.showwarning("No File Loaded", "Please open a C3D file first.")
            return

        out_path = filedialog.asksaveasfilename(
            title="Save Modified C3D As",
            defaultextension=".c3d",
            filetypes=[("C3D Files", "*.c3d")],
            initialfile=f"{Path(self.current_c3d_path).stem}_meta.c3d",
            parent=self,
        )
        if not out_path:
            return

        try:
            point_rate = (
                float(self.point_fps_var.get().strip())
                if self.point_fps_var.get().strip()
                else None
            )
            analog_rate = (
                float(self.analog_rate_var.get().strip())
                if self.analog_rate_var.get().strip()
                else None
            )
            mfg = self.mfg_var.get().strip() or None
            sw = self.software_var.get().strip() or None
            units = self.point_units_var.get().strip() or None
            scale = self.scale_coords_var.get()
            a_units = self.analog_units_var.get().strip() or None

            update_c3d_metadata(
                self.current_c3d_path,
                out_path,
                manufacturer=mfg,
                software=sw,
                point_rate=point_rate,
                point_units=units,
                analog_rate=analog_rate,
                scale_coordinates=scale,
                analog_units=a_units,
            )

            # Print CLI mirror
            argv = [
                "uv",
                "run",
                "vaila/c3d_metadata.py",
                "-i",
                self.current_c3d_path,
                "-o",
                out_path,
            ]
            if point_rate:
                argv.extend(["--fps", str(point_rate)])
            if mfg:
                argv.extend(["--manufacturer", mfg])
            if units:
                argv.extend(["--point-units", units])
            if scale:
                argv.append("--scale-coords")
            print_gui_cli_mirror("vaila/c3d_metadata", argv)

            messagebox.showinfo("Metadata Saved", f"Successfully saved updated C3D to:\n{out_path}")
            self.status_var.set(f"Saved: {os.path.basename(out_path)}")
        except Exception as e:
            messagebox.showerror("Error Saving C3D", f"Could not update C3D:\n{e}")
            self.status_var.set("Error during save.")

    def open_create_dialog(self) -> None:
        dlg = tk.Toplevel(self)
        dlg.title("Create Blank C3D Template")
        dlg.geometry("450x380")
        dlg.transient(self)

        ttk.Label(dlg, text="New C3D Parameters", font=("Arial", 11, "bold")).pack(pady=10)

        f = ttk.Frame(dlg, padding=10)
        f.pack(fill=tk.BOTH, expand=True)

        ttk.Label(f, text="Frames:").grid(row=0, column=0, sticky="w", pady=4)
        frames_var = tk.StringVar(value="100")
        ttk.Entry(f, textvariable=frames_var, width=12).grid(row=0, column=1, sticky="w", pady=4)

        ttk.Label(f, text="FPS (Hz):").grid(row=1, column=0, sticky="w", pady=4)
        fps_var = tk.StringVar(value="100.0")
        ttk.Entry(f, textvariable=fps_var, width=12).grid(row=1, column=1, sticky="w", pady=4)

        ttk.Label(f, text="Number of Markers:").grid(row=2, column=0, sticky="w", pady=4)
        markers_var = tk.StringVar(value="4")
        ttk.Entry(f, textvariable=markers_var, width=12).grid(row=2, column=1, sticky="w", pady=4)

        ttk.Label(f, text="Units:").grid(row=3, column=0, sticky="w", pady=4)
        units_var = tk.StringVar(value="mm")
        ttk.Combobox(f, textvariable=units_var, values=["mm", "m"], width=10).grid(
            row=3, column=1, sticky="w", pady=4
        )

        ttk.Label(f, text="Analog Channels:").grid(row=4, column=0, sticky="w", pady=4)
        analogs_var = tk.StringVar(value="0")
        ttk.Entry(f, textvariable=analogs_var, width=12).grid(row=4, column=1, sticky="w", pady=4)

        ttk.Label(f, text="Manufacturer:").grid(row=5, column=0, sticky="w", pady=4)
        mfg_var = tk.StringVar(value="vailá")
        ttk.Entry(f, textvariable=mfg_var, width=20).grid(row=5, column=1, sticky="w", pady=4)

        def do_create():
            out = filedialog.asksaveasfilename(
                title="Save Blank C3D As",
                defaultextension=".c3d",
                filetypes=[("C3D Files", "*.c3d")],
                initialfile="blank_template.c3d",
                parent=dlg,
            )
            if not out:
                return
            try:
                create_c3d(
                    out,
                    num_frames=int(frames_var.get().strip()),
                    point_rate=float(fps_var.get().strip()),
                    num_markers=int(markers_var.get().strip()),
                    point_units=units_var.get().strip(),
                    num_analogs=int(analogs_var.get().strip()),
                    manufacturer=mfg_var.get().strip(),
                )
                dlg.destroy()
                self.load_file(out)
                messagebox.showinfo("C3D Created", f"Successfully created C3D:\n{out}")
            except Exception as e:
                messagebox.showerror("Error Creating C3D", f"Could not create C3D:\n{e}")

        ttk.Button(dlg, text="Create C3D File...", command=do_create).pack(pady=15)


def run_c3d_metadata_gui(initial_file: str | None = None) -> None:
    """Launch the C3D Metadata GUI editor."""
    root = getattr(tk, "_default_root", None)
    created = False
    if root is None:
        root = tk.Tk()
        root.withdraw()
        created = True

    app = C3DMetadataGUI(root, initial_file=initial_file)
    if created:
        app.protocol("WM_DELETE_WINDOW", root.destroy)
        root.mainloop()


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect, modify, or create C3D file metadata.")
    parser.add_argument("--show", help="Inspect and show C3D metadata")
    parser.add_argument("-i", "--input", dest="input_file", help="Input C3D file to modify")
    parser.add_argument("-o", "--output", dest="output_file", help="Output C3D file")
    parser.add_argument("--fps", type=float, help="Point frequency (FPS in Hz) to set")
    parser.add_argument("--point-units", choices=["mm", "m"], help="Point coordinates unit")
    parser.add_argument(
        "--scale-coords",
        action="store_true",
        help="Scale 3D points when changing units (mm ↔ m)",
    )
    parser.add_argument("--analog-rate", type=float, help="Analog sampling frequency in Hz")
    parser.add_argument(
        "--manufacturer", help="Manufacturer company name (e.g. Vicon, Qualisys, vailá)"
    )
    parser.add_argument("--software", help="Software name to store in metadata")
    parser.add_argument("--create", action="store_true", help="Create a new template C3D file")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames for new C3D")
    parser.add_argument("--markers", type=int, default=4, help="Number of markers for new C3D")
    parser.add_argument(
        "--analogs", type=int, default=0, help="Number of analog channels for new C3D"
    )

    parsed_args = parser.parse_args(args)

    # Mode 1: Inspection
    if parsed_args.show:
        meta = read_c3d_metadata(parsed_args.show)
        print(format_c3d_metadata_summary(meta))
        return 0

    # Mode 2: Create
    if parsed_args.create:
        if not parsed_args.output_file:
            print("Error: --output is required when --create is used.")
            return 1
        out = create_c3d(
            parsed_args.output_file,
            num_frames=parsed_args.frames,
            point_rate=parsed_args.fps or 100.0,
            point_units=parsed_args.point_units or "mm",
            num_markers=parsed_args.markers,
            num_analogs=parsed_args.analogs,
            manufacturer=parsed_args.manufacturer or "vailá",
            software=parsed_args.software or "vailá multimodal toolbox",
        )
        print(f"Successfully created C3D: {out}")
        return 0

    # Mode 3: Modify
    if parsed_args.input_file:
        if not parsed_args.output_file:
            print("Error: --output is required when modifying a C3D file.")
            return 1
        out = update_c3d_metadata(
            parsed_args.input_file,
            parsed_args.output_file,
            manufacturer=parsed_args.manufacturer,
            software=parsed_args.software,
            point_rate=parsed_args.fps,
            point_units=parsed_args.point_units,
            analog_rate=parsed_args.analog_rate,
            scale_coordinates=parsed_args.scale_coords,
        )
        print(f"Successfully updated C3D metadata: {out}")
        return 0

    # Default: Run GUI
    run_c3d_metadata_gui()
    return 0


if __name__ == "__main__":
    sys.exit(main())
