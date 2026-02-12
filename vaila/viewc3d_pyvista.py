"""
================================================================================
Script: viewc3d_pyvista.py
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/paulopreto/vaila-multimodaltoolbox
Please see AUTHORS for contributors.

================================================================================
Author: Paulo Santiago
Version: 0.2.0
Created: 06 February 2025
Last Updated: 11 February 2026

To run:
  uv run viewc3d_pyvista.py [path/to/file.c3d]
  python -m vaila.viewc3d_pyvista [path/to/file.c3d]
  CLI help:  python -m vaila.viewc3d_pyvista --help
  GUI help:  press H in the viewer (opens HTML)

Dependencies:
  pip install pyvista ezc3d numpy

Description:
------------
VTK-based 3D viewer for C3D and CSV marker data (PyVista backend).
Timeline, interactive marker picking, skeleton connections, trails,
export (screenshot, PNG sequence, MP4), quality stats. Same palette
and marker visibility options as the Open3D viewer (viewc3d.py).

    Architecture:
    - MokkaLikeViewer: single class (state, init_gui, update_frame, key handlers)
    - Load from C3D (load_c3d) or from arrays (from_array for CSV)

    Key Features:
    - C3D and CSV support; automatic unit detection (mm/m)
    - Left-click to select marker (name shown on screen)
    - C cycles marker color (Orange, Blue, Green, Red, White, Yellow, …)
    - M: dialog to show/hide markers
    - View presets (1–4), background cycle (B), grid (G), labels (X)
    - Trail (T), speed ([ ]), marker size (+ −), skeleton from JSON (J)
    - Export: K screenshot, Z PNG sequence, V MP4
    - Distance mode (D): click two markers to measure
    - Info (I), quality stats (A), help (H)

    Keyboard Shortcuts (see H in viewer for full list):
    Navigation: Space Play | ← → ±1 | ↑ ↓ ±10 | PgUp/PgDn ±100 | S Start | End End
    View:      R Reset | 1–4 Presets | B Background | G Grid | X Labels | C Colors
    Data:      T Trail | { } Trail length | [ ] Speed | + − Size | M Markers
    Skeleton:  J Load JSON
    Export:    K Screenshot | Z PNG seq | V MP4
    Info:      I Info | A Stats | D Distance | H Help | Escape Clear

    Mouse:
    Left click – Select marker (shows name)
    Left drag – Rotate | Middle/Right drag – Pan | Wheel – Zoom

License:
    Affero General Public License v3.0 - AGPL-3.0
"""

import argparse
import contextlib
import json
import os
import sys
import tempfile
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog

import ezc3d
import numpy as np
import pyvista as pv

# ---------------------------------------------------------------------------
#  Colors: same palette as viewc3d (Open3D viewer) for C key cycle
# ---------------------------------------------------------------------------
AVAILABLE_COLORS = [
    ([1.0, 0.65, 0.0], "Orange"),
    ([0.0, 0.5, 1.0], "Blue"),
    ([0.0, 1.0, 0.0], "Green"),
    ([1.0, 0.0, 0.0], "Red"),
    ([1.0, 1.0, 1.0], "White"),
    ([1.0, 1.0, 0.0], "Yellow"),
    ([0.5, 0.0, 1.0], "Purple"),
    ([0.0, 1.0, 1.0], "Cyan"),
    ([1.0, 0.0, 1.0], "Pink"),
    ([0.5, 0.5, 0.5], "Gray"),
    ([0.0, 0.0, 0.0], "Black"),
]


def _marker_color(name, idx=0):
    """Return a consistent RGB tuple [0-1] for a marker name (used for per-marker hash)."""
    h = hash(name) if name else idx * 137
    r = ((h & 0xFF0000) >> 16) / 255.0
    g = ((h & 0x00FF00) >> 8) / 255.0
    b = (h & 0x0000FF) / 255.0
    mx = max(r, g, b)
    if mx < 0.3:
        r, g, b = r + 0.3, g + 0.3, b + 0.3
    return (r, g, b)


# ═══════════════════════════════════════════════════════════════════════════
#  MokkaLikeViewer
# ═══════════════════════════════════════════════════════════════════════════
class MokkaLikeViewer:
    """PyVista-based 3D viewer for C3D and CSV marker data."""

    # ── Shared default state factory ──────────────────────────────────────
    def _init_state(self):
        """Initialise ALL viewer state (called from __init__ and from_array)."""
        # Data
        self.points_data = None
        self.labels: list[str] = []
        self.n_frames = 0
        self.n_markers = 0
        self.current_frame = 0
        self.playing = False
        self.frame_rate = 60.0
        self.units = "m"
        self.c3d_events: list[dict] = []  # [{frame, label, context}, ...]
        self.analog_info: dict = {}  # {n_channels, rate}

        # PyVista objects
        self.plotter = None
        self.point_cloud = None
        self.point_cloud_actor = None
        self._valid_indices: np.ndarray = np.array([], dtype=int)  # Bug fix #1

        # UI state
        self._sel_text_actor = None
        self.trail_frames = 0
        self.trail_actor = None
        self._last_trail_frame = -1  # Bug fix #2: cache guard
        self.play_speed = 1.0
        self._speed_accumulator = 0.0
        self._timer_step = 0  # for Windows TimerEvent observer
        self._feedback_actor = None
        self._point_size = 10
        self._plotter_title = "Vaila - PyVista Viewer"

        # Phase 1 display toggles
        self._show_labels = False
        self._label_actors: list = []
        self._bg_mode = 0  # 0=dark 1=gray 2=white
        self._grid_visible = True
        self._grid_actor = None
        self._floor_actor = None
        self._axes_widget = None

        # Phase 2 skeleton
        self._skeleton_pairs: list[tuple[str, str]] = []
        self._skeleton_actor = None

        # Phase 3 info / colors: C cycles through AVAILABLE_COLORS (same as viewc3d)
        self._color_index = 0
        self._last_color_index: int | None = None  # force rebuild when color changes

        # Phase 5
        self._distance_mode = False
        self._distance_picks: list = []
        self._distance_actor = None

        # Marker visibility: None = all visible, else set of marker names to show
        self._visible_markers: set[str] | None = None

    # ── Constructor (C3D) ─────────────────────────────────────────────────
    def __init__(self, c3d_path=None):
        self.c3d_path = c3d_path
        self._init_state()
        self._plotter_title = "Vaila - PyVista C3D Viewer"

        if not self.c3d_path:
            self.select_file()
        if self.c3d_path:
            self.load_c3d()
            self.init_gui()
        else:
            print("No file selected.")

    # ── Constructor (numpy array – used by CSV viewer) ────────────────────
    @classmethod
    def from_array(cls, points, labels, frame_rate=60.0, title=None):
        """Create viewer from numpy array (n_frames, n_markers, 3)."""
        self = cls.__new__(cls)
        self.c3d_path = None
        self._init_state()
        self.points_data = np.nan_to_num(np.asarray(points, dtype=np.float64), nan=-999.0)
        self.labels = list(labels)
        self.n_frames = self.points_data.shape[0]
        self.n_markers = self.points_data.shape[1]
        self.frame_rate = float(frame_rate)
        self._plotter_title = title or "Vaila - PyVista CSV Viewer"
        self.init_gui()
        return self

    # ── File selection ────────────────────────────────────────────────────
    def select_file(self):
        root = tk.Tk()
        root.withdraw()
        self.c3d_path = filedialog.askopenfilename(
            title="Select a C3D file",
            filetypes=[("C3D Files", "*.c3d")],
        )
        root.destroy()

    # ── C3D loading ───────────────────────────────────────────────────────
    def load_c3d(self):
        print(f"Loading: {self.c3d_path}...")
        c = ezc3d.c3d(self.c3d_path)

        points = c["data"]["points"]
        self.labels = c["parameters"]["POINT"]["LABELS"]["value"]
        self.frame_rate = c["parameters"]["POINT"]["RATE"]["value"][0]
        self.n_frames = points.shape[2]
        self.n_markers = points.shape[1]
        print(f"Loaded {self.n_markers} markers, {self.n_frames} frames.")

        self.points_data = np.transpose(points[:3, :, :], (2, 1, 0))

        # Auto-detect units
        max_coord = np.nanmax(np.abs(self.points_data))
        if max_coord > 5000:
            self.units = "mm"
            self.points_data *= 0.001
            print("Detected: Millimeters (converting to meters)")
        else:
            self.units = "m"
            print("Detected: Meters")

        self.points_data = np.nan_to_num(self.points_data, nan=-999.0)

        # ── Load events ──
        try:
            ep = c["parameters"].get("EVENT", {})
            if ep:
                evt_labels = ep.get("LABELS", {}).get("value", [])
                evt_contexts = ep.get("CONTEXTS", {}).get("value", [])
                evt_times = ep.get("TIMES", {}).get("value", [])
                first_frame = (
                    c["parameters"]["POINT"].get("FRAMES", {}).get("value", [0])[0]
                    if "FRAMES" in c["parameters"]["POINT"]
                    else 0
                )
                for i, lab in enumerate(evt_labels):
                    t = evt_times[1][i] if len(evt_times) > 1 else 0
                    fr = int(round(t * self.frame_rate)) - first_frame
                    ctx = evt_contexts[i] if i < len(evt_contexts) else ""
                    self.c3d_events.append({"frame": fr, "label": lab, "context": ctx})
                if self.c3d_events:
                    print(f"Loaded {len(self.c3d_events)} events.")
        except Exception:
            pass

        # ── Analog info ──
        try:
            an = c["parameters"].get("ANALOG", {})
            n_ch = an.get("USED", {}).get("value", [0])[0] if an else 0
            a_rate = an.get("RATE", {}).get("value", [0])[0] if an else 0
            self.analog_info = {"n_channels": int(n_ch), "rate": float(a_rate)}
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════
    #  Frame update
    # ══════════════════════════════════════════════════════════════════════
    def update_frame(self, frame_idx):
        """Update geometry for the given frame."""
        frame_idx = int(frame_idx)
        if frame_idx >= self.n_frames:
            frame_idx = 0
        self.current_frame = frame_idx

        current_points = self.points_data[frame_idx].copy()
        valid_mask = ~np.all(current_points == -999.0, axis=1)
        # Apply marker visibility filter
        if self._visible_markers is not None:
            for i in range(len(self.labels)):
                if (
                    i < len(valid_mask)
                    and valid_mask[i]
                    and self.labels[i] not in self._visible_markers
                ):
                    valid_mask[i] = False
        self._valid_indices = np.where(valid_mask)[0]  # Bug fix #1
        valid_points = current_points[valid_mask]
        if valid_points.size == 0:
            valid_points = np.array([[0.0, 0.0, 0.0]])
            self._valid_indices = np.array([], dtype=int)

        # Point cloud: replace actor when point count or color index changes
        n_pts = len(valid_points)
        color_changed = self._last_color_index != self._color_index
        do_rebuild = (
            self.point_cloud_actor is None
            or (self.point_cloud.n_points != n_pts if self.point_cloud is not None else True)
            or color_changed
        )
        if do_rebuild:
            if self.point_cloud_actor is not None:
                self.plotter.remove_actor(self.point_cloud_actor)
                self.point_cloud_actor = None
            self.point_cloud = pv.PolyData(valid_points.copy())
            rgb_tuple, _ = AVAILABLE_COLORS[self._color_index]
            self.point_cloud_actor = self.plotter.add_mesh(
                self.point_cloud,
                color=rgb_tuple,
                point_size=self._point_size,
                render_points_as_spheres=True,
                pickable=True,
            )
            self._last_color_index = self._color_index
        else:
            self.point_cloud.points = valid_points
            if self.point_cloud_actor is not None:
                self.point_cloud_actor.GetMapper().Update()

        # Status text
        time_s = frame_idx / self.frame_rate if self.frame_rate > 0 else 0
        speed_str = f"  Speed: {self.play_speed}x" if self.play_speed != 1.0 else ""
        n_vis = len(self._valid_indices)
        self.text_actor.set_text(
            "lower_right",
            f"Frame {frame_idx}/{self.n_frames - 1}  t={time_s:.3f}s  vis={n_vis}/{self.n_markers}{speed_str}",
        )

        # Trail
        if self.trail_frames > 0 and frame_idx != self._last_trail_frame:
            self._update_trail(frame_idx)
            self._last_trail_frame = frame_idx

        # Skeleton
        self._update_skeleton(frame_idx)

        # Labels
        if self._show_labels:
            self._draw_labels(valid_points, self._valid_indices)

    # ══════════════════════════════════════════════════════════════════════
    #  Playback
    # ══════════════════════════════════════════════════════════════════════
    def toggle_play(self, state):
        self.playing = state

    def _win_timer_observer(self, _obj, _evt):
        """TimerEvent observer for Windows when add_timer_event does not fire reliably."""
        self._timer_step += 1
        self.animation_callback(self._timer_step)

    def animation_callback(self, step):
        if not self.playing:
            return
        self._speed_accumulator += self.play_speed
        while self._speed_accumulator >= 1.0:
            self._speed_accumulator -= 1.0
            idx = self.current_frame + 1
            if idx >= self.n_frames:
                idx = 0
            self.slider_widget.GetRepresentation().SetValue(idx)
            self.update_frame(idx)

    # ══════════════════════════════════════════════════════════════════════
    #  Picking  (Bug fix #1: use _valid_indices)
    # ══════════════════════════════════════════════════════════════════════
    def on_pick(self, *args, **kwargs):
        """Callback for point picking (use_picker=True -> args=(point, picker))."""
        if len(args) >= 2 and hasattr(args[1], "GetPointId"):
            pick_idx = args[1].GetPointId()
        elif len(args) >= 1 and isinstance(args[0], (int, np.integer)):
            pick_idx = int(args[0])
        else:
            return

        # Map filtered index -> original marker index
        if pick_idx < 0 or len(self._valid_indices) == 0:
            return
        if pick_idx < len(self._valid_indices):
            orig_idx = self._valid_indices[pick_idx]
        else:
            return
        if orig_idx >= len(self.labels):
            return

        marker_name = self.labels[orig_idx]

        # Distance mode
        if self._distance_mode:
            self._distance_picks.append(
                (marker_name, self.points_data[self.current_frame, orig_idx].copy())
            )
            if len(self._distance_picks) == 2:
                self._draw_distance()
                self._distance_mode = False
            else:
                self._show_feedback(f"D: First marker = {marker_name}. Click second.")
            return

        print(f"Marker selected: {marker_name} (idx={orig_idx})")
        with contextlib.suppress(Exception):
            self.plotter.remove_actor("sel_text")
        self._sel_text_actor = self.plotter.add_text(
            f"Selected: {marker_name}",
            position="upper_left",
            name="sel_text",
            font_size=12,
            color="yellow",
        )

    # ══════════════════════════════════════════════════════════════════════
    #  Feedback
    # ══════════════════════════════════════════════════════════════════════
    def _show_feedback(self, msg):
        print(f"[Viewer] {msg}")
        if not self.plotter:
            return
        with contextlib.suppress(Exception):
            self.plotter.remove_actor("key_feedback")
        self._feedback_actor = self.plotter.add_text(
            msg,
            position="upper_right",
            name="key_feedback",
            font_size=11,
            color="wheat",
        )

    def show_help(self, _key=None):
        help_path = Path(__file__).resolve().parent / "help" / "view3d_pyvista_help.html"
        if help_path.exists():
            webbrowser.open(help_path.as_uri())
            self._show_feedback("H: Help opened")
            return
        html = "<html><body><h2>See view3d_pyvista_help.html</h2></body></html>"
        tmp = Path(tempfile.gettempdir()) / "view3d_pyvista_help.html"
        tmp.write_text(html, encoding="utf-8")
        webbrowser.open(tmp.as_uri())

    # ══════════════════════════════════════════════════════════════════════
    #  KEY HANDLERS — Navigation
    # ══════════════════════════════════════════════════════════════════════
    def _goto_frame(self, idx):
        idx = idx % self.n_frames
        self.slider_widget.GetRepresentation().SetValue(idx)
        self.update_frame(idx)

    def _key_prev_frame(self, _key=None):
        self._goto_frame(self.current_frame - 1)

    def _key_next_frame(self, _key=None):
        self._goto_frame(self.current_frame + 1)

    def _key_back10(self, _key=None):
        self._goto_frame(self.current_frame - 10)
        self._show_feedback("↑: −10 frames")

    def _key_fwd10(self, _key=None):
        self._goto_frame(self.current_frame + 10)
        self._show_feedback("↓: +10 frames")

    def _key_back100(self, _key=None):
        self._goto_frame(self.current_frame - 100)
        self._show_feedback("PgUp: −100 frames")

    def _key_fwd100(self, _key=None):
        self._goto_frame(self.current_frame + 100)
        self._show_feedback("PgDn: +100 frames")

    def _key_toggle_play(self, _key=None):
        self.playing = not self.playing
        with contextlib.suppress(Exception):
            self._play_checkbox.GetRepresentation().SetState(self.playing)
        self._show_feedback("Space: Play" if self.playing else "Space: Pause")

    def _win_key_observer(self, obj, _evt):
        """Low-level key observer for Windows: VTK may not forward Space/arrows to add_key_event.
        On Windows we handle Space only here (add_key_event does not register Space) to avoid double-fire."""
        try:
            key_sym = obj.GetKeySym()
            key_code = obj.GetKeyCode()
        except Exception:
            return
        if key_sym in ("space", " ") or key_code == 32:
            self._key_toggle_play()
            return
        if key_sym == "Left":
            self._key_prev_frame()
        elif key_sym == "Right":
            self._key_next_frame()
        elif key_sym == "Up":
            self._key_back10()
        elif key_sym == "Down":
            self._key_fwd10()

    def _key_start(self, _key=None):
        self._goto_frame(0)
        self._show_feedback("S: Start (frame 0)")

    def _key_end(self, _key=None):
        self._goto_frame(self.n_frames - 1)
        self._show_feedback("End: Last frame")

    # ══════════════════════════════════════════════════════════════════════
    #  KEY HANDLERS — View presets  (1-4)
    # ══════════════════════════════════════════════════════════════════════
    def _apply_background(self):
        """Re-apply current background to avoid VTK state corruption after view changes."""
        bgs = [(0.1, 0.1, 0.1), (0.3, 0.3, 0.3), (1.0, 1.0, 1.0)]
        self.plotter.set_background(bgs[self._bg_mode])
        self.plotter.render()

    def _key_view_front(self, _key=None):
        self.plotter.view_xz()
        self.plotter.camera.up = (0, 0, 1)
        self._apply_background()
        self._show_feedback("1: Front view (XZ)")

    def _key_view_right(self, _key=None):
        self.plotter.view_yz()
        self.plotter.camera.up = (0, 0, 1)
        self._apply_background()
        self._show_feedback("2: Right view (YZ)")

    def _key_view_top(self, _key=None):
        self.plotter.view_xy()
        self.plotter.camera.up = (0, 1, 0)
        self._apply_background()
        self._show_feedback("3: Top view (XY)")

    def _key_view_iso(self, _key=None):
        self.plotter.view_isometric()
        self.plotter.camera.up = (0, 0, 1)
        self._apply_background()
        self._show_feedback("4: Isometric view")

    def _key_reset_camera(self, _key=None):
        self.plotter.reset_camera()
        self.plotter.camera.view_angle = 30.0
        self._show_feedback("R: Reset camera")

    # ══════════════════════════════════════════════════════════════════════
    #  KEY HANDLERS — Display toggles
    # ══════════════════════════════════════════════════════════════════════
    def _key_escape(self, _key=None):
        with contextlib.suppress(Exception):
            self.plotter.remove_actor("sel_text")
        self._sel_text_actor = None
        self._distance_mode = False
        self._distance_picks.clear()
        self._show_feedback("Escape: Cleared")

    def _key_toggle_labels(self, _key=None):
        self._show_labels = not self._show_labels
        if not self._show_labels:
            self._clear_labels()
        else:
            self.update_frame(self.current_frame)
        self._show_feedback(f"X: Labels {'ON' if self._show_labels else 'OFF'}")

    def _draw_labels(self, valid_points, valid_indices):
        self._clear_labels()
        for i, orig_idx in enumerate(valid_indices):
            if i < len(valid_points) and orig_idx < len(self.labels):
                pos = valid_points[i]
                name = self.labels[orig_idx]
                self.plotter.add_point_labels(
                    pv.PolyData(pos.reshape(1, 3)),
                    [name],
                    font_size=14,
                    point_color="yellow",
                    point_size=0,
                    render_points_as_spheres=False,
                    name=f"_lbl_{orig_idx}",
                    text_color="white",
                    shape=None,
                    always_visible=True,
                )
                self._label_actors.append(f"_lbl_{orig_idx}")

    def _clear_labels(self):
        for name in self._label_actors:
            with contextlib.suppress(Exception):
                self.plotter.remove_actor(name)
        self._label_actors.clear()

    def _key_toggle_bg(self, _key=None):
        bgs = [(0.1, 0.1, 0.1), (0.3, 0.3, 0.3), (1.0, 1.0, 1.0)]
        names = ["Dark", "Gray", "White"]
        self._bg_mode = (self._bg_mode + 1) % len(bgs)
        self._apply_background()
        self._show_feedback(f"B: Background {names[self._bg_mode]}")

    def _key_toggle_grid(self, _key=None):
        self._grid_visible = not self._grid_visible
        if self._floor_actor is not None:
            self._floor_actor.SetVisibility(self._grid_visible)
        self._show_feedback(f"G: Grid {'ON' if self._grid_visible else 'OFF'}")

    def _key_toggle_colors(self, _key=None):
        self._color_index = (self._color_index + 1) % len(AVAILABLE_COLORS)
        _, color_name = AVAILABLE_COLORS[self._color_index]
        self._show_feedback(f"C: {color_name}")
        self.update_frame(self.current_frame)

    def _key_marker_visibility(self, _key=None):
        """Open dialog to choose which markers to display. Key: M."""
        if not self.labels:
            self._show_feedback("M: No markers")
            return
        win = tk.Toplevel()
        win.title("Show / hide markers")
        win.geometry("380x420")
        vars_by_name = {}
        inner = tk.Frame(win, padx=10, pady=10)
        inner.pack(fill=tk.BOTH, expand=True)
        tk.Label(inner, text="Check markers to display (uncheck to hide):", font=("", 10)).pack(
            anchor=tk.W
        )
        scroll_frame = tk.Frame(inner)
        scroll_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        scrollbar = tk.Scrollbar(scroll_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas = tk.Canvas(scroll_frame, yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=canvas.yview)
        cb_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=cb_frame, anchor=tk.NW)
        for name in self.labels:
            var = tk.BooleanVar(
                value=self._visible_markers is None or name in self._visible_markers
            )
            vars_by_name[name] = var
            tk.Checkbutton(cb_frame, text=name, variable=var).pack(anchor=tk.W)

        def on_configure(_event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        cb_frame.bind("<Configure>", on_configure)

        def all_none(all_checked):
            for v in vars_by_name.values():
                v.set(all_checked)

        def apply_and_close():
            selected = {name for name, v in vars_by_name.items() if v.get()}
            if len(selected) == len(self.labels):
                self._visible_markers = None
            else:
                self._visible_markers = selected
            win.destroy()
            self._last_color_index = None  # force rebuild with new marker set
            self.update_frame(self.current_frame)
            n = len(self._visible_markers) if self._visible_markers else len(self.labels)
            self._show_feedback(f"M: Showing {n}/{len(self.labels)} markers")

        btn_frame = tk.Frame(inner)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="All", command=lambda: all_none(True)).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="None", command=lambda: all_none(False)).pack(
            side=tk.LEFT, padx=4
        )
        tk.Button(btn_frame, text="OK", command=apply_and_close).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Cancel", command=win.destroy).pack(side=tk.LEFT, padx=4)

    # ══════════════════════════════════════════════════════════════════════
    #  Trail
    # ══════════════════════════════════════════════════════════════════════
    def _key_toggle_trail(self, _key=None):
        self.trail_frames = 0 if self.trail_frames else 50
        self._last_trail_frame = -1
        if self.trail_frames == 0 and self.trail_actor is not None:
            self.plotter.remove_actor(self.trail_actor)
            self.trail_actor = None
        else:
            self._update_trail(self.current_frame)
        self._show_feedback(
            f"T: Trail {'ON (' + str(self.trail_frames) + ')' if self.trail_frames else 'OFF'}"
        )

    def _key_trail_shorter(self, _key=None):
        lengths = [0, 25, 50, 100, 200]
        i = next((j for j, v in enumerate(lengths) if v >= self.trail_frames), len(lengths) - 1)
        self.trail_frames = lengths[max(0, i - 1)]
        self._last_trail_frame = -1
        self.update_frame(self.current_frame)
        self._show_feedback(
            f"{{: Trail {self.trail_frames} frames" if self.trail_frames else "{: Trail OFF"
        )

    def _key_trail_longer(self, _key=None):
        lengths = [0, 25, 50, 100, 200]
        i = next((j for j, v in enumerate(lengths) if v >= self.trail_frames), 0)
        self.trail_frames = lengths[min(len(lengths) - 1, i + 1)]
        self._last_trail_frame = -1
        self.update_frame(self.current_frame)
        self._show_feedback(f"}}: Trail {self.trail_frames} frames")

    def _update_trail(self, frame_idx):
        if self.trail_actor is not None:
            self.plotter.remove_actor(self.trail_actor)
            self.trail_actor = None
        if self.trail_frames <= 0:
            return
        start_f = max(0, frame_idx - self.trail_frames)
        if start_f >= frame_idx:
            return
        pts, cells, idx = [], [], 0
        for m in range(self.n_markers):
            for f in range(start_f, frame_idx):
                p0 = self.points_data[f, m]
                p1 = self.points_data[f + 1, m]
                if np.all(p0 != -999.0) and np.all(p1 != -999.0):
                    pts.append(p0)
                    pts.append(p1)
                    cells.extend([2, idx, idx + 1])
                    idx += 2
        if not pts:
            return
        mesh = pv.PolyData(np.array(pts))
        mesh.lines = np.array(cells, dtype=np.int64)
        self.trail_actor = self.plotter.add_mesh(mesh, color="cyan", line_width=2)

    # ══════════════════════════════════════════════════════════════════════
    #  Speed / Size
    # ══════════════════════════════════════════════════════════════════════
    _SPEEDS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    def _key_speed_down(self, _key=None):
        i = next(
            (j for j, s in enumerate(self._SPEEDS) if s >= self.play_speed), len(self._SPEEDS) - 1
        )
        self.play_speed = self._SPEEDS[max(0, i - 1)]
        self._show_feedback(f"[: Speed {self.play_speed}x")

    def _key_speed_up(self, _key=None):
        i = next((j for j, s in enumerate(self._SPEEDS) if s >= self.play_speed), 0)
        self.play_speed = self._SPEEDS[min(len(self._SPEEDS) - 1, i + 1)]
        self._show_feedback(f"]: Speed {self.play_speed}x")

    def _key_marker_size_up(self, _key=None):
        self._point_size = min(50, self._point_size + 2)
        with contextlib.suppress(Exception):
            self.point_cloud_actor.GetProperty().SetPointSize(self._point_size)
        self._show_feedback(f"+: Size {self._point_size}")

    def _key_marker_size_down(self, _key=None):
        self._point_size = max(2, self._point_size - 2)
        with contextlib.suppress(Exception):
            self.point_cloud_actor.GetProperty().SetPointSize(self._point_size)
        self._show_feedback(f"−: Size {self._point_size}")

    # ══════════════════════════════════════════════════════════════════════
    #  Skeleton connections  (Phase 2)
    # ══════════════════════════════════════════════════════════════════════
    def _key_load_skeleton(self, _key=None):
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Load skeleton connections JSON",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
        )
        root.destroy()
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self._skeleton_pairs = [(p[0], p[1]) for p in data if len(p) >= 2]
            elif isinstance(data, dict) and "connections" in data:
                self._skeleton_pairs = [(p[0], p[1]) for p in data["connections"] if len(p) >= 2]
            self._show_feedback(f"J: Loaded {len(self._skeleton_pairs)} connections")
            self.update_frame(self.current_frame)
        except Exception as exc:
            self._show_feedback(f"J: Error loading skeleton: {exc}")

    def _update_skeleton(self, frame_idx):
        if self._skeleton_actor is not None:
            self.plotter.remove_actor(self._skeleton_actor)
            self._skeleton_actor = None
        if not self._skeleton_pairs:
            return
        label_to_idx = {lab: i for i, lab in enumerate(self.labels)}
        pts, cells, idx = [], [], 0
        fp = self.points_data[frame_idx]
        for a, b in self._skeleton_pairs:
            ia, ib = label_to_idx.get(a), label_to_idx.get(b)
            if ia is None or ib is None:
                continue
            pa, pb = fp[ia], fp[ib]
            if np.all(pa != -999.0) and np.all(pb != -999.0):
                pts.append(pa)
                pts.append(pb)
                cells.extend([2, idx, idx + 1])
                idx += 2
        if not pts:
            return
        mesh = pv.PolyData(np.array(pts))
        mesh.lines = np.array(cells, dtype=np.int64)
        self._skeleton_actor = self.plotter.add_mesh(mesh, color="orange", line_width=3)

    # ══════════════════════════════════════════════════════════════════════
    #  Info & Stats  (Phase 3)
    # ══════════════════════════════════════════════════════════════════════
    def _key_info(self, _key=None):
        t = self.current_frame / self.frame_rate if self.frame_rate else 0
        fname = os.path.basename(self.c3d_path) if self.c3d_path else "array"
        lines = [
            f"File: {fname}",
            f"Frame: {self.current_frame}/{self.n_frames - 1}  Time: {t:.3f} s",
            f"Markers: {len(self._valid_indices)} visible / {self.n_markers} total",
            f"Frame rate: {self.frame_rate} Hz   Units: {self.units}",
            f"Speed: {self.play_speed}x   Trail: {self.trail_frames}",
        ]
        if self.c3d_events:
            lines.append(f"Events: {len(self.c3d_events)}")
        if self.analog_info.get("n_channels"):
            lines.append(
                f"Analog: {self.analog_info['n_channels']} ch @ {self.analog_info['rate']} Hz"
            )
        info = "\n".join(lines)
        print(f"\n{'=' * 50}\n{info}\n{'=' * 50}")
        self._show_feedback("I: Info (see console)")

    def _key_stats(self, _key=None):
        """Print quality stats: gaps, range, events."""
        data = self.points_data  # (F, M, 3)
        gaps_per_marker = []
        for m in range(self.n_markers):
            missing = np.sum(np.all(data[:, m, :] == -999.0, axis=1))
            gaps_per_marker.append(missing)
        valid_vals = data[data != -999.0]
        lines = [
            "=== Quality Stats ===",
            f"Frames: {self.n_frames}   Markers: {self.n_markers}",
            f"Coord range: [{valid_vals.min():.4f}, {valid_vals.max():.4f}] m"
            if valid_vals.size
            else "No valid data",
            f"Mean abs: {np.mean(np.abs(valid_vals)):.4f} m" if valid_vals.size else "",
        ]
        total_gaps = sum(gaps_per_marker)
        lines.append(f"Total gap frames: {total_gaps} / {self.n_frames * self.n_markers}")
        worst = np.argmax(gaps_per_marker)
        lines.append(f"Worst marker: {self.labels[worst]} ({gaps_per_marker[worst]} gaps)")
        if self.c3d_events:
            lines.append(f"\nEvents ({len(self.c3d_events)}):")
            for ev in self.c3d_events[:20]:
                lines.append(f"  Frame {ev['frame']:>5d}  {ev['label']:>15s}  {ev['context']}")
        print("\n".join(lines))
        self._show_feedback("A: Stats (see console)")

    # ══════════════════════════════════════════════════════════════════════
    #  Export  (Phase 4)
    # ══════════════════════════════════════════════════════════════════════
    def _key_screenshot(self, _key=None):
        out = Path(self.c3d_path).with_suffix(".png") if self.c3d_path else Path("screenshot.png")
        self.plotter.screenshot(str(out))
        self._show_feedback(f"K: Screenshot → {out.name}")
        print(f"Screenshot saved: {out}")

    def _key_png_sequence(self, _key=None):
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="Select folder for PNG sequence")
        root.destroy()
        if not folder:
            return
        self._show_feedback("Z: Exporting PNGs…")
        was_playing = self.playing
        self.playing = False
        for f in range(self.n_frames):
            self.update_frame(f)
            self.slider_widget.GetRepresentation().SetValue(f)
            self.plotter.render()
            self.plotter.screenshot(os.path.join(folder, f"frame_{f:06d}.png"))
        self.playing = was_playing
        self._show_feedback(f"Z: {self.n_frames} PNGs saved")
        print(f"PNG sequence saved to {folder}")

    def _key_mp4_export(self, _key=None):
        root = tk.Tk()
        root.withdraw()
        path = filedialog.asksaveasfilename(
            title="Save MP4",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4")],
        )
        root.destroy()
        if not path:
            return
        self._show_feedback("V: Recording MP4…")
        was_playing = self.playing
        self.playing = False
        try:
            self.plotter.open_movie(path, framerate=int(self.frame_rate))
            for f in range(self.n_frames):
                self.update_frame(f)
                self.slider_widget.GetRepresentation().SetValue(f)
                self.plotter.render()
                self.plotter.write_frame()
            self.plotter.mwriter.close()
            self._show_feedback(f"V: MP4 saved → {os.path.basename(path)}")
            print(f"MP4 saved: {path}")
        except Exception as exc:
            self._show_feedback(f"V: MP4 error: {exc}")
            print(f"MP4 export error: {exc}")
        self.playing = was_playing

    # ══════════════════════════════════════════════════════════════════════
    #  Distance measurement  (Phase 5)
    # ══════════════════════════════════════════════════════════════════════
    def _key_distance(self, _key=None):
        self._distance_mode = True
        self._distance_picks.clear()
        if self._distance_actor:
            self.plotter.remove_actor(self._distance_actor)
            self._distance_actor = None
        self._show_feedback("D: Click two markers to measure distance")

    def _draw_distance(self):
        n1, p1 = self._distance_picks[0]
        n2, p2 = self._distance_picks[1]
        if np.all(p1 == -999.0) or np.all(p2 == -999.0):
            self._show_feedback("D: One marker is occluded")
            return
        dist = float(np.linalg.norm(p2 - p1))
        print(f"Distance {n1} → {n2}: {dist:.4f} m ({dist * 1000:.1f} mm)")
        mesh = pv.Line(p1, p2)
        if self._distance_actor:
            self.plotter.remove_actor(self._distance_actor)
        self._distance_actor = self.plotter.add_mesh(mesh, color="magenta", line_width=3)
        self._show_feedback(f"D: {n1}→{n2} = {dist:.4f} m")

    # ══════════════════════════════════════════════════════════════════════
    #  GUI Initialization
    # ══════════════════════════════════════════════════════════════════════
    def init_gui(self):
        pv.set_plot_theme("dark")
        self.plotter = pv.Plotter(
            window_size=(1280, 800),
            title=self._plotter_title,
        )

        # ── Adaptive ground plane ──
        (
            self.points_data[self.points_data != -999.0].reshape(-1, 3)
            if np.any(self.points_data != -999.0)
            else np.zeros((1, 3))
        )
        # compute bounding box from valid data
        all_valid = self.points_data.copy().reshape(-1, 3)
        all_valid = all_valid[~np.all(all_valid == -999.0, axis=1)]
        if len(all_valid) > 0:
            dmin = all_valid.min(axis=0)
            dmax = all_valid.max(axis=0)
            span = max(dmax[0] - dmin[0], dmax[1] - dmin[1], 2.0)
            center_x = (dmin[0] + dmax[0]) / 2
            center_y = (dmin[1] + dmax[1]) / 2
            floor_size = span * 1.5
        else:
            center_x, center_y, floor_size = 0, 0, 10

        # Initial point cloud
        fp0 = self.points_data[0]
        valid_mask = ~np.all(fp0 == -999.0, axis=1)
        self._valid_indices = np.where(valid_mask)[0]
        valid_pts0 = fp0[valid_mask] if valid_mask.any() else np.array([[0.0, 0.0, 0.0]])
        self.point_cloud = pv.PolyData(valid_pts0)
        self.point_cloud_actor = self.plotter.add_mesh(
            self.point_cloud,
            color="#00ff00",
            point_size=self._point_size,
            render_points_as_spheres=True,
            pickable=True,
        )

        # Floor / grid
        self.plotter.show_grid(color="gray")
        floor = pv.Plane(
            center=(center_x, center_y, 0),
            direction=(0, 0, 1),
            i_size=floor_size,
            j_size=floor_size,
        )
        self._floor_actor = self.plotter.add_mesh(
            floor, color="gray", opacity=0.15, show_edges=True
        )

        # Axes widget
        self._axes_widget = self.plotter.add_axes(interactive=False)

        # Status text
        self.text_actor = self.plotter.add_text(
            f"Frame 0/{self.n_frames - 1}",
            position="lower_right",
            font_size=10,
        )

        # Shortcuts reminder
        shortcuts = (
            "H Help | Space Play | ←→ ±1 | ↑↓ ±10 | PgUp/Dn ±100 | S Start | End End\n"
            "1-4 Views | R Reset | B Bg | G Grid | X Labels | C Colors | T Trail\n"
            "[ ] Speed | +− Size | M Markers | J Skeleton | K Shot | D Dist | I Info | A Stats"
        )
        self.plotter.add_text(shortcuts, position="lower_left", font_size=8, color="gray")

        # Timeline slider
        self.slider_widget = self.plotter.add_slider_widget(
            self.update_frame,
            [0, self.n_frames - 1],
            value=0,
            title="Timeline",
            pointa=(0.35, 0.92),
            pointb=(0.9, 0.92),
            style="modern",
            fmt="%.0f",
        )

        # Play checkbox
        self._play_checkbox = self.plotter.add_checkbox_button_widget(
            self.toggle_play,
            value=False,
            position=(10, 730),
            size=30,
            border_size=1,
            color_on="green",
            color_off="grey",
        )
        self.plotter.add_text("Play (Space)", position=(50, 740), font_size=9)

        # Camera
        self.plotter.view_isometric()
        self.plotter.camera.up = (0, 0, 1)
        self.plotter.reset_camera()
        self._apply_background()

        # Picking: pre-set 'pickpoint' so PyVista's internal left_button_down can assign to it
        pv.set_new_attribute(self.plotter, "pickpoint", np.zeros((1, 3)))
        self.plotter.enable_point_picking(
            callback=self.on_pick,
            show_message=False,
            use_picker=True,
            show_point=True,
            color="red",
        )

        # ── Register ALL key events ──
        ke = self.plotter.add_key_event
        # Help
        ke("h", self.show_help)
        ke("H", self.show_help)
        # Play (on Windows only the observer handles Space to avoid double-fire when both fire)
        if sys.platform != "win32":
            ke(" ", self._key_toggle_play)
            ke("space", self._key_toggle_play)
        # Navigation
        ke("Left", self._key_prev_frame)
        ke("Right", self._key_next_frame)
        ke("Up", self._key_back10)
        ke("Down", self._key_fwd10)
        ke("Prior", self._key_back100)
        ke("Next", self._key_fwd100)  # PageUp/PageDown
        ke("s", self._key_start)
        ke("S", self._key_start)
        ke("End", self._key_end)
        # View presets
        ke("1", self._key_view_front)
        ke("2", self._key_view_right)
        ke("3", self._key_view_top)
        ke("4", self._key_view_iso)
        ke("r", self._key_reset_camera)
        ke("R", self._key_reset_camera)
        # Display
        ke("Escape", self._key_escape)
        ke("x", self._key_toggle_labels)
        ke("X", self._key_toggle_labels)
        ke("b", self._key_toggle_bg)
        ke("B", self._key_toggle_bg)
        ke("g", self._key_toggle_grid)
        ke("G", self._key_toggle_grid)
        ke("c", self._key_toggle_colors)
        ke("C", self._key_toggle_colors)
        # Trail
        ke("t", self._key_toggle_trail)
        ke("T", self._key_toggle_trail)
        ke("braceleft", self._key_trail_shorter)
        ke("braceright", self._key_trail_longer)
        # Speed / Size ([ ] may be reported as bracketleft/bracketright on some systems)
        ke("[", self._key_speed_down)
        ke("]", self._key_speed_up)
        ke("bracketleft", self._key_speed_down)
        ke("bracketright", self._key_speed_up)
        ke("plus", self._key_marker_size_up)
        ke("equal", self._key_marker_size_up)
        ke("minus", self._key_marker_size_down)
        # Skeleton
        ke("j", self._key_load_skeleton)
        ke("J", self._key_load_skeleton)
        # Info
        ke("i", self._key_info)
        ke("I", self._key_info)
        ke("a", self._key_stats)
        ke("A", self._key_stats)
        # Export
        ke("k", self._key_screenshot)
        ke("K", self._key_screenshot)
        ke("z", self._key_png_sequence)
        ke("Z", self._key_png_sequence)
        ke("v", self._key_mp4_export)
        ke("V", self._key_mp4_export)
        # Distance
        ke("d", self._key_distance)
        ke("D", self._key_distance)
        # Marker visibility
        ke("m", self._key_marker_visibility)
        ke("M", self._key_marker_visibility)

        # Windows: use low-level key observer and timer (add_timer_event often does not fire)
        if sys.platform == "win32":
            iren = getattr(self.plotter, "iren", None)
            if (
                iren is None
                and hasattr(self.plotter, "render_window")
                and self.plotter.render_window is not None
            ):
                with contextlib.suppress(Exception):
                    iren = self.plotter.render_window.GetInteractor()
            if iren is not None:
                iren.add_observer("KeyPressEvent", self._win_key_observer)
                duration_ms = max(1, int(1000 / self.frame_rate))
                iren.create_timer(duration_ms, repeating=True)
                iren.add_observer("TimerEvent", self._win_timer_observer)

        print("\n--- Controls (press H for full help) ---")
        print("Space Play | ←→ ±1 | ↑↓ ±10 | PgUp/Dn ±100 | S Start | End End")
        print("1-4 Views | R Reset | B Bg | G Grid | X Labels | C Colors | T Trail")
        print(
            "{ } Trail len | [ ] Speed | +− Size | M Markers | J Skeleton | K Shot | Z PNGs | V MP4"
        )
        print("I Info | A Stats | D Distance | Escape Clear | H Help\n")

        # Timer (Linux/macOS; on Windows we use TimerEvent observer above)
        if sys.platform != "win32":
            self.plotter.add_timer_event(
                max_steps=100_000,
                duration=max(1, int(1000 / self.frame_rate)),
                callback=self.animation_callback,
            )

        self.plotter.show()


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="vailá PyVista 3D viewer for C3D/CSV marker data. Press H in the viewer for full HTML help.",
        prog="viewc3d_pyvista",
    )
    parser.add_argument(
        "c3d_path",
        nargs="?",
        default=None,
        help="Path to a C3D file (optional; file dialog opens if omitted)",
    )
    args = parser.parse_args()
    viewer = MokkaLikeViewer(args.c3d_path)
