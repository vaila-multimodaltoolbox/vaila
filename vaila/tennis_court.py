"""
Project: vailá Multimodal Toolbox
Script: tennis_court.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 19 March 2026
Version: 0.0.1

Description:
    This script draws an ITF-standard tennis court and allows overlaying
    player position trajectories (from 2D/3D reconstruction CSV files)
    and generating KDE heatmaps of player movement patterns.
    Mirrors the architecture of soccerfield.py.

    Court dimensions (ITF Standard):
      - Total length: 23.77 m
      - Doubles width: 10.97 m
      - Singles width: 8.23 m  (1.37 m alley on each side)
      - Service box length: 6.40 m  (net to service line)
      - Net height at center: 0.914 m (posts: 1.07 m)

Usage:
    GUI:
        python tennis_court.py
    CLI:
        python tennis_court.py --court <path_to_csv>
        python tennis_court.py --markers <path_to_markers_csv>
        python tennis_court.py --markers <path_to_markers_csv> --heatmap
        python tennis_court.py --color clay

Requirements:
    - Python 3.12
    - pandas, matplotlib, numpy, tkinter, rich

License:
    GNU Affero General Public License v3.0
"""

import os
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import Frame, filedialog, messagebox

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from rich import print

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Court geometry helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def draw_line(ax, p1, p2, **kw):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **kw)


def draw_rectangle(ax, bottom_left, width, height, **kw):
    ax.add_patch(patches.Rectangle(bottom_left, width, height, **kw))


# Court surface colours
COURT_COLORS = {
    "hard_blue": {"surface": "#2E6DB4", "lines": "white", "out": "#1A4872"},
    "hard_green": {"surface": "#3A7D44", "lines": "white", "out": "#2C5F35"},
    "clay": {"surface": "#C2603A", "lines": "white", "out": "#9E4D2E"},
    "grass": {"surface": "#4B8B3B", "lines": "white", "out": "#3A6B2D"},
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Court drawing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def plot_court(df, show_reference_points=True, show_axis_values=False, court_color="hard_blue"):
    """
    Plots an ITF tennis court using reference coordinates from *df*.

    The CSV points are used only to position reference labels; the actual
    court geometry is drawn from hard-coded ITF measurements so the court
    is always metric-correct regardless of the CSV content.

    Returns
    -------
    fig, ax
    """
    colors = COURT_COLORS.get(court_color, COURT_COLORS["hard_blue"])
    line_color = colors["lines"]

    # Court ITF dimensions (meters)
    L = 23.77  # total length (x-axis)
    W = 10.97  # doubles width (y-axis)
    sa = 1.37  # singles alley width each side
    sl = 6.40  # service-box length (net → service line)  = L/2 - 0.5*L + sl ≈ 5.485+0.915? Let us use ITF directly
    # Net is at x = L/2 = 11.885
    # Service line is at x = 11.885 - 6.40 = 5.485 and x = 11.885 + 6.40 = 18.285
    net_x = L / 2
    svc_left_x = net_x - sl  # 5.485
    svc_right_x = net_x + sl  # 18.285
    singles_y_low = sa  # 1.37
    singles_y_high = W - sa  # 9.60
    center_y = W / 2  # 5.485

    margin = 3.66  # standard runback/runoff
    fig_w, fig_h = 12, 8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-margin, L + margin)
    ax.set_ylim(-margin, W + margin)
    ax.set_aspect("equal")

    if show_axis_values:
        ax.grid(True, alpha=0.3, color="gray", linestyle="-", linewidth=0.5)
        ax.set_xlabel("X (meters)", fontsize=10)
        ax.set_ylabel("Y (meters)", fontsize=10)
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.set_xticks(np.arange(-margin, L + margin + 1, 2))
        ax.set_yticks(np.arange(-margin, W + margin + 1, 2))
    else:
        ax.axis("off")

    # ── Background (out-of-bounds area) ──────────────────────────────────
    draw_rectangle(
        ax,
        (-margin, -margin),
        L + 2 * margin,
        W + 2 * margin,
        edgecolor="none",
        facecolor=colors["out"],
        zorder=0,
    )

    # ── Court playing surface ─────────────────────────────────────────────
    draw_rectangle(ax, (0, 0), L, W, edgecolor="none", facecolor=colors["surface"], zorder=0.5)

    lw = 2  # line width
    line_kw = {"color": line_color, "linewidth": lw, "zorder": 1}

    # ── Outer boundary (doubles) ──────────────────────────────────────────
    draw_line(ax, (0, 0), (L, 0), **line_kw)  # bottom baseline
    draw_line(ax, (0, W), (L, W), **line_kw)  # top baseline
    draw_line(ax, (0, 0), (0, W), **line_kw)  # left side
    draw_line(ax, (L, 0), (L, W), **line_kw)  # right side

    # ── Singles sidelines ─────────────────────────────────────────────────
    draw_line(ax, (0, singles_y_low), (L, singles_y_low), **line_kw)
    draw_line(ax, (0, singles_y_high), (L, singles_y_high), **line_kw)

    # ── Service lines (left court → net, right court → net) ──────────────
    draw_line(ax, (svc_left_x, singles_y_low), (svc_left_x, singles_y_high), **line_kw)
    draw_line(ax, (svc_right_x, singles_y_low), (svc_right_x, singles_y_high), **line_kw)

    # ── Center service line ───────────────────────────────────────────────
    draw_line(ax, (svc_left_x, center_y), (net_x, center_y), **line_kw)
    draw_line(ax, (net_x, center_y), (svc_right_x, center_y), **line_kw)

    # ── Center marks on baselines (0.10 m long, inset from baseline) ──────
    mark = 0.20  # half-length of center tick
    draw_line(ax, (0, center_y - mark), (0, center_y + mark), **dict(line_kw, linewidth=3))
    draw_line(ax, (L, center_y - mark), (L, center_y + mark), **dict(line_kw, linewidth=3))

    # ── Net ───────────────────────────────────────────────────────────────
    net_kw = {"color": line_color, "linewidth": 4, "zorder": 2}
    draw_line(ax, (net_x, 0), (net_x, W), **net_kw)

    # Net strap (center thicker mark)
    strap_kw = {"color": line_color, "linewidth": 6, "zorder": 2}
    draw_line(ax, (net_x, center_y - 0.15), (net_x, center_y + 0.15), **strap_kw)

    # ── Reference point labels ────────────────────────────────────────────
    if show_reference_points and df is not None:
        points = {
            row["point_name"]: (row["x"], row["y"], row["point_number"]) for _, row in df.iterrows()
        }
        for _name, (x, y, num) in points.items():
            ax.text(
                x + 0.15,
                y + 0.15,
                str(num),
                color="black",
                fontsize=7,
                weight="bold",
                bbox={"facecolor": "white", "alpha": 0.7, "boxstyle": "round", "pad": 0.2},
                zorder=10,
            )

    # ── Court label ───────────────────────────────────────────────────────
    style_label = court_color.replace("_", " ").title()
    ax.set_title(
        f"Tennis Court — ITF Standard  ({L} m × {W} m)  [{style_label}]", fontsize=11, pad=8
    )

    return fig, ax


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Marker loading / plotting (same as soccerfield)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def load_and_plot_markers(
    field_ax,
    csv_path,
    canvas,
    manual_marker_artists_ref,
    frame_markers_ref,
    current_frame_ref,
    selected_markers=None,
):
    """
    Load vaila-format CSV (frame, p1_x, p1_y, p2_x, p2_y, …) and plot
    trajectories on top of the tennis court.
    """
    # Clear previous CSV/manual markers
    if field_ax:
        to_remove = [
            a for a in field_ax.get_children() if hasattr(a, "get_zorder") and a.get_zorder() >= 50
        ]
        for a in to_remove:
            a.remove()

    manual_marker_artists_ref.clear()
    frame_markers_ref.clear()
    current_frame_ref[0] = 0

    markers_df = pd.read_csv(csv_path)
    print(f"Loaded: {csv_path}  ({len(markers_df)} frames)")
    markers_df = markers_df.replace("", np.nan)

    cols = markers_df.columns
    marker_names = sorted(
        {
            col.rsplit("_", 1)[0]
            for col in cols
            if col != "frame" and (col.endswith("_x") or col.endswith("_y"))
        }
    )

    field_ax._all_marker_names = marker_names
    colors = plt.cm.rainbow(np.linspace(0, 1, max(len(marker_names), 1)))

    # Valid range for a tennis court with margin
    X_MIN, X_MAX = -5, 29
    Y_MIN, Y_MAX = -5, 16

    for idx, marker in enumerate(marker_names):
        if selected_markers is not None and marker not in selected_markers:
            continue
        x_col = f"{marker}_x"
        y_col = f"{marker}_y"
        if x_col not in cols or y_col not in cols:
            continue

        valid = markers_df[[x_col, y_col]].dropna()
        mask = (valid[x_col].between(X_MIN, X_MAX)) & (valid[y_col].between(Y_MIN, Y_MAX))
        vx = valid.loc[mask, x_col].values
        vy = valid.loc[mask, y_col].values

        if len(vx) == 0:
            continue

        c = colors[idx]
        field_ax.plot(vx, vy, "-", color=c, linewidth=1.5, alpha=0.7, zorder=50)
        field_ax.scatter(
            vx,
            vy,
            color=c,
            s=50,
            marker="o",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85,
            zorder=51,
        )
        label = marker.replace("M", "p") if marker.startswith("M") else marker
        field_ax.text(
            vx[-1] + 0.2,
            vy[-1] + 0.2,
            label,
            fontsize=7,
            color="black",
            weight="bold",
            bbox={
                "facecolor": c,
                "alpha": 0.7,
                "edgecolor": "black",
                "boxstyle": "round",
                "pad": 0.1,
            },
            zorder=52,
        )
        print(f"  Plotted {marker}: {len(vx)} points")

    canvas.draw()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main GUI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_tenniscourt():
    """Launch the Tennis Court Visualization GUI."""
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    root = tk.Tk()
    root.title("Tennis Court Visualization — vailá")
    root.geometry("1300x850")

    # ── State variables ───────────────────────────────────────────────────
    current_ax = [None]
    current_canvas = [None]
    show_ref_points = [True]
    show_axis_values = [False]
    current_court_csv = [None]
    current_markers_csv = [None]
    selected_markers = [None]
    current_court_color = ["hard_blue"]

    manual_marker_mode = [False]
    current_marker_number = [1]
    current_frame = [0]
    frame_markers = {}
    manual_marker_artists = []

    # ── Helpers ───────────────────────────────────────────────────────────

    def load_court(custom_csv=None, color=None):
        try:
            if custom_csv:
                csv_path = custom_csv
            else:
                models_dir = os.path.join(os.path.dirname(__file__), "models")
                csv_path = os.path.join(models_dir, "tenniscourt_ref3d.csv")

            current_court_csv[0] = csv_path
            df = pd.read_csv(csv_path)
            use_color = color or current_court_color[0]
            current_court_color[0] = use_color

            fig, ax = plot_court(
                df,
                show_reference_points=show_ref_points[0],
                show_axis_values=show_axis_values[0],
                court_color=use_color,
            )
            current_ax[0] = ax

            for w in plot_frame.winfo_children():
                w.destroy()

            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            current_canvas[0] = canvas

            toolbar = NavigationToolbar2Tk(canvas, plot_frame)
            toolbar.update()

            frame_markers.clear()
            manual_marker_artists.clear()
            setup_manual_marker_events(canvas)

            if current_markers_csv[0]:
                load_and_plot_markers(
                    current_ax[0],
                    current_markers_csv[0],
                    current_canvas[0],
                    manual_marker_artists,
                    frame_markers,
                    current_frame,
                    selected_markers[0],
                )
                select_markers_btn.config(state=tk.NORMAL)

            print("Court plotted successfully!")

        except Exception as e:
            print(f"Error plotting court: {e}")
            import traceback

            traceback.print_exc()

    def load_custom_court():
        p = filedialog.askopenfilename(
            title="Select CSV file with court coordinates",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if p:
            load_court(custom_csv=p)

    def toggle_ref_points():
        show_ref_points[0] = not show_ref_points[0]
        ref_btn.config(text="Hide Ref Points" if show_ref_points[0] else "Show Ref Points")
        if current_court_csv[0]:
            load_court(custom_csv=current_court_csv[0])
        if current_markers_csv[0]:
            load_and_plot_markers(
                current_ax[0],
                current_markers_csv[0],
                current_canvas[0],
                manual_marker_artists,
                frame_markers,
                current_frame,
                selected_markers[0],
            )

    def toggle_axis():
        show_axis_values[0] = not show_axis_values[0]
        axis_btn.config(text="Hide Axis" if show_axis_values[0] else "Show Axis")
        if current_court_csv[0]:
            load_court(custom_csv=current_court_csv[0])
        if current_markers_csv[0]:
            load_and_plot_markers(
                current_ax[0],
                current_markers_csv[0],
                current_canvas[0],
                manual_marker_artists,
                frame_markers,
                current_frame,
                selected_markers[0],
            )

    def change_court_color():
        colors_order = list(COURT_COLORS.keys())
        idx = colors_order.index(current_court_color[0])
        next_color = colors_order[(idx + 1) % len(colors_order)]
        color_btn.config(text=f"Surface: {next_color.replace('_', ' ').title()}")
        if current_court_csv[0]:
            load_court(custom_csv=current_court_csv[0], color=next_color)
        if current_markers_csv[0]:
            load_and_plot_markers(
                current_ax[0],
                current_markers_csv[0],
                current_canvas[0],
                manual_marker_artists,
                frame_markers,
                current_frame,
                selected_markers[0],
            )

    def load_markers_csv_action():
        if current_ax[0] is None:
            print("Load the court first.")
            return
        p = filedialog.askopenfilename(
            title="Select vaila CSV with player coordinates",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not p:
            return
        try:
            current_markers_csv[0] = p
            selected_markers[0] = None
            load_and_plot_markers(
                current_ax[0],
                p,
                current_canvas[0],
                manual_marker_artists,
                frame_markers,
                current_frame,
                None,
            )
            select_markers_btn.config(state=tk.NORMAL)
        except Exception as e:
            print(f"Error loading markers: {e}")
            import traceback

            traceback.print_exc()

    def open_marker_selection_dialog():
        if not current_markers_csv[0] or not os.path.exists(current_markers_csv[0]):
            messagebox.showerror("Error", "No marker CSV loaded.")
            return

        df = pd.read_csv(current_markers_csv[0])
        names = sorted(
            {
                col.rsplit("_", 1)[0]
                for col in df.columns
                if col != "frame" and (col.endswith("_x") or col.endswith("_y"))
            }
        )
        if not names:
            messagebox.showerror("Error", "No markers found in CSV.")
            return

        win = tk.Toplevel(root)
        win.title("Select Markers")
        win.geometry("280x380")

        frm = Frame(win)
        frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        sb = tk.Scrollbar(frm)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        lb = tk.Listbox(frm, selectmode=tk.MULTIPLE, yscrollcommand=sb.set)
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=lb.yview)

        for n in names:
            lb.insert(tk.END, n)
        initial = selected_markers[0] if selected_markers[0] else names
        for i, n in enumerate(names):
            if n in initial:
                lb.selection_set(i)

        bf = Frame(win)
        bf.pack(fill=tk.X, padx=10, pady=8)

        def apply_sel():
            sel = [lb.get(i) for i in lb.curselection()]
            selected_markers[0] = sel if sel else None
            load_and_plot_markers(
                current_ax[0],
                current_markers_csv[0],
                current_canvas[0],
                manual_marker_artists,
                frame_markers,
                current_frame,
                selected_markers[0] if selected_markers[0] else [],
            )
            current_canvas[0].draw()
            win.destroy()

        tk.Button(
            bf,
            text="Select All",
            command=lambda: lb.select_set(0, tk.END),
            bg="#4CAF50",
            fg="white",
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            bf,
            text="Deselect All",
            command=lambda: lb.selection_clear(0, tk.END),
            bg="#f44336",
            fg="white",
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Apply", command=apply_sel, bg="#2196F3", fg="white").pack(
            side=tk.RIGHT, padx=4
        )
        win.transient(root)
        win.grab_set()

    # ── Manual markers ────────────────────────────────────────────────────

    def toggle_manual_marker_mode():
        manual_marker_mode[0] = not manual_marker_mode[0]
        if manual_marker_mode[0]:
            manual_btn.config(text="Disable Manual Markers")
            print(
                "Manual marker mode ON — left-click to add, right-click to delete, Ctrl+S to save."
            )
        else:
            manual_btn.config(text="Create Manual Markers")
            print("Manual marker mode OFF.")

    def create_marker(event):
        if not manual_marker_mode[0] or current_ax[0] is None:
            return
        try:
            ax = current_ax[0]
            x, y_bad = ax.transData.inverted().transform((event.x, event.y))
            y_min, y_max = ax.get_ylim()
            y = y_min + y_max - y_bad + 0.3
            if x < -5 or x > 29 or y < -5 or y > 16:
                return
            if event.state & 0x1:
                current_marker_number[0] += 1
            mn = current_marker_number[0]
            fi = current_frame[0]
            frame_markers.setdefault(fi, {})[mn] = (x, y)

            circ = patches.Circle(
                (x, y),
                0.25,
                color=plt.cm.tab10(mn % 10),
                edgecolor="black",
                linewidth=1,
                alpha=0.85,
                zorder=100,
            )
            current_ax[0].add_patch(circ)
            txt = current_ax[0].text(
                x + 0.3,
                y + 0.3,
                f"p{mn}",
                fontsize=7,
                color="black",
                weight="bold",
                bbox={
                    "facecolor": plt.cm.tab10(mn % 10),
                    "alpha": 0.7,
                    "edgecolor": "black",
                    "boxstyle": "round",
                    "pad": 0.1,
                },
                zorder=101,
            )
            manual_marker_artists.append((circ, txt, x, y, mn, fi))
            current_canvas[0].draw()
            current_frame[0] += 1
        except Exception as e:
            print(f"Error creating marker: {e}")

    def delete_marker(event):
        if not manual_marker_mode[0] or current_ax[0] is None:
            return
        try:
            ax = current_ax[0]
            x, y_bad = ax.transData.inverted().transform((event.x, event.y))
            y_min, y_max = ax.get_ylim()
            y = y_min + y_max - y_bad
            best, best_dist = -1, float("inf")
            for i, (_, _, mx, my, _, _) in enumerate(manual_marker_artists):
                d = np.hypot(x - mx, y - my)
                if d < best_dist and d < 2:
                    best_dist, best = d, i
            if best >= 0:
                circ, txt, _, _, mn, fi = manual_marker_artists[best]
                circ.remove()
                txt.remove()
                del manual_marker_artists[best]
                if fi in frame_markers and mn in frame_markers[fi]:
                    del frame_markers[fi][mn]
                    if not frame_markers[fi]:
                        del frame_markers[fi]
                current_canvas[0].draw()
        except Exception as e:
            print(f"Error deleting marker: {e}")

    def save_markers_csv(event=None):
        if not frame_markers:
            print("No markers to save.")
            return
        p = filedialog.asksaveasfilename(
            title="Save Markers CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not p:
            return
        max_mn = max(mn for fd in frame_markers.values() for mn in fd)
        max_frame = max(frame_markers.keys())
        data = {"frame": list(range(max_frame + 1))}
        for mn in range(1, max_mn + 1):
            data[f"p{mn}_x"] = [None] * (max_frame + 1)
            data[f"p{mn}_y"] = [None] * (max_frame + 1)
        for fi, fd in frame_markers.items():
            for mn, (x, y) in fd.items():
                data[f"p{mn}_x"][fi] = x
                data[f"p{mn}_y"][fi] = y
        pd.DataFrame(data).to_csv(p, index=False)
        print(f"Markers saved to: {p}")
        messagebox.showinfo("Saved", f"Markers saved to:\n{p}")

    def clear_all_markers():
        if current_ax[0] is None:
            return
        to_rm = [
            a
            for a in current_ax[0].get_children()
            if hasattr(a, "get_zorder") and a.get_zorder() >= 50
        ]
        for a in to_rm:
            a.remove()
        manual_marker_artists.clear()
        frame_markers.clear()
        current_frame[0] = 0
        current_markers_csv[0] = None
        selected_markers[0] = None
        select_markers_btn.config(state=tk.DISABLED)
        if current_canvas[0]:
            current_canvas[0].draw()
        print("All markers cleared.")

    def setup_manual_marker_events(canvas):
        w = canvas.get_tk_widget()
        w.bind("<Button-1>", create_marker)
        w.bind("<Button-3>", delete_marker)
        root.bind("<Control-s>", save_markers_csv)

    def open_tennis_court_help():
        """Open bundled HTML help in the default browser (no extra Tk window)."""
        html_path = Path(__file__).resolve().parent / "help" / "tennis_court.html"
        if html_path.is_file():
            webbrowser.open_new_tab(html_path.as_uri())
        else:
            messagebox.showinfo(
                "Help",
                f"Help file not found:\n{html_path}\n\nSee vaila/help/tennis_court.md",
            )

    def show_heatmap():
        """Generate a KDE heatmap from loaded marker trajectories on the court."""
        if current_ax[0] is None or current_canvas[0] is None:
            messagebox.showwarning("Warning", "Load the court first.")
            return
        if not current_markers_csv[0] or not os.path.exists(current_markers_csv[0]):
            messagebox.showwarning("No data", "Load a Markers CSV first to generate a heatmap.")
            return

        win = tk.Toplevel(root)
        win.title("Heatmap — Tennis Court")
        win.geometry("850x620")
        win.resizable(True, True)

        ctrl = Frame(win)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ALL = "All"

        mdf = pd.read_csv(current_markers_csv[0]).replace("", np.nan)
        marker_names = sorted(
            {
                c.rsplit("_", 1)[0]
                for c in mdf.columns
                if c != "frame" and (c.endswith("_x") or c.endswith("_y"))
            }
        )

        tk.Label(ctrl, text="Marker:", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        filter_var = tk.StringVar(value=ALL)
        tk.OptionMenu(ctrl, filter_var, ALL, *marker_names).pack(side=tk.LEFT, padx=4)

        tk.Label(ctrl, text="Cmap:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(8, 0))
        cmap_var = tk.StringVar(value="Blues")
        tk.OptionMenu(
            ctrl,
            cmap_var,
            "Blues",
            "Reds",
            "Greens",
            "Oranges",
            "YlOrRd",
            "viridis",
            "plasma",
            "inferno",
        ).pack(side=tk.LEFT, padx=4)

        fig_hm, ax_hm = plt.subplots(figsize=(7.5, 5.0))
        canvas_hm = FigureCanvasTkAgg(fig_hm, master=win)
        canvas_hm.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def draw_heatmap():
            try:
                ax_hm.clear()
                _draw_court_on_ax(ax_hm, current_court_color[0])

                flt = filter_var.get()
                xs, ys = [], []
                for nm in marker_names:
                    if flt != ALL and nm != flt:
                        continue
                    xc, yc = f"{nm}_x", f"{nm}_y"
                    if xc in mdf.columns and yc in mdf.columns:
                        sub = mdf[[xc, yc]].dropna()
                        xs.extend(sub[xc].tolist())
                        ys.extend(sub[yc].tolist())

                if len(xs) < 2:
                    ax_hm.set_title("Not enough points for heatmap")
                    canvas_hm.draw()
                    return

                hm_df = pd.DataFrame({"x": xs, "y": ys})
                sns.kdeplot(
                    data=hm_df,
                    x="x",
                    y="y",
                    cmap=cmap_var.get(),
                    fill=True,
                    alpha=0.6,
                    bw_method="scott",
                    thresh=0.05,
                    ax=ax_hm,
                )

                ax_hm.set_xlim(-2, 25.77)
                ax_hm.set_ylim(-2, 12.97)
                ax_hm.set_aspect("equal")
                title = "Heatmap"
                if flt != ALL:
                    title += f" — {flt}"
                ax_hm.set_title(title)
                canvas_hm.draw()
            except Exception as exc:
                messagebox.showerror("Error", f"Heatmap failed: {exc}")

        tk.Button(
            ctrl, text="Show", command=draw_heatmap, bg="#4CAF50", fg="white", padx=8, pady=2
        ).pack(side=tk.LEFT, padx=8)
        draw_heatmap()

    def _draw_court_on_ax(ax, color_name="hard_blue"):
        """Minimal re-draw of the tennis court on a given axes (for heatmap overlay)."""
        cs = COURT_COLORS.get(color_name, COURT_COLORS["hard_blue"])
        L, W = 23.77, 10.97
        SW = 8.23
        SL = 6.40
        alley = (W - SW) / 2

        ax.set_facecolor(cs["out"])
        ax.add_patch(
            patches.Rectangle(
                (0, 0), L, W, facecolor=cs["surface"], edgecolor=cs["lines"], linewidth=2, zorder=0
            )
        )
        lc = cs["lines"]
        # Singles sidelines
        ax.plot([0, L], [alley, alley], color=lc, lw=1.5, zorder=1)
        ax.plot([0, L], [W - alley, W - alley], color=lc, lw=1.5, zorder=1)
        # Service lines
        mid_x = L / 2
        ax.plot([mid_x - SL, mid_x - SL], [alley, W - alley], color=lc, lw=1.5, zorder=1)
        ax.plot([mid_x + SL, mid_x + SL], [alley, W - alley], color=lc, lw=1.5, zorder=1)
        # Centre service line
        ax.plot([mid_x - SL, mid_x + SL], [W / 2, W / 2], color=lc, lw=1.5, zorder=1)
        # Net
        ax.plot([mid_x, mid_x], [-0.5, W + 0.5], color=lc, lw=2.5, zorder=2)
        # Centre marks
        ax.plot([0, 0.15], [W / 2, W / 2], color=lc, lw=1.5, zorder=1)
        ax.plot([L - 0.15, L], [W / 2, W / 2], color=lc, lw=1.5, zorder=1)

        ax.set_xlim(-2, L + 2)
        ax.set_ylim(-2, W + 2)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

    # ── Layout ────────────────────────────────────────────────────────────
    btn_bar = Frame(root)
    btn_bar.pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)

    plot_frame = Frame(root)
    plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    btn_kw = {"fg": "white", "font": ("Arial", 9, "bold"), "padx": 6, "pady": 4}

    tk.Button(btn_bar, text="Load Default Court", bg="#1565C0", command=load_court, **btn_kw).pack(
        side=tk.LEFT, padx=3
    )

    tk.Button(
        btn_bar, text="Load Custom Court", bg="#1976D2", command=load_custom_court, **btn_kw
    ).pack(side=tk.LEFT, padx=3)

    color_btn = tk.Button(
        btn_bar, text="Surface: Hard Blue", bg="#455A64", command=change_court_color, **btn_kw
    )
    color_btn.pack(side=tk.LEFT, padx=3)

    tk.Button(
        btn_bar, text="Load Markers CSV", bg="#2E7D32", command=load_markers_csv_action, **btn_kw
    ).pack(side=tk.LEFT, padx=3)

    select_markers_btn = tk.Button(
        btn_bar,
        text="Select Markers",
        bg="#388E3C",
        command=open_marker_selection_dialog,
        state=tk.DISABLED,
        **btn_kw,
    )
    select_markers_btn.pack(side=tk.LEFT, padx=3)

    tk.Button(btn_bar, text="Heatmap", bg="#D84315", command=show_heatmap, **btn_kw).pack(
        side=tk.LEFT, padx=3
    )

    ref_btn = tk.Button(
        btn_bar, text="Hide Ref Points", bg="#6A1B9A", command=toggle_ref_points, **btn_kw
    )
    ref_btn.pack(side=tk.LEFT, padx=3)

    axis_btn = tk.Button(btn_bar, text="Show Axis", bg="#4A148C", command=toggle_axis, **btn_kw)
    axis_btn.pack(side=tk.LEFT, padx=3)

    manual_btn = tk.Button(
        btn_bar,
        text="Create Manual Markers",
        bg="#E65100",
        command=toggle_manual_marker_mode,
        **btn_kw,
    )
    manual_btn.pack(side=tk.LEFT, padx=3)

    tk.Button(
        btn_bar, text="Clear All Markers", bg="#B71C1C", command=clear_all_markers, **btn_kw
    ).pack(side=tk.LEFT, padx=3)

    tk.Button(btn_bar, text="Help", bg="#37474F", command=open_tennis_court_help, **btn_kw).pack(
        side=tk.RIGHT, padx=3
    )

    # Auto-load the default court
    load_court()

    root.mainloop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="vailá — Tennis Court Visualization")
    parser.add_argument("--court", type=str, help="Path to court model CSV")
    parser.add_argument("--markers", type=str, help="Path to vaila-format markers CSV")
    parser.add_argument(
        "--color",
        type=str,
        default="hard_blue",
        choices=list(COURT_COLORS.keys()),
        help="Court surface color (default: hard_blue)",
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        help="Generate a heatmap from --markers data (requires --markers)",
    )
    args = parser.parse_args()

    if args.heatmap and args.markers:
        _fig, _ax = plt.subplots(figsize=(8, 5.5))

        cs = COURT_COLORS.get(args.color, COURT_COLORS["hard_blue"])
        L, W = 23.77, 10.97
        SW = 8.23
        SL = 6.40
        _alley = (W - SW) / 2
        _ax.set_facecolor(cs["out"])
        _ax.add_patch(
            patches.Rectangle(
                (0, 0), L, W, facecolor=cs["surface"], edgecolor=cs["lines"], linewidth=2, zorder=0
            )
        )
        _lc = cs["lines"]
        _ax.plot([0, L], [_alley, _alley], color=_lc, lw=1.5, zorder=1)
        _ax.plot([0, L], [W - _alley, W - _alley], color=_lc, lw=1.5, zorder=1)
        _mid = L / 2
        _ax.plot([_mid - SL, _mid - SL], [_alley, W - _alley], color=_lc, lw=1.5, zorder=1)
        _ax.plot([_mid + SL, _mid + SL], [_alley, W - _alley], color=_lc, lw=1.5, zorder=1)
        _ax.plot([_mid - SL, _mid + SL], [W / 2, W / 2], color=_lc, lw=1.5, zorder=1)
        _ax.plot([_mid, _mid], [-0.5, W + 0.5], color=_lc, lw=2.5, zorder=2)

        _mdf = pd.read_csv(args.markers).replace("", np.nan)
        _names = sorted(
            {
                c.rsplit("_", 1)[0]
                for c in _mdf.columns
                if c != "frame" and (c.endswith("_x") or c.endswith("_y"))
            }
        )
        _xs, _ys = [], []
        for _nm in _names:
            _xc, _yc = f"{_nm}_x", f"{_nm}_y"
            if _xc in _mdf.columns and _yc in _mdf.columns:
                _sub = _mdf[[_xc, _yc]].dropna()
                _xs.extend(_sub[_xc].tolist())
                _ys.extend(_sub[_yc].tolist())
        if len(_xs) >= 2:
            sns.kdeplot(
                data=pd.DataFrame({"x": _xs, "y": _ys}),
                x="x",
                y="y",
                cmap="Blues",
                fill=True,
                alpha=0.6,
                bw_method="scott",
                thresh=0.05,
                ax=_ax,
            )
        _ax.set_xlim(-2, L + 2)
        _ax.set_ylim(-2, W + 2)
        _ax.set_aspect("equal")
        _ax.set_title("Heatmap — Tennis Court")
        plt.tight_layout()
        plt.show()
    else:
        run_tenniscourt()
