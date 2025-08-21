"""
Project: vailá Multimodal Toolbox
Script: readcsv.py - Read CSV File

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 29 July 2024
Update Date: 20 July 2025
Version: 0.0.3

Description:
    This script provides tools for reading CSV files and displaying their contents.
    It includes functions for:
    - Detecting the delimiter used in the file.
    - Detecting if the file has a header.
    - Selecting markers to display.
    - Selecting headers to display.
    - Visualizing the data using Matplotlib or Open3D.

Usage:
    Run the script from the command line:
        python readcsv.py

Requirements:
    - Python 3.x
    - pandas
    - numpy
    - matplotlib
    - tkinter
    - rich

License:
    This project is licensed under the terms of GNU General Public License v3.0.

Change History:
    - v0.0.3: Added support for CSV, TXT and TSV files, improved UI
    - v0.0.2: Added support for CSV, TXT and TSV files, improved UI
    - v0.0.1: First version
"""

import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, Toplevel, Button, Label, Listbox, Frame, messagebox
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button as MplButton, TextBox
from matplotlib import animation
from rich import print
from typing import Union, Optional, List, cast
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backend_bases import TimerBase
# ==================== Performance & Cache Helpers (Matplotlib path) ====================
# Float32 + rounding + NPZ cache to avoid re-parsing CSV on subsequent runs.

import numpy as _np
import os as _os

def _npz_path_for(csv_path: str) -> str:
    base = _os.path.basename(csv_path)
    stem, _ = _os.path.splitext(base)
    return _os.path.join(_os.path.dirname(csv_path), f"{stem}.npz")

def _save_npz(npz_file: str, points_f32, markers, index_vec, meta: dict):
    _os.makedirs(_os.path.dirname(npz_file) or ".", exist_ok=True)
    _np.savez_compressed(
        npz_file,
        points=points_f32,
        markers=_np.array(markers),
        index=_np.array(index_vec),
        meta=_np.array(meta, dtype=object),
    )

def _load_npz(npz_file: str):
    data = _np.load(npz_file, allow_pickle=True)
    meta = {k: data[k].item() if k == "meta" else data[k] for k in data.files}
    return meta["points"], meta["markers"], meta["index"], meta["meta"]

def _quantize_points(points: _np.ndarray, precision: int = 3) -> _np.ndarray:
    if points.dtype != _np.float32:
        points = points.astype(_np.float32, copy=False)
    _np.around(points, decimals=int(precision), out=points)
    return points

def _basic_loader(csv_path: str):
    """
    Minimal fast loader: expects header with triplets *_x,*_y,*_z and first column as index/time.
    """
    import pandas as pd
    # Try pyarrow engine for speed when available
    try:
        df = pd.read_csv(csv_path, engine="pyarrow")
    except Exception:
        df = pd.read_csv(csv_path)
    if df.shape[1] < 4:
        raise RuntimeError("CSV must contain at least one marker with X,Y,Z columns plus an index/time column.")
    index_vec = df.iloc[:, 0].to_numpy()
    names = df.columns.tolist()
    coord_map = {}
    for c in names[1:]:
        base, sep, suf = c.rpartition('_')
        if sep and suf.lower() in ('x','y','z'):
            coord_map.setdefault(base, {})[suf.lower()] = c
    markers = [m for m,v in coord_map.items() if set(v.keys())=={'x','y','z'}]
    if not markers:
        raise RuntimeError("No complete triplets *_x,*_y,*_z were found.")
    # Read only the required columns in one go as float32
    arrs = [df[[coord_map[m]['x'],coord_map[m]['y'],coord_map[m]['z']]].to_numpy(dtype=_np.float32, copy=False) for m in markers]
    points = _np.stack(arrs, axis=1)  # (T, M, 3)
    # Auto mm->m heuristic
    try:
        mean_abs = _np.nanmean(_np.abs(points))
        if _np.isfinite(mean_abs) and mean_abs > 100.0:
            points *= _np.float32(0.001)
    except Exception:
        pass
    return index_vec, markers, points

def _load_with_cache(csv_path: str, precision: int = 3, use_cache: bool = True, force_recache: bool = False):
    """
    Returns (index_vec, markers, points_f32, from_cache, delimiter_or_None)
    - NPZ cache checked by mtime/size to avoid re-parsing CSV
    - Fast path uses _basic_loader; falls back to read_csv_generic if needed
    """
    csv_path = _os.path.abspath(csv_path)
    npz_file = _npz_path_for(csv_path)
    src_mtime = _os.path.getmtime(csv_path)
    src_size = _os.path.getsize(csv_path)

    if use_cache and _os.path.exists(npz_file) and not force_recache:
        try:
            points, markers, index_vec, meta = _load_npz(npz_file)
            if (meta.get("src_path") == csv_path and
                abs(meta.get("src_mtime", -1) - src_mtime) < 1e-6 and
                meta.get("src_size") == src_size):
                cached_precision = int(meta.get("precision", 3))
                if precision < cached_precision:
                    points = _quantize_points(points, precision)
                return index_vec, list(markers), points, True, meta.get("delimiter", None)
        except Exception as e:
            print(f"[readcsv] NPZ cache ignored (reason: {e})")

    # Try fast basic loader first
    delimiter = None
    try:
        index_vec, markers, points = _basic_loader(csv_path)
    except Exception:
        # Fall back to canonical generic reader (preserves behavior)
        index_vec, marker_data, valid_markers, delimiter = read_csv_generic(csv_path)
        markers = list(valid_markers.keys())
        points = _np.stack([marker_data[m] for m in markers], axis=1)

    points = _quantize_points(points, precision)

    if use_cache:
        meta = {
            "src_path": csv_path,
            "src_mtime": src_mtime,
            "src_size": src_size,
            "precision": int(precision),
            "dtype": "float32",
            "shape": points.shape,
            "delimiter": delimiter,
            "index_name": getattr(index_vec, "name", "Index"),
        }
        try:
            _save_npz(npz_file, points, markers, index_vec, meta)
            print(f"[readcsv] Saved NPZ cache at: {npz_file}")
        except Exception as e:
            print(f"[readcsv] Failed to save NPZ cache ({e})")

    return index_vec, markers, points, False, delimiter

def csv2npz_inplace(csv_path: str, precision: int = 3):
    """
    In-script CSV→NPZ converter (no separate script required).
    Uses the fast basic loader and falls back to read_csv_generic on failure.
    """
    csv_path = _os.path.abspath(csv_path)
    try:
        index_vec, markers, points = _basic_loader(csv_path)
        delimiter = None
    except Exception:
        index_vec, marker_data, valid_markers, delimiter = read_csv_generic(csv_path)
        markers = list(valid_markers.keys())
        points = _np.stack([marker_data[m] for m in markers], axis=1)
    points = _quantize_points(points, precision)

    src_mtime = _os.path.getmtime(csv_path)
    src_size = _os.path.getsize(csv_path)
    meta = {
        "src_path": csv_path,
        "src_mtime": src_mtime,
        "src_size": src_size,
        "precision": int(precision),
        "dtype": "float32",
        "shape": points.shape,
        "delimiter": delimiter,
        "index_name": getattr(index_vec, "name", "Index"),
    }
    out_npz = _npz_path_for(csv_path)
    _save_npz(out_npz, points, markers, index_vec, meta)
    print(f"[csv2npz] Generated: {out_npz} (float32, precision={precision})")
    return out_npz
# ==================== End Helpers ====================



###############################################################################
# Function: headersidx
# (Added to maintain compatibility with vaila/__init__.py)
###############################################################################
def headersidx(headers, prefix):
    """
    Given a list of headers and a prefix, returns the indices of headers that start with the prefix.

    Args:
        headers (list): List of header names.
        prefix (str): Prefix to check.

    Returns:
        List[int]: List of indices of headers that start with the prefix.
    """
    return [i for i, header in enumerate(headers) if header.startswith(prefix)]


###############################################################################
# Function: reshapedata
# (Added to maintain compatibility with vaila/__init__.py)
###############################################################################
def reshapedata(df, selected_markers):
    """
    Given a DataFrame `df` containing the time column and marker columns in the format:
      marker_x, marker_y, marker_z,
    and a list with the names of selected markers, returns a NumPy array of shape:
      (num_frames, num_markers, 3)

    If the mean absolute value is high (> 100), data is converted from millimeters to meters.

    Args:
        df (DataFrame): DataFrame containing the data, where the first column is Time.
        selected_markers (list): List of marker names.

    Returns:
        numpy.ndarray: Array with shape (num_frames, num_markers, 3) containing marker data.
    """
    num_frames = df.shape[0]
    num_markers = len(selected_markers)
    points = np.zeros((num_frames, num_markers, 3))
    for i, marker in enumerate(selected_markers):
        x_col = f"{marker}_x"
        y_col = f"{marker}_y"
        z_col = f"{marker}_z"
        if x_col in df.columns and y_col in df.columns and z_col in df.columns:
            points[:, i, 0] = df[x_col].values
            points[:, i, 1] = df[y_col].values
            points[:, i, 2] = df[z_col].values
        else:
            raise ValueError(
                f"Columns for marker '{marker}' not found in expected format."
            )
    if np.mean(np.abs(points)) > 100:
        points = points * 0.001  # Convert from millimeters to meters
    return points


###############################################################################
# Function: detect_delimiter
# (Added to maintain compatibility with vaila/__init__.py)
###############################################################################
def detect_delimiter(file_path):
    """
    Detects the delimiter used in the file by trying common delimiters.
    Analyzes multiple lines for more accurate detection.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: Detected delimiter (',', ';', '\t', or ' ')
    """
    import csv
    
    # Try different delimiters and check consistency
    delimiters = [',', ';', '\t', ' ']
    delimiter_scores = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read first few lines
            sample_lines = []
            file.seek(0)
            for i, line in enumerate(file):
                if i >= 10:  # Read max 10 lines
                    break
                sample_lines.append(line.strip())
            
            for delimiter in delimiters:
                scores = []
                try:
                    # Use csv.Sniffer for better detection
                    file.seek(0)
                    sample = file.read(1024)
                    if sample:  # Check if sample is not empty
                        dialect = csv.Sniffer().sniff(sample, delimiters=[delimiter])
                        if dialect.delimiter == delimiter:
                            scores.append(10)  # High score for csv.Sniffer detection
                except Exception:
                    pass
                
                # Check consistency of column count
                column_counts = []
                for line in sample_lines:
                    if line:
                        column_counts.append(len(line.split(delimiter)))
                
                if column_counts:
                    # Prefer delimiters that give consistent column counts
                    most_common_count = max(set(column_counts), key=column_counts.count)
                    consistency = column_counts.count(most_common_count) / len(column_counts)
                    avg_columns = sum(column_counts) / len(column_counts)
                    
                    # Score based on consistency and reasonable column count
                    score = consistency * 10 + (avg_columns if avg_columns > 1 else 0)
                    delimiter_scores[delimiter] = score
            
            # Return delimiter with highest score
            if delimiter_scores:
                best_delimiter = max(delimiter_scores, key=delimiter_scores.get)
                print(f"Delimiter detected: '{best_delimiter}' (score: {delimiter_scores[best_delimiter]:.2f})")
                return best_delimiter
            
    except Exception as e:
        print(f"Warning: Error detecting delimiter: {e}")
    
    print("Using default delimiter: ','")
    return ','  # Default to comma


###############################################################################
# Function: detect_has_header
# (Added to maintain compatibility with vaila/__init__.py)
###############################################################################
def detect_has_header(file_path, delimiter):
    """
    Detects if the file has a header by checking if the first line contains non-numeric values.
    
    Args:
        file_path (str): Path to the file
        delimiter (str): Delimiter used in the file
        
    Returns:
        bool: True if file has header, False otherwise
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        first_line = file.readline().strip()
        values = first_line.split(delimiter)
        
        # Check if any value in the first line is not numeric
        for value in values:
            try:
                float(value)
            except ValueError:
                return True
    return False


###############################################################################
# Function: select_file
# (Added to maintain compatibility with vaila/__init__.py)
###############################################################################
def select_file():
    """Displays the dialog box for selecting CSV, TXT or TSV file."""
    return filedialog.askopenfilename(
        title="Select data file",
        filetypes=[
            ("Data files", ("*.csv", "*.txt", "*.tsv")),
            ("CSV files", "*.csv"),
            ("Text files", "*.txt"),
            ("TSV files", "*.tsv"),
            ("All files", "*.*")
        ]
    )


###############################################################################
# Function: choose_visualizer
# Opens a window for the user to choose the visualization method.
###############################################################################
def choose_visualizer():
    """
    Opens a window for the user to choose the visualization method.
    Returns:
        "matplotlib" or "open3d" based on user choice.
    """
    root = tk.Tk()
    root.title("Select Visualizer")
    choice: list[str | None] = [None]  # Add type annotation

    def choose_matplotlib():
        choice[0] = "matplotlib"
        root.quit()

    def choose_open3d():
        choice[0] = "open3d"
        root.quit()

    Label(root, text="Choose visualization method:").pack(pady=10)
    Button(root, text="Matplotlib Visualizer", command=choose_matplotlib).pack(pady=5)
    Button(root, text="Open3D Visualizer", command=choose_open3d).pack(pady=5)
    root.mainloop()
    root.destroy()
    return choice[0]


###############################################################################
# Function: select_markers_csv
# Displays a marker selection window using a Listbox.
###############################################################################
def select_markers_csv(marker_labels):
    """
    Displays a Tkinter window with a Listbox for marker selection.

    Args:
        marker_labels (list): list of marker names.

    Returns:
        List of selected marker names.
    """
    root = tk.Tk()
    root.title("Select Markers to Display")
    listbox = Listbox(root, selectmode="multiple", width=50, height=15)
    for label in marker_labels:
        listbox.insert(tk.END, label)
    listbox.pack(padx=10, pady=10)

    btn_frame = Frame(root)
    btn_frame.pack(pady=5)

    def select_all():
        listbox.select_set(0, tk.END)

    def unselect_all():
        listbox.selection_clear(0, tk.END)

    Button(btn_frame, text="Select All", command=select_all).pack(side="left", padx=5)
    Button(btn_frame, text="Unselect All", command=unselect_all).pack(
        side="left", padx=5
    )

    Button(root, text="Select", command=root.quit).pack(pady=10)
    root.mainloop()
    selected_indices = listbox.curselection()
    root.destroy()
    # Return the marker names corresponding to the selected indices
    return [marker_labels[int(i)] for i in selected_indices]


###############################################################################
# Function: select_headers_gui
# (Added to maintain compatibility with vaila/__init__.py)
###############################################################################
def select_headers_gui(headers):
    """
    Displays a Tkinter window with a Listbox for header (column) selection.

    Args:
        headers (list): List of CSV headers.

    Returns:
        List of selected headers.
    """
    root = tk.Tk()
    root.title("Select CSV Headers")
    listbox = Listbox(root, selectmode="multiple", width=50, height=15)
    for header in headers:
        listbox.insert(tk.END, header)
    listbox.pack(padx=10, pady=10)

    btn_frame = Frame(root)
    btn_frame.pack(pady=5)

    def select_all():
        listbox.select_set(0, tk.END)

    def unselect_all():
        listbox.selection_clear(0, tk.END)

    Button(btn_frame, text="Select All", command=select_all).pack(side="left", padx=5)
    Button(btn_frame, text="Unselect All", command=unselect_all).pack(
        side="left", padx=5
    )

    Button(root, text="OK", command=root.quit).pack(pady=10)
    root.mainloop()
    selected_indices = listbox.curselection()
    root.destroy()
    return [headers[int(i)] for i in selected_indices]


###############################################################################
# Function: get_csv_headers
# (Added to maintain compatibility with vaila/__init__.py)
###############################################################################
def get_csv_headers(file_path):
    """
    Reads the CSV file at the given file_path and returns its headers (column names).

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        List of headers (str).
    """
    try:
        df = pd.read_csv(file_path, nrows=0)
        return list(df.columns)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read CSV headers: {e}")
        return []


###############################################################################
# Function: show_csv_open3d
# Visualizes marker data using Open3D (similar to viewc3d.py).
###############################################################################
def show_csv_open3d(points, marker_names, fps=30):
    """
    Visualizes CSV marker data using Open3D.

    Args:
        points: numpy array of shape (num_frames, num_markers, 3)
        marker_names: list of marker names corresponding to the second dimension
        fps: frames per second for animation
    """
    try:
        import open3d as o3d
    except ImportError:
        print("open3d is not installed. Install it with 'pip install open3d'.")
        return

    num_frames, num_markers, _ = points.shape
    print(f"Open3D visualization stub: {num_frames} frames, {num_markers} markers.")
    # Here you would implement the actual Open3D visualization.
    # For this stub, we just simulate a delay.
    time.sleep(2)
    print("Open3D visualization complete.")


###############################################################################
# Function: show_csv_matplotlib
# Visualizes marker data using Matplotlib (similar to showc3d.py).
###############################################################################
def show_csv_matplotlib(points, marker_names, fps=60, *, turbo2d=False, stride=1, marker_stride=1, hide_axes=True, dpi=100):
    """
    Fast Matplotlib playback (no Speed slider). Two modes:
    - 3D minimal (default): single scatter, updates offsets only.
    - 2D turbo (turbo2d=True): XY projection with blitting for high FPS.
    Perf knobs: stride (frames), marker_stride, hide_axes, dpi.
    """
    import numpy as np
    import time

    try:
        import matplotlib
        try:
            matplotlib.use("QtAgg")  # usually faster
        except Exception:
            pass
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Please 'pip install matplotlib'.")
        return

    stride = max(1, int(stride))
    marker_stride = max(1, int(marker_stride))
    pts = points[::stride, ::marker_stride, :]

    T, M, _ = pts.shape
    if T == 0 or M == 0:
        print("[readcsv] Nothing to display after decimation."); return

    valid_mask = ~np.isnan(pts).any(axis=2)
    pts = pts.copy()
    pts[~valid_mask] = np.nan

    x = pts[:,:,0].reshape(-1)
    y = pts[:,:,1].reshape(-1)
    z = pts[:,:,2].reshape(-1)
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    if np.any(valid):
        x_valid = x[valid]; y_valid = y[valid]; z_valid = z[valid]
        pad = 0.05
        xr = (np.nanmin(x_valid), np.nanmax(x_valid)); dx = xr[1]-xr[0] or 1.0
        yr = (np.nanmin(y_valid), np.nanmax(y_valid)); dy = yr[1]-yr[0] or 1.0
        zr = (np.nanmin(z_valid), np.nanmax(z_valid)); dz = zr[1]-zr[0] or 1.0
        xr = (xr[0]-dx*pad, xr[1]+dx*pad)
        yr = (yr[0]-dy*pad, yr[1]+dy*pad)
        zr = (zr[0]-dz*pad, zr[1]+dz*pad)
    else:
        xr = yr = zr = (-1, 1)

    # ----- Turbo 2D with blitting -----
    if turbo2d:
        fig, ax = plt.subplots(figsize=(9, 6), dpi=dpi)
        if hide_axes:
            ax.set_axis_off()
        else:
            ax.set_xlabel("X"); ax.set_ylabel("Y")

        f0 = 0
        xy0 = np.c_[pts[f0, :, 0], pts[f0, :, 1]]
        scat = ax.scatter(xy0[:,0], xy0[:,1], s=10, antialiased=False)
        ax.set_xlim(*xr); ax.set_ylim(*yr)

        plt.ion()
        fig.canvas.draw()
        try:
            bg = fig.canvas.copy_from_bbox(fig.bbox)
            can_blit = True
        except Exception:
            can_blit = False
            print("[readcsv] Blitting not supported; using normal redraws.")

        cur = {"frame": 0, "playing": True}
        target = 1.0/float(fps)
        last = time.perf_counter()

        def set_offsets2d(coll, arr2):
            coll.set_offsets(arr2)

        def draw_frame():
            nonlocal last
            f = cur["frame"]
            arr2 = np.c_[pts[f, :, 0], pts[f, :, 1]]
            if can_blit:
                fig.canvas.restore_region(bg)
                set_offsets2d(scat, arr2)
                ax.draw_artist(scat)
                fig.canvas.blit(fig.bbox)
                fig.canvas.flush_events()
            else:
                set_offsets2d(scat, arr2)
                fig.canvas.draw_idle()
                plt.pause(0.001)
            now = time.perf_counter()
            elapsed = now - last
            if elapsed < target:
                time.sleep(max(0.0, target - elapsed))
            last = time.perf_counter()

        def on_key(event):
            if event.key == ' ':
                cur["playing"] = not cur["playing"]
            elif event.key == 'right':
                cur["frame"] = (cur["frame"] + 1) % T; draw_frame()
            elif event.key == 'left':
                cur["frame"] = (cur["frame"] - 1) % T; draw_frame()
            elif event.key == 'up':
                cur["frame"] = (cur["frame"] + 10) % T; draw_frame()
            elif event.key == 'down':
                cur["frame"] = (cur["frame"] - 10) % T; draw_frame()
            elif event.key in ('q', 'Q', 'escape'):
                cur["playing"] = False; plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)

        try:
            while plt.fignum_exists(fig.number):
                if cur["playing"]:
                    cur["frame"] = (cur["frame"] + 1) % T
                    draw_frame()
                else:
                    plt.pause(0.01)
        except KeyboardInterrupt:
            pass
        return

    # ----- 3D minimal -----
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(9, 6), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    if hide_axes:
        try: ax.set_axis_off(); ax._axis3don = False
        except Exception: pass
    else:
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xlim(*xr); ax.set_ylim(*yr); ax.set_zlim(*zr)
    try: ax.set_proj_type('ortho')
    except Exception: pass

    f0 = 0
    scat = ax.scatter(pts[f0,:,0], pts[f0,:,1], pts[f0,:,2], s=10, antialiased=False, depthshade=False)

    def set_offsets3d(coll, xs, ys, zs):
        coll._offsets3d = (xs, ys, zs)

    cur = {"frame": 0, "playing": True}
    import time as _time
    target = 1.0/float(fps)
    last = _time.perf_counter()

    def draw_frame():
        nonlocal last
        f = cur["frame"]
        set_offsets3d(scat, pts[f,:,0], pts[f,:,1], pts[f,:,2])
        fig.canvas.draw_idle()
        now = _time.perf_counter()
        elapsed = now - last
        if elapsed < target:
            _time.sleep(max(0.0, target - elapsed))
        last = _time.perf_counter()

    def on_key(event):
        if event.key == ' ':
            cur["playing"] = not cur["playing"]
        elif event.key == 'right':
            cur["frame"] = (cur["frame"] + 1) % T; draw_frame()
        elif event.key == 'left':
            cur["frame"] = (cur["frame"] - 1) % T; draw_frame()
        elif event.key == 'up':
            cur["frame"] = (cur["frame"] + 10) % T; draw_frame()
        elif event.key == 'down':
            cur["frame"] = (cur["frame"] - 10) % T; draw_frame()
        elif event.key in ('q', 'Q', 'escape'):
            cur["playing"] = False; plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    try:
        while plt.fignum_exists(fig.number):
            if cur["playing"]:
                cur["frame"] = (cur["frame"] + 1) % T
                draw_frame()
            else:
                plt.pause(0.01)
    except KeyboardInterrupt:
        pass



###############################################################################
# Function: detect_units
# (Added to maintain compatibility with vaila/__init__.py)
###############################################################################
def detect_units(points):
    """
    Detects if the data is in millimeters or meters based on the magnitude of values.
    
    Args:
        points (numpy.ndarray): Array of shape (num_frames, num_markers, 3) containing the coordinates
        
    Returns:
        bool: True if data is in millimeters (needs conversion), False if already in meters
    """
    # Calculate the mean absolute value of all coordinates
    mean_abs = np.mean(np.abs(points))
    
    # If mean absolute value is greater than 100, likely in millimeters
    return mean_abs > 100


def ask_user_units():
    """
    Ask user about the units of the coordinate data.
    
    Returns:
        str: 'mm' for millimeters, 'm' for meters, 'auto' for automatic detection
    """
    import tkinter as tk
    from tkinter import messagebox, simpledialog
    
    try:
        # Try simple dialog first (more compatible)
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        # Simple approach using messagebox
        response = messagebox.askyesnocancel(
            "Select Coordinate Units",
            "What are the units of your coordinate data?\n\n"
            "• YES = Millimeters (mm) - typical values like 1234.56\n"
            "• NO = Meters (m) - typical values like 1.23\n"
            "• CANCEL = Auto-detect (let program guess)\n\n"
            "Choose based on your data range shown in terminal."
        )
        
        root.destroy()
        
        if response is True:
            return 'mm'
        elif response is False:
            return 'm'
        else:
            return 'auto'
            
    except Exception as e:
        print(f"Warning: Could not show unit selection dialog: {e}")
        print("Using auto-detection for units.")
        return 'auto'


def read_csv_generic(file_path):
    """
    Reads a CSV or TXT file considering:
      - Automatically detects the delimiter (',', ';', '\t', ' ')
      - Detects if the file has a header
      - If it doesn't have a header, uses default names (Index, p1_x, p1_y, p1_z, p2_x, ...)
      - The first column contains any index data (time, frames, or any identifier)
      - The subsequent columns are organized in groups of three (x, y, z)
      - Asks user about units and converts accordingly

    Returns:
      index_vector: pd.Series with the first column data (time/frames/index)
      marker_data: dictionary mapping the marker name to a numpy array Nx3
      valid_markers: dictionary mapping the marker name to the list of columns used
      delimiter: the detected delimiter
    """
    print(f"Processing file: {file_path}")
    
    # Detect delimiter
    delimiter = detect_delimiter(file_path)
    
    # Detect if file has header
    has_header = detect_has_header(file_path, delimiter)
    print(f"Header detected: {has_header}")
    
    # Read the file
    try:
        if has_header:
            df = pd.read_csv(file_path, delimiter=delimiter)
        else:
            # Create default column names for files without headers
            # First read to count columns
            temp_df = pd.read_csv(file_path, delimiter=delimiter, nrows=1)
            num_columns = len(temp_df.columns)
            
            default_columns = ['Index']  # Generic name for first column
            for i in range(1, (num_columns - 1) // 3 + 1):
                default_columns.extend([f'p{i}_x', f'p{i}_y', f'p{i}_z'])
            
            # Add remaining columns if any
            while len(default_columns) < num_columns:
                default_columns.append(f'col_{len(default_columns)}')
                
            df = pd.read_csv(file_path, delimiter=delimiter, names=default_columns)
        
        print(f"Successfully read CSV with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    if df.empty:
        raise ValueError("The file is empty or could not be read.")

    # The first column is generic (time/frames/index/etc.)
    index_vector = df.iloc[:, 0]

    # Process the remaining columns: each column must have the format marker_coord (ex.: MARKER_X)
    # Support multiple naming conventions
    marker_headers = {}
    for col in df.columns[1:]:
        marker_name = None
        coord = None
        
        # Try different naming patterns
        # Pattern 1: marker_X, marker_Y, marker_Z or marker_x, marker_y, marker_z
        if "_" in col:
            parts = col.rsplit("_", 1)  # split by the last occurrence of '_'
            if len(parts) == 2 and parts[1].upper() in ["X", "Y", "Z"]:
                marker_name = parts[0]
                coord = parts[1].upper()
        
        # Pattern 2: markerX, markerY, markerZ (no underscore)
        elif col[-1].upper() in ["X", "Y", "Z"]:
            marker_name = col[:-1]
            coord = col[-1].upper()
        
        # Pattern 3: marker.x, marker.y, marker.z
        elif "." in col:
            parts = col.rsplit(".", 1)
            if len(parts) == 2 and parts[1].upper() in ["X", "Y", "Z"]:
                marker_name = parts[0]
                coord = parts[1].upper()
        
        # Pattern 4: marker:x, marker:y, marker:z
        elif ":" in col:
            parts = col.rsplit(":", 1)
            if len(parts) == 2 and parts[1].upper() in ["X", "Y", "Z"]:
                marker_name = parts[0]
                coord = parts[1].upper()
        
        if marker_name:
            if marker_name not in marker_headers:
                marker_headers[marker_name] = {}
            marker_headers[marker_name][coord] = col
        else:
            print(f"Warning: Could not parse column '{col}' - no matching pattern found")

    print(f"Found marker patterns: {list(marker_headers.keys())}")
    
    # Debug: Show details of marker parsing
    if len(marker_headers) <= 10:  # Only show details if reasonable number
        for marker, coords in marker_headers.items():
            print(f"  {marker}: {coords}")

    # Select only the markers that have the complete set of 3 columns
    valid_markers = {}
    for marker, coord_dict in marker_headers.items():
        if len(coord_dict) == 3 and all(coord in coord_dict for coord in ['X', 'Y', 'Z']):
            # Ensure correct order: X, Y, Z
            ordered_cols = [coord_dict['X'], coord_dict['Y'], coord_dict['Z']]
            valid_markers[marker] = ordered_cols
        else:
            available_coords = list(coord_dict.keys()) if coord_dict else []
            print(f"Warning: The marker '{marker}' has incomplete data. Available coordinates: {available_coords}")

    print(f"Valid markers found: {list(valid_markers.keys())}")

    # Extract the data for each marker into an Nx3 array
    marker_data = {}
    for marker, cols in valid_markers.items():
        marker_data[marker] = df[cols].to_numpy()

    # Handle unit conversion based on user input
    if valid_markers:
        # Create a temporary array with all points to check units
        temp_points = np.stack([marker_data[marker] for marker in valid_markers.keys()], axis=1)
        
        # Calculate statistics for user information
        valid_coords = temp_points[~np.isnan(temp_points)]
        if len(valid_coords) > 0:
            mean_abs_value = np.mean(np.abs(valid_coords))
            min_value = np.min(valid_coords)
            max_value = np.max(valid_coords)
            
            print(f"\nCoordinate data statistics:")
            print(f"  Mean absolute value: {mean_abs_value:.3f}")
            print(f"  Range: [{min_value:.3f}, {max_value:.3f}]")
            print(f"  Data preview (first marker, first 3 records):")
            first_marker = list(marker_data.keys())[0]
            preview_data = marker_data[first_marker][:3]
            for i, coords in enumerate(preview_data):
                if not np.isnan(coords).any():
                    print(f"    Record {i}: X={coords[0]:.6f}, Y={coords[1]:.6f}, Z={coords[2]:.6f}")
            
            # Ask user about units
            print(f"\nPlease select units in the dialog box...")
            user_units = ask_user_units()
            print(f"User selected: {user_units}")
            
            if user_units == 'mm':
                print("Converting from millimeters to meters...")
                # Convert all marker data from millimeters to meters
                for marker in marker_data:
                    marker_data[marker] = marker_data[marker] * 0.001
                print("Unit conversion completed: mm → m")
                
                # Show converted preview
                print(f"After conversion (first marker, first 3 records):")
                preview_data = marker_data[first_marker][:3]
                for i, coords in enumerate(preview_data):
                    if not np.isnan(coords).any():
                        print(f"    Record {i}: X={coords[0]:.6f}, Y={coords[1]:.6f}, Z={coords[2]:.6f}")
            
        elif user_units == 'm':
            print("No conversion needed - data is already in meters.")
            
        elif user_units == 'auto':
            print("Using auto-detection for units...")
            if detect_units(temp_points):
                print("Auto-detection: Data appears to be in millimeters. Converting to meters...")
                # Convert all marker data from millimeters to meters
                for marker in marker_data:
                    marker_data[marker] = marker_data[marker] * 0.001
                print("Unit conversion completed: mm → m")
            else:
                print("Auto-detection: Data appears to be already in meters.")

        # Check for and filter extreme outliers after unit conversion
        print(f"\nChecking for extreme outliers...")
        temp_points_converted = np.stack([marker_data[marker] for marker in valid_markers.keys()], axis=1)
        valid_coords = temp_points_converted[~np.isnan(temp_points_converted)]
        
        if len(valid_coords) > 0:
            # Calculate statistics to detect outliers
            q25 = np.percentile(valid_coords, 25)
            q75 = np.percentile(valid_coords, 75)
            iqr = q75 - q25
            median = np.median(valid_coords)
            
            # Define outlier bounds (more conservative for coordinate data)
            outlier_threshold = 10 * iqr  # Allow larger range for coordinate data
            lower_bound = q25 - outlier_threshold
            upper_bound = q75 + outlier_threshold
            
            # Count outliers
            outliers = (valid_coords < lower_bound) | (valid_coords > upper_bound)
            outlier_count = np.sum(outliers)
            outlier_percentage = (outlier_count / len(valid_coords)) * 100
            
            print(f"Data quality check:")
            print(f"  Median: {median:.3f}")
            print(f"  Q25-Q75: [{q25:.3f}, {q75:.3f}]")
            print(f"  IQR: {iqr:.3f}")
            print(f"  Outlier bounds: [{lower_bound:.3f}, {upper_bound:.3f}]")
            print(f"  Outliers found: {outlier_count}/{len(valid_coords)} ({outlier_percentage:.1f}%)")
            
            if outlier_percentage > 5:  # If more than 5% outliers
                print(f"Warning: High percentage of outliers detected!")
                print(f"This may indicate data quality issues or incorrect units.")
                print(f"Consider reviewing your data source.")

    else:
        print("Warning: No valid coordinate data found.")

    return index_vector, marker_data, valid_markers, delimiter


###############################################################################
# Function: show_csv (Main Function)
# - Opens a file selection dialog to pick the CSV file.
# - Extracts the marker names (ignoring the first "Time" column).
# - Opens a marker selection dialog.
# - Constructs an array of marker positions of shape (num_frames, num_markers, 3).
# - Prompts the user to choose a visualization method.
# - Launches the visualization using either Open3D or Matplotlib.
###############################################################################
def show_csv(file_path=None, *, precision: int = 3, use_cache: bool = True, force_recache: bool = False, turbo2d: bool = False, stride: int = 1, marker_stride: int = 1, hide_axes: bool = True, dpi: int = 100):
    """
    Fast path: load CSV with NPZ cache, let user pick markers, then minimal Matplotlib playback.
    """
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    
    # File selection
    if file_path is None:
        root = tk.Tk(); root.withdraw()
        file_path = select_file()
        if not file_path:
            print("No file selected."); return
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        print(f"[bold red]Error:[/bold red] Invalid file: {file_path}"); return
    
    # Load (prefer cache)
    try:
        index_vector, all_markers, all_points, from_cache, delim = _load_with_cache(
            file_path, precision=precision, use_cache=use_cache, force_recache=force_recache
        )
        print(f"[readcsv] Loaded {'from NPZ cache' if from_cache else 'from CSV'} | dtype={all_points.dtype} | shape={all_points.shape}")
    except Exception as e:
        messagebox.showerror("Error", f"Error loading data: {e}")
        return
    
    # Marker selection
    selected_markers = select_markers_csv(all_markers)
    if not selected_markers:
        messagebox.showwarning("Warning", "No markers selected."); return
    
    sel_idx = [all_markers.index(m) for m in selected_markers]
    points = all_points[:, sel_idx, :]
    
    # Show with performance optimizations
    show_csv_matplotlib(points, selected_markers, fps=60, turbo2d=turbo2d, stride=stride, marker_stride=marker_stride, hide_axes=hide_axes, dpi=dpi)

if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser(
        description="Fast CSV viewer with NPZ caching and performance optimizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Performance Tips:
  --turbo2d          Use 2D projection with blitting (much faster than 3D)
  --stride 2         Skip every 2nd frame (reduces data by 50%%)
  --marker-stride 2  Skip every 2nd marker (reduces data by 50%%)
  --dpi 72          Lower DPI = faster rendering
  --precision 3     Float32 with 3 decimal places (default)

Examples:
  # Turbo mode (fastest)
  python readcsv.py data.csv --turbo2d --dpi 72 --stride 2
  
  # 3D mode with optimizations
  python readcsv.py data.csv --precision 3 --dpi 72 --stride 2
  
  # Convert to NPZ for faster loading
  python readcsv.py --csv2npz data.csv --precision 3
        """
    )
    ap.add_argument("file", nargs="?", help="Path to CSV/TXT/TSV")
    ap.add_argument("--precision", type=int, default=3, help="Decimals to round (float32)")
    ap.add_argument("--no-cache", action="store_true", help="Disable NPZ cache")
    ap.add_argument("--force-recache", action="store_true", help="Rebuild NPZ even if fresh")
    ap.add_argument("--csv2npz", metavar="CSV", help="Convert CSV to NPZ and exit")
    # Performance flags
    ap.add_argument("--turbo2d", action="store_true", help="Blitted 2D XY projection (much faster)")
    ap.add_argument("--stride", type=int, default=1, help="Frame decimation (>=1)")
    ap.add_argument("--marker-stride", type=int, default=1, help="Marker decimation (>=1)")
    ap.add_argument("--show-axes", action="store_true", help="Show axes/ticks (default hides)")
    ap.add_argument("--dpi", type=int, default=100, help="Figure DPI (lower=faster)")
    args = ap.parse_args()

    if args.csv2npz:
        out = csv2npz_inplace(args.csv2npz, precision=args.precision)
        print(f"NPZ file created: {out}")
        sys.exit(0)

    file_path = args.file if args.file else None
    show_csv(file_path, precision=args.precision, use_cache=(not args.no_cache), force_recache=args.force_recache,
             turbo2d=args.turbo2d, stride=args.stride, marker_stride=args.marker_stride, hide_axes=(not args.show_axes), dpi=args.dpi)