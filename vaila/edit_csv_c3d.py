"""
===============================================================================
edit_csv_c3d.py
===============================================================================
Author: Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 24 August 2026
Update Date: 23 September 2026
Version: 0.4.5

Description:
Edit CSV/C3D (Frame C button `C_A_r1_c1`). Opens a directory that holds
`.csv` and/or `.c3d` files and applies the same column-editing tools from
`rearrange_data.ColumnReorderGUI` to both:

- `.csv` files go straight into the editor, exactly like the old "Edit CSV"
  button.
- `.c3d` files are converted to a marker CSV first (`readc3d_export.
  c3d_markers_to_dataframe`, headless), edited alongside the CSVs, then
  converted back to `.c3d` (`readcsv_export.auto_create_c3d_from_csv`),
  preserving POINT RATE, ANALOG RATE, POINT UNITS, analog channels, and
  occlusion residual flags (NaN <-> negative residual).

Source files are never overwritten. Every run writes into a fresh
`processed_edit_csv_c3d_YYYYMMDD_HHMMSS/` directory.

The **C3D <--> CSV** button (`C_A_r1_c2`, `readc3d_export.py` /
`readcsv_export.py` batch converters) is a separate tool and is not touched
or replaced by this module.

Usage:
    GUI (same as the button):
        uv run vaila/edit_csv_c3d.py

    GUI pre-filled with a directory:
        uv run vaila/edit_csv_c3d.py -i INPUT_DIR [-o OUTPUT_DIR]

    Headless (no Tk, used by tests and scripting):
        uv run vaila/edit_csv_c3d.py -i INPUT_DIR -o OUTPUT_DIR --identity
        uv run vaila/edit_csv_c3d.py -i INPUT_DIR -o OUTPUT_DIR --columns COL1,COL2,...
        uv run vaila/edit_csv_c3d.py -i INPUT_DIR -o OUTPUT_DIR --identity -r
        uv run vaila/edit_csv_c3d.py -i INPUT_DIR -o OUTPUT_DIR --identity -d 2

Notes:
- `--columns` keeps/reorders headers by exact name (mirrors `rearrange_data.
  reshapedata`'s filtering). For `.c3d`-derived data, `auto_create_c3d_from_csv`
  derives marker labels from complete `LABEL_X/Y/Z` triples in column order,
  so a `--columns` list touching C3D markers must keep whole X/Y/Z triples
  together and in order or the round-tripped C3D will be malformed.
- `-r/--recursive` (equivalent to `-d -1`) and `-d/--depth N` scan
  subdirectories headlessly: `-1` unlimited, `0` `INPUT_DIR` only (default),
  `N` levels down. Output mirrors each file's original subdirectory.
- Interactively (GUI), after picking a directory a "Select Files to Edit"
  dialog lets you set the same depth, scan, then choose "Process All" or
  "Process Selected File" to edit just one file.
- Bulk column rename/renumber (e.g. `p1_x..p70_y` -> `p0_x..p69_y`) is
  GUI-only: `ColumnReorderGUI`'s Edit menu -> "Rename Column(s)...".
"""

import argparse
import contextlib
import glob
import os
import re
import shutil
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import pandas as pd

try:
    from .cli_highlight import print_gui_cli_mirror
    from .readc3d_export import c3d_markers_to_dataframe
    from .readcsv_export import auto_create_c3d_from_csv
    from .rearrange_data import ColumnReorderGUI, get_headers
except ImportError:  # standalone execution
    from cli_highlight import print_gui_cli_mirror  # ty: ignore[unresolved-import]
    from readc3d_export import c3d_markers_to_dataframe  # ty: ignore[unresolved-import]
    from readcsv_export import auto_create_c3d_from_csv  # ty: ignore[unresolved-import]
    from rearrange_data import ColumnReorderGUI, get_headers  # ty: ignore[unresolved-import]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_output_dir(input_dir: str) -> str:
    return os.path.join(input_dir, f"processed_edit_csv_c3d_{_timestamp()}")


_PROCESSED_OUTPUT_DIR_RE = re.compile(r"^processed_edit_csv_c3d_\d{8}_\d{6}$")
_SKIP_STAGING_DIR_NAMES = frozenset({"_staging", "data_rearranged"})


def _should_prune_dir(name: str) -> bool:
    """True for directories a recursive scan must never descend into: hidden
    dirs, this module's own `processed_edit_csv_c3d_*` output dirs, and the
    `_staging`/`data_rearranged` working dirs a prior run left behind."""
    return (
        name.startswith(".")
        or name in _SKIP_STAGING_DIR_NAMES
        or bool(_PROCESSED_OUTPUT_DIR_RE.match(name))
    )


def find_edit_csv_c3d_files(input_dir: str, max_depth: int = 0) -> list[str]:
    """Depth-limited recursive scan for `.csv`/`.c3d` files under `input_dir`.

    Same pruning idiom as `find_markerless_batch_directories`
    (markerless_2d_analysis.py) and `find_videos_recursive`
    (compress_videos_h264.py): `os.walk` with in-place `dirnames[:]`
    pruning so a rerun never walks into its own prior output.

    max_depth: -1 unlimited, 0 `input_dir` only (legacy top-level-only
    behavior), N = N levels below `input_dir`.

    Returns sorted POSIX-relative paths from `input_dir`, e.g.
    `"sub/dir/file.csv"` (or bare `"file.csv"` for a root-level file).
    """
    root = os.path.abspath(input_dir)
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        depth = 0 if rel_dir == "." else len(rel_dir.split(os.sep))
        dirnames[:] = sorted(
            name
            for name in dirnames
            if not _should_prune_dir(name) and (max_depth < 0 or depth < max_depth)
        )
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            if name.lower().endswith((".csv", ".c3d")):
                rel_path = name if rel_dir == "." else f"{rel_dir}/{name}"
                found.append(rel_path.replace(os.sep, "/"))
    return sorted(found)


def _list_input_files(input_dir: str) -> list[str]:
    """Back-compat thin wrapper: top-level-only scan (`max_depth=0`)."""
    return find_edit_csv_c3d_files(input_dir, max_depth=0)


def _select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Keep/reorder columns by exact header name (existing ones only)."""
    existing = [c for c in columns if c in df.columns]
    return df[existing]


def _flatten_staged_name(rel_stem: str) -> str:
    """Collision-safe flat staged CSV name for a POSIX-relative stem (no
    extension), e.g. `"sub/dir/file"` -> `"sub__dir__file.csv"`.
    `ColumnReorderGUI` stages every file into one flat directory (it has no
    subdirectory awareness), so files with the same basename in different
    subdirectories must not collide once staged."""
    return f"{rel_stem.replace('/', '__')}.csv"


def _stage_inputs(input_dir: str, staging_dir: str, rel_paths: list[str]) -> dict:
    """Convert every `.c3d` to a staged marker CSV, copy every `.csv` as-is.

    `rel_paths` are POSIX-relative paths (as returned by
    `find_edit_csv_c3d_files`) naming exactly which files to stage — a
    single-element list is how "process just this one file" is expressed.

    Returns `{rel_stem: entry}` where `entry["kind"]` is "csv" or "c3d",
    `entry["staged_name"]` is the flattened file name inside `staging_dir`,
    `entry["rel_dir"]` is the file's original subdirectory (POSIX, `""` for
    `input_dir`'s root), and (for "c3d") `entry["meta"]` carries the
    round-trip metadata from `c3d_markers_to_dataframe`.
    """
    os.makedirs(staging_dir, exist_ok=True)
    entries: dict = {}
    for rel_path in rel_paths:
        source = os.path.join(input_dir, *rel_path.split("/"))
        rel_stem, ext = os.path.splitext(rel_path)
        ext = ext.lower()
        rel_dir = os.path.dirname(rel_stem)
        staged_name = _flatten_staged_name(rel_stem)
        dest = os.path.join(staging_dir, staged_name)
        if ext == ".csv":
            shutil.copyfile(source, dest)
            entries[rel_stem] = {
                "kind": "csv",
                "staged_name": staged_name,
                "source": source,
                "rel_dir": rel_dir,
            }
        elif ext == ".c3d":
            markers_df, meta = c3d_markers_to_dataframe(source)
            markers_df.to_csv(dest, index=False)
            entries[rel_stem] = {
                "kind": "c3d",
                "staged_name": staged_name,
                "source": source,
                "rel_dir": rel_dir,
                "meta": meta,
            }
    return entries


def _newest_edited_csv(rearranged_dir: str, stem: str) -> str | None:
    """Most recently written `data_rearranged/{stem}_*.csv`, if any.

    An explicit Save & Exit (`ColumnReorderGUI.save_and_exit`, suffix
    `_final`) always wins over a plain intermediate Ctrl+S save when both
    exist, since it is the user's deliberate end state."""
    if not os.path.isdir(rearranged_dir):
        return None
    final_candidates = glob.glob(os.path.join(rearranged_dir, f"{stem}_*_final.csv"))
    if final_candidates:
        final_candidates.sort(key=os.path.getmtime, reverse=True)
        return final_candidates[0]
    candidates = glob.glob(os.path.join(rearranged_dir, f"{stem}_*.csv"))
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _write_output(rel_stem: str, entry: dict, df: pd.DataFrame, output_dir: str) -> str:
    """Write one entry's (possibly edited) DataFrame to `output_dir`, mirroring
    the file's original subdirectory (`entry["rel_dir"]`)."""
    rel_dir = entry.get("rel_dir", "")
    dest_dir = os.path.join(output_dir, *rel_dir.split("/")) if rel_dir else output_dir
    os.makedirs(dest_dir, exist_ok=True)
    base = os.path.basename(rel_stem)

    if entry["kind"] == "csv":
        out_path = os.path.join(dest_dir, f"{base}_final.csv")
        df.to_csv(out_path, index=False)
        return out_path

    meta = entry["meta"]
    out_path = os.path.join(dest_dir, f"{base}.c3d")
    auto_create_c3d_from_csv(
        df,
        out_path,
        analog_df=meta.get("analog_df"),
        point_rate=meta.get("marker_freq", 100.0),
        analog_rate=meta.get("analog_freq", 1000.0),
        point_units=meta.get("point_units"),
    )
    return out_path


def _finalize_from_staging(entries: dict, staging_dir: str, output_dir: str) -> list[str]:
    """After the GUI editor closes: pick each entry's edited CSV and write it out."""
    os.makedirs(output_dir, exist_ok=True)
    rearranged_dir = os.path.join(staging_dir, "data_rearranged")
    written = []
    for rel_stem, entry in entries.items():
        staged_stem = os.path.splitext(entry["staged_name"])[0]
        edited = _newest_edited_csv(rearranged_dir, staged_stem)
        source_csv = edited or os.path.join(staging_dir, entry["staged_name"])
        df = pd.read_csv(source_csv)
        written.append(_write_output(rel_stem, entry, df, output_dir))
    return written


def _headless_process(
    input_dir: str,
    output_dir: str,
    columns: list[str] | None,
    max_depth: int = 0,
) -> list[str]:
    """No Tk, no GUI editor: read, optionally filter/reorder columns, write.

    max_depth: -1 unlimited, 0 `input_dir` only (legacy default), N = N
    levels below `input_dir`. Output mirrors each file's original
    subdirectory under `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)
    written = []
    for rel_path in find_edit_csv_c3d_files(input_dir, max_depth):
        source = os.path.join(input_dir, *rel_path.split("/"))
        rel_stem, ext = os.path.splitext(rel_path)
        ext = ext.lower()
        rel_dir = os.path.dirname(rel_stem)
        if ext == ".csv":
            df = pd.read_csv(source)
            entry = {"kind": "csv", "rel_dir": rel_dir}
        else:
            df, meta = c3d_markers_to_dataframe(source)
            entry = {"kind": "c3d", "meta": meta, "rel_dir": rel_dir}
        if columns:
            df = _select_columns(df, columns)
        written.append(_write_output(rel_stem, entry, df, output_dir))
    return written


def _prompt_file_selection(parent: tk.Tk, input_dir: str) -> list[str] | None:
    """Modal: pick a scan depth, list matching files, then either process all
    of them or just the one selected in the listbox.

    Returns the chosen list of POSIX-relative paths (as from
    `find_edit_csv_c3d_files`), or `None` if the user cancelled.
    """
    result: list[str] | None = None
    rel_paths: list[str] = []

    window = tk.Toplevel(parent)
    window.title("Select Files to Edit")
    window.geometry("520x480")

    depth_frame = tk.Frame(window)
    depth_frame.pack(fill=tk.X, padx=10, pady=10)
    tk.Label(depth_frame, text="Depth (0=this dir, N=N levels, -1=unlimited):").pack(side=tk.LEFT)
    depth_var = tk.StringVar(value="0")
    tk.Entry(depth_frame, textvariable=depth_var, width=6).pack(side=tk.LEFT, padx=5)

    list_frame = tk.Frame(window)
    list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    scrollbar = tk.Scrollbar(list_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    file_list = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, exportselection=False)
    file_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=file_list.yview)

    status_var = tk.StringVar(value="Click Scan to list files.")
    tk.Label(window, textvariable=status_var, anchor="w").pack(fill=tk.X, padx=10)

    def do_scan():
        nonlocal rel_paths
        try:
            depth = int(depth_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Depth", "Depth must be an integer.", parent=window)
            return
        if depth < -1 or depth > 99:
            messagebox.showerror(
                "Invalid Depth", "Depth must be -1 (unlimited) or between 0 and 99.", parent=window
            )
            return
        rel_paths = find_edit_csv_c3d_files(input_dir, depth)
        file_list.delete(0, tk.END)
        for rel_path in rel_paths:
            file_list.insert(tk.END, rel_path)
        status_var.set(f"{len(rel_paths)} file(s) found.")

    def process_all():
        nonlocal result
        if not rel_paths:
            messagebox.showinfo("Edit CSV/C3D", "No files to process. Scan first.", parent=window)
            return
        result = list(rel_paths)
        window.destroy()

    def process_selected():
        nonlocal result
        selection = file_list.curselection()
        if not selection:
            messagebox.showinfo("Edit CSV/C3D", "Select one file first.", parent=window)
            return
        result = [rel_paths[selection[0]]]
        window.destroy()

    def cancel():
        window.destroy()

    do_scan()

    button_frame = tk.Frame(window)
    button_frame.pack(fill=tk.X, padx=10, pady=10)
    tk.Button(depth_frame, text="Scan", command=do_scan).pack(side=tk.LEFT)
    tk.Button(button_frame, text="Process All", command=process_all).pack(side=tk.LEFT)
    tk.Button(button_frame, text="Process Selected File", command=process_selected).pack(
        side=tk.LEFT, padx=5
    )
    tk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.RIGHT)

    # No `.transient(parent)`: `parent` is a withdrawn `tk.Tk()` root (only
    # used to host `filedialog.askdirectory`), and on X11 a Toplevel made
    # transient for a withdrawn master gets forced back into the withdrawn
    # state itself (even after an explicit `deiconify()`) whenever no window
    # manager remaps it — the dialog would never become visible, and
    # `wait_window()` would then block forever on an invisible window.
    window.deiconify()
    window.lift()
    window.focus_force()
    window.grab_set()
    parent.wait_window(window)
    return result


def run_edit_csv_c3d(
    preset_input_dir: str | None = None, preset_output_dir: str | None = None
) -> None:
    """GUI entry point, called from the `Edit CSV/C3D` button and standalone."""
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Running Edit CSV/C3D")
    print("================================================")

    existing_root = getattr(tk, "_default_root", None)
    owns_root = existing_root is None
    dialog_root = tk.Tk() if owns_root else existing_root
    if owns_root:
        dialog_root.withdraw()

    input_dir = preset_input_dir or filedialog.askdirectory(
        title="Select Directory Containing CSV/C3D Files", parent=dialog_root
    )
    if not input_dir:
        print("No directory selected.")
        if owns_root:
            dialog_root.destroy()
        return

    if preset_input_dir is not None:
        rel_paths = find_edit_csv_c3d_files(input_dir, max_depth=0)
    else:
        rel_paths = _prompt_file_selection(dialog_root, input_dir)
        if rel_paths is None:
            print("Edit CSV/C3D: cancelled.")
            if owns_root:
                dialog_root.destroy()
            return

    if not rel_paths:
        messagebox.showinfo("Edit CSV/C3D", "No .csv or .c3d files found in that directory.")
        print("No .csv or .c3d files found.")
        if owns_root:
            dialog_root.destroy()
        return

    output_dir = preset_output_dir or _default_output_dir(input_dir)

    argv = ["uv", "run", "vaila/edit_csv_c3d.py", "-i", input_dir, "-o", output_dir]
    print_gui_cli_mirror("vaila/edit_csv_c3d", argv)

    if owns_root:
        dialog_root.destroy()

    staging_dir = os.path.join(output_dir, "_staging")
    entries = _stage_inputs(input_dir, staging_dir, rel_paths)

    staged_names = sorted(entry["staged_name"] for entry in entries.values())
    original_headers = get_headers(os.path.join(staging_dir, staged_names[0]))

    # ColumnReorderGUI subclasses tk.Tk directly and manages its own
    # mainloop()/destroy() lifecycle; the picker root above is already gone
    # by the time this runs, so this is the only live Tk root at this point.
    app = ColumnReorderGUI(original_headers, staged_names, staging_dir)
    app.original_input_dir = input_dir
    app.mainloop()

    # Check whether user saved changes (either marked saved or wrote into data_rearranged)
    rearranged_dir = os.path.join(staging_dir, "data_rearranged")
    has_edits = getattr(app, "saved", False) or (
        os.path.isdir(rearranged_dir) and bool(os.listdir(rearranged_dir))
    )

    if has_edits:
        written = _finalize_from_staging(entries, staging_dir, output_dir)
        print(f"Edit CSV/C3D: wrote {len(written)} file(s) to {output_dir}")
        for path in written:
            print(f"  - {path}")
        if os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)
    else:
        print("Edit CSV/C3D: closed without saving. No files written.")
        if os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)
        if os.path.isdir(output_dir) and not os.listdir(output_dir):
            with contextlib.suppress(OSError):
                os.rmdir(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Edit CSV/C3D: apply rearrange_data column edits to .csv and .c3d "
            "files in a directory, round-tripping .c3d through CSV without "
            "touching the source files."
        )
    )
    parser.add_argument("-i", "--input", dest="input_dir", help="Input directory (.csv/.c3d)")
    parser.add_argument("-o", "--output", dest="output_dir", help="Output directory")
    parser.add_argument(
        "--identity",
        action="store_true",
        help="Headless: round-trip every file with no column changes.",
    )
    parser.add_argument(
        "--columns",
        help="Headless: comma-separated header names to keep/reorder (CSV and C3D markers).",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Headless: scan subdirectories too (equivalent to --depth -1 unless --depth is set).",
    )
    parser.add_argument(
        "-d",
        "--depth",
        type=int,
        default=None,
        help="Headless: recursion depth. -1 unlimited, 0 input dir only (default), N levels down.",
    )
    args = parser.parse_args()

    if not args.input_dir:
        run_edit_csv_c3d()
        return

    if args.depth is not None:
        max_depth = args.depth
    elif args.recursive:
        max_depth = -1
    else:
        max_depth = 0
    if max_depth < -1 or max_depth > 99:
        parser.error("--depth must be -1 (unlimited) or between 0 and 99")

    if args.identity or args.columns:
        output_dir = args.output_dir or _default_output_dir(args.input_dir)
        columns = [c.strip() for c in args.columns.split(",")] if args.columns else None
        written = _headless_process(args.input_dir, output_dir, columns, max_depth)
        print(f"Edit CSV/C3D: wrote {len(written)} file(s) to {output_dir}")
        for path in written:
            print(f"  - {path}")
        return

    run_edit_csv_c3d(preset_input_dir=args.input_dir, preset_output_dir=args.output_dir)


if __name__ == "__main__":
    main()
