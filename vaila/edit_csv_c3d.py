"""
===============================================================================
edit_csv_c3d.py
===============================================================================
Author: Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 24 August 2026
Update Date: 24 August 2026
Version: 0.3.111

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

Notes:
- `--columns` keeps/reorders headers by exact name (mirrors `rearrange_data.
  reshapedata`'s filtering). For `.c3d`-derived data, `auto_create_c3d_from_csv`
  derives marker labels from complete `LABEL_X/Y/Z` triples in column order,
  so a `--columns` list touching C3D markers must keep whole X/Y/Z triples
  together and in order or the round-tripped C3D will be malformed.
"""

import argparse
import glob
import os
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


def _list_input_files(input_dir: str) -> list[str]:
    """Sorted `.csv`/`.c3d` file names in `input_dir`, hidden files ignored."""
    names = []
    for name in sorted(os.listdir(input_dir)):
        if name.startswith("."):
            continue
        full = os.path.join(input_dir, name)
        if not os.path.isfile(full):
            continue
        if name.lower().endswith((".csv", ".c3d")):
            names.append(name)
    return names


def _select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Keep/reorder columns by exact header name (existing ones only)."""
    existing = [c for c in columns if c in df.columns]
    return df[existing]


def _stage_inputs(input_dir: str, staging_dir: str) -> dict:
    """Convert every `.c3d` to a staged marker CSV, copy every `.csv` as-is.

    Returns `{stem: entry}` where `entry["kind"]` is "csv" or "c3d",
    `entry["staged_name"]` is the file name inside `staging_dir`, and (for
    "c3d") `entry["meta"]` carries the round-trip metadata from
    `c3d_markers_to_dataframe`.
    """
    os.makedirs(staging_dir, exist_ok=True)
    entries: dict = {}
    for name in _list_input_files(input_dir):
        source = os.path.join(input_dir, name)
        stem, ext = os.path.splitext(name)
        ext = ext.lower()
        staged_name = f"{stem}.csv"
        dest = os.path.join(staging_dir, staged_name)
        if ext == ".csv":
            shutil.copyfile(source, dest)
            entries[stem] = {"kind": "csv", "staged_name": staged_name, "source": source}
        elif ext == ".c3d":
            markers_df, meta = c3d_markers_to_dataframe(source)
            markers_df.to_csv(dest, index=False)
            entries[stem] = {
                "kind": "c3d",
                "staged_name": staged_name,
                "source": source,
                "meta": meta,
            }
    return entries


def _newest_edited_csv(rearranged_dir: str, stem: str) -> str | None:
    """Most recently written `data_rearranged/{stem}_*.csv`, if any."""
    if not os.path.isdir(rearranged_dir):
        return None
    candidates = glob.glob(os.path.join(rearranged_dir, f"{stem}_*.csv"))
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _write_output(stem: str, entry: dict, df: pd.DataFrame, output_dir: str) -> str:
    """Write one entry's (possibly edited) DataFrame to `output_dir`."""
    if entry["kind"] == "csv":
        out_path = os.path.join(output_dir, f"{stem}_final.csv")
        df.to_csv(out_path, index=False)
        return out_path

    meta = entry["meta"]
    out_path = os.path.join(output_dir, f"{stem}.c3d")
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
    """After the GUI editor closes: pick each stem's edited CSV and write it out."""
    os.makedirs(output_dir, exist_ok=True)
    rearranged_dir = os.path.join(staging_dir, "data_rearranged")
    written = []
    for stem, entry in entries.items():
        edited = _newest_edited_csv(rearranged_dir, stem)
        source_csv = edited or os.path.join(staging_dir, entry["staged_name"])
        df = pd.read_csv(source_csv)
        written.append(_write_output(stem, entry, df, output_dir))
    return written


def _headless_process(input_dir: str, output_dir: str, columns: list[str] | None) -> list[str]:
    """No Tk, no GUI editor: read, optionally filter/reorder columns, write."""
    os.makedirs(output_dir, exist_ok=True)
    written = []
    for name in _list_input_files(input_dir):
        source = os.path.join(input_dir, name)
        stem, ext = os.path.splitext(name)
        ext = ext.lower()
        if ext == ".csv":
            df = pd.read_csv(source)
            entry = {"kind": "csv"}
        else:
            df, meta = c3d_markers_to_dataframe(source)
            entry = {"kind": "c3d", "meta": meta}
        if columns:
            df = _select_columns(df, columns)
        written.append(_write_output(stem, entry, df, output_dir))
    return written


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

    file_names = _list_input_files(input_dir)
    if not file_names:
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
    entries = _stage_inputs(input_dir, staging_dir)

    staged_names = sorted(entry["staged_name"] for entry in entries.values())
    original_headers = get_headers(os.path.join(staging_dir, staged_names[0]))

    # ColumnReorderGUI subclasses tk.Tk directly and manages its own
    # mainloop()/destroy() lifecycle; the picker root above is already gone
    # by the time this runs, so this is the only live Tk root at this point.
    app = ColumnReorderGUI(original_headers, staged_names, staging_dir)
    app.mainloop()

    written = _finalize_from_staging(entries, staging_dir, output_dir)
    print(f"Edit CSV/C3D: wrote {len(written)} file(s) to {output_dir}")
    for path in written:
        print(f"  - {path}")


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
    args = parser.parse_args()

    if not args.input_dir:
        run_edit_csv_c3d()
        return

    if args.identity or args.columns:
        output_dir = args.output_dir or _default_output_dir(args.input_dir)
        columns = [c.strip() for c in args.columns.split(",")] if args.columns else None
        written = _headless_process(args.input_dir, output_dir, columns)
        print(f"Edit CSV/C3D: wrote {len(written)} file(s) to {output_dir}")
        for path in written:
            print(f"  - {path}")
        return

    run_edit_csv_c3d(preset_input_dir=args.input_dir, preset_output_dir=args.output_dir)


if __name__ == "__main__":
    main()
