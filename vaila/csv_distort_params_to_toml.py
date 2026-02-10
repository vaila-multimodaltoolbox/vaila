"""
================================================================================
csv_distort_params_to_toml.py
================================================================================
vailá - Multimodal Toolbox
Author: Prof. Dr. Paulo R. P. Santiago
https://github.com/paulopreto/vaila-multimodaltoolbox
Date: 10 February 2026
Version: 0.0.1
Python Version: 3.12.12

Description:
------------
Converts distortpar_*.csv (distortion parameters) to TOML under a root directory.
Walks recursively, converts each distortpar_*.csv to distortpar_*.toml with the
same parameter values (fx, fy, cx, cy, k1, k2, k3, p1, p2), then removes the
original CSV. Use this to migrate calibration files to TOML for use with
vaila_datdistort, vaila_lensdistortvideo and vaila_distortvideo_gui.

How to use:
-----------
    From project root:
    uv run vaila/vaila/csv_distort_params_to_toml.py <root_dir>
    python vaila/vaila/csv_distort_params_to_toml.py <root_dir>

    Example:
    uv run vaila/vaila/csv_distort_params_to_toml.py /path/to/markerpixel_step1/

License:
--------
This program is licensed under the GNU Lesser General Public License v3.0.
For more details, visit: https://www.gnu.org/licenses/lgpl-3.0.html
================================================================================
"""

import csv
import os
import sys
from pathlib import Path

KEYS = ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2")


def read_params_csv(path):
    """Read first data row from a distortion parameters CSV. Returns dict of float or None."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline()
    sep = ";" if ";" in first and "," not in first else ","
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f, delimiter=sep)
        row = next(r, None)
    if not row:
        return None
    out = {}
    for k in KEYS:
        if k not in row:
            return None
        try:
            out[k] = float(row[k].strip())
        except (ValueError, TypeError):
            return None
    return out


def write_toml(path, params):
    """Write distortion parameters to a TOML file."""
    lines = [
        "# Camera distortion parameters (fx, fy, cx, cy, k1, k2, k3, p1, p2)",
        f"fx = {params['fx']}",
        f"fy = {params['fy']}",
        f"cx = {params['cx']}",
        f"cy = {params['cy']}",
        f"k1 = {params['k1']}",
        f"k2 = {params['k2']}",
        f"k3 = {params['k3']}",
        f"p1 = {params['p1']}",
        f"p2 = {params['p2']}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python vaila/vaila/csv_distort_params_to_toml.py <root_dir>")
        sys.exit(1)
    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        print(f"Error: not a directory: {root}")
        sys.exit(1)

    converted = 0
    errors = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if not name.startswith("distortpar_") or not name.lower().endswith(".csv"):
                continue
            csv_path = Path(dirpath) / name
            toml_path = csv_path.with_suffix(".toml")
            try:
                params = read_params_csv(csv_path)
                if params is None:
                    errors.append(f"{csv_path}: invalid or missing parameters")
                    continue
                write_toml(toml_path, params)
                csv_path.unlink()
                converted += 1
                print(f"OK {csv_path} -> {toml_path}")
            except Exception as e:
                errors.append(f"{csv_path}: {e}")

    print(f"\nConverted {converted} file(s) to TOML.")
    if errors:
        print("Errors:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
