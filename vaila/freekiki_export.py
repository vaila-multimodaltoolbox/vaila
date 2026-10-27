"""
================================================================================
Script: freekiki_export.py - C3D file of the FreeKiki 3D markers
================================================================================
vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.11
Created: 26 October 2026
Update Date: 26 October 2026

Description:
    Writes the 3D markers of a FreeKiki run (field keypoints, poses and ball, all in
    pitch metres) to a C3D file with ``ezc3d``. A marker missing in a frame is stored
    as NaN coordinates with residual -1, which is how C3D marks an invalid sample; a
    viewer then draws nothing for it.

    * :func:`write_c3d` - ``labels`` plus a ``(frames, markers, 3)`` array in NaN where
      a marker is missing, at ``rate_hz``.
    * :func:`write_c3d_csv` - the same markers as a CSV, one row per frame.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

INVALID_RESIDUAL = -1.0


def write_c3d(
    path: Path | str,
    labels: list[str],
    points: np.ndarray,
    *,
    rate_hz: float,
) -> Path:
    """Save ``points`` ``(frames, markers, 3)`` in metres as a C3D file at ``rate_hz``."""
    import ezc3d

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"points must be (frames, markers, 3), got {points.shape}")
    if points.shape[1] != len(labels):
        raise ValueError(f"{len(labels)} labels for {points.shape[1]} markers")
    if len(set(labels)) != len(labels):
        raise ValueError("marker labels must be unique")

    n_frames, n_markers, _ = points.shape
    valid = np.isfinite(points).all(axis=2)  # (frames, markers)
    # A missing sample stays NaN on all three axes: viewers skip a NaN point, while zeros
    # would be drawn as a marker at the origin.
    data = np.full((4, n_markers, n_frames), np.nan)
    data[:3] = points.transpose(2, 1, 0)
    data[3] = 1.0
    residuals = np.zeros((1, n_markers, n_frames))
    residuals[0] = np.where(valid.T, 0.0, INVALID_RESIDUAL)

    c3d = ezc3d.c3d()
    c3d["parameters"]["POINT"]["RATE"]["value"] = [float(rate_hz)]
    c3d["parameters"]["POINT"]["LABELS"]["value"] = list(labels)
    c3d["parameters"]["POINT"]["UNITS"]["value"] = ["m"]
    c3d["data"]["points"] = data
    c3d["data"]["meta_points"]["residuals"] = residuals

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    c3d.write(str(path))
    return path


def write_c3d_csv(
    path: Path | str,
    labels: list[str],
    points: np.ndarray,
    *,
    rate_hz: float,
) -> Path:
    """Save the same markers as :func:`write_c3d` to a CSV: one row per frame.

    Columns are ``frame, time_s`` and then ``<label>_X, <label>_Y, <label>_Z`` in metres.
    A missing sample is an empty cell, as a NaN in the C3D (which draws nothing).
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"points must be (frames, markers, 3), got {points.shape}")
    if points.shape[1] != len(labels):
        raise ValueError(f"{len(labels)} labels for {points.shape[1]} markers")
    header = ["frame", "time_s"]
    for label in labels:
        header += [f"{label}_X", f"{label}_Y", f"{label}_Z"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for f in range(points.shape[0]):
            cells: list[str] = [str(f), f"{f / rate_hz:.6f}"]
            for m in range(points.shape[1]):
                for c in range(3):
                    value = points[f, m, c]
                    cells.append(f"{value:.6f}" if np.isfinite(value) else "")
            writer.writerow(cells)
    return path
