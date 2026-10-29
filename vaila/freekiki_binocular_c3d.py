"""
================================================================================
Script: freekiki_binocular_c3d.py - C3D files of a two-camera run (field, people, ball)
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.12
Created: 28 October 2026
Update Date: 28 October 2026

Description:
    The C3D files of a two-camera run, in metres, with the same layout as the single-camera
    folders (:func:`freekiki_detect3d._c3d_groups`):

    * ``field_ball.c3d``: the 49 field points (``FKP_<name>``), each triangulated from the accepted
      2D points of the two cameras with the DLT3D of each frame, and the ball (``BALL``);
    * ``person_<id>_<role>.c3d``: one file per person paired between the two cameras, with the 70
      MHR keypoints triangulated in :mod:`freekiki_pose_binocular`.

    A point is kept only when both views reproduce it within ``MAX_KP_RESIDUAL_PX`` and the two rays
    are at least ``MIN_ANGLE_DEG`` apart; anything else is NaN, never a zero. A person seen in one
    camera only cannot be triangulated, so it has no file: ``README.txt`` lists it and the reason.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

try:
    from .freekiki_export import write_c3d
    from .freekiki_pose_binocular import MAX_KP_RESIDUAL_PX
    from .freekiki_stereo import capture_time, cut_frame_at, project
    from .monocular_dlt_align import dlt_projection_matrix
except ImportError:
    from freekiki_export import write_c3d  # ty: ignore[unresolved-import]
    from freekiki_pose_binocular import MAX_KP_RESIDUAL_PX  # ty: ignore[unresolved-import]
    from freekiki_stereo import (  # ty: ignore[unresolved-import]
        capture_time,
        cut_frame_at,
        project,
    )
    from monocular_dlt_align import dlt_projection_matrix  # ty: ignore[unresolved-import]

FIELD_PREFIX = "FKP_"
BALL_LABEL = "BALL"
FIELD_BALL = "field_ball.c3d"


def _field_names(field_csv: Path) -> dict[int, str]:
    """Point index -> field name, from the 3D run's ``field_kps_3d.csv``."""
    names: dict[int, str] = {}
    with field_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            names.setdefault(int(row["point"]), row["name"])
    return names


def _field_2d(path: Path, n_points: int) -> dict[int, dict[int, tuple[float, float]]]:
    """Frame -> {point index: (u, v)} of the accepted field points (empty cells are missing)."""
    out: dict[int, dict[int, tuple[float, float]]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pts: dict[int, tuple[float, float]] = {}
            for k in range(n_points):
                x, y = row.get(f"p{k}_x", ""), row.get(f"p{k}_y", "")
                if x and y:
                    pts[k] = (float(x), float(y))
            out[int(row["frame"])] = pts
    return out


def ground_triangulate(
    L_a: np.ndarray, L_b: np.ndarray, uv_a: tuple[float, float], uv_b: tuple[float, float]
) -> tuple[np.ndarray, float, float]:
    """A pitch point on the ground (z = 0) seen at ``uv_a`` and ``uv_b``: ``(xyz, res_a_px, res_b_px)``.

    The field points lie on the ground, so only (x, y) are unknown: each view gives two linear
    equations and the least-squares solution fits both. Free triangulation is not used for them,
    because a small angle between the rays puts a point metres off the ground.
    """
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for L, (u, v) in ((L_a, uv_a), (L_b, uv_b)):
        P = dlt_projection_matrix(L)
        A = P[:, [0, 1, 3]]  # homogeneous (x, y, 1) of the ground point
        for row, pixel in ((A[0], u), (A[1], v)):
            rows.append(row[:2] - pixel * A[2, :2])
            rhs.append(-(row[2] - pixel * A[2, 2]))
    xy, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
    xyz = np.array([xy[0], xy[1], 0.0])
    res_a = float(np.linalg.norm(project(dlt_projection_matrix(L_a), xyz) - np.asarray(uv_a)))
    res_b = float(np.linalg.norm(project(dlt_projection_matrix(L_b), xyz) - np.asarray(uv_b)))
    return xyz, res_a, res_b


def triangulate_field(
    ref_view, other_view, ref_2d: dict, other_2d: dict, names: dict[int, str], n: int
) -> tuple[np.ndarray, list[str], dict]:
    """``(n, points, 3)`` metres of the field points; the stats count the kept ones."""
    indices = sorted(names)
    labels = [f"{FIELD_PREFIX}{names[k]}" for k in indices]
    points = np.full((n, len(indices), 3), np.nan)
    kept = rejected = 0
    for f in range(n):
        cut = cut_frame_at(other_view, capture_time(ref_view, f))
        if cut is None:
            continue
        o = int(round(cut))
        L_a, L_b = ref_view.dlt.get(f), other_view.dlt.get(o)
        if L_a is None or L_b is None:
            continue
        uv_a, uv_b = ref_2d.get(f, {}), other_2d.get(o, {})
        for col, k in enumerate(indices):
            if k not in uv_a or k not in uv_b:
                continue
            xyz, res_a, res_b = ground_triangulate(L_a, L_b, uv_a[k], uv_b[k])
            if res_a <= MAX_KP_RESIDUAL_PX and res_b <= MAX_KP_RESIDUAL_PX:
                points[f, col] = xyz
                kept += 1
            else:
                rejected += 1
    return points, labels, {"kept": kept, "rejected": rejected}


def _ball_points(ball_csv: Path, n: int) -> np.ndarray:
    arr = np.full((n, 1, 3), np.nan)
    with ball_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            f = int(row["frame"])
            if 0 <= f < n and row["valid"] == "True" and row["x_m"] != "":
                arr[f, 0] = (float(row["x_m"]), float(row["y_m"]), float(row["z_m"]))
    return arr


def write_binocular_c3d(
    ref_view,
    other_view,
    *,
    data_dir: Path | str,
    ref_2d_csv: Path,
    other_2d_csv: Path,
    field_names_csv: Path,
    ball_csv: Path,
    fps: float,
) -> dict:
    """Writes ``<data_dir>/field_ball.c3d``: the field points (on the ground) and the ball.

    The people are not written here: each team folder gets its own ``.c3d`` files
    (:mod:`freekiki_team_analysis`).
    """
    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = int(np.asarray(ref_view.pts_cut).size)
    names = _field_names(field_names_csv)
    field, field_labels, stats = triangulate_field(
        ref_view,
        other_view,
        _field_2d(ref_2d_csv, max(names) + 1),
        _field_2d(other_2d_csv, max(names) + 1),
        names,
        n,
    )
    ball = _ball_points(ball_csv, n)
    path = write_c3d(
        out / FIELD_BALL,
        field_labels + [BALL_LABEL],
        np.concatenate([field, ball], axis=1),
        rate_hz=fps,
    )
    summary = {
        "frames": n,
        "rate_hz": fps,
        "units": "m",
        "field_kept": stats["kept"],
        "field_rejected": stats["rejected"],
        "ball_valid_frames": int(np.isfinite(ball[:, 0, 0]).sum()),
        "max_residual_px": MAX_KP_RESIDUAL_PX,
    }
    (out / "field_ball_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"out": str(path), "summary": summary}


def field_2d_csv(work: Path) -> Path:
    """The accepted field keypoints of a camera's detect run (``field_kps_getpixelvideo.csv``)."""
    found = sorted(Path(work).glob("freekiki_predict_*/field_kps_getpixelvideo.csv"))
    if not found:
        raise FileNotFoundError(f"no field_kps_getpixelvideo.csv under {work}")
    return found[-1]
