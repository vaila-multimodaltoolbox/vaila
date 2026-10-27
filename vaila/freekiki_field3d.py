"""
================================================================================
Script: freekiki_field3d.py - field keypoints, poses and ball in pitch metres
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
    Puts the 3D outputs of a FreeKiki run in the pitch frame of ``kiki49``: origin
    at the centre, +x towards the right goal, +y towards the far touchline and +z
    up, in metres. Every function is pure (arrays in, rows out); the files are
    written by :mod:`vaila.freekiki_detect3d`.

    * :func:`dlt_for_frames` - frame -> 11 DLT coefficients of a ``.dlt3d`` file;
    * :func:`project_points` - world points -> pixels with one frame's DLT3D;
    * :func:`pixel_to_plane` - pixel -> world (x, y) on the plane ``z = const``;
    * :func:`field_keypoints_3d` - the 49 schema points per frame with the
      reprojection error against the detected pixels;
    * :func:`ball_3d` - ball centre on the pitch, its radius taken as its height;
    * :func:`pose_to_pitch` - one SAM 3D Body pose in pitch metres: the camera
      points are rotated into the field frame by the DLT3D, then the feet are
      moved onto the ground point under the bounding box.

    Why the pose needs both steps: ``sam3dinov3`` returns ``xcam_m`` in the camera
    frame of the DLT (checked: the points project onto the person's box through
    the DLT intrinsics). Its ``x_m`` is only that frame shifted by the camera
    translation, so the field orientation is missing. The depth of each person is
    also estimated by the body model, so the feet can float; anchoring them to the
    box base fixes both.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:
    from .freekiki_players import PITCH_HALF_LENGTH_M, PITCH_HALF_WIDTH_M, feet_point
    from .monocular_dlt_align import decompose_dlt3d, dlt_projection_matrix, load_dlt3d_file
except ImportError:
    from freekiki_players import (  # ty: ignore[unresolved-import]
        PITCH_HALF_LENGTH_M,
        PITCH_HALF_WIDTH_M,
        feet_point,
    )
    from monocular_dlt_align import (  # ty: ignore[unresolved-import]
        decompose_dlt3d,
        dlt_projection_matrix,
        load_dlt3d_file,
    )

BALL_RADIUS_M = 0.11  # a size 5 ball is about 22 cm across; its centre is this high on the ground
FEET_NAMES = frozenset(
    {
        "left-ankle",
        "right-ankle",
        "left-heel",
        "right-heel",
        "left-big-toe-tip",
        "right-big-toe-tip",
        "left-small-toe-tip",
        "right-small-toe-tip",
    }
)
FIELD_KP_COLUMNS = [
    "frame",
    "point",
    "name",
    "x_m",
    "y_m",
    "z_m",
    "u_px",
    "v_px",
    "det_u_px",
    "det_v_px",
    "err_px",
]
BALL_3D_COLUMNS = ["frame", "cx", "cy", "score", "x_m", "y_m", "z_m", "on_field"]


def dlt_for_frames(path: Path | str | None) -> dict[int, np.ndarray]:
    """Frame -> the 11 DLT coefficients of a ``.dlt3d`` file, rows with NaN left out."""
    if path is None:
        return {}
    frames, coeffs = load_dlt3d_file(path)
    out: dict[int, np.ndarray] = {}
    for frame, L in zip(frames, coeffs, strict=True):
        if np.isfinite(frame) and np.isfinite(L).all():
            out[int(frame)] = L
    return out


def project_points(L: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """``(n, 2)`` pixels of the world points ``xyz`` ``(n, 3)`` seen through DLT ``L``.

    A point whose homogeneous scale is not positive is behind the camera: NaN.
    """
    P = dlt_projection_matrix(L)
    homog = np.column_stack((np.asarray(xyz, dtype=np.float64), np.ones(len(xyz)))) @ P.T
    w = homog[:, 2]
    uv = np.full((len(xyz), 2), np.nan)
    ok = w > 1e-12
    uv[ok] = homog[ok, :2] / w[ok, None]
    return uv


def pixel_to_plane(L: np.ndarray, uv, z: float) -> tuple[float, float] | None:
    """World ``(x, y)`` of pixel ``uv`` on the horizontal plane ``Z = z``; None if degenerate."""
    P = dlt_projection_matrix(L)
    u, v = float(uv[0]), float(uv[1])
    # u * (P[2] . X) = P[0] . X and v * (P[2] . X) = P[1] . X, with X = (x, y, z, 1).
    A = np.array(
        [
            [u * P[2, 0] - P[0, 0], u * P[2, 1] - P[0, 1]],
            [v * P[2, 0] - P[1, 0], v * P[2, 1] - P[1, 1]],
        ]
    )
    b = np.array(
        [
            P[0, 2] * z + P[0, 3] - u * (P[2, 2] * z + P[2, 3]),
            P[1, 2] * z + P[1, 3] - v * (P[2, 2] * z + P[2, 3]),
        ]
    )
    if not np.isfinite(A).all() or abs(np.linalg.det(A)) < 1e-12:
        return None
    x, y = np.linalg.solve(A, b)
    return float(x), float(y)


def field_keypoints_3d(
    names: list[str],
    xyz: np.ndarray,
    cameras: dict[int, np.ndarray],
    pixels: dict[int, np.ndarray],
) -> list[dict]:
    """The schema points of every frame with a camera, with their reprojection error.

    ``pixels[frame]`` is the ``(K, 2)`` detected field keypoints (NaN where not
    accepted); the error is NaN for a point that was not detected.
    """
    rows: list[dict] = []
    for frame in sorted(cameras):
        uv = project_points(cameras[frame], xyz)
        det = pixels.get(frame)
        for i, name in enumerate(names):
            du, dv = det[i] if det is not None else (np.nan, np.nan)
            rows.append(
                {
                    "frame": frame,
                    "point": i,
                    "name": name,
                    "x_m": xyz[i, 0],
                    "y_m": xyz[i, 1],
                    "z_m": xyz[i, 2],
                    "u_px": uv[i, 0],
                    "v_px": uv[i, 1],
                    "det_u_px": du,
                    "det_v_px": dv,
                    "err_px": float(np.hypot(uv[i, 0] - du, uv[i, 1] - dv)),
                }
            )
    return rows


def ball_3d(
    ball_rows: list[dict],
    cameras: dict[int, np.ndarray],
    *,
    radius_m: float = BALL_RADIUS_M,
) -> list[dict]:
    """The ball of every row in pitch metres, placed at height ``radius_m``.

    ``on_field`` is True when that ground point is inside the pitch rectangle.
    A frame without a camera or without a ball keeps empty coordinates.
    """
    rows: list[dict] = []
    for row in ball_rows:
        frame = int(row["frame"])
        out = {
            "frame": frame,
            "cx": row["cx"],
            "cy": row["cy"],
            "score": row["score"],
            "x_m": np.nan,
            "y_m": np.nan,
            "z_m": np.nan,
            "on_field": False,
        }
        L = cameras.get(frame)
        if L is not None and row["cx"] != "":
            xy = pixel_to_plane(L, (float(row["cx"]), float(row["cy"])), radius_m)
            if xy is not None:
                x, y = xy
                out.update(
                    x_m=x,
                    y_m=y,
                    z_m=radius_m,
                    on_field=abs(x) <= PITCH_HALF_LENGTH_M and abs(y) <= PITCH_HALF_WIDTH_M,
                )
        rows.append(out)
    return rows


def pitch_ball_candidates(
    candidates: list[list[tuple[float, np.ndarray]]],
    cameras: dict[int, np.ndarray],
    *,
    radius_m: float = BALL_RADIUS_M,
) -> list[list[tuple[float, float, float, np.ndarray]]]:
    """Per frame, the ``(score, cx, cy, box)`` candidates whose ground point is on the pitch.

    The ball sits ``radius_m`` above the ground, so each box centre is lifted to that
    plane through the frame's DLT3D; the candidate counts when that point is inside the
    pitch rectangle. A frame without a camera has no candidates. The stands and the spare
    balls off the pitch are dropped here, before the track is chosen.
    """
    out: list[list[tuple[float, float, float, np.ndarray]]] = []
    for frame, detections in enumerate(candidates):
        kept: list[tuple[float, float, float, np.ndarray]] = []
        L = cameras.get(frame)
        if L is not None:
            for score, box in detections:
                x1, y1, x2, y2 = (float(v) for v in box)
                cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                xy = pixel_to_plane(L, (cx, cy), radius_m)
                if xy is None:
                    continue
                if abs(xy[0]) <= PITCH_HALF_LENGTH_M and abs(xy[1]) <= PITCH_HALF_WIDTH_M:
                    kept.append((score, cx, cy, box))
        out.append(kept)
    return out


def box_ground(L: np.ndarray, bbox_xyxy) -> tuple[float, float] | None:
    """The pitch point under a box's bottom centre, from the frame's DLT3D (None if degenerate)."""
    xy = pixel_to_plane(L, feet_point(bbox_xyxy), 0.0)
    return None if xy is None else (float(xy[0]), float(xy[1]))


def pose_to_pitch(
    kpt_names: list[str],
    xcam: np.ndarray,
    L: np.ndarray,
    bbox_xyxy,
    *,
    ground_xy: tuple[float, float] | None = None,
) -> np.ndarray | None:
    """One SAM 3D Body pose ``(K, 3)`` in pitch metres, or None without an anchor.

    ``xcam`` are the camera-frame points of the ``sam3dinov3`` CSV, ``L`` the DLT3D of
    the frame, and ``bbox_xyxy`` the person's box in that frame. The camera points
    are rotated and moved by the DLT pose, then shifted so the mean of the feet
    lands on the ground point under the box base (z = 0). ``ground_xy`` replaces that
    per-frame ground point, e.g. with a smoothed track (see :func:`freekiki_physics.smooth_track`).
    """
    cam = decompose_dlt3d(L)
    world = (cam.R.T @ np.asarray(xcam, dtype=np.float64).T).T + cam.C
    feet = [
        i for i, name in enumerate(kpt_names) if name in FEET_NAMES and np.isfinite(world[i]).all()
    ]
    if not feet:
        return None
    ground = ground_xy if ground_xy is not None else box_ground(L, bbox_xyxy)
    if ground is None:
        return None
    foot = world[feet].mean(axis=0)
    shift = np.array([ground[0] - foot[0], ground[1] - foot[1], -foot[2]])
    return world + shift


def write_table(path: Path | str, columns: list[str], rows: list[dict]) -> Path:
    """Write ``rows`` as CSV with ``columns`` in order; NaN and None become empty cells."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            cells = {}
            for key in columns:
                value = row.get(key, "")
                if isinstance(value, (float, np.floating)):
                    cells[key] = "" if not np.isfinite(value) else round(float(value), 6)
                elif value is None:
                    cells[key] = ""
                else:
                    cells[key] = value
            writer.writerow(cells)
    return path
