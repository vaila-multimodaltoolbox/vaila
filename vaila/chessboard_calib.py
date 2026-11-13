"""
Project: vailá
Script: chessboard_calib.py
Authors: Paulo Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 06 November 2026
Update Date: 13 November 2026
Version: 0.4.18

Description:
    Multi-camera (or single-camera) calibration from chessboard videos, exported for
    vailá 3D reconstruction.

    One video per camera (same session, same chessboard). With a single video the
    multi-view sync / bundle-adjustment steps are skipped; the same
    ``<stem>.dlt3d`` and ``<stem>_intrinsics.toml`` files are still written for
    ``yolo_depth3d`` and ``sam3dinov3``. The pipeline:

    1. Detect the inner corners in every ``--step``-th frame. When the board is small, the
       detector retries on an upscaled crop around the last detection (``--zoom``). Corners
       are refined to sub-pixel accuracy and put in a fixed order (corner 0 at the top-left
       of an upright board, rows left to right), so the same physical corner has the same
       index in every camera even for boards that look the same after a 180-degree turn.
    2. Intrinsics per camera (``cv2.calibrateCameraExtended``). With ``--intrinsics auto``
       the lens model is restricted when the corners cover little of the image (the
       distortion at the image border cannot be measured from views near the centre).
       Precomputed intrinsics can be loaded instead (``--intrinsics-toml-dir``; matched by the
       camera name contained in the video name, so cut and equalized copies find their lens).
       ``board_tilt_p95_deg`` warns when the board kept one orientation (focal length and
       board distance cannot be separated then).
    3. Frame offsets between cameras (``--sync auto``): the board's angular speed does not
       depend on the camera, so its cross-correlation between cameras gives the lag, refined to
       a fraction of a frame. Manual ``--frame-offsets`` may have decimals. The fractions feed
       ``undistort --subframe`` and a printed Video Event Equalizer (RIFE) command.
    4. Extrinsics: board poses per camera (PnP), a camera graph for the initial poses, and a
       bundle adjustment of all camera and board poses with fixed intrinsics.
    5. World frame: the board at the frame seen by most cameras (``--world board``), or
       measured control points (``--world control``: a CSV of name,x,y,z in metres plus one
       vailá pixel CSV per camera with the same points).
    6. Export per camera: ``<stem>.dlt3d`` (Rec3D 1DLT / MultiDLT), ``<stem>_intrinsics.toml``
       (Distort Video format), ``<stem>_board_pixels.csv`` (detected corners in
       vailá pixel format), plus ``cameras.json``, ``calibration_report.json``, a README and
       coverage images. The validation triangulates the board corners with the exported DLTs
       after undistortion (``vaila.rec3d.rec3d_available_views``) and compares neighbouring
       corner distances with the square size. It uses the calibration frames themselves, so
       it is an in-sample consistency check, not an independent accuracy measurement.

    3D reconstruction afterwards: ``chessboard_calib.py undistort`` removes the lens
    distortion from the pixel CSVs in the camera matrix of the ``.dlt3d`` files (optionally
    shifting frames by the offsets) and prints the Rec3D 1DLT command. Distort CSV
    (``vaila_datdistort.py``) must not be used for this step: it reprojects with
    ``cv2.getOptimalNewCameraMatrix`` at a fixed 1920x1080, a different camera matrix.

Usage:
    uv run vaila/chessboard_calib.py                      # GUI
    uv run vaila/chessboard_calib.py -i videos/ --pattern 6x8 --square 0.098
    uv run vaila/chessboard_calib.py -i videos/ -o out/ --step 1 --zoom 2 --sync auto
    uv run vaila/chessboard_calib.py -i videos/ --world control \\
        --control-points floor_points.csv --control-pixels floor_pixels/
    uv run vaila/chessboard_calib.py undistort --calib out/processed_chessboard_calib_* \\
        -i pixels/ --apply-offsets          # or --subframe (offsets before rounding)

    On NVIDIA/CUDA workstations use ``uv run --no-sync`` so uv does not swap torch wheels.

License:
    GNU Affero General Public License v3.0 (AGPL-3.0).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

try:
    from .cli_highlight import print_gui_cli_mirror
    from .ffmpeg_utils import TimestampFrameCounter, numbering_fps, probe_frame_rates
    from .fifa_to_dlt import compute_dlt3d_from_KRt
    from .rec3d import rec3d_available_views
except ImportError:  # standalone execution: python vaila/chessboard_calib.py
    from cli_highlight import print_gui_cli_mirror  # ty: ignore[unresolved-import]
    from ffmpeg_utils import (  # ty: ignore[unresolved-import]
        TimestampFrameCounter,
        numbering_fps,
        probe_frame_rates,
    )
    from fifa_to_dlt import compute_dlt3d_from_KRt  # ty: ignore[unresolved-import]
    from rec3d import rec3d_available_views  # ty: ignore[unresolved-import]

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv")
DEFAULT_PATTERN = (6, 8)  # inner corners: points per row, points per column
DEFAULT_SQUARE_M = 0.098  # lab board: 98 mm squares (measure yours over several squares)
DETECT_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
# A lens model is only fully estimable when the corners cover a good part of the image.
AUTO_FULL_MODEL_COVERAGE = 0.30  # >= : k1, k2, p1, p2 and principal point free
AUTO_CONSTRAINED_COVERAGE = 0.05  # >= : f, k1, k2; below: f and k1 only
SYNC_MIN_CORR = 0.8  # pairwise lags with a weaker angular-speed correlation are ignored
MIN_VIEWS_INTRINSICS = 8
PNP_MAX_ERROR_PX = 5.0
# Board normal spread (p95, deg) below which one orientation dominates: the focal length cannot
# be told apart from the board distance (frontal static board: 7-8 deg; moved board: 54-69 deg).
STATIC_BOARD_TILT_DEG = 20.0
# Fraction of a frame above which the README offers a RIFE command that aligns the videos.
RIFE_MIN_FRACTION = 0.1
# Board upright in the image: world X along the rows, Z up (against the board's +y), Y = Z x X
UPRIGHT_AXES = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def log(msg: str) -> None:
    print(f">> vaila/chessboard_calib: {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Board geometry and corner detection
# --------------------------------------------------------------------------- #


def parse_pattern(text: str) -> tuple[int, int]:
    """'6x8' -> (6, 8): inner corners per row, inner corners per column."""
    parts = str(text).lower().replace(",", "x").replace("*", "x").split("x")
    if len(parts) != 2:
        raise ValueError(f"pattern must look like 6x8, got {text!r}")
    cols, rows = int(parts[0]), int(parts[1])
    if cols < 2 or rows < 2:
        raise ValueError(f"pattern needs at least 2x2 inner corners, got {text!r}")
    return cols, rows


def board_object_points(cols: int, rows: int, square_m: float) -> np.ndarray:
    """Board corners in metres, row by row (index = row * cols + col), z = 0."""
    obj = np.zeros((cols * rows, 3), np.float64)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square_m)
    return obj


def canonical_corner_order(corners: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """Reorder detected corners so corner 0 is top-left and rows run left to right.

    ``cv2.findChessboardCorners`` may start the numbering at either end of the board (a
    board with an even number of inner corners on both sides looks the same after a
    180-degree turn) and may traverse it mirrored. For an upright board seen from the front
    the four candidates are told apart by the image directions of its rows and columns.
    """
    g = np.asarray(corners, np.float64).reshape(rows, cols, 2)
    best, best_score = g, -np.inf
    for cand in (g, g[::-1, ::-1], g[:, ::-1], g[::-1, :]):
        u = (cand[:, -1] - cand[:, 0]).mean(axis=0)
        v = (cand[-1, :] - cand[0, :]).mean(axis=0)
        score = u[0] / (np.linalg.norm(u) + 1e-12) + v[1] / (np.linalg.norm(v) + 1e-12)
        if score > best_score:
            best, best_score = cand, score
    return best.reshape(-1, 2)


def _subpix_window(corners: np.ndarray, cols: int) -> int:
    g = corners.reshape(-1, cols, 2)
    spacing = float(np.median(np.linalg.norm(np.diff(g, axis=1), axis=2)))
    return int(np.clip(0.4 * spacing, 2, 11))


def detect_board(
    gray: np.ndarray,
    pattern: tuple[int, int],
    roi: tuple[int, int, int, int] | None = None,
    zoom: float = 2.0,
    full_frame: bool = True,
) -> np.ndarray | None:
    """Detect, refine and order the inner corners; ``None`` when the board is not found.

    With ``roi`` (x0, y0, x1, y1) the crop is upscaled by ``zoom`` and searched first; the
    corners are mapped back and refined on the full-resolution image.
    """
    h, w = gray.shape[:2]
    found, corners = False, None
    if roi is not None:
        x0, y0, x1, y1 = (int(round(v)) for v in roi)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 > 16 and y1 - y0 > 16:
            crop = gray[y0:y1, x0:x1]
            # Upscale only small crops; a large board does not need it.
            scale = zoom if zoom and zoom > 1.0 and (x1 - x0) * (y1 - y0) < 0.25 * w * h else 1.0
            if scale > 1.0:
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            found, c = cv2.findChessboardCorners(crop, pattern, flags=DETECT_FLAGS)
            if found:
                corners = c.reshape(-1, 2) / scale + (x0, y0)
    if not found and full_frame:
        found, c = cv2.findChessboardCorners(
            gray, pattern, flags=DETECT_FLAGS | cv2.CALIB_CB_FAST_CHECK
        )
        if found:
            corners = c.reshape(-1, 2)
    if not found or corners is None:
        return None
    win = _subpix_window(corners, pattern[0])
    refined = cv2.cornerSubPix(
        gray, corners.astype(np.float32).reshape(-1, 1, 2), (win, win), (-1, -1), SUBPIX_CRITERIA
    )
    return canonical_corner_order(refined.reshape(-1, 2), *pattern)


def _bbox(corners: np.ndarray, margin: float, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return (
        int(max(0, x0 - mx)),
        int(max(0, y0 - my)),
        int(min(w, x1 + mx)),
        int(min(h, y1 + my)),
    )


@dataclass
class CameraDetections:
    name: str
    path: str
    width: int
    height: int
    fps: float
    n_frames: int
    corners: dict[int, np.ndarray] = field(default_factory=dict)
    numbering: str = "timestamp"  # "timestamp" (round(t * fps)) or "decode" (no timestamps)
    dropped: list[int] = field(default_factory=list)  # frame numbers missing from the video


def detect_video(
    path: str | Path,
    pattern: tuple[int, int],
    step: int = 1,
    zoom: float = 2.0,
    search_every: int = 10,
    max_frames: int | None = None,
) -> CameraDetections:
    """Detect the board in one video, reading it once.

    While the board was seen recently, only an upscaled crop around it is searched (fast
    and better for small boards). Otherwise the full frame is searched every
    ``search_every`` frames.

    Frames are numbered by their timestamps (round(t * fps), first frame 0), the numbering
    of OpenCV's seek, getpixelvideo and a constant-frame-rate copy. Frames the camera
    dropped leave gaps in the numbers (``dropped``) instead of shifting every later frame,
    which also keeps the offsets between cameras constant.
    """
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, n = _frame_rate_and_span(path, cap)
    det = CameraDetections(path.stem, str(path), w, h, fps, n)
    step = max(1, int(step))
    search_every = max(step, int(search_every))
    last_corners: np.ndarray | None = None
    last_hit = -(10**9)
    counter = TimestampFrameCounter(fps)
    decoded = 0
    idx = -1
    while True:
        if max_frames is not None and decoded >= max_frames:
            break
        if not cap.grab():
            break
        decoded += 1
        idx = counter.next(cap.get(cv2.CAP_PROP_POS_MSEC))
        if idx % step == 0:
            recent = last_corners is not None and idx - last_hit <= 3 * search_every
            full = idx % search_every == 0
            if recent or full:
                ok, frame = cap.retrieve()
                if ok:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    roi = (
                        _bbox(last_corners, 0.6, w, h)
                        if recent and last_corners is not None
                        else None
                    )
                    c = detect_board(
                        gray, pattern, roi=roi, zoom=zoom, full_frame=full or roi is None
                    )
                    if c is not None:
                        det.corners[idx] = c
                        last_corners, last_hit = c, idx
    cap.release()
    det.n_frames = max(n, idx + 1)
    det.numbering = counter.mode
    det.dropped = list(counter.dropped)
    return det


def _frame_rate_and_span(path: Path, cap: cv2.VideoCapture) -> tuple[float, int]:
    """Numbering frame rate (nominal rate when only drops lower the average) and frame span."""
    cv_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    fps = numbering_fps(cv_fps, *probe_frame_rates(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, int(round(n * fps / cv_fps))


def _detect_worker(args: tuple) -> CameraDetections:
    return detect_video(*args)


FRAME_NUMBERING_RULE = (
    "video timestamps: frame = round(t * fps), first frame 0 (OpenCV seek, getpixelvideo, "
    "constant-frame-rate copy); frames the camera dropped are missing numbers"
)


def _restore_frame_numbering(prev: Path, dets: list[CameraDetections], log) -> None:
    """Copy the numbering and dropped frames of a reused run; warn about older runs."""
    try:
        old = json.loads((prev / "calibration_report.json").read_text(encoding="utf-8"))
        per_cam = old["frame_numbering"]["per_camera"]
    except (OSError, KeyError, TypeError, ValueError):
        log(
            "WARNING: the reused run does not record its frame numbering; runs before this "
            "version counted frames in decode order, which is off by the number of dropped "
            "frames after each drop. Run the detection again (without --reuse-detections)."
        )
        return
    for d in dets:
        info = per_cam.get(d.name, {})
        d.numbering = str(info.get("numbering", d.numbering))
        d.dropped = [int(f) for f in info.get("dropped_frames", [])]


def load_detections(video: Path, pixels_csv: Path, n_corners: int) -> CameraDetections:
    """Rebuild detections from a previous run's ``<stem>_board_pixels.csv``."""
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, n = _frame_rate_and_span(video, cap)
    cap.release()
    df = pd.read_csv(pixels_csv)
    cols = [f"p{k + 1}_{a}" for k in range(n_corners) for a in ("x", "y")]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{pixels_csv}: missing columns {missing[:4]}... (pattern mismatch?)")
    det = CameraDetections(video.stem, str(video), w, h, fps, n)
    for f, row in zip(df["frame"].astype(int), df[cols].to_numpy(np.float64), strict=True):
        det.corners[int(f)] = row.reshape(-1, 2)
    return det


# --------------------------------------------------------------------------- #
# Intrinsics
# --------------------------------------------------------------------------- #


def corner_coverage(points: np.ndarray, width: int, height: int) -> float:
    """Fraction of the image inside the convex hull of all detected corners."""
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32).reshape(-1, 1, 2))
    return float(cv2.contourArea(hull)) / float(width * height)


INTRINSIC_MODES = ("auto", "full", "standard", "constrained", "minimal")


def _intrinsic_flags(mode: str) -> int:
    fixed_center = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K3
    )
    return {
        "full": 0,
        "standard": cv2.CALIB_FIX_K3,
        "constrained": fixed_center,
        "minimal": fixed_center | cv2.CALIB_FIX_K2,
    }[mode]


def calibrate_intrinsics(
    obj: np.ndarray,
    corners: dict[int, np.ndarray],
    image_size: tuple[int, int],
    mode: str = "auto",
    max_views: int = 80,
) -> dict[str, Any]:
    """Calibrate one camera; drops views whose error is far above the median and refits."""
    frames = sorted(corners)
    if len(frames) < MIN_VIEWS_INTRINSICS:
        raise ValueError(f"only {len(frames)} board views; need >= {MIN_VIEWS_INTRINSICS}")
    if len(frames) > max_views:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, max_views).astype(int)]
    w, h = image_size
    coverage = corner_coverage(np.vstack([corners[f] for f in frames]), w, h)
    if mode == "auto":
        if coverage >= AUTO_FULL_MODEL_COVERAGE:
            mode = "standard"
        elif coverage >= AUTO_CONSTRAINED_COVERAGE:
            mode = "constrained"
        else:
            mode = "minimal"
    flags = _intrinsic_flags(mode)

    def fit(sel: list[int]):
        objp = [obj.astype(np.float32)] * len(sel)
        imgp = [corners[f].astype(np.float32).reshape(-1, 1, 2) for f in sel]
        K0 = cv2.initCameraMatrix2D(objp, imgp, (w, h))
        if flags & cv2.CALIB_USE_INTRINSIC_GUESS:
            f0 = float(np.sqrt(K0[0, 0] * K0[1, 1]))
            K0 = np.array([[f0, 0, (w - 1) / 2.0], [0, f0, (h - 1) / 2.0], [0, 0, 1]], np.float64)
        return cv2.calibrateCameraExtended(objp, imgp, (w, h), K0, np.zeros(5), flags=flags)

    rms, K, dist, _r, _t, std_int, _std_ext, per_view = fit(frames)
    errs = np.asarray(per_view).ravel()
    keep = [f for f, e in zip(frames, errs, strict=True) if e <= max(3.0 * np.median(errs), 1.0)]
    dropped = len(frames) - len(keep)
    if dropped and len(keep) >= MIN_VIEWS_INTRINSICS:
        frames = keep
        rms, K, dist, _r, _t, std_int, _std_ext, per_view = fit(frames)
    dist = np.asarray(dist, np.float64).ravel()[:5]
    std = np.asarray(std_int, np.float64).ravel()
    names = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")
    return {
        "K": np.asarray(K, np.float64),
        "dist": dist,
        "rms_px": float(rms),
        "mode": mode,
        "n_views": len(frames),
        "views_dropped": int(dropped),
        "coverage": coverage,
        "std": {n: float(s) for n, s in zip(names, std[: len(names)], strict=False)},
        "source": "chessboard",
    }


def load_intrinsics_toml(path: Path) -> dict[str, Any]:
    """Read fx, fy, cx, cy, k1, k2, k3, p1, p2 (the Distort Video / Distort CSV TOML format)."""
    import tomllib

    with open(path, "rb") as f:
        data = tomllib.load(f)
    flat: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    K = np.array(
        [[flat["fx"], 0.0, flat["cx"]], [0.0, flat["fy"], flat["cy"]], [0.0, 0.0, 1.0]], np.float64
    )
    dist = np.array([flat.get(k, 0.0) for k in ("k1", "k2", "p1", "p2", "k3")], np.float64)
    return {
        "K": K,
        "dist": dist,
        "rms_px": float("nan"),
        "mode": "loaded",
        "n_views": 0,
        "views_dropped": 0,
        "coverage": float("nan"),
        "std": {},
        "source": str(path),
    }


def _match_camera(file_stem: str, cameras: list[str]) -> str | None:
    """Camera whose name starts ``file_stem``, else the one contained in it (longest wins).

    Containment lets a cut copy (``<camera>_sync_..._frames_...``) or an equalized copy
    (``source_2_<camera>_..._equalized``) use the files of the original camera.
    """
    hits = [c for c in cameras if file_stem.startswith(c)] or [c for c in cameras if c in file_stem]
    return max(hits, key=len) if hits else None


def find_intrinsics_toml(toml_dir: Path, video_name: str) -> Path | None:
    """``<camera>_intrinsics.toml`` in ``toml_dir`` for ``video_name`` (exact name first)."""
    suffix = "_intrinsics.toml"
    exact = toml_dir / f"{video_name}{suffix}"
    if exact.is_file():
        return exact
    names = {p.name[: -len(suffix)]: p for p in toml_dir.glob(f"*{suffix}")}
    cam = _match_camera(video_name, list(names))
    return names[cam] if cam is not None else None


def board_tilt_p95_deg(rotations: list[np.ndarray]) -> float:
    """95th percentile of the angle (deg) between each view's board normal and the mean normal.

    A board that keeps one orientation (small value) does not separate the focal length from
    the board distance, so the lens is unreliable away from the board.
    """
    if len(rotations) < 2:
        return float("nan")
    normals = np.array([np.asarray(R, np.float64)[:, 2] for R in rotations])
    mean = normals.mean(axis=0)
    mean /= np.linalg.norm(mean)
    angles = np.degrees(np.arccos(np.clip(normals @ mean, -1.0, 1.0)))
    return float(np.percentile(angles, 95))


# --------------------------------------------------------------------------- #
# Poses, synchronization and bundle adjustment
# --------------------------------------------------------------------------- #


def _to_mat4(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, np.float64).ravel()
    return T


def _rot_angle(R: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


def board_pose(obj: np.ndarray, img: np.ndarray, K: np.ndarray, dist: np.ndarray):
    """Board -> camera pose by planar PnP; returns (R, t, rms_px) or None."""
    ok, rvec, tvec = cv2.solvePnP(obj, img.reshape(-1, 1, 2), K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img.reshape(-1, 1, 2), K, dist, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    if tvec.ravel()[2] <= 0:
        return None
    return cv2.Rodrigues(rvec)[0], tvec.ravel(), rms


def rife_alignment_command(
    paths: list[str],
    counts: list[int],
    exact: list[float],
    ref: int,
    output_dir: Path,
) -> list[str] | None:
    """Event-equalizer command that puts every camera on the reference instants (RIFE).

    Reference frame n (1-based) is at camera position n + e (camera frame = reference frame +
    e, ``exact``). Events A,A,B on the reference and A+e,A+e,B+e on each camera shift that
    camera by e; the equalizer generates the in-between images with RIFE. A and B keep every
    position inside every video. None when no camera is a sizeable fraction of a frame off.
    """
    exact = [round(float(e), 3) for e in exact]
    if all(abs(e - round(e)) <= RIFE_MIN_FRACTION for e in exact):
        return None
    first = max([1, *(math.ceil(1 - e) for e in exact)])
    last = min(math.floor(n - e) for n, e in zip(counts, exact, strict=True))
    if last - first < 2:
        return None

    def events(e: float) -> str:
        if abs(e - round(e)) < 1e-9:
            a, b = first + round(e), last + round(e)
            return f"{a},{a},{b}"
        return f"{first + e:.3f},{first + e:.3f},{last + e:.3f}"

    cmd = ["uv", "run", "--no-sync", "vaila/video_event_equalizer.py"]
    cmd += ["--reference", str(paths[ref]), "--reference-events", events(exact[ref])]
    for k, (path, e) in enumerate(zip(paths, exact, strict=True)):
        if k != ref:
            cmd += ["--source", str(path), "--source-events", events(e)]
    return [*cmd, "--output-dir", str(output_dir)]


def angular_speed_series(
    rotations: dict[int, np.ndarray], n_frames: int, lag: int = 3
) -> np.ndarray:
    """Rotation angle of the board between frames f and f+lag (rad); NaN where missing.

    The relative rotation R(f+lag) R(f)^T is the board's rotation expressed in the camera
    frame, so its angle is the same in every camera.
    """
    out = np.full(n_frames, np.nan)
    for f, R in rotations.items():
        R2 = rotations.get(f + lag)
        if R2 is not None and 0 <= f < n_frames:
            out[f] = _rot_angle(R2 @ R.T)
    return out


def pairwise_lag(a: np.ndarray, b: np.ndarray, max_lag: int, min_overlap: int = 30):
    """Lag L maximizing corr(a[f], b[f+L]); returns (L, corr, overlap) or None.

    L is refined to a fraction of a frame with a parabola through the correlation at the
    best integer lag and its two neighbours: cameras that are not genuinely synchronized
    are offset by fractions of a frame, and rounding every pair separately would make
    the pairs disagree with each other.
    """
    best = None
    corr: dict[int, float] = {}
    n = min(len(a), len(b))
    for L in range(-max_lag, max_lag + 1):
        i0, i1 = max(0, -L), min(n, n - L)
        if i1 - i0 < min_overlap:
            continue
        x, y = a[i0:i1], b[i0 + L : i1 + L]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < min_overlap or np.std(x[m]) < 1e-9 or np.std(y[m]) < 1e-9:
            continue
        r = float(np.corrcoef(x[m], y[m])[0, 1])
        corr[L] = r
        if best is None or r > best[1]:
            best = (L, r, int(m.sum()))
    if best is None:
        return None
    L, r, m = best
    lo, hi = corr.get(L - 1), corr.get(L + 1)
    frac = 0.0
    if lo is not None and hi is not None and lo - 2 * r + hi < 0:
        frac = float(np.clip(0.5 * (lo - hi) / (lo - 2 * r + hi), -0.5, 0.5))
    return (L + frac, r, m)


def solve_offsets(
    n_cams: int, ref: int, lags: dict[tuple[int, int], tuple[float, float, int]]
) -> tuple[list[int], list[int], list[float]]:
    """Frame offsets o (o_ref = 0) from pairwise o_j - o_i = L_ij by weighted least squares.

    Returns the integer offsets, the cameras without a lag, and the offsets before rounding
    (fractions of a frame that a whole-frame shift cannot remove)."""
    rows, rhs, wts = [], [], []
    for (i, j), (L, r, m) in lags.items():
        row = np.zeros(n_cams)
        row[j], row[i] = 1.0, -1.0
        rows.append(row)
        rhs.append(L)
        wts.append(max(r, 1e-3) * np.sqrt(m))
    anchor = np.zeros(n_cams)
    anchor[ref] = 1.0
    rows.append(anchor)
    rhs.append(0.0)
    wts.append(1e3)
    A = np.asarray(rows) * np.asarray(wts)[:, None]
    b = np.asarray(rhs) * np.asarray(wts)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    linked = {ref} | {i for i, _ in lags} | {j for _, j in lags}
    exact = [float(v - sol[ref]) if k in linked else 0.0 for k, v in enumerate(sol)]
    return (
        [int(round(v)) for v in exact],
        sorted(set(range(n_cams)) - linked),
        exact,
    )


def _average_rotation(Rs: list[np.ndarray]) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def initial_camera_poses(
    poses: list[dict[int, tuple[np.ndarray, np.ndarray]]], ref: int
) -> list[np.ndarray | None]:
    """Camera poses (4x4, reference camera -> camera) from shared board views (BFS tree)."""
    n = len(poses)
    shared: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for f in set(poses[i]) & set(poses[j]):
                Ti = _to_mat4(*poses[i][f])
                Tj = _to_mat4(*poses[j][f])
                shared[(i, j)].append(Tj @ np.linalg.inv(Ti))  # camera i -> camera j
    T: list[np.ndarray | None] = [None] * n
    T[ref] = np.eye(4)
    queue = deque([ref])
    while queue:
        i = queue.popleft()
        Ti = T[i]
        if Ti is None:
            continue
        cands = sorted(
            (j for j in range(n) if T[j] is None and shared.get((i, j))),
            key=lambda j: -len(shared[(i, j)]),
        )
        for j in cands:
            rel = shared[(i, j)]
            Rs = [m[:3, :3] for m in rel]
            R_med = _average_rotation(Rs)
            inl = [m for m in rel if _rot_angle(m[:3, :3] @ R_med.T) < np.radians(5.0)] or rel
            Tij = _to_mat4(
                _average_rotation([m[:3, :3] for m in inl]),
                np.median([m[:3, 3] for m in inl], axis=0),
            )
            T[j] = Tij @ Ti
            queue.append(j)
    return T


def bundle_adjust(
    obj: np.ndarray,
    observations: list[tuple[int, int, np.ndarray]],
    intrinsics: list[dict[str, Any]],
    cam_T: list[np.ndarray],
    board_T: dict[int, np.ndarray],
    ref: int,
    max_nfev: int = 200,
) -> tuple[list[np.ndarray], dict[int, np.ndarray], dict[str, Any]]:
    """Refine camera and board poses with fixed intrinsics.

    Plain least squares (a Huber loss converged ~100x slower on real data and ended with a
    higher error), then one refit without the board views whose error is far above the
    median (> max(3 x median, 2 px)).
    """
    cam_out, board_out, info = _bundle_adjust_once(
        obj, observations, intrinsics, cam_T, board_T, ref, max_nfev
    )
    per_obs = info["per_observation_rms_px"]
    limit = max(3.0 * float(np.median(per_obs)), 2.0)
    keep = [o for o, e in zip(observations, per_obs, strict=True) if e <= limit]
    rejected = len(observations) - len(keep)
    if rejected:
        kept_frames = {f for _c, f, _i in keep}
        cam_out, board_kept, info = _bundle_adjust_once(
            obj,
            keep,
            intrinsics,
            cam_out,
            {f: T for f, T in board_out.items() if f in kept_frames},
            ref,
            max_nfev,
        )
        board_out = board_kept
    info["views_rejected"] = int(rejected)
    info["reject_limit_px"] = float(limit)
    return cam_out, board_out, info


def _bundle_adjust_once(
    obj: np.ndarray,
    observations: list[tuple[int, int, np.ndarray]],
    intrinsics: list[dict[str, Any]],
    cam_T: list[np.ndarray],
    board_T: dict[int, np.ndarray],
    ref: int,
    max_nfev: int,
) -> tuple[list[np.ndarray], dict[int, np.ndarray], dict[str, Any]]:
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    from scipy.spatial.transform import Rotation

    n_cams = len(cam_T)
    cam_ids = [c for c in range(n_cams) if c != ref]
    cam_col = {c: 6 * k for k, c in enumerate(cam_ids)}
    frames = sorted(board_T)
    frame_col = {f: 6 * (len(cam_ids) + k) for k, f in enumerate(frames)}

    def pack(Tm: np.ndarray) -> np.ndarray:
        return np.concatenate([cv2.Rodrigues(Tm[:3, :3])[0].ravel(), Tm[:3, 3]])

    def unpack(p: np.ndarray) -> np.ndarray:
        return _to_mat4(cv2.Rodrigues(p[:3])[0], p[3:6])

    x0 = np.concatenate([pack(cam_T[c]) for c in cam_ids] + [pack(board_T[f]) for f in frames])
    n_pts = len(obj)
    frame_index = {f: k for k, f in enumerate(frames)}
    o_cam = np.array([c for c, _f, _i in observations])
    o_frame = np.array([frame_index[f] for _c, f, _i in observations])
    o_img = np.stack([img for _c, _f, img in observations]).astype(np.float64)  # (O, N, 2)
    K_all = np.stack([intrinsics[c]["K"] for c in range(n_cams)])
    D_all = np.stack([np.asarray(intrinsics[c]["dist"], np.float64)[:5] for c in range(n_cams)])
    fx, fy = K_all[o_cam, 0, 0][:, None], K_all[o_cam, 1, 1][:, None]
    cx, cy = K_all[o_cam, 0, 2][:, None], K_all[o_cam, 1, 2][:, None]
    k1, k2, p1, p2, k3 = (D_all[o_cam, i][:, None] for i in range(5))
    obj64 = np.asarray(obj, np.float64)

    def residuals(x: np.ndarray) -> np.ndarray:
        """OpenCV pinhole + (k1, k2, p1, p2, k3) projection of every view at once."""
        cam_p = np.zeros((n_cams, 6))
        cam_p[cam_ids] = x[: 6 * len(cam_ids)].reshape(-1, 6)
        board_p = x[6 * len(cam_ids) :].reshape(-1, 6)
        Rc = Rotation.from_rotvec(cam_p[o_cam, :3]).as_matrix()
        Rb = Rotation.from_rotvec(board_p[o_frame, :3]).as_matrix()
        Xr = np.einsum("oij,nj->oni", Rb, obj64) + board_p[o_frame, None, 3:]
        Xc = np.einsum("oij,onj->oni", Rc, Xr) + cam_p[o_cam, None, 3:]
        z = np.where(np.abs(Xc[..., 2]) < 1e-9, 1e-9, Xc[..., 2])
        xn, yn = Xc[..., 0] / z, Xc[..., 1] / z
        r2 = xn * xn + yn * yn
        radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
        yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
        proj = np.stack([fx * xd + cx, fy * yd + cy], axis=-1)
        return (proj - o_img).ravel()

    sparsity = lil_matrix((2 * n_pts * len(observations), len(x0)), dtype=int)
    for k, (c, f, _img) in enumerate(observations):
        r0, r1 = 2 * n_pts * k, 2 * n_pts * (k + 1)
        if c != ref:
            sparsity[r0:r1, cam_col[c] : cam_col[c] + 6] = 1
        sparsity[r0:r1, frame_col[f] : frame_col[f] + 6] = 1
    res = least_squares(
        residuals,
        x0,
        jac_sparsity=sparsity,
        loss="linear",
        x_scale="jac",
        method="trf",
        ftol=1e-6,
        xtol=1e-6,
        max_nfev=max_nfev,
    )
    cam_out = [
        np.eye(4) if c == ref else unpack(res.x[cam_col[c] : cam_col[c] + 6]) for c in range(n_cams)
    ]
    board_out = {f: unpack(res.x[frame_col[f] : frame_col[f] + 6]) for f in frames}
    r = res.fun.reshape(-1, n_pts, 2)
    per_obs = np.sqrt(np.mean(np.sum(r**2, axis=2), axis=1))
    per_cam = {}
    for c in range(n_cams):
        sel = [k for k, (cc, _f, _i) in enumerate(observations) if cc == c]
        if sel:
            per_cam[c] = float(np.sqrt(np.mean(np.sum(r[sel] ** 2, axis=2))))
    info = {
        "rms_px": float(np.sqrt(np.mean(np.sum(r**2, axis=2)))),
        "per_camera_rms_px": per_cam,
        "per_observation_rms_px": per_obs,
        "n_observations": len(observations),
        "n_frames": len(frames),
        "status": int(res.status),
        "nfev": int(res.nfev),
    }
    return cam_out, board_out, info


# --------------------------------------------------------------------------- #
# World frame, export and validation
# --------------------------------------------------------------------------- #


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = False):
    """Similarity (s, R, t) minimizing ||dst - (s R src + t)||; src, dst are (N, 3)."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / np.mean(np.sum(xs**2, axis=1))) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def undistort_pixels(pts: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Distorted pixels -> ideal pinhole pixels of the same camera (NaN rows kept)."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    out = np.full_like(pts, np.nan)
    ok = np.isfinite(pts).all(axis=1)
    if ok.any():
        und = cv2.undistortPoints(pts[ok].reshape(-1, 1, 2), K, dist, P=K)
        out[ok] = und.reshape(-1, 2)
    return out


def triangulate(
    dlts: list[np.ndarray | None], pixels: list[np.ndarray | None]
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate N points from undistorted pixels; returns (N, 3) xyz and (N,) camera counts."""
    n_pts = next(len(p) for p in pixels if p is not None)
    xyz = np.full((n_pts, 3), np.nan)
    nv = np.zeros(n_pts, int)
    for k in range(n_pts):
        # rec3d_available_views skips non-finite pixels; a camera without a view gets NaN.
        pix = [(np.nan, np.nan) if p is None else tuple(p[k]) for p in pixels]
        xyz[k], nv[k] = rec3d_available_views(dlts, pix)
    return xyz, nv


def board_validation(
    xyz: np.ndarray, cols: int, rows: int, square_m: float
) -> dict[str, float] | None:
    """Neighbour-corner distances and planarity of one reconstructed board."""
    g = xyz.reshape(rows, cols, 3)
    d = np.concatenate(
        [
            np.linalg.norm(np.diff(g, axis=1), axis=2).ravel(),
            np.linalg.norm(np.diff(g, axis=0), axis=2).ravel(),
        ]
    )
    d = d[np.isfinite(d)]
    if not len(d):
        return None
    pts = xyz[np.isfinite(xyz).all(axis=1)]
    plan = float("nan")
    if len(pts) >= 4:
        c = pts - pts.mean(axis=0)
        plan = float(np.sqrt(np.mean((c @ np.linalg.svd(c)[2][-1]) ** 2)))
    err = d - square_m
    return {
        "n_distances": int(len(d)),
        "mean_error_m": float(err.mean()),
        "rmse_m": float(np.sqrt(np.mean(err**2))),
        "max_abs_error_m": float(np.abs(err).max()),
        "planarity_rms_m": plan,
    }


def write_intrinsics_toml(path: Path, intr: dict[str, Any], image_size: tuple[int, int]) -> None:
    K, d = intr["K"], intr["dist"]
    lines = [
        "# Camera intrinsics from vaila/chessboard_calib.py",
        "# TOML format of Distort Video (vaila_lensdistortvideo.py). For Rec3D use chessboard_calib.py undistort.",
        f"# image_size = {image_size[0]} x {image_size[1]}; model = {intr['mode']}; "
        f"rms = {intr['rms_px']:.4f} px; coverage = {intr['coverage'] * 100:.2f} % of the image",
        f"fx = {K[0, 0]:.10g}",
        f"fy = {K[1, 1]:.10g}",
        f"cx = {K[0, 2]:.10g}",
        f"cy = {K[1, 2]:.10g}",
        f"k1 = {d[0]:.10g}",
        f"k2 = {d[1]:.10g}",
        f"k3 = {d[4]:.10g}",
        f"p1 = {d[2]:.10g}",
        f"p2 = {d[3]:.10g}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dlt3d(path: Path, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    dlt = compute_dlt3d_from_KRt(K, R, t)
    cols = ["frame"] + [f"p{j + 1}" for j in range(11)]
    pd.DataFrame([[0, *dlt.tolist()]], columns=pd.Index(cols)).to_csv(path, index=False)
    return dlt


def write_board_pixels(path: Path, corners: dict[int, np.ndarray]) -> None:
    """Detected (distorted) corners in vailá pixel format: frame, p1_x, p1_y, ..."""
    frames = sorted(corners)
    n = len(next(iter(corners.values()))) if corners else 0
    cols = ["frame"] + [f"p{k + 1}_{a}" for k in range(n) for a in ("x", "y")]
    rows = [[f, *corners[f].ravel().tolist()] for f in frames]
    pd.DataFrame(rows, columns=pd.Index(cols)).to_csv(path, index=False, float_format="%.4f")


def save_coverage_image(path: Path, video: str, corners: dict[int, np.ndarray]) -> None:
    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    cap.release()
    if not ok or not corners:
        return
    pts = np.vstack(list(corners.values()))
    for x, y in pts[:: max(1, len(pts) // 4000)]:
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
    hull = cv2.convexHull(pts.astype(np.float32).reshape(-1, 1, 2)).astype(np.int32)
    cv2.polylines(frame, [hull], True, (0, 0, 255), 3)
    label = f"{len(corners)} board views; red = region where the calibration is measured"
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.rectangle(frame, (10, 10), (30 + tw, 30 + th + base), (0, 0, 0), -1)
    cv2.putText(frame, label, (20, 20 + th), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(str(path), frame)


def read_control_points(points_csv: Path) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(points_csv)
    lower = {c.lower().strip(): c for c in df.columns}
    for need in ("x", "y", "z"):
        if need not in lower:
            raise ValueError(f"{points_csv}: needs columns name,x,y,z (metres)")
    names = (
        df[lower["name"]].astype(str).tolist()
        if "name" in lower
        else [f"p{k + 1}" for k in range(len(df))]
    )
    return names, df[[lower["x"], lower["y"], lower["z"]]].to_numpy(np.float64)


def read_control_pixels(pixels_csv: Path, n_points: int) -> np.ndarray:
    """First row of a vailá pixel CSV (frame, p1_x, p1_y, ...) -> (n_points, 2); NaN if absent."""
    df = pd.read_csv(pixels_csv)
    row = df.iloc[0]
    out = np.full((n_points, 2), np.nan)
    for k in range(n_points):
        for a, axis in enumerate(("x", "y")):
            col = f"p{k + 1}_{axis}"
            if col in df.columns:
                out[k, a] = float(row[col])
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def find_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.iterdir() if p.suffix.lower() in VIDEO_EXTS and p.is_file())


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_ready(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _finish_monocular_calibration(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    obj: np.ndarray,
    cols: int,
    rows: int,
    square: float,
    d: CameraDetections,
    ci: dict[str, Any],
    poses: dict[int, tuple[np.ndarray, np.ndarray]],
    errs: dict[int, float],
) -> dict[str, Any]:
    """Export the same DLT/TOML artifacts from one camera (PnP board world, no BA)."""
    if getattr(args, "world", "board") == "control":
        raise ValueError(
            "--world control needs at least two cameras; use --world board for a single video"
        )
    if not poses:
        raise RuntimeError(
            f"{d.name}: no usable board PnP poses (all views above {PNP_MAX_ERROR_PX} px)"
        )

    # World = board at the lowest-reprojection PnP frame (camera away from origin).
    f_world = min(poses, key=lambda f: errs.get(f, float("inf")))
    R_bc, t_bc = poses[f_world]
    board_T = _to_mat4(R_bc, t_bc)  # board -> camera
    M = UPRIGHT_AXES if args.board_axes == "upright" else np.eye(3)
    # Same framing as multi-cam with a single reference camera (cam_T = I):
    # T_world_ref = M @ inv(board_T); cams_world = inv(T_world_ref) = board_T @ inv(M)
    Tw = board_T @ np.linalg.inv(_to_mat4(M, np.zeros(3)))
    R, t = Tw[:3, :3], Tw[:3, 3]
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt

    dlt = write_dlt3d(out_dir / f"{d.name}.dlt3d", ci["K"], R, t)
    write_intrinsics_toml(out_dir / f"{d.name}_intrinsics.toml", ci, (d.width, d.height))
    write_board_pixels(out_dir / f"{d.name}_board_pixels.csv", d.corners)
    save_coverage_image(out_dir / "qa" / f"{d.name}_coverage.png", d.path, d.corners)
    center = (-R.T @ t).ravel()

    # Validation: PnP reprojection + DLT reprojects undistorted corners at f_world.
    err_vals = np.array(list(errs.values()), dtype=float) if errs else np.array([])
    und = undistort_pixels(d.corners[f_world], ci["K"], ci["dist"])
    # Ideal pinhole projection of object points in the upright/board world.
    obj_w = (M @ obj.T).T  # board object -> world
    ones = np.ones((len(obj_w), 1))
    Xh = np.hstack([obj_w, ones])
    P = np.vstack([ci["K"] @ np.column_stack([R, t.ravel()]), [0.0, 0.0, 0.0, 1.0]])
    proj = (P[:3] @ Xh.T).T
    proj_uv = proj[:, :2] / proj[:, 2:3]
    reproj = float(np.sqrt(np.mean(np.sum((proj_uv - und) ** 2, axis=1))))

    # Neighbour lengths from PnP camera-frame 3D (sanity; should match square size).
    per_frame_val: list[dict[str, float]] = []
    for _f, (Rb, tb) in poses.items():
        xyz_cam = (Rb @ obj.T).T + tb.ravel()
        v = board_validation(xyz_cam, cols, rows, square)
        if v is not None:
            per_frame_val.append(v)
    dist_errs = np.array([p["rmse_m"] for p in per_frame_val]) if per_frame_val else np.array([])

    cameras_json = {
        d.name: {
            "video": d.path,
            "image_size": [d.width, d.height],
            "fps": d.fps,
            "K": ci["K"],
            "dist_k1_k2_p1_p2_k3": ci["dist"],
            "R_world_to_camera": R,
            "t_world_to_camera_m": t,
            "camera_center_world_m": center,
            "frame_offset": 0,
            "intrinsics": {k: v for k, v in ci.items() if k not in ("K", "dist")},
            "board_views": len(d.corners),
            "ba_rms_px": None,
            "pnp_world_frame": int(f_world),
            "dlt_reproj_rms_px": reproj,
        }
    }
    sync_report: dict[str, Any] = {
        "mode": "monocular",
        "frame_offsets": {d.name: 0},
    }
    world_report: dict[str, Any] = {
        "mode": "board",
        "board_frame": int(f_world),
        "board_axes": args.board_axes,
        "origin": "board corner 0 (top-left inner corner of the upright board)",
        "cameras": 1,
    }
    ba_report: dict[str, Any] = {
        "mode": "monocular_pnp",
        "status": 1,
        "rms_px": float(np.median(err_vals)) if len(err_vals) else float("nan"),
        "nfev": 0,
        "views_rejected": 0,
        "reject_limit_px": PNP_MAX_ERROR_PX,
        "per_camera_rms_px": {0: float(np.median(err_vals)) if len(err_vals) else None},
        "world_frame_pnp_rms_px": float(errs.get(f_world, float("nan"))),
        "dlt_reproj_rms_px": reproj,
    }
    validation: dict[str, Any] = {
        "kind": (
            "monocular: PnP reprojection and neighbour-corner lengths from board poses "
            "(no multi-view triangulation)"
        ),
        "frames": int(len(per_frame_val)),
        "square_m": square,
        "pnp_rms_median_px": float(np.median(err_vals)) if len(err_vals) else None,
        "pnp_rms_p95_px": float(np.percentile(err_vals, 95)) if len(err_vals) else None,
        "dlt_reproj_rms_px": reproj,
        "distance_rmse_median_m": float(np.median(dist_errs)) if len(dist_errs) else None,
        "distance_rmse_p95_m": float(np.percentile(dist_errs, 95)) if len(dist_errs) else None,
        "distance_mean_error_median_m": (
            float(np.median([p["mean_error_m"] for p in per_frame_val])) if per_frame_val else None
        ),
        "planarity_rms_median_m": (
            float(np.nanmedian([p["planarity_rms_m"] for p in per_frame_val]))
            if per_frame_val
            else None
        ),
        "holdout": {"enabled": False, "skipped": "monocular calibration has no holdout blocks"},
    }
    report: dict[str, Any] = {
        "schema": "vaila_chessboard_calib_v1",
        "created": datetime.now().isoformat(timespec="seconds"),
        "input": str(args.input),
        "pattern_inner_corners": [cols, rows],
        "square_m": square,
        "step": args.step,
        "zoom": args.zoom,
        "intrinsics_mode": args.intrinsics,
        "mode": "monocular",
        "reference_camera_for_ba": d.name,
        "frame_numbering": {
            "rule": FRAME_NUMBERING_RULE,
            "per_camera": {
                d.name: {"numbering": d.numbering, "dropped_frames": d.dropped},
            },
        },
        "sync": sync_report,
        "world": world_report,
        "bundle_adjustment": ba_report,
        "validation": validation,
        "cameras": cameras_json,
        "pnp_rms_median_px": {
            d.name: float(np.median(list(errs.values()))) if errs else None,
        },
        "dlt_coefficients": dlt.tolist(),
    }
    (out_dir / "cameras.json").write_text(
        json.dumps(_json_ready(cameras_json), indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "calibration_report.json").write_text(
        json.dumps(_json_ready(report), indent=2) + "\n", encoding="utf-8"
    )
    _write_readme(out_dir, report, [d.name])
    log(
        f"monocular: world board frame {f_world}, PnP rms median "
        f"{ba_report['rms_px']:.3f} px, DLT reproj {reproj:.3f} px"
    )
    if len(dist_errs):
        log(
            "validation (monocular): neighbour-corner distance error median "
            f"{np.median(dist_errs) * 1000:.2f} mm "
            f"(p95 {np.percentile(dist_errs, 95) * 1000:.2f} mm) for {square * 1000:.0f} mm squares"
        )
    log(f"done -> {out_dir}")
    report["output_dir"] = str(out_dir)
    return report


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    pattern = parse_pattern(args.pattern)
    cols, rows = pattern
    square = float(args.square)
    obj = board_object_points(cols, rows, square)
    videos = find_videos(Path(args.input))
    if not videos:
        raise ValueError(f"no camera videos found in {args.input}")
    out_parent = (
        Path(args.output)
        if args.output
        else (Path(args.input) if Path(args.input).is_dir() else Path(args.input).parent)
    )
    out_dir = out_parent / f"processed_chessboard_calib_{datetime.now():%Y%m%d_%H%M%S}"
    (out_dir / "qa").mkdir(parents=True, exist_ok=True)
    mode_label = "monocular" if len(videos) == 1 else f"{len(videos)} cameras"
    log(f"{mode_label}; pattern {cols}x{rows} inner corners; square {square} m -> {out_dir}")

    # 1. detection (one process per video)
    jobs = [
        (str(v), pattern, args.step, args.zoom, args.search_every, args.max_frames) for v in videos
    ]
    workers = max(1, min(len(jobs), int(args.workers or os.cpu_count() or 1)))
    if args.reuse_detections:
        prev = Path(args.reuse_detections)
        dets = [
            load_detections(v, prev / f"{v.stem}_board_pixels.csv", cols * rows) for v in videos
        ]
        log(f"reusing detections from {prev}")
        _restore_frame_numbering(prev, dets, log)
    elif workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            dets = list(pool.map(_detect_worker, jobs))
    else:
        dets = [_detect_worker(j) for j in jobs]
    for d in dets:
        msg = f"{d.name}: board in {len(d.corners)} of {d.n_frames} frames (step {args.step})"
        if d.dropped:
            shown = ", ".join(str(f) for f in d.dropped[:8]) + (
                ", ..." if len(d.dropped) > 8 else ""
            )
            msg += f"; {len(d.dropped)} frame(s) missing from the video timestamps ({shown})"
        if d.numbering != "timestamp":
            msg += "; no usable timestamps, frames counted in decode order"
        log(msg)
    have_toml = bool(args.intrinsics_toml_dir)
    dropped_cams = [d.name for d in dets if len(d.corners) < MIN_VIEWS_INTRINSICS and not have_toml]
    if dropped_cams:
        log(
            f"WARNING: fewer than {MIN_VIEWS_INTRINSICS} board views, camera left out: {dropped_cams}"
        )
        dets = [d for d in dets if d.name not in dropped_cams]
    if not dets:
        raise RuntimeError("no camera saw the board often enough for intrinsics")
    monocular = len(dets) == 1
    if monocular:
        log("one camera: monocular path (intrinsics + PnP board world; no sync/BA)")

    # 2. intrinsics
    intr: list[dict[str, Any]] = []
    toml_dir = Path(args.intrinsics_toml_dir) if args.intrinsics_toml_dir else None
    intr_dets = dets
    if args.intrinsics_input:
        ivids = {p.stem: p for p in find_videos(Path(args.intrinsics_input))}
        ijobs = [
            (
                str(ivids.get(d.name, d.path)),
                pattern,
                args.step,
                args.zoom,
                args.search_every,
                args.max_frames,
            )
            for d in dets
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            intr_dets = list(pool.map(_detect_worker, ijobs))
    for d, di in zip(dets, intr_dets, strict=True):
        toml = find_intrinsics_toml(toml_dir, d.name) if toml_dir else None
        if toml is not None:
            ci = load_intrinsics_toml(toml)
            log(f"{d.name}: lens loaded from {toml}")
        else:
            if toml_dir is not None:
                log(
                    f"WARNING {d.name}: no <camera>_intrinsics.toml in {toml_dir} matches this "
                    "video name; the lens is estimated from its own board views"
                )
            ci = calibrate_intrinsics(obj, di.corners, (di.width, di.height), mode=args.intrinsics)
        intr.append(ci)
        cov = ci["coverage"]
        log(
            f"{d.name}: intrinsics {ci['mode']} f={ci['K'][0, 0]:.1f}px "
            f"k1={ci['dist'][0]:.4f} k2={ci['dist'][1]:.4f} "
            f"rms={'n/a' if not np.isfinite(ci['rms_px']) else f'{ci["rms_px"]:.3f}px'} "
            f"coverage={'n/a' if not np.isfinite(cov) else f'{cov * 100:.1f}%'}"
        )
        if np.isfinite(cov) and cov < AUTO_FULL_MODEL_COVERAGE:
            log(
                f"WARNING {d.name}: the board covers {cov * 100:.1f}% of the image; lens distortion "
                "outside that region is extrapolated (see qa/*_coverage.png)."
            )

    # 3. board poses per camera
    pnp: list[dict[int, tuple[np.ndarray, np.ndarray]]] = []
    pnp_err: list[dict[int, float]] = []
    for d, ci in zip(dets, intr, strict=True):
        poses, errs = {}, {}
        for f, c in d.corners.items():
            res = board_pose(obj, c, ci["K"], ci["dist"])
            if res is not None and res[2] <= PNP_MAX_ERROR_PX:
                poses[f], errs[f] = (res[0], res[1]), res[2]
        pnp.append(poses)
        pnp_err.append(errs)
        tilt = board_tilt_p95_deg([R for R, _t in poses.values()])
        ci["board_tilt_p95_deg"] = tilt
        if ci["mode"] != "loaded" and np.isfinite(tilt) and tilt < STATIC_BOARD_TILT_DEG:
            log(
                f"WARNING {d.name}: the board kept almost one orientation ({tilt:.0f} deg, p95), "
                "so the focal length cannot be told apart from the board distance; 3D points "
                "away from the board can be off by centimetres. Tilt the board 30-45 deg in "
                "several directions, or load the lens of a calibration where it moved: "
                "--intrinsics-toml-dir <processed_chessboard_calib_* folder>."
            )

    if monocular:
        return _finish_monocular_calibration(
            args=args,
            out_dir=out_dir,
            obj=obj,
            cols=cols,
            rows=rows,
            square=square,
            d=dets[0],
            ci=intr[0],
            poses=pnp[0],
            errs=pnp_err[0],
        )

    # 4. frame offsets (multi-camera)
    n_cams = len(dets)
    n_frames = max(d.n_frames for d in dets)
    sync_report: dict[str, Any] = {"mode": args.sync}
    exact_offsets: list[float] | None = None  # offsets before rounding, when known
    rife_ref = 0
    if args.frame_offsets:
        manual = [float(v) for v in str(args.frame_offsets).split(",")]
        if len(manual) != n_cams:
            raise ValueError(
                f"--frame-offsets needs {n_cams} values (camera order: {[d.name for d in dets]})"
            )
        offsets = [math.floor(e + 0.5) for e in manual]
        sync_report["mode"] = "manual"
        if any(abs(e - round(e)) > 1e-9 for e in manual):
            # decimals, e.g. measured where the board moved: kept for --subframe and RIFE
            exact_offsets = manual
            rife_ref = int(np.argmin(np.abs(manual)))
            sync_report["frame_offsets_subframe"] = {
                d.name: round(e, 3) for d, e in zip(dets, manual, strict=True)
            }
    elif args.sync == "auto" and args.step == 1:
        series = [
            angular_speed_series({f: p[0] for f, p in cams.items()}, n_frames) for cams in pnp
        ]
        lags = {}
        for i in range(n_cams):
            for j in range(i + 1, n_cams):
                best = pairwise_lag(series[i], series[j], args.max_lag)
                if best is not None and best[1] >= SYNC_MIN_CORR:
                    lags[(i, j)] = best
        ref_sync = int(np.argmax([len(p) for p in pnp]))
        offsets, unlinked, exact = solve_offsets(n_cams, ref_sync, lags)
        sync_report.update(
            {
                "reference_camera": dets[ref_sync].name,
                "pairwise": {
                    f"{dets[i].name}|{dets[j].name}": {"lag": L, "corr": r, "overlap": m}
                    for (i, j), (L, r, m) in lags.items()
                },
                "unlinked_cameras": [dets[k].name for k in unlinked],
                "frame_offsets_subframe": {
                    d.name: round(e, 3) for d, e in zip(dets, exact, strict=True)
                },
            }
        )
        if unlinked:
            log(f"WARNING: no reliable lag for {[dets[k].name for k in unlinked]}; offset 0 used")
        exact_offsets, rife_ref = list(exact), ref_sync
    else:
        if args.sync == "auto":
            log("auto sync needs --step 1; using offset 0 for every camera")
        offsets = [0] * n_cams
    rife = (
        rife_alignment_command(
            [d.path for d in dets],
            [d.n_frames for d in dets],
            exact_offsets,
            rife_ref,
            Path(args.input),
        )
        if exact_offsets is not None
        else None
    )
    if rife is not None and any(d.dropped for d in dets):
        sync_report["rife_note"] = (
            "cameras dropped frames: make constant-frame-rate copies before aligning the "
            "videos with the event equalizer, which numbers frames in reading order"
        )
    elif rife is not None:
        sync_report["rife_equalizer_command"] = shlex.join(rife)
        log(
            "optional: align the videos below one frame with RIFE (CUDA; generated images "
            "are estimates):\n>>   " + shlex.join(rife)
        )
    sync_report["frame_offsets"] = {d.name: o for d, o in zip(dets, offsets, strict=True)}
    log(
        "frame offsets (camera frame = reference frame + offset): "
        + ", ".join(f"{d.name}={o:+d}" for d, o in zip(dets, offsets, strict=True))
    )

    # 5. extrinsics: synchronized frames -> camera graph -> bundle adjustment
    sync_pnp = [{f - o: p for f, p in cams.items()} for cams, o in zip(pnp, offsets, strict=True)]
    shared_counts = [
        sum(len(set(sync_pnp[i]) & set(sync_pnp[j])) for j in range(n_cams) if j != i)
        for i in range(n_cams)
    ]
    ref = int(np.argmax(shared_counts))
    cam_T0 = initial_camera_poses(sync_pnp, ref)
    missing = [dets[k].name for k, T in enumerate(cam_T0) if T is None]
    if missing:
        raise RuntimeError(f"cameras without a shared board view with the others: {missing}")
    cam_T0_ok: list[np.ndarray] = [T for T in cam_T0 if T is not None]
    board_T0: dict[int, np.ndarray] = {}
    observations: list[tuple[int, int, np.ndarray]] = []
    for c, d in enumerate(dets):
        for f, p in sync_pnp[c].items():
            if f not in board_T0:
                board_T0[f] = np.linalg.inv(cam_T0_ok[c]) @ _to_mat4(*p)
            observations.append((c, f, d.corners[f + offsets[c]]))
    views_per_frame: dict[int, int] = defaultdict(int)
    for _c, f, _img in observations:
        views_per_frame[f] += 1
    multi = sorted(f for f, k in views_per_frame.items() if k >= 2)
    if len(multi) < 3:
        raise RuntimeError("fewer than 3 frames with the board seen by two or more cameras")
    observations_all = [o for o in observations if views_per_frame[o[1]] >= 2]
    ba_frames = set(multi)
    if len(multi) > args.max_ba_frames:
        ba_frames = {
            multi[i] for i in np.linspace(0, len(multi) - 1, args.max_ba_frames).astype(int)
        }
    observations = [o for o in observations_all if o[1] in ba_frames]
    board_T0_all = dict(board_T0)
    board_T0 = {f: T for f, T in board_T0.items() if f in ba_frames}
    log(
        f"bundle adjustment: {n_cams} cameras, {len(ba_frames)} of {len(multi)} multi-camera frames, "
        f"{len(observations)} board views"
    )
    t0 = time.perf_counter()
    cam_T, board_T, ba = bundle_adjust(
        obj, observations, intr, cam_T0_ok, board_T0, ref, max_nfev=args.max_nfev
    )
    state = "converged" if ba["status"] > 0 else "stopped at --max-nfev"
    log(
        f"bundle adjustment rms {ba['rms_px']:.3f} px ({state}, {ba['nfev']} evaluations, "
        f"{ba['views_rejected']} views rejected above {ba['reject_limit_px']:.2f} px, "
        f"{time.perf_counter() - t0:.1f} s)"
    )

    # 6. world frame
    counts = defaultdict(int)
    for _c, f, _i in observations:
        counts[f] += 1
    f_world = max(sorted(counts), key=lambda f: counts[f])
    M = UPRIGHT_AXES if args.board_axes == "upright" else np.eye(3)
    T_world_ref = _to_mat4(M, np.zeros(3)) @ np.linalg.inv(board_T[f_world])  # ref camera -> world
    world_report: dict[str, Any] = {
        "mode": "board",
        "board_frame": int(f_world),
        "board_axes": args.board_axes,
        "origin": "board corner 0 (top-left inner corner of the upright board)",
    }
    cams_world = [Tc @ np.linalg.inv(T_world_ref) for Tc in cam_T]  # world -> camera
    if args.world == "control":
        names, pts_world = read_control_points(Path(args.control_points))
        pix_dir = Path(args.control_pixels)
        dl, px = [], []
        for d, ci, Tc in zip(dets, intr, cams_world, strict=True):
            files = sorted(pix_dir.glob(f"{d.name}*.csv"))
            if not files:
                dl.append(None)
                px.append(None)
                continue
            raw = read_control_pixels(files[0], len(names))
            px.append(undistort_pixels(raw, ci["K"], ci["dist"]))
            dl.append(compute_dlt3d_from_KRt(ci["K"], Tc[:3, :3], Tc[:3, 3]))
        if sum(p is not None for p in px) < 2:
            raise ValueError(f"control pixels for at least two cameras are needed in {pix_dir}")
        xyz_old, nv = triangulate(dl, px)
        ok = np.isfinite(xyz_old).all(axis=1)
        if ok.sum() < 3:
            raise ValueError("fewer than 3 control points were seen by two or more cameras")
        s, R_u, t_u = umeyama(xyz_old[ok], pts_world[ok], with_scale=args.control_scale)
        T_new_old = np.eye(4)
        T_new_old[:3, :3] = s * R_u
        T_new_old[:3, 3] = t_u
        resid = np.linalg.norm((s * (R_u @ xyz_old[ok].T)).T + t_u - pts_world[ok], axis=1)
        # world_new = s R world_old + t  ->  camera = Tc_old @ inv(T_new_old)
        S_inv = np.linalg.inv(T_new_old)
        cams_world = []
        for Tc in [Tc @ np.linalg.inv(T_world_ref) for Tc in cam_T]:
            A = Tc @ S_inv  # 3x3 block = R_c R_u^T / s
            cams_world.append(A)
        world_report = {
            "mode": "control",
            "control_points": str(args.control_points),
            "control_pixels": str(pix_dir),
            "scale_applied": bool(args.control_scale),
            "scale": s,
            "residual_m": {
                str(n): float(r) for n, r in zip(np.array(names)[ok], resid, strict=True)
            },
            "residual_rms_m": float(np.sqrt(np.mean(resid**2))),
            "views_per_point": {n: int(v) for n, v in zip(names, nv, strict=True)},
        }
        log(
            f"control-point alignment: residual rms {world_report['residual_rms_m'] * 1000:.1f} mm, scale {s:.4f}"
        )

    # 7. export
    cameras_json: dict[str, Any] = {}
    dlts: list[np.ndarray] = []
    for c, (d, ci, Tw) in enumerate(zip(dets, intr, cams_world, strict=True)):
        A = Tw[:3, :3]
        if args.world == "control" and args.control_scale:
            sc = np.cbrt(np.linalg.det(A))
            R, t = A / sc, Tw[:3, 3] / sc
        else:
            R, t = A, Tw[:3, 3]
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        dlt = write_dlt3d(out_dir / f"{d.name}.dlt3d", ci["K"], R, t)
        dlts.append(dlt)
        write_intrinsics_toml(out_dir / f"{d.name}_intrinsics.toml", ci, (d.width, d.height))
        write_board_pixels(out_dir / f"{d.name}_board_pixels.csv", d.corners)
        save_coverage_image(out_dir / "qa" / f"{d.name}_coverage.png", d.path, d.corners)
        center = (-R.T @ t).ravel()
        cameras_json[d.name] = {
            "video": d.path,
            "image_size": [d.width, d.height],
            "fps": d.fps,
            "K": ci["K"],
            "dist_k1_k2_p1_p2_k3": ci["dist"],
            "R_world_to_camera": R,
            "t_world_to_camera_m": t,
            "camera_center_world_m": center,
            "frame_offset": offsets[c],
            "intrinsics": {k: v for k, v in ci.items() if k not in ("K", "dist")},
            "board_views": len(d.corners),
            "ba_rms_px": ba["per_camera_rms_px"].get(c),
        }

    # 8. validation: triangulate the board corners with the exported DLTs
    t0 = time.perf_counter()
    by_frame: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for c, f, img in observations_all:
        by_frame[f][c] = undistort_pixels(img, intr[c]["K"], intr[c]["dist"])
    per_frame, rows_3d = [], []
    for f in sorted(by_frame):
        pix = [by_frame[f].get(c) for c in range(n_cams)]
        xyz, nv = triangulate(list(dlts), pix)
        v = board_validation(xyz, cols, rows, square)
        if v is not None:
            per_frame.append(v)
        for k in range(len(xyz)):
            rows_3d.append([f, k + 1, *xyz[k].tolist(), int(nv[k])])
    pd.DataFrame(
        rows_3d, columns=pd.Index(["frame", "corner", "x_m", "y_m", "z_m", "n_views"])
    ).to_csv(out_dir / "validation_board_3d.csv", index=False, float_format="%.6f")
    errs = np.array([p["rmse_m"] for p in per_frame])
    means = np.array([p["mean_error_m"] for p in per_frame])
    validation = {
        "kind": "in-sample: board corners of the calibration frames, triangulated with the exported DLTs after undistortion",
        "frames": len(per_frame),
        "square_m": square,
        "distance_rmse_median_m": float(np.median(errs)) if len(errs) else None,
        "distance_rmse_p95_m": float(np.percentile(errs, 95)) if len(errs) else None,
        "distance_mean_error_median_m": float(np.median(means)) if len(means) else None,
        "planarity_rms_median_m": float(np.nanmedian([p["planarity_rms_m"] for p in per_frame]))
        if per_frame
        else None,
    }
    if len(errs):
        log(
            "validation (in-sample): neighbour-corner distance error median "
            f"{np.median(errs) * 1000:.2f} mm (p95 {np.percentile(errs, 95) * 1000:.2f} mm) "
            f"for {square * 1000:.0f} mm squares, {len(errs)} frames ({time.perf_counter() - t0:.1f} s)"
        )

    # 9. held-out check: extrinsics from alternate time blocks, board triangulated in the others
    holdout: dict[str, Any] = {"enabled": args.holdout_block > 0}
    if args.holdout_block > 0:
        blk = int(args.holdout_block)
        train = [f for f in multi if (f // blk) % 2 == 0]
        test = [f for f in multi if (f // blk) % 2 == 1]
        if len(train) >= 3 and test:
            if len(train) > args.max_ba_frames:
                train = [
                    train[i] for i in np.linspace(0, len(train) - 1, args.max_ba_frames).astype(int)
                ]
            train_set = set(train)
            obs_tr = [o for o in observations_all if o[1] in train_set]
            cam_tr, _b, info_tr = bundle_adjust(
                obj,
                obs_tr,
                intr,
                cam_T0_ok,
                {f: board_T0_all[f] for f in train},
                ref,
                max_nfev=args.max_nfev,
            )
            # Express the cameras in the board world frame: in the reference-camera frame the
            # reference camera has t = 0, which the DLT normalization (P[2,3] = 1) cannot hold.
            # Distances do not depend on this rigid transform.
            dl_tr: list[np.ndarray | None] = []
            for c in range(n_cams):
                Tw = cam_tr[c] @ np.linalg.inv(T_world_ref)
                dl_tr.append(compute_dlt3d_from_KRt(intr[c]["K"], Tw[:3, :3], Tw[:3, 3]))
            h_err = []
            for f in test:
                xyz, _nv = triangulate(dl_tr, [by_frame[f].get(c) for c in range(n_cams)])
                v = board_validation(xyz, cols, rows, square)
                if v is not None:
                    h_err.append(v["rmse_m"])
            holdout.update(
                {
                    "kind": "extrinsics fitted on alternate time blocks; board triangulated in the "
                    "other blocks (same session; intrinsics from all views)",
                    "block_frames": blk,
                    "train_frames": len(train),
                    "test_frames": len(h_err),
                    "train_ba_rms_px": info_tr["rms_px"],
                    "distance_rmse_median_m": float(np.median(h_err)) if h_err else None,
                    "distance_rmse_p95_m": float(np.percentile(h_err, 95)) if h_err else None,
                }
            )
            if h_err:
                log(
                    f"held-out blocks: neighbour-corner distance error median {np.median(h_err) * 1000:.2f} mm "
                    f"(p95 {np.percentile(h_err, 95) * 1000:.2f} mm) in {len(h_err)} frames not used for the extrinsics"
                )
        else:
            holdout["skipped"] = "not enough multi-camera frames in alternate blocks"
    validation["holdout"] = holdout

    report = {
        "schema": "vaila_chessboard_calib_v1",
        "created": datetime.now().isoformat(timespec="seconds"),
        "input": str(args.input),
        "pattern_inner_corners": [cols, rows],
        "square_m": square,
        "step": args.step,
        "zoom": args.zoom,
        "intrinsics_mode": args.intrinsics,
        "reference_camera_for_ba": dets[ref].name,
        "frame_numbering": {
            "rule": FRAME_NUMBERING_RULE,
            "per_camera": {
                d.name: {"numbering": d.numbering, "dropped_frames": d.dropped} for d in dets
            },
        },
        "sync": sync_report,
        "world": world_report,
        "bundle_adjustment": {k: v for k, v in ba.items() if k != "per_observation_rms_px"},
        "validation": validation,
        "cameras": cameras_json,
        "pnp_rms_median_px": {
            d.name: float(np.median(list(e.values()))) if e else None
            for d, e in zip(dets, pnp_err, strict=True)
        },
    }
    (out_dir / "cameras.json").write_text(
        json.dumps(_json_ready(cameras_json), indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "calibration_report.json").write_text(
        json.dumps(_json_ready(report), indent=2) + "\n", encoding="utf-8"
    )
    _write_readme(out_dir, report, [d.name for d in dets])
    log(f"done -> {out_dir}")
    report["output_dir"] = str(out_dir)
    return report


def _write_readme(out_dir: Path, report: dict[str, Any], names: list[str]) -> None:
    v = report["validation"]
    lines = [
        "vailá chessboard calibration",
        "=" * 28,
        f"Created: {report['created']}",
        f"Input: {report['input']}",
        f"Board: {report['pattern_inner_corners'][0]}x{report['pattern_inner_corners'][1]} inner corners, square {report['square_m']} m",
        f"World frame: {report['world']['mode']} ({report['world'].get('origin', report['world'].get('control_points', ''))})",
        "",
        "Lens per camera (board tilt = p95 spread of the board orientation):",
    ]
    static = []
    for n in names:
        cam = report["cameras"][n]
        intr = cam["intrinsics"]
        tilt = float(intr.get("board_tilt_p95_deg", float("nan")))
        loaded = intr.get("mode") == "loaded"
        source = (
            f"loaded from {intr.get('source')}" if loaded else f"estimated ({intr.get('mode')})"
        )
        lines.append(
            f"  {n}: f={float(cam['K'][0][0]):.1f} px, {source}; board tilt {tilt:.0f} deg"
        )
        if not loaded and np.isfinite(tilt) and tilt < STATIC_BOARD_TILT_DEG:
            static.append(n)
    if static:
        lines += [
            f"  WARNING {', '.join(static)}: the board kept almost one orientation. The focal",
            "  length cannot be told apart from the board distance, so 3D points away from the",
            "  board can be off by centimetres even when the validation below looks good (it",
            "  measures the board itself). Tilt the board 30-45 deg in several directions, or",
            "  load the lens of a calibration where it moved:",
            "    --intrinsics-toml-dir <processed_chessboard_calib_* folder of the same cameras>",
        ]
    lines += [
        "",
        "Per camera:",
        "  <camera>.dlt3d             DLT for Rec3D 1DLT / MultiDLT / yolo_depth3d (ideal pinhole)",
        "  <camera>_intrinsics.toml   fx, fy, cx, cy, k1, k2, k3, p1, p2 (Distort Video; "
        "yolo_depth3d / sam3dinov3)",
        "  <camera>_board_pixels.csv  detected board corners (distorted pixels, vailá pixel format)",
        "  qa/<camera>_coverage.png   corners used; the calibration is measured inside the red outline",
        "",
    ]
    if report.get("mode") == "monocular" or report["sync"].get("mode") == "monocular":
        lines += [
            "Mode: monocular (one video). Sync and bundle adjustment were skipped; the world",
            "  frame is the board at the lowest-PnP-error frame. Use:",
            "    yolo_depth3d --intrinsics-toml <camera>_intrinsics.toml --dlt3d <camera>.dlt3d",
            "    sam3dinov3 --intrinsics-toml <camera>_intrinsics.toml",
            "",
        ]
    lines += [
        "3D reconstruction:",
        "  1. Undistort the pixel CSVs with this calibration (each name starts with or contains the",
        "     camera video name):",
        f"       uv run vaila/chessboard_calib.py undistort --calib {out_dir} -i <pixels> --apply-offsets",
        "     (or Chessboard Calib -> 'Undistort pixels for Rec3D...'). Do not use Distort CSV for this:",
        "     it reprojects with a different camera matrix than the one in the .dlt3d files.",
        "  2. Run Rec3D 1DLT with the <camera>.dlt3d files and the undistorted CSVs; the command is",
        "     printed by step 1.",
        "  --apply-offsets shifts frames by the offsets below (camera frame = reference frame + offset).",
        "  --subframe uses the offsets before rounding: each camera is interpolated linearly to the",
        "  reference instants (no gap filling). Use it for points that move.",
        "",
        "Frame offsets: "
        + ", ".join(f"{n}={report['sync']['frame_offsets'][n]:+d}" for n in names),
    ]
    sub = report["sync"].get("frame_offsets_subframe")
    if sub:
        offs = report["sync"]["frame_offsets"]
        lines.append(
            "Offsets before rounding: " + ", ".join(f"{n}={sub[n]:+.2f}" for n in names if n in sub)
        )
        far = [n for n in names if n in sub and abs(sub[n] - offs[n]) > 0.25]
        if far:
            fps = report["cameras"][names[0]]["fps"]
            lines += [
                f"  {', '.join(far)}: a quarter to half a frame from the reference after rounding.",
                "  The cameras are not genuinely synchronized; whole-frame offsets leave up to",
                f"  half a frame ({500.0 / fps:.0f} ms) between cameras for moving points.",
            ]
    rife = report["sync"].get("rife_equalizer_command")
    if rife:
        lines += [
            "Align the videos below one frame (optional; CUDA RIFE; generated images are estimates):",
            f"  {rife}",
            "  Pixel CSVs of the equalized videos use this calibration without --apply-offsets or",
            "  --subframe (the equalized videos are already aligned).",
        ]
    elif report["sync"].get("rife_note"):
        lines.append(f"Video alignment with RIFE: {report['sync']['rife_note']}.")
    ba = report.get("bundle_adjustment") or {}
    if ba.get("mode") == "monocular_pnp":
        rms = ba.get("rms_px")
        rms_txt = "n/a" if rms is None or not np.isfinite(rms) else f"{float(rms):.3f} px"
        lines.append(f"Monocular PnP rms (median): {rms_txt} (no multi-camera bundle adjustment)")
        if ba.get("dlt_reproj_rms_px") is not None:
            lines.append(f"DLT reprojection at world frame: {ba['dlt_reproj_rms_px']:.3f} px")
    elif ba.get("rms_px") is not None:
        lines.append(f"Bundle adjustment rms: {ba['rms_px']:.3f} px")
    lines += [
        "",
        f"Frame numbers: {report['frame_numbering']['rule']}.",
    ]
    per_cam = report["frame_numbering"]["per_camera"]
    drops = {n: per_cam[n]["dropped_frames"] for n in names if per_cam[n]["dropped_frames"]}
    if drops:
        fps = report["cameras"][names[0]]["fps"]
        lines += [
            "Frames missing from the video timestamps (dropped by the camera):",
            *[f"  {n}: {', '.join(str(f) for f in fr)}" for n, fr in drops.items()],
            "  Tools that count frames while reading (Markerless 2D, Video to PNG) number every",
            "  frame after a drop differently. For multi-camera 3D, make constant-frame-rate",
            "  copies first; their frame numbers match the timestamps (the drop repeats the",
            "  previous image):",
            f"    ffmpeg -i <camera>.mp4 -fps_mode cfr -r {fps:g} -c:v libx264 -crf 12 <camera>_cfr.mp4",
        ]
    if v.get("distance_rmse_median_m") is not None:
        lines.append(
            f"Validation (in-sample): neighbour-corner distance rmse median {v['distance_rmse_median_m'] * 1000:.2f} mm, "
            f"p95 {v['distance_rmse_p95_m'] * 1000:.2f} mm over {v['frames']} frames"
        )
    lines.append(
        "  (uses the calibration frames: a consistency check, not an accuracy measurement)"
    )
    h = v.get("holdout", {})
    if h.get("distance_rmse_median_m") is not None:
        lines.append(
            f"Held-out blocks: neighbour-corner distance rmse median {h['distance_rmse_median_m'] * 1000:.2f} mm, "
            f"p95 {h['distance_rmse_p95_m'] * 1000:.2f} mm over {h['test_frames']} frames"
        )
        lines.append(
            "  (extrinsics fitted without these frames; same session and board, intrinsics from all views)"
        )
    (out_dir / "README_calibration.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI and GUI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chessboard_calib",
        description="Multi-camera chessboard calibration -> .dlt3d + intrinsics TOML for vailá Rec3D.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "-i",
        "--input",
        help="Folder with chessboard videos (one per camera), or a single video file "
        "(monocular: same .dlt3d + _intrinsics.toml for yolo_depth3d / sam3dinov3).",
    )
    p.add_argument("-o", "--output", help="Output parent folder (default: the input folder).")
    p.add_argument(
        "--pattern", default="6x8", help="Inner corners per row x per column (default 6x8)."
    )
    p.add_argument(
        "--square",
        type=float,
        default=DEFAULT_SQUARE_M,
        help="Square side in metres (default 0.098); the scale of the whole reconstruction.",
    )
    p.add_argument(
        "--step", type=int, default=1, help="Use every N-th frame (default 1; auto sync needs 1)."
    )
    p.add_argument(
        "--zoom",
        type=float,
        default=2.0,
        help="Upscale factor of the crop around the board (default 2).",
    )
    p.add_argument(
        "--search-every",
        type=int,
        default=10,
        help="Full-frame search interval when the board is lost.",
    )
    p.add_argument(
        "--max-frames", type=int, default=None, help="Read only the first N frames (testing)."
    )
    p.add_argument(
        "--intrinsics", choices=INTRINSIC_MODES, default="auto", help="Lens model (default auto)."
    )
    p.add_argument(
        "--intrinsics-input", help="Folder with separate intrinsics videos (same file stems)."
    )
    p.add_argument(
        "--intrinsics-toml-dir",
        help="Folder with <stem>_intrinsics.toml to load instead of estimating.",
    )
    p.add_argument(
        "--sync", choices=("auto", "none"), default="auto", help="Frame offsets between cameras."
    )
    p.add_argument(
        "--frame-offsets",
        help=(
            "Manual offsets, comma-separated in camera (file name) order; decimals (e.g. "
            "measured where the board moved) are kept for undistort --subframe and RIFE."
        ),
    )
    p.add_argument(
        "--max-lag", type=int, default=30, help="Largest frame offset searched by --sync auto."
    )
    p.add_argument(
        "--world", choices=("board", "control"), default="board", help="World frame definition."
    )
    p.add_argument(
        "--board-axes",
        choices=("upright", "board"),
        default="upright",
        help="Axes for --world board.",
    )
    p.add_argument("--control-points", help="CSV with name,x,y,z (metres) of floor/control points.")
    p.add_argument(
        "--control-pixels", help="Folder with one vailá pixel CSV per camera (file stem prefix)."
    )
    p.add_argument(
        "--control-scale", action="store_true", help="Also fit a scale to the control points."
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel detection processes (default: CPU count).",
    )
    p.add_argument(
        "--reuse-detections",
        help="Previous processed_chessboard_calib_* folder: reuse its <stem>_board_pixels.csv.",
    )
    p.add_argument(
        "--max-nfev",
        type=int,
        default=1000,
        help="Bundle adjustment evaluation limit (default 1000).",
    )
    p.add_argument(
        "--max-ba-frames",
        type=int,
        default=1000,
        help="Most multi-camera frames used by the bundle adjustment (default 1000).",
    )
    p.add_argument(
        "--holdout-block",
        type=int,
        default=60,
        help="Held-out check: frames per alternate time block (default 60; 0 disables).",
    )
    return p


def cli_argv(args: argparse.Namespace) -> list[str]:
    """argv reproducing ``args`` (for the GUI -> CLI mirror)."""
    argv = ["uv", "run", "--no-sync", "vaila/chessboard_calib.py", "-i", str(args.input)]
    defaults = vars(build_parser().parse_args(["-i", "x"]))
    for key, value in vars(args).items():
        if key == "input" or value == defaults.get(key) or value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    return argv


# --------------------------------------------------------------------------- #
# Undistort pixel CSVs for Rec3D
# --------------------------------------------------------------------------- #


def build_undistort_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chessboard_calib undistort",
        description=(
            "Undistort vailá pixel CSVs with a chessboard calibration, in the camera matrix "
            "that the exported .dlt3d files expect, and print the Rec3D 1DLT command."
        ),
    )
    p.add_argument("--calib", required=True, help="processed_chessboard_calib_* folder.")
    p.add_argument(
        "-i",
        "--input",
        required=True,
        help="Pixel CSV or folder of CSVs; each file name starts with its camera video name.",
    )
    p.add_argument("-o", "--output", help="Output folder (default: <calib>/undistorted_<time>).")
    p.add_argument(
        "--apply-offsets",
        action="store_true",
        help="Rewrite the frame column to the reference timeline (frame - offset).",
    )
    p.add_argument(
        "--subframe",
        action="store_true",
        help=(
            "Use the offsets before rounding: interpolate each camera linearly to the reference "
            "instants (implies --apply-offsets; gaps stay empty)."
        ),
    )
    return p


def shift_to_reference(df: pd.DataFrame, offset: float) -> pd.DataFrame:
    """Rows on the reference timeline: reference frame n takes the camera at frame n + offset.

    ``offset`` is the camera's offset before rounding (camera frame = reference frame +
    offset). A fractional part interpolates linearly between the two neighbouring camera
    frames; a missing or NaN neighbour gives NaN (gaps and dropped frames are not filled).
    An integer offset only renumbers the frames. Rows that end up all-NaN are dropped.
    """
    frames = df["frame"].astype(int).to_numpy()
    cols = [c for c in df.columns if c != "frame" and pd.api.types.is_numeric_dtype(df[c])]
    first, last = int(frames.min()), int(frames.max())
    table = df.assign(frame=frames).drop_duplicates("frame").set_index("frame")[cols]
    values = table.reindex(range(first, last + 1)).to_numpy(np.float64)
    base = math.floor(offset)
    frac = float(offset - base)
    if frac > 1 - 1e-6:
        base, frac = base + 1, 0.0
    rows = values if frac < 1e-6 else (1.0 - frac) * values[:-1] + frac * values[1:]
    ref_frames = np.arange(first, first + len(rows)) - base
    out = pd.DataFrame(rows, columns=pd.Index(cols))
    out.insert(0, "frame", ref_frames)
    return out[~np.isnan(rows).all(axis=1)].reset_index(drop=True)


def undistort_pixel_files(
    calib_dir: Path,
    inputs: Path,
    out_dir: Path,
    apply_offsets: bool = False,
    subframe: bool = False,
) -> list[tuple[str, Path, Path]]:
    """Undistort every ``p*_x``/``p*_y`` pair with the matching camera's K and distortion.

    The result is in ideal pinhole pixels of the same K, which is what the exported
    ``.dlt3d`` files expect. (Distort CSV reprojects with ``getOptimalNewCameraMatrix`` and
    a fixed 1920x1080 size, a different camera matrix, so it must not be used here.)
    """
    cams = json.loads((calib_dir / "cameras.json").read_text(encoding="utf-8"))
    names = list(cams)
    exact: dict[str, float] = {}
    if subframe:
        report_path = calib_dir / "calibration_report.json"
        report = (
            json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        )
        exact = report.get("sync", {}).get("frame_offsets_subframe") or {}
        if not exact:
            raise ValueError(
                f"{calib_dir.name} has no offsets before rounding (sync.frame_offsets_subframe "
                "in calibration_report.json): it was calibrated with --sync none or manual "
                "offsets, or by an older version. Use --apply-offsets instead."
            )
    files = sorted(inputs.glob("*.csv")) if inputs.is_dir() else [inputs]
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, Path, Path]] = []
    for path in files:
        cam = _match_camera(path.stem, names)
        if cam is None:
            log(f"skip {path.name}: its name does not start with a camera of {calib_dir.name}")
            continue
        K = np.asarray(cams[cam]["K"], np.float64)
        dist = np.asarray(cams[cam]["dist_k1_k2_p1_p2_k3"], np.float64)
        df = pd.read_csv(path)
        pairs = [
            (c, c[:-2] + "_y")
            for c in map(str, df.columns)
            if c.endswith("_x") and c[:-2] + "_y" in df.columns
        ]
        if not pairs:
            log(f"skip {path.name}: no p*_x / p*_y columns")
            continue
        for cx, cy in pairs:
            und = undistort_pixels(df[[cx, cy]].to_numpy(np.float64), K, dist)
            df[cx], df[cy] = und[:, 0], und[:, 1]
        if subframe and "frame" in df.columns:
            if cam not in exact:
                log(f"skip {path.name}: no offset before rounding for {cam}")
                continue
            dropped = [
                c for c in df.columns if c != "frame" and not pd.api.types.is_numeric_dtype(df[c])
            ]
            n_in = len(df)
            df = shift_to_reference(df, float(exact[cam]))
            log(
                f"{path.name}: shifted by {float(exact[cam]):+.3f} frames to the reference "
                f"timeline ({n_in} -> {len(df)} rows)"
                + (f"; text columns left out: {dropped}" if dropped else "")
            )
        elif apply_offsets and "frame" in df.columns:
            df["frame"] = df["frame"].astype(int) - int(cams[cam]["frame_offset"])
        out = out_dir / path.name
        df.to_csv(out, index=False, float_format="%.6f")
        written.append((cam, path, out))
        log(f"{path.name}: {len(pairs)} points undistorted with {cam} -> {out}")
    return written


def run_undistort(argv: list[str], *, from_gui: bool = False) -> int:
    args = build_undistort_parser().parse_args(argv)
    calib = Path(args.calib)
    out = (
        Path(args.output) if args.output else calib / f"undistorted_{datetime.now():%Y%m%d_%H%M%S}"
    )
    cmd = ["uv", "run", "--no-sync", "vaila/chessboard_calib.py", "undistort", *argv]
    if from_gui:
        print_gui_cli_mirror("vaila/chessboard_calib", cmd)
    written = undistort_pixel_files(
        calib, Path(args.input), out, args.apply_offsets, subframe=args.subframe
    )
    if not written:
        log("no pixel file was undistorted")
        return 1
    cams = json.loads((calib / "cameras.json").read_text(encoding="utf-8"))
    fps = cams[written[0][0]].get("fps") or 30.0
    rec = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "vaila.rec3d_one_dlt3d",
        "--dlt3d",
        *[str(calib / f"{cam}.dlt3d") for cam, _src, _dst in written],
        "--pixels",
        *[str(dst) for _cam, _src, dst in written],
        "--fps",
        f"{float(fps):g}",
        "-o",
        str(out),
    ]
    log("next step, Rec3D 1DLT with these files:\n>>   " + shlex.join(rec))
    if not (args.apply_offsets or args.subframe):
        log(
            "frames were kept as recorded; add --apply-offsets (or --subframe) if the videos "
            "are not synchronized"
        )
    return 0


def run_gui() -> tuple[str, Any] | None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("vailá — Chessboard Calib")
    values: dict[str, tk.Variable] = {
        "input": tk.StringVar(),
        "output": tk.StringVar(),
        "pattern": tk.StringVar(value="6x8"),
        "square": tk.StringVar(value=str(DEFAULT_SQUARE_M)),
        "step": tk.StringVar(value="1"),
        "zoom": tk.StringVar(value="2.0"),
        "intrinsics": tk.StringVar(value="auto"),
        "intrinsics_toml_dir": tk.StringVar(),
        "sync": tk.StringVar(value="auto"),
        "frame_offsets": tk.StringVar(),
        "world": tk.StringVar(value="board"),
        "control_points": tk.StringVar(),
        "control_pixels": tk.StringVar(),
    }
    result: dict[str, tuple[str, Any] | None] = {"args": None}
    frm = ttk.Frame(root, padding=12)
    frm.grid(sticky="nsew")

    def browse(key: str, kind: str) -> None:
        if kind == "dir":
            path = filedialog.askdirectory(parent=root)
        else:
            path = filedialog.askopenfilename(
                parent=root, filetypes=[("CSV", "*.csv"), ("All", "*.*")]
            )
        if path:
            values[key].set(path)

    rows_spec = [
        ("Camera videos folder", "input", "dir"),
        ("Output parent folder (empty = input)", "output", "dir"),
        ("Inner corners (per row x per column)", "pattern", None),
        ("Square side (m)", "square", None),
        ("Frame step", "step", None),
        ("Zoom around the board", "zoom", None),
        ("Intrinsics model", "intrinsics", list(INTRINSIC_MODES)),
        ("Intrinsics TOML folder (optional)", "intrinsics_toml_dir", "dir"),
        ("Sync", "sync", ["auto", "none"]),
        ("Manual frame offsets (optional, comma list; decimals ok)", "frame_offsets", None),
        ("World frame", "world", ["board", "control"]),
        ("Control points CSV (name,x,y,z m)", "control_points", "file"),
        ("Control pixels folder", "control_pixels", "dir"),
    ]
    for r, (label, key, kind) in enumerate(rows_spec):
        ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", pady=2)
        if isinstance(kind, list):
            ttk.Combobox(
                frm, textvariable=values[key], values=kind, state="readonly", width=46
            ).grid(row=r, column=1, sticky="we")
        else:
            ttk.Entry(frm, textvariable=values[key], width=48).grid(row=r, column=1, sticky="we")
        if kind in ("dir", "file"):
            ttk.Button(frm, text="Browse", command=lambda k=key, t=str(kind): browse(k, t)).grid(
                row=r, column=2, padx=4
            )

    def on_run() -> None:
        if not values["input"].get():
            messagebox.showerror(
                "Chessboard Calib", "Choose the camera videos folder.", parent=root
            )
            return
        argv = ["-i", values["input"].get()]
        for key, var in values.items():
            val = str(var.get()).strip()
            if key == "input" or not val:
                continue
            argv.extend(["--" + key.replace("_", "-"), val])
        try:
            result["args"] = ("calibrate", build_parser().parse_args(argv))
        except SystemExit:
            messagebox.showerror(
                "Chessboard Calib", "Invalid parameters; check the fields.", parent=root
            )
            return
        root.destroy()

    def on_undistort() -> None:
        calib = filedialog.askdirectory(
            parent=root, title="Calibration folder (processed_chessboard_calib_*)"
        )
        if not calib:
            return
        pixels = filedialog.askdirectory(
            parent=root, title="Folder with pixel CSVs (names start with the camera video name)"
        )
        if not pixels:
            return
        argv = ["--calib", calib, "-i", pixels]
        if messagebox.askyesno(
            "Chessboard Calib",
            "Apply the frame offsets of the calibration (videos not synchronized)?",
            parent=root,
        ):
            if messagebox.askyesno(
                "Chessboard Calib",
                "Use the sub-frame offsets (interpolate the pixels between frames)?",
                parent=root,
            ):
                argv.append("--subframe")
            else:
                argv.append("--apply-offsets")
        result["args"] = ("undistort", argv)
        root.destroy()

    def on_help() -> None:
        help_html = Path(__file__).resolve().parent / "help" / "chessboard_calib.html"
        if help_html.is_file():
            import webbrowser

            webbrowser.open(help_html.as_uri())

    btns = ttk.Frame(frm)
    btns.grid(row=len(rows_spec), column=0, columnspan=3, pady=(10, 0))
    ttk.Button(btns, text="Run calibration", command=on_run).pack(side="left", padx=4)
    ttk.Button(btns, text="Undistort pixels for Rec3D...", command=on_undistort).pack(
        side="left", padx=4
    )
    ttk.Button(btns, text="Help", command=on_help).pack(side="left", padx=4)
    ttk.Button(btns, text="Cancel", command=root.destroy).pack(side="left", padx=4)
    root.mainloop()
    return result["args"]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    if argv and argv[0] == "undistort":
        return run_undistort(argv[1:])
    if not argv:
        choice = run_gui()
        if choice is None:
            log("cancelled")
            return 1
        mode, payload = choice
        if mode == "undistort":
            return run_undistort(payload, from_gui=True)
        args = payload
        print_gui_cli_mirror("vaila/chessboard_calib", cli_argv(args))
    else:
        args = parser.parse_args(argv)
        if not args.input:
            parser.error("-i/--input is required (or run without arguments for the GUI)")
        log("command: " + shlex.join(cli_argv(args)))
    if args.world == "control" and not (args.control_points and args.control_pixels):
        parser.error("--world control needs --control-points and --control-pixels")
    run_calibration(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
