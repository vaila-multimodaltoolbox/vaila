"""
================================================================================
Script: freekiki_openplay_kinematics.py - planar path, speed and COD metrics
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.14
Created: 02 November 2026
Update Date: 02 November 2026

Description:
    Pure planar kinematics for open-play FreeKiki analyses (dribble 1vN and pass):

    * load person ground tracks (``ground_tracks*.csv`` smoothed or raw / ankle mean from poses)
      and ball XY (``ball_3d_on_field.csv`` flight model, or ``ball_3d.csv`` ground ray when the
      flight model puts the ball under the pitch);
    * frame rate from the run's JSON, else its source video (README ``video:`` line);
    * outlier-rejected, low-pass smoothing of noisy tracks (:func:`smooth_track`);
    * path length, speed, acceleration / deceleration, heading, path curvature (rad/m),
      change-of-direction (COD) events;
    * one change of direction between two straight stretches chosen on the video
      (:func:`turn_between`: angle, time, angular speed, entry/exit speed) and the body turn
      from the hip line of SAM 3D Body (:func:`body_yaw_deg`, :func:`body_rotation`), with the
      COD and speed zones of Monteiro, Santiago et al. 2025 (Sci Rep);
    * interpersonal distance and relative speed between two tracks.

    Scientific framing (operational metrics on filmed clips, not lab closed-skill batteries):
    Wilson et al. 2018 (curved-path dribbling speed); Wilson & Santiago predator–prey COD;
    Corrêa / Duarte interpersonal distance and relative velocity in 1v1;
    Monteiro, Santiago et al. 2025 Sci Rep (COD as angular speed in zones, crossed with speed zones).

    Dual-import pattern for package and standalone use.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

try:
    from .freekiki_layout import data_dir as analysis_data_dir
    from .freekiki_layout import resolve
except ImportError:
    from freekiki_layout import data_dir as analysis_data_dir  # ty: ignore[unresolved-import]
    from freekiki_layout import resolve  # ty: ignore[unresolved-import]

DEG = 180.0 / math.pi
DEFAULT_FPS = 30.0
COD_MIN_DEG = 25.0  # operational: heading change that counts as a COD event
CURVATURE_MIN_STEP_M = 0.05  # skip curvature on near-zero steps


def _finite_xy(arr: np.ndarray) -> np.ndarray:
    """``(n, 2)`` float array; non-finite rows become NaN."""
    out = np.asarray(arr, dtype=float)
    if out.ndim != 2 or out.shape[1] != 2:
        raise ValueError(f"expected (n, 2) xy, got shape {out.shape}")
    bad = ~np.isfinite(out).all(axis=1)
    out = out.copy()
    out[bad] = np.nan
    return out


def path_length(xy: np.ndarray) -> float:
    """Sum of finite consecutive steps in metres."""
    pts = _finite_xy(xy)
    total = 0.0
    prev = None
    for row in pts:
        if not np.isfinite(row).all():
            prev = None
            continue
        if prev is not None:
            total += float(np.linalg.norm(row - prev))
        prev = row
    return total


def speed(xy: np.ndarray, fps: float) -> np.ndarray:
    """Instantaneous speed (m/s) per frame; NaN where the step is undefined."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    pts = _finite_xy(xy)
    n = len(pts)
    out = np.full(n, np.nan)
    dt = 1.0 / float(fps)
    for i in range(1, n):
        if np.isfinite(pts[i]).all() and np.isfinite(pts[i - 1]).all():
            out[i] = float(np.linalg.norm(pts[i] - pts[i - 1]) / dt)
    return out


def accel(v: np.ndarray, fps: float) -> np.ndarray:
    """Acceleration (m/s²) from a speed series; NaN where either neighbour speed is missing."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    vv = np.asarray(v, dtype=float)
    n = len(vv)
    out = np.full(n, np.nan)
    dt = 1.0 / float(fps)
    for i in range(1, n):
        if np.isfinite(vv[i]) and np.isfinite(vv[i - 1]):
            out[i] = float((vv[i] - vv[i - 1]) / dt)
    return out


def heading(xy: np.ndarray) -> np.ndarray:
    """Travel heading (rad, atan2(dy, dx)) per frame from the previous point; NaN at starts/gaps."""
    pts = _finite_xy(xy)
    n = len(pts)
    out = np.full(n, np.nan)
    for i in range(1, n):
        if np.isfinite(pts[i]).all() and np.isfinite(pts[i - 1]).all():
            d = pts[i] - pts[i - 1]
            if float(np.linalg.norm(d)) > 1e-9:
                out[i] = float(math.atan2(float(d[1]), float(d[0])))
    return out


def _unwrap_delta(a: float, b: float) -> float:
    """Smallest signed angle from ``a`` to ``b`` (rad)."""
    d = b - a
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


def curvature_rad_per_m(xy: np.ndarray) -> np.ndarray:
    """Local path curvature |dθ/ds| in rad/m (Wilson-style operational measure)."""
    pts = _finite_xy(xy)
    heads = heading(pts)
    n = len(pts)
    out = np.full(n, np.nan)
    for i in range(2, n):
        if not (np.isfinite(heads[i]) and np.isfinite(heads[i - 1])):
            continue
        if not (np.isfinite(pts[i]).all() and np.isfinite(pts[i - 1]).all()):
            continue
        step = float(np.linalg.norm(pts[i] - pts[i - 1]))
        if step < CURVATURE_MIN_STEP_M:
            continue
        out[i] = abs(_unwrap_delta(float(heads[i - 1]), float(heads[i]))) / step
    return out


def cod_events(
    heads: np.ndarray,
    *,
    min_deg: float = COD_MIN_DEG,
    fps: float | None = None,
) -> list[dict]:
    """Frames where |Δheading| exceeds ``min_deg`` (change of direction events)."""
    hh = np.asarray(heads, dtype=float)
    thr = abs(float(min_deg)) / DEG
    events: list[dict] = []
    prev = None
    prev_i = None
    for i, h in enumerate(hh):
        if not np.isfinite(h):
            prev, prev_i = None, None
            continue
        if prev is not None and prev_i is not None:
            delta = _unwrap_delta(float(prev), float(h))
            if abs(delta) >= thr:
                ev = {
                    "frame": int(i),
                    "from_frame": int(prev_i),
                    "delta_deg": float(delta * DEG),
                    "abs_deg": float(abs(delta) * DEG),
                }
                if fps and fps > 0:
                    ev["t_s"] = float(i / fps)
                events.append(ev)
        prev, prev_i = float(h), int(i)
    return events


# Monteiro, Santiago et al. 2025 (Sci Rep 10.1038/s41598-025-31587-w): COD as angular speed (deg/s)
# from filtered x-y positions, in zones; speed zones in km/h. Their data are 10 Hz GNSS low-passed at
# 1 Hz, so on a 24 fps broadcast clip the zones are a qualitative reading, not a numeric comparison.
COD_ZONES_DEG_S = ((30.0, "very_low"), (60.0, "low"), (90.0, "moderate"), (math.inf, "high"))
SPEED_ZONES_KMH = (6.0, 15.0, 20.0, 25.0)  # zone 1 below 6, zone 5 from 25 km/h


def cod_zone(deg_per_s: float | None) -> str | None:
    """Monteiro et al. 2025 COD zone of an angular speed (|deg/s|)."""
    if deg_per_s is None or not math.isfinite(deg_per_s):
        return None
    value = abs(float(deg_per_s))
    return next(name for limit, name in COD_ZONES_DEG_S if value < limit)


def speed_zone(kmh: float | None) -> int | None:
    """Monteiro et al. 2025 speed zone (1..5) of a speed in km/h."""
    if kmh is None or not math.isfinite(kmh):
        return None
    return 1 + sum(float(kmh) >= limit for limit in SPEED_ZONES_KMH)


def chord(xy: np.ndarray, a: int, b: int) -> tuple[float, float] | None:
    """Heading (deg, atan2(dy, dx)) and length (m) of the straight line from frame ``a`` to ``b``."""
    pts = _finite_xy(xy)
    if not (0 <= a < b < len(pts)) or not np.isfinite(pts[[a, b]]).all():
        return None
    d = pts[b] - pts[a]
    length = float(np.linalg.norm(d))
    if length <= 1e-9:
        return None
    return math.degrees(math.atan2(float(d[1]), float(d[0]))), length


def turn_between(
    xy: np.ndarray,
    fps: float,
    before: tuple[int, int],
    after: tuple[int, int],
) -> dict | None:
    """One change of direction of a path, between two straight stretches chosen on the video.

    The angle is the smallest signed difference between the heading of the stretch ``after`` and
    that of ``before`` (each a chord, frame to frame, so the noise of single steps averages out).
    The turn lasts from the end of ``before`` to the start of ``after``; its mean angular speed is
    the angle over that time. The entry and exit speeds are the chord lengths over their times,
    and the slowest point between is taken from the frame-to-frame speed. Returns None when a chord
    is missing.
    """
    (a0, a1), (b0, b1) = (int(v) for v in before), (int(v) for v in after)
    if not (a1 <= b0) or fps <= 0:
        raise ValueError("before must end at or before the start of after")
    cin, cout = chord(xy, a0, a1), chord(xy, b0, b1)
    if cin is None or cout is None:
        return None
    angle = _unwrap_delta(math.radians(cin[0]), math.radians(cout[0])) * DEG
    duration = (b0 - a1) / float(fps)
    v_in = cin[1] / ((a1 - a0) / float(fps))
    v_out = cout[1] / ((b1 - b0) / float(fps))
    v = speed(_finite_xy(xy), fps)[a1 : b0 + 1]
    finite = np.flatnonzero(np.isfinite(v))
    rate = abs(angle) / duration if duration > 0 else None
    return {
        "method": "path_chords",
        "before_frames": [a0, a1],
        "after_frames": [b0, b1],
        "heading_in_deg": cin[0],
        "heading_out_deg": cout[0],
        "angle_deg": angle,
        "abs_deg": abs(angle),
        "t_start_s": a1 / float(fps),
        "t_end_s": b0 / float(fps),
        "duration_s": duration,
        "angular_speed_deg_s": rate,
        "cod_zone": cod_zone(rate),
        "v_in_mps": v_in,
        "v_out_mps": v_out,
        "v_in_zone": speed_zone(v_in * 3.6),
        "v_out_zone": speed_zone(v_out * 3.6),
        "v_min_mps": float(v[finite].min()) if finite.size else None,
        "v_min_frame": int(a1 + finite[np.argmin(v[finite])]) if finite.size else None,
    }


def body_yaw_deg(left_hip_xz: np.ndarray, right_hip_xz: np.ndarray) -> np.ndarray:
    """Yaw (deg) of the hip line about the vertical, from camera-frame (x, z) of both hips.

    SAM 3D Body keypoints are in the camera frame (x right, y down, z forward), so rotation about
    the camera y axis approximates rotation about the vertical for a near-level broadcast camera.
    The series is unwrapped; NaN where a hip is missing.
    """
    left = np.asarray(left_hip_xz, dtype=float)
    right = np.asarray(right_hip_xz, dtype=float)
    d = right - left
    yaw = np.full(len(d), np.nan)
    ok = np.isfinite(d).all(axis=1)
    yaw[ok] = np.degrees(np.unwrap(np.arctan2(d[ok, 1], d[ok, 0])))
    return yaw


def body_rotation(yaw_deg: np.ndarray, fps: float, start: int, end: int) -> dict | None:
    """Rotation of the body (hip line) from frame ``start`` to ``end``: total, time and mean rate."""
    yaw = np.asarray(yaw_deg, dtype=float)
    if not (0 <= start < end < len(yaw)) or not np.isfinite(yaw[[start, end]]).all():
        return None
    total = float(yaw[end] - yaw[start])
    duration = (end - start) / float(fps)
    rate = abs(total) / duration
    return {
        "method": "hip_line_yaw_camera_frame",
        "frames": [int(start), int(end)],
        "angle_deg": total,
        "abs_deg": abs(total),
        "t_start_s": start / float(fps),
        "t_end_s": end / float(fps),
        "duration_s": duration,
        "angular_speed_deg_s": rate,
        "cod_zone": cod_zone(rate),
    }


def interpersonal_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance (m) between two XY tracks of equal length."""
    aa, bb = _finite_xy(a), _finite_xy(b)
    if len(aa) != len(bb):
        raise ValueError(f"track length mismatch: {len(aa)} vs {len(bb)}")
    out = np.full(len(aa), np.nan)
    both = np.isfinite(aa).all(axis=1) & np.isfinite(bb).all(axis=1)
    out[both] = np.linalg.norm(aa[both] - bb[both], axis=1)
    return out


def relative_speed(a: np.ndarray, b: np.ndarray, fps: float) -> np.ndarray:
    """Closing / separating rate (m/s): positive when distance increases."""
    d = interpersonal_distance(a, b)
    return accel(d, fps)  # d/dt of distance shares the same finite-difference as accel


def series_summary(v: np.ndarray, a: np.ndarray | None = None) -> dict:
    """Compact speed / accel peaks from finite samples."""
    vv = np.asarray(v, dtype=float)
    finite = vv[np.isfinite(vv)]
    out: dict = {
        "v_mean_mps": float(finite.mean()) if finite.size else None,
        "v_max_mps": float(finite.max()) if finite.size else None,
        "v_min_mps": float(finite.min()) if finite.size else None,
        "samples": int(finite.size),
    }
    if a is not None:
        aa = np.asarray(a, dtype=float)
        pos = aa[np.isfinite(aa) & (aa > 0)]
        neg = aa[np.isfinite(aa) & (aa < 0)]
        out["a_peak_mps2"] = float(pos.max()) if pos.size else None
        out["decel_peak_mps2"] = float((-neg).max()) if neg.size else None
    return out


def track_metrics(xy: np.ndarray, fps: float, *, min_cod_deg: float = COD_MIN_DEG) -> dict:
    """Full planar metric pack for one XY track."""
    pts = _finite_xy(xy)
    v = speed(pts, fps)
    a = accel(v, fps)
    heads = heading(pts)
    curv = curvature_rad_per_m(pts)
    curv_f = curv[np.isfinite(curv)]
    codes = cod_events(heads, min_deg=min_cod_deg, fps=fps)
    summary = series_summary(v, a)
    # lateral deviation from the chord first→last finite point
    valid = np.isfinite(pts).all(axis=1)
    lat_max = None
    if valid.sum() >= 2:
        idx = np.flatnonzero(valid)
        p0, p1 = pts[idx[0]], pts[idx[-1]]
        chord = p1 - p0
        length = float(np.linalg.norm(chord))
        if length > 1e-6:
            unit = chord / length
            perp = np.array([-unit[1], unit[0]])
            lat = (pts[valid] - p0) @ perp
            lat_max = float(np.nanmax(np.abs(lat)))
    return {
        "path_length_m": path_length(pts),
        "curvature_mean_rad_per_m": float(curv_f.mean()) if curv_f.size else None,
        "curvature_max_rad_per_m": float(curv_f.max()) if curv_f.size else None,
        "cod_count": len(codes),
        "cod_events": codes,
        "lateral_dev_max_m": lat_max,
        **summary,
        "speed_mps": [None if not np.isfinite(x) else float(x) for x in v],
        "accel_mps2": [None if not np.isfinite(x) else float(x) for x in a],
        "heading_rad": [None if not np.isfinite(x) else float(x) for x in heads],
        "curvature_rad_per_m": [None if not np.isfinite(x) else float(x) for x in curv],
        "xy_m": [
            [None, None] if not np.isfinite(row).all() else [float(row[0]), float(row[1])]
            for row in pts
        ],
    }


def _frames_extent(*maps: dict[int, tuple[float, float]]) -> int:
    last = -1
    for m in maps:
        if m:
            last = max(last, max(m))
    return last + 1


def _to_array(series: dict[int, tuple[float, float]], n: int) -> np.ndarray:
    out = np.full((n, 2), np.nan)
    for f, (x, y) in series.items():
        if 0 <= f < n and math.isfinite(x) and math.isfinite(y):
            out[f] = [x, y]
    return out


def read_ground_person(
    folder: Path | str, person_id: int, *, raw: bool = False
) -> dict[int, tuple[float, float]]:
    """Frame -> (x, y) for ``person_id`` from ground_tracks (players file preferred).

    ``raw=True`` reads the unsmoothed box-base columns ``raw_x_m/raw_y_m`` when the file has
    them (the ``x_m/y_m`` columns can be a near constant-velocity fit that hides when a player
    starts to run); smooth them with :func:`smooth_track`.
    """
    folder = Path(folder)
    for name in ("ground_tracks_players.csv", "ground_tracks.csv"):
        path = resolve(folder, name)
        if not path.is_file():
            path = folder / name
        if not path.is_file():
            continue
        out: dict[int, tuple[float, float]] = {}
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    pid = int(row["person_id"])
                    if pid != int(person_id):
                        continue
                    frame = int(row["frame"])
                    if raw and row.get("raw_x_m") not in (None, ""):
                        x, y = float(row["raw_x_m"]), float(row["raw_y_m"])
                    else:
                        x, y = float(row["x_m"]), float(row["y_m"])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(x) and math.isfinite(y):
                    out[frame] = (x, y)
        if out:
            return out
    # ankle mean from poses
    for name in ("poses_3d_on_field_players.csv", "poses_3d_on_field.csv"):
        path = folder / name
        if not path.is_file():
            continue
        sums: dict[int, list[float]] = {}
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if "ankle" not in row.get("kpt_name", ""):
                    continue
                try:
                    if int(row["person_id"]) != int(person_id):
                        continue
                    frame = int(row["frame"])
                    x, y = float(row["x_m"]), float(row["y_m"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                acc = sums.setdefault(frame, [0.0, 0.0, 0.0])
                acc[0] += x
                acc[1] += y
                acc[2] += 1.0
        return {f: (s[0] / s[2], s[1] / s[2]) for f, s in sums.items() if s[2] > 0}
    return {}


BALL_SOURCES = ("auto", "ground", "flight")
BALL_BELOW_GROUND_M = -0.1  # a flight-file z below this means the flight model failed
BALL_MIN_SCORE = 0.5  # ground-ray rows below this detection score are dropped


def _ball_file(folder: Path, name: str) -> Path:
    path = resolve(folder, name)
    return path if path.is_file() else folder / name


def _flight_file_below_ground(path: Path) -> bool:
    """True when ``ball_3d_on_field.csv`` puts the ball under the pitch (failed flight fit)."""
    if not path.is_file():
        return False
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                z = float(row.get("z_m") or "nan")
            except ValueError:
                continue
            if math.isfinite(z) and z < BALL_BELOW_GROUND_M:
                return True
    return False


def ball_source(folder: Path | str, source: str = "auto") -> tuple[Path | None, str]:
    """``(file, name)`` of the ball track to use.

    ``flight`` is ``ball_3d_on_field.csv`` (the 3D flight model on the pitch); ``ground`` is
    ``ball_3d.csv`` (each detection's camera ray cut at ball height, right for a ground pass).
    ``auto`` keeps the flight file unless it puts the ball below the pitch, then uses the ground
    file. ``name`` is ``flight``, ``ground`` or ``none``.
    """
    if source not in BALL_SOURCES:
        raise ValueError(f"ball source must be one of {', '.join(BALL_SOURCES)}, got {source!r}")
    folder = Path(folder)
    flight = _ball_file(folder, "ball_3d_on_field.csv")
    ground = _ball_file(folder, "ball_3d.csv")
    if source == "flight":
        return (flight, "flight") if flight.is_file() else (None, "none")
    if source == "ground":
        return (ground, "ground") if ground.is_file() else (None, "none")
    if flight.is_file() and not (ground.is_file() and _flight_file_below_ground(flight)):
        return flight, "flight"
    if ground.is_file():
        return ground, "ground"
    return None, "none"


def read_ball_xy(folder: Path | str, source: str = "auto") -> dict[int, tuple[float, float]]:
    """Frame -> (x, y) metres of the ball (file chosen by :func:`ball_source`)."""
    path, name = ball_source(folder, source)
    out: dict[int, tuple[float, float]] = {}
    if path is None:
        return out
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                frame = int(row["frame"])
                if row.get("x_m") in (None, "") or row.get("y_m") in (None, ""):
                    continue
                x, y = float(row["x_m"]), float(row["y_m"])
                if (
                    name == "ground"
                    and row.get("score") not in (None, "")
                    and float(row["score"]) < BALL_MIN_SCORE
                ):
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                out[frame] = (x, y)
    return out


def load_xy_series(
    folder: Path | str, person_id: int, n: int | None = None, *, raw: bool = False
) -> np.ndarray:
    """``(n, 2)`` ground track for ``person_id`` (NaN gaps); ``raw`` as in read_ground_person."""
    series = read_ground_person(folder, person_id, raw=raw)
    if n is None:
        n = _frames_extent(series)
    return _to_array(series, max(n, 0))


def load_ball_xy(folder: Path | str, n: int | None = None, *, source: str = "auto") -> np.ndarray:
    """``(n, 2)`` ball track on the pitch plane (NaN gaps); ``source`` as in ball_source."""
    series = read_ball_xy(folder, source)
    if n is None:
        n = _frames_extent(series)
    return _to_array(series, max(n, 0))


def video_fps(video: Path | str) -> float | None:
    """Frame rate of a video file (cv2), or None when it cannot be read."""
    video = Path(video)
    if not video.is_file():
        return None
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(video))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    return fps if fps > 0 else None


def _readme_video(folder: Path) -> Path | None:
    """The ``video:`` line of the run's README.txt (the source video of the FreeKiki run)."""
    readme = folder / "README.txt"
    if not readme.is_file():
        return None
    for line in readme.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lower().startswith("video:"):
            return Path(line.split(":", 1)[1].strip())
    return None


def detect_fps_source(folder: Path | str, default: float = DEFAULT_FPS) -> tuple[float, str]:
    """``(fps, source)``: summary / flight JSON key, else the README video, else the overlay video.

    ``source`` is ``json:<file>``, ``video``, ``overlay_video`` or ``default``.
    """
    folder = Path(folder)
    for name in ("summary.json", "ball_flight.json"):
        path = resolve(folder, name)
        if not path.is_file():
            path = folder / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for key in ("fps", "rate_hz", "frame_rate"):
            val = data.get(key)
            if val is not None:
                try:
                    fps = float(val)
                except (TypeError, ValueError):
                    continue
                if fps > 0:
                    return fps, f"json:{name}"
    video = _readme_video(folder)
    if video is not None:
        fps = video_fps(video)
        if fps:
            return fps, "video"
    for overlay in sorted(folder.glob("*_3d_overlay.mp4")):
        fps = video_fps(overlay)
        if fps:
            return fps, "overlay_video"
    return float(default), "default"


def detect_fps(folder: Path | str, default: float = DEFAULT_FPS) -> float:
    """Best-effort fps (see :func:`detect_fps_source`); else ``default``."""
    return detect_fps_source(folder, default)[0]


def _hampel_mask(xy: np.ndarray, half: int, n_sigma: float, floor_m: float) -> np.ndarray:
    """Boolean outlier mask: residual from a line through each point's nearest observed
    neighbours (``half`` on each side, across gaps).

    Removes the single worst point per pass while it exceeds ``n_sigma`` robust SDs (global MAD
    of the residuals, at least ``floor_m``), at most a quarter of the points.
    """
    pts = xy.copy()
    n = len(pts)
    bad = np.zeros(n, dtype=bool)
    for _ in range(max(1, n // 4)):
        res = np.full(n, np.nan)
        ok = np.isfinite(pts).all(axis=1)
        good = np.flatnonzero(ok)
        for i in good:
            before = [int(j) for j in good[good < i][-half:]]
            after = [int(j) for j in good[good > i][:half]]
            nb = before + after
            if len(nb) < 2:
                continue
            t = np.asarray(nb, dtype=float)
            pred = [np.polyval(np.polyfit(t, pts[nb, k], 1), float(i)) for k in range(2)]
            res[i] = float(np.hypot(pts[i, 0] - pred[0], pts[i, 1] - pred[1]))
        finite = res[np.isfinite(res)]
        if finite.size < 4:
            break
        scale = max(1.4826 * float(np.median(np.abs(finite - np.median(finite)))), floor_m)
        worst = int(np.nanargmax(res))
        if res[worst] - float(np.median(finite)) <= n_sigma * scale:
            break
        bad[worst] = True
        pts[worst] = np.nan
    return bad


def smooth_track(
    xy: np.ndarray,
    fps: float,
    *,
    cutoff_hz: float = 2.0,
    order: int = 4,
    hampel_half: int = 2,
    n_sigma: float = 3.0,
    floor_m: float = 0.3,
) -> np.ndarray:
    """Outlier-rejected, gap-filled, low-pass ``(n, 2)`` track (NaN outside the observed span).

    Hampel-type rejection against a local line, linear fill of interior gaps, then a zero-lag
    Butterworth low-pass (scipy ``sosfiltfilt`` with odd padding) when the span is long enough.
    """
    pts = _finite_xy(xy)
    n = len(pts)
    out = np.full((n, 2), np.nan)
    if n == 0:
        return out
    pts[_hampel_mask(pts, hampel_half, n_sigma, floor_m)] = np.nan
    ok = np.flatnonzero(np.isfinite(pts).all(axis=1))
    if ok.size == 0:
        return out
    lo, hi = int(ok[0]), int(ok[-1])
    idx = np.arange(lo, hi + 1)
    span = np.column_stack([np.interp(idx, ok, pts[ok, k]) for k in range(2)])
    if fps > 0 and cutoff_hz < 0.5 * fps and len(span) >= 8:
        from scipy.signal import butter, sosfiltfilt

        sos = butter(order, cutoff_hz / (0.5 * fps), btype="low", output="sos")
        # odd padding keeps the trend at the clip edges (an even reflection would force the
        # speed to zero in the first and last frames)
        span = sosfiltfilt(sos, span, axis=0, padtype="odd", padlen=min(15, len(span) - 1))
    out[lo : hi + 1] = span
    return out


def align_n(*arrays: np.ndarray) -> int:
    """Common length for several ``(n, …)`` arrays (max length; callers pad by slicing)."""
    return max((len(a) for a in arrays), default=0)


def pad_to(xy: np.ndarray, n: int) -> np.ndarray:
    """Pad or truncate an ``(m, 2)`` track to length ``n``."""
    pts = _finite_xy(xy) if len(xy) else np.zeros((0, 2))
    if len(pts) == n:
        return pts
    out = np.full((n, 2), np.nan)
    m = min(n, len(pts))
    out[:m] = pts[:m]
    return out


def pair_approach(a: np.ndarray, b: np.ndarray, fps: float) -> dict:
    """Minimum interpersonal distance and relative speed at that instant."""
    dist = interpersonal_distance(a, b)
    rel = relative_speed(a, b, fps)
    finite = np.isfinite(dist)
    if not finite.any():
        return {
            "min_distance_m": None,
            "min_frame": None,
            "min_t_s": None,
            "rel_speed_at_min_mps": None,
            "distance_m": [None if not np.isfinite(x) else float(x) for x in dist],
            "rel_speed_mps": [None if not np.isfinite(x) else float(x) for x in rel],
        }
    i = int(np.nanargmin(dist))
    return {
        "min_distance_m": float(dist[i]),
        "min_frame": i,
        "min_t_s": float(i / fps) if fps > 0 else None,
        "rel_speed_at_min_mps": None if not np.isfinite(rel[i]) else float(rel[i]),
        "distance_m": [None if not np.isfinite(x) else float(x) for x in dist],
        "rel_speed_mps": [None if not np.isfinite(x) else float(x) for x in rel],
    }


def write_json(folder: Path | str, name: str, payload: dict) -> Path:
    """Write ``data/<name>`` under the analysis folder."""
    path = analysis_data_dir(folder) / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
