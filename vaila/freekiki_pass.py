"""
================================================================================
Script: freekiki_pass.py - two-player pass analysis on FreeKiki open-play runs
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
    Analyse a filmed pass between two typed SAM ``person_id`` tracks and the ball:

    * planar ball / passer / receiver path length, speed, accel / decel;
    * pairwise distances over time (passer–ball, receiver–ball, passer–receiver);
    * release / receive window: release ≈ local minimum of passer–ball distance followed by a
      rise in ball speed; receive ≈ minimum of receiver–ball distance after release.

    Writes ``data/pass_analysis.json``. Framing: Wilson / Santiago passing & control skill line
    (IJSPP 2026) applied as clip kinematics, not rebound-board closed skills.
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
    from .freekiki_openplay_kinematics import (
        ball_source,
        detect_fps_source,
        interpersonal_distance,
        load_ball_xy,
        load_xy_series,
        pad_to,
        path_length,
        smooth_track,
        speed,
        track_metrics,
        write_json,
    )
except ImportError:
    from freekiki_layout import data_dir as analysis_data_dir  # ty: ignore[unresolved-import]
    from freekiki_layout import resolve  # ty: ignore[unresolved-import]
    from freekiki_openplay_kinematics import (  # ty: ignore[unresolved-import]
        ball_source,
        detect_fps_source,
        interpersonal_distance,
        load_ball_xy,
        load_xy_series,
        pad_to,
        path_length,
        smooth_track,
        speed,
        track_metrics,
        write_json,
    )

ANALYSIS_NAME = "pass_analysis.json"
TIMESERIES_NAME = "pass_timeseries.csv"
TOUCH_NEAR_M = 1.5  # operational: ball within this of a foot counts as a touch candidate
RUN_ONSET_MPS = 2.0  # operational: the receiver "is running" at or above this planar speed
A_MAX_MPS2 = 5.0  # typical peak sprint-start acceleration, for the run-onset lower bound
A_MAX_RANGE_MPS2 = (4.0, 7.0)  # sensitivity of that bound
BALL_MOVING_MPS = 3.0  # the ball counts as already moving in the first frames above this


def _strip_series(metrics: dict, *, keep_series: bool) -> dict:
    if keep_series:
        return metrics
    skip = {"speed_mps", "accel_mps2", "heading_rad", "curvature_rad_per_m", "xy_m", "cod_events"}
    out = {k: v for k, v in metrics.items() if k not in skip}
    out["cod_events"] = metrics.get("cod_events") or []
    return out


def _local_minima(d: np.ndarray) -> list[int]:
    """Interior indices that are finite local minima."""
    out = []
    for i in range(1, len(d) - 1):
        if not np.isfinite(d[i]):
            continue
        left = d[i - 1] if np.isfinite(d[i - 1]) else d[i] + 1.0
        right = d[i + 1] if np.isfinite(d[i + 1]) else d[i] + 1.0
        if d[i] <= left and d[i] <= right:
            out.append(i)
    return out


def find_pass_window(
    passer_xy: np.ndarray,
    receiver_xy: np.ndarray,
    ball_xy: np.ndarray,
    fps: float,
    *,
    near_m: float = TOUCH_NEAR_M,
) -> dict:
    """Heuristic release / receive frames from pairwise distances and ball speed."""
    d_pass = interpersonal_distance(passer_xy, ball_xy)
    d_recv = interpersonal_distance(receiver_xy, ball_xy)
    d_pair = interpersonal_distance(passer_xy, receiver_xy)
    v_ball = speed(ball_xy, fps)

    release = None
    # prefer a near-ball local min followed by a speed rise
    mins = [i for i in _local_minima(d_pass) if d_pass[i] <= near_m]
    if not mins:
        mins = _local_minima(d_pass)
    best = None
    for i in mins:
        # look ahead ~0.25 s for a ball-speed increase
        ahead = int(max(2, round(0.25 * fps)))
        j1, j2 = i + 1, min(len(v_ball), i + 1 + ahead)
        window = v_ball[j1:j2]
        finite = window[np.isfinite(window)]
        boost = (
            float(finite.max() - (v_ball[i] if np.isfinite(v_ball[i]) else 0.0))
            if finite.size
            else -1.0
        )
        score = boost - float(d_pass[i])  # prefer close + accelerating ball
        if best is None or score > best[0]:
            best = (score, i)
    if best is not None:
        release = int(best[1])
    elif np.isfinite(d_pass).any():
        release = int(np.nanargmin(d_pass))

    receive = None
    start = (release + 1) if release is not None else 0
    recv_mins = [
        i
        for i in _local_minima(d_recv)
        if i >= start and np.isfinite(d_recv[i]) and d_recv[i] <= near_m
    ]
    if not recv_mins:
        recv_mins = [i for i in _local_minima(d_recv) if i >= start]
    if recv_mins:
        receive = int(min(recv_mins, key=lambda i: float(d_recv[int(i)])))
    elif start < len(d_recv) and np.isfinite(d_recv[start:]).any():
        receive = int(start + int(np.nanargmin(d_recv[start:])))

    pass_length = None
    flight_s = None
    if release is not None and receive is not None and receive > release:
        if np.isfinite(ball_xy[release]).all() and np.isfinite(ball_xy[receive]).all():
            pass_length = float(np.linalg.norm(ball_xy[receive] - ball_xy[release]))
        flight_s = float((receive - release) / fps) if fps > 0 else None
        # also accumulate path length of the ball between release and receive
        path = 0.0
        prev = None
        for f in range(release, receive + 1):
            if not np.isfinite(ball_xy[f]).all():
                continue
            if prev is not None:
                path += float(np.linalg.norm(ball_xy[f] - prev))
            prev = ball_xy[f]
        ball_path = path if prev is not None else None
    else:
        ball_path = None

    return {
        "release_frame": release,
        "receive_frame": receive,
        "release_t_s": None if release is None or fps <= 0 else float(release / fps),
        "receive_t_s": None if receive is None or fps <= 0 else float(receive / fps),
        "pass_chord_m": pass_length,
        "ball_path_release_to_receive_m": ball_path,
        "flight_s": flight_s,
        "passer_ball_at_release_m": None
        if release is None or not np.isfinite(d_pass[release])
        else float(d_pass[release]),
        "receiver_ball_at_receive_m": None
        if receive is None or not np.isfinite(d_recv[receive])
        else float(d_recv[receive]),
        "passer_receiver_at_release_m": None
        if release is None or not np.isfinite(d_pair[release])
        else float(d_pair[release]),
        "distance_passer_ball_m": [None if not np.isfinite(x) else float(x) for x in d_pass],
        "distance_receiver_ball_m": [None if not np.isfinite(x) else float(x) for x in d_recv],
        "distance_passer_receiver_m": [None if not np.isfinite(x) else float(x) for x in d_pair],
        "ball_speed_mps": [None if not np.isfinite(x) else float(x) for x in v_ball],
    }


def _f(x) -> float | None:
    """JSON-safe float (None for NaN / missing)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _first_finite(values: np.ndarray, start: int = 0) -> int | None:
    for i in range(max(0, start), len(values)):
        if np.isfinite(values[i]).all():
            return i
    return None


def speed_headline_kmh(launch_kmh: float | None, mean_kmh: float | None = None) -> int | None:
    """One round number for TV: the ball speed at the start (off the foot, from the flight fit),
    rounded to the nearest 5 km/h; the mean over the path when there is no start value.

    After the kick the ball only loses speed (drag), so the start value is the largest one and
    a per-frame "peak" taken after it is a sampling artefact, not a reference.
    """
    for value in (launch_kmh, mean_kmh):
        if value is not None and math.isfinite(float(value)):
            return int(5 * round(float(value) / 5))
    return None


def _read_flight(folder: Path) -> dict:
    path = resolve(folder, "ball_flight.json")
    if not path.is_file():
        path = folder / "ball_flight.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def pass_timing(
    passer_xy: np.ndarray,
    receiver_xy: np.ndarray,
    ball_xy: np.ndarray,
    fps: float,
    *,
    kick_time_s: float | None = None,
    heuristic_release_frame: int | None = None,
    arrival_frame: int | None = None,
    onset_mps: float = RUN_ONSET_MPS,
    a_max: float = A_MAX_MPS2,
    a_range: tuple[float, float] = A_MAX_RANGE_MPS2,
) -> dict:
    """Release, arrival and receiver-run timing on smoothed planar tracks.

    Time zero is the ball release. When the ball already moves in the first frames (the kick is
    at or before the clip start), the release time is the flight fit's ``kick_time_s`` (seconds
    from frame 0, ≤ 0); otherwise it is the heuristic release frame. Arrival is the closest
    receiver–ball approach after release, unless ``arrival_frame`` (e.g. the first touch seen in
    the video) is given. The run onset is the first frame at which the receiver
    reaches ``onset_mps``; when he already runs in the first observed frame, only a lower bound
    is possible: reaching speed ``v`` from rest takes at least ``v / a_max``.
    """
    n = len(ball_xy)
    v_ball = speed(ball_xy, fps)
    v_pass = speed(passer_xy, fps)
    v_recv = speed(receiver_xy, fps)
    d_pass = interpersonal_distance(passer_xy, ball_xy)
    d_recv = interpersonal_distance(receiver_xy, ball_xy)
    d_pair = interpersonal_distance(passer_xy, receiver_xy)

    first_v = _first_finite(v_ball)
    early = v_ball[first_v : first_v + 3] if first_v is not None else np.zeros(0)
    moving_at_start = bool(early.size and np.nanmax(early) >= BALL_MOVING_MPS)
    if kick_time_s is not None and moving_at_start and float(kick_time_s) < 1.0 / fps:
        release_t = min(float(kick_time_s), 0.0)
        release_source = "flight_fit"
    elif heuristic_release_frame is not None:
        release_t = float(heuristic_release_frame) / fps
        release_source = "heuristic"
    else:
        release_t = 0.0
        release_source = "first_frame"
    release_frame = max(0, int(math.ceil(release_t * fps - 1e-9)))  # first frame at/after release

    # back-extrapolation cross-check: ball leaving the passer's ground point at its first speed
    back_t = None
    fb = _first_finite(ball_xy, release_frame)
    if fb is not None and first_v is not None and np.isfinite(d_pass[fb]) and v_ball[first_v] > 0:
        back_t = fb / fps - float(d_pass[fb]) / float(v_ball[first_v])

    # arrival: the given frame, else the closest receiver–ball approach after release
    arrival = None
    arrival_source = None
    if arrival_frame is not None:
        if not 0 <= int(arrival_frame) < n:
            raise ValueError(f"arrival frame {arrival_frame} is outside the clip (0..{n - 1})")
        arrival, arrival_source = int(arrival_frame), "user"
    else:
        after = np.arange(n) >= release_frame
        cand = np.where(after & np.isfinite(d_recv), d_recv, np.inf)
        if np.isfinite(cand).any():
            arrival, arrival_source = int(np.argmin(cand)), "closest_approach"
    last_ball = max((i for i in range(n) if np.isfinite(ball_xy[i]).all()), default=None)
    at_end = arrival is not None and last_ball is not None and arrival >= last_ball - 1

    def t_rel(frame: int | None) -> float | None:
        return None if frame is None else float(frame) / fps - release_t

    p_rel = _first_finite(passer_xy, release_frame)
    r_rel = _first_finite(receiver_xy, release_frame)
    out: dict = {
        "fps": fps,
        "release_t_s": release_t,
        "release_source": release_source,
        "release_frame": release_frame,
        "release_t_backextrap_s": _f(back_t),
        "arrival_frame": arrival,
        "arrival_t_s": t_rel(arrival),
        "arrival_source": arrival_source,
        "arrival_at_clip_end": bool(at_end),
        "flight_s": t_rel(arrival),
        "t_rel_s": [float(i) / fps - release_t for i in range(n)],
    }
    passer_rel = passer_xy[p_rel] if p_rel is not None else None
    arrival_pt = ball_xy[arrival] if arrival is not None else None
    out["passer_at_release_xy"] = None if passer_rel is None else [float(v) for v in passer_rel]
    out["receiver_at_release_xy"] = (
        None if r_rel is None else [float(v) for v in receiver_xy[r_rel]]
    )
    out["arrival_xy"] = None if arrival_pt is None else [float(v) for v in arrival_pt]
    out["passer_to_arrival_m"] = (
        None
        if passer_rel is None or arrival_pt is None
        else float(np.linalg.norm(arrival_pt - passer_rel))
    )
    out["passer_receiver_at_release_m"] = (
        _f(d_pair[min(p_rel, r_rel)]) if p_rel is not None and r_rel is not None else None
    )
    out["receiver_ball_at_arrival_m"] = None if arrival is None else _f(d_recv[arrival])
    out["receiver_at_arrival_xy"] = (
        None
        if arrival is None or not np.isfinite(receiver_xy[arrival]).all()
        else [float(v) for v in receiver_xy[arrival]]
    )

    # ball
    observed = None
    if fb is not None and arrival is not None and arrival > fb:
        observed = path_length(ball_xy[fb : arrival + 1])
    vb = v_ball[first_v] if first_v is not None else np.nan
    out["ball"] = {
        "first_frame": fb,
        "chord_first_to_arrival_m": None
        if fb is None or arrival_pt is None
        else float(np.linalg.norm(arrival_pt - ball_xy[fb])),
        "path_observed_m": _f(observed),
        "v_release_mps": _f(vb),
        "v_arrival_mps": None if arrival is None else _f(v_ball[arrival]),
        "v_mean_mps": None
        if observed is None or arrival is None or fb is None or arrival <= fb
        else float(observed / ((arrival - fb) / fps)),
        "v_max_mps": _f(np.nanmax(v_ball)) if np.isfinite(v_ball).any() else None,
    }

    # passer
    out["passer"] = {
        "v_release_mps": _f(v_pass[_first_finite(v_pass, release_frame) or 0])
        if np.isfinite(v_pass).any()
        else None,
        "v_max_after_mps": _f(np.nanmax(v_pass[release_frame:]))
        if np.isfinite(v_pass[release_frame:]).any()
        else None,
        "displacement_to_arrival_m": None
        if p_rel is None or arrival is None or not np.isfinite(passer_xy[arrival]).all()
        else float(np.linalg.norm(passer_xy[arrival] - passer_xy[p_rel])),
    }

    # receiver run
    fv = _first_finite(v_recv)
    run: dict = {
        "threshold_mps": onset_mps,
        "v_first_mps": None if fv is None else _f(v_recv[fv]),
        "t_first_s": t_rel(fv),
        "v_arrival_mps": None if arrival is None else _f(v_recv[arrival]),
        "v_peak_mps": _f(np.nanmax(v_recv)) if np.isfinite(v_recv).any() else None,
        "t_peak_s": t_rel(int(np.nanargmax(v_recv))) if np.isfinite(v_recv).any() else None,
        "displacement_m": None
        if r_rel is None or arrival is None or not np.isfinite(receiver_xy[arrival]).all()
        else float(np.linalg.norm(receiver_xy[arrival] - receiver_xy[r_rel])),
        "path_m": None
        if r_rel is None or arrival is None
        else _f(path_length(receiver_xy[r_rel : arrival + 1])),
        "onset_frame": None,
        "onset_t_s": None,
        "onset_before_release": None,
        "onset_observed": False,
        "lead_lower_bound_s": None,
        "lead_lower_bound_range_s": None,
        "a_max_mps2": a_max,
    }
    if fv is not None and np.isfinite(v_recv[fv]):
        if v_recv[fv] >= onset_mps:
            v0 = float(v_recv[fv])
            t0 = float(t_rel(fv) or 0.0)
            run["onset_before_release"] = t0 - v0 / a_max < 0.0
            run["lead_lower_bound_s"] = v0 / a_max - t0
            lo, hi = sorted(a_range)
            run["lead_lower_bound_range_s"] = [v0 / hi - t0, v0 / lo - t0]
        else:
            idx = [i for i in range(fv, n) if np.isfinite(v_recv[i]) and v_recv[i] >= onset_mps]
            if idx:
                run["onset_frame"] = idx[0]
                run["onset_t_s"] = t_rel(idx[0])
                run["onset_before_release"] = bool((t_rel(idx[0]) or 0.0) < 0.0)
                run["onset_observed"] = True
    out["receiver_run"] = run
    out["series"] = {
        "ball_speed_mps": [_f(v) for v in v_ball],
        "passer_speed_mps": [_f(v) for v in v_pass],
        "receiver_speed_mps": [_f(v) for v in v_recv],
        "distance_passer_ball_m": [_f(v) for v in d_pass],
        "distance_receiver_ball_m": [_f(v) for v in d_recv],
        "distance_passer_receiver_m": [_f(v) for v in d_pair],
    }
    return out


def shot_link(shot_folder: Path | str, timing: dict) -> dict:
    """Numbers of the shot that follows the pass, from a FreeKiki kick-analysis folder.

    Uses the shot report's own metric code (:func:`freekiki_report.compute_metrics`) so the
    numbers match that report, plus ``data/setpiece.json`` (distance to the goal line) and
    ``data/kicker_metrics.json``. The time between the two clips is not measured (separate
    clips); ``gap_estimate_s`` brackets it from the arrival → shot distance and the ball and
    receiver speeds at arrival.
    """
    folder = Path(shot_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"shot folder not found: {folder}")
    out: dict = {"folder": str(folder)}
    try:
        try:
            from .freekiki_report import compute_metrics, load_run
        except ImportError:
            from freekiki_report import compute_metrics, load_run  # ty: ignore[unresolved-import]
        run = load_run(folder)
        m = compute_metrics(run)
        kf = int(m["kick_frame"])
        ef = int(m["end_frame"])
        point = run.track[kf, :2]
        end = run.track[ef, :2]
        out.update(
            {
                "fps": float(run.fps),
                "shot_xy": [float(point[0]), float(point[1])],
                "shot_end_xy": [float(end[0]), float(end[1])],
                "launch_kmh": _f(m.get("launch_kmh")),
                "peak_kmh": _f(m.get("peak_kmh")),
                "speed_headline_kmh": speed_headline_kmh(m.get("launch_kmh"), m.get("mean_kmh")),
                "mean_kmh": _f(m.get("mean_kmh")),
                "end_kmh": _f(m.get("end_kmh")),
                "flight_s": _f(m.get("flight_s")),
                "distance_m": _f(m.get("distance_m")),
                "zmax_m": _f(m.get("zmax_m")),
                "spin_rps": _f(m.get("spin_rps")),
                "arrival_status": m.get("arrival_status"),
                "source": "freekiki_report.compute_metrics",
            }
        )
    except Exception as exc:  # the shot run may lack a flight fit; keep the set-piece numbers
        out["metrics_error"] = f"{type(exc).__name__}: {exc}"
    data = folder / "data"
    setpiece = data / "setpiece.json"
    if setpiece.is_file():
        try:
            sp = json.loads(setpiece.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            sp = {}
        out["distance_to_goal_m"] = _f(sp.get("distance_to_goal_m"))
        out["goal_x_m"] = _f(sp.get("goal_x_m"))
        if "shot_xy" not in out and sp.get("ball_xy_m"):
            out["shot_xy"] = [float(v) for v in sp["ball_xy_m"]]
    kicker = data / "kicker_metrics.json"
    if kicker.is_file():
        try:
            km = json.loads(kicker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            km = {}
        metrics = km.get("metrics") or {}
        events = km.get("events") or {}
        out["kicker_id"] = events.get("kicker_id")
        out["kick_foot"] = metrics.get("kick_foot") or events.get("kick_foot")
        out["v_ball_post_mps"] = _f(metrics.get("v_ball_post_mps") or None)
    shot = out.get("shot_xy")
    arrival = timing.get("arrival_xy")
    passer = timing.get("passer_at_release_xy")
    if shot is not None and arrival is not None:
        d = float(math.dist(shot, arrival))
        out["arrival_to_shot_m"] = d
        v_ball = (timing.get("ball") or {}).get("v_arrival_mps")
        v_recv = (timing.get("receiver_run") or {}).get("v_arrival_mps")
        speeds = [v for v in (v_ball, v_recv) if v and v > 0.5]
        out["gap_estimate_s"] = sorted(d / v for v in speeds) if speeds else None
    if shot is not None and passer is not None:
        out["passer_to_shot_m"] = float(math.dist(shot, passer))
    return out


def write_timeseries(
    folder: Path | str,
    passer_id: int,
    receiver_id: int,
    ball: np.ndarray,
    passer: np.ndarray,
    receiver: np.ndarray,
    timing: dict,
) -> Path:
    """``data/pass_timeseries.csv``: one row per frame (positions, speeds, distances)."""
    path = analysis_data_dir(folder) / TIMESERIES_NAME
    series = timing["series"]
    head = [
        "frame",
        "t_rel_release_s",
        "ball_x_m",
        "ball_y_m",
        f"p{passer_id}_x_m",
        f"p{passer_id}_y_m",
        f"p{receiver_id}_x_m",
        f"p{receiver_id}_y_m",
        "ball_speed_mps",
        f"p{passer_id}_speed_mps",
        f"p{receiver_id}_speed_mps",
        "passer_ball_m",
        "receiver_ball_m",
        "passer_receiver_m",
    ]

    def cell(v) -> str:
        return "" if v is None or not math.isfinite(float(v)) else f"{float(v):.4f}"

    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        for i, t in enumerate(timing["t_rel_s"]):
            w.writerow(
                [
                    i,
                    cell(t),
                    cell(ball[i, 0]),
                    cell(ball[i, 1]),
                    cell(passer[i, 0]),
                    cell(passer[i, 1]),
                    cell(receiver[i, 0]),
                    cell(receiver[i, 1]),
                    cell(series["ball_speed_mps"][i]),
                    cell(series["passer_speed_mps"][i]),
                    cell(series["receiver_speed_mps"][i]),
                    cell(series["distance_passer_ball_m"][i]),
                    cell(series["distance_receiver_ball_m"][i]),
                    cell(series["distance_passer_receiver_m"][i]),
                ]
            )
    return path


def analyze_pass(
    folder: Path | str,
    passer_id: int,
    receiver_id: int,
    *,
    fps: float | None = None,
    keep_series: bool = True,
    ball: str = "auto",
    smooth: bool = True,
    shot_folder: Path | str | None = None,
    arrival_frame: int | None = None,
) -> dict:
    """Planar pass metrics for two SAM person ids + ball.

    ``ball`` picks the ball file (``auto`` / ``ground`` / ``flight``, see
    :func:`freekiki_openplay_kinematics.ball_source`). ``smooth`` uses the raw box-base player
    tracks and the ball track through :func:`smooth_track` (outlier rejection + 2 Hz low-pass).
    ``shot_folder`` links the shot that follows (see :func:`shot_link`). ``arrival_frame``
    (0-based clip frame of the first touch, from the video) replaces the closest-approach rule.
    """
    folder = Path(folder)
    if int(passer_id) == int(receiver_id):
        raise ValueError("passer_id and receiver_id must differ")
    if fps and fps > 0:
        rate, fps_src = float(fps), "user"
    else:
        rate, fps_src = detect_fps_source(folder)
    _, ball_name = ball_source(folder, ball)
    ball_xy = load_ball_xy(folder, source=ball)
    passer = load_xy_series(folder, int(passer_id), raw=smooth)
    receiver = load_xy_series(folder, int(receiver_id), raw=smooth)
    n = max(len(ball_xy), len(passer), len(receiver))
    if n <= 0:
        raise FileNotFoundError(
            f"no ball or person tracks in {folder}: need ball_3d_on_field.csv / ball_3d.csv "
            "and ground_tracks"
        )
    ball_xy = pad_to(ball_xy, n)
    passer = pad_to(passer, n)
    receiver = pad_to(receiver, n)
    if smooth:
        ball_xy = smooth_track(ball_xy, rate)
        passer = smooth_track(passer, rate)
        receiver = smooth_track(receiver, rate)
    window = find_pass_window(passer, receiver, ball_xy, rate)
    flight = _read_flight(folder)
    timing = pass_timing(
        passer,
        receiver,
        ball_xy,
        rate,
        kick_time_s=flight.get("kick_time_s"),
        heuristic_release_frame=window.get("release_frame"),
        arrival_frame=arrival_frame,
    )
    if not keep_series:
        for key in (
            "distance_passer_ball_m",
            "distance_receiver_ball_m",
            "distance_passer_receiver_m",
            "ball_speed_mps",
        ):
            window.pop(key, None)
    result = {
        "kind": "pass",
        "fps": rate,
        "fps_source": fps_src,
        "ball_source": ball_name,
        "smoothed": bool(smooth),
        "n_frames": n,
        "passer_id": int(passer_id),
        "receiver_id": int(receiver_id),
        "ball": _strip_series(track_metrics(ball_xy, rate), keep_series=keep_series),
        "passer": _strip_series(track_metrics(passer, rate), keep_series=keep_series),
        "receiver": _strip_series(track_metrics(receiver, rate), keep_series=keep_series),
        "window": window,
        "timing": timing,
        "notes": (
            "Planar pass metrics from FreeKiki tracks. Time zero is the ball release: the flight "
            "fit's kick time when the ball already moves in the first frame, else the heuristic "
            "(passer–ball min + ball speed rise). Arrival is the closest receiver–ball approach. "
            "Run onset before the clip is a lower bound from v / a_max (estimate). The 'window' "
            "block keeps the older heuristic. Not a rebound-board closed-skill test (Wilson / "
            "Santiago passing line)."
        ),
    }
    if keep_series:
        result["ball"] = track_metrics(ball_xy, rate)
        result["passer"] = track_metrics(passer, rate)
        result["receiver"] = track_metrics(receiver, rate)
    if shot_folder:
        result["shot"] = shot_link(shot_folder, timing)
    result["_arrays"] = (ball_xy, passer, receiver)
    return result


def run_pass(
    folder: Path | str,
    passer_id: int,
    receiver_id: int,
    *,
    fps: float | None = None,
    keep_series: bool = True,
    ball: str = "auto",
    smooth: bool = True,
    shot_folder: Path | str | None = None,
    arrival_frame: int | None = None,
) -> dict:
    """Analyse and write ``data/pass_analysis.json`` + ``data/pass_timeseries.csv``."""
    result = analyze_pass(
        folder,
        passer_id,
        receiver_id,
        fps=fps,
        keep_series=keep_series,
        ball=ball,
        smooth=smooth,
        shot_folder=shot_folder,
        arrival_frame=arrival_frame,
    )
    ball_xy, passer, receiver = result.pop("_arrays")
    ts = write_timeseries(
        folder, int(passer_id), int(receiver_id), ball_xy, passer, receiver, result["timing"]
    )
    result["timeseries"] = str(ts)
    path = write_json(folder, ANALYSIS_NAME, result)
    result["path"] = str(path)
    return result
