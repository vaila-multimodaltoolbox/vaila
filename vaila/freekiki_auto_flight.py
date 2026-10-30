"""
================================================================================
Script: freekiki_auto_flight.py - monocular ball flight with automatic start and arrival
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.13
Created: 27 October 2026
Update Date: 29 October 2026

Description:
    The monocular ball is placed on the ground plane, so a ball in the air leaves the pitch. A
    flight fitted to its pixels, with the two real ends of the kick as anchors, fixes that:

    * start: the kick contact, the first frame in which the ball leaves its rest position. The
      flight is counted from there. The rest point (the median of the still frames before the
      contact, where the ball lies on the ground and its lift is exact) is the only start anchor;
      the kick time is free in the fit.
    * arrival: the goal plane is reached (x = +/-52.45 m) inside the goal mouth: the entry point
      is the ball on the plane at the crossing time, mapped with the goal homography of
      :mod:`vaila.pynalty_analysis` (``dlt2d`` / ``rec2d``) from the four goal corners projected
      by the frame's projection. Otherwise (a save) the arrival is the goalkeeper's hand, the mean
      of the wrists of the player nearest the goal in the last frame, and it is an anchor of a
      second pass (it sits on an integer frame, the last one).

    The entry is not an anchor: the crossing time comes from the fit, which is continuous in
    time, while an anchor sits on a whole frame and would push the fit by up to half a frame
    (about 0.4 m at ball speed).
"""

from __future__ import annotations

import numpy as np

try:
    from .freekiki_field3d import BALL_RADIUS_M, FEET_NAMES
    from .freekiki_physics import fit_flight, seed_flight
    from .monocular_dlt_align import dlt_projection_matrix
    from .pynalty_analysis import dlt2d, rec2d
except ImportError:
    from freekiki_field3d import (  # ty: ignore[unresolved-import]
        BALL_RADIUS_M,
        FEET_NAMES,
    )
    from freekiki_physics import fit_flight, seed_flight  # ty: ignore[unresolved-import]
    from monocular_dlt_align import dlt_projection_matrix  # ty: ignore[unresolved-import]
    from pynalty_analysis import dlt2d, rec2d  # ty: ignore[unresolved-import]

GOAL_X_M = 52.45  # the goal line (x), as in soccerfield_kiki.csv
GOAL_HALF_M = 3.66  # half the goal width
GOAL_HEIGHT_M = 2.44  # the crossbar
GOAL_MARGIN_M = 0.2  # tolerance on the entry point (ball radius and calibration)
ONSET_SHIFT_PX = 5.0  # movement from the rest pixel that marks the kick contact
REST_FRAMES = 5  # frames used for the rest pixel
ARRIVAL_ENTRY_SIGMA_M = 0.2
ARRIVAL_KEEPER_SIGMA_M = 0.3
PIXEL_REJECT_PX = 3.0  # a fit above this RMS is reported as not reliable
WALL_DISTANCE_M = 9.15  # the wall stands 9.15 m from the ball (FIFA rule)
PITCH_HALF_X_M = 52.45
PITCH_HALF_Y_M = 33.95


def kick_onset(
    pixels: dict[int, tuple[float, float]],
    *,
    min_shift_px: float = ONSET_SHIFT_PX,
    rest_frames: int = REST_FRAMES,
) -> int | None:
    """The first frame in which the ball has moved ``min_shift_px`` from its rest pixel.

    The rest pixel is the median of the first ``rest_frames`` observed frames. None when the
    ball never moves in the record.
    """
    frames = sorted(pixels)
    if len(frames) < rest_frames + 1:
        return None
    rest = np.median([pixels[f] for f in frames[:rest_frames]], axis=0)
    for f in frames[rest_frames:]:
        if float(np.hypot(*(np.asarray(pixels[f]) - rest))) > min_shift_px:
            return int(f)
    return None


def _pixel_at(pixels: dict[int, tuple[float, float]], frame: float) -> np.ndarray | None:
    """The pixel at a fractional frame, linear between the two observed frames."""
    lo = int(np.floor(frame))
    w = frame - lo
    a = pixels.get(lo)
    if abs(w) < 1e-9:
        return None if a is None else np.asarray(a, dtype=float)
    b = pixels.get(lo + 1)
    if a is None or b is None:
        return None
    return (1.0 - w) * np.asarray(a, dtype=float) + w * np.asarray(b, dtype=float)


def rest_point(lifted: dict[int, dict], onset: int) -> np.ndarray:
    """The ball's rest point on the ground: the median of the still frames before the contact.

    The ball does not move there, so its lift to the ground is exact. The contact frame is the
    fallback when no frame before it has the ball.
    """
    still = [(float(r["x_m"]), float(r["y_m"])) for f, r in lifted.items() if f < onset]
    if still:
        return np.median(np.asarray(still, dtype=float), axis=0)
    return np.array([float(lifted[onset]["x_m"]), float(lifted[onset]["y_m"])])


def goal_plane_point(P: np.ndarray, uv: np.ndarray, x_goal: float) -> np.ndarray | None:
    """``(y, z)`` in metres where the pixel ``uv`` meets the goal plane ``x = x_goal``.

    ``P`` is the frame's 3x4 projection. The four goal corners (the posts' base and top,
    ``y = +/-3.66``, ``z = 0`` and ``2.44``) are projected by ``P``; their pixels give the
    homography of the plane (the same :func:`vaila.pynalty_analysis.dlt2d` used by the penalty
    tool), and ``rec2d`` maps the ball's pixel through it. Exact only for a point on the plane: the
    ball centre at the crossing.
    """
    corners_yz = np.array(
        [
            [-GOAL_HALF_M, 0.0],
            [-GOAL_HALF_M, GOAL_HEIGHT_M],
            [GOAL_HALF_M, GOAL_HEIGHT_M],
            [GOAL_HALF_M, 0.0],
        ]
    )
    pixels = []
    for y, z in corners_yz:
        h = P @ np.array([x_goal, y, z, 1.0])
        if h[2] <= 1e-9:
            return None
        pixels.append([h[0] / h[2], h[1] / h[2]])
    H = dlt2d(corners_yz, np.asarray(pixels, dtype=float))
    yz = rec2d(H, np.asarray(uv, dtype=float).reshape(1, 2))[0]
    if not np.isfinite(yz).all():
        return None
    return yz


def inside_goal_mouth(yz: np.ndarray) -> bool:
    """True when the point is inside the goal mouth, with the entry margin."""
    y, z = float(yz[0]), float(yz[1])
    return (
        abs(y) <= GOAL_HALF_M + GOAL_MARGIN_M
        and -GOAL_MARGIN_M <= z <= GOAL_HEIGHT_M + GOAL_MARGIN_M
    )


def keeper_hand(pose_rows: list[dict], frame: int, goal_x: float) -> np.ndarray | None:
    """The goalkeeper's hand at ``frame``: the mean of the wrists of the player whose feet are
    nearest the goal centre. None when no player has both feet and a wrist in that frame."""
    people: dict[int, dict[str, np.ndarray]] = {}
    for row in pose_rows:
        if int(row["frame"]) != frame or not np.isfinite(row["z_m"]):
            continue
        people.setdefault(int(row["person_id"]), {})[str(row["kpt_name"])] = np.array(
            [row["x_m"], row["y_m"], row["z_m"]], dtype=float
        )
    best: tuple[float, np.ndarray] | None = None
    for parts in people.values():
        feet = [parts[name] for name in FEET_NAMES if name in parts]
        wrists = [parts[name] for name in ("left-wrist", "right-wrist") if name in parts]
        if not feet or not wrists:
            continue
        ground = np.mean(np.array(feet), axis=0)
        distance = float(np.hypot(ground[0] - goal_x, ground[1]))
        if best is None or distance < best[0]:
            best = (distance, np.mean(np.array(wrists), axis=0))
    return None if best is None else best[1]


def auto_flight(
    ball_all: list[dict],
    pixels: dict[int, tuple[float, float]],
    cameras: dict[int, np.ndarray],
    pose_rows: list[dict],
    fps: float,
    n_frames: int,
) -> dict | None:
    """The monocular flight with the automatic start and arrival, or None when the kick cannot
    be found (no movement, or no camera for the start frame).

    ``ball_all`` is the lifted track (:func:`freekiki_field3d.ball_3d`): the rest point, the seed
    and the side of the goal are read from it. ``cameras`` maps frame -> the 11-coefficient DLT, as everywhere in
    the pipeline; it is turned into the 3x4 projections the fit and the goal homography use.
    Returns ``{"fit", "onset_frame", "side", "arrival", "points", "barrier", "rms_flag"}``.
    """
    projections = {f: dlt_projection_matrix(L) for f, L in cameras.items()}
    onset = kick_onset(pixels)
    if onset is None:
        return None
    lifted = {int(r["frame"]): r for r in ball_all if np.isfinite(r["x_m"])}
    start = lifted.get(onset)
    if start is None:
        return None
    last_frame = max(pixels)
    rest_xy = rest_point(lifted, onset)
    # the contact is anywhere before the last frame. A window around the first moved frame forced the
    # fit to a wrong kick time: a hand-marked ball drifts a few frames before the real contact, and on
    # a real penalty that window gave 28 px rms and a 13.6 m anchor residual (2.3 px with this window)
    kick_window = (0.0, (last_frame - 1) / fps)
    x0 = float(start["x_m"])
    after = {f: uv for f, uv in pixels.items() if f >= onset}
    frames = np.array(sorted(f for f in lifted if f >= onset), dtype=float)
    xyz = np.array([[lifted[int(f)]["x_m"], lifted[int(f)]["y_m"], BALL_RADIUS_M] for f in frames])
    seed = seed_flight(frames, xyz, fps, start_frame=onset)
    side = 1.0
    if last_frame in lifted:
        dx = float(lifted[last_frame]["x_m"]) - x0
        side = -1.0 if dx < 0 else 1.0
    start_anchor = {
        "frame": int(onset),
        "xyz": [float(rest_xy[0]), float(rest_xy[1]), BALL_RADIUS_M],
        "label": "start: contato do pé (início do movimento)",
    }
    first = fit_flight(
        {f: uv for f, uv in after.items() if f in projections},
        projections,
        fps,
        n_frames,
        anchors={"rest": rest_xy},
        theta0=seed,
        kick_window=kick_window,
    )
    positions = first["positions"]
    crossing = _goal_crossing(positions, side, onset)
    arrival: dict | None = None
    if crossing is not None:
        _, frame_f = crossing
        uv_c = _pixel_at(pixels, frame_f)
        P_c = projections.get(int(round(frame_f)))
        if uv_c is not None and P_c is not None:
            yz = goal_plane_point(P_c, uv_c, side * GOAL_X_M)
            if yz is not None and inside_goal_mouth(yz):
                arrival = {
                    "type": "entry",
                    "frame": int(round(frame_f)),
                    "xyz": [side * GOAL_X_M, float(yz[0]), float(yz[1])],
                    "sigma": ARRIVAL_ENTRY_SIGMA_M,
                    "label": "chegada: entrada no gol (plano da baliza)",
                }
    if arrival is None and last_frame in projections:
        hand = keeper_hand(pose_rows, last_frame, side * GOAL_X_M)
        if hand is not None:
            arrival = {
                "type": "keeper_hand",
                "frame": int(last_frame),
                "xyz": [float(v) for v in hand],
                "sigma": ARRIVAL_KEEPER_SIGMA_M,
                "label": "chegada: mão do goleiro no último quadro",
            }
    final = first
    keeper_points: list[dict] = []
    if arrival is not None and arrival["type"] == "keeper_hand":
        keeper_points = [{k: arrival[k] for k in ("frame", "xyz", "sigma", "label")}]
        final = fit_flight(
            {f: uv for f, uv in after.items() if f in projections},
            projections,
            fps,
            n_frames,
            anchors={"rest": rest_xy, "points": keeper_points},
            theta0=np.asarray(first["theta"], dtype=float),
            kick_window=kick_window,
        )
    return {
        "fit": final,
        "onset_frame": int(onset),
        "side": side,
        "arrival": arrival,
        "start": start_anchor,
        "points": keeper_points,
        "barrier": barrier_point(final["positions"], onset),
        "rms_flag": final["rms_px"] is not None and final["rms_px"] > PIXEL_REJECT_PX,
    }


def barrier_point(positions: dict[int, tuple[float, float, float]], onset: int) -> dict:
    """The wall plane: the fitted ball is 9.15 m from its position at the kick contact.

    Reported as a measured point of the flight (the rule of the wall distance), not as an anchor.
    """
    if onset not in positions:
        return {}
    x0, y0 = positions[onset][0], positions[onset][1]
    for f in sorted(f for f in positions if f >= onset):
        x, y, z = positions[f]
        if float(np.hypot(x - x0, y - y0)) >= WALL_DISTANCE_M:
            return {
                "frame": int(f),
                "xyz": [float(x), float(y), float(z)],
                "label": "barreira: plano a 9,15 m do chute (regra FIFA)",
            }
    return {}


def auto_rows_and_record(result: dict, pixels: dict[int, tuple[float, float]], n_frames: int):
    """The ball rows of every frame (``ball_3d_on_field.csv``) and the JSON record
    (``ball_flight.json``) of an automatic flight, in the layout the report reads."""
    fit = result["fit"]
    positions = fit["positions"]
    rows: list[dict] = []
    for f in range(n_frames):
        x, y, z = positions[f]
        uv = pixels.get(f)
        rows.append(
            {
                "frame": f,
                "cx": round(float(uv[0]), 2) if uv is not None else "",
                "cy": round(float(uv[1]), 2) if uv is not None else "",
                "score": 1.0 if uv is not None else "",
                "x_m": float(x),
                "y_m": float(y),
                "z_m": float(z),
                "on_field": abs(x) <= PITCH_HALF_X_M and abs(y) <= PITCH_HALF_Y_M,
            }
        )
    record = {k: v for k, v in fit.items() if k not in ("positions", "per_frame_px")}
    record["per_frame_px"] = {str(f): round(e, 3) for f, e in fit["per_frame_px"].items()}
    record["method"] = "monocular_auto"
    record["anchor_mode"] = "auto"
    record["onset_frame"] = result["onset_frame"]
    record["side"] = result["side"]
    record["arrival"] = result["arrival"]
    anchors = {"p0": {k: result["start"][k] for k in ("label", "frame", "xyz")}}
    if result["barrier"]:
        anchors["p_wall"] = result["barrier"]
    if result["arrival"] is not None and result["arrival"]["type"] == "keeper_hand":
        anchors["p_arrival"] = {k: result["arrival"][k] for k in ("label", "frame", "xyz")}
    record["anchor_points"] = anchors
    record["quality"] = "rms above the reject limit" if result["rms_flag"] else "ok"
    return rows, record


def _goal_crossing(
    positions: dict[int, tuple[float, float, float]], side: float, onset: int
) -> tuple[int, float] | None:
    """The first frame after the onset where the fitted ball reaches the goal plane, and the
    fractional frame of the crossing. None when the fitted ball never reaches the plane."""
    frames = sorted(f for f in positions if f >= onset)
    for prev, cur in zip(frames, frames[1:], strict=False):
        x_prev = side * positions[prev][0]
        x_cur = side * positions[cur][0]
        if x_prev < GOAL_X_M <= x_cur:
            w = (GOAL_X_M - x_prev) / (x_cur - x_prev)
            return int(cur), float(prev + w * (cur - prev))
    return None
