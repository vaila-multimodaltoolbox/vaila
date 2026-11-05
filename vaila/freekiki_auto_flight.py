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
Version: 0.4.14
Created: 27 October 2026
Update Date: 02 November 2026

Description:
    The monocular ball is placed on the ground plane, so a ball in the air leaves the pitch. A
    flight fitted to its pixels, with the two real ends of the kick as anchors, fixes that:

    * start: the kick contact, the first frame in which the ball leaves its rest position. The
      flight is counted from there. The rest point (the median of the still frames before the
      contact, where the ball lies on the ground and its lift is exact) is the only start anchor;
      the kick time is free in the fit.
    * arrival: a manual entry uses the observed ball and four confirmed kiki49 goal corners
      on the crossing frame. Its local homography supplies a physical anchor with uncertainty.
      Later frames in the net do not constrain the pre-entry flight. A keeper-hand anchor needs
      explicit confirmation. With neither, a model crossing may be reported; an unrecognized
      arrival remains undetermined.

    The contact is decided once (:func:`find_contact`) and written to ``ball_flight.json``, so the
    player ids, the kicker and the set-piece rule read the same frame (:func:`flight_contact`):

    * ``still`` (a set piece): the ball lies still in the first frames; the contact is
      :func:`kick_onset`, as before.
    * ``rolling`` (chute a gol / ``kick``): the ball rolls into the kick. Its ground track is two straight
      segments (constant velocity) and the contact is the break between them
      (:func:`rolling_break`). The flight starts there; the frames before it are the ball that came
      in (``incoming``), measured on the ground, where the lift is exact.
    * ``unseen``: the ball already flies in the first frame (the kick is at or before it).
    * a contact frame given by hand (``manual``) wins over all of these.
"""

from __future__ import annotations

import json
from pathlib import Path

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
ROLLING_MAX_MPS = 15.0  # a ball rolling into the kick is slower; faster on the ground lift = flying
KINK_SEGMENT = 3  # frames of roll before the break, and of flight after it to read the launch
ROLL_BREAK_M = (
    0.3  # a frame this far (metres) off the rolling line has left it (hand marks: ~0.1 m)
)
KINK_MIN_DV_MPS = 3.0  # the kick speeds the ball up by at least this much...
KINK_MIN_TURN_DEG = 30.0  # ...or turns it by at least this angle


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


def is_still(
    pixels: dict[int, tuple[float, float]],
    *,
    min_shift_px: float = ONSET_SHIFT_PX,
    rest_frames: int = REST_FRAMES,
) -> bool:
    """True when the ball lies still in the first ``rest_frames`` observed frames (a set piece):
    every one of them is within ``min_shift_px`` of their median pixel."""
    frames = sorted(pixels)[:rest_frames]
    if len(frames) < rest_frames:
        return False
    pts = np.array([pixels[f] for f in frames], dtype=float)
    rest = np.median(pts, axis=0)
    return bool(np.all(np.hypot(*(pts - rest).T) <= min_shift_px))


def _line_fit(t: np.ndarray, xy: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """``(sse, velocity per frame, intercept)`` of the straight line (constant velocity) through
    ``xy`` (n x 2) at the frames ``t``, by least squares."""
    A = np.column_stack([np.ones_like(t), t])
    coef, *_ = np.linalg.lstsq(A, xy, rcond=None)
    return float(((A @ coef - xy) ** 2).sum()), coef[1], coef[0]


def _turn_deg(a: np.ndarray, b: np.ndarray) -> float:
    """The angle between two horizontal velocities, in degrees (0 when one of them is ~zero)."""
    na, nb = float(np.hypot(*a[:2])), float(np.hypot(*b[:2]))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    cos = float(np.dot(a[:2], b[:2]) / (na * nb))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _roll_in(
    ground: dict[int, tuple[float, float]], before: list[int], contact: int, fps: float
) -> dict:
    """The ball that rolled into the kick: its straight line over the frames ``before`` the contact.

    ``kick_xy`` is that line at the middle of the last rolling frame and the contact (the kick
    lies between them); ``incoming`` holds its speed, heading and frames.
    """
    t = np.asarray(before, dtype=float)
    _, v, c = _line_fit(t, np.asarray([ground[f] for f in before], dtype=float))
    t_kick = 0.5 * (before[-1] + contact)
    kick_xy = c + v * t_kick
    vel = v * fps
    return {
        "kick_xy": [float(kick_xy[0]), float(kick_xy[1])],
        "incoming": {
            "speed_mps": round(float(np.hypot(*vel)), 2),
            "heading_deg": round(float(np.degrees(np.arctan2(vel[1], vel[0]))), 1),
            "velocity_mps": [round(float(vel[0]), 3), round(float(vel[1]), 3)],
            "frames": [int(before[0]), int(before[-1])],
        },
    }


def rolling_break(ground: dict[int, tuple[float, float]], fps: float) -> dict | None:
    """The kick of a ball that rolls into it: where its ground track leaves the rolling line.

    ``ground`` maps frame -> ``(x, y)``, the ball lifted to the pitch: exact while it rolls (after
    the kick the ball rises and its lift runs away, which only makes it leave the line sooner). The
    rolling line (constant velocity) is fitted to the frames so far; the contact is the first frame
    that is more than ``ROLL_BREAK_M`` off its prediction while the next frame is off too (one badly
    marked frame is not a kick). The change must look like a kick: the first ``KINK_SEGMENT``
    frames after it are faster by ``KINK_MIN_DV_MPS`` or turned by ``KINK_MIN_TURN_DEG``, and the
    roll itself is at most ``ROLLING_MAX_MPS`` (faster on the ground lift is a ball already flying).
    Returns ``{"frame", "kick_xy", "incoming"}``, or None.
    """
    frames = sorted(ground)
    if len(frames) < KINK_SEGMENT + 2:
        return None
    t = np.asarray(frames, dtype=float)
    xy = np.asarray([ground[f] for f in frames], dtype=float)
    for k in range(KINK_SEGMENT, len(frames) - 1):
        _, v, c = _line_fit(t[:k], xy[:k])
        off = [float(np.hypot(*(xy[j] - (c + v * t[j])))) for j in (k, k + 1)]
        if min(off) <= ROLL_BREAK_M:
            continue  # still rolling, or one badly marked frame
        if float(np.hypot(*v)) * fps > ROLLING_MAX_MPS:
            return None  # too fast for a roll: the ball already flies
        after = slice(k, min(k + KINK_SEGMENT, len(frames)))
        _, v_after, _ = _line_fit(t[after], xy[after])
        faster = (float(np.hypot(*v_after)) - float(np.hypot(*v))) * fps >= KINK_MIN_DV_MPS
        if faster or _turn_deg(v, v_after) >= KINK_MIN_TURN_DEG:
            return {"frame": int(frames[k]), **_roll_in(ground, frames[:k], frames[k], fps)}
        return None  # the ball left its line without a kick (it stopped, or a bounce)
    return None


def find_contact(
    pixels: dict[int, tuple[float, float]],
    ground: dict[int, tuple[float, float]],
    fps: float,
    *,
    contact_frame: int | None = None,
) -> dict:
    """The kick contact, decided once for the whole pipeline.

    Returns ``{"frame", "source", "pre_contact", "warning"}`` (plus ``kick_xy`` and ``incoming``
    for a rolling ball). ``source``: ``manual`` (``contact_frame``, given by hand), ``rest`` (a
    still ball, :func:`kick_onset`), ``kink`` (:func:`rolling_break`) or ``first_frame`` (the ball
    already flies in the first frame). ``pre_contact``: ``still``, ``rolling`` or ``unseen``.
    ``frame`` is None when the track is too short or the still ball never moves.
    """
    frames = sorted(pixels)
    if len(frames) < REST_FRAMES + 1:
        return {"frame": None, "source": None, "pre_contact": None,
                "warning": f"fewer than {REST_FRAMES + 1} ball frames"}  # fmt: skip
    still = is_still(pixels)
    if contact_frame is not None:
        contact = int(contact_frame)
        before = [f for f in sorted(ground) if f < contact]
        pre = "still" if still else ("rolling" if len(before) >= 2 else "unseen")
        found: dict = {"frame": contact, "source": "manual", "pre_contact": pre, "warning": None}
        if pre == "rolling":
            found.update(_roll_in(ground, before, contact, fps))
        return found
    if still:
        onset = kick_onset(pixels)
        return {"frame": onset, "source": "rest", "pre_contact": "still",
                "warning": None if onset is not None else "the ball never moves"}  # fmt: skip
    rolled = rolling_break(ground, fps)
    if rolled is not None:
        return {"source": "kink", "pre_contact": "rolling", "warning": None, **rolled}
    return {
        "frame": int(frames[0]),
        "source": "first_frame",
        "pre_contact": "unseen",
        "warning": "the ball already moves fast in the first frame: the kick is at or before it",
    }


def flight_contact(folder) -> dict | None:
    """The contact written by the automatic flight in a 3D folder's ``ball_flight.json``.

    ``{"frame", "source", "pre_contact", "kick_xy", "incoming"}``; None for a folder without
    that file or for an older flight that did not record the contact (the callers then fall back
    to :func:`kick_onset` and :func:`rest_point`).
    """
    path = Path(folder) / "ball_flight.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("contact_frame") is None:
        return None
    xyz = ((data.get("anchor_points") or {}).get("p0") or {}).get("xyz")
    return {
        "frame": int(data["contact_frame"]),
        "source": data.get("contact_source"),
        "pre_contact": data.get("pre_contact"),
        "kick_xy": [float(xyz[0]), float(xyz[1])] if xyz else None,
        "incoming": data.get("incoming"),
    }


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
    *,
    contact_frame: int | None = None,
    arrival_observation: dict | None = None,
) -> dict | None:
    """The monocular flight with the automatic start and arrival, or None when the kick cannot
    be found (no movement, or no camera for the start frame).

    The contact comes from :func:`find_contact` (``contact_frame`` is a contact given by hand).
    A still ball keeps today's fit: the rest point is the start anchor and the kick time is free.
    A rolling ball starts at its kick point on the ground (the end of its incoming line) and the
    kick time is held between the last rolling frame and the contact. A ball already flying in
    the first frame starts at its lift in that frame, with the kick up to a frame before it.

    ``ball_all`` is the lifted track (:func:`freekiki_field3d.ball_3d`): the rest point, the seed
    and the side of the goal are read from it. ``cameras`` maps frame -> the 11-coefficient DLT, as everywhere in
    the pipeline; it is turned into the 3x4 projections the fit and the goal homography use.
    Returns ``{"fit", "onset_frame", "side", "arrival", "points", "barrier", "rms_flag"}``.
    """
    try:
        from .freekiki_arrival import entry_anchor, validate_observation
    except ImportError:
        from freekiki_arrival import entry_anchor, validate_observation  # ty: ignore[unresolved-import]

    observation = (
        validate_observation(arrival_observation, n_frames)
        if arrival_observation is not None
        else None
    )
    projections = {f: dlt_projection_matrix(L) for f, L in cameras.items()}
    arrival: dict | None = None
    if observation is not None:
        frame = observation["frame"]
        if frame not in projections:
            raise ValueError(f"no calibrated camera for observed arrival frame {frame}")
        if observation["type"] == "goal_entry":
            arrival = entry_anchor(observation, projections[frame])
            if not inside_goal_mouth(np.asarray(arrival["xyz"][1:])):
                raise ValueError(
                    f"observed entry is outside the goal mouth: {arrival['xyz']}; review ball/corners/frame"
                )
        else:
            arrival = {
                "type": "keeper_hand",
                "frame": frame,
                "xyz": observation["xyz"],
                "sigma": observation["sigma_m"],
                "label": "chegada: defesa confirmada",
                "provenance": "manual",
                "observation": observation,
            }
        # The observed centre also replaces this frame's old ball marker in the fit.
        pixels = {f: uv for f, uv in pixels.items() if f <= frame}
        if observation["type"] == "goal_entry":
            pixels[frame] = tuple(observation["pixel"])
    lifted = {int(r["frame"]): r for r in ball_all if np.isfinite(r["x_m"])}
    ground = {f: (float(r["x_m"]), float(r["y_m"])) for f, r in lifted.items()}
    contact = find_contact(pixels, ground, fps, contact_frame=contact_frame)
    onset = contact["frame"]
    if onset is None:
        if observation is not None:
            raise ValueError(
                "cannot fit the observed arrival: the marked pre-arrival segment does not establish a kick contact"
            )
        return None
    if arrival is not None and arrival["frame"] <= onset:
        raise ValueError("observed arrival must be after the kick contact")
    start = lifted.get(onset)
    if start is None:
        return None
    last_frame = max(pixels)
    pre = contact["pre_contact"]
    if pre == "still":
        rest_xy = rest_point(lifted, onset)
        # the contact is anywhere before the last frame. A window around the first moved frame forced
        # the fit to a wrong kick time: a hand-marked ball drifts a few frames before the real contact,
        # and on a real penalty that window gave 28 px rms and a 13.6 m anchor residual (2.3 px with
        # this window). A contact given by hand is trusted: the kick is in the frame before it.
        kick_window = (
            (0.0, (last_frame - 1) / fps)
            if contact["source"] == "rest"
            else ((onset - 1) / fps, onset / fps)
        )
    elif pre == "rolling":
        rest_xy = np.asarray(contact["kick_xy"], dtype=float)
        kick_window = ((onset - 1) / fps, onset / fps)
    else:  # unseen: the kick is at the first frame or just before it
        rest_xy = np.array([float(start["x_m"]), float(start["y_m"])])
        kick_window = ((onset - 1) / fps, (onset + 0.5) / fps)
    x0 = float(start["x_m"])
    after = {f: uv for f, uv in pixels.items() if f >= onset}
    frames = np.array(sorted(f for f in lifted if f >= onset), dtype=float)
    xyz = np.array([[lifted[int(f)]["x_m"], lifted[int(f)]["y_m"], BALL_RADIUS_M] for f in frames])
    seed = seed_flight(frames, xyz, fps, start_frame=onset)
    side = 1.0
    if last_frame in lifted:
        dx = float(lifted[last_frame]["x_m"]) - x0
        side = -1.0 if dx < 0 else 1.0
    if observation is not None and observation["type"] == "goal_entry":
        side = float(observation["goal_side"])
    start_anchor = {
        "frame": int(onset),
        "xyz": [float(rest_xy[0]), float(rest_xy[1]), BALL_RADIUS_M],
        "label": {
            "still": "start: contato do pé (início do movimento)",
            "rolling": "start: contato do pé (a bola chegava rolando)",
        }.get(pre, "start: primeiro quadro (o chute é nele ou logo antes)"),
    }
    arrival_points = (
        [] if arrival is None else [{k: arrival[k] for k in ("frame", "xyz", "sigma", "label")}]
    )
    first = fit_flight(
        {f: uv for f, uv in after.items() if f in projections},
        projections,
        fps,
        n_frames,
        anchors={"rest": rest_xy, "points": arrival_points},
        theta0=seed,
        kick_window=kick_window,
    )
    positions = first["positions"]
    crossing = _goal_crossing(positions, side, onset)
    if arrival is None and crossing is not None:
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
                    "provenance": "model",
                }
    final = first
    if observation is not None:
        assert arrival is not None
        pos = np.asarray(final["positions"][observation["frame"]])
        arrival.setdefault("diagnostics", {}).update(
            fit_residual_m=float(np.linalg.norm(pos - np.asarray(arrival["xyz"]))),
            fit_plane_error_m=float(abs(pos[0] - arrival["xyz"][0])),
            fitted_crossing_frame=None if crossing is None else crossing[1],
        )
    incoming = contact.get("incoming")
    if incoming is not None:  # the turn of the ball at the kick: incoming line vs fitted launch
        launch = np.asarray(final["theta"][2:5], dtype=float)
        incoming = {
            **incoming,
            "out_speed_mps": round(float(np.linalg.norm(launch)), 2),
            "turn_deg": round(_turn_deg(np.asarray(incoming["velocity_mps"]), launch), 1),
        }
    return {
        "fit": final,
        "onset_frame": int(onset),
        "contact": {k: contact.get(k) for k in ("frame", "source", "pre_contact", "warning")},
        "incoming": incoming,
        "ground": ground if pre == "rolling" else {},
        "side": side,
        "arrival": arrival,
        "start": start_anchor,
        "points": arrival_points,
        "observation": observation,
        # the wall is a set-piece rule: 9.15 m from a ball that lay still
        "barrier": barrier_point(final["positions"], onset) if pre == "still" else {},
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
    onset = int(result["onset_frame"])
    ground = result.get("ground") or {}
    observation = result.get("observation")
    rows: list[dict] = []
    for f in range(n_frames):
        x, y, z = positions[f]
        if observation is not None and f > observation["frame"]:
            x = y = z = float("nan")  # no free-flight extrapolation after net/keeper contact
        if f < onset and f in ground:  # a rolling ball before the kick: its ground track (exact)
            x, y, z = ground[f][0], ground[f][1], BALL_RADIUS_M
        uv = pixels.get(f)
        if (
            observation is not None
            and observation["type"] == "goal_entry"
            and f == observation["frame"]
        ):
            uv = observation["pixel"]
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
    contact = result.get("contact") or {}
    record["contact_frame"] = contact.get("frame", result["onset_frame"])
    record["contact_source"] = contact.get("source")
    record["pre_contact"] = contact.get("pre_contact")
    if contact.get("warning"):
        record["contact_warning"] = contact["warning"]
    if result.get("incoming"):
        record["incoming"] = result["incoming"]
    record["side"] = result["side"]
    record["arrival"] = result["arrival"]
    record["arrival_status"] = result["arrival"]["type"] if result["arrival"] else "undetermined"
    record["arrival_observation"] = observation
    record["valid_until_frame"] = observation["frame"] if observation else n_frames - 1
    anchors = {"p0": {k: result["start"][k] for k in ("label", "frame", "xyz")}}
    if result["barrier"]:
        anchors["p_wall"] = result["barrier"]
    if observation is not None:
        key = "goal_entry" if observation["type"] == "goal_entry" else "p_arrival"
        anchors[key] = {
            k: result["arrival"][k]
            for k in ("label", "frame", "xyz", "sigma", "type", "provenance")
        }
    record["anchor_points"] = anchors
    problems = []
    if result["rms_flag"]:
        problems.append("rms above the reject limit")
    if observation is not None:
        diag = result["arrival"]["diagnostics"]
        if diag["fit_residual_m"] > 3 * result["arrival"]["sigma"]:
            problems.append("observed arrival residual above 3 sigma: review calibration/marks")
        if max(diag.get("goal_corner_errors_px", {"none": 0}).values()) > 3 * observation.get(
            "pixel_sigma", 2
        ):
            problems.append("goal corner reprojection needs review")
    record["quality"] = "; ".join(problems) if problems else "ok"
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
