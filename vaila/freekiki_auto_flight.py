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
Version: 0.4.19
Created: 27 October 2026
Update Date: 24 November 2026

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
    * ``feet`` (open play, the ball is dribbled or passed before the kick): the SAM 3D Body 2D feet
      of the people say who touches the ball; the contact is the last frame in which the ball is at
      the kicker's foot, followed by a sustained separation and a change of the ball's velocity
      (:func:`contact_from_feet`). Needs ``feet`` (:func:`read_feet`) and, best, the ``kicker``.
    * ``unseen``: the ball already flies in the first frame (the kick is at or before it).
    * a contact frame given by hand (``manual``) wins over all of these.

    Cameras. ``cameras`` maps frame -> the 11 DLT coefficients or a 3x4 projection
    (:func:`as_dlt_cameras`); a per-frame camera track (:func:`cameras_from_track`) replaces the
    few FreeKiki DLT3Ds of a panning broadcast camera, with the frames that are only copies of a
    neighbour (``approximate``) left out of the fit.
"""

from __future__ import annotations

import gzip
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
FEET_SIDE_FROM = 2  # frames after a feet contact from which the side of the goal is read...
FEET_SIDE_TO = (
    8  # ...to here (the ball is going away; further on it may be caught and carried back)
)
KINK_SEGMENT = 3  # frames of roll before the break, and of flight after it to read the launch
ROLL_BREAK_M = (
    0.3  # a frame this far (metres) off the rolling line has left it (hand marks: ~0.1 m)
)
KINK_MIN_DV_MPS = 3.0  # the kick speeds the ball up by at least this much...
KINK_MIN_TURN_DEG = 30.0  # ...or turns it by at least this angle
# A moving (panning/zooming) camera moves a resting ball in the image: its stillness and kick are then
# read on the pitch (the ball lifted with each frame's own camera), where only the ball moves.
ONSET_SHIFT_M = (
    0.3  # pitch movement from the rest point that marks the kick (calibration noise ~0.1 m)
)
CAMERA_PAN_PX = 2.0  # a fixed pitch point moving more than this in the image: the camera moves
# A save (the keeper's hands stop the ball): the ball sits on its pixel's camera ray, at the depth of
# the keeper's palms from the 3D pose. Across the ray the pixels fix it; along it (the depth a single
# camera cannot see) the pose decides, with this uncertainty.
SAVE_SIGMA_ACROSS_M = 0.05
SAVE_SIGMA_ALONG_M = 0.4
# Contact from the feet (open play): thresholds are fractions of the person's box height, so they do
# not depend on the zoom of the broadcast camera. A foot touches the ball when a foot keypoint is
# within FEET_NEAR_FRAC of it; a kick is the last such frame followed by a sustained separation
# (the ball leaves the foot at FEET_SEPARATION_SPEED_H_S box heights per second or more over
# FEET_SEPARATION_FRAMES frames, without coming back) and by a change of the ball's image velocity
# of at least FEET_KICK_DV_H_S box heights per second (a ball that merely passes a foot keeps
# its velocity, a dribble touch changes it little; Vargas, Fla x CAP 2026-11-19: kick 14.2, his dribble
# touches at most 2.0).
FEET_NEAR_FRAC = 0.20
FEET_SEPARATION_FRAMES = 3
FEET_SEPARATION_SPEED_H_S = 3.5
FEET_KICK_DV_H_S = 3.0
FEET_BEFORE_FRAMES = 3  # frames of the ball up to the contact that give the velocity going in
FEET_MIN_BEFORE = 2  # fewer ball frames than this before the contact: the ball cannot be judged
FEET_MAX_GAP_FRAMES = 3  # the first frame after the contact must be at most this far from it
FEET_GROWTH_TOL_H = 0.05  # the distance may dip this much (box heights) between frames
PALM_PARTS = {
    "left": ("left-wrist", "left-index-tip", "left-pinky-tip"),
    "right": ("right-wrist", "right-index-tip", "right-pinky-tip"),
}


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


def kick_onset_ground(
    ground: dict[int, tuple[float, float]],
    *,
    min_shift_m: float = ONSET_SHIFT_M,
    rest_frames: int = REST_FRAMES,
) -> int | None:
    """The first frame in which the ball has moved ``min_shift_m`` on the pitch from its rest point.

    ``ground`` maps frame -> ``(x, y)``, the ball lifted to the pitch with each frame's own camera,
    so a panning or zooming camera, which moves a resting ball in the image, does not move it here.
    The rest point is the median of the first ``rest_frames`` frames. The next observed frame must
    be off too (one badly lifted frame is not a kick), unless the moved frame is the last one.
    None when the ball never moves in the record.
    """
    frames = sorted(ground)
    if len(frames) < rest_frames + 1:
        return None
    rest = np.median(np.asarray([ground[f] for f in frames[:rest_frames]], dtype=float), axis=0)

    def moved(f: int) -> bool:
        return float(np.hypot(*(np.asarray(ground[f], dtype=float) - rest))) > min_shift_m

    for i in range(rest_frames, len(frames)):
        if moved(frames[i]) and (i + 1 == len(frames) or moved(frames[i + 1])):
            return int(frames[i])
    return None


def is_still_ground(
    ground: dict[int, tuple[float, float]],
    *,
    min_shift_m: float = ONSET_SHIFT_M,
    rest_frames: int = REST_FRAMES,
) -> bool:
    """True when the ball lies still on the pitch in the first ``rest_frames`` frames: every one is
    within ``min_shift_m`` of their median (the pitch twin of :func:`is_still`)."""
    frames = sorted(ground)[:rest_frames]
    if len(frames) < rest_frames:
        return False
    pts = np.asarray([ground[f] for f in frames], dtype=float)
    rest = np.median(pts, axis=0)
    return bool(np.all(np.hypot(*(pts - rest).T) <= min_shift_m))


def camera_moves(
    projections: dict[int, np.ndarray],
    point_xyz,
    *,
    tol_px: float = CAMERA_PAN_PX,
) -> bool:
    """True when a fixed pitch point moves more than ``tol_px`` in the image across the frames.

    ``projections`` maps frame -> 3x4 projection. A broadcast camera that pans or zooms moves every
    pitch point; a fixed camera (the same projection every frame) moves none.
    """
    X = np.append(np.asarray(point_xyz, dtype=float), 1.0)
    pixels = []
    for P in projections.values():
        h = np.asarray(P, dtype=float) @ X
        if abs(h[2]) > 1e-12:
            pixels.append(h[:2] / h[2])
    if len(pixels) < 2:
        return False
    pts = np.asarray(pixels)
    spread = np.hypot(*(pts - np.median(pts, axis=0)).T)
    return bool(np.nanmax(spread) > tol_px)


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


def as_dlt(camera) -> np.ndarray:
    """The 11 DLT coefficients of a camera given as them (11 values) or as a 3x4 projection.

    A projection (3x4 or its 12 values, any scale and sign) is divided by its ``[2, 3]`` entry,
    the DLT convention ``P[2, 3] = 1``, so the point in front of the camera keeps ``w > 0``.
    """
    arr = np.asarray(camera, dtype=float)
    if arr.size == 11:
        return arr.reshape(11)
    if arr.size == 12:
        P = arr.reshape(3, 4)
        if not np.isfinite(P).all() or abs(P[2, 3]) < 1e-12:
            raise ValueError("a projection with P[2, 3] = 0 or not finite has no DLT form")
        return (P / P[2, 3]).reshape(12)[:11]
    raise ValueError(f"a camera is 11 DLT coefficients or a 3x4 projection, got {arr.shape}")


def as_dlt_cameras(cameras: dict[int, np.ndarray] | None) -> dict[int, np.ndarray]:
    """``cameras`` (frame -> 11 coefficients or 3x4) as frame -> 11 coefficients, the form the
    pipeline reads. Frames whose camera is not finite are left out."""
    out: dict[int, np.ndarray] = {}
    for frame, camera in (cameras or {}).items():
        try:
            out[int(frame)] = as_dlt(camera)
        except ValueError:
            continue
    return out


def cameras_from_track(
    track, *, exclude: tuple[str, ...] = ("approximate",)
) -> dict[int, np.ndarray]:
    """Frame -> the 11 DLT coefficients of every frame of a camera track that can be trusted.

    ``track`` is a :class:`vaila.kick_measure.CameraTrack` (``n_frames``, ``info(frame)`` with
    ``source`` and ``matrix``). Frames whose ``source`` is in ``exclude`` (default ``approximate``:
    a copy of the nearest good frame) are left out, so the flight is never fitted through a camera
    that is only a guess.
    """
    out: dict[int, np.ndarray] = {}
    for frame in range(int(track.n_frames)):
        info = track.info(frame)
        if info.source in exclude:
            continue
        out[frame] = as_dlt(info.matrix)
    return out


def read_feet(
    predictions,
) -> tuple[dict[int, dict[int, list[tuple[float, float]]]], dict[int, dict[int, float]]]:
    """``(feet, heights_px)`` of a SAM 3 + SAM 3D Body ``*_predictions.json.gz``.

    ``feet[frame][sam_id]`` are the pixels of that person's ankle, heel and toe keypoints
    (:data:`vaila.freekiki_field3d.FEET_NAMES`, those that are finite); ``heights_px[frame][sam_id]``
    is the height of the person's box, the scale of :func:`contact_from_feet`. A person is keyed by
    ``sam_obj_id`` (``person_id`` when it is absent).
    """
    with gzip.open(predictions, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    names = list(data["keypoint_names"])
    idx = [i for i, n in enumerate(names) if n in FEET_NAMES]
    feet: dict[int, dict[int, list[tuple[float, float]]]] = {}
    heights: dict[int, dict[int, float]] = {}
    for frame in data["frames"]:
        f = int(frame["frame"])
        for inst in frame["instances"]:
            pid = inst.get("sam_obj_id", inst.get("person_id"))
            box = inst.get("bbox_xyxy")
            if pid is None or box is None:
                continue
            kps = np.asarray(inst["keypoints_2d_px"], dtype=float).reshape(-1, 2)
            pts = [(float(x), float(y)) for x, y in kps[idx] if np.isfinite([x, y]).all()]
            height = float(box[3]) - float(box[1])
            if pts and height > 0:
                feet.setdefault(f, {})[int(pid)] = pts
                heights.setdefault(f, {})[int(pid)] = height
    return feet, heights


def _feet_candidates(
    person: int,
    pixels: dict[int, tuple[float, float]],
    feet: dict[int, dict[int, list[tuple[float, float]]]],
    heights_px: dict[int, dict[int, float]],
    fps: float,
    near_frac: float,
) -> list[dict]:
    """The kicks of one person: each last frame of a run at the foot that the ball then leaves."""
    dist: dict[int, float] = {}
    dist_px: dict[int, float] = {}
    height: dict[int, float] = {}
    for f, uv in pixels.items():
        pts = feet.get(f, {}).get(person)
        h = heights_px.get(f, {}).get(person)
        if not pts or not h or h <= 0:
            continue
        d = float(np.min(np.hypot(*(np.asarray(pts, dtype=float) - np.asarray(uv, dtype=float)).T)))
        dist[f], dist_px[f], height[f] = d / h, d, h
    seen = sorted(dist)
    out: list[dict] = []
    for c in seen:
        if dist[c] > near_frac:
            continue
        later = [g for g in seen if c < g <= c + 2 * FEET_SEPARATION_FRAMES][
            :FEET_SEPARATION_FRAMES
        ]
        if len(later) < 2 or later[0] - c > FEET_MAX_GAP_FRAMES or dist[later[0]] <= near_frac:
            continue  # not the last frame at the foot, or the ball is not followed
        span = later[-1] - c
        speed_h_s = (dist[later[-1]] - dist[c]) / span * fps
        if speed_h_s < FEET_SEPARATION_SPEED_H_S:
            continue
        chain = [c, *later]
        if any(
            dist[b] < dist[a] - FEET_GROWTH_TOL_H for a, b in zip(chain, chain[1:], strict=False)
        ):
            continue  # it comes back to the foot: not a separation
        before = [g for g in sorted(pixels) if c - FEET_BEFORE_FRAMES <= g <= c]
        if len(before) < FEET_MIN_BEFORE:
            continue  # the ball was not seen going in: a pass-by cannot be told from a kick
        v_in = (np.asarray(pixels[c], float) - np.asarray(pixels[before[0]], float)) / (
            c - before[0]
        )
        v_out = (np.asarray(pixels[later[-1]], float) - np.asarray(pixels[c], float)) / span
        dv_h_s = float(np.hypot(*(v_out - v_in))) * fps / height[c]
        if dv_h_s < FEET_KICK_DV_H_S:
            continue
        out.append(
            {
                "frame": int(c),
                "person": int(person),
                "distance_px": round(dist_px[c], 2),
                "distance_frac": round(dist[c], 3),
                "separation_h_per_s": round(speed_h_s, 2),
                "dv_h_per_s": round(dv_h_s, 2),
                "source": "feet",
            }
        )
    return out


def contact_from_feet(
    ball_px: dict[int, tuple[float, float]],
    feet: dict[int, dict[int, list[tuple[float, float]]]],
    heights_px: dict[int, dict[int, float]],
    fps: float,
    *,
    kicker: int | None = None,
    near_frac: float = FEET_NEAR_FRAC,
) -> dict | None:
    """The kick contact read from the feet of the people, for a ball that is not at rest.

    ``ball_px`` maps frame -> ball pixel; ``feet[frame][person]`` lists the pixels of the
    person's foot keypoints (:func:`read_feet`) and ``heights_px[frame][person]`` the person's box
    height in pixels. ``kicker`` is the SAM id of the kicker (roster); None looks at everybody.

    A candidate contact of a person is the last frame of a run in which a foot keypoint is within
    ``near_frac`` of the person's box height from the ball, if: the next ball frame is not at the
    foot and the ball keeps going away over ``FEET_SEPARATION_FRAMES`` frames at no less than
    ``FEET_SEPARATION_SPEED_H_S`` box heights per second; and the ball's image velocity changes by at
    least ``FEET_KICK_DV_H_S`` box heights per second between the frames going in and out (a ball
    that only passes a foot, or a gentle dribble touch, does not). The ball must be seen in at least
    ``FEET_MIN_BEFORE`` frames up to the contact, else it cannot be judged (a ball that already
    flies in the first frame, or a clip that starts at the kick: None).

    With several candidates the one with the largest velocity change wins (a shot beats a dribble
    touch before it or a keeper's parry after it; ties go to the later frame); the others are in
    ``others``. Returns ``{"frame", "person", "distance_px", "distance_frac",
    "separation_h_per_s", "dv_h_per_s", "source": "feet", "others"}`` (the contact frame is the last
    frame with the ball at the foot: the kick lies between it and the next frame), or None.
    """
    pixels = {int(f): (float(uv[0]), float(uv[1])) for f, uv in ball_px.items()}
    if len(pixels) < FEET_MIN_BEFORE + 2:
        return None
    people = {
        int(p) for f in pixels for p in feet.get(f, {}) if kicker is None or int(p) == int(kicker)
    }
    found: list[dict] = []
    for person in sorted(people):
        found += _feet_candidates(person, pixels, feet, heights_px, fps, near_frac)
    if not found:
        return None
    found.sort(key=lambda r: (r["dv_h_per_s"], r["frame"]))
    best = dict(found[-1])
    best["others"] = [
        {k: r[k] for k in ("frame", "person", "dv_h_per_s")} for r in reversed(found[:-1])
    ]
    return best


def _feet_roll_in(ground: dict[int, tuple[float, float]], contact: int, fps: float) -> dict | None:
    """Where the ball is at the kick of a feet contact, and the way it came in.

    A straight line (constant velocity) through the last ``FEET_BEFORE_FRAMES + 1`` ground frames up
    to the contact, taken at the middle of the contact frame and the next one (the kick lies between
    them): a dribbled ball is not straight over its whole run, only over the last touch. With fewer
    than three frames the ball is taken where it was seen. None without a ground frame there.
    """
    frames = [f for f in sorted(ground) if contact - FEET_BEFORE_FRAMES <= f <= contact]
    if not frames:
        return None
    if len(frames) < 3:
        xy = np.asarray(ground[frames[-1]], dtype=float)
        return {"kick_xy": [float(xy[0]), float(xy[1])], "incoming": None}
    t = np.asarray(frames, dtype=float)
    _, v, c = _line_fit(t, np.asarray([ground[f] for f in frames], dtype=float))
    kick_xy = c + v * (contact + 0.5)
    vel = v * fps
    return {
        "kick_xy": [float(kick_xy[0]), float(kick_xy[1])],
        "incoming": {
            "speed_mps": round(float(np.hypot(*vel)), 2),
            "heading_deg": round(float(np.degrees(np.arctan2(vel[1], vel[0]))), 1),
            "velocity_mps": [round(float(vel[0]), 3), round(float(vel[1]), 3)],
            "frames": [int(frames[0]), int(frames[-1])],
        },
    }


def find_contact(
    pixels: dict[int, tuple[float, float]],
    ground: dict[int, tuple[float, float]],
    fps: float,
    *,
    contact_frame: int | None = None,
    panning: bool = False,
    feet: dict[int, dict[int, list[tuple[float, float]]]] | None = None,
    heights_px: dict[int, dict[int, float]] | None = None,
    kicker: int | None = None,
) -> dict:
    """The kick contact, decided once for the whole pipeline.

    Returns ``{"frame", "source", "pre_contact", "warning"}`` (plus ``kick_xy`` and ``incoming``
    for a rolling ball). ``source``: ``manual`` (``contact_frame``, given by hand), ``rest`` (a
    still ball, :func:`kick_onset`), ``feet`` (a ball that is not at rest and the feet of the
    people, :func:`contact_from_feet`; also ``person`` and ``distance_px``), ``kink``
    (:func:`rolling_break`) or ``first_frame`` (the ball already flies in the first frame).
    ``pre_contact``: ``still``, ``rolling`` or ``unseen``. ``frame`` is None when the track is too
    short or the still ball never moves.

    ``feet`` / ``heights_px`` (:func:`read_feet`) and ``kicker`` (SAM id) switch the feet reading on
    for a ball that is not at rest; when it finds no kick the rolling break and the first frame
    decide, as before. Without ``feet`` nothing changes. For a ``feet`` contact ``frame`` is the
    last frame with the ball at the foot: the kick is between it and the next frame.

    ``panning`` (the camera moves, :func:`camera_moves`): the image moves a resting ball, so its
    stillness and kick are read on the pitch (:func:`is_still_ground`, :func:`kick_onset_ground`)
    whenever ``ground`` has enough frames. A fixed camera keeps the pixel rules.
    """
    frames = sorted(pixels)
    if len(frames) < REST_FRAMES + 1:
        return {"frame": None, "source": None, "pre_contact": None,
                "warning": f"fewer than {REST_FRAMES + 1} ball frames"}  # fmt: skip
    on_pitch = panning and len(ground) >= REST_FRAMES + 1
    still = is_still_ground(ground) if on_pitch else is_still(pixels)
    if contact_frame is not None:
        contact = int(contact_frame)
        before = [f for f in sorted(ground) if f < contact]
        pre = "still" if still else ("rolling" if len(before) >= 2 else "unseen")
        found: dict = {"frame": contact, "source": "manual", "pre_contact": pre, "warning": None}
        if pre == "rolling":
            found.update(_roll_in(ground, before, contact, fps))
        return found
    if still:
        onset = kick_onset_ground(ground) if on_pitch else kick_onset(pixels)
        return {"frame": onset, "source": "rest", "pre_contact": "still",
                "warning": None if onset is not None else "the ball never moves"}  # fmt: skip
    if feet:
        touched = contact_from_feet(pixels, feet, heights_px or {}, fps, kicker=kicker)
        if touched is not None:
            before = [f for f in sorted(ground) if f <= touched["frame"]]
            rolled_in = _feet_roll_in(ground, touched["frame"], fps)
            return {
                "frame": int(touched["frame"]),
                "source": "feet",
                "pre_contact": "rolling" if len(before) >= 2 else "unseen",
                "warning": None,
                "person": touched["person"],
                "distance_px": touched["distance_px"],
                "others": touched["others"],
                **(rolled_in or {}),
            }
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

    ``{"frame", "source", "pre_contact", "kick_xy", "incoming", "person", "side"}`` (``person``: the
    SAM id of a ``feet`` contact, else None; ``side``: +1 or -1, the goal line x = side * 52.45 the
    fitted ball goes to, the attacked goal of an open-play kick, None in an older file);
    None for a folder without
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
        "person": data.get("contact_person"),
        "side": data.get("side"),
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


def camera_centre(P: np.ndarray) -> np.ndarray:
    """The camera centre (pitch metres) of a 3x4 projection: ``P @ [C, 1] = 0``."""
    P = np.asarray(P, dtype=float)
    return -np.linalg.solve(P[:, :3], P[:, 3])


def pixel_ray(P: np.ndarray, uv) -> np.ndarray:
    """The unit direction, from the camera into the scene, of the pixel ``uv``'s camera ray."""
    P = np.asarray(P, dtype=float)
    d = np.linalg.solve(P[:, :3], np.array([float(uv[0]), float(uv[1]), 1.0]))
    return d / np.linalg.norm(d)


def keeper_palms(pose_rows: list[dict], frame: int, person_id: int) -> np.ndarray | None:
    """The keeper's palms at ``frame``: the mean over the hands found of each hand's centroid of
    wrist, index tip and pinky tip (:data:`PALM_PARTS`). None without any of them."""
    parts: dict[str, np.ndarray] = {}
    for row in pose_rows:
        if int(row["frame"]) != frame or int(row["person_id"]) != int(person_id):
            continue
        xyz = np.array([row["x_m"], row["y_m"], row["z_m"]], dtype=float)
        if np.isfinite(xyz).all():
            parts[str(row["kpt_name"])] = xyz
    palms = []
    for names in PALM_PARTS.values():
        found = [parts[n] for n in names if n in parts]
        if found:
            palms.append(np.mean(np.asarray(found), axis=0))
    return np.mean(np.asarray(palms), axis=0) if palms else None


def save_anchor(
    pose_rows: list[dict],
    frame: int,
    P: np.ndarray,
    ball_uv,
    toward_xy,
    *,
    person_id: int | None = None,
    goal_x: float = GOAL_X_M,
    sigma_along: float = SAVE_SIGMA_ALONG_M,
    shift_m: float = 0.0,
) -> dict | None:
    """The ball at the save: a ``points`` anchor on its pixel's camera ray at the keeper's hands.

    The ball centre is a ball radius in front of the palms (:func:`keeper_palms` of ``person_id``,
    or the wrists of the player nearest the goal, :func:`keeper_hand`), toward ``toward_xy`` (the
    kick spot). That point is moved to the nearest point of the ball's camera ray: the pixel
    fixes the ball across the ray, the pose only its depth along it, which is uncertain by
    ``sigma_along``. ``shift_m`` moves the anchor along the ray (the +/- sigma refits of the
    uncertainty). ``ray_gap_m`` is how far the pose's ball was from the ray (the pose's own error
    across it). None when the keeper has no hands in that frame.
    """
    if person_id is not None:
        hands = keeper_palms(pose_rows, frame, person_id)
    else:
        hands = keeper_hand(pose_rows, frame, goal_x)
    if hands is None:
        return None
    toward = np.asarray(toward_xy, dtype=float)[:2] - hands[:2]
    norm = float(np.linalg.norm(toward))
    step = np.zeros(3) if norm < 1e-9 else np.append(toward / norm, 0.0) * BALL_RADIUS_M
    ball = hands + step
    C = camera_centre(P)
    ray = pixel_ray(P, ball_uv)
    on_ray = C + float((ball - C) @ ray) * ray
    anchor = on_ray + float(shift_m) * ray
    return {
        "frame": int(frame),
        "xyz": [float(v) for v in anchor],
        "sigma": SAVE_SIGMA_ACROSS_M,
        "ray": [float(v) for v in ray],
        "sigma_along": float(sigma_along),
        "label": "chegada: mãos do goleiro (defesa)",
        "hands_xyz": [float(v) for v in hands],
        "ray_gap_m": float(np.linalg.norm(ball - on_ray)),
        "shift_m": float(shift_m),
    }


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
    save_frame: int | None = None,
    keeper_id: int | None = None,
    save_sigma_along: float = SAVE_SIGMA_ALONG_M,
    save_shift_m: float = 0.0,
    feet: dict[int, dict[int, list[tuple[float, float]]]] | None = None,
    heights_px: dict[int, dict[int, float]] | None = None,
    kicker: int | None = None,
) -> dict | None:
    """The monocular flight with the automatic start and arrival, or None when the kick cannot
    be found (no movement, or no camera for the start frame).

    The contact comes from :func:`find_contact` (``contact_frame`` is a contact given by hand;
    ``feet``, ``heights_px`` and ``kicker`` let it read the kick from the people's feet when the ball
    is dribbled or passed before it). A ``feet`` contact starts the flight at the ball's position at
    the contact frame, with the kick time free between that frame and the next, and the side of the
    goal is read from the ball a few frames after the kick.
    A still ball keeps today's fit: the rest point is the start anchor and the kick time is free.
    A rolling ball starts at its kick point on the ground (the end of its incoming line) and the
    kick time is held between the last rolling frame and the contact. A ball already flying in
    the first frame starts at its lift in that frame, with the kick up to a frame before it.

    ``ball_all`` is the lifted track (:func:`freekiki_field3d.ball_3d`): the rest point, the seed
    and the side of the goal are read from it. ``cameras`` maps frame -> the 11-coefficient DLT, as
    everywhere in the pipeline (a 3x4 projection is also accepted, :func:`as_dlt_cameras`); it is
    turned into the 3x4 projections the fit and the goal homography use. Only the frames in it are
    fitted: leave the frames of an ``approximate`` camera out (:func:`cameras_from_track`).
    Returns ``{"fit", "onset_frame", "side", "arrival", "points", "barrier", "rms_flag"}``.

    ``save_frame`` (0-based, optional): the keeper's hands stop the ball in that frame. The pixels
    after it are dropped (the ball no longer flies freely) and, unless an arrival observation is
    given by hand, the end is anchored on the keeper's hands (:func:`save_anchor`, the keeper is
    the SAM ``keeper_id`` or the player nearest the goal). ``save_shift_m`` moves that anchor along
    the camera ray, for the +/- ``save_sigma_along`` refits of the uncertainty.
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
    projections = {f: dlt_projection_matrix(L) for f, L in as_dlt_cameras(cameras).items()}
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
    panning = bool(lifted) and camera_moves(
        projections,
        (*ground[min(lifted)], BALL_RADIUS_M),
    )
    contact = find_contact(
        pixels,
        ground,
        fps,
        contact_frame=contact_frame,
        panning=panning,
        feet=feet,
        heights_px=heights_px,
        kicker=kicker,
    )
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
    from_feet = contact["source"] == "feet"
    if from_feet:
        # the ball is at the foot in the contact frame and has left in the next: the kick is between
        kick_xy = contact.get("kick_xy")
        rest_xy = (
            np.asarray(kick_xy, dtype=float)
            if kick_xy is not None
            else np.array([float(start["x_m"]), float(start["y_m"])])
        )
        kick_window = (onset / fps, (onset + 1) / fps)
    elif pre == "still":
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
    if from_feet:  # the goal the ball goes to just after the kick, not where the clip ends
        soon = [
            float(lifted[f]["x_m"])
            for f in sorted(lifted)
            if onset + FEET_SIDE_FROM <= f <= onset + FEET_SIDE_TO
        ]
        if len(soon) >= 2:
            side = -1.0 if float(np.median(soon)) - x0 < 0 else 1.0
    if observation is not None and observation["type"] == "goal_entry":
        side = float(observation["goal_side"])
    start_anchor = {
        "frame": int(onset),
        "xyz": [float(rest_xy[0]), float(rest_xy[1]), BALL_RADIUS_M],
        "label": (
            "start: contato do pé (a bola era conduzida; último quadro no pé do chutador)"
            if from_feet
            else {
                "still": "start: contato do pé (início do movimento)",
                "rolling": "start: contato do pé (a bola chegava rolando)",
            }.get(pre, "start: primeiro quadro (o chute é nele ou logo antes)")
        ),
    }
    save_info: dict | None = None
    if save_frame is not None:
        save_frame = int(save_frame)
        if save_frame <= onset:
            raise ValueError(
                f"the save (frame {save_frame}) must be after the kick contact ({onset})"
            )
        after = {f: uv for f, uv in after.items() if f <= save_frame}  # no free flight after it
        save_info = {"frame": save_frame, "anchor": None}
        if observation is None:
            P_s, uv_s = projections.get(save_frame), pixels.get(save_frame)
            if P_s is None or uv_s is None:
                raise ValueError(
                    f"no calibrated camera or ball pixel at the save frame {save_frame}"
                )
            anchor = save_anchor(
                pose_rows,
                save_frame,
                P_s,
                uv_s,
                rest_xy,
                person_id=keeper_id,
                sigma_along=save_sigma_along,
                shift_m=save_shift_m,
            )
            save_info["anchor"] = anchor
            if anchor is not None:
                arrival = {
                    "type": "keeper_hand",
                    "frame": save_frame,
                    "xyz": anchor["xyz"],
                    "sigma": anchor["sigma"],
                    "ray": anchor["ray"],
                    "sigma_along": anchor["sigma_along"],
                    "label": anchor["label"],
                    "provenance": "auto",
                    "diagnostics": {
                        "ray_gap_m": round(anchor["ray_gap_m"], 3),
                        "hands_xyz": [round(v, 3) for v in anchor["hands_xyz"]],
                        "shift_m": anchor["shift_m"],
                    },
                }
    arrival_points = (
        []
        if arrival is None
        else [
            {
                k: arrival[k]
                for k in ("frame", "xyz", "sigma", "label", "ray", "sigma_along")
                if k in arrival
            }
        ]
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
    if save_info is not None and crossing is not None and crossing[1] > save_info["frame"]:
        crossing = None  # the keeper stopped the ball first: a later crossing is not an entry
    if arrival is None and crossing is not None and save_info is None:
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
    is_penalty_spot = (
        pre == "still"
        and abs(abs(float(rest_xy[0])) - 41.45) <= 2.0
        and abs(float(rest_xy[1])) <= 2.0
    )
    if (
        save_info is None
        and observation is None
        and arrival is None
        and is_penalty_spot
        and pose_rows
        and last_frame in projections
        and last_frame in pixels
    ):
        auto_anchor = save_anchor(
            pose_rows,
            last_frame,
            projections[last_frame],
            pixels[last_frame],
            rest_xy,
            person_id=keeper_id,
            goal_x=side * GOAL_X_M,
            sigma_along=save_sigma_along,
            shift_m=save_shift_m,
        )
        if auto_anchor is not None and auto_anchor["ray_gap_m"] <= 0.45:
            save_info = {"frame": int(last_frame), "anchor": auto_anchor}
            arrival = {
                "type": "keeper_hand",
                "frame": int(last_frame),
                "xyz": auto_anchor["xyz"],
                "sigma": auto_anchor["sigma"],
                "ray": auto_anchor["ray"],
                "sigma_along": auto_anchor["sigma_along"],
                "label": auto_anchor["label"],
                "provenance": "auto",
                "diagnostics": {
                    "ray_gap_m": round(auto_anchor["ray_gap_m"], 3),
                    "hands_xyz": [round(v, 3) for v in auto_anchor["hands_xyz"]],
                    "shift_m": auto_anchor["shift_m"],
                },
            }
            arrival_points = [
                {
                    k: arrival[k]
                    for k in ("frame", "xyz", "sigma", "label", "ray", "sigma_along")
                    if k in arrival
                }
            ]
            first = fit_flight(
                {f: uv for f, uv in after.items() if f in projections},
                projections,
                fps,
                n_frames,
                anchors={"rest": rest_xy, "points": arrival_points},
                theta0=seed,
                kick_window=kick_window,
            )
    final = first
    if observation is not None:
        assert arrival is not None
        pos = np.asarray(final["positions"][observation["frame"]])
        arrival.setdefault("diagnostics", {}).update(
            fit_residual_m=float(np.linalg.norm(pos - np.asarray(arrival["xyz"]))),
            fit_plane_error_m=float(abs(pos[0] - arrival["xyz"][0])),
            fitted_crossing_frame=None if crossing is None else crossing[1],
        )
    elif save_info is not None and arrival is not None:
        pos = np.asarray(final["positions"][save_info["frame"]])
        arrival["diagnostics"]["fit_residual_m"] = round(
            float(np.linalg.norm(pos - np.asarray(arrival["xyz"]))), 3
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
        "contact": {
            k: contact.get(k)
            for k in ("frame", "source", "pre_contact", "warning", "person", "distance_px")
        },
        "camera_moves": panning,
        "save_frame": None if save_info is None else save_info["frame"],
        "incoming": incoming,
        "ground": ground if pre == "rolling" else {},
        "side": side,
        "arrival": arrival,
        "start": start_anchor,
        "points": arrival_points,
        "observation": observation,
        # the wall is a set-piece rule: 9.15 m from a free kick (never on a penalty)
        "barrier": (
            barrier_point(final["positions"], onset)
            if (pre == "still" and not is_penalty_spot and save_info is None)
            else {}
        ),
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
    save_frame = result.get("save_frame")
    # the last frame of free flight: an observed arrival (net or keeper) or the save
    cut = observation["frame"] if observation is not None else save_frame
    rows: list[dict] = []
    for f in range(n_frames):
        x, y, z = positions[f]
        if cut is not None and f > cut:
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
    if contact.get("person") is not None:  # a feet contact: who touched the ball last
        record["contact_person"] = contact["person"]
        record["contact_distance_px"] = contact.get("distance_px")
    record["camera_moves"] = bool(result.get("camera_moves", False))
    if contact.get("warning"):
        record["contact_warning"] = contact["warning"]
    if result.get("incoming"):
        record["incoming"] = result["incoming"]
    record["side"] = result["side"]
    record["arrival"] = result["arrival"]
    record["arrival_status"] = result["arrival"]["type"] if result["arrival"] else "undetermined"
    record["arrival_observation"] = observation
    record["valid_until_frame"] = cut if cut is not None else n_frames - 1
    if save_frame is not None:
        record["save_frame"] = int(save_frame)
        record["shot_outcome"] = "save"
    anchors = {"p0": {k: result["start"][k] for k in ("label", "frame", "xyz")}}
    if result["barrier"]:
        anchors["p_wall"] = result["barrier"]
    if observation is not None:
        key = "goal_entry" if observation["type"] == "goal_entry" else "p_arrival"
        anchors[key] = {
            k: result["arrival"][k]
            for k in ("label", "frame", "xyz", "sigma", "type", "provenance")
        }
    elif save_frame is not None and result["arrival"] is not None:
        anchors["p_arrival"] = {
            k: result["arrival"][k]
            for k in ("label", "frame", "xyz", "sigma", "sigma_along", "ray", "type", "provenance")
        }
    record["anchor_points"] = anchors
    problems = []
    if result["rms_flag"]:
        problems.append("rms above the reject limit")
    if observation is None and save_frame is not None:
        if result["arrival"] is None:
            problems.append(
                "save without the keeper's hands in the 3D pose: the depth of the ball at the save "
                "is free (one camera); give a keeper id, an arrival observation or a second view"
            )
        else:
            along = abs(float(fit["anchor_residuals"].get("p0_along_m") or 0.0))
            if along > 3 * float(result["arrival"]["sigma_along"]):
                problems.append(
                    "the flight disagrees with the keeper's hands along the camera ray (> 3 sigma): "
                    "review the keeper's 3D pose or add a second view"
                )
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
