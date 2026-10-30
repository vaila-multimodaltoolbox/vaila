"""
================================================================================
Script: freekiki_topview.py - top view of the players and the ball on the pitch
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
    The ground track of every kept person (the bottom centre of the box, smoothed in
    ``freekiki_physics.smooth_track``) and the ball on the pitch plane, drawn
    frame by frame over the field lines. Each person leaves a coloured path behind;
    the ball leaves a yellow one. Written as ``<stem>_topview.mp4`` and as the last
    frame, ``<stem>_topview.png``, in the output folder.

    * :func:`write_topview` - reads ``ground_tracks.csv`` and ``ball_3d_on_field.csv``;
    * :func:`render_frame` - one frame on the pitch canvas (pure drawing).

    Uses OpenCV only; the field is drawn in pitch metres (origin at the centre spot,
    +x to the right goal, +y to the far touchline).
"""

from __future__ import annotations

import colorsys
import csv
from pathlib import Path

import numpy as np

try:
    from .freekiki_players import PITCH_HALF_LENGTH_M, PITCH_HALF_WIDTH_M
except ImportError:
    from freekiki_players import (  # ty: ignore[unresolved-import]
        PITCH_HALF_LENGTH_M,
        PITCH_HALF_WIDTH_M,
    )

PX_PER_M = 12
MARGIN_PX = 60
GRASS_BGR = (63, 143, 65)
LINE_BGR = (255, 255, 255)
BALL_BGR = (0, 230, 255)
TRAIL_THICKNESS = 3


def pitch_to_px(x: float, y: float) -> tuple[int, int]:
    """Pitch metres -> canvas pixels (y grows upwards on the pitch, downwards on the image)."""
    u = MARGIN_PX + (x + PITCH_HALF_LENGTH_M) * PX_PER_M
    v = MARGIN_PX + (PITCH_HALF_WIDTH_M - y) * PX_PER_M
    return int(round(u)), int(round(v))


def canvas_size() -> tuple[int, int]:
    """``(width, height)`` of the topview canvas in pixels."""
    w = int(round(2 * PITCH_HALF_LENGTH_M * PX_PER_M)) + 2 * MARGIN_PX
    h = int(round(2 * PITCH_HALF_WIDTH_M * PX_PER_M)) + 2 * MARGIN_PX
    return w, h


def person_colour(pid: int) -> tuple[int, int, int]:
    """A stable BGR colour per person id."""
    hue = (pid * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_pitch() -> np.ndarray:
    """The pitch in metres, FIFA dimensions, as a BGR image (the canvas of every frame)."""
    import cv2

    w, h = canvas_size()
    img = np.full((h, w, 3), GRASS_BGR, dtype=np.uint8)
    half_l, half_w = PITCH_HALF_LENGTH_M, PITCH_HALF_WIDTH_M

    def line(p, q):
        cv2.line(img, pitch_to_px(*p), pitch_to_px(*q), LINE_BGR, 2, cv2.LINE_AA)

    def rect(x0, y0, x1, y1):
        line((x0, y0), (x1, y0))
        line((x1, y0), (x1, y1))
        line((x1, y1), (x0, y1))
        line((x0, y1), (x0, y0))

    rect(-half_l, -half_w, half_l, half_w)  # touchlines and goal lines
    line((0.0, -half_w), (0.0, half_w))  # halfway line
    cv2.circle(img, pitch_to_px(0.0, 0.0), int(9.15 * PX_PER_M), LINE_BGR, 2, cv2.LINE_AA)
    cv2.circle(img, pitch_to_px(0.0, 0.0), 4, LINE_BGR, -1, cv2.LINE_AA)
    for side in (-1.0, 1.0):
        goal_x = side * half_l
        inner = side * (half_l - 16.5)  # penalty area is 16.5 m deep
        rect(min(goal_x, inner), -20.16, max(goal_x, inner), 20.16)
        inner = side * (half_l - 5.5)  # goal area is 5.5 m deep
        rect(min(goal_x, inner), -9.16, max(goal_x, inner), 9.16)
        spot = (side * (half_l - 11.0), 0.0)
        cv2.circle(img, pitch_to_px(*spot), 4, LINE_BGR, -1, cv2.LINE_AA)
        goal_u, goal_v = pitch_to_px(goal_x, 3.66), pitch_to_px(goal_x - side * 2.0, -3.66)
        cv2.rectangle(img, goal_u, goal_v, LINE_BGR, 2)
    return img


def render_frame(
    base: np.ndarray,
    frame: int,
    tracks: dict[int, dict[int, tuple[float, float]]],
    ball: dict[int, tuple[float, float]],
    fps: float,
    labels: dict[int, int] | None = None,
) -> np.ndarray:
    """The pitch with every person's path up to ``frame`` and the ball's path.

    ``labels`` maps a track id to the number to write beside the person (the player id of a run
    with a player map: ID0 is the keeper and ID1 the kicker of a penalty); the track id otherwise.
    """
    import cv2

    img = base.copy()

    def trail(points: dict[int, tuple[float, float]], colour):
        seen = [f for f in sorted(points) if f <= frame]
        if len(seen) >= 2:
            pts = np.array([pitch_to_px(*points[f]) for f in seen], dtype=np.int32)
            cv2.polylines(img, [pts], False, colour, TRAIL_THICKNESS, cv2.LINE_AA)
        return seen

    for pid in sorted(tracks):
        colour = person_colour(pid)
        seen = trail(tracks[pid], colour)
        if seen:
            x, y = tracks[pid][seen[-1]]
            if seen[-1] == frame:
                centre = pitch_to_px(x, y)
                cv2.circle(img, centre, 8, colour, -1, cv2.LINE_AA)
                cv2.circle(img, centre, 8, LINE_BGR, 1, cv2.LINE_AA)
                shown = labels.get(pid, pid) if labels else pid
                cv2.putText(
                    img,
                    f"{shown}",
                    (centre[0] + 10, centre[1] - 8),
                    0,
                    0.6,
                    LINE_BGR,
                    2,
                    cv2.LINE_AA,
                )
    trail(ball, BALL_BGR)
    if frame in ball:
        cv2.circle(img, pitch_to_px(*ball[frame]), 7, BALL_BGR, -1, cv2.LINE_AA)
    cv2.putText(
        img,
        f"frame {frame}  t={frame / fps:.2f} s",
        (MARGIN_PX, MARGIN_PX - 20),
        0,
        0.8,
        LINE_BGR,
        2,
        cv2.LINE_AA,
    )
    return img


def _read_tracks(path: Path) -> dict[int, dict[int, tuple[float, float]]]:
    tracks: dict[int, dict[int, tuple[float, float]]] = {}
    if not path.is_file():
        return tracks
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["x_m"] == "":
                continue
            tracks.setdefault(int(row["person_id"]), {})[int(row["frame"])] = (
                float(row["x_m"]),
                float(row["y_m"]),
            )
    return tracks


def _read_ball(path: Path) -> dict[int, tuple[float, float]]:
    ball: dict[int, tuple[float, float]] = {}
    if not path.is_file():
        return ball
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["on_field"] == "True" and row["x_m"] != "":
                ball[int(row["frame"])] = (float(row["x_m"]), float(row["y_m"]))
    return ball


def write_topview(
    out_dir: Path | str,
    stem: str,
    *,
    fps: float,
    n_frames: int,
    labels: dict[int, int] | None = None,
) -> dict[str, str]:
    """Write ``<stem>_topview.mp4`` and ``<stem>_topview.png`` from the run's CSVs.

    Reads ``ground_tracks.csv`` (the smoothed ground of each kept person) and
    ``ball_3d_on_field.csv`` from ``out_dir``. ``labels`` is as in :func:`render_frame`. Returns the
    names of the two files.
    """
    import cv2

    out_dir = Path(out_dir)
    tracks = _read_tracks(out_dir / "ground_tracks.csv")
    ball = _read_ball(out_dir / "ball_3d_on_field.csv")
    base = draw_pitch()
    video = out_dir / f"{stem}_topview.mp4"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"mp4v"),  # ty: ignore[unresolved-attribute]
        fps,
        (base.shape[1], base.shape[0]),
    )
    last = None
    try:
        for f in range(n_frames):
            last = render_frame(base, f, tracks, ball, fps, labels)
            writer.write(last)
    finally:
        writer.release()
    png = out_dir / f"{stem}_topview.png"
    if last is not None:
        cv2.imwrite(str(png), last)
    return {"video": video.name, "png": png.name}
