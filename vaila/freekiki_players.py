"""
================================================================================
Script: freekiki_players.py - keep only the people whose feet are on the pitch
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.11
Created: 25 October 2026
Update Date: 26 October 2026

Description:
    A broadcast frame shows the pitch plus the crowd, the bench and the staff.
    With ``--players-on-field`` a person counts as a player when the bottom
    centre of the person's box (the feet) lies inside the pitch outline of the
    frame, projected with that frame's DLT2D.

    * :func:`homography_from_dlt2d` - 3x3 world-plane -> pixel homography from the
      8 ``L1..L8`` coefficients of a ``.dlt2d`` row;
    * :func:`pitch_outline_px` - the pitch rectangle projected into the image;
    * :func:`feet_point` - bottom centre of an XYXY box;
    * :func:`inside_fraction` - share of a track's frames whose feet lie on the pitch;
    * :func:`players_on_field` - keep the tracks whose share reaches ``min_frac``.

    The pitch is the 105 x 68 m rectangle of ``soccerfield_kiki.csv`` (corners at
    x = +-52.45 m, y = +-33.95 m). Everything here is pure NumPy/OpenCV geometry.
"""

from __future__ import annotations

import numpy as np

PITCH_HALF_LENGTH_M = 52.45  # x, towards the goals
PITCH_HALF_WIDTH_M = 33.95  # y, towards the touchlines


def homography_from_dlt2d(coeffs: np.ndarray) -> np.ndarray:
    """3x3 homography (world X, Y, 0 -> pixel) from ``L1..L8`` with H[2,2] = 1."""
    c = np.asarray(coeffs, dtype=np.float64).reshape(8)
    return np.array(
        [[c[0], c[1], c[2]], [c[3], c[4], c[5]], [c[6], c[7], 1.0]],
        dtype=np.float64,
    )


def pitch_outline_px(H: np.ndarray) -> np.ndarray:
    """The four pitch corners in image pixels, in order (4, 2), or NaN if degenerate."""
    corners = np.array(
        [
            [-PITCH_HALF_LENGTH_M, -PITCH_HALF_WIDTH_M, 1.0],
            [PITCH_HALF_LENGTH_M, -PITCH_HALF_WIDTH_M, 1.0],
            [PITCH_HALF_LENGTH_M, PITCH_HALF_WIDTH_M, 1.0],
            [-PITCH_HALF_LENGTH_M, PITCH_HALF_WIDTH_M, 1.0],
        ]
    )
    proj = corners @ np.asarray(H, dtype=np.float64).T
    if not np.isfinite(proj).all() or np.any(np.abs(proj[:, 2]) < 1e-12):
        return np.full((4, 2), np.nan)
    return proj[:, :2] / proj[:, 2:3]


def feet_point(box_xyxy) -> tuple[float, float]:
    """Bottom centre of an XYXY box: the point where the person touches the ground."""
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    return 0.5 * (x1 + x2), max(y1, y2)


def point_on_pitch(outline_px: np.ndarray, point: tuple[float, float]) -> bool:
    """True when ``point`` lies inside (or on) the projected pitch quadrilateral."""
    import cv2

    if not np.isfinite(outline_px).all():
        return False
    contour = np.asarray(outline_px, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def inside_fraction(
    boxes: list[tuple[int, np.ndarray]],
    homographies: dict[int, np.ndarray],
) -> tuple[float, int, int]:
    """``(fraction, frames_on_pitch, frames_checked)`` of one track.

    ``boxes`` is ``[(frame_index, xyxy), ...]`` and ``homographies`` maps a frame
    index to its DLT2D homography. Frames without a homography are not checked
    and do not count for or against the track.
    """
    on = 0
    checked = 0
    for frame, box in boxes:
        H = homographies.get(int(frame))
        if H is None:
            continue
        checked += 1
        if point_on_pitch(pitch_outline_px(H), feet_point(box)):
            on += 1
    if checked == 0:
        return 0.0, 0, 0
    return on / checked, on, checked


def players_on_field(
    tracks: dict[int, list[tuple[int, np.ndarray]]],
    homographies: dict[int, np.ndarray],
    *,
    min_frac: float = 0.5,
) -> dict[int, dict]:
    """Verdict for every track: ``{track_id: {fraction, on, checked, keep}}``.

    A track is kept when at least ``min_frac`` of its checked frames have the feet
    on the pitch. A track with no checked frame is not kept.
    """
    if not 0.0 <= min_frac <= 1.0:
        raise ValueError("min_frac must be between 0 and 1")
    verdicts: dict[int, dict] = {}
    for track_id, boxes in tracks.items():
        fraction, on, checked = inside_fraction(boxes, homographies)
        keep = checked > 0 and fraction >= min_frac
        verdicts[int(track_id)] = {
            "fraction": round(fraction, 4),
            "on": on,
            "checked": checked,
            "keep": bool(keep),
        }
    return verdicts
