"""
================================================================================
Script: freekiki_snap.py - FreeKiki keypoints snapped to the painted lines
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.8
Created: 15 October 2026
Update Date: 15 October 2026

Description:
    On cameras that are rare in the training set (high, lateral, close to the
    penalty box) the network places the box corners with a systematic offset
    of 20-30 px: the box comes out larger than the painted one. The paint in
    the image is the truth. For every painted straight line of the pitch
    (touchlines, goal lines, halfway line, penalty-box and goal-area lines)
    that has at least two accepted network points, the line is refitted to the
    white paint inside a band around the predicted segment; then

    * a keypoint at the crossing of two refitted lines (a box corner, a goal
      area corner, a pitch corner) moves to their intersection;
    * a keypoint on one refitted line only (penalty-arc crossings, post bases,
      centre circle top / bottom, centre spot) moves to its foot on that line
      (the position along the line is kept).

    A move larger than ``max_move_px`` (at 1920 px width) is refused: the paint
    found is then not the line the network meant. Pure numpy / OpenCV; nothing
    here changes the network CSVs (``freekiki.py snap`` writes its own).
"""

from __future__ import annotations

import numpy as np

# Painted straight lines of the pitch, as the kiki49 keypoints that lie on them.
PAINTED_LINES: dict[str, tuple[int, ...]] = {
    "touchline_top": (0, 13, 24),
    "touchline_bottom": (5, 16, 29),
    "goal_line_left": (0, 1, 2, 3, 4, 5, 32, 33),
    "goal_line_right": (24, 25, 26, 27, 28, 29, 40, 41),
    "halfway": (13, 14, 15, 16, 48),
    "box_left_side": (9, 10, 11, 12),
    "box_left_top": (1, 9),
    "box_left_bottom": (4, 12),
    "goal_area_left_side": (6, 7),
    "goal_area_left_top": (2, 6),
    "goal_area_left_bottom": (3, 7),
    "box_right_side": (17, 18, 19, 20),
    "box_right_top": (25, 17),
    "box_right_bottom": (28, 20),
    "goal_area_right_side": (22, 23),
    "goal_area_right_top": (26, 22),
    "goal_area_right_bottom": (27, 23),
}
BAND_PX = 35.0  # half width of the search band around a predicted line, px at 1920
MAX_MOVE_PX = 40.0  # a snapped keypoint may move at most this much, px at 1920
MAX_ANGLE_DEG = 8.0  # refitted line vs predicted line
MIN_PAINT_PX = 40  # paint pixels needed to refit a line
ON_LINE_PX = 3.0  # px at 1920: a network point this close to its refitted line(s) is kept
LINE_ON_PAINT = 0.6  # share of a refitted segment that must lie on paint
MIN_CROSS_DEG = 15.0  # two lines crossing at a shallower angle give an unstable corner


def lines_of(point: int) -> list[str]:
    return [name for name, pts in PAINTED_LINES.items() if point in pts]


def paint_mask(image_bgr) -> np.ndarray:
    """Bool mask of thin bright markings on grass (white-top-hat, grass only)."""
    import cv2

    try:
        from .kiki49_build.line_support import grass_mask
    except ImportError:
        from kiki49_build.line_support import grass_mask  # ty: ignore[unresolved-import]

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    k = max(3, int(round(15 * gray.shape[1] / 1920.0)) | 1)
    tophat = cv2.morphologyEx(
        gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    return (tophat > 25) & grass_mask(image_bgr).astype(bool)


def fit_line_in_band(paint_xy: np.ndarray, a, b, band: float, paint: np.ndarray | None = None):
    """``(centre, unit direction)`` of the paint inside a band around segment a-b, or None.

    ``paint_xy`` is (N, 2) pixel coordinates of paint. The band follows the
    predicted segment (5 % beyond each end); a robust refit keeps the dominant
    straight stroke when the band also catches another marking or a player.
    With ``paint`` (the bool mask), the refitted segment between a and b must
    lie on paint for at least ``LINE_ON_PAINT`` of its length: a line pulled by
    goal posts, the net or an advertising board does not.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = b - a
    length = float(np.hypot(*d))
    if length < 1.0 or len(paint_xy) == 0:
        return None
    u = d / length
    n = np.array([-u[1], u[0]])
    rel = paint_xy - a
    along, off = rel @ u, rel @ n
    q = paint_xy[(np.abs(off) < band) & (along > -0.05 * length) & (along < 1.05 * length)]
    if len(q) < MIN_PAINT_PX:
        return None
    for _ in range(3):
        c = q.mean(axis=0)
        _, _, vt = np.linalg.svd(q - c, full_matrices=False)
        dirv = vt[0]
        resid = np.abs((q - c) @ np.array([-dirv[1], dirv[0]]))
        keep = resid < max(3.0, float(np.percentile(resid, 70)))
        if keep.sum() < MIN_PAINT_PX:
            break
        q = q[keep]
    c = q.mean(axis=0)
    _, _, vt = np.linalg.svd(q - c, full_matrices=False)
    dirv = vt[0]
    angle = np.degrees(np.arccos(min(1.0, abs(float(dirv @ u)))))
    if angle > MAX_ANGLE_DEG:
        return None
    if paint is not None:
        h, w = paint.shape
        r = max(1, int(round(2 * w / 1920.0)))
        ta, tb = float((a - c) @ dirv), float((b - c) @ dirv)
        on = total = 0
        for t in np.linspace(min(ta, tb), max(ta, tb), 60):
            x, y = (c + t * dirv).round().astype(int)
            if 0 <= x < w and 0 <= y < h:
                total += 1
                on += bool(paint[max(0, y - r) : y + r + 1, max(0, x - r) : x + r + 1].any())
        if total < 10 or on / total < LINE_ON_PAINT:
            return None
    return c, dirv


def _intersect(l1, l2):
    (c1, d1), (c2, d2) = l1, l2
    if abs(float(d1[0] * d2[1] - d1[1] * d2[0])) < np.sin(np.radians(MIN_CROSS_DEG)):
        return None  # near-parallel lines: a tiny tilt moves the corner a lot
    A = np.c_[d1, -d2]
    s, _ = np.linalg.solve(A, c2 - c1)
    return c1 + s * d1


def _foot(line, p):
    c, d = line
    return c + ((np.asarray(p, dtype=float) - c) @ d) * d


def snap_keypoints(
    xy,
    accepted,
    paint: np.ndarray,
    *,
    band_px: float = BAND_PX,
    max_move_px: float = MAX_MOVE_PX,
) -> tuple[np.ndarray, dict[int, str]]:
    """Accepted keypoints moved onto the painted lines.

    ``xy`` (49, 2) network points, ``accepted`` (49,) bool, ``paint`` the bool
    mask of :func:`paint_mask` (its width sets the px scale). Returns the new
    (49, 2) points (unchanged where nothing was snapped) and ``{point: how}``
    with ``how`` = ``corner`` (two lines) or ``line`` (one line). A point
    already within ``ON_LINE_PX`` of its refitted line(s) is left as is (on a
    camera the network knows well, the snap must not move anything).
    """
    xy = np.asarray(xy, dtype=float)
    acc = np.asarray(accepted, dtype=bool)
    scale = paint.shape[1] / 1920.0
    ys, xs = np.nonzero(paint)
    pts = np.c_[xs, ys].astype(float)
    fitted = {}
    for name, members in PAINTED_LINES.items():
        on = [i for i in members if acc[i] and np.isfinite(xy[i]).all()]
        if len(on) < 2:
            continue
        P = xy[on]
        # the two members farthest apart span the predicted segment
        i, j = max(
            ((a, b) for a in range(len(on)) for b in range(a + 1, len(on))),
            key=lambda ab: float(np.hypot(*(P[ab[0]] - P[ab[1]]))),
        )
        line = fit_line_in_band(pts, P[i], P[j], band_px * scale, paint)
        if line is not None:
            fitted[name] = line
    out, how = xy.copy(), {}
    for k in np.flatnonzero(acc):
        mine = [fitted[n] for n in lines_of(int(k)) if n in fitted]
        # already on its refitted line(s): the network is right, leave it alone
        if mine and all(
            abs(float((xy[k] - c) @ np.array([-d[1], d[0]]))) <= ON_LINE_PX * scale for c, d in mine
        ):
            continue
        new = None
        if len(mine) >= 2:
            new, kind = _intersect(mine[0], mine[1]), "corner"
        elif len(mine) == 1:
            new, kind = _foot(mine[0], xy[k]), "line"
        if new is None or not np.isfinite(new).all():
            continue
        if np.hypot(*(new - xy[k])) <= max_move_px * scale:
            out[k] = new
            how[int(k)] = kind
    return out, how


STABLE_WINDOW = 15  # frames of the running mean of a point's correction


def _jitter(track: np.ndarray) -> float:
    """Mean |second difference| of a (T, 2) track, NaN-aware (frame-to-frame shake).

    A mean, not a median: a single-frame pop must count.
    """
    d2 = track[2:] - 2 * track[1:-1] + track[:-2]
    v = np.hypot(d2[:, 0], d2[:, 1])
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if len(v) else float("nan")


def stabilize(
    network: np.ndarray,
    snapped: np.ndarray,
    width: float | None = None,
    *,
    window: int = STABLE_WINDOW,
) -> np.ndarray:
    """Pick, for every keypoint, the steadier of two snapped tracks for the whole clip.

    ``network`` and ``snapped`` are (T, 49, 2) with NaN where a point is not
    accepted. The candidates of a point are

    * the per-frame snap (best where its painted lines are sharp: the paint holds
      the point still even when the network shakes), and
    * the network plus the running mean (``window`` frames) of its correction
      (best where the snap hesitates between two strokes, e.g. a short goal-area
      line half hidden by players; a mean, because a median of two alternating
      values alternates too).

    Both are computed, and compared, inside each stretch where the point is
    present (a new shot or a pan back starts a new stretch); the one with the
    lower mean frame-to-frame shake is used for the whole stretch, so a point
    never switches source mid-stretch (a switch is itself a jump).
    A stretch whose median correction is below ``ON_LINE_PX`` (scaled by
    ``width``) keeps the network points exactly: the snap never adds shake to a
    camera the network already gets right.
    """
    import warnings

    net = np.asarray(network, dtype=float)
    snp = np.asarray(snapped, dtype=float)
    off = snp - net
    half = max(1, window // 2)
    smooth = net.copy()
    present = [_runs(np.asarray(np.isfinite(net[:, i]).all(axis=1))) for i in range(net.shape[1])]
    # the running mean of the correction stays inside one stretch (never mixes two shots)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, runs in enumerate(present):
            for a, b in runs:
                for t in range(a, b):
                    lo, hi = max(a, t - half), min(b, t + half + 1)
                    smooth[t, i] = net[t, i] + np.nanmean(off[lo:hi, i], axis=0)
    out = snp.copy()
    keep_px = ON_LINE_PX * (width or 1920.0) / 1920.0
    for i, runs in enumerate(present):
        for a, b in runs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                typical = float(np.nanmedian(np.hypot(*off[a:b, i].T)))
            if not np.isfinite(typical) or typical < keep_px:
                out[a:b, i] = net[a:b, i]  # the network is right here: never touch it
                continue
            j_snap, j_smooth = _jitter(snp[a:b, i]), _jitter(smooth[a:b, i])
            if np.isfinite(j_smooth) and (not np.isfinite(j_snap) or j_smooth < j_snap):
                out[a:b, i] = smooth[a:b, i]
    return out


def _runs(present: np.ndarray, max_gap: int = 2) -> list[tuple[int, int]]:
    """``[start, stop)`` stretches where a point is present (gaps up to ``max_gap`` bridged)."""
    idx = np.flatnonzero(present)
    if not len(idx):
        return []
    runs, start = [], idx[0]
    for prev, cur in zip(idx[:-1], idx[1:], strict=True):
        if cur - prev > max_gap + 1:
            runs.append((int(start), int(prev) + 1))
            start = cur
    runs.append((int(start), int(idx[-1]) + 1))
    return runs
