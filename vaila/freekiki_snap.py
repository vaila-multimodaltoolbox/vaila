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
Version: 0.4.9
Created: 15 October 2026
Update Date: 17 October 2026

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
      after transporting its along-line position with the snapped shift of the
      line's endpoints (so an oblique shift on a tilted line never slides the
      point along the line).

    A move larger than ``max_move_px`` (at 1920 px width) is refused: the paint
    found is then not the line the network meant. Pure numpy / OpenCV; nothing
    here changes the network CSVs (``freekiki.py snap`` writes its own).
"""

from __future__ import annotations

from functools import lru_cache

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
# Parallel line pencils on the soccer field rectangle hierarchy:
# longitudinal lines (parallel to X axis, converging at VP_x) and
# transverse lines (parallel to Y axis, converging at VP_y).
LONGITUDINAL_LINES: tuple[str, ...] = (
    "touchline_top",
    "box_left_top",
    "box_right_top",
    "goal_area_left_top",
    "goal_area_right_top",
    "goal_area_left_bottom",
    "goal_area_right_bottom",
    "box_left_bottom",
    "box_right_bottom",
    "touchline_bottom",
)
TRANSVERSE_LINES: tuple[str, ...] = (
    "goal_line_left",
    "goal_area_left_side",
    "box_left_side",
    "halfway",
    "box_right_side",
    "goal_area_right_side",
    "goal_line_right",
)
# Vertical pole/post pairs: ground base (z=0) -> elevated top (z>0).
# A ground corner/post base must never snap upward along its pole toward the top.
VERTICAL_POLE_PAIRS: dict[int, int] = {
    0: 38,
    5: 39,
    24: 46,
    29: 47,
    32: 34,
    33: 35,
    40: 42,
    41: 43,
}
# Center circle diamond (top, left, bottom, right, center spot) and longitudinal axis.
CENTER_DIAMOND: tuple[int, ...] = (14, 30, 15, 31, 48)
LONGITUDINAL_CENTER_AXIS: tuple[int, ...] = (8, 30, 48, 31, 21)

BAND_PX = 35.0  # half width of the search band around a predicted line, px at 1920
MAX_MOVE_PX = 40.0  # a snapped keypoint may move at most this much, px at 1920
MAX_ANGLE_DEG = 8.0  # refitted line vs predicted line
MIN_PAINT_PX = 40  # paint pixels needed to refit a line
ON_LINE_PX = 3.0  # px at 1920: a network point this close to its refitted line(s) is kept
LINE_ON_PAINT = 0.6  # share of a refitted segment that must lie on paint
MIN_CROSS_DEG = 15.0  # two lines crossing at a shallower angle give an unstable corner


@lru_cache(maxsize=1)
def _field_xyz() -> tuple[np.ndarray, np.ndarray]:
    """Cached ``(xyz (49, 3), planar_mask (49,))`` from ``soccerfield_kiki.csv``."""
    try:
        from . import freekiki_diag as diag
    except ImportError:
        import freekiki_diag as diag  # ty: ignore[unresolved-import]
    _, _, xyz = diag.load_field_points()
    xyz.setflags(write=False)
    planar = xyz[:, 2] == 0.0
    planar.setflags(write=False)
    return xyz, planar


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
    predicted segment (5 % beyond each end); a 1D normal-offset ridge search
    over quantized angles finds the sharpest straight stroke (immune to curved
    penalty arcs, centre circles or players inside the band), followed by a
    tight SVD refit on the stroke inliers. With ``paint`` (the bool mask), the
    refitted segment between a and b must lie on paint for at least
    ``LINE_ON_PAINT`` of its length: a line pulled by goal posts, the net or an
    advertising board does not.
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
    scale = (paint.shape[1] / 1920.0) if paint is not None else max(1.0, band / BAND_PX)
    half_stroke = max(2.0, 2.5 * scale)
    mid = 0.5 * (a + b)
    rel_m = q - mid
    u_ang = float(np.degrees(np.arctan2(u[1], u[0])))
    a_lo = np.ceil((u_ang - MAX_ANGLE_DEG) * 4.0) / 4.0
    a_hi = np.floor((u_ang + MAX_ANGLE_DEG) * 4.0) / 4.0
    angles_deg = np.arange(a_lo, a_hi + 1e-9, 0.25)
    if len(angles_deg) == 0:
        angles_deg = np.array([u_ang])
    rads = np.radians(angles_deg)
    normals = np.c_[-np.sin(rads), np.cos(rads)]
    proj = rel_m @ normals.T
    step = max(0.25, 0.5 * scale)
    bins = np.arange(-band, band + half_stroke, step)
    best_inliers = None
    if len(bins) >= 4:
        win = max(1, int(round(half_stroke / step)))
        kernel = np.bartlett(2 * win + 1)
        best_score = -1.0
        for j in range(len(angles_deg)):
            col = proj[:, j]
            hist, _ = np.histogram(col, bins=bins)
            smooth = np.convolve(hist, kernel, mode="same")
            k_max = int(np.argmax(smooth))
            score = float(smooth[k_max])
            if score > best_score:
                center_off = 0.5 * (bins[k_max] + bins[k_max + 1])
                inl = np.abs(col - center_off) <= half_stroke
                if int(inl.sum()) >= MIN_PAINT_PX:
                    best_score = score
                    best_inliers = inl
    q_in = q[best_inliers] if best_inliers is not None else q
    for _ in range(2):
        c = q_in.mean(axis=0)
        _, _, vt = np.linalg.svd(q_in - c, full_matrices=False)
        dirv = vt[0]
        resid = np.abs((q - c) @ np.array([-dirv[1], dirv[0]]))
        keep = resid <= max(2.5, 3.0 * scale)
        if int(keep.sum()) < MIN_PAINT_PX:
            break
        q_in = q[keep]
    c = q_in.mean(axis=0)
    _, _, vt = np.linalg.svd(q_in - c, full_matrices=False)
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


def _perp_dist(line: tuple[np.ndarray, np.ndarray], p: np.ndarray) -> float:
    c, d = line
    return abs(float((np.asarray(p, dtype=float) - c) @ np.array([-d[1], d[0]])))


def _projective_1d_along_line(
    lname: str,
    k: int,
    on: list[int],
    xy: np.ndarray,
    line: tuple[np.ndarray, np.ndarray],
    scale: float,
) -> np.ndarray | None:
    """Predict keypoint ``k`` on ``line`` from the 1D projective cross-ratio of its colinear peers.

    Any >= 3 colinear points on a straight pitch line determine a 1D projective
    homography ``s(S) = (a*S + b) / (c1*S + 1)`` from known metric pitch coordinates
    ``S`` (metres) to 1D pixel arclength ``s`` along ``line``.
    """
    others = [m for m in on if m != k and np.isfinite(xy[m]).all()]
    if len(others) < 3:
        return None
    xyz, _ = _field_xyz()
    axis = 0 if lname in LONGITUDINAL_LINES else 1
    S_others = np.array([float(xyz[m, axis]) for m in others], dtype=float)
    if float(np.ptp(S_others)) < 3.0:
        return None
    c, d = line
    s_others = np.array([float((xy[m] - c) @ d) for m in others], dtype=float)
    # Normalize S for numerical conditioning
    S_mean = float(np.mean(S_others))
    S_scale = max(1.0, float(np.std(S_others)))
    Sn = (S_others - S_mean) / S_scale
    A = np.c_[Sn, np.ones(len(Sn)), -s_others * Sn]
    try:
        params, _, _, _ = np.linalg.lstsq(A, s_others, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, b, c1 = float(params[0]), float(params[1]), float(params[2])
    denom_others = c1 * Sn + 1.0
    if np.any(denom_others <= 0.15):
        return None
    s_fit = (a * Sn + b) / denom_others
    if float(np.median(np.abs(s_fit - s_others))) > 6.0 * scale:
        return None
    Sn_k = (float(xyz[k, axis]) - S_mean) / S_scale
    denom_k = c1 * Sn_k + 1.0
    if denom_k <= 0.15:
        return None
    s_k = (a * Sn_k + b) / denom_k
    return c + s_k * d


def _loo_planar_prediction(
    xy: np.ndarray, acc: np.ndarray, k: int, width: float
) -> np.ndarray | None:
    """Leave-one-out planar homography prediction for keypoint ``k`` using all other ground points."""
    xyz, planar = _field_xyz()
    if not planar[k]:
        return None
    use = acc.copy() & planar & np.isfinite(xy).all(axis=1)
    use[k] = False
    if int(use.sum()) < 6:
        return None
    try:
        from . import freekiki_diag as diag
    except ImportError:
        import freekiki_diag as diag  # ty: ignore[unresolved-import]
    fit = diag.fit_field_homography(
        xy[use], xyz[use, :2], width=width, ransac_px=10.0, max_rmse_px=6.0
    )
    if fit.get("status") != "ok" or fit.get("H") is None:
        return None
    H = np.asarray(fit["H"], dtype=float)
    ph = H @ np.array([float(xyz[k, 0]), float(xyz[k, 1]), 1.0])
    if abs(float(ph[2])) <= 1e-9:
        return None
    return ph[:2] / float(ph[2])


def _vanishing_point(
    fitted: dict[str, tuple[np.ndarray, np.ndarray]],
    members_on: dict[str, list[int]],
    pencil: tuple[str, ...],
) -> np.ndarray | None:
    """Weighted least-squares vanishing point of parallel pitch lines in ``pencil``."""
    lines = [(fitted[n], len(members_on.get(n, ()))) for n in pencil if n in fitted]
    if len(lines) < 2:
        return None
    # Prefer lines with >= 3 colinear points when available
    strong = [(ln, cnt) for ln, cnt in lines if cnt >= 3]
    use_lines = strong if len(strong) >= 2 else lines
    A = np.zeros((2, 2), dtype=float)
    b = np.zeros(2, dtype=float)
    for (c, d), cnt in use_lines:
        n_vec = np.array([-d[1], d[0]], dtype=float)
        w = float(cnt**2)
        A += w * np.outer(n_vec, n_vec)
        b += w * float(n_vec @ c) * n_vec
    if abs(float(np.linalg.det(A))) <= 1e-8:
        return None
    try:
        vp = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return vp if np.isfinite(vp).all() else None


LOCAL_SPAN_PX = 320.0  # local reach along a long distorted line when refitting near a corner


def _accept_local_line(
    loc: tuple[np.ndarray, np.ndarray] | None,
    global_line: tuple[np.ndarray, np.ndarray],
    xy: np.ndarray,
    k: int,
    nb_indices: tuple[int, ...],
    on: list[int],
    scale: float,
) -> bool:
    """Guard against a local 2-point refit jumping onto a parallel advertising board / goalpost.

    On a line with >= 3 accepted points, ``global_line`` is anchored by multiple
    colinear landmarks. A local refit around ``k`` is accepted for radial-distortion
    tangent tracking (e.g. ``p9``/``p17``) only when it stays faithful to the
    neighboring anchor(s) ``nb_indices`` and does not jump to a false stroke far
    from ``xy[k]`` when ``global_line`` already passes through ``xy[k]``.
    """
    if loc is None:
        return False
    if len(on) < 3:
        return True
    p_glob = _perp_dist(global_line, xy[k])
    p_loc = _perp_dist(loc, xy[k])
    for nb in nb_indices:
        p_nb_glob = _perp_dist(global_line, xy[nb])
        p_nb_loc = _perp_dist(loc, xy[nb])
        if p_nb_loc > max(8.0 * scale, p_nb_glob + 4.0 * scale):
            return False
    return not (p_glob <= 6.0 * scale and p_loc > p_glob + 4.0 * scale)


def _local_line_for_point(
    pts: np.ndarray,
    xy: np.ndarray,
    k: int,
    on: list[int],
    band: float,
    paint: np.ndarray,
    global_line: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Refit the line locally around keypoint ``k`` so radial lens distortion on long lines
    (such as ``box_left_side`` p9..p12 or ``box_right_side`` p17..p20) never pulls a corner
    along the crossing line, while rejecting false parallel strokes (perimeter boards).
    """
    scale = paint.shape[1] / 1920.0
    is_corner = len(lines_of(k)) >= 2
    if not is_corner and k in on:
        idx = on.index(k)
        if 0 < idx < len(on) - 1:
            ia, ib = on[idx - 1], on[idx + 1]
            if float(np.hypot(*(xy[ib] - xy[ia]))) >= 80.0 * scale:
                loc = fit_line_in_band(pts, xy[ia], xy[ib], band, paint)
                if _accept_local_line(loc, global_line, xy, k, (ia, ib), on, scale):
                    assert loc is not None
                    return loc
    others = [m for m in on if m != k and float(np.hypot(*(xy[m] - xy[k]))) >= 80.0 * scale]
    if not others:
        others = [m for m in on if m != k]
    if not others:
        return global_line
    nb = min(others, key=lambda m: float(np.hypot(*(xy[m] - xy[k]))))
    a, b = xy[k], xy[nb]
    seg_len = float(np.hypot(*(b - a)))
    if is_corner and seg_len > LOCAL_SPAN_PX * scale:
        b_local = a + (b - a) * (LOCAL_SPAN_PX * scale / seg_len)
        loc = fit_line_in_band(pts, a, b_local, band, paint)
        if _accept_local_line(loc, global_line, xy, k, (nb,), on, scale):
            assert loc is not None
            return loc
    if (k, nb) != (on[0], on[-1]) and (nb, k) != (on[0], on[-1]):
        loc = fit_line_in_band(pts, a, b, band, paint)
        if _accept_local_line(loc, global_line, xy, k, (nb,), on, scale):
            assert loc is not None
            return loc
    return global_line


def _violates_pole_guard(k: int, new: np.ndarray, xy: np.ndarray, scale: float) -> bool:
    """True when ``new`` pulls ground base ``k`` upward along its flagpole/goalpost toward the top."""
    top_k = VERTICAL_POLE_PAIRS.get(k)
    if top_k is None or not np.isfinite(xy[top_k]).all():
        return False
    pole_vec = xy[top_k] - xy[k]
    pole_len = float(np.hypot(*pole_vec))
    if pole_len < 6.0 * scale:
        return False
    u_pole = pole_vec / pole_len
    move_vec = new - xy[k]
    along_pole = float(move_vec @ u_pole)
    perp_pole = abs(float(move_vec @ np.array([-u_pole[1], u_pole[0]])))
    return along_pole > 6.0 * scale and along_pole > 1.5 * perp_pole


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
    width = float(paint.shape[1])
    scale = width / 1920.0
    ys, xs = np.nonzero(paint)
    pts = np.c_[xs, ys].astype(float)
    fitted = {}
    spans = {}
    members_on = {}
    for name, members in PAINTED_LINES.items():
        on = [i for i in members if acc[i] and np.isfinite(xy[i]).all()]
        if len(on) < 2:
            continue
        # the two members farthest apart span the predicted segment
        ia, ib = max(
            ((on[a], on[b]) for a in range(len(on)) for b in range(a + 1, len(on))),
            key=lambda ab: float(np.hypot(*(xy[ab[0]] - xy[ab[1]]))),
        )
        line = fit_line_in_band(pts, xy[ia], xy[ib], band_px * scale, paint)
        if line is not None:
            fitted[name] = line
            spans[name] = (ia, ib)
            members_on[name] = on
    out, how = xy.copy(), {}
    for k in np.flatnonzero(acc):
        k = int(k)
        lns = [n for n in lines_of(k) if n in fitted]
        mine = [
            _local_line_for_point(pts, xy, k, members_on[n], band_px * scale, paint, fitted[n])
            for n in lns
        ]
        if not mine:
            continue
        all_lns = lines_of(k)
        thr = (ON_LINE_PX if len(mine) >= 2 or len(all_lns) == 1 else 1.5 * ON_LINE_PX) * scale
        perps = [_perp_dist(m, xy[k]) for m in mine]
        if all(p <= thr for p in perps) or max(perps) > max_move_px * scale:
            continue
        new = None
        kind = "line"
        if len(mine) >= 2:
            new, kind = _intersect(mine[0], mine[1]), "corner"
            # Structural rectangle / colinearity guard: when one line has >= 3 colinear points
            # and the crossing line has only 2 points (e.g. touchline near advertising boards),
            # verify that the crossing line does not slide the corner along the main line away
            # from the 1D projective cross-ratio and LOO planar homography.
            counts = [len(members_on[n]) for n in lns[:2]]
            if new is not None and max(counts) >= 3 and min(counts) == 2:
                idx_main = 0 if counts[0] >= counts[1] else 1
                lname_main = lns[idx_main]
                m_main = mine[idx_main]
                d_main = m_main[1]
                perp_main = perps[idx_main]
                along_shift = abs(float((new - xy[k]) @ d_main))
                if along_shift > 10.0 * scale:
                    p_cr = _projective_1d_along_line(
                        lname_main, k, members_on[lname_main], xy, m_main, scale
                    )
                    p_loo = _loo_planar_prediction(xy, acc, k, width)
                    refs = [p for p in (p_cr, p_loo) if p is not None and np.isfinite(p).all()]
                    bad_cross = False
                    if refs:
                        ref_pt = np.mean(refs, axis=0)
                        err_new = abs(float((new - ref_pt) @ d_main))
                        err_raw = abs(float((xy[k] - ref_pt) @ d_main))
                        if err_new > max(10.0 * scale, err_raw + 6.0 * scale):
                            bad_cross = True
                    elif perp_main <= 5.0 * scale and along_shift > 3.0 * max(
                        perp_main, 2.0 * scale
                    ):
                        bad_cross = True
                    if bad_cross:
                        if perp_main > ON_LINE_PX * scale:
                            new, kind = _foot(m_main, xy[k]), "line"
                        else:
                            new = None
        elif len(mine) == 1:
            lname = lns[0]
            ia, ib = spans[lname]
            cross_a = [fitted[n][1] for n in lines_of(ia) if n != lname and n in fitted]
            cross_b = [fitted[n][1] for n in lines_of(ib) if n != lname and n in fitted]
            if len(all_lns) == 1 and (cross_a or cross_b):
                da = cross_a[0] if cross_a else cross_b[0]
                db = cross_b[0] if cross_b else cross_a[0]
                if float(da @ db) < 0:
                    db = -db
                seg = xy[ib] - xy[ia]
                seg_len2 = float(seg @ seg)
                t_param = (
                    float(np.clip(((xy[k] - xy[ia]) @ seg) / seg_len2, 0.0, 1.0))
                    if seg_len2 > 1.0
                    else 0.5
                )
                dc = (1.0 - t_param) * da + t_param * db
                dc_norm = float(np.hypot(*dc))
                if dc_norm > 1e-6:
                    new = _intersect(mine[0], (xy[k], dc / dc_norm))
            if new is None:
                new = _foot(mine[0], xy[k])
            kind = "line"
        if new is None or not np.isfinite(new).all():
            continue
        if _violates_pole_guard(k, new, xy, scale):
            continue
        if np.hypot(*(new - xy[k])) <= max_move_px * scale:
            out[k] = new
            how[k] = kind
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
    snp_filled = snp.copy()
    present = [_runs(np.asarray(np.isfinite(net[:, i]).all(axis=1))) for i in range(net.shape[1])]
    keep_px = ON_LINE_PX * (width or 1920.0) / 1920.0
    # the running mean of the correction stays inside one stretch (never mixes two shots)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, runs in enumerate(present):
            for a, b in runs:
                dists = np.hypot(*off[a:b, i].T)
                typical = float(np.nanmedian(dists))
                off_run = off[a:b, i].copy()
                dropout = dists < 1e-4
                if np.isfinite(typical) and typical >= keep_px and np.any(~dropout):
                    off_run[dropout] = np.nan
                for t in range(a, b):
                    lo, hi = max(a, t - half) - a, min(b, t + half + 1) - a
                    mean_off = np.nanmean(off_run[lo:hi], axis=0)
                    if not np.isfinite(mean_off).all():
                        smooth[t, i] = net[t, i]
                        continue
                    smooth[t, i] = net[t, i] + mean_off
                    if np.isfinite(typical) and typical >= keep_px and dropout[t - a]:
                        snp_filled[t, i] = smooth[t, i]
    out = snp_filled.copy()
    for i, runs in enumerate(present):
        for a, b in runs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                typical = float(np.nanmedian(np.hypot(*off[a:b, i].T)))
            if not np.isfinite(typical) or typical < keep_px:
                out[a:b, i] = net[a:b, i]  # the network is right here: never touch it
                continue
            j_snap, j_smooth = _jitter(snp_filled[a:b, i]), _jitter(smooth[a:b, i])
            if np.isfinite(j_smooth) and (not np.isfinite(j_snap) or j_smooth < j_snap):
                out[a:b, i] = smooth[a:b, i]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                final_typ = float(np.nanmedian(np.hypot(*(out[a:b, i] - net[a:b, i]).T)))
            if not np.isfinite(final_typ) or final_typ < keep_px:
                out[a:b, i] = net[a:b, i]
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
