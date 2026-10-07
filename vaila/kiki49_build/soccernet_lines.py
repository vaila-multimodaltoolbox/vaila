"""
================================================================================
Script: soccernet_lines.py - kiki49_build SoccerNet line annotations to keypoints
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.7
Created: 06 October 2026
Update Date: 06 October 2026

Ported from mkvis3d ``openbiomech/soccer_field`` (the builder of the kiki49
dataset) so FreeKiki can convert external field datasets by itself.

Description:
    SoccerNet calibration-2023 line annotations -> kiki49 points + metric homography.

    Each annotation JSON maps a line class to a polyline of normalized image
    points. Class names and their pitch geometry follow the SoccerNet
    ``sn-calibration`` ``soccerpitch.py`` model, re-expressed in the kiki frame
    (its y axis points to the near touchline and z down; kiki +y is the far
    "top" touchline and z is up).

    Keypoints come from the annotations themselves: straight-line intersections,
    circle x line crossings (penalty-arc and centre-circle points), post x goal
    line (post bases) and post x crossbar (post tops). A metric ground-plane
    homography is fitted to all planar evidence at once -- intersection points,
    straight-line samples and circle samples -- so that the remaining points can
    be projected. The TLS line fit and line intersection follow vailá
    ``vaila/fifa_dataset_builder.py`` (``_fit_line_abc``, ``_line_intersection``).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .kiki49 import (
    BOX_DEPTH,
    BOX_HALF_WIDTH,
    CIRCLE_RADIUS,
    GOAL_AREA_DEPTH,
    GOAL_AREA_HALF_WIDTH,
    HALF_LENGTH,
    HALF_WIDTH,
    N_KPT,
    PENALTY_SPOT,
    load_kiki49,
)

Point2 = tuple[float, float]


def _ground_lines() -> dict[str, tuple[Point2, Point2]]:
    L, W = HALF_LENGTH, HALF_WIDTH
    out: dict[str, tuple[Point2, Point2]] = {
        "Side line top": ((-L, W), (L, W)),
        "Side line bottom": ((-L, -W), (L, -W)),
        "Side line left": ((-L, -W), (-L, W)),
        "Side line right": ((L, -W), (L, W)),
        "Middle line": ((0.0, -W), (0.0, W)),
    }
    for side, sx in (("left", -1.0), ("right", 1.0)):
        for rect, depth, half in (
            ("Big rect.", BOX_DEPTH, BOX_HALF_WIDTH),
            ("Small rect.", GOAL_AREA_DEPTH, GOAL_AREA_HALF_WIDTH),
        ):
            gx, ix = sx * L, sx * (L - depth)
            out[f"{rect} {side} top"] = ((gx, half), (ix, half))
            out[f"{rect} {side} main"] = ((ix, -half), (ix, half))
            out[f"{rect} {side} bottom"] = ((gx, -half), (ix, -half))
    return out


GROUND_LINES = _ground_lines()
CIRCLES: dict[str, Point2] = {
    "Circle central": (0.0, 0.0),
    "Circle left": (-(HALF_LENGTH - PENALTY_SPOT), 0.0),
    "Circle right": (HALF_LENGTH - PENALTY_SPOT, 0.0),
}
# circle, straight line -> (kiki index of the far (+y) crossing, near (-y) crossing)
CIRCLE_CROSSINGS: dict[tuple[str, str], tuple[int, int]] = {
    ("Circle left", "Big rect. left main"): (10, 11),
    ("Circle right", "Big rect. right main"): (18, 19),
    ("Circle central", "Middle line"): (14, 15),
}
# post class -> (kiki base index, kiki top index). sn-calibration names the
# posts as seen from the main camera: "left goal, left post" is the near post
# (y = -3.66); pinned empirically by ``post_named_err`` vs ``post_swap_err``.
POSTS: dict[str, tuple[int, int]] = {
    "Goal left post left": (32, 34),
    "Goal left post right": (33, 35),
    "Goal right post left": (41, 43),
    "Goal right post right": (40, 42),
}
POST_SWAP = {32: 33, 33: 32, 40: 41, 41: 40}
GOAL_LINE = {"left": "Side line left", "right": "Side line right"}
CROSSBAR = {"left": "Goal left crossbar", "right": "Goal right crossbar"}


def _on_segment(p: np.ndarray, seg: tuple[Point2, Point2], tol: float = 1e-6) -> bool:
    a, b = np.array(seg[0]), np.array(seg[1])
    return bool(
        np.all(p >= np.minimum(a, b) - tol)
        and np.all(p <= np.maximum(a, b) + tol)
        and abs((b - a)[0] * (p - a)[1] - (b - a)[1] * (p - a)[0]) <= tol
    )


def _intersection_table() -> dict[int, tuple[str, str]]:
    """kiki index -> the two annotated straight lines whose crossing it is."""
    xyz = load_kiki49().xyz
    table: dict[int, tuple[str, str]] = {}
    for (na, sa), (nb, sb) in itertools.combinations(GROUND_LINES.items(), 2):
        la = np.cross([*sa[0], 1.0], [*sa[1], 1.0])
        lb = np.cross([*sb[0], 1.0], [*sb[1], 1.0])
        q = np.cross(la, lb)
        if abs(q[2]) < 1e-12:
            continue
        p = q[:2] / q[2]
        if not (_on_segment(p, sa) and _on_segment(p, sb)):
            continue
        hit = np.flatnonzero((np.abs(xyz[:, :2] - p).max(axis=1) < 1e-6) & (xyz[:, 2] == 0.0))
        for k in hit:
            table[int(k)] = (na, nb)
    return table


INTERSECTIONS = _intersection_table()


def fit_line(P: np.ndarray) -> np.ndarray:
    """Total-least-squares line ``a x + b y + c = 0`` with ``a^2 + b^2 = 1``."""
    m = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - m)
    n = Vt[-1]
    return np.array([n[0], n[1], -float(n @ m)])


def intersect(l1: np.ndarray, l2: np.ndarray, min_angle_deg: float = 2.0) -> np.ndarray | None:
    """Crossing of two normalized lines; None when they are nearly parallel."""
    if abs(l1[0] * l2[1] - l1[1] * l2[0]) < np.sin(np.radians(min_angle_deg)):
        return None
    q = np.cross(l1, l2)
    return q[:2] / q[2]


def circle_line_crossings(P: np.ndarray, line: np.ndarray, tol: float) -> list[np.ndarray]:
    """Where an annotated circle polyline meets a line (sign changes, plus
    polyline ends lying within ``tol`` of it, projected onto it)."""
    s = P @ line[:2] + line[2]
    out: list[np.ndarray] = []
    for i in range(len(P) - 1):
        if s[i] * s[i + 1] < 0:
            t = s[i] / (s[i] - s[i + 1])
            out.append(P[i] + t * (P[i + 1] - P[i]))
    for i in (0, len(P) - 1):
        if abs(s[i]) <= tol:
            out.append(P[i] - s[i] * line[:2])
    uniq: list[np.ndarray] = []
    for q in out:
        if all(np.hypot(*(q - u)) > 2.0 * tol for u in uniq):
            uniq.append(q)
    return uniq


def _normalizers(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    Ti = np.array([[2.0 / width, 0.0, -1.0], [0.0, 2.0 / width, -height / width], [0.0, 0.0, 1.0]])
    Tw = np.diag([1.0 / 50.0, 1.0 / 50.0, 1.0])
    return Ti, Tw


def dlt_points_lines(
    world_pts: np.ndarray,
    img_pts: np.ndarray,
    world_lines: np.ndarray,
    img_lines: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray | None:
    """Homography world -> image from point and line correspondences (normalized DLT).

    Points satisfy ``p ~ H X``; lines satisfy ``l_world ~ H^T l_image``.
    """
    if 2 * len(world_pts) + 2 * len(world_lines) < 8:
        return None
    Ti, Tw = _normalizers(width, height)
    rows = []
    for X, p in zip(world_pts, img_pts, strict=True):
        x, y, _ = Tw @ [X[0], X[1], 1.0]
        u, v, _ = Ti @ [p[0], p[1], 1.0]
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
    Ti_T, Tw_T = np.linalg.inv(Ti).T, np.linalg.inv(Tw).T
    for lw, li in zip(world_lines, img_lines, strict=True):
        lw = Tw_T @ lw
        li = Ti_T @ li
        lw, li = lw / np.linalg.norm(lw), li / np.linalg.norm(li)
        # m = H^T li, m_j = sum_k li_k h[3k + j]; rows of lw x m = 0.
        for a, b in ((1, 2), (2, 0), (0, 1)):
            r = np.zeros(9)
            for k in range(3):
                r[3 * k + b] += lw[a] * li[k]
                r[3 * k + a] -= lw[b] * li[k]
            rows.append(r)
    A = np.asarray(rows, dtype=np.float64)
    _, s, Vt = np.linalg.svd(A)
    if s[-2] < 1e-9 * s[0]:
        return None
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Ti) @ Hn @ Tw
    if not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-12:
        return None
    return H / H[2, 2]


@dataclass
class _Evidence:
    pt_world: list[np.ndarray] = field(default_factory=list)
    pt_img: list[np.ndarray] = field(default_factory=list)
    line_world: list[np.ndarray] = field(default_factory=list)
    line_img: list[np.ndarray] = field(default_factory=list)
    line_samples: list[tuple[np.ndarray, np.ndarray]] = field(
        default_factory=list
    )  # (world line, px)
    circle_samples: list[tuple[np.ndarray, np.ndarray]] = field(
        default_factory=list
    )  # (centre, px)


def _residuals(H: np.ndarray, ev: _Evidence) -> np.ndarray:
    """Pixel residuals of every planar observation under ``H``."""
    res = []
    if ev.pt_world:
        X = np.column_stack([np.asarray(ev.pt_world), np.ones(len(ev.pt_world))]) @ H.T
        res.append((X[:, :2] / X[:, 2:3] - np.asarray(ev.pt_img)).ravel())
    Hinv_T = np.linalg.inv(H).T
    for lw, P in ev.line_samples:
        li = Hinv_T @ lw
        res.append((P @ li[:2] + li[2]) / np.hypot(li[0], li[1]))
    if ev.circle_samples:
        Hinv = np.linalg.inv(H)
        detH = np.linalg.det(H)
        for c, P in ev.circle_samples:
            q = np.column_stack([P, np.ones(len(P))]) @ Hinv.T
            X = q[:, :2] / q[:, 2:3]
            w = np.column_stack([X, np.ones(len(X))]) @ H[2]
            scale = np.sqrt(np.abs(detH / w**3))  # px per metre at X
            res.append((np.hypot(X[:, 0] - c[0], X[:, 1] - c[1]) - CIRCLE_RADIUS) * scale)
    return np.concatenate(res) if res else np.zeros(0)


def refine_homography(H0: np.ndarray, ev: _Evidence, f_scale_px: float) -> np.ndarray | None:
    """Robust refinement of ``H0``; None when ``H0`` is degenerate for the evidence."""

    def fun(h: np.ndarray) -> np.ndarray:
        return _residuals(np.append(h, 1.0).reshape(3, 3), ev)

    try:
        with np.errstate(all="ignore"):
            h0 = (H0 / H0[2, 2]).ravel()[:8]
            if not np.all(np.isfinite(h0)) or not np.all(np.isfinite(fun(h0))):
                return None
            sol = least_squares(fun, h0, loss="soft_l1", f_scale=f_scale_px)
    except (ValueError, np.linalg.LinAlgError):  # singular / non-finite during the fit
        return None
    return np.append(sol.x, 1.0).reshape(3, 3)


@dataclass
class LineLabels:
    annotated: np.ndarray  # (49, 3) pixels, v = 2 where the annotation defines the point
    H: np.ndarray | None  # metric ground plane -> pixels
    residual_px: np.ndarray  # per planar observation, pixels (empty without H)
    stats: dict[str, float]


def parse_annotation(
    ann: dict[str, list[dict[str, float]]],
    width: int,
    height: int,
    seed_points: dict[int, np.ndarray] | None = None,
) -> LineLabels:
    """kiki49 evidence from one calibration-2023 JSON (normalized polylines).

    ``seed_points`` ({kiki index: pixel}, e.g. network keypoints) anchor the
    homography when the human lines alone do not define it; they never become
    annotated points and are left out of ``residual_px``, so the residual
    measures how well the human lines agree with the anchored plane.
    """
    diag = float(np.hypot(width, height))
    tol = 0.01 * diag
    polys: dict[str, np.ndarray] = {}
    for name, pts in ann.items():
        P = np.array([[p["x"] * width, p["y"] * height] for p in pts], dtype=np.float64)
        if len(P) >= 2 and np.all(np.isfinite(P)):
            polys[name.strip()] = P
    lines = {n: fit_line(P) for n, P in polys.items() if not n.startswith("Circle")}

    annotated = np.zeros((N_KPT, 3))
    xyz = load_kiki49().xyz
    margin = 0.5 * np.array([width, height])
    ev = _Evidence()

    def add_point(k: int, p: np.ndarray) -> None:
        if np.all(p > -margin) and np.all(p < np.array([width, height]) + margin):
            annotated[k] = [p[0], p[1], 2.0]
            ev.pt_world.append(xyz[k, :2])
            ev.pt_img.append(p)

    for k, (na, nb) in INTERSECTIONS.items():
        if na in lines and nb in lines:
            p = intersect(lines[na], lines[nb])
            if p is not None:
                add_point(k, p)
    for name, seg in GROUND_LINES.items():
        if name in polys:
            lw = np.cross([*seg[0], 1.0], [*seg[1], 1.0])
            ev.line_world.append(lw)
            ev.line_img.append(lines[name])
            ev.line_samples.append((lw, polys[name]))
    for name, c in CIRCLES.items():
        if name in polys:
            ev.circle_samples.append((np.array(c), polys[name]))

    H = dlt_points_lines(
        np.asarray(ev.pt_world).reshape(-1, 2),
        np.asarray(ev.pt_img).reshape(-1, 2),
        np.asarray(ev.line_world).reshape(-1, 3),
        np.asarray(ev.line_img).reshape(-1, 3),
        width,
        height,
    )
    seed = _Evidence()
    if seed_points:
        seed = _Evidence(
            pt_world=ev.pt_world + [xyz[k, :2] for k in seed_points],
            pt_img=ev.pt_img + [np.asarray(p, dtype=float) for p in seed_points.values()],
            line_world=ev.line_world,
            line_img=ev.line_img,
            line_samples=ev.line_samples,
            circle_samples=ev.circle_samples,
        )
        if H is None:
            H = dlt_points_lines(
                np.asarray(seed.pt_world).reshape(-1, 2),
                np.asarray(seed.pt_img).reshape(-1, 2),
                np.asarray(seed.line_world).reshape(-1, 3),
                np.asarray(seed.line_img).reshape(-1, 3),
                width,
                height,
            )
    fit_ev = seed if seed_points else ev
    if H is not None and ev.circle_samples:
        H = refine_homography(H, fit_ev, 2.0)

    for (circ, lname), (far, near) in CIRCLE_CROSSINGS.items():
        if circ not in polys or lname not in lines:
            continue
        cand = circle_line_crossings(polys[circ], lines[lname], tol)
        if H is not None:
            pred = {k: _apply(H, xyz[k, :2]) for k in (far, near)}
            for q in cand:
                k = min(pred, key=lambda j: np.hypot(*(pred[j] - q)))
                if np.hypot(*(pred[k] - q)) < 0.05 * diag:
                    add_point(k, q)
        elif len(cand) == 2:
            lo, hi = sorted(cand, key=lambda q: q[1])
            add_point(far, lo)
            add_point(near, hi)

    stats: dict[str, float] = {
        "n_points": float(len(ev.pt_world)),
        "n_lines": float(len(ev.line_world)),
    }

    # Goal posts: bases on the goal line are planar evidence, tops are 3D.
    named, swapped = [], []
    for name, (base, top) in POSTS.items():
        if name not in lines:
            continue
        side = "left" if " left post" in name else "right"
        gl, cb = GOAL_LINE[side], CROSSBAR[side]
        if gl in lines:
            p = intersect(lines[name], lines[gl], min_angle_deg=10.0)
            if p is not None:
                if H is not None:
                    named.append(np.hypot(*(_apply(H, xyz[base, :2]) - p)))
                    swapped.append(np.hypot(*(_apply(H, xyz[POST_SWAP[base], :2]) - p)))
                add_point(base, p)
        if cb in lines:
            p = intersect(lines[name], lines[cb], min_angle_deg=10.0)
            if (
                p is not None
                and np.all(p > -margin)
                and np.all(p < np.array([width, height]) + margin)
            ):
                annotated[top] = [p[0], p[1], 2.0]
    if named:
        stats["post_named_err"] = float(np.median(named))
        stats["post_swap_err"] = float(np.median(swapped))

    residual = np.zeros(0)
    if H is not None:
        if seed_points:  # human points found after the seed fit join the anchors
            n = len(seed_points)
            fit_ev.pt_world = ev.pt_world + fit_ev.pt_world[-n:]
            fit_ev.pt_img = ev.pt_img + fit_ev.pt_img[-n:]
        H = refine_homography(H, fit_ev, 2.0)
        if H is not None:
            with np.errstate(all="ignore"):
                residual = np.abs(_residuals(H, ev))
            if not (np.all(np.isfinite(H)) and np.all(np.isfinite(residual))):
                H, residual = None, np.zeros(0)
    return LineLabels(annotated=annotated, H=H, residual_px=residual, stats=stats)


def _apply(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    p = H @ np.array([xy[0], xy[1], 1.0])
    return p[:2] / p[2]
