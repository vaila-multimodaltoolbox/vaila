"""
================================================================================
Script: freekiki_geom.py - FreeKiki field geometry (camera from keypoints)
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.7
Created: 04 October 2026
Update Date: 05 October 2026

Description:
    The pitch is a known 3D object (``vaila/models/soccerfield_kiki.csv``,
    metres, z up). Once some of the 49 keypoints are found in a frame, a camera
    is fitted to them and predicts where every other keypoint must be:

    * planar homography (world z = 0 -> image), RANSAC, from
      ``freekiki_diag.fit_field_homography``;
    * full camera for the points off the ground (flag tops z = 1.5 m, goal
      post tops z = 2.44 m):
        - DLT3D (11 parameters, ``dlt3d.calculate_dlt3d_params``) when >= 7
          points including >= 2 off the ground are available, pruned of
          points that disagree and checked with ``decompose_dlt3d``
          (square pixels, no skew, camera above the ground);
        - otherwise a pinhole camera recovered from the homography: principal
          point at the image centre, square pixels, focal from Zhang's two
          constraints (h1'w h2 = 0, h1'w h1 = h2'w h2 with w = diag(1/f^2,
          1/f^2, 1)), then R = [r1 r2 r1 x r2] and t from K^-1 H.

    ``dlt_coefficients`` turns each frame camera into the vailá DLT2D (8) and
    DLT3D (11) coefficients that ``rec2d.py`` / ``rec3d.py`` read, so the
    field calibration can drive per-frame reconstruction of player tracks.

    ``refine_keypoints`` uses that camera on the network output:
      fill  a point the network did not accept (conf < kp_conf, or a
            same-pixel duplicate) is filled when its projection lies inside
            the image and the network still half-sees it (raw conf >=
            min_conf); it keeps the network position when that agrees with
            the projection, else takes the projection (code G);
      fix   fill + an accepted point farther than fix_px from its projection
            is replaced by the projection (Gx); in fill mode it is only
            flagged (Dg);
      off   unchanged.
    A geometry point says where the keypoint must be; it may be occluded.
    Distances are px@1920 (pixels rescaled to a 1920-px-wide image).

License:
    GNU Affero General Public License v3.0 (AGPLv3).
================================================================================
"""

from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache
from typing import Any

import numpy as np

try:
    from . import freekiki_diag as diag
    from .dlt3d import calculate_dlt3d_params
    from .monocular_dlt_align import decompose_dlt3d, dlt_projection_matrix
except ImportError:
    import freekiki_diag as diag  # ty: ignore[unresolved-import]
    from dlt3d import calculate_dlt3d_params  # ty: ignore[unresolved-import]
    from monocular_dlt_align import (  # ty: ignore[unresolved-import]
        decompose_dlt3d,
        dlt_projection_matrix,
    )

NKP = diag.NKP
GEOM_MODES = ("fill", "fix", "off")
GEOM_CODES = {
    "G": "filled by field geometry (network conf < kp_conf but >= min_conf; inside the image)",
    "Gx": "replaced by field geometry (accepted point farther than fix_px from the camera)",
    "Dg": "accepted, but farther than fix_px from the field geometry (kept in fill mode)",
}
GEOM_DEFAULTS: dict[str, Any] = {
    "mode": "fill",
    "min_conf": 0.05,  # network conf a point needs to be filled (not invented)
    "agree_px": 15.0,  # keep the network xy within this of the projection (px@1920)
    "fix_px": 20.0,  # accepted point farther than this disagrees with the camera
    "fit_px": 15.0,  # RANSAC / DLT3D pruning threshold (px@1920)
    "max_rmse_px": 8.0,  # camera fit RMSE on its own points (px@1920)
    "min_dlt_points": 7,  # DLT3D: 11 unknowns, 7 points keep redundancy
    "min_offplane": 2,  # DLT3D: points off the ground among them
    "min_focal_ratio": 0.2,  # focal / image width
    "max_focal_ratio": 50.0,
    "min_cam_height_m": 1.0,
    "max_cam_height_m": 300.0,
    "max_aspect_dev": 0.25,  # |fy/fx - 1| of a DLT3D camera
    "max_skew": 0.15,  # |K01| / focal of a DLT3D camera
    "margin": 0.01,  # of the image width kept from every border
}
FILLABLE = ("Rk", "Rd")  # network codes geometry may fill


@lru_cache(maxsize=1)
def _field() -> tuple[np.ndarray, np.ndarray]:
    """``(xyz (49, 3) metres, planar mask)``; read once, never modified."""
    _, _, xyz = diag.load_field_points()
    xyz.setflags(write=False)
    planar = xyz[:, 2] == 0
    planar.setflags(write=False)
    return xyz, planar


def _settings(settings: dict | None) -> dict:
    return GEOM_DEFAULTS | dict(settings or {})


def _project(P: np.ndarray, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixels and the in-front mask of world points through a 3x4 camera."""
    uvw = np.c_[xyz, np.ones(len(xyz))] @ P.T
    with np.errstate(invalid="ignore", divide="ignore"):
        uv = uvw[:, :2] / uvw[:, 2:3]
    return uv, uvw[:, 2] > 0


def camera_from_homography(
    H, width: float, height: float, s: dict | None = None, obj=None, img=None
) -> dict | None:
    """Pinhole camera (square pixels, centred principal point) behind a field homography.

    The closed form (Zhang) starts a non-linear refinement on the planar
    points ``obj`` (metres) / ``img`` (pixels) when given (>= 4): the closed
    form alone is exact for perfect data but sensitive to label noise.
    Returns ``{"P", "focal_px", "C"}`` or None when the focal or the camera
    position is not physical (camera below the pitch, focal out of range).
    """
    s = _settings(s)
    cx, cy = width / 2.0, height / 2.0
    T = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    G = T @ np.asarray(H, dtype=float)
    G = G / np.linalg.norm(G)
    h1, h2, h3 = G[:, 0], G[:, 1], G[:, 2]
    # Zhang with K = diag(f, f, 1): unknown x = 1/f^2, two linear equations.
    a = np.array([h1[0] * h2[0] + h1[1] * h2[1], h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2])
    b = -np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
    den = float(a @ a)
    if not den > 0:
        return None
    x = float(a @ b) / den
    if not x > 0:
        return None
    f = 1.0 / math.sqrt(x)
    if not s["min_focal_ratio"] * width <= f <= s["max_focal_ratio"] * width:
        return None
    Kinv = np.diag([1.0 / f, 1.0 / f, 1.0])
    m1, m2, m3 = Kinv @ h1, Kinv @ h2, Kinv @ h3
    lam = 2.0 / (np.linalg.norm(m1) + np.linalg.norm(m2))
    if (lam * m3)[2] < 0:  # the pitch centre is in front of the camera
        lam = -lam
    r1, r2, t = lam * m1, lam * m2, lam * m3
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        return None
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    if obj is not None and img is not None and len(obj) >= 4:
        R, t, f, K = _refine_pinhole(obj, img, width, height, K, R, t)
    if not s["min_focal_ratio"] * width <= f <= s["max_focal_ratio"] * width:
        return None
    C = -R.T @ t
    if not s["min_cam_height_m"] <= C[2] <= s["max_cam_height_m"]:
        return None
    P = K @ np.c_[R, t]
    return {"P": P / P[2, 3], "focal_px": f, "C": C}


def _refine_pinhole(obj, img, width, height, K, R, t):
    """One-view calibration: focal + pose, principal point / aspect fixed, no distortion."""
    import cv2

    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K1
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
    )
    try:
        _, K2, _, rvecs, tvecs = cv2.calibrateCamera(
            [np.asarray(obj, dtype=np.float32)],
            [np.asarray(img, dtype=np.float32)],
            (int(width), int(height)),
            K.copy(),
            np.zeros(5),
            flags=flags,
        )
    except cv2.error:
        return R, t, float(K[0, 0]), K
    R2, _ = cv2.Rodrigues(rvecs[0])
    return R2, tvecs[0].ravel(), float(K2[0, 0]), K2


def _fit_dlt3d(xy, xyz, use, offplane, width: float, s: dict) -> dict | None:
    """Pruned DLT3D on ``use`` points (needs points off the ground), or None."""
    use = use.copy()
    scale = diag.REF_WIDTH / max(1.0, width)
    res = np.zeros(0)
    L = None
    for _ in range(4):
        if use.sum() < s["min_dlt_points"] or (use & offplane).sum() < s["min_offplane"]:
            return None
        L = calculate_dlt3d_params(xy[use], xyz[use])
        uv, front = _project(dlt_projection_matrix(L), xyz[use])
        res = np.linalg.norm(uv - xy[use], axis=1) * scale
        if not front.all():
            return None
        worst = int(np.argmax(res))
        if res[worst] <= s["fit_px"]:
            break
        use[np.flatnonzero(use)[worst]] = False
    else:
        return None
    rmse = float(np.sqrt(np.mean(res**2)))
    if L is None or rmse > s["max_rmse_px"]:
        return None
    try:
        cam = decompose_dlt3d(L)
    except (ValueError, np.linalg.LinAlgError):
        return None
    fx, fy = float(cam.K[0, 0]), float(cam.K[1, 1])
    if not (fx > 0 and fy > 0) or abs(fy / fx - 1.0) > s["max_aspect_dev"]:
        return None
    if abs(float(cam.K[0, 1])) / ((fx + fy) / 2.0) > s["max_skew"]:
        return None
    f = (fx + fy) / 2.0
    if not s["min_focal_ratio"] * width <= f <= s["max_focal_ratio"] * width:
        return None
    if not s["min_cam_height_m"] <= float(cam.C[2]) <= s["max_cam_height_m"]:
        return None
    return {"P": dlt_projection_matrix(L), "focal_px": f, "C": cam.C, "rmse_px": rmse}


def fit_field_camera(xy, ok, width: float, height: float, settings: dict | None = None) -> dict:
    """Camera of one frame from its trusted keypoints ``xy[ok]`` (pixels).

    ``source``: dlt3d (points off the ground used) | homography (pinhole
    camera recovered from the planar homography) | planar (homography only:
    z != 0 points cannot be projected) | none. ``status`` is ok or the reason.
    """
    s = _settings(settings)
    xyz, planar = _field()
    xy = np.asarray(xy, dtype=float).reshape(NKP, 2)
    ok = np.asarray(ok, dtype=bool) & np.isfinite(xy).all(axis=1)
    out: dict = {
        "status": "few_points",
        "source": "none",
        "P": None,
        "H": None,
        "h_sign": 1.0,
        "focal_px": None,
        "cam_height_m": None,
        "rmse_px": None,
        "n_points": int(ok.sum()),
    }
    use = ok & planar
    fit = diag.fit_field_homography(
        xy[use], xyz[use, :2], width=width, ransac_px=s["fit_px"], max_rmse_px=s["max_rmse_px"]
    )
    if fit["status"] != "ok":
        out["status"] = fit["status"]
        return out
    H = fit["H"]
    inliers = np.zeros(NKP, dtype=bool)
    inliers[np.flatnonzero(use)[fit["inliers"]]] = True
    w = (np.c_[xyz[inliers, :2], np.ones(int(inliers.sum()))] @ H.T)[:, 2]
    out.update(status="ok", source="planar", H=H, h_sign=float(np.sign(np.median(w)) or 1.0))
    out["rmse_px"] = fit["rmse_px"]
    offplane = ok & ~planar
    cam = None
    if offplane.sum() >= s["min_offplane"]:
        cam = _fit_dlt3d(xy, np.asarray(xyz), inliers | offplane, offplane, width, s)
        if cam is not None:
            out["source"] = "dlt3d"
    if cam is None:
        cam = camera_from_homography(H, width, height, s, xyz[inliers], xy[inliers])
        if cam is not None:
            uv, _ = _project(cam["P"], xyz[inliers])
            scale = diag.REF_WIDTH / max(1.0, width)
            rmse = float(np.sqrt(np.mean(np.sum((uv - xy[inliers]) ** 2, axis=1)))) * scale
            if rmse <= s["max_rmse_px"]:
                out["source"] = "homography"
                cam["rmse_px"] = rmse
            else:
                cam = None
    if cam is not None:
        out.update(
            P=cam["P"],
            focal_px=round(float(cam["focal_px"]), 1),
            cam_height_m=round(float(cam["C"][2]), 2),
            rmse_px=round(float(cam["rmse_px"]), 3),
        )
    return out


def project_field(model: dict, width: float, height: float, margin: float = 0.01):
    """``(xy (49, 2), inside (49,))``: every keypoint through the frame's camera.

    ``inside``: projected in front of the camera and inside the image (minus
    ``margin`` of the width). Without a full camera only z = 0 points project.
    """
    xyz, planar = _field()
    xy = np.full((NKP, 2), np.nan)
    front = np.zeros(NKP, dtype=bool)
    if model.get("P") is not None:
        xy, front = _project(model["P"], xyz)
    elif model.get("H") is not None:
        ph = np.c_[xyz[:, :2], np.ones(NKP)] @ np.asarray(model["H"]).T
        with np.errstate(invalid="ignore", divide="ignore"):
            xy = np.where(planar[:, None], ph[:, :2] / ph[:, 2:3], np.nan)
        front = planar & (np.sign(ph[:, 2]) == model["h_sign"])
    m = margin * width
    with np.errstate(invalid="ignore"):
        inside = (
            front
            & np.isfinite(xy).all(axis=1)
            & (xy[:, 0] >= m)
            & (xy[:, 0] <= width - m)
            & (xy[:, 1] >= m)
            & (xy[:, 1] <= height - m)
        )
    return xy, inside


def dlt_coefficients(model: dict | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """vailá DLT2D and DLT3D coefficients of a frame camera (``rec2d`` / ``rec3d`` layout).

    DLT2D (8): ground plane (X, Y, Z = 0, metres) -> image homography with
    H[2,2] = 1, ``[H00 H01 H02 H10 H11 H12 H20 H21]`` (same as fifa_to_dlt).
    DLT3D (11): 3x4 camera with P[2,3] = 1, ``L1..L11`` row by row. Only the
    camera sources that model z != 0 (dlt3d, homography) give a DLT3D. World
    frame: origin at the centre spot, x towards the right goal, y towards the
    "top" touchline, z up (soccerfield_kiki.csv).
    """
    if not model or model.get("status") != "ok":
        return None, None
    d2 = d3 = None
    H = model.get("H")
    if H is not None and abs(float(H[2, 2])) > 1e-12:
        d2 = (np.asarray(H, dtype=float) / float(H[2, 2])).ravel()[:8]
    P = model.get("P")
    if P is not None and abs(float(P[2, 3])) > 1e-12:
        d3 = (np.asarray(P, dtype=float) / float(P[2, 3])).ravel()[:11]
    return d2, d3


def refine_keypoints(
    raw_xy, raw_kc, codes, width: float, height: float, *, mode: str, kp_conf: float, settings=None
):
    """Fill / fix one frame's network keypoints with the field camera.

    ``codes`` are detect's per-point codes (``D`` accepted). Returns
    ``(xy, kc, gcodes, model)``; ``gcodes`` adds G / Gx / Dg (``GEOM_CODES``).
    Nothing changes without a valid camera, or with ``mode='off'``.
    """
    gcodes = list(codes)
    if mode not in GEOM_MODES:
        raise ValueError(f"geometry mode must be one of {GEOM_MODES}")
    if raw_xy is None or raw_kc is None or mode == "off":
        return raw_xy, raw_kc, gcodes, None
    s = _settings(settings)
    xy = np.array(raw_xy, dtype=float).reshape(NKP, 2)
    kc = np.array(raw_kc, dtype=float).reshape(NKP)
    ok = np.array([c == "D" for c in codes])
    model = fit_field_camera(xy, ok, width, height, s)
    if model["status"] != "ok":
        return xy, kc, gcodes, model
    proj, inside = project_field(model, width, height, s["margin"])
    scale = diag.REF_WIDTH / max(1.0, width)
    with np.errstate(invalid="ignore"):
        dist = np.linalg.norm(xy - proj, axis=1) * scale
    for i in range(NKP):
        if codes[i] == "D":
            if np.isfinite(dist[i]) and dist[i] > s["fix_px"]:
                if mode == "fix" and inside[i]:
                    xy[i] = proj[i]
                    gcodes[i] = "Gx"
                else:
                    gcodes[i] = "Dg"
        elif codes[i] in FILLABLE and inside[i] and np.nan_to_num(kc[i], nan=-1.0) >= s["min_conf"]:
            if not dist[i] <= s["agree_px"]:
                xy[i] = proj[i]
            kc[i] = max(float(kc[i]), kp_conf)
            gcodes[i] = "G"
    return xy, kc, gcodes, model


def refine_predictions(
    pred: dict, *, det_conf: float, kp_conf: float, mode: str, settings=None
) -> tuple[dict, dict]:
    """``refine_keypoints`` on every image of an ``evaluate`` prediction set.

    Returns ``(pred with refined pred_xy / pred_kc, stats)``; images without
    a box above ``det_conf`` are left as they are.
    """
    xy_all = np.array(pred["pred_xy"], dtype=float)
    kc_all = np.array(pred["pred_kc"], dtype=float)
    box = np.nan_to_num(np.asarray(pred["box_conf"], dtype=float), nan=-1.0)
    sources: Counter = Counter()
    fills: Counter = Counter()
    fixes: Counter = Counter()
    for n in range(len(xy_all)):
        if box[n] < det_conf:
            continue
        w, h = float(pred["width"][n]), float(pred["height"][n])
        xy, kc = xy_all[n], kc_all[n]
        codes = [
            "D"
            if np.nan_to_num(kc[i], nan=-1.0) >= kp_conf and 0 <= xy[i, 0] < w and 0 <= xy[i, 1] < h
            else "Rk"
            for i in range(NKP)
        ]
        new_xy, new_kc, gcodes, model = refine_keypoints(
            xy, kc, codes, w, h, mode=mode, kp_conf=kp_conf, settings=settings
        )
        sources[model["source"] if model else "none"] += 1
        fills.update(f"p{i}" for i, c in enumerate(gcodes) if c == "G")
        fixes.update(f"p{i}" for i, c in enumerate(gcodes) if c == "Gx")
        xy_all[n], kc_all[n] = new_xy, new_kc
    n_box = int((box >= det_conf).sum())
    stats = {
        "mode": mode,
        "images_with_box": n_box,
        "camera_sources": dict(sorted(sources.items())),
        "camera_ok_rate": round(
            sum(v for k, v in sources.items() if k != "none") / max(1, n_box), 4
        ),
        "fills": dict(fills.most_common()),
        "fixes": dict(fixes.most_common()),
    }
    return pred | {"pred_xy": xy_all, "pred_kc": kc_all}, stats


def nonplanar_label_outliers(px, vis, width: float, height: float, *, tol_px: float = 15.0):
    """Labelled flag / post-top points far from the camera of the other labels.

    Leave-one-out: for each labelled z != 0 point, the camera is fitted
    without it. Returns ``[(index, residual_px@1920)]`` above ``tol_px``.
    """
    xyz, planar = _field()
    px = np.asarray(px, dtype=float).reshape(NKP, 2)
    vis = np.asarray(vis, dtype=bool)
    scale = diag.REF_WIDTH / max(1.0, width)
    out = []
    for i in np.flatnonzero(vis & ~planar):
        keep = vis.copy()
        keep[i] = False
        model = fit_field_camera(px, keep, width, height)
        if model["status"] != "ok" or model.get("P") is None:
            continue
        uv, front = _project(model["P"], xyz[i : i + 1])
        if front[0]:
            err = float(np.linalg.norm(uv[0] - px[i])) * scale
            if err > tol_px:
                out.append((int(i), round(err, 1)))
    return out
