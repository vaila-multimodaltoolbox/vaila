"""
================================================================================
Script: freekiki_diag.py - FreeKiki metrics, geometry and dataset diagnostics
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.7
Created: 27 September 2026
Update Date: 05 October 2026

Description:
    Verifiable measurement helpers for ``freekiki.py`` (49-point soccer-field
    keypoints). Pure numpy / OpenCV, no Ultralytics import.

    * Keypoint matching with an explicit distance gate. Every labelled or
      predicted keypoint of every image gets one status:
        TP  labelled, predicted (conf >= kp_conf) and within ``match_px``;
        MIS labelled and predicted but farther than ``match_px``
            (counted as one false positive AND one false negative);
        FN  labelled, not predicted;
        FP  predicted, not labelled;
        TN  neither.
      All distances are in px@1920 (pixels rescaled to a 1920-px-wide image).
      A missing prediction is never an error of zero: its distance stays NaN.
    * Per-keypoint tables: precision = TP / (TP+MIS+FP), recall =
      TP / (TP+MIS+FN), error mean/median/P90/P95 on TP, PCK@t with two
      denominators (``pck<t>_pred``: labelled and predicted points;
      ``pck<t>_all``: every labelled point, misses count as failures).
    * Identity-swap diagnosis: a labelled point that was missed or
      mislocalized but lies within ``match_px`` of ANOTHER predicted index is
      tagged mirror (flip_idx partner, x -> -x), rot180 ((x, y) -> (-x, -y))
      or other.
    * Field homography (world metres -> image) from planar (z = 0) keypoints
      only, with RANSAC, inlier count, RMSE in px@1920 and metres, spread
      (degeneracy) and orientation checks. A camera above the pitch always
      gives a negative Jacobian determinant for the schema axes; a positive
      one means the left/right (or near/far) identities are mirrored.
    * Temporal metrics: raw displacement between frames and the residual after
      removing camera motion with an inter-frame homography; shots are split
      at detected cuts.
    * Dataset audit (read-only): split/source/group counts, match-level
      leakage, label validity, flip_idx consistency, flag/post above its base,
      per-image label homography, near-duplicate images across splits (dHash).
    * Repeat-factor oversampling (LVIS) of rare keypoints into a versioned
      train list. The dataset itself is not modified.
    * Oversampling manifest: repeat-factor sampling (LVIS) with keypoints as
      categories writes a repeated train image list plus counts and hashes;
      the dataset itself is never modified.
    * Candidate-vs-baseline promotion gate on the same evaluation split.
    * Label completeness: the field homography of a frame's labelled points
      lists the points projected inside the image but neither labelled nor
      hidden (a "not visible" label on a visible point teaches the network to
      ignore it). Used by review, export, ingest and the audit.

License:
    GNU Affero General Public License v3.0 (AGPLv3).
================================================================================
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_CSV = Path(__file__).resolve().parent / "models" / "soccerfield_kiki.csv"
NKP = 49
REF_WIDTH = 1920.0
UNITS = "px@1920"

TN, TP, MIS, FN, FP = 0, 1, 2, 3, 4
STATUS_NAMES = {TN: "tn", TP: "tp", MIS: "mislocalized", FN: "fn", FP: "fp"}
SWAP_NONE, SWAP_MIRROR, SWAP_ROT180, SWAP_OTHER = 0, 1, 2, 3
SWAP_NAMES = {SWAP_NONE: "", SWAP_MIRROR: "mirror", SWAP_ROT180: "rot180", SWAP_OTHER: "other"}

# Camera above the pitch, schema axes (x towards the right goal, y towards the
# "top" touchline): image u grows with x and v shrinks with y for the main
# camera, and both flip for a reverse-angle camera, so det(J) < 0 always.
EXPECTED_ORIENTATION = -1

# (elevated point, its ground point): flag tops over corners, post tops over bases.
ABOVE_PAIRS = ((38, 0), (39, 5), (46, 24), (47, 29), (34, 32), (35, 33), (42, 40), (43, 41))

HOMOGRAPHY_DEFAULTS = {
    "ransac_px": 8.0,  # RANSAC reprojection threshold, px@1920
    "min_inliers": 6,  # 4 points always fit exactly; 6 leave redundancy to check
    "max_rmse_px": 6.0,  # inlier RMSE, px@1920
    "min_minor_std_m": 1.0,  # spread of inliers across their weakest direction
    "min_axis_ratio": 0.05,  # minor/major principal spread (near-collinear below)
}

# A homography can pass every self-consistency check above and still be the
# wrong one (a model that outputs the mean field layout does). With labels, it
# counts as correct only when the labelled points reproject within this.
CALIB_CORRECT_PX = 10.0  # px@1920

# Bump when eval_summary.json gains a metric the promotion gate reads, so
# older reports are recomputed instead of silently skipping that check.
EVAL_SCHEMA = 2

PROMOTION_DEFAULTS: dict[str, Any] = {
    "mode": "gate",  # gate | map | never
    "split": "val",
    "pck_px": 10,
    "max_map_drop": 0.0,
    "max_pck_all_drop": 0.0,
    "critical_kps": [0, 5, 13, 16, 24, 29, 32, 33, 40, 41, 48],
    "min_support": 30,
    "max_recall_drop": 0.02,
    "max_median_err_increase_px": 1.0,
    "max_calib_correct_drop": 0.01,
}


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def load_field_points(csv_path: Path = SCHEMA_CSV) -> tuple[list[str], np.ndarray, np.ndarray]:
    """``(names, flip_idx (49,), xyz (49, 3) metres)`` in point_number order."""
    with Path(csv_path).open(encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["point_number"]))
    names = [r["point_name"] for r in rows]
    flip = np.array([int(r["flip_idx"]) for r in rows])
    xyz = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
    return names, flip, xyz


def symmetry_maps(xyz: np.ndarray, tol: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Mirror (x -> -x) and 180-degree rotation ((x, y) -> (-x, -y)) partners.

    Returns -1 where no partner exists.
    """
    mirror = np.full(len(xyz), -1)
    rot = np.full(len(xyz), -1)
    for i, (x, y, z) in enumerate(xyz):
        for target, out in (((-x, y, z), mirror), ((-x, -y, z), rot)):
            d = np.linalg.norm(xyz - np.array(target), axis=1)
            j = int(np.argmin(d))
            if d[j] < tol:
                out[i] = j
    return mirror, rot


# --------------------------------------------------------------------------- #
# Matching + per-keypoint metrics
# --------------------------------------------------------------------------- #
def match_keypoints(pred_xy, pred_conf, gt_xy, gt_vis, widths, kp_conf: float, match_px: float):
    """Status + distance of every keypoint of N images.

    ``pred_xy`` (N, 49, 2) and ``gt_xy`` (N, 49, 2) in image pixels;
    ``pred_conf`` (N, 49), NaN when the image has no accepted detection;
    ``gt_vis`` (N, 49), > 0 when labelled; ``widths`` (N,) image widths.
    Returns ``status`` (N, 49) int codes and ``dist`` (N, 49) px@1920, NaN
    unless the point is both labelled and predicted.
    """
    pred_xy = np.asarray(pred_xy, dtype=float).reshape(-1, NKP, 2)
    gt_xy = np.asarray(gt_xy, dtype=float).reshape(-1, NKP, 2)
    conf = np.asarray(pred_conf, dtype=float).reshape(-1, NKP)
    labelled = np.asarray(gt_vis).reshape(-1, NKP) > 0
    predicted = np.nan_to_num(conf, nan=-1.0) >= kp_conf
    scale = REF_WIDTH / np.maximum(1.0, np.asarray(widths, dtype=float).reshape(-1, 1))
    dist = np.linalg.norm(pred_xy - gt_xy, axis=2) * scale
    dist = np.where(labelled & predicted, dist, np.nan)
    status = np.full(labelled.shape, TN)
    status[labelled & ~predicted] = FN
    status[~labelled & predicted] = FP
    both = labelled & predicted
    status[both & (dist <= match_px)] = TP
    status[both & ~(dist <= match_px)] = MIS
    return status, dist


def identity_swaps(pred_xy, predicted, gt_xy, status, widths, match_px, mirror, rot):
    """For missed/mislocalized labelled points, which OTHER predicted index sits on them.

    Returns (N, 49) codes: SWAP_MIRROR, SWAP_ROT180, SWAP_OTHER or SWAP_NONE,
    and (N, 49) the index found (-1 when none).
    """
    pred_xy = np.asarray(pred_xy, dtype=float).reshape(-1, NKP, 2)
    gt_xy = np.asarray(gt_xy, dtype=float).reshape(-1, NKP, 2)
    predicted = np.asarray(predicted, dtype=bool).reshape(-1, NKP)
    scale = REF_WIDTH / np.maximum(1.0, np.asarray(widths, dtype=float))
    codes = np.full(predicted.shape, SWAP_NONE)
    found = np.full(predicted.shape, -1)
    todo = np.isin(status, (MIS, FN))
    eye = np.eye(NKP, dtype=bool)
    for n in np.flatnonzero(todo.any(axis=1)):
        d = np.linalg.norm(gt_xy[n][:, None, :] - pred_xy[n][None, :, :], axis=2) * scale[n]
        d[:, ~predicted[n]] = np.inf
        d[eye] = np.inf
        j = np.argmin(d, axis=1)
        hit = todo[n] & (d[np.arange(NKP), j] <= match_px)
        for i in np.flatnonzero(hit):
            found[n, i] = j[i]
            codes[n, i] = (
                SWAP_MIRROR if j[i] == mirror[i] else SWAP_ROT180 if j[i] == rot[i] else SWAP_OTHER
            )
    return codes, found


def _num(value, digits: int = 4):
    """Round a finite number; None stays None (blank in CSV)."""
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def _ratio(num: int, den: int):
    return round(num / den, 4) if den else None


def _stats_block(status, dist, pck: tuple, groups=None, swaps=None) -> dict:
    """Metrics for one flattened set of (image, keypoint) cells."""
    status = np.asarray(status).ravel()
    dist = np.asarray(dist, dtype=float).ravel()
    n_tp = int((status == TP).sum())
    n_mis = int((status == MIS).sum())
    n_fn = int((status == FN).sum())
    n_fp = int((status == FP).sum())
    n_ref = n_tp + n_mis + n_fn
    n_pred = n_tp + n_mis + n_fp
    e_tp = dist[status == TP]
    e_any = dist[(status == TP) | (status == MIS)]
    out = {
        "units": UNITS,
        "n_ref": n_ref,
        "n_pred": n_pred,
        "n_tp": n_tp,
        "n_mislocalized": n_mis,
        "n_fn": n_fn,
        "n_fp": n_fp,
        "precision": _ratio(n_tp, n_pred),
        "recall": _ratio(n_tp, n_ref),
        "err_mean": _num(e_tp.mean(), 2) if e_tp.size else None,
        "err_median": _num(np.median(e_tp), 2) if e_tp.size else None,
        "err_p90": _num(np.percentile(e_tp, 90), 2) if e_tp.size else None,
        "err_p95": _num(np.percentile(e_tp, 95), 2) if e_tp.size else None,
        "err_any_median": _num(np.median(e_any), 2) if e_any.size else None,
        "err_any_p90": _num(np.percentile(e_any, 90), 2) if e_any.size else None,
    }
    for t in pck:
        ok = int((e_any <= t).sum())
        out[f"pck{t:g}_pred"] = _ratio(ok, e_any.size)
        out[f"pck{t:g}_all"] = _ratio(ok, n_ref)
    if groups is not None:
        ref = np.isin(status, (TP, MIS, FN))
        out["n_groups_ref"] = len(
            {g for g, r in zip(np.asarray(groups).ravel(), ref, strict=True) if r}
        )
    if swaps is not None:
        sw = np.asarray(swaps).ravel()
        out["swap_mirror"] = int((sw == SWAP_MIRROR).sum())
        out["swap_rot180"] = int((sw == SWAP_ROT180).sum())
        out["swap_other"] = int((sw == SWAP_OTHER).sum())
    return out


def keypoint_table(status, dist, names, *, pck=(5, 10, 25), groups=None, swaps=None):
    """``(overall, per-keypoint rows)`` from :func:`match_keypoints` output.

    ``groups`` (N,) recording/match id of each image adds ``n_groups_ref``;
    ``swaps`` (N, 49) from :func:`identity_swaps` adds swap counts.
    """
    status = np.asarray(status).reshape(-1, NKP)
    dist = np.asarray(dist, dtype=float).reshape(-1, NKP)
    g2 = None if groups is None else np.repeat(np.asarray(groups)[:, None], NKP, axis=1)
    rows = []
    for i in range(NKP):
        rows.append(
            {"kp": f"p{i}", "name": names[i]}
            | _stats_block(
                status[:, i],
                dist[:, i],
                pck,
                None if g2 is None else g2[:, i],
                None if swaps is None else np.asarray(swaps)[:, i],
            )
        )
    overall = {"images": int(status.shape[0])} | _stats_block(status, dist, pck, g2, swaps)
    return overall, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def _orientation(H: np.ndarray, world_xy: np.ndarray) -> int:
    """Sign of the Jacobian determinant of world -> image at the points' centroid."""
    if len(world_xy) == 0:
        return 0
    c = world_xy.mean(axis=0)
    w = H[2, 0] * c[0] + H[2, 1] * c[1] + H[2, 2]
    sign = np.sign(np.linalg.det(H) * w)
    return int(sign) if np.isfinite(sign) else 0


def _project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ph = np.c_[pts, np.ones(len(pts))] @ H.T
    return ph[:, :2] / ph[:, 2:3]


def _spread(world_xy: np.ndarray) -> np.ndarray:
    """Principal standard deviations (minor, major) of field points, metres."""
    if len(world_xy) < 3:
        return np.zeros(2)
    ev = np.linalg.eigvalsh(np.cov(np.asarray(world_xy, dtype=float).T))
    return np.sqrt(np.clip(ev, 0.0, None))


def _is_degenerate(world_xy: np.ndarray, lim: dict) -> bool:
    """Near-collinear points: too little spread across the weakest direction."""
    minor, major = _spread(world_xy)
    return minor < float(lim["min_minor_std_m"]) or minor < float(lim["min_axis_ratio"]) * major


def fit_field_homography(
    img_xy,
    world_xy,
    *,
    width: float = REF_WIDTH,
    expected_orientation: int | None = EXPECTED_ORIENTATION,
    **limits,
) -> dict:
    """Robust world (metres, z = 0 only) -> image homography with explicit checks.

    ``status``: ok | few_points | degenerate | ransac_fail | few_inliers |
    high_error | mirrored. Errors are px@1920 (image) and metres (field).
    """
    import cv2

    lim = HOMOGRAPHY_DEFAULTS | limits
    img_xy = np.asarray(img_xy, dtype=float).reshape(-1, 2)
    world_xy = np.asarray(world_xy, dtype=float).reshape(-1, 2)
    scale = REF_WIDTH / max(1.0, float(width))
    out: dict = {
        "status": "few_points",
        "n_points": len(img_xy),
        "n_inliers": 0,
        "rmse_px": None,
        "rmse_m": None,
        "minor_std_m": None,
        "orientation": 0,
        "H": None,
        "inliers": np.zeros(len(img_xy), dtype=bool),
    }
    if len(img_xy) < 4:
        return out
    if _is_degenerate(world_xy, lim):
        # e.g. only goal-line points (all x = -52.45): collinear, no homography
        out["status"] = "degenerate"
        out["minor_std_m"] = _num(_spread(world_xy)[0], 3)
        return out
    H, mask = cv2.findHomography(
        world_xy.astype(np.float32),
        img_xy.astype(np.float32),
        cv2.RANSAC,
        float(lim["ransac_px"]) / scale,
    )
    if H is None or not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
        out["status"] = "ransac_fail"
        return out
    inl = mask.ravel().astype(bool) if mask is not None else np.zeros(len(img_xy), dtype=bool)
    wi, ii = world_xy[inl], img_xy[inl]
    out.update({"H": H, "inliers": inl, "n_inliers": int(inl.sum())})
    if inl.sum() < 4:  # RANSAC can return an H that no point supports
        out["status"] = "few_inliers"
        return out
    ev = _spread(wi)
    out["minor_std_m"] = _num(ev[0], 3)
    out["orientation"] = _orientation(H, wi)
    err_px = np.linalg.norm(_project(H, wi) - ii, axis=1) * scale
    out["rmse_px"] = _num(np.sqrt(np.mean(err_px**2)), 3)
    try:
        err_m = np.linalg.norm(_project(np.linalg.inv(H), ii) - wi, axis=1)
        out["rmse_m"] = _num(np.sqrt(np.mean(err_m**2)), 3)
    except np.linalg.LinAlgError:
        pass
    if out["n_inliers"] < int(lim["min_inliers"]):
        out["status"] = "few_inliers"
    elif _is_degenerate(wi, lim):
        out["status"] = "degenerate"
    elif out["rmse_px"] is None or out["rmse_px"] > float(lim["max_rmse_px"]):
        out["status"] = "high_error"
    elif expected_orientation and out["orientation"] != expected_orientation:
        out["status"] = "mirrored"
    else:
        out["status"] = "ok"
    return out


def frame_homography(xy, conf, kp_conf, planar, world_xy, width, **limits) -> dict:
    """:func:`fit_field_homography` on one frame's accepted planar keypoints."""
    if xy is None or conf is None:
        return fit_field_homography(np.zeros((0, 2)), np.zeros((0, 2)), width=width)
    use = planar & (np.nan_to_num(np.asarray(conf, dtype=float), nan=-1.0) >= kp_conf)
    return fit_field_homography(np.asarray(xy)[use], world_xy[use], width=width, **limits)


def calibration_error(H, gt_xy, gt_vis, planar, world_xy, width) -> float | None:
    """Median px@1920 distance between labelled planar points and their H projection."""
    if H is None:
        return None
    use = planar & (np.asarray(gt_vis) > 0)
    if not use.any():
        return None
    proj = _project(H, world_xy[use])
    err = np.linalg.norm(proj - np.asarray(gt_xy)[use], axis=1) * (REF_WIDTH / max(1.0, width))
    return float(np.median(err))


# --------------------------------------------------------------------------- #
# Offline scoring of saved predictions (evaluate + threshold sweep)
# --------------------------------------------------------------------------- #
# Human labels are exact up to click error and lens distortion, so the fit that
# checks a labelled frame is looser than the one that checks a prediction.
COMPLETENESS_LIMITS: dict[str, Any] = {"ransac_px": 15.0, "min_inliers": 4, "max_rmse_px": 15.0}
COMPLETENESS_MARGIN = 0.02  # of the image width, kept from every border


def label_completeness(points, hidden, width: float, height: float) -> dict:
    """Points a human label probably misses in one frame.

    A point left empty is written ``0 0 0`` ("not visible") and the pose loss
    then pushes its confidence to 0, so a partial frame teaches the network to
    ignore visible points. The labelled planar points give a field homography;
    a point is a *suspect* when its projection (the ground base for flag and
    post tops, see ``ABOVE_PAIRS``) lies inside the image while it is neither
    labelled nor in ``hidden`` (confirmed occluded / not there).

    ``status``: complete | incomplete, or the homography status (few_points,
    degenerate, few_inliers, high_error, mirrored, ...) when the frame cannot
    be checked. ``projected`` maps unlabelled planar indices inside the image
    to their projected xy (pixels).
    """
    names, _, xyz = load_field_points()
    planar = xyz[:, 2] == 0
    pts = np.full((NKP, 2), np.nan)
    for i, p in enumerate(points):
        if p is not None:
            pts[i] = p
    labelled = np.isfinite(pts[:, 0])
    use = labelled & planar
    fit = fit_field_homography(pts[use], xyz[use, :2], width=width, **COMPLETENESS_LIMITS)
    out: dict = {
        "status": fit["status"],
        "n_planar": int(use.sum()),
        "rmse_px": fit["rmse_px"],
        "suspects": [],
        "projected": {},
    }
    if fit["status"] != "ok":
        return out
    ground = xyz[:, :2].copy()
    for top, base in ABOVE_PAIRS:
        ground[top] = xyz[base, :2]
    ph = np.c_[ground, np.ones(NKP)] @ fit["H"].T
    proj = ph[:, :2] / ph[:, 2:3]
    # Points behind the camera project with the opposite sign of w.
    front = np.sign(ph[:, 2]) == np.sign(np.median(ph[use, 2]))
    m = COMPLETENESS_MARGIN * float(width)
    inside = (
        front
        & (proj[:, 0] >= m)
        & (proj[:, 0] <= width - m)
        & (proj[:, 1] >= m)
        & (proj[:, 1] <= height - m)
    )
    hidden = {int(i) for i in hidden or ()}
    missing = inside & ~labelled
    out["suspects"] = [int(i) for i in np.flatnonzero(missing) if int(i) not in hidden]
    out["projected"] = {
        int(i): [float(proj[i, 0]), float(proj[i, 1])] for i in np.flatnonzero(missing & planar)
    }
    out["status"] = "incomplete" if out["suspects"] else "complete"
    return out


def accepted_confidence(pred: dict, det_conf: float) -> np.ndarray:
    """Keypoint confidences with images whose best box is below ``det_conf`` set to NaN."""
    kc = np.asarray(pred["pred_kc"], dtype=float).copy()
    kc[~(np.nan_to_num(np.asarray(pred["box_conf"], dtype=float), nan=-1.0) >= det_conf)] = np.nan
    return kc


def calibration_table(pred: dict, kc: np.ndarray, kp_conf: float, **limits) -> dict:
    """Predicted-keypoint homography quality on images whose labels calibrate.

    Denominator: images where the LABELLED planar points give status ``ok``
    (same checks). ``ok_rate`` = share of those where the PREDICTED points
    also give ``ok`` (self-consistent only); ``correct_rate`` = share where
    the predicted homography is ``ok`` AND reprojects the labelled planar
    points within ``CALIB_CORRECT_PX`` (median); ``err_*`` = that median
    px@1920 distance over the ``ok`` images.
    """
    names, _, xyz = load_field_points()
    planar, world = xyz[:, 2] == 0, xyz[:, :2]
    statuses: Counter = Counter()
    errs, per_image = [], []
    n_gt_ok = n_ok = 0
    for n in range(len(kc)):
        w = float(pred["width"][n])
        gt_vis = np.asarray(pred["gt_vis"][n])
        use = planar & (gt_vis > 0)
        gt_fit = fit_field_homography(pred["gt_xy"][n][use], world[use], width=w, **limits)
        fit = frame_homography(pred["pred_xy"][n], kc[n], kp_conf, planar, world, w, **limits)
        err = calibration_error(fit["H"], pred["gt_xy"][n], gt_vis, planar, world, w)
        per_image.append((gt_fit["status"], fit["status"], err))
        if gt_fit["status"] != "ok":
            continue
        n_gt_ok += 1
        statuses[fit["status"]] += 1
        if fit["status"] == "ok":
            n_ok += 1
            if err is not None:
                errs.append(err)
    e = np.asarray(errs)
    return {
        "units": UNITS,
        "n_images_gt_calibratable": n_gt_ok,
        "ok_rate": _ratio(n_ok, n_gt_ok),
        "pred_status": dict(statuses),
        "err_median_px": _num(np.median(e), 2) if e.size else None,
        "err_p90_px": _num(np.percentile(e, 90), 2) if e.size else None,
        "correct_px": CALIB_CORRECT_PX,
        "correct_rate": _ratio(int((e <= CALIB_CORRECT_PX).sum()), n_gt_ok),
        "per_image": per_image,
    }


def score_predictions(
    pred: dict,
    *,
    det_conf: float,
    kp_conf: float,
    match_px: float,
    pck=(5, 10, 25),
    with_calib: bool = True,
) -> dict:
    """All FreeKiki keypoint tables from saved raw predictions (see ``evaluate``).

    ``pred`` holds arrays: images, groups, sources, width, box_conf (NaN = no
    box), pred_xy, pred_kc, gt_xy (pixels), gt_vis. Returns overall,
    per_keypoint, per_source, failures, calib.
    """
    names, _, xyz = load_field_points()
    mirror, rot = symmetry_maps(xyz)
    kc = accepted_confidence(pred, det_conf)
    status, dist = match_keypoints(
        pred["pred_xy"], kc, pred["gt_xy"], pred["gt_vis"], pred["width"], kp_conf, match_px
    )
    predicted = np.nan_to_num(kc, nan=-1.0) >= kp_conf
    swaps, found = identity_swaps(
        pred["pred_xy"], predicted, pred["gt_xy"], status, pred["width"], match_px, mirror, rot
    )
    groups = np.asarray(pred["groups"])
    overall, per_kp = keypoint_table(status, dist, names, pck=pck, groups=groups, swaps=swaps)
    has_box = np.nan_to_num(np.asarray(pred["box_conf"], dtype=float), nan=-1.0) >= det_conf
    overall = {"det_conf": det_conf, "kp_conf": kp_conf, "match_px": match_px} | overall
    overall["images_with_detection"] = int(has_box.sum())
    per_source = []
    sources = np.asarray(pred["sources"])
    for src in sorted(set(sources.tolist())):
        m = sources == src
        block = _stats_block(status[m], dist[m], pck, np.repeat(groups[m][:, None], NKP, axis=1))
        per_source.append({"source": src, "images": int(m.sum())} | block)
    failures = []
    scale = REF_WIDTH / np.maximum(1.0, np.asarray(pred["width"], dtype=float))
    for n, i in zip(*np.nonzero(np.isin(status, (MIS, FN, FP))), strict=True):
        failures.append(
            {
                "image": pred["images"][n],
                "source": sources[n],
                "group": groups[n],
                "kp": f"p{i}",
                "name": names[i],
                "status": STATUS_NAMES[int(status[n, i])],
                "kp_conf": _num(kc[n, i], 3),
                "box_conf": _num(float(pred["box_conf"][n]), 3),
                "err_px": _num(dist[n, i], 1),
                "gt_x": _num(pred["gt_xy"][n, i, 0], 1) if status[n, i] != FP else None,
                "gt_y": _num(pred["gt_xy"][n, i, 1], 1) if status[n, i] != FP else None,
                "pred_x": _num(pred["pred_xy"][n, i, 0], 1),
                "pred_y": _num(pred["pred_xy"][n, i, 1], 1),
                "swap": SWAP_NAMES[int(swaps[n, i])],
                "swap_with": f"p{found[n, i]}" if found[n, i] >= 0 else "",
                "px_scale": _num(scale[n], 4),
            }
        )
    calib = calibration_table(pred, kc, kp_conf) if with_calib else {}
    return {
        "overall": overall,
        "per_keypoint": per_kp,
        "per_source": per_source,
        "failures": failures,
        "calib": calib,
    }


# --------------------------------------------------------------------------- #
# Temporal behaviour
# --------------------------------------------------------------------------- #
def frame_signature(frame) -> np.ndarray:
    """Normalised Hue-Saturation histogram used for cut detection."""
    import cv2

    small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 16], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).astype(np.float32)


def is_cut(prev_sig, sig, threshold: float = 0.4) -> bool:
    """Bhattacharyya distance between consecutive signatures above ``threshold``."""
    import cv2

    if prev_sig is None:
        return False
    return float(cv2.compareHist(prev_sig, sig, cv2.HISTCMP_BHATTACHARYYA)) > threshold


def temporal_metrics(pts, cuts, planar, width: float, *, ransac_px: float = 8.0) -> dict:
    """Displacement vs residual jitter between consecutive processed frames.

    ``pts`` (T, 49, 2) accepted keypoints (NaN when absent); ``cuts`` (T,)
    True where frame t starts a new shot (pairs across a cut are skipped).
    For each pair (t-1, t) and keypoint i visible in both:

      displacement_i = |x_i[t] - x_i[t-1]|
      residual_i     = |x_i[t] - M_t(x_i[t-1])|

    where M_t is the camera motion: an image-to-image homography fitted with
    RANSAC on the common planar keypoints (>= 4), else the median translation
    of the common keypoints (>= 3 points), else the pair is skipped. The
    residual is what remains after camera motion: detector noise plus real
    errors. All values px@1920.
    """
    import cv2

    pts = np.asarray(pts, dtype=float)
    scale = REF_WIDTH / max(1.0, float(width))
    disp, resid = [], []
    methods: Counter = Counter()
    for t in range(1, len(pts)):
        if cuts[t]:
            methods["cut"] += 1
            continue
        common = np.isfinite(pts[t]).all(axis=1) & np.isfinite(pts[t - 1]).all(axis=1)
        if common.sum() < 3:
            methods["skipped"] += 1
            continue
        a, b = pts[t - 1][common], pts[t][common]
        disp.extend((np.linalg.norm(b - a, axis=1) * scale).tolist())
        cp = common & planar
        moved = None
        if cp.sum() >= 4:
            H, _ = cv2.findHomography(
                pts[t - 1][cp].astype(np.float32),
                pts[t][cp].astype(np.float32),
                cv2.RANSAC,
                ransac_px / scale,
            )
            if H is not None and np.all(np.isfinite(H)):
                moved = _project(H, a)
                methods["homography"] += 1
        if moved is None:
            moved = a + np.median(b - a, axis=0)
            methods["translation"] += 1
        resid.extend((np.linalg.norm(b - moved, axis=1) * scale).tolist())
    d, r = np.asarray(disp), np.asarray(resid)
    return {
        "pairs_homography": methods["homography"],
        "pairs_translation": methods["translation"],
        "pairs_skipped": methods["skipped"],
        "pairs_across_cut": methods["cut"],
        "displacement_median_px": _num(np.median(d), 2) if d.size else None,
        "displacement_p90_px": _num(np.percentile(d, 90), 2) if d.size else None,
        "residual_median_px": _num(np.median(r), 2) if r.size else None,
        "residual_p90_px": _num(np.percentile(r, 90), 2) if r.size else None,
    }


def fill_short_gaps(pts, cuts, max_gap: int):
    """Linear interpolation of gaps of <= ``max_gap`` frames inside one shot.

    Returns ``(filled pts, interpolated mask)``. Gaps touching a shot edge,
    longer gaps and gaps across a cut stay NaN (no extrapolation).
    """
    pts = np.asarray(pts, dtype=float).copy()
    interp = np.zeros(pts.shape[:2], dtype=bool)
    if max_gap <= 0:
        return pts, interp
    starts = [0] + [t for t in range(1, len(pts)) if cuts[t]] + [len(pts)]
    for s, e in zip(starts[:-1], starts[1:], strict=True):
        for i in range(pts.shape[1]):
            ok = np.flatnonzero(np.isfinite(pts[s:e, i, 0])) + s
            for a, b in zip(ok[:-1], ok[1:], strict=True):
                if 1 < b - a <= max_gap + 1:
                    w = (np.arange(a + 1, b) - a) / (b - a)
                    pts[a + 1 : b, i] = pts[a, i] + w[:, None] * (pts[b, i] - pts[a, i])
                    interp[a + 1 : b, i] = True
    return pts, interp


def duplicate_points(xy, accepted, width, min_px: float = 3.0) -> list[tuple[int, int]]:
    """Pairs of different accepted indices predicted on the same pixel (identity clash)."""
    idx = np.flatnonzero(accepted)
    if len(idx) < 2:
        return []
    p = np.asarray(xy, dtype=float)[idx]
    d = np.linalg.norm(p[:, None] - p[None], axis=2) * (REF_WIDTH / max(1.0, width))
    a, b = np.nonzero(np.triu(d < min_px, 1))
    return [(int(idx[i]), int(idx[j])) for i, j in zip(a, b, strict=True)]


# --------------------------------------------------------------------------- #
# Dataset audit (read-only)
# --------------------------------------------------------------------------- #
def match_key(source: str, group: str) -> str:
    """Recording/match unit for leakage checks (finer groups share a match)."""
    if source == "ts_worldcup":
        m = re.search(r"(\d{4}_Match_Highlights\d+)", group)
        return f"tswc:{m.group(1)}" if m else group
    if source == "soccernet":
        return "|".join(group.split("|")[:3])
    return group


def dhash_image(path: Path, size: int = 16):
    """256-bit difference hash (4 x uint64) + reduced image size; None if unreadable."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if img is None:
        return None, (0, 0)
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    words = np.packbits(bits).view(">u8").astype(np.uint64)
    return words, (img.shape[1] * 8, img.shape[0] * 8)


def _near_duplicates(ha: np.ndarray, hb: np.ndarray, max_bits: int, chunk: int = 256):
    """Nearest hb for every ha with Hamming distance <= max_bits: (ia, ib, bits)."""
    out = []
    for s in range(0, len(ha), chunk):
        x = ha[s : s + chunk, None, :] ^ hb[None, :, :]
        d = np.bitwise_count(x).sum(axis=2)
        j = d.argmin(axis=1)
        dm = d[np.arange(len(j)), j]
        for k in np.flatnonzero(dm <= max_bits):
            out.append((s + int(k), int(j[k]), int(dm[k])))
    return out


def _geom_module():
    """``freekiki_geom`` (imports this module, so it is loaded lazily)."""
    try:
        from . import freekiki_geom
    except ImportError:
        import freekiki_geom  # ty: ignore[unresolved-import]
    return freekiki_geom


def audit_dataset(ds_dir, out_dir, *, dup_bits: int = 10, hash_images: bool = True, log=print):
    """Read-only integrity + leakage audit of a kiki49 dataset. Returns the summary dict."""
    import yaml

    ds_dir, out_dir = Path(ds_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names, flip, xyz = load_field_points()
    mirror, rot = symmetry_maps(xyz)
    planar = xyz[:, 2] == 0
    data = yaml.safe_load((ds_dir / "data.yaml").read_text(encoding="utf-8")) or {}
    summary: dict = {"dataset": str(ds_dir), "date": datetime.now().isoformat(timespec="seconds")}

    # schema / flip_idx
    yflip = list(data.get("flip_idx") or [])
    summary["schema"] = {
        "flip_idx_matches_yaml": yflip == flip.tolist(),
        "flip_idx_involution": bool(np.all(flip[flip] == np.arange(NKP))),
        "flip_idx_equals_mirror_geometry": bool(np.all(flip == mirror)),
        "flip_idx_mismatches": [int(i) for i in np.flatnonzero(flip != mirror)],
        "rot180_complete": bool(np.all(rot >= 0)),
        "non_planar": [int(i) for i in np.flatnonzero(~planar)],
        "kpt_names_match": list((data.get("kpt_names") or {}).get(0, [])) in ([], names),
    }

    rows = list(csv.DictReader((ds_dir / "manifest.csv").open(encoding="utf-8")))
    summary["counts"] = {
        f"{s}/{src}": n
        for (s, src), n in sorted(Counter((r["split"], r["source"]) for r in rows).items())
    }
    # leakage by group and by match
    leaks = []
    for level, key in (
        ("group", lambda r: r["group"]),
        ("match", lambda r: match_key(r["source"], r["group"])),
    ):
        splits = defaultdict(Counter)
        for r in rows:
            splits[key(r)][r["split"]] += 1
        for k, c in sorted(splits.items()):
            if len(c) > 1:
                leaks.append(
                    {"level": level, "key": k}
                    | {s: c.get(s, 0) for s in ("train", "val", "test", "hard")}
                )
    write_csv(out_dir / "leakage_groups.csv", leaks or [{"level": "", "key": ""}])
    summary["leakage"] = {
        "groups_across_splits": sum(1 for x in leaks if x["level"] == "group"),
        "matches_across_splits": sum(1 for x in leaks if x["level"] == "match"),
    }

    # labels
    vis_values: Counter = Counter()
    issues, geom, outliers, missing = [], [], [], []
    avail = defaultdict(lambda: np.zeros(NKP, dtype=int))
    groups_kp = defaultdict(lambda: [set() for _ in range(NKP)])
    above_bad = Counter()
    above_n = Counter()
    sizes: dict[str, tuple[int, int]] = {}
    hashes: dict[str, np.ndarray] = {}
    if hash_images:
        log(f"audit: hashing {len(rows)} images (dHash 256-bit) ...")
        with ThreadPoolExecutor(max_workers=8) as pool:
            for r, (h, wh) in zip(
                rows, pool.map(lambda r: dhash_image(ds_dir / r["image"]), rows), strict=True
            ):
                if h is not None:
                    hashes[r["image"]] = h
                    sizes[r["image"]] = wh
    log("audit: checking labels ...")
    for r in rows:
        lp = ds_dir / r["label"]
        lines = (
            [x.split() for x in lp.read_text(encoding="utf-8").splitlines() if x.strip()]
            if lp.is_file()
            else []
        )
        base = {"split": r["split"], "image": r["image"], "source": r["source"]}
        if not lines:
            issues.append(base | {"issue": "no_label"})
            continue
        if len(lines) > 1:
            issues.append(base | {"issue": f"instances={len(lines)}"})
        if len(lines[0]) != 5 + 3 * NKP:
            issues.append(base | {"issue": f"columns={len(lines[0])}"})
            continue
        k = np.asarray(lines[0][5:], dtype=float).reshape(NKP, 3)
        vis_values.update(k[:, 2].astype(int).tolist())
        vis = k[:, 2] > 0
        if not np.isin(k[:, 2], (0, 1, 2)).all():
            issues.append(base | {"issue": "visibility_not_0_1_2"})
        if ((k[vis, :2] < 0) | (k[vis, :2] > 1)).any():
            issues.append(base | {"issue": "visible_out_of_range"})
        if (k[~vis, :2] != 0).any():
            issues.append(base | {"issue": "hidden_with_coords"})
        w, h = sizes.get(r["image"], (1920, 1080))
        px = k[:, :2] * (w, h)
        for a, b in duplicate_points(px, vis, w, min_px=1.0):
            issues.append(base | {"issue": f"duplicate p{a}=p{b}"})
        avail[(r["split"], r["source"])] += vis
        for i in np.flatnonzero(vis):
            groups_kp[r["split"]][i].add(r["group"])
        for top, bottom in ABOVE_PAIRS:
            if vis[top] and vis[bottom]:
                above_n[(top, bottom)] += 1
                if px[top, 1] >= px[bottom, 1]:
                    above_bad[(top, bottom)] += 1
                    issues.append(base | {"issue": f"p{top} not above p{bottom}"})
        use = vis & planar
        fit = fit_field_homography(px[use], xyz[use, :2], width=w, ransac_px=15.0, max_rmse_px=1e9)
        geom.append(
            base
            | {
                "group": r["group"],
                "n_planar": int(use.sum()),
                "status": fit["status"],
                "n_inliers": fit["n_inliers"],
                "rmse_px": fit["rmse_px"],
                "orientation": fit["orientation"],
            }
        )
        if fit["H"] is not None:
            res = np.linalg.norm(_project(fit["H"], xyz[use, :2]) - px[use], axis=1) * (
                REF_WIDTH / w
            )
            for i, e in zip(np.flatnonzero(use), res, strict=True):
                if e > 15.0:
                    outliers.append(
                        base
                        | {
                            "kp": f"p{i}",
                            "name": names[i],
                            "residual_px": round(float(e), 1),
                            "kind": "planar",
                        }
                    )
        # Flag / post-top labels vs the camera of the other labels (leave-one-out).
        if (vis & ~planar).any():
            for i, e in _geom_module().nonplanar_label_outliers(px, vis, w, h):
                outliers.append(
                    base | {"kp": f"p{i}", "name": names[i], "residual_px": e, "kind": "geom3d"}
                )
        # Visible-looking points labelled "not visible" (negative supervision).
        comp = label_completeness(
            [p if v else None for p, v in zip(px, vis, strict=True)], (), w, h
        )
        for i in comp["suspects"]:
            xy = comp["projected"].get(i, [None, None])
            missing.append(
                base
                | {
                    "group": r["group"],
                    "kp": f"p{i}",
                    "name": names[i],
                    "proj_x": _num(xy[0], 1),
                    "proj_y": _num(xy[1], 1),
                }
            )
    write_csv(out_dir / "label_issues.csv", issues or [{"split": "", "issue": ""}])
    write_csv(out_dir / "label_geometry.csv", geom)
    write_csv(out_dir / "label_outliers.csv", outliers or [{"image": "", "kp": ""}])
    write_csv(out_dir / "label_missing_suspects.csv", missing or [{"image": "", "kp": ""}])
    orient = Counter((g["status"], g["orientation"]) for g in geom)
    summary["labels"] = {
        "visibility_values": {str(k): v for k, v in sorted(vis_values.items())},
        "issues": dict(
            Counter(
                i["issue"].split(" ")[0] if i["issue"].startswith("duplicate") else i["issue"]
                for i in issues
            )
        ),
        "above_violations": {
            f"p{a}>p{b}": f"{above_bad[(a, b)]}/{above_n[(a, b)]}" for a, b in ABOVE_PAIRS
        },
        "label_homography": {f"{s}/{o}": n for (s, o), n in sorted(orient.items())},
        "mirrored_label_images": sum(1 for g in geom if g["status"] == "mirrored"),
        "keypoint_outliers_gt15px": len(outliers),
        "geom3d_label_outliers": sum(o.get("kind") == "geom3d" for o in outliers),
        # label_missing_suspects.csv: works as ``manifest --exclude`` (image column)
        "missing_label_suspects": {
            "images": len({m["image"] for m in missing}),
            "by_kp_split": {
                f"{kp}/{sp}": n
                for (kp, sp), n in sorted(
                    Counter((m["kp"], m["split"]) for m in missing).items(),
                    key=lambda item: -item[1],
                )
            },
        },
    }
    avail_rows = []
    for i in range(NKP):
        row = {"kp": f"p{i}", "name": names[i]}
        for s in ("train", "val", "test", "hard"):
            row[s] = int(sum(v[i] for (sp, _), v in avail.items() if sp == s))
            row[f"{s}_groups"] = len(groups_kp[s][i])
        for (sp, src), v in sorted(avail.items()):
            row[f"{sp}:{src}"] = int(v[i])
        avail_rows.append(row)
    write_csv(out_dir / "kp_availability.csv", avail_rows)

    # near-duplicates across splits
    if hashes:
        by_split = defaultdict(list)
        for r in rows:
            if r["image"] in hashes:
                by_split[r["split"]].append(r)
        dups = []
        for a, b in (
            ("val", "train"),
            ("test", "train"),
            ("test", "val"),
            ("hard", "train"),
            ("hard", "val"),
        ):
            ra, rb = by_split.get(a, []), by_split.get(b, [])
            if not ra or not rb:
                continue
            ha = np.stack([hashes[r["image"]] for r in ra])
            hb = np.stack([hashes[r["image"]] for r in rb])
            for ia, ib, bits in _near_duplicates(ha, hb, dup_bits):
                dups.append(
                    {
                        "pair": f"{a}-{b}",
                        "bits": bits,
                        "image_a": ra[ia]["image"],
                        "group_a": ra[ia]["group"],
                        "image_b": rb[ib]["image"],
                        "group_b": rb[ib]["group"],
                        "same_match": match_key(ra[ia]["source"], ra[ia]["group"])
                        == match_key(rb[ib]["source"], rb[ib]["group"]),
                    }
                )
        write_csv(out_dir / "near_duplicates.csv", dups or [{"pair": "", "bits": ""}])
        summary["near_duplicates"] = {
            "max_bits": dup_bits,
            **{
                p: sum(1 for d in dups if d["pair"] == p)
                for p in ("val-train", "test-train", "test-val")
            },
        }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------- #
# Oversampling manifest (repeat-factor sampling per keypoint)
# --------------------------------------------------------------------------- #
RFS_SCHEMA = 1


def repeat_factors(vis: np.ndarray, *, t: float, cap: float):
    """Repeat-factor sampling (Gupta et al., LVIS 2019) with keypoints as categories.

    ``vis`` is (n_images, NKP) bool. For keypoint k, ``f_k`` = fraction of
    images where k is labelled visible and ``r_k = max(1, sqrt(t / f_k))``
    (undefined, NaN, when no image has k). An image repeats
    ``r_i = min(cap, max(1, max r_k over its visible k))`` times on average.
    Returns ``(r_img, r_kp, f_kp)``.
    """
    vis = np.asarray(vis, dtype=bool)
    f_kp = vis.mean(axis=0) if len(vis) else np.zeros(vis.shape[1])
    safe = np.where(f_kp > 0, f_kp, 1.0)
    r_kp = np.where(f_kp > 0, np.maximum(1.0, np.sqrt(t / safe)), np.nan)
    per_img = np.where(vis, np.nan_to_num(r_kp, nan=1.0)[None, :], 1.0)
    r_img = np.minimum(cap, per_img.max(axis=1, initial=1.0))
    return r_img, r_kp, f_kp


def stochastic_copies(r_img: np.ndarray, seed: int) -> np.ndarray:
    """Integer copies per image: ``floor(r) + Bernoulli(r - floor(r))``, seeded."""
    r_img = np.asarray(r_img, dtype=float)
    base = np.floor(r_img)
    draw = np.random.default_rng(seed).random(len(r_img))
    return (base + (draw < r_img - base)).astype(int)


def _first_label_vis(text: str) -> np.ndarray | None:
    for line in text.splitlines():
        values = line.split()
        if len(values) == 5 + 3 * NKP:
            return np.asarray(values[5:], dtype=float).reshape(NKP, 3)[:, 2] > 0
    return None


def train_labels_digest(ds_dir) -> str:
    """SHA-256 over (label path, label SHA-1) of every train row of ``manifest.csv``."""
    ds_dir = Path(ds_dir)
    h = hashlib.sha256()
    with (ds_dir / "manifest.csv").open(encoding="utf-8") as f:
        rows = sorted(
            (r for r in csv.DictReader(f) if r["split"] == "train"), key=lambda r: r["label"]
        )
    for r in rows:
        lp = ds_dir / r["label"]
        digest = hashlib.sha1(lp.read_bytes()).hexdigest() if lp.is_file() else "missing"
        h.update(f"{r['label']}\t{digest}\n".encode())
    return h.hexdigest()


def build_rfs_manifest(
    ds_dir,
    out_dir,
    *,
    t: float = 0.05,
    cap: float = 4.0,
    seed: int = 0,
    exclude: set[str] | None = None,
    version: str = "",
    log=print,
) -> dict:
    """Write an oversampled train list for ``ds_dir`` into ``out_dir`` (read-only on the dataset).

    Files: ``train_list.txt`` (dataset-relative image paths, each repeated
    ``copies`` times), ``repeats.csv`` (per image), ``kp_counts.csv``
    (per keypoint, before/after) and ``manifest.json`` (parameters, input
    hashes, counts). ``exclude`` holds image file names left out (copies 0).
    Returns the manifest dict.
    """
    ds_dir, out_dir = Path(ds_dir), Path(out_dir)
    exclude = exclude or set()
    names, _, _ = load_field_points()
    with (ds_dir / "manifest.csv").open(encoding="utf-8") as f:
        rows = sorted(
            (r for r in csv.DictReader(f) if r["split"] == "train"), key=lambda r: r["image"]
        )
    if not rows:
        raise ValueError(f"No train rows in {ds_dir / 'manifest.csv'}")
    keep = [r for r in rows if Path(r["image"]).name not in exclude]
    vis = np.zeros((len(keep), NKP), dtype=bool)
    no_label = 0
    for i, r in enumerate(keep):
        lp = ds_dir / r["label"]
        v = _first_label_vis(lp.read_text(encoding="utf-8")) if lp.is_file() else None
        if v is None:
            no_label += 1
        else:
            vis[i] = v
    r_img, r_kp, f_kp = repeat_factors(vis, t=t, cap=cap)
    copies = stochastic_copies(r_img, seed)
    rarest = np.where(vis, np.nan_to_num(r_kp, nan=0.0)[None, :], 0.0).argmax(axis=1)

    out_dir.mkdir(parents=True, exist_ok=False)
    lines = [r["image"] for r, c in zip(keep, copies, strict=True) for _ in range(c)]
    (out_dir / "train_list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rep_rows = []
    by_image = {r["image"]: i for i, r in enumerate(keep)}
    for r in rows:
        i = by_image.get(r["image"])
        row = {"image": r["image"], "source": r["source"], "group": r["group"]}
        if i is None:
            rep_rows.append(row | {"repeat_factor": 0, "copies": 0, "reason": "excluded"})
            continue
        boosted = r_img[i] > 1.0
        k = int(rarest[i])
        rep_rows.append(
            row
            | {
                "repeat_factor": round(float(r_img[i]), 4),
                "copies": int(copies[i]),
                "reason": f"p{k} {names[k]}" if boosted else "",
            }
        )
    write_csv(out_dir / "repeats.csv", rep_rows)

    before = vis.sum(axis=0)
    after = (vis * copies[:, None]).sum(axis=0)
    n_after = int(copies.sum())
    kp_rows = [
        {
            "kp": f"p{k}",
            "name": names[k],
            "images_before": int(before[k]),
            "freq_before": _num(float(f_kp[k]), 5),
            "repeat_factor_kp": _num(float(r_kp[k])),
            "instances_after": int(after[k]),
            "freq_after": _num(float(after[k] / n_after), 5) if n_after else None,
        }
        for k in range(NKP)
    ]
    write_csv(out_dir / "kp_counts.csv", kp_rows)

    src_before = Counter(r["source"] for r in keep)
    src_after: Counter = Counter()
    for r, c in zip(keep, copies, strict=True):
        src_after[r["source"]] += int(c)
    info = {
        "schema": RFS_SCHEMA,
        "version": version or out_dir.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(ds_dir.resolve()),
        "method": {
            "name": "repeat_factor_sampling",
            "reference": "Gupta, Dollar, Girshick. LVIS. CVPR 2019 (keypoints as categories)",
            "t": t,
            "cap": cap,
            "seed": seed,
            "rounding": "floor(r) + Bernoulli(frac(r)), numpy default_rng(seed), images sorted",
            "visible": "label visibility > 0 (first instance)",
        },
        "inputs": {
            "manifest_csv_sha256": hashlib.sha256(
                (ds_dir / "manifest.csv").read_bytes()
            ).hexdigest(),
            "data_yaml_sha256": (
                hashlib.sha256((ds_dir / "data.yaml").read_bytes()).hexdigest()
                if (ds_dir / "data.yaml").is_file()
                else None
            ),
            "train_labels_digest": train_labels_digest(ds_dir),
        },
        "counts": {
            "train_images": len(rows),
            "excluded": len(rows) - len(keep),
            "without_label": no_label,
            "kept_images": len(keep),
            "images_repeated": int((copies > 1).sum()),
            "entries_after": n_after,
            "by_source_before": dict(sorted(src_before.items())),
            "by_source_after": dict(sorted(src_after.items())),
        },
        "val_test": "unchanged (data.yaml val/test of the dataset)",
        "train_list_sha256": hashlib.sha256((out_dir / "train_list.txt").read_bytes()).hexdigest(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log(
        f"manifest {info['version']}: {len(keep)} train images -> {n_after} entries "
        f"({info['counts']['images_repeated']} repeated, {len(rows) - len(keep)} excluded)"
    )
    for k in np.argsort(before, kind="stable"):
        r = kp_rows[k]
        if not np.isfinite(r_kp[k]) or r_kp[k] <= 1.0:
            continue
        log(
            f"  {r['kp']:>4} {r['name']:<38} images {r['images_before']:>5} -> "
            f"{r['instances_after']:>5}  (r_k {r['repeat_factor_kp']})"
        )
    return info


# --------------------------------------------------------------------------- #
# Promotion gate
# --------------------------------------------------------------------------- #
def names_digest(names) -> str:
    """SHA-1 of the sorted image names: two evaluations are comparable only if equal."""
    return hashlib.sha1("\n".join(sorted(map(str, names))).encode()).hexdigest()


def _read_eval(eval_dir: Path) -> tuple[dict, dict[str, dict]]:
    summary = json.loads((Path(eval_dir) / "eval_summary.json").read_text(encoding="utf-8"))
    with (Path(eval_dir) / "per_keypoint.csv").open(encoding="utf-8") as f:
        per_kp = {r["kp"]: r for r in csv.DictReader(f)}
    return summary, per_kp


def _f(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def compare_evals(baseline_dir, candidate_dir, gate: dict | None = None) -> dict:
    """Decide whether the candidate may replace the baseline (same split, same thresholds).

    Checks: pose mAP50-95, overall PCK@t (misses = failures), the recall and
    median error of every critical keypoint with enough support in both, and
    the rate of images whose predicted homography is valid and correct
    (labelled points reproject within ``CALIB_CORRECT_PX``). Returns a decision dict
    with ``promote`` and the list of ``reasons`` (failed checks) / ``notes``.
    """
    g: dict[str, Any] = PROMOTION_DEFAULTS | (gate or {})
    bs, bk = _read_eval(Path(baseline_dir))
    cs, ck = _read_eval(Path(candidate_dir))
    reasons, notes, checks = [], [], []

    def check(name, base, cand, max_drop, higher_better=True):
        if base is None or cand is None:
            notes.append(f"{name}: undefined (baseline={base}, candidate={cand})")
            return
        delta = cand - base if higher_better else base - cand
        ok = delta >= -max_drop
        checks.append(
            {"check": name, "baseline": base, "candidate": cand, "tolerance": max_drop, "pass": ok}
        )
        if not ok:
            reasons.append(f"{name}: {base} -> {cand} (tolerance {max_drop})")

    for key in ("split", "n_images", "det_conf", "kp_conf", "match_px", "images_sha1"):
        if bs.get(key) != cs.get(key):
            reasons.append(f"not comparable: {key} differs ({bs.get(key)} vs {cs.get(key)})")
    if not reasons:
        pck = f"pck{g['pck_px']:g}_all"
        check(
            "pose_map50_95",
            _f(bs["ultralytics"].get("metrics/mAP50-95(P)")),
            _f(cs["ultralytics"].get("metrics/mAP50-95(P)")),
            g["max_map_drop"],
        )
        check(
            f"overall_{pck}",
            _f(bs["keypoints"].get(pck)),
            _f(cs["keypoints"].get(pck)),
            g["max_pck_all_drop"],
        )
        check(
            "calib_correct_rate",
            _f(bs.get("calib", {}).get("correct_rate")),
            _f(cs.get("calib", {}).get("correct_rate")),
            g["max_calib_correct_drop"],
        )
        for i in g["critical_kps"]:
            b, c = bk.get(f"p{i}", {}), ck.get(f"p{i}", {})
            support = min(int(_f(b.get("n_ref")) or 0), int(_f(c.get("n_ref")) or 0))
            if support < int(g["min_support"]):
                notes.append(f"p{i}: support {support} < {g['min_support']}, not gated")
                continue
            check(f"p{i}_recall", _f(b.get("recall")), _f(c.get("recall")), g["max_recall_drop"])
            check(
                f"p{i}_err_median",
                _f(b.get("err_median")),
                _f(c.get("err_median")),
                g["max_median_err_increase_px"],
                higher_better=False,
            )
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "baseline": str(baseline_dir),
        "candidate": str(candidate_dir),
        "baseline_model": bs.get("model"),
        "candidate_model": cs.get("model"),
        "split": cs.get("split"),
        "gate": g,
        "promote": not reasons,
        "reasons": reasons,
        "notes": notes,
        "checks": checks,
    }
