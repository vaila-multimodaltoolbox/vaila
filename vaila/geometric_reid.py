"""Shared geometric Re-ID helpers (Hungarian matching, IoU, velocity direction).

Used by yolov26track, vaila_sam, and reid_markers to keep ID-stabilization logic
consistent across bbox tracking, SAM exports, and marker CSV workflows.

Update Date: 07 July 2026
Version: 0.3.79
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def apply_homography_to_xy(points: np.ndarray, homography: np.ndarray | None) -> np.ndarray:
    """Map Nx2 points through a 3x3 homography when available."""
    pts = np.asarray(points, dtype=float)
    if homography is None or pts.size == 0:
        return pts
    h_mat = np.asarray(homography, dtype=float)
    if h_mat.shape != (3, 3):
        raise ValueError("Homography matrix must be 3x3.")
    pts_h = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
    mapped = (h_mat @ pts_h.T).T
    denom = mapped[:, 2:3]
    bad = np.isclose(denom, 0.0)
    denom[bad] = np.nan
    return mapped[:, :2] / denom


def assignment_min_cost(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Rectangular minimum-cost assignment (Hungarian with greedy fallback)."""
    if cost_matrix.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost_matrix)
        return [(int(r), int(c)) for r, c in zip(rows, cols, strict=True)]
    except Exception:
        remaining_rows = set(range(cost_matrix.shape[0]))
        remaining_cols = set(range(cost_matrix.shape[1]))
        pairs: list[tuple[int, int]] = []
        while remaining_rows and remaining_cols:
            best: tuple[float, int, int] | None = None
            for r in remaining_rows:
                for c in remaining_cols:
                    val = float(cost_matrix[r, c])
                    if best is None or val < best[0]:
                        best = (val, r, c)
            if best is None:
                break
            _val, r_best, c_best = best
            pairs.append((r_best, c_best))
            remaining_rows.remove(r_best)
            remaining_cols.remove(c_best)
        return pairs


def bbox_iou_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """IoU between axis-aligned boxes in xyxy pixel coordinates."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def bbox_iou_xywh(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """IoU between xywh boxes (x, y, w, h)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return bbox_iou_xyxy((ax, ay, ax + aw, ay + ah), (bx, by, bx + bw, by + bh))


def centroid_xyxy(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x_min, y_min, x_max, y_max = bbox
    return (x_min + x_max) * 0.5, (y_min + y_max) * 0.5


# COCO-17 body keypoint indices inside Sapiens2 308-kp topology (Sociopticon).
COCO17_KPT_INDICES: tuple[int, ...] = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
)

# Default per-keypoint sigmas (COCO pose OKS).
COCO17_OKS_SIGMAS: tuple[float, ...] = (
    0.026,
    0.025,
    0.025,
    0.035,
    0.035,
    0.079,
    0.079,
    0.072,
    0.072,
    0.062,
    0.062,
    0.107,
    0.107,
    0.087,
    0.087,
    0.089,
    0.089,
)


def keypoint_oks_similarity(
    kpts_a: np.ndarray | list | None,
    scores_a: np.ndarray | list | None,
    kpts_b: np.ndarray | list | None,
    scores_b: np.ndarray | list | None,
    bbox_area: float,
    *,
    kpt_thr: float = 0.3,
    kpt_indices: tuple[int, ...] = COCO17_KPT_INDICES,
    sigmas: tuple[float, ...] = COCO17_OKS_SIGMAS,
) -> float:
    """Object Keypoint Similarity (COCO-style) on a keypoint subset.

    Returns 0.0 when no confident overlapping keypoints exist.
    """
    if kpts_a is None or kpts_b is None or scores_a is None or scores_b is None:
        return 0.0
    ka = np.asarray(kpts_a, dtype=float)
    kb = np.asarray(kpts_b, dtype=float)
    sa = np.asarray(scores_a, dtype=float).reshape(-1)
    sb = np.asarray(scores_b, dtype=float).reshape(-1)
    if ka.ndim != 2 or kb.ndim != 2 or ka.shape[1] < 2 or kb.shape[1] < 2:
        return 0.0
    area = max(float(bbox_area), 1.0)
    scale = 2.0 * np.sqrt(area)
    if scale <= 0.0:
        return 0.0
    oks_sum = 0.0
    n_valid = 0
    for idx, sigma in zip(kpt_indices, sigmas, strict=False):
        if idx >= len(ka) or idx >= len(kb) or idx >= len(sa) or idx >= len(sb):
            continue
        if sa[idx] < kpt_thr or sb[idx] < kpt_thr:
            continue
        dx = float(ka[idx, 0] - kb[idx, 0])
        dy = float(ka[idx, 1] - kb[idx, 1])
        d2 = dx * dx + dy * dy
        var = (float(sigma) * scale) ** 2
        oks_sum += float(np.exp(-d2 / (2.0 * var + 1e-9)))
        n_valid += 1
    return float(oks_sum / n_valid) if n_valid > 0 else 0.0


def mask_iou_u8(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """IoU between two uint8 binary masks (same shape)."""
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    inter = int(np.count_nonzero((a > 0) & (b > 0)))
    if inter <= 0:
        return 0.0
    union = int(np.count_nonzero((a > 0) | (b > 0)))
    return float(inter / union) if union > 0 else 0.0


@dataclass
class GeometricLinkerConfig:
    max_gap: int = 12
    max_centroid_dist_px: float = 180.0
    min_iou: float = 0.05
    direction_weight: float = 0.0
    homography_matrix: np.ndarray | None = None
    mask_iou_weight: float = 0.0
    # Per-frame detectors (Sapiens2 DETR): lock low-speed tracks to an anchor so
    # crossing athletes do not steal static IDs (SAM3 has native video IDs first).
    static_speed_threshold: float = 0.0
    static_anchor_radius_px: float = 70.0
    static_mismatch_penalty: float = 4.0
    mobility_warmup_frames: int = 6
    # Sapiens2 pose: OKS on COCO-17 subset inside 308-kp topology.
    kpt_oks_weight: float = 0.0
    kpt_thr: float = 0.3


def detection_link_xy(det: dict[str, Any]) -> tuple[float, float]:
    """Link point for Hungarian matching — optional ``link_xy``, else bbox center."""
    link = det.get("link_xy")
    if link is not None:
        return float(link[0]), float(link[1])
    return centroid_xyxy(det["xyxy"])


def pairwise_link_cost(
    det_xy: np.ndarray,
    trk_xy: np.ndarray,
    *,
    det_bbox_xyxy: tuple[float, float, float, float],
    trk_bbox_xyxy: tuple[float, float, float, float],
    trk_vel: np.ndarray,
    config: GeometricLinkerConfig,
    det_mask: np.ndarray | None = None,
    trk_mask: np.ndarray | None = None,
    det_keypoints: np.ndarray | list | None = None,
    det_keypoint_scores: np.ndarray | list | None = None,
    trk_keypoints: np.ndarray | list | None = None,
    trk_keypoint_scores: np.ndarray | list | None = None,
) -> tuple[float, float]:
    """Return (cost, distance) for one detection-track pair; inf if gated out."""
    det_h = apply_homography_to_xy(det_xy.reshape(1, 2), config.homography_matrix)[0]
    trk_h = apply_homography_to_xy(trk_xy.reshape(1, 2), config.homography_matrix)[0]
    dvec = det_h - trk_h
    dist = float(np.linalg.norm(dvec))
    iou = bbox_iou_xyxy(det_bbox_xyxy, trk_bbox_xyxy)
    if dist > config.max_centroid_dist_px and iou < config.min_iou:
        return float("inf"), dist

    norm_prod = float(np.linalg.norm(trk_vel) * np.linalg.norm(dvec))
    cosine = float(np.dot(dvec, trk_vel) / norm_prod) if norm_prod > 1e-9 else 1.0
    cosine = max(-1.0, min(1.0, cosine))
    alignment_penalty = 1.0 - cosine
    cost = (dist / config.max_centroid_dist_px) + (1.0 - iou)
    if config.direction_weight > 0.0:
        cost += float(config.direction_weight) * alignment_penalty
    if config.mask_iou_weight > 0.0 and det_mask is not None and trk_mask is not None:
        miou = mask_iou_u8(det_mask, trk_mask)
        cost += float(config.mask_iou_weight) * (1.0 - miou)
    if config.kpt_oks_weight > 0.0:
        ax1, ay1, ax2, ay2 = det_bbox_xyxy
        bx1, by1, bx2, by2 = trk_bbox_xyxy
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        bbox_area = max(area_a, area_b, 1.0)
        oks = keypoint_oks_similarity(
            det_keypoints,
            det_keypoint_scores,
            trk_keypoints,
            trk_keypoint_scores,
            bbox_area,
            kpt_thr=config.kpt_thr,
        )
        cost += float(config.kpt_oks_weight) * (1.0 - oks)
    return cost, dist


class GeometricFrameLinker:
    """Frame-to-frame geometric ID stabilizer with Hungarian assignment."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        config: GeometricLinkerConfig | None = None,
        start_stable_id: int = 1,
    ) -> None:
        self.enabled = enabled
        self.config = config or GeometricLinkerConfig()
        self.active: dict[int, dict[str, Any]] = {}
        self.next_stable_id = start_stable_id
        self.reid_links: list[tuple[int, int, int]] = []

    def assign_frame(
        self, frame_idx: int, detections: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Assign ``stable_id`` to each detection; log ``(frame, raw_id, stable_id)``."""
        if not self.enabled:
            for det in detections:
                det["stable_id"] = int(det.get("tracker_id", det.get("raw_id", 0)))
            return detections

        if not detections:
            return []

        active_ids = [
            tid
            for tid, tr in self.active.items()
            if 0 <= frame_idx - int(tr["frame"]) <= self.config.max_gap
        ]

        if not active_ids:
            out: list[dict[str, Any]] = []
            for det in detections:
                det_out = self._spawn_track(frame_idx, det)
                out.append(det_out)
            return out

        n_det = len(detections)
        n_trk = len(active_ids)
        cost = np.full((n_det, n_trk), 1e6, dtype=float)
        dist_m = np.full((n_det, n_trk), np.inf, dtype=float)

        for i, det in enumerate(detections):
            x_min, y_min, x_max, y_max = det["xyxy"]
            bbox = (float(x_min), float(y_min), float(x_max), float(y_max))
            det_xy = np.array(detection_link_xy(det), dtype=float)
            det_mask = det.get("mask_u8")
            for j, tid in enumerate(active_ids):
                tr = self.active[tid]
                tr_bbox = tr["bbox"]
                tr_xy = np.asarray(tr["centroid"], dtype=float)
                tr_vel = np.asarray(tr.get("vel", np.zeros(2)), dtype=float)
                c, d = pairwise_link_cost(
                    det_xy,
                    tr_xy,
                    det_bbox_xyxy=bbox,
                    trk_bbox_xyxy=tr_bbox,
                    trk_vel=tr_vel,
                    config=self.config,
                    det_mask=det_mask if isinstance(det_mask, np.ndarray) else None,
                    trk_mask=tr.get("mask_u8"),
                    det_keypoints=det.get("keypoints"),
                    det_keypoint_scores=det.get("keypoint_scores"),
                    trk_keypoints=tr.get("keypoints"),
                    trk_keypoint_scores=tr.get("keypoint_scores"),
                )
                c = self._mobility_cost_adjustment(tid, tr, det_xy, d, c)
                cost[i, j] = c
                dist_m[i, j] = d

        assignments = assignment_min_cost(cost)
        assigned_det: set[int] = set()
        assigned_trk: set[int] = set()
        out = [dict(d) for d in detections]

        for det_idx, trk_idx in assignments:
            if cost[det_idx, trk_idx] >= 1e5:
                continue
            if dist_m[det_idx, trk_idx] > self.config.max_centroid_dist_px:
                iou = bbox_iou_xyxy(
                    tuple(float(v) for v in detections[det_idx]["xyxy"]),  # type: ignore[arg-type]
                    self.active[active_ids[trk_idx]]["bbox"],
                )
                if iou < self.config.min_iou:
                    continue
            tid = active_ids[trk_idx]
            assigned_det.add(det_idx)
            assigned_trk.add(trk_idx)
            det_out = self._attach_track(frame_idx, tid, detections[det_idx])
            out[det_idx] = det_out

        for i, det in enumerate(detections):
            if i not in assigned_det:
                out[i] = self._spawn_track(frame_idx, det)

        for tid in active_ids:
            if (
                tid not in assigned_trk
                and frame_idx - int(self.active[tid]["frame"]) > self.config.max_gap
            ):
                del self.active[tid]

        return out

    def _mobility_cost_adjustment(
        self,
        tid: int,
        tr: dict[str, Any],
        det_xy: np.ndarray,
        dist: float,
        base_cost: float,
    ) -> float:
        """Penalize static/mobile mismatches when ``static_speed_threshold`` is set."""
        cfg = self.config
        if cfg.static_speed_threshold <= 0.0 or base_cost >= 1e5:
            return base_cost
        mobility = str(tr.get("mobility", "unknown"))
        if mobility == "static":
            anchor = np.asarray(tr.get("anchor", tr["centroid"]), dtype=float)
            anchor_dist = float(np.linalg.norm(det_xy - anchor))
            if anchor_dist > cfg.static_anchor_radius_px:
                base_cost += cfg.static_mismatch_penalty * (
                    anchor_dist / max(cfg.max_centroid_dist_px, 1.0)
                )
            tr_speed = float(np.linalg.norm(np.asarray(tr.get("vel", np.zeros(2)), dtype=float)))
            if tr_speed < cfg.static_speed_threshold and dist > cfg.static_anchor_radius_px * 0.6:
                base_cost += cfg.static_mismatch_penalty * 0.75
        elif mobility == "mobile":
            tr_vel = np.asarray(tr.get("vel", np.zeros(2)), dtype=float)
            tr_speed = float(np.linalg.norm(tr_vel))
            if tr_speed > cfg.static_speed_threshold:
                pred = np.asarray(tr["centroid"], dtype=float) + tr_vel
                pred_miss = float(np.linalg.norm(det_xy - pred))
                if (
                    pred_miss > cfg.max_centroid_dist_px * 0.45
                    and dist > cfg.static_anchor_radius_px
                ):
                    base_cost += cfg.static_mismatch_penalty * 0.5
        return base_cost

    def _spawn_track(self, frame_idx: int, det: dict[str, Any]) -> dict[str, Any]:
        tid = self.next_stable_id
        self.next_stable_id += 1
        return self._attach_track(frame_idx, tid, det)

    def _attach_track(self, frame_idx: int, tid: int, det: dict[str, Any]) -> dict[str, Any]:
        x_min, y_min, x_max, y_max = det["xyxy"]
        bbox = (float(x_min), float(y_min), float(x_max), float(y_max))
        cx, cy = detection_link_xy(det)
        prev = self.active.get(tid)
        vel = np.zeros(2, dtype=float)
        birth = frame_idx
        speed_ema = 0.0
        mobility = "unknown"
        anchor = (cx, cy)
        if prev is not None:
            prev_c = np.asarray(prev["centroid"], dtype=float)
            vel = np.array([cx, cy], dtype=float) - prev_c
            birth = int(prev.get("birth", frame_idx))
            speed_ema = float(prev.get("speed_ema", 0.0))
            mobility = str(prev.get("mobility", "unknown"))
            anchor = tuple(prev.get("anchor", (cx, cy)))
        speed_ema = 0.65 * speed_ema + 0.35 * float(np.linalg.norm(vel))
        cfg = self.config
        if cfg.static_speed_threshold > 0.0 and frame_idx - birth >= cfg.mobility_warmup_frames:
            if mobility == "unknown":
                if speed_ema > cfg.static_speed_threshold * 2.5:
                    mobility = "mobile"
                elif speed_ema < cfg.static_speed_threshold:
                    mobility = "static"
                    anchor = (cx, cy)
            elif mobility == "static":
                anchor = (cx, cy)
        self.active[tid] = {
            "frame": frame_idx,
            "birth": birth,
            "bbox": bbox,
            "centroid": (cx, cy),
            "vel": vel,
            "speed_ema": speed_ema,
            "mobility": mobility,
            "anchor": anchor,
            "mask_u8": det.get("mask_u8"),
            "keypoints": det.get("keypoints"),
            "keypoint_scores": det.get("keypoint_scores"),
        }
        raw_id = int(det.get("raw_id", det.get("tracker_id", 0)))
        self.reid_links.append((frame_idx, raw_id, tid))
        det_out = dict(det)
        det_out["stable_id"] = tid
        return det_out


def write_reid_links_csv(
    path: str, links: list[tuple[int, int, int]], header: tuple[str, ...]
) -> None:
    """Write audit CSV for ID remapping."""
    import csv

    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in links:
            writer.writerow(row)
