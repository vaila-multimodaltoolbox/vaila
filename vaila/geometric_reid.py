"""Shared geometric Re-ID helpers (Hungarian matching, IoU, velocity direction).

Used by yolov26track, vaila_sam, and reid_markers to keep ID-stabilization logic
consistent across bbox tracking, SAM exports, and marker CSV workflows.

Update Date: 04 July 2026
Version: 0.3.68
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

    def assign_frame(self, frame_idx: int, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
            cx, cy = centroid_xyxy(bbox)
            det_xy = np.array([cx, cy], dtype=float)
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
                )
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
            if tid not in assigned_trk and frame_idx - int(self.active[tid]["frame"]) > self.config.max_gap:
                del self.active[tid]

        return out

    def _spawn_track(self, frame_idx: int, det: dict[str, Any]) -> dict[str, Any]:
        tid = self.next_stable_id
        self.next_stable_id += 1
        return self._attach_track(frame_idx, tid, det)

    def _attach_track(self, frame_idx: int, tid: int, det: dict[str, Any]) -> dict[str, Any]:
        x_min, y_min, x_max, y_max = det["xyxy"]
        bbox = (float(x_min), float(y_min), float(x_max), float(y_max))
        cx, cy = centroid_xyxy(bbox)
        prev = self.active.get(tid)
        vel = np.zeros(2, dtype=float)
        if prev is not None:
            prev_c = np.asarray(prev["centroid"], dtype=float)
            vel = np.array([cx, cy], dtype=float) - prev_c
        self.active[tid] = {
            "frame": frame_idx,
            "bbox": bbox,
            "centroid": (cx, cy),
            "vel": vel,
            "mask_u8": det.get("mask_u8"),
        }
        raw_id = int(det.get("raw_id", det.get("tracker_id", 0)))
        self.reid_links.append((frame_idx, raw_id, tid))
        det_out = dict(det)
        det_out["stable_id"] = tid
        return det_out


def write_reid_links_csv(path: str, links: list[tuple[int, int, int]], header: tuple[str, ...]) -> None:
    """Write audit CSV for ID remapping."""
    import csv

    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in links:
            writer.writerow(row)
