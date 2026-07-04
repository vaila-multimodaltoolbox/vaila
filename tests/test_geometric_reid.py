"""Unit tests for shared geometric Re-ID (Hungarian linker, velocity cost)."""

from __future__ import annotations

import numpy as np
import pytest

from vaila.geometric_reid import (
    GeometricFrameLinker,
    GeometricLinkerConfig,
    assignment_min_cost,
    bbox_iou_xyxy,
    pairwise_link_cost,
)


def test_assignment_min_cost_prefers_cheaper_pairs() -> None:
    cost = np.array([[0.1, 2.0], [2.0, 0.1]], dtype=float)
    pairs = assignment_min_cost(cost)
    assert sorted(pairs) == [(0, 0), (1, 1)]


def test_hungarian_avoids_swap_on_crossing() -> None:
    """Two tracks crossing: list order inverted; velocity keeps correct IDs."""
    config = GeometricLinkerConfig(
        max_gap=5,
        max_centroid_dist_px=120.0,
        min_iou=0.01,
        direction_weight=0.8,
    )
    linker = GeometricFrameLinker(enabled=True, config=config, start_stable_id=1)

    out0 = linker.assign_frame(
        0,
        [
            {"raw_id": 1, "tracker_id": 1, "xyxy": (90, 90, 110, 110)},
            {"raw_id": 2, "tracker_id": 2, "xyxy": (190, 90, 210, 110)},
        ],
    )
    stable_a = out0[0]["stable_id"]
    stable_b = out0[1]["stable_id"]

    # Positions swapped in detection list (simulates greedy order bug).
    out1 = linker.assign_frame(
        1,
        [
            {"raw_id": 4, "tracker_id": 4, "xyxy": (188, 88, 212, 112)},
            {"raw_id": 3, "tracker_id": 3, "xyxy": (88, 88, 112, 112)},
        ],
    )
    ids_frame1 = sorted(d["stable_id"] for d in out1)
    assert ids_frame1 == sorted([stable_a, stable_b])


def test_velocity_penalty_favors_aligned_motion() -> None:
    config = GeometricLinkerConfig(
        max_centroid_dist_px=200.0,
        direction_weight=1.0,
    )
    trk_vel = np.array([10.0, 0.0])
    trk_xy = np.array([100.0, 100.0])
    det_aligned = np.array([112.0, 100.0])
    det_perp = np.array([100.0, 112.0])
    bbox_a = (105.0, 95.0, 115.0, 105.0)
    bbox_b = (95.0, 105.0, 105.0, 115.0)
    cost_aligned, _ = pairwise_link_cost(
        det_aligned,
        trk_xy,
        det_bbox_xyxy=bbox_a,
        trk_bbox_xyxy=(90.0, 90.0, 110.0, 110.0),
        trk_vel=trk_vel,
        config=config,
    )
    cost_perp, _ = pairwise_link_cost(
        det_perp,
        trk_xy,
        det_bbox_xyxy=bbox_b,
        trk_bbox_xyxy=(90.0, 90.0, 110.0, 110.0),
        trk_vel=trk_vel,
        config=config,
    )
    assert cost_aligned < cost_perp


def test_bbox_iou_xyxy_identical_boxes() -> None:
    box = (0.0, 0.0, 10.0, 10.0)
    assert bbox_iou_xyxy(box, box) == pytest.approx(1.0)
