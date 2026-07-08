"""Unit tests for shared geometric Re-ID (Hungarian linker, velocity cost)."""

from __future__ import annotations

import numpy as np
import pytest

from vaila.geometric_reid import (
    GeometricFrameLinker,
    GeometricLinkerConfig,
    assignment_min_cost,
    bbox_iou_xyxy,
    keypoint_oks_similarity,
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


def test_detection_link_xy_prefers_foot_proxy() -> None:
    from vaila.geometric_reid import detection_link_xy

    xy = detection_link_xy({"xyxy": (10.0, 20.0, 30.0, 100.0), "link_xy": (22.0, 95.0)})
    assert xy == (22.0, 95.0)


def test_static_mobility_penalty_reduces_far_jump() -> None:
    """Static track should not grab a detection far from its anchor."""
    config = GeometricLinkerConfig(
        max_gap=8,
        max_centroid_dist_px=180.0,
        min_iou=0.05,
        direction_weight=1.0,
        static_speed_threshold=5.0,
        static_anchor_radius_px=60.0,
        static_mismatch_penalty=5.0,
        mobility_warmup_frames=3,
    )
    linker = GeometricFrameLinker(enabled=True, config=config, start_stable_id=0)
    for fi, xy in enumerate([(500.0, 400.0), (501.0, 400.0), (499.0, 400.0), (500.0, 400.0)]):
        linker.assign_frame(
            fi,
            [{"raw_id": 1, "tracker_id": 1, "xyxy": (xy[0] - 20, 300, xy[0] + 20, 420), "link_xy": xy}],
        )
    out = linker.assign_frame(
        4,
        [
            {"raw_id": 1, "tracker_id": 1, "xyxy": (120, 300, 160, 420), "link_xy": (140.0, 420.0)},
            {"raw_id": 2, "tracker_id": 2, "xyxy": (480, 300, 520, 420), "link_xy": (500.0, 420.0)},
        ],
    )
    static_ids = [d["stable_id"] for d in out if d["link_xy"][0] > 400]
    moving_ids = [d["stable_id"] for d in out if d["link_xy"][0] < 200]
    assert static_ids == [0]
    assert moving_ids != [0]


def test_keypoint_oks_similarity_identical_pose() -> None:
    kpts = np.array([[100.0, 100.0], [110.0, 90.0], [90.0, 90.0]], dtype=float)
    scores = np.array([0.9, 0.9, 0.9], dtype=float)
    oks = keypoint_oks_similarity(
        kpts,
        scores,
        kpts.copy(),
        scores.copy(),
        bbox_area=4000.0,
        kpt_indices=(0, 1, 2),
        sigmas=(0.05, 0.05, 0.05),
    )
    assert oks == pytest.approx(1.0, abs=0.01)


def test_oks_cost_prefers_same_pose_over_crossing_bbox() -> None:
    """Swapped bboxes but consistent keypoints should keep track IDs."""
    n_kpts = 17
    kpts_left = np.zeros((n_kpts, 2), dtype=float)
    kpts_right = np.zeros((n_kpts, 2), dtype=float)
    scores = np.full(n_kpts, 0.95, dtype=float)
    for i in range(n_kpts):
        kpts_left[i] = [100.0 + i, 200.0 + i * 0.5]
        kpts_right[i] = [300.0 + i, 200.0 + i * 0.5]

    config = GeometricLinkerConfig(
        max_gap=5,
        max_centroid_dist_px=200.0,
        min_iou=0.01,
        direction_weight=0.5,
        kpt_oks_weight=0.8,
        kpt_thr=0.3,
    )
    linker = GeometricFrameLinker(enabled=True, config=config, start_stable_id=0)
    out0 = linker.assign_frame(
        0,
        [
            {
                "raw_id": 1,
                "tracker_id": 1,
                "xyxy": (80, 180, 120, 220),
                "keypoints": kpts_left,
                "keypoint_scores": scores,
            },
            {
                "raw_id": 2,
                "tracker_id": 2,
                "xyxy": (280, 180, 320, 220),
                "keypoints": kpts_right,
                "keypoint_scores": scores,
            },
        ],
    )
    id_left = out0[0]["stable_id"]
    id_right = out0[1]["stable_id"]

    # Bboxes swapped in list order; keypoints stay with correct person.
    out1 = linker.assign_frame(
        1,
        [
            {
                "raw_id": 4,
                "tracker_id": 4,
                "xyxy": (278, 178, 322, 222),
                "keypoints": kpts_right + np.array([2.0, 1.0]),
                "keypoint_scores": scores,
            },
            {
                "raw_id": 3,
                "tracker_id": 3,
                "xyxy": (78, 178, 122, 222),
                "keypoints": kpts_left + np.array([2.0, 1.0]),
                "keypoint_scores": scores,
            },
        ],
    )
    by_kpt_x = {int(np.mean(d["keypoints"][:, 0])): d["stable_id"] for d in out1}
    assert by_kpt_x[min(by_kpt_x)] == id_left
    assert by_kpt_x[max(by_kpt_x)] == id_right
