"""Unit tests for yolov26track pose ROI helpers and geometric ID linker."""

from __future__ import annotations

import numpy as np
import pytest

from vaila.yolov26track import (
    POSE_KEYPOINT_NAMES,
    _GeometricTrackLinker,
    map_pose_keypoints_to_global,
    prepare_pose_roi,
    select_pose_person,
)


def test_prepare_pose_roi_upscales_small_bbox() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    xyxy = (100, 200, 140, 280)  # 40 x 80
    roi, scale, off_x, off_y = prepare_pose_roi(frame, xyxy, pad_pct=0.0, min_side=256)
    assert off_x == 100
    assert off_y == 200
    assert roi.shape[0] >= 256 or roi.shape[1] >= 256
    assert scale > 1.0
    assert roi.shape[0] == int(round(80 * scale))
    assert roi.shape[1] == int(round(40 * scale))


def test_map_pose_keypoints_to_global_roundtrip() -> None:
    scale = 2.0
    off_x, off_y = 10, 20
    kps_roi = np.array([[100.0, 50.0, 0.9], [120.0, 60.0, 0.8]], dtype=np.float32)
    global_kps = map_pose_keypoints_to_global(kps_roi, scale, off_x, off_y)
    assert global_kps[0][0] == pytest.approx(60.0)
    assert global_kps[0][1] == pytest.approx(45.0)
    assert global_kps[1][0] == pytest.approx(70.0)
    assert global_kps[1][1] == pytest.approx(50.0)


def test_select_pose_person_picks_center() -> None:
    # Two people: one at corner, one at center of 200x200 ROI
    person_corner = np.zeros((len(POSE_KEYPOINT_NAMES), 3), dtype=np.float32)
    person_corner[0] = [10.0, 10.0, 0.95]
    person_center = np.zeros((len(POSE_KEYPOINT_NAMES), 3), dtype=np.float32)
    person_center[0] = [100.0, 100.0, 0.95]
    person_center[5] = [90.0, 110.0, 0.9]
    person_center[6] = [110.0, 110.0, 0.9]
    kps_list = np.stack([person_corner, person_center])
    idx = select_pose_person(kps_list, (200, 200))
    assert idx == 1


def test_geometric_linker_merges_id_switch() -> None:
    linker = _GeometricTrackLinker(enabled=True, max_gap=12)
    det0 = {
        "raw_id": 5,
        "tracker_id": 1,
        "xyxy": (100, 100, 200, 200),
        "label": "person",
        "conf": 0.9,
    }
    out0 = linker.assign_frame(0, [det0])
    stable0 = out0[0]["stable_id"]

    det1 = {
        "raw_id": 12,
        "tracker_id": 2,
        "xyxy": (102, 102, 202, 202),
        "label": "person",
        "conf": 0.9,
    }
    out1 = linker.assign_frame(1, [det1])
    stable1 = out1[0]["stable_id"]
    assert stable1 == stable0
    assert len(linker.reid_links) == 2


def test_geometric_linker_disabled_passthrough() -> None:
    linker = _GeometricTrackLinker(enabled=False)
    det = {"raw_id": 3, "tracker_id": 7, "xyxy": (0, 0, 50, 50), "label": "person", "conf": 0.5}
    out = linker.assign_frame(0, [det])
    assert out[0]["stable_id"] == 7
