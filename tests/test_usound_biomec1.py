"""Focused tests for vaila.usound_biomec1 helper logic."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vaila.usound_biomec1 import (
    ImageRecord,
    adjust_edge_parameters,
    build_comparison_plan,
    combine_rois,
    detect_ultrasound_roi,
    discover_before_after_groups,
    select_unique_best_pairs,
)


def _write_ultrasound_like_image(
    path: Path, rect: tuple[int, int, int, int], intensity: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    x, y, w, h = rect
    img[y : y + h, x : x + w] = intensity
    cv2.imwrite(str(path), img)


def test_detect_ultrasound_roi_returns_expected_bbox() -> None:
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[50:180, 80:220] = 180

    roi = detect_ultrasound_roi(img, padding=0)

    assert roi is not None
    x, y, w, h = roi
    assert abs(x - 80) <= 1
    assert abs(y - 50) <= 1
    assert abs(w - 140) <= 2
    assert abs(h - 130) <= 2


def test_combine_rois_returns_union_bbox() -> None:
    rois = [(80, 50, 140, 130), (70, 60, 100, 100), (90, 45, 50, 80)]
    assert combine_rois(rois, (240, 320)) == (70, 45, 150, 135)


def test_build_comparison_plan_marks_best_unique_pairs(tmp_path: Path) -> None:
    before_1 = tmp_path / "before_1.jpg"
    before_2 = tmp_path / "before_2.jpg"
    after_1 = tmp_path / "after_1.jpg"
    after_2 = tmp_path / "after_2.jpg"

    _write_ultrasound_like_image(before_1, (70, 50, 120, 120), 160)
    _write_ultrasound_like_image(before_2, (140, 60, 80, 120), 180)
    _write_ultrasound_like_image(after_1, (70, 50, 120, 120), 160)
    _write_ultrasound_like_image(after_2, (140, 60, 80, 120), 180)

    before_records = [
        ImageRecord(path=str(before_1), condition="before"),
        ImageRecord(path=str(before_2), condition="before"),
    ]
    after_records = [
        ImageRecord(path=str(after_1), condition="after"),
        ImageRecord(path=str(after_2), condition="after"),
    ]

    plan = build_comparison_plan(before_records, after_records)
    selected = [row for row in plan if row["selected_pair"]]

    assert len(plan) == 4
    assert len(selected) == 2
    assert {(row["before_file"], row["after_file"]) for row in selected} == {
        ("before_1.jpg", "after_1.jpg"),
        ("before_2.jpg", "after_2.jpg"),
    }
    assert selected == select_unique_best_pairs(plan)


def test_discover_before_after_groups_filters_incomplete_muscles(tmp_path: Path) -> None:
    valid_before = tmp_path / "rectus_femoris" / "before" / "before_1.jpg"
    valid_after = tmp_path / "rectus_femoris" / "after" / "after_1.jpg"
    missing_after = tmp_path / "vastus_lateralis" / "before" / "before_1.jpg"
    empty_before_dir = tmp_path / "gastrocnemius" / "before"
    empty_after_file = tmp_path / "gastrocnemius" / "after" / "after_1.jpg"

    _write_ultrasound_like_image(valid_before, (60, 40, 100, 110), 170)
    _write_ultrasound_like_image(valid_after, (60, 40, 100, 110), 170)
    _write_ultrasound_like_image(missing_after, (70, 50, 90, 100), 160)
    empty_before_dir.mkdir(parents=True, exist_ok=True)
    _write_ultrasound_like_image(empty_after_file, (80, 50, 80, 90), 150)

    groups, skipped = discover_before_after_groups(str(tmp_path))

    assert groups == [
        {
            "muscle": "rectus_femoris",
            "before_dir": str(tmp_path / "rectus_femoris" / "before"),
            "after_dir": str(tmp_path / "rectus_femoris" / "after"),
            "before_count": 1,
            "after_count": 1,
        }
    ]
    assert {row["muscle"] for row in skipped} == {"vastus_lateralis", "gastrocnemius"}
    skipped_by_muscle = {row["muscle"]: row["reason"] for row in skipped}
    assert skipped_by_muscle["vastus_lateralis"] == "missing after/"
    assert skipped_by_muscle["gastrocnemius"] == "empty before/"


def test_adjust_edge_parameters_survives_immediate_trackbar_callbacks(monkeypatch) -> None:
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    img[20:100, 40:120] = 180

    monkeypatch.setattr(cv2, "namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *args, **kwargs: 32)

    def fake_create_trackbar(name, window, value, max_value, callback):
        callback(value)

    monkeypatch.setattr(cv2, "createTrackbar", fake_create_trackbar)

    params = adjust_edge_parameters(img, img.copy())

    assert params == {"threshold1": 30, "threshold2": 100, "blur": 5}
