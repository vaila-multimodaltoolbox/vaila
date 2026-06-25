"""Tests for FIFA / vailá marker CSV parsing (``pN_x`` columns including ``p0``)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vaila import getpixelvideo as gpv


def test_pitch_p_indices_includes_zero() -> None:
    cols = ["frame", "p0_x", "p0_y", "p31_x", "p31_y"]
    assert gpv._vaila_pitch_p_indices_from_columns(cols) == [0, 31]


def test_load_vaila_p_xy_markers_df_zero_based_fifa_slot() -> None:
    """``p0`` must map to list slot 0 (YOLO keypoint 0 / top_left_corner)."""
    df = pd.DataFrame(
        [
            {"frame": 0, "p0_x": 10.0, "p0_y": 20.0, "p1_x": 100.0, "p1_y": 200.0},
        ]
    )
    coords, labels = gpv.load_vaila_p_xy_markers_df(df, total_frames=5)
    assert coords[0][0] == (10.0, 20.0)
    assert coords[0][1] == (100.0, 200.0)
    assert labels[0] == "Pixel 0"


def test_load_vaila_p_xy_markers_df_legacy_one_based() -> None:
    df = pd.DataFrame([{"frame": 1, "p1_x": 5.0, "p1_y": 6.0, "p2_x": 7.0, "p2_y": 8.0}])
    coords, _ = gpv.load_vaila_p_xy_markers_df(df, total_frames=3)
    assert coords[1][0] == (5.0, 6.0)
    assert coords[1][1] == (7.0, 8.0)


def test_load_vaila_p_xy_sparse_middle_missing() -> None:
    df = pd.DataFrame([{"frame": 0, "p0_x": 1.0, "p0_y": 2.0, "p2_x": 9.0, "p2_y": 8.0}])
    coords, _ = gpv.load_vaila_p_xy_markers_df(df, total_frames=2)
    assert coords[0][0] == (1.0, 2.0)
    assert coords[0][1] == (None, None)
    assert coords[0][2] == (9.0, 8.0)


def test_sorted_frames_with_visible_markers_sparse_slots() -> None:
    coords = {
        0: [(1.0, 2.0), (None, None)],
        1: [(None, None), (None, None)],
        2: [(None, None), (3.0, 4.0)],
    }
    del_pos: dict[int, set[int]] = {0: set(), 1: set(), 2: set()}
    got = gpv.sorted_frames_with_visible_markers(
        coordinates=coords,
        deleted_positions=del_pos,
        one_line_mode=False,
        one_line_markers=[],
        deleted_markers=set(),
        total_frames=3,
    )
    assert got == [0, 2]


def test_frame_index_from_marker_timeline_snap_middle_of_bin() -> None:
    marked = [0, 4, 5, 6, 99]
    # strip_width 100, denom 100 → column 5 covers frames [5,6)
    assert (
        gpv.frame_index_from_marker_timeline_x(
            mouse_x=5,
            strip_left=0,
            strip_width=100,
            denom_frames=100,
            marked_sorted=marked,
        )
        == 5
    )


def test_frame_index_from_marker_timeline_proportional_when_bin_empty() -> None:
    marked = [50]
    assert (
        gpv.frame_index_from_marker_timeline_x(
            mouse_x=10,
            strip_left=0,
            strip_width=100,
            denom_frames=100,
            marked_sorted=marked,
        )
        == 10
    )


def test_sorted_frames_with_visible_markers_respects_deleted() -> None:
    coords = {0: [(9.0, 9.0)]}
    del_pos = {0: {0}}
    got = gpv.sorted_frames_with_visible_markers(
        coordinates=coords,
        deleted_positions=del_pos,
        one_line_mode=False,
        one_line_markers=[],
        deleted_markers=set(),
        total_frames=1,
    )
    assert got == []


def test_sam_tracks_bboxes_parse_and_keep_id_separate() -> None:
    df = pd.DataFrame(
        [
            {
                "frame": 0,
                "obj_id": 7,
                "x_px": 10.5,
                "y_px": 20.0,
                "w_px": 30.0,
                "h_px": 40.0,
                "score": 0.876,
            }
        ]
    )
    assert gpv._detect_tracking_format(df) == "sam_tracks"
    rows = gpv._iter_bboxes_from_df(df, "sam_tracks", video_width=100, video_height=100)
    assert rows == [
        {
            "frame": 0,
            "obj_id": 7,
            "x1": 10.5,
            "y1": 20.0,
            "x2": 40.5,
            "y2": 60.0,
            "label": "id7",
            "score": 0.876,
        }
    ]


def test_tracking_bbox_label_text_modes_do_not_duplicate_sam_id() -> None:
    box = {"label": "object", "id": 7, "conf": 0.876}
    assert gpv.tracking_bbox_label_text(box, "colors") == ""
    assert gpv.tracking_bbox_label_text(box, "id") == "ID 7"
    assert gpv.tracking_bbox_label_text(box, "id_conf") == "ID 7 0.88"


def test_normalize_bboxes_for_labeling_accepts_tracking_xyxy() -> None:
    tracking = {
        0: [
            {"x1": 10.4, "y1": 20.4, "x2": 40.6, "y2": 60.6, "label": "object", "id": 7}
        ]
    }
    assert gpv.normalize_bboxes_for_labeling(tracking) == {
        0: [{"x": 10, "y": 20, "w": 30, "h": 40, "label": "object", "id": 7}]
    }


def test_export_labeling_dataset_accepts_loaded_tracking_bboxes(tmp_path) -> None:
    import cv2
    import numpy as np

    video_path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48)
    )
    assert writer.isOpened()
    for i in range(3):
        frame = np.full((48, 64, 3), i * 30, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    tracking = {0: [{"x1": 8, "y1": 10, "x2": 28, "y2": 30, "label": "object", "id": 2}]}
    bboxes = gpv.normalize_bboxes_for_labeling(tracking)
    dataset_dir, msg = gpv.export_labeling_dataset(
        str(video_path), bboxes, 3, 64, 48, output_dataset_dir=None
    )

    assert dataset_dir is not None, msg
    dataset = Path(dataset_dir)
    assert (dataset / "data.yaml").is_file()
    labels = list(dataset.glob("*/labels/*.txt"))
    assert len(labels) == 1
    parts = labels[0].read_text(encoding="utf-8").strip().split()
    assert parts[0] == "0"
    assert len(parts) == 5
