"""Tests for Sapiens2 CSV loading in getpixelvideo smart tracking loader."""

from __future__ import annotations

import pandas as pd
import pytest

from vaila import getpixelvideo as gpv


def test_detect_tracking_format_sapiens_pose_long() -> None:
    df = pd.DataFrame(
        {
            "frame": [0, 0],
            "person_id": [1, 1],
            "kpt_idx": [0, 1],
            "x": [10.0, 11.0],
            "y": [20.0, 21.0],
            "score": [0.9, 0.8],
        }
    )
    assert gpv._detect_tracking_format(df) == "sapiens_pose_long"


def test_detect_tracking_format_sapiens_bbox_tracks() -> None:
    df = pd.DataFrame(
        {
            "frame": [0],
            "obj_id": [1],
            "x_px": [10.0],
            "y_px": [20.0],
            "w_px": [40.0],
            "h_px": [80.0],
            "score": [0.9],
        }
    )
    assert gpv._detect_tracking_format(df) == "sam_tracks"


def test_detect_tracking_format_sapiens_tracks_legacy() -> None:
    df = pd.DataFrame(
        {
            "frame": [0],
            "stable_id": [1],
            "x1": [10.0],
            "y1": [20.0],
            "x2": [50.0],
            "y2": [100.0],
            "mean_kpt_score": [0.9],
        }
    )
    assert gpv._detect_tracking_format(df) == "sapiens_tracks"


def test_sapiens_long_to_marker_df_pivot() -> None:
    df = pd.DataFrame(
        {
            "frame": [0, 0, 1],
            "person_id": [1, 1, 1],
            "kpt_idx": [0, 1, 0],
            "x": [10.0, 11.0, 12.0],
            "y": [20.0, 21.0, 22.0],
            "score": [0.9, 0.8, 0.7],
        }
    )
    wide = gpv._sapiens_long_to_marker_df(df, 1, kpt_thr=0.3)
    assert list(wide.columns[:3]) == ["frame", "kp000_x", "kp000_y"]
    assert wide.loc[0, "kp000_x"] == "10.0000"
    assert wide.loc[0, "kp001_y"] == "21.0000"
    assert wide.loc[1, "kp000_x"] == "12.0000"


def test_iter_bboxes_from_sapiens_bbox_tracks() -> None:
    df = pd.DataFrame(
        {
            "frame": [0],
            "obj_id": [3],
            "x_px": [10.0],
            "y_px": [20.0],
            "w_px": [40.0],
            "h_px": [80.0],
            "score": [0.95],
        }
    )
    boxes = gpv._iter_bboxes_from_df(df, "sam_tracks")
    assert len(boxes) == 1
    assert boxes[0]["obj_id"] == 3
    assert boxes[0]["x2"] == pytest.approx(50.0)
    assert boxes[0]["y2"] == pytest.approx(100.0)


def test_iter_bboxes_from_sapiens_tracks_legacy() -> None:
    df = pd.DataFrame(
        {
            "frame": [0],
            "stable_id": [2],
            "x1": [10.0],
            "y1": [20.0],
            "x2": [50.0],
            "y2": [100.0],
            "mean_kpt_score": [0.88],
        }
    )
    boxes = gpv._iter_bboxes_from_df(df, "sapiens_tracks")
    assert len(boxes) == 1
    assert boxes[0]["obj_id"] == 2
    assert boxes[0]["score"] == pytest.approx(0.88)
