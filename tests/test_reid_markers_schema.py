"""Schema auto-detection + long-format extraction for reid_markers.py.

Covers the formats Geometric ReID (2D + velocity) must recognize without a
manual flag: bbox wide-per-slot (all_id_detection.csv), vailá's pN_x/pN_y
point convention, row-per-detection bbox (xyxy/xywh), and SAM long tracks.
"""

from __future__ import annotations

import pandas as pd
import pytest

from vaila.reid_markers import (
    INPUT_SCHEMA_BBOX_ROW_XYWH,
    INPUT_SCHEMA_BBOX_ROW_XYXY,
    INPUT_SCHEMA_BBOX_WIDE_SLOT,
    INPUT_SCHEMA_POINT_ROW,
    INPUT_SCHEMA_SAM_TRACKS,
    detect_input_schema,
    extract_long_detections,
)


def _bbox_wide_slot_df(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Frame": range(n),
            "Tracker ID_person_id_01": [1] * n,
            "Label_person_id_01": ["person"] * n,
            "X_min_person_id_01": [10 + i for i in range(n)],
            "Y_min_person_id_01": [10 + i for i in range(n)],
            "X_max_person_id_01": [30 + i for i in range(n)],
            "Y_max_person_id_01": [30 + i for i in range(n)],
            "Confidence_person_id_01": [0.9] * n,
            "Tracker ID_person_id_02": [2] * n,
            "Label_person_id_02": ["person"] * n,
            "X_min_person_id_02": [500 + i for i in range(n)],
            "Y_min_person_id_02": [500 + i for i in range(n)],
            "X_max_person_id_02": [520 + i for i in range(n)],
            "Y_max_person_id_02": [520 + i for i in range(n)],
            "Confidence_person_id_02": [0.8] * n,
        }
    )


def test_detects_bbox_wide_slot_schema() -> None:
    df = _bbox_wide_slot_df()
    assert detect_input_schema(df) == INPUT_SCHEMA_BBOX_WIDE_SLOT


def test_extract_bbox_wide_slot_produces_one_row_per_slot_per_frame() -> None:
    df = _bbox_wide_slot_df(n=3)
    long_df = extract_long_detections(df, INPUT_SCHEMA_BBOX_WIDE_SLOT)
    assert len(long_df) == 6  # 2 slots x 3 frames
    assert set(long_df["raw_slot"]) == {"person_id_01", "person_id_02"}
    row = long_df.loc[(long_df["frame"] == 0) & (long_df["raw_slot"] == "person_id_01")].iloc[0]
    assert row["x1"] == 10 and row["x2"] == 30
    assert row["cx"] == pytest.approx(20.0)
    # Extra per-slot metadata (Confidence, Tracker ID, Label) carried through.
    assert row["Confidence"] == pytest.approx(0.9)
    assert row["Tracker ID"] == 1
    assert row["Label"] == "person"


def test_detects_point_row_schema() -> None:
    df = pd.DataFrame({"frame": [0, 1], "p1_x": [1.0, 2.0], "p1_y": [3.0, 4.0]})
    assert detect_input_schema(df) == INPUT_SCHEMA_POINT_ROW


def test_extract_point_row() -> None:
    df = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "p1_x": [10.0, 11.0, 12.0],
            "p1_y": [20.0, 21.0, 22.0],
            "p2_x": [100.0, 101.0, None],
            "p2_y": [200.0, 201.0, None],
        }
    )
    long_df = extract_long_detections(df, INPUT_SCHEMA_POINT_ROW)
    # p2 has only 2 valid frames (frame 2 is NaN) -> 3 + 2 = 5 rows.
    assert len(long_df) == 5
    assert long_df["x1"].isna().all()  # point-only: no bbox columns
    row = long_df.loc[(long_df["frame"] == 0) & (long_df["raw_slot"] == "p1")].iloc[0]
    assert row["cx"] == 10.0 and row["cy"] == 20.0


@pytest.mark.parametrize(
    "columns,schema",
    [
        (["frame", "id", "x1", "y1", "x2", "y2"], INPUT_SCHEMA_BBOX_ROW_XYXY),
        (["frame", "id", "x", "y", "w", "h"], INPUT_SCHEMA_BBOX_ROW_XYWH),
    ],
)
def test_detects_row_per_detection_bbox_schemas(columns, schema) -> None:
    df = pd.DataFrame({c: [0] for c in columns})
    assert detect_input_schema(df) == schema


def test_extract_bbox_row_xyxy() -> None:
    df = pd.DataFrame(
        {
            "frame": [0, 0, 1],
            "id": [1, 2, 1],
            "x1": [10, 500, 11],
            "y1": [10, 500, 11],
            "x2": [30, 520, 31],
            "y2": [30, 520, 31],
        }
    )
    long_df = extract_long_detections(df, INPUT_SCHEMA_BBOX_ROW_XYXY)
    assert len(long_df) == 3
    assert set(long_df["raw_slot"]) == {"1", "2"}
    assert long_df["orig_index"].tolist() == sorted(long_df["orig_index"].tolist())


def test_extract_bbox_row_xywh_converts_to_xyxy() -> None:
    df = pd.DataFrame({"frame": [0], "id": [1], "x": [10], "y": [20], "w": [5], "h": [8]})
    long_df = extract_long_detections(df, INPUT_SCHEMA_BBOX_ROW_XYWH)
    row = long_df.iloc[0]
    assert (row["x1"], row["y1"], row["x2"], row["y2"]) == (10, 20, 15, 28)


def test_detects_sam_tracks_schema() -> None:
    df = pd.DataFrame(
        {
            "frame": [0],
            "obj_id": [1],
            "x_px": [1.0],
            "y_px": [2.0],
            "w_px": [3.0],
            "h_px": [4.0],
            "score": [0.9],
            "cx_px": [2.5],
            "cy_px": [4.0],
        }
    )
    assert detect_input_schema(df) == INPUT_SCHEMA_SAM_TRACKS


def test_extract_sam_tracks() -> None:
    df = pd.DataFrame(
        {
            "frame": [0, 1],
            "obj_id": [5, 5],
            "x_px": [10.0, 12.0],
            "y_px": [20.0, 22.0],
            "w_px": [4.0, 4.0],
            "h_px": [8.0, 8.0],
            "score": [0.9, 0.9],
            "cx_px": [12.0, 14.0],
            "cy_px": [24.0, 26.0],
        }
    )
    long_df = extract_long_detections(df, INPUT_SCHEMA_SAM_TRACKS)
    assert len(long_df) == 2
    assert long_df.iloc[0]["raw_slot"] == "5"
    assert long_df.iloc[0]["x2"] == pytest.approx(14.0)  # x_px + w_px


def test_unsupported_schema_raises_with_actionable_message() -> None:
    df = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
    with pytest.raises(ValueError, match="Could not auto-detect"):
        detect_input_schema(df)
