"""Tests for geometric marker ReID helpers."""

import numpy as np
import pandas as pd

from vaila.reid_markers import (
    detect_markers_dynamic,
    estimate_max_ids,
    extract_long_detections,
    geometric_reid_align_markers,
    load_homography_matrix,
    merge_fragmented_ids_geometric,
    normalize_marker_input,
    sam_tracks_to_marker_points,
    write_bbox_wide_slot_output,
)


def test_geometric_reid_keeps_identity_through_column_swap() -> None:
    df = pd.DataFrame(
        {
            "frame": [0, 1, 2, 3],
            "p1_x": [0.0, 1.0, 8.0, 3.0],
            "p1_y": [0.0, 0.0, 0.0, 0.0],
            "p2_x": [10.0, 9.0, 2.0, 7.0],
            "p2_y": [0.0, 0.0, 0.0, 0.0],
        }
    )
    markers = detect_markers_dynamic(df, ["p1_x", "p1_y", "p2_x", "p2_y"])

    corrected, stats = geometric_reid_align_markers(
        df,
        markers,
        ["p1", "p2"],
        start_frame=0,
        end_frame=3,
        max_dist=5.0,
        direction_weight=0.5,
    )

    assert corrected.loc[2, "p1_x"] == 2.0
    assert corrected.loc[2, "p2_x"] == 8.0
    assert stats["matches"] >= 4


def test_load_homography_matrix_csv(tmp_path) -> None:
    h_path = tmp_path / "H.csv"
    np.savetxt(h_path, np.eye(3), delimiter=",")

    H = load_homography_matrix(h_path)

    assert H.shape == (3, 3)
    assert np.allclose(H, np.eye(3))


def test_sam_tracks_to_marker_points_uses_sorted_obj_ids_and_foot_point() -> None:
    tracks = pd.DataFrame(
        {
            "frame": [0, 0, 1],
            "obj_id": [7, 3, 7],
            "x_px": [10.0, 100.0, 20.0],
            "y_px": [5.0, 50.0, 6.0],
            "w_px": [4.0, 20.0, 6.0],
            "h_px": [8.0, 30.0, 10.0],
            "score": [0.8, 0.9, 0.7],
            "area_px": [32, 600, 60],
            "n_polygons": [1, 1, 1],
            "largest_polygon_pts": [4, 4, 4],
            "cx_px": [12.0, 110.0, 23.0],
            "cy_px": [9.0, 65.0, 11.0],
        }
    )

    points, id_map = sam_tracks_to_marker_points(tracks)

    assert list(points.columns[:5]) == ["frame", "p1_x", "p1_y", "p1_cx", "p1_cy"]
    assert id_map.to_dict("records") == [
        {"pN": 1, "obj_id": 3, "n_frames": 1, "first_frame": 0, "last_frame": 0},
        {"pN": 2, "obj_id": 7, "n_frames": 2, "first_frame": 0, "last_frame": 1},
    ]
    assert points.loc[0, "p1_x"] == 110.0
    assert points.loc[0, "p1_y"] == 80.0
    assert points.loc[0, "p2_x"] == 12.0
    assert points.loc[0, "p2_y"] == 13.0


def test_normalize_marker_input_prefers_sibling_sam_points(tmp_path) -> None:
    tracks_path = tmp_path / "sam_tracks.csv"
    pd.DataFrame(
        {
            "frame": [0],
            "obj_id": [1],
            "x_px": [10.0],
            "y_px": [20.0],
            "w_px": [2.0],
            "h_px": [4.0],
            "score": [0.5],
            "cx_px": [11.0],
            "cy_px": [22.0],
        }
    ).to_csv(tracks_path, index=False)
    sibling = tmp_path / "sam_points.csv"
    sibling.write_text("frame,p1_x,p1_y\n0,123,456\n", encoding="utf-8")

    df = pd.read_csv(tracks_path)
    out_df, out_path = normalize_marker_input(df, str(tracks_path))

    assert out_path == str(sibling)
    assert out_df.loc[0, "p1_x"] == 123


# =============================================================================
# Geometric ReID v2: max_ids-bounded merge engine (hand-computed ground truth)
# =============================================================================


def _bbox_wide_slot_df(**slots: dict) -> pd.DataFrame:
    """Build a minimal bbox_wide_slot dataframe.

    ``slots`` maps a slot label to ``{"frames": [...], "x1": [...], "y1": [...]}``
    (20px square boxes, one row per listed frame; other frames get NaN).
    """
    all_frames = sorted({f for spec in slots.values() for f in spec["frames"]})
    df = pd.DataFrame({"Frame": all_frames})
    for slot, spec in slots.items():
        by_frame = dict(zip(spec["frames"], zip(spec["x1"], spec["y1"], strict=True), strict=True))
        x1 = [by_frame.get(f, (np.nan, np.nan))[0] for f in all_frames]
        y1 = [by_frame.get(f, (np.nan, np.nan))[1] for f in all_frames]
        df[f"X_min_{slot}"] = x1
        df[f"Y_min_{slot}"] = y1
        df[f"X_max_{slot}"] = [v + 20 if pd.notna(v) else v for v in x1]
        df[f"Y_max_{slot}"] = [v + 20 if pd.notna(v) else v for v in y1]
    return df


def test_merge_keeps_two_well_separated_simultaneous_ids_apart() -> None:
    """Hand-computed ground truth: 2 raw ids, always >400px apart, always
    co-occurring -> must NEVER merge into 1, regardless of max_ids."""
    df = _bbox_wide_slot_df(
        a={"frames": range(5), "x1": [10, 11, 12, 13, 14], "y1": [10] * 5},
        b={"frames": range(5), "x1": [500, 501, 502, 503, 504], "y1": [500] * 5},
    )
    from vaila.reid_markers import detect_input_schema

    schema = detect_input_schema(df)
    long_df = extract_long_detections(df, schema)
    merged, stats = merge_fragmented_ids_geometric(long_df, max_ids=2)
    assert stats == {
        "frames": 5,
        "raw_ids": 2,
        "stable_ids": 2,
        "forced_reassignments": 0,
        "max_ids": 2,
        "dropped_rows": 0,
    }
    # Each raw slot keeps ONE stable id across all its frames (no flapping).
    for slot in ("a", "b"):
        ids = merged.loc[merged["raw_slot"] == slot, "stable_id"]
        assert ids.nunique() == 1


def test_merge_consolidates_id_switch_into_one_stable_trajectory() -> None:
    """Hand-computed ground truth: raw slot 'a' moves right frames 0-2, then
    (simulating an occlusion-driven ID switch) raw slot 'b' continues the
    identical trajectory frames 3-5. A single real subject -> must merge to
    exactly 1 stable id with max_ids=1, matching the real yolov26track
    16693-frame fixture's own fragmentation pattern at small scale."""
    df = _bbox_wide_slot_df(
        a={"frames": [0, 1, 2], "x1": [100, 110, 120], "y1": [200, 200, 200]},
        b={"frames": [3, 4, 5], "x1": [130, 140, 150], "y1": [200, 200, 200]},
    )
    from vaila.reid_markers import detect_input_schema

    schema = detect_input_schema(df)
    long_df = extract_long_detections(df, schema)
    merged, stats = merge_fragmented_ids_geometric(long_df, max_ids=1, max_gap=3, max_dist=50.0)
    assert stats["raw_ids"] == 2
    assert stats["stable_ids"] == 1
    assert stats["dropped_rows"] == 0
    assert merged["stable_id"].nunique() == 1


def test_merge_never_drops_a_row_when_max_ids_covers_true_peak_concurrency() -> None:
    """3 simultaneous well-separated ids, max_ids=3 (== true peak) -> every
    row keeps a stable id, nothing forced/dropped."""
    df = _bbox_wide_slot_df(
        a={"frames": range(3), "x1": [10, 11, 12], "y1": [10] * 3},
        b={"frames": range(3), "x1": [500, 501, 502], "y1": [500] * 3},
        c={"frames": range(3), "x1": [1000, 1001, 1002], "y1": [1000] * 3},
    )
    from vaila.reid_markers import detect_input_schema

    schema = detect_input_schema(df)
    long_df = extract_long_detections(df, schema)
    assert estimate_max_ids(long_df) == 3
    merged, stats = merge_fragmented_ids_geometric(long_df, max_ids=3)
    assert stats["stable_ids"] == 3
    assert stats["forced_reassignments"] == 0
    assert stats["dropped_rows"] == 0
    assert len(merged) == len(long_df)  # every input row survives


def test_merge_documents_forced_reassignment_when_max_ids_too_tight() -> None:
    """3 simultaneous well-separated ids but max_ids=2 (< true peak) is an
    unsatisfiable request -- must not silently invent a 3rd id; must report
    the honest, audited cost (forced_reassignments / dropped_rows) instead."""
    df = _bbox_wide_slot_df(
        a={"frames": [0], "x1": [10], "y1": [10]},
        b={"frames": [0], "x1": [500], "y1": [500]},
        c={"frames": [0], "x1": [1000], "y1": [1000]},
    )
    from vaila.reid_markers import detect_input_schema

    schema = detect_input_schema(df)
    long_df = extract_long_detections(df, schema)
    merged, stats = merge_fragmented_ids_geometric(long_df, max_ids=2)
    assert stats["stable_ids"] <= 2
    assert stats["forced_reassignments"] >= 1
    assert stats["dropped_rows"] >= 1  # the unsatisfiable 3rd concurrent slot


def test_merge_supports_point_only_input_via_synthetic_bbox() -> None:
    """point_row (no bbox columns) must still merge via the same engine."""
    from vaila.reid_markers import INPUT_SCHEMA_POINT_ROW

    df = pd.DataFrame(
        {
            "frame": [0, 1, 2, 3, 4],
            "p1_x": [100.0, 110.0, 120.0, np.nan, np.nan],
            "p1_y": [200.0, 200.0, 200.0, np.nan, np.nan],
            "p2_x": [np.nan, np.nan, np.nan, 130.0, 140.0],
            "p2_y": [np.nan, np.nan, np.nan, 200.0, 200.0],
        }
    )
    long_df = extract_long_detections(df, INPUT_SCHEMA_POINT_ROW)
    merged, stats = merge_fragmented_ids_geometric(long_df, max_ids=1, max_gap=3, max_dist=50.0)
    assert stats["stable_ids"] == 1  # same continuous trajectory, id-switched


def test_writer_roundtrip_preserves_all_detection_cells_bounded_to_max_ids() -> None:
    """write_bbox_wide_slot_output: cell count preserved, columns bounded to
    the actual number of stable ids produced (<= max_ids)."""
    from vaila.reid_markers import detect_input_schema

    df = _bbox_wide_slot_df(
        a={"frames": range(4), "x1": [10, 11, 12, 13], "y1": [10] * 4},
        b={"frames": range(4), "x1": [500, 501, 502, 503], "y1": [500] * 4},
    )
    schema = detect_input_schema(df)
    long_df = extract_long_detections(df, schema)
    merged, stats = merge_fragmented_ids_geometric(long_df, max_ids=2)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "out.csv"
        out = write_bbox_wide_slot_output(df, merged, out_path)
        out_read = pd.read_csv(out_path)

    assert stats["stable_ids"] == 2
    slot_cols = [c for c in out.columns if c.startswith("X_min_")]
    assert len(slot_cols) == 2  # bounded to stable_ids actually produced
    # Every original detection cell (8 = 2 slots x 4 frames) is present.
    non_null_bbox_cells = out_read.filter(regex=r"^X_min_").notna().to_numpy().sum()
    assert non_null_bbox_cells == 8
    assert len(out_read) == 4  # Frame range preserved exactly


def test_estimate_max_ids_empty_long_df_returns_one() -> None:
    empty = pd.DataFrame(columns=["frame", "raw_slot", "cx", "cy", "x1", "y1", "x2", "y2"])
    assert estimate_max_ids(empty) == 1


def test_merge_empty_long_df_is_a_clean_noop() -> None:
    empty = pd.DataFrame(columns=["frame", "raw_slot", "cx", "cy", "x1", "y1", "x2", "y2"])
    merged, stats = merge_fragmented_ids_geometric(empty, max_ids=5)
    assert len(merged) == 0
    assert stats["frames"] == 0
    assert stats["stable_ids"] == 0
