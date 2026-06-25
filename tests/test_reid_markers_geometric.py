"""Tests for geometric marker ReID helpers."""

import numpy as np
import pandas as pd

from vaila.reid_markers import (
    detect_markers_dynamic,
    geometric_reid_align_markers,
    load_homography_matrix,
    normalize_marker_input,
    sam_tracks_to_marker_points,
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
