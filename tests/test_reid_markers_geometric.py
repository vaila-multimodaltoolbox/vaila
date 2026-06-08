"""Tests for geometric marker ReID helpers."""

import numpy as np
import pandas as pd

from vaila.reid_markers import (
    detect_markers_dynamic,
    geometric_reid_align_markers,
    load_homography_matrix,
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
