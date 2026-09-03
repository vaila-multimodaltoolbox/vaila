"""Tests for FIFA soccer field model expansion (48 KP) and centered reference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vaila import getpixelvideo as gpv
from vaila import soccerfield_vitruvian_dlt3d as svd

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "vaila" / "models"


def test_fifa_center_reference_exists_and_valid() -> None:
    """Validate soccerfield_ref3d_fifa_center.csv structure and coordinates."""
    path = MODELS_DIR / "soccerfield_ref3d_fifa_center.csv"
    assert path.exists(), f"File missing: {path}"

    df = pd.read_csv(path)
    assert len(df) == 37
    assert set(df.columns) == {"point_name", "point_number", "x", "y", "z"}

    # Validate center point
    center_row = df[df["point_name"] == "center_field"].iloc[0]
    assert float(center_row["x"]) == 0.0
    assert float(center_row["y"]) == 0.0
    assert float(center_row["z"]) == 0.0

    # Validate bounds
    assert np.isclose(df["x"].min(), -52.45)
    assert np.isclose(df["x"].max(), 52.45)
    assert np.isclose(df["y"].min(), -33.95)
    assert np.isclose(df["y"].max(), 33.95)
    assert (df["z"] == 0.0).all()


def test_fifa_dataset_expanded_to_48_points() -> None:
    """Validate soccerfield_ref3d_fifa_dataset.csv 48-keypoint expansion."""
    path = MODELS_DIR / "soccerfield_ref3d_fifa_dataset.csv"
    assert path.exists(), f"File missing: {path}"

    df = pd.read_csv(path)
    assert len(df) == 48
    assert list(df["point_number"].values) == list(range(48))

    # Test flip_idx involution: flip(flip(i)) == i for all 48 keypoints
    flips = df["flip_idx"].tolist()
    assert len(flips) == 48
    for i in range(48):
        mapped = flips[i]
        assert 0 <= mapped < 48
        assert flips[mapped] == i, f"Flip asymmetry for point {i}: flips[{i}]={mapped}, flips[{mapped}]={flips[mapped]}"

    # Verify new 3D points
    by_name = {row["point_name"]: row for _, row in df.iterrows()}

    # Ground posts
    assert np.isclose(by_name["left_goal_bottom_post_base"]["z"], 0.0)
    assert np.isclose(by_name["left_goal_bottom_post_base"]["x"], -52.45)
    assert np.isclose(by_name["left_goal_bottom_post_base"]["y"], -3.66)

    assert np.isclose(by_name["right_goal_top_post_base"]["z"], 0.0)
    assert np.isclose(by_name["right_goal_top_post_base"]["x"], 52.45)
    assert np.isclose(by_name["right_goal_top_post_base"]["y"], 3.66)

    # Crossbar height = 2.44 m
    assert np.isclose(by_name["left_goal_bottom_post_top"]["z"], 2.44)
    assert np.isclose(by_name["left_goal_top_post_top"]["z"], 2.44)
    assert np.isclose(by_name["right_goal_bottom_post_top"]["z"], 2.44)
    assert np.isclose(by_name["right_goal_top_post_top"]["z"], 2.44)

    # Net ground backing (2.0 m depth behind goal line)
    assert np.isclose(by_name["left_goal_net_bottom_ground"]["x"], -54.45)
    assert np.isclose(by_name["left_goal_net_bottom_ground"]["z"], 0.0)
    assert np.isclose(by_name["right_goal_net_bottom_ground"]["x"], 54.45)
    assert np.isclose(by_name["right_goal_net_bottom_ground"]["z"], 0.0)

    # Corner flag height = 1.50 m
    assert np.isclose(by_name["left_corner_flag_top"]["z"], 1.50)
    assert np.isclose(by_name["left_corner_flag_top"]["x"], -52.45)
    assert np.isclose(by_name["left_corner_flag_top"]["y"], 33.95)

    assert np.isclose(by_name["right_corner_flag_bottom"]["z"], 1.50)
    assert np.isclose(by_name["right_corner_flag_bottom"]["x"], 52.45)
    assert np.isclose(by_name["right_corner_flag_bottom"]["y"], -33.95)


def test_getpixelvideo_pitch_guide_loads_all_48_points() -> None:
    """Test getpixelvideo loads all 48 points in FIFA dataset mode."""
    pts, src, flips = gpv.load_pitch_guide_points(prefer_fifa_dataset=True)
    assert len(pts) == 48
    assert len(flips) == 48
    assert pts[0]["point_name"] == "top_left_corner"
    assert pts[47]["point_name"] == "right_corner_flag_bottom"
    assert "soccerfield_ref3d_fifa_dataset.csv" in src


def test_getpixelvideo_roundtrip_and_continue_marking(tmp_path: Path) -> None:
    """Test save and reload with 48 keypoints allows continuing marking."""
    total_frames = 2
    # Frame 0 has points 0..31 marked, 32..47 empty
    coords: dict[int, list[tuple[float | None, float | None]]] = {
        0: [(float(i * 10), float(i * 5)) for i in range(32)] + [(None, None)] * 16,
        1: [(None, None)] * 48,
    }

    dummy_video = tmp_path / "test_match.mp4"
    dummy_video.touch()

    # Save via save_coordinates
    gpv.save_coordinates(
        str(dummy_video),
        coords,
        total_frames=total_frames,
        fixed_keypoints_count=48,
        keypoint_start_idx=0,
        keypoint_index_base=0,
    )

    out_csv = tmp_path / "test_match_markers.csv"
    assert out_csv.exists()

    # Check columns
    df = pd.read_csv(out_csv)
    assert "frame" in df.columns
    assert "p0_x" in df.columns
    assert "p47_x" in df.columns
    assert "p47_y" in df.columns

    # Load via load_marker_csv_df
    loaded_coords, loaded_labels, msg = gpv.load_marker_csv_df(df, total_frames=total_frames)
    assert len(loaded_coords[0]) == 48
    assert loaded_coords[0][0] == (0.0, 0.0)
    assert loaded_coords[0][31] == (310.0, 155.0)
    assert loaded_coords[0][32] == (None, None)

    # Simulate continuing marking: mark point 32
    loaded_coords[0][32] = (500.0, 250.0)

    # Re-save
    gpv.save_coordinates(
        str(dummy_video),
        loaded_coords,
        total_frames=total_frames,
        fixed_keypoints_count=48,
        keypoint_start_idx=0,
        keypoint_index_base=0,
    )

    df2 = pd.read_csv(out_csv)
    assert df2.loc[0, "p32_x"] == 500.0
    assert df2.loc[0, "p32_y"] == 250.0


def test_vitruvian_load_field_reference_supports_expanded() -> None:
    """Validate vitruvian dlt3d load_field_reference with 48 points."""
    ref_path = MODELS_DIR / "soccerfield_ref3d_fifa_dataset.csv"
    ref = svd.load_field_reference(ref_path)
    assert "top_left_corner" in ref
    assert "right_corner_flag_bottom" in ref
    assert "p0" in ref or "p1" in ref
    assert "p48" in ref
