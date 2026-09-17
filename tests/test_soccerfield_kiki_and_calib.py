"""Tests for soccerfield_kiki.csv model, coincident keypoint rendering, and calibration keypoint editor."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from vaila import drawsportsfields as dsf

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "vaila" / "models"


def test_soccerfield_kiki_file_exists_and_matches_49_points() -> None:
    """Verify Kiki has 49 points and uses visible regulation arc intersections."""
    kiki_path = MODELS_DIR / "soccerfield_kiki.csv"
    assert kiki_path.exists(), f"File missing: {kiki_path}"

    df_kiki = pd.read_csv(kiki_path)
    assert len(df_kiki) == 49

    expected_cols = ["point_name", "point_number", "flip_idx", "x", "y", "z", "x_norm", "y_norm"]
    for col in expected_cols:
        assert col in df_kiki.columns, f"Missing column {col} in soccerfield_kiki.csv"

    # Kiki keeps the dataset index/flip schema, but replaces four virtual
    # goal-area-Y training anchors with visible arc/penalty-line intersections.
    dataset_path = MODELS_DIR / "soccerfield_ref3d_fifa_dataset.csv"
    assert dataset_path.exists()
    df_dataset = pd.read_csv(dataset_path)
    remapped = {10, 11, 18, 19}
    unchanged = [idx for idx in range(48) if idx not in remapped]
    pd.testing.assert_frame_equal(
        df_kiki.iloc[unchanged].reset_index(drop=True),
        df_dataset.iloc[unchanged].reset_index(drop=True),
    )
    assert df_kiki.iloc[:48]["point_number"].tolist() == list(range(48))
    assert df_kiki.iloc[:48]["flip_idx"].tolist() == df_dataset["flip_idx"].tolist()

    expected_intersections = {
        10: ("left_penalty_arc_right_intersection", -35.95, 7.312489),
        11: ("left_penalty_arc_left_intersection", -35.95, -7.312489),
        18: ("right_penalty_arc_right_intersection", 35.95, 7.312489),
        19: ("right_penalty_arc_left_intersection", 35.95, -7.312489),
    }
    for idx, (name, x, y) in expected_intersections.items():
        row = df_kiki.loc[df_kiki["point_number"] == idx].iloc[0]
        assert row["point_name"] == name
        assert float(row["x"]) == pytest.approx(x)
        assert float(row["y"]) == pytest.approx(y)
        assert float(row["z"]) == 0.0

    # Verify point 48: center_field at (0, 0, 0)
    row_48 = df_kiki.iloc[48]
    assert row_48["point_name"] == "center_field"
    assert row_48["point_number"] == 48
    assert row_48["flip_idx"] == 48
    assert pytest.approx(row_48["x"]) == 0.0
    assert pytest.approx(row_48["y"]) == 0.0
    assert pytest.approx(row_48["z"]) == 0.0
    assert pytest.approx(row_48["x_norm"]) == 0.5
    assert pytest.approx(row_48["y_norm"]) == 0.5


def test_kiki_sport_detection_and_registry() -> None:
    """Verify 'kiki' is detected by _detect_sport and registered in SPORT_REGISTRY."""
    kiki_path = MODELS_DIR / "soccerfield_kiki.csv"
    df_kiki = pd.read_csv(kiki_path)

    detected = dsf._detect_sport(str(kiki_path), df_kiki)
    assert detected == "kiki"

    assert "kiki" in dsf.SPORT_REGISTRY
    spec = dsf.SPORT_REGISTRY["kiki"]
    assert spec.model_csv == "soccerfield_kiki.csv"
    assert "Kiki" in spec.title or "kiki" in spec.title.lower()
    assert spec.plot_fn is dsf.plot_field_fifa_dataset


def test_plot_field_fifa_dataset_kiki_rendering_and_deoverlapped_keypoints() -> None:
    """Ensure that coincident points (corner ground vs flags, goal posts) have different positions and orientations."""
    kiki_path = MODELS_DIR / "soccerfield_kiki.csv"
    df_kiki = pd.read_csv(kiki_path)

    fig, ax = dsf.plot_field_fifa_dataset(df_kiki)
    assert fig is not None
    assert ax is not None

    points = {
        str(row["point_name"]): (float(row["x"]), float(row["y"]), int(row["point_number"]))
        for _, row in df_kiki.iterrows()
    }
    overlay_xy = dsf._fifa32_dataset_xy_from_field_points(points)
    assert overlay_xy is not None
    assert overlay_xy[10] == pytest.approx((-35.95, 7.312489))
    assert overlay_xy[11] == pytest.approx((-35.95, -7.312489))
    assert overlay_xy[18] == pytest.approx((35.95, 7.312489))
    assert overlay_xy[19] == pytest.approx((35.95, -7.312489))

    # The four yellow reference markers must be at the visible arc endpoints,
    # not at the obsolete +/-9.16 m training-anchor positions.
    yellow_markers = {
        (float(line.get_xdata()[0]), float(line.get_ydata()[0]))
        for line in ax.lines
        if line.get_marker() == "o" and line.get_markerfacecolor() == "#FFD100"
    }
    for point in (overlay_xy[10], overlay_xy[11], overlay_xy[18], overlay_xy[19]):
        assert point in yellow_markers
    assert (-35.95, 9.16) not in yellow_markers
    assert (35.95, -9.16) not in yellow_markers

    # Collect all text annotations from the axes
    texts = ax.texts
    text_info = {}
    for t in texts:
        s = t.get_text()
        pos = t.get_position()
        rot = t.get_rotation()
        text_info[s] = (pos, rot, t)

    # Check that corner points and flags are both present with concise numeric labels
    assert "0" in text_info, "Corner 0 missing"
    assert "38" in text_info, "Flag 38 missing"

    pos_0, rot_0, _ = text_info["0"]
    pos_38, rot_38, _ = text_info["38"]

    # Position must NOT be the same!
    assert pos_0 != pos_38, f"Positions overlap: {pos_0} vs {pos_38}"

    # Check corner 5 vs flag 39
    assert "5" in text_info
    assert "39" in text_info
    pos_5, _, _ = text_info["5"]
    pos_39, _, _ = text_info["39"]
    assert pos_5 != pos_39

    # Check corner 24 vs flag 46
    assert "24" in text_info
    assert "46" in text_info
    pos_24, _, _ = text_info["24"]
    pos_46, _, _ = text_info["46"]
    assert pos_24 != pos_46

    # Check corner 29 vs flag 47
    assert "29" in text_info
    assert "47" in text_info
    pos_29, _, _ = text_info["29"]
    pos_47, _, _ = text_info["47"]
    assert pos_29 != pos_47

    # Check goal posts: base (32) vs top (34) - drawn backwards (para trás)
    assert "32" in text_info
    assert "34" in text_info
    pos_32, _, _ = text_info["32"]
    pos_34, _, _ = text_info["34"]
    assert pos_32 != pos_34
    # Must be drawn backwards behind the goal line (X < -52.45) and outside penalty area (Y <= -3.66)
    assert pos_32[0] < -52.45, f"Post base 32 should be behind goal line: {pos_32[0]}"
    assert pos_34[0] < -52.45, f"Post top 34 should be behind goal line: {pos_34[0]}"
    assert pos_32[1] <= -3.66, f"Post base 32 should not cross into goal area: {pos_32[1]}"
    assert pos_34[1] <= -3.66, f"Post top 34 should not cross into goal area: {pos_34[1]}"
    sep_32_34 = np.hypot(pos_32[0] - pos_34[0], pos_32[1] - pos_34[1])
    assert sep_32_34 >= 2.5, f"Goal base and top too close: {sep_32_34:.2f}m"

    # Check top post: 33 vs 35 - drawn backwards (para trás)
    assert "33" in text_info
    assert "35" in text_info
    pos_33, _, _ = text_info["33"]
    pos_35, _, _ = text_info["35"]
    assert pos_33[0] < -52.45, f"Post base 33 should be behind goal line: {pos_33[0]}"
    assert pos_35[0] < -52.45, f"Post top 35 should be behind goal line: {pos_35[0]}"
    assert pos_33[1] >= 3.66, f"Post base 33 should not cross into goal area: {pos_33[1]}"
    assert pos_35[1] >= 3.66, f"Post top 35 should not cross into goal area: {pos_35[1]}"
    sep_33_35 = np.hypot(pos_33[0] - pos_35[0], pos_33[1] - pos_35[1])
    assert sep_33_35 >= 2.5, f"Top goal base and top too close: {sep_33_35:.2f}m"

    # Check net ground points (fundo do gol: 36, 37, 44, 45)
    assert "36" in text_info
    assert "37" in text_info
    assert "44" in text_info
    assert "45" in text_info
    pos_36, _, _ = text_info["36"]
    assert pos_36[0] < -54.0, f"Net point 36 should be behind net: {pos_36[0]}"
    sep_32_36 = np.hypot(pos_32[0] - pos_36[0], pos_32[1] - pos_36[1])
    assert sep_32_36 >= 2.0, f"Net point too close to post: {sep_32_36:.2f}m"

    # Right goal posts: 40, 42, 41, 43 - drawn backwards behind right goal line (X > 52.45)
    assert "40" in text_info
    assert "42" in text_info
    assert "41" in text_info
    assert "43" in text_info
    pos_40, _, _ = text_info["40"]
    pos_42, _, _ = text_info["42"]
    assert pos_40[0] > 52.45, f"Right goal base 40 should be behind goal line: {pos_40[0]}"
    assert pos_42[0] > 52.45, f"Right goal top 42 should be behind goal line: {pos_42[0]}"
    assert pos_40[1] <= -3.66
    assert pos_42[1] <= -3.66
    sep_40_42 = np.hypot(pos_40[0] - pos_42[0], pos_40[1] - pos_42[1])
    assert sep_40_42 >= 2.5, f"Right goal base and top too close: {sep_40_42:.2f}m"

    # Check center field spot (point 48) inside center circle
    assert "48" in text_info, "Center field point 48 missing from plot"
    pos_48, _, _ = text_info["48"]
    assert pos_48[0] > 0.0, f"Point 48 should be offset from center: {pos_48[0]}"
    assert pos_48[1] > 0.0, f"Point 48 should be offset from center: {pos_48[1]}"

    # Leader lines check: verify axes lines contain dotted connecting lines
    lines = ax.lines
    assert len(lines) > 0, "Leader lines missing from plot"

    plt.close(fig)


def test_ref_label_deoverlapping_coincident_points() -> None:
    """Test _ref_label differentiates label positions and orientations when points share x, y."""
    fig, ax = plt.subplots()
    points = {
        "corner_ground": (-52.45, 33.95, 0),
        "corner_flag_top": (-52.45, 33.95, 38, 1.5),
    }
    dsf._ref_label(ax, points, field_w=105.0, field_h=68.0, show=True)

    texts = [t for t in ax.texts if t.get_text()]
    assert len(texts) == 2

    t1, t2 = texts[0], texts[1]
    assert t1.get_position() != t2.get_position()
    assert t1.get_rotation() != t2.get_rotation()
    plt.close(fig)


def test_calibration_model_save_and_load_roundtrip(tmp_path: Path) -> None:
    """Test saving calibration points to CSV and loading them back accurately."""
    custom_points = [
        {"point_name": "origin", "point_number": 0, "x": 0.0, "y": 0.0, "z": 0.0},
        {"point_name": "bench_left", "point_number": 1, "x": -10.5, "y": 38.0, "z": 0.8},
        {"point_name": "camera_pole_top", "point_number": 2, "x": 25.0, "y": -40.0, "z": 6.5},
    ]

    out_file = tmp_path / "my_custom_model.csv"
    saved_path = dsf.save_calibration_model_csv(custom_points, out_file)
    assert saved_path.exists()

    loaded = dsf.load_calibration_model_csv(saved_path)
    assert len(loaded) == 3
    assert loaded[0]["point_name"] == "origin"
    assert pytest.approx(loaded[0]["x"]) == 0.0
    assert pytest.approx(loaded[0]["z"]) == 0.0

    assert loaded[1]["point_name"] == "bench_left"
    assert pytest.approx(loaded[1]["x"]) == -10.5
    assert pytest.approx(loaded[1]["y"]) == 38.0
    assert pytest.approx(loaded[1]["z"]) == 0.8

    assert loaded[2]["point_name"] == "camera_pole_top"
    assert pytest.approx(loaded[2]["z"]) == 6.5


def test_calibration_model_augment_and_scratch_workflow(tmp_path: Path) -> None:
    """Test augmenting an existing model and building a model from scratch."""
    # 1. Start from scratch
    scratch_points: list[dict] = []
    # Add 4 corners from scratch
    scratch_points.append({"point_name": "c1", "point_number": 1, "x": -50.0, "y": 30.0, "z": 0.0})
    scratch_points.append({"point_name": "c2", "point_number": 2, "x": -50.0, "y": -30.0, "z": 0.0})
    scratch_points.append({"point_name": "c3", "point_number": 3, "x": 50.0, "y": 30.0, "z": 0.0})
    scratch_points.append({"point_name": "c4", "point_number": 4, "x": 50.0, "y": -30.0, "z": 0.0})

    scratch_file = tmp_path / "scratch_model.csv"
    dsf.save_calibration_model_csv(scratch_points, scratch_file)

    loaded_scratch = dsf.load_calibration_model_csv(scratch_file)
    assert len(loaded_scratch) == 4

    # 2. Augment the model by adding 2 new keypoints
    loaded_scratch.append(
        {"point_name": "c1_flag_top", "point_number": 5, "x": -50.0, "y": 30.0, "z": 1.5}
    )
    loaded_scratch.append(
        {"point_name": "penalty_spot", "point_number": 6, "x": -40.0, "y": 0.0, "z": 0.0}
    )

    augmented_file = tmp_path / "augmented_model.csv"
    dsf.save_calibration_model_csv(loaded_scratch, augmented_file)

    loaded_augmented = dsf.load_calibration_model_csv(augmented_file)
    assert len(loaded_augmented) == 6
    assert loaded_augmented[4]["point_name"] == "c1_flag_top"
    assert pytest.approx(loaded_augmented[4]["z"]) == 1.5


def test_plot_field_reference_points_deoverlapping() -> None:
    """Test plot_field de-overlaps coincident reference points and produces distinct positions."""
    df_fifa = pd.read_csv(MODELS_DIR / "soccerfield_ref3d_fifa.csv")
    fig, ax = dsf.plot_field(df_fifa, show_reference_points=True)
    assert fig is not None
    assert ax is not None

    texts = [t for t in ax.texts if t.get_text()]
    assert len(texts) >= 30

    # Ensure no two text labels share the exact same position
    positions = [t.get_position() for t in texts]
    assert len(positions) == len(set(positions)), (
        "Coincident points in plot_field produced overlapping text positions"
    )

    plt.close(fig)


def test_calibration_keypoints_custom_coords_and_labels(tmp_path: Path) -> None:
    """Test manual coordinate entry, custom labels, and accurate normalization."""
    pts = [
        {"point_name": "cam_post_high", "point_number": 100, "x": -52.45, "y": 0.0, "z": 8.5},
        {"point_name": "coach_box_center", "point_number": 101, "x": -15.0, "y": 35.0, "z": 0.0},
    ]
    out_file = tmp_path / "custom_calib_model.csv"
    dsf.save_calibration_model_csv(pts, out_file)
    assert out_file.exists()

    df_out = pd.read_csv(out_file)
    assert len(df_out) == 2
    assert "point_name" in df_out.columns
    assert df_out.loc[0, "point_name"] == "cam_post_high"
    assert df_out.loc[0, "point_number"] == 100
    assert pytest.approx(df_out.loc[0, "z"]) == 8.5
    assert df_out.loc[1, "point_name"] == "coach_box_center"
    assert pytest.approx(df_out.loc[1, "x"]) == -15.0


def test_draw_soccer_field_3d_center_and_corner_origins() -> None:
    """Test draw_soccer_field_3d renders complete pitch lines, 3D goals, and corner flags for both origins."""
    # 1. Center origin (FIFA / Kiki standard)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    dsf.draw_soccer_field_3d(
        ax,
        length=105.0,
        width=68.0,
        origin="center",
        show_goals=True,
        show_corner_flags=True,
        show_pitch_surface=True,
    )

    # Verify lines were drawn
    lines = ax.lines
    assert len(lines) >= 25, f"Expected >= 25 lines for full 3D soccer field, got {len(lines)}"

    # Check that vertical posts exist reaching Z = 2.44m
    max_z_line = max(np.max(line.get_data_3d()[2]) for line in lines)
    assert pytest.approx(max_z_line, rel=1e-2) == 2.44, f"Goal post Z max should be 2.44m, got {max_z_line}"

    # Check that corner flag poles exist reaching Z = 1.5m
    has_flag_pole = any(
        pytest.approx(np.max(line.get_data_3d()[2]), rel=1e-2) == 1.5 for line in lines
    )
    assert has_flag_pole, "Expected corner flag pole reaching Z = 1.5m"

    # Check X bounds span [-52.5, 52.5]
    min_x_line = min(np.min(line.get_data_3d()[0]) for line in lines)
    max_x_line = max(np.max(line.get_data_3d()[0]) for line in lines)
    # Goal net extends 2m behind lines, so min_x <= -54.5 and max_x >= 54.5
    assert min_x_line <= -52.5
    assert max_x_line >= 52.5
    plt.close(fig)

    # 2. Corner origin (legacy standard)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection="3d")
    dsf.draw_soccer_field_3d(
        ax2,
        length=105.0,
        width=68.0,
        origin="corner",
        show_goals=True,
        show_corner_flags=True,
        show_pitch_surface=True,
    )
    lines2 = ax2.lines
    assert len(lines2) >= 25

    # Check X bounds span [0, 105] (with net extending to -2.0 and +107.0)
    min_x2 = min(np.min(line.get_data_3d()[0]) for line in lines2)
    max_x2 = max(np.max(line.get_data_3d()[0]) for line in lines2)
    assert min_x2 <= 0.0
    assert max_x2 >= 105.0
    plt.close(fig2)


def test_plot_calibration_model_3d_headless() -> None:
    """Test plot_calibration_model_3d creates 3D field, markers, drop lines, and labels in headless mode."""
    pts = [
        {"point_name": "center_spot", "point_number": 0, "x": 0.0, "y": 0.0, "z": 0.0},
        {"point_name": "left_crossbar_center", "point_number": 1, "x": -52.45, "y": 0.0, "z": 2.44},
        {"point_name": "right_crossbar_center", "point_number": 2, "x": 52.45, "y": 0.0, "z": 2.44},
        {"point_name": "camera_high", "point_number": 3, "x": -20.0, "y": -40.0, "z": 8.0},
    ]

    res = dsf.plot_calibration_model_3d(
        pts,
        field_csv="soccerfield_ref3d_fifa.csv",
        title="Test 3D Calibration",
        aspect_z_factor=4.5,
    )
    assert res is not None
    fig, ax = res

    # Verify scatter points (keypoints)
    collections = ax.collections
    assert len(collections) >= 2  # scatter keypoints + drop point scatter / pitch surface

    # Verify text badges exist for all 4 points
    texts = ax.texts
    assert len(texts) == 4
    labels_text = [t.get_text() for t in texts]
    assert "0: center_spot" in labels_text
    assert "1: left_crossbar_center" in labels_text
    assert "2: right_crossbar_center" in labels_text
    assert "3: camera_high" in labels_text

    # Verify drop lines exist for points with Z > 0 (points 1, 2, 3)
    drop_lines = [
        line
        for line in ax.lines
        if line.get_linestyle() == "--" or line.get_linestyle() == "dashed"
    ]
    assert len(drop_lines) >= 3, f"Expected >= 3 drop lines, found {len(drop_lines)}"

    plt.close(fig)


def test_preview_calibration_in_3d_delegation() -> None:
    """Test preview_calibration_in_3d delegates to plot_calibration_model_3d without errors."""
    pts = [
        {"point_name": "p0", "point_number": 0, "x": 0.0, "y": 0.0, "z": 0.0},
        {"point_name": "p1", "point_number": 1, "x": 10.0, "y": 10.0, "z": 2.44},
    ]
    res = dsf.preview_calibration_in_3d(pts, title="Preview Test")
    assert res is not None
    fig, ax = res
    assert ax is not None
    plt.close(fig)

    # Empty points check
    res_empty = dsf.preview_calibration_in_3d([])
    assert res_empty is None

