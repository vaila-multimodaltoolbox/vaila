"""Unit + smoke tests for vaila.planar_geometry_tracker."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vaila.planar_geometry_tracker import (
    has_non_collinear_quad,
    is_collinear_triplet,
    load_measurements_csv,
    load_target_geometry,
    project_points,
    run_planar_geometry_tracker,
    sample_arc_world,
    sample_circle_world,
    solve_frame_homographies,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TATAME_TOML = REPO_ROOT / "vaila" / "models" / "planar_targets" / "tatame_1x1m.toml"
SOCCER_TOML = REPO_ROOT / "vaila" / "models" / "planar_targets" / "soccerfield_broadcast.toml"
TATAME_CSV = REPO_ROOT / "tests" / "interp_geometry" / "tatame_8markers.csv"


# ------------------------------------------------------------------------- #
# Non-collinearity
# ------------------------------------------------------------------------- #


def test_is_collinear_triplet_true_for_colinear_points() -> None:
    p1 = np.array([0.0, 0.0])
    p2 = np.array([1.0, 1.0])
    p3 = np.array([2.0, 2.0])
    assert is_collinear_triplet(p1, p2, p3)


def test_is_collinear_triplet_false_for_triangle() -> None:
    p1 = np.array([0.0, 0.0])
    p2 = np.array([1.0, 0.0])
    p3 = np.array([0.0, 1.0])
    assert not is_collinear_triplet(p1, p2, p3)


def test_has_non_collinear_quad_requires_four_points() -> None:
    world = {0: np.array([0.0, 0.0]), 1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}
    assert not has_non_collinear_quad([0, 1, 2], world)


def test_has_non_collinear_quad_true_for_square() -> None:
    world = {
        0: np.array([0.0, 0.0]),
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 1.0]),
        3: np.array([0.0, 1.0]),
    }
    assert has_non_collinear_quad([0, 1, 2, 3], world)


def test_has_non_collinear_quad_false_when_all_collinear() -> None:
    world = {
        0: np.array([0.0, 0.0]),
        1: np.array([1.0, 0.0]),
        2: np.array([2.0, 0.0]),
        3: np.array([3.0, 0.0]),
    }
    assert not has_non_collinear_quad([0, 1, 2, 3], world)


# ------------------------------------------------------------------------- #
# TOML profile parsing
# ------------------------------------------------------------------------- #


def test_load_target_geometry_tatame() -> None:
    geometry = load_target_geometry(TATAME_TOML)
    assert geometry.name == "tatame_eva_1x1m"
    assert len(geometry.points) == 8
    assert len(geometry.lines) == 8
    assert geometry.points[4].name == "corner_ne"


def test_load_target_geometry_soccer_has_circles_and_arcs() -> None:
    geometry = load_target_geometry(SOCCER_TOML)
    assert len(geometry.points) == 17
    assert len(geometry.circles) == 1
    assert len(geometry.arcs) == 2
    assert geometry.circles[0].name == "center_circle"


# ------------------------------------------------------------------------- #
# Homography fit + reprojection round-trip (identity-ish transform)
# ------------------------------------------------------------------------- #


def test_homography_reprojects_measured_points_with_low_error() -> None:
    geometry = load_target_geometry(TATAME_TOML)
    # Synthetic pixel measurements: world (meters) scaled by 500 px/m and
    # shifted -- an affine map is a valid (degenerate) homography, so a
    # RANSAC fit against it should reproject with ~0 residual error.
    scale, shift = 500.0, np.array([100.0, 50.0])
    measurements: dict[int, dict[int, tuple[float, float]]] = {
        0: {pid: tuple(geometry.world_xy(pid) * scale + shift) for pid in geometry.sorted_ids}
    }
    solved = solve_frame_homographies(geometry, measurements, ransac_thresh=3.0)
    assert 0 in solved
    assert solved[0].method == "ransac"

    world_xy = np.array([geometry.world_xy(pid) for pid in geometry.sorted_ids])
    projected = project_points(solved[0].homography, world_xy)
    expected = world_xy * scale + shift
    assert np.allclose(projected, expected, atol=1e-6)


def test_solve_frame_homographies_on_tatame_fixture_covers_all_frames() -> None:
    geometry = load_target_geometry(TATAME_TOML)
    measurements = load_measurements_csv(TATAME_CSV)
    solved = solve_frame_homographies(geometry, measurements, ransac_thresh=3.0)
    assert set(solved) == set(measurements)
    # p5 is missing at both edges of the fixture (frames 0-9, 224-330) -- those
    # frames must still resolve a homography via affine fallback/propagation,
    # not be dropped.
    assert 0 in solved
    assert 330 in solved


def test_measured_points_reproject_with_small_residual_on_fixture() -> None:
    geometry = load_target_geometry(TATAME_TOML)
    measurements = load_measurements_csv(TATAME_CSV)
    solved = solve_frame_homographies(geometry, measurements, ransac_thresh=3.0)

    world_xy = np.array([geometry.world_xy(pid) for pid in geometry.sorted_ids])
    errors = []
    for frame, obs in measurements.items():
        projected = project_points(solved[frame].homography, world_xy)
        for idx, pid in enumerate(geometry.sorted_ids):
            if pid in obs:
                u_meas, v_meas = obs[pid]
                errors.append(
                    float(np.hypot(u_meas - projected[idx, 0], v_meas - projected[idx, 1]))
                )
    # RANSAC-solved keyframes should reproject their own inliers tightly;
    # affine-fallback/propagated frames may drift more, so check the median
    # rather than the max.
    assert np.median(errors) < 15.0


# ------------------------------------------------------------------------- #
# Circle / arc projection sanity
# ------------------------------------------------------------------------- #


def test_sample_circle_world_passes_through_cardinal_points() -> None:
    geometry = load_target_geometry(SOCCER_TOML)
    circle = geometry.circles[0]
    pts = sample_circle_world(circle, geometry)
    center = geometry.world_xy(circle.center_point)
    # First sample is theta=0 -> (center.x + radius, center.y)
    assert np.allclose(pts[0], center + np.array([circle.radius, 0.0]))
    # All samples must lie exactly `radius` from the center.
    dists = np.linalg.norm(pts - center, axis=1)
    assert np.allclose(dists, circle.radius, atol=1e-9)


def test_sample_circle_projects_as_ellipse_under_identity_homography() -> None:
    geometry = load_target_geometry(SOCCER_TOML)
    circle = geometry.circles[0]
    world = sample_circle_world(circle, geometry)
    projected = project_points(np.eye(3), world)
    assert np.allclose(projected, world)


def test_sample_arc_world_respects_angle_bounds() -> None:
    geometry = load_target_geometry(SOCCER_TOML)
    arc = geometry.arcs[0]
    pts = sample_arc_world(arc, geometry)
    center = geometry.world_xy(arc.center_point)
    start_expected = center + arc.radius * np.array(
        [np.cos(np.radians(arc.angle_start_deg)), np.sin(np.radians(arc.angle_start_deg))]
    )
    end_expected = center + arc.radius * np.array(
        [np.cos(np.radians(arc.angle_end_deg)), np.sin(np.radians(arc.angle_end_deg))]
    )
    assert np.allclose(pts[0], start_expected)
    assert np.allclose(pts[-1], end_expected)


# ------------------------------------------------------------------------- #
# Full pipeline + CLI smoke tests
# ------------------------------------------------------------------------- #


def test_run_planar_geometry_tracker_produces_all_outputs_no_nans(tmp_path: Path) -> None:
    written = run_planar_geometry_tracker(
        TATAME_TOML,
        TATAME_CSV,
        tmp_path,
        ransac_thresh=3.0,
    )
    assert set(written) == {"imputed_csv", "long_csv", "homographies_npz", "ref3d", "ref3d_map"}
    for path in written.values():
        assert path.exists()

    imputed = pd.read_csv(written["imputed_csv"])
    assert len(imputed) == 331
    assert not imputed.isna().to_numpy().any()

    npz = np.load(written["homographies_npz"])
    assert npz["H"].shape == (331, 3, 3)
    assert npz["frame_ids"].shape == (331,)

    ref3d = pd.read_csv(written["ref3d"])
    assert len(ref3d) == 1
    assert "p1_x" in ref3d.columns


def test_cli_smoke_run(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_out"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vaila.planar_geometry_tracker",
            "--config",
            str(TATAME_TOML),
            "--measurements-csv",
            str(TATAME_CSV),
            "--output-dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / "tatame_8markers_imputed.csv").exists()
    assert (out_dir / "homographies.npz").exists()
    assert (out_dir / "target_calibration.ref3d").exists()


def test_load_target_geometry_missing_points_raises(tmp_path: Path) -> None:
    bad_toml = tmp_path / "empty.toml"
    bad_toml.write_text('[target]\nname = "x"\n')
    with pytest.raises(ValueError, match="no \\[points\\]"):
        load_target_geometry(bad_toml)
