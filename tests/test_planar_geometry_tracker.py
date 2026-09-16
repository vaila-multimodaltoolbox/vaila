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
    # shifted — DLT2D / projective map should reproject with ~0 residual.
    scale, shift = 500.0, np.array([100.0, 50.0])
    measurements: dict[int, dict[int, tuple[float, float]]] = {
        0: {pid: tuple(geometry.world_xy(pid) * scale + shift) for pid in geometry.sorted_ids}
    }
    solved = solve_frame_homographies(geometry, measurements, ransac_thresh=3.0)
    assert 0 in solved
    assert solved[0].method in ("dlt", "topology")

    world_xy = np.array([geometry.world_xy(pid) for pid in geometry.sorted_ids])
    projected = project_points(solved[0].homography, world_xy)
    expected = world_xy * scale + shift
    assert np.allclose(projected, expected, atol=1e-4)


def test_solve_frame_homographies_on_tatame_fixture_covers_all_frames() -> None:
    geometry = load_target_geometry(TATAME_TOML)
    measurements = load_measurements_csv(TATAME_CSV)
    solved = solve_frame_homographies(geometry, measurements, ransac_thresh=3.0)
    assert set(solved) == set(measurements)
    # p5 is missing at both edges of the fixture (frames 0-9, 224-330) — those
    # frames must still resolve via topology midpoint / temporal DLT.
    assert 0 in solved
    assert 330 in solved


def test_measured_points_reproject_with_small_residual_on_fixture() -> None:
    """DLT residual is diagnostic; output uses full reprojection of TOML geometry."""
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
    assert set(written) == {
        "imputed_csv",
        "long_csv",
        "homographies_npz",
        "ref3d",
        "ref3d_map",
        "animation_html",
    }
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


# ------------------------------------------------------------------------- #
# Session helpers (Geo Homog wizard)
# ------------------------------------------------------------------------- #


def test_write_rectangle_toml_round_trip(tmp_path: Path) -> None:
    from vaila.planar_geometry_tracker import write_rectangle_toml

    path = tmp_path / "rect.toml"
    geometry = write_rectangle_toml(path, 2.0, 1.5, name="mat")
    assert path.is_file()
    loaded = load_target_geometry(path)
    assert loaded.name == "mat"
    assert len(loaded.points) == 4
    assert loaded.points[1].x == pytest.approx(2.0)
    assert loaded.points[2].y == pytest.approx(1.5)
    assert geometry.lines == [(0, 1), (1, 2), (2, 3), (3, 0)]


def test_scale_target_geometry_bounding_box() -> None:
    from vaila.planar_geometry_tracker import geometry_bounding_box, scale_target_geometry

    geometry = load_target_geometry(TATAME_TOML)
    scaled = scale_target_geometry(geometry, 1.92, 1.92)
    min_x, min_y, max_x, max_y = geometry_bounding_box(scaled)
    assert max_x - min_x == pytest.approx(1.92)
    assert max_y - min_y == pytest.approx(1.92)
    # Relative midpoint of south edge stays mid-width.
    assert scaled.points[1].x == pytest.approx(0.96)


def test_remap_measurements_csv_then_solve(tmp_path: Path) -> None:
    from vaila.planar_geometry_tracker import (
        make_rectangle_geometry,
        remap_measurements_csv,
        write_target_geometry_toml,
    )

    # Source CSV uses marker columns 10..13; map them onto geom ids 0..3.
    frames = 5
    rows = []
    for f in range(frames):
        row = {"frame": f}
        # Pixel square that matches a 1x1m rectangle under scale=100, shift=(50,20)
        corners = [(50.0, 20.0), (150.0, 20.0), (150.0, 120.0), (50.0, 120.0)]
        for i, (u, v) in enumerate(corners):
            row[f"p{10 + i}_x"] = u
            row[f"p{10 + i}_y"] = v
        rows.append(row)
    src = tmp_path / "src_markers.csv"
    pd.DataFrame(rows).to_csv(src, index=False)

    mapping = {10: 0, 11: 1, 12: 2, 13: 3}
    dst = tmp_path / "session_measurements.csv"
    remap_measurements_csv(src, mapping, dst)
    remapped = pd.read_csv(dst)
    assert list(remapped.columns) == [
        "frame",
        "p0_x",
        "p0_y",
        "p1_x",
        "p1_y",
        "p2_x",
        "p2_y",
        "p3_x",
        "p3_y",
    ]

    geom_path = tmp_path / "session_geometry.toml"
    geometry = make_rectangle_geometry(1.0, 1.0)
    write_target_geometry_toml(geometry, geom_path)

    written = run_planar_geometry_tracker(geom_path, dst, tmp_path / "out")
    assert written["imputed_csv"].is_file()
    assert written["homographies_npz"].is_file()
    measurements = load_measurements_csv(dst)
    solved = solve_frame_homographies(geometry, measurements, ransac_thresh=3.0)
    assert set(solved) == set(range(frames))
    assert all(s.method in ("dlt", "topology") for s in solved.values())


def test_parse_marker_geom_mapping() -> None:
    from vaila.planar_geometry_tracker import parse_marker_geom_mapping

    assert parse_marker_geom_mapping("0:0, 1:2 ; 3:4") == {0: 0, 1: 2, 3: 4}
    with pytest.raises(ValueError):
        parse_marker_geom_mapping("0=0")


# ------------------------------------------------------------------------- #
# Topology + DLT imputation invariants
# ------------------------------------------------------------------------- #


def test_measured_pixels_never_changed_in_imputed_csv(tmp_path: Path) -> None:
    """Imputed CSV preserves measured cells; missing filled; edges stay collinear."""
    geometry = load_target_geometry(TATAME_TOML)
    # Build a short CSV with p5 missing on frame 0
    scale, shift = 400.0, np.array([50.0, 30.0])
    rows = []
    for f in range(3):
        row: dict[str, float | int] = {"frame": f}
        for pid in geometry.sorted_ids:
            if pid == 5 and f == 0:
                row[f"p{pid}_x"] = float("nan")
                row[f"p{pid}_y"] = float("nan")
            else:
                xy = geometry.world_xy(pid) * scale + shift
                row[f"p{pid}_x"] = float(xy[0])
                row[f"p{pid}_y"] = float(xy[1])
        rows.append(row)
    src = tmp_path / "partial.csv"
    pd.DataFrame(rows).to_csv(src, index=False)

    written = run_planar_geometry_tracker(TATAME_TOML, src, tmp_path / "out")
    assert written["imputed_csv"] != src
    assert written["imputed_csv"].name.endswith("_imputed.csv")
    assert "animation_html" in written and written["animation_html"].is_file()

    src_df = pd.read_csv(src)
    imp_df = pd.read_csv(written["imputed_csv"])
    # Noise-free: measured cells match DLT reprojection
    for col in src_df.columns:
        if col == "frame" or not (col.endswith("_x") or col.endswith("_y")):
            continue
        for i in range(len(src_df)):
            s = src_df.at[i, col]
            if pd.isna(s):
                continue
            assert float(imp_df.at[i, col]) == pytest.approx(float(s), abs=1e-4)

    # South edge p0-p1-p2 collinear on every frame
    for i in range(len(imp_df)):
        a = np.array([imp_df.at[i, "p0_x"], imp_df.at[i, "p0_y"]], dtype=float)
        b = np.array([imp_df.at[i, "p1_x"], imp_df.at[i, "p1_y"]], dtype=float)
        c = np.array([imp_df.at[i, "p2_x"], imp_df.at[i, "p2_y"]], dtype=float)
        ba, bc = a - b, c - b
        ang = float(
            np.degrees(
                np.arccos(np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)), -1, 1))
            )
        )
        assert ang == pytest.approx(180.0, abs=0.05)


def test_topology_midpoint_imputes_p5_from_p4_p6() -> None:
    from vaila.planar_geometry_tracker import build_topology_index, resolve_frame_pixels

    geometry = load_target_geometry(TATAME_TOML)
    topo = build_topology_index(geometry)
    assert 5 in topo.metric_midpoints
    assert topo.metric_midpoints[5] == (4, 6)

    obs = {
        0: (100.0, 100.0),
        1: (200.0, 100.0),
        2: (300.0, 100.0),
        3: (300.0, 200.0),
        4: (300.0, 300.0),
        6: (100.0, 300.0),
        7: (100.0, 200.0),
    }
    # Without temporal prior, missing mid is filled by DLT (affine → exact mid)
    resolved, _dlt, method = resolve_frame_pixels(geometry, obs, topo=topo)
    assert 5 in resolved
    assert resolved[5][0] == pytest.approx(200.0, abs=1.0)
    assert resolved[5][1] == pytest.approx(300.0, abs=1.0)
    assert method in ("topology", "dlt", "dlt_after_topology", "temporal_gap")
    for pid, xy in obs.items():
        assert resolved[pid] == xy

    # Temporal prior wins over perspective-wrong image midpoints
    resolved2, _dlt2, method2 = resolve_frame_pixels(
        geometry, obs, topo=topo, temporal_guess={5: (201.0, 299.0)}
    )
    assert resolved2[5] == (201.0, 299.0)
    assert method2 == "temporal_gap"


def test_line_intersection_corner() -> None:
    from vaila.planar_geometry_tracker import (
        build_topology_index,
        line_intersection_2d,
        resolve_frame_pixels,
    )

    # Unit: axis-aligned lines meet at (10, 0)
    hit = line_intersection_2d(
        np.array([0.0, 0.0]),
        np.array([10.0, 0.0]),
        np.array([10.0, -5.0]),
        np.array([10.0, 5.0]),
    )
    assert hit is not None
    assert hit[0] == pytest.approx(10.0)
    assert hit[1] == pytest.approx(0.0)

    # Tatame: drop corner_ne (4); edges through mids/corners still define the
    # east and north lines, so DLT / topology recovers p4.
    g = load_target_geometry(TATAME_TOML)
    topo_t = build_topology_index(g)
    scale, shift = 100.0, np.array([0.0, 0.0])
    full = {pid: tuple(g.world_xy(pid) * scale + shift) for pid in g.sorted_ids}
    obs2 = {pid: xy for pid, xy in full.items() if pid != 4}
    resolved, _dlt, method = resolve_frame_pixels(g, obs2, topo=topo_t)
    assert 4 in resolved
    expected = full[4]
    assert resolved[4][0] == pytest.approx(expected[0], abs=1.0)
    assert resolved[4][1] == pytest.approx(expected[1], abs=1.0)
    assert method in ("topology", "dlt", "dlt_after_topology")
    for pid, xy in obs2.items():
        assert resolved[pid][0] == pytest.approx(xy[0], abs=1.0)
        assert resolved[pid][1] == pytest.approx(xy[1], abs=1.0)


def test_dlt_per_frame_uses_visible_only() -> None:
    from vaila.planar_geometry_tracker import fit_dlt2d_visible, resolve_frame_pixels

    geometry = load_target_geometry(TATAME_TOML)
    scale, shift = 250.0, np.array([40.0, 20.0])
    # 5 of 8 visible (drop 5,6,7)
    visible = [0, 1, 2, 3, 4]
    obs = {pid: tuple(geometry.world_xy(pid) * scale + shift) for pid in visible}
    params = fit_dlt2d_visible(geometry, obs)
    assert params is not None
    assert params.size == 8

    resolved, dlt, method = resolve_frame_pixels(geometry, obs)
    assert dlt is not None
    assert method in ("dlt", "topology", "dlt_after_topology")
    # All 8 points present after resolve
    assert set(resolved) == set(geometry.sorted_ids)
    # Visible match DLT reprojection (exact for noise-free affine)
    for pid in visible:
        assert resolved[pid][0] == pytest.approx(obs[pid][0], abs=1e-4)
        assert resolved[pid][1] == pytest.approx(obs[pid][1], abs=1e-4)


def test_extra_csv_marker_columns_ignored(tmp_path: Path) -> None:
    """CSV may contain p8+ from other tools; only TOML point IDs are used."""
    geometry = load_target_geometry(TATAME_TOML)
    scale, shift = 200.0, np.array([10.0, 10.0])
    rows = []
    for f in range(2):
        row: dict[str, float | int] = {"frame": f}
        for pid in range(14):
            if pid in geometry.points:
                xy = geometry.world_xy(pid) * scale + shift
                row[f"p{pid}_x"] = float(xy[0])
                row[f"p{pid}_y"] = float(xy[1])
            else:
                row[f"p{pid}_x"] = 999.0
                row[f"p{pid}_y"] = 999.0
        rows.append(row)
    src = tmp_path / "extra.csv"
    pd.DataFrame(rows).to_csv(src, index=False)
    written = run_planar_geometry_tracker(TATAME_TOML, src, tmp_path / "out")
    assert written["imputed_csv"].is_file()
    imp = pd.read_csv(written["imputed_csv"])
    assert "p0_x" in imp.columns and "p7_x" in imp.columns


def test_noisy_measured_full_dlt_keeps_south_edge_collinear() -> None:
    """Missing corner via topology stays collinear; no DLT slam into interior."""
    from vaila.planar_geometry_tracker import resolve_frame_pixels

    geometry = load_target_geometry(TATAME_TOML)
    scale, shift = 400.0, np.array([50.0, 30.0])
    # Drop p0; keep clean affine observations on the rest
    obs = {}
    for pid in (1, 2, 3, 4, 5, 6, 7):
        xy = geometry.world_xy(pid) * scale + shift
        obs[pid] = (float(xy[0]), float(xy[1]))
    resolved, dlt, method = resolve_frame_pixels(geometry, obs)
    assert 0 in resolved
    assert method in ("topology", "dlt", "temporal_gap")
    # Measured unchanged
    for pid, xy in obs.items():
        assert resolved[pid] == xy
    a = np.array(resolved[0])
    b = np.array(resolved[1])
    c = np.array(resolved[2])
    ba, bc = a - b, c - b
    ang = float(
        np.degrees(
            np.arccos(np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)), -1, 1))
        )
    )
    assert ang == pytest.approx(180.0, abs=0.05)
    # Topology places p0 near the affine SW corner, not the mat center
    expected = geometry.world_xy(0) * scale + shift
    assert float(np.linalg.norm(a - expected)) < 5.0


def test_missing_midpoint_prefers_temporal_hold_on_stabilized() -> None:
    """Occluded mids must not jump via image midpoint; temporal hold stays put."""
    from vaila.planar_geometry_tracker import resolve_all_frames

    geometry = load_target_geometry(TATAME_TOML)
    scale, shift = 400.0, np.array([50.0, 30.0])
    full = {pid: tuple(geometry.world_xy(pid) * scale + shift) for pid in geometry.sorted_ids}
    # p5 measured only on frames 0 and 5; missing 1..4 should hold/interp near 802-ish
    measurements = {
        0: dict(full),
        1: {pid: xy for pid, xy in full.items() if pid != 5},
        2: {pid: xy for pid, xy in full.items() if pid != 5},
        3: {pid: xy for pid, xy in full.items() if pid != 5},
        4: {pid: xy for pid, xy in full.items() if pid != 5},
        5: dict(full),
    }
    resolved, _solved = resolve_all_frames(geometry, measurements)
    expected = np.array(full[5])
    for f in (1, 2, 3, 4):
        got = np.array(resolved[f][5])
        assert float(np.linalg.norm(got - expected)) < 2.0


def test_missing_corner_no_large_jump_vs_previous_frame() -> None:
    """Occluding p0 must not jump tens of pixels vs the previous measured frame."""
    from vaila.planar_geometry_tracker import resolve_all_frames

    geometry = load_target_geometry(TATAME_TOML)
    scale, shift = 400.0, np.array([50.0, 30.0])
    full = {pid: tuple(geometry.world_xy(pid) * scale + shift) for pid in geometry.sorted_ids}
    measurements = {
        0: dict(full),
        1: {pid: xy for pid, xy in full.items() if pid != 0},
    }
    resolved, solved = resolve_all_frames(geometry, measurements)
    jump = float(
        np.hypot(
            resolved[1][0][0] - resolved[0][0][0],
            resolved[1][0][1] - resolved[0][0][1],
        )
    )
    assert jump < 10.0
    assert solved[1].method in ("topology", "temporal_gap", "dlt")


def test_frame_42_tatame_collinearity_and_continuity() -> None:
    """Frame 42 in tatame fixture must keep p0 collinear with (p1, p2) and (p7, p6)."""
    from vaila.planar_geometry_tracker import resolve_all_frames

    geometry = load_target_geometry(TATAME_TOML)
    fixture_csv = Path(__file__).parent / "video_stabilizer" / "tatame_markers.csv"
    if not fixture_csv.is_file():
        fixture_csv = Path("/home/preto/data/kabuto_teste/tatame_markers.csv")
    if not fixture_csv.is_file():
        pytest.skip("Tatame CSV fixture not available")

    measurements = load_measurements_csv(fixture_csv)
    resolved, _solved = resolve_all_frames(geometry, measurements)

    # In frame 42, p0 is occluded (missing from raw CSV)
    assert 42 in resolved
    p0 = np.array(resolved[42][0])
    p1 = np.array(resolved[42][1])
    p2 = np.array(resolved[42][2])
    p6 = np.array(resolved[42][6])
    p7 = np.array(resolved[42][7])

    # Collinearity along South edge (p0, p1, p2)
    v_south = p2 - p1
    dist_south = float(abs((p2[0] - p1[0]) * (p1[1] - p0[1]) - (p2[1] - p1[1]) * (p1[0] - p0[0])) / np.linalg.norm(v_south))
    assert dist_south < 0.05, f"Expected collinear South edge, got distance {dist_south} px"

    # Collinearity along West edge (p6, p7, p0)
    v_west = p6 - p7
    dist_west = float(abs((p6[0] - p7[0]) * (p7[1] - p0[1]) - (p6[1] - p7[1]) * (p7[0] - p0[0])) / np.linalg.norm(v_west))
    assert dist_west < 0.05, f"Expected collinear West edge, got distance {dist_west} px"

    # Smooth continuity from frame 40 (measured) to 41, 42
    p0_40 = np.array(resolved[40][0])
    p0_41 = np.array(resolved[41][0])
    step_40_41 = float(np.linalg.norm(p0_41 - p0_40))
    step_41_42 = float(np.linalg.norm(p0 - p0_41))
    assert step_40_41 < 10.0
    assert step_41_42 < 10.0


def test_html_animation_frame_counting_and_keys(tmp_path: Path) -> None:
    """HTML animation slider must show consistent 0-based frame counting and key listener."""
    from vaila.planar_geometry_tracker import resolve_all_frames, write_geometry_animation_html

    geometry = load_target_geometry(TATAME_TOML)
    scale, shift = 400.0, np.array([50.0, 30.0])
    full = {pid: tuple(geometry.world_xy(pid) * scale + shift) for pid in geometry.sorted_ids}
    measurements = {
        0: dict(full),
        1: dict(full),
    }
    resolved, solved = resolve_all_frames(geometry, measurements)
    out_html = tmp_path / "test_anim.html"
    write_geometry_animation_html(out_html, geometry, resolved, solved)
    assert out_html.is_file()
    content = out_html.read_text(encoding="utf-8")
    assert "Frame: ' + fr.frame + ' (' + fr.frame + '/" in content
    assert "Total: ' + DATA.frames.length + ' frames" in content
    assert "ArrowLeft" in content and "ArrowRight" in content


def test_html_animation_isometric_square_aspect_ratio(tmp_path: Path) -> None:
    """HTML animation canvas must preserve 1:1 isometric square scaling without distortion."""
    from vaila.planar_geometry_tracker import resolve_all_frames, write_geometry_animation_html

    geometry = load_target_geometry(TATAME_TOML)
    scale, shift = 400.0, np.array([50.0, 30.0])
    full = {pid: tuple(geometry.world_xy(pid) * scale + shift) for pid in geometry.sorted_ids}
    measurements = {0: dict(full)}
    resolved, solved = resolve_all_frames(geometry, measurements)
    out_html = tmp_path / "test_isometric.html"
    write_geometry_animation_html(out_html, geometry, resolved, solved)
    content = out_html.read_text(encoding="utf-8")
    # Must have equal width and height on canvas and uniform scale
    assert 'width="600" height="600"' in content
    assert "Math.min(availW / b.spanX, availH / b.spanY)" in content
    assert "worldBounds = bounds(DATA.ideal_world)" in content


def test_direct_dlt_reliable_filter_rejects_one_sided_clusters() -> None:
    """DLT reliability filter rejects frames missing an entire edge while accepting well-spread frames."""
    from vaila.planar_geometry_tracker import (
        _is_direct_dlt_reliable,
        build_topology_index,
        fit_dlt2d_visible,
    )

    geometry = load_target_geometry(TATAME_TOML)
    topo = build_topology_index(geometry)
    scale, shift = 500.0, np.array([100.0, 100.0])
    # Full clean frame
    clean_obs = {pid: tuple(geometry.world_xy(pid) * scale + shift) for pid in geometry.sorted_ids}
    dlt_clean = fit_dlt2d_visible(geometry, clean_obs)
    assert dlt_clean is not None
    assert _is_direct_dlt_reliable(geometry, clean_obs, dlt_clean, topo) is True

    # Clustered frame: missing both west corners (0 and 6)
    one_sided_obs = {p: clean_obs[p] for p in [1, 2, 3, 4, 5, 7]}
    dlt_one_sided = fit_dlt2d_visible(geometry, one_sided_obs)
    assert dlt_one_sided is not None
    # West edge has only point 7 (missing corners 0 and 6), so should be rejected
    assert _is_direct_dlt_reliable(geometry, one_sided_obs, dlt_one_sided, topo) is False


