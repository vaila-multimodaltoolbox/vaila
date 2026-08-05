"""Synthetic, CPU-only tests for vaila.monocular_dlt_align.

Everything here is built from a KNOWN synthetic camera and a KNOWN body
placement, so each test asserts recovery of a ground truth rather than
agreement with whatever the code currently produces.

The real-data behaviour these back up (measured on the 3-camera COD fixture,
camera c1, 631 frames, 2026-08-05):
  * DLT decomposition reproduced its own 12 control points to 2.35 px mean;
  * the placement fit reached 1.18 px mean / 3.45 px max reprojection;
  * the lowest foot landed +0.050 m from the floor -- a constraint the fit
    never uses, so it independently validates absolute depth;
  * 100 % of markers fell inside the calibrated volume.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

try:
    from vaila.monocular_dlt_align import (
        DEFAULT_PLACEMENT_ORIGIN_MARKERS,
        _placement_origin,
        decompose_dlt3d,
        dlt_project,
        dlt_projection_matrix,
        refine_placement,
        smooth_placement,
        solve_placement_translation,
    )
except ImportError:
    from monocular_dlt_align import (  # ty: ignore[unresolved-import]
        DEFAULT_PLACEMENT_ORIGIN_MARKERS,
        _placement_origin,
        decompose_dlt3d,
        dlt_project,
        dlt_projection_matrix,
        refine_placement,
        smooth_placement,
        solve_placement_translation,
    )

# A real .dlt3d row from the COD fixture (camera c1).
L_REAL = np.array(
    [
        -144.261763,
        18.045030,
        2.272899,
        885.950018,
        -49.440975,
        -34.688465,
        -95.554227,
        650.457952,
        -0.092946,
        -0.065014,
        0.000358,
    ]
)


def _synthetic_dlt(focal=900.0, cx=960.0, cy=540.0, centre=(5.0, 4.0, 1.5)):
    """Build DLT coefficients for a known camera looking at the origin."""
    centre = np.asarray(centre, dtype=np.float64)
    forward = -centre / np.linalg.norm(centre)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.vstack([right, down, forward])  # world -> camera (OpenCV: +y down)
    K = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
    P = K @ np.hstack([R, (-R @ centre).reshape(3, 1)])
    P = P / P[2, 3]
    return np.concatenate([P[0, :], P[1, :], P[2, :3]]), R, centre, K


def _body(n=70, seed=0):
    """A stand-in body: a compact metric point cloud, hips at indices 9/10."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(scale=0.25, size=(n, 3))
    pts[9] = [-0.10, 0.0, 0.0]
    pts[10] = [0.10, 0.0, 0.0]
    return pts


def test_projection_matrix_layout_and_scale():
    P = dlt_projection_matrix(L_REAL)
    assert P.shape == (3, 4)
    assert P[2, 3] == 1.0  # the DLT convention that fixes the homogeneous scale
    np.testing.assert_allclose(P[0], L_REAL[:4])


def test_decompose_recovers_a_known_synthetic_camera():
    L, R_true, C_true, K_true = _synthetic_dlt()
    cam = decompose_dlt3d(L)
    np.testing.assert_allclose(cam.C, C_true, atol=1e-6)
    np.testing.assert_allclose(cam.R, R_true, atol=1e-6)
    assert cam.focal_px == pytest.approx(K_true[0, 0], rel=1e-6)
    assert cam.principal_point[0] == pytest.approx(K_true[0, 2], rel=1e-6)


def test_decomposed_rotation_is_a_proper_rotation():
    cam = decompose_dlt3d(L_REAL)
    np.testing.assert_allclose(cam.R @ cam.R.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(cam.R) == pytest.approx(1.0, abs=1e-9)


def test_real_camera_frame_is_opencv_y_down():
    """A world point 1.5 m HIGHER must land at a SMALLER camera Y, and every
    point in front of the lens must have positive camera Z. That is what makes
    camera -> world a plain rigid transform with no axis flip."""
    cam = decompose_dlt3d(L_REAL)
    low = (np.array([0.0, 0.0, 0.07]) - cam.C) @ cam.R.T
    high = (np.array([0.0, 0.0, 1.60]) - cam.C) @ cam.R.T
    assert high[1] < low[1]
    assert low[2] > 0 and high[2] > 0


def test_projection_round_trips_through_the_decomposition():
    cam = decompose_dlt3d(L_REAL)
    pts = np.array([[0.0, 0.0, 0.07], [2.5, -5.0, 1.6], [1.0, 2.0, 0.9]])
    from_dlt = dlt_project(L_REAL, pts)
    cam_pts = (pts - cam.C) @ cam.R.T
    hom = cam_pts @ cam.K.T
    np.testing.assert_allclose(from_dlt, hom[:, :2] / hom[:, 2:3], atol=1e-6)


def test_solve_translation_recovers_a_known_placement():
    L, R_true, _, _ = _synthetic_dlt()
    shape = _body()
    true_T = np.array([1.2, -0.4, 0.9])
    uv = dlt_project(L, shape + true_T)
    solved = solve_placement_translation(L, shape, uv)
    assert solved is not None
    np.testing.assert_allclose(solved, true_T, atol=1e-6)


def test_solve_translation_ignores_missing_pixels():
    L, _, _, _ = _synthetic_dlt()
    shape = _body()
    true_T = np.array([0.3, 0.7, 1.1])
    uv = dlt_project(L, shape + true_T)
    uv[5:40] = np.nan
    solved = solve_placement_translation(L, shape, uv)
    assert solved is not None
    np.testing.assert_allclose(solved, true_T, atol=1e-6)


def test_solve_translation_returns_none_when_too_few_points():
    L, _, _, _ = _synthetic_dlt()
    shape = _body()
    uv = np.full((70, 2), np.nan)
    uv[:2] = dlt_project(L, shape[:2])
    assert solve_placement_translation(L, shape, uv) is None


def test_refine_recovers_a_known_rotation_and_translation():
    L, _, _, _ = _synthetic_dlt()
    shape = _body()
    true_rot = Rotation.from_euler("xyz", [7.0, -12.0, 20.0], degrees=True)
    true_T = np.array([0.8, -1.1, 1.0])
    uv = dlt_project(L, shape @ true_rot.as_matrix().T + true_T)

    out = refine_placement(L, shape, np.zeros(3), uv)
    assert out is not None
    rotvec, translation = out
    placed = shape @ Rotation.from_rotvec(rotvec).as_matrix().T + translation
    # The pose parameters can differ by a symmetry of the point set; what must
    # be recovered is the placed geometry itself.
    np.testing.assert_allclose(placed, shape @ true_rot.as_matrix().T + true_T, atol=1e-4)
    np.testing.assert_allclose(dlt_project(L, placed), uv, atol=1e-4)


def test_refine_returns_none_when_too_few_points():
    L, _, _, _ = _synthetic_dlt()
    shape = _body()
    uv = np.full((70, 2), np.nan)
    uv[:3] = dlt_project(L, shape[:3])
    assert refine_placement(L, shape, np.zeros(3), uv) is None


def test_smoothing_removes_a_depth_spike_without_moving_the_trend():
    n = 200
    t = np.arange(n)
    translations = np.column_stack([t * 0.01, np.zeros(n), np.ones(n)])
    translations[100, 0] += 0.5  # one-frame depth spike
    rotvecs = np.zeros((n, 3))

    _, smoothed = smooth_placement(rotvecs, translations, cutoff_hz=6.0, fps=120.0)
    spike_before = np.abs(np.diff(translations[:, 0])).max()
    spike_after = np.abs(np.diff(smoothed[:, 0])).max()
    assert spike_after < spike_before / 3
    # the underlying ramp must survive
    assert smoothed[-1, 0] - smoothed[0, 0] == pytest.approx(
        translations[-1, 0] - translations[0, 0], rel=0.1
    )


def test_smoothing_is_a_noop_for_too_few_frames_or_bad_cutoff():
    rotvecs = np.zeros((4, 3))
    translations = np.ones((4, 3))
    for cutoff, fps in ((6.0, 120.0), (100.0, 120.0), (0.0, 120.0)):
        r, tr = smooth_placement(rotvecs, translations, cutoff, fps)
        np.testing.assert_array_equal(tr, translations)
        np.testing.assert_array_equal(r, rotvecs)


def test_smoothing_keeps_quaternions_continuous_across_a_sign_flip():
    """q and -q are the same rotation; without hemisphere alignment the filter
    reads the flip as a 360 deg swing and smears it over neighbouring frames."""
    n = 120
    angles = np.linspace(0, np.pi * 1.5, n)
    rotvecs = np.column_stack([np.zeros(n), np.zeros(n), angles])
    translations = np.zeros((n, 3))
    smoothed_rv, _ = smooth_placement(rotvecs, translations, 10.0, 120.0)
    err = np.degrees(
        (Rotation.from_rotvec(smoothed_rv) * Rotation.from_rotvec(rotvecs).inv()).magnitude()
    )
    assert err.max() < 5.0


def test_placement_origin_prefers_the_requested_markers():
    pts = _body()
    finite = np.ones(len(pts), dtype=bool)
    origin = _placement_origin(pts, finite, DEFAULT_PLACEMENT_ORIGIN_MARKERS)
    np.testing.assert_allclose(origin, np.array([0.0, 0.0, 0.0]), atol=1e-12)


def test_placement_origin_uses_the_surviving_marker_when_one_is_missing():
    """One hip missing still gives a stable body landmark -- better than
    dropping all the way back to the hand-heavy centroid."""
    pts = _body()
    pts[9] = np.nan
    finite = np.isfinite(pts).all(axis=1)
    origin = _placement_origin(pts, finite, DEFAULT_PLACEMENT_ORIGIN_MARKERS)
    np.testing.assert_allclose(origin, pts[10], atol=1e-12)


def test_placement_origin_falls_back_to_centroid_when_all_markers_missing():
    pts = _body()
    pts[9] = np.nan
    pts[10] = np.nan
    finite = np.isfinite(pts).all(axis=1)
    origin = _placement_origin(pts, finite, DEFAULT_PLACEMENT_ORIGIN_MARKERS)
    np.testing.assert_allclose(origin, np.nanmean(pts[finite], axis=0), atol=1e-12)


def test_placement_origin_survives_out_of_range_markers():
    pts = _body()
    finite = np.isfinite(pts).all(axis=1)
    origin = _placement_origin(pts, finite, (999,))
    np.testing.assert_allclose(origin, np.nanmean(pts[finite], axis=0), atol=1e-12)


def test_placement_origin_does_not_change_the_raw_fit():
    """The origin only matters for smoothing -- the fitted world points must be
    identical either way, which is exactly what makes the wrong choice easy to
    miss (see the module docstring's gotcha)."""
    L, _, _, _ = _synthetic_dlt()
    shape = _body()
    true_T = np.array([0.5, 0.2, 1.0])
    world_true = shape + true_T
    uv = dlt_project(L, world_true)

    results = []
    for origin in (np.zeros(3), np.array([3.0, -2.0, 5.0])):
        centred = shape - origin
        T = solve_placement_translation(L, centred, uv)
        results.append(centred + T)
    np.testing.assert_allclose(results[0], results[1], atol=1e-6)
    np.testing.assert_allclose(results[0], world_true, atol=1e-6)
