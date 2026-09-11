"""
Tests for Bidirectional Subspace Tracker and RTS Zero-Phase Smoother.

Verifies sub-pixel accuracy, numerical stability, Huber loss M-estimator
occlusion robustness, zero-phase distortion (Δϕ = 0), and trajectory recovery
on synthetic biomechanical benchmark data.

Author: Prof. Dr. Paulo R. P. Santiago
Update Date: 10 September 2026
Version: 0.3.131
"""

import numpy as np
import pytest

from vaila.tracking import (
    AnchorBlock,
    BidirectionalSubspaceTracker,
    CWNAStateSpace,
    RTSSmoother,
    TrackingResult,
    extract_normalized_patch,
)


def _render_synthetic_target(
    size: tuple[int, int],
    center: tuple[float, float],
    sigma: float = 6.0,
    intensity: float = 200.0,
    bg_noise: float = 5.0,
    seed: int = 42,
) -> np.ndarray:
    """Renders a synthetic image with a textured Gaussian marker target."""
    rng = np.random.default_rng(seed)
    w, h = size
    cx, cy = center
    x = np.arange(w, dtype=np.float64)
    y = np.arange(h, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    # 2D Gaussian profile
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    blob = intensity * np.exp(-0.5 * dist2 / (sigma**2))

    # Add high-frequency texture intrinsic to the marker target
    texture = (
        25.0 * np.sin((xx - cx) * 0.8) * np.cos((yy - cy) * 0.8) * np.exp(-0.5 * dist2 / (sigma**2))
    )

    # Background noise
    noise = rng.normal(0, bg_noise, (h, w))
    img = np.clip(blob + texture + noise + 30.0, 0, 255).astype(np.uint8)
    return img


def test_cwna_state_space_matrices() -> None:
    """Validates dimension, symmetry, and positive definiteness of state-space matrices."""
    fps = 60.0
    cwna = CWNAStateSpace(fps=fps, sigma_a=50.0)

    assert cwna.F.shape == (6, 6)
    assert cwna.Q.shape == (6, 6)
    assert cwna.H.shape == (2, 6)

    # Transition dt terms
    dt = 1.0 / fps
    assert np.isclose(cwna.F[0, 2], dt)
    assert np.isclose(cwna.F[1, 3], dt)
    assert np.isclose(cwna.F[0, 4], 0.5 * dt**2)

    # Q must be symmetric and positive semi-definite
    assert np.allclose(cwna.Q, cwna.Q.T)
    eigvals = np.linalg.eigvalsh(cwna.Q)
    assert np.all(eigvals >= -1e-10)


def test_patch_extraction_and_normalization() -> None:
    """Tests sub-pixel bicubic patch sampling and Z-score normalization."""
    img = _render_synthetic_target((100, 100), (50.0, 50.0))
    p_norm, raw_patch = extract_normalized_patch(img, (50.25, 49.75), patch_size=(31, 31))

    assert p_norm.shape == (31 * 31,)
    assert raw_patch.shape == (31, 31)

    # Z-score: mean must be ~0 and std must be ~1
    assert np.isclose(np.mean(p_norm), 0.0, atol=1e-5)
    assert np.isclose(np.std(p_norm), 1.0, atol=1e-3)


def test_subspace_tracker_calibration() -> None:
    """Tests SVD eigenbasis decomposition across 30 anchor frames."""
    frames = []
    # 30 frames with slight illumination change and jitter
    for i in range(30):
        intensity = 180.0 + 20.0 * np.sin(i / 5.0)
        frames.append(_render_synthetic_target((80, 80), (40.0, 40.0), intensity=intensity, seed=i))

    anchors = [
        AnchorBlock(frame_indices=list(range(0, 10)), points=[(40.0, 40.0)] * 10),
        AnchorBlock(frame_indices=list(range(10, 20)), points=[(40.0, 40.0)] * 10),
        AnchorBlock(frame_indices=list(range(20, 30)), points=[(40.0, 40.0)] * 10),
    ]

    tracker = BidirectionalSubspaceTracker(
        patch_size=(25, 25), n_components=4, energy_threshold=0.95
    )
    tracker.calibrate(frames, anchors)

    assert tracker.is_calibrated
    assert tracker.mean_template is not None
    assert tracker.mean_template.shape == (25 * 25,)
    assert tracker.eigenbasis is not None
    assert tracker.eigenbasis.shape[0] == 25 * 25
    assert tracker.eigenbasis.shape[1] >= 1
    assert tracker.spatial_gradients is not None
    assert tracker.spatial_gradients.shape == (25 * 25, 2)


def test_iclk_huber_subpixel_convergence() -> None:
    """Tests sub-pixel convergence error < 0.08 px for shift (dx=0.35, dy=-0.42)."""
    true_x, true_y = 50.0, 50.0
    img_ref = _render_synthetic_target((100, 100), (true_x, true_y), seed=10)

    # 30 identical templates to calibrate
    anchors = [
        AnchorBlock(frame_indices=list(range(0, 10)), points=[(true_x, true_y)] * 10),
        AnchorBlock(frame_indices=list(range(10, 20)), points=[(true_x, true_y)] * 10),
        AnchorBlock(frame_indices=list(range(20, 30)), points=[(true_x, true_y)] * 10),
    ]
    tracker = BidirectionalSubspaceTracker(patch_size=(31, 31), n_components=4)
    tracker.calibrate([img_ref] * 30, anchors)

    # Frame with known sub-pixel shift
    dx_true, dy_true = 0.35, -0.42
    target_x = true_x + dx_true
    target_y = true_y + dy_true
    img_shifted = _render_synthetic_target((100, 100), (target_x, target_y), seed=11)

    # Initial guess starts at true_x, true_y
    opt_pos, conf, _ = tracker.align_patch(img_shifted, (true_x, true_y))

    err_x = abs(opt_pos[0] - target_x)
    err_y = abs(opt_pos[1] - target_y)
    total_err = np.hypot(err_x, err_y)

    assert total_err < 0.08, f"Sub-pixel error {total_err} exceeds threshold"
    assert conf > 0.90, f"Confidence score {conf} too low for clear target"


def test_iclk_huber_occlusion_rejection() -> None:
    """Tests that Huber M-estimator downweights occluded/outlier pixels without divergence."""
    true_x, true_y = 50.0, 50.0
    img_ref = _render_synthetic_target((100, 100), (true_x, true_y), seed=20)

    anchors = [
        AnchorBlock(frame_indices=list(range(0, 10)), points=[(true_x, true_y)] * 10),
        AnchorBlock(frame_indices=list(range(10, 20)), points=[(true_x, true_y)] * 10),
        AnchorBlock(frame_indices=list(range(20, 30)), points=[(true_x, true_y)] * 10),
    ]
    tracker = BidirectionalSubspaceTracker(patch_size=(31, 31), n_components=4)
    tracker.calibrate([img_ref] * 30, anchors)

    # Create target with sub-pixel shift and 5% severe outlier pixels (salt-and-pepper / specular noise)
    target_x, target_y = true_x + 0.25, true_y - 0.20
    img_occluded = _render_synthetic_target((100, 100), (target_x, target_y), seed=21)
    rng = np.random.default_rng(999)
    outlier_mask = rng.uniform(0, 1, img_occluded.shape) < 0.05
    img_occluded[outlier_mask] = 250

    opt_pos, conf, _ = tracker.align_patch(img_occluded, (true_x, true_y))

    total_err = np.hypot(opt_pos[0] - target_x, opt_pos[1] - target_y)
    assert total_err < 0.08, f"Huber IRLS failed under occlusion, error={total_err}"
    assert conf > 0.80, f"Confidence score {conf} unexpectedly low"


def test_rts_smoother_zero_phase_lag() -> None:
    """Validates that the RTS Smoother guarantees zero-phase distortion (Δϕ = 0).

    A sinusoidal ground truth trajectory x(t) = A * sin(omega * t) is corrupted by
    additive white noise. Cross-correlation between true and smoothed velocity must peak
    at exactly lag = 0.
    """
    fps = 100.0
    dt = 1.0 / fps
    n_frames = 200
    t_arr = np.arange(n_frames) * dt
    omega = 2.0 * np.pi * 1.5  # 1.5 Hz oscillation
    amp = 50.0

    true_x = 100.0 + amp * np.sin(omega * t_arr)
    true_y = 100.0 + amp * np.cos(omega * t_arr)
    true_vx = amp * omega * np.cos(omega * t_arr)

    rng = np.random.default_rng(123)
    meas_noise = rng.normal(0, 1.2, (n_frames, 2))
    measurements = np.column_stack([true_x, true_y]) + meas_noise

    smoother = RTSSmoother(fps=fps, sigma_a=500.0, sigma_manual=0.5, sigma_track_base=1.2)
    smoothed_states, _ = smoother.smooth(measurements)

    smoothed_x = smoothed_states[:, 0]
    smoothed_vx = smoothed_states[:, 2]

    # Position RMSE should be lower than measurement noise
    rmse_meas = np.sqrt(np.mean((measurements[:, 0] - true_x) ** 2))
    rmse_smooth = np.sqrt(np.mean((smoothed_x - true_x) ** 2))
    assert rmse_smooth < rmse_meas

    # Check zero phase delay (Δϕ = 0): compute cross-correlation of velocity
    # Trim edges to ignore initial transient
    v_true_crop = true_vx[20:-20] - np.mean(true_vx[20:-20])
    v_smooth_crop = smoothed_vx[20:-20] - np.mean(smoothed_vx[20:-20])

    corr = np.correlate(v_smooth_crop, v_true_crop, mode="full")
    lag = np.argmax(corr) - (len(v_true_crop) - 1)

    assert lag == 0, f"RTS Smoother suffered non-zero phase delay! Peak lag = {lag}"


def test_full_synthetic_video_tracking_pipeline() -> None:
    """End-to-end benchmark: 35-frame video with moving target, 3 anchor blocks."""
    n_frames = 35
    fps = 60.0
    dt = 1.0 / fps

    # Target moves linearly with slight acceleration: x(t) = 30 + 15*t + 5*t^2
    t_vals = np.arange(n_frames) * dt
    traj_x = 30.0 + 15.0 * t_vals + 5.0 * (t_vals**2)
    traj_y = 40.0 + 10.0 * t_vals - 3.0 * (t_vals**2)

    frames = []
    for i in range(n_frames):
        frames.append(_render_synthetic_target((120, 120), (traj_x[i], traj_y[i]), seed=100 + i))

    # 3 anchor blocks: Start (0..9), Mid (12..21), End (25..34)
    anchors = [
        AnchorBlock(
            frame_indices=list(range(0, 10)), points=[(traj_x[i], traj_y[i]) for i in range(0, 10)]
        ),
        AnchorBlock(
            frame_indices=list(range(12, 22)),
            points=[(traj_x[i], traj_y[i]) for i in range(12, 22)],
        ),
        AnchorBlock(
            frame_indices=list(range(25, 35)),
            points=[(traj_x[i], traj_y[i]) for i in range(25, 35)],
        ),
    ]

    tracker = BidirectionalSubspaceTracker(
        patch_size=(31, 31),
        n_components=4,
        fps=fps,
        sigma_a=50.0,
    )
    tracker.calibrate(frames, anchors)
    result = tracker.process(frames, total_frames=n_frames, anchors=anchors)

    assert isinstance(result, TrackingResult)
    assert result.trajectory.shape == (n_frames, 2)
    assert result.velocities.shape == (n_frames, 2)
    assert result.accelerations.shape == (n_frames, 2)
    assert result.covariances.shape == (n_frames, 6, 6)
    assert result.confidence_scores.shape == (n_frames,)

    # Trajectory RMSE against ground truth must be sub-pixel (< 0.15 px)
    rmse_x = np.sqrt(np.mean((result.trajectory[:, 0] - traj_x) ** 2))
    rmse_y = np.sqrt(np.mean((result.trajectory[:, 1] - traj_y) ** 2))
    total_rmse = np.hypot(rmse_x, rmse_y)

    assert total_rmse < 0.15, f"End-to-end tracking RMSE {total_rmse:.4f} px exceeds sub-pixel bar"
    # Mean confidence should be high (> 0.90)
    assert np.mean(result.confidence_scores) > 0.90


def test_tracking_error_handling_and_edge_cases() -> None:
    """Validates boundary handling, uncalibrated calls, and empty inputs."""
    tracker = BidirectionalSubspaceTracker()

    # align_patch before calibrate must raise RuntimeError
    with pytest.raises(RuntimeError, match="must be calibrated"):
        tracker.align_patch(np.zeros((50, 50), dtype=np.uint8), (25.0, 25.0))

    # calibrate with empty anchors must raise ValueError
    with pytest.raises(ValueError, match="At least one AnchorBlock"):
        tracker.calibrate([], [])

    # process before calibrate must raise RuntimeError
    with pytest.raises(RuntimeError, match="must be calibrated"):
        tracker.process([])

    # RTS smoother with empty measurements
    smoother = RTSSmoother()
    states, covs = smoother.smooth(np.zeros((0, 2)))
    assert states.shape == (0, 6)
    assert covs.shape == (0, 6, 6)

    # RTS smoother with all NaNs
    states_nan, covs_nan = smoother.smooth(np.full((10, 2), np.nan))
    assert states_nan.shape == (10, 6)
    assert covs_nan.shape == (10, 6, 6)
