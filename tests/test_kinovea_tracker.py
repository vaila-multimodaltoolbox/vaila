"""
Tests for Kinovea Block Matching + Deep Feature Embedding (ResNet50) Tracker.

Validates:
1. Parabolic sub-pixel peak refinement accuracy (<0.05 px error).
2. Kinovea 3-tier template update policy against drift.
3. ResNet50 visual feature cosine similarity verification.
4. Bidirectional gap infilling and RTS zero-phase kinematic smoothing on synthetic trajectories.
5. Real-data test on JJ_Kabuto video with sparse keyframe anchors.

Author: Prof. Dr. Paulo R. P. Santiago
Update Date: 10 September 2026
Version: 0.3.134
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pandas as pd
import pytest

from vaila.tracking.ai_tracker import (
    AITracker,
    AITrackerParameters,
    DeepFeatureExtractor,
    KinoveaTrackerParameters,
    infill_and_smooth,
    refine_location_parabola,
)


def _render_synthetic_marker(
    size: tuple[int, int],
    center: tuple[float, float],
    sigma: float = 6.0,
    intensity: float = 220.0,
    bg_noise: float = 3.0,
    seed: int = 42,
) -> np.ndarray:
    """Render a synthetic 3-channel BGR image with a high-contrast textured Gaussian marker."""
    rng = np.random.default_rng(seed)
    w, h = size
    cx, cy = center
    x = np.arange(w, dtype=np.float64)
    y = np.arange(h, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    blob = intensity * np.exp(-0.5 * dist2 / (sigma**2))
    texture = (
        20.0 * np.sin((xx - cx) * 0.7) * np.cos((yy - cy) * 0.7) * np.exp(-0.5 * dist2 / (sigma**2))
    )
    noise = rng.normal(0, bg_noise, (h, w))

    gray = np.clip(blob + texture + noise + 25.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return bgr


class MockVideoSource:
    """Mock cv2.VideoCapture for synthetic video testing."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.idx = 0

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def set(self, prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.idx = int(value)
            return True
        return False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if 0 <= self.idx < len(self.frames):
            frame = self.frames[self.idx]
            self.idx += 1
            return True, frame
        return False, None


def test_parabolic_subpixel_refinement() -> None:
    """Validates that parabolic fitting accurately resolves sub-pixel peak shifts."""
    # Construct a 2D quadratic similarity map with a known sub-pixel peak at (10.35, 10.25)
    # y = -A * (x - x_peak)^2 - B * (y - y_peak)^2 + 1.0
    true_x = 10.35
    true_y = 10.25
    x = np.arange(21, dtype=np.float32)
    y = np.arange(21, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    smap = 1.0 - 0.05 * (xx - true_x) ** 2 - 0.05 * (yy - true_y) ** 2
    max_loc = (10, 10)  # Discrete maximum integer cell
    central_val = float(smap[10, 10])

    ref_x, ref_y = refine_location_parabola(smap, max_loc, central_val)

    assert np.isclose(ref_x, true_x, atol=0.01), f"Expected x={true_x}, got {ref_x}"
    assert np.isclose(ref_y, true_y, atol=0.01), f"Expected y={true_y}, got {ref_y}"


def test_ai_template_update_policy() -> None:
    """Validates adaptive running template update policy: EMA blends on confident matches, freezes on occlusion."""
    img1 = _render_synthetic_marker((150, 150), (75.0, 75.0), seed=1)
    # img2 is slightly shifted
    img2 = _render_synthetic_marker((150, 150), (76.0, 75.0), seed=1)

    params = AITrackerParameters(
        search_window=(80, 80),
        block_window=(30, 30),
        similarity_threshold=0.45,
        template_update_threshold=0.70,
        template_learning_rate=0.12,
        use_deep_features=False,
    )
    tracker = AITracker(params)
    tracker.set_reference(img1, (75.0, 75.0))

    assert tracker.template is not None
    initial_tpl = tracker.template.copy()

    # Step 1: Confident match (score >= 0.70) -> template updates via adaptive EMA blending
    res1 = tracker.track_frame(img2, (75.0, 75.0))
    assert res1.similarity >= 0.70
    assert res1.template_updated, (
        "Template SHOULD update via EMA blending on confident matches to adapt to deformation"
    )
    assert not np.array_equal(tracker.template, initial_tpl)

    # Step 2: Poor match / Occlusion (black image, score < 0.45)
    assert tracker.template is not None
    tpl_before_occlusion = tracker.template.copy()
    img_black = np.zeros_like(img1)
    res_poor = tracker.track_frame(img_black, (75.0, 75.0))
    assert res_poor.similarity < 0.45
    assert not res_poor.template_updated, (
        "Template should NEVER update on tracking failure / occlusion"
    )
    assert np.array_equal(tracker.template, tpl_before_occlusion)


# Backward compatibility alias
test_kinovea_template_update_policy = test_ai_template_update_policy


def test_deep_feature_extractor() -> None:
    """Validates ResNet50 visual feature extractor and cosine similarity."""
    extractor = DeepFeatureExtractor.get_shared()
    if not extractor.enabled:
        pytest.skip("Torch / Torchvision ResNet50 not available.")

    patch1 = _render_synthetic_marker((40, 40), (20.0, 20.0), seed=10)
    patch2 = _render_synthetic_marker((40, 40), (20.5, 20.5), seed=10)
    patch_diff = np.zeros((40, 40, 3), dtype=np.uint8)

    emb1 = extractor.extract_embedding(patch1)
    emb2 = extractor.extract_embedding(patch2)
    emb_diff = extractor.extract_embedding(patch_diff)

    assert emb1 is not None
    assert emb2 is not None
    assert emb1.shape == (2048,)
    assert np.isclose(np.linalg.norm(emb1), 1.0, atol=1e-5)

    sim_high = extractor.cosine_similarity(emb1, emb2)
    sim_low = extractor.cosine_similarity(emb1, emb_diff)

    assert sim_high > 0.80, f"Expected high similarity for similar patches, got {sim_high}"
    assert sim_high > sim_low, "Similar patches must have higher similarity than black image"


def test_bidirectional_gap_infilling_and_rts_smoother() -> None:
    """Validates bidirectional tracking across gaps with RTS zero-phase smoothing on synthetic video."""
    n_frames = 30
    fps = 60.0
    w, h = 200, 200

    # True parabolic trajectory
    t = np.linspace(0, 1.0, n_frames)
    true_x = 50.0 + 80.0 * t
    true_y = 60.0 + 50.0 * np.sin(np.pi * t)

    frames = []
    for i in range(n_frames):
        frm = _render_synthetic_marker((w, h), (true_x[i], true_y[i]), seed=i)
        frames.append(frm)

    cap = MockVideoSource(frames)

    # Sparse keyframes: only anchor at frame 0 and frame 29 (gap of 28 frames)
    known = {0: (float(true_x[0]), float(true_y[0])), 29: (float(true_x[29]), float(true_y[29]))}

    params = KinoveaTrackerParameters(
        search_window=(80, 80),
        block_window=(32, 32),
        use_deep_features=False,  # Test core geometry & blending fast
    )

    result = infill_and_smooth(
        cap=cap,
        total_frames=n_frames,
        known_points=known,
        fps=fps,
        parameters=params,
    )

    assert result.trajectory.shape == (n_frames, 2)
    assert result.velocities.shape == (n_frames, 2)
    assert result.accelerations.shape == (n_frames, 2)

    # Keyframes must match input exactly
    assert np.allclose(result.trajectory[0], known[0])
    assert np.allclose(result.trajectory[29], known[29])

    # Trajectory RMSE against ground truth must be sub-pixel (<1.0 px)
    err_x = np.abs(result.trajectory[:, 0] - true_x)
    err_y = np.abs(result.trajectory[:, 1] - true_y)
    rmse = float(np.sqrt(np.mean(err_x**2 + err_y**2)))

    assert rmse < 1.0, f"Trajectory tracking RMSE must be < 1.0 px, got {rmse:.3f} px"


def test_real_data_jj_kabuto_sample() -> None:
    """Validates tracker on the real vertical jump video /home/preto/data/jjkabuto/JJ_Kabuto.mp4."""
    video_path = "/home/preto/data/jjkabuto/JJ_Kabuto.mp4"
    csv_path = (
        "/home/preto/data/jjkabuto/JJ_Kabuto_markers_corrected.csv"
        if os.path.isfile("/home/preto/data/jjkabuto/JJ_Kabuto_markers_corrected.csv")
        else "/home/preto/data/jjkabuto/JJ_Kabuto_markers.csv"
    )

    if not os.path.isfile(video_path) or not os.path.isfile(csv_path):
        pytest.skip("JJ_Kabuto sample video or markers CSV not found on this machine.")

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Failed to open JJ_Kabuto.mp4"

    df = pd.read_csv(csv_path)
    # Read user markers
    known: dict[int, tuple[float, float]] = {}
    for _, row in df.iterrows():
        f = int(row["frame"])
        if pd.notna(row.get("p0_x")) and pd.notna(row.get("p0_y")):
            known[f] = (float(row["p0_x"]), float(row["p0_y"]))

    assert len(known) > 0, "No valid markers found in JJ_Kabuto markers CSV"

    # Test tracking on the first 50 frames containing gaps
    sub_frames = 50
    sub_known = {f: pt for f, pt in known.items() if f < sub_frames}

    params = AITrackerParameters(
        search_window=(120, 120),
        block_window=(36, 36),
        use_deep_features=True,
    )

    result = infill_and_smooth(
        cap=cap,
        total_frames=sub_frames,
        known_points=sub_known,
        fps=60.0,
        parameters=params,
    )

    cap.release()

    assert not np.isnan(result.trajectory).any(), "Result trajectory contains NaNs!"
    assert result.trajectory.shape == (sub_frames, 2)
    # Velocities and accelerations must be finite
    assert np.all(np.isfinite(result.velocities))
    assert np.all(np.isfinite(result.accelerations))


def test_jj_kabuto_gap_229_290_drift_fixed() -> None:
    """Validates that the drift on frames 231-291 is completely resolved (<2.0 px mean error).

    Before fix: drift exceeded 100+ px jumping onto distractor foot.
    After fix (Gaussian motion prior sigma=22 + adaptive EMA blending):
    mean error is ~1.36 px, max error < 5.0 px.
    """
    video_path = "/home/preto/data/jjkabuto/JJ_Kabuto.mp4"
    csv_path = "/home/preto/data/jjkabuto/JJ_Kabuto_markers_corrected.csv"

    if not os.path.isfile(video_path) or not os.path.isfile(csv_path):
        pytest.skip("JJ_Kabuto sample video or corrected markers CSV not found.")

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Failed to open JJ_Kabuto.mp4"

    df = pd.read_csv(csv_path)
    gt: dict[int, tuple[float, float]] = {}
    for _, row in df.iterrows():
        f = int(row["frame"])
        if pd.notna(row.get("p0_x")) and pd.notna(row.get("p0_y")):
            gt[f] = (float(row["p0_x"]), float(row["p0_y"]))

    assert 229 in gt and 290 in gt, "Ground truth keyframes 229 and 290 missing from corrected CSV"

    # Set anchor reference at frame 229
    cap.set(cv2.CAP_PROP_POS_FRAMES, 229)
    ret_anchor, anchor_img = cap.read()
    assert ret_anchor and anchor_img is not None

    params = AITrackerParameters(
        search_window=(140, 140),
        block_window=(36, 36),
        similarity_threshold=0.45,
        template_update_threshold=0.70,
        spatial_sigma=22.0,
        template_learning_rate=0.12,
        use_mask=True,
        use_deep_features=True,
        deep_weight=0.25,
    )
    tracker = AITracker(parameters=params)
    tracker.set_reference(anchor_img, gt[229])

    # Sequential frame-by-frame live tracking (mirroring getpixelvideo Space playback)
    cur_pt = gt[229]
    errors = []
    for f in range(230, 291):
        ret, frame = cap.read()
        assert ret and frame is not None
        res = tracker.track_frame(frame, cur_pt)
        cur_pt = res.location
        if f in gt:
            true_pt = gt[f]
            err = float(np.hypot(cur_pt[0] - true_pt[0], cur_pt[1] - true_pt[1]))
            errors.append(err)

    cap.release()

    assert len(errors) == 61
    mean_err = float(np.mean(errors))
    max_err = float(np.max(errors))

    print(
        f"\n>> JJ_Kabuto live tracking gap 230-290: "
        f"mean error = {mean_err:.2f} px, max error = {max_err:.2f} px"
    )
    assert mean_err < 2.0, (
        f"Mean error ({mean_err:.2f} px) must be < 2.0 px (previously drifted > 100 px)"
    )
    assert max_err < 5.0, f"Max error ({max_err:.2f} px) must be < 5.0 px"


def test_ai_tracker_online_discriminator() -> None:
    """Validate fast online model (<1ms) retraining and scoring on user anchors."""
    params = AITrackerParameters(use_deep_features=False)
    tracker = AITracker(parameters=params)

    frame1 = _render_synthetic_marker((160, 160), (80.0, 80.0))
    tracker.set_reference(frame1, (80.0, 80.0))

    assert len(tracker.anchors) == 1
    assert tracker.discriminator_w is not None

    # Add second anchor
    frame2 = _render_synthetic_marker((160, 160), (82.0, 81.0))
    tracker.add_anchor(frame2, (82.0, 81.0), frame_idx=10)
    assert len(tracker.anchors) == 2

    # Retrain online model
    tracker.retrain_online_model()
    assert tracker.discriminator_w is not None
    assert tracker.discriminator_w.shape == (768,)

    # Score positive patch vs background patch
    pos_patch = frame2[81 - 18 : 81 + 18, 82 - 18 : 82 + 18]
    bg_patch = frame2[10 : 10 + 36, 10 : 10 + 36]

    score_pos = tracker.score_patch_discriminator(pos_patch)
    score_bg = tracker.score_patch_discriminator(bg_patch)

    assert 0.0 <= score_pos <= 1.0
    assert 0.0 <= score_bg <= 1.0
    assert score_pos > score_bg, f"Positive score ({score_pos:.3f}) should exceed BG ({score_bg:.3f})"


def test_ai_tracker_toml_save_load(tmp_path: os.PathLike) -> None:
    """Validate serializing and deserializing AITrackerParameters to/from .toml."""
    from pathlib import Path

    params = AITrackerParameters(
        search_window=(180, 180),
        block_window=(40, 40),
        similarity_threshold=0.55,
        use_deep_features=True,
        deep_weight=0.35,
    )
    toml_path = Path(tmp_path) / "test_config.toml"
    params.to_toml(str(toml_path))

    assert toml_path.exists()
    content = toml_path.read_text(encoding="utf-8")
    assert "search_window_w = 180" in content
    assert "use_deep_features = true" in content

    loaded = AITrackerParameters.from_toml(str(toml_path))
    assert loaded.search_window == (180, 180)
    assert loaded.block_window == (40, 40)
    assert abs(loaded.similarity_threshold - 0.55) < 1e-5
    assert loaded.use_deep_features is True
    assert abs(loaded.deep_weight - 0.35) < 1e-5
