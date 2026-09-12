"""
Deterministic round-trip + transfer-learning blend tests for the AI Tracker's
on-disk discriminator checkpoint (`AITracker.save_checkpoint` / `load_checkpoint`).

No video/GT fixtures needed — pure in-memory arrays, always runs.

Author: Prof. Dr. Paulo R. P. Santiago
Update Date: 11 September 2026
Version: 0.3.137
"""

from __future__ import annotations

import numpy as np
import pytest

from vaila.tracking import ai_tracker as ait
from vaila.tracking.ai_tracker import AITracker, AITrackerParameters, default_checkpoint_path


def _tracker() -> AITracker:
    return AITracker(parameters=AITrackerParameters(use_deep_features=False))


def test_checkpoint_round_trip_bit_identical(tmp_path) -> None:
    """save_checkpoint -> load_checkpoint must reproduce w/b/n exactly (float32 npz)."""
    rng = np.random.default_rng(42)
    w = rng.standard_normal(ait._FEATURE_DIM).astype(np.float32)

    src = _tracker()
    src.discriminator_w = w
    src.discriminator_b = -0.375
    src._live_n_samples = 17.0

    path = tmp_path / "discriminator_test.npz"
    assert src.save_checkpoint(path) is True
    assert path.is_file()

    dst = _tracker()
    assert dst.load_checkpoint(path) is True
    assert np.array_equal(dst._checkpoint_w, w)
    assert dst._checkpoint_b == pytest.approx(-0.375, abs=1e-6)
    assert dst._checkpoint_n == pytest.approx(17.0, abs=1e-6)


def test_save_checkpoint_without_discriminator_returns_false(tmp_path) -> None:
    """Nothing to persist yet (discriminator_w is None) -> save is a documented no-op."""
    src = _tracker()
    assert src.discriminator_w is None
    assert src.save_checkpoint(tmp_path / "discriminator_test.npz") is False
    assert not (tmp_path / "discriminator_test.npz").exists()


def test_load_missing_checkpoint_returns_false(tmp_path) -> None:
    dst = _tracker()
    assert dst.load_checkpoint(tmp_path / "does_not_exist.npz") is False
    assert dst._checkpoint_w is None


def test_load_incompatible_feat_dim_rejected(tmp_path) -> None:
    """A checkpoint from a different feature extractor/config must be silently ignored,
    never corrupt-merged into the live discriminator (per loop spec scientific-validity gate)."""
    path = tmp_path / "discriminator_bad.npz"
    np.savez(
        path,
        w=np.zeros(64, dtype=np.float32),  # wrong dim (not _FEATURE_DIM=808)
        b=np.float32(0.0),
        n_samples=np.float32(5.0),
        feat_dim=np.int32(64),
    )
    dst = _tracker()
    assert dst.load_checkpoint(path) is False
    assert dst._checkpoint_w is None


def test_default_checkpoint_path_under_models_ai_tracker() -> None:
    p = default_checkpoint_path()
    assert p.name == "discriminator_default.npz"
    assert p.parent.name == "ai_tracker"
    assert p.parent.parent.name == "models"


def test_retrain_blends_checkpoint_by_sample_count_weighted_average() -> None:
    """Core transfer-learning contract: after loading a checkpoint, retrain_online_model()
    must blend session weights with checkpoint weights via
    (n_session*w_session + n_checkpoint*w_checkpoint) / (n_session + n_checkpoint) —
    not simply overwrite, and not ignore the checkpoint."""
    rng = np.random.default_rng(7)
    feat_dim = ait._FEATURE_DIM

    # Build a session-only tracker and capture its plain (unblended) ridge solution.
    baseline = _tracker()
    feats = [rng.standard_normal(feat_dim).astype(np.float32) for _ in range(4)]
    labels = [1.0, 1.0, 0.0, 0.0]
    baseline.training_feats = list(feats)
    baseline.training_labels = list(labels)
    baseline.retrain_online_model()
    w_session = baseline.discriminator_w.copy()
    b_session = baseline.discriminator_b
    n_session = float(len(feats))
    assert baseline.discriminator_w is not None

    # Now a fresh tracker with an injected checkpoint, given the SAME session data.
    checkpoint_w = rng.standard_normal(feat_dim).astype(np.float32)
    checkpoint_b = 0.42
    checkpoint_n = 9.0

    blended = _tracker()
    blended._checkpoint_w = checkpoint_w
    blended._checkpoint_b = checkpoint_b
    blended._checkpoint_n = checkpoint_n
    blended.training_feats = list(feats)
    blended.training_labels = list(labels)
    blended.retrain_online_model()

    total_n = n_session + checkpoint_n
    expected_w = (n_session * w_session + checkpoint_n * checkpoint_w) / total_n
    expected_b = (n_session * b_session + checkpoint_n * checkpoint_b) / total_n

    assert blended.discriminator_w is not None
    np.testing.assert_allclose(blended.discriminator_w, expected_w, rtol=1e-5, atol=1e-6)
    assert blended.discriminator_b == pytest.approx(expected_b, abs=1e-5)
    assert blended._live_n_samples == pytest.approx(total_n, abs=1e-6)

    # And the blended result must differ from the plain session-only solution — proves
    # the checkpoint actually influenced the outcome (transfer learning happened, not a
    # silent no-op).
    assert not np.allclose(blended.discriminator_w, w_session)


def test_get_available_resnet50_checkpoints() -> None:
    """Check that get_available_resnet50_checkpoints returns existing models/ai_tracker weights."""
    from vaila.tracking.ai_tracker import get_available_resnet50_checkpoints

    ckpts = get_available_resnet50_checkpoints()
    assert isinstance(ckpts, list)
    local_ai_tracker = ait._ai_tracker_resnet50_local_path()
    if local_ai_tracker.is_file():
        assert any(c.name == "resnet50_imagenet.pth" for c in ckpts)
        assert any("ai_tracker" in str(c) for c in ckpts)

