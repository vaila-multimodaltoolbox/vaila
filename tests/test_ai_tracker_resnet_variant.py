"""
Tests for the configurable ResNet variant (Part 2) and the frozen ResNet embedding
feeding into the retrained online discriminator (Part 3) of the AI Tracker.

Uses a stubbed DeepFeatureExtractor (no real weights download) so it always runs in CI.

Author: Prof. Dr. Paulo R. P. Santiago
Update Date: 13 September 2026
Version: 0.3.138
"""

from __future__ import annotations

import numpy as np
import pytest

from vaila.tracking import ai_tracker as ait
from vaila.tracking.ai_tracker import AITracker, AITrackerParameters


class _StubExtractor:
    """Fake DeepFeatureExtractor: fixed-size embedding, no torch/model involved."""

    def __init__(self, enabled: bool = True, dim: int = 2048, variant: str = "resnet50") -> None:
        self.enabled = enabled
        self._dim = dim
        self.feature_dim = dim
        self.variant = variant

    def extract_embedding(self, patch: np.ndarray) -> np.ndarray | None:
        if patch.size == 0:
            return None
        rng = np.random.default_rng(int(patch.sum()) % (2**31))
        feat = rng.standard_normal(self._dim).astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-7)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-7))


def _tracker_with_stub_extractor() -> AITracker:
    tracker = AITracker(parameters=AITrackerParameters(use_deep_features=False))
    tracker.params.use_deep_features = True
    tracker.extractor = _StubExtractor()  # ty: ignore[invalid-assignment]
    return tracker


def test_effective_feature_dim_without_deep_features() -> None:
    tracker = AITracker(parameters=AITrackerParameters(use_deep_features=False))
    assert tracker.effective_feature_dim() == ait._FEATURE_DIM


def test_effective_feature_dim_with_deep_features() -> None:
    tracker = _tracker_with_stub_extractor()
    assert tracker.effective_feature_dim() == ait._FEATURE_DIM + 2048


def test_discriminator_feature_length_matches_effective_dim() -> None:
    tracker = _tracker_with_stub_extractor()
    patch = np.zeros((36, 36, 3), dtype=np.uint8)
    patch[:] = 128
    feat = tracker._discriminator_feature(patch, "point")
    assert feat.shape == (ait._FEATURE_DIM + 2048,)


def test_discriminator_feature_falls_back_to_zeros_on_extraction_failure() -> None:
    tracker = _tracker_with_stub_extractor()

    class _FailingExtractor(_StubExtractor):
        def extract_embedding(self, patch: np.ndarray) -> np.ndarray | None:
            raise RuntimeError("boom")

    tracker.extractor = _FailingExtractor()  # ty: ignore[invalid-assignment]
    patch = np.zeros((36, 36, 3), dtype=np.uint8)
    patch[:] = 200
    feat = tracker._discriminator_feature(patch, "point")
    assert feat.shape == (ait._FEATURE_DIM + 2048,)
    # Deep half must be the zero-fallback (last 2048 entries).
    np.testing.assert_array_equal(feat[ait._FEATURE_DIM :], np.zeros(2048, dtype=np.float32))


def test_add_anchor_and_retrain_produce_combined_dim_discriminator() -> None:
    tracker = _tracker_with_stub_extractor()
    frame = (np.random.default_rng(0).random((200, 200, 3)) * 255).astype(np.uint8)
    tracker.add_anchor(frame, (100.0, 100.0), frame_idx=0)
    assert len(tracker.training_feats) >= 2
    for feat in tracker.training_feats:
        assert feat.shape == (ait._FEATURE_DIM + 2048,)

    elapsed_ms = tracker.retrain_online_model()
    assert elapsed_ms >= 0.0
    assert tracker.discriminator_w is not None
    assert tracker.discriminator_w.shape == (ait._FEATURE_DIM + 2048,)


def test_score_patch_discriminator_does_not_raise_with_deep_features() -> None:
    tracker = _tracker_with_stub_extractor()
    frame = (np.random.default_rng(1).random((200, 200, 3)) * 255).astype(np.uint8)
    tracker.add_anchor(frame, (100.0, 100.0), frame_idx=0)
    tracker.retrain_online_model()

    patch = frame[82:118, 82:118]
    score = tracker.score_patch_discriminator(patch)
    assert 0.0 <= score <= 1.0


def test_save_and_load_checkpoint_round_trip_with_deep_features(tmp_path) -> None:
    tracker = _tracker_with_stub_extractor()
    frame = (np.random.default_rng(2).random((200, 200, 3)) * 255).astype(np.uint8)
    tracker.add_anchor(frame, (100.0, 100.0), frame_idx=0)
    tracker.retrain_online_model()

    ckpt_path = tmp_path / "discriminator_test.npz"
    assert tracker.save_checkpoint(ckpt_path) is True

    reloaded = _tracker_with_stub_extractor()
    assert reloaded.load_checkpoint(ckpt_path) is True
    assert reloaded._checkpoint_w is not None
    assert reloaded._checkpoint_w.shape == (ait._FEATURE_DIM + 2048,)


def test_get_available_resnet_checkpoints_variant_filtering(tmp_path, monkeypatch) -> None:
    ai_dir = tmp_path / "ai_tracker"
    ai_dir.mkdir()
    big = b"0" * 10_000_001
    (ai_dir / "resnet50_imagenet.pth").write_bytes(big)
    (ai_dir / "resnet152_imagenet.pth").write_bytes(big)
    # A stray file directly under vaila/models/ (not ai_tracker/) must never be picked up.
    stray_dir = tmp_path
    (stray_dir / "resnet50_stray.pth").write_bytes(big)

    monkeypatch.setattr(ait, "_default_checkpoint_dir", lambda: ai_dir)
    monkeypatch.setattr(ait.Path, "home", staticmethod(lambda: tmp_path / "__no_torch_hub__"))

    ckpts50 = ait.get_available_resnet_checkpoints("resnet50")
    ckpts152 = ait.get_available_resnet_checkpoints("resnet152")

    assert any(p.name == "resnet50_imagenet.pth" for p in ckpts50)
    assert all("resnet152" not in p.name for p in ckpts50)
    assert any(p.name == "resnet152_imagenet.pth" for p in ckpts152)
    assert all(p.name != "resnet50_stray.pth" for p in ckpts50 + ckpts152)


def test_ai_tracker_parameters_resnet_variant_toml_round_trip(tmp_path) -> None:
    toml_path = tmp_path / "track_ai.toml"
    params = AITrackerParameters(resnet_variant="resnet152")
    params.to_toml(toml_path)
    reloaded = AITrackerParameters.from_toml(toml_path)
    assert reloaded.resnet_variant == "resnet152"


def test_ai_tracker_parameters_resnet_variant_invalid_falls_back() -> None:
    params = AITrackerParameters(resnet_variant="resnet9000")
    assert params.resnet_variant == "resnet50"


# --- Part 1-5: multi-backbone cascade (mobilenet_v3_small / efficientnet_b0) ---


def test_backbone_variants_includes_lightweight_variants() -> None:
    assert ait._BACKBONE_VARIANTS == (
        "resnet50",
        "resnet152",
        "mobilenet_v3_small",
        "efficientnet_b0",
    )
    # Backward-compatible alias must still exist and match.
    assert ait._RESNET_VARIANTS == ait._BACKBONE_VARIANTS


def test_get_available_resnet_checkpoints_filters_lightweight_variants(
    tmp_path, monkeypatch
) -> None:
    ai_dir = tmp_path / "ai_tracker"
    ai_dir.mkdir()
    big = b"0" * 2_000_000  # above the 1 MB floor, below the old 10 MB floor
    (ai_dir / "mobilenet_v3_small_imagenet.pth").write_bytes(big)
    (ai_dir / "efficientnet_b0_imagenet.pth").write_bytes(big)

    monkeypatch.setattr(ait, "_default_checkpoint_dir", lambda: ai_dir)
    monkeypatch.setattr(ait.Path, "home", staticmethod(lambda: tmp_path / "__no_torch_hub__"))

    mnv3 = ait.get_available_resnet_checkpoints("mobilenet_v3_small")
    effnet = ait.get_available_resnet_checkpoints("efficientnet_b0")

    assert any(p.name == "mobilenet_v3_small_imagenet.pth" for p in mnv3)
    assert any(p.name == "efficientnet_b0_imagenet.pth" for p in effnet)
    assert all("efficientnet" not in p.name for p in mnv3)


def test_ai_tracker_parameters_fallback_variant_defaults_disabled() -> None:
    params = AITrackerParameters()
    assert params.fallback_variant == ""
    assert params.fallback_threshold == pytest.approx(0.48)


def test_ai_tracker_parameters_fallback_variant_invalid_disables() -> None:
    params = AITrackerParameters(fallback_variant="not_a_backbone")
    assert params.fallback_variant == ""


def test_ai_tracker_parameters_fallback_variant_toml_round_trip(tmp_path) -> None:
    toml_path = tmp_path / "track_ai.toml"
    params = AITrackerParameters(fallback_variant="efficientnet_b0", fallback_threshold=0.55)
    params.to_toml(toml_path)
    reloaded = AITrackerParameters.from_toml(toml_path)
    assert reloaded.fallback_variant == "efficientnet_b0"
    assert reloaded.fallback_threshold == pytest.approx(0.55)


def _tracker_with_dual_stub_extractors(
    fallback_variant: str = "efficientnet_b0",
    fallback_threshold: float = 0.48,
) -> AITracker:
    tracker = AITracker(
        parameters=AITrackerParameters(
            use_deep_features=True,
            fallback_variant=fallback_variant,
            fallback_threshold=fallback_threshold,
        )
    )
    tracker.extractor = _StubExtractor(dim=2048, variant="resnet50")  # ty: ignore[invalid-assignment]
    tracker.anchor_embedding = np.ones(2048, dtype=np.float32) / np.sqrt(2048)
    if fallback_variant:
        tracker.extractor_fallback = _StubExtractor(  # ty: ignore[invalid-assignment]
            dim=1280, variant=fallback_variant
        )
        tracker.anchor_embedding_fallback = np.ones(1280, dtype=np.float32) / np.sqrt(1280)
    return tracker


def test_effective_feature_dim_unaffected_by_fallback_variant() -> None:
    tracker = _tracker_with_dual_stub_extractors()
    # Only the primary (resnet50, 2048-d) extractor feeds the discriminator.
    assert tracker.effective_feature_dim() == ait._FEATURE_DIM + 2048


def test_discriminator_feature_length_unaffected_by_fallback_variant() -> None:
    tracker = _tracker_with_dual_stub_extractors()
    patch = np.full((36, 36, 3), 128, dtype=np.uint8)
    feat = tracker._discriminator_feature(patch, "point")
    assert feat.shape == (ait._FEATURE_DIM + 2048,)


def test_fallback_cascade_not_invoked_when_disabled() -> None:
    tracker = _tracker_with_dual_stub_extractors(fallback_variant="")
    assert tracker.extractor_fallback is None
    assert tracker.anchor_embedding_fallback is None


def test_discriminator_feature_never_touches_fallback_extractor(monkeypatch) -> None:
    """`_discriminator_feature()`/`effective_feature_dim()` must only ever read
    `self.extractor` (primary) -- the fallback extractor exists solely for
    `track_frame()`'s per-frame cascade re-score, never for discriminator training.
    """
    tracker = _tracker_with_dual_stub_extractors(fallback_threshold=0.99)

    calls: list[str] = []
    orig_extract = tracker.extractor_fallback.extract_embedding

    def _tracking_extract(patch: np.ndarray) -> np.ndarray | None:
        calls.append("fallback_called")
        return orig_extract(patch)

    monkeypatch.setattr(tracker.extractor_fallback, "extract_embedding", _tracking_extract)

    patch = np.full((36, 36, 3), 100, dtype=np.uint8)
    feat = tracker._discriminator_feature(patch, "point")
    assert feat.shape == (ait._FEATURE_DIM + 2048,)
    assert calls == []  # fallback never invoked, regardless of fallback_threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
