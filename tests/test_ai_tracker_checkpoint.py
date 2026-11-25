"""
Deterministic round-trip + transfer-learning blend tests for the AI Tracker's
on-disk discriminator checkpoint (`AITracker.save_checkpoint` / `load_checkpoint`).

No video/GT fixtures needed — pure in-memory arrays, always runs.

Also covers the per-feature-space checkpoint key (a Deep checkpoint is never overwritten or
loaded by a tracker of another feature space), the user-chosen AI Track home, the versioned
saves and the bank of human-placed samples.

Author: Prof. Dr. Paulo R. P. Santiago
Update Date: 24 November 2026
Version: 0.4.19
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vaila import ai_tracker_store as store
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


def test_default_checkpoint_path_lives_in_the_chosen_home(_isolate_ai_tracker_home) -> None:
    home = _isolate_ai_tracker_home
    p = default_checkpoint_path()
    assert p == home / "ai_tracker" / "discriminators" / "discriminator_default__classic.npz"
    keyed = default_checkpoint_path("ball", feature_key="resnet50-0123abcd")
    assert keyed.name == "discriminator_ball__resnet50-0123abcd.npz"
    assert keyed.parent == p.parent


def test_default_checkpoint_dir_is_the_resolved_root(_isolate_ai_tracker_home) -> None:
    assert ait._default_checkpoint_dir() == _isolate_ai_tracker_home / "ai_tracker"


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


class _StubExtractor:
    """Frozen-embedding stand-in so the deep path runs without ResNet weights."""

    enabled = True
    feature_dim = 16

    def extract_embedding(self, patch):
        return np.full(self.feature_dim, float(patch.mean()) / 255.0, dtype=np.float32)


def _frame() -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)


def test_toggling_deep_features_mid_session_does_not_break_retraining() -> None:
    """Old (short) and new (long) features used to make np.stack fail:
    'all input arrays must have the same shape'."""
    tr = _tracker()
    frame = _frame()
    tr.add_anchor(frame, (100.0, 100.0), 0)
    tr.retrain_online_model()
    assert tr.discriminator_w is not None and tr.discriminator_w.shape == (ait._FEATURE_DIM,)

    tr.params.use_deep_features = True  # the checkbox is switched on
    tr.extractor = _StubExtractor()
    tr.add_anchor(frame, (150.0, 120.0), 1)
    assert len({f.shape[0] for f in tr.training_feats}) == 2  # the bug's precondition
    tr.retrain_online_model()  # must not raise
    assert tr.discriminator_w.shape == (ait._FEATURE_DIM + _StubExtractor.feature_dim,)
    assert len(tr.training_feats) == len(tr.training_labels) == len(tr.training_patches) > 2

    tr.params.use_deep_features = False  # and off again
    tr.add_anchor(frame, (200.0, 150.0), 2)
    tr.retrain_online_model()
    assert tr.discriminator_w.shape == (ait._FEATURE_DIM,)


def test_constant_feature_dimension_is_left_untouched() -> None:
    tr = _tracker()
    frame = _frame()
    tr.add_anchor(frame, (100.0, 100.0), 0)
    before = [f.copy() for f in tr.training_feats]
    tr.retrain_online_model()
    assert all(np.array_equal(a, b) for a, b in zip(before, tr.training_feats, strict=True))


# --- Defects 1 and 4: one discriminator file per feature space -----------------------------


class _FakeDeep:
    """Enabled extractor with a fixed variant and a real (tmp) weights file."""

    enabled = True

    def __init__(self, weights: Path, variant: str = "resnet50", dim: int = 64) -> None:
        self.variant = variant
        self.weights_path = str(weights)
        self.feature_dim = dim

    def extract_embedding(self, patch):
        v = np.full(self.feature_dim, float(patch.mean()) / 255.0, dtype=np.float32)
        v[0] += float(patch.std()) / 255.0
        return v


def _deep_tracker(weights: Path, variant: str = "resnet50", dim: int = 64) -> AITracker:
    tr = AITracker(parameters=AITrackerParameters(use_deep_features=False))
    tr.params.use_deep_features = True
    tr.extractor = _FakeDeep(weights, variant, dim)  # ty: ignore[invalid-assignment]
    return tr


def _teach(tr: AITracker, n_points: int = 3) -> None:
    frame = _frame()
    for i in range(n_points):
        tr.add_anchor(frame, (60.0 + 40 * i, 100.0), i)
    tr.retrain_online_model()


def _weights(tmp_path: Path, name: str, content: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(content)
    return p


def test_deep_off_persist_does_not_overwrite_deep_checkpoint(tmp_path, _isolate_ai_tracker_home):
    """Defect 1: Deep ON and Deep OFF used the same file; the OFF save replaced the Deep one."""
    home = _isolate_ai_tracker_home
    deep = _deep_tracker(_weights(tmp_path, "w.pth", b"backbone-A" * 50))
    _teach(deep, 3)
    out_deep = deep.persist()
    deep_file = out_deep["checkpoint"]
    assert deep_file is not None and deep_file.is_file()
    before = np.load(deep_file)
    w_before, n_before = before["w"].copy(), float(before["n_samples"])
    assert w_before.shape == (ait._FEATURE_DIM + 64,)

    off = _tracker()
    _teach(off, 2)
    out_off = off.persist()
    off_file = out_off["checkpoint"]
    assert off_file is not None and off_file != deep_file
    assert off_file.name == "discriminator_default__classic.npz"
    assert out_off["key"] == "classic" and out_off["root"] == home / "ai_tracker"

    after = np.load(deep_file)
    assert np.array_equal(after["w"], w_before)
    assert float(after["n_samples"]) == n_before
    assert np.load(off_file)["w"].shape == (ait._FEATURE_DIM,)


def test_checkpoint_of_another_feature_space_is_not_loaded(tmp_path) -> None:
    """Defect 4: same dimension, different backbone weights -> different feature space."""
    a = _deep_tracker(_weights(tmp_path, "a.pth", b"AAAA" * 100))
    b = _deep_tracker(_weights(tmp_path, "b.pth", b"BBBB" * 100))
    assert a.checkpoint_key() != b.checkpoint_key()
    assert a.checkpoint_key().startswith("resnet50-")
    _teach(a, 3)
    path = tmp_path / "discriminator_x.npz"
    assert a.save_checkpoint(path) is True
    assert b.load_checkpoint(path) is False
    assert b._checkpoint_w is None
    twin = _deep_tracker(tmp_path / "a.pth")
    assert twin.load_checkpoint(path) is True


def test_checkpoint_key_classic_when_deep_is_off_or_disabled(tmp_path) -> None:
    assert _tracker().checkpoint_key() == "classic"
    fake = _FakeDeep(_weights(tmp_path, "w.pth", b"x" * 100))
    tr = AITracker(parameters=AITrackerParameters(use_deep_features=False))
    tr.params.use_deep_features = True
    tr.extractor = fake  # ty: ignore[invalid-assignment]
    assert tr.checkpoint_key().startswith("resnet50-")
    fake.enabled = False
    assert tr.checkpoint_key() == "classic"
    fake.enabled = True
    tr.params.use_deep_features = False
    assert tr.checkpoint_key() == "classic"


def test_checkpoint_without_key_field_still_loads_when_shape_matches(tmp_path) -> None:
    """Files written before the key field existed keep working (shape-checked only)."""
    path = tmp_path / "discriminator_old.npz"
    np.savez(
        path,
        w=np.zeros(ait._FEATURE_DIM, dtype=np.float32),
        b=np.float32(0.0),
        n_samples=np.float32(3.0),
        feat_dim=np.int32(ait._FEATURE_DIM),
    )
    assert _tracker().load_checkpoint(path) is True


def test_save_checkpoint_stores_key_and_variant(tmp_path) -> None:
    tr = _deep_tracker(_weights(tmp_path, "w.pth", b"k" * 100), variant="mobilenet_v3_small")
    _teach(tr, 2)
    path = tmp_path / "ai_tracker" / "discriminators" / "d.npz"
    assert tr.save_checkpoint(path) is True
    data = np.load(path)
    assert str(data["key"]) == tr.checkpoint_key()
    assert str(data["variant"]) == "mobilenet_v3_small"


def test_persist_twice_writes_history_and_updates_manifest(tmp_path) -> None:
    root = tmp_path / "root" / "ai_tracker"
    tr = _tracker()
    _teach(tr, 2)
    first = tr.persist(root=root)
    assert first["checkpoint"] == root / "discriminators" / "discriminator_default__classic.npz"
    assert store.history_versions(first["checkpoint"]) == []
    _teach(tr, 1)
    second = tr.persist(root=root)
    assert second["checkpoint"] == first["checkpoint"]
    assert len(store.history_versions(first["checkpoint"])) == 1
    entry = store.read_manifest(root)["files"]["discriminators/discriminator_default__classic.npz"]
    assert entry["key"] == "classic" and entry["feat_dim"] == ait._FEATURE_DIM
    assert entry["n_samples"] > 0 and len(entry["sha256"]) == 64


def test_persist_without_anything_learned_writes_nothing(tmp_path) -> None:
    out = _tracker().persist(root=tmp_path / "ai_tracker")
    assert out["checkpoint"] is None and out["samples"] is None


# --- Sample bank: only human anchors ---------------------------------------------------------


def test_only_human_anchors_enter_the_sample_log() -> None:
    tr = _tracker()
    frame = _frame()
    tr.sample_context = {"video": "clip", "marker": 3}
    tr.set_reference(frame, (160.0, 120.0), frame_idx=5)  # human anchor
    n_after_ref = len(tr.sample_log)
    assert n_after_ref >= 2
    tr.track_frame(frame, (160.0, 120.0))  # a prediction: never recorded
    tr.track_frame(frame, (162.0, 121.0))
    assert len(tr.sample_log) == n_after_ref
    tr.update_from_correction(frame, (170.0, 130.0), frame_idx=9)  # human correction
    assert len(tr.sample_log) > n_after_ref
    labels = {e["label"] for e in tr.sample_log}
    assert labels == {1, -1}
    first = tr.sample_log[0]
    assert first["video"] == "clip" and first["marker"] == 3 and first["frame"] == 5
    bw, bh = tr.params.block_window
    assert first["context"].shape[:2] == (2 * bh, 2 * bw)
    assert first["context"].dtype == np.uint8 and first["context"].shape[2] == 3


def test_record_sample_false_trains_the_model_but_logs_nothing() -> None:
    """Points that no person clicked in this session never enter the sample bank."""
    frame = _frame()

    tr = _tracker()
    tr.set_reference(frame, (160.0, 120.0), frame_idx=1, record_sample=False)
    assert tr.sample_log == []
    assert len(tr.training_feats) >= 2  # the online model still learned from it

    n_feats = len(tr.training_feats)
    tr.update_from_correction(frame, (170.0, 130.0), frame_idx=2, record_sample=False)
    assert tr.sample_log == []
    assert len(tr.training_feats) > n_feats

    n_feats = len(tr.training_feats)
    known = {3: (150.0, 110.0), 4: (180.0, 140.0)}
    assert tr.seed_anchors_from_known(lambda _i: frame, known, record_sample=False) == 2
    assert tr.sample_log == []
    assert len(tr.training_feats) > n_feats


def test_record_sample_defaults_to_true_on_every_entry_point() -> None:
    frame = _frame()
    tr = _tracker()

    tr.set_reference(frame, (160.0, 120.0), frame_idx=1)
    n_ref = len(tr.sample_log)
    assert n_ref >= 2
    tr.update_from_correction(frame, (170.0, 130.0), frame_idx=2)
    n_corr = len(tr.sample_log)
    assert n_corr > n_ref
    tr.seed_anchors_from_known(lambda _i: frame, {3: (150.0, 110.0)})
    assert len(tr.sample_log) > n_corr


def test_context_crop_is_padded_at_the_border() -> None:
    tr = _tracker()
    frame = _frame()
    tr.add_anchor(frame, (2.0, 3.0), 0)
    bw, bh = tr.params.block_window
    pos = next(e for e in tr.sample_log if e["label"] == 1)
    assert pos["context"].shape[:2] == (2 * bh, 2 * bw)
    # replicate padding: the top-left corner of the crop repeats the frame corner pixel
    assert np.array_equal(pos["context"][0, 0], frame[0, 0])


def test_save_samples_is_one_cumulative_file_without_duplicates(tmp_path) -> None:
    tr = _tracker()
    frame = _frame()
    tr.sample_context = {"video": "my clip!", "marker": 1}
    assert tr.save_samples(tmp_path) is None  # empty log
    tr.add_anchor(frame, (100.0, 100.0), 0)
    first = tr.save_samples(tmp_path, "my clip!")
    assert first is not None and first.parent == tmp_path
    n1 = len(store.load_samples(first)["labels"])
    again = tr.save_samples(tmp_path, "my clip!")  # nothing new
    assert again == first and len(store.load_samples(first)["labels"]) == n1
    tr.add_anchor(frame, (200.0, 120.0), 1)
    third = tr.save_samples(tmp_path, "my clip!")
    assert third == first
    assert len(list(tmp_path.glob("*.npz"))) == 1
    data = store.load_samples(first)
    assert len(data["labels"]) > n1
    assert data["patches"].ndim == 4 and data["contexts"].ndim == 4
    assert data["patches"].shape[0] == data["contexts"].shape[0] == data["labels"].shape[0]
    assert set(np.unique(data["labels"]).tolist()) == {-1, 1}
    assert set(data["videos"].tolist()) == {"my clip!"}
    # re-adding the same human point (e.g. gap seeding) is not stored twice
    n_before = len(tr.sample_log)
    tr.add_anchor(frame, (200.0, 120.0), 1)
    assert len(tr.sample_log) == n_before
    assert first.name.endswith("_my_clip_.npz")


def test_sample_log_is_capped_with_a_warning(capsys, monkeypatch) -> None:
    monkeypatch.setattr(ait, "_MAX_SAMPLE_LOG", 5)
    tr = _tracker()
    frame = _frame()
    for i in range(4):
        tr.add_anchor(frame, (60.0 + 20 * i, 100.0), i)
    assert len(tr.sample_log) == 5
    assert "sample" in capsys.readouterr().out.lower()


def test_persist_returns_paths_and_saves_samples(tmp_path) -> None:
    root = tmp_path / "ai_tracker"
    tr = _tracker()
    tr.sample_context = {"video": "vid", "marker": 0}
    _teach(tr, 2)
    out = tr.persist(root=root)
    assert out["samples"] is not None
    assert out["samples"].parent == root / "samples"
    assert out["samples"].name.endswith("_vid.npz")
    assert set(out) == {"checkpoint", "samples", "key", "root"}


# --- Legacy folder, active backbone, weight search -------------------------------------------


def test_legacy_mode_exposes_flat_checkpoint_under_its_key(tmp_path, monkeypatch) -> None:
    """Without a chosen home the flat legacy discriminator is copied (not moved) to the
    sub-folder named after its feature space, and default_checkpoint_path finds it."""
    legacy = tmp_path / "legacy" / "ai_tracker"
    legacy.mkdir(parents=True)
    official = legacy / "resnet50_imagenet.pth"
    official.write_bytes(b"R" * 4000)
    np.savez(
        legacy / "discriminator_default.npz",
        w=np.zeros(ait._FEATURE_DIM + 2048, dtype=np.float32),
        b=np.float32(0.0),
        n_samples=np.float32(600.0),
        feat_dim=np.int32(ait._FEATURE_DIM + 2048),
    )
    monkeypatch.delenv(store.ENV_HOME)
    monkeypatch.setattr(store, "LEGACY_DIR", legacy)
    monkeypatch.setattr(ait, "_LEGACY_MIGRATED", set())
    key = store.feature_key("resnet50", official)
    path = default_checkpoint_path("default", feature_key=key)
    assert path == legacy / "discriminators" / f"discriminator_default__{key}.npz"
    assert path.is_file()
    assert (legacy / "discriminator_default.npz").is_file()  # flat original kept
    assert ait._default_checkpoint_dir() == legacy
    # the official weights are found where they are
    assert ait._ai_tracker_resnet_local_path("resnet50") == official


def test_active_backbone_is_preferred_by_ensure_backbone_weights(tmp_path) -> None:
    root = tmp_path / "home" / "ai_tracker"
    store.layout(root)
    tuned = root / "backbones" / "resnet50_finetuned_20261124_100000.pth"
    tuned.write_bytes(b"T" * 1_200_000)
    assert ait.ensure_backbone_weights("resnet50") != tuned.resolve()  # not active yet
    home = Path(root).parent
    store.set_session_home(home)
    try:
        assert ait.get_available_resnet_checkpoints("resnet50")[0] == tuned.resolve()
        store.set_active_backbone(root, "resnet50", tuned)
        assert ait.ensure_backbone_weights("resnet50") == tuned.resolve()
        assert ait.DeepFeatureExtractor._resolve_weights_path(None, "resnet50") == tuned.resolve()
        explicit = tmp_path / "mine.pth"
        explicit.write_bytes(b"E" * 1_200_000)
        assert ait.ensure_backbone_weights("resnet50", weights_path=explicit) == explicit.resolve()
    finally:
        store.set_session_home(None)


def test_official_weights_still_found_in_legacy_folder(tmp_path) -> None:
    """Even with the home elsewhere, the official ImageNet weights in the checkout are found."""
    official = ait._ai_tracker_resnet_local_path("resnet50")
    if not official.is_file():
        pytest.skip("no local resnet50_imagenet.pth in the checkout")
    assert official.resolve() in ait.get_available_resnet_checkpoints("resnet50")
    assert ait.ensure_backbone_weights("resnet50") is not None
