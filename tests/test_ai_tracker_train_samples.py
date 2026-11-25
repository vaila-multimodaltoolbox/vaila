"""
Tests for the ``--from-samples`` retrain of vaila.ai_tracker_train (CPU, seconds, no network).

A synthetic sample bank (a bright disc on noise = marker, noise = background) is written
into a temporary AI Track home, a tiny random-initialised MobileNetV3-small plays the
backbone, and the real ``main([...])`` is run. ``tests/conftest.py`` points the home at a
temporary folder, so no real user file is touched.

Author: Paulo Roberto Pereira Santiago
Update Date: 24 November 2026
Version: 0.4.19
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
tv_models = pytest.importorskip("torchvision.models")

from vaila import ai_tracker_store as store  # noqa: E402
from vaila import ai_tracker_train as train  # noqa: E402
from vaila.tracking.ai_tracker import (  # noqa: E402
    AITracker,
    AITrackerParameters,
    DeepFeatureExtractor,
)

VARIANT = "mobilenet_v3_small"
BLOCK = 36


def _feature_tracker(weights: Path, variant: str = VARIANT) -> AITracker:
    """A deep-feature tracker on CPU with the given backbone (what the live tracker loads)."""
    tracker = AITracker(AITrackerParameters(use_deep_features=False, resnet_variant=variant))
    tracker.params.use_deep_features = True
    tracker.extractor = DeepFeatureExtractor(use_cuda=False, weights_path=weights, variant=variant)
    assert tracker.extractor.enabled
    return tracker


def _classic_tracker() -> AITracker:
    return AITracker(AITrackerParameters(use_deep_features=False))


@pytest.fixture(scope="module")
def tiny_weights(tmp_path_factory) -> Path:
    """Random-initialised MobileNetV3-small state_dict (no download)."""
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny") / "mobilenet_v3_small_random.pth"
    torch.save(tv_models.get_model(VARIANT, weights=None).state_dict(), str(path))
    assert path.stat().st_size > 1_000_000
    return path


def _make_mini_detect_dataset(root: Path) -> Path:
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        img[40:120, 60:140] = (40, 180, 40)
        img[130:200, 180:260] = (180, 40, 40)
        cv2.imwrite(str(root / split / "images" / f"{split}_0.jpg"), img)
        (root / split / "labels" / f"{split}_0.txt").write_text(
            "0 0.3125 0.3333 0.25 0.3333\n1 0.6875 0.6875 0.25 0.2917\n", encoding="utf-8"
        )
    (root / "classes.txt").write_text("person\nsports ball\n", encoding="utf-8")
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f"path: .\ntrain: {root / 'train' / 'images'}\nval: {root / 'val' / 'images'}\n"
        "nc: 2\nnames: ['person', 'sports ball']\n",
        encoding="utf-8",
    )
    return yaml_path


# ------------------------------------------------------------------ synthetic sample bank


def _blob(rng: np.random.Generator, size: int, marker: bool) -> np.ndarray:
    """Noise (background) or noise with a bright disc in the middle (marker), BGR uint8."""
    img = rng.integers(0, 120, (size, size, 3), dtype=np.uint8)
    if marker:
        cv2.circle(img, (size // 2, size // 2), size // 4, (255, 255, 255), -1)
    return np.ascontiguousarray(img)


def write_bank(
    root: Path,
    videos: tuple[str, ...] = ("clip_a", "clip_b"),
    frames_per_video: int = 30,
    seed: int = 0,
) -> Path:
    """One samples/*.npz per video in the schema of ``AITracker.save_samples``: at every frame
    one marker patch (+1) and one background patch (-1)."""
    rng = np.random.default_rng(seed)
    folder = root / "samples"
    folder.mkdir(parents=True, exist_ok=True)
    for k, video in enumerate(videos):
        labels, frames, patches, contexts = [], [], [], []
        for f in range(frames_per_video):
            for marker in (True, False):
                labels.append(1 if marker else -1)
                frames.append(f)
                patches.append(_blob(rng, BLOCK, marker))
                contexts.append(_blob(rng, 2 * BLOCK, marker))
        n = len(labels)
        np.savez_compressed(
            folder / f"20261124_10000{k}_{video}.npz",
            patches=np.stack(patches),
            contexts=np.stack(contexts),
            labels=np.array(labels, dtype=np.int8),
            frames=np.array(frames, dtype=np.int32),
            markers=np.zeros(n, dtype=np.int32),
            block_w=np.full(n, BLOCK, dtype=np.int32),
            block_h=np.full(n, BLOCK, dtype=np.int32),
            points=np.full((n, 2), 50.0, dtype=np.float32),
            videos=np.array([video] * n, dtype=str),
            shapes=np.array(["point"] * n, dtype=str),
            session=np.array("20261124_100000"),
        )
    return folder


@pytest.fixture
def root() -> Path:
    path = store.resolve_home().root
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run(root: Path, weights: Path, *extra: str) -> int:
    return train.main(
        [
            "--from-samples",
            str(root),
            "--mode",
            "both",
            "--weights",
            str(weights),
            "--variant",
            VARIANT,
            "--epochs",
            "1",
            "--batch",
            "16",
            "--device",
            "cpu",
            *extra,
        ]
    )


def _snapshot(root: Path) -> list[tuple[str, int]]:
    return sorted((str(p.relative_to(root)), p.stat().st_size) for p in root.rglob("*"))


def _last_retrain(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text())["last_retrain"]


# ------------------------------------------------------------------ defect 3 (YOLO path)


def test_yolo_discriminator_is_loadable_by_the_live_tracker(tmp_path, tiny_weights) -> None:
    """Defect 3: the offline YOLO discriminator used to be a 576/2048-dim file the live
    tracker (808 + deep, keyed by backbone) never loaded."""
    from vaila.ai_tracker_train import TrainConfig, run_training

    yaml_path = _make_mini_detect_dataset(tmp_path / "ds")
    cfg = TrainConfig(
        data_yaml=yaml_path,
        weights=tiny_weights,
        mode="discriminator",
        profile="pytest",
        max_samples=50,
        device="cpu",
    )
    run_training(cfg)

    root = store.resolve_home().root
    deep = _feature_tracker(tiny_weights)
    deep_path = store.discriminator_path(root, "pytest", deep.checkpoint_key())
    assert deep_path.is_file(), "no discriminator written for the backbone feature key"
    assert deep.load_checkpoint(deep_path) is True
    assert _classic_tracker().load_checkpoint(deep_path) is False
    # the deep file is 808 + 576 wide, the classic one 808, both keyed
    assert np.load(deep_path)["w"].shape == (808 + 576,)
    classic_path = store.discriminator_path(root, "pytest", store.CLASSIC_KEY)
    assert np.load(classic_path)["w"].shape == (808,)
    assert _classic_tracker().load_checkpoint(classic_path) is True


# ------------------------------------------------------------------ end to end


def test_from_samples_both_end_to_end(root, tiny_weights, capsys) -> None:
    write_bank(root)
    rc = _run(root, tiny_weights)
    assert rc == 0
    last = _last_retrain(root)
    assert last["status"] in {"activated", "kept_previous"}
    assert last["mode"] == "both" and last["variant"] == VARIANT and last["device"] == "cpu"
    assert last["n_samples"] == 120 and last["n_videos"] == 2
    assert last["metric"] == "heldout_auc"
    assert isinstance(last["before"], float) and isinstance(last["after"], float)
    assert 0.0 <= last["before"] <= 1.0 and 0.0 <= last["after"] <= 1.0
    assert last["started"] <= last["finished"] and last["reason"]
    assert last["backbone"].startswith("backbones/mobilenet_v3_small_finetuned_")
    assert (root / last["backbone"]).is_file()
    assert (root / last["log"]).is_file()
    log_text = (root / last["log"]).read_text()
    assert train.PREFIX in log_text and "Equivalent CLI" in log_text
    assert "--variant mobilenet_v3_small" in log_text and "<auto>" not in log_text
    assert not list(root.glob(".tmp_*"))  # the staging folder is gone
    assert f"saved backbone -> {root / last['backbone']}" in log_text
    out = capsys.readouterr().out
    assert f"{train.PREFIX} hold-out by videos" in out

    # the discriminators are loadable by the live tracker (defect 3), keyed by the backbone
    chosen = root / last["backbone"] if last["status"] == "activated" else tiny_weights
    deep = _feature_tracker(chosen)
    deep_path = store.discriminator_path(root, "default", deep.checkpoint_key())
    classic_path = store.discriminator_path(root, "default", store.CLASSIC_KEY)
    assert deep_path.is_file() and classic_path.is_file()
    assert deep.load_checkpoint(deep_path) is True
    assert _classic_tracker().load_checkpoint(classic_path) is True
    assert _classic_tracker().load_checkpoint(deep_path) is False
    other = tiny_weights if chosen != tiny_weights else root / last["backbone"]
    assert _feature_tracker(other).load_checkpoint(deep_path) is False  # another backbone key
    entry = store.read_manifest(root)["files"][deep_path.relative_to(root).as_posix()]
    assert entry["kind"] == "discriminator" and entry["key"] == deep.checkpoint_key()
    assert entry["n_samples"] == 120


@pytest.mark.parametrize(
    ("aucs", "status", "active"),
    [
        ((0.60, 0.80), "activated", True),
        ((0.70, 0.70), "activated", True),  # not worse is enough
        ((0.80, 0.60), "kept_previous", False),
        ((None, 0.60), "kept_previous", False),  # no validation signal: do not activate
    ],
)
def test_activation_rule(root, tiny_weights, monkeypatch, aucs, status, active) -> None:
    write_bank(root, frames_per_video=12)
    calls = iter(aucs)
    monkeypatch.setattr(train, "_heldout_auc", lambda *a, **k: next(calls))
    assert _run(root, tiny_weights, "--min-samples", "10") == 0
    last = _last_retrain(root)
    assert last["status"] == status
    assert last["before"] == aucs[0] and last["after"] == aucs[1]
    new_file = root / last["backbone"]
    assert new_file.is_file()  # kept for inspection even when refused
    assert (store.active_backbone(root, VARIANT) == new_file) is active
    if aucs[0] is not None and status == "kept_previous":
        assert f"{aucs[0]:.4f}" in last["reason"] and f"{aucs[1]:.4f}" in last["reason"]
    manifest = store.read_manifest(root)
    assert (manifest["active_backbones"].get(VARIANT) is not None) is active
    # the discriminators belong to the backbone that stays active
    chosen = new_file if active else tiny_weights
    key = _feature_tracker(chosen).checkpoint_key()
    assert store.discriminator_path(root, "default", key).is_file()
    if not active:
        new_key = _feature_tracker(new_file).checkpoint_key()
        assert not store.discriminator_path(root, "default", new_key).is_file()


def test_discriminator_mode_keeps_the_backbone(root, tiny_weights) -> None:
    write_bank(root, frames_per_video=12)
    assert (
        train.main(
            [
                "--from-samples",
                str(root),
                "--mode",
                "discriminator",
                "--weights",
                str(tiny_weights),
                "--variant",
                VARIANT,
                "--device",
                "cpu",
                "--min-samples",
                "10",
            ]
        )
        == 0
    )
    last = _last_retrain(root)
    assert last["status"] == "activated" and last["backbone"] is None
    assert last["before"] is None and isinstance(last["after"], float)
    assert not list((root / "backbones").glob("*")) if (root / "backbones").exists() else True
    key = _feature_tracker(tiny_weights).checkpoint_key()
    assert store.discriminator_path(root, "default", key).is_file()


# ------------------------------------------------------------------ errors, dry run, contract


def test_too_few_samples_is_an_error(root, tiny_weights, capsys) -> None:
    write_bank(root, frames_per_video=5)  # 10 samples per video: train split < 40
    assert _run(root, tiny_weights) != 0
    last = _last_retrain(root)
    assert last["status"] == "error"
    assert "minimum 40" in last["reason"]
    assert last["n_samples"] == 20 and last["finished"]
    assert store.active_backbone(root, VARIANT) is None
    assert (
        not list((root / "discriminators").glob("*"))
        if (root / "discriminators").exists()
        else True
    )
    assert "ERROR" in capsys.readouterr().err


def test_no_samples_is_an_error(root, tiny_weights) -> None:
    assert _run(root, tiny_weights) != 0
    assert _last_retrain(root)["status"] == "error"


def test_missing_root_is_an_error(tmp_path, tiny_weights) -> None:
    assert _run(tmp_path / "nowhere", tiny_weights) != 0
    assert not (tmp_path / "nowhere").exists()


def test_dry_run_writes_nothing(root, tiny_weights) -> None:
    write_bank(root, frames_per_video=12)
    before = _snapshot(root)
    assert _run(root, tiny_weights, "--dry-run", "--min-samples", "10") == 0
    assert _snapshot(root) == before
    assert not (root / "manifest.json").exists()
    # a dry run that would fail does not record the failure either
    assert _run(root, tiny_weights, "--dry-run", "--min-samples", "500") != 0
    assert _snapshot(root) == before


def test_retrain_argv_is_frozen(tmp_path) -> None:
    import sys

    assert train.retrain_argv(tmp_path) == [
        sys.executable,
        "-m",
        "vaila.ai_tracker_train",
        "--from-samples",
        str(tmp_path),
        "--mode",
        "both",
        "--device",
        "auto",
    ]
    assert train.retrain_argv(tmp_path, "backbone", "cpu")[-4:] == [
        "--mode",
        "backbone",
        "--device",
        "cpu",
    ]
    # the CLI it describes parses
    args = train.build_arg_parser().parse_args(train.retrain_argv(tmp_path)[3:])
    assert args.from_samples == str(tmp_path) and args.mode == "both" and args.device == "auto"


def test_data_and_from_samples_are_exclusive() -> None:
    with pytest.raises(SystemExit):
        train.build_arg_parser().parse_args(["--data", "x.yaml", "--from-samples", "r"])


def test_ai_models_option_sets_the_session_home(tmp_path) -> None:
    other = tmp_path / "other_home"
    other.mkdir()
    train.main(["--from-samples", str(tmp_path / "none"), "--ai-models", str(other)])
    assert store.resolve_home().root == other / "ai_tracker"


def test_defaults_follow_the_active_backbone(root, tiny_weights) -> None:
    active = root / "backbones" / "mobilenet_v3_small_finetuned_20261124_000000.pth"
    store.atomic_copy(tiny_weights, active)
    store.set_active_backbone(root, VARIANT, active)
    cfg = train.SamplesConfig(root=root)
    assert train._pick_variant(cfg) == VARIANT
    assert train._pick_weights(cfg, VARIANT) == active
    explicit = train.SamplesConfig(root=root, variant="efficientnet_b0", weights=tiny_weights)
    assert train._pick_variant(explicit) == "efficientnet_b0"
    assert train._pick_weights(explicit, "efficientnet_b0") == tiny_weights


# ------------------------------------------------------------------ hold-out


def test_split_whole_videos_never_mixed() -> None:
    videos = ["a"] * 40 + ["b"] * 30 + ["c"] * 30
    frames = np.arange(100)
    tr, va, kind = train.split_holdout(videos, frames, 0.3, seed=1)
    assert kind == "videos" and len(va) > 0 and len(tr) > 0
    assert set(tr).isdisjoint(va) and len(tr) + len(va) == 100
    assert {videos[i] for i in tr}.isdisjoint({videos[i] for i in va})
    # reproducible, and a different seed may pick a different video
    again = train.split_holdout(videos, frames, 0.3, seed=1)
    assert np.array_equal(tr, again[0]) and np.array_equal(va, again[1])


def test_split_two_videos_holds_out_one() -> None:
    videos = ["a"] * 10 + ["b"] * 10
    tr, va, _ = train.split_holdout(videos, np.zeros(20, dtype=int), 0.2, seed=0)
    assert len(va) == 10 and len(tr) == 10
    assert len({videos[i] for i in va}) == 1


def test_split_one_video_holds_out_a_contiguous_frame_block() -> None:
    frames = np.repeat(np.arange(50), 4)  # a click and 3 negatives share each frame
    videos = ["only"] * len(frames)
    tr, va, kind = train.split_holdout(videos, frames, 0.2, seed=0)
    assert kind == "frames"
    assert set(frames[tr]).isdisjoint(frames[va])  # no frame on both sides
    assert frames[va].min() > frames[tr].max()  # one block at the end
    assert sorted(set(frames[va])) == list(range(40, 50))


def test_split_one_video_one_frame_is_refused() -> None:
    with pytest.raises(ValueError, match="hold out"):
        train.split_holdout(["v"] * 4, np.zeros(4, dtype=int))


def test_bank_split_through_the_loader(root) -> None:
    write_bank(root, videos=("clip_a", "clip_b", "clip_c"), frames_per_video=10)
    bank = train.load_sample_bank(root)
    assert len(bank) == 60 and bank.n_videos == 3
    tr, va, _ = train.split_holdout(bank.videos, bank.frames, 0.34, seed=3)
    assert {bank.videos[i] for i in tr}.isdisjoint({bank.videos[i] for i in va})


# ------------------------------------------------------------------ pieces


def test_ridge_dual_and_primal_agree_and_match_the_tracker() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 12)).astype(np.float32)
    y = np.where(rng.random(30) > 0.5, 1.0, -1.0).astype(np.float32)
    w_primal, b_primal = train.ridge_fit(X, y)  # N > d: primal
    lam = train.RIDGE_LAMBDA
    alpha = np.linalg.solve(X.astype(np.float64) @ X.T + lam * np.eye(30), y.astype(np.float64))
    w_dual = X.T @ alpha
    assert np.allclose(w_primal, w_dual, atol=1e-4)
    # the tracker's own online ridge gives the same weights
    tracker = _classic_tracker()
    tracker.training_feats = [np.pad(r, (0, 796)) for r in X]  # pad to the 808 classic width
    tracker.training_labels = list(y)
    tracker.retrain_online_model()
    w_tracker, _ = train.ridge_fit(np.stack(tracker.training_feats), y)  # N < d: dual
    assert tracker.discriminator_w is not None
    assert np.allclose(tracker.discriminator_w, w_tracker, atol=1e-3)
    assert np.isfinite(b_primal)


def test_roc_auc_values() -> None:
    assert train.roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([-1, -1, 1, 1])) == 1.0
    assert train.roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([-1, -1, 1, 1])) == 0.0
    assert train.roc_auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([-1, 1, -1, 1])) == 0.5
    assert train.roc_auc(np.array([1.0, 2.0]), np.array([1, 1])) is None
    scores = np.array([0.1, 0.4, 0.35, 0.8])  # the classic sklearn example: AUC = 0.75
    assert train.roc_auc(scores, np.array([0, 0, 1, 1])) == pytest.approx(0.75)


def test_live_features_have_the_tracker_layout(tiny_weights) -> None:
    rng = np.random.default_rng(1)
    patches = [_blob(rng, BLOCK, True), _blob(rng, BLOCK, False)]
    tracker = train.make_feature_tracker(VARIANT, tiny_weights, "cpu")
    X = train._embed_patches(tracker, patches, ["point", "point"])
    assert X.shape == (2, 808 + 576) == (2, tracker.effective_feature_dim())
    classic = AITracker.extract_patch_feature(patches[0], "point")
    assert np.allclose(X[0, :808], classic)


def test_set_trainable_freezes_all_but_the_last_stage() -> None:
    mobile = tv_models.get_model(VARIANT, weights=None)
    trainable = train._set_trainable(mobile, VARIANT, "last")
    assert trainable
    assert all(p.requires_grad for p in mobile.features[-1].parameters())
    assert all(p.requires_grad for p in mobile.classifier.parameters())
    assert not any(p.requires_grad for p in mobile.features[0].parameters())
    assert not any(p.requires_grad for p in mobile.features[-4].parameters())
    everything = train._set_trainable(tv_models.get_model(VARIANT, weights=None), VARIANT, "all")
    assert len(everything) > len(trainable)
    resnet = tv_models.get_model("resnet50", weights=None)
    train._set_trainable(resnet, "resnet50", "last")
    assert all(p.requires_grad for p in resnet.layer4.parameters())
    assert all(p.requires_grad for p in resnet.fc.parameters())
    assert not any(p.requires_grad for p in resnet.layer3.parameters())


def test_train_loop_learns_and_keeps_frozen_batchnorm_in_eval() -> None:
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    nn = torch.nn
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.Flatten(), nn.Linear(4 * 6 * 6, 2)
    )
    for p in list(model[0].parameters()) + list(model[1].parameters()):
        p.requires_grad = False
    x = torch.randn(32, 3, 8, 8)
    labels = (x.mean(dim=(1, 2, 3)) > 0).long()
    x[:, 0] += labels.view(-1, 1, 1) * 2.0
    bn_before = model[1].running_mean.clone()
    loader = DataLoader(TensorDataset(x, labels), batch_size=8)
    history = train._train_loop(model, loader, 8, 1e-2, "cpu", model[3].parameters())
    assert [h["epoch"] for h in history] == list(range(1, 9))
    assert history[-1]["loss"] < history[0]["loss"]
    assert torch.equal(model[1].running_mean, bn_before)  # frozen BN did not drift


def test_train_backbone_still_uses_the_shared_loop(tmp_path, tiny_weights, monkeypatch) -> None:
    """The YOLO path trains through ``_train_loop`` and saves into backbones/."""
    calls: list[int] = []
    real = train._train_loop

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(train, "_train_loop", spy)
    cfg = train.TrainConfig(
        data_yaml=_make_mini_detect_dataset(tmp_path / "ds"),
        weights=tiny_weights,
        mode="backbone",
        epochs=1,
        batch=4,
        device="cpu",
        max_samples=20,
    )
    out = train.train_backbone(cfg)
    assert calls and out is not None and out.is_file()
    assert out.parent.name == "backbones"


def test_device_auto_needs_free_gpu_memory(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1 * 1024**3, 8 * 1024**3))
    assert train._resolve_device("auto") == "cpu"  # 1 GiB free < 3 GiB
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (6 * 1024**3, 8 * 1024**3))
    assert train._resolve_device("auto") == "cuda"
    assert train._resolve_device("cpu") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert train._resolve_device("auto") == "cpu"
