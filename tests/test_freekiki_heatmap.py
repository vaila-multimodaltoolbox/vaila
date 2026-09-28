"""Tests for vaila/freekiki_heatmap.py and its FreeKiki wiring (train/resume/register/predict).

Geometry and decoding are checked on synthetic values; training runs a tiny
ResNet-18 from scratch on CPU (64 px canvas, a few real images), so no
weights are downloaded and no GPU is needed.

Update Date: 28 September 2026
Version: 0.4.6
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import torch
import yaml

from vaila import freekiki as fk
from vaila import freekiki_heatmap as fh

FRAME_W, FRAME_H = 160, 90


# --------------------------------------------------------------------------- #
# Geometry / heatmaps
# --------------------------------------------------------------------------- #
def test_input_size_is_16_9_and_multiple_of_32() -> None:
    assert fh.input_size(1024) == (1024, 576)
    assert fh.input_size(1000) == (1024, 576)
    assert fh.input_size(64) == (64, 64)  # floor 64 on both sides
    for imgsz in (320, 640, 1280):
        w, h = fh.input_size(imgsz)
        assert w % 32 == 0 and h % 32 == 0 and h >= w * 9 / 16


def test_letterbox_inverse_round_trip() -> None:
    m = fh.letterbox_matrix(1920, 1080, 1024, 576)
    xy = np.array([[0.0, 0.0], [1919.0, 1079.0], [960.0, 540.0], [13.5, 700.25]])
    back = fh.apply_affine(np.linalg.inv(m), fh.apply_affine(m, xy))
    np.testing.assert_allclose(back, xy, atol=1e-9)
    # 4:3 frame on a 16:9 canvas: centered with side padding, aspect kept.
    m = fh.letterbox_matrix(640, 480, 1024, 576)
    corners = fh.apply_affine(m, np.array([[0.0, 0.0], [640.0, 480.0]]))
    np.testing.assert_allclose(corners, [[128.0, 0.0], [896.0, 576.0]])


def test_flip_matrix_is_involution_and_flip_idx_swaps_pairs() -> None:
    names, flip_idx = fk.load_schema()
    flip_idx = np.asarray(flip_idx)
    np.testing.assert_array_equal(flip_idx[flip_idx], np.arange(len(names)))

    class NoAug:  # uniform() -> midpoint: no rotation/scale/shift, flip only
        def uniform(self, lo, hi):
            return (lo + hi) / 2

    w, h = 1024, 576
    m = fh.augment_matrix(NoAug(), w, h, flip=True)
    np.testing.assert_allclose(m @ m, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(fh.apply_affine(m, np.array([[100.0, 50.0]])), [[924.0, 50.0]])


def test_cells_round_trip() -> None:
    xy = np.array([[0.0, 0.0], [17.3, 41.9], [1023.0, 575.0]])
    np.testing.assert_allclose(fh.from_cells(fh.to_cells(xy)), xy, atol=1e-12)


def test_decode_recovers_subcell_peaks() -> None:
    uv = torch.tensor([[[10.3, 7.8], [0.0, 0.0], [30.75, 20.1]]])
    vis = torch.tensor([[1.0, 0.0, 1.0]])
    hm = fh.render_heatmaps(uv, vis, (32, 48))
    assert hm.shape == (1, 3, 32, 48)
    assert float(hm[0, 1].max()) == 0.0  # v=0 -> empty target
    got, peak = fh.decode_heatmaps(hm[0].numpy())
    np.testing.assert_allclose(got[[0, 2]], uv[0, [0, 2]].numpy(), atol=0.05)
    assert peak[1] == 0.0 and peak[0] > 0.9 and peak[2] > 0.9
    # < 1 px at stride 4 after mapping back to canvas pixels.
    err = np.abs(fh.from_cells(got[[0, 2]]) - fh.from_cells(uv[0, [0, 2]].numpy()))
    assert err.max() < 1.0


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def _label_line(xy: np.ndarray, vis: np.ndarray) -> str:
    kps = np.column_stack([xy, vis]).ravel()
    return " ".join(["0", "0.5", "0.5", "1", "1"] + [f"{v:.6f}" for v in kps])


def _write_split(root: Path, split: str, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        frame = rng.integers(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
        xy = rng.uniform(0.05, 0.95, (fk.NKP, 2))
        vis = (rng.random(fk.NKP) < 0.7).astype(float) * 2
        for x, y in (xy * (FRAME_W, FRAME_H))[vis > 0].astype(int):
            cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 255), -1)
        cv2.imwrite(str(root / "images" / split / f"{split}_{i}.png"), frame)
        (root / "labels" / split / f"{split}_{i}.txt").write_text(
            _label_line(xy, vis) + "\n", encoding="utf-8"
        )


def _real_kiki49(root: Path) -> Path:
    """Tiny YOLO-pose kiki49 build with readable images (unlike the byte stubs elsewhere)."""
    names, flip_idx = fk.load_schema()
    root.mkdir(parents=True)
    data = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "kpt_shape": [fk.NKP, 3],
        "flip_idx": flip_idx,
        "names": {0: "football_pitch"},
        "kpt_names": {0: names},
    }
    (root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    for seed, (split, n) in enumerate((("train", 4), ("val", 2), ("test", 2))):
        _write_split(root, split, n, seed)
    return root


def test_read_label_and_label_path(tmp_path: Path) -> None:
    ds = _real_kiki49(tmp_path / "ds")
    img = fh.list_images(ds, "images/train")[0]
    assert fh.label_path(img) == ds / "labels" / "train" / f"{img.stem}.txt"
    xy, vis = fh.read_label(fh.label_path(img), fk.NKP)
    assert xy.shape == (fk.NKP, 2) and vis.shape == (fk.NKP,)
    assert set(np.unique(vis)) <= {0.0, 2.0}
    xy, vis = fh.read_label(tmp_path / "missing.txt", fk.NKP)
    assert not xy.any() and not vis.any()


def test_dataset_targets_follow_image_and_flip(tmp_path: Path) -> None:
    ds = _real_kiki49(tmp_path / "ds")
    images = fh.list_images(ds, "images/train")
    _, flip_idx = fk.load_schema()
    plain = fh.KikiHeatDataset(images, flip_idx, (64, 64), train=False)
    x, uv, vis = plain[0]
    assert x.shape == (3, 64, 64) and uv.shape == (fk.NKP, 2) and vis.shape == (fk.NKP,)
    xy, v = fh.read_label(fh.label_path(images[0]), fk.NKP)
    m = fh.letterbox_matrix(FRAME_W, FRAME_H, 64, 64)
    expect = fh.to_cells(fh.apply_affine(m, xy * (FRAME_W, FRAME_H)))
    np.testing.assert_allclose(uv.numpy()[v > 0], expect[v > 0], atol=1e-4)
    np.testing.assert_array_equal(vis.numpy() > 0, v > 0)
    # Augmented samples are deterministic per (seed, epoch, index).
    aug = fh.KikiHeatDataset(images, flip_idx, (64, 64), train=True, seed=3)
    a, b = aug[1], aug[1]
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    aug.epoch = 1
    assert not torch.equal(aug[1][0], a[0])


# --------------------------------------------------------------------------- #
# Training / checkpoints / predictor
# --------------------------------------------------------------------------- #
class _StopError(Exception):
    pass


def _stop_after_epoch_1(message: str) -> None:
    if message.startswith("heatmap epoch 1/"):
        raise _StopError


def test_train_resume_and_predict_on_cpu(tmp_path: Path) -> None:
    ds = _real_kiki49(tmp_path / "ds")
    _, flip_idx = fk.load_schema()
    images = fh.list_images(ds, "images/train")
    run = tmp_path / "run"
    calls: list[int] = []

    def val_fn(predictor) -> dict:
        calls.append(1)
        conf, xy, kc = predictor.predict(cv2.imread(str(images[0])))
        assert xy.shape == (fk.NKP, 2) and kc.shape == (fk.NKP,)
        return {"recall": 0.5, "precision": 0.5, "err_median": 3.0, "pck10_all": 0.1 * len(calls)}

    kwargs: dict[str, Any] = {
        "train_images": images,
        "flip_idx": flip_idx,
        "val_fn": val_fn,
        "imgsz": 64,
        "batch": 2,
        "workers": 0,
        "backbone": "resnet18",
        "device": "cpu",
    }
    with pytest.raises(_StopError):
        fh.train_heatmap(
            run,
            epochs=2,
            pretrained=False,
            extra_args={"fraction": 0.5, "manifest": None},
            log=_stop_after_epoch_1,
            **kwargs,
        )
    args = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
    assert args["backend"] == "heatmap" and args["epochs"] == 2 and args["fraction"] == 0.5
    assert args["model"] == "resnet18-scratch" and args["input_wh"] == [64, 64]
    last = torch.load(run / "weights" / "last.pt", weights_only=False)
    assert last["epoch"] == 1 and last["optimizer"] is not None
    assert fh.model_backend(run / "weights" / "last.pt") == "heatmap"
    assert fk._checkpoint_resumable(run / "weights" / "last.pt")

    best = fh.train_heatmap(run, resume=True, log=lambda m: None, **kwargs)
    assert best == run / "weights" / "best.pt"
    with (run / "results.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert list(rows[0]) == fh.RESULTS_FIELDS
    assert fh.best_fitness(run / "results.csv") == (pytest.approx(0.2), 2)
    last = torch.load(run / "weights" / "last.pt", weights_only=False)
    assert last["epoch"] == 2 and last["optimizer"] is None  # finished: not resumable
    assert last["args"]["fraction"] == 0.5
    assert torch.load(best, weights_only=False)["best_epoch"] == 2
    assert (run / "heatmap_summary.json").is_file()

    predictor = fk.load_predictor(str(best), device="cpu")
    assert predictor.backend == "heatmap" and predictor.imgsz == 64
    conf, xy, kc = predictor.predict(np.zeros((FRAME_H, FRAME_W, 3), np.uint8))
    assert isinstance(conf, float) and xy.shape == (fk.NKP, 2) and kc.shape == (fk.NKP,)
    assert np.isfinite(xy).all() and ((kc >= 0) & (kc <= 1)).all()


def test_resume_refuses_finished_or_foreign_checkpoint(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    torch.save({"epoch": 3, "optimizer": {"state": {}}}, run / "weights" / "last.pt")
    assert fh.model_backend(run / "weights" / "last.pt") == "yolo"
    assert fh.model_backend(tmp_path / "nothing.pt") == "yolo"
    with pytest.raises(ValueError, match="not a resumable heatmap"):
        fh.train_heatmap(run, train_images=[], flip_idx=[0], resume=True, device="cpu")


# --------------------------------------------------------------------------- #
# FreeKiki wiring
# --------------------------------------------------------------------------- #
def _heatmap_results(path: Path, values: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fh.RESULTS_FIELDS)
        w.writeheader()
        for epoch, v in enumerate(values, 1):
            w.writerow({"epoch": epoch, "val/pck10_all": v, "train/loss": 1.0})


def test_read_best_metrics_heatmap(tmp_path: Path) -> None:
    _heatmap_results(tmp_path / "results.csv", [0.2, 0.61, 0.61, 0.4])
    best = fk.read_best_metrics(tmp_path / "results.csv")
    assert best == {"best_epoch": "2", "pose_map50": "", "pose_map50_95": "", "fitness": 0.61}


def test_run_backend(tmp_path: Path) -> None:
    assert fk.run_backend(tmp_path) == "yolo"
    (tmp_path / "args.yaml").write_text("model: yolo26m-pose.pt\n", encoding="utf-8")
    assert fk.run_backend(tmp_path) == "yolo"
    (tmp_path / "args.yaml").write_text("backend: heatmap\n", encoding="utf-8")
    assert fk.run_backend(tmp_path) == "heatmap"


def test_heatmap_run_is_registered_but_never_promoted(tmp_path: Path) -> None:
    ws = fk.init_workspace(tmp_path / "ws")
    run = ws / "runs" / "hm"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"hm")
    (run / "args.yaml").write_text("backend: heatmap\nepochs: 4\nimgsz: 64\n", encoding="utf-8")
    _heatmap_results(run / "results.csv", [0.3, 0.9])
    row = fk.register_run(ws, run, base="resnet50-imagenet", epochs=4, imgsz=64)
    assert row["backend"] == "heatmap" and row["promoted"] is False
    assert not (ws / fk.ACTIVE_MODEL).exists()  # even the first model stays a candidate
    with (ws / fk.REGISTRY_CSV).open(encoding="utf-8") as f:
        (logged,) = list(csv.DictReader(f))
    assert logged["backend"] == "heatmap" and logged["fitness"] == "0.9"
    assert fk.run_state(ws, run)["state"] == "registered"


def test_older_registry_gains_backend_and_fitness_columns(tmp_path: Path) -> None:
    """A pre-heatmap registry.csv must not silently drop the new columns."""
    ws = fk.init_workspace(tmp_path / "ws")
    old_header = [c for c in fk.REGISTRY_FIELDS if c not in ("fitness", "backend")]
    old_row = dict.fromkeys(old_header, "x") | {"run": "yolo_old"}
    with (ws / fk.REGISTRY_CSV).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=old_header)
        writer.writeheader()
        writer.writerow(old_row)
    run = ws / "runs" / "hm"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"hm")
    (run / "args.yaml").write_text("backend: heatmap\n", encoding="utf-8")
    _heatmap_results(run / "results.csv", [0.3, 0.9])
    fk.register_run(ws, run, base="resnet50-imagenet", epochs=2, imgsz=64)
    with (ws / fk.REGISTRY_CSV).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0] == old_row | {"fitness": "", "backend": "yolo"}  # old values kept
    assert rows[1]["backend"] == "heatmap" and rows[1]["fitness"] == "0.9"


def test_collect_predictions_uses_predictor_interface(tmp_path: Path) -> None:
    ds = _real_kiki49(tmp_path / "ds")
    images = fh.list_images(ds, "images/val")

    class FakePredictor:
        backend = "fake"

        def __init__(self):
            self.frames = []

        def predict(self, frame):
            self.frames.append(frame.shape)
            if len(self.frames) == 1:
                return 0.0, None, None  # nothing detected
            return 0.8, np.ones((fk.NKP, 2)), np.full(fk.NKP, 0.7)

    pred_fn = FakePredictor()
    pred = fk.collect_predictions(pred_fn, images, ds / "labels" / "val", {})
    assert pred_fn.frames == [(FRAME_H, FRAME_W, 3)] * 2
    assert pred["pred_xy"].shape == (2, fk.NKP, 2)
    assert np.isnan(pred["pred_xy"][0]).all() and np.isnan(pred["pred_kc"][0]).all()
    assert pred["box_conf"].tolist() == [0.0, 0.8]
    xy, _ = fh.read_label(fh.label_path(images[0]), fk.NKP)
    np.testing.assert_allclose(pred["gt_xy"][0], xy * (FRAME_W, FRAME_H), atol=1e-4)


def test_parser_heatmap_options() -> None:
    args = fk.build_parser().parse_args(
        ["train", "-w", "ws", "--backend", "heatmap", "--no-pretrained", "--lr", "5e-4"]
    )
    assert (args.backend, args.pretrained, args.lr, args.backbone) == (
        "heatmap",
        False,
        5e-4,
        "resnet50",
    )
    args = fk.build_parser().parse_args(["train", "-w", "ws"])
    assert (args.backend, args.pretrained, args.lr) == ("yolo", True, None)
    with pytest.raises(SystemExit):
        fk.build_parser().parse_args(["train", "-w", "ws", "--backend", "mmpose"])


def test_workspace_heatmap_train_interrupt_and_resume(tmp_path: Path, monkeypatch) -> None:
    ws = fk.init_workspace(tmp_path / "ws")
    fk.import_dataset(ws, _real_kiki49(tmp_path / "src"))
    real_log = fk._log

    def log(message: str) -> None:
        real_log(message)
        _stop_after_epoch_1(message)

    monkeypatch.setattr(fk, "_log", log)
    opts: dict[str, Any] = {"backend": "heatmap", "backbone": "resnet18", "pretrained": False}
    with pytest.raises(_StopError):
        fk.train(ws, epochs=2, imgsz=64, batch=2, workers=0, device="cpu", name="hm", **opts)
    monkeypatch.setattr(fk, "_log", real_log)
    run = ws / "runs" / "hm"
    info = fk.run_state(ws, run)
    assert info["state"] == "resumable" and info["epochs_done"] == 1 and info["epochs"] == 2
    manifest = yaml.safe_load((run / "train_manifest.json").read_text(encoding="utf-8"))
    assert manifest["args"]["backend"] == "heatmap" and "torchvision" in manifest["versions"]
    with pytest.raises(fk.RunStateError, match="resume --name hm"):
        fk.train(ws, epochs=2, imgsz=64, name="hm", **opts)

    row = fk.resume(ws, name="hm", device="cpu")
    assert row["backend"] == "heatmap" and row["promoted"] is False
    assert fk.run_state(ws, run)["state"] == "registered"
    assert (ws / row["model"]).is_file() and not (ws / fk.ACTIVE_MODEL).exists()
    with (run / "results.csv").open(encoding="utf-8") as f:
        assert [r["epoch"] for r in csv.DictReader(f)] == ["1", "2"]
