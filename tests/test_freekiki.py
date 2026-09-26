"""Tests for vaila/freekiki.py (workspace, dataset, registry, CSV rows, quality metrics).

Update Date: 25 September 2026
Version: 0.4.5
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from vaila import freekiki as fk


def _fake_kiki49(root: Path, *, nkp: int = fk.NKP, flip_idx: list[int] | None = None) -> Path:
    names, schema_flip = fk.load_schema()
    data = {
        "path": "/somewhere/else/kiki49_dataset",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "kpt_shape": [nkp, 3],
        "flip_idx": flip_idx if flip_idx is not None else schema_flip,
        "names": {0: "football_pitch"},
        "kpt_names": {0: names},
    }
    root.mkdir(parents=True)
    (root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    label = " ".join(["0", "0.5", "0.5", "1", "1"] + ["0.5", "0.5", "2"] * nkp)
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        (root / "images" / split / "a.jpg").write_bytes(b"jpg")
        (root / "labels" / split / "a.txt").write_text(label + "\n", encoding="utf-8")
        (root / "labels" / f"{split}.cache").write_bytes(b"cache")
    (root / "preview").mkdir()
    (root / "preview" / "big.jpg").write_bytes(b"skip me")
    (root / "manifest.csv").write_text("image\n", encoding="utf-8")
    return root


def test_schema_matches_skeleton() -> None:
    names, flip_idx = fk.load_schema()
    assert len(names) == fk.NKP and sorted(flip_idx) == list(range(fk.NKP))
    bones = fk.load_bones()
    assert bones and all(0 <= a < fk.NKP and 0 <= b < fk.NKP for a, b in bones)


def test_init_workspace_layout_and_settings(tmp_path: Path) -> None:
    ws = fk.init_workspace(tmp_path / "ws")
    for sub in ("spec", "datasets", "runs", "models", "outputs"):
        assert (ws / sub).is_dir()
    assert (ws / "spec" / "soccerfield_kiki.csv").is_file()
    settings = fk.load_settings(ws)
    settings["train"]["epochs"] = 7
    fk.save_settings(ws, settings)
    fk.init_workspace(ws)  # re-init keeps user settings
    assert fk.load_settings(ws)["train"]["epochs"] == 7
    assert fk.load_settings(ws)["train"]["imgsz"] == 1280


def test_import_dataset_copies_and_rewrites_path(tmp_path: Path) -> None:
    src = _fake_kiki49(tmp_path / "src")
    ws = fk.init_workspace(tmp_path / "ws")
    dst = fk.import_dataset(ws, src)
    data = yaml.safe_load((dst / "data.yaml").read_text(encoding="utf-8"))
    assert Path(data["path"]) == dst.resolve()
    assert (dst / "images" / "val" / "a.jpg").is_file()
    assert not (dst / "preview").exists()
    assert not list(dst.rglob("*.cache"))
    # Source untouched.
    assert "/somewhere/else" in (src / "data.yaml").read_text(encoding="utf-8")
    assert fk.check_dataset(ws) == []
    fk.import_dataset(ws, src)  # resumable re-run


def test_check_dataset_flags_wrong_schema(tmp_path: Path) -> None:
    bad_flip = list(range(fk.NKP))
    ws = fk.init_workspace(tmp_path / "ws")
    fk.import_dataset(ws, _fake_kiki49(tmp_path / "src", flip_idx=bad_flip))
    assert any("flip_idx" in issue for issue in fk.check_dataset(ws))

    ws2 = fk.init_workspace(tmp_path / "ws2")
    fk.import_dataset(ws2, _fake_kiki49(tmp_path / "src2", nkp=32, flip_idx=list(range(32))))
    issues = fk.check_dataset(ws2)
    assert any("kpt_shape" in issue for issue in issues)
    assert any("columns" in issue for issue in issues)


def _fake_run(ws: Path, name: str, score: float) -> Path:
    run = ws / "runs" / name
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(name.encode())
    with (run / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "  metrics/mAP50(P)", "  metrics/mAP50-95(P)"])
        w.writerow([1, score + 0.1, score / 2])
        w.writerow([2, score + 0.2, score])
    return run


def test_register_run_promotes_only_better(tmp_path: Path) -> None:
    ws = fk.init_workspace(tmp_path / "ws")
    first = fk.register_run(
        ws, _fake_run(ws, "r1", 0.40), base="yolo26m-pose.pt", epochs=2, imgsz=640
    )
    assert first["promoted"] and first["best_epoch"] == "2"
    worse = fk.register_run(ws, _fake_run(ws, "r2", 0.30), base="active", epochs=2, imgsz=640)
    assert not worse["promoted"]
    assert (ws / fk.ACTIVE_MODEL).read_bytes() == b"r1"
    better = fk.register_run(ws, _fake_run(ws, "r3", 0.55), base="active", epochs=2, imgsz=640)
    assert better["promoted"]
    assert (ws / fk.ACTIVE_MODEL).read_bytes() == b"r3"
    assert fk.load_settings(ws)["active"]["run"] == "r3"
    with (ws / fk.REGISTRY_CSV).open(encoding="utf-8") as f:
        assert [r["run"] for r in csv.DictReader(f)] == ["r1", "r2", "r3"]
    assert fk.resolve_model(ws, "active").endswith("active.pt")


def _interrupted_run(ws: Path, name: str, *, optimizer: dict | None, epochs: int = 150) -> Path:
    """A run as left on disk after 2 epochs: args.yaml, results.csv, weights/last.pt."""
    import torch

    run = _fake_run(ws, name, 0.2)
    (run / "args.yaml").write_text(
        yaml.safe_dump({"model": "/x/yolo26m-pose.pt", "epochs": epochs, "imgsz": 1280}),
        encoding="utf-8",
    )
    torch.save(
        {"epoch": 1 if optimizer else -1, "optimizer": optimizer}, run / "weights" / "last.pt"
    )
    return run


def test_run_states(tmp_path: Path) -> None:
    ws = fk.init_workspace(tmp_path / "ws")
    _interrupted_run(ws, "cut", optimizer={"state": {}})
    _interrupted_run(ws, "ended", optimizer=None)
    _fake_run(ws, "early", 0.1)  # killed before the first last.pt
    (ws / "runs" / "early" / "weights" / "best.pt").unlink()
    fk.register_run(ws, _fake_run(ws, "done", 0.4), base="yolo26m-pose.pt", epochs=2, imgsz=640)
    live = _interrupted_run(ws, "live", optimizer={"state": {}})
    (live / fk.RUNNING_MARKER).write_text(str(__import__("os").getpid()), encoding="utf-8")
    states = {r["run"]: r for r in fk.list_runs(ws)}
    assert states["cut"]["state"] == "resumable"
    assert (states["cut"]["epochs_done"], states["cut"]["epochs"]) == (2, 150)
    assert states["cut"]["base"] == "yolo26m-pose.pt"
    assert states["ended"]["state"] == "finished"
    assert states["early"]["state"] == "no-checkpoint"
    assert states["done"]["state"] == "registered"
    # the marker names this pytest process, whose command line is not FreeKiki's
    assert states["live"]["state"] == "resumable"
    (live / fk.RUNNING_MARKER).write_text("999999999", encoding="utf-8")  # dead PID
    assert fk.run_state(ws, live)["state"] == "resumable"


def test_running_marker_blocks_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = fk.init_workspace(tmp_path / "ws")
    _interrupted_run(ws, "live", optimizer={"state": {}})
    monkeypatch.setattr(fk, "_is_running", lambda run_dir: True)
    assert fk.run_state(ws, ws / "runs" / "live")["state"] == "running"
    with pytest.raises(fk.RunStateError, match="still training"):
        fk.resume(ws, name="live")
    with pytest.raises(fk.RunStateError, match="resume --name live"):
        fk.train(ws, name="live")


def test_resume_finished_run_only_registers(tmp_path: Path) -> None:
    ws = fk.init_workspace(tmp_path / "ws")
    _interrupted_run(ws, "ended", optimizer=None)
    row = fk.resume(ws)
    assert row["run"] == "ended" and row["promoted"]
    assert fk.run_state(ws, ws / "runs" / "ended")["state"] == "registered"
    with pytest.raises(fk.RunStateError, match="already finished"):
        fk.resume(ws, name="ended")


def test_resume_interrupted_run_continues_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types

    ws = fk.init_workspace(tmp_path / "moved_ws")
    fk.import_dataset(ws, _fake_kiki49(tmp_path / "src"))
    run = _interrupted_run(ws, "cut", optimizer={"state": {}})
    calls: list = []

    class FakeYOLO:
        def __init__(self, weights):
            calls.append(("load", weights))

        def add_callback(self, *args):
            pass

        def train(self, **kwargs):
            calls.append(("train", kwargs))

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    row = fk.resume(ws, batch=8)
    assert calls[0] == ("load", str(run / "weights" / "last.pt"))
    kwargs = calls[1][1]
    assert kwargs["resume"] is True and kwargs["batch"] == 8
    assert kwargs["save_dir"] == str(run)
    assert Path(kwargs["data"]) == fk.dataset_dir(ws) / "data.yaml"
    assert not (run / fk.RUNNING_MARKER).exists()
    assert row["run"] == "cut" and row["epochs"] == 150 and row["imgsz"] == 1280


def test_resolve_model_active_missing(tmp_path: Path) -> None:
    ws = fk.init_workspace(tmp_path / "ws")
    with pytest.raises(FileNotFoundError):
        fk.resolve_model(ws, "active")
    assert fk.resolve_model(ws, "yolo26m-pose.pt") == "yolo26m-pose.pt"


def test_keypoints_row_blanks_low_confidence() -> None:
    header = fk.getpixelvideo_header()
    assert header[:3] == ["frame", "p0_x", "p0_y"] and header[-1] == "p48_y"
    xy = np.arange(fk.NKP * 2, dtype=float).reshape(fk.NKP, 2)
    kconf = np.full(fk.NKP, 0.9)
    kconf[3] = 0.1
    row = fk.keypoints_row(12, xy, kconf, 0.5)
    assert len(row) == len(header) and row[0] == 12
    assert row[1:3] == ["0.00", "1.00"]
    assert row[7:9] == ["", ""]
    assert fk.keypoints_row(5, None, None, 0.5)[1:] == [""] * (fk.NKP * 2)


def test_read_label_keypoints(tmp_path: Path) -> None:
    label = tmp_path / "a.txt"
    kps = ["0.25", "0.5", "2"] + ["0", "0", "0"] * (fk.NKP - 1)
    label.write_text(" ".join(["0", "0.5", "0.5", "1", "1"] + kps) + "\n", encoding="utf-8")
    xy, vis = fk.read_label_keypoints(label)
    assert xy.shape == (fk.NKP, 2) and xy[0].tolist() == [0.25, 0.5]
    assert vis[0] == 2 and vis[1:].sum() == 0
    label.write_text("0 0.5 0.5 1 1\n", encoding="utf-8")
    assert fk.read_label_keypoints(label) == (None, None)


def test_keypoint_errors_and_summary() -> None:
    gt_xy = np.zeros((fk.NKP, 2))
    gt_vis = np.zeros(fk.NKP)
    gt_vis[:3] = 2  # p0..p2 labelled
    pred_xy = gt_xy.copy()
    pred_xy[1] = [3.0, 4.0]  # 5 px off at 960 px wide -> 10 px @1920
    conf = np.zeros(fk.NKP)
    conf[[0, 1, 5]] = 0.9  # p2 missed, p5 false positive
    res = fk.keypoint_errors(pred_xy, conf, gt_xy, gt_vis, 960, 0.5)
    assert res["found"][:3].tolist() == [True, True, False]
    assert res["false_pos"][5] and res["false_pos"].sum() == 1
    assert res["err_px"][1] == pytest.approx(10.0) and np.isnan(res["err_px"][2])

    missed = fk.keypoint_errors(None, None, gt_xy, gt_vis, 960, 0.5)
    overall, per_kp = fk.summarize_keypoint_errors([res, missed])
    assert overall["images"] == 2 and overall["labelled"] == 6
    assert overall["recall"] == pytest.approx(2 / 6, abs=1e-3)
    assert overall["precision"] == pytest.approx(2 / 3, abs=1e-3)
    assert overall["pck10"] == 1.0 and overall["median_err_px"] == 5.0
    assert per_kp[1]["kp"] == "p1" and per_kp[1]["recall"] == 0.5
    assert per_kp[10]["labelled"] == 0 and per_kp[10]["recall"] is None


def test_video_quality_indicators() -> None:
    kc_on = np.zeros(fk.NKP)
    kc_on[:5] = 0.8
    xy = np.zeros((fk.NKP, 2))
    xy_seq = [xy + [t, 0.0] for t in range(4)] + [None]  # smooth pan, then a miss
    kc_seq = [kc_on] * 4 + [None]
    q = fk.video_quality(xy_seq, kc_seq, 0.5)
    assert q["frames"] == 5 and q["detection_rate"] == 0.8
    assert q["mean_visible_kps"] == 4.0 and q["calib_ready_rate"] == 0.8
    assert q["mean_kp_conf"] == pytest.approx(0.8) and q["jitter_px"] == 0.0
    assert fk.video_quality([], [], 0.5)["jitter_px"] is None
