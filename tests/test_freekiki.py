"""Tests for vaila/freekiki.py (workspace, dataset, registry, CSV rows, quality metrics).

Update Date: 28 September 2026
Version: 0.4.6
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from vaila import freekiki as fk
from vaila import freekiki_diag as diag


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


def test_detection_checks_workspace_and_active_model_before_creating_output(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "match.mp4").touch()
    output_dir = tmp_path / "results"

    with pytest.raises(FileNotFoundError, match="freekiki.toml"):
        fk.detect_videos(tmp_path / "not_a_workspace", video_dir, output_dir=output_dir)
    assert not output_dir.exists()

    ws = fk.init_workspace(tmp_path / "workspace")
    with pytest.raises(FileNotFoundError, match="No active model"):
        fk.detect_videos(ws, video_dir, output_dir=output_dir)
    assert not output_dir.exists()


def test_model_picker_can_find_its_workspace(tmp_path: Path) -> None:
    ws = fk.init_workspace(tmp_path / "workspace")
    model = ws / fk.slot_file("m")
    model.touch()
    assert fk.workspace_for_model_file(model) == ws
    assert fk.workspace_for_model_file(tmp_path / "missing.pt") is None
    fk.validate_detection_workspace(ws, "active")


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
    settings = fk.load_settings(ws)
    settings["promotion"]["mode"] = "map"  # legacy policy: pose mAP50-95 only
    fk.save_settings(ws, settings)
    first = fk.register_run(
        ws, _fake_run(ws, "r1", 0.40), base="yolo26m-pose.pt", epochs=2, imgsz=640
    )
    assert first["promoted"] and first["best_epoch"] == "2"
    worse = fk.register_run(ws, _fake_run(ws, "r2", 0.30), base="active", epochs=2, imgsz=640)
    assert not worse["promoted"]
    assert (ws / fk.slot_file("m")).read_bytes() == b"r1"
    better = fk.register_run(ws, _fake_run(ws, "r3", 0.55), base="active", epochs=2, imgsz=640)
    assert better["promoted"]
    assert (ws / fk.slot_file("m")).read_bytes() == b"r3"
    assert fk.load_settings(ws)["models"]["m"]["run"] == "r3"
    with (ws / fk.REGISTRY_CSV).open(encoding="utf-8") as f:
        assert [r["run"] for r in csv.DictReader(f)] == ["r1", "r2", "r3"]
    assert fk.resolve_model(ws, "active").endswith("freekiki_m.pt")
    assert fk.resolve_model(ws, "freekiki_m") == fk.resolve_model(ws, "m")


def test_register_run_gate_and_never_keep_active(tmp_path: Path, monkeypatch) -> None:
    ws = fk.init_workspace(tmp_path / "ws")
    assert fk.load_settings(ws)["promotion"]["mode"] == "gate"
    first = fk.register_run(ws, _fake_run(ws, "r1", 0.40), base="b.pt", epochs=2, imgsz=640)
    assert first["promoted"]  # first model of the workspace, no evaluation
    calls: list[str] = []

    def fake_decision(ws_, candidate, settings, baseline=None):
        calls.append(candidate)
        return {"promote": False, "reasons": ["p5_recall: 0.9 -> 0.1 (tolerance 0.02)"]}

    monkeypatch.setattr(fk, "promotion_decision", fake_decision)
    row = fk.register_run(ws, _fake_run(ws, "r2", 0.90), base="active", epochs=2, imgsz=640)
    assert not row["promoted"] and calls and calls[0].endswith("kiki49_r2.pt")
    assert (ws / fk.slot_file("m")).read_bytes() == b"r1"  # better mAP alone is not enough
    assert (ws / "models" / "kiki49_r2.pt").is_file()  # candidate kept for later compare

    settings = fk.load_settings(ws)
    settings["promotion"]["mode"] = "never"
    fk.save_settings(ws, settings)
    row = fk.register_run(ws, _fake_run(ws, "r3", 0.99), base="active", epochs=2, imgsz=640)
    assert not row["promoted"] and (ws / fk.slot_file("m")).read_bytes() == b"r1"
    with (ws / fk.PROMOTION_LOG).open(encoding="utf-8") as f:
        log = list(csv.DictReader(f))
    assert [r["promoted"] for r in log] == ["True", "False", "False"]
    assert "p5_recall" in log[1]["reasons"] and log[2]["mode"] == "never"


def test_read_best_metrics_uses_pose_plus_box_fitness(tmp_path: Path) -> None:
    results = tmp_path / "results.csv"
    with results.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["epoch", "  metrics/mAP50(P)", "  metrics/mAP50-95(P)", "  metrics/mAP50-95(B)"]
        )
        w.writerow([139, 0.95, 0.858, 0.890])  # best pose mAP
        w.writerow([148, 0.95, 0.857, 0.898])  # best pose + box: what best.pt holds
        w.writerow([149, 0.95, 0.857, 0.898])  # tie: the first maximum wins
    best = fk.read_best_metrics(results)
    assert best["best_epoch"] == "148" and best["pose_map50_95"] == 0.857
    assert best["fitness"] == pytest.approx(1.755)
    assert fk.read_best_metrics(tmp_path / "missing.csv")["best_epoch"] == ""


def test_parser_new_subcommands() -> None:
    p = fk.build_parser()
    a = p.parse_args(["evaluate", "-w", "ws"])
    assert a.split == "val" and a.match_px == 25
    a = p.parse_args(["evaluate", "-w", "ws", "--split", "test", "--kp-conf", "0.3"])
    assert a.split == "test" and a.kp_conf == 0.3
    a = p.parse_args(["compare", "-w", "ws", "--candidate", "models/x.pt"])
    assert a.baseline is None and not a.promote
    assert p.parse_args(["models", "-w", "ws", "--default", "l"]).default == "l"
    a = p.parse_args(["grow", "-w", "ws"])
    assert (a.src, a.to, a.out) == ("m", "l", "models/freekiki_l_init.pt")
    a = p.parse_args(["sweep", "-w", "ws", "--eval-dir", "e"])
    assert not a.allow_test
    assert p.parse_args(["audit", "-w", "ws", "--no-hash"]).no_hash
    assert p.parse_args(["bench", "-w", "ws"]).batches == "2,8,16"
    assert p.parse_args(["train", "-w", "ws"]).seed == 0
    assert (
        p.parse_args(["detect", "-w", "ws", "--video", "v.mp4", "--fill-gaps", "3"]).fill_gaps == 3
    )


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


def test_bench_reports_effective_batch_after_oom_reduction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ultralytics halves the batch after OOM; throughput must use the batch trained."""
    import sys
    import types

    ws = fk.init_workspace(tmp_path / "ws")
    fk.import_dataset(ws, _fake_kiki49(tmp_path / "src"))
    clock = iter(float(i) for i in range(1000))
    monkeypatch.setattr("time.perf_counter", lambda: next(clock))

    class FakeYOLO:
        def __init__(self, weights):
            self.cb = None

        def add_callback(self, name, fn):
            assert name == "on_train_batch_end"
            self.cb = fn

        def train(self, **kwargs):
            bs = kwargs["batch"]
            trainer = types.SimpleNamespace(batch_size=bs)
            if bs > 8:  # OOM after 2 batches -> halved, like trainer.py
                for _ in range(2):
                    self.cb(trainer)
                trainer.batch_size = bs // 2
            for _ in range(12):
                self.cb(trainer)

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    out = fk.bench(ws, batches=(8, 16), workers=(2,), warmup=2)
    with (out / "bench.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["status"] for r in rows] == ["ok", "oom_reduced"]
    assert [int(r["batch_effective"]) for r in rows] == [8, 8]
    # 1 s between timed batches -> images_per_s == effective batch, not the requested 16
    assert [float(r["images_per_s"]) for r in rows] == [8.0, 8.0]
    assert [int(r["batches_timed"]) for r in rows] == [9, 9]


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


def test_point_codes_rejection_reasons() -> None:
    xy = np.zeros((fk.NKP, 2))
    xy[:, 0] = np.arange(fk.NKP) * 30.0 + 5.0
    xy[:, 1] = 100.0
    kc = np.full(fk.NKP, 0.9)
    kc[2] = 0.2  # below kp_conf
    xy[3] = [-10.0, 100.0]  # outside the image
    xy[5] = xy[4] + [0.5, 0.0]  # identity clash with p4, lower confidence
    kc[5] = 0.8
    codes = fk.point_codes(xy, kc, 0.9, 0.25, 0.5, (1920, 1080))
    assert codes[:6] == ["D", "D", "Rk", "Ro", "D", "Rd"]
    assert fk.point_codes(xy, kc, 0.1, 0.25, 0.5, (1920, 1080)) == ["Rb"] * fk.NKP
    assert fk.point_codes(None, None, float("nan"), 0.25, 0.5, (1920, 1080)) == ["N"] * fk.NKP
    assert set(fk.POINT_CODES) >= {"D", "N", "Rb", "Rk", "Ro", "Rd", "I"}


def test_video_quality_indicators() -> None:
    kc_on = np.zeros(fk.NKP)
    kc_on[:5] = 0.8
    xy = np.zeros((fk.NKP, 2))
    xy[:5] = [[100, 100], [500, 120], [900, 400], [300, 700], [1200, 800]]
    xy_seq = [xy + [t, 0.0] for t in range(4)] + [None]  # smooth pan, then a miss
    kc_seq = [kc_on] * 4 + [None]
    q = fk.video_quality(xy_seq, kc_seq, 0.5)
    assert q["frames"] == 5 and q["detection_rate"] == 0.8
    assert q["mean_visible_kps"] == 4.0 and q["min4_kps_rate"] == 0.8
    assert q["mean_kp_conf"] == pytest.approx(0.8)
    assert q["displacement_median_px"] == 1.0 and q["residual_median_px"] == 0.0
    assert "calib_ok_rate" not in q and q["cuts"] == 0
    q = fk.video_quality(xy_seq, kc_seq, 0.5, calib=["ok", "ok", "degenerate", "ok", "few_points"])
    assert q["calib_ok_rate"] == 0.6 and q["calib_status"]["degenerate"] == 1
    assert fk.video_quality([], [], 0.5)["residual_median_px"] is None


def test_train_cli_passes_workers_and_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = fk.init_workspace(tmp_path / "ws")
    seen: dict = {}
    monkeypatch.setattr(fk, "train", lambda ws, **kw: seen.update(kw) or {})
    fk.main(["train", "-w", str(ws), "--batch", "8", "--workers", "16", "--seed", "3"])
    assert (seen["batch"], seen["workers"], seen["seed"]) == (8, 16, 3)
    fk.main(["train", "-w", str(ws), "--manifest", "v001"])
    assert seen["manifest"] == "v001"


def _visible_label(on: set[int]) -> str:
    parts = ["0", "0.5", "0.5", "1", "1"]
    for k in range(fk.NKP):
        parts += ["0.2", "0.3", "2"] if k in on else ["0", "0", "0"]
    return " ".join(parts) + "\n"


def _write_rfs_dataset(ds: Path, *, n_common: int = 39) -> None:
    """One train image with only p5 visible, plus ``n_common`` with only p0."""
    names, flip = fk.load_schema()
    rows = ["image,label,split,source,group"]
    for split in ("train", "val"):
        (ds / "images" / split).mkdir(parents=True)
        (ds / "labels" / split).mkdir(parents=True)
    specs = [("rare.jpg", {5})] + [(f"common_{i:02d}.jpg", {0}) for i in range(n_common)]
    for name, vis in specs:
        (ds / "images" / "train" / name).write_bytes(b"jpg")
        label_name = Path(name).with_suffix(".txt").name
        (ds / "labels" / "train" / label_name).write_text(_visible_label(vis), encoding="utf-8")
        rows.append(f"images/train/{name},labels/train/{label_name},train,synth,g")
    (ds / "images" / "val" / "v.jpg").write_bytes(b"jpg")
    (ds / "labels" / "val" / "v.txt").write_text(_visible_label({0}), encoding="utf-8")
    rows.append("images/val/v.jpg,labels/val/v.txt,val,synth,g")
    (ds / "manifest.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    data = {
        "path": str(ds),
        "train": "images/train",
        "val": "images/val",
        "test": "images/val",
        "kpt_shape": [fk.NKP, 3],
        "flip_idx": flip,
        "names": {0: "football_pitch"},
        "kpt_names": {0: names},
    }
    (ds / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_manifest_is_versioned_immutable_and_trains_from_a_copy(tmp_path: Path) -> None:
    ws = fk.init_workspace(tmp_path / "ws")
    ds = fk.dataset_dir(ws)
    _write_rfs_dataset(ds)
    label_before = (ds / "labels" / "train" / "rare.txt").read_bytes()
    exclude = tmp_path / "exclude.txt"
    exclude.write_text("common_00.jpg\n", encoding="utf-8")

    first = fk.make_manifest(ws, t=0.05, cap=4.0, seed=0, exclude=exclude)
    assert first.name == "v001"
    assert (ds / "labels" / "train" / "rare.txt").read_bytes() == label_before
    info = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert info["counts"]["excluded"] == 1
    assert info["counts"]["entries_after"] >= info["counts"]["kept_images"]
    assert info["inputs"]["data_yaml_sha256"]
    listed = (first / "train_list.txt").read_text(encoding="utf-8")
    assert "common_00.jpg" not in listed
    assert "rare.jpg" in listed
    again = fk.make_manifest(ws, t=0.05, cap=4.0, seed=0)
    assert again.name == "v002"
    assert (again / "train_list.txt").read_text(encoding="utf-8") != listed  # exclude differs
    same = tmp_path / "same"
    diag.build_rfs_manifest(ds, same / "a", t=0.05, cap=4, seed=3, log=lambda *_a, **_k: None)
    diag.build_rfs_manifest(ds, same / "b", t=0.05, cap=4, seed=3, log=lambda *_a, **_k: None)
    assert (same / "a" / "train_list.txt").read_bytes() == (
        same / "b" / "train_list.txt"
    ).read_bytes()
    with pytest.raises(FileExistsError):
        fk.make_manifest(ws, name="v001")
    with pytest.raises(ValueError):
        fk.make_manifest(ws, cap=0.5)

    yaml_path, sampling = fk.materialize_manifest(ws, "v001", ws / "runs" / "dry")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    train_txt = Path(data["train"])
    lines = [ln for ln in train_txt.read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) == info["counts"]["entries_after"] == sampling["entries"]
    assert all(ln.startswith(str(ds.resolve())) for ln in lines)
    assert data["val"].endswith("images/val")
    rare_line = next(ln for ln in lines if ln.endswith("rare.jpg"))
    reps = list(csv.DictReader((first / "repeats.csv").open(encoding="utf-8")))
    rare = next(r for r in reps if r["image"].endswith("rare.jpg"))
    # common_00 excluded: f5 = 1/39, r = sqrt(0.05 * 39)
    assert float(rare["repeat_factor"]) == pytest.approx((0.05 * 39) ** 0.5, abs=1e-3)
    assert lines.count(rare_line) == int(rare["copies"])

    (first / "train_list.txt").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(fk.RunStateError):
        fk.materialize_manifest(ws, "v001", ws / "runs" / "dry2")


def test_retrain_from_active_uses_low_adamw_lr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = fk.init_workspace(tmp_path / "ws")
    ds = fk.dataset_dir(ws)
    ds.mkdir(parents=True)
    (ds / "data.yaml").write_text("path: /tmp\ntrain: images/train\n", encoding="utf-8")
    (ws / fk.slot_file("m")).write_bytes(b"checkpoint")
    seen: dict = {}

    def fake_train(*_args, **kwargs) -> None:
        seen.clear()
        seen.update(kwargs)

    monkeypatch.setattr("vaila.yolotrain.train_yolo_dataset", fake_train)
    monkeypatch.setattr(fk, "register_run", lambda *_a, **_k: {})
    fk.train(ws, base="active", epochs=2, imgsz=640, batch=2, name="ft", workers=4, seed=0)
    extra = seen["extra_train_args"]
    assert extra["optimizer"] == "AdamW"
    assert extra["lr0"] == 1e-4
    assert extra["lrf"] == 0.01
    assert extra["warmup_epochs"] == 1.0
    assert extra["cos_lr"] is True
    assert extra["mosaic"] == 0.0
    assert "freeze" not in extra
    assert (extra["workers"], extra["seed"], extra["patience"]) == (4, 0, 30)

    fk.train(ws, base="yolo26n-pose.pt", epochs=1, imgsz=640, batch=2, name="from_coco")
    assert "optimizer" not in seen["extra_train_args"]
    assert seen["model"] == "yolo26n-pose.pt"


def test_ultralytics_train_list_keeps_duplicate_lines(tmp_path: Path) -> None:
    pytest.importorskip("ultralytics")
    from ultralytics.data.base import BaseDataset

    img = tmp_path / "images" / "train" / "a.jpg"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"\xff\xd8\xff\xd9")
    listed = img.resolve().as_posix()
    txt = tmp_path / "train.txt"
    txt.write_text(f"{listed}\n{listed}\n", encoding="utf-8")
    dataset = BaseDataset.__new__(BaseDataset)
    dataset.prefix = ""
    dataset.fraction = 1.0
    files = BaseDataset.get_img_files(dataset, str(txt))
    assert len(files) == 2 and files[0] == files[1]


def _schema1(ws: Path, *, with_file: bool = True) -> None:
    raw = fk.toml.load(ws / fk.CONFIG_NAME)
    raw.pop("models")
    raw["active"] = {"model": fk.LEGACY_ACTIVE_MODEL, "run": "r0", "pose_map50_95": 0.8}
    (ws / fk.CONFIG_NAME).write_text(fk.toml.dumps(raw), encoding="utf-8")
    if with_file:
        (ws / fk.LEGACY_ACTIVE_MODEL).write_bytes(b"old")


def test_schema1_migration_copies_active_into_its_slot(tmp_path: Path) -> None:
    ws = fk.init_workspace(tmp_path / "ws")
    _schema1(ws)
    settings = fk.load_settings(ws)
    assert settings["models"]["default"] == "m"
    assert settings["models"]["m"]["run"] == "r0"
    assert settings["models"]["m"]["sha256"] == fk.file_sha256(ws / fk.LEGACY_ACTIVE_MODEL)
    assert (ws / fk.slot_file("m")).read_bytes() == b"old"
    assert (ws / fk.LEGACY_ACTIVE_MODEL).read_bytes() == b"old"  # copied, never moved
    raw = fk.toml.load(ws / fk.CONFIG_NAME)
    assert "active" not in raw and raw["legacy_active"]["run"] == "r0"
    before = (ws / fk.CONFIG_NAME).read_text(encoding="utf-8")
    assert fk.load_settings(ws)["models"] == settings["models"]  # idempotent
    assert (ws / fk.CONFIG_NAME).read_text(encoding="utf-8") == before
    assert fk.resolve_model(ws, "active").endswith("freekiki_m.pt")

    clash = fk.init_workspace(tmp_path / "clash")
    _schema1(clash)
    (clash / fk.slot_file("m")).write_bytes(b"other")
    with pytest.raises(FileExistsError, match="differs"):
        fk.load_settings(clash)
    assert (clash / fk.slot_file("m")).read_bytes() == b"other"

    empty = fk.init_workspace(tmp_path / "empty")
    _schema1(empty, with_file=False)
    assert fk.load_settings(empty)["models"] == {"default": "m"}


def test_runs_compete_only_inside_their_size_slot(tmp_path: Path) -> None:
    ws = fk.init_workspace(tmp_path / "ws")
    settings = fk.load_settings(ws)
    settings["promotion"]["mode"] = "map"
    fk.save_settings(ws, settings)
    reg = fk.register_run
    assert reg(ws, _fake_run(ws, "m1", 0.40), base="yolo26m-pose.pt", epochs=2, imgsz=640)[
        "promoted"
    ]  # first model of the workspace
    low = reg(ws, _fake_run(ws, "l1", 0.30), base="yolo26l-pose.pt", epochs=2, imgsz=640)
    assert not low["promoted"]  # empty slot, below new_slot_min_pose_map50_95
    high = reg(ws, _fake_run(ws, "l2", 0.60), base="yolo26l-pose.pt", epochs=2, imgsz=640)
    assert high["promoted"]
    assert (ws / fk.slot_file("l")).read_bytes() == b"l2"
    assert (ws / fk.slot_file("m")).read_bytes() == b"m1"
    models = fk.load_settings(ws)["models"]
    assert models["default"] == "m" and models["l"]["arch"] == "yolo26l-pose"
    with (ws / fk.REGISTRY_CSV).open(encoding="utf-8") as f:
        assert [(r["slot"], r["arch"]) for r in csv.DictReader(f)] == [
            ("m", "yolo26m-pose"),
            ("l", "yolo26l-pose"),
            ("l", "yolo26l-pose"),
        ]
    assert [r["slot"] for r in fk.list_models(ws)] == ["m", "l"]
    fk.list_models(ws, default="l")
    assert fk.resolve_model(ws, "active").endswith("freekiki_l.pt")
    with pytest.raises(ValueError, match="no model"):
        fk.list_models(ws, default="x")


def test_any_trained_base_is_a_continued_finetune(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = fk.init_workspace(tmp_path / "ws")
    ds = fk.dataset_dir(ws)
    ds.mkdir(parents=True)
    (ds / "data.yaml").write_text("path: /tmp\ntrain: images/train\n", encoding="utf-8")
    (ws / "models" / "freekiki_l_init.pt").write_bytes(b"grown")
    seen: dict = {}
    monkeypatch.setattr(
        "vaila.yolotrain.train_yolo_dataset", lambda *_a, **k: seen.update(k) or None
    )
    monkeypatch.setattr(fk, "register_run", lambda *_a, **_k: {})
    fk.train(ws, base="models/freekiki_l_init.pt", epochs=1, imgsz=64, batch=2, name="g")
    assert seen["extra_train_args"]["lr0"] == 1e-4
    assert seen["extra_train_args"]["mosaic"] == 0.0
    seen.clear()
    fk.train(ws, base="yolo26l-pose.pt", epochs=1, imgsz=64, batch=2, name="o")
    assert "lr0" not in seen["extra_train_args"]


def test_grow_m_to_l_keeps_the_function(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from vaila import freekiki_sizes as fs

    class Like:
        yaml = {"kpt_shape": [49, 3], "nc": 1}
        nc = 1

    src = tmp_path / "m.pt"
    torch.save({"model": fs.build("m", Like())}, src)
    stats = fs.grow(src, tmp_path / "l.pt")
    assert stats["padded"] > 0 and stats["identity_blocks"] > 0
    assert stats["max_diff"] < 1e-3
    assert fk.checkpoint_slot(tmp_path / "l.pt") == ("l", "yolo26l-pose")
    from ultralytics import YOLO

    assert YOLO(str(tmp_path / "l.pt")).task == "pose"  # "detect" reads pose labels as corrupt
    with pytest.raises(FileExistsError):
        fs.grow(src, tmp_path / "l.pt")
    with pytest.raises(ValueError, match="widths differ"):
        fs.grow(tmp_path / "l.pt", tmp_path / "x.pt")
