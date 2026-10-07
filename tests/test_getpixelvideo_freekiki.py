"""FreeKiki human review, completeness gate, label queue and train/hard ingest.

Version: 0.4.7
Update Date: 03 October 2026
"""

import csv
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from vaila import freekiki, freekiki_diag


def _field_points(w: float = 48, h: float = 32) -> list[list[float]]:
    """Every kiki49 point of a synthetic full-pitch view (flag/post tops 1 px above)."""
    _, _, xyz = freekiki_diag.load_field_points()
    return [
        [w / 2 + x * 0.8 * w / 105, h / 2 - y * 0.8 * h / 68 - (1.0 if z else 0.0)]
        for x, y, z in xyz
    ]


def _complete(session: dict, frame: int, conf=None) -> None:
    """AI draft with every point of the synthetic view (a complete frame)."""
    freekiki.apply_review_prediction(
        session, frame, _field_points(), conf or [0.9] * 49, "model-hash"
    )


def _workspace(root: Path) -> Path:
    ws = freekiki.init_workspace(root)
    ds = ws / "datasets/kiki49"
    names, flips = freekiki.load_schema()
    (ds / "images/val").mkdir(parents=True)
    (ds / "images/test").mkdir(parents=True)
    (ds / "labels/val").mkdir(parents=True)
    (ds / "labels/test").mkdir(parents=True)
    (ds / "images/train").mkdir(parents=True)
    (ds / "labels/train").mkdir(parents=True)
    (ds / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(ds),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "kpt_shape": [49, 3],
                "flip_idx": flips,
                "kpt_names": names,
                "names": {0: "football_pitch"},
            }
        )
    )
    rows = []
    for split, seed in (("val", 1), ("test", 2)):
        image = np.random.default_rng(seed).integers(0, 255, (32, 48, 3), dtype=np.uint8)
        cv2.imwrite(str(ds / f"images/{split}/{split}.png"), image)
        pts = [[10, 10]] + [None] * 48
        (ds / f"labels/{split}/{split}.txt").write_text(freekiki.pose_label_line(pts, 48, 32))
        rows.append(
            {
                "split": split,
                "image": f"images/{split}/{split}.png",
                "label": f"labels/{split}/{split}.txt",
                "source": "fixture",
                "group": split,
                "origin": "fixture",
                "n_visible": "1",
                "aux3d": "0",
                "qa_score": "1",
            }
        )
    with (ds / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return ws


class _FrameCapture:
    def __init__(self, *_):
        self.frame = 0

    def isOpened(self):  # noqa: N802 - OpenCV VideoCapture API
        return True

    def set(self, _prop, frame):
        self.frame = int(frame)

    def read(self):
        image = np.random.default_rng(self.frame).integers(0, 255, (32, 48, 3), dtype=np.uint8)
        return True, image

    def release(self):
        pass

    def get(self, prop):
        return {
            cv2.CAP_PROP_FRAME_WIDTH: 48,
            cv2.CAP_PROP_FRAME_HEIGHT: 32,
            cv2.CAP_PROP_FRAME_COUNT: 20000,
            cv2.CAP_PROP_FPS: 30.0,
        }[prop]


def test_review_export_ingest_and_reopen(tmp_path, monkeypatch):
    ws = _workspace(tmp_path / "workspace")
    ds = ws / "datasets/kiki49"
    reserved = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for kind in ("images", "labels")
        for split in ("val", "test")
        for p in (ds / kind / split).iterdir()
    }
    yaml_before = (ds / "data.yaml").read_bytes()
    video = tmp_path / "match_a.mp4"
    video.write_bytes(b"video fixture")
    session = freekiki.new_review_session(video, 48, 32, 30, workspace=ws)
    _complete(session, 17608)
    _complete(session, 17609)
    x0, y0 = _field_points()[0]
    freekiki.edit_review_point(session, 17608, 0, [x0 + 0.2, y0])
    freekiki.mark_reviewed(session, 17608)
    session_path = freekiki.save_review_session(session)
    reopened = freekiki.load_review_session(session_path, video, 48, 32, 30)
    assert reopened["frames"]["17608"]["state"] == "HUMAN_REVIEWED"
    assert reopened["frames"]["17609"]["state"] == "AI_DRAFT"

    monkeypatch.setattr(cv2, "VideoCapture", _FrameCapture)
    out = freekiki.export_reviewed_session(reopened)
    images = list((out / "images").glob("*.png"))
    labels = list((out / "labels").glob("*.txt"))
    assert len(images) == len(labels) == 1
    assert "f00017608" in images[0].name
    assert len(labels[0].read_text().split()) == 152
    xy, vis = freekiki.read_label_keypoints(labels[0])
    assert vis[0] == 2 and np.count_nonzero(vis) == 49
    assert xy[0, 0] == pytest.approx((x0 + 0.2) / 48, abs=1e-7)
    assert freekiki.ingest_reviewed(ws, out, "MATCH_A")["committed"] is False
    assert not list((ds / "images/train").iterdir())
    assert freekiki.ingest_reviewed(ws, out, "MATCH_A", commit=True)["frames"] == 1
    assert freekiki.ingest_reviewed(ws, out, "MATCH_A", commit=True)["frames"] == 0
    with pytest.raises(ValueError, match="already ingested as train/MATCH_A"):
        freekiki.ingest_reviewed(ws, out, "DIFFERENT_MATCH", commit=True)
    assert freekiki.check_dataset(ws) == []
    assert all(
        hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in reserved.items()
    )
    assert (ds / "data.yaml").read_bytes() == yaml_before
    assert (out / "human_vs_ai.csv").is_file()


def test_review_guards_and_match_leakage(tmp_path, monkeypatch):
    ws = _workspace(tmp_path / "workspace")
    video = tmp_path / "match.mp4"
    video.write_bytes(b"video")
    session = freekiki.new_review_session(video, 48, 32, 30, workspace=ws)
    _complete(session, 4)
    with pytest.raises(ValueError, match="No complete human-reviewed"):
        freekiki.export_reviewed_session(session)
    x1, y1 = _field_points()[1]
    freekiki.edit_review_point(session, 4, 1, [x1 + 0.1, y1])
    freekiki.mark_reviewed(session, 4)
    assert session["frames"]["4"]["state"] == "HUMAN_REVIEWED"
    freekiki.edit_review_point(session, 4, 1, [x1 + 0.2, y1])
    assert session["frames"]["4"]["state"] == "DRAFT_MANUAL"
    freekiki.mark_reviewed(session, 4)
    monkeypatch.setattr(cv2, "VideoCapture", _FrameCapture)
    out = freekiki.export_reviewed_session(session)
    with pytest.raises(ValueError, match="another split"):
        freekiki.ingest_reviewed(ws, out, "val", commit=True)
    with pytest.raises(ValueError, match="video identity"):
        freekiki.load_review_session(out / "session.json", tmp_path / "other.mp4")


def test_pose_line_rejects_invalid_and_has_tiny_bbox():
    line = freekiki.pose_label_line([[0, 0]] + [None] * 48, 48, 32)
    assert len(line.split()) == 152
    assert float(line.split()[3]) > 0 and float(line.split()[4]) > 0
    with pytest.raises(ValueError, match="outside image"):
        freekiki.pose_label_line([[49, 0]] + [None] * 48, 48, 32)


def test_same_frame_from_two_videos_has_distinct_names(tmp_path, monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", _FrameCapture)
    names = []
    for folder in ("camera_a", "camera_b"):
        video = tmp_path / folder / "clip.mp4"
        video.parent.mkdir()
        video.write_bytes(folder.encode())
        session = freekiki.new_review_session(video, 48, 32, 30, workspace=tmp_path)
        _complete(session, 17608)
        freekiki.mark_reviewed(session, 17608)
        out = freekiki.export_reviewed_session(session)
        names.append(next((out / "images").glob("*.png")).name)
    assert names[0] != names[1]


def test_ingest_recovers_after_copy_interruption(tmp_path, monkeypatch):
    ws = _workspace(tmp_path / "workspace")
    ds = ws / "datasets/kiki49"
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"clip")
    session = freekiki.new_review_session(video, 48, 32, 30, workspace=ws)
    _complete(session, 8)
    freekiki.mark_reviewed(session, 8)
    monkeypatch.setattr(cv2, "VideoCapture", _FrameCapture)
    out = freekiki.export_reviewed_session(session)
    manifest_before = (ds / "manifest.csv").read_bytes()
    original_link = os.link
    calls = 0

    def interrupted_link(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated interruption")
        return original_link(source, target)

    monkeypatch.setattr(freekiki.os, "link", interrupted_link)
    with pytest.raises(OSError, match="simulated interruption"):
        freekiki.ingest_reviewed(ws, out, "MATCH", commit=True)
    assert (ds / "manifest.csv").read_bytes() == manifest_before
    monkeypatch.setattr(freekiki.os, "link", original_link)
    assert freekiki.ingest_reviewed(ws, out, "MATCH", commit=True)["frames"] == 1
    assert freekiki.check_dataset(ws) == []


def test_ingest_rejects_reserved_near_duplicate(tmp_path, monkeypatch):
    ws = _workspace(tmp_path / "workspace")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"clip")
    session = freekiki.new_review_session(video, 48, 32, 30, workspace=ws)
    _complete(session, 8)
    freekiki.mark_reviewed(session, 8)
    reserved_frame = cv2.imread(str(ws / "datasets/kiki49/images/val/val.png"))

    class _DuplicateCapture(_FrameCapture):
        def read(self):
            return True, reserved_frame.copy()

    monkeypatch.setattr(cv2, "VideoCapture", _DuplicateCapture)
    out = freekiki.export_reviewed_session(session)
    with pytest.raises(ValueError, match="Near duplicate"):
        freekiki.ingest_reviewed(ws, out, "NEW_MATCH", commit=True)
    image = next((out / "images").glob("*.png"))
    image.write_bytes(image.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="changed after review"):
        freekiki.ingest_reviewed(ws, out, "NEW_MATCH", commit=True)


def test_import_raw_predictions_keeps_original_model_hash(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"clip")
    session = freekiki.new_review_session(video, 48, 32, 30)
    output = tmp_path / "detect"
    output.mkdir()
    (output / "README.txt").write_text(
        f"video: {video.resolve()}\nmodel: {tmp_path / 'promoted.pt'}\n"
        "model_sha256: original-model-hash\ndimensions: 48x32\n"
    )
    fields = ["frame", "box_conf"] + [
        f"p{i}_{field}" for i in range(49) for field in ("x", "y", "conf")
    ]
    row = {"frame": "17608", "box_conf": "0.9"}
    for i in range(49):
        row.update({f"p{i}_x": "10", f"p{i}_y": "10", f"p{i}_conf": "0.8"})
    with (output / "field_kps_raw.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    monkeypatch.setattr(cv2, "VideoCapture", _FrameCapture)
    assert freekiki.load_raw_review_predictions(session, output) == 1
    draft = session["frames"]["17608"]
    assert draft["state"] == "AI_DRAFT"
    assert draft["model_sha256"] == "original-model-hash"
    assert draft["points"][0] == [10, 10]


def test_ingest_cli_requires_explicit_commit_flag():
    parser = freekiki.build_parser()
    preview = parser.parse_args(["ingest", "-w", "WS", "--src", "SESSION", "--match-id", "MATCH"])
    committed = parser.parse_args(
        ["ingest", "-w", "WS", "--src", "SESSION", "--match-id", "MATCH", "--commit"]
    )
    assert preview.commit is False
    assert committed.commit is True


def test_raw_predictions_resolve_video_inside_batch_folder(tmp_path, monkeypatch):
    fields = ["frame", "box_conf"] + [
        f"p{i}_{field}" for i in range(49) for field in ("x", "y", "conf")
    ]
    batch = tmp_path / "processed_freekiki_batch_20260928_163632"
    videos = {}
    for stem, x in (("clip_a", "10"), ("clip_b", "20")):
        video = tmp_path / f"{stem}.mp4"
        video.write_bytes(stem.encode())
        videos[stem] = video
        output = batch / f"processed_freekiki_{stem}_20260928_163633"
        output.mkdir(parents=True)
        (output / "README.txt").write_text(
            f"video: {video.resolve()}\nmodel_sha256: hash-{stem}\ndimensions: 48x32\n"
        )
        row = {"frame": "3", "box_conf": "0.9"}
        for i in range(49):
            row.update({f"p{i}_x": x, f"p{i}_y": "10", f"p{i}_conf": "0.8"})
        with (output / "field_kps_raw.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)
    monkeypatch.setattr(cv2, "VideoCapture", _FrameCapture)
    session = freekiki.new_review_session(videos["clip_b"], 48, 32, 30)
    assert freekiki.load_raw_review_predictions(session, batch) == 1
    draft = session["frames"]["3"]
    assert draft["model_sha256"] == "hash-clip_b"
    assert draft["points"][0] == [20, 10]
    other = tmp_path / "clip_c.mp4"
    other.write_bytes(b"clip_c")
    with pytest.raises(ValueError, match="No detect output for clip_c.mp4"):
        freekiki.load_raw_review_predictions(freekiki.new_review_session(other, 48, 32, 30), batch)


def test_edit_review_point_clamps_subpixel_drift_and_handles_nan(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"clip")
    session = freekiki.new_review_session(video, 48, 32, 30)

    # Negative coordinate (drift slightly left of 0) clamped to 0.0
    freekiki.edit_review_point(session, 1, 0, [-0.6163528, 15.5])
    assert session["frames"]["1"]["points"][0] == [0.0, 15.5]

    # Coordinate exceeding frame dimension clamped to width/height
    freekiki.edit_review_point(session, 1, 1, [50.2, 35.1])
    assert session["frames"]["1"]["points"][1] == [48.0, 32.0]

    # NaN / Inf gracefully converted to None (absent)
    freekiki.edit_review_point(session, 1, 2, [float("nan"), 10.0])
    assert session["frames"]["1"]["points"][2] is None
    assert session["frames"]["1"]["point_sources"][2] == "absent"


def test_completeness_gate_flags_missing_points_until_labelled_or_hidden(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"clip")
    session = freekiki.new_review_session(video, 48, 32, 30)
    _complete(session, 1)
    freekiki.edit_review_point(session, 1, 5, None)
    with pytest.raises(ValueError, match="p5 bottom_left_corner probably visible"):
        freekiki.mark_reviewed(session, 1)
    session["frames"]["1"]["hidden"] = [5]  # Del in getpixelvideo: confirmed not visible
    freekiki.mark_reviewed(session, 1)
    assert session["frames"]["1"]["state"] == "HUMAN_REVIEWED"

    freekiki.edit_review_point(session, 2, 0, _field_points()[0])
    with pytest.raises(ValueError, match="not verifiable \\(few_points"):
        freekiki.mark_reviewed(session, 2)


def test_export_keeps_only_complete_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", _FrameCapture)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"clip")
    session = freekiki.new_review_session(video, 48, 32, 30, workspace=tmp_path)
    _complete(session, 10)
    freekiki.mark_reviewed(session, 10)
    # A frame reviewed by an older version: one point only.
    freekiki.edit_review_point(session, 11, 0, _field_points()[0])
    session["frames"]["11"]["state"] = "HUMAN_REVIEWED"
    out = freekiki.export_reviewed_session(session)
    assert [p.name[-13:] for p in (out / "images").glob("*.png")] == ["f00000010.png"]
    assert session["frames"]["11"]["state"] == "DRAFT_MANUAL"
    with (out / "incomplete_frames.csv").open() as f:
        assert [r["frame"] for r in csv.DictReader(f)] == ["11"]


def test_ingest_rejects_incomplete_frames_of_an_old_export(tmp_path, monkeypatch):
    ws = _workspace(tmp_path / "workspace")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"clip")
    session = freekiki.new_review_session(video, 48, 32, 30, workspace=ws)
    _complete(session, 8)
    freekiki.mark_reviewed(session, 8)
    monkeypatch.setattr(cv2, "VideoCapture", _FrameCapture)
    out = freekiki.export_reviewed_session(session)
    old = freekiki.load_review_session(out / "session.json")
    old["frames"]["8"]["points"][5] = None
    freekiki.save_review_session(old)
    with pytest.raises(ValueError, match="1 incomplete frame"):
        freekiki.ingest_reviewed(ws, out, "MATCH")


def test_hard_split_is_a_holdout_never_shared_with_train(tmp_path, monkeypatch):
    ws = _workspace(tmp_path / "workspace")
    ds = ws / "datasets/kiki49"
    monkeypatch.setattr(cv2, "VideoCapture", _FrameCapture)
    video = tmp_path / "hard_clip.mp4"
    video.write_bytes(b"hard")
    session = freekiki.new_review_session(video, 48, 32, 30, workspace=ws)
    _complete(session, 8)
    freekiki.mark_reviewed(session, 8)
    out = freekiki.export_reviewed_session(session)
    report = freekiki.ingest_reviewed(ws, out, "PALXSCRI", split="hard", commit=True)
    assert report["frames"] == 1 and report["split"] == "hard"
    assert len(list((ds / "images/hard").glob("*.png"))) == 1
    assert not list((ds / "images/train").iterdir())
    assert yaml.safe_load((ds / "data.yaml").read_text())["hard"] == "images/hard"
    with (ds / "manifest.csv").open() as f:
        assert [r["split"] for r in csv.DictReader(f)][-1] == "hard"
    assert not [i for i in freekiki.check_dataset(ws) if i.startswith("hard")]  # train is empty

    # The same footage copied elsewhere cannot reach train.
    copy = tmp_path / "copy" / "hard_clip_copy.mp4"
    copy.parent.mkdir()
    copy.write_bytes(b"copy")
    session = freekiki.new_review_session(copy, 48, 32, 30, workspace=ws)
    _complete(session, 8)
    freekiki.mark_reviewed(session, 8)
    out = freekiki.export_reviewed_session(session)
    with pytest.raises(ValueError, match="Near duplicate"):
        freekiki.ingest_reviewed(ws, out, "OTHER_MATCH")
    with pytest.raises(ValueError, match="another split"):
        freekiki.ingest_reviewed(ws, out, "PALXSCRI")


def test_review_suggestions_ai_geometry_and_both(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"clip")
    session = freekiki.new_review_session(video, 48, 32, 30)
    xy = _field_points()
    xy[29] = [xy[29][0] - 5, xy[29][1]]  # AI puts p29 5 px (200 px@1920) off: geometry wins
    conf = [0.9] * 49
    for i in (5, 29, 39):
        conf[i] = 0.2  # below kp_conf: not drafted, kept as a ghost
    freekiki.apply_review_prediction(session, 3, xy, conf, "hash")
    hints = freekiki.review_suggestions(session, 3)
    by_index = {g["index"]: g for g in hints["suggestions"]}
    assert set(by_index) == {5, 29, 39}
    assert by_index[5]["source"] == "ai+geometry" and by_index[5]["conf"] == 0.2
    assert by_index[29]["source"] == "geometry"
    assert by_index[29]["xy"] == pytest.approx(_field_points()[29])
    assert by_index[39]["source"] == "ai"  # flag top: no ground projection
    assert hints["status"] == "incomplete" and set(hints["suspects"]) == {5, 29, 39}


def test_queue_frames_priority_gap_and_cap():
    status = [
        {"frame": str(f), "homography": "few_points" if f < 10 else "ok", "n_accepted": "2"}
        if f < 10
        else {"frame": str(f), "homography": "ok", "n_accepted": "12"}
        for f in range(40)
    ]
    raw = [{"frame": str(f), "p5_conf": "0.2" if 20 <= f < 30 else "0.9"} for f in range(40)]
    picked = freekiki.queue_frames(status, raw, per_video=6, min_gap=3, kp_conf=0.5)
    frames = [r["frame"] for r in picked]
    assert len(frames) == 6
    assert all(b - a >= 3 for a, b in zip(frames, frames[1:], strict=False))
    tiers = [r["tier"] for r in picked]
    assert tiers.count(0) >= 3 and set(tiers) <= {0, 1}
    assert all(r["reason"].startswith("low-confidence rare p5") for r in picked if r["tier"] == 1)


def test_build_label_queue_drafts_only_queued_frames(tmp_path, monkeypatch):
    ws = _workspace(tmp_path / "workspace")
    monkeypatch.setattr(cv2, "VideoCapture", _FrameCapture)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"clip")
    out = tmp_path / "processed_freekiki_batch_x" / "processed_freekiki_clip_1"
    out.mkdir(parents=True)
    (out / "README.txt").write_text(
        f"video: {video.resolve()}\nmodel_sha256: hash\ndimensions: 48x32\n"
    )
    pts = _field_points()
    fields = ["frame", "box_conf"] + [f"p{i}_{k}" for i in range(49) for k in ("x", "y", "conf")]
    with (out / "field_kps_raw.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for frame in range(30):
            row = {"frame": frame, "box_conf": 0.9}
            for i, (x, y) in enumerate(pts):
                row |= {f"p{i}_x": x, f"p{i}_y": y, f"p{i}_conf": 0.9}
            writer.writerow(row)
    with (out / "field_kps_status.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "n_accepted", "homography"])
        writer.writeheader()
        for frame in range(30):
            writer.writerow({"frame": frame, "n_accepted": 3, "homography": "few_points"})
    (path,) = freekiki.build_label_queue(ws, out.parent, per_video=4, min_gap=5)
    session = freekiki.load_review_session(path)
    drafted = sorted(int(k) for k, r in session["frames"].items() if r["state"] == "AI_DRAFT")
    assert len(drafted) == 4
    assert path.parent.parent == ws / "incoming"
    with (path.parent / "queue.csv").open() as f:
        assert sorted(int(r["frame"]) for r in csv.DictReader(f)) == drafted


def test_audit_lists_missing_label_suspects(tmp_path):
    ws = _workspace(tmp_path / "workspace")
    ds = ws / "datasets/kiki49"
    data = yaml.safe_load((ds / "data.yaml").read_text())
    (ds / "data.yaml").write_text(yaml.safe_dump(data | {"kpt_names": {0: data["kpt_names"]}}))
    pts: list = _field_points()
    pts[5] = None  # bottom_left_corner visible in the view but labelled "not visible"
    cv2.imwrite(str(ds / "images/train/t.png"), np.zeros((32, 48, 3), np.uint8))
    (ds / "labels/train/t.txt").write_text(freekiki.pose_label_line(pts, 48, 32))
    with (ds / "manifest.csv").open("a", newline="") as f:
        f.write("train,images/train/t.png,labels/train/t.txt,fixture,g_train,fixture,48,0,1\n")
    summary = freekiki_diag.audit_dataset(ds, tmp_path / "audit", hash_images=False, log=print)
    with (tmp_path / "audit/label_missing_suspects.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [(r["image"], r["kp"]) for r in rows] == [("images/train/t.png", "p5")]
    assert summary["labels"]["missing_label_suspects"]["by_kp_split"] == {"p5/train": 1}


def test_cli_queue_ingest_split_and_evaluate_hard():
    parser = freekiki.build_parser()
    args = parser.parse_args(["queue", "-w", "WS", "--batch", "B", "--per-video", "30"])
    assert (args.batch_dir, args.per_video, args.min_gap) == (["B"], 30, None)
    args = parser.parse_args(
        ["queue", "-w", "WS", "--batch", "B1", "B2", "--need", "p5,p29", "--montage"]
    )
    assert (args.batch_dir, args.need, args.montage, args.per_video) == (
        ["B1", "B2"],
        "p5,p29",
        True,
        None,
    )
    assert parser.parse_args(["ingest", "-w", "WS", "--src", "S"]).match_id is None
    args = parser.parse_args(
        ["ingest", "-w", "WS", "--src", "S", "--match-id", "M", "--split", "hard"]
    )
    assert args.split == "hard" and args.commit is False
    assert parser.parse_args(["evaluate", "-w", "WS", "--split", "hard"]).split == "hard"


def test_detect_output_dir_recognises_folder_and_its_csv(tmp_path):
    run = tmp_path / "processed_freekiki_clip_1"
    run.mkdir()
    (run / "README.txt").write_text("FreeKiki field keypoints (vailá)\nvideo: /v.mp4\n")
    (run / "field_kps_getpixelvideo.csv").write_text("frame\n")
    assert freekiki.detect_output_dir(run) == run.resolve()
    assert freekiki.detect_output_dir(run / "field_kps_getpixelvideo.csv") == run.resolve()
    plain = tmp_path / "markers.csv"
    plain.write_text("frame\n")
    assert freekiki.detect_output_dir(plain) is None


def test_save_dataset_folder_then_train_ingests_it(tmp_path, monkeypatch):
    ws = _workspace(tmp_path / "workspace")
    monkeypatch.setattr(cv2, "VideoCapture", _FrameCapture)
    video = tmp_path / "Australia vs Brazil - clip.mp4"
    video.write_bytes(b"clip")
    session = freekiki.new_review_session(video, 48, 32, 30)
    _complete(session, 8)
    freekiki.mark_reviewed(session, 8)
    folder = tmp_path / "freekiki_corrections_clip"
    freekiki.relocate_review_session(session, folder)
    assert session["session_id"] == folder.name and (folder / "session.json").is_file()
    out = freekiki.export_reviewed_session(session)
    assert out == folder.resolve() and (folder / "data.yaml").is_file()
    assert len(list((folder / "labels").glob("*.txt"))) == 1
    freekiki.relocate_review_session(session, folder)  # its own folder: fine
    other = tmp_path / "busy"
    other.mkdir()
    (other / "x.txt").write_text("x")
    with pytest.raises(ValueError, match="empty folder"):
        freekiki.relocate_review_session(session, other)
    assert freekiki.default_match_id(session) == "australia_vs_brazil_clip"

    calls = []
    monkeypatch.setattr(freekiki, "train", lambda *a, **k: calls.append("train"))
    real_add = freekiki.add_corrections
    monkeypatch.setattr(
        freekiki, "add_corrections", lambda *a, **k: calls.append("add") or real_add(*a, **k)
    )
    freekiki.main(["train", "-w", str(ws), "--add-dataset", str(folder), "--epochs", "1"])
    assert calls == ["add", "train"]
    ds = ws / "datasets/kiki49"
    with (ds / "manifest.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "train"]
    assert [r["group"] for r in rows] == ["australia_vs_brazil_clip"]
    with pytest.raises(ValueError, match="manifest is a fixed train list"):
        freekiki.main(["train", "-w", str(ws), "--add-dataset", str(folder), "--manifest", "v001"])


def test_train_stops_when_a_correction_is_incomplete(tmp_path, monkeypatch):
    ws = _workspace(tmp_path / "workspace")
    monkeypatch.setattr(cv2, "VideoCapture", _FrameCapture)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"clip")
    session = freekiki.new_review_session(video, 48, 32, 30)
    _complete(session, 8)
    freekiki.mark_reviewed(session, 8)
    folder = tmp_path / "corr"
    freekiki.relocate_review_session(session, folder)
    freekiki.export_reviewed_session(session)
    old = freekiki.load_review_session(folder / "session.json")
    old["frames"]["8"]["points"][5] = None
    freekiki.save_review_session(old)
    monkeypatch.setattr(freekiki, "train", lambda *a, **k: pytest.fail("must not train"))
    with pytest.raises(ValueError, match="incomplete"):
        freekiki.main(["train", "-w", str(ws), "--add-dataset", str(folder)])


def _detect_run(folder: Path, video: Path, model: str = "") -> Path:
    folder.mkdir(parents=True)
    text = f"FreeKiki field keypoints (vailá)\nvideo: {video}\n"
    (folder / "README.txt").write_text(text + (f"model: {model}\n" if model else ""))
    (folder / "field_kps_getpixelvideo.csv").write_text("frame\n")
    return folder


def test_detect_runs_single_file_and_batch_newest_per_video(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    batch = tmp_path / "processed_freekiki_batch_1"
    old_a = _detect_run(batch / "processed_freekiki_a_20260930_060000", a)
    new_a = _detect_run(batch / "processed_freekiki_a_20260930_070000", a)
    run_b = _detect_run(batch / "processed_freekiki_b_20260930_060000", b)
    (batch / "quality_summary.csv").write_text("video\n")
    assert freekiki.detect_runs(old_a) == [(old_a.resolve(), a.resolve())]
    assert freekiki.detect_runs(run_b / "field_kps_getpixelvideo.csv") == [
        (run_b.resolve(), b.resolve())
    ]
    expected = [(new_a.resolve(), a.resolve()), (run_b.resolve(), b.resolve())]
    assert freekiki.detect_runs(batch) == expected
    assert freekiki.detect_runs(batch / "quality_summary.csv") == expected
    assert freekiki.detect_runs(tmp_path / "a.mp4") == []


def test_detect_runs_finds_freekiki_predict_and_prefers_newer_stamp(tmp_path):
    video = tmp_path / "mirassol.mp4"
    video.write_bytes(b"v")
    batch = tmp_path / "processed_freekiki_batch_1"
    old = _detect_run(batch / "processed_freekiki_mirassol_20260930_060000", video)
    new = _detect_run(batch / "freekiki_predict_mirassol_20260930_135426", video)
    assert freekiki.detect_runs(new) == [(new.resolve(), video.resolve())]
    assert freekiki.detect_runs(batch) == [(new.resolve(), video.resolve())]
    assert old.is_dir()


def test_run_video_follows_a_moved_video_and_run_options(tmp_path):
    ws = freekiki.init_workspace(tmp_path / "ws")
    model = ws / "models" / "freekiki_m.pt"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"pt")
    moved = tmp_path / "videos" / "clip.mp4"
    moved.parent.mkdir()
    moved.write_bytes(b"v")
    run = _detect_run(
        moved.parent / "batch" / "processed_freekiki_clip_1", tmp_path / "gone" / "clip.mp4", model
    )
    assert freekiki.run_video(run) == moved.resolve()
    assert freekiki.freekiki_run_options(run) == {
        "workspace": str(ws),
        "predictions": str(run),
        "session": None,
    }
    lost = _detect_run(tmp_path / "processed_freekiki_lost_1", tmp_path / "nowhere" / "x.mp4")
    with pytest.raises(ValueError, match="Original video not found: x.mp4"):
        freekiki.run_video(lost)
    assert freekiki.detect_runs(tmp_path) == []  # the lost run is skipped, not fatal


def test_freekiki_panel_background_is_translucent():
    import pygame

    from vaila.getpixelvideo import FREEKIKI_PANEL_ALPHA, blit_translucent

    screen = pygame.Surface((20, 10))
    screen.fill((200, 200, 200))  # the video behind the panel
    blit_translucent(screen, pygame.Rect(0, 0, 10, 10), (0, 0, 0), FREEKIKI_PANEL_ALPHA)
    covered, free = screen.get_at((5, 5)), screen.get_at((15, 5))
    assert 0 < covered.r < free.r == 200  # darker but the image still shows through
    assert 0 < FREEKIKI_PANEL_ALPHA < 255


def test_find_corrections_after_relocation_and_reopen(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    run = _detect_run(tmp_path / "detect", video)
    session = freekiki.new_review_session(video, 48, 32, 30)
    session["prediction_source"] = str(run)
    _complete(session, 8)
    freekiki.edit_review_point(session, 8, 5, None)
    session["frames"]["8"]["hidden"] = [5]
    session["cursor"] = {"frame": 8, "point": 5}
    freekiki.save_review_session(session)
    folder = tmp_path / "elsewhere" / "corrections"
    freekiki.relocate_review_session(session, folder)
    found = freekiki.find_review_session(video, run / "field_kps_getpixelvideo.csv")
    assert found == folder / "session.json"
    restored = freekiki.load_review_session(found, video, 48, 32, 30)
    assert restored["cursor"] == {"frame": 8, "point": 5}
    assert restored["frames"]["8"]["points"][5] is None
    assert restored["frames"]["8"]["hidden"] == [5]
    (folder / "data.yaml").write_text("kpt_shape: [49, 3]\n")
    assert freekiki.find_review_session(video, folder / "data.yaml") == found
    other = tmp_path / "other.mp4"
    other.write_bytes(b"other")
    assert freekiki.find_review_session(other, run) is None


def test_delete_key_executes_selected_frame_only_with_undo():
    import ast
    from types import SimpleNamespace

    import pygame

    # Execute the actual event branch without starting the interactive video loop.
    tree = ast.parse((Path(__file__).parents[1] / "vaila/getpixelvideo.py").read_text())
    branch = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If) and "pygame.K_DELETE" in ast.unparse(n.test)
    )
    undo = []
    synced = []
    deleted = {8: set(), 9: set()}
    env = {
        "pygame": pygame,
        "event": SimpleNamespace(key=pygame.K_DELETE),
        "freekiki_session": {},
        "_require_insert": lambda: True,
        "selected_marker_idx": 5,
        "frame_count": 8,
        "_push_undo": lambda: undo.append({f: set(v) for f, v in deleted.items()}),
        "deleted_positions": deleted,
        "_review_sync": synced.append,
    }
    code = compile(
        ast.fix_missing_locations(ast.Module(body=branch.body, type_ignores=[])), "delete", "exec"
    )
    exec(code, env)
    assert deleted == {8: {5}, 9: set()}
    assert synced == [8] and undo == [{8: set(), 9: set()}]
    assert env["selected_marker_idx"] == 5
    env["_require_insert"] = lambda: False
    exec(code, env)
    assert len(undo) == 1


def test_resume_restores_editor_grid_and_cursor(tmp_path):
    import ast

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    session = freekiki.new_review_session(video, 48, 32, 30)
    _complete(session, 8)
    session["frames"]["8"]["hidden"] = [5]
    session["cursor"] = {"frame": 8, "point": 17}
    path = freekiki.save_review_session(session)
    tree = ast.parse((Path(__file__).parents[1] / "vaila/getpixelvideo.py").read_text())
    names = {"_resume_freekiki", "_coordinates_from_session", "_enter_correction_mode"}
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    # Lift the real closures into a shared namespace for a headless integration test.
    for func in funcs:
        func.body = [
            ast.Global(names=n.names) if isinstance(n, ast.Nonlocal) else n for n in func.body
        ]
    grid, hidden = {1: [(999, 999)]}, {1: {0}}
    env = {
        "__name__": "vaila.getpixelvideo",
        "__package__": "vaila",
        "Path": Path,
        "video_path": str(video),
        "original_width": 48,
        "original_height": 32,
        "fps": 30,
        "total_frames": 20,
        "freekiki_session": {},
        "freekiki_api": freekiki,
        "coordinates": grid,
        "deleted_positions": hidden,
        "editor_mode": "insert",
        "frame_count": 0,
        "selected_marker_idx": 0,
        "_review_hints": lambda: {"suspects": [5]},
        "_refresh_restore_snapshot": lambda: None,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[])), "resume", "exec"
        ),
        env,
    )
    env["_resume_freekiki"](path)
    assert env["frame_count"] == 8 and env["selected_marker_idx"] == 17
    assert env["paused"] is True
    assert grid[1] == [] and hidden[1] == set()
    assert grid[8][0] == tuple(session["frames"]["8"]["points"][0])
    assert hidden[8] == {5}


def test_export_reviewed_session_full_vs_only_correct_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", _FrameCapture)
    video = tmp_path / "match_clip.mp4"
    video.write_bytes(b"match_clip_bytes")
    session = freekiki.new_review_session(video, 48, 32, 30)

    # Frame 2: AI draft only (uncorrected, not human-reviewed)
    pts_ai = [[12.0, 14.0]] + [None] * 48
    conf_ai = [0.85] + [0.0] * 48
    freekiki.apply_review_prediction(session, 2, pts_ai, conf_ai, "test-model-sha")

    # Frame 5: Complete frame reviewed by human
    _complete(session, 5)
    freekiki.mark_reviewed(session, 5)

    folder = tmp_path / "export_corr"
    freekiki.relocate_review_session(session, folder)

    # In only_correct mode: only frame 5 is exported
    out_corr = freekiki.export_reviewed_session(session, mode="only_correct")
    assert out_corr == folder.resolve()
    assert session.get("export_mode") == "only_correct"
    exported_images_corr = sorted(p.name for p in (folder / "images").glob("*.png"))
    assert len(exported_images_corr) == 1
    assert "f00000005.png" in exported_images_corr[0]

    with (folder / "reviewed_frames.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["frame"] == "5"
    assert rows[0]["reviewed_at"] != ""

    # In full mode: both frame 2 (AI draft) and frame 5 (human-reviewed) are exported!
    folder_full = tmp_path / "export_full"
    freekiki.relocate_review_session(session, folder_full)
    out_full = freekiki.export_reviewed_session(session, mode="full")
    assert out_full == folder_full.resolve()
    assert session.get("export_mode") == "full"

    exported_images_full = sorted(p.name for p in (folder_full / "images").glob("*.png"))
    assert len(exported_images_full) == 2
    assert any("f00000002.png" in name for name in exported_images_full)
    assert any("f00000005.png" in name for name in exported_images_full)

    with (folder_full / "reviewed_frames.csv").open(encoding="utf-8") as f:
        rows_full = sorted(csv.DictReader(f), key=lambda r: int(r["frame"]))
    assert len(rows_full) == 2
    assert rows_full[0]["frame"] == "2"
    assert rows_full[0]["reviewed_at"] == ""
    assert int(rows_full[0]["n_ai"]) == 1
    assert int(rows_full[0]["n_corrected"]) == 0

    assert rows_full[1]["frame"] == "5"
    assert rows_full[1]["reviewed_at"] != ""


def test_ingest_reviewed_accepts_full_export_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", _FrameCapture)
    ws = _workspace(tmp_path / "ws")
    video = tmp_path / "video_full.mp4"
    video.write_bytes(b"video_full_content")
    session = freekiki.new_review_session(video, 48, 32, 30, workspace=ws)

    # Frame 3: uncorrected AI prediction with few points (incomplete under completeness_problem)
    pts_ai = [[10.0, 10.0], [20.0, 20.0]] + [None] * 47
    conf_ai = [0.9, 0.9] + [0.0] * 47
    freekiki.apply_review_prediction(session, 3, pts_ai, conf_ai, "test-model-sha")

    # Frame 7: complete human reviewed frame
    _complete(session, 7)
    freekiki.mark_reviewed(session, 7)

    folder = tmp_path / "full_session_dir"
    freekiki.relocate_review_session(session, folder)
    out = freekiki.export_reviewed_session(session, mode="full")

    # Ingesting full mode session into train split succeeds!
    report = freekiki.ingest_reviewed(ws, out, "MATCH_FULL", split="train", commit=True)
    assert report["frames"] == 2
    assert report["split"] == "train"
    ds = freekiki.dataset_dir(ws)
    with (ds / "manifest.csv").open(encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    train_frames = [r for r in manifest_rows if r.get("group") == "MATCH_FULL"]
    assert len(train_frames) == 2


def test_freekiki_export_cli_mode_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", _FrameCapture)
    video = tmp_path / "cli_vid.mp4"
    video.write_bytes(b"cli_vid")
    session = freekiki.new_review_session(video, 48, 32, 30)
    _complete(session, 1)
    freekiki.mark_reviewed(session, 1)
    pts_ai = [[15.0, 15.0]] + [None] * 48
    freekiki.apply_review_prediction(session, 2, pts_ai, [0.9] + [0.0] * 48, "sha")
    folder = tmp_path / "cli_session"
    freekiki.relocate_review_session(session, folder)
    freekiki.save_review_session(session)

    res = freekiki.main(["export", "--session", str(folder / "session.json"), "--mode", "full"])
    assert res == 0
    saved = freekiki.load_review_session(folder / "session.json")
    assert saved.get("export_mode") == "full"
    with (folder / "reviewed_frames.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


def test_freekiki_export_mode_lite_alias(tmp_path, monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", _FrameCapture)
    video = tmp_path / "lite_vid.mp4"
    video.write_bytes(b"lite_vid")
    session = freekiki.new_review_session(video, 48, 32, 30)
    _complete(session, 1)
    freekiki.mark_reviewed(session, 1)
    pts_ai = [[15.0, 15.0]] + [None] * 48
    freekiki.apply_review_prediction(session, 2, pts_ai, [0.9] + [0.0] * 48, "sha")
    folder = tmp_path / "lite_session"
    freekiki.relocate_review_session(session, folder)
    freekiki.save_review_session(session)

    # API export with mode="lite"
    out = freekiki.export_reviewed_session(session, mode="lite")
    assert out == folder
    assert session.get("export_mode") == "only_correct"
    with (folder / "reviewed_frames.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["frame"] == "1"

    # CLI export with --mode lite
    res = freekiki.main(["export", "--session", str(folder / "session.json"), "--mode", "lite"])
    assert res == 0
    saved = freekiki.load_review_session(folder / "session.json")
    assert saved.get("export_mode") == "only_correct"


def test_ingest_accepts_a_box_touching_the_border_after_rounding(tmp_path):
    # Real label (freekiki_dataset2 frame 51): 8 decimals make cx + bw/2 = 1.000000005.
    box = "0 0.53947266 0.55955556 0.92105469 0.68100000"
    points = " 0.50000000 0.50000000 2 1.00000000 0.40000000 2" + " 0 0 0" * 47
    label = tmp_path / "edge.txt"
    label.write_text(box + points + "\n")
    assert 0.53947266 + 0.92105469 / 2 > 1  # the rounding overshoot is real
    assert freekiki._validate_ingest_label(label) == 2


def test_montage_session_ingests_one_match_per_source_video(tmp_path, monkeypatch):
    ws = _workspace(tmp_path / "workspace")
    ds = ws / "datasets/kiki49"
    monkeypatch.setattr(cv2, "VideoCapture", _FrameCapture)
    video = tmp_path / "montage.mp4"
    video.write_bytes(b"montage")
    session = freekiki.new_review_session(video, 48, 32, 5, workspace=ws)
    for k in (10, 11, 12):  # frames 1 and 2 would repeat the val/test fixture images
        _complete(session, k)
        freekiki.mark_reviewed(session, k)
    out = freekiki.export_reviewed_session(session)
    rows = [
        {"montage_frame": 10, "match": "match_a", "video": "/v/a.mp4", "frame": 10},
        {"montage_frame": 11, "match": "match_b", "video": "/v/b.mp4", "frame": 20},
        {"montage_frame": 12, "match": "match_a", "video": "/v/a2.mp4", "frame": 30},
    ]
    freekiki_diag.write_csv(out / freekiki.MONTAGE_CSV, rows)
    report = freekiki.ingest_reviewed(ws, out, None, commit=True)
    assert report["groups"] == {"match_a": 2, "match_b": 1}
    with (ds / "manifest.csv").open() as f:
        train = [r for r in csv.DictReader(f) if r["split"] == "train"]
    assert sorted((r["group"], r["source_video"], r["source_frame"]) for r in train) == [
        ("match_a", "/v/a.mp4", "10"),
        ("match_a", "/v/a2.mp4", "30"),
        ("match_b", "/v/b.mp4", "20"),
    ]
    # a montage match that is reserved in val is refused
    rows[1]["match"] = "val"
    freekiki_diag.write_csv(out / freekiki.MONTAGE_CSV, rows)
    with pytest.raises(ValueError, match="Match val already belongs"):
        freekiki.ingest_reviewed(ws, out, None)
    (out / freekiki.MONTAGE_CSV).unlink()
    with pytest.raises(ValueError, match="--match-id is required"):
        freekiki.ingest_reviewed(ws, out, None)


def test_relocated_montage_session_keeps_its_match_map(tmp_path):
    video = tmp_path / "montage.mp4"
    video.write_bytes(b"m")
    session = freekiki.new_review_session(
        video, 48, 32, 5, session_path=tmp_path / "montage_1" / "session.json"
    )
    freekiki.save_review_session(session)
    freekiki_diag.write_csv(
        tmp_path / "montage_1" / freekiki.MONTAGE_CSV, [{"montage_frame": 0, "match": "m"}]
    )
    freekiki.relocate_review_session(session, tmp_path / "corrections")
    assert (tmp_path / "corrections" / freekiki.MONTAGE_CSV).is_file()
    assert freekiki.montage_rows(tmp_path / "corrections") == {
        0: {"montage_frame": "0", "match": "m"}
    }


def test_discarded_frame_leaves_the_dataset_and_can_be_restored(tmp_path, monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", _FrameCapture)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"clip")
    session = freekiki.new_review_session(video, 48, 32, 30, workspace=tmp_path)
    for k in (10, 11):
        _complete(session, k)
        freekiki.mark_reviewed(session, k)
    out = freekiki.export_reviewed_session(session)
    assert len(list((out / "images").glob("*.png"))) == 2
    assert freekiki.discard_review_frame(session, 11) == "DISCARDED"
    assert session["frames"]["11"]["points"] == [None] * 49
    assert len(list((out / "images").glob("*.png"))) == 1  # stale export removed
    out = freekiki.export_reviewed_session(session)
    with (out / "reviewed_frames.csv").open() as f:
        assert [r["frame"] for r in csv.DictReader(f)] == ["10"]
    assert freekiki.discard_review_frame(session, 11) == "AI_DRAFT"  # X again: back to the draft
    assert sum(p is not None for p in session["frames"]["11"]["points"]) == 49
