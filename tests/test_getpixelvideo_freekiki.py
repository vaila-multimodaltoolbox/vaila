"""FreeKiki human review, completeness gate, label queue and train/hard ingest.

Version: 0.4.6
Update Date: 30 September 2026
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
    assert (args.batch_dir, args.per_video, args.min_gap) == ("B", 30, 5)
    args = parser.parse_args(
        ["ingest", "-w", "WS", "--src", "S", "--match-id", "M", "--split", "hard"]
    )
    assert args.split == "hard" and args.commit is False
    assert parser.parse_args(["evaluate", "-w", "WS", "--split", "hard"]).split == "hard"
