"""FreeKiki human review and train-only ingest round trip.

Version: 0.4.6
Update Date: 28 September 2026
"""

import csv
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from vaila import freekiki


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
    ai_xy = [[10.0, 10.0]] * 49
    ai_conf = [0.9] * 49
    freekiki.apply_review_prediction(session, 17608, ai_xy, ai_conf, "model-hash")
    freekiki.apply_review_prediction(session, 17609, ai_xy, ai_conf, "model-hash")
    freekiki.edit_review_point(session, 17608, 0, [11, 10])
    for i in range(1, 49):
        freekiki.edit_review_point(session, 17608, i, None)
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
    assert vis[0] == 2 and np.count_nonzero(vis) == 1
    assert xy[0, 0] == pytest.approx(11 / 48, abs=1e-7)
    assert freekiki.ingest_reviewed(ws, out, "MATCH_A")["committed"] is False
    assert not list((ds / "images/train").iterdir())
    assert freekiki.ingest_reviewed(ws, out, "MATCH_A", commit=True)["frames"] == 1
    assert freekiki.ingest_reviewed(ws, out, "MATCH_A", commit=True)["frames"] == 0
    with pytest.raises(ValueError, match="already ingested under group"):
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
    freekiki.apply_review_prediction(session, 4, [[10, 10]] * 49, [0.9] * 49, "hash")
    with pytest.raises(ValueError, match="No human-reviewed"):
        freekiki.export_reviewed_session(session)
    freekiki.edit_review_point(session, 4, 1, [12, 10])
    freekiki.mark_reviewed(session, 4)
    assert session["frames"]["4"]["state"] == "HUMAN_REVIEWED"
    freekiki.edit_review_point(session, 4, 1, [13, 10])
    assert session["frames"]["4"]["state"] == "DRAFT_MANUAL"
    freekiki.mark_reviewed(session, 4)
    monkeypatch.setattr(cv2, "VideoCapture", _FrameCapture)
    out = freekiki.export_reviewed_session(session)
    with pytest.raises(ValueError, match="reserved"):
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
        freekiki.edit_review_point(session, 17608, 0, [10, 10])
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
    freekiki.edit_review_point(session, 8, 0, [12, 8])
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
    freekiki.edit_review_point(session, 8, 0, [12, 8])
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
