"""FreeKiki label queue: --need (field camera says the keypoint is in the picture) and montage.

Version: 0.4.7
Update Date: 05 October 2026
"""

import csv
import json

import cv2
import numpy as np
import pytest

from vaila import freekiki, freekiki_diag, freekiki_geom

W, H = 1920, 1080


def _camera(C=(10.0, -60.0, 25.0), target=(30.0, 0.0, 0.0), f=2000.0):
    C, target = np.asarray(C, float), np.asarray(target, float)
    z = (target - C) / np.linalg.norm(target - C)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    R = np.vstack([x, np.cross(z, x), z])
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    P = K @ np.c_[R, -R @ C]
    return P / P[2, 3]


def _view():
    xyz, _ = freekiki_geom._field()
    uv, front = freekiki_geom._project(_camera(), xyz)
    inside = front & (uv[:, 0] > 5) & (uv[:, 0] < W - 5) & (uv[:, 1] > 5) & (uv[:, 1] < H - 5)
    return uv, inside


def _rows(frame, uv, kc, codes, n_acc=None):
    raw = {"frame": str(frame), "box_conf": "0.9"}
    for i in range(49):
        raw |= {
            f"p{i}_x": f"{uv[i, 0]:.3f}",
            f"p{i}_y": f"{uv[i, 1]:.3f}",
            f"p{i}_conf": str(kc[i]),
        }
    status = {"frame": str(frame), "n_accepted": str(n_acc or codes.count("D"))}
    status |= {f"p{i}": codes[i] for i in range(49)}
    return status, raw


def test_match_of_a_cut_is_its_source_video():
    assert freekiki._match_from_name("003_Melhores Momentos_frame_13654_to_13681") == (
        "003_melhores_momentos"
    )
    assert freekiki._match_from_name("Clip A") == "clip_a"


def test_parse_keypoints():
    assert freekiki.parse_keypoints("p5, p29,39,p47,p5") == (5, 29, 39, 47)
    for bad in ("p49", "x5", ""):
        with pytest.raises(ValueError):
            freekiki.parse_keypoints(bad)


def test_need_frames_use_the_field_camera():
    uv, inside = _view()
    assert inside[24] and inside[46] and not inside[29]  # top-right corner / flag in view
    statuses = []
    kc_ok = np.where(inside, 0.9, 0.0)
    # frame 0: camera ok, p24 in the picture but the network misses it -> tier 0
    kc = kc_ok.copy()
    kc[24] = 0.02
    codes = ["D" if inside[i] and i != 24 else "Rk" for i in range(49)]
    statuses.append(_rows(0, uv, kc, codes))
    # frame 10: camera ok, p24 and p46 accepted -> tier 2
    statuses.append(_rows(10, uv, kc_ok, ["D" if inside[i] else "Rk" for i in range(49)]))
    # frame 20: three points only (no camera), network half-sees p29 -> tier 1
    kc = np.zeros(49)
    few = np.flatnonzero(inside)[:3]
    kc[few] = 0.9
    kc[29] = 0.2
    statuses.append(_rows(20, uv, kc, ["D" if i in few else "Rk" for i in range(49)]))
    # frame 30: camera ok, nothing needed in the picture -> skipped
    statuses.append(_rows(30, uv, kc_ok, ["D" if inside[i] else "Rk" for i in range(49)]))
    status_rows, raw_rows = zip(*statuses, strict=True)
    picked = freekiki.queue_need_frames(
        status_rows, raw_rows, (24, 29), W, H, per_video=5, min_gap=1, half_seen=True
    )
    by_frame = {r["frame"]: r for r in picked}
    default = freekiki.queue_need_frames(
        status_rows, raw_rows, (24, 29), W, H, per_video=5, min_gap=1
    )
    assert 20 not in {r["frame"] for r in default}  # no camera + half-seen: off by default
    assert by_frame[0]["tier"] == 0 and by_frame[0]["need_missing"] == "p24"
    assert by_frame[0]["camera"] in ("dlt3d", "homography")
    assert by_frame[20]["tier"] == 1 and "p29" in by_frame[20]["reason"]
    assert by_frame[10]["tier"] == 2
    assert 30 in by_frame  # p24 accepted in view: still a tier-2 candidate
    only_flag = freekiki.queue_need_frames(
        status_rows[1:2], raw_rows[1:2], (47,), W, H, per_video=5, min_gap=1
    )
    assert only_flag == []  # p47 is not in this view
    capped = freekiki.queue_need_frames(
        status_rows, raw_rows, (24, 29), W, H, per_video=2, min_gap=1, half_seen=True
    )
    assert [r["tier"] for r in capped] == [0, 1]  # best tiers first


def _clip(path, size, n, color_of):
    w, h = size
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))  # ty: ignore[unresolved-attribute]
    for i in range(n):
        out.write(np.full((h, w, 3), color_of(i), np.uint8))
    out.release()


def _run(folder, video, size, frames, xy):
    folder.mkdir(parents=True)
    (folder / "README.txt").write_text(
        f"FreeKiki field keypoints (vailá)\nvideo: {video}\nmodel_sha256: sha-{video.stem}\n"
        f"dimensions: {size[0]}x{size[1]}\n"
    )
    fields = ["frame", "box_conf"] + [f"p{i}_{k}" for i in range(49) for k in ("x", "y", "conf")]
    with (folder / "field_kps_raw.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for fr in frames:
            row = {"frame": fr, "box_conf": 0.9}
            for i in range(49):
                row |= {f"p{i}_x": xy[0], f"p{i}_y": xy[1], f"p{i}_conf": 0.9 if i == 0 else 0.1}
            writer.writerow(row)


def test_montage_reads_exact_frames_letterboxes_and_drafts(tmp_path):
    ws = freekiki.init_workspace(tmp_path / "ws")
    a, b = tmp_path / "Clip A.mp4", tmp_path / "clip_b.mp4"
    _clip(a, (320, 180), 12, lambda i: (20 * i, 0, 0))  # 16:9, the common size
    _clip(b, (240, 180), 8, lambda i: (0, 30 * i, 0))  # 4:3 -> letterboxed
    run_a, run_b = tmp_path / "run_a", tmp_path / "run_b"
    _run(run_a, a, (320, 180), range(12), (100.0, 50.0))
    _run(run_b, b, (240, 180), range(8), (120.0, 90.0))
    items = [
        (
            run_a,
            a,
            [{"frame": 3, "tier": 0, "reason": "x"}, {"frame": 9, "tier": 2, "reason": "y"}],
        ),
        (run_b, b, [{"frame": 5, "tier": 1, "reason": "z"}]),
    ]
    path = freekiki.build_montage(ws, items, kp_conf=0.5, folder=tmp_path / "montage_1")
    folder = path.parent
    cap = cv2.VideoCapture(str(folder / "montage.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (
        320,
        180,
    )
    colors = []
    for k in range(3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)  # exact seeking: every frame is a key frame
        ok, img = cap.read()
        assert ok
        colors.append(img[90, 160].astype(int))
    cap.release()
    assert abs(colors[0][0] - 60) <= 8 and abs(colors[1][0] - 180) <= 8  # a frames 3, 9
    assert abs(colors[2][1] - 150) <= 8  # b frame 5 (centre of the letterbox)
    with (folder / freekiki.MONTAGE_CSV).open() as f:
        rows = list(csv.DictReader(f))
    assert [(r["match"], r["frame"]) for r in rows] == [
        ("clip_a", "3"),
        ("clip_a", "9"),
        ("clip_b", "5"),
    ]
    assert rows[2]["offset_x"] == "40" and float(rows[2]["scale"]) == 1.0
    session = json.loads(path.read_text())
    assert session["session_id"] == folder.name and session["dataset_folder"] == str(folder)
    states = {k: r["state"] for k, r in session["frames"].items()}
    assert states == {"0": "AI_DRAFT", "1": "AI_DRAFT", "2": "AI_DRAFT"}
    assert session["frames"]["2"]["points"][0] == [160.0, 90.0]  # 120 + offset 40
    assert session["frames"]["0"]["model_sha256"] == "sha-Clip A"
    loaded = freekiki.load_review_session(path)
    assert loaded["video"] == str(folder / "montage.mp4")


def test_queue_need_montage_end_to_end(tmp_path, monkeypatch):
    ws = freekiki.init_workspace(tmp_path / "ws")
    uv, inside = _view()
    w, h = 480, 270
    s = w / W
    batch = tmp_path / "processed_freekiki_batch_1"
    for name in ("m1", "m2"):
        video = tmp_path / f"{name}.mp4"
        _clip(video, (w, h), 6, lambda i: (i, i, i))
        run = batch / f"freekiki_predict_{name}_20261005_000000"
        run.mkdir(parents=True)
        (run / "README.txt").write_text(
            f"FreeKiki field keypoints (vailá)\nvideo: {video}\nmodel_sha256: x\ndimensions: {w}x{h}\n"
        )
        kc = np.where(inside, 0.9, 0.0)
        kc[24] = 0.1
        codes = ["D" if inside[i] and i != 24 else "Rk" for i in range(49)]
        rows = [_rows(fr, uv * s, kc, codes) for fr in range(6)]
        for fname, idx in (("field_kps_status.csv", 0), ("field_kps_raw.csv", 1)):
            freekiki_diag.write_csv(run / fname, [r[idx] for r in rows])
    sessions = freekiki.build_label_queue(
        ws, [batch], need=(24,), montage=True, per_video=2, min_gap=3
    )
    assert len(sessions) == 1
    with (sessions[0].parent / freekiki.MONTAGE_CSV).open() as f:
        rows = list(csv.DictReader(f))
    assert [r["match"] for r in rows] == ["m1", "m1", "m2", "m2"]
    assert all(r["tier"] == "0" for r in rows)
    assert abs(int(rows[1]["frame"]) - int(rows[0]["frame"])) >= 3


def test_describe_model_names_slot_run_and_hash(tmp_path):
    ws = freekiki.init_workspace(tmp_path / "ws")
    run = ws / "runs" / "r1" / "weights"
    run.mkdir(parents=True)
    (run / "best.pt").write_bytes(b"weights")
    text = freekiki.describe_model(ws, str(run / "best.pt"))
    assert "run r1" in text and "not a promoted slot" in text
    assert freekiki.file_sha256(run / "best.pt")[:12] in text


def test_queue_never_offers_a_frame_twice(tmp_path, monkeypatch):
    test_queue_need_montage_end_to_end(tmp_path, monkeypatch)  # first montage: 2 frames x 2 videos
    ws = tmp_path / "ws"
    first = freekiki.already_queued(ws)
    assert len(first) == 4
    batch = tmp_path / "processed_freekiki_batch_1"
    sessions = freekiki.build_label_queue(
        ws, [batch], need=(24,), montage=True, per_video=2, min_gap=1
    )
    with (sessions[0].parent / freekiki.MONTAGE_CSV).open() as f:
        rows = list(csv.DictReader(f))
    second = {(str(r["video"]), int(r["frame"])) for r in rows}
    assert rows and not second & {(str(v), f) for v, f in first}
