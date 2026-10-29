"""Hand-marked keypoints replace the network for one video (vaila/freekiki.py, ``--keypoints-csv``).

A small synthetic video and a 49-point CSV in the getpixelvideo layout (``frame,p0_x,p0_y,...``).
The checks: no model is loaded, the marked points are the accepted points, an unmarked point is
status ``A``, a wrong layout or a frame outside the video is refused, a batch does not screen a
video that has its ``<stem>_kiki49.csv``, the ball reader refuses the 49-point file, and the
export writes the 49 points to ``<stem>_kiki49.csv``. Nothing here needs a GPU or the weights.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from vaila import freekiki as fk
from vaila import freekiki_detect3d as d3

W, H, N_FRAMES = 640, 360, 6
MARKS = {
    0: {0: (120.0, 150.0), 5: (300.0, 140.0), 12: (500.0, 200.0), 20: (250.0, 250.0)},
    2: {3: (200.0, 120.0)},
}


def _video(path: Path) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),  # ty: ignore[unresolved-attribute]
        25.0,
        (W, H),
    )
    rng = np.random.default_rng(0)
    for _ in range(N_FRAMES):
        writer.write(rng.integers(0, 255, (H, W, 3), dtype=np.uint8))
    writer.release()
    return path


def _markers(path: Path, marks: dict, *, header: list[str] | None = None) -> Path:
    head = header or ["frame"] + [f"p{i}_{k}" for i in range(fk.NKP) for k in ("x", "y")]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(head)
        for frame in sorted(set(range(N_FRAMES)) | set(marks)):
            pts = marks.get(frame, {})
            row = [frame]
            for i in range(fk.NKP):
                row += list(pts[i]) if i in pts else ["", ""]
            writer.writerow(row)
    return path


def _no_network(*args, **kwargs):
    raise AssertionError("the network must not run for hand-marked keypoints")


def _rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    return fk.init_workspace(tmp_path / "ws")


def test_marked_points_are_the_accepted_points_and_no_model_runs(
    tmp_path: Path, ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fk, "load_predictor", _no_network)
    video = _video(tmp_path / "clip.mp4")
    csv_path = _markers(tmp_path / "clip_kiki49.csv", MARKS)
    out = fk.detect_video(
        ws, video, keypoints_csv=csv_path, output_dir=tmp_path / "out", overlay=False, diag_frames=0
    )

    accepted = _rows(out / "field_kps_getpixelvideo.csv")
    frame0 = accepted[1]  # header, then frame 0
    assert frame0[0] == "0"
    assert float(frame0[1 + 2 * 0]) == pytest.approx(120.0, abs=0.01)
    assert float(frame0[2 + 2 * 0]) == pytest.approx(150.0, abs=0.01)
    assert float(frame0[1 + 2 * 12]) == pytest.approx(500.0, abs=0.01)
    assert frame0[1 + 2 * 1] == ""  # p1 was not marked: not an accepted point

    quality = json.loads((out / "quality.json").read_text(encoding="utf-8"))
    assert quality["keypoint_source"] == "manual"
    assert quality["model"] is None
    assert "keypoints: manual" in (out / "README.txt").read_text(encoding="utf-8")


def test_a_mark_outside_the_image_is_rejected_and_reported(
    tmp_path: Path, ws: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(fk, "load_predictor", _no_network)
    video = _video(tmp_path / "clip.mp4")
    marks = {1: {0: (W + 900.0, 100.0), 4: (300.0, 200.0)}}  # p0 is right of the frame
    csv_path = _markers(tmp_path / "clip_kiki49.csv", marks)
    out = fk.detect_video(
        ws, video, keypoints_csv=csv_path, output_dir=tmp_path / "out", overlay=False, diag_frames=0
    )

    status = _rows(out / "field_kps_status.csv")
    p0, p4 = status[0].index("p0"), status[0].index("p4")
    assert status[2][p0] == "Ro"  # frame 1
    assert status[2][p4] == "D"
    assert "hand mark(s) outside the image were rejected" in capsys.readouterr().out


def test_unmarked_points_get_status_a_and_frames_without_marks_get_n(
    tmp_path: Path, ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fk, "load_predictor", _no_network)
    video = _video(tmp_path / "clip.mp4")
    csv_path = _markers(tmp_path / "clip_kiki49.csv", MARKS)
    out = fk.detect_video(
        ws, video, keypoints_csv=csv_path, output_dir=tmp_path / "out", overlay=False, diag_frames=0
    )

    status = _rows(out / "field_kps_status.csv")
    header = status[0]
    p0, p1 = header.index("p0"), header.index("p1")
    frame0, frame1 = status[1], status[2]  # frame 0 has marks, frame 1 none
    assert frame0[p0] == "D"
    assert frame0[p1] == "A"
    assert frame1[p0] == "N"


@pytest.mark.parametrize(
    ("header", "marks", "message"),
    [
        (["frame", "p0_x", "p0_y"], {}, "not the 49-keypoint layout"),
        (None, {N_FRAMES + 3: {0: (10.0, 10.0)}}, "outside the video"),
    ],
)
def test_a_wrong_layout_or_a_frame_outside_the_video_is_refused(
    tmp_path: Path,
    ws: Path,
    monkeypatch: pytest.MonkeyPatch,
    header: list[str] | None,
    marks: dict,
    message: str,
) -> None:
    monkeypatch.setattr(fk, "load_predictor", _no_network)
    video = _video(tmp_path / "clip.mp4")
    csv_path = _markers(tmp_path / "clip_kiki49.csv", marks, header=header)
    with pytest.raises(ValueError, match=message):
        fk.detect_video(
            ws, video, keypoints_csv=csv_path, output_dir=tmp_path / "out", overlay=False
        )


def test_manual_keypoints_for_finds_the_kiki49_file_next_to_the_video(tmp_path: Path) -> None:
    video = tmp_path / "c2.mp4"
    assert fk.manual_keypoints_for(video) is None
    kiki = tmp_path / "c2_kiki49.csv"
    kiki.write_text("frame\n", encoding="utf-8")
    assert fk.manual_keypoints_for(video) == kiki


def test_a_batch_does_not_screen_a_video_with_its_kiki49_file(
    tmp_path: Path, ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "clips"
    source.mkdir()
    (source / "a.mp4").write_bytes(b"")  # has its keypoints: no screen
    (source / "b.mp4").write_bytes(b"")  # the network screen decides: skipped here
    _markers(source / "a_kiki49.csv", MARKS)

    screened: list[str] = []
    detected: list[tuple[str, Path | None]] = []

    def screen(video, *args, **kwargs):
        screened.append(Path(video).name)
        return {"camera_ok": 0.0, "few_points": 1.0, "bad_camera": 0.0, "sampled": 24}

    def detect(ws_arg, video, *, output_dir=None, keypoints_csv=None, **kwargs):
        assert output_dir is not None
        detected.append((Path(video).name, keypoints_csv))
        folder = Path(output_dir) / f"freekiki_predict_{Path(video).stem}_x"
        folder.mkdir(parents=True)
        (folder / "quality.json").write_text(json.dumps({"video": Path(video).name}))
        return folder

    monkeypatch.setattr(fk, "validate_detection_workspace", lambda *a, **k: None)
    monkeypatch.setattr(fk, "resolve_model", lambda *a, **k: Path("model.pt"))
    monkeypatch.setattr(fk, "load_predictor", lambda *a, **k: object())
    monkeypatch.setattr(fk, "screen_video", screen)
    monkeypatch.setattr(fk, "detect_video", detect)

    batch = fk.detect_videos(ws, source, output_dir=tmp_path / "out")

    assert screened == ["b.mp4"]
    assert detected == [("a.mp4", source / "a_kiki49.csv")]
    assert (batch / "skipped_videos.csv").is_file()


def test_keypoints_csv_names_one_video_not_a_folder(
    tmp_path: Path, ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fk, "validate_detection_workspace", lambda *a, **k: None)
    source = tmp_path / "clips"
    source.mkdir()
    (source / "a.mp4").write_bytes(b"")
    with pytest.raises(ValueError, match="names the keypoints of one video"):
        fk.detect_videos(ws, source, output_dir=tmp_path / "out", keypoints_csv=source / "x.csv")


def test_the_ball_reader_refuses_the_49_point_file_and_reads_the_ball(tmp_path: Path) -> None:
    keypoints = _markers(tmp_path / "clip_kiki49.csv", MARKS)
    with pytest.raises(ValueError, match="holds 49 field keypoints"):
        d3.read_corrected_ball(keypoints, N_FRAMES)

    ball = tmp_path / "clip_markers.csv"
    ball.write_text("frame,p0_x,p0_y\n1,300,200\n", encoding="utf-8")
    rows = d3.read_corrected_ball(ball, N_FRAMES)
    assert len(rows) == N_FRAMES


def test_the_export_writes_the_49_points_to_kiki49_and_not_to_the_ball_file(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    session = {
        "video": str(video),
        "frames": {"0": {"points": [(120.0, 150.0)] + [None] * (fk.NKP - 1)}},
    }
    root = tmp_path / "session"
    root.mkdir()
    written = fk.export_wide_markers_csv(session, root, total_video_frames=N_FRAMES)

    names = {path.name for path in written}
    assert "clip_kiki49.csv" in names
    assert "clip_markers.csv" not in names
    assert not (tmp_path / "clip_markers.csv").exists()
