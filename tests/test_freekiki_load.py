"""Tests for read_result_settings: the GUI 'Load previous result' reader (no GUI, no model)."""

from __future__ import annotations

from pathlib import Path

from vaila import freekiki

DETECT_README = (
    "FreeKiki field keypoints (vailá)\n"
    "video: /data/clip.mp4\n"
    "model: /models/freekiki_hm_hr32.pt\n"
    "dimensions: 1920x1080\n"
    "frames: 80 (start=0, stride=1)\n"
    "detection (box) conf: 0.25\n"
    "keypoint conf: 0.5\n"
    "imgsz: 1024\n"
    "fill gaps: 0 frames\n"
    "geometry: fill\n"
    "snap: True (snapped onto painted pitch lines)\n"
)


def _run(folder: Path, name: str = "freekiki_predict_clip_20260101_000000") -> Path:
    run = folder / name
    run.mkdir(parents=True)
    (run / "README.txt").write_text(DETECT_README, encoding="utf-8")
    return run


def test_detect_run_gives_the_gui_settings(tmp_path: Path) -> None:
    run = _run(tmp_path)
    got = freekiki.read_result_settings(run)
    assert got is not None
    assert got["run"] == str(run)
    assert got["video"] == "/data/clip.mp4"
    assert got["model"] == "/models/freekiki_hm_hr32.pt"
    assert got["stride"] == "1"
    assert got["conf"] == "0.25"
    assert got["kp_conf"] == "0.5"
    assert got["fill_gaps"] == "0"
    assert got["geom"] == "fill"
    assert got["snap"] is True


def test_3d_folder_gives_its_run_video_and_pitch_and_ball_choices(tmp_path: Path) -> None:
    prev = tmp_path / "clip_3d_20260101_000000"
    run = _run(prev / "work")
    prev.mkdir(exist_ok=True)
    (prev / "README.txt").write_text(
        "FreeKiki 3D output (vailá)\nvideo: /data/clip.mp4\n"
        "ball: {'mode': 'sam3', 'frames': 80, 'found': 60}\n",
        encoding="utf-8",
    )
    (prev / "players_on_field.csv").write_text(
        "person_id,fraction,on,checked,keep\n1,1.0,32,32,True\n2,0.0,0,32,False\n",
        encoding="utf-8",
    )
    got = freekiki.read_result_settings(prev)
    assert got is not None
    assert got["folder3d"] == str(prev)
    assert got["out"] == str(tmp_path)
    assert got["run"] == str(run)
    assert got["video"] == "/data/clip.mp4"
    assert got["model"] == "/models/freekiki_hm_hr32.pt"
    assert got["pitch_on"] is True
    assert got["ball_mode"] == "sam3"


def test_corrected_ball_does_not_change_the_ball_choice(tmp_path: Path) -> None:
    prev = tmp_path / "clip_3d_20260101_000000"
    _run(prev / "work")
    (prev / "README.txt").write_text(
        "FreeKiki 3D output (vailá)\nvideo: /data/clip.mp4\n"
        "ball: {'mode': 'corrected', 'frames': 80, 'found': 80}\n",
        encoding="utf-8",
    )
    got = freekiki.read_result_settings(prev)
    assert got is not None
    assert "ball_mode" not in got
    assert "pitch_on" not in got  # no players_on_field.csv in this folder


def test_parent_with_several_3d_folders_uses_the_newest(tmp_path: Path) -> None:
    for stamp in ("20260101_000000", "20260103_000000", "20260102_000000"):
        folder = tmp_path / f"clip_3d_{stamp}"
        folder.mkdir()
        (folder / "README.txt").write_text(
            f"FreeKiki 3D output (vailá)\nvideo: /data/{stamp}.mp4\n", encoding="utf-8"
        )
    got = freekiki.read_result_settings(tmp_path)
    assert got is not None
    assert got["folder3d"] == str(tmp_path / "clip_3d_20260103_000000")


def test_batch_folder_uses_the_newest_run(tmp_path: Path) -> None:
    _run(tmp_path, "freekiki_predict_a_20260101_000000")
    newest = _run(tmp_path, "freekiki_predict_b_20260102_000000")
    got = freekiki.read_result_settings(tmp_path)
    assert got is not None
    assert got["run"] == str(newest)


def test_folder_that_is_not_a_result_gives_none(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    assert freekiki.read_result_settings(tmp_path) is None
    assert freekiki.read_result_settings(tmp_path / "missing") is None


def test_incomplete_readme_keeps_only_what_it_states(tmp_path: Path) -> None:
    run = tmp_path / "freekiki_predict_x_20260101_000000"
    run.mkdir()
    (run / "README.txt").write_text(
        "FreeKiki field keypoints (vailá)\nvideo: /data/clip.mp4\n", encoding="utf-8"
    )
    got = freekiki.read_result_settings(run)
    assert got == {"run": str(run), "video": "/data/clip.mp4"}


def test_rebuilt_folder_without_work_uses_its_sibling(tmp_path: Path) -> None:
    source = tmp_path / "clip_3d_20260101_000000"
    run = _run(source / "work")
    (source / "players_on_field.csv").write_text(
        "person_id,fraction,on,checked,keep\n1,1.0,32,32,True\n", encoding="utf-8"
    )
    (source / "README.txt").write_text(
        "FreeKiki 3D output (vailá)\nvideo: /data/clip.mp4\nball: {'mode': 'sam3', 'frames': 80}\n",
        encoding="utf-8",
    )
    rebuilt = tmp_path / "clip_3d_20260102_000000"
    rebuilt.mkdir()
    (rebuilt / "README.txt").write_text(
        "FreeKiki 3D output (vailá)\nvideo: /data/clip.mp4\nball: {'mode': 'corrected', 'frames': 80}\n",
        encoding="utf-8",
    )
    got = freekiki.read_result_settings(rebuilt)
    assert got is not None
    assert got["folder3d"] == str(rebuilt)
    assert got["work_from"] == str(source)
    assert got["run"] == str(run)
    assert got["model"] == "/models/freekiki_hm_hr32.pt"
    assert got["pitch_on"] is True
    assert "ball_mode" not in got  # corrected: the choice of the original is not restored


def test_ball_rebuild_accepts_the_workspace_the_gui_passes() -> None:
    parser = freekiki.build_parser()
    with_w = parser.parse_args(["ball-rebuild", "-w", "/ws", "--out", "A", "--ball-csv", "B"])
    without_w = parser.parse_args(["ball-rebuild", "--out", "A", "--ball-csv", "B"])
    assert with_w.out == without_w.out == "A"
    assert with_w.ball_csv == without_w.ball_csv == "B"


def test_rebuilt_folder_is_read_from_the_log_line() -> None:
    line = "[freekiki] ball rebuilt: /data/clip_3d_20260102_000000 | {'mode': 'corrected', 'frames': 80}\n"
    assert freekiki.rebuilt_folder_from_log(line) == Path("/data/clip_3d_20260102_000000")
    assert freekiki.rebuilt_folder_from_log("[freekiki] finished (exit code 0)\n") is None
    assert freekiki.rebuilt_folder_from_log("[freekiki] batch done: 1/4 videos\n") is None
