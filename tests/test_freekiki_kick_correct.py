"""Open a finished kick analysis, correct its ball or field keypoints, and run it again in place.

Update Date: 01 November 2026
Version: 0.4.14

A small kick folder is built by hand (a short video, ``pipeline_status.json``, a detect run and a
3D folder): no GPU, no model. getpixelvideo is never opened; the commands it would get are checked.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from vaila import freekiki as fk

STEM = "c2_chute_frame_6_to_20"
TS = "20261101_195640"
N_FRAMES = 5


def _header() -> list[str]:
    return ["frame"] + [f"p{i}_{k}" for i in range(fk.NKP) for k in ("x", "y")]


def _write_kps(path: Path, points: dict[int, dict[int, tuple[float, float]]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_header())
        for frame in range(N_FRAMES):
            row = [str(frame)]
            for i in range(fk.NKP):
                xy = points.get(frame, {}).get(i)
                row += [f"{xy[0]:.1f}", f"{xy[1]:.1f}"] if xy else ["", ""]
            writer.writerow(row)


@pytest.fixture
def kick_run(tmp_path: Path) -> dict:
    clip = tmp_path / "clip"
    clip.mkdir()
    video = clip / f"{STEM}.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (64, 48))  # ty: ignore[unresolved-attribute]
    for i in range(N_FRAMES):
        writer.write(np.full((48, 64, 3), 20 * i, np.uint8))
    writer.release()
    (clip / f"{STEM}_markers.csv").write_text("frame,p0_x,p0_y\n0,10,10\n", encoding="utf-8")
    run = clip / f"kick_analysis_{TS}"
    predict = run / "work" / STEM / f"freekiki_predict_{STEM}_{TS}"
    predict.mkdir(parents=True)
    _write_kps(predict / "field_kps_getpixelvideo.csv", {0: {3: (10.0, 20.0)}, 2: {7: (30.0, 5.0)}})
    threed = run / f"{STEM}_3d_{TS}"
    threed.mkdir()
    (threed / "ball.csv").write_text("frame,cx,cy\n0,10.0,10.0\n1,,\n", encoding="utf-8")
    status = {
        "ts": TS,
        "stages": {
            f"detect:{STEM}": {
                "status": "ok",
                "inputs": {"video": {"path": str(video)}},
                "result": {"predict": str(predict)},
            }
        },
        "options": {
            "videos": [str(video)],
            "set_piece": "open_play",
            "contact_frame": 3,
            "analysis": None,
            "sync": None,
        },  # fmt: skip
    }
    (run / "pipeline_status.json").write_text(json.dumps(status), encoding="utf-8")
    return {"run": run, "video": video, "predict": predict, "threed": threed}


def test_the_detect_run_of_a_kick_folder_and_of_its_3d_folder_is_found(kick_run: dict) -> None:
    assert fk.resolve_detect_predictions(kick_run["run"]) == kick_run["predict"].resolve()
    assert fk.resolve_detect_predictions(kick_run["threed"]) == kick_run["predict"].resolve()


def test_one_camera_of_a_kick_run(kick_run: dict) -> None:
    cam = fk.kick_run_camera(kick_run["run"])
    assert cam["video"] == kick_run["video"] and cam["stem"] == STEM
    assert cam["threed"] == kick_run["threed"] and cam["predict"] == kick_run["predict"].resolve()
    with pytest.raises(ValueError, match="one camera"):
        fk.kick_run_camera(kick_run["run"], "other")
    with pytest.raises(ValueError, match="not a kick analysis folder"):
        fk.kick_run_camera(kick_run["video"].parent)


def test_the_field_session_drafts_the_points_the_analysis_used_and_is_resumed(
    kick_run: dict,
) -> None:
    path = fk.kick_review_session(kick_run["run"])
    assert path == kick_run["run"] / fk.KICK_CORRECTIONS / STEM / "session.json"
    session = json.loads(path.read_text(encoding="utf-8"))
    assert session["draft_source"].endswith("field_kps_getpixelvideo.csv")
    assert session["frames"]["0"]["points"][3] == [10.0, 20.0]
    assert session["frames"]["2"]["points"][7] == [30.0, 5.0]
    assert fk.kick_review_session(kick_run["run"]) == path  # made once, then resumed


def test_a_second_round_drafts_the_keypoints_marked_before(kick_run: dict) -> None:
    _write_kps(kick_run["video"].with_name(f"{STEM}{fk.KIKI49_SUFFIX}"), {1: {5: (11.0, 12.0)}})
    session = json.loads(fk.kick_review_session(kick_run["run"]).read_text(encoding="utf-8"))
    assert session["draft_source"].endswith(fk.KIKI49_SUFFIX)
    assert session["frames"]["1"]["points"][5] == [11.0, 12.0]


@pytest.mark.parametrize("existing", [False, True])
def test_f9_keeps_the_session_in_the_run_and_reopens_the_saved_edit(
    kick_run: dict, existing: bool
) -> None:
    path = fk.kick_review_session(kick_run["run"])
    if existing:
        session = fk.load_review_session(path)
        session.pop("dataset_folder")  # a session created before the relocation fix
        fk.save_review_session(session)
        path = fk.kick_review_session(kick_run["run"])
    session = fk.load_review_session(path)
    assert session["dataset_folder"] == str(path.parent)  # F9 skips its relocation dialog
    fk.edit_review_point(session, 0, 3, (12.0, 22.0))
    assert fk.export_reviewed_session(session, mode="full") == path.parent
    reopened = fk.load_review_session(fk.kick_review_session(kick_run["run"]))
    assert reopened["frames"]["0"]["points"][3] == [12.0, 22.0]
    assert reopened["frames"]["0"]["point_sources"][3] == "corrected"


def test_the_getpixelvideo_commands_of_the_ball_and_of_the_field(kick_run: dict) -> None:
    argv = fk.kick_correction_argv(kick_run["run"], "ball")
    track = kick_run["threed"] / "ball_for_correction.csv"
    assert argv == ["-f", str(kick_run["video"]), "--track-csv", str(track)]
    assert track.read_text(encoding="utf-8").splitlines()[:2] == ["frame,p0_x,p0_y", "0,10.0,10.0"]
    field = fk.kick_correction_argv(kick_run["run"], "field")
    assert field == ["--freekiki-session", str(fk.kick_review_session(kick_run["run"]))]
    with pytest.raises(ValueError, match="ball or field"):
        fk.kick_correction_argv(kick_run["run"], "goal")


def test_kick_correct_copies_the_file_it_will_replace_first(
    kick_run: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[list[str]] = []
    monkeypatch.setattr(fk.subprocess, "Popen", lambda argv: opened.append(argv))
    result = fk.kick_correct(kick_run["run"], "ball", wait=False)
    assert opened and opened[0][-4:] == result["argv"]
    backup = Path(result["backup"])
    assert backup.parent == kick_run["run"] / fk.KICK_CORRECTIONS / STEM
    assert backup.read_text(encoding="utf-8") == "frame,p0_x,p0_y\n0,10,10\n"


def test_resume_takes_the_stored_options_and_the_command_line_wins(kick_run: dict) -> None:
    parser = fk.build_parser()
    run = str(kick_run["run"])
    stored = fk._kick_analysis_kwargs(
        parser.parse_args(["kick-analysis", "-w", "ws", "--resume", run])
    )
    assert stored["videos"] == str(kick_run["video"]) and stored["out_dir"] == run
    assert stored["set_piece"] == "open_play" and stored["contact_frame"] == 3
    given = fk._kick_analysis_kwargs(
        parser.parse_args(
            [
                "kick-analysis",
                "-w",
                "ws",
                "--resume",
                run,
                "--set-piece",
                "free_kick",
                "--contact-frame",
                "auto",
            ]
        )  # fmt: skip
    )
    assert given["set_piece"] == "free_kick" and given["contact_frame"] is None
    with pytest.raises(Exception, match="--videos"):
        fk._kick_analysis_kwargs(parser.parse_args(["kick-analysis", "-w", "ws"]))
