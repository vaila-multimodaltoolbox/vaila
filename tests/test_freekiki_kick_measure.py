"""Open a finished kick analysis in getpixelvideo to measure on it (``kick-measure``).

Update Date: 23 November 2026
Version: 0.4.19

The kick folder is the small hand-made one of ``test_freekiki_kick_correct.py`` (a short video,
``pipeline_status.json``, a detect run and a 3D folder): no GPU, no model. getpixelvideo and the
Tk window are never opened; the command it would get is checked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests import test_freekiki_kick_correct as _kick_correct
from vaila import freekiki as fk

# the hand-made kick folder of the correction tests, as a fixture of this file
kick_run = _kick_correct.kick_run


def test_the_getpixelvideo_command_opens_the_camera_with_the_kick_run(kick_run: dict) -> None:
    argv = fk.kick_measure_argv(kick_run["run"])
    assert argv == ["-f", str(kick_run["video"]), "--kick-run", str(kick_run["run"])]
    assert fk.kick_measure_argv(kick_run["run"], "reference") == argv
    # the 3D folder and the clip folder resolve to the same run
    assert Path(argv[-1]).name.startswith("kick_analysis_")


def test_the_camera_must_exist(kick_run: dict) -> None:
    with pytest.raises(ValueError, match="one camera"):
        fk.kick_measure_argv(kick_run["run"], "other")
    with pytest.raises(ValueError, match="reference or other"):
        fk.kick_measure_argv(kick_run["run"], "third")
    with pytest.raises(ValueError, match="not a kick analysis folder"):
        fk.kick_measure_argv(kick_run["video"].parent)


def test_the_second_camera_is_the_other_video(kick_run: dict, tmp_path: Path) -> None:
    status_path = kick_run["run"] / fk.KICK_STATUS
    status = json.loads(status_path.read_text(encoding="utf-8"))
    second = tmp_path / "clip" / "c1_other.mp4"
    second.write_bytes(b"")
    status["options"]["videos"].append(str(second))
    status_path.write_text(json.dumps(status), encoding="utf-8")
    argv = fk.kick_measure_argv(kick_run["run"], "other")
    assert argv[:2] == ["-f", str(second)]


def test_the_subcommand_parses_the_camera() -> None:
    parser = fk.build_parser()
    args = parser.parse_args(["kick-measure", "some_run"])
    assert (args.command, args.run, args.camera, args.no_wait) == (
        "kick-measure",
        "some_run",
        "reference",
        False,
    )
    args = parser.parse_args(["kick-measure", "some_run", "--camera", "other", "--no-wait"])
    assert args.camera == "other" and args.no_wait is True
    args = parser.parse_args(["kick-measure", "some_run", "-w", "ws"])  # the GUI passes -w
    assert args.workspace == "ws"
    with pytest.raises(SystemExit):
        parser.parse_args(["kick-measure", "some_run", "--camera", "third"])


def test_kick_measure_opens_getpixelvideo_and_prints_the_mirror(
    kick_run: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[list[str]] = []
    mirrored: list[tuple] = []
    monkeypatch.setattr(fk.subprocess, "Popen", lambda argv: opened.append(argv))
    monkeypatch.setattr(fk, "print_gui_cli_mirror", lambda *a, **k: mirrored.append(a))
    result = fk.kick_measure(kick_run["run"], wait=False)
    assert opened and opened[0][-4:] == result["argv"]
    assert opened[0][1].endswith("getpixelvideo.py")
    command = mirrored[0][1]
    assert command[:4] == ["uv", "run", "--no-sync", "vaila/getpixelvideo.py"]
    assert "--kick-run" in command
    assert result["video"] == str(kick_run["video"]) and result["run"] == str(kick_run["run"])


def test_main_returns_one_for_a_folder_that_is_not_a_kick_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert fk.main(["kick-measure", str(tmp_path)]) == 1


# --- "Measure kick: video + SAM folder": a video and a folder, no workspace --------------------


def test_the_video_and_folder_command_is_the_same_getpixelvideo_call(kick_run: dict) -> None:
    argv, camera = fk.kick_measure_video_argv(kick_run["video"], kick_run["run"])
    assert camera == "reference"
    assert argv == ["-f", str(kick_run["video"].resolve()), "--kick-run", str(kick_run["run"])]
    assert argv == fk.kick_measure_argv(kick_run["run"])
    # the 3D folder of the run is accepted as the folder, as in kick-measure
    threed = next(kick_run["run"].glob("*_3d_*"))
    assert fk.kick_measure_video_argv(kick_run["video"], threed)[0][-1] == str(kick_run["run"])


def test_a_video_that_is_no_camera_of_the_folder_is_refused(kick_run: dict, tmp_path: Path) -> None:
    stranger = tmp_path / "elsewhere" / "other_game.mp4"
    stranger.parent.mkdir()
    stranger.write_bytes(b"")
    with pytest.raises(ValueError, match="not a camera of kick_analysis_"):
        fk.kick_measure_video_argv(stranger, kick_run["run"])
    with pytest.raises(ValueError, match="not a kick analysis folder"):
        fk.kick_measure_video_argv(kick_run["video"], tmp_path)


def test_a_wrong_video_warns_and_opens_nothing(
    kick_run: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stranger = tmp_path / "other_game.mp4"
    stranger.write_bytes(b"")
    opened: list[list[str]] = []
    monkeypatch.setattr(fk.subprocess, "Popen", lambda argv: opened.append(argv))
    monkeypatch.setattr(fk.subprocess, "run", lambda argv, **k: opened.append(argv))
    monkeypatch.setattr(fk, "print_gui_cli_mirror", lambda *a, **k: opened.append(list(a)))
    warnings: list[str] = []
    result = fk.measure_kick_video_folder(
        lambda start: str(stranger),
        lambda start: str(kick_run["run"]),
        warnings.append,
    )
    assert result is None
    assert len(warnings) == 1 and "other_game.mp4 is not a camera" in warnings[0]
    assert opened == []


def test_two_dialogs_then_getpixelvideo_with_the_mirror(
    kick_run: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[list[str]] = []
    mirrored: list[tuple] = []
    monkeypatch.setattr(fk.subprocess, "Popen", lambda argv: opened.append(argv))
    monkeypatch.setattr(fk, "print_gui_cli_mirror", lambda *a, **k: mirrored.append(a))
    starts: list[tuple[str, str | None]] = []
    result = fk.measure_kick_video_folder(
        lambda start: starts.append(("video", start)) or str(kick_run["video"]),
        lambda start: starts.append(("folder", start)) or str(kick_run["run"]),
        lambda message: pytest.fail(message),
        initial_video=kick_run["video"],
    )
    assert starts == [
        ("video", str(kick_run["video"].parent)),
        ("folder", str(kick_run["video"].parent)),
    ]
    assert result is not None and result["camera"] == "reference"
    assert opened[0][1].endswith("getpixelvideo.py") and opened[0][-4:] == result["argv"]
    assert mirrored[0][1] == [
        "uv",
        "run",
        "--no-sync",
        "vaila/getpixelvideo.py",
        "-f",
        str(kick_run["video"].resolve()),
        "--kick-run",
        str(kick_run["run"]),
    ]


def test_cancelling_a_dialog_opens_nothing(kick_run: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[list[str]] = []
    monkeypatch.setattr(fk.subprocess, "Popen", lambda argv: opened.append(argv))
    asked: list[str] = []
    assert (
        fk.measure_kick_video_folder(lambda s: "", lambda s: asked.append("f") or "", print) is None
    )
    assert (
        fk.measure_kick_video_folder(lambda s: str(kick_run["video"]), lambda s: "", print) is None
    )
    assert asked == []
    assert opened == []


def test_the_button_is_in_the_main_window() -> None:
    import inspect

    assert callable(fk.measure_kick_video_folder)
    assert fk.KICK_MEASURE_BUTTON == "Measure kick: video + SAM folder"
    assert fk.KICK_VIDEO_TYPES[0][1].split()[0] == "*.mp4" and fk.KICK_VIDEO_TYPES[-1][1] == "*.*"
    source = inspect.getsource(fk.run_freekiki)
    # first-level button of the "Kick analysis" box, not of a secondary window
    assert "text=KICK_MEASURE_BUTTON" in source
    assert "command=do_measure_kick_video_folder" in source
    assert source.index("text=KICK_MEASURE_BUTTON") > source.index('text="Kick analysis"')
    # no workspace is asked: the flow does not go through run_cli
    flow = inspect.getsource(fk.measure_kick_video_folder)
    assert "run_cli" not in flow and 'v["ws"]' not in flow
