"""Common event grid, exact FPS, export provenance and Syncvid launcher regressions.

Update Date: 30 October 2026
Version: 0.4.13
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from fractions import Fraction
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from vaila import rife_paths
from vaila import video_event_equalizer as equalizer


def test_19_to_22_exact_mapping():
    rows = equalizer.frame_map([1, 9, 22], [1, 9, 19], 22, 19, "2997/100")
    assert len(rows) == 22
    assert [r["source_position"] for r in rows[:9]] == [str(i) for i in range(1, 10)]
    assert rows[9]["source_position"] == "127/13"
    assert rows[-1]["source_position"] == "19"
    assert sum(r["kind"] == "generated" for r in rows) == 12
    for j, row in enumerate(rows[9:], 10):
        assert Fraction(row["source_position"]) == 9 + Fraction((j - 9) * 10, 13)
    assert rows[8]["output_time_rational"] == str(Fraction(800, 2997))


@pytest.mark.parametrize(
    "ref,src",
    [
        ([1, 9], [1, 9, 19]),
        ([1, 9, 23], [1, 9, 19]),
        ([1, 1, 22], [1, 9, 19]),
        ([1, 9, 22], [0, 9, 19]),
        ([1, 9, 22], [1, 19, 9]),
    ],
)
def test_bad_anchors(ref, src):
    with pytest.raises(equalizer.EventSyncError):
        equalizer.frame_map(ref, src, 22, 19, "2997/100")


def test_non_one_start():
    rows = equalizer.frame_map([4, 7, 12], [2, 5, 9], 20, 15, "30")
    assert len(rows) == 9
    assert [rows[i]["source_position"] for i in (0, 3, 8)] == ["2", "5", "9"]
    assert rows[0]["output_time_rational"] == "0"


def write_video(path, count):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg needed for lossless export proof")
    images = np.stack([np.full((48, 64, 3), i * 8, np.uint8) for i in range(count)])
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            "64x48",
            "-framerate",
            "2997/100",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            str(path),
        ],
        input=images.tobytes(),
        check=True,
        capture_output=True,
    )
    return images


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    images = []
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                return images
            images.append(image)
    finally:
        cap.release()


@pytest.fixture
def video_args(tmp_path):
    write_video(tmp_path / "ref.mp4", 22)
    write_video(tmp_path / "src.mp4", 19)
    return argparse.Namespace(
        reference=tmp_path / "ref.mp4",
        source=tmp_path / "src.mp4",
        reference_events="1,9,22",
        source_events="1,9,19",
        rife_dir=tmp_path,
        weights=tmp_path,
        output_dir=tmp_path / "out",
    )


class StandInRife:
    """Deterministic blend to verify export plumbing; not a model accuracy test."""

    def __init__(self, *_args):
        self.metadata = {"test_double": True}

    def __call__(self, left, right, alpha):
        return cv2.addWeighted(left, 1 - float(alpha), right, float(alpha), 0)


@pytest.mark.parametrize(
    "fps,count,expected",
    [
        (None, None, 22),
        ("29.970", None, 22),
        ("30000/1001", None, 22),
        ("60", None, 43),
        ("120", None, 85),
        (None, "31", 31),
        ("120", "31", 31),
        ("60", "3", 3),
    ],
)
def test_export_lossless_copies_and_maps(video_args, monkeypatch, fps, count, expected):
    video_args.output_fps, video_args.output_frame_count = fps, count
    preview = equalizer.prepare(video_args)
    monkeypatch.setattr(equalizer, "RifeInterpolator", StandInRife)
    out = equalizer.export(video_args)
    assert not (out / "INCOMPLETE.txt").exists()
    manifest = json.loads((out / "event_equalizer.json").read_text())
    rate = Fraction(fps or "2997/100")
    assert manifest["output_fps"] == str(rate)
    assert manifest["requested_output_fps"] == fps
    assert manifest["requested_output_frame_count"] == count
    assert manifest["output_frame_count"] == expected
    assert manifest["output_duration_rational"] == str(Fraction(expected - 1) / rate)
    for role, source in (("reference", video_args.reference), ("source", video_args.source)):
        camera = manifest["cameras"][role]
        frames = read_video(out / camera["video"])
        originals = read_video(source)
        assert len(frames) == expected
        assert equalizer.probe(out / camera["video"])["fps"] == str(rate)
        assert camera["original_output_count"] == preview["cameras"][role]["original_output_count"]
        assert (
            camera["generated_output_count"] == preview["cameras"][role]["generated_output_count"]
        )
        with (out / f"{role}_frames.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        for anchor, original in zip(manifest["output_event_frames"], camera["events"], strict=True):
            assert rows[anchor - 1]["kind"] == "original"
            assert rows[anchor - 1]["source_position"] == str(original)
            np.testing.assert_array_equal(frames[anchor - 1], originals[original - 1])
        for row in rows:
            assert row["output_fps"] == str(rate)
            assert int(row["output_frame_count"]) == expected
            assert int(row["output_contact_frame"]) == manifest["output_event_frames"][1]
            assert (
                row["contact_rounding_adjustment_rational"]
                == manifest["contact_rounding_adjustment_rational"]
            )
            assert row["output_time_rational"] == str(Fraction(int(row["output_frame"]) - 1) / rate)
        for frame, row in zip(frames, rows, strict=True):
            if row["kind"] == "original":
                np.testing.assert_array_equal(frame, originals[int(row["source_frame_left"]) - 1])
            else:
                expected_image = StandInRife()(
                    originals[int(row["source_frame_left"]) - 1],
                    originals[int(row["source_frame_right"]) - 1],
                    Fraction(row["interpolation_fraction"]),
                )
                np.testing.assert_array_equal(frame, expected_image)
        result = subprocess.run(
            [
                equalizer.get_ffprobe_path(),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-show_packets",
                "-of",
                "json",
                str(out / camera["video"]),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        media = json.loads(result.stdout)
        stream = media["streams"][0]
        time_base = Fraction(stream["time_base"])
        assert Fraction(stream["avg_frame_rate"]) == rate
        assert Fraction(stream["r_frame_rate"]) == rate
        assert int(stream["nb_frames"]) == expected
        assert int(stream["duration_ts"]) * time_base == Fraction(expected) / rate
        packets = sorted(media["packets"], key=lambda p: int(p["pts"]))
        assert len(packets) == expected
        for index, packet in enumerate(packets):
            assert int(packet["pts"]) * time_base == Fraction(index) / rate
            assert int(packet["duration"]) * time_base == 1 / rate
    if fps is None and count is None:
        assert manifest["cameras"]["source"]["generated_output_count"] == 12


@pytest.mark.parametrize(
    "fps,count,n,contact",
    [
        (None, None, 22, 9),
        ("29.970", None, 22, 9),
        ("30000/1001", None, 22, 9),
        ("60", None, 43, 17),
        ("120", None, 85, 33),
        (None, 31, 31, 12),
        ("60", 31, 31, 12),
    ],
)
def test_common_grid_policy(fps, count, n, contact):
    grid = equalizer.output_grid([1, 9, 22], 22, "2997/100", fps, count)
    assert grid["output_frame_count"] == n
    assert grid["output_event_frames"] == [1, contact, n]
    assert grid["output_fps"] == str(Fraction(fps or "2997/100"))
    for events, size in (([1, 9, 22], 22), ([1, 9, 19], 19)):
        rows = equalizer.frame_map(
            [1, 9, 22], events, 22, size, "2997/100", output_fps=fps, output_frame_count=count
        )
        assert len(rows) == n
        assert [rows[i - 1]["source_position"] for i in (1, contact, n)] == list(map(str, events))
        positions = [Fraction(row["source_position"]) for row in rows]
        assert positions == sorted(positions)
        assert all(
            row["event"] == event
            for row, event in zip(
                (rows[0], rows[contact - 1], rows[-1]), ("start", "contact", "end"), strict=True
            )
        )


def test_half_up_contact_clamping_and_non_one_start():
    # 2.5 intervals rounds to 3, with contact halfway through the reference.
    grid = equalizer.output_grid([4, 6, 8], 20, "4", "4.5")
    assert grid["output_frame_count"] == 6
    assert grid["output_event_frames"] == [1, 4, 6]
    assert grid["contact_rounding_adjustment_rational"] == "1/9"
    for events, expected_contact in (([1, 2, 101], 2), ([1, 100, 101], 2)):
        grid = equalizer.output_grid(events, 101, "30", output_frame_count=3)
        assert grid["output_event_frames"] == [1, expected_contact, 3]
        assert Fraction(grid["contact_rounding_adjustment_rational"]) != 0
    rows = equalizer.frame_map([4, 7, 12], [2, 5, 9], 20, 15, "30", output_frame_count=6)
    assert [rows[i - 1]["source_position"] for i in (1, 3, 6)] == ["2", "5", "9"]
    assert rows[1]["reference_position"] == "11/2"


@pytest.mark.parametrize(
    "fps,count",
    [
        ("0", None),
        ("-30", None),
        ("nan", None),
        ("inf", None),
        ("1/0", None),
        ("bad", None),
        ("1", None),
        (None, "2"),
        (None, "0"),
        (None, "-3"),
        (None, "3.5"),
        (None, "3_0"),
        (None, 3.0),
        (None, True),
        (None, "nan"),
    ],
)
def test_bad_output_grid(fps, count):
    with pytest.raises(equalizer.EventSyncError):
        equalizer.output_grid([1, 9, 22], 22, "30", fps, count)


def test_preflight_covers_both_cameras_before_creating_output(video_args, monkeypatch):
    video_args.output_fps = "60"
    calls = []

    class PreflightRife(StandInRife):
        def __call__(self, left, right, alpha):
            calls.append(alpha)
            if len(calls) == 2:
                raise RuntimeError("second camera fails")
            return super().__call__(left, right, alpha)

    monkeypatch.setattr(equalizer, "RifeInterpolator", PreflightRife)
    with pytest.raises(RuntimeError, match="second camera"):
        equalizer.export(video_args)
    assert len(calls) == 2
    assert not video_args.output_dir.exists()


def test_invalid_input_and_preflight_failure_create_no_output(video_args, monkeypatch):
    video_args.source_events = "1,9,99"
    with pytest.raises(equalizer.EventSyncError):
        equalizer.export(video_args)
    assert not video_args.output_dir.exists()
    video_args.source_events = "1,9,19"

    def fail(*_args):
        raise RuntimeError("CUDA preflight failed")

    monkeypatch.setattr(equalizer, "RifeInterpolator", fail)
    with pytest.raises(RuntimeError, match="CUDA preflight"):
        equalizer.export(video_args)
    assert not video_args.output_dir.exists()


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_failed_export_stays_marked(video_args, monkeypatch, failure):
    monkeypatch.setattr(equalizer, "RifeInterpolator", StandInRife)

    def fail(*_args, **_kwargs):
        raise failure("encoder failed")

    monkeypatch.setattr(equalizer, "_encode_samples", fail)
    with pytest.raises(failure, match="encoder failed"):
        equalizer.export(video_args)
    assert len(list(video_args.output_dir.glob("*/INCOMPLETE.txt"))) == 1
    assert not list(video_args.output_dir.glob("*/event_equalizer.json"))


@pytest.mark.parametrize(
    "fps,count",
    [
        (None, None),
        ("29.970", None),
        ("30000/1001", None),
        ("60", None),
        ("120", "31"),
        (None, "31"),
    ],
)
def test_cli_dry_run_uses_same_plan(video_args, capsys, fps, count):
    video_args.output_fps, video_args.output_frame_count = fps, count
    argv = equalizer.cli_command(video_args)[3:] + ["--dry-run"]
    assert equalizer.main(argv) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert (
        manifest["cameras"]["source"]["rows"]
        == equalizer.prepare(video_args)["cameras"]["source"]["rows"]
    )
    assert not video_args.output_dir.exists()


class CaptureRife(StandInRife):
    """Records which folders the network is loaded from."""

    loaded: dict = {}

    def __init__(self, checkout, weights):
        super().__init__()
        CaptureRife.loaded = {"checkout": checkout, "weights": weights}


def _export_argv(video_args, *extra):
    return [
        "--reference", str(video_args.reference),
        "--source", str(video_args.source),
        "--reference-events", video_args.reference_events,
        "--source-events", video_args.source_events,
        "--output-dir", str(video_args.output_dir),
        *extra,
    ]  # fmt: skip


def test_export_loads_rife_from_the_chosen_home(video_args, monkeypatch, tmp_path, capsys):
    monkeypatch.delenv(rife_paths.ENV_HOME, raising=False)
    monkeypatch.setattr(equalizer, "RifeInterpolator", CaptureRife)
    home = tmp_path / "rife on another disk"
    assert equalizer.main(_export_argv(video_args, "--rife-home", str(home))) == 0
    assert CaptureRife.loaded == {
        "checkout": home / "Practical-RIFE",
        "weights": home / "train_log",
    }
    assert "--rife-home" in capsys.readouterr().out  # the log says where it came from


def test_export_uses_the_environment_home_and_flags_override_it(video_args, monkeypatch, tmp_path):
    monkeypatch.setattr(equalizer, "RifeInterpolator", CaptureRife)
    monkeypatch.setenv(rife_paths.ENV_HOME, str(tmp_path / "env"))
    assert equalizer.main(_export_argv(video_args)) == 0
    assert CaptureRife.loaded["checkout"] == tmp_path / "env" / "Practical-RIFE"
    assert equalizer.main(_export_argv(video_args, "--weights", str(tmp_path / "w"))) == 0
    assert CaptureRife.loaded == {
        "checkout": tmp_path / "env" / "Practical-RIFE",
        "weights": tmp_path / "w",
    }


def test_default_still_finds_the_install_inside_the_checkout(video_args, monkeypatch):
    monkeypatch.delenv(rife_paths.ENV_HOME, raising=False)
    monkeypatch.setattr(equalizer, "RifeInterpolator", CaptureRife)
    assert equalizer.main(_export_argv(video_args)) == 0
    assert CaptureRife.loaded == {
        "checkout": rife_paths.LEGACY_CHECKOUT,
        "weights": rife_paths.LEGACY_WEIGHTS,
    }


def test_a_dry_run_needs_no_rife_at_all(video_args, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(equalizer, "RifeInterpolator", CaptureRife)
    CaptureRife.loaded = {}
    argv = _export_argv(video_args, "--rife-home", str(tmp_path / "nowhere"), "--dry-run")
    assert equalizer.main(argv) == 0
    assert CaptureRife.loaded == {}
    assert json.loads(capsys.readouterr().out)["cameras"]


def test_missing_rife_error_explains_how_to_choose_a_folder(
    video_args, tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv(rife_paths.ENV_HOME, raising=False)
    monkeypatch.setitem(sys.modules, "torch", MagicMock())  # CUDA is checked before the files
    argv = _export_argv(video_args, "--rife-home", str(tmp_path / "empty"))
    assert equalizer.main(argv) == 1
    err = capsys.readouterr().err
    assert "--rife-home" in err and "VAILA_RIFE_HOME" in err and "--export" in err


def test_preview_reports_counts_and_timing(video_args):
    video_args.output_fps, video_args.output_frame_count = "120", "31"
    manifest = equalizer.prepare(video_args)
    text = equalizer.preview_text(manifest)
    assert "120 FPS" in text and "31 frames per camera" in text
    assert "First-to-last frame: 1/4 s" in text
    assert "File duration: 31/120 s" in text
    assert "Duration change:" in text
    assert "Contact rounding/clamping adjustment:" in text
    assert "reference:" in text and "source:" in text


@pytest.mark.parametrize("change_during_preview", [False, True])
def test_gui_reviews_current_settings_before_run(video_args, monkeypatch, change_during_preview):
    """Exercise callbacks without a display; worker output uses the real planner."""
    import tkinter as tk
    from tkinter import messagebox, ttk

    variables, buttons, callbacks, launches = [], {}, [], []
    root, preview = MagicMock(), MagicMock()
    root.after.side_effect = lambda _delay, callback: callbacks.append(callback)

    class Variable:
        def __init__(self, _root, value):
            self.value, self.traces = value, []
            variables.append(self)

        def get(self):
            return self.value

        def set(self, value):
            self.value = value
            for callback in self.traces:
                callback()

        def trace_add(self, _mode, callback):
            self.traces.append(callback)

    def button(_parent, **options):
        buttons[options["text"]] = options
        widget = MagicMock()
        widget.configure.side_effect = lambda **kw: options.update(kw)
        return widget

    # Avoid replacing subprocess used by probe: mock only preparation's probes.
    manifest = equalizer.prepare(video_args)
    infos = {
        camera["path"]: {k: camera[k] for k in ("path", "frame_count", "width", "height", "fps")}
        for camera in manifest["cameras"].values()
    }
    monkeypatch.setattr(equalizer, "probe", lambda path: infos[str(path)])

    def worker(command, *, stdout, **_kwargs):
        launches.append(command)
        if "--dry-run" in command:
            parsed = dict(zip(command[3::2], command[4::2], strict=False))
            args = argparse.Namespace(
                **{key[2:].replace("-", "_"): value for key, value in parsed.items()}
            )
            stdout.write(json.dumps(equalizer.prepare(args)))
        else:
            stdout.write(">> Completed test export\n")
        stdout.flush()
        result = MagicMock()
        result.poll.return_value = 0
        return result

    monkeypatch.setattr(tk, "Tk", lambda: root)
    monkeypatch.setattr(tk, "Text", lambda *_args, **_kw: preview)
    monkeypatch.setattr(tk, "StringVar", Variable)
    for name in ("Label", "Entry", "Frame", "Progressbar"):
        monkeypatch.setattr(ttk, name, MagicMock())
    monkeypatch.setattr(ttk, "Button", button)
    monkeypatch.setattr(messagebox, "showinfo", MagicMock())
    monkeypatch.setattr(messagebox, "showerror", MagicMock())
    monkeypatch.setattr(equalizer.subprocess, "Popen", worker)

    def interact():
        assert buttons["Run"]["state"] == "disabled"
        buttons["120"]["command"]()
        assert variables[4].get() == "120"
        buttons["Preview"]["command"]()
        assert buttons["Run"]["state"] == "disabled"
        assert "--dry-run" in launches[-1]
        if change_during_preview:
            variables[5].set("31")
        callbacks.pop(0)()
        if change_during_preview:
            assert buttons["Run"]["state"] == "disabled"
            buttons["Preview"]["command"]()
            callbacks.pop(0)()
        else:
            assert "85 frames per camera" in preview.insert.call_args.args[1]
            variables[5].set("31")
            assert buttons["Run"]["state"] == "disabled"
            buttons["Preview"]["command"]()
            callbacks.pop(0)()
        assert "31 frames per camera" in preview.insert.call_args.args[1]
        assert buttons["Run"]["state"] == "normal"
        buttons["Run"]["command"]()
        command = launches[-1]
        assert "--dry-run" not in command
        assert command[command.index("--output-fps") + 1] == "120"
        assert command[command.index("--output-frame-count") + 1] == "31"
        callbacks.pop(0)()
        messagebox.showerror.assert_not_called()
        messagebox.showinfo.assert_called_once()

    root.mainloop.side_effect = interact
    equalizer.show_gui(video_args)


def test_syncvid_button_launches_prefilled_gui(monkeypatch, tmp_path):
    from vaila import syncvid

    player = object.__new__(syncvid.PygameSyncPlayer)
    player.video_infos = [
        syncvid.VideoInfo(tmp_path / name, count, 29.97, 1920, 1080)
        for name, count in (("c1.mp4", 19), ("c2.mp4", 22))
    ]
    player.sync_fields = {
        info.name: syncvid.PygameTextField(info.name, "9") for info in player.video_infos
    }
    monkeypatch.setattr(player, "_reference_name_or_none", lambda: "c2.mp4")
    monkeypatch.setattr(player, "_set_status", lambda *_args, **_kwargs: None)
    captured = []
    monkeypatch.setattr(syncvid.subprocess, "Popen", lambda cmd, **kw: captured.append(cmd))
    player._handle_button("side:equalize_rife")
    command = captured[0]
    assert "--gui" in command
    assert command[command.index("--reference") + 1] == str(tmp_path / "c2.mp4")
    assert command[command.index("--reference-events") + 1] == "1,9,22"
    assert command[command.index("--source-events") + 1] == "1,9,19"
