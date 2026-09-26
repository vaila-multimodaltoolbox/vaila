"""Focused tests for multicam_recorder command building, naming, and stop logic.

No test spawns a real ffmpeg process or opens a real camera (subprocess.Popen
and cv2 access are monkeypatched throughout), matching the repo-wide pattern
for hardware-dependent video modules.

Update Date: 26 September 2026
Version: 0.4.5
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from vaila import multicam_recorder


def _camera(backend: str, device_id: str = "/dev/video0") -> multicam_recorder.CameraDevice:
    return multicam_recorder.CameraDevice(id=device_id, label="Test Camera", backend=backend)


# ── build_record_command ─────────────────────────────────────────────────────


def test_build_record_command_linux_uses_v4l2_input_format():
    cmd = multicam_recorder.build_record_command(
        "ffmpeg",
        _camera("v4l2", "/dev/video2"),
        framerate=30,
        video_size="1280x720",
        output_path=Path("/tmp/out/cam1_trial.mkv"),
    )
    assert cmd[0] == "ffmpeg"
    assert "-nostdin" not in cmd
    assert "-t" not in cmd
    assert "-input_format" in cmd
    assert cmd[cmd.index("-input_format") + 1] == "mjpeg"
    assert "-i" in cmd
    assert cmd[cmd.index("-i") + 1] == "/dev/video2"
    assert cmd[-3:] == ["-c:v", "copy", "/tmp/out/cam1_trial.mkv"]


def test_build_record_command_macos_uses_vcodec_not_input_format():
    cmd = multicam_recorder.build_record_command(
        "ffmpeg",
        _camera("avfoundation", "0"),
        framerate=30,
        video_size="1280x720",
        output_path=Path("/tmp/out/cam1_trial.mkv"),
    )
    assert "-input_format" not in cmd
    assert "-vcodec" in cmd
    assert cmd[cmd.index("-vcodec") + 1] == "mjpeg"
    assert cmd[cmd.index("-i") + 1] == "0"
    assert "-nostdin" not in cmd
    assert "-t" not in cmd


def test_build_record_command_windows_uses_dshow_video_prefix():
    cmd = multicam_recorder.build_record_command(
        "ffmpeg",
        _camera("dshow", "@device_pnp_abc"),
        framerate=30,
        video_size="1280x720",
        output_path=Path("/tmp/out/cam1_trial.mkv"),
    )
    assert "-input_format" not in cmd
    assert cmd[cmd.index("-vcodec") + 1] == "mjpeg"
    assert cmd[cmd.index("-i") + 1] == "video=@device_pnp_abc"


# ── output naming ────────────────────────────────────────────────────────────


def test_build_output_filename_sanitizes_session_name():
    name = multicam_recorder.build_output_filename(1, "trial 01/subject A")
    assert name == "cam1_trial_01_subject_A.mp4"


def test_build_session_dir_uses_sanitized_name_and_timestamp(tmp_path):
    result = multicam_recorder.build_session_dir(tmp_path, "My Trial!", timestamp="20260926_120000")
    assert result == tmp_path / "My_Trial_20260926_120000"


# ── GUI to CLI mirror ─────────────────────────────────────────────────────────


def test_format_record_cli_command_quotes_paths_with_spaces():
    cmd = ["ffmpeg", "-i", "/dev/video0", "/tmp/my session/cam1.mkv"]
    rendered = multicam_recorder.format_record_cli_command(cmd)
    assert shlex.split(rendered) == cmd


# ── device listing parsers ────────────────────────────────────────────────────


def test_enumerate_cameras_linux_globs_dev_video_sorted_numerically(monkeypatch):
    monkeypatch.setattr(
        multicam_recorder.glob,
        "glob",
        lambda pattern: ["/dev/video4", "/dev/video2", "/dev/video10"],
    )
    monkeypatch.setattr(multicam_recorder, "_is_capturable_v4l2_device", lambda path: True)
    devices = multicam_recorder._enumerate_cameras_linux()
    assert [d.id for d in devices] == ["/dev/video2", "/dev/video4", "/dev/video10"]
    assert all(d.backend == "v4l2" for d in devices)


def test_enumerate_cameras_linux_filters_out_non_capture_nodes(monkeypatch):
    """Metadata-only nodes exposed by some UVC webcams must not appear as
    "ghost cameras" alongside the real capture node."""
    monkeypatch.setattr(
        multicam_recorder.glob,
        "glob",
        lambda pattern: ["/dev/video0", "/dev/video1"],
    )
    capturable = {"/dev/video0"}
    monkeypatch.setattr(
        multicam_recorder, "_is_capturable_v4l2_device", lambda path: path in capturable
    )
    devices = multicam_recorder._enumerate_cameras_linux()
    assert [d.id for d in devices] == ["/dev/video0"]


def test_parse_avfoundation_listing_stops_before_audio_section():
    stderr_text = (
        "[AVFoundation indev @ 0x1] AVFoundation video devices:\n"
        "[AVFoundation indev @ 0x1] [0] FaceTime HD Camera\n"
        "[AVFoundation indev @ 0x1] [1] USB Camera\n"
        "[AVFoundation indev @ 0x1] AVFoundation audio devices:\n"
        "[AVFoundation indev @ 0x1] [0] Built-in Microphone\n"
    )
    devices = multicam_recorder._parse_avfoundation_listing(stderr_text)
    assert [(d.id, d.label) for d in devices] == [
        ("0", "FaceTime HD Camera"),
        ("1", "USB Camera"),
    ]
    assert all(d.backend == "avfoundation" for d in devices)


def test_parse_dshow_listing_disambiguates_duplicate_display_names():
    stderr_text = (
        "[dshow @ 0x1] DirectShow video devices (some may be both video and audio devices)\n"
        '[dshow @ 0x1]  "Integrated Camera"\n'
        '[dshow @ 0x1]     Alternative name "@device_pnp_A"\n'
        '[dshow @ 0x1]  "Integrated Camera"\n'
        '[dshow @ 0x1]     Alternative name "@device_pnp_B"\n'
        "[dshow @ 0x1] DirectShow audio devices\n"
        '[dshow @ 0x1]  "Microphone"\n'
    )
    devices = multicam_recorder._parse_dshow_listing(stderr_text)
    assert [d.id for d in devices] == ["@device_pnp_A", "@device_pnp_B"]
    assert all(d.label == "Integrated Camera" for d in devices)
    assert all(d.backend == "dshow" for d in devices)


# ── recording controller (fake Popen, no real subprocess) ───────────────────


class _FakeStdin:
    def __init__(self) -> None:
        self.written = b""
        self.closed = False

    def write(self, data: bytes) -> None:
        if self.closed:
            raise BrokenPipeError("stdin closed")
        self.written += data

    def flush(self) -> None:
        pass


class _FakePopenProcess:
    def __init__(self, already_dead: bool = False, timeout_then_succeed: bool = False) -> None:
        self.stdin = _FakeStdin()
        self._already_dead = already_dead
        self._timeout_then_succeed = timeout_then_succeed
        self._wait_calls = 0
        self.terminated = False
        self.killed = False

    def poll(self):
        return 0 if self._already_dead else None

    def wait(self, timeout: float | None = None):
        self._wait_calls += 1
        if self._already_dead:
            return 0
        if self._timeout_then_succeed and self._wait_calls == 1:
            raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=timeout or 0.0)
        return 0

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def _start_one_camera(monkeypatch, tmp_path, process):
    monkeypatch.setattr(multicam_recorder.subprocess, "Popen", lambda *a, **k: process)
    controller = multicam_recorder.MulticamRecorderController("ffmpeg")
    camera = _camera("v4l2", "/dev/video0")
    controller.start([(camera, 30, "1280x720")], tmp_path / "session", "trial")
    return controller, camera


def test_controller_stop_sends_q_to_alive_process_and_skips_dead_one(monkeypatch, tmp_path):
    alive = _FakePopenProcess()
    dead = _FakePopenProcess(already_dead=True)
    processes = iter([alive, dead])
    monkeypatch.setattr(multicam_recorder.subprocess, "Popen", lambda *a, **k: next(processes))

    controller = multicam_recorder.MulticamRecorderController("ffmpeg")
    cam1 = _camera("v4l2", "/dev/video0")
    cam2 = _camera("v4l2", "/dev/video2")
    controller.start(
        [(cam1, 30, "1280x720"), (cam2, 30, "1280x720")], tmp_path / "session", "trial"
    )

    results = controller.stop(timeout=1)

    assert alive.stdin.written == b"q"
    assert dead.stdin.written == b""
    assert set(results.keys()) == {cam1.id, cam2.id}


def test_controller_stop_escalates_to_terminate_on_timeout(monkeypatch, tmp_path):
    proc = _FakePopenProcess(timeout_then_succeed=True)
    controller, _camera_dev = _start_one_camera(monkeypatch, tmp_path, proc)

    controller.stop(timeout=0.01)

    assert proc.terminated is True
    assert proc.killed is False


def test_controller_poll_disconnected_reports_dead_process_once(monkeypatch, tmp_path):
    proc = _FakePopenProcess()
    controller, camera = _start_one_camera(monkeypatch, tmp_path, proc)

    assert controller.poll_disconnected() == []

    proc._already_dead = True
    dead = controller.poll_disconnected()
    assert len(dead) == 1
    assert dead[0].camera.id == camera.id

    assert controller.poll_disconnected() == []
