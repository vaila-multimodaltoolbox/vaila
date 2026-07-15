"""Focused tests for cutvideo timeline and cancellable render helpers."""

from vaila import cutvideo


def test_timeline_x_for_frame_reaches_both_edges():
    assert cutvideo.timeline_x_for_frame(0, 101, 10, 200) == 10
    assert cutvideo.timeline_x_for_frame(100, 101, 10, 200) == 210
    assert cutvideo.timeline_x_for_frame(999, 101, 10, 200) == 210
    assert cutvideo.timeline_x_for_frame(0, 0, 10, 200) == 10


def test_cut_timeline_click_snaps_to_marker_in_pixel_column():
    assert (
        cutvideo.frame_index_from_cut_timeline_x(
            mouse_x=2,
            strip_left=0,
            strip_width=10,
            total_frames=100,
            cut_markers=[25, 80],
        )
        == 25
    )
    assert (
        cutvideo.frame_index_from_cut_timeline_x(
            mouse_x=10,
            strip_left=0,
            strip_width=10,
            total_frames=100,
            cut_markers=[],
        )
        == 99
    )


def test_adjacent_cut_marker_frame_wraps_in_both_directions():
    markers = [20, 5, 10, 20]
    assert cutvideo.adjacent_cut_marker_frame(5, markers, 1) == 10
    assert cutvideo.adjacent_cut_marker_frame(20, markers, 1) == 5
    assert cutvideo.adjacent_cut_marker_frame(20, markers, -1) == 10
    assert cutvideo.adjacent_cut_marker_frame(5, markers, -1) == 20
    assert cutvideo.adjacent_cut_marker_frame(5, [], 1) is None


def test_parse_basename_list_one_name_per_line(tmp_path):
    path = tmp_path / "cuts.txt"
    path.write_text(
        "CarlosMiguel_cod_02\nCoutinho_cod_02\nHeittor_cod_02\n",
        encoding="utf-8",
    )
    names = cutvideo.parse_basename_list(path)
    assert names == ["CarlosMiguel_cod_02", "Coutinho_cod_02", "Heittor_cod_02"]


def test_build_cut_output_filenames_uses_per_cut_names(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    cuts = [(0, 10), (20, 30), (40, 50)]
    per_cut = ["CarlosMiguel_cod_02", "Coutinho_cod_02", "Heittor_cod_02"]
    out = cutvideo.build_cut_output_filenames(video, cuts, None, per_cut)
    assert out == [
        "CarlosMiguel_cod_02.mp4",
        "Coutinho_cod_02.mp4",
        "Heittor_cod_02.mp4",
    ]


def test_playback_speed_steps_include_normal_speed():
    assert cutvideo._step_playback_speed(0.5, 1) == 1.0
    assert cutvideo._step_playback_speed(2.0, -1) == 1.0
    assert cutvideo._step_playback_speed(1.0, 1) == 2.0
    assert cutvideo._step_playback_speed(1.0, -1) == 0.5


def test_ffmpeg_render_can_be_cancelled(monkeypatch, tmp_path):
    class FakeProcess:
        returncode = None
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.returncode = -15
            return self.returncode

    process = FakeProcess()
    monkeypatch.setattr(cutvideo.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(cutvideo.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(cutvideo, "encoders_with_cpu_fallback", lambda selected=None: ["libx264"])
    monkeypatch.setattr(cutvideo, "get_ffmpeg_path", lambda: "ffmpeg")
    monkeypatch.setattr(cutvideo, "get_video_encode_ffmpeg_path", lambda encoder=None: "ffmpeg")
    monkeypatch.setattr(
        cutvideo, "get_ffmpeg_video_encoding_args", lambda encoder=None: ["-c:v", "libx264"]
    )

    success = cutvideo.cut_video_with_ffmpeg(
        tmp_path / "input.mp4",
        tmp_path / "output.mp4",
        0,
        10,
        {"fps": 30.0},
        progress_callback=lambda: False,
    )

    assert success is False
    assert process.terminated is True


def test_save_cuts_to_toml_uses_manual_fps_for_times(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    cuts = [(0, 185)]  # frames 1-186 inclusive

    monkeypatch.setattr(
        cutvideo,
        "get_precise_video_metadata",
        lambda _path: {"fps": 30.0, "fps_num": 30, "fps_den": 1},
    )

    toml_path = cutvideo.save_cuts_to_toml(str(video), cuts, fps=120.0)
    content = toml_path.read_text(encoding="utf-8")

    assert "fps = 120.000000" in content
    assert "start_time = 0.008333" in content
    assert "duration = 1.550000" in content
    assert "end_time = 1.558333" in content


def test_cut_times_for_toml_matches_ui_convention():
    start, end, duration = cutvideo._cut_times_for_toml(1, 186, 120, 1)
    assert start == 1 / 120
    assert duration == 186 / 120
    assert end == start + duration
