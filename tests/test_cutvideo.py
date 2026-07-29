"""Focused tests for cutvideo timeline, sync handoff, and render helpers.

Update Date: 28 July 2026
Version: 0.3.85
"""

from pathlib import Path

import pytest

from vaila import cutvideo, syncvid


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


def test_parse_sync_v2_uses_zero_based_internal_frames(tmp_path: Path):
    video = tmp_path / "camera one.mp4"
    video.write_bytes(b"x")
    sync_file = syncvid.write_sync_file(
        [syncvid.SyncPlanEntry(video.name, "camera one sync.mp4", 10, 20, 15)],
        tmp_path / "sync.txt",
    )
    cuts, is_sync, data = cutvideo.parse_sync_file_content(sync_file, video, strict=True)
    assert cuts == [(10, 20)]
    assert is_sync is True
    assert data[video.name]["initial_frame"] == 10
    assert data[video.name]["final_frame"] == 20


def test_load_cuts_or_sync_uses_explicit_file(tmp_path: Path):
    video = tmp_path / "camera.mp4"
    video.write_bytes(b"x")
    selected = syncvid.write_sync_file(
        [syncvid.SyncPlanEntry(video.name, "selected.mp4", 5, 15, 8)],
        tmp_path / "selected.txt",
    )
    syncvid.write_sync_file(
        [syncvid.SyncPlanEntry(video.name, "automatic.mp4", 100, 110, 105)],
        tmp_path / "automatic_sync.txt",
    )
    cuts, is_sync, data = cutvideo.load_cuts_or_sync(video, sync_file=selected)
    assert cuts == [(5, 15)]
    assert is_sync is True
    assert data[video.name]["new_name"] == "selected.mp4"


def test_safe_sync_child_path_rejects_traversal_and_external_symlink(tmp_path: Path):
    with pytest.raises(syncvid.SyncFileError):
        cutvideo._safe_sync_child_path(tmp_path, "../camera.mp4", field="video_file")

    outside = tmp_path.parent / f"{tmp_path.name}_outside.mp4"
    outside.write_bytes(b"x")
    link = tmp_path / "camera.mp4"
    link.symlink_to(outside)
    with pytest.raises(syncvid.SyncFileError, match="escapes"):
        cutvideo._safe_sync_child_path(
            tmp_path,
            link.name,
            field="video_file",
            must_exist=True,
        )


def test_validate_sync_handoff_checks_all_source_videos(tmp_path: Path):
    reference = tmp_path / "reference.mp4"
    other = tmp_path / "other.mp4"
    reference.write_bytes(b"x")
    other.write_bytes(b"x")
    sync_file = syncvid.write_sync_file(
        [
            syncvid.SyncPlanEntry(reference.name, "reference_sync.mp4", 0, 9, 4),
            syncvid.SyncPlanEntry(other.name, "other_sync.mp4", 2, 11, 6),
        ],
        tmp_path / "sync.txt",
    )
    assert cutvideo.validate_sync_handoff(reference, sync_file) == (
        reference.resolve(),
        sync_file.resolve(),
    )
    other.unlink()
    with pytest.raises(syncvid.SyncFileError):
        cutvideo.validate_sync_handoff(reference, sync_file)


def test_run_cutvideo_direct_handoff_skips_file_dialog(monkeypatch, tmp_path: Path):
    video = tmp_path / "camera.mp4"
    video.write_bytes(b"x")
    sync_file = syncvid.write_sync_file(
        [syncvid.SyncPlanEntry(video.name, "camera_sync.mp4", 0, 9, 4)],
        tmp_path / "sync.txt",
    )
    calls = []
    monkeypatch.setattr(
        cutvideo,
        "play_video_with_cuts",
        lambda selected, *, sync_file=None: calls.append((selected, sync_file)),
    )
    monkeypatch.setattr(cutvideo, "cleanup_resources", lambda: None)
    monkeypatch.setattr(
        cutvideo,
        "get_video_path",
        lambda: pytest.fail("file dialog must not open during direct handoff"),
    )
    assert cutvideo.run_cutvideo(video, sync_file) is True
    assert calls == [(str(video.resolve()), sync_file.resolve())]


class _FakeProgressDialog:
    def __init__(self, _title, _total_steps):
        self.cancelled = False

    def update(self, *_args, **_kwargs):
        return True

    def close(self):
        return None


def test_batch_sync_rejects_path_traversal_before_render(monkeypatch, tmp_path: Path):
    reference = tmp_path / "reference.mp4"
    reference.write_bytes(b"x")
    monkeypatch.setattr(cutvideo, "RenderProgressDialog", _FakeProgressDialog)
    monkeypatch.setattr(
        cutvideo,
        "cut_video_with_ffmpeg",
        lambda *_args, **_kwargs: pytest.fail("unsafe entry must not render"),
    )
    data = {
        "../outside.mp4": {
            "new_name": "safe.mp4",
            "initial_frame": 0,
            "final_frame": 9,
        }
    }
    assert cutvideo.batch_process_sync_videos(reference, data) is False


def test_batch_sync_rejects_out_of_bounds_end_instead_of_truncating(monkeypatch, tmp_path: Path):
    video = tmp_path / "camera.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(cutvideo, "RenderProgressDialog", _FakeProgressDialog)
    monkeypatch.setattr(
        cutvideo,
        "get_precise_video_metadata",
        lambda _path: {"nb_frames": 10, "fps": 30.0},
    )
    monkeypatch.setattr(
        cutvideo,
        "cut_video_with_ffmpeg",
        lambda *_args, **_kwargs: pytest.fail("out-of-bounds range must not render"),
    )
    data = {
        video.name: {
            "new_name": "camera_sync.mp4",
            "initial_frame": 0,
            "final_frame": 10,
        }
    }
    assert cutvideo.batch_process_sync_videos(video, data) is False


def test_batch_sync_reports_success_only_after_every_video(monkeypatch, tmp_path: Path):
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"x")
    second.write_bytes(b"x")
    rendered = []
    monkeypatch.setattr(cutvideo, "RenderProgressDialog", _FakeProgressDialog)
    monkeypatch.setattr(
        cutvideo,
        "get_precise_video_metadata",
        lambda _path: {"nb_frames": 100, "fps": 30.0},
    )
    monkeypatch.setattr(
        cutvideo,
        "cut_video_with_ffmpeg",
        lambda source, *_args, **_kwargs: rendered.append(Path(source).name) or True,
    )
    data = {
        name: {"new_name": f"{Path(name).stem}_sync.mp4", "initial_frame": 0, "final_frame": 9}
        for name in (first.name, second.name)
    }
    assert cutvideo.batch_process_sync_videos(first, data) is True
    assert rendered == [first.name, second.name]
