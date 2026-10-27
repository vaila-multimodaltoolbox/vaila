"""Unit tests for interactive multi-video synchronization.

Update Date: 03 September 2026
Version: 0.3.120
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vaila import syncvid


def _video_info(path: Path, frame_count: int = 1000) -> syncvid.VideoInfo:
    return syncvid.VideoInfo(path=path, frame_count=frame_count, fps=120.0, width=1920, height=1080)


def test_discover_video_files_is_sorted_and_ignores_non_video(tmp_path: Path) -> None:
    (tmp_path / "b.mp4").write_bytes(b"x")
    (tmp_path / "A.mov").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    assert [path.name for path in syncvid.discover_video_files(tmp_path)] == ["A.mov", "b.mp4"]


def test_discover_video_files_rejects_symlink_escape(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"x")
    (target / "camera.mp4").symlink_to(outside)
    with pytest.raises(syncvid.SyncWorkflowError, match="escapes"):
        syncvid.discover_video_files(target)


def test_build_sync_plan_preserves_offsets_and_equal_duration(tmp_path: Path) -> None:
    infos = [
        _video_info(tmp_path / "camera one.mp4"),
        _video_info(tmp_path / "camera_two.mp4", frame_count=800),
    ]
    frames = {"camera one.mp4": 100, "camera_two.mp4": 120}
    assert syncvid.common_reference_bounds(infos, frames, "camera one.mp4") == (0, 779)
    plan = syncvid.build_sync_plan(infos, frames, "camera one.mp4", 50, 200)
    assert [(entry.start_frame, entry.end_frame) for entry in plan] == [(50, 200), (70, 220)]
    assert {entry.end_frame - entry.start_frame for entry in plan} == {150}


def test_build_sync_plan_rejects_missing_marker_and_out_of_bounds(tmp_path: Path) -> None:
    infos = [_video_info(tmp_path / "a.mp4"), _video_info(tmp_path / "b.mp4")]
    with pytest.raises(syncvid.SyncWorkflowError, match="every video"):
        syncvid.build_sync_plan(infos, {"a.mp4": 10}, "a.mp4", 0, 20)
    with pytest.raises(syncvid.SyncWorkflowError, match="common interval"):
        syncvid.build_sync_plan(
            infos,
            {"a.mp4": 10, "b.mp4": 998},
            "a.mp4",
            0,
            20,
        )


def test_resolve_reference_video_accepts_row_name_stem_and_unique_fragment(
    tmp_path: Path,
) -> None:
    infos = [
        _video_info(tmp_path / "camera_diagonal.mp4"),
        _video_info(tmp_path / "camera_frontal.mp4"),
    ]
    assert syncvid.resolve_reference_video("2", infos) == "camera_frontal.mp4"
    assert syncvid.resolve_reference_video("CAMERA_FRONTAL.MP4", infos) == "camera_frontal.mp4"
    assert syncvid.resolve_reference_video("camera_frontal", infos) == "camera_frontal.mp4"
    assert syncvid.resolve_reference_video("frontal", infos) == "camera_frontal.mp4"

    with pytest.raises(syncvid.SyncWorkflowError, match="more than one"):
        syncvid.resolve_reference_video("camera", infos)


@pytest.mark.parametrize("text", ["", "zero", "0", "101"])
def test_parse_frame_field_rejects_empty_noninteger_or_out_of_bounds(text: str) -> None:
    with pytest.raises(syncvid.SyncWorkflowError):
        syncvid.parse_frame_field(text, field="sync frame", frame_count=100)


def test_build_sync_plan_from_typed_fields(tmp_path: Path) -> None:
    infos = [
        _video_info(tmp_path / "camera_diagonal.mp4", frame_count=1000),
        _video_info(tmp_path / "camera_frontal.mp4", frame_count=800),
    ]
    reference, plan = syncvid.build_sync_plan_from_fields(
        infos,
        {
            "camera_diagonal.mp4": "101",
            "camera_frontal.mp4": "121",
        },
        "frontal",
        "71",
        "221",
    )
    assert reference == "camera_frontal.mp4"
    assert [(entry.start_frame, entry.end_frame) for entry in plan] == [
        (50, 200),
        (70, 220),
    ]
    assert {entry.end_frame - entry.start_frame + 1 for entry in plan} == {151}


def test_build_sync_plan_from_fields_requires_every_sync_value(tmp_path: Path) -> None:
    infos = [
        _video_info(tmp_path / "camera_a.mp4"),
        _video_info(tmp_path / "camera_b.mp4"),
    ]
    with pytest.raises(syncvid.SyncWorkflowError, match="camera_b"):
        syncvid.build_sync_plan_from_fields(
            infos,
            {"camera_a.mp4": "10"},
            "1",
            "1",
            "20",
        )


def test_pygame_navigation_matches_cutvideo_style_and_uses_ascii_minus() -> None:
    assert syncvid.frame_after_navigation(50, 100, -60) == 0
    assert syncvid.frame_after_navigation(50, 100, 60) == 99
    assert syncvid.step_playback_speed(1.0, -1) == 0.5
    assert syncvid.step_playback_speed(1.0, 1) == 2.0
    labels = [label for label, _delta in syncvid.PLAYER_STEP_BUTTONS]
    assert labels == ["-60", "-1", "+1", "+60"]
    assert all("−" not in label and "@" not in label for label in labels)


def test_pygame_text_field_replaces_selected_text_and_commits() -> None:
    pygame = syncvid.pygame
    field = syncvid.PygameTextField("reference", "old-camera.mp4")
    field.activate()
    typed = pygame.event.Event(
        pygame.KEYDOWN,
        key=pygame.K_f,
        mod=0,
        unicode="f",
    )
    assert field.handle_key(typed) is False
    assert field.text == "f"
    committed = pygame.event.Event(
        pygame.KEYDOWN,
        key=pygame.K_RETURN,
        mod=0,
        unicode="\r",
    )
    assert field.handle_key(committed) is True


def test_pygame_player_minus_and_up_keys_move_sixty_frames() -> None:
    pygame = syncvid.pygame
    player = object.__new__(syncvid.PygameSyncPlayer)
    player.playing = False
    deltas = []
    player._seek_relative = deltas.append

    minus = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_MINUS, mod=0, unicode="-")
    up = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP, mod=0, unicode="")
    player._handle_player_key(minus)
    player._handle_player_key(up)
    assert deltas == [-60, 60]


def test_sync_file_round_trip_supports_spaces_and_replaces_existing(tmp_path: Path) -> None:
    path = tmp_path / "session sync.txt"
    entries = [
        syncvid.SyncPlanEntry("camera one.mp4", "camera one synced.mp4", 10, 20, 15),
        syncvid.SyncPlanEntry("camera two.mp4", "camera two synced.mp4", 30, 40, 35),
    ]
    syncvid.write_sync_file(entries, path)
    assert syncvid.read_sync_file(path) == entries

    syncvid.write_sync_file(entries[:1], path)
    assert syncvid.read_sync_file(path) == entries[:1]
    assert path.read_text(encoding="utf-8").count("camera one.mp4") == 1


def test_read_sync_file_supports_legacy_quoted_rows(tmp_path: Path) -> None:
    path = tmp_path / "legacy.txt"
    path.write_text('"camera one.mp4" "camera one sync.mp4" 11 21\n', encoding="utf-8")
    assert syncvid.read_sync_file(path) == [
        syncvid.SyncPlanEntry("camera one.mp4", "camera one sync.mp4", 10, 20, None)
    ]


@pytest.mark.parametrize(
    "row",
    [
        "../camera.mp4 output.mp4 1 10",
        "camera.mp4 ../../output.mp4 1 10",
        "camera.mp4 output.mp4 10 1",
        "camera.mp4 output.mp4 0 10",
    ],
)
def test_read_sync_file_rejects_unsafe_or_invalid_rows(tmp_path: Path, row: str) -> None:
    path = tmp_path / "unsafe.txt"
    path.write_text(row + "\n", encoding="utf-8")
    with pytest.raises(syncvid.SyncFileError):
        syncvid.read_sync_file(path)


def test_read_sync_file_rejects_duplicate_video_names(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.txt"
    path.write_text(
        "camera.mp4 out1.mp4 1 10\nCAMERA.mp4 out2.mp4 2 11\n",
        encoding="utf-8",
    )
    with pytest.raises(syncvid.SyncFileError, match="duplicate"):
        syncvid.read_sync_file(path)


def test_find_sync_entry_never_uses_substring_matching() -> None:
    entries = [
        syncvid.SyncPlanEntry("cam1.mp4", "out1.mp4", 0, 9),
        syncvid.SyncPlanEntry("cam10.mp4", "out10.mp4", 0, 9),
    ]
    assert syncvid.find_sync_entry(entries, "cam1.mp4") == entries[0]
    assert syncvid.find_sync_entry(entries, "prefix_cam1_extra.mp4") is None


def test_scrub_cv2_qt_plugin_env_removes_only_cv2s_own_leaked_paths(monkeypatch) -> None:
    # cv2's import unconditionally overwrites these two vars with paths into
    # its own bundled (often system-incompatible) Qt runtime. Every child
    # process this module spawns -- Help's webbrowser tab, the Cut Video
    # handoff subprocess -- inherits whatever is left in os.environ, and a
    # Qt-based helper anywhere in that chain crashes with "Could not load the
    # Qt platform plugin xcb" / "Aborted (core dumped)" if the leaked value
    # survives. A legitimately-set, unrelated Qt path must be left alone.
    monkeypatch.setenv(
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        "/home/user/venv/lib/site-packages/cv2/qt/plugins",
    )
    monkeypatch.setenv("QT_QPA_FONTDIR", "C:\\Users\\me\\venv\\Lib\\site-packages\\cv2\\qt\\fonts")
    syncvid._scrub_cv2_qt_plugin_env()
    assert "QT_QPA_PLATFORM_PLUGIN_PATH" not in os.environ
    assert "QT_QPA_FONTDIR" not in os.environ

    monkeypatch.setenv("QT_QPA_PLATFORM_PLUGIN_PATH", "/usr/lib/x86_64-linux-gnu/qt5/plugins")
    syncvid._scrub_cv2_qt_plugin_env()
    assert os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] == "/usr/lib/x86_64-linux-gnu/qt5/plugins"


def test_checkpoint_round_trips_through_build_write_read(tmp_path: Path) -> None:
    state = syncvid.build_checkpoint_state(
        reference="camera one.mp4",
        start="10",
        end="200",
        sync_frames={"camera one.mp4": "50", "camera two.mp4": "60"},
        current_video="camera two.mp4",
        current_frame=123,
        output_file=tmp_path / "sync.txt",
        completed=True,
    )
    checkpoint_path = syncvid.write_checkpoint_file(state, tmp_path / "sync.json")
    assert checkpoint_path.is_file()

    loaded = syncvid.read_checkpoint_file(checkpoint_path)
    assert loaded["vaila_checkpoint"] == syncvid.CHECKPOINT_SCHEMA_VERSION
    assert loaded["reference"] == "camera one.mp4"
    assert loaded["sync_frames"] == {"camera one.mp4": "50", "camera two.mp4": "60"}
    assert loaded["current_video"] == "camera two.mp4"
    assert loaded["current_frame"] == 123
    assert loaded["output_file"] == str(tmp_path / "sync.txt")
    assert loaded["completed"] is True


def test_checkpoint_state_accepts_none_output_file(tmp_path: Path) -> None:
    state = syncvid.build_checkpoint_state(
        reference="a.mp4",
        start="1",
        end="10",
        sync_frames={"a.mp4": "5"},
        current_video="a.mp4",
        current_frame=0,
        output_file=None,
        completed=False,
    )
    checkpoint_path = syncvid.write_checkpoint_file(state, tmp_path / "in_progress.json")
    loaded = syncvid.read_checkpoint_file(checkpoint_path)
    assert loaded["output_file"] is None
    assert loaded["completed"] is False


def test_write_checkpoint_file_replaces_existing_atomically(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    first = syncvid.build_checkpoint_state(
        reference="a.mp4",
        start="1",
        end="10",
        sync_frames={"a.mp4": "5"},
        current_video="a.mp4",
        current_frame=0,
        output_file=None,
        completed=False,
    )
    syncvid.write_checkpoint_file(first, path)
    second = dict(first)
    second["current_frame"] = 99
    syncvid.write_checkpoint_file(second, path)
    assert syncvid.read_checkpoint_file(path)["current_frame"] == 99
    # No leftover temp files from the atomic write/replace dance.
    assert list(tmp_path.glob(".*.tmp")) == []


def test_read_checkpoint_file_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(syncvid.SyncFileError, match="not found"):
        syncvid.read_checkpoint_file(tmp_path / "missing.json")


def test_read_checkpoint_file_rejects_oversized_file(tmp_path: Path) -> None:
    path = tmp_path / "huge.json"
    path.write_text("x" * (syncvid.MAX_CHECKPOINT_FILE_BYTES + 1), encoding="utf-8")
    with pytest.raises(syncvid.SyncFileError, match="larger than"):
        syncvid.read_checkpoint_file(path)


def test_read_checkpoint_file_rejects_corrupt_json(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(syncvid.SyncFileError, match="not valid JSON"):
        syncvid.read_checkpoint_file(path)


def test_read_checkpoint_file_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "array.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(syncvid.SyncFileError, match="JSON object"):
        syncvid.read_checkpoint_file(path)


def test_read_checkpoint_file_rejects_missing_keys(tmp_path: Path) -> None:
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"vaila_checkpoint": 1, "reference": "a.mp4"}), encoding="utf-8")
    with pytest.raises(syncvid.SyncFileError, match="missing key"):
        syncvid.read_checkpoint_file(path)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"reference": 123}, "must be text"),
        ({"current_frame": "5"}, "non-negative integer"),
        ({"current_frame": -1}, "non-negative integer"),
        ({"current_frame": True}, "non-negative integer"),
        ({"sync_frames": {"a.mp4": 5}}, "map video names to text"),
        ({"sync_frames": "not-a-dict"}, "map video names to text"),
        ({"output_file": 5}, "must be text or null"),
    ],
)
def test_read_checkpoint_file_rejects_wrong_field_types(
    tmp_path: Path, overrides: dict, match: str
) -> None:
    base = {
        "vaila_checkpoint": 1,
        "reference": "a.mp4",
        "start": "1",
        "end": "10",
        "sync_frames": {"a.mp4": "5"},
        "current_video": "a.mp4",
        "current_frame": 0,
    }
    base.update(overrides)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(syncvid.SyncFileError, match=match):
        syncvid.read_checkpoint_file(path)


def test_resilient_open_local_html_uses_browser_env_var_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    help_file = tmp_path / "help.html"
    help_file.write_text("<html></html>", encoding="utf-8")
    monkeypatch.setenv("BROWSER", "my-custom-browser")
    monkeypatch.setattr(syncvid.shutil, "which", lambda _name: None)
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        raise AssertionError("stop after capturing the first candidate")

    monkeypatch.setattr(syncvid.subprocess, "Popen", fake_popen)
    with pytest.raises(AssertionError, match="stop after"):
        syncvid._resilient_open_local_html(help_file)
    assert captured["command"][0] == "my-custom-browser"


def test_resilient_open_local_html_tries_native_browsers_before_wslview(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """wslview must never be tried while a native Linux browser is available,
    even when WSL interop looks viable -- native browsers are strictly
    preferred, matching the task's ordering requirement."""
    help_file = tmp_path / "help.html"
    help_file.write_text("<html></html>", encoding="utf-8")
    monkeypatch.delenv("BROWSER", raising=False)
    monkeypatch.setattr(
        syncvid.shutil, "which", lambda name: "/usr/bin/xdg-open" if name == "xdg-open" else None
    )
    monkeypatch.setattr(syncvid.Path, "exists", lambda self: True)
    launched: list[list[str]] = []

    def fake_popen(command, **kwargs):
        launched.append(command)

        class _Proc:
            pass

        return _Proc()

    monkeypatch.setattr(syncvid.subprocess, "Popen", fake_popen)
    syncvid._resilient_open_local_html(help_file)
    assert launched
    assert launched[0][0] == "xdg-open"


def test_resilient_open_local_html_only_tries_wslview_when_interop_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    help_file = tmp_path / "help.html"
    help_file.write_text("<html></html>", encoding="utf-8")
    monkeypatch.delenv("BROWSER", raising=False)
    monkeypatch.setattr(
        syncvid.shutil, "which", lambda name: "/usr/bin/wslview" if name == "wslview" else None
    )

    # Interop marker absent: wslview must never be attempted, so Popen is
    # never called and we fall through to the print-URI last resort.
    monkeypatch.setattr(syncvid.Path, "exists", lambda self: False)

    def fail_popen(command, **kwargs):
        raise AssertionError(f"must not launch a browser when interop is unavailable: {command}")

    monkeypatch.setattr(syncvid.subprocess, "Popen", fail_popen)
    monkeypatch.setattr(
        syncvid.webbrowser,
        "open_new_tab",
        lambda _uri: (_ for _ in ()).throw(syncvid.webbrowser.Error("no browser")),
    )
    syncvid._resilient_open_local_html(help_file)  # must not raise


def test_resilient_open_local_html_falls_back_to_printing_uri(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    help_file = tmp_path / "help.html"
    help_file.write_text("<html></html>", encoding="utf-8")
    monkeypatch.delenv("BROWSER", raising=False)
    monkeypatch.setattr(syncvid.shutil, "which", lambda _name: None)
    monkeypatch.setattr(syncvid.Path, "exists", lambda self: False)
    monkeypatch.setattr(
        syncvid.webbrowser,
        "open_new_tab",
        lambda _uri: (_ for _ in ()).throw(syncvid.webbrowser.Error("no browser")),
    )
    syncvid._resilient_open_local_html(help_file)
    # rich.print wraps long lines at the console width, so the URI itself may
    # be split across lines in the captured output -- compare with newlines
    # collapsed instead of a literal substring match.
    out = capsys.readouterr().out.replace("\n", "")
    assert help_file.resolve().as_uri() in out


def test_resilient_open_local_html_launch_failure_falls_through_to_next_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A candidate whose Popen() raises OSError must not propagate or crash
    the caller -- the next candidate (or the final print fallback) is tried."""
    help_file = tmp_path / "help.html"
    help_file.write_text("<html></html>", encoding="utf-8")
    monkeypatch.delenv("BROWSER", raising=False)
    monkeypatch.setattr(
        syncvid.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in ("xdg-open", "firefox") else None,
    )
    monkeypatch.setattr(syncvid.Path, "exists", lambda self: False)
    attempted: list[str] = []

    def flaky_popen(command, **kwargs):
        attempted.append(command[0])
        if command[0] == "xdg-open":
            raise OSError("no display")
        return object()

    monkeypatch.setattr(syncvid.subprocess, "Popen", flaky_popen)
    syncvid._resilient_open_local_html(help_file)  # must not raise
    assert attempted == ["xdg-open", "firefox"]


def test_cutvideo_handoff_uses_argument_list_without_shell(monkeypatch, tmp_path: Path) -> None:
    video = tmp_path / "camera.mp4"
    video.write_bytes(b"x")
    sync_file = syncvid.write_sync_file(
        [syncvid.SyncPlanEntry(video.name, "camera_sync.mp4", 0, 9, 4)],
        tmp_path / "sync.txt",
    )
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(syncvid.subprocess, "Popen", fake_popen)
    syncvid.launch_cutvideo(video, sync_file)
    assert captured["command"][1:3] == ["-m", "vaila.cutvideo"]
    assert captured["command"][-4:] == ["--video", str(video), "--sync-file", str(sync_file)]
    assert captured["kwargs"]["shell"] is False


def _write_video_with_repeats(path: Path, unique: int, repeat_after: set[int]) -> int:
    """Write a noisy moving-square video; frames in repeat_after get one re-encoded copy.

    The copy adds grey-level noise, as an editor's H.264 re-encode does, so it is
    not pixel-identical to the frame it repeats.
    """
    import cv2
    import numpy as np

    rng = np.random.default_rng(0)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 60.0, (160, 120))
    written = 0
    for index in range(unique):
        frame = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
        cv2.rectangle(frame, (index * 3, 20), (index * 3 + 30, 60), (255, 255, 255), -1)
        writer.write(frame)
        written += 1
        if index in repeat_after:
            noise = rng.normal(0.0, 6.0, frame.shape)
            writer.write(np.clip(frame + noise, 0, 255).astype(np.uint8))
            written += 1
    writer.release()
    return written


def test_repeat_threshold_cuts_at_noise_motion_gap() -> None:
    # Scores measured on the real cobrador.mp4: copies <= 16.5, motion >= 40.2.
    copies = [0.2, 0.6, 3.0, 8.9, 16.5]
    motion = [40.2, 61.4, 95.8, 133.7]
    threshold, ratio = syncvid.repeat_threshold(copies + motion)
    assert threshold == 16.5
    assert ratio == pytest.approx(40.2 / 16.5)

    # A video without repeats (cabine1.mp4 range) has no wide gap: only invisible changes repeat.
    continuous = [11.3, 12.4, 13.7, 15.1, 16.9, 18.9, 21.1, 23.7, 26.3, 29.2]
    assert syncvid.repeat_threshold([*continuous, 0.4])[0] == syncvid.REPEAT_INVISIBLE_SCORE


def test_remove_repeated_frames_drops_only_the_copies(tmp_path: Path) -> None:
    video = tmp_path / "spliced.avi"
    total = _write_video_with_repeats(video, 20, {3, 9, 10})
    report = syncvid.find_repeated_frames(video)
    assert report.total_frames == total == 23
    # copies sit right after frames 3, 9 and 10 (0-based in the written stream: 4, 11, 13)
    assert report.repeated_frames == (4, 11, 13)
    assert report.threshold > syncvid.REPEAT_INVISIBLE_SCORE
    assert len(report.kept_frames) == 20
    assert report.estimated_fps == pytest.approx(60.0 * 20 / 23)

    out_dir, reports = syncvid.remove_repeated_frames([video], tmp_path / "out")
    assert [r.video for r in reports] == [video.resolve()]
    output = out_dir / f"spliced_norepeat_{60.0 * 20 / 23:.2f}fps.mp4"
    out_report = syncvid.find_repeated_frames(output)
    assert out_report.total_frames == 20
    assert out_report.fps == pytest.approx(60.0 * 20 / 23, rel=1e-3)
    rows = (out_dir / "spliced_norepeat_frames.csv").read_text().splitlines()
    assert rows[0] == "new_frame,original_frame,original_time_s"
    assert rows[5].split(",")[:2] == ["5", "6"]
    scores = (out_dir / "spliced_motion_scores.csv").read_text().splitlines()
    assert scores[5].split(",")[2] == "1"
    summary = (out_dir / "norepeat_summary.csv").read_text().splitlines()
    assert summary[1].split(",")[:4] == ["spliced.avi", "23", "20", "3"]


def test_remove_repeated_frames_leaves_clean_video_unwritten(tmp_path: Path) -> None:
    video = tmp_path / "clean.avi"
    _write_video_with_repeats(video, 12, set())
    out_dir, reports = syncvid.remove_repeated_frames([video], tmp_path / "out")
    assert reports[0].repeated_frames == ()
    assert not list(out_dir.glob("clean_norepeat*.mp4"))


def test_pygame_player_held_arrow_starts_repeat_and_zero_resets_zoom() -> None:
    pygame = syncvid.pygame
    player = object.__new__(syncvid.PygameSyncPlayer)
    player.playing = False
    deltas = []
    player._seek_relative = deltas.append
    player.zoom_level = 3.0
    player.offset_x = player.offset_y = 50.0

    right = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT, mod=0, unicode="")
    player._handle_player_key(right)
    assert deltas == [1]
    assert player.nav_hold_key == pygame.K_RIGHT

    zero = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_0, mod=0, unicode="0")
    player._handle_player_key(zero)
    assert (player.zoom_level, player.offset_x, player.offset_y) == (1.0, 0.0, 0.0)


def test_sync_file_round_trips_removed_frames_inside_range(tmp_path: Path) -> None:
    infos = [_video_info(tmp_path / "a.mp4", 200), _video_info(tmp_path / "b.mp4", 200)]
    _reference, entries = syncvid.build_sync_plan_from_fields(
        infos,
        {"a.mp4": "11", "b.mp4": "21"},
        "a",
        "11",
        "60",
        {"b.mp4": {5, 25, 30, 199}},  # 5 and 199 fall outside b's range 21..70
    )
    assert entries[0].removed_frames == ()
    assert entries[1].removed_frames == (25, 30)

    sync = syncvid.write_sync_file(entries, tmp_path / "sync.txt")
    text = sync.read_text()
    assert "\tremoved_frames\n" in text
    assert text.splitlines()[-1].endswith("\t26 31")
    assert syncvid.read_sync_file(sync) == entries
    data = syncvid.sync_entries_to_cutvideo_data(entries)
    assert data["b.mp4"]["removed_frames"] == (25, 30)

    # Without marks the file keeps the five classic columns.
    plain = syncvid.write_sync_file(
        syncvid.build_sync_plan_from_fields(infos, {"a.mp4": "1", "b.mp4": "1"}, "a", "1", "9")[1],
        tmp_path / "plain.txt",
    )
    assert "removed_frames" not in plain.read_text()


@pytest.mark.parametrize("removed", ["5", "12 12", "x"])
def test_read_sync_file_rejects_bad_removed_frames(tmp_path: Path, removed: str) -> None:
    sync = tmp_path / "sync.txt"
    sync.write_text(
        "video_file\toutput_file\tstart_frame\tend_frame\tsync_frame\tremoved_frames\n"
        f"a.mp4\ta_out.mp4\t10\t20\t10\t{removed}\n"
    )
    with pytest.raises(syncvid.SyncFileError):
        syncvid.read_sync_file(sync)


def test_checkpoint_round_trips_removed_frames(tmp_path: Path) -> None:
    state = syncvid.build_checkpoint_state(
        reference="a.mp4",
        start="1",
        end="9",
        sync_frames={"a.mp4": "1"},
        current_video="a.mp4",
        current_frame=0,
        output_file=None,
        completed=False,
        removed_frames={"a.mp4": {4, 2}, "b.mp4": set()},
    )
    assert state["removed_frames"] == {"a.mp4": [3, 5]}
    path = syncvid.write_checkpoint_file(state, tmp_path / "c.json")
    assert syncvid.read_checkpoint_file(path)["removed_frames"] == {"a.mp4": [3, 5]}

    state["removed_frames"] = {"a.mp4": [0]}
    syncvid.write_checkpoint_file(state, path)
    with pytest.raises(syncvid.SyncFileError):
        syncvid.read_checkpoint_file(path)


def test_pygame_player_d_toggles_the_current_frame_mark(tmp_path: Path) -> None:
    pygame = syncvid.pygame
    player = object.__new__(syncvid.PygameSyncPlayer)
    player.video_infos = [_video_info(tmp_path / "a.mp4", 50)]
    player.current_index = 0
    player.current_frame = 7
    player.playing = False
    player.removed_frames = {"a.mp4": set()}

    key_d = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d, mod=0, unicode="d")
    player._handle_player_key(key_d)
    assert player.removed_frames["a.mp4"] == {7}
    delete = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DELETE, mod=0, unicode="")
    player._handle_player_key(delete)
    assert player.removed_frames["a.mp4"] == set()


def test_write_video_without_frames_maps_cut_frames_to_source(tmp_path: Path) -> None:
    cut = tmp_path / "cut.avi"
    _write_video_with_repeats(cut, 20, {3, 9, 10})  # copies at 4, 11, 13
    info = syncvid.probe_video(cut)
    output = syncvid.write_video_without_frames(info, {4, 11, 13}, tmp_path, frame_offset=100)
    assert syncvid.find_repeated_frames(output).total_frames == 20
    rows = (tmp_path / "cut_norepeat_frames.csv").read_text().splitlines()
    # new frame 5 is cut frame 5 (0-based), i.e. source frame 106 (1-based)
    assert rows[5].split(",")[:2] == ["5", "106"]


def _rate_info(path: Path, frame_count: int, fps: float) -> syncvid.VideoInfo:
    return syncvid.VideoInfo(path=path, frame_count=frame_count, fps=fps, width=640, height=360)


def test_build_sync_plan_maps_keyframes_across_frame_rates(tmp_path: Path) -> None:
    infos = [
        _rate_info(tmp_path / "ref.mp4", 600, 60.0),
        _rate_info(tmp_path / "phone.mp4", 300, 30.0),
        _rate_info(tmp_path / "slow.mp4", 300, 22.956),
    ]
    # Reference keyframe 100, cut 80..200 (2 s); phone keyframe 40 at 30 fps.
    plan = syncvid.build_sync_plan(
        infos, {"ref.mp4": 100, "phone.mp4": 40, "slow.mp4": 30}, "ref.mp4", 80, 200
    )
    ranges = {entry.video_file: (entry.start_frame, entry.end_frame) for entry in plan}
    assert ranges["ref.mp4"] == (80, 200)
    assert ranges["phone.mp4"] == (30, 90)  # 40 - 10, then 2 s at 30 fps
    assert ranges["slow.mp4"] == (30 + round(-20 / 60 * 22.956), 22 + round(2 * 22.956))
    for entry in plan:
        fps = next(info.fps for info in infos if info.name == entry.video_file)
        assert abs((entry.end_frame - entry.start_frame) / fps - 2.0) <= 0.5 / fps
    lines = syncvid.describe_sync_plan(plan, infos, "ref.mp4")
    assert lines[0].endswith("[reference]")
    assert all("imply" not in line for line in lines)


def test_build_sync_plan_uses_own_start_end_without_sync_frame(tmp_path: Path) -> None:
    infos = [_rate_info(tmp_path / "ref.mp4", 600, 60.0), _rate_info(tmp_path / "b.mp4", 300, 30.0)]
    reference, plan = syncvid.build_sync_plan_from_fields(
        infos,
        {"ref.mp4": "", "b.mp4": ""},
        "ref",
        "81",
        "201",
        range_texts={"b.mp4": ("31", "100")},
    )
    assert reference == "ref.mp4"
    assert [(e.start_frame, e.end_frame, e.sync_frame) for e in plan] == [
        (80, 200, None),
        (30, 99, None),
    ]
    assert plan[1].output_file == "b_frames_31_to_100.mp4"
    # 69 frames over the reference's 2 s: a 34.5 fps camera, not 30.
    assert "imply 34.500 fps" in syncvid.describe_sync_plan(plan, infos, reference)[1]


@pytest.mark.parametrize(
    ("sync_texts", "range_texts", "message"),
    [
        ({"ref.mp4": "100", "b.mp4": ""}, {"b.mp4": ("31", "")}, "both Start and End"),
        ({"ref.mp4": "", "b.mp4": "40"}, {}, "sync frame for ref.mp4"),
        ({"ref.mp4": "100", "b.mp4": ""}, {}, "sync frame for b.mp4"),
        ({"ref.mp4": "100", "b.mp4": "295"}, {}, "common interval"),
    ],
)
def test_build_sync_plan_from_fields_reports_missing_or_bad_values(
    tmp_path: Path, sync_texts: dict[str, str], range_texts: dict, message: str
) -> None:
    infos = [_rate_info(tmp_path / "ref.mp4", 600, 60.0), _rate_info(tmp_path / "b.mp4", 300, 30.0)]
    with pytest.raises(syncvid.SyncWorkflowError, match=message):
        syncvid.build_sync_plan_from_fields(
            infos, sync_texts, "ref", "81", "201", range_texts=range_texts
        )


def test_common_reference_bounds_is_time_based(tmp_path: Path) -> None:
    infos = [_rate_info(tmp_path / "ref.mp4", 600, 60.0), _rate_info(tmp_path / "b.mp4", 100, 30.0)]
    # b covers 1.0 s before its keyframe 30 and 2.3 s after: reference frames 40..238.
    assert syncvid.common_reference_bounds(infos, {"ref.mp4": 100, "b.mp4": 30}, "ref.mp4") == (
        40,
        238,
    )


def test_apply_fps_overrides_and_fill_range_texts(tmp_path: Path) -> None:
    """Matching overrides apply; mismatched ones are ignored (keep file fps)."""
    infos = [
        _rate_info(tmp_path / "ref.mp4", 300, 60.0),
        _rate_info(tmp_path / "phone.mp4", 300, 30.0),
        _rate_info(tmp_path / "norepeat.mp4", 100, 13.91),
    ]
    overridden, warnings = syncvid.apply_fps_overrides(
        infos, {"phone.mp4": 30.0, "norepeat.mp4": 29.97}
    )
    assert overridden[0].fps == 60.0
    assert overridden[1].fps == 30.0
    assert overridden[2].fps == 13.91  # 29.97 mismatch ignored
    assert any("norepeat.mp4" in text for text in warnings)

    _reference, plan = syncvid.build_sync_plan_from_fields(
        overridden[:2],
        {"ref.mp4": "101", "phone.mp4": "51"},
        "ref",
        "81",
        "201",
        range_texts={"ref.mp4": ("", ""), "phone.mp4": ("", "")},
    )
    ranges = {entry.video_file: (entry.start_frame, entry.end_frame) for entry in plan}
    assert ranges["ref.mp4"] == (80, 200)
    assert ranges["phone.mp4"] == (40, 100)
    filled = syncvid.fill_range_texts_from_plan(plan, reference_video="ref.mp4")
    assert filled == {"phone.mp4": ("41", "101")}


def test_flamengo_mismatch_override_keeps_time_not_equal_frames(tmp_path: Path) -> None:
    """29.97 typed on a 13.91 norepeat file must not force 19 frames (duration would drift)."""
    infos = [
        _rate_info(tmp_path / "c1.mp4", 81, 29.970030),
        _rate_info(tmp_path / "c2_norepeat.mp4", 71, 13.907661),
    ]
    effective, warnings = syncvid.apply_fps_overrides(
        infos, {"c2_norepeat.mp4": 29.97}
    )
    assert effective[1].fps == 13.907661
    assert warnings
    plan = syncvid.build_sync_plan(
        effective, {"c1.mp4": 70, "c2_norepeat.mp4": 57}, "c1.mp4", 62, 80
    )
    ref_n = plan[0].end_frame - plan[0].start_frame + 1
    c2_n = plan[1].end_frame - plan[1].start_frame + 1
    assert ref_n == 19
    assert c2_n != ref_n
    ref_s = (plan[0].end_frame - plan[0].start_frame) / effective[0].fps
    c2_s = (plan[1].end_frame - plan[1].start_frame) / effective[1].fps
    assert abs(ref_s - c2_s) <= 0.5 / effective[1].fps


def test_equal_fps_forces_identical_frame_count_even_with_noise(tmp_path: Path) -> None:
    """Cameras at ~same fps share the reference frame count (time-aligned start)."""
    infos = [
        _rate_info(tmp_path / "ref.mp4", 200, 29.970030),
        _rate_info(tmp_path / "cam.mp4", 200, 29.97),
    ]
    plan = syncvid.build_sync_plan(
        infos, {"ref.mp4": 70, "cam.mp4": 70}, "ref.mp4", 62, 80
    )
    ref_entry, cam_entry = plan
    assert ref_entry.end_frame - ref_entry.start_frame == cam_entry.end_frame - cam_entry.start_frame
    assert (ref_entry.end_frame - ref_entry.start_frame + 1) == 19
    # Hand-typed End that disagrees is snapped when fps matches.
    plan2 = syncvid.build_sync_plan(
        infos,
        {"ref.mp4": 70, "cam.mp4": 70},
        "ref.mp4",
        62,
        80,
        ranges={"cam.mp4": (50, 99)},  # wrong length on purpose
    )
    assert plan2[1].start_frame == 50
    assert plan2[1].end_frame == 50 + (80 - 62)
    assert plan2[1].end_frame - plan2[1].start_frame == plan2[0].end_frame - plan2[0].start_frame


def test_different_fps_keeps_time_align_with_different_frame_counts(tmp_path: Path) -> None:
    infos = [
        _rate_info(tmp_path / "ref.mp4", 200, 29.970030),
        _rate_info(tmp_path / "slow.mp4", 100, 13.907661),
    ]
    plan = syncvid.build_sync_plan(
        infos, {"ref.mp4": 70, "slow.mp4": 57}, "ref.mp4", 62, 80
    )
    ref_n = plan[0].end_frame - plan[0].start_frame + 1
    slow_n = plan[1].end_frame - plan[1].start_frame + 1
    assert ref_n != slow_n
    ref_s = (plan[0].end_frame - plan[0].start_frame) / infos[0].fps
    slow_s = (plan[1].end_frame - plan[1].start_frame) / infos[1].fps
    assert abs(ref_s - slow_s) <= 0.5 / infos[1].fps


def test_parse_fps_field_rejects_bad_values() -> None:
    assert syncvid.parse_fps_field("29.97") == pytest.approx(29.97)
    assert syncvid.parse_fps_field("13,91") == pytest.approx(13.91)
    with pytest.raises(syncvid.SyncWorkflowError, match="Enter a"):
        syncvid.parse_fps_field("  ")
    with pytest.raises(syncvid.SyncWorkflowError, match="Invalid"):
        syncvid.parse_fps_field("fast")


def test_checkpoint_round_trips_fps_overrides(tmp_path: Path) -> None:
    state = syncvid.build_checkpoint_state(
        reference="a.mp4",
        start="1",
        end="9",
        sync_frames={"a.mp4": "1", "b.mp4": "2"},
        current_video="a.mp4",
        current_frame=0,
        output_file=None,
        completed=False,
        fps_overrides={"a.mp4": 29.97, "b.mp4": 13.91},
    )
    path = syncvid.write_checkpoint_file(state, tmp_path / "c.json")
    loaded = syncvid.read_checkpoint_file(path)
    assert loaded["fps_overrides"] == {"a.mp4": 29.97, "b.mp4": 13.91}

    state["fps_overrides"] = {"a.mp4": -1.0}
    syncvid.write_checkpoint_file(state, path)
    with pytest.raises(syncvid.SyncFileError, match="fps_overrides"):
        syncvid.read_checkpoint_file(path)


def test_flamengo_style_norepeat_mix_fill_aligns_durations(tmp_path: Path) -> None:
    """Mixed container/norepeat fps: Fill Start/End keeps one shared time window."""
    # Mirrors the flamengo_estudiantes/bruno_henrique/new mix (c1@29.97 + c2@13.91).
    infos = [
        _rate_info(tmp_path / "c1_main.mp4", 81, 29.970030),
        _rate_info(tmp_path / "c2_norepeat.mp4", 71, 13.907661),
    ]
    _reference, plan = syncvid.build_sync_plan_from_fields(
        infos,
        {"c1_main.mp4": "71", "c2_norepeat.mp4": "58"},
        "c1_main",
        "63",
        "81",
    )
    ref_entry, other = plan
    ref_s = (ref_entry.end_frame - ref_entry.start_frame) / infos[0].fps
    other_s = (other.end_frame - other.start_frame) / infos[1].fps
    assert abs(ref_s - other_s) <= 0.5 / infos[1].fps
    filled = syncvid.fill_range_texts_from_plan(plan, reference_video="c1_main.mp4")
    assert filled["c2_norepeat.mp4"] == (
        str(other.start_frame + 1),
        str(other.end_frame + 1),
    )


def test_checkpoint_round_trips_per_camera_ranges(tmp_path: Path) -> None:
    state = syncvid.build_checkpoint_state(
        reference="a.mp4",
        start="1",
        end="9",
        sync_frames={"a.mp4": "1", "b.mp4": ""},
        current_video="a.mp4",
        current_frame=0,
        output_file=None,
        completed=False,
        ranges={"a.mp4": ("", ""), "b.mp4": ("3", "8")},
    )
    assert state["ranges"] == {"b.mp4": ["3", "8"]}
    path = syncvid.write_checkpoint_file(state, tmp_path / "c.json")
    assert syncvid.read_checkpoint_file(path)["ranges"] == {"b.mp4": ["3", "8"]}

    state["ranges"] = {"b.mp4": ["3"]}
    syncvid.write_checkpoint_file(state, path)
    with pytest.raises(syncvid.SyncFileError, match="ranges"):
        syncvid.read_checkpoint_file(path)


def test_pygame_player_g_focuses_go_to_and_enter_seeks(tmp_path: Path) -> None:
    pygame = syncvid.pygame
    player = object.__new__(syncvid.PygameSyncPlayer)
    info = _video_info(tmp_path / "a.mp4", 50)
    player.video_infos = [info]
    player.current_index = 0
    player.current_frame = 4
    player.playing = False
    player.reference_field = syncvid.PygameTextField("reference", "a.mp4")
    player.start_field = syncvid.PygameTextField("reference_start", "1")
    player.end_field = syncvid.PygameTextField("reference_end", "50")
    player.sync_fields = {"a.mp4": syncvid.PygameTextField("sync:a.mp4")}
    player.range_start_fields = {"a.mp4": syncvid.PygameTextField("start:a.mp4")}
    player.range_end_fields = {"a.mp4": syncvid.PygameTextField("end:a.mp4")}
    player.goto_field = syncvid.PygameTextField("goto")
    player.active_field = None
    decoded = []

    def decode(frame: int) -> bool:
        decoded.append(frame)
        player.current_frame = frame
        return True

    player._decode_at = decode

    player._handle_player_key(
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_g, mod=0, unicode="g")
    )
    assert player.active_field is player.goto_field
    assert player.goto_field.text == "5"  # starts at the current frame, selected
    for char in "37":
        key = getattr(pygame, f"K_{char}")
        player.goto_field.handle_key(
            pygame.event.Event(pygame.KEYDOWN, key=key, mod=0, unicode=char)
        )
    player._commit_active_field()
    assert decoded == [36]
    assert player.active_field is None

    player._handle_button("goto")  # Go button re-uses the typed frame
    assert decoded == [36, 36]
    player.goto_field.text = "999"
    player._handle_button("goto")
    assert decoded == [36, 36]


def test_find_repeated_frames_compensates_field_shift_but_not_pans(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    rng = np.random.default_rng(3)
    # Broadcast-like detail: a 1 px shift alone changes the raw score a lot (~30).
    base = rng.integers(0, 255, (90, 160), dtype=np.uint8)
    texture = cv2.cvtColor(
        cv2.resize(base, (640, 360), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR
    )

    def write(path: Path, frames: list) -> None:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 60.0, (320, 180))
        for frame in frames:
            writer.write(frame)
        writer.release()

    # Interlaced-style copies: frames 4, 9 and 14 repeat the previous one shifted 1 px down.
    frames = []
    for index in range(18):
        if index in (4, 9, 14):
            frames.append(np.roll(frames[-1], 1, axis=0))
            continue
        frame = texture[40:220, 40:360].copy()
        cv2.circle(frame, (20 + index * 15, 90), 14, (255, 255, 255), -1)
        frames.append(frame)
    write(tmp_path / "field.avi", frames)
    report = syncvid.find_repeated_frames(tmp_path / "field.avi")
    assert report.repeated_frames == (4, 9, 14)
    assert abs(report.shifts[3][1]) > 0.5  # the measured vertical field shift

    # A slow horizontal pan over a static scene is real motion: nothing is removed.
    pan = [texture[40:220, 40 + index : 360 + index].copy() for index in range(18)]
    write(tmp_path / "pan.avi", pan)
    assert syncvid.find_repeated_frames(tmp_path / "pan.avi").repeated_frames == ()


def test_choose_output_file_uses_portable_dialog_and_releases_sdl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Save/Load must release SDL and use the portable desktop chooser.

    In-process Tk filedialog under pygame left the mouse frozen on GNOME/X11.
    Load/Save now quit the SDL display, open zenity (or isolated Tk), then
    rebuild ``self.screen``.
    """
    import cv2
    import numpy as np

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    for name in ("camera_one.avi", "camera_two.avi"):
        writer = cv2.VideoWriter(
            str(tmp_path / name), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48)
        )
        for _ in range(5):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        writer.release()

    player = syncvid.PygameSyncPlayer(syncvid.probe_video_directory(tmp_path))
    events: list[str] = []
    old_screen = player.screen
    old_size = old_screen.get_size()

    def fake_portable(kind: str, **kwargs: object) -> str:
        events.append(f"portable:{kind}")
        assert not pygame.display.get_init()
        assert kwargs.get("title")
        return str(tmp_path / "chosen.txt")

    monkeypatch.setattr(syncvid, "_portable_path_dialog", fake_portable)

    try:
        result = player._choose_output_file()
        assert result == (tmp_path / "chosen.txt").resolve()
        assert events == ["portable:save"]
        assert pygame.display.get_init()
        assert player.screen is not old_screen
        assert player.screen.get_size() == old_size

        events.clear()

        def fake_cancel(kind: str, **_kwargs: object) -> str:
            events.append(f"portable:{kind}")
            return ""

        monkeypatch.setattr(syncvid, "_portable_path_dialog", fake_cancel)
        assert player._choose_output_file() is None
        assert events == ["portable:save"]
    finally:
        player._release_capture()
        pygame.display.quit()
        pygame.font.quit()


def test_portable_path_dialog_prefers_zenity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """zenity is preferred over the Tk subprocess when available."""
    monkeypatch.setattr(
        syncvid,
        "_zenity_path_dialog",
        lambda kind, **_kwargs: f"/tmp/zenity-{kind}.txt",
    )
    called: list[str] = []
    monkeypatch.setattr(
        syncvid,
        "_isolated_tk_path_dialog",
        lambda kind, **_kwargs: called.append(kind) or "/tmp/tk.txt",
    )
    assert syncvid._portable_path_dialog("open", title="Pick") == "/tmp/zenity-open.txt"
    assert called == []
