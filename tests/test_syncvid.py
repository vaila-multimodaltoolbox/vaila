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
