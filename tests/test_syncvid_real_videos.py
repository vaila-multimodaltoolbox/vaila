"""Integration tests for vaila/syncvid.py against the real fixture videos.

These tests decode the actual H.264 clips in tests/syncvid/ (not synthetic
VideoInfo fixtures) to exercise probing, frame-accurate seeking, staggered
multi-camera sync-plan math, the --dry-run report, and a headless
PygameSyncPlayer session end to end. tests/test_syncvid.py keeps the fast
synthetic-fixture unit tests; this file covers the real-decode paths the
project's own goal asked for.

Update Date: 03 September 2026
Version: 0.3.120
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path

import cv2
import pygame
import pytest

from vaila import syncvid

REAL_VIDEO_DIR = Path(__file__).resolve().parent / "syncvid"
REAL_VIDEO_NAMES = [
    "palxscri_h264_frame_17608_to_17689.mp4",
    "palxscri_h264_frame_17953_to_18064.mp4",
    "palxscri_h264_frame_18208_to_18347.mp4",
    "palxscri_h264_frame_18515_to_18656.mp4",
    "palxscri_h264_frame_18897_to_19134.mp4",
]
# Ground truth captured directly with cv2.VideoCapture(...).get(...); guards
# against a regression in probe_video()'s own frame_count/fps/size reporting.
EXPECTED_FRAME_COUNTS = {
    "palxscri_h264_frame_17608_to_17689.mp4": 82,
    "palxscri_h264_frame_17953_to_18064.mp4": 112,
    "palxscri_h264_frame_18208_to_18347.mp4": 140,
    "palxscri_h264_frame_18515_to_18656.mp4": 142,
    "palxscri_h264_frame_18897_to_19134.mp4": 238,
}

pytestmark = pytest.mark.skipif(
    not REAL_VIDEO_DIR.is_dir() or len(list(REAL_VIDEO_DIR.glob("*.mp4"))) < 2,
    reason="tests/syncvid/ real video fixtures are not present",
)


@pytest.fixture
def headless_display(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force SDL to use its dummy video/audio driver so pygame needs no display."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")


def test_discover_video_files_real_directory_ignores_toml_and_output_dirs() -> None:
    # tests/syncvid/ also holds *_cuts.toml files and *_vailacut_*/ output
    # directories left behind by live Cut Video sessions; discover_video_files
    # must only surface direct-child videos, never recurse or pick up siblings.
    names = syncvid.get_video_files(REAL_VIDEO_DIR)
    assert names == sorted(REAL_VIDEO_NAMES, key=str.casefold)


def test_probe_video_directory_reports_correct_metadata_for_real_clips() -> None:
    infos = syncvid.probe_video_directory(REAL_VIDEO_DIR)
    assert [info.name for info in infos] == sorted(REAL_VIDEO_NAMES, key=str.casefold)
    for info in infos:
        assert info.frame_count == EXPECTED_FRAME_COUNTS[info.name]
        assert info.width == 1920
        assert info.height == 1080
        assert 59.9 < info.fps < 60.0


def test_build_dry_run_report_real_videos_lists_every_clip() -> None:
    lines = syncvid.build_dry_run_report(REAL_VIDEO_DIR)
    assert lines[0].startswith(">> vaila/syncvid: Dry-run")
    assert f"videos={len(REAL_VIDEO_NAMES)}" in lines
    report = "\n".join(lines)
    for name in REAL_VIDEO_NAMES:
        assert name in report


def test_frame_seek_matches_sequential_decode_for_real_h264_clips() -> None:
    """cap.set(CAP_PROP_POS_FRAMES) must land on the same bytes as sequential read()."""
    path = REAL_VIDEO_DIR / "palxscri_h264_frame_18208_to_18347.mp4"

    sequential_hashes: list[str] = []
    cap = cv2.VideoCapture(str(path))
    try:
        ok = True
        while ok:
            ok, frame = cap.read()
            if ok:
                sequential_hashes.append(hashlib.md5(frame.tobytes()).hexdigest())
    finally:
        cap.release()

    assert len(sequential_hashes) == EXPECTED_FRAME_COUNTS[path.name]

    cap = cv2.VideoCapture(str(path))
    try:
        for index in (0, 1, 10, 50, len(sequential_hashes) - 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            assert ok, f"seek to frame {index} failed to decode"
            seeked_hash = hashlib.md5(frame.tobytes()).hexdigest()
            assert seeked_hash == sequential_hashes[index], (
                f"seeked frame {index} does not match sequentially decoded frame"
            )
    finally:
        cap.release()


def test_build_sync_plan_with_staggered_offsets_across_real_videos() -> None:
    """Five real clips with different lengths and independently chosen sync frames."""
    infos = syncvid.probe_video_directory(REAL_VIDEO_DIR)
    # Each camera's kickoff/whistle frame lands at a different point in its own
    # timeline; pick frames near the middle of each real clip's timeline.
    sync_frames = {
        "palxscri_h264_frame_17608_to_17689.mp4": 40,
        "palxscri_h264_frame_17953_to_18064.mp4": 55,
        "palxscri_h264_frame_18208_to_18347.mp4": 70,
        "palxscri_h264_frame_18515_to_18656.mp4": 71,
        "palxscri_h264_frame_18897_to_19134.mp4": 119,
    }
    reference = "palxscri_h264_frame_17608_to_17689.mp4"
    start, end = syncvid.common_reference_bounds(infos, sync_frames, reference)
    assert start <= sync_frames[reference] <= end

    plan = syncvid.build_sync_plan(infos, sync_frames, reference, start, end)
    assert len(plan) == len(infos)
    durations = {entry.end_frame - entry.start_frame for entry in plan}
    assert len(durations) == 1, "every synchronized clip must share the same duration"

    by_name = {entry.video_file: entry for entry in plan}
    for name, info in {info.name: info for info in infos}.items():
        entry = by_name[name]
        assert 0 <= entry.start_frame <= entry.end_frame < info.frame_count
        # The offset from each camera's own sync frame to its start/end must
        # equal the reference camera's offset (the whole point of the sync).
        assert entry.start_frame - sync_frames[name] == start - sync_frames[reference]
        assert entry.end_frame - sync_frames[name] == end - sync_frames[reference]


def test_pygame_sync_player_opens_and_navigates_real_videos(headless_display: None) -> None:
    pygame = pytest.importorskip("pygame")
    infos = syncvid.probe_video_directory(REAL_VIDEO_DIR)
    player = syncvid.PygameSyncPlayer(infos)
    try:
        assert player.current_index == 0
        assert player.current_frame == 0

        # Seek forward within the first (82-frame) clip.
        player._seek_relative(10)
        assert player.current_frame == 10

        # Seeking past the end must clamp, not raise or wrap silently past bounds.
        player._decode_at(10_000)
        assert player.current_frame == infos[0].frame_count - 1

        # Switch to the last camera (238 frames) and confirm independent position.
        player._open_video(len(infos) - 1)
        assert player.current_index == len(infos) - 1
        assert player.current_frame == 0
        player._seek_relative(50)
        assert player.current_frame == 50

        # Switching back to the first camera preserves its remembered position.
        player._open_video(0)
        assert player.current_frame == infos[0].frame_count - 1

        surface = player._get_video_surface(player._video_viewport())
        assert surface is not None
        assert isinstance(surface, pygame.Surface)

        # "vailá" branding: the side-panel title must render "vailá" in
        # italic per the project's writing convention.
        assert player.title_font_italic.get_italic() is True
        player._draw_side_panel()  # must not raise with the real window/panel geometry
    finally:
        player._release_capture()
        pygame.display.quit()
        pygame.font.quit()


def test_pygame_sync_player_sets_window_icon_from_packaged_images(
    headless_display: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The window icon must come from vaila/images/ (bundled with the package),
    not docs/images/ (docs-only, not guaranteed to ship in an install)."""
    import pygame

    icon_path = Path(syncvid.__file__).resolve().parent / "images" / "vaila_ico_mac.png"
    assert icon_path.is_file()

    calls: list[pygame.Surface] = []
    real_set_icon = pygame.display.set_icon
    monkeypatch.setattr(
        pygame.display, "set_icon", lambda surface: (calls.append(surface), real_set_icon(surface))
    )

    infos = syncvid.probe_video_directory(REAL_VIDEO_DIR)
    player = syncvid.PygameSyncPlayer(infos)
    try:
        assert len(calls) == 1
        assert calls[0].get_size() == (256, 256)
    finally:
        player._release_capture()
        pygame.display.quit()
        pygame.font.quit()


def test_choose_output_file_iconifies_before_and_restores_after_tk_dialog(
    headless_display: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A native Tk save dialog blocks this process from pumping the SDL
    window's own event queue, which is a known trigger for desktop-wide
    freezes on some X11 window managers (especially multi-monitor) when a
    second toolkit's top-level window competes for the WM's attention at the
    same time. _choose_output_file() must iconify the SDL window before the
    Tk dialog opens and restore it (via set_mode(), mirroring the resize
    handler) once the dialog returns -- regardless of Save or Cancel."""
    import pygame

    from vaila.syncvid import filedialog

    infos = syncvid.probe_video_directory(REAL_VIDEO_DIR)
    player = syncvid.PygameSyncPlayer(infos)
    try:
        calls: list[str] = []
        real_iconify = pygame.display.iconify
        real_set_mode = pygame.display.set_mode

        def spy_iconify() -> object:
            calls.append("iconify")
            return real_iconify()

        def spy_set_mode(size: tuple[int, int], flags: int = 0) -> pygame.Surface:
            calls.append("set_mode")
            return real_set_mode(size, flags)

        def fake_ask(**_kwargs: object) -> str:
            # The dialog itself must run strictly between iconify and restore.
            assert calls == ["iconify"]
            return str(tmp_path / "chosen.txt")

        monkeypatch.setattr(pygame.display, "iconify", spy_iconify)
        monkeypatch.setattr(pygame.display, "set_mode", spy_set_mode)
        monkeypatch.setattr(filedialog, "asksaveasfilename", fake_ask)

        result = player._choose_output_file()
        assert result == (tmp_path / "chosen.txt").resolve()
        assert calls == ["iconify", "set_mode"]

        # Cancelling (asksaveasfilename returns "") must still restore the window.
        calls.clear()
        monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **_kwargs: "")
        assert player._choose_output_file() is None
        assert calls == ["iconify", "set_mode"]
    finally:
        player._release_capture()
        pygame.display.quit()
        pygame.font.quit()


def test_save_workflow_writes_checkpoint_sidecar_and_load_checkpoint_restores_state(
    headless_display: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Save writes a .json checkpoint next to the sync file; a fresh player
    can then Load/Resume it and end up with the same field/position state."""
    infos = syncvid.probe_video_directory(REAL_VIDEO_DIR)
    player = syncvid.PygameSyncPlayer(infos)
    try:
        for field in player.sync_fields.values():
            field.text = "5"
        player.reference_field.text = infos[0].name
        player.start_field.text = "1"
        player.end_field.text = str(infos[0].frame_count)
        player._open_video(len(infos) - 1, target_frame=7)

        output_path = tmp_path / "vaila_sync_checkpoint.txt"
        monkeypatch.setattr(player, "_choose_output_file", lambda: output_path)
        player._save(open_cutvideo=False)

        checkpoint_path = output_path.with_suffix(".json")
        assert checkpoint_path.is_file()
        data = syncvid.read_checkpoint_file(checkpoint_path)
        assert data["completed"] is False  # "Save sync" (not Save + Cut Video)
        assert data["current_video"] == infos[-1].name
        assert data["current_frame"] == 7
        assert data["sync_frames"] == {info.name: "5" for info in infos}
    finally:
        player._release_capture()
        pygame.display.quit()
        pygame.font.quit()

    # A brand new player (simulating relaunching syncvid) loads that checkpoint.
    resumed = syncvid.PygameSyncPlayer(infos)
    try:
        monkeypatch.setattr(
            resumed,
            "_sdl_window_minimized_for_dialog",
            lambda: contextlib.nullcontext(),
        )
        monkeypatch.setattr(
            syncvid.filedialog, "askopenfilename", lambda **_kwargs: str(checkpoint_path)
        )
        resumed._load_checkpoint()

        assert resumed.reference_field.text == infos[0].name
        assert resumed.start_field.text == "1"
        assert resumed.end_field.text == str(infos[0].frame_count)
        assert all(field.text == "5" for field in resumed.sync_fields.values())
        assert resumed.current_info.name == infos[-1].name
        assert resumed.current_frame == 7
        assert resumed.output_file == output_path
        assert "Resumed" in resumed.status_text
        assert "in-progress" in resumed.status_text
    finally:
        resumed._release_capture()
        pygame.display.quit()
        pygame.font.quit()


def test_load_checkpoint_cancel_leaves_state_untouched(
    headless_display: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    infos = syncvid.probe_video_directory(REAL_VIDEO_DIR)
    player = syncvid.PygameSyncPlayer(infos)
    try:
        monkeypatch.setattr(
            player, "_sdl_window_minimized_for_dialog", lambda: contextlib.nullcontext()
        )
        monkeypatch.setattr(syncvid.filedialog, "askopenfilename", lambda **_kwargs: "")
        before_frame, before_index = player.current_frame, player.current_index
        player._load_checkpoint()
        assert player.current_frame == before_frame
        assert player.current_index == before_index
        assert player.status_text == "Load cancelled"
    finally:
        player._release_capture()
        pygame.display.quit()
        pygame.font.quit()


def test_load_checkpoint_reports_clean_error_for_corrupt_file(
    headless_display: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt/missing-key checkpoint must produce a status-bar error, not
    crash the player."""
    infos = syncvid.probe_video_directory(REAL_VIDEO_DIR)
    player = syncvid.PygameSyncPlayer(infos)
    try:
        corrupt_path = tmp_path / "corrupt.json"
        corrupt_path.write_text("{not valid json", encoding="utf-8")
        monkeypatch.setattr(
            player, "_sdl_window_minimized_for_dialog", lambda: contextlib.nullcontext()
        )
        monkeypatch.setattr(
            syncvid.filedialog, "askopenfilename", lambda **_kwargs: str(corrupt_path)
        )
        before_frame = player.current_frame
        player._load_checkpoint()  # must not raise
        assert player.current_frame == before_frame
        assert "Could not load checkpoint" in player.status_text
        assert player.status_color == (255, 130, 120)  # _set_status(..., error=True)
    finally:
        player._release_capture()
        pygame.display.quit()
        pygame.font.quit()


def test_load_checkpoint_skips_unknown_cameras_gracefully(
    headless_display: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint referencing a camera not present in this directory must
    not crash -- its sync-frame entry is skipped and the status bar names it."""
    infos = syncvid.probe_video_directory(REAL_VIDEO_DIR)
    state = syncvid.build_checkpoint_state(
        reference=infos[0].name,
        start="1",
        end=str(infos[0].frame_count),
        sync_frames={infos[0].name: "5", "unknown_camera.mp4": "9"},
        current_video=infos[0].name,
        current_frame=1,
        output_file=None,
        completed=False,
    )
    checkpoint_path = syncvid.write_checkpoint_file(state, tmp_path / "partial.json")

    player = syncvid.PygameSyncPlayer(infos)
    try:
        monkeypatch.setattr(
            player, "_sdl_window_minimized_for_dialog", lambda: contextlib.nullcontext()
        )
        monkeypatch.setattr(
            syncvid.filedialog, "askopenfilename", lambda **_kwargs: str(checkpoint_path)
        )
        player._load_checkpoint()
        assert player.sync_fields[infos[0].name].text == "5"
        assert "unknown_camera.mp4" in player.status_text
    finally:
        player._release_capture()
        pygame.display.quit()
        pygame.font.quit()


def test_pygame_sync_player_save_workflow_produces_valid_sync_file(
    headless_display: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full field-entry -> save flow against real videos, writing only to tmp_path."""
    infos = syncvid.probe_video_directory(REAL_VIDEO_DIR)
    player = syncvid.PygameSyncPlayer(infos)
    try:
        for field in player.sync_fields.values():
            field.text = "5"
        player.reference_field.text = infos[0].name
        player.start_field.text = "1"
        player.end_field.text = str(infos[0].frame_count)

        output_path = tmp_path / "vaila_sync_real.txt"
        monkeypatch.setattr(player, "_choose_output_file", lambda: output_path)
        player._save(open_cutvideo=False)

        assert player.result is not None
        assert player.result.sync_file == output_path
        assert output_path.is_file()

        entries = syncvid.read_sync_file(output_path)
        assert len(entries) == len(infos)
        durations = {entry.end_frame - entry.start_frame for entry in entries}
        assert len(durations) == 1

        # Round-trip: cutvideo's own dict-shape helper must accept these entries.
        cutvideo_data = syncvid.sync_entries_to_cutvideo_data(entries)
        assert set(cutvideo_data) == {info.name for info in infos}
        for data in cutvideo_data.values():
            assert isinstance(data["initial_frame"], int)
            assert isinstance(data["final_frame"], int)
            assert data["initial_frame"] <= data["final_frame"]
    finally:
        player._release_capture()
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame

        pygame.display.quit()
        pygame.font.quit()
