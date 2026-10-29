"""Independent side-by-side editing and exact exported frame identities.

Update Date: 28 October 2026
Version: 0.4.12
"""

import csv
import json
import shutil
import subprocess
from contextlib import nullcontext
from copy import deepcopy

import cv2
import numpy as np
import pygame
import pytest

from vaila import manual_video_cuts as manual
from vaila import sync_events, syncvid


def write_video(path, count=12, fps=30000 / 1001):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (96, 64))
    assert writer.isOpened()
    for index in range(count):
        image = np.full((64, 96, 3), (index * 12, 255 - index * 10, index * 5), np.uint8)
        cv2.putText(image, str(index + 1), (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1)
        writer.write(image)
    writer.release()


@pytest.fixture
def videos(tmp_path):
    write_video(tmp_path / "a.avi")
    write_video(tmp_path / "b.avi", 10, 60000 / 1001)
    return syncvid.probe_video_directory(tmp_path)


def cuts(videos):
    return manual.build_manual_cuts(
        videos, {"a.avi": ("3", "10"), "b.avi": ("2", "8")}, {"a.avi": [4, 8, 12], "b.avi": [5]}
    )


def decode(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def test_independent_ranges_and_marks_without_equal_fps_snapping(videos):
    data = cuts(videos)
    assert manual.selected_frames(data["cameras"]["a.avi"]) == [3, 5, 6, 7, 9, 10]
    assert manual.selected_frames(data["cameras"]["b.avi"]) == [2, 3, 4, 6, 7, 8]
    assert data["cameras"]["a.avi"]["output_frame_count"] == 6
    assert data["cameras"]["b.avi"]["output_frame_count"] == 6
    # OpenCV's AVI writer declares decimal 29.97/59.94; preserve those exact rates.
    assert data["cameras"]["a.avi"]["source_fps"] == "2997/100"
    assert data["cameras"]["b.avi"]["source_fps"] == "2997/50"
    # Matching container FPS must not adjust independently chosen intervals/counts.
    same_fps = [info.with_fps(30) for info in videos]
    unequal = manual.build_manual_cuts(same_fps, {"a.avi": ("1", "9"), "b.avi": ("2", "8")}, {})
    assert [c["output_frame_count"] for c in unequal["cameras"].values()] == [9, 7]


@pytest.mark.parametrize(
    "change,message",
    [
        ({"start": 0}, "Start/End"),
        ({"end": 13}, "Start/End"),
        ({"start": 11, "end": 3}, "Start/End"),
        ({"start": True}, "Start/End"),
        ({"removed_frames": [4, 4]}, "unique"),
        ({"removed_frames": [0]}, "1-based"),
        ({"removed_frames": list(range(3, 11))}, "empty"),
        ({"frame_count": 13}, "frame count differs"),
        ({"source_fps": "25"}, "FPS differs"),
    ],
)
def test_invalid_requests_create_no_directory(videos, tmp_path, change, message):
    data = cuts(videos)
    data["cameras"]["a.avi"].update(change)
    with pytest.raises(sync_events.EventSyncError, match=message):
        manual.export_manual_cuts(data, videos, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required"
)
def test_headless_export_keeps_pixels_frames_source_fps_and_existing_files(
    videos, tmp_path, monkeypatch, capsys
):
    data = cuts(videos)
    path = tmp_path / "edits.json"
    path.write_text(json.dumps(data))
    original = path.read_bytes()
    monkeypatch.setattr(
        syncvid, "PygameSyncPlayer", lambda *_a, **_k: pytest.fail("CLI opened GUI")
    )
    monkeypatch.setattr(
        syncvid, "launch_cutvideo", lambda *_a, **_k: pytest.fail("manual export synchronized")
    )
    out = tmp_path / "exports"
    assert (
        syncvid.main(
            [
                "--manual-cuts",
                "-i",
                str(tmp_path),
                "--cuts-file",
                str(path),
                "--output-dir",
                str(out),
            ]
        )
        == 0
    )
    for info in videos:
        camera = data["cameras"][info.name]
        output_path = out / camera["output_file"]
        output, source = decode(output_path), decode(info.path)
        with (out / f"{info.name}_manual_frames.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert [int(r["original_frame"]) for r in rows] == manual.selected_frames(camera)
        assert len(output) == 6
        for frame, row in zip(output, rows, strict=True):
            np.testing.assert_array_equal(frame, source[int(row["original_frame"]) - 1])
        metadata = json.loads(
            subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output_path)]
            )
        )["streams"]
        assert len(metadata) == 1
        assert metadata[0]["r_frame_rate"] == camera["source_fps"]
        assert metadata[0]["nb_frames"] == "6"
    assert path.read_bytes() == original
    assert not (out / "INCOMPLETE.txt").exists()
    assert not list(out.glob("*.events.json"))
    assert not list(out.glob("*.checkpoint.json"))
    assert manual.read_manual_cuts(out / "manual_cuts.json", videos) == data
    assert ">> Equivalent CLI" in capsys.readouterr().out
    before = (out / "a.avi_manual.mp4").read_bytes()
    with pytest.raises(FileExistsError):
        manual.export_manual_cuts(data, videos, out)
    assert (out / "a.avi_manual.mp4").read_bytes() == before


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_cancelled_export_removes_active_partial_video(videos, tmp_path):
    out = tmp_path / "cancelled"
    with pytest.raises(sync_events.EventExportCancelledError):
        manual.export_manual_cuts(
            cuts(videos), videos, out, progress_callback=lambda _name, done, _total: done < 2
        )
    assert (out / "INCOMPLETE.txt").exists()
    assert not list(out.glob("*.mp4"))
    assert manual.read_manual_cuts(out / "manual_cuts.json", videos) == cuts(videos)
    untouched = tmp_path / "early_cancel"
    with pytest.raises(sync_events.EventExportCancelledError):
        manual.export_manual_cuts(
            cuts(videos), videos, untouched, progress_callback=lambda *_a: False
        )
    assert not untouched.exists()


def test_source_changed_since_open_is_rejected_before_export(videos, tmp_path):
    data = cuts(videos)
    write_video(videos[1].path, 9, 60000 / 1001)
    with pytest.raises(sync_events.EventSyncError, match="source changed"):
        manual.export_manual_cuts(data, videos, tmp_path / "changed")
    assert not (tmp_path / "changed").exists()


@pytest.fixture
def player(videos, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    editor = syncvid.PygameSyncPlayer(videos)
    editor._handle_button("side:manual_mode")
    yield editor
    editor._release_capture()
    pygame.display.quit()
    pygame.font.quit()


def key(player, code, mod=0):
    player._handle_key(pygame.event.Event(pygame.KEYDOWN, key=code, mod=mod, unicode=""))


def test_side_by_side_positions_i_o_and_d_are_independent(player):
    player._draw()
    assert list(player.manual_view_rects) == ["a.avi", "b.avi"]
    player._decode_at(2)
    key(player, pygame.K_i)
    player._decode_at(9)
    key(player, pygame.K_o)
    player._decode_at(3)
    key(player, pygame.K_d)
    key(player, pygame.K_d)
    assert player.manual_removed_frames["a.avi"] == set()
    key(player, pygame.K_d)
    position_a = player.current_frame
    rect = player.manual_view_rects["b.avi"]
    player._handle_mouse_down(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=rect.center, button=1))
    assert player.current_info.name == "b.avi"
    assert player.current_frame == 0
    assert player.positions["a.avi"] == position_a
    player._decode_at(1)
    key(player, pygame.K_i)
    player._decode_at(7)
    key(player, pygame.K_o)
    key(player, pygame.K_d)
    assert player.manual_removed_frames == {"a.avi": {3}, "b.avi": {7}}
    assert player.removed_frames == {"a.avi": set(), "b.avi": set()}
    cameras = player._manual_manifest()["cameras"]
    assert (
        cameras["a.avi"]["start"],
        cameras["a.avi"]["end"],
        cameras["a.avi"]["output_frame_count"],
    ) == (3, 10, 7)
    assert (
        cameras["b.avi"]["start"],
        cameras["b.avi"]["end"],
        cameras["b.avi"]["output_frame_count"],
    ) == (2, 8, 6)
    player._draw()
    assert player.manual_start_fields["b.avi"] in player.visible_fields
    assert player.reference_field not in player.visible_fields
    assert "side:save_cutvideo" not in player.button_rects
    pane = player.manual_view_rects["b.avi"]
    player._zoom_at(pane.center, 2.0)
    player._draw()
    before_pan = player.offset_x
    player._handle_mouse_down(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pane.center, button=1))
    assert player.panning
    player._handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=pane.center, rel=(-20, 0)))
    assert player.offset_x > before_pan
    assert player.positions["a.avi"] == position_a
    player._handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1))
    saved = deepcopy(player.manual_removed_frames)
    key(player, pygame.K_m)
    player._draw()
    assert not player.manual_mode
    assert player.manual_removed_frames == saved
    assert "side:manual_save" not in player.button_rects


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_gui_save_all_current_load_and_cancel_without_sync(player, tmp_path, monkeypatch):
    # Existing automatic/sync marks never become manual deletions.
    player.removed_frames["a.avi"] = {2, 5}
    player.manual_start_fields["a.avi"].text = "3"
    player.manual_end_fields["a.avi"].text = "5"
    player.manual_removed_frames["a.avi"].add(3)
    monkeypatch.setattr(syncvid, "_portable_path_dialog", lambda *_a, **_k: str(tmp_path))
    monkeypatch.setattr(player, "_sdl_released_for_dialog", nullcontext)
    monkeypatch.setattr(
        syncvid, "write_sync_file", lambda *_a: pytest.fail("manual Save wrote sync")
    )
    monkeypatch.setattr(
        syncvid, "launch_cutvideo", lambda *_a: pytest.fail("manual Save launched sync")
    )
    key(player, pygame.K_s, pygame.KMOD_CTRL)
    exported = list(tmp_path.glob("manual_cuts_*/manual_cuts.json"))
    assert len(exported) == 1
    saved = manual.read_manual_cuts(exported[0], player.video_infos)
    assert len(saved["cameras"]) == 2
    assert manual.selected_frames(saved["cameras"]["a.avi"]) == [3, 5]
    assert player.result is None and player.running
    assert "Saved 2 video(s)" in player.status_text
    player.manual_start_fields["a.avi"].text = "1"
    player.manual_removed_frames["a.avi"].clear()
    player._load_manual_cuts(exported[0])
    assert player.manual_start_fields["a.avi"].text == "3"
    assert player.manual_removed_frames["a.avi"] == {3}
    player.manual_end_fields["b.avi"].text = "invalid"
    player._handle_button("side:manual_save_current")
    exported = list(tmp_path.glob("manual_cuts_*/manual_cuts.json"))
    assert sorted(len(json.loads(path.read_text())["cameras"]) for path in exported) == [1, 2]
    player.manual_end_fields["b.avi"].text = "10"
    player.manual_exporting = True
    player._draw()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    assert player._manual_progress("a.avi", 1, 10) is False
    assert player.manual_cancelled and player.running


@pytest.mark.parametrize("extra", [[], ["--events"], ["--output-dir", "unused"]])
def test_manual_cli_rejects_conflicting_or_incomplete_requests(tmp_path, extra):
    args = ["--cuts-file", str(tmp_path / "cuts.json")] if not extra else ["--manual-cuts", *extra]
    with pytest.raises(SystemExit) as error:
        syncvid.main(args)
    assert error.value.code == 2


@pytest.mark.parametrize("count", [1, 3])
def test_single_camera_and_pagination_in_pairs(videos, tmp_path, monkeypatch, count):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    if count == 3:
        write_video(tmp_path / "c.avi", 7)
        videos = syncvid.probe_video_directory(tmp_path)
    editor = syncvid.PygameSyncPlayer(videos[:count])
    try:
        editor._handle_button("side:manual_mode")
        editor._draw()
        assert len(editor.manual_view_rects) == min(2, count)
        if count == 3:
            key(editor, pygame.K_PAGEDOWN)
            key(editor, pygame.K_PAGEDOWN)
            editor._draw()
            assert list(editor.manual_view_rects) == ["c.avi"]
            key(editor, pygame.K_d)
            assert editor.manual_removed_frames == {"a.avi": set(), "b.avi": set(), "c.avi": {0}}
        editor._handle_event(pygame.event.Event(pygame.VIDEORESIZE, w=900, h=650))
        editor._draw()
        assert (
            editor.manual_view_rects[editor.current_info.name].width
            == editor._video_viewport().width
        )
    finally:
        editor._release_capture()
        pygame.display.quit()
        pygame.font.quit()
