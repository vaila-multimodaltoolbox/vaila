"""Event timing, GUI persistence and lossless end-to-end export regression proof.

Update Date: 27 October 2026
Version: 0.4.12
"""

import csv
import json
import shutil
import subprocess
from copy import deepcopy
from fractions import Fraction

import cv2
import numpy as np
import pygame
import pytest

from vaila import cutvideo, sync_events, syncvid

REMOVED = [
    2,
    5,
    8,
    11,
    15,
    18,
    22,
    26,
    30,
    34,
    38,
    42,
    46,
    50,
    54,
    58,
    62,
    66,
    70,
    73,
    77,
    81,
    85,
    88,
    92,
    95,
    98,
    101,
    103,
    106,
    108,
    111,
]


def checkpoint():
    return syncvid.build_checkpoint_state(
        reference="a.avi",
        start="1",
        end="79",
        sync_frames={"a.avi": "15", "b.avi": "20"},
        ranges={"b.avi": ("1", "109")},
        current_video="b.avi",
        current_frame=108,
        output_file=None,
        completed=False,
        removed_frames={"b.avi": {f - 1 for f in REMOVED}},
    )


def infos(tmp_path):
    return [
        syncvid.VideoInfo(tmp_path / name, frame_count=112, fps=59.94005994, width=96, height=64)
        for name in ("a.avi", "b.avi")
    ]


def model(tmp_path):
    return sync_events.manifest_from_checkpoint(checkpoint(), infos(tmp_path))


def test_three_confirmed_events_and_physical_durations(tmp_path):
    manifest = model(tmp_path)
    assert manifest["output_fps"] == "30000/1001"
    assert manifest["output_frame_count"] == 40
    assert manifest["sample_span_seconds"] == pytest.approx(1.3013)
    assert manifest["playback_duration_seconds"] == pytest.approx(1.3346666666666667)
    assert manifest["cameras"]["a.avi"]["retained_count"] == 79
    assert manifest["cameras"]["b.avi"]["retained_count"] == 78
    assert manifest["cameras"]["b.avi"]["removed_frames"] == REMOVED
    for name, expected in (("a.avi", [1, 15, 79]), ("b.avi", [1, 20, 109])):
        timeline, rows = sync_events.temporal_maps(manifest, name)
        assert [rows[i - 1]["original_frame"] for i in (1, 8, 40)] == expected
        assert [rows[i - 1]["temporal_error_s"] for i in (1, 8, 40)] == [0, 0, 0]
        for row in rows:
            time = Fraction(row["requested_time_rational"])
            index = min(
                range(len(timeline.frames)), key=lambda i: (abs(timeline.times[i] - time), i)
            )
            assert row["original_frame"] == timeline.frames[index]
            assert Fraction(row["temporal_error_rational"]) == timeline.times[index] - time
            assert 0 <= time <= Fraction("13013/10000")


@pytest.mark.parametrize(
    "events,removed,message",
    [
        ([1, 21, 20], [], "out of order"),
        ([1, 20, 109], [20], "event 2, frame 20, is marked"),
        ([1, 20, 113], [], "exceeds"),
        ([1, 109], [], "same ordered events"),
        ([0, 20, 109], [], "positive 1-based"),
        ([1, True, 109], [], "positive 1-based"),
    ],
)
def test_invalid_events_identify_camera_and_correction(tmp_path, events, removed, message):
    manifest = model(tmp_path)
    manifest["cameras"]["b.avi"].update(events=events, removed_frames=removed)
    with pytest.raises(sync_events.EventSyncError, match=f"b.avi.*{message}"):
        sync_events.validate_manifest(manifest)


def test_three_cameras_different_intervals_more_events_and_earlier_ties(tmp_path):
    manifest = model(tmp_path)
    manifest["reference_fps"], manifest["output_fps"] = "4", "8"
    manifest["cameras"] = {
        "a.avi": {"events": [4, 8, 12, 16], "removed_frames": [], "frame_count": 16},
        "b.avi": {"events": [2, 5, 8, 10], "removed_frames": [3, 6], "frame_count": 16},
        "c.avi": {"events": [1, 3, 5, 7], "removed_frames": [], "frame_count": 16},
    }
    manifest = sync_events.validate_manifest(manifest)
    for name in manifest["cameras"]:
        timeline, rows = sync_events.temporal_maps(manifest, name)
        assert rows[0]["original_frame"] == timeline.frames[0]
        assert rows[-1]["original_frame"] == timeline.frames[-1]
        assert [rows[i]["original_frame"] for i in (0, 8, 16, 24)] == manifest["cameras"][name][
            "events"
        ]
    _, rows = sync_events.temporal_maps(manifest, "c.avi")
    assert rows[2]["original_frame"] == 1  # exact midpoint: earlier sample
    assert rows[2]["reused"] == 1
    assert rows[2]["temporal_error_s"] == -0.25


@pytest.mark.parametrize("clock", ["0", "-3", "1/0", "nan", "garbage"])
def test_reject_invalid_clocks(tmp_path, clock):
    manifest = model(tmp_path)
    manifest["reference_fps"] = clock
    with pytest.raises(sync_events.EventSyncError, match="Reference clock"):
        sync_events.validate_manifest(manifest)


def write_source(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 60000 / 1001, (96, 64))
    assert writer.isOpened()
    for i in range(112):
        frame = np.full((64, 96, 3), (i * 2, i, 255 - i), np.uint8)
        cv2.putText(frame, str(i + 1), (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
        writer.write(frame)
    writer.release()


def decode(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required"
)
def test_cutvideo_cli_exports_exact_grid_maps_and_retained_images(tmp_path, monkeypatch):
    for info in infos(tmp_path):
        write_source(info.path)
    state_path = syncvid.write_checkpoint_file(checkpoint(), tmp_path / "old_checkpoint.json")
    manifest_path = tmp_path / "test.events.json"
    assert (
        syncvid.main(
            [
                "--events",
                "--checkpoint",
                str(state_path),
                "-i",
                str(tmp_path),
                "-o",
                str(manifest_path),
            ]
        )
        == 0
    )
    cuts, is_sync, data = cutvideo.parse_sync_file_content(
        manifest_path, tmp_path / "a.avi", strict=True
    )
    assert cuts == [(0, 78)] and is_sync
    assert data["b.avi"]["final_frame"] == 108
    monkeypatch.setattr(
        cutvideo, "play_video_with_cuts", lambda *_a, **_k: pytest.fail("event CLI opened GUI")
    )
    out = tmp_path / "out"
    assert (
        cutvideo.main(
            [
                "--video",
                str(tmp_path / "a.avi"),
                "--sync-file",
                str(manifest_path),
                "--output-dir",
                str(out),
            ]
        )
        == 0
    )
    for name in ("a.avi", "b.avi"):
        source = decode(tmp_path / name)
        output = decode(out / f"{name}_events.mp4")
        assert len(output) == 40
        metadata = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_streams",
                    "-of",
                    "json",
                    str(out / f"{name}_events.mp4"),
                ]
            )
        )
        assert metadata["streams"][0]["r_frame_rate"] == "30000/1001"
        assert metadata["streams"][0]["nb_frames"] == "40"
        with (out / f"{name}_time_map.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        for frame, row in zip(output, rows, strict=True):
            np.testing.assert_array_equal(frame, source[int(row["original_frame"]) - 1])
        with (out / f"{name}_retained_times.csv").open() as handle:
            retained = list(csv.DictReader(handle))
        assert len(retained) == (79 if name == "a.avi" else 78)
        for row in retained:
            frame = cv2.imread(str(out / f"{name}_retained" / row["image"]))
            np.testing.assert_array_equal(frame, source[int(row["original_frame"]) - 1])
    assert json.loads(state_path.read_text()) == checkpoint()  # source checkpoint remains intact


def test_invalid_later_camera_creates_no_output(tmp_path):
    manifest = model(tmp_path)
    manifest["cameras"]["b.avi"]["removed_frames"].append(20)
    with pytest.raises(sync_events.EventSyncError, match="marked for removal"):
        sync_events.export_event_videos(manifest, tmp_path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_gui_load_old_checkpoint_event_edit_save_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    for info in infos(tmp_path):
        write_source(info.path)
    player = syncvid.PygameSyncPlayer(syncvid.probe_video_directory(tmp_path))
    try:
        player._apply_checkpoint(checkpoint())
        player._handle_button("side:event_mode")
        assert player.event_fields["a.avi"].text == "1, 15, 79"
        assert player.event_fields["b.avi"].text == "1, 20, 109"
        player._draw()
        assert player.event_fields["b.avi"] in player.visible_fields
        assert player.reference_clock_field in player.visible_fields
        player._set_active_field(player.event_fields["b.avi"])
        player._commit_active_field()
        assert "40 output frames" in player.status_text
        saved = tmp_path / "gui.txt"
        monkeypatch.setattr(player, "_choose_output_file", lambda: saved)
        player._save(False)
        assert player.running is True
        assert player.result.sync_file == tmp_path / "gui.events.json"
        assert player.result.open_cutvideo is False
        assert "Saved events" in player.status_text
        assert (
            sync_events.read_event_manifest(player.result.sync_file)["cameras"]["b.avi"][
                "retained_count"
            ]
            == 78
        )
        restored = syncvid.read_checkpoint_file(tmp_path / "gui.checkpoint.json")
        player.event_fields["b.avi"].text = "bad"
        player._apply_checkpoint(restored)
        assert player.event_mode
        assert player.event_fields["b.avi"].text == "1, 20, 109"
        # Invalid event removal is displayed and prevents saving over an existing valid manifest.
        player.removed_frames["b.avi"].add(19)
        before = player.result.sync_file.read_bytes()
        player._save(False)
        assert player.running is True
        assert "event 2, frame 20, is marked" in player.status_text
        assert player.result.sync_file.read_bytes() == before
        player.removed_frames["b.avi"].discard(19)
        player._save(True)
        assert player.running is False
        assert player.result.open_cutvideo is True
    finally:
        player._release_capture()
        pygame.display.quit()
        pygame.font.quit()


def test_legacy_equal_fps_behavior_preserved(tmp_path):
    plan = syncvid.build_sync_plan(
        infos(tmp_path), {"a.avi": 14, "b.avi": 19}, "a.avi", 0, 78, ranges={"b.avi": (0, 108)}
    )
    assert plan[1].end_frame == 78
    path = syncvid.write_sync_file(plan, tmp_path / "old.txt")
    assert syncvid.read_sync_file(path) == plan


def test_untrusted_manifest_derivations_are_recomputed(tmp_path):
    manifest = deepcopy(model(tmp_path))
    manifest["output_frame_count"] = 123
    manifest["event_times"] = ["0", "10", "20"]
    assert sync_events.validate_manifest(manifest)["output_frame_count"] == 40
