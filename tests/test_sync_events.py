"""Event timing, GUI persistence and lossless end-to-end export regression proof.

Update Date: 28 October 2026
Version: 0.4.12
"""

import csv
import json
import shutil
import subprocess
from contextlib import nullcontext
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
        player._handle_button("side:mode_nearest")
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


def interpolation_checkpoint():
    state = checkpoint()
    state.update(
        start="63",
        end="81",
        sync_frames={"a.avi": "71", "b.avi": "58"},
        ranges={"b.avi": ["54", "62"]},
        removed_frames={},
    )
    return state


def interpolated_model(tmp_path):
    return sync_events.manifest_from_checkpoint(
        interpolation_checkpoint(),
        infos(tmp_path),
        reference_fps="30000/1001",
        interpolation_method=sync_events.INTERPOLATION,
    )


def test_interpolation_preserves_every_original_and_all_three_events(tmp_path):
    manifest = interpolated_model(tmp_path)
    assert manifest["output_fps"] == "60000/1001"
    assert manifest["output_frame_count"] == 37
    assert manifest["event_output_frames"] == [1, 17, 37]
    for name, events in (("a.avi", [63, 71, 81]), ("b.avi", [54, 58, 62])):
        timeline, rows = sync_events.temporal_maps(manifest, name)
        originals = [r for r in rows if r["type"] == "original"]
        assert [r["original_frame"] for r in originals] == list(timeline.frames)
        assert [rows[i - 1]["original_frame"] for i in (1, 17, 37)] == events
        for row in rows:
            time = Fraction(row["requested_time_rational"])
            lo, hi = row["source_frame_left"], row["source_frame_right"]
            alpha = Fraction(row["interpolation_fraction"])
            assert 0 <= alpha < 1
            left_time = timeline.times[timeline.frames.index(lo)]
            right_time = timeline.times[timeline.frames.index(hi)]
            assert time == left_time + alpha * (right_time - left_time)
    # Untrusted derived fields cannot force frame loss or a rounded grid.
    manifest.update(output_fps="30", output_frame_count=19)
    assert sync_events.validate_manifest(manifest)["output_fps"] == "60000/1001"


def test_interpolation_additional_events_and_minimum_rational_rate(tmp_path):
    manifest = interpolated_model(tmp_path)
    manifest["reference_fps"] = "10"
    manifest["cameras"]["a.avi"]["events"] = [1, 5, 9, 13]
    manifest["cameras"]["b.avi"]["events"] = [2, 4, 7, 9]
    manifest = sync_events.validate_manifest(manifest)
    assert manifest["output_fps"] == "30"
    assert manifest["event_output_frames"] == [1, 13, 25, 37]
    for name, camera in manifest["cameras"].items():
        _, rows = sync_events.temporal_maps(manifest, name)
        assert [rows[i - 1]["original_frame"] for i in manifest["event_output_frames"]] == camera[
            "events"
        ]


@pytest.mark.parametrize(
    "reference,rate,count,anchors",
    [
        ("a.avi", "12727013/32666", 235, [1, 105, 235]),
        ("b.avi", "4895005/16333", 211, [1, 81, 211]),
    ],
)
def test_configurable_ceiling_preserves_both_shorter_and_longer_reference(
    tmp_path, reference, rate, count, anchors
):
    state = interpolation_checkpoint()
    state.update(
        reference=reference,
        reference_fps="979001/32666",
        events={"a.avi": [1, 9, 19], "b.avi": [1, 9, 22]},
    )
    with pytest.raises(sync_events.EventSyncError, match="above 240 FPS"):
        sync_events.manifest_from_checkpoint(
            state,
            infos(tmp_path),
            reference_fps=state["reference_fps"],
            interpolation_method=sync_events.INTERPOLATION,
        )
    manifest = sync_events.manifest_from_checkpoint(
        state,
        infos(tmp_path),
        reference_fps=state["reference_fps"],
        interpolation_method=sync_events.INTERPOLATION,
        max_output_fps="480",
    )
    assert manifest["output_fps"] == rate  # Ceiling authorizes the minimum, never forces 480.
    assert manifest["max_output_fps"] == "480"
    assert manifest["reference_fps"] == state["reference_fps"]
    assert manifest["output_frame_count"] == count
    assert manifest["event_output_frames"] == anchors
    for name, source_count in (("a.avi", 19), ("b.avi", 22)):
        _, rows = sync_events.temporal_maps(manifest, name)
        assert [r["original_frame"] for r in rows if r["type"] == "original"] == list(
            range(1, source_count + 1)
        )
        assert [rows[i - 1]["original_frame"] for i in anchors] == state["events"][name]
        assert manifest["cameras"][name]["original_output_count"] == source_count
        assert manifest["cameras"][name]["interpolated_output_count"] == count - source_count
    # Persisted audit must retain the ceiling when revalidated by Cut Video.
    saved = syncvid.write_checkpoint_file(manifest, tmp_path / "high_rate.events.json")
    assert sync_events.read_event_manifest(saved) == manifest
    manifest["max_output_fps"] = rate  # Exact ceiling is inclusive.
    assert sync_events.validate_manifest(manifest)["output_fps"] == rate
    manifest["max_output_fps"] = str(Fraction(rate) - Fraction(1, 1000))
    with pytest.raises(sync_events.EventSyncError, match="above"):
        sync_events.validate_manifest(manifest)


@pytest.mark.parametrize("ceiling", ["0", "-480", "invalid", "1/0", "10000", True, None])
def test_invalid_ceiling_rejected_in_manifest_and_checkpoint(tmp_path, ceiling):
    manifest = interpolated_model(tmp_path)
    manifest["max_output_fps"] = ceiling
    with pytest.raises(sync_events.EventSyncError, match="Max output FPS"):
        sync_events.validate_manifest(manifest)
    state = interpolation_checkpoint()
    state["max_output_fps"] = ceiling
    path = syncvid.write_checkpoint_file(state, tmp_path / "invalid.json")
    with pytest.raises(syncvid.SyncFileError, match="Max output FPS"):
        syncvid.read_checkpoint_file(path)


def test_legacy_manifest_and_checkpoint_default_to_240(tmp_path):
    state = interpolation_checkpoint()
    path = syncvid.write_checkpoint_file(state, tmp_path / "legacy.json")
    assert syncvid.read_checkpoint_file(path)["max_output_fps"] == "240"
    manifest = interpolated_model(tmp_path)
    manifest.pop("max_output_fps")
    assert sync_events.validate_manifest(manifest)["max_output_fps"] == "240"
    manifest["reference_fps"] = "121"
    with pytest.raises(sync_events.EventSyncError, match="above 240 FPS"):
        sync_events.validate_manifest(manifest)


def test_interpolation_accepts_exactly_240_fps_and_rejects_two_anchors(tmp_path):
    manifest = interpolated_model(tmp_path)
    manifest["reference_fps"] = "120"
    assert sync_events.validate_manifest(manifest)["output_fps"] == "240"
    for camera in manifest["cameras"].values():
        camera["events"] = [camera["events"][0], camera["events"][-1]]
    with pytest.raises(sync_events.EventSyncError, match="Start, Sync and End"):
        sync_events.validate_manifest(manifest)


@pytest.mark.parametrize(
    "events,message",
    [
        ([54, 62], "same ordered events"),
        ([54, 58, 58], "out of order"),
        ([54, 58, 113], "exceeds"),
        ([54, 58, 69], "requires 330 FPS"),
    ],
)
def test_interpolation_invalid_correspondences_and_fps_limit(tmp_path, events, message):
    manifest = interpolated_model(tmp_path)
    manifest["reference_fps"] = "30"
    manifest["cameras"]["b.avi"]["events"] = events
    with pytest.raises(sync_events.EventSyncError, match=message):
        sync_events.validate_manifest(manifest)


def test_interpolation_rejects_removal_marks_inside_only(tmp_path):
    manifest = interpolated_model(tmp_path)
    manifest["cameras"]["b.avi"]["removed_frames"] = [1, 53, 63, 112]
    sync_events.validate_manifest(manifest)
    manifest["cameras"]["b.avi"]["removed_frames"].append(56)
    with pytest.raises(sync_events.EventSyncError, match="b.avi: removal marks inside"):
        sync_events.validate_manifest(manifest)


def test_farneback_translates_texture_to_known_intermediate_positions():
    rng = np.random.default_rng(17)
    left = np.zeros((96, 128, 3), np.uint8)
    left[24:72, 30:78] = rng.integers(35, 255, (48, 48, 3), dtype=np.uint8)
    right = cv2.warpAffine(left, np.float32([[1, 0, 8], [0, 1, 0]]), (128, 96))
    flows = sync_events.bidirectional_flow(left, right)
    assert np.median(flows[0][30:66, 36:72, 0]) == pytest.approx(8, abs=0.25)
    for alpha in (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)):
        expected = cv2.warpAffine(
            left, np.float32([[1, 0, float(alpha) * 8], [0, 1, 0]]), (128, 96)
        )
        generated = sync_events.interpolate_image(left, right, alpha, flows)
        error = np.mean(np.abs(generated[30:66, 40:74].astype(float) - expected[30:66, 40:74]))
        assert error < 3
        assert np.mean(np.abs(left.astype(float) - expected)) > np.mean(
            np.abs(generated.astype(float) - expected)
        )
    np.testing.assert_array_equal(
        sync_events.interpolate_image(left, right, Fraction(0), flows), left
    )
    np.testing.assert_array_equal(
        sync_events.interpolate_image(left, right, Fraction(1), flows), right
    )


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required"
)
def test_interpolation_cli_lossless_pixels_progress_and_cancel(tmp_path, monkeypatch, capsys):
    for info in infos(tmp_path):
        write_source(info.path)
    state_path = syncvid.write_checkpoint_file(interpolation_checkpoint(), tmp_path / "source.json")
    path = tmp_path / "interpolated.events.json"
    assert (
        syncvid.main(
            [
                "--events",
                "--interpolate",
                "--checkpoint",
                str(state_path),
                "-i",
                str(tmp_path),
                "-o",
                str(path),
                "--reference-fps",
                "30000/1001",
            ]
        )
        == 0
    )
    monkeypatch.setattr(
        cutvideo, "RenderProgressDialog", lambda *_a: pytest.fail("CLI created a window")
    )
    out = tmp_path / "export"
    assert (
        cutvideo.main(
            ["--video", str(tmp_path / "a.avi"), "--sync-file", str(path), "--output-dir", str(out)]
        )
        == 0
    )
    manifest = sync_events.read_event_manifest(path)
    for name in manifest["cameras"]:
        source, result = decode(tmp_path / name), decode(out / f"{name}_events.mp4")
        assert len(result) == 37
        with (out / f"{name}_time_map.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        for image, row in zip(result, rows, strict=True):
            if row["type"] == "original":
                np.testing.assert_array_equal(image, source[int(row["original_frame"]) - 1])
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
        )["streams"]
        assert len(metadata) == 1
        assert metadata[0]["r_frame_rate"] == "60000/1001"
        assert metadata[0]["nb_frames"] == "37"
    assert "Event export:" in capsys.readouterr().out
    assert not (out / "INCOMPLETE.txt").exists()
    original_manifest = (out / "event_sync_manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        sync_events.export_event_videos(manifest, tmp_path, out)
    assert (out / "event_sync_manifest.json").read_bytes() == original_manifest
    cancelled = tmp_path / "cancelled"
    progress = []

    def stop(name, done, total):
        progress.append((name, done, total))
        return done < 2

    with pytest.raises(sync_events.EventExportCancelledError):
        sync_events.export_event_videos(manifest, tmp_path, cancelled, progress_callback=stop)
    assert progress[-1][1:] == (2, 74)
    assert (cancelled / "INCOMPLETE.txt").exists()
    assert not list(cancelled.glob("*.mp4"))
    assert not (cancelled / "event_sync_manifest.json").exists()


def test_clock_confirmation_for_legacy_checkpoint_and_new_source(tmp_path):
    state = interpolation_checkpoint()
    state["reference_fps"] = "60000/1001"
    videos = [
        syncvid.VideoInfo(i.path, i.frame_count, 30000 / 1001, i.width, i.height)
        for i in infos(tmp_path)
    ]
    with pytest.raises(sync_events.EventSyncError, match="Confirm reference clock"):
        sync_events.manifest_from_checkpoint(
            state, videos, interpolation_method=sync_events.INTERPOLATION
        )
    manifest = sync_events.manifest_from_checkpoint(
        state, videos, reference_fps="30000/1001", interpolation_method=sync_events.INTERPOLATION
    )
    assert manifest["output_frame_count"] == 37
    state.update(reference_clock_confirmed=True, reference_source_fps="30000/1001")
    assert not sync_events.clock_needs_confirmation(state, videos[0])
    different_source = syncvid.VideoInfo(videos[0].path, 112, 120, 96, 64)
    assert sync_events.clock_needs_confirmation(state, different_source)


def test_gui_interpolation_clock_confirmation_and_checkpoint_roundtrip(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    for info in infos(tmp_path):
        write_source(info.path)
    player = syncvid.PygameSyncPlayer(syncvid.probe_video_directory(tmp_path))
    try:
        state = interpolation_checkpoint()
        state["reference_fps"] = "120"
        player._apply_checkpoint(state)
        player._handle_button("side:mode_interpolate")
        assert player.interpolate_events
        with pytest.raises(sync_events.EventSyncError, match="Confirm reference clock"):
            player._event_manifest()
        player.reference_clock_field.text = "30000/1001"
        player._handle_button("side:confirm_clock")
        manifest = player._event_manifest()
        assert manifest["output_frame_count"] == 37
        player._draw()
        assert player.output_rate_field not in player.visible_fields
        saved = tmp_path / "gui_interpolation.txt"
        monkeypatch.setattr(player, "_choose_output_file", lambda: saved)
        player._save(False)
        restored = syncvid.read_checkpoint_file(tmp_path / "gui_interpolation.checkpoint.json")
        assert restored["interpolation_method"] == sync_events.INTERPOLATION
        assert restored["fps_policy"] == sync_events.AUTO_FPS
        assert restored["reference_clock_confirmed"] is True
        assert "Equivalent event export CLI" in capsys.readouterr().out
        player._apply_checkpoint(restored)
        assert player.interpolate_events
        assert not player.clock_confirmation_required
        assert player._event_manifest()["output_fps"] == "60000/1001"
        cli_path = tmp_path / "resumed_cli.events.json"
        assert (
            syncvid.main(
                [
                    "--events",
                    "--checkpoint",
                    str(tmp_path / "gui_interpolation.checkpoint.json"),
                    "-i",
                    str(tmp_path),
                    "-o",
                    str(cli_path),
                ]
            )
            == 0
        )
        cli_manifest = sync_events.read_event_manifest(cli_path)
        gui_manifest = player._event_manifest()
        for key in (
            "reference_fps",
            "output_fps",
            "interpolation_method",
            "fps_policy",
            "event_times",
            "event_output_frames",
            "cameras",
        ):
            assert cli_manifest[key] == gui_manifest[key]
        player._handle_button("side:mode_fps")
        assert not player.event_mode
        invalid = dict(restored, reference_fps="invalid-clock")
        invalid_path = syncvid.write_checkpoint_file(invalid, tmp_path / "invalid_clock.json")
        monkeypatch.setattr(syncvid, "_portable_path_dialog", lambda *_a, **_k: invalid_path)
        monkeypatch.setattr(player, "_sdl_released_for_dialog", nullcontext)
        player._load_checkpoint()
        assert "Could not load checkpoint" in player.status_text
        assert "Checkpoint clock" in player.status_text
    finally:
        player._release_capture()
        pygame.display.quit()
        pygame.font.quit()


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required"
)
@pytest.mark.parametrize("reference,count", [("a.avi", 235), ("b.avi", 211)])
def test_gui_high_ceiling_cli_and_decoded_originals(
    tmp_path, monkeypatch, capsys, reference, count
):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    for info in infos(tmp_path):
        write_source(info.path)
    state = interpolation_checkpoint()
    state.update(
        reference=reference,
        sync_mode="events",
        reference_fps="979001/32666",
        event_texts={"a.avi": "1, 9, 19", "b.avi": "1, 9, 22"},
    )
    player = syncvid.PygameSyncPlayer(syncvid.probe_video_directory(tmp_path))
    try:
        player._apply_checkpoint(state)
        player._handle_button("side:mode_interpolate")
        player._handle_button("side:confirm_clock")
        with pytest.raises(sync_events.EventSyncError, match="above 240 FPS"):
            player._event_manifest()
        player.max_output_rate_field.text = "480"
        player._handle_button("side:mode_interpolate")  # Re-selecting is idempotent.
        assert player.event_mode and player.interpolate_events
        player._set_active_field(player.max_output_rate_field)
        player._commit_active_field()
        assert f"{count} output frames" in player.status_text
        player._draw()
        assert player.max_output_rate_field in player.visible_fields
        assert player.max_output_rate_field in player._all_fields()
        assert player.output_rate_field not in player.visible_fields
        assert "side:event_mode" not in player.button_rects
        for key in ("fps", "nearest", "interpolate"):
            assert f"side:mode_{key}" in player.button_rects
        monkeypatch.setattr(player, "_choose_output_file", lambda: tmp_path / "gui.txt")
        player._save(False)
        output = capsys.readouterr().out
        assert f"{count} frames per camera" in output
        assert f"19 originals + {count - 19} interpolated" in output
        assert f"22 originals + {count - 22} interpolated" in output
        saved_checkpoint = tmp_path / "gui.checkpoint.json"
        restored = syncvid.read_checkpoint_file(saved_checkpoint)
        assert restored["max_output_fps"] == "480"
        player.max_output_rate_field.text = "240"
        player._apply_checkpoint(restored)
        assert player.max_output_rate_field.text == "480"
        manifest = player._event_manifest()
        cli_args = ["--events", "--checkpoint", str(saved_checkpoint), "-i", str(tmp_path)]
        # Saved ceiling is used without repeating the flag; an explicit override wins.
        with pytest.raises(SystemExit):
            syncvid.main(
                cli_args + ["--max-output-fps", "240", "-o", str(tmp_path / "blocked.json")]
            )
        assert not (tmp_path / "blocked.json").exists()
        cli_output = tmp_path / "cli.events.json"
        assert syncvid.main(cli_args + ["-o", str(cli_output)]) == 0
        cli_manifest = sync_events.read_event_manifest(cli_output)
        for key in ("max_output_fps", "output_fps", "event_output_frames", "cameras"):
            assert cli_manifest[key] == manifest[key]
        export_dir = tmp_path / "export"
        assert (
            cutvideo.main(
                [
                    "--video",
                    str(tmp_path / reference),
                    "--sync-file",
                    str(cli_output),
                    "--output-dir",
                    str(export_dir),
                ]
            )
            == 0
        )
        for name, source_count in (("a.avi", 19), ("b.avi", 22)):
            source = decode(tmp_path / name)
            result = decode(export_dir / f"{name}_events.mp4")
            assert len(result) == count
            with (export_dir / f"{name}_time_map.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            originals = [row for row in rows if row["type"] == "original"]
            assert len(originals) == source_count
            for row in originals:
                np.testing.assert_array_equal(
                    result[int(row["output_frame"]) - 1], source[int(row["original_frame"]) - 1]
                )
            metadata = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_streams",
                        "-show_packets",
                        "-of",
                        "json",
                        str(export_dir / f"{name}_events.mp4"),
                    ]
                )
            )
            stream = metadata["streams"][0]
            assert Fraction(stream["r_frame_rate"]) == Fraction(manifest["output_fps"])
            assert Fraction(stream["avg_frame_rate"]) == Fraction(manifest["output_fps"])
            for index, packet in enumerate(metadata["packets"]):
                assert int(packet["pts"]) * Fraction(stream["time_base"]) == (
                    Fraction(index) / Fraction(manifest["output_fps"])
                )
        player._handle_button("side:mode_nearest")
        player._handle_button("side:mode_nearest")
        assert player.event_mode and not player.interpolate_events
        assert player._event_manifest()["interpolation_method"] == "nearest"
        assert player._event_manifest()["output_fps"] == "30000/1001"
        player._handle_button("side:mode_fps")
        assert not player.event_mode and not player.interpolate_events
        # Returning to a legacy checkpoint must reset a previously raised ceiling.
        player._apply_checkpoint(state)
        assert player.max_output_rate_field.text == "240"
    finally:
        player._release_capture()
        pygame.display.quit()
        pygame.font.quit()
