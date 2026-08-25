"""
================================================================================
Test: test_viewc3d_multifile.py
================================================================================
vailá - Multimodal Toolbox
Tests for vaila/viewc3d.py's multi-file loading helpers: master-FPS
resolution, master-timeline frame count, nearest-index frame mapping, and
per-file color assignment.

Expected values below are hand-computed from the real fixtures'
known FPS/frame counts (tests/viewc3d/rec3d_200hz.c3d = 200 Hz, 8074 frames;
tests/viewc3d/rec3d_240hz.c3d = 240 Hz, 8298 frames), independently of the
implementation under test -- see loops/viewc3d-multifile-loop.md
"Verification (Governing Check)" for the arithmetic.
================================================================================
"""

import pytest

from vaila.viewc3d import (
    assign_file_color,
    map_master_frame_to_local,
    master_frame_count,
    resolve_master_fps,
)

FPS_200 = 200.0
FPS_240 = 240.0
FRAMES_200 = 8074
FRAMES_240 = 8298
DURATION_200 = FRAMES_200 / FPS_200  # 40.37 s
DURATION_240 = FRAMES_240 / FPS_240  # 34.575 s


def test_resolve_master_fps_defaults_to_highest():
    assert resolve_master_fps([FPS_200, FPS_240]) == FPS_240


def test_resolve_master_fps_downsample_to_lowest():
    assert resolve_master_fps([FPS_200, FPS_240], downsample_to_lowest=True) == FPS_200


def test_resolve_master_fps_empty_raises():
    with pytest.raises(ValueError):
        resolve_master_fps([])


def test_master_frame_count_at_highest_fps():
    # Longest recording (200 Hz file, 40.37 s) expressed at 240 Hz:
    # 40.37 * 240 = 9688.8 -> 9689
    count = master_frame_count([DURATION_200, DURATION_240], FPS_240)
    assert count == 9689


def test_master_frame_count_at_lowest_fps():
    # Same longest duration (40.37 s) expressed at 200 Hz -> 8074 exactly
    count = master_frame_count([DURATION_200, DURATION_240], FPS_200)
    assert count == FRAMES_200


def test_map_master_frame_to_local_last_frame_no_clamp():
    # Master timeline at 240 Hz, last index 9688; the 200 Hz file's own
    # last frame is 8073 -- lands exactly on it, no clamping needed.
    local_idx = map_master_frame_to_local(9688, FPS_240, FPS_200, FRAMES_200)
    assert local_idx == FRAMES_200 - 1  # 8073


def test_map_master_frame_to_local_clamps_shorter_recording():
    # Master timeline at 200 Hz (downsampled), last index 8073; the 240 Hz
    # file's raw mapped index is round(8073*240/200) = 9688, which exceeds
    # its own last frame (8297) since it is a shorter recording -- clamped.
    local_idx = map_master_frame_to_local(8073, FPS_200, FPS_240, FRAMES_240)
    assert local_idx == FRAMES_240 - 1  # 8297


def test_map_master_frame_to_local_never_negative_or_out_of_range():
    for master_idx in (0, 1, 100, 9688):
        local_idx = map_master_frame_to_local(master_idx, FPS_240, FPS_200, FRAMES_200)
        assert 0 <= local_idx <= FRAMES_200 - 1


def test_assign_file_color_cycles_and_wraps():
    palette = [f"color{i}" for i in range(11)]
    assigned = [assign_file_color(i, palette) for i in range(12)]
    assert assigned[:11] == palette
    assert assigned[11] == palette[0]  # 12th load wraps back to the start


def test_assign_file_color_empty_raises():
    with pytest.raises(ValueError):
        assign_file_color(0, [])
