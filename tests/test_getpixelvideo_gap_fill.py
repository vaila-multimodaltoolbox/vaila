"""Unit tests for getpixelvideo marker Gap Fill (Kalman/RTS + linear)."""

from __future__ import annotations

import numpy as np

from vaila.getpixelvideo import (
    _interior_gap_segments,
    gap_fill_marker_coordinates,
)


def test_interior_gap_segments_respects_max_gap() -> None:
    valid = np.array([True, False, False, True, False, False, False, True], dtype=bool)
    segs = _interior_gap_segments(valid, start=0, end=7, max_gap=2)
    assert segs == [(1, 2)]
    segs_all = _interior_gap_segments(valid, start=0, end=7, max_gap=3)
    assert segs_all == [(1, 2), (4, 6)]


def test_interior_gap_segments_skips_leading_trailing() -> None:
    valid = np.array([False, False, True, False, True, False], dtype=bool)
    segs = _interior_gap_segments(valid, start=0, end=5, max_gap=10)
    assert segs == [(3, 3)]


def test_gap_fill_linear_fills_deleted_interior() -> None:
    coordinates: dict[int, list] = {
        0: [(0.0, 0.0)],
        1: [(10.0, 10.0)],
        2: [(20.0, 20.0)],
        3: [(30.0, 30.0)],
        4: [(40.0, 40.0)],
    }
    deleted = {1: {0}, 2: {0}, 3: {0}}
    n, msg = gap_fill_marker_coordinates(
        coordinates,
        deleted,
        marker_idx=0,
        total_frames=5,
        fps=30.0,
        method="linear",
        max_gap=10,
    )
    assert n == 3
    assert "Gap Fill" in msg
    assert 0 not in deleted.get(1, set())
    assert 0 not in deleted.get(2, set())
    x, y = coordinates[2][0]
    assert abs(float(x) - 20.0) < 1e-6
    assert abs(float(y) - 20.0) < 1e-6


def test_gap_fill_linear_preserves_anchors() -> None:
    coordinates: dict[int, list] = {
        0: [(0.0, 100.0)],
        1: [(None, None)],
        2: [(20.0, 100.0)],
    }
    deleted: dict[int, set[int]] = {1: {0}}
    n, _ = gap_fill_marker_coordinates(
        coordinates,
        deleted,
        marker_idx=0,
        total_frames=3,
        method="linear",
        max_gap=5,
    )
    assert n == 1
    assert coordinates[0][0] == (0.0, 100.0)
    assert coordinates[2][0] == (20.0, 100.0)
    x, y = coordinates[1][0]
    assert abs(float(x) - 10.0) < 1e-6
    assert abs(float(y) - 100.0) < 1e-6


def test_gap_fill_kalman_rts_fills_gap() -> None:
    # Constant-velocity trajectory with a hole
    coordinates: dict[int, list] = {}
    deleted: dict[int, set[int]] = {}
    for f in range(20):
        if 5 <= f <= 9:
            coordinates[f] = [(None, None)]
            deleted[f] = {0}
        else:
            coordinates[f] = [(float(f) * 2.0, 50.0)]
    n, msg = gap_fill_marker_coordinates(
        coordinates,
        deleted,
        marker_idx=0,
        total_frames=20,
        fps=60.0,
        method="kalman_rts",
        max_gap=10,
    )
    assert n == 5
    assert "Kalman" in msg or "RTS" in msg
    for f in range(5, 10):
        x, y = coordinates[f][0]
        assert x is not None and y is not None
        assert np.isfinite(x) and np.isfinite(y)
        # Should stay near the linear motion x = 2*f
        assert abs(float(x) - 2.0 * f) < 3.0
    # Anchors unchanged
    assert coordinates[4][0] == (8.0, 50.0)
    assert coordinates[10][0] == (20.0, 50.0)


def test_gap_fill_respects_max_gap() -> None:
    coordinates: dict[int, list] = {
        0: [(0.0, 0.0)],
        1: [(None, None)],
        2: [(None, None)],
        3: [(None, None)],
        4: [(40.0, 0.0)],
    }
    deleted = {1: {0}, 2: {0}, 3: {0}}
    n, _ = gap_fill_marker_coordinates(
        coordinates,
        deleted,
        marker_idx=0,
        total_frames=5,
        method="linear",
        max_gap=2,
    )
    assert n == 0
    assert 0 in deleted[2]
