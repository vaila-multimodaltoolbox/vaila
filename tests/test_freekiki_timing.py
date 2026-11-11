"""Camera clocks (vaila/freekiki_timing.py): the same interval counted on cameras of different rates.

The numbers of the first test are the three counts of the Lescano x Carlos Miguel penalty: 11 frames
of the main camera at 29.97 fps, 19 of the back camera (a replay captured at 50 fps) and 42 of the
slow-motion camera (119.88 fps). Synthetic data otherwise.

Version: 0.4.16
Update Date: 11 November 2026
"""

from __future__ import annotations

import numpy as np
import pytest

from vaila import freekiki_timing as ft

MAIN = ft.CameraSpan(
    "main", file_fps=30000 / 1001, capture_fps=30000 / 1001, frames=11, source="measured"
)
BACK = ft.CameraSpan("back", file_fps=30000 / 1001, capture_fps=50.0, frames=19, source="measured")
SLOW = ft.CameraSpan(
    "slow", file_fps=30000 / 1001, capture_fps=119.88, frames=42, source="declared"
)


def test_frames_are_seconds_only_at_the_capture_rate() -> None:
    assert MAIN.seconds() == pytest.approx(0.3670, abs=1e-4)
    assert MAIN.tolerance() == pytest.approx(0.03337, abs=1e-5)  # one frame at 29.97 fps
    assert BACK.seconds() == pytest.approx(0.380, abs=1e-9)
    assert BACK.tolerance() == pytest.approx(0.020, abs=1e-9)
    assert SLOW.seconds() == pytest.approx(0.3503, abs=1e-4)
    assert SLOW.tolerance() == pytest.approx(0.00834, abs=1e-5)
    # the replay is shown slower than it happened: 19 frames take 0.634 s to play at 29.97 fps
    assert BACK.shown_seconds() == pytest.approx(19 / (30000 / 1001), abs=1e-9)
    assert BACK.slowdown() == pytest.approx(50.0 / (30000 / 1001), abs=1e-9)
    assert SLOW.slowdown() == pytest.approx(4.0, abs=1e-3)


def test_the_three_counts_measure_one_interval_within_their_tolerances() -> None:
    out = ft.reconcile([MAIN, BACK, SLOW])
    assert [r["name"] for r in out["cameras"]] == ["main", "back", "slow"]
    pairs = {(p["a"], p["b"]): p for p in out["pairs"]}
    assert pairs[("main", "back")]["consistent"] and pairs[("main", "slow")]["consistent"]
    # back (0.3800 s) and slow (0.3503 s) differ by 29.7 ms and tolerate 28.3 ms: they miss by ~1.4 ms
    gap = pairs[("back", "slow")]
    assert not gap["consistent"] and gap["gap_ms"] == pytest.approx(1.4, abs=0.2)
    assert out["common_interval_s"] is None and out["max_gap_ms"] == pytest.approx(1.4, abs=0.2)
    # the camera with the finest clock weighs the most
    assert out["weighted_mean_s"] == pytest.approx(0.353, abs=0.004)
    assert out["weighted_mean_s"] < MAIN.seconds()
    assert out["span_s"] == pytest.approx(0.380 - 0.3503, abs=1e-3)


def test_counts_that_overlap_have_a_common_interval() -> None:
    a = ft.CameraSpan("a", 30.0, 30.0, 11)
    b = ft.CameraSpan("b", 30.0, 60.0, 21)
    out = ft.reconcile([a, b])
    lo, hi = out["common_interval_s"]
    assert lo <= 11 / 30 <= hi and lo <= 21 / 60 <= hi and out["max_gap_ms"] == 0.0


def test_the_effective_rate_of_a_replay_comes_from_the_events_both_cameras_see() -> None:
    rate, offset = 50.0, 7.3  # the replay: 50 captured frames per second, its frame 0 at 7.3 s
    frames = np.array(
        [30.0, 33.0, 41.0, 60.0]
    )  # onset, support foot, contact, save (replay frames)
    times = offset + frames / rate  # the same events on the main camera's clock (seconds)
    fit = ft.fit_clock(times, frames)
    assert fit["rate_hz"] == pytest.approx(rate, rel=1e-9)
    assert fit["offset_s"] == pytest.approx(offset, abs=1e-9)
    assert fit["rms_ms"] == pytest.approx(0.0, abs=1e-6) and fit["n"] == 4
    # a one-frame slip on one event shows in the residuals instead of bending the rate silently
    slipped = times.copy()
    slipped[1] += 1 / (30000 / 1001)
    worse = ft.fit_clock(slipped, frames)
    assert worse["rms_ms"] > 5.0 and max(abs(r) for r in worse["residual_ms"]) > 10.0
    with pytest.raises(ValueError, match="two events"):
        ft.fit_clock([1.0], [1.0])


def test_the_slowdown_of_a_replay_follows_from_its_effective_rate() -> None:
    fit = ft.fit_clock(np.array([0.0, 0.3592]), np.array([41.0, 60.0]))  # 19 frames in 0.3592 s
    assert fit["rate_hz"] == pytest.approx(19 / 0.3592, rel=1e-9)  # 52.9 fps
    assert ft.slowdown(fit["rate_hz"], 30000 / 1001) == pytest.approx(52.9 / 29.97, rel=2e-3)


def test_bad_inputs_are_refused() -> None:
    with pytest.raises(ValueError, match="rate"):
        ft.CameraSpan("x", 30.0, 0.0, 5).seconds()
    with pytest.raises(ValueError, match="frames"):
        ft.reconcile([ft.CameraSpan("x", 30.0, 30.0, 0), MAIN])
    with pytest.raises(ValueError, match="two cameras"):
        ft.reconcile([MAIN])
