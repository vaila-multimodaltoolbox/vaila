"""Regression tests for the Lock (B) marker-selection invariant.

Bug (phase 1): right-clicking to delete a marker reclamped
``selected_marker_idx`` into the current frame's coordinate-list range
even when the selection was locked, silently reassigning selection away
from an unmeasured, out-of-range marker (e.g. marker 3 selected+locked
while the current frame only has slots 0, 1, 2).

Bug (phase 2): the phase-1 fix only guarded the right-click delete
handler. Plain frame navigation (e.g. jumping back to frame 0, where the
locked marker has no data yet) bypassed it entirely and could still let
downstream code observe a stale/out-of-range ``selected_marker_idx``.
The fix adds a centralized ``clamp_selected_marker_index`` call once per
``while running:`` iteration, right after ``frame_count`` settles for
that tick and before any rendering/event handling runs, so the lock
invariant is re-asserted on every frame regardless of how the frame was
reached.
"""

from __future__ import annotations

import inspect

from vaila.getpixelvideo import clamp_selected_marker_index, play_video_with_controls


def test_locked_out_of_range_selection_stays_pinned() -> None:
    # Marker 3 selected and locked; current frame only has 3 slots (0,1,2).
    assert clamp_selected_marker_index(3, True, 3) == 3


def test_locked_selection_pinned_even_with_zero_markers_in_frame() -> None:
    assert clamp_selected_marker_index(3, True, 0) == 3


def test_unlocked_out_of_range_selection_clamps_to_last_slot() -> None:
    assert clamp_selected_marker_index(3, False, 3) == 2


def test_unlocked_selection_resets_to_none_when_frame_has_no_markers() -> None:
    assert clamp_selected_marker_index(3, False, 0) == -1


def test_unlocked_in_range_selection_is_unchanged() -> None:
    assert clamp_selected_marker_index(1, False, 3) == 1


def test_locked_in_range_selection_is_unchanged() -> None:
    assert clamp_selected_marker_index(1, True, 3) == 1


def test_main_loop_reasserts_lock_invariant_every_tick() -> None:
    """clamp_selected_marker_index must run once per main-loop iteration,
    not only inside the right-click delete handler, so plain frame
    navigation cannot desync a locked selection from its pinned marker.
    """
    source = inspect.getsource(play_video_with_controls)
    while_running_idx = source.index("while running:")
    loop_body = source[while_running_idx:]
    call_count = loop_body.count("= clamp_selected_marker_index(")
    # 2 call sites inside the right-click delete handler (one_line_mode
    # and normal-mode branches) + 2 in the centralized per-tick guard
    # (same two branches) = 4 total within the main loop body.
    assert call_count == 4
