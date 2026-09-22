"""Regression tests for the Lock (B) marker-selection invariant.

Bug: right-clicking to delete a marker reclamped ``selected_marker_idx``
into the current frame's coordinate-list range even when the selection
was locked, silently reassigning selection away from an unmeasured,
out-of-range marker (e.g. marker 3 selected+locked while the current
frame only has slots 0, 1, 2).
"""

from __future__ import annotations

from vaila.getpixelvideo import clamp_selected_marker_index


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
