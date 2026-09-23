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

Bug (phase 3): ``one_line_mode``'s ``selected_marker_idx`` is a sparse
global position in ``one_line_markers`` (append order), not a dense
0..N-1 per-frame slot count. Both the right-click delete handler and
the phase-2 per-tick guard passed it ``len(markers_in_frame)`` (a
count) as ``clamp_selected_marker_index``'s ``n_markers`` bound —
``if selected_marker_idx >= n_markers: return n_markers - 1``. A
freshly created marker's global index almost always exceeds its own
frame's local marker count, so pressing ``a`` to add a marker in
one_line_mode, then leaving it unlocked for even one tick, silently
reverted the selection to an earlier marker. The fix is
``clamp_selected_marker_index_sparse``, which checks membership in the
list of marker indices present on the current frame instead of a
count, and is used at both one_line_mode call sites.
"""

from __future__ import annotations

import inspect

from vaila.getpixelvideo import (
    clamp_selected_marker_index,
    clamp_selected_marker_index_sparse,
    play_video_with_controls,
)


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


def test_sparse_clamp_locked_pinned_even_if_absent_from_frame() -> None:
    # Marker 5 selected+locked; current frame's markers are [0, 1] only.
    assert clamp_selected_marker_index_sparse(5, True, [0, 1]) == 5


def test_sparse_clamp_unlocked_keeps_selection_when_present_despite_high_global_index() -> None:
    # The actual phase-3 bug: a just-created marker's global index (5)
    # exceeds its own frame's local marker count (1), but the marker IS
    # present in this frame's list — it must stay selected.
    assert clamp_selected_marker_index_sparse(5, False, [5]) == 5


def test_sparse_clamp_unlocked_falls_back_to_last_present_when_absent() -> None:
    assert clamp_selected_marker_index_sparse(5, False, [0, 1, 3]) == 3


def test_sparse_clamp_unlocked_empty_frame_returns_minus_one() -> None:
    assert clamp_selected_marker_index_sparse(5, False, []) == -1


def test_main_loop_reasserts_lock_invariant_every_tick() -> None:
    """clamp_selected_marker_index[_sparse] must run once per main-loop
    iteration, not only inside the right-click delete handler, so plain
    frame navigation cannot desync a locked selection from its pinned
    marker, and one_line_mode must use the sparse (membership) variant
    rather than the dense (count) variant.
    """
    source = inspect.getsource(play_video_with_controls)
    while_running_idx = source.index("while running:")
    loop_body = source[while_running_idx:]
    dense_call_count = loop_body.count("= clamp_selected_marker_index(")
    sparse_call_count = loop_body.count("= clamp_selected_marker_index_sparse(")
    # Dense (normal-mode, count-based): right-click delete handler +
    # centralized per-tick guard = 2 total.
    assert dense_call_count == 2
    # Sparse (one_line_mode, membership-based): right-click delete
    # handler + centralized per-tick guard = 2 total.
    assert sparse_call_count == 2
