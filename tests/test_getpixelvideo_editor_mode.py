"""Unit tests for getpixelvideo VISUAL/INSERT undo and restore helpers."""

from __future__ import annotations

from vaila.getpixelvideo import (
    GETPIXELVIDEO_UPDATE_DATE,
    GETPIXELVIDEO_VERSION,
    MARKER_UNDO_STACK_MAX,
    clone_marker_editor_state,
    pop_marker_undo,
    push_marker_undo,
)


def test_version_stamp_0_4_2() -> None:
    assert GETPIXELVIDEO_VERSION == "0.4.2"
    assert GETPIXELVIDEO_UPDATE_DATE == "15 September 2026"


def test_clone_marker_editor_state_is_deep() -> None:
    coordinates = {0: [(10.0, 20.0)], 1: [(None, None)]}
    deleted_positions = {0: {1}, 1: set()}
    one_line_markers = [(0, 1.0, 2.0)]
    deleted_markers = {3}
    bboxes = {0: [{"x": 1, "y": 2, "w": 3, "h": 4, "label": "obj"}]}
    snap = clone_marker_editor_state(
        coordinates=coordinates,
        deleted_positions=deleted_positions,
        one_line_markers=one_line_markers,
        deleted_markers=deleted_markers,
        bboxes=bboxes,
        selected_marker_idx=2,
    )
    coordinates[0][0] = (99.0, 99.0)
    deleted_positions[0].add(9)
    one_line_markers.append((1, 0.0, 0.0))
    deleted_markers.add(7)
    bboxes[0][0]["x"] = 999
    assert snap["coordinates"][0][0] == (10.0, 20.0)
    assert snap["deleted_positions"][0] == {1}
    assert snap["one_line_markers"] == [(0, 1.0, 2.0)]
    assert snap["deleted_markers"] == {3}
    assert snap["bboxes"][0][0]["x"] == 1
    assert snap["selected_marker_idx"] == 2


def test_push_pop_marker_undo_lifo_and_trim() -> None:
    stack: list[dict] = []
    assert pop_marker_undo(stack) is None
    for i in range(MARKER_UNDO_STACK_MAX + 5):
        push_marker_undo(
            stack,
            clone_marker_editor_state(
                coordinates={0: [(float(i), 0.0)]},
                deleted_positions={},
                one_line_markers=[],
                deleted_markers=set(),
                bboxes={},
                selected_marker_idx=i,
            ),
        )
    assert len(stack) == MARKER_UNDO_STACK_MAX
    last = pop_marker_undo(stack)
    assert last is not None
    assert last["selected_marker_idx"] == MARKER_UNDO_STACK_MAX + 4
    assert last["coordinates"][0][0][0] == float(MARKER_UNDO_STACK_MAX + 4)


def test_restore_snapshot_semantics_independent_of_undo() -> None:
    open_state = clone_marker_editor_state(
        coordinates={0: [(1.0, 1.0)]},
        deleted_positions={0: set()},
        one_line_markers=[],
        deleted_markers=set(),
        bboxes={},
        selected_marker_idx=0,
    )
    undo: list[dict] = []
    push_marker_undo(
        undo,
        clone_marker_editor_state(
            coordinates={0: [(2.0, 2.0)]},
            deleted_positions={0: set()},
            one_line_markers=[],
            deleted_markers=set(),
            bboxes={},
            selected_marker_idx=0,
        ),
    )
    # Successful Save refreshes restore without clearing prior undo until Save path clears it.
    after_save = clone_marker_editor_state(
        coordinates={0: [(3.0, 3.0)]},
        deleted_positions={0: set()},
        one_line_markers=[],
        deleted_markers=set(),
        bboxes={},
        selected_marker_idx=1,
    )
    restore = after_save
    undo.clear()
    assert restore["coordinates"][0][0] == (3.0, 3.0)
    assert restore["selected_marker_idx"] == 1
    assert open_state["coordinates"][0][0] == (1.0, 1.0)
    assert pop_marker_undo(undo) is None
