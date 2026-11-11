"""
================================================================================
Test: test_viewc3d_skeleton.py
================================================================================
vailá - Multimodal Toolbox
Tests for vaila/viewc3d.py's skeleton-JSON marker resolution
(``resolve_skeleton_marker``, used by the ``J`` key of the Open3D viewer).

vailá's skeleton presets (``vaila/skeletons/*.json``) reference markers as
``pN`` with **0-based** N, the same numbering rec3d writes to its CSV/C3D
(``p0..pN-1``). The viewer used to treat ``pN`` as 1-based and drew every
MediaPipe connection shifted by one marker (p11-p12 -> p10-p11).

Expected values are hand-derived from that convention, independently of the
implementation under test.

Update Date: 11 November 2026
Version: 0.4.16
================================================================================
"""

import json
from pathlib import Path

from vaila.viewc3d import resolve_skeleton_marker

REPO_ROOT = Path(__file__).resolve().parents[1]
MEDIAPIPE_JSON = REPO_ROOT / "vaila" / "skeletons" / "mediapipe_pose33.json"

LABELS_0_BASED = [f"p{i}" for i in range(33)]  # what rec3d writes
ALL_33 = list(range(33))


def _p_index(token):
    return int(token[1:])


def test_mediapipe_json_resolves_to_same_zero_based_indices():
    connections = json.loads(MEDIAPIPE_JSON.read_text(encoding="utf-8"))["connections"]
    assert len(connections) == 32

    for a, b in connections:
        assert resolve_skeleton_marker(a, LABELS_0_BASED, ALL_33) == _p_index(a)
        assert resolve_skeleton_marker(b, LABELS_0_BASED, ALL_33) == _p_index(b)


def test_shoulders_are_p11_p12_not_p10_p11():
    # MediaPipe: p11 = left_shoulder, p12 = right_shoulder (0-based).
    assert resolve_skeleton_marker("p11", LABELS_0_BASED, ALL_33) == 11
    assert resolve_skeleton_marker("p12", LABELS_0_BASED, ALL_33) == 12
    assert resolve_skeleton_marker("p0", LABELS_0_BASED, ALL_33) == 0


def test_selected_subset_returns_position_in_selected_space():
    # Viewer shows only markers 11..16; positions in the selected space are 0..5.
    selected = [11, 12, 13, 14, 15, 16]
    assert resolve_skeleton_marker("p11", LABELS_0_BASED, selected) == 0
    assert resolve_skeleton_marker("p16", LABELS_0_BASED, selected) == 5
    # Present in the file but not selected -> cannot be drawn.
    assert resolve_skeleton_marker("p23", LABELS_0_BASED, selected) is None


def test_exact_label_wins_over_position_for_legacy_one_based_labels():
    # Legacy C3D labelled p1..p33: a JSON token "p1" names the label "p1"
    # (position 0), not the marker at index 1.
    labels = [f"p{i}" for i in range(1, 34)]
    assert resolve_skeleton_marker("p1", labels, ALL_33) == 0
    assert resolve_skeleton_marker("p33", labels, ALL_33) == 32


def test_deselected_label_is_not_replaced_by_a_neighbour():
    # Labels p1..p33 with "p3" (index 2) deselected: "p3" must be None, never
    # silently fall back to the marker at position 3 (label "p4").
    labels = [f"p{i}" for i in range(1, 34)]
    selected = [i for i in range(33) if i != 2]
    assert resolve_skeleton_marker("p3", labels, selected) is None


def test_named_labels_resolve_by_normalised_name():
    labels = ["Nose", "Left_Shoulder", "right shoulder", "L.ASIS"]
    selected = [0, 1, 2, 3]
    assert resolve_skeleton_marker("nose", labels, selected) == 0
    assert resolve_skeleton_marker("left_shoulder", labels, selected) == 1
    assert resolve_skeleton_marker("RightShoulder", labels, selected) == 2
    assert resolve_skeleton_marker("lasis", labels, selected) == 3
    assert resolve_skeleton_marker("sacrum", labels, selected) is None


def test_positional_fallback_is_zero_based_when_labels_are_not_pn():
    # Marker names unrelated to pN: "pN" falls back to the 0-based index N.
    labels = ["nose", "leye", "reye", "lear"]
    selected = [0, 1, 2, 3]
    assert resolve_skeleton_marker("p0", labels, selected) == 0
    assert resolve_skeleton_marker("p3", labels, selected) == 3
    assert resolve_skeleton_marker("p4", labels, selected) is None


def test_pn_token_uses_position_not_substring():
    # No label "p1" here: the token is an index (0-based -> position 1, label
    # "p11"), never a substring hit on "p10"/"p11" (which would be ambiguous).
    labels = ["p10", "p11", "p12"]
    assert resolve_skeleton_marker("p1", labels, [0, 1, 2]) == 1
    assert resolve_skeleton_marker("p0", labels, [0, 1, 2]) == 0
