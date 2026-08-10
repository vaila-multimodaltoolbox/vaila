"""Tests for yolov26track.py's additive --reid-postprocess hook.

The hook is additive-only: --max-ids (drop-based, live) and --stabilize-ids
(GeometricFrameLinker, live) are untouched regardless of this flag -- see
loops/reid-markers-geometric-max-ids-loop.md Goal §8. No GPU/model/video is
needed to test the resolution logic or CLI wiring; the actual merge call is
covered end-to-end by tests/test_reid_markers_cli.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from vaila.yolov26track import resolve_reid_postprocess_max_ids


def test_resolve_prefers_explicit_postprocess_max_ids() -> None:
    assert resolve_reid_postprocess_max_ids(7, 16) == 7


def test_resolve_falls_back_to_live_max_ids_when_postprocess_unset() -> None:
    assert resolve_reid_postprocess_max_ids(None, 16) == 16


def test_resolve_falls_back_to_none_when_neither_set() -> None:
    assert resolve_reid_postprocess_max_ids(None, None) is None


def test_resolve_ignores_a_zero_or_negative_live_max_ids() -> None:
    """--max-ids 0 means 'cap disabled' for the live tracker -- must not leak
    into the postprocess merge as an actual max_ids=0."""
    assert resolve_reid_postprocess_max_ids(None, 0) is None
    assert resolve_reid_postprocess_max_ids(None, -1) is None


def test_resolve_explicit_zero_postprocess_falls_through_to_live_max_ids() -> None:
    # 0 is falsy -> "not explicitly set" per the same convention as --max-ids.
    assert resolve_reid_postprocess_max_ids(0, 16) == 16


def test_track_cli_help_registers_reid_postprocess_flags() -> None:
    """Real subprocess against the actual CLI -- no mocking of argparse
    internals, confirms the flags are genuinely registered end to end."""
    proc = subprocess.run(
        [sys.executable, "-u", "-m", "vaila.yolov26track", "track", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert proc.returncode == 0
    assert "--reid-postprocess" in proc.stdout
    assert "--reid-postprocess-max-ids" in proc.stdout
    # Existing live flags must still be present, unchanged, alongside the new ones.
    assert "--max-ids" in proc.stdout
    assert "--stabilize-ids" in proc.stdout
