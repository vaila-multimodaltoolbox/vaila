"""Unit tests for the post-tracking ID cap (global rerank) helpers in
vaila.yolov26track.

The helpers are pure-Python: they do not require a GPU, Ultralytics, or a
real video. We mock just enough of the Ultralytics ``Results`` API for the
``buffer_tracking_stream`` path.

Run with::

    uv run pytest tests/test_yolov26track_idcap.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the vaila package importable when tests are run from the repo root.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vaila.yolov26track import (  # noqa: E402
    _BufferedFrame,
    _memory_snapshot,
    _release_yolo_gpu_memory,
    build_id_rerank_map,
    buffer_tracking_stream,
    rerank_buffered_stream,
    rewrite_ultralytics_boxes_id,
)


class _FakeBoxes:
    """Minimal stand-in for Ultralytics ``result.boxes``."""

    def __init__(self, ids: list[int], xyxys: list[tuple[float, float, float, float]]):
        import torch  # local import; tests assume torch is available

        self.id = torch.as_tensor(ids, dtype=torch.int32)
        self.xyxy = torch.as_tensor([list(b) for b in xyxys], dtype=torch.float32)
        self.conf = torch.ones((len(ids),), dtype=torch.float32)
        self.cls = torch.zeros((len(ids),), dtype=torch.float32)
        self.device = self.id.device


class _FakeResult:
    def __init__(self, ids: list[int], xyxys: list[tuple[float, float, float, float]]):
        self.boxes = _FakeBoxes(ids, xyxys)

    def plot(self) -> np.ndarray:
        return np.zeros((4, 4, 3), dtype=np.uint8)


def _stream(frames_data: list[tuple[list[int], list[tuple[float, float, float, float]]]]):
    for ids, xyxys in frames_data:
        yield _FakeResult(ids, xyxys)


def test_build_id_rerank_map_disabled_when_zero():
    id_counts = {1: 50, 2: 30, 3: 5}
    assert build_id_rerank_map(id_counts, 0) == {}
    assert build_id_rerank_map(id_counts, -1) == {}


def test_build_id_rerank_map_keeps_top_n_by_persistence():
    id_counts = {1: 50, 2: 30, 3: 5, 4: 99, 5: 1}
    m = build_id_rerank_map(id_counts, 2)
    assert m == {4: 1, 1: 2}


def test_build_id_rerank_map_deterministic_ties():
    # When persistence is equal, lower raw id wins (deterministic).
    id_counts = {7: 10, 3: 10, 5: 10}
    m = build_id_rerank_map(id_counts, 2)
    assert list(m.keys()) == [3, 5]
    assert list(m.values()) == [1, 2]


def test_buffer_tracking_stream_counts_ids():
    frames = [
        ([1, 2], [(0, 0, 1, 1), (2, 2, 3, 3)]),
        ([1, 2], [(0, 0, 1, 1), (2, 2, 3, 3)]),
        ([1, 3], [(0, 0, 1, 1), (4, 4, 5, 5)]),
        ([2, 3], [(2, 2, 3, 3), (4, 4, 5, 5)]),
    ]
    buffer, counts = buffer_tracking_stream(_stream(frames), save_annotated=False)
    assert counts == {1: 3, 2: 3, 3: 2}
    assert len(buffer) == 4
    assert all(isinstance(bf, _BufferedFrame) for bf in buffer)
    assert [len(bf.detections) for bf in buffer] == [2, 2, 2, 2]


def test_buffer_tracking_stream_lightweight_drops_raw_result():
    frames = [([1], [(0, 0, 1, 1)])]
    buffer, _ = buffer_tracking_stream(
        _stream(frames),
        save_annotated=False,
        keep_raw_result=False,
    )
    assert len(buffer) == 1
    assert buffer[0].raw_result is None
    assert buffer[0].detections[0]["raw_id"] == 1


def test_rerank_buffered_stream_drops_unmapped_ids():
    frames = [
        ([1, 2, 7], [(0, 0, 1, 1), (2, 2, 3, 3), (9, 9, 10, 10)]),
        ([1, 7], [(0, 0, 1, 1), (9, 9, 10, 10)]),
    ]
    buffer, counts = buffer_tracking_stream(_stream(frames), save_annotated=False)
    # Keep only the top-1 (raw id 1, observed in both frames).
    m = build_id_rerank_map(counts, 1)
    out = rerank_buffered_stream(buffer, m)
    # Frame 0: only raw id 1 -> 1 det. Frame 1: only raw id 1 -> 1 det.
    assert [len(bf.detections) for bf in out] == [1, 1]
    assert all(bf.detections[0]["raw_id"] == 1 for bf in out)


def test_rerank_buffered_stream_renumbers_to_1_n():
    frames = [
        ([10, 20], [(0, 0, 1, 1), (2, 2, 3, 3)]),
        ([10, 20], [(0, 0, 1, 1), (2, 2, 3, 3)]),
        ([10], [(0, 0, 1, 1)]),
    ]
    buffer, counts = buffer_tracking_stream(_stream(frames), save_annotated=False)
    m = build_id_rerank_map(counts, 2)
    out = rerank_buffered_stream(buffer, m)
    # Top-2 by persistence: 10 (3 frames) and 20 (2 frames) -> 1 and 2.
    assert m == {10: 1, 20: 2}
    new_ids_per_frame = [[d["raw_id"] for d in bf.detections] for bf in out]
    assert new_ids_per_frame == [[1, 2], [1, 2], [1]]


def test_rewrite_ultralytics_boxes_id_keeps_only_mapped():
    result = _FakeResult([1, 2, 7], [(0, 0, 1, 1), (2, 2, 3, 3), (9, 9, 10, 10)])
    rewrite_ultralytics_boxes_id(result, {1: 1, 2: 2})
    # The unmapped id (7) must be removed; the kept ids must be re-mapped.
    assert result.boxes.id is not None
    assert result.boxes.id.tolist() == [1, 2]
    assert result.boxes.xyxy.shape[0] == 2


def test_rewrite_ultralytics_boxes_id_no_op_when_empty_map():
    result = _FakeResult([1, 2], [(0, 0, 1, 1), (2, 2, 3, 3)])
    n_before = result.boxes.xyxy.shape[0]
    rewrite_ultralytics_boxes_id(result, {})
    assert result.boxes.xyxy.shape[0] == n_before
    assert result.boxes.id.tolist() == [1, 2]


def test_release_yolo_gpu_memory_smoke():
    _release_yolo_gpu_memory()


def test_memory_snapshot_returns_ram_or_vram_keys():
    snap = _memory_snapshot()
    assert isinstance(snap, dict)
    assert snap  # at least RAM on any test runner
