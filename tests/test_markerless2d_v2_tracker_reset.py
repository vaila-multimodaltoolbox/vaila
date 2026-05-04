"""Ensure YOLO tracker reset between videos does not leave empty trackers (Ultralytics persist)."""

from __future__ import annotations


def test_reset_yolo_tracker_state_deletes_tracker_attrs() -> None:
    from vaila.markerless2d_analysis_v2 import reset_yolo_tracker_state

    class Pred:
        pass

    class Model:
        predictor = Pred()

    m = Model()
    m.predictor.trackers = []
    m.predictor.vid_path = [None]

    reset_yolo_tracker_state(m)

    assert not hasattr(m.predictor, "trackers")
    assert not hasattr(m.predictor, "vid_path")
