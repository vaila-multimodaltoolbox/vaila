"""MediaPipe pose through the Tasks API (mediapipe 1.x has no ``mp.solutions``)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vaila import mp_pose_tasks as mpt

pytest.importorskip("mediapipe")


def _model_or_skip(complexity: int) -> str:
    try:
        return mpt.get_pose_model_path(complexity, download=False)
    except RuntimeError:
        pytest.skip(f"pose model (complexity {complexity}) is not in vaila/models")


def _blank_video(path: Path, frames: int = 4, size: tuple[int, int] = (160, 120)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    for _ in range(frames):
        writer.write(np.full((size[1], size[0], 3), 128, dtype=np.uint8))
    writer.release()


def test_model_path_uses_the_file_that_is_there(tmp_path: Path) -> None:
    (tmp_path / "pose_landmarker_lite.task").write_bytes(b"x")
    assert mpt.get_pose_model_path(0, models_dir=tmp_path) == str(
        tmp_path / "pose_landmarker_lite.task"
    )


def test_model_path_without_download_names_the_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="pose_landmarker_heavy.task"):
        mpt.get_pose_model_path(2, models_dir=tmp_path, download=False)


def test_landmark_visibility_is_a_float_even_when_the_model_gives_none() -> None:
    assert mpt.landmark_visibility(SimpleNamespace(visibility=None)) == 0.0
    assert mpt.landmark_visibility(SimpleNamespace(visibility=0.75)) == 0.75
    assert mpt.landmark_visibility(SimpleNamespace()) == 0.0


def test_timestamps_grow_even_when_the_same_one_is_given_twice() -> None:
    seen: list[int] = []

    class FakeLandmarker:
        def detect_for_video(self, _image, ts):
            seen.append(ts)
            return SimpleNamespace(pose_landmarks=[])

        def close(self):
            pass

    # A fake `mp` as well: the real mp.Image loads libmediapipe.so, which needs libGLESv2 on
    # Linux (absent on the CI runner), and this test is only about the timestamp logic.
    fake_mp = SimpleNamespace(
        Image=lambda image_format, data: SimpleNamespace(format=image_format, data=data),
        ImageFormat=SimpleNamespace(SRGB="SRGB"),
    )
    landmarker = object.__new__(mpt.PoseVideoLandmarker)
    landmarker._mp = fake_mp
    landmarker._landmarker = FakeLandmarker()
    landmarker._last_ms = -1
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    assert landmarker.detect(frame, timestamp_ms=100.0) is None
    assert landmarker.detect(frame, timestamp_ms=100.0) is None
    assert landmarker.detect(frame, timestamp_ms=99.0) is None
    assert seen == [100, 101, 102]


def test_blank_frame_gives_no_person() -> None:
    _model_or_skip(0)
    with mpt.PoseVideoLandmarker(complexity=0) as landmarker:
        frame = np.full((120, 160, 3), 128, dtype=np.uint8)
        assert landmarker.detect(frame, timestamp_ms=0.0) is None


def test_extract_mediapipe_csv_writes_one_row_of_33_points_per_frame(tmp_path: Path) -> None:
    """The 3D pipeline's 2D extractor runs on the Tasks API; no person -> NaN rows."""
    _model_or_skip(2)
    from vaila.markerless_3d_analysis import extract_mediapipe_csv, load_csv33

    video = tmp_path / "blank.mp4"
    _blank_video(video, frames=4)
    w, h, fps, n = extract_mediapipe_csv(str(video), str(tmp_path / "out.csv"))

    assert (w, h, n) == (160, 120, 4)
    assert fps == pytest.approx(10.0)
    points = load_csv33(str(tmp_path / "out.csv"))
    assert points.shape == (4, 33, 2)
    assert np.isnan(points).all()


def test_live_angle_calculator_runs_on_the_tasks_landmarker() -> None:
    """MediaPipeAngleCalculator (webcam mode) no longer needs mp.solutions."""
    _model_or_skip(0)
    from vaila.markerless_live import MediaPipeAngleCalculator

    calculator = MediaPipeAngleCalculator(model_complexity=0)
    assert calculator.mp_pose.PoseLandmark.RIGHT_SHOULDER == 12
    angles, connections, visible = calculator.process_frame(
        np.full((120, 160, 3), 128, dtype=np.uint8)
    )
    assert (angles, connections, visible) == ({}, [], [])
    assert calculator.last_landmarks is None
