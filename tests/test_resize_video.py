"""Regression tests for crop-only video resizing.

Version: 0.4.0
Update Date: 14 September 2026
"""

import json
from pathlib import Path

import cv2
import numpy as np

from vaila.resize_video import convert_coordinates, resize_with_opencv, run_resize_cli


def _make_video(path: Path, width: int = 32, height: int = 24, frames: int = 4) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"XVID"), 10.0, (width, height))
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((height, width, 3), index * 20, dtype=np.uint8))
    writer.release()


def test_crop_only_clamps_roi_and_writes_expected_dimensions(tmp_path):
    source = tmp_path / "source.avi"
    output = tmp_path / "cropped.avi"
    _make_video(source)
    metadata = resize_with_opencv(str(source), str(output), 1, (4, 5, 40, 30))
    assert metadata is not None
    assert metadata["crop"] == {"x": 4, "y": 5, "width": 28, "height": 19}
    assert metadata["output_width"] == 28
    assert metadata["output_height"] == 20
    capture = cv2.VideoCapture(str(output))
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 28
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 20
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    capture.release()
    saved = json.loads((tmp_path / "cropped_metadata.json").read_text())
    assert saved["crop_applied"] is True


def test_full_resize_callback_is_safe_for_worker_use(tmp_path):
    source = tmp_path / "source.avi"
    output = tmp_path / "full.avi"
    _make_video(source)
    messages = []
    metadata = resize_with_opencv(str(source), str(output), 1, progress_callback=messages.append)
    assert metadata is not None
    assert any("Resizing" in message for message in messages)
    assert output.exists()


def test_coordinate_conversion_preserves_crop_offset():
    original_x, original_y = convert_coordinates(
        20,
        30,
        {"crop_applied": True, "scale_factor": 2, "crop": {"x": 5, "y": 7}},
    )
    assert (original_x, original_y) == (15.0, 22.0)


def test_cli_applies_same_roi_to_directory(tmp_path):
    input_dir = tmp_path / "videos"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    _make_video(input_dir / "first.avi", width=32, height=24)
    _make_video(input_dir / "second.avi", width=32, height=24)

    result = run_resize_cli(input_dir, output_dir, scale_factor=1, roi=(4, 5, 12, 10))

    assert result == 0
    outputs = sorted(output_dir.glob("*_metadata.json"))
    assert len(outputs) == 2
    for metadata_path in outputs:
        metadata = json.loads(metadata_path.read_text())
        assert metadata["crop"] == {"x": 4, "y": 5, "width": 12, "height": 10}


def test_cli_returns_failure_for_empty_input(tmp_path):
    assert run_resize_cli(tmp_path, tmp_path / "outputs") == 1
