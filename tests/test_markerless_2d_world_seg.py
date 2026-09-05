"""Tests for MediaPipe world landmarks + segmentation config/helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vaila.markerless_2d_analysis import (
    apply_batch_cli_world_seg_overrides,
    blend_segmentation_mask,
    build_world_csv_columns,
    extract_segmentation_mask_array,
    extract_world_landmarks_xyzv,
    get_flat_default_pose_config,
    landmark_names,
    load_config_from_toml,
    pose_landmarker_wants_segmentation,
    save_config_to_toml,
    world_landmarks_to_row,
)


def test_defaults_include_world_and_seg_keys() -> None:
    cfg = get_flat_default_pose_config()
    assert cfg["export_world_landmarks"] is True
    assert cfg["enable_segmentation"] is False
    assert cfg["smooth_segmentation"] is False
    assert cfg["save_segmentation_mask"] is False


def test_toml_roundtrip_world_seg_keys(tmp_path) -> None:
    flat = get_flat_default_pose_config()
    flat["export_world_landmarks"] = True
    flat["enable_segmentation"] = True
    flat["save_segmentation_mask"] = True
    path = tmp_path / "world_seg.toml"
    assert save_config_to_toml(flat, str(path))
    loaded = load_config_from_toml(str(path))
    assert loaded is not None
    assert loaded["export_world_landmarks"] is True
    assert loaded["enable_segmentation"] is True
    assert loaded["save_segmentation_mask"] is True
    assert loaded["smooth_segmentation"] is False


def test_build_world_csv_columns() -> None:
    cols = build_world_csv_columns()
    assert cols[0] == "frame_index"
    assert len(cols) == 1 + len(landmark_names) * 4
    assert cols[1:5] == ["nose_x", "nose_y", "nose_z", "nose_visibility"]


def test_world_landmarks_to_row_known_values() -> None:
    lm = SimpleNamespace(x=0.1, y=0.2, z=0.3, visibility=0.9)
    row = world_landmarks_to_row(7, [lm] + [None] * (len(landmark_names) - 1))
    assert row[0] == 7.0
    assert row[1:5] == pytest.approx([0.1, 0.2, 0.3, 0.9])
    assert np.isnan(row[5]) and np.isnan(row[6])


def test_world_landmarks_to_row_missing_pose() -> None:
    row = world_landmarks_to_row(0, None)
    assert row[0] == 0.0
    assert len(row) == 1 + len(landmark_names) * 4
    assert all(np.isnan(v) for v in row[1:])


def test_extract_world_landmarks_xyzv() -> None:
    pts = [
        SimpleNamespace(x=1.0, y=2.0, z=3.0, visibility=0.5),
        SimpleNamespace(x=4.0, y=5.0, z=6.0, visibility=0.6),
    ]
    result = SimpleNamespace(world_landmarks=[pts])
    out = extract_world_landmarks_xyzv(result)
    assert out is not None
    assert out[0] == pytest.approx([1.0, 2.0, 3.0, 0.5])
    assert out[1] == pytest.approx([4.0, 5.0, 6.0, 0.6])
    assert extract_world_landmarks_xyzv(SimpleNamespace(world_landmarks=[])) is None


def test_pose_landmarker_wants_segmentation() -> None:
    assert pose_landmarker_wants_segmentation({"enable_segmentation": True}) is True
    assert pose_landmarker_wants_segmentation({"enable_segmentation": False}) is False
    assert pose_landmarker_wants_segmentation({}) is False


def test_blend_segmentation_mask_noop_and_tint() -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    assert np.array_equal(blend_segmentation_mask(frame, None), frame)

    mask = np.zeros((4, 4), dtype=np.float32)
    mask[1:3, 1:3] = 1.0
    blended = blend_segmentation_mask(frame, mask, alpha=1.0, color=(0, 255, 0))
    assert blended[2, 2, 1] == 255
    assert blended[0, 0, 1] == 0


def test_extract_segmentation_mask_array() -> None:
    raw = np.ones((2, 2), dtype=np.float32) * 0.8
    result = SimpleNamespace(segmentation_masks=[SimpleNamespace(numpy_view=lambda: raw)])
    arr = extract_segmentation_mask_array(result)
    assert arr is not None
    assert arr.shape == (2, 2)
    assert extract_segmentation_mask_array(SimpleNamespace(segmentation_masks=None)) is None


def test_cli_overrides_world_seg() -> None:
    cfg = get_flat_default_pose_config()
    args = SimpleNamespace(
        export_world_landmarks=False,
        enable_segmentation=True,
        save_segmentation_mask=True,
    )
    out = apply_batch_cli_world_seg_overrides(cfg, args)
    assert out["export_world_landmarks"] is False
    assert out["enable_segmentation"] is True
    assert out["save_segmentation_mask"] is True

    args_none = SimpleNamespace(
        export_world_landmarks=None,
        enable_segmentation=None,
        save_segmentation_mask=None,
    )
    untouched = apply_batch_cli_world_seg_overrides(dict(cfg), args_none)
    assert untouched["export_world_landmarks"] is True
    assert untouched["enable_segmentation"] is False
