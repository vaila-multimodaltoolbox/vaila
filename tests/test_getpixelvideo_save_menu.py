"""
Unit tests for GetPixelVideo unified Save Menu and AI tracking weights integration.

Author: Prof. Dr. Paulo R. P. Santiago and Rafael L. M. Monteiro
Update Date: 30 September 2026
Version: 0.4.6
"""

from __future__ import annotations

from pathlib import Path

import vaila.getpixelvideo as gpv
from vaila.tracking import AITrackerParameters
from vaila.tracking.ai_tracker import _default_checkpoint_dir


def test_vaila_tracking_exports_scan_weights() -> None:
    import vaila.tracking as vt

    assert hasattr(vt, "scan_all_ai_tracker_weights")
    assert "scan_all_ai_tracker_weights" in vt.__all__


def test_ai_tracker_checkpoint_dir_exists_or_creatable() -> None:
    ckpt_dir = _default_checkpoint_dir()
    assert isinstance(ckpt_dir, Path)
    assert ckpt_dir.name == "ai_tracker"


def test_ai_tracker_parameters_deep_weights_path() -> None:
    params = AITrackerParameters(
        use_deep_features=True,
        deep_weight=0.25,
        deep_weights_path="/path/to/custom_weights.pth",
        resnet_variant="resnet50",
    )
    assert params.deep_weights_path == "/path/to/custom_weights.pth"
    assert params.use_deep_features is True

    # Test TOML round-trip
    toml_str = params.to_toml()
    loaded = AITrackerParameters.from_toml(toml_str)
    assert loaded.deep_weights_path == "/path/to/custom_weights.pth"
    assert loaded.resnet_variant == "resnet50"
    assert loaded.use_deep_features is True


def test_save_coordinates_round_trip(tmp_path: Path) -> None:
    # Verify save_coordinates functions properly with mock video path
    video_path = str(tmp_path / "mock_video.mp4")
    coordinates = {
        0: [(10.0, 20.0), (30.0, 40.0)],
        1: [(12.0, 22.0), (32.0, 42.0)],
    }
    deleted_positions = {0: set(), 1: set()}

    out_file = gpv.save_coordinates(
        video_path=video_path,
        coordinates=coordinates,
        total_frames=2,
        deleted_positions=deleted_positions,
        is_sequential=False,
    )
    assert Path(out_file).exists()
    assert out_file.endswith("_markers.csv")
