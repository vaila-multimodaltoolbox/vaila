"""
Unit tests for AI Tracker model selection and directory scanning.

Tests `scan_all_ai_tracker_weights` discovery, ordering, and DeepFeatureExtractor loading.

Author: Prof. Dr. Paulo R. P. Santiago and Rafael L. M. Monteiro
Update Date: 02 October 2026
Version: 0.4.7
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from vaila.tracking import (
    DeepFeatureExtractor,
    scan_all_ai_tracker_weights,
)


def test_scan_all_ai_tracker_weights_empty_dir(tmp_path: Path) -> None:
    empty_dir = tmp_path / "models"
    empty_dir.mkdir()
    weights = scan_all_ai_tracker_weights(empty_dir)
    assert weights == []


def test_scan_all_ai_tracker_weights_discovery_and_ordering(tmp_path: Path) -> None:
    model_dir = tmp_path / "ai_tracker"
    model_dir.mkdir()

    # Create various model files
    (model_dir / "custom_tracker.pt").touch()
    (model_dir / "my_resnet.pth").touch()
    (model_dir / "resnet50_imagenet.pth").touch()
    (model_dir / "mobilenet_v3_small_imagenet.pth").touch()
    (model_dir / "unrelated.txt").touch()
    (model_dir / "notes.csv").touch()

    found = scan_all_ai_tracker_weights(model_dir)
    found_names = [p.name for p in found]

    # Verify unrelated files are ignored
    assert "unrelated.txt" not in found_names
    assert "notes.csv" not in found_names

    # Verify canonical models come first
    assert found_names[0] == "resnet50_imagenet.pth"
    assert found_names[1] == "mobilenet_v3_small_imagenet.pth"

    # Verify custom models are present
    assert "custom_tracker.pt" in found_names
    assert "my_resnet.pth" in found_names
    assert len(found) == 4


def test_scan_all_ai_tracker_weights_default_directory() -> None:
    # Scanning default checkpoint directory should return a list without errors
    weights = scan_all_ai_tracker_weights(None)
    assert isinstance(weights, list)
    for p in weights:
        assert p.suffix.lower() in (".pth", ".pt")


def test_deep_feature_extractor_with_dummy_weights(tmp_path: Path) -> None:
    # Create a small dummy module and save its state_dict
    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

    dummy = TinyBackbone()
    weights_path = tmp_path / "tiny_weights.pth"
    torch.save(dummy.state_dict(), str(weights_path))

    # Initialize extractor with dummy weights
    extractor = DeepFeatureExtractor(
        resnet_variant="resnet50",
        device="cpu",
        weights_path=str(weights_path),
    )
    assert extractor.weights_path == str(weights_path)
    assert extractor.model is not None
    assert extractor.enabled is True
