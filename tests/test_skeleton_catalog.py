"""Tests for vaila.skeleton_catalog (Tpl: pose presets).

Update Date: 16 September 2026
Version: 0.4.3
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vaila import skeleton_catalog

EXPECTED_POSE_IDS = [
    "mediapipe",
    "yolo",
    "openpose25",
    "halpe26",
    "fifa15",
    "sam3d70",
    "sapiens308",
    "hand21",
    "hands42",
    "holistic75",
    "wholebody133",
]

EXPECTED_COUNTS = {
    "mediapipe": 33,
    "yolo": 17,
    "openpose25": 25,
    "halpe26": 26,
    "fifa15": 15,
    "sam3d70": 70,
    "sapiens308": 308,
    "hand21": 21,
    "hands42": 42,
    "holistic75": 75,
    "wholebody133": 133,
}


@pytest.fixture(autouse=True)
def _clear_catalog_cache() -> None:
    skeleton_catalog.clear_catalog_cache()
    yield
    skeleton_catalog.clear_catalog_cache()


def test_skeletons_dir_exists() -> None:
    root = skeleton_catalog.skeletons_dir()
    assert root.is_dir()
    assert (root / "mediapipe_pose33.json").is_file()
    assert (root / "sapiens2_goliath308.json").is_file()


def test_list_pose_templates_order_and_counts() -> None:
    specs = skeleton_catalog.list_pose_templates()
    ids = [s.id for s in specs]
    assert ids == EXPECTED_POSE_IDS
    for spec in specs:
        assert spec.num_keypoints == EXPECTED_COUNTS[spec.id]
        assert len(spec.keypoints) == EXPECTED_COUNTS[spec.id]
        assert len(spec.connections) > 0
        assert all(
            0 <= a < spec.num_keypoints and 0 <= b < spec.num_keypoints for a, b in spec.connections
        )
        assert not Path(spec.path).name.startswith("soccerfield_")


def test_soccerfield_jsons_excluded_from_catalog() -> None:
    root = skeleton_catalog.skeletons_dir()
    assert (root / "soccerfield_calib29.json").is_file()
    assert (root / "soccerfield_pitch32.json").is_file()
    names = {s.path.name for s in skeleton_catalog.list_pose_templates()}
    assert "soccerfield_calib29.json" not in names
    assert "soccerfield_pitch32.json" not in names


def test_template_labels_include_specials_and_pose() -> None:
    labels = skeleton_catalog.template_labels()
    assert labels["free"] == "Free"
    assert labels["fifa"] == "Soccer-Kiki"
    assert labels["sam3d70"] == "SAM3D70"
    assert labels["sapiens308"] == "Sapiens308"
    assert labels["mediapipe"] == "MediaPipe33"


def test_dialog_prompt_is_column_layout() -> None:
    prompt = skeleton_catalog.format_template_dialog_prompt()
    lines = prompt.splitlines()
    assert lines[0].startswith("Tpl —")
    assert "0 = Free" in lines[1]
    assert "1 = Soccer-Kiki" in lines[2]
    assert "FIFA Soccer-Field" not in prompt
    assert any("SAM3+DINOv3" in line for line in lines)
    assert any("Sapiens2 Goliath" in line for line in lines)
    # One option per line after the header.
    assert len(lines) == 1 + len(skeleton_catalog.dialog_choices())


def test_resolve_dialog_choice_numeric_and_id() -> None:
    assert skeleton_catalog.resolve_dialog_choice(None) is None
    assert skeleton_catalog.resolve_dialog_choice("") is None
    assert skeleton_catalog.resolve_dialog_choice("0") == "free"
    assert skeleton_catalog.resolve_dialog_choice("1") == "fifa"
    assert skeleton_catalog.resolve_dialog_choice("2") == "mediapipe"
    assert skeleton_catalog.resolve_dialog_choice("7") == "sam3d70"
    assert skeleton_catalog.resolve_dialog_choice("8") == "sapiens308"
    assert skeleton_catalog.resolve_dialog_choice("sam3d70") == "sam3d70"
    assert skeleton_catalog.resolve_dialog_choice("99") is None


def test_get_template_sam3_and_sapiens() -> None:
    sam = skeleton_catalog.get_template("sam3d70")
    assert sam is not None
    assert sam.keypoints[0] == "nose"
    assert sam.num_keypoints == 70
    sapiens = skeleton_catalog.get_template("sapiens308")
    assert sapiens is not None
    assert sapiens.num_keypoints == 308
    assert skeleton_catalog.get_template("missing") is None


def test_template_button_caption_shows_soccer_kiki() -> None:
    from vaila.getpixelvideo import template_button_caption

    # In FIFA / pitch-guide mode, show only Soccer-Kiki (no "Tpl:", no "FIFA").
    assert template_button_caption("fifa") == "Soccer-Kiki"
    assert "FIFA" not in template_button_caption("fifa")
    assert "Tpl:" not in template_button_caption("fifa")

    # In other modes, use the Tpl: prefix.
    assert template_button_caption("free") == "Tpl: Free"
    assert template_button_caption("mediapipe") == "Tpl: MediaPipe33"
    assert template_button_caption("yolo") == "Tpl: YOLO17"
    assert template_button_caption("sam3d70") == "Tpl: SAM3D70"
