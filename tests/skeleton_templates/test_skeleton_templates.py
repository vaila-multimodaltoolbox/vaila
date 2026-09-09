"""Tests for skeleton templates and presets in vaila/skeletons and tests/skeleton_templates.

Update Date: 24 August 2026
Version: 0.3.112
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from vaila.rec3d import generate_blender_companion_script

REPO_ROOT = Path(__file__).resolve().parents[1]
SKEL_DIR = REPO_ROOT / "vaila" / "skeletons"
TEMPLATE_DIR = REPO_ROOT / "tests" / "skeleton_templates"
REC3D_DIR = REPO_ROOT / "tests" / "rec3d_one_dlt3d"

EXPECTED_PRESETS = [
    ("fifa_body15.json", 15),
    ("yolo_coco17.json", 17),
    ("mediapipe_hand21.json", 21),
    ("openpose_body25.json", 25),
    ("halpe26.json", 26),
    ("soccerfield_calib29.json", 29),
    ("soccerfield_pitch32.json", 32),
    ("mediapipe_pose33.json", 33),
    ("mediapipe_hands42.json", 42),
    ("sam3dinov3_mhr70.json", 70),
    ("mediapipe_holistic75.json", 75),
    ("coco_wholebody133.json", 133),
    ("sapiens2_goliath308.json", 308),
]


def _check_skeleton_json_structure(file_path: Path, expected_count: int | None = None) -> dict:
    assert file_path.is_file(), f"File does not exist: {file_path}"
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    assert "schema" in data, f"Missing 'schema' in {file_path.name}"
    assert isinstance(data["schema"], str) and len(data["schema"]) > 0

    assert "note" in data, f"Missing 'note' in {file_path.name}"
    assert isinstance(data["note"], str)

    assert "connections" in data, f"Missing 'connections' in {file_path.name}"
    connections = data["connections"]
    assert isinstance(connections, list), f"'connections' must be list in {file_path.name}"
    assert len(connections) > 0, f"'connections' is empty in {file_path.name}"

    num_kp = data.get("num_keypoints")
    if expected_count is not None and num_kp is not None:
        assert num_kp == expected_count, (
            f"num_keypoints mismatch in {file_path.name}: {num_kp} != {expected_count}"
        )

    if "keypoints" in data:
        kps = data["keypoints"]
        assert isinstance(kps, list)
        if expected_count is not None:
            assert len(kps) == expected_count, f"keypoints list length mismatch in {file_path.name}"

    seen_edges = set()
    for pair in connections:
        assert isinstance(pair, list) and len(pair) == 2, (
            f"Invalid connection pair {pair} in {file_path.name}"
        )
        a, b = pair[0], pair[1]
        assert isinstance(a, str) and isinstance(b, str)
        assert re.match(r"^p\d+$", a), f"Invalid joint format {a!r} in {file_path.name}"
        assert re.match(r"^p\d+$", b), f"Invalid joint format {b!r} in {file_path.name}"

        ia = int(a[1:])
        ib = int(b[1:])
        assert ia != ib, f"Self-connection {pair} in {file_path.name}"
        assert ia >= 1 and ib >= 1, f"Non-positive index in {pair} in {file_path.name}"
        if num_kp is not None:
            assert ia <= num_kp, f"Index {ia} exceeds num_keypoints ({num_kp}) in {file_path.name}"
            assert ib <= num_kp, f"Index {ib} exceeds num_keypoints ({num_kp}) in {file_path.name}"

        edge_key = tuple(sorted((ia, ib)))
        assert edge_key not in seen_edges, f"Duplicate connection {pair} in {file_path.name}"
        seen_edges.add(edge_key)

    return data


@pytest.mark.parametrize(("filename", "expected_count"), EXPECTED_PRESETS)
def test_vaila_skeletons_presets(filename: str, expected_count: int) -> None:
    path = SKEL_DIR / filename
    _check_skeleton_json_structure(path, expected_count)


@pytest.mark.parametrize(("filename", "expected_count"), EXPECTED_PRESETS)
def test_tests_skeleton_templates_presets(filename: str, expected_count: int) -> None:
    path = TEMPLATE_DIR / filename
    _check_skeleton_json_structure(path, expected_count)


@pytest.mark.parametrize(
    ("alias_name", "target_file"),
    [
        ("skeleton_pose_mediapipe.json", "mediapipe_pose33.json"),
        ("skeleton_pose_yolo.json", "yolo_coco17.json"),
        ("skeleton_pose_sam3dinov3.json", "sam3dinov3_mhr70.json"),
        ("skeleton_pose_sapiens2.json", "sapiens2_goliath308.json"),
        ("skeleton_pose_fifa.json", "fifa_body15.json"),
        ("skeleton_pose_openpose25.json", "openpose_body25.json"),
        ("skeleton_pose_hand21.json", "mediapipe_hand21.json"),
    ],
)
def test_template_aliases(alias_name: str, target_file: str) -> None:
    alias_path = TEMPLATE_DIR / alias_name
    target_path = TEMPLATE_DIR / target_file
    _check_skeleton_json_structure(alias_path)
    assert json.loads(alias_path.read_text(encoding="utf-8")) == json.loads(
        target_path.read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    "fixture_name",
    [
        "skeleton_pose_mediapipe.json",
        "skeleton_pose_yolo.json",
        "skeleton_pose_sam3dinov3.json",
        "skeleton_pose_sapiens2.json",
        "skeleton_pose_fifa.json",
    ],
)
def test_rec3d_test_fixtures(fixture_name: str) -> None:
    path = REC3D_DIR / fixture_name
    _check_skeleton_json_structure(path)


@pytest.mark.parametrize(("filename", "expected_count"), EXPECTED_PRESETS)
def test_generate_blender_companion_script_with_each_template(
    tmp_path: Path, filename: str, expected_count: int
) -> None:
    json_path = TEMPLATE_DIR / filename
    bvh_path = tmp_path / "trial.bvh"
    bvh_path.write_text(
        "HIERARCHY\nROOT p1\n{\n}\nMOTION\nFrames: 10\nFrame Time: 0.016666667\n0 0 0\n",
        encoding="utf-8",
    )
    script_path = generate_blender_companion_script(
        output_dir=str(tmp_path),
        file_base="trial",
        skeleton_json_path=str(json_path),
        point_rate=60.0,
        n_frames=10,
    )
    assert script_path is not None
    script_text = Path(script_path).read_text(encoding="utf-8")
    assert "Vaila_Skeleton" in script_text
    assert "connections = [" in script_text


@pytest.mark.parametrize(
    "template_file",
    [
        "mediapipe_pose33.json",
        "sam3dinov3_mhr70.json",
        "sapiens2_goliath308.json",
        "fifa_body15.json",
        "yolo_coco17.json",
    ],
)
def test_rec3d_one_dlt3d_cli_with_skeleton_templates(tmp_path: Path, template_file: str) -> None:
    import subprocess
    import sys

    dlt1 = str(REC3D_DIR / "cam1_dlt_calib.dlt3d")
    dlt2 = str(REC3D_DIR / "cam2_dlt_calib.dlt3d")
    pix1 = str(REC3D_DIR / "cam01_makerless.csv")
    pix2 = str(REC3D_DIR / "cam02_markerless.csv")
    skel_file = str(TEMPLATE_DIR / template_file)
    output_dir = tmp_path / "out"

    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "vaila.rec3d_one_dlt3d",
            "--dlt3d",
            dlt1,
            dlt2,
            "--pixels",
            pix1,
            pix2,
            "--fps",
            "100",
            "--output",
            str(output_dir),
            "--skeleton",
            skel_file,
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"rec3d_one_dlt3d failed with {template_file}: {res.stderr}"

    # Check that companion script was generated and contains custom connections
    viz_scripts = list(output_dir.rglob("*_blender_skeleton_viz.py"))
    assert len(viz_scripts) >= 1
    script_content = viz_scripts[0].read_text(encoding="utf-8")
    assert "Vaila_Skeleton" in script_content
    assert "connections = [" in script_content


def test_tugturn_loads_skeleton_template() -> None:
    from vaila.tugturn import load_mediapipe_pose_connections

    template_path = TEMPLATE_DIR / "mediapipe_pose33.json"
    conns = load_mediapipe_pose_connections(template_path)
    assert len(conns) >= 30
    # verify 0-based indices for tugturn
    assert all(0 <= a < 33 and 0 <= b < 33 for a, b in conns)
