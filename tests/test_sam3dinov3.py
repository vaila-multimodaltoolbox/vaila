"""Tests for :mod:`vaila.sam3dinov3` (SAM3 + DINOv3 / SAM 3D Body markerless 3D).

Only the CPU-side logic is covered: keypoint metadata, the SAM->SAM 3D Body
batch bridge, and the CSV/JSON writers. The GPU inference itself needs the gated
``facebook/sam-3d-body-dinov3`` weights and CUDA, so it is out of scope here.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from vaila.sam3dinov3 import (
    COLOR_CENTER_RGB,
    COLOR_LEFT_RGB,
    COLOR_RIGHT_RGB,
    MHR70_NAMES,
    _collect_person_ids,
    _draw_pose_overlay,
    _frame_batch_from_guidance,
    _instances_from_outputs,
    _rgb_to_bgr,
    _side_color_bgr,
    _write_readme,
    build_worker_command,
    keypoint_names,
    resolve_sam3d_assets,
    skeleton_edges,
    write_camera_csv,
    write_long_keypoints_csvs,
    write_predictions_json,
    write_wide_person_csvs,
)

W, H = 640, 480
N_KPTS = len(MHR70_NAMES)


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def _track(obj_id: int, x: float, y: float, w: float, h: float) -> dict[str, object]:
    return {
        "frame": 0,
        "obj_id": obj_id,
        "x_px": x,
        "y_px": y,
        "w_px": w,
        "h_px": h,
        "score": 0.9,
        "area_px": int(w * h),
        "cx_px": x + w / 2,
        "cy_px": y + h / 2,
    }


def _contour(obj_id: int, x: float, y: float, w: float, h: float) -> dict[str, object]:
    return {
        "obj_id": obj_id,
        "polygons": [
            [
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h],
            ]
        ],
    }


def _fake_output(seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    return {
        "bbox": np.array([10.0, 20.0, 110.0, 220.0], dtype=np.float32),
        "focal_length": np.array([1234.5], dtype=np.float32),
        "pred_keypoints_3d": rng.normal(size=(N_KPTS, 3)).astype(np.float32),
        "pred_keypoints_2d": rng.uniform(0, 400, size=(N_KPTS, 2)).astype(np.float32),
        "pred_cam_t": np.array([0.1, -0.2, 4.5], dtype=np.float32),
        "pred_vertices": rng.normal(size=(16, 3)).astype(np.float32),
    }


def _timeline(n_frames: int = 3, person_ids: tuple[int, ...] = (1, 2)) -> dict:
    timeline: dict[int, list[dict]] = {}
    for frame_idx in range(n_frames):
        outputs = [_fake_output(frame_idx * 10 + pid) for pid in person_ids]
        timeline[frame_idx] = _instances_from_outputs(
            list(person_ids), outputs, frame_idx=frame_idx
        )
    return timeline


# --------------------------------------------------------------------------- #
# keypoint metadata
# --------------------------------------------------------------------------- #
def test_mhr70_has_70_unique_names():
    assert len(MHR70_NAMES) == 70
    assert len(set(MHR70_NAMES)) == 70


def test_keypoint_names_pads_beyond_mhr70():
    names = keypoint_names(75)
    assert names[:70] == list(MHR70_NAMES)
    assert names[70:] == ["kpt_070", "kpt_071", "kpt_072", "kpt_073", "kpt_074"]
    assert keypoint_names(5) == list(MHR70_NAMES[:5])


def test_skeleton_edges_resolve_to_valid_indices():
    names = keypoint_names(N_KPTS)
    edges = skeleton_edges(names)
    assert edges, "expected a non-empty body skeleton"
    for a, b in edges:
        assert 0 <= a < N_KPTS
        assert 0 <= b < N_KPTS
        assert a != b
    # A truncated name list must silently drop the unresolvable edges.
    assert skeleton_edges(["nose", "neck"]) == [(1, 0)]


def test_side_color_helper_maps_prefixes_to_palette():
    assert _side_color_bgr("left-knee") == _rgb_to_bgr(COLOR_LEFT_RGB)
    assert _side_color_bgr("right-knee") == _rgb_to_bgr(COLOR_RIGHT_RGB)
    assert _side_color_bgr("neck") == _rgb_to_bgr(COLOR_CENTER_RGB)
    # Case- and separator-insensitive (defensive against upstream renames).
    assert _side_color_bgr("LEFT-elbow") == _rgb_to_bgr(COLOR_LEFT_RGB)
    assert _side_color_bgr("right_elbow") == _rgb_to_bgr(COLOR_RIGHT_RGB)


def test_draw_pose_overlay_colors_left_and_right_differently():
    """Regression test: the live overlay video used one solid color per
    person for the whole skeleton (monochromatic, no left/right cue). It
    must now match the left=green/right=orange/center=blue palette shared
    with sam3dinov3_visualize.
    """
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    names = ["nose", "left-shoulder", "right-shoulder"]
    edges = [(0, 1), (0, 2)]
    instance = {
        "person_id": 0,
        "keypoints_2d": np.array([[40.0, 5.0], [20.0, 30.0], [60.0, 30.0]], dtype=np.float32),
        "cam_t": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "bbox": np.array([0.0, 0.0, 80.0, 60.0], dtype=np.float32),
    }
    out = _draw_pose_overlay(image, [instance], edges, names, draw_ids=False)
    left_point = tuple(int(v) for v in out[30, 20])
    right_point = tuple(int(v) for v in out[30, 60])
    assert left_point == _rgb_to_bgr(COLOR_LEFT_RGB)
    assert right_point == _rgb_to_bgr(COLOR_RIGHT_RGB)
    assert left_point != right_point


# --------------------------------------------------------------------------- #
# SAM -> SAM 3D Body bridge
# --------------------------------------------------------------------------- #
def test_frame_batch_builds_boxes_and_binary_masks():
    guidance = [
        (_track(3, 100, 50, 80, 200), _contour(3, 100, 50, 80, 200)),
        (_track(7, 300, 60, 90, 210), None),
    ]
    obj_ids, boxes, masks = _frame_batch_from_guidance(
        guidance,
        frame_width=W,
        frame_height=H,
        bbox_padding=0.1,
        contour_margin=4,
        use_mask=True,
    )
    assert obj_ids == [3, 7]
    assert boxes is not None and boxes.shape == (2, 4)
    assert boxes.dtype == np.float32
    # Boxes stay inside the frame and keep x1<x2, y1<y2.
    assert np.all(boxes[:, 0] < boxes[:, 2])
    assert np.all(boxes[:, 1] < boxes[:, 3])
    assert boxes.min() >= 0.0
    assert boxes[:, 2].max() <= W
    assert boxes[:, 3].max() <= H

    assert masks is not None and masks.shape == (2, H, W)
    assert masks.dtype == np.uint8
    # SAM 3D Body expects 0/1 masks, never 0/255.
    assert set(np.unique(masks)).issubset({0, 1})
    assert masks[0].any()


def test_frame_batch_without_masks_and_when_empty():
    guidance = [(_track(1, 10, 10, 50, 100), None)]
    obj_ids, boxes, masks = _frame_batch_from_guidance(
        guidance,
        frame_width=W,
        frame_height=H,
        bbox_padding=0.0,
        contour_margin=0,
        use_mask=False,
    )
    assert obj_ids == [1]
    assert boxes is not None
    assert masks is None

    obj_ids, boxes, masks = _frame_batch_from_guidance(
        [],
        frame_width=W,
        frame_height=H,
        bbox_padding=0.0,
        contour_margin=0,
        use_mask=True,
    )
    assert obj_ids == []
    assert boxes is None and masks is None


def test_instances_keep_sam_identity_and_add_camera_frame_coords():
    outputs = [_fake_output(1), _fake_output(2)]
    instances = _instances_from_outputs([5, 9], outputs, frame_idx=0)
    assert [i["person_id"] for i in instances] == [5, 9]
    assert [i["sam_obj_id"] for i in instances] == [5, 9]
    for inst, out in zip(instances, outputs, strict=True):
        expected = np.asarray(out["pred_keypoints_3d"]) + np.asarray(out["pred_cam_t"])[None, :]
        np.testing.assert_allclose(inst["keypoints_3d_cam"], expected, rtol=1e-6)
        assert inst["focal_length"] == pytest.approx(1234.5)


def test_instances_pair_common_prefix_when_counts_mismatch(capsys):
    instances = _instances_from_outputs([1, 2, 3], [_fake_output(1)], frame_idx=4)
    assert len(instances) == 1
    assert instances[0]["person_id"] == 1
    assert "WARNING" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #
def test_long_keypoint_csvs_shape_and_units(tmp_path: Path):
    timeline = _timeline(n_frames=2, person_ids=(1, 4))
    names = keypoint_names(N_KPTS)
    path3d, path2d = write_long_keypoints_csvs(tmp_path, "clip", timeline, names)

    with path3d.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2 * 2 * N_KPTS
    assert rows[0]["kpt_name"] == "nose"
    assert set(rows[0]) == {
        "frame",
        "person_id",
        "kpt_idx",
        "kpt_name",
        "x_m",
        "y_m",
        "z_m",
        "xcam_m",
        "ycam_m",
        "zcam_m",
    }
    # Camera-frame z is the root-relative z plus cam_t z (4.5 in the fixture).
    first = timeline[0][0]
    assert float(rows[0]["zcam_m"]) == pytest.approx(
        float(first["keypoints_3d"][0][2]) + 4.5, abs=1e-5
    )

    with path2d.open(newline="", encoding="utf-8") as fh:
        rows2d = list(csv.DictReader(fh))
    assert len(rows2d) == 2 * 2 * N_KPTS
    assert set(rows2d[0]) == {"frame", "person_id", "kpt_idx", "kpt_name", "x_px", "y_px"}


def test_camera_csv_one_row_per_person_frame(tmp_path: Path):
    timeline = _timeline(n_frames=3, person_ids=(2,))
    path = write_camera_csv(tmp_path, "clip", timeline)
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    assert [int(r["frame"]) for r in rows] == [0, 1, 2]
    assert all(int(r["person_id"]) == 2 for r in rows)
    assert float(rows[0]["cam_t_z_m"]) == pytest.approx(4.5)
    assert float(rows[0]["focal_length_px"]) == pytest.approx(1234.5)


def test_wide_person_csvs_follow_vaila_conventions(tmp_path: Path):
    timeline = _timeline(n_frames=2, person_ids=(1, 3))
    names = keypoint_names(N_KPTS)
    written = write_wide_person_csvs(tmp_path, "clip", timeline, names, n_frames=4)

    # Three files per identity.
    assert len(written) == 6
    for person_id in (1, 3):
        named = tmp_path / f"clip_id_{person_id:02d}_mhr70_3d.csv"
        rec3d = tmp_path / f"clip_id_{person_id:02d}_mhr70_rec3d.csv"
        markers = tmp_path / f"clip_id_{person_id:02d}_markers.csv"
        assert named.is_file() and rec3d.is_file() and markers.is_file()

        named_lines = named.read_text(encoding="utf-8").strip().splitlines()
        header = named_lines[0].split(",")
        assert header[0] == "frame"
        assert header[1:4] == ["nose_x", "nose_y", "nose_z"]
        # Hyphens are not valid in vailá column names.
        assert "-" not in named_lines[0]
        assert len(header) == 1 + 3 * N_KPTS
        # n_frames=4 but only 2 frames have data: the tail is padded with blanks.
        assert len(named_lines) == 5
        assert named_lines[-1].split(",")[1] == ""

        rec_header = rec3d.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert rec_header[:4] == ["frame", "p1_x", "p1_y", "p1_z"]
        assert rec_header[-1] == f"p{N_KPTS}_z"

        mk_header = markers.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert mk_header[:3] == ["frame", "p1_x", "p1_y"]
        assert len(mk_header) == 1 + 2 * N_KPTS


def test_wide_csv_uses_camera_frame_coordinates(tmp_path: Path):
    timeline = _timeline(n_frames=1, person_ids=(1,))
    names = keypoint_names(N_KPTS)
    write_wide_person_csvs(tmp_path, "clip", timeline, names, n_frames=1)
    row = (tmp_path / "clip_id_01_mhr70_3d.csv").read_text(encoding="utf-8").splitlines()[1]
    cells = row.split(",")
    expected = timeline[0][0]["keypoints_3d_cam"][0]
    assert float(cells[1]) == pytest.approx(float(expected[0]), abs=1e-5)
    assert float(cells[3]) == pytest.approx(float(expected[2]), abs=1e-5)


def test_predictions_json_roundtrip(tmp_path: Path):
    timeline = _timeline(n_frames=2, person_ids=(6,))
    names = keypoint_names(N_KPTS)
    path = write_predictions_json(
        tmp_path, "clip", timeline, names, meta={"video": "clip.mp4", "fps": 30.0}
    )
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["schema"] == "vaila_sam3dinov3_v1"
    assert payload["keypoint_names"][:1] == ["nose"]
    assert payload["fps"] == 30.0
    assert len(payload["frames"]) == 2
    inst = payload["frames"][0]["instances"][0]
    assert inst["person_id"] == 6 and inst["sam_obj_id"] == 6
    assert len(inst["keypoints_3d_m"]) == N_KPTS
    assert len(inst["keypoints_2d_px"]) == N_KPTS
    # Meshes are intentionally excluded from the JSON (they go to meshes/*.npz).
    assert "vertices" not in inst


def test_collect_person_ids_sorted_and_unique():
    timeline = _timeline(n_frames=2, person_ids=(9, 2))
    assert _collect_person_ids(timeline) == [2, 9]


def test_readme_documents_units_and_scale_caveat(tmp_path: Path):
    path = _write_readme(
        tmp_path,
        video_path=Path("clip.mp4"),
        sam_dir=Path("/tmp/sam"),
        weights_dir=Path("/tmp/w"),
        inference_type="full",
        stride=1,
        use_mask=True,
        bbox_padding=0.12,
        contour_margin=8,
        focal_px=None,
        n_keypoints=N_KPTS,
    )
    text = path.read_text(encoding="utf-8")
    assert "identity_authority=SAM3 obj_id" in text
    assert "Scale caveat" in text
    assert "--focal-px" in text
    assert "xcam_m,ycam_m,zcam_m" in text


# --------------------------------------------------------------------------- #
# CLI plumbing
# --------------------------------------------------------------------------- #
def test_resolve_assets_error_is_actionable(tmp_path: Path):
    with pytest.raises(FileNotFoundError) as exc:
        resolve_sam3d_assets(tmp_path)
    message = str(exc.value)
    assert "model.ckpt" in message
    assert "mhr_model.pt" in message
    assert "setup_fifa_sam3d.sh" in message
    assert "facebook/sam-3d-body-dinov3" in message


def test_checkout_candidates_honour_env_override(tmp_path: Path, monkeypatch):
    from vaila.sam3dinov3 import _sam3d_checkout_candidates

    monkeypatch.setenv("VAILA_SAM3D_BODY_DIR", str(tmp_path / "elsewhere"))
    candidates = _sam3d_checkout_candidates()
    # The override must be searched first.
    assert candidates[0] == tmp_path / "elsewhere"
    # Repo-root conventions must still be covered, both spellings.
    names = {c.name for c in candidates}
    assert {"sam_3d_body", "sam-3d-body"} <= names
    # No duplicates.
    assert len(candidates) == len(set(candidates))


def test_ensure_importable_finds_checkout_root(tmp_path: Path, monkeypatch):
    """Upstream ships no packaging metadata, so the checkout ROOT goes on sys.path."""
    import sys

    from vaila.sam3dinov3 import ensure_sam3d_importable

    checkout = tmp_path / "sam_3d_body"
    pkg = checkout / "sam_3d_body"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("VALUE = 42\n", encoding="utf-8")

    monkeypatch.setenv("VAILA_SAM3D_BODY_DIR", str(checkout))
    monkeypatch.delitem(sys.modules, "sam_3d_body", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))

    found = ensure_sam3d_importable()
    assert found == checkout.resolve()
    # It must add the checkout root, NOT the inner package dir.
    assert str(checkout.resolve()) in sys.path
    assert str(pkg.resolve()) not in sys.path


def test_ensure_importable_returns_none_without_checkout(tmp_path: Path, monkeypatch):
    import sys

    from vaila.sam3dinov3 import ensure_sam3d_importable

    monkeypatch.setenv("VAILA_SAM3D_BODY_DIR", str(tmp_path / "nope"))
    monkeypatch.delitem(sys.modules, "sam_3d_body", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.chdir(tmp_path)
    # Also neutralise the repo-root candidates for this test.
    monkeypatch.setattr("vaila.sam3dinov3._module_dir", lambda: tmp_path / "fake_pkg", raising=True)
    assert ensure_sam3d_importable() is None


def test_worker_command_round_trips_flags(tmp_path: Path):
    from vaila.sam3dinov3 import _build_parser

    args = _build_parser().parse_args(
        [
            "-i",
            "clip.mp4",
            "-o",
            str(tmp_path),
            "--save-mesh",
            "--no-mask",
            "--focal-px",
            "1400",
            "--inference-type",
            "body",
        ]
    )
    cmd = build_worker_command(
        Path("clip.mp4"),
        tmp_path,
        tmp_path / "clip",
        args,
        sam_dir=tmp_path / "sam",
    )
    assert "--save-mesh" in cmd
    assert "--no-mask" in cmd
    assert cmd[cmd.index("--focal-px") + 1] == "1400.0"
    assert cmd[cmd.index("--inference-type") + 1] == "body"
    assert cmd[cmd.index("--worker-sam-dir") + 1] == str(tmp_path / "sam")
    # The prompt frame must survive into the worker.
    assert cmd[cmd.index("--sam-frame") + 1] == "0"
    # The worker must never re-enter the GUI branch.
    assert "--worker-output-dir" in cmd

    # Flags left at their defaults must not appear.
    plain = _build_parser().parse_args(["-i", "clip.mp4", "-o", str(tmp_path)])
    plain_cmd = build_worker_command(
        Path("clip.mp4"), tmp_path, tmp_path / "clip", plain, sam_dir=None
    )
    assert "--save-mesh" not in plain_cmd
    assert "--focal-px" not in plain_cmd
    assert "--worker-sam-dir" not in plain_cmd


def test_parser_rejects_invalid_values(tmp_path: Path):
    from vaila.sam3dinov3 import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--inference-type", "torso"])


def test_gui_cli_mirror_is_reproducible(tmp_path: Path):
    from vaila.sam3dinov3 import Sam3dGuiSettings, _format_gui_cli

    settings = Sam3dGuiSettings(
        input_path=tmp_path / "clip.mp4",
        output_parent=tmp_path / "out",
        resume=None,
        sam_results=None,
        prompt="athlete",
        device=0,
        stride=2,
        bbox_padding=0.12,
        contour_margin=8,
        max_persons=4,
        inference_type="full",
        use_mask=True,
        save_overlay=True,
        save_mesh=True,
        focal_px=1400.0,
        weights_dir=None,
    )
    cmd = _format_gui_cli(settings)
    assert cmd[:5] == ["uv", "run", "python", "-u", "vaila/sam3dinov3.py"]
    assert cmd[cmd.index("-t") + 1] == "athlete"
    assert "--save-mesh" in cmd
    assert "--no-mask" not in cmd  # use_mask=True is the default
    assert cmd[cmd.index("--focal-px") + 1] == "1400.0"

    # The printed CLI must be parseable by this module's own parser.
    from vaila.sam3dinov3 import _build_parser

    parsed = _build_parser().parse_args(cmd[5:])
    assert parsed.text == "athlete"
    assert parsed.save_mesh is True
    assert parsed.stride == 2


def test_worker_namespace_has_every_attribute_the_pipeline_reads(tmp_path: Path):
    """Guards against a flag being read in the hot loop but missing from the parser."""
    from vaila.sam3dinov3 import _build_parser

    args = _build_parser().parse_args(["-i", "clip.mp4", "-o", str(tmp_path)])
    for attr in (
        "weights_dir",
        "fov_estimator",
        "focal_px",
        "stride",
        "min_sam_score",
        "min_sam_area",
        "max_persons",
        "bbox_padding",
        "contour_margin",
        "no_mask",
        "inference_type",
        "verbose_model",
        "save_mesh",
        "no_overlay",
        "no_draw_id",
        "worker_sam_dir",
        "sam_results",
        "text",
        "sam_frame",
        "sam_checkpoint",
        "sam_max_frames",
        "sam_max_input_long_edge",
        "keep_sam_masks",
    ):
        assert hasattr(args, attr), f"parser is missing --{attr.replace('_', '-')}"
    assert isinstance(args, argparse.Namespace)
