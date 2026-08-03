"""Tests for the selected-ID SAM3+DINOv3 3D rerenderer.

Update Date: 03 August 2026
Version: 0.3.98
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from vaila import sam3dinov3
from vaila import sam3dinov3_visualize as viz

NAMES = ["nose", "left-eye", "right-eye", "neck", "left-shoulder", "right-shoulder"]


def _fixture_run(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "processed_sam3dinov3_20260802" / "clip"
    sam = run / "sam3"
    sam.mkdir(parents=True)

    frames = []
    for frame in range(2):
        instances = []
        for pid in (2, 9):
            kp2d = [[10 + pid + frame, 10 + i] for i in range(len(NAMES))]
            kp3d = [[0.01 * i, 0.02 * i, 1.5] for i in range(len(NAMES))]
            instances.append(
                {
                    "person_id": pid,
                    "sam_obj_id": pid,
                    "bbox_xyxy": [10.0 + pid, 10.0, 30.0 + pid, 40.0],
                    "focal_length_px": 1400.0,
                    "cam_t_m": [0.0, 0.0, 1.5],
                    "keypoints_3d_m": kp3d,
                    "keypoints_3d_cam_m": kp3d,
                    "keypoints_2d_px": kp2d,
                }
            )
        frames.append({"frame": frame, "instances": instances})

    payload = {
        "schema": "vaila_sam3dinov3_v1",
        "keypoint_names": NAMES,
        "video": "clip.mp4",
        "sam_results": str(sam),
        "width": 80,
        "height": 60,
        "fps": 10.0,
        "n_frames": 2,
        "stride": 1,
        "inference_type": "full",
        "mask_conditioned": True,
        "focal_px": None,
        "frames": frames,
    }
    pred_path = run / "clip_sam3dinov3_predictions.json.gz"
    with gzip.open(pred_path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)

    contours = {
        "schema": "vaila_sam_contours_v1",
        "object_ids": [2, 9],
        "frames": [
            {
                "frame": frame,
                "objects": [
                    {"obj_id": 2, "polygons": [[[10, 10], [30, 10], [30, 40], [10, 40]]]},
                    {"obj_id": 9, "polygons": []},
                ],
            }
            for frame in range(2)
        ],
    }
    (sam / "sam_contours.json").write_text(json.dumps(contours), encoding="utf-8")
    with (sam / "sam_tracks.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "obj_id", "x1", "y1", "x2", "y2"])
        writer.writerow([0, 2, 10, 10, 30, 40])
        writer.writerow([0, 9, 40, 40, 50, 50])

    for pid in (2, 9):
        (run / f"clip_id_{pid:02d}_mhr70_3d.csv").write_text(
            "frame,nose_x,nose_y,nose_z\n0,0,0,1.5\n1,0,0,1.5\n", encoding="utf-8"
        )
        (run / f"clip_id_{pid:02d}_mhr70_rec3d.csv").write_text(
            "frame,p1_x,p1_y,p1_z\n0,0,0,1.5\n1,0,0,1.5\n", encoding="utf-8"
        )
        (run / f"clip_id_{pid:02d}_markers.csv").write_text(
            "frame,p1_x,p1_y\n0,10,10\n1,11,10\n", encoding="utf-8"
        )

    with (run / "clip_sam3dinov3_keypoints3d.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
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
            ]
        )
        for frame in range(2):
            for pid in (2, 9):
                writer.writerow([frame, pid, 0, "nose", 0.0, 0.0, 1.5, 0.0, 0.0, 1.5])

    mesh_dir = run / "meshes"
    mesh_dir.mkdir()
    rng = np.random.default_rng(0)
    for frame in range(2):
        np.savez_compressed(
            mesh_dir / f"frame_{frame:06d}.npz",
            obj_ids=np.asarray([2, 9], dtype=np.int32),
            vertices=rng.random((2, 4, 3)).astype(np.float32),
            cam_t=rng.random((2, 3)).astype(np.float32),
        )
    np.save(run / "mesh_faces.npy", np.arange(6).reshape(2, 3))

    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (80, 60))
    for _ in range(2):
        writer.write(np.zeros((60, 80, 3), dtype=np.uint8))
    writer.release()
    return run, video


def test_discover_and_resolve_selected_id(tmp_path: Path) -> None:
    run, video = _fixture_run(tmp_path)
    assert viz.resolve_run_dir(run.parent, video) == run.resolve()
    assert viz.discover_ids(run) == [2, 9]


def test_prompt_selected_id_reprompts_until_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter(["", "abc", "99", "2"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert viz.prompt_selected_id([2, 9]) == 2


def test_cli_prompts_for_id_when_flag_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, video = _fixture_run(tmp_path)
    output = tmp_path / "out_id_prompt"
    monkeypatch.setattr(viz, "prompt_selected_id", lambda _ids: 2)
    code = viz.main(
        [
            "--sam3d-results",
            str(run),
            "--video",
            str(video),
            "--output",
            str(output),
            "--dry-run",
        ]
    )
    assert code == 0
    assert output.exists() is False


def test_discovers_recorded_source_video_and_validates_alignment(tmp_path: Path) -> None:
    run, video = _fixture_run(tmp_path)
    (run / "sam3dinov3_summary.json").write_text(
        json.dumps({"video": str(video), "person_ids": [2, 9]}), encoding="utf-8"
    )
    payload = viz.load_predictions(run)
    assert viz.discover_source_video(run, payload) == video.resolve()
    assert viz.validate_source_video(video, payload)["frames"] == 2

    wrong = tmp_path / "wrong.mp4"
    writer = cv2.VideoWriter(str(wrong), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (80, 60))
    for _ in range(3):
        writer.write(np.zeros((60, 80, 3), dtype=np.uint8))
    writer.release()
    try:
        viz.validate_source_video(wrong, payload)
    except ValueError as exc:
        assert "frames=3, expected 2" in str(exc)
    else:
        raise AssertionError("A frame-count mismatch must be rejected")


def test_gui_output_is_new_child_of_selected_parent(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    first = viz._unique_gui_output_dir(tmp_path, video, 4)
    assert first.parent == tmp_path.resolve()
    assert first.name == "clip_sam3dinov3_visualized_id_04"
    first.mkdir()
    assert viz._unique_gui_output_dir(tmp_path, video, 4).name.endswith("_2")


def test_selected_artifacts_filter_json_csv_and_mesh(tmp_path: Path) -> None:
    run, video = _fixture_run(tmp_path)
    output = tmp_path / "selected"
    result = viz.visualize_selected_id(run, video, 2, output)
    assert result["selected_id"] == 2

    rows = list(csv.DictReader((output / "sam_tracks.csv").open(encoding="utf-8")))
    assert rows and {int(row["obj_id"]) for row in rows} == {2}

    kp3d_rows = list(
        csv.DictReader((output / "clip_sam3dinov3_keypoints3d.csv").open(encoding="utf-8"))
    )
    assert kp3d_rows and {int(row["person_id"]) for row in kp3d_rows} == {2}

    selected = json.loads((output / "sam_contours.json").read_text(encoding="utf-8"))
    assert selected["object_ids"] == [2]
    assert all(obj["obj_id"] == 2 for frame in selected["frames"] for obj in frame["objects"])

    with gzip.open(output / "clip_sam3dinov3_predictions.json.gz", "rt", encoding="utf-8") as fh:
        pred = json.load(fh)
    ids = {inst["person_id"] for fr in pred["frames"] for inst in fr["instances"]}
    assert ids == {2}

    assert (output / "clip_id_02_mhr70_3d.csv").exists()
    assert not (output / "clip_id_09_mhr70_3d.csv").exists()

    with np.load(output / "meshes" / "frame_000000.npz") as mesh:
        assert list(mesh["obj_ids"]) == [2]
        assert mesh["vertices"].shape[0] == 1
    assert (output / "mesh_faces.npy").exists()

    assert (output / "source_artifacts" / "sam3" / "sam_contours.json").exists()
    assert (output / "sam3dinov3_selected_id_manifest.json").exists()


def test_render_selected_video_draws_only_selected_instance(tmp_path: Path) -> None:
    run, video = _fixture_run(tmp_path)
    payload = viz.load_predictions(run)
    output = tmp_path / "render.mp4"
    edges = [(0, 3), (3, 4), (3, 5)]
    rendered, frames, drawn = viz.render_selected_video(
        video,
        output,
        viz._records_by_frame(payload, 2),
        viz._contours_by_frame(run, 2),
        edges,
        NAMES,
        selected_id=2,
    )
    assert rendered.exists()
    assert frames == drawn == 2


def test_side_color_helper_maps_prefixes_to_palette() -> None:
    assert viz._side_color_bgr("left-shoulder") == viz._rgb_to_bgr(sam3dinov3.COLOR_LEFT_RGB)
    assert viz._side_color_bgr("right-elbow") == viz._rgb_to_bgr(sam3dinov3.COLOR_RIGHT_RGB)
    assert viz._side_color_bgr("nose") == viz._rgb_to_bgr(viz.COLOR_CENTER_RGB)


def test_draw_mhr_skeleton_colors_left_and_right_differently() -> None:
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    names = ["nose", "left-shoulder", "right-shoulder"]
    instance = {
        "keypoints_2d_px": [[40.0, 5.0], [20.0, 30.0], [60.0, 30.0]],
    }
    out = viz._draw_mhr_skeleton(image, instance, [(0, 1), (0, 2)], names)
    left_point = out[30, 20]
    right_point = out[30, 60]
    assert tuple(int(v) for v in left_point) == viz._rgb_to_bgr(sam3dinov3.COLOR_LEFT_RGB)
    assert tuple(int(v) for v in right_point) == viz._rgb_to_bgr(sam3dinov3.COLOR_RIGHT_RGB)


def test_export_mesh_sequence_writes_obj_and_ply_with_translation(tmp_path: Path) -> None:
    run, _video = _fixture_run(tmp_path)
    faces = np.load(run / "mesh_faces.npy")

    obj_out = tmp_path / "meshes_obj"
    written_obj = viz.export_mesh_sequence(
        run / "meshes", run / "mesh_faces.npy", obj_out, person_id=2, fmt="obj"
    )
    assert len(written_obj) == 2
    text = written_obj[0].read_text(encoding="utf-8")
    n_v_lines = sum(1 for line in text.splitlines() if line.startswith("v "))
    n_f_lines = sum(1 for line in text.splitlines() if line.startswith("f "))
    assert n_v_lines == 4  # 4 vertices per fixture person
    assert n_f_lines == len(faces)

    with np.load(run / "meshes" / "frame_000000.npz") as mesh:
        idx = int(np.nonzero(np.asarray(mesh["obj_ids"]) == 2)[0][0])
        expected = np.asarray(mesh["vertices"][idx]) + np.asarray(mesh["cam_t"][idx])
    first_vertex_line = next(line for line in text.splitlines() if line.startswith("v "))
    got = [float(v) for v in first_vertex_line.split()[1:]]
    assert got == pytest.approx(expected[0].tolist(), abs=1e-5)

    ply_out = tmp_path / "meshes_ply"
    written_ply = viz.export_mesh_sequence(
        run / "meshes", run / "mesh_faces.npy", ply_out, person_id=2, fmt="ply"
    )
    assert len(written_ply) == 2
    assert written_ply[0].read_bytes().startswith(b"ply\n")


def test_visualize_selected_id_exports_mesh_when_requested(tmp_path: Path) -> None:
    run, video = _fixture_run(tmp_path)
    output = tmp_path / "selected_with_mesh"
    result = viz.visualize_selected_id(run, video, 2, output, export_mesh="obj")
    assert result["mesh_export_format"] == "obj"
    assert result["n_mesh_frames_exported"] == 2
    mesh_out = output / "meshes_obj"
    assert sorted(p.name for p in mesh_out.glob("*.obj")) == [
        "frame_000000.obj",
        "frame_000001.obj",
    ]
    readme = (output / "README_sam3dinov3_selected_id.txt").read_text(encoding="utf-8")
    assert "Stop Motion OBJ" in readme
