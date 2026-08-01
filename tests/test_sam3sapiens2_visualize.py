"""Tests for the selected-ID SAM3+Sapiens2 rerenderer.

Update Date: 01 August 2026
Version: 0.3.89
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from vaila import sam3sapiens2_visualize as viz


def _fixture_run(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "processed_sam3sapiens2_20260731" / "clip"
    sam = run / "sam3"
    sam.mkdir(parents=True)
    predictions = {
        "schema": "vaila_sam3sapiens2_v1",
        "video": "clip.mp4",
        "image_size": [60, 80],
        "fps": 10.0,
        "frames": [
            {
                "frame_index": frame,
                "instances": [
                    {
                        "stable_id": 2,
                        "sam_obj_id": 2,
                        "sam_bbox_xyxy": [10 + frame, 10, 30 + frame, 40],
                        "keypoints": [[12 + frame, 12], [14 + frame, 14]] + [[0, 0]] * 19,
                        "keypoint_scores": [0.9, 0.9] + [0.0] * 19,
                    },
                    {"stable_id": 9, "sam_obj_id": 9, "keypoints": [[50, 50]]},
                ],
            }
            for frame in range(2)
        ],
    }
    (run / "clip_sam3sapiens2_predictions.json").write_text(
        json.dumps(predictions), encoding="utf-8"
    )
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
        writer.writerow(["frame", "obj_id", "x_px", "y_px", "w_px", "h_px"])
        writer.writerow([0, 2, 10, 10, 20, 30])
        writer.writerow([0, 9, 40, 40, 10, 10])
    with (run / "sapiens_tracks.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "stable_id", "x1", "y1", "x2", "y2"])
        writer.writerow([0, 2, 10, 10, 30, 40])
        writer.writerow([0, 9, 40, 40, 50, 50])
    (run / "sapiens_id_map.csv").write_text(
        "pN,stable_id,n_frames,first_frame,last_frame\n1,2,2,0,1\n2,9,2,0,1\n",
        encoding="utf-8",
    )
    (run / "sapiens_points.csv").write_text(
        "frame,p1_x,p1_y,p2_x,p2_y\n0,12,12,50,50\n1,13,12,51,50\n",
        encoding="utf-8",
    )
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


def test_discovers_recorded_source_video_and_validates_alignment(tmp_path: Path) -> None:
    run, video = _fixture_run(tmp_path)
    (run / "sam3sapiens2_summary.json").write_text(
        json.dumps({"video": str(video)}), encoding="utf-8"
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
    assert first.name == "clip_sam3sapiens2_visualized_id_04"
    first.mkdir()
    assert viz._unique_gui_output_dir(tmp_path, video, 4).name.endswith("_2")


def test_selected_artifacts_filter_json_and_csv(tmp_path: Path) -> None:
    run, _video = _fixture_run(tmp_path)
    output = tmp_path / "selected"
    result = viz.visualize_selected_id(run, tmp_path / "clip.mp4", 2, output)
    assert result["selected_id"] == 2
    rows = list(csv.DictReader((output / "sapiens_tracks.csv").open(encoding="utf-8")))
    assert rows and {int(row["stable_id"]) for row in rows} == {2}
    selected = json.loads((output / "sam_contours.json").read_text(encoding="utf-8"))
    assert selected["object_ids"] == [2]
    assert all(obj["obj_id"] == 2 for frame in selected["frames"] for obj in frame["objects"])
    assert (output / "source_artifacts" / "sam3" / "sam_contours.json").exists()
    assert (output / "sam3sapiens2_selected_id_manifest.json").exists()


def test_render_selected_video_draws_only_selected_instance(tmp_path: Path) -> None:
    run, video = _fixture_run(tmp_path)
    payload = viz.load_predictions(run)
    output = tmp_path / "render.mp4"
    rendered, frames, drawn = viz.render_selected_video(
        video,
        output,
        viz._records_by_frame(payload, 2),
        viz._contours_by_frame(run, 2),
        selected_id=2,
    )
    assert rendered.exists()
    assert frames == drawn == 2
