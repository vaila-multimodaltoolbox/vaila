"""Tests for the SAM3-guided Sapiens2 pipeline.

Update Date: 31 July 2026
Version: 0.3.86
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vaila import sam3sapiens2 as combo
from vaila import vaila_sapiens as vs


def test_find_videos_accepts_folder_batch(tmp_path: Path) -> None:
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "b.MOV").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("skip", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "c.mp4").write_bytes(b"")
    (tmp_path / "a_sam_overlay.mp4").write_bytes(b"")

    found = combo._find_videos(tmp_path)
    assert [p.name for p in found] == ["a.mp4", "b.MOV"]


def test_find_videos_accepts_single_file(tmp_path: Path) -> None:
    clip = tmp_path / "solo.mkv"
    clip.write_bytes(b"")
    assert combo._find_videos(clip) == [clip.resolve()]


def _write_sam_fixture(root: Path, *, frames: int = 2, obj_id: int = 7) -> Path:
    root.mkdir(parents=True)
    with (root / "sam_tracks.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "frame",
                "obj_id",
                "x_px",
                "y_px",
                "w_px",
                "h_px",
                "score",
                "area_px",
                "cx_px",
                "cy_px",
            ]
        )
        for frame in range(frames):
            writer.writerow([frame, obj_id, 10 + frame * 10, 20, 20, 40, 0.9, 600, 20, 40])
    payload = {
        "schema": "vaila_sam_contours_v1",
        "video": "clip.mp4",
        "width": 100,
        "height": 80,
        "fps": 30.0,
        "n_frames": frames,
        "frames": [
            {
                "frame": frame,
                "objects": [
                    {
                        "obj_id": obj_id,
                        "score": 0.9,
                        "polygons": [
                            [
                                [12 + frame * 10, 22],
                                [28 + frame * 10, 22],
                                [28 + frame * 10, 58],
                                [12 + frame * 10, 58],
                            ]
                        ],
                    }
                ],
            }
            for frame in range(frames)
        ],
    }
    (root / "sam_contours.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_load_sam_guidance_tracks_contours_and_metadata(tmp_path: Path) -> None:
    sam_dir = _write_sam_fixture(tmp_path / "sam")
    guidance = combo.load_sam_guidance(sam_dir)

    assert guidance.width == 100
    assert guidance.height == 80
    assert guidance.n_frames == 2
    assert guidance.tracks_by_frame[0][0]["obj_id"] == 7
    assert guidance.contours_by_frame[1][7]["polygons"]


def test_load_sam_guidance_rejects_missing_tracks(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="sam_tracks.csv"):
        combo.load_sam_guidance(tmp_path)


def test_resolve_sam_results_batch_parent(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    batch = tmp_path / "processed_sam_123"
    expected = _write_sam_fixture(batch / video.stem)
    assert combo.resolve_sam_results_dir(batch, video) == expected.resolve()


def test_pose_bbox_uses_contour_and_padding() -> None:
    track = {"x_px": 10.0, "y_px": 20.0, "w_px": 30.0, "h_px": 40.0}
    contour = {
        "polygons": [[[15, 25], [35, 25], [35, 55], [15, 55]]],
    }
    box = combo._pose_bbox_from_sam(
        track,
        contour,
        frame_width=100,
        frame_height=80,
        padding_fraction=0.1,
    )
    assert box.tolist() == pytest.approx([12.9, 21.9, 38.1, 59.1], abs=0.11)


def test_contour_mask_and_score_attenuation() -> None:
    track = {"x_px": 2.0, "y_px": 2.0, "w_px": 8.0, "h_px": 8.0}
    contour = {"polygons": [[[2, 2], [9, 2], [9, 9], [2, 9]]]}
    mask = combo._contour_mask((16, 16, 3), track, contour, margin_px=0)
    scores, inside = combo._attenuate_scores_outside_contour(
        np.asarray([[5.0, 5.0], [14.0, 14.0]], dtype=np.float32),
        np.asarray([0.8, 0.8], dtype=np.float32),
        mask,
        outside_factor=0.25,
    )
    assert inside == [True, False]
    assert scores.tolist() == pytest.approx([0.8, 0.2])


def test_expand_pose_keeps_sam_identity_and_updates_bbox() -> None:
    guidance = combo.SamGuidance(
        sam_dir=Path("/sam"),
        tracks_by_frame={
            0: [
                {
                    "obj_id": 5,
                    "x_px": 0.0,
                    "y_px": 0.0,
                    "w_px": 10.0,
                    "h_px": 10.0,
                    "score": 0.9,
                    "area_px": 100,
                }
            ],
            1: [
                {
                    "obj_id": 5,
                    "x_px": 10.0,
                    "y_px": 0.0,
                    "w_px": 10.0,
                    "h_px": 10.0,
                    "score": 0.8,
                    "area_px": 100,
                }
            ],
        },
        contours_by_frame={},
        width=40,
        height=20,
        fps=30.0,
        n_frames=2,
        contour_path=None,
    )
    inferred = {
        0: [
            {
                "stable_id": 5,
                "sam_obj_id": 5,
                "bbox": [0.0, 0.0, 10.0, 10.0],
                "sam_bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
                "keypoints": [[5.0, 5.0]],
                "keypoint_scores": [0.9],
            }
        ]
    }
    timeline = combo._expand_pose_with_sam_guidance(
        inferred,
        guidance,
        n_frames=2,
        frame_width=40,
        frame_height=20,
        bbox_padding=0.0,
        min_score=0.0,
        min_area=1,
        max_persons=4,
    )
    moved = timeline[1][0]
    assert moved["stable_id"] == moved["sam_obj_id"] == 5
    assert moved["bbox"] == pytest.approx([10.0, 0.0, 20.0, 10.0])
    assert moved["keypoints"][0] == pytest.approx([15.0, 5.0])


def test_build_sam_command_exports_guidance_without_overlay(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    command = combo.build_sam_command(
        video,
        tmp_path / "out" / "sam3",
        prompt="person",
        prompt_frame=0,
        checkpoint=None,
        max_frames=256,
        max_input_long_edge=1280,
        keep_masks=False,
    )
    assert "--video-output-dir" in command
    assert "--save-contours" in command
    assert "--save-tracks-csv" in command
    assert "--no-overlay" in command
    assert "--no-png" in command
    assert command[command.index("--max-frames") + 1] == "256"


def test_pose_session_without_detector_rejects_detector_entrypoint() -> None:
    session = object.__new__(vs.PoseInferenceSession)
    session.use_detector = False
    with pytest.raises(RuntimeError, match="process_frame_with_bboxes"):
        session.process_frame(np.zeros((8, 8, 3), dtype=np.uint8))


def test_prepare_gui_root_maps_tiny_owner_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    root = SimpleNamespace(
        deiconify=lambda: calls.append("deiconify"),
        geometry=lambda value: calls.append(("geometry", value)),
        update_idletasks=lambda: calls.append("update_idletasks"),
    )
    monkeypatch.setattr(combo.sys, "platform", "linux")
    combo._prepare_gui_root(root, owns_root=True)
    assert calls == ["deiconify", ("geometry", "1x1+100+100"), "update_idletasks"]


def test_prepare_gui_root_leaves_embedded_owner_alone() -> None:
    root = SimpleNamespace(
        deiconify=lambda: pytest.fail("embedded root must not be remapped"),
        geometry=lambda _value: pytest.fail("embedded root must not be resized"),
        update_idletasks=lambda: pytest.fail("embedded root must not be updated"),
    )
    combo._prepare_gui_root(root, owns_root=False)
