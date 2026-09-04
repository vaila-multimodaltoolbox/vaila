"""Tests for the SAM3-guided Sapiens2 pipeline.

Update Date: 24 August 2026
Version: 0.3.112
"""

from __future__ import annotations

import csv
import json
import sys
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


@pytest.mark.parametrize(
    "overlay_name",
    [
        "a_sam_overlay.mp4",
        "a_sapiens_overlay.mp4",
        "a_sam3sapiens2_overlay.mp4",
        "c1_cod_sam3sapiens2_id_04_overlay.mp4",  # sam3sapiens2_visualize.py's real output
        "a_sam3dinov3_overlay.mp4",
        "c1_cod_sam3dinov3_id_07_overlay.mp4",  # sam3dinov3_visualize.py's real output
        "a_sapiens2_3d_overlay.mp4",  # sapiens2_3d.py's own overlay
    ],
)
def test_is_derived_video_matches_every_known_overlay_suffix(
    tmp_path: Path, overlay_name: str
) -> None:
    """Regression for the 2026-08-07 bug: the old substring-only check matched
    '_sam3sapiens2_overlay' but not '_sam3sapiens2_id_04_overlay' -- the
    actual filename sam3sapiens2_visualize.py writes -- so a rendered
    overlay got queued by _find_videos() as if it were raw input."""
    path = tmp_path / overlay_name
    path.write_bytes(b"")
    assert combo._is_derived_video(path) is True
    assert combo._find_videos(tmp_path) == []


@pytest.mark.parametrize(
    "raw_name",
    ["c1_cod.mp4", "clip.mp4", "athlete_sam3sapiens2_run.mp4", "sapiens2_intro.mov"],
)
def test_is_derived_video_does_not_exclude_real_raw_video_names(
    tmp_path: Path, raw_name: str
) -> None:
    """Regression guard the other direction: the broadened filter must not
    start excluding legitimate raw video names that merely mention a
    pipeline name without ending in one of the known overlay suffixes."""
    path = tmp_path / raw_name
    path.write_bytes(b"")
    assert combo._is_derived_video(path) is False
    assert combo._find_videos(tmp_path) == [path.resolve()]


def test_is_derived_video_excludes_files_inside_a_processed_batch_dir(tmp_path: Path) -> None:
    batch_dir = tmp_path / "processed_sam3sapiens2_20260806_233956"
    batch_dir.mkdir()
    # A file that doesn't even match the suffix regex is still caught by the
    # parent-directory signal -- the safety net for a future overlay-writing
    # tool this suffix list hasn't been updated for yet.
    stray = batch_dir / "some_future_writer_output.mp4"
    stray.write_bytes(b"")
    assert combo._is_derived_video(stray) is True


def test_is_derived_video_excludes_files_inside_a_visualized_id_dir(tmp_path: Path) -> None:
    visualize_dir = tmp_path / "c1_cod_sam3sapiens2_visualized_id_04"
    visualize_dir.mkdir()
    stray = visualize_dir / "markers.mp4"
    stray.write_bytes(b"")
    assert combo._is_derived_video(stray) is True
    other_dir = tmp_path / "c2_cod_sam3dinov3_visualized_id_08"
    other_dir.mkdir()
    stray2 = other_dir / "whatever.mp4"
    stray2.write_bytes(b"")
    assert combo._is_derived_video(stray2) is True


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
    with (root / "sam_frames_meta.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "frame",
                f"box_x_{obj_id}",
                f"box_y_{obj_id}",
                f"box_w_{obj_id}",
                f"box_h_{obj_id}",
                f"prob_{obj_id}",
            ]
        )
        for frame in range(frames):
            writer.writerow([frame, 0.1, 0.2, 0.2, 0.5, 0.9])
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
    assert "--output-base" in command
    assert "--video-output-dir" not in command
    assert "--save-contours" in command
    assert "--save-tracks-csv" in command
    assert "--no-overlay" in command
    assert "--no-png" in command
    assert command[command.index("--max-frames") + 1] == "256"


def test_plan_video_processing_resume_skip_reuse_and_rerun(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(combo, "_video_frame_count", lambda _path: 2)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    done = tmp_path / "done"
    done.mkdir()
    (done / "sam3sapiens2_summary.json").write_text(
        json.dumps(
            {
                "video": str(video),
                "ok": True,
                "completed": True,
                "expected_frames": 2,
                "n_frames": 2,
            }
        ),
        encoding="utf-8",
    )
    action, sam_dir, summary = combo.plan_video_processing(
        video,
        done,
        resume=True,
        sam_results=None,
        single_video=True,
    )
    assert action == "skip"
    assert sam_dir is None
    assert summary is not None and summary["ok"] is True

    partial = tmp_path / "partial"
    local_sam = _write_sam_fixture(partial / "sam3")
    action, sam_dir, summary = combo.plan_video_processing(
        video,
        partial,
        resume=True,
        sam_results=None,
        single_video=True,
    )
    assert action == "reuse_sam"
    assert sam_dir == local_sam.resolve()
    assert summary is None

    failed = tmp_path / "failed"
    (failed / "sam3" / "_chunks").mkdir(parents=True)
    (failed / "FAILED_sam3sapiens2.txt").write_text("boom", encoding="utf-8")
    (failed / "sam3" / "FAILED_sam.txt").write_text("boom", encoding="utf-8")
    action, sam_dir, summary = combo.plan_video_processing(
        video,
        failed,
        resume=True,
        sam_results=None,
        single_video=True,
    )
    assert action == "run"
    assert sam_dir is None
    assert summary is None
    combo.prepare_sam_rerun_dir(failed)
    assert not (failed / "FAILED_sam3sapiens2.txt").exists()
    assert not (failed / "sam3" / "FAILED_sam.txt").exists()
    assert not (failed / "sam3" / "_chunks").exists()


def test_write_batch_input_marker_and_find_batch_input_path_roundtrip(tmp_path: Path) -> None:
    output_base = tmp_path / "processed_sam3sapiens2_20260101_000000"
    output_base.mkdir()
    source = tmp_path / "videos"
    source.mkdir()
    combo.write_batch_input_marker(output_base, source, "sam3sapiens2")
    marker = output_base / "BATCH_INPUT.json"
    assert marker.is_file()
    recorded = json.loads(marker.read_text(encoding="utf-8"))
    assert recorded["schema"] == "vaila_sam3sapiens2_batch_input_v1"
    assert Path(recorded["input"]) == source
    assert combo.find_batch_input_path(output_base) == source


def test_find_batch_input_path_falls_back_to_batch_summary_when_marker_missing(
    tmp_path: Path,
) -> None:
    # Older/completed runs from before the BATCH_INPUT.json marker existed
    # still record --input in the end-of-run batch summary.
    output_base = tmp_path / "processed_sam3sapiens2_20260101_000000"
    output_base.mkdir()
    source = tmp_path / "videos"
    source.mkdir()
    (output_base / "sam3sapiens2_batch_summary.json").write_text(
        json.dumps({"schema": "vaila_sam3sapiens2_batch_v1", "input": str(source)}),
        encoding="utf-8",
    )
    assert combo.find_batch_input_path(output_base) == source


def test_find_batch_input_path_returns_none_for_unmarked_directory(tmp_path: Path) -> None:
    output_base = tmp_path / "processed_sam3sapiens2_20260101_000000"
    output_base.mkdir()
    assert combo.find_batch_input_path(output_base) is None


def test_resolve_auto_resume_output_base_matches_existing_run_by_input(tmp_path: Path) -> None:
    output_parent = tmp_path / "out"
    output_parent.mkdir()
    source = tmp_path / "videos"
    source.mkdir()

    older = output_parent / "processed_sam3sapiens2_20260101_000000"
    older.mkdir()
    combo.write_batch_input_marker(older, source, "sam3sapiens2")

    newer = output_parent / "processed_sam3sapiens2_20260102_000000"
    newer.mkdir()
    combo.write_batch_input_marker(newer, source, "sam3sapiens2")

    output_base, is_resume = combo.resolve_auto_resume_output_base(
        output_parent, source, "sam3sapiens2", fresh=False
    )
    assert is_resume is True
    assert output_base == newer  # newest match wins


def test_resolve_auto_resume_output_base_ignores_unrelated_input(tmp_path: Path) -> None:
    output_parent = tmp_path / "out"
    output_parent.mkdir()
    other_source = tmp_path / "other_videos"
    other_source.mkdir()
    unrelated = output_parent / "processed_sam3sapiens2_20260101_000000"
    unrelated.mkdir()
    combo.write_batch_input_marker(unrelated, other_source, "sam3sapiens2")

    this_source = tmp_path / "videos"
    this_source.mkdir()
    output_base, is_resume = combo.resolve_auto_resume_output_base(
        output_parent, this_source, "sam3sapiens2", fresh=False
    )
    assert is_resume is False
    assert output_base != unrelated
    assert output_base.name.startswith("processed_sam3sapiens2_")


def test_resolve_auto_resume_output_base_fresh_forces_new_dir(tmp_path: Path) -> None:
    output_parent = tmp_path / "out"
    output_parent.mkdir()
    source = tmp_path / "videos"
    source.mkdir()
    existing = output_parent / "processed_sam3sapiens2_20260101_000000"
    existing.mkdir()
    combo.write_batch_input_marker(existing, source, "sam3sapiens2")

    output_base, is_resume = combo.resolve_auto_resume_output_base(
        output_parent, source, "sam3sapiens2", fresh=True
    )
    assert is_resume is False
    assert output_base != existing


def test_build_parser_exposes_fresh_flag_default_false(tmp_path: Path) -> None:
    args = combo._build_parser().parse_args(["-i", "clip.mp4", "-o", str(tmp_path)])
    assert args.fresh is False
    args = combo._build_parser().parse_args(["-i", "clip.mp4", "-o", str(tmp_path), "--fresh"])
    assert args.fresh is True


def test_main_rejects_fresh_with_resume(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    resume_dir = tmp_path / "processed_sam3sapiens2_20260101_000000"
    resume_dir.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        ["sam3sapiens2.py", "-i", "clip.mp4", "--resume", str(resume_dir), "--fresh"],
    )
    with pytest.raises(SystemExit):
        combo.main()


def test_plan_video_processing_honours_external_sam_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(combo, "_video_frame_count", lambda _path: 2)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    batch = tmp_path / "processed_sam_123"
    expected = _write_sam_fixture(batch / video.stem)
    action, sam_dir, summary = combo.plan_video_processing(
        video,
        tmp_path / "fresh_out" / video.stem,
        resume=False,
        sam_results=batch,
        single_video=False,
    )
    assert action == "reuse_sam"
    assert sam_dir == expected.resolve()
    assert summary is None


def test_resume_rejects_local_partial_sam(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(combo, "_video_frame_count", lambda _path: 2)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    out = tmp_path / "out"
    sam_dir = _write_sam_fixture(out / "sam3", frames=1)
    action, reused, summary = combo.plan_video_processing(
        video,
        out,
        resume=True,
        sam_results=None,
        single_video=True,
    )
    assert sam_dir.is_dir()
    assert action == "run"
    assert reused is None
    assert summary is None


def test_resume_accepts_verified_legacy_complete_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(combo, "_video_frame_count", lambda _path: 2)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    out = tmp_path / "out"
    out.mkdir()
    artifacts: dict[str, str] = {}
    for key, name in (
        ("predictions", "predictions.json"),
        ("long_csv", "poses.csv"),
        ("identity_audit", "audit.csv"),
    ):
        path = out / name
        path.write_text("ok", encoding="utf-8")
        artifacts[key] = str(path)
    (out / "sam3sapiens2_summary.json").write_text(
        json.dumps({"video": str(video), "n_frames": 2, **artifacts}),
        encoding="utf-8",
    )
    action, _sam, summary = combo.plan_video_processing(
        video,
        out,
        resume=True,
        sam_results=None,
        single_video=True,
    )
    assert action == "skip"
    assert summary is not None


def test_dry_run_resume_plan_reports_skip_and_rerun(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(combo, "_video_frame_count", lambda _path: 2)
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    done_video = videos_dir / "done.mp4"
    fail_video = videos_dir / "fail.mp4"
    done_video.write_bytes(b"")
    fail_video.write_bytes(b"")
    run_dir = tmp_path / "processed_sam3sapiens2_20260731_000000"
    done_out = run_dir / done_video.stem
    fail_out = run_dir / fail_video.stem
    done_out.mkdir(parents=True)
    fail_out.mkdir(parents=True)
    (done_out / "sam3sapiens2_summary.json").write_text(
        json.dumps(
            {
                "video": str(done_video),
                "completed": True,
                "expected_frames": 2,
                "n_frames": 2,
            }
        ),
        encoding="utf-8",
    )
    (fail_out / "FAILED_sam3sapiens2.txt").write_text("failed", encoding="utf-8")
    (fail_out / "sam3" / "_chunks").mkdir(parents=True)

    args = SimpleNamespace(
        input=videos_dir,
        resume=run_dir,
        sam_results=None,
        model="1b",
        stride=1,
        device=0,
        bbox_padding=0.12,
        contour_margin=8,
        no_contour_focus=False,
        text="person",
        sam_frame=0,
        sam_checkpoint=None,
        sam_max_frames=None,
        sam_max_input_long_edge=None,
        keep_sam_masks=False,
    )
    lines = combo._build_dry_run_report(
        [done_video.resolve(), fail_video.resolve()],
        run_dir,
        args,
    )
    joined = "\n".join(lines)
    assert "action=skip" in joined
    assert "action=run" in joined
    assert "--output-base" in joined
    assert "--video-output-dir" not in joined


class _OverreportingCapture:
    """Capture whose container metadata claims more frames than decode yields."""

    meta_frames = 10
    decodable_frames = 7
    width = 100
    height = 80

    def __init__(self, _path: str) -> None:
        self._pos = 0

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def get(self, prop: int) -> float:
        if prop == combo.cv2.CAP_PROP_FRAME_COUNT:
            return float(self.meta_frames)
        if prop == combo.cv2.CAP_PROP_FPS:
            return 30.0
        if prop == combo.cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == combo.cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        if prop == combo.cv2.CAP_PROP_POS_FRAMES:
            self._pos = int(value)
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._pos >= self.decodable_frames:
            return False, None
        self._pos += 1
        return True, np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def release(self) -> None:
        return None


class _SessionSentinelError(Exception):
    """Raised by the stubbed Sapiens2 session to prove the gate was passed."""


def test_run_sapiens_from_sam_gates_on_decodable_frames_not_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """JJ_Kabuto class: nb_frames=10 but only 7 frames decode and SAM covered all 7."""
    sam_dir = _write_sam_fixture(tmp_path / "sam", frames=_OverreportingCapture.decodable_frames)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    monkeypatch.setattr(combo.cv2, "VideoCapture", _OverreportingCapture)

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise _SessionSentinelError("Sapiens2 session reached")

    monkeypatch.setattr(combo, "PoseInferenceSession", _explode)
    with pytest.raises(_SessionSentinelError):
        combo.run_sapiens_from_sam(video, tmp_path / "out", sam_dir)


def test_run_sapiens_from_sam_still_rejects_a_real_mid_video_hole(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The coverage gate must keep failing when a decodable frame is truly missing."""
    sam_dir = _write_sam_fixture(tmp_path / "sam", frames=_OverreportingCapture.decodable_frames)
    meta = sam_dir / "sam_frames_meta.csv"
    lines = meta.read_text(encoding="utf-8").splitlines(keepends=True)
    meta.write_text(
        "".join(line for line in lines if not line.startswith("3,")),
        encoding="utf-8",
    )
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    monkeypatch.setattr(combo.cv2, "VideoCapture", _OverreportingCapture)

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise _SessionSentinelError("Sapiens2 must not load for an incomplete SAM run")

    monkeypatch.setattr(combo, "PoseInferenceSession", _explode)
    with pytest.raises(RuntimeError, match="SAM3 guidance is incomplete"):
        combo.run_sapiens_from_sam(video, tmp_path / "out", sam_dir)


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
