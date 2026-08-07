"""Tests for :mod:`vaila.sapiens2_3d` (Sapiens2-guided SAM 3D Body markerless 3D).

CPU-only: bbox-tightening math, the SAM->SAM 3D Body batch bridge with
Sapiens2 guidance, the guidance CSV writer, and the sam3sapiens2-results
loader/resolver. The GPU inference itself (SAM 3D Body estimator, real video
decode) needs CUDA and the gated weights, exactly like ``test_sam3dinov3.py``
-- out of scope here; see ``loops/sapiens2-3d-pipeline-loop.md`` tier 3 for
the real-video evidence this module still needs before ``success``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vaila.sapiens2_3d import (
    DEFAULT_KPT_SCORE_THRESH,
    DEFAULT_MIN_GUIDANCE_KEYPOINTS,
    DEFAULT_MIN_SANITY_IOU,
    DLT_DEFAULT_EXPORT_MESH,
    DLT_DEFAULT_ORIGIN_MARKERS,
    DLT_DEFAULT_SMOOTH_HZ,
    GuidanceRecord,
    _auto_locate_video_from_results,
    _find_raw_video_by_name,
    _frame_batch_from_guidance_sapiens2,
    _resolve_sapiens2_front_end,
    _run_dlt_chain,
    find_sapiens2_predictions_json,
    load_sapiens2_predictions,
    sapiens2_keypoint_bbox,
    sapiens2_keypoints_by_frame,
    tighten_bbox_with_sapiens2,
    write_guidance_csv,
)

W, H = 640, 480


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def _track(obj_id: int, x: float, y: float, w: float, h: float) -> dict[str, Any]:
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


def _contour(obj_id: int, x: float, y: float, w: float, h: float) -> dict[str, Any]:
    return {
        "obj_id": obj_id,
        "polygons": [[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]],
    }


def _keypoints_inside(x: float, y: float, w: float, h: float, n: int = 10) -> np.ndarray:
    """n keypoints spread inside the box (x, y, x+w, y+h)."""
    rng = np.random.default_rng(0)
    xs = rng.uniform(x + 0.1 * w, x + 0.9 * w, size=n)
    ys = rng.uniform(y + 0.1 * h, y + 0.9 * h, size=n)
    return np.stack([xs, ys], axis=1)


# --------------------------------------------------------------------------- #
# sapiens2_keypoint_bbox
# --------------------------------------------------------------------------- #
def test_keypoint_bbox_ignores_low_score_points():
    kpts = np.array([[10.0, 10.0], [100.0, 100.0], [500.0, 400.0]])
    scores = np.array([0.9, 0.9, 0.05])  # third point below threshold
    bbox, num_used = sapiens2_keypoint_bbox(
        kpts, scores, score_thresh=0.3, frame_width=W, frame_height=H, padding_frac=0.0
    )
    assert num_used == 2
    assert bbox is not None
    np.testing.assert_allclose(bbox, [10.0, 10.0, 100.0, 100.0], atol=1e-4)


def test_keypoint_bbox_returns_none_when_all_below_threshold():
    kpts = np.array([[10.0, 10.0], [100.0, 100.0]])
    scores = np.array([0.1, 0.2])
    bbox, num_used = sapiens2_keypoint_bbox(
        kpts, scores, score_thresh=0.3, frame_width=W, frame_height=H, padding_frac=0.0
    )
    assert bbox is None
    assert num_used == 0


def test_keypoint_bbox_ignores_nan_points():
    kpts = np.array([[10.0, 10.0], [np.nan, np.nan], [100.0, 100.0]])
    scores = np.array([0.9, 0.9, 0.9])
    bbox, num_used = sapiens2_keypoint_bbox(
        kpts, scores, score_thresh=0.3, frame_width=W, frame_height=H, padding_frac=0.0
    )
    assert num_used == 2
    assert bbox is not None
    assert np.isfinite(bbox).all()


def test_keypoint_bbox_padding_and_clipping():
    kpts = np.array([[5.0, 5.0], [50.0, 40.0]])
    scores = np.array([0.9, 0.9])
    bbox, num_used = sapiens2_keypoint_bbox(
        kpts, scores, score_thresh=0.3, frame_width=W, frame_height=H, padding_frac=1.0
    )
    assert num_used == 2
    assert bbox is not None
    # width=45, height=35 -> pad_x=45, pad_y=35; x0 clipped to 0 (5-45<0)
    assert bbox[0] == pytest.approx(0.0)
    assert bbox[1] == pytest.approx(0.0)
    assert bbox[2] == pytest.approx(95.0)
    assert bbox[3] == pytest.approx(75.0)


def test_keypoint_bbox_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        sapiens2_keypoint_bbox(
            np.zeros((3, 2)),
            np.zeros(2),
            score_thresh=0.3,
            frame_width=W,
            frame_height=H,
            padding_frac=0.0,
        )


# --------------------------------------------------------------------------- #
# tighten_bbox_with_sapiens2
# --------------------------------------------------------------------------- #
def test_tighten_falls_back_when_keypoints_are_none():
    sam_bbox = np.array([0.0, 0.0, 200.0, 200.0])
    bbox, guided, num_used = tighten_bbox_with_sapiens2(
        sam_bbox, None, None, frame_width=W, frame_height=H
    )
    np.testing.assert_allclose(bbox, sam_bbox)
    assert guided is False
    assert num_used == 0


def test_tighten_falls_back_below_min_guidance_keypoints():
    sam_bbox = np.array([0.0, 0.0, 200.0, 200.0])
    kpts = _keypoints_inside(20, 20, 60, 60, n=DEFAULT_MIN_GUIDANCE_KEYPOINTS - 1)
    scores = np.full(len(kpts), 0.9)
    bbox, guided, num_used = tighten_bbox_with_sapiens2(
        sam_bbox, kpts, scores, frame_width=W, frame_height=H
    )
    np.testing.assert_allclose(bbox, sam_bbox)
    assert guided is False
    assert num_used == DEFAULT_MIN_GUIDANCE_KEYPOINTS - 1


def test_tighten_accepts_confident_keypoints_agreeing_with_sam_bbox():
    # SAM bbox is loose; Sapiens2 keypoints cluster tightly inside it and
    # overlap it well enough to pass the sanity IoU check.
    sam_bbox = np.array([0.0, 0.0, 300.0, 300.0])
    kpts = _keypoints_inside(50, 50, 150, 150, n=12)
    scores = np.full(len(kpts), 0.9)
    bbox, guided, num_used = tighten_bbox_with_sapiens2(
        sam_bbox, kpts, scores, frame_width=W, frame_height=H, padding_frac=0.08
    )
    assert guided is True
    assert num_used == 12
    # Tightened box must be strictly inside the loose SAM box and non-trivial.
    assert bbox[0] > sam_bbox[0]
    assert bbox[1] > sam_bbox[1]
    assert bbox[2] < sam_bbox[2]
    assert bbox[3] < sam_bbox[3]
    assert bbox[2] > bbox[0]
    assert bbox[3] > bbox[1]


def test_tighten_falls_back_when_keypoints_disagree_with_sam_bbox():
    # Sapiens2 keypoints land far from the SAM bbox -- e.g. a misassigned
    # track for a different, nearby person. Must not silently guide there.
    sam_bbox = np.array([0.0, 0.0, 50.0, 50.0])
    kpts = _keypoints_inside(400, 350, 100, 100, n=10)
    scores = np.full(len(kpts), 0.95)
    bbox, guided, num_used = tighten_bbox_with_sapiens2(
        sam_bbox, kpts, scores, frame_width=W, frame_height=H, min_sanity_iou=DEFAULT_MIN_SANITY_IOU
    )
    np.testing.assert_allclose(bbox, sam_bbox)
    assert guided is False
    assert num_used == 10


def test_tighten_uses_module_default_score_threshold():
    # A point exactly at the module default threshold counts; just below does not.
    # Deterministic corner keypoints (not the random helper) so the derived
    # bbox reliably overlaps the SAM bbox well enough to clear the sanity IoU
    # check regardless of RNG draw -- the threshold behaviour is what this
    # test targets, not the IoU sanity check (covered separately above).
    sam_bbox = np.array([40.0, 40.0, 210.0, 210.0])
    assert DEFAULT_MIN_GUIDANCE_KEYPOINTS == 4, "test assumes 4 corner points"
    kpts = np.array([[60.0, 60.0], [190.0, 60.0], [190.0, 190.0], [60.0, 190.0]])
    scores = np.full(len(kpts), DEFAULT_KPT_SCORE_THRESH)
    _, guided, num_used = tighten_bbox_with_sapiens2(
        sam_bbox, kpts, scores, frame_width=W, frame_height=H
    )
    assert guided is True
    assert num_used == DEFAULT_MIN_GUIDANCE_KEYPOINTS

    scores_below = np.full(len(kpts), DEFAULT_KPT_SCORE_THRESH - 1e-3)
    _, guided_below, num_used_below = tighten_bbox_with_sapiens2(
        sam_bbox, kpts, scores_below, frame_width=W, frame_height=H
    )
    assert guided_below is False
    assert num_used_below == 0


# --------------------------------------------------------------------------- #
# _frame_batch_from_guidance_sapiens2
# --------------------------------------------------------------------------- #
def test_frame_batch_guides_one_person_and_falls_back_for_another():
    track_a = _track(1, 10, 10, 100, 100)
    contour_a = _contour(1, 10, 10, 100, 100)
    track_b = _track(2, 300, 200, 80, 80)
    contour_b = _contour(2, 300, 200, 80, 80)
    frame_guidance = [(track_a, contour_a), (track_b, contour_b)]

    # Person 1 has confident, well-agreeing Sapiens2 keypoints -> guided.
    # Person 2 has no entry in the lookup at all -> falls back to SAM bbox.
    kpts_a = _keypoints_inside(20, 20, 80, 80, n=8)
    scores_a = np.full(len(kpts_a), 0.9)
    sapiens2_lookup = {1: (kpts_a, scores_a)}

    obj_ids, boxes, masks, records = _frame_batch_from_guidance_sapiens2(
        frame_guidance,
        sapiens2_lookup,
        frame_idx=7,
        frame_width=W,
        frame_height=H,
        bbox_padding=0.1,
        contour_margin=4,
        use_mask=True,
        min_guidance_keypoints=DEFAULT_MIN_GUIDANCE_KEYPOINTS,
        kpt_score_thresh=DEFAULT_KPT_SCORE_THRESH,
        kpt_padding_frac=0.08,
        min_sanity_iou=DEFAULT_MIN_SANITY_IOU,
    )

    assert obj_ids == [1, 2]
    assert boxes is not None and boxes.shape == (2, 4)
    assert masks is not None and masks.shape[0] == 2
    # Masks must be binary 0/1 (SAM 3D Body convention), never 0/255.
    assert set(np.unique(masks)).issubset({0, 1})

    assert len(records) == 2
    rec_by_person = {r.person_id: r for r in records}
    assert rec_by_person[1].guided is True
    assert rec_by_person[1].num_keypoints_used == 8
    assert rec_by_person[1].frame == 7
    assert rec_by_person[2].guided is False
    assert rec_by_person[2].num_keypoints_used == 0


def test_frame_batch_skips_degenerate_boxes():
    # A zero-area SAM box (buggy upstream track) with no keypoints to help it
    # must be dropped, not passed to the estimator as a 0-pixel crop.
    track = _track(9, 100, 100, 0, 0)
    contour = _contour(9, 100, 100, 0, 0)
    obj_ids, boxes, masks, records = _frame_batch_from_guidance_sapiens2(
        [(track, contour)],
        {},
        frame_idx=0,
        frame_width=W,
        frame_height=H,
        bbox_padding=0.0,
        contour_margin=0,
        use_mask=False,
        min_guidance_keypoints=DEFAULT_MIN_GUIDANCE_KEYPOINTS,
        kpt_score_thresh=DEFAULT_KPT_SCORE_THRESH,
        kpt_padding_frac=0.08,
        min_sanity_iou=DEFAULT_MIN_SANITY_IOU,
    )
    assert obj_ids == []
    assert boxes is None
    assert masks is None
    assert records == []


def test_frame_batch_without_mask_returns_none_masks():
    track = _track(1, 10, 10, 100, 100)
    contour = _contour(1, 10, 10, 100, 100)
    obj_ids, boxes, masks, records = _frame_batch_from_guidance_sapiens2(
        [(track, contour)],
        {},
        frame_idx=0,
        frame_width=W,
        frame_height=H,
        bbox_padding=0.1,
        contour_margin=4,
        use_mask=False,
        min_guidance_keypoints=DEFAULT_MIN_GUIDANCE_KEYPOINTS,
        kpt_score_thresh=DEFAULT_KPT_SCORE_THRESH,
        kpt_padding_frac=0.08,
        min_sanity_iou=DEFAULT_MIN_SANITY_IOU,
    )
    assert obj_ids == [1]
    assert boxes is not None
    assert masks is None
    assert records[0].guided is False


# --------------------------------------------------------------------------- #
# write_guidance_csv
# --------------------------------------------------------------------------- #
def test_write_guidance_csv_round_trips(tmp_path: Path):
    records = [
        GuidanceRecord(frame=0, person_id=1, guided=True, num_keypoints_used=12),
        GuidanceRecord(frame=0, person_id=2, guided=False, num_keypoints_used=1),
        GuidanceRecord(frame=1, person_id=1, guided=False, num_keypoints_used=0),
    ]
    path = write_guidance_csv(tmp_path, "clip", records)
    assert path.name == "clip_sapiens2_3d_guidance.csv"

    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    assert rows[0] == {
        "frame": "0",
        "person_id": "1",
        "guided": "1",
        "num_keypoints_used": "12",
    }
    assert rows[1]["guided"] == "0"
    assert rows[1]["num_keypoints_used"] == "1"


# --------------------------------------------------------------------------- #
# find_sapiens2_predictions_json / load_sapiens2_predictions / lookup
# --------------------------------------------------------------------------- #
def _write_fake_payload(path: Path, *, sam_results: str = "/fake/sam3") -> dict:
    payload = {
        "schema": "vaila_sam3sapiens2_v1",
        "video": "clip.mp4",
        "image_size": [H, W],
        "fps": 30.0,
        "n_frames": 2,
        "sam_results": sam_results,
        "frames": [
            {
                "frame_index": 0,
                "instances": [
                    {
                        "sam_obj_id": 1,
                        "sam_bbox_xyxy": [10.0, 10.0, 110.0, 110.0],
                        "keypoints": [[20.0, 20.0], [90.0, 90.0]],
                        "keypoint_scores": [0.9, 0.8],
                    }
                ],
            },
            {
                "frame_index": 1,
                "instances": [
                    {
                        "sam_obj_id": 1,
                        "sam_bbox_xyxy": [12.0, 12.0, 112.0, 112.0],
                        "keypoints": [[22.0, 22.0], [92.0, 92.0]],
                        "keypoint_scores": [0.9, 0.1],
                    }
                ],
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_find_sapiens2_predictions_json_direct_file(tmp_path: Path):
    json_path = tmp_path / "clip_sam3sapiens2_predictions.json"
    _write_fake_payload(json_path)
    found = find_sapiens2_predictions_json(tmp_path, "clip")
    assert found == json_path


def test_find_sapiens2_predictions_json_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        find_sapiens2_predictions_json(tmp_path, "clip")


def test_find_sapiens2_predictions_json_ambiguous_raises(tmp_path: Path):
    (tmp_path / "sub_a").mkdir()
    (tmp_path / "sub_b").mkdir()
    _write_fake_payload(tmp_path / "sub_a" / "other_sam3sapiens2_predictions.json")
    _write_fake_payload(tmp_path / "sub_b" / "yet_another_sam3sapiens2_predictions.json")
    with pytest.raises(FileNotFoundError):
        find_sapiens2_predictions_json(tmp_path, video_stem=None)


def test_find_sapiens2_predictions_json_unambiguous_direct_child_ignores_stem_mismatch(
    tmp_path: Path, capsys
):
    """Regression for the 2026-08-07 bug: a directory holding exactly one
    predictions JSON is used even when the caller's video_stem doesn't match
    it -- this is exactly the real case where a Visualize-ID rerender dir
    preserves a copy of the original combined run's JSON under a name that
    doesn't match the (wrong) video that got queued from that same dir."""
    json_path = tmp_path / "c1_cod_sam3sapiens2_predictions.json"
    _write_fake_payload(json_path)
    found = find_sapiens2_predictions_json(tmp_path, video_stem="c1_cod_sam3sapiens2_id_04_overlay")
    assert found == json_path
    assert "does not match" in capsys.readouterr().out


def test_find_sapiens2_predictions_json_resolves_visualized_id_sibling(tmp_path: Path, capsys):
    combined_dir = tmp_path / "c1_cod"
    combined_dir.mkdir()
    sibling_json = combined_dir / "c1_cod_sam3sapiens2_predictions.json"
    _write_fake_payload(sibling_json)

    visualize_dir = tmp_path / "c1_cod_sam3sapiens2_visualized_id_04"
    visualize_dir.mkdir()
    # No predictions JSON directly inside visualize_dir this time (unlike the
    # real fixture) -- forces resolution through the sibling-directory path.
    (visualize_dir / "some_other_file.csv").write_text("x", encoding="utf-8")

    found = find_sapiens2_predictions_json(visualize_dir)
    assert found == sibling_json
    assert "resolved to its combined run" in capsys.readouterr().out


def test_find_sapiens2_predictions_json_missing_suggests_sibling_in_error(tmp_path: Path):
    visualize_dir = tmp_path / "c1_cod_sam3dinov3_visualized_id_08"
    visualize_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="did you mean the combined run"):
        find_sapiens2_predictions_json(visualize_dir)


# --------------------------------------------------------------------------- #
# _find_raw_video_by_name / _auto_locate_video_from_results
# --------------------------------------------------------------------------- #
def test_find_raw_video_by_name_walks_up_to_a_unique_match(tmp_path: Path):
    (tmp_path / "c1_cod.mp4").write_bytes(b"")
    nested = tmp_path / "processed_sam3sapiens2_20260806" / "c1_cod"
    nested.mkdir(parents=True)
    found = _find_raw_video_by_name([nested], "c1_cod.mp4")
    assert found == (tmp_path / "c1_cod.mp4").resolve()


def test_find_raw_video_by_name_returns_none_when_not_found(tmp_path: Path):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert _find_raw_video_by_name([nested], "does_not_exist.mp4") is None


def test_find_raw_video_by_name_returns_none_on_ambiguity(tmp_path: Path):
    (tmp_path / "left").mkdir()
    (tmp_path / "left" / "clip.mp4").write_bytes(b"")
    (tmp_path / "right").mkdir()
    (tmp_path / "right" / "clip.mp4").write_bytes(b"")
    found = _find_raw_video_by_name([tmp_path / "left", tmp_path / "right"], "clip.mp4")
    assert found is None


def test_auto_locate_video_from_results_finds_it_via_payload_video_field(tmp_path: Path):
    (tmp_path / "c1_cod.mp4").write_bytes(b"")
    results_dir = tmp_path / "processed_sam3sapiens2_20260806" / "c1_cod"
    results_dir.mkdir(parents=True)
    _write_fake_payload(results_dir / "c1_cod_sam3sapiens2_predictions.json")
    # _write_fake_payload's video field is "clip.mp4"; rename the raw file to match.
    (tmp_path / "c1_cod.mp4").rename(tmp_path / "clip.mp4")

    found = _auto_locate_video_from_results(
        sapiens2_results=results_dir, sam_results=None, input_hint=None
    )
    assert found == (tmp_path / "clip.mp4").resolve()


def test_auto_locate_video_from_results_returns_none_without_sapiens2_results(tmp_path: Path):
    assert (
        _auto_locate_video_from_results(
            sapiens2_results=None, sam_results=tmp_path, input_hint=None
        )
        is None
    )


def test_auto_locate_video_from_results_returns_none_when_video_field_missing(tmp_path: Path):
    results_dir = tmp_path / "c1_cod"
    results_dir.mkdir()
    payload = _write_fake_payload(results_dir / "c1_cod_sam3sapiens2_predictions.json")
    del payload["video"]
    (results_dir / "c1_cod_sam3sapiens2_predictions.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    assert (
        _auto_locate_video_from_results(
            sapiens2_results=results_dir, sam_results=None, input_hint=None
        )
        is None
    )


def test_load_and_index_sapiens2_predictions(tmp_path: Path):
    json_path = tmp_path / "clip_sam3sapiens2_predictions.json"
    _write_fake_payload(json_path)
    payload = load_sapiens2_predictions(json_path)
    lookup = sapiens2_keypoints_by_frame(payload)

    assert set(lookup.keys()) == {0, 1}
    kpts0, scores0 = lookup[0][1]
    np.testing.assert_allclose(kpts0, [[20.0, 20.0], [90.0, 90.0]])
    np.testing.assert_allclose(scores0, [0.9, 0.8])
    assert 2 not in lookup[0]  # only obj_id 1 present in this fixture


def test_load_sapiens2_predictions_warns_on_unexpected_schema(tmp_path: Path, capsys):
    json_path = tmp_path / "clip_sam3sapiens2_predictions.json"
    json_path.write_text(json.dumps({"schema": "something_else", "frames": []}), encoding="utf-8")
    load_sapiens2_predictions(json_path)  # must not raise
    assert "WARNING" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# _resolve_sapiens2_front_end branching (mocked -- no GPU, no real files)
# --------------------------------------------------------------------------- #
def test_resolve_front_end_requires_one_source(tmp_path: Path):
    args = argparse.Namespace(sapiens2_results=None, sam_results=None)
    with pytest.raises(ValueError):
        _resolve_sapiens2_front_end(tmp_path / "clip.mp4", tmp_path / "out", args)


def test_resolve_front_end_prefers_existing_sapiens2_results(tmp_path: Path, monkeypatch):
    json_path = tmp_path / "clip_sam3sapiens2_predictions.json"
    _write_fake_payload(json_path, sam_results=str(tmp_path / "sam3"))

    called = {"run_sapiens_from_sam": False}

    def _fail_if_called(*args, **kwargs):
        called["run_sapiens_from_sam"] = True
        raise AssertionError("must not run the Sapiens2 stage when results already exist")

    monkeypatch.setattr("vaila.sapiens2_3d.run_sapiens_from_sam", _fail_if_called)
    monkeypatch.setattr(
        "vaila.sapiens2_3d.load_sam_guidance",
        lambda sam_dir: SamGuidanceStub(sam_dir),
    )

    args = argparse.Namespace(sapiens2_results=tmp_path, sam_results=None)
    guidance, lookup, resolved_json = _resolve_sapiens2_front_end(
        tmp_path / "clip.mp4", tmp_path / "out", args
    )
    assert resolved_json == json_path
    assert called["run_sapiens_from_sam"] is False
    assert set(lookup.keys()) == {0, 1}
    assert guidance.sam_dir == Path(tmp_path / "sam3")


def test_resolve_front_end_rejects_video_payload_mismatch(tmp_path: Path, monkeypatch):
    """Correctness guard: if the resolved predictions JSON was built from a
    different video than the one about to be processed, frame-by-frame
    guidance would silently misalign with the wrong footage -- must raise,
    not proceed."""
    json_path = tmp_path / "c1_cod_sam3sapiens2_predictions.json"
    _write_fake_payload(json_path)  # payload["video"] == "clip.mp4"
    monkeypatch.setattr(
        "vaila.sapiens2_3d.load_sam_guidance", lambda sam_dir: SamGuidanceStub(sam_dir)
    )

    args = argparse.Namespace(sapiens2_results=tmp_path, sam_results=None)
    wrong_video = tmp_path / "a_completely_different_video.mp4"
    with pytest.raises(ValueError, match="must be the same video"):
        _resolve_sapiens2_front_end(wrong_video, tmp_path / "out", args)


def test_resolve_front_end_runs_sapiens2_stage_when_only_sam_results_given(
    tmp_path: Path, monkeypatch
):
    sam_dir = tmp_path / "sam3"
    sam_dir.mkdir()
    video_path = tmp_path / "clip.mp4"
    output_dir = tmp_path / "out"

    def _fake_resolve_sam_results_dir(sam_results, video_path, *, single_video):
        return sam_dir

    def _fake_run_sapiens_from_sam(video_path, sapiens2_dir, sam_dir_arg):
        sapiens2_dir.mkdir(parents=True, exist_ok=True)
        _write_fake_payload(
            sapiens2_dir / f"{video_path.stem}_sam3sapiens2_predictions.json",
            sam_results=str(sam_dir_arg),
        )
        return {}

    monkeypatch.setattr("vaila.sapiens2_3d.resolve_sam_results_dir", _fake_resolve_sam_results_dir)
    monkeypatch.setattr("vaila.sapiens2_3d.run_sapiens_from_sam", _fake_run_sapiens_from_sam)
    monkeypatch.setattr(
        "vaila.sapiens2_3d.load_sam_guidance", lambda sam_dir_arg: SamGuidanceStub(sam_dir_arg)
    )

    args = argparse.Namespace(sapiens2_results=None, sam_results=tmp_path / "raw_sam")
    guidance, lookup, resolved_json = _resolve_sapiens2_front_end(video_path, output_dir, args)

    assert resolved_json == output_dir / "sam3sapiens2" / "clip_sam3sapiens2_predictions.json"
    assert guidance.sam_dir == sam_dir
    assert set(lookup.keys()) == {0, 1}


class SamGuidanceStub:
    """Minimal stand-in for sam3sapiens2.SamGuidance in the resolver tests above."""

    def __init__(self, sam_dir: Path) -> None:
        self.sam_dir = Path(sam_dir)


# --------------------------------------------------------------------------- #
# _run_dlt_chain (mocked align_monocular_to_world -- no real DLT math here,
# that belongs to test_monocular_dlt_align.py; this only covers the call site)
# --------------------------------------------------------------------------- #
def _dlt_chain_args(**overrides: Any) -> argparse.Namespace:
    defaults = {
        "dlt3d": Path("/fake/cam.dlt3d"),
        "ref3d": None,
        "smooth_hz": DLT_DEFAULT_SMOOTH_HZ,
        "no_smooth": False,
        "no_refine": False,
        "origin_markers": list(DLT_DEFAULT_ORIGIN_MARKERS),
        "skeleton": None,
        "export_mesh": DLT_DEFAULT_EXPORT_MESH,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_run_dlt_chain_calls_align_once_per_person_with_right_paths(tmp_path: Path, monkeypatch):
    for pid in (1, 2):
        (tmp_path / f"clip_id_{pid:02d}_mhr70_rec3d.csv").write_text("frame\n", encoding="utf-8")
        (tmp_path / f"clip_id_{pid:02d}_markers.csv").write_text("frame\n", encoding="utf-8")

    calls: list[dict[str, Any]] = []

    def _fake_align(mono3d_path, dlt3d_path, output_directory, **kwargs):
        calls.append(
            {
                "mono3d_path": mono3d_path,
                "dlt3d_path": dlt3d_path,
                "output_directory": output_directory,
                **kwargs,
            }
        )
        return (output_directory, "clip")

    monkeypatch.setattr("vaila.sapiens2_3d.align_monocular_to_world", _fake_align)

    args = _dlt_chain_args()
    results = _run_dlt_chain(tmp_path, "clip", [1, 2], 119.88, args)

    assert set(results.keys()) == {1, 2}
    assert len(calls) == 2
    call1 = next(c for c in calls if c["mono3d_path"] == tmp_path / "clip_id_01_mhr70_rec3d.csv")
    assert call1["dlt3d_path"] == args.dlt3d
    assert call1["output_directory"] == tmp_path / "dlt_world" / "id_01"
    assert call1["pixels_path"] == tmp_path / "clip_id_01_markers.csv"
    assert call1["ref3d_path"] is None
    assert call1["point_rate"] == 119.88
    assert call1["smooth_hz"] == DLT_DEFAULT_SMOOTH_HZ
    assert call1["refine"] is True
    assert call1["origin_markers"] == tuple(DLT_DEFAULT_ORIGIN_MARKERS)
    assert call1["export_mesh"] == DLT_DEFAULT_EXPORT_MESH
    assert call1["gui"] is False


def test_run_dlt_chain_no_smooth_and_no_refine_flags(tmp_path: Path, monkeypatch):
    (tmp_path / "clip_id_01_mhr70_rec3d.csv").write_text("frame\n", encoding="utf-8")

    captured: dict[str, Any] = {}

    def _fake_align(mono3d_path, dlt3d_path, output_directory, **kwargs):
        captured.update(kwargs)
        return (output_directory, "clip")

    monkeypatch.setattr("vaila.sapiens2_3d.align_monocular_to_world", _fake_align)

    args = _dlt_chain_args(no_smooth=True, no_refine=True)
    _run_dlt_chain(tmp_path, "clip", [1], 30.0, args)

    assert captured["smooth_hz"] == 0.0
    assert captured["refine"] is False


def test_run_dlt_chain_skips_missing_mono3d_file(tmp_path: Path, monkeypatch):
    called = False

    def _fake_align(*a, **kw):
        nonlocal called
        called = True
        return None

    monkeypatch.setattr("vaila.sapiens2_3d.align_monocular_to_world", _fake_align)
    args = _dlt_chain_args()
    results = _run_dlt_chain(tmp_path, "clip", [1], 30.0, args)
    assert results == {}
    assert called is False


def test_run_dlt_chain_continues_after_one_person_fails(tmp_path: Path, monkeypatch, capsys):
    for pid in (1, 2):
        (tmp_path / f"clip_id_{pid:02d}_mhr70_rec3d.csv").write_text("frame\n", encoding="utf-8")

    def _fake_align(mono3d_path, *a, **kw):
        if "id_01" in str(mono3d_path):
            raise RuntimeError("placement blew up for person 1")
        return (Path("/fake/out"), "clip")

    monkeypatch.setattr("vaila.sapiens2_3d.align_monocular_to_world", _fake_align)
    args = _dlt_chain_args()
    results = _run_dlt_chain(tmp_path, "clip", [1, 2], 30.0, args)

    assert 1 not in results
    assert 2 in results
    assert "WARNING" in capsys.readouterr().out


def test_run_dlt_chain_handles_none_result(tmp_path: Path, monkeypatch):
    (tmp_path / "clip_id_01_mhr70_rec3d.csv").write_text("frame\n", encoding="utf-8")
    monkeypatch.setattr("vaila.sapiens2_3d.align_monocular_to_world", lambda *a, **kw: None)
    args = _dlt_chain_args()
    results = _run_dlt_chain(tmp_path, "clip", [1], 30.0, args)
    assert results == {}
