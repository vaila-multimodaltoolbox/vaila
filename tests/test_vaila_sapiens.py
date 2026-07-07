"""Unit tests for vaila.vaila_sapiens (Sapiens2 Pose video wrapper)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

import numpy as np
import pytest

from vaila import vaila_sapiens as vs


def test_normalize_model_key_default() -> None:
    assert vs._normalize_model_key("1b") == "1b"
    assert vs._normalize_model_key("sapiens2_1b") == "1b"


def test_normalize_model_key_invalid() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        vs._normalize_model_key("99b")


def test_resolve_model_spec_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    sapiens2 = repo / "sapiens2" / "sapiens" / "pose"
    cfg = (
        sapiens2
        / "configs/keypoints308/shutterstock_goliath_3po"
        / "sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768.py"
    )
    cfg.parent.mkdir(parents=True)
    cfg.write_text("# stub config\n", encoding="utf-8")
    ckpt_root = repo / "vaila" / "models" / "sapiens2"
    pose_ckpt = ckpt_root / "pose" / "sapiens2_1b_pose.safetensors"
    pose_ckpt.parent.mkdir(parents=True)
    pose_ckpt.write_bytes(b"stub")
    det = ckpt_root / "detector" / vs.DETECTOR_DIRNAME
    det.mkdir(parents=True)
    (det / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(vs, "_repo_root", lambda: repo)
    monkeypatch.setattr(vs, "_sapiens2_root", lambda: repo / "sapiens2")
    monkeypatch.setattr(vs, "_checkpoint_root", lambda: ckpt_root)

    spec = vs.resolve_model_spec("1b")
    assert spec.model_key == "1b"
    assert spec.config_path.is_file()
    assert spec.checkpoint_path.is_file()
    assert spec.detector_path.is_dir()


def test_flatten_instances_to_csv_rows() -> None:
    instances = [
        {
            "keypoints": [[10.0, 20.0], [30.0, 40.0]],
            "keypoint_scores": [0.9, 0.8],
        }
    ]
    rows = vs.flatten_instances_to_csv_rows(5, instances)
    assert rows == [
        (5, 0, 0, 10.0, 20.0, 0.9),
        (5, 0, 1, 30.0, 40.0, 0.8),
    ]


def test_nearest_pose_frame() -> None:
    keys = [0, 10, 20]
    assert vs._nearest_pose_frame(7, keys) == 10
    assert vs._nearest_pose_frame(3, keys) == 0


def test_build_dry_run_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"\x00")
    monkeypatch.setattr(
        vs,
        "resolve_model_spec",
        lambda model="1b": vs.SapiensModelSpec(
            model_key="1b",
            arch="sapiens2_1b",
            config_path=tmp_path / "cfg.py",
            checkpoint_path=tmp_path / "ckpt.safetensors",
            detector_path=tmp_path / "det",
        ),
    )
    lines = vs.build_dry_run_report(vid, tmp_path / "out", model="1b", stride=2)
    assert any("Dry-run" in line for line in lines)
    assert any("clip.mp4" in line for line in lines)


def test_open_sam3_video_writer_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "test_overlay.mp4"
    writer, actual = vs._open_sam3_video_writer(out, 30.0, (64, 48), purpose="test")
    try:
        assert writer.isOpened()
        assert actual.exists() or actual.parent == tmp_path
    finally:
        writer.release()


def test_markerless_2d_analysis_videos_dry_run() -> None:
    """Smoke: dry-run on committed test clips (no GPU inference)."""
    root = Path(__file__).resolve().parents[1]
    test_dir = root / "tests" / "markerless_2d_analysis"
    videos = list(test_dir.glob("*.mp4"))
    if not videos:
        pytest.skip("no test mp4 in tests/markerless_2d_analysis/")
    lines = vs.build_dry_run_report(test_dir, test_dir / "out_dry", model="1b", stride=1)
    assert any("videos=" in line for line in lines)


def test_format_sapiens_cli_command_quotes_paths() -> None:
    cmd = vs._format_sapiens_cli_command(
        "/path/with spaces/in.mp4",
        "/out folder",
        model="1b",
        stride=2,
        kpt_thr=0.35,
        save_overlay=False,
    )
    assert "'/path/with spaces/in.mp4'" in cmd
    assert "'/out folder'" in cmd
    assert "--model 1b" in cmd
    assert "--stride 2" in cmd
    assert "--no-overlay" in cmd


def test_build_sapiens_cli_argv_output_base() -> None:
    argv = vs._build_sapiens_cli_argv(
        input_path=Path("/in/vid.mp4"),
        out_parent=Path("/out"),
        output_base=Path("/out/processed_sapiens_123"),
        model="1b",
        for_subprocess=True,
    )
    assert "--output-base" in argv
    assert "/out/processed_sapiens_123" in argv
    assert argv[0] == sys.executable


def test_find_videos_skips_sapiens_outputs(tmp_path: Path) -> None:
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00")
    overlay = tmp_path / "clip_sapiens_overlay.mp4"
    overlay.write_bytes(b"\x00")
    nested = tmp_path / "processed_sapiens_20260101_120000" / "other.mp4"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"\x00")
    found = vs._find_videos(tmp_path)
    assert found == [src.resolve()]


def test_is_sapiens_derived_video() -> None:
    assert vs._is_sapiens_derived_video(Path("a_sapiens_overlay.mp4"))
    assert vs._is_sapiens_derived_video(
        Path("out/processed_sapiens_123/clip/clip_sapiens_overlay.mp4")
    )
    assert not vs._is_sapiens_derived_video(Path("clip.mp4"))


@patch.object(vs, "run_sapiens_on_video")
def test_run_sapiens_batch_emits_progress_lines(mock_run: MagicMock, tmp_path: Path) -> None:
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"\x00")
    out_base = tmp_path / "batch"
    out_base.mkdir()
    succeeded, failed = vs._run_sapiens_batch(
        [vid],
        out_base,
        model="1b",
        stride=1,
        save_overlay=False,
        bbox_thr=0.3,
        nms_thr=0.3,
        kpt_thr=0.3,
        device="cuda:0",
    )
    assert succeeded == 1
    assert failed == []
    mock_run.assert_called_once()


@patch("huggingface_hub.snapshot_download")
@patch("huggingface_hub.hf_hub_download")
def test_download_weights_uses_hf_hub_api(
    mock_hf_download: MagicMock,
    mock_snapshot: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt_root = tmp_path / "models"
    monkeypatch.setattr(vs, "_checkpoint_root", lambda: ckpt_root)
    mock_hf_download.return_value = str(ckpt_root / "pose" / "sapiens2_1b_pose.safetensors")

    vs.download_weights("1b")

    mock_hf_download.assert_called_once()
    assert mock_hf_download.call_args.kwargs["repo_id"] == "facebook/sapiens2-pose-1b"
    mock_snapshot.assert_called_once()
    assert mock_snapshot.call_args.kwargs["repo_id"] == vs.DETECTOR_HF_REPO


@patch.object(vs, "PoseInferenceSession")
def test_run_sapiens_on_video_writes_outputs(
    mock_session_cls: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cv2

    repo = tmp_path / "repo"
    video = tmp_path / "sample.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # ty: ignore[unresolved-attribute]
    writer = cv2.VideoWriter(str(video), fourcc, 10.0, (32, 24))
    for _ in range(3):
        writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()

    cfg = repo / "sapiens2" / "sapiens" / "pose" / "configs" / "c.py"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("# x", encoding="utf-8")
    ckpt_root = repo / "vaila" / "models" / "sapiens2"
    (ckpt_root / "pose").mkdir(parents=True)
    (ckpt_root / "pose" / "sapiens2_1b_pose.safetensors").write_bytes(b"x")
    det = ckpt_root / "detector" / vs.DETECTOR_DIRNAME
    det.mkdir(parents=True)
    (det / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(vs, "_repo_root", lambda: repo)
    monkeypatch.setattr(vs, "_sapiens2_root", lambda: repo / "sapiens2")
    monkeypatch.setattr(vs, "_checkpoint_root", lambda: ckpt_root)
    monkeypatch.setattr(vs, "_require_cuda", lambda: "cpu")

    session = MagicMock()
    session.process_frame.return_value = (
        [np.array([[1.0, 2.0]], dtype=float)],
        [np.array([0.9], dtype=float)],
        [np.array([0, 0, 10, 10], dtype=float)],
    )
    session.render_overlay.side_effect = lambda img, *_a, **_k: img
    mock_session_cls.return_value = session

    out_dir = tmp_path / "out"
    vs.run_sapiens_on_video(video, out_dir, model="1b", stride=1, device="cpu")

    assert (out_dir / "sample_predictions.json").is_file()
    assert (out_dir / "sample_sapiens_vaila.csv").is_file()
    assert (out_dir / "sample_sapiens_overlay.mp4").is_file()
    assert (out_dir / "sample_markers.csv").is_file()
    assert (out_dir / "sapiens_vaila_center.csv").is_file()
    assert (out_dir / "sapiens_points.csv").is_file()
    assert (out_dir / "README_sapiens.txt").is_file()


def test_write_sapiens_biomechanics_csvs(tmp_path: Path) -> None:
    timeline = {
        0: [
            {
                "stable_id": 1,
                "bbox": [10.0, 20.0, 50.0, 100.0],
                "keypoints": [[30.0, 60.0], [35.0, 65.0]] + [[0.0, 0.0]] * 10,
                "keypoint_scores": [0.9, 0.85] + [0.8] * 10,
            }
        ],
        1: [
            {
                "stable_id": 1,
                "bbox": [12.0, 22.0, 52.0, 102.0],
                "keypoints": [[31.0, 61.0], [36.0, 66.0]] + [[0.0, 0.0]] * 10,
                "keypoint_scores": [0.9, 0.85] + [0.8] * 10,
            }
        ],
    }
    written = vs.write_sapiens_biomechanics_csvs(tmp_path, "clip", timeline, kpt_thr=0.3)
    assert len(written) >= 9
    assert (tmp_path / "clip_markers.csv").is_file()
    center = (tmp_path / "sapiens_vaila_center.csv").read_text(encoding="utf-8")
    assert "frame,x1,y1" in center
    assert "0,30.0000,60.0000" in center or "0,30.0000" in center
    assert (tmp_path / "sapiens_bbox_tracks.csv").is_file()
    bbox_txt = (tmp_path / "sapiens_bbox_tracks.csv").read_text(encoding="utf-8")
    assert "obj_id,x_px,y_px,w_px,h_px" in bbox_txt.replace(" ", "")


def test_write_getpixelvideo_pose_csv_wide(tmp_path: Path) -> None:
    n_kp = 4
    timeline = {
        0: [
            {
                "stable_id": 1,
                "bbox": [10.0, 20.0, 50.0, 100.0],
                "keypoints": [[30.0 + i, 60.0 + i] for i in range(n_kp)],
                "keypoint_scores": [0.9] * n_kp,
            }
        ],
    }
    written = vs.write_sapiens_getpixelvideo_pose_csvs(
        tmp_path, "clip", timeline, kpt_thr=0.3, keypoint_names=[f"joint{i}" for i in range(n_kp)]
    )
    assert len(written) == 2
    pose_path = tmp_path / "clip_id_01_sapiens_pose.csv"
    alias_path = tmp_path / "clip_getpixelvideo_pose.csv"
    assert pose_path.is_file()
    assert alias_path.is_file()
    header = pose_path.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("frame,")
    assert "joint0_x,joint0_y" in header
    assert "30.0000,60.0000" in pose_path.read_text(encoding="utf-8")


def test_sapiens_bbox_tracks_sam_schema(tmp_path: Path) -> None:
    timeline = {
        0: [
            {
                "stable_id": 2,
                "bbox": [10.0, 20.0, 50.0, 100.0],
                "keypoints": [[30.0, 60.0]],
                "keypoint_scores": [0.9],
            }
        ],
    }
    vs.write_sapiens_biomechanics_csvs(tmp_path, "clip", timeline, kpt_thr=0.3)
    import pandas as pd

    df = pd.read_csv(tmp_path / "sapiens_bbox_tracks.csv")
    assert list(df.columns) == ["frame", "obj_id", "x_px", "y_px", "w_px", "h_px", "score"]
    assert int(df.iloc[0]["obj_id"]) == 2
    assert float(df.iloc[0]["w_px"]) == 40.0
    assert float(df.iloc[0]["h_px"]) == 80.0


def test_frame_progress_interval() -> None:
    assert vs._frame_progress_interval(0) == 30
    assert vs._frame_progress_interval(100) == 5
    assert vs._frame_progress_interval(5000) == 60


def test_progress_quiet_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TQDM_DISABLE", raising=False)
    assert not vs._progress_quiet()
    monkeypatch.setenv("TQDM_DISABLE", "1")
    assert vs._progress_quiet()
    monkeypatch.setenv("TQDM_DISABLE", "true")
    assert vs._progress_quiet()
