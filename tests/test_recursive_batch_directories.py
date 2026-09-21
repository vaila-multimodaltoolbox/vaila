"""Tests for the --recursive/--depth directory-walker helpers added to the SAM
family (sam3sapiens2.py / sam3dinov3.py) and markerless_2d_analysis.py: they
must find every leaf directory with raw videos, skip already-processed
leaves, and never descend into (or return) a directory that is itself a prior
run's own output — the infinite-loop case batch mode must avoid.

Update Date: 20 September 2026
Version: 0.4.4
"""

from __future__ import annotations

from pathlib import Path

from vaila.markerless_2d_analysis import find_markerless_batch_directories
from vaila.sam3dinov3_visualize import find_run_directories as find_dinov3_run_directories
from vaila.sam3sapiens2 import find_batch_directories
from vaila.sam3sapiens2_visualize import find_run_directories as find_sapiens2_run_directories


def _touch_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_find_batch_directories_finds_leaves_and_skips_processed_output(
    tmp_path: Path,
) -> None:
    root = tmp_path
    _touch_video(root / "raw_leaf" / "clip1.mp4")
    _touch_video(root / "raw_leaf" / "clip2.mov")

    processed_leaf = root / "already_done"
    _touch_video(processed_leaf / "clip.mp4")
    output_dir = processed_leaf / "processed_sam3dinov3_20260917_194522"
    _touch_video(output_dir / "clip" / "clip_sam3dinov3_overlay.mp4")

    directories = find_batch_directories(root, -1)

    assert (root / "raw_leaf") in directories
    assert processed_leaf in directories
    assert output_dir not in directories
    assert not any(output_dir in d.parents or d == output_dir for d in directories)


def test_find_batch_directories_respects_depth(tmp_path: Path) -> None:
    root = tmp_path
    _touch_video(root / "l1.mp4")
    _touch_video(root / "a" / "l2.mp4")
    _touch_video(root / "a" / "b" / "l3.mp4")

    assert find_batch_directories(root, 0) == [root]
    depth1 = find_batch_directories(root, 1)
    assert root in depth1
    assert (root / "a") in depth1
    assert (root / "a" / "b") not in depth1
    depth_unlimited = find_batch_directories(root, -1)
    assert (root / "a" / "b") in depth_unlimited


def test_find_markerless_batch_directories_skips_prior_gui_and_cli_output(
    tmp_path: Path,
) -> None:
    root = tmp_path
    _touch_video(root / "raw" / "clip.mp4")

    gui_output = (
        root / "raw" / "mediapipe_resize_2x_filter_linear_butterworth_c10_fs100_20260920_143000"
    )
    _touch_video(gui_output / "clip" / "clip_mp_overlay.mp4")

    cli_output = root / "other" / "mediapipe_cli_20260920_150000"
    _touch_video(root / "other" / "clip.mp4")
    _touch_video(cli_output / "clip" / "clip_mp_overlay.mp4")

    directories = find_markerless_batch_directories(root, -1)

    assert (root / "raw") in directories
    assert (root / "other") in directories
    assert gui_output not in directories
    assert cli_output not in directories
    assert not any(gui_output in d.parents for d in directories)
    assert not any(cli_output in d.parents for d in directories)


def test_find_markerless_batch_directories_respects_depth(tmp_path: Path) -> None:
    root = tmp_path
    _touch_video(root / "l1.mp4")
    _touch_video(root / "a" / "l2.mp4")
    _touch_video(root / "a" / "b" / "l3.mp4")

    assert find_markerless_batch_directories(root, 0) == [root]
    depth1 = find_markerless_batch_directories(root, 1)
    assert root in depth1
    assert (root / "a") in depth1
    assert (root / "a" / "b") not in depth1
    depth_unlimited = find_markerless_batch_directories(root, -1)
    assert (root / "a" / "b") in depth_unlimited


def test_find_sam3sapiens2_run_directories_collects_runs_and_skips_visualized_output(
    tmp_path: Path,
) -> None:
    root = tmp_path
    run_dir = root / "s1_L_pre_ah" / "processed_sam3sapiens2_20260917_110023"
    _touch_video(run_dir / "clip" / "sam3" / "clip_sam3sapiens2_predictions.json")

    visualized = run_dir / "clip" / "clip_sam3sapiens2_visualized_id_01" / "clip_overlay.mp4"
    _touch_video(visualized)

    directories = find_sapiens2_run_directories(root, -1)

    assert run_dir in directories
    assert not any(d.name.endswith("_visualized_id_01") for d in directories)
    assert not any((run_dir / "clip") in d.parents and d != run_dir for d in directories)


def test_find_sam3sapiens2_run_directories_respects_depth(tmp_path: Path) -> None:
    root = tmp_path
    shallow = root / "processed_sam3sapiens2_20260917_110023"
    _touch_video(shallow / "clip" / "sam3" / "clip_sam3sapiens2_predictions.json")
    deep = root / "a" / "b" / "processed_sam3sapiens2_20260918_120000"
    _touch_video(deep / "clip" / "sam3" / "clip_sam3sapiens2_predictions.json")

    assert find_sapiens2_run_directories(root, 0) == []
    depth1 = find_sapiens2_run_directories(root, 1)
    assert shallow in depth1
    assert deep not in depth1
    depth_unlimited = find_sapiens2_run_directories(root, -1)
    assert shallow in depth_unlimited
    assert deep in depth_unlimited


def test_find_sam3dinov3_run_directories_collects_runs_and_skips_visualized_output(
    tmp_path: Path,
) -> None:
    root = tmp_path
    run_dir = root / "s1_L_pre_ah" / "processed_sam3dinov3_20260917_194522"
    (run_dir / "clip").mkdir(parents=True)
    (run_dir / "clip" / "clip_sam3dinov3_predictions.json.gz").write_bytes(b"")

    visualized = run_dir / "clip" / "clip_sam3dinov3_visualized_id_01" / "clip_overlay.mp4"
    _touch_video(visualized)

    directories = find_dinov3_run_directories(root, -1)

    assert run_dir in directories
    assert not any(d.name.endswith("_visualized_id_01") for d in directories)
    assert not any((run_dir / "clip") in d.parents and d != run_dir for d in directories)
