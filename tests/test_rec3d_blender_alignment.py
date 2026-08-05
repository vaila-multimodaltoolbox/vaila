"""Tests for the Blender alignment fix in rec3d.py.

Root cause these guard against (found 2026-08-04 on real data): Blender's
BVH importer defaults to ``update_scene_fps=False`` and
``update_scene_duration=False``, so importing a 631-frame / 120 Hz capture
leaves the scene at 24 fps with frame_end=250 -- the BVH and OBJ mesh
sequence then play in slow motion and stop a third of the way through, while
an imported C3D (whose importer reads POINT:RATE) plays correctly. The
exported data was never wrong; only the Blender scene settings were. The
generated companion script now sets the scene rate and frame range
explicitly.

These are pure-Python tests (no bpy); they check the generator's arithmetic
and the emitted script's configuration.
"""

import shutil
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

try:
    from vaila.rec3d import (
        blender_scene_fps,
        generate_blender_companion_script,
        save_rec3d_as_bvh,
    )
except ImportError:  # standalone execution
    from rec3d import (  # ty: ignore[unresolved-import]
        blender_scene_fps,
        generate_blender_companion_script,
        save_rec3d_as_bvh,
    )


@pytest.mark.parametrize(
    ("rate", "expected_fps", "expected_base"),
    [
        (120.0, 120, 1.0),
        (100.0, 100, 1.0),
        (60.0, 60, 1.0),
        # NTSC-derived rates must come back as the standard fps/1.001 pair.
        (119.88012001, 120, 1.001),
        (29.97002997, 30, 1.001),
    ],
)
def test_blender_scene_fps_splits_rate_into_fps_and_base(rate, expected_fps, expected_base):
    fps, base = blender_scene_fps(rate)
    assert fps == expected_fps
    assert base == pytest.approx(expected_base, abs=1e-6)
    # The whole point: fps / fps_base must reproduce the true rate exactly.
    assert fps / base == pytest.approx(rate, rel=1e-9)


@pytest.mark.parametrize("bad_rate", [0, -5, None, "abc"])
def test_blender_scene_fps_falls_back_on_invalid_rate(bad_rate):
    fps, base = blender_scene_fps(bad_rate)
    assert fps > 0
    assert base > 0


def _make_rec3d_df(n_frames=631, n_markers=3):
    data = {"frame": np.arange(n_frames, dtype=float)}
    for m in range(1, n_markers + 1):
        for axis in ("x", "y", "z"):
            data[f"p{m}_{axis}"] = np.linspace(0.0, 1.0, n_frames)
    return pd.DataFrame(data)


def test_companion_script_sets_scene_rate_and_frame_range(tmp_path):
    script_path = generate_blender_companion_script(
        str(tmp_path),
        "rec3d_test",
        skeleton_json_path=None,
        point_rate=119.88012001,
        n_frames=631,
        mesh_dir="meshes_obj",
    )
    assert script_path is not None
    with open(script_path, encoding="utf-8") as fh:
        text = fh.read()

    assert "SCENE_FPS = 120" in text
    assert "FRAME_START = 1" in text
    assert "FRAME_END = 631" in text
    # The two importer flags whose False defaults caused the original bug.
    assert "update_scene_fps=True" in text
    assert "update_scene_duration=True" in text
    # Scene setup must exist and be invoked.
    assert "def setup_scene(" in text
    assert "scene.render.fps_base" in text
    assert "meshes_obj" in text


def test_companion_script_finds_its_files_after_the_run_folder_moves(tmp_path):
    """The script records absolute paths, so a moved or copied output folder
    used to leave it importing nothing -- or worse, silently importing the
    files still sitting at the old location while the user assumed they were
    looking at the new run. It must fall back to its own directory."""
    original = tmp_path / "original_run"
    (original / "meshes_obj").mkdir(parents=True)
    script_path = generate_blender_companion_script(
        str(original),
        "rec3d_test",
        skeleton_json_path=None,
        point_rate=120.0,
        n_frames=631,
        mesh_dir="meshes_obj",
    )
    text = Path(script_path).read_text(encoding="utf-8")
    assert "BVH_NAME = 'rec3d_test.bvh'" in text
    assert "MESH_DIR_NAME = 'meshes_obj'" in text
    assert "def _resolve(" in text

    fake_bpy = types.ModuleType("bpy")
    fake_bpy.data = types.SimpleNamespace(texts=[])
    fake_bpy.path = types.SimpleNamespace(abspath=lambda p: p)
    namespace = {"__file__": str(Path(script_path))}
    header, _, _ = text.partition("def setup_scene(")
    with mock.patch.dict(sys.modules, {"bpy": fake_bpy}):
        exec(compile(header, "<companion>", "exec"), namespace)  # noqa: S102

    moved = tmp_path / "moved_run"
    moved.mkdir()
    (moved / "meshes_obj").mkdir()
    (moved / "rec3d_test.bvh").write_text("HIERARCHY\n", encoding="utf-8")
    namespace["__file__"] = str(moved / "rec3d_test_blender_skeleton_viz.py")
    shutil.rmtree(original)  # the run folder was moved, not copied

    resolve = namespace["_resolve"]
    assert resolve(str(original / "rec3d_test.bvh"), "rec3d_test.bvh") == str(
        moved / "rec3d_test.bvh"
    )
    assert resolve(str(original / "meshes_obj"), "meshes_obj", is_dir=True) == str(
        moved / "meshes_obj"
    )


def test_companion_script_is_valid_python(tmp_path):
    """The emitted script is built by string concatenation -- make sure it
    always parses, so a template typo cannot ship a broken Blender script."""
    import ast

    script_path = generate_blender_companion_script(
        str(tmp_path),
        "rec3d_test",
        skeleton_json_path=None,
        point_rate=120.0,
        n_frames=250,
        mesh_dir=None,
    )
    with open(script_path, encoding="utf-8") as fh:
        ast.parse(fh.read())


def test_bvh_frame_time_precise_enough_for_fractional_rates(tmp_path):
    """A 6-decimal frame time turns 119.88012001 Hz into 119.875330 Hz on
    re-read; 9 decimals keeps the round-trip within a frame-accurate margin."""
    rate = 119.88012001
    save_rec3d_as_bvh(_make_rec3d_df(n_frames=10), str(tmp_path), "rec3d_test", rate, gui=False)

    bvh_text = (tmp_path / "rec3d_test.bvh").read_text(encoding="utf-8")
    frame_time_line = next(ln for ln in bvh_text.splitlines() if ln.startswith("Frame Time:"))
    frame_time = float(frame_time_line.split(":")[1])

    recovered_rate = 1.0 / frame_time
    assert recovered_rate == pytest.approx(rate, abs=1e-3), (
        f"BVH frame time {frame_time} reads back as {recovered_rate} Hz, not {rate} Hz"
    )


def test_bvh_frame_count_matches_dataframe(tmp_path):
    """BVH must declare and contain exactly one motion row per reconstructed
    frame -- the companion script sets scene frame_end from the same count."""
    n_frames = 631
    save_rec3d_as_bvh(
        _make_rec3d_df(n_frames=n_frames), str(tmp_path), "rec3d_test", 120.0, gui=False
    )

    lines = (tmp_path / "rec3d_test.bvh").read_text(encoding="utf-8").splitlines()
    declared = next(int(ln.split(":")[1]) for ln in lines if ln.startswith("Frames:"))
    motion_idx = lines.index("MOTION")
    actual_rows = len([ln for ln in lines[motion_idx + 3 :] if ln.strip()])

    assert declared == n_frames
    assert actual_rows == n_frames
