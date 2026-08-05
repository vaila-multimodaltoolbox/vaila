"""Tests for vaila/blender_viz.py — the Animation Blender button.

These are pure-Python: no ``bpy``, and Blender itself is never invoked
(``subprocess`` is monkeypatched). What they pin down is the plumbing that
decides *which* script gets handed to *which* executable, since getting either
wrong is silent — Blender simply opens an empty scene.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    from vaila import blender_viz
except ImportError:  # standalone execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from vaila import blender_viz


def _write_bvh(path, n_frames, rate):
    """Minimal BVH with the same MOTION header the exporter writes."""
    frame_time = 1.0 / rate
    path.write_text(
        "HIERARCHY\nROOT p1\n{\n}\n"
        f"MOTION\nFrames: {n_frames}\nFrame Time: {frame_time:.9f}\n"
        "0.000000 0.000000 0.000000\n",
        encoding="utf-8",
    )


def _write_rec3d_csv(path, n_markers=70, n_frames=5):
    data = {"frame": np.arange(n_frames, dtype=float)}
    for i in range(1, n_markers + 1):
        for axis in ("x", "y", "z"):
            data[f"p{i}_{axis}"] = np.full(n_frames, float(i))
    pd.DataFrame(data).to_csv(path, index=False)


@pytest.fixture
def run_dir(tmp_path):
    """A rec3d output folder with a BVH + CSV but no companion script."""
    directory = tmp_path / "vaila_rec3d_20260805_101010"
    directory.mkdir()
    _write_bvh(directory / "rec3d_20260805_101010.bvh", 631, 120.0)
    _write_rec3d_csv(directory / "rec3d_20260805_101010.csv")
    return directory


# ---------------------------------------------------------------------------
# BVH header parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rate", [120.0, 119.88012001, 60.0, 239.76024002])
def test_parse_bvh_header_recovers_frames_and_rate(tmp_path, rate):
    """A fractional NTSC-derived rate has to survive the frame-time round trip.

    The exporter writes 9 decimals precisely for this: at 6 decimals,
    119.88012001 Hz comes back as 119.875330 Hz and Blender ends up on a
    subtly wrong scene rate.
    """
    bvh = tmp_path / "trial.bvh"
    _write_bvh(bvh, 631, rate)
    n_frames, parsed_rate = blender_viz.parse_bvh_header(bvh)
    assert n_frames == 631
    assert parsed_rate == pytest.approx(rate, rel=1e-6)


def test_parse_bvh_header_returns_none_for_garbage(tmp_path):
    bvh = tmp_path / "broken.bvh"
    bvh.write_text("HIERARCHY\nROOT p1\n{\n}\n", encoding="utf-8")
    assert blender_viz.parse_bvh_header(bvh) is None
    assert blender_viz.parse_bvh_header(tmp_path / "missing.bvh") is None


# ---------------------------------------------------------------------------
# Skeleton preset inference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_markers", "expected"),
    [
        (17, "yolo_coco17.json"),
        (33, "mediapipe_pose33.json"),
        (70, "sam3dinov3_mhr70.json"),
        (308, "sapiens2_goliath308.json"),
    ],
)
def test_infer_skeleton_preset_matches_shipped_layouts(n_markers, expected):
    """Marker COUNT is the discriminator, not the max marker index.

    sapiens2_goliath308's connection list tops out at p63 despite the layout
    having 308 markers, so an index-based rule would misfile it as MHR70.
    """
    resolved = blender_viz.infer_skeleton_preset(n_markers)
    assert resolved is not None
    assert Path(resolved).name == expected
    assert Path(resolved).is_file()


def test_infer_skeleton_preset_unknown_count_falls_back_to_default():
    """An unrecognised layout must not block the launch — the generator has
    its own default connections."""
    assert blender_viz.infer_skeleton_preset(99) is None


# ---------------------------------------------------------------------------
# Companion-script resolution
# ---------------------------------------------------------------------------


def test_resolve_accepts_the_script_directly(tmp_path):
    script = tmp_path / "rec3d_x_blender_skeleton_viz.py"
    script.write_text("print('hi')\n", encoding="utf-8")
    assert blender_viz.resolve_companion_script(script) == str(script)


def test_resolve_rejects_a_non_python_file(tmp_path):
    other = tmp_path / "rec3d_x.bvh"
    other.write_text("HIERARCHY\n", encoding="utf-8")
    assert blender_viz.resolve_companion_script(other) is None


def test_resolve_missing_path_returns_none(tmp_path):
    assert blender_viz.resolve_companion_script(tmp_path / "nope") is None


def test_resolve_picks_the_newest_script_in_a_folder(tmp_path):
    older = tmp_path / "rec3d_a_blender_skeleton_viz.py"
    newer = tmp_path / "rec3d_b_blender_skeleton_viz.py"
    older.write_text("# a\n", encoding="utf-8")
    newer.write_text("# b\n", encoding="utf-8")
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))
    assert blender_viz.resolve_companion_script(tmp_path) == str(newer)


def test_resolve_regenerates_when_the_folder_has_no_script(run_dir):
    """Runs produced before the companion script existed must still work."""
    assert not list(run_dir.glob(f"*{blender_viz.COMPANION_SUFFIX}"))
    script_path = blender_viz.resolve_companion_script(run_dir)
    assert script_path is not None
    text = Path(script_path).read_text(encoding="utf-8")
    assert "FRAME_END = 631" in text
    assert "SCENE_FPS = 120" in text


def test_regenerate_forces_a_rebuild_over_an_existing_script(run_dir):
    """--regenerate is the fix for a script whose recorded paths went stale."""
    stale = run_dir / "rec3d_20260805_101010_blender_skeleton_viz.py"
    stale_text = "# stale, points at a directory that no longer exists\n"
    stale.write_text(stale_text, encoding="utf-8")
    script_path = blender_viz.resolve_companion_script(run_dir, regenerate=True)
    rebuilt = Path(script_path).read_text(encoding="utf-8")
    assert rebuilt != stale_text
    assert "FRAME_END = 631" in rebuilt


def test_regenerate_picks_up_the_mesh_directory(run_dir):
    (run_dir / "meshes_obj").mkdir()
    script_path = blender_viz.regenerate_companion_script(run_dir)
    assert "meshes_obj" in Path(script_path).read_text(encoding="utf-8")


def test_find_run_file_base_prefers_a_bvh_with_a_matching_csv(run_dir):
    assert blender_viz.find_run_file_base(run_dir) == "rec3d_20260805_101010"


# ---------------------------------------------------------------------------
# User config
# ---------------------------------------------------------------------------


def test_config_round_trips_through_the_home_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert blender_viz.load_vaila_config() == {}
    assert blender_viz.remember_blender_executable("/opt/blender/blender")
    assert blender_viz.vaila_config_path().is_file()
    assert blender_viz.load_vaila_config()["blender"]["executable"] == "/opt/blender/blender"


def test_unreadable_config_is_ignored_rather_than_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    path = blender_viz.vaila_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("this is not = valid = toml [[[\n", encoding="utf-8")
    assert blender_viz.load_vaila_config() == {}


# ---------------------------------------------------------------------------
# Blender discovery
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_blender(tmp_path, monkeypatch):
    """Make only the paths we name look like a working Blender."""
    working = set()

    def fake_version(executable):
        return "5.2.0" if str(executable) in working else None

    monkeypatch.setattr(blender_viz, "blender_version", fake_version)
    monkeypatch.setattr(blender_viz.shutil, "which", lambda _name: None)
    monkeypatch.setattr(blender_viz, "candidate_blender_paths", list)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("VAILA_BLENDER", raising=False)
    return working


def test_explicit_path_wins(fake_blender, monkeypatch):
    fake_blender.update({"/explicit/blender", "/env/blender"})
    monkeypatch.setenv("VAILA_BLENDER", "/env/blender")
    assert blender_viz.find_blender_executable(explicit="/explicit/blender") == "/explicit/blender"


def test_env_var_beats_the_saved_config(fake_blender, monkeypatch):
    fake_blender.update({"/env/blender", "/saved/blender"})
    blender_viz.remember_blender_executable("/saved/blender")
    monkeypatch.setenv("VAILA_BLENDER", "/env/blender")
    assert blender_viz.find_blender_executable() == "/env/blender"


def test_saved_config_is_used_when_nothing_else_is_set(fake_blender):
    fake_blender.add("/saved/blender")
    blender_viz.remember_blender_executable("/saved/blender")
    assert blender_viz.find_blender_executable() == "/saved/blender"


def test_stale_saved_path_falls_through_to_autodetection(fake_blender, monkeypatch):
    """A remembered path that no longer works must not dead-end the button."""
    blender_viz.remember_blender_executable("/uninstalled/blender")
    fake_blender.add("/usr/bin/blender")
    monkeypatch.setattr(blender_viz, "candidate_blender_paths", lambda: ["/usr/bin/blender"])
    assert blender_viz.find_blender_executable() == "/usr/bin/blender"


def test_headless_returns_none_when_blender_is_absent(fake_blender):
    assert blender_viz.find_blender_executable(gui=False) is None


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------


def test_build_argv_runs_the_script_on_startup():
    argv = blender_viz.build_blender_argv("/runs/viz.py", "/snap/bin/blender")
    assert argv == ["/snap/bin/blender", "--python", "/runs/viz.py"]


def test_build_argv_background_adds_the_headless_flag():
    argv = blender_viz.build_blender_argv("/runs/viz.py", "/snap/bin/blender", background=True)
    assert argv == ["/snap/bin/blender", "-b", "--python", "/runs/viz.py"]


def test_launch_does_not_block_the_caller(monkeypatch):
    """The vailá Tk loop has to keep running while Blender is open."""
    seen = {}

    def fake_popen(argv, *args, **kwargs):
        seen["argv"] = argv
        return "process"

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    assert blender_viz.launch_blender("/runs/viz.py", "/snap/bin/blender") == "process"
    assert seen["argv"] == ["/snap/bin/blender", "--python", "/runs/viz.py"]


def test_cli_mirror_is_copy_pasteable():
    command = blender_viz.format_blender_viz_cli(
        "/runs/vaila_rec3d_1", "/snap/bin/blender", background=True, regenerate=True
    )
    assert command == (
        "uv run python -m vaila.blender_viz -i /runs/vaila_rec3d_1 "
        "--blender /snap/bin/blender --regenerate --background"
    )
