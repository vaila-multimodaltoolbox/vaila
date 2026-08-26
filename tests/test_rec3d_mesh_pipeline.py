"""
Project: vailá
Script: test_rec3d_mesh_pipeline.py
Update Date: 26 August 2026
Version: 0.3.116

Description:
    Unit tests for vaila/rec3d_mesh_pipeline.py's TOML manifest validation
    (load_config) and vaila/gpu_subprocess.py's ensure_cuda_nvrtc_env() pure
    path-patching logic. No GPU/torch required.
"""

import os

import pytest
import toml

from vaila.gpu_subprocess import ensure_cuda_nvrtc_env
from vaila.rec3d_mesh_pipeline import MIN_CAMERAS, PipelineError, load_config


def _write_manifest(tmp_path, data):
    path = tmp_path / "manifest.toml"
    path.write_text(toml.dumps(data))
    return path


def _valid_camera(tmp_path, name):
    video = tmp_path / f"{name}.mp4"
    dlt3d = tmp_path / f"{name}.dlt3d"
    results = tmp_path / f"{name}_results"
    video.write_bytes(b"")
    dlt3d.write_text("0\n")
    results.mkdir()
    return {
        "video": str(video),
        "dlt3d": str(dlt3d),
        "sapiens2_results": str(results),
    }


def test_load_config_valid_two_cameras(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        {
            "output_dir": str(tmp_path / "out"),
            "camera": [_valid_camera(tmp_path, "c1"), _valid_camera(tmp_path, "c2")],
        },
    )
    config = load_config(manifest)
    assert len(config.cameras) == 2
    assert config.export_mesh == "obj"
    assert config.overwrite is False


def test_load_config_rejects_single_camera(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        {
            "output_dir": str(tmp_path / "out"),
            "camera": [_valid_camera(tmp_path, "c1")],
        },
    )
    with pytest.raises(PipelineError, match=f"at least {MIN_CAMERAS} cameras"):
        load_config(manifest)


def test_load_config_rejects_zero_cameras(tmp_path):
    manifest = _write_manifest(tmp_path, {"output_dir": str(tmp_path / "out")})
    with pytest.raises(PipelineError, match="at least"):
        load_config(manifest)


def test_load_config_missing_output_dir(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        {"camera": [_valid_camera(tmp_path, "c1"), _valid_camera(tmp_path, "c2")]},
    )
    with pytest.raises(PipelineError, match="output_dir"):
        load_config(manifest)


def test_load_config_missing_video_file(tmp_path):
    cam1 = _valid_camera(tmp_path, "c1")
    cam2 = _valid_camera(tmp_path, "c2")
    cam2["video"] = str(tmp_path / "does_not_exist.mp4")
    manifest = _write_manifest(
        tmp_path, {"output_dir": str(tmp_path / "out"), "camera": [cam1, cam2]}
    )
    with pytest.raises(PipelineError, match="video not found"):
        load_config(manifest)


def test_load_config_missing_dlt3d_file(tmp_path):
    cam1 = _valid_camera(tmp_path, "c1")
    cam2 = _valid_camera(tmp_path, "c2")
    cam2["dlt3d"] = str(tmp_path / "missing.dlt3d")
    manifest = _write_manifest(
        tmp_path, {"output_dir": str(tmp_path / "out"), "camera": [cam1, cam2]}
    )
    with pytest.raises(PipelineError, match="dlt3d file not found"):
        load_config(manifest)


def test_load_config_missing_results_dir(tmp_path):
    cam1 = _valid_camera(tmp_path, "c1")
    cam2 = _valid_camera(tmp_path, "c2")
    cam2["sapiens2_results"] = str(tmp_path / "no_such_dir")
    manifest = _write_manifest(
        tmp_path, {"output_dir": str(tmp_path / "out"), "camera": [cam1, cam2]}
    )
    with pytest.raises(PipelineError, match="results dir not found"):
        load_config(manifest)


def test_load_config_requires_results_or_sam(tmp_path):
    cam1 = _valid_camera(tmp_path, "c1")
    cam2 = _valid_camera(tmp_path, "c2")
    del cam2["sapiens2_results"]
    manifest = _write_manifest(
        tmp_path, {"output_dir": str(tmp_path / "out"), "camera": [cam1, cam2]}
    )
    with pytest.raises(PipelineError, match="sapiens2_results.*sam_results"):
        load_config(manifest)


def test_load_config_rejects_bad_export_mesh(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        {
            "output_dir": str(tmp_path / "out"),
            "export_mesh": "stl",
            "camera": [_valid_camera(tmp_path, "c1"), _valid_camera(tmp_path, "c2")],
        },
    )
    with pytest.raises(PipelineError, match="export_mesh"):
        load_config(manifest)


def test_load_config_accepts_sam_results_alias(tmp_path):
    cam1 = _valid_camera(tmp_path, "c1")
    cam2 = _valid_camera(tmp_path, "c2")
    cam2["sam_results"] = cam2.pop("sapiens2_results")
    manifest = _write_manifest(
        tmp_path, {"output_dir": str(tmp_path / "out"), "camera": [cam1, cam2]}
    )
    config = load_config(manifest)
    assert config.cameras[1].sam_results is not None
    assert config.cameras[1].sapiens2_results is None


def test_ensure_cuda_nvrtc_env_noop_when_dir_missing(tmp_path, monkeypatch):
    # Point purelib at an empty directory (no nvidia/cu13/lib) regardless of
    # what the real venv running this test happens to have installed.
    monkeypatch.setattr(
        "vaila.gpu_subprocess.sysconfig.get_paths", lambda: {"purelib": str(tmp_path)}
    )
    env = {"LD_LIBRARY_PATH": "/some/existing/path", "PATH": "/usr/bin"}
    patched = ensure_cuda_nvrtc_env(env)
    assert patched == env
    assert patched is not env  # pure function: never returns the same object


def test_ensure_cuda_nvrtc_env_prepends_when_present(tmp_path, monkeypatch):
    purelib = tmp_path / "site-packages"
    nvrtc_dir = purelib / "nvidia" / "cu13" / "lib"
    nvrtc_dir.mkdir(parents=True)
    (nvrtc_dir / "libnvrtc-builtins.so.13.0").write_bytes(b"")
    monkeypatch.setattr(
        "vaila.gpu_subprocess.sysconfig.get_paths", lambda: {"purelib": str(purelib)}
    )
    env = {"LD_LIBRARY_PATH": "/other/path"}
    patched = ensure_cuda_nvrtc_env(env)
    assert patched["LD_LIBRARY_PATH"] == f"{nvrtc_dir}{os.pathsep}/other/path"
    assert env["LD_LIBRARY_PATH"] == "/other/path"  # input untouched


def test_ensure_cuda_nvrtc_env_idempotent(tmp_path, monkeypatch):
    purelib = tmp_path / "site-packages"
    nvrtc_dir = purelib / "nvidia" / "cu13" / "lib"
    nvrtc_dir.mkdir(parents=True)
    (nvrtc_dir / "libnvrtc-builtins.so.13.0").write_bytes(b"")
    monkeypatch.setattr(
        "vaila.gpu_subprocess.sysconfig.get_paths", lambda: {"purelib": str(purelib)}
    )
    env = {"LD_LIBRARY_PATH": str(nvrtc_dir)}
    patched = ensure_cuda_nvrtc_env(env)
    assert patched["LD_LIBRARY_PATH"] == str(nvrtc_dir)  # not duplicated
