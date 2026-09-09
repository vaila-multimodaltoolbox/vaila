"""Tests for vaila/vaila_env.py environment, dependencies, and AI models inspector."""

from __future__ import annotations

import pytest

from vaila.vaila_env import (
    CORE_PACKAGES,
    VAILA_UPDATE_DATE,
    VAILA_VERSION,
    format_env_summary,
    format_size,
    get_activation_command,
    get_ai_models_status,
    get_hardware_info,
    get_huggingface_info,
    get_ipython_guide,
    get_package_version,
    get_system_info,
    get_vaila_info,
)
from vaila.vaila_env import (
    main as vaila_env_main,
)


def test_get_vaila_info():
    v_info = get_vaila_info()
    assert v_info["version"] == VAILA_VERSION
    assert v_info["update_date"] == VAILA_UPDATE_DATE
    assert "branch" in v_info
    assert "commit" in v_info
    assert "root_dir" in v_info


def test_format_size():
    assert format_size(0) == "0 B"
    assert format_size(500) == "500.0 B"
    assert format_size(1024) == "1.0 KB"
    assert format_size(1048576) == "1.0 MB"
    assert format_size(1073741824) == "1.0 GB"


def test_get_hardware_info():
    hw = get_hardware_info()
    assert "nvidia_smi_available" in hw
    assert isinstance(hw["smi_gpus"], list)
    assert "torch_cuda_available" in hw
    assert "gpu_summary" in hw
    assert isinstance(hw["gpu_summary"], str)


def test_get_huggingface_info():
    hf = get_huggingface_info()
    assert "installed" in hf
    assert "version" in hf
    assert "has_token" in hf
    assert "username" in hf
    assert "cache_dir" in hf


def test_get_ai_models_status():
    models = get_ai_models_status()
    assert "models_dir" in models
    for key in ("sam3", "sam_3d_dinov3", "sapiens2", "mediapipe", "yolo", "reid"):
        assert key in models
        fam = models[key]
        assert "title" in fam
        assert "items" in fam
        assert isinstance(fam["items"], list)
        for it in fam["items"]:
            assert "name" in it
            assert "status" in it
            assert "size_str" in it
            assert "exists" in it


def test_get_system_info():
    info = get_system_info()
    assert "os" in info
    assert "python_version" in info
    assert "python_path" in info
    assert "in_venv" in info
    assert "has_cuda" in info
    assert "hardware" in info
    assert info["python_version"].startswith("3.")


def test_get_package_version():
    # Numpy and ezc3d are core dependencies
    ver, is_installed = get_package_version("numpy")
    assert is_installed is True
    assert ver != "Not installed"

    ver_ez, is_installed_ez = get_package_version("ezc3d")
    assert is_installed_ez is True
    assert ver_ez == "1.7.2"

    # Non-existent package
    ver_fake, is_installed_fake = get_package_version("fake_nonexistent_pkg_xyz")
    assert is_installed_fake is False
    assert ver_fake == "Not installed"


def test_core_dependencies():
    names = [pkg for pkg, _ in CORE_PACKAGES]
    assert "ezc3d" in names
    assert "numpy" in names
    assert "pandas" in names
    assert "scipy" in names
    assert "huggingface_hub" in names


def test_format_env_summary():
    summary = format_env_summary()
    assert "vailá - Environment" in summary
    assert VAILA_VERSION in summary
    assert VAILA_UPDATE_DATE in summary
    assert "Hugging Face Hub Status:" in summary
    assert "AI Tracking Models & Checkpoints Inventory" in summary
    assert "SAM 3" in summary
    assert "SAM-3D-DINOv3" in summary
    assert "Sapiens2" in summary
    assert "MediaPipe" in summary
    assert "YOLO" in summary
    assert "ezc3d" in summary
    assert "How to activate .venv on your system:" in summary
    assert "How to run IPython for calculations and coding:" in summary


def test_get_activation_command_and_ipython_guide():
    cmd = get_activation_command()
    assert "activate" in cmd.lower()

    guide = get_ipython_guide()
    assert "ipython" in guide.lower()
    assert "ezc3d" in guide


def test_vaila_env_cli(capsys: pytest.CaptureFixture[str]):
    res = vaila_env_main()
    assert res == 0
    captured = capsys.readouterr()
    assert "vailá - Environment" in captured.out
    assert VAILA_VERSION in captured.out


def test_vaila_env_gui_builds():
    import tkinter as tk

    from vaila.vaila_env import VailaEnvGUI

    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:
        pytest.skip("No display available for Tkinter GUI test")

    try:
        gui = VailaEnvGUI(root)
        assert gui.title() == "vailá - Imagination & Environment Info"
        gui.destroy()
    finally:
        root.destroy()
