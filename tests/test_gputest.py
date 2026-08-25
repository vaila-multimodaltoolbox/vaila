"""Tests for GPU, PyTorch, CUDA, and AI Stack diagnostics (vaila/gputest.py).

Update Date: 24 August 2026
Version: 0.3.113
"""

from __future__ import annotations

import pytest

from vaila import gputest
from vaila.vaila_cli_hints import CLI_HINTS_BY_HANDLER
from vaila.vaila_cli_menu import VAILA_MENU_ENTRIES


def test_check_pytorch_and_cuda_returns_valid_result() -> None:
    result = gputest.check_pytorch_and_cuda()
    assert isinstance(result, gputest.DiagnosticResult)
    assert result.target == "PyTorch & CUDA Core"
    assert result.status in ("OK", "WARNING", "FAIL")
    assert len(result.details) > 0
    assert result.summary != ""


def test_check_markerless2d_yolo26_returns_valid_result() -> None:
    result = gputest.check_markerless2d_yolo26()
    assert isinstance(result, gputest.DiagnosticResult)
    assert "markerless2d_yolo26" in result.script_path
    assert result.status in ("OK", "WARNING", "FAIL")
    assert len(result.details) > 0


def test_check_yolov26track_returns_valid_result() -> None:
    result = gputest.check_yolov26track()
    assert isinstance(result, gputest.DiagnosticResult)
    assert "yolov26track" in result.script_path
    assert result.status in ("OK", "WARNING", "FAIL")
    assert len(result.details) > 0


def test_check_sam3sapiens2_returns_valid_result() -> None:
    result = gputest.check_sam3sapiens2()
    assert isinstance(result, gputest.DiagnosticResult)
    assert "sam3sapiens2" in result.script_path
    assert result.status in ("OK", "WARNING", "FAIL")
    assert len(result.details) > 0


def test_check_sam3dinov3_returns_valid_result() -> None:
    result = gputest.check_sam3dinov3()
    assert isinstance(result, gputest.DiagnosticResult)
    assert "sam3dinov3" in result.script_path
    assert result.status in ("OK", "WARNING", "FAIL")
    assert len(result.details) > 0


def test_run_gpu_diagnostics_aggregates_all_checks() -> None:
    results = gputest.run_gpu_diagnostics(verbose=False)
    assert isinstance(results, list)
    assert len(results) == 5
    targets = [r.target for r in results]
    assert "PyTorch & CUDA Core" in targets
    assert any("markerless2d_yolo26" in t for t in targets)
    assert any("yolov26track" in t for t in targets)
    assert any("sam3sapiens2" in t for t in targets)
    assert any("sam3dinov3" in t for t in targets)


def test_diagnostic_remediation_on_missing_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    r_sapiens = gputest.check_sam3sapiens2()
    assert r_sapiens.status == "FAIL"
    assert any("CUDA" in d for d in r_sapiens.details)
    assert len(r_sapiens.remediation) > 0

    r_sam3d = gputest.check_sam3dinov3()
    assert r_sam3d.status == "FAIL"
    assert any("CUDA" in d for d in r_sam3d.details)
    assert len(r_sam3d.remediation) > 0


def test_cli_menu_and_hints_registration() -> None:
    menu_codes = {entry.code: entry for entry in VAILA_MENU_ENTRIES}
    assert "GPUTEST" in menu_codes
    assert menu_codes["GPUTEST"].handler == "run_gpu_test"
    assert "run_gpu_test" in CLI_HINTS_BY_HANDLER


def test_vaila_app_has_gpu_test_handler() -> None:
    import importlib.util
    from pathlib import Path

    root_vaila = Path(__file__).resolve().parents[1] / "vaila.py"
    spec = importlib.util.spec_from_file_location("vaila_main", root_vaila)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module.Vaila, "run_gpu_test")
