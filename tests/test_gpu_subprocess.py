"""Tests for CUDA worker process isolation and recovery barriers.

Update Date: 01 August 2026
Version: 0.3.89
"""

from __future__ import annotations

import sys

import pytest

from vaila import gpu_subprocess as gpu


def test_wait_for_gpu_memory_recovery_waits_for_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readings = iter((300, 920))
    monkeypatch.setattr(gpu, "gpu_free_memory_mib", lambda *_a, **_k: next(readings))
    monkeypatch.setattr(gpu.time, "sleep", lambda _seconds: None)
    assert gpu.wait_for_gpu_memory_recovery(1000, tolerance_mib=100) == 920


def test_wait_for_gpu_memory_recovery_rejects_persistent_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter((0.0, 1.0))
    monkeypatch.setattr(gpu, "gpu_free_memory_mib", lambda *_a, **_k: 100)
    monkeypatch.setattr(gpu.time, "monotonic", lambda: next(ticks))
    with pytest.raises(gpu.GpuMemoryRecoveryError, match="refusing to start the next GPU stage"):
        gpu.wait_for_gpu_memory_recovery(1000, timeout_seconds=0.5, tolerance_mib=100)


def test_run_isolated_gpu_subprocess_preserves_return_code_without_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu, "gpu_free_memory_mib", lambda *_a, **_k: None)
    result = gpu.run_isolated_gpu_subprocess([sys.executable, "-c", "raise SystemExit(7)"])
    assert result.returncode == 7
    assert result.baseline_free_mib is None
    assert result.final_free_mib is None
