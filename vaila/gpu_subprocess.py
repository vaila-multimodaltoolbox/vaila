"""
Project: vailá
Script: gpu_subprocess.py
Authors: Paulo Santiago et al.
Update Date: 01 August 2026
Version: 0.3.89

Description:
    Process-group isolation and GPU-memory recovery barriers for CUDA workers.
    This internal helper avoids cascading OOM failures when a worker exits but
    leaves descendant processes or a CUDA context alive.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GpuSubprocessResult:
    """Result of an isolated GPU worker after the recovery barrier passed."""

    returncode: int
    baseline_free_mib: int | None
    final_free_mib: int | None


class GpuMemoryRecoveryError(RuntimeError):
    """Raised when a worker ended but its GPU memory was not reclaimed."""


def _physical_gpu_index(device: int, env: Mapping[str, str] | None = None) -> str:
    """Map a logical CUDA device to the physical index visible to nvidia-smi."""
    source = os.environ if env is None else env
    visible = source.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        entries = [part.strip() for part in visible.split(",") if part.strip()]
        if 0 <= int(device) < len(entries):
            return entries[int(device)]
    return str(int(device))


def gpu_free_memory_mib(
    device: int = 0,
    *,
    env: Mapping[str, str] | None = None,
) -> int | None:
    """Return free VRAM without importing torch or creating a CUDA context."""
    gpu_index = _physical_gpu_index(device, env)
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                gpu_index,
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
            env=dict(env) if env is not None else None,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    first = completed.stdout.strip().splitlines()
    if not first:
        return None
    try:
        return int(float(first[0].strip()))
    except ValueError:
        return None


def popen_process_group_kwargs() -> dict[str, Any]:
    """Keyword arguments that place a worker in its own process group."""
    if os.name == "posix":
        return {"start_new_session": True}
    creationflags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    return {"creationflags": creationflags} if creationflags else {}


def terminate_process_tree(proc: subprocess.Popen[Any], *, grace_seconds: float = 2.0) -> None:
    """Terminate descendants that survived their direct worker process."""
    if os.name == "posix":
        pgid = proc.pid
        try:
            os.killpg(pgid, 0)
        except (ProcessLookupError, PermissionError):
            return
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGTERM)
        deadline = time.monotonic() + max(0.0, float(grace_seconds))
        while time.monotonic() < deadline:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return
            except PermissionError:
                break
            time.sleep(0.1)
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)
        return

    # On Windows, taskkill /T is the standard way to include descendants. It
    # can report a non-zero code when the tree is already gone; that is benign.
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True,
            check=False,
            timeout=max(2.0, float(grace_seconds) + 1.0),
        )


def wait_for_gpu_memory_recovery(
    baseline_free_mib: int | None,
    *,
    device: int = 0,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float = 45.0,
    tolerance_mib: int = 512,
    poll_seconds: float = 0.5,
) -> int | None:
    """Wait until free VRAM returns close to its pre-worker baseline."""
    if baseline_free_mib is None:
        return None
    target = max(0, int(baseline_free_mib) - max(0, int(tolerance_mib)))
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    last: int | None = None
    while True:
        last = gpu_free_memory_mib(device, env=env)
        if last is not None and last >= target:
            return last
        if time.monotonic() >= deadline:
            raise GpuMemoryRecoveryError(
                "GPU memory did not recover after the isolated worker exited: "
                f"baseline={baseline_free_mib} MiB, current={last} MiB, "
                f"required>={target} MiB after {timeout_seconds:.0f}s. "
                "A descendant process or CUDA context is still alive; refusing "
                "to start the next GPU stage and cascade into OOM."
            )
        time.sleep(max(0.05, float(poll_seconds)))


def run_isolated_gpu_subprocess(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    device: int = 0,
    cleanup_timeout_seconds: float = 45.0,
    cleanup_tolerance_mib: int = 512,
    descendant_grace_seconds: float = 2.0,
    log: Callable[[str], None] | None = None,
    **popen_kwargs: Any,
) -> GpuSubprocessResult:
    """Run one CUDA worker in a process group and verify VRAM reclamation.

    The worker's return code is preserved. A cleanup failure raises
    :class:`GpuMemoryRecoveryError` even when the worker returned zero, because
    launching another model in that state would create a misleading cascade.
    """
    child_env = dict(env) if env is not None else os.environ.copy()
    baseline = gpu_free_memory_mib(device, env=child_env)
    group_kwargs = popen_process_group_kwargs()
    for key, value in group_kwargs.items():
        popen_kwargs.setdefault(key, value)
    proc = subprocess.Popen(list(cmd), env=child_env, **popen_kwargs)
    try:
        returncode = int(proc.wait())
    except BaseException:
        with contextlib.suppress(Exception):
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=max(1.0, descendant_grace_seconds))
        terminate_process_tree(proc, grace_seconds=descendant_grace_seconds)
        raise

    terminate_process_tree(proc, grace_seconds=descendant_grace_seconds)
    final_free = wait_for_gpu_memory_recovery(
        baseline,
        device=device,
        env=child_env,
        timeout_seconds=cleanup_timeout_seconds,
        tolerance_mib=cleanup_tolerance_mib,
    )
    if log is not None and baseline is not None and final_free is not None:
        log(f"GPU recovery barrier passed: {final_free} MiB free (baseline {baseline} MiB)")
    return GpuSubprocessResult(returncode, baseline, final_free)
