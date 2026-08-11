#!/usr/bin/env python3
"""
===============================================================================
bin/verify_cuda_libs.py
===============================================================================
Update Date: 11 August 2026
Version: 0.3.104

Verifies that installed NVIDIA CUDA wheels (``nvidia-*-cu12``) and the
PyTorch stack (``torch``, ``torchvision``, ``torchaudio``, ``triton``) have
every file their own installed RECORD on disk — catches a real corruption
class where a package reports as installed (dist-info present, ``uv sync``
considers it satisfied and skips it) but its large binary payload
(``.so``/``.dll``) is missing, so ``import torch`` fails with e.g.:

    ImportError: libcusparseLt.so.0: cannot open shared object file
    ImportError: libnvshmem_host.so.3: cannot open shared object file

Observed in production on a real RTX 3090 machine (2026-08-10): three
separate ``nvidia-*-cu12`` wheels (``cusparselt``, ``nvshmem``, ``cudnn``)
were simultaneously missing their ``lib/*.so`` payload while their
``.dist-info`` metadata was intact, so plain ``uv sync`` reported nothing to
install/reinstall — the breakage is invisible to uv's own bookkeeping and
only surfaces as an ``ImportError`` at ``import torch`` time. Likely causes:
an interrupted extraction, a hardlink broken by external cache pruning (uv
hardlinks from its cache into ``.venv`` when both live on the same
filesystem — see the "Failed to hardlink files" warning), or the disk
running out of space mid-install without a loud error.

Usage:
    uv run python bin/verify_cuda_libs.py            # human-readable report; exit 1 if broken
    uv run python bin/verify_cuda_libs.py --quiet     # print only broken package names, one per line (for scripting)

Exit codes:
    0  every watched package has all its recorded files on disk
    1  one or more watched packages are missing files
"""

from __future__ import annotations

import importlib.metadata as im
import sys
from pathlib import Path

# Packages known to ship large binary payloads (.so/.dll) as part of the
# CUDA / PyTorch wheel stack — these are the ones observed to go missing
# after a corrupted `uv sync` (partial hardlink / interrupted extraction /
# disk full mid-install). Checking only these keeps the scan fast and avoids
# false positives from unrelated packages with unusual RECORD layouts.
WATCHED_PREFIXES = ("nvidia-",)
WATCHED_NAMES = {"torch", "torchvision", "torchaudio", "triton"}


def _is_watched(name: str) -> bool:
    lname = name.lower()
    return lname.startswith(WATCHED_PREFIXES) or lname in WATCHED_NAMES


def find_broken_packages() -> list[tuple[str, int, int]]:
    """Return (name, missing_count, total_count) for every watched package
    that has at least one recorded file missing on disk."""
    broken: list[tuple[str, int, int]] = []
    for dist in im.distributions():
        name = dist.metadata.get("Name") or dist.name
        if not name or not _is_watched(name):
            continue
        files = dist.files or []
        missing = 0
        for f in files:
            try:
                path = Path(str(dist.locate_file(f)))
            except Exception:
                continue
            if not path.exists():
                missing += 1
        if missing:
            broken.append((name, missing, len(files)))
    return broken


def main(argv: list[str]) -> int:
    quiet = "--quiet" in argv
    broken = find_broken_packages()

    if not broken:
        if not quiet:
            print("OK: all NVIDIA/PyTorch wheel payloads present.", file=sys.stderr)
        return 0

    for name, missing, total in broken:
        if quiet:
            print(name)
        else:
            print(
                f"BROKEN: {name} -- {missing}/{total} recorded files missing on disk",
                file=sys.stderr,
            )

    if not quiet:
        reinstall_flags = " ".join(f"--reinstall-package {n}" for n, _, _ in broken)
        print(
            "\nThese packages report as installed (dist-info present) but are "
            "missing their actual files on disk. Fix (reinstalls only the "
            "broken packages, not the whole environment):\n"
            f"  uv sync {reinstall_flags} --extra gpu ...  # keep your usual --extra flags\n"
            "\nIf it recurs right after: check disk space (`df -h`) and, if the uv "
            "cache and .venv live on different filesystems, `export UV_LINK_MODE=copy`.",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
