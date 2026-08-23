"""
===============================================================================
update_checker.py
===============================================================================
Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 21 August 2026
Update Date: 23 August 2026
Version: 0.3.110

Description:
------------
Lightweight, stdlib-only GitHub "main branch" update checker for vailá.

vailá is installed/updated via a one-line install script (see README.md and
install_vaila_linux.sh / install_vaila_mac.sh / install_vaila_win.ps1), not a
package manager, so the GUI has no other way to learn a newer version landed
on GitHub. This module:

  - Reads the installed version from the local pyproject.toml (single source
    of truth already used by vaila.py's header/banner).
  - Fetches the version declared in pyproject.toml on the GitHub `main`
    branch (raw.githubusercontent.com, no auth, no API rate limit).
  - Compares them and reports whether an update is available, plus the
    OS-specific one-line command to re-run to update.
  - Caches the last check (~/.vaila/update_check_cache.json) so vailá does
    not hit GitHub on every launch, and remembers a "skip this version".

All network I/O fails soft (returns None / checked=False) - no internet
access must never block or crash the GUI. Call check_for_updates_async()
from a background thread; deliver its result back to the GUI thread via a
queue polled with Tk's `after` (Tkinter is not thread-safe).
===============================================================================
"""

from __future__ import annotations

import json
import platform
import threading
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO_URL = "https://github.com/vaila-multimodaltoolbox/vaila"
RAW_PYPROJECT_URL = (
    "https://raw.githubusercontent.com/vaila-multimodaltoolbox/vaila/main/pyproject.toml"
)
CACHE_FILE = Path.home() / ".vaila" / "update_check_cache.json"
CHECK_INTERVAL_SECONDS = 20 * 60 * 60  # ~20h between automatic checks
REQUEST_TIMEOUT = 4.0  # seconds

# Exact one-liners from README.md "Install vaila with a single command!"
INSTALL_COMMANDS = {
    "Linux": (
        "wget -qO- https://raw.githubusercontent.com/vaila-multimodaltoolbox/vaila/main/"
        "install_vaila_linux.sh | bash"
    ),
    "Darwin": (
        '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/vaila-multimodaltoolbox/'
        'vaila/main/install_vaila_mac.sh)"'
    ),
    "Windows": (
        "[Net.ServicePointManager]::SecurityProtocol = "
        "[Net.ServicePointManager]::SecurityProtocol -bor 3072; "
        "irm https://raw.githubusercontent.com/vaila-multimodaltoolbox/vaila/main/"
        "install_vaila_win.ps1 | iex"
    ),
}


def get_install_command(system: str | None = None) -> str:
    """OS-specific one-line (re)install/update command, matching README.md."""
    system = system or platform.system()
    return INSTALL_COMMANDS.get(system, INSTALL_COMMANDS["Linux"])


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_local_version() -> str | None:
    """Installed version, read from the local pyproject.toml."""
    pyproject_path = _project_root() / "pyproject.toml"
    try:
        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)
        return data["project"]["version"]
    except Exception:
        return None


def fetch_remote_version(timeout: float = REQUEST_TIMEOUT) -> str | None:
    """Version declared in pyproject.toml on the GitHub `main` branch, or None."""
    try:
        req = urllib.request.Request(
            RAW_PYPROJECT_URL, headers={"User-Agent": "vaila-update-checker"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https URL)
            text = resp.read().decode("utf-8")
        data = tomllib.loads(text)
        return data["project"]["version"]
    except (urllib.error.URLError, TimeoutError, OSError, tomllib.TOMLDecodeError, KeyError):
        return None
    except Exception:
        return None


def parse_version(version: str) -> tuple[int, ...]:
    """'0.3.110' -> (0, 3, 110); tolerant of a non-numeric suffix on the last
    chunk (e.g. '0.3.110rc1' -> (0, 3, 110)), by reading only the leading
    digit run of each chunk rather than every digit in it."""
    parts = []
    for chunk in version.split("."):
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def is_newer(remote: str, local: str) -> bool:
    return parse_version(remote) > parse_version(local)


def _load_cache() -> dict:
    try:
        with CACHE_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(data: dict) -> None:
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass  # cache is a convenience, never fatal


@dataclass
class UpdateCheckResult:
    checked: bool  # True if we have a confirmed remote_version (fresh or cached)
    local_version: str | None = None
    remote_version: str | None = None
    update_available: bool = False
    error: str | None = None


def check_for_updates(force: bool = False) -> UpdateCheckResult:
    """Synchronous check (does network I/O) - call from a background thread.

    Respects the on-disk cache unless force=True (manual "Check for Updates").
    """
    cache = _load_cache()
    now = time.time()
    local_version = get_local_version()

    fresh_enough = not force and (now - cache.get("last_check", 0)) < CHECK_INTERVAL_SECONDS
    if fresh_enough:
        remote_version = cache.get("last_remote_version")
        if remote_version and local_version:
            available = is_newer(remote_version, local_version) and remote_version != cache.get(
                "skip_version"
            )
            return UpdateCheckResult(
                checked=True,
                local_version=local_version,
                remote_version=remote_version,
                update_available=available,
            )

    remote_version = fetch_remote_version()
    cache["last_check"] = now
    if remote_version:
        cache["last_remote_version"] = remote_version
    _save_cache(cache)

    if remote_version is None:
        return UpdateCheckResult(
            checked=False, local_version=local_version, error="offline_or_unreachable"
        )

    update_available = local_version is not None and is_newer(remote_version, local_version)
    if update_available and remote_version == cache.get("skip_version"):
        update_available = False

    return UpdateCheckResult(
        checked=True,
        local_version=local_version,
        remote_version=remote_version,
        update_available=update_available,
    )


def skip_version(version: str) -> None:
    """Remember the user dismissed the notice for this remote version."""
    cache = _load_cache()
    cache["skip_version"] = version
    _save_cache(cache)


def check_for_updates_async(
    on_result: Callable[[UpdateCheckResult], None], force: bool = False
) -> None:
    """Run check_for_updates() on a background thread.

    ``on_result`` is invoked from that worker thread - callers touching
    Tkinter must marshal back onto the GUI thread themselves (e.g. push the
    result onto a queue.Queue and poll it with Tk's ``after``).
    """

    def worker() -> None:
        result = check_for_updates(force=force)
        on_result(result)

    threading.Thread(target=worker, daemon=True, name="vaila-update-check").start()
