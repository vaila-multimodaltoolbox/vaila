"""
===============================================================================
update_checker.py
===============================================================================
Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 21 August 2026
Update Date: 24 August 2026
Version: 0.3.113

Description:
------------
Git-based update checker for vailá git clones.

When the project directory is a git repository, the GUI compares the local
``HEAD`` with ``origin/main`` (``git fetch`` + rev-list) so any new commit is
detected — not only pyproject.toml version bumps. Manual **Check for Updates**
always fetches; automatic startup checks respect a ~20 h cache.

Non-git installs (one-line installer trees without ``.git``) fall back to
comparing the local pyproject version with GitHub's ``main`` pyproject.toml.

All network / git I/O fails soft. Run ``check_for_updates_async()`` and
``git_pull_async()`` from background threads; marshal results to Tk via a
queue polled with ``after``.
===============================================================================
"""

from __future__ import annotations

import json
import platform
import subprocess
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
GIT_TIMEOUT = 30.0  # seconds for fetch / rev-parse
GIT_PULL_TIMEOUT = 180.0  # seconds
REQUEST_TIMEOUT = 4.0  # seconds (non-git fallback)

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


def get_git_pull_command(project_root: Path | None = None) -> str:
    """Copy-paste command for updating a git clone."""
    root = project_root or _project_root()
    return f"cd {root} && git pull --ff-only origin main"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def is_git_repository(project_root: Path | None = None) -> bool:
    root = project_root or _project_root()
    return (root / ".git").is_dir()


def _run_git(
    args: list[str],
    project_root: Path,
    *,
    timeout: float = GIT_TIMEOUT,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def git_fetch_main(project_root: Path | None = None) -> tuple[bool, str | None]:
    """Fetch origin/main. Returns (ok, error_code_or_message)."""
    root = project_root or _project_root()
    if not is_git_repository(root):
        return False, "not_a_git_repo"
    try:
        proc = _run_git(["fetch", "--quiet", "origin", "main"], root)
    except (OSError, subprocess.TimeoutExpired):
        return False, "git_fetch_failed"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        return False, detail or "git_fetch_failed"
    return True, None


def git_status_behind_main(
    project_root: Path | None = None,
) -> tuple[int, str | None, str | None, str | None]:
    """Commits on origin/main not in HEAD. Returns (behind, local_short, remote_short, error)."""
    root = project_root or _project_root()
    if not is_git_repository(root):
        return 0, None, None, "not_a_git_repo"

    local_proc = _run_git(["rev-parse", "--short", "HEAD"], root)
    remote_proc = _run_git(["rev-parse", "--short", "origin/main"], root)
    if local_proc.returncode != 0:
        return 0, None, None, "git_head_unavailable"
    if remote_proc.returncode != 0:
        return 0, None, None, "origin_main_unavailable"

    local_short = local_proc.stdout.strip()
    remote_short = remote_proc.stdout.strip()

    count_proc = _run_git(["rev-list", "--count", "HEAD..origin/main"], root)
    if count_proc.returncode != 0:
        return 0, local_short, remote_short, "git_rev_list_failed"
    try:
        behind = int(count_proc.stdout.strip())
    except ValueError:
        return 0, local_short, remote_short, "git_rev_list_failed"
    return behind, local_short, remote_short, None


def git_pull_main(project_root: Path | None = None) -> tuple[bool, str]:
    """Fast-forward pull from origin/main. Returns (success, combined output)."""
    root = project_root or _project_root()
    if not is_git_repository(root):
        return False, "Not a git repository."
    try:
        proc = _run_git(
            ["pull", "--ff-only", "origin", "main"],
            root,
            timeout=GIT_PULL_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return False, "git pull timed out."
    except OSError as exc:
        return False, str(exc)

    output = "\n".join(part.strip() for part in (proc.stdout, proc.stderr) if part.strip())
    if proc.returncode != 0:
        return False, output or "git pull failed."
    return True, output or "Already up to date."


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
    """Version declared in pyproject.toml on the GitHub ``main`` branch, or None."""
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
    """'0.3.110' -> (0, 3, 110); tolerant of a non-numeric suffix on the last chunk."""
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
    checked: bool
    update_available: bool = False
    local_version: str | None = None
    remote_version: str | None = None  # git: origin/main short sha; fallback: remote semver
    commits_behind: int = 0
    local_commit: str | None = None
    remote_commit: str | None = None
    is_git_repo: bool = False
    error: str | None = None


@dataclass
class GitPullResult:
    success: bool
    message: str


def _check_git_updates(force: bool, local_version: str | None) -> UpdateCheckResult:
    project_root = _project_root()
    cache = _load_cache()
    now = time.time()

    fresh_enough = not force and (now - cache.get("last_check", 0)) < CHECK_INTERVAL_SECONDS
    if fresh_enough and cache.get("last_remote_commit"):
        behind, local_short, remote_short, err = git_status_behind_main(project_root)
        if err is None:
            skip = cache.get("skip_commit")
            available = behind > 0 and remote_short != skip
            return UpdateCheckResult(
                checked=True,
                local_version=local_version,
                remote_version=remote_short,
                commits_behind=behind,
                local_commit=local_short,
                remote_commit=remote_short,
                is_git_repo=True,
                update_available=available,
            )

    ok, fetch_err = git_fetch_main(project_root)
    cache["last_check"] = now
    if not ok:
        _save_cache(cache)
        return UpdateCheckResult(
            checked=False,
            local_version=local_version,
            is_git_repo=True,
            error=fetch_err or "git_fetch_failed",
        )

    behind, local_short, remote_short, err = git_status_behind_main(project_root)
    if err:
        _save_cache(cache)
        return UpdateCheckResult(
            checked=False,
            local_version=local_version,
            is_git_repo=True,
            error=err,
        )

    cache["last_remote_commit"] = remote_short
    _save_cache(cache)

    skip = cache.get("skip_commit")
    available = behind > 0 and remote_short != skip
    return UpdateCheckResult(
        checked=True,
        local_version=local_version,
        remote_version=remote_short,
        commits_behind=behind,
        local_commit=local_short,
        remote_commit=remote_short,
        is_git_repo=True,
        update_available=available,
    )


def _check_pyproject_updates(force: bool, local_version: str | None) -> UpdateCheckResult:
    """Fallback for install trees without a .git directory."""
    cache = _load_cache()
    now = time.time()

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


def check_for_updates(force: bool = False) -> UpdateCheckResult:
    """Synchronous check — call from a background thread."""
    local_version = get_local_version()
    if is_git_repository():
        return _check_git_updates(force, local_version)
    return _check_pyproject_updates(force, local_version)


def skip_version(version: str) -> None:
    """Remember the user dismissed the notice (commit sha or semver string)."""
    cache = _load_cache()
    if is_git_repository():
        cache["skip_commit"] = version
    else:
        cache["skip_version"] = version
    _save_cache(cache)


def check_for_updates_async(
    on_result: Callable[[UpdateCheckResult], None], force: bool = False
) -> None:
    """Run check_for_updates() on a background thread."""

    def worker() -> None:
        on_result(check_for_updates(force=force))

    threading.Thread(target=worker, daemon=True, name="vaila-update-check").start()


def git_pull_async(on_result: Callable[[GitPullResult], None]) -> None:
    """Run git_pull_main() on a background thread."""

    def worker() -> None:
        ok, message = git_pull_main()
        on_result(GitPullResult(success=ok, message=message))

    threading.Thread(target=worker, daemon=True, name="vaila-git-pull").start()
