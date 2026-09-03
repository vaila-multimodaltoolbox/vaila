"""
===============================================================================
update_checker.py
===============================================================================
Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 21 August 2026
Update Date: 03 September 2026
Version: 0.3.120

Description:
------------
Git-based update checker for vailá git clones.

When the project directory is a git repository, the GUI compares the local
``HEAD`` with ``origin/main`` (``git fetch`` + rev-list) so any new commit is
detected — not only pyproject.toml version bumps. Manual **Check for Updates**
always fetches; automatic startup checks respect a ~20 h cache.

Non-git installs (one-line installer trees without ``.git``) fall back to
comparing the local pyproject version with GitHub's ``main`` pyproject.toml.

An ``origin`` remote set to SSH (``git@github.com:...``) has no key on most
client machines, so ``git fetch`` fails with "Permission denied (publickey)".
``git_fetch_main`` classifies that failure as ``"ssh_permission_denied"``
instead of surfacing the raw git stderr; ``_check_git_updates`` then falls
back to a read-only ``git ls-remote`` over HTTPS so the comparison can still
succeed. ``describe_update_error`` turns any error code into one actionable
message shared by the GUI and any headless/CLI caller, and
``fix_ssh_remote`` can rewrite the SSH origin to HTTPS in place (only ever
called after explicit user confirmation — never automatically).

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
HTTPS_REMOTE_URL = "https://github.com/vaila-multimodaltoolbox/vaila.git"
RAW_PYPROJECT_URL = (
    "https://raw.githubusercontent.com/vaila-multimodaltoolbox/vaila/main/pyproject.toml"
)
CACHE_FILE = Path.home() / ".vaila" / "update_check_cache.json"
_SSH_REMOTE_PREFIXES = ("git@github.com:", "ssh://git@github.com/")
_SSH_AUTH_FAILURE_MARKERS = (
    "permission denied (publickey)",
    "could not read from remote repository",
)
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
        "[Net.ServicePointManager]::SecurityProtocol -bor 3072\n"
        "$i = Join-Path $env:TEMP 'install_vaila_win.ps1'\n"
        "Invoke-WebRequest -Uri "
        "'https://raw.githubusercontent.com/vaila-multimodaltoolbox/vaila/main/"
        "install_vaila_win.ps1' -OutFile $i -UseBasicParsing\n"
        "Unblock-File -Path $i -ErrorAction SilentlyContinue\n"
        '& powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$i"'
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


def get_remote_url(project_root: Path | None = None, remote: str = "origin") -> str | None:
    """Current URL configured for `remote`, or None if unavailable."""
    root = project_root or _project_root()
    if not is_git_repository(root):
        return None
    try:
        proc = _run_git(["remote", "get-url", remote], root)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def is_ssh_github_remote(url: str | None) -> bool:
    """True if `url` is an SSH github.com remote (needs a key most machines lack)."""
    if not url:
        return False
    return url.startswith(_SSH_REMOTE_PREFIXES)


def sanitize_remote_url(url: str | None) -> str | None:
    """HTTPS equivalent for an SSH vailá github.com remote, else None (no change needed)."""
    if not is_ssh_github_remote(url):
        return None
    return HTTPS_REMOTE_URL


def fix_ssh_remote(project_root: Path | None = None, remote: str = "origin") -> tuple[bool, str]:
    """Rewrite an SSH `remote` to HTTPS in place. Returns (changed, message).

    Only call this after explicit user confirmation (e.g. a GUI dialog) —
    never automatically, since it mutates the user's real git configuration.
    """
    root = project_root or _project_root()
    url = get_remote_url(root, remote)
    https_url = sanitize_remote_url(url)
    if https_url is None:
        return False, f"'{remote}' is already HTTPS (or unavailable); nothing to change."
    try:
        proc = _run_git(["remote", "set-url", remote, https_url], root)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        return False, detail or f"failed to rewrite remote '{remote}'."
    return True, f"'{remote}' rewritten from SSH to {https_url}"


def _is_ssh_auth_failure(detail: str) -> bool:
    lowered = detail.lower()
    return any(marker in lowered for marker in _SSH_AUTH_FAILURE_MARKERS)


def ls_remote_main_https() -> tuple[str | None, str | None]:
    """Read-only `git ls-remote` of origin/main over HTTPS (no local remote change).

    Used as a fallback comparison when the local `origin` remote can't be
    fetched (e.g. SSH auth failure). Returns (short_sha, error).
    """
    try:
        proc = subprocess.run(
            ["git", "ls-remote", HTTPS_REMOTE_URL, "main"],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, "ls_remote_failed"
    if proc.returncode != 0 or not proc.stdout.strip():
        return None, "ls_remote_failed"
    sha_full = proc.stdout.split()[0]
    return sha_full[:7], None


def describe_update_error(error: str | None, ssh_origin_detected: bool = False) -> str:
    """Human-actionable text for an `UpdateCheckResult.error` code.

    Shared by the GUI and any headless/CLI caller so both surfaces give the
    same guidance instead of a raw git stack trace.
    """
    if error == "ssh_permission_denied" or (error and ssh_origin_detected):
        return (
            "git could not authenticate over SSH to GitHub "
            "(Permission denied (publickey)).\n\n"
            "Your 'origin' remote is set to SSH (git@github.com:...), which "
            "needs a GitHub SSH key most machines don't have configured.\n\n"
            f"Fix: git remote set-url origin {HTTPS_REMOTE_URL}"
        )
    if error == "not_a_git_repo":
        return "This install is not a git clone; use the one-line installer to update."
    if error == "origin_main_unavailable":
        return "origin/main isn't available locally yet. Run: git fetch origin main"
    if error == "ls_remote_failed":
        return "Could not reach GitHub over HTTPS either. Check your internet connection."
    if error:
        return f"Details: {error}"
    return "Unknown error."


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
        if _is_ssh_auth_failure(detail):
            return False, "ssh_permission_denied"
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
    ssh_origin_detected: bool = False  # origin is git@github.com:... — GUI can offer a fix
    used_https_fallback: bool = False  # comparison came from ls-remote, not a local fetch


@dataclass
class GitPullResult:
    success: bool
    message: str


def _check_git_updates(force: bool, local_version: str | None) -> UpdateCheckResult:
    project_root = _project_root()
    cache = _load_cache()
    now = time.time()
    ssh_detected = is_ssh_github_remote(get_remote_url(project_root))

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
                ssh_origin_detected=ssh_detected,
            )

    ok, fetch_err = git_fetch_main(project_root)
    cache["last_check"] = now
    if not ok:
        if fetch_err == "ssh_permission_denied":
            remote_short, _ls_err = ls_remote_main_https()
            if remote_short is not None:
                local_proc = _run_git(["rev-parse", "--short", "HEAD"], project_root)
                local_short = local_proc.stdout.strip() if local_proc.returncode == 0 else None
                _save_cache(cache)
                # ls-remote gives no rev-list distance over HTTPS without a
                # local fetch, so "behind" is a 0/1 different-or-not signal.
                behind = 1 if (local_short and local_short != remote_short) else 0
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
                    error=fetch_err,
                    ssh_origin_detected=True,
                    used_https_fallback=True,
                )
        _save_cache(cache)
        return UpdateCheckResult(
            checked=False,
            local_version=local_version,
            is_git_repo=True,
            error=fetch_err or "git_fetch_failed",
            ssh_origin_detected=ssh_detected,
        )

    behind, local_short, remote_short, err = git_status_behind_main(project_root)
    if err:
        _save_cache(cache)
        return UpdateCheckResult(
            checked=False,
            local_version=local_version,
            is_git_repo=True,
            error=err,
            ssh_origin_detected=ssh_detected,
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
        ssh_origin_detected=ssh_detected,
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
