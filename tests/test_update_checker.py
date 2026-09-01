"""Tests for vaila.update_checker (offline-safe helpers + local git status)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from vaila.update_checker import (
    HTTPS_REMOTE_URL,
    describe_update_error,
    fix_ssh_remote,
    get_git_pull_command,
    get_install_command,
    get_local_version,
    get_remote_url,
    git_fetch_main,
    git_status_behind_main,
    is_git_repository,
    is_newer,
    is_ssh_github_remote,
    parse_version,
    sanitize_remote_url,
)


def test_parse_version_basic():
    assert parse_version("0.3.109") == (0, 3, 109)
    assert parse_version("1.2.3") == (1, 2, 3)


def test_parse_version_tolerant_of_suffix():
    assert parse_version("0.3.109rc1") == (0, 3, 109)


def test_is_newer():
    assert is_newer("0.3.109", "0.3.108") is True
    assert is_newer("0.3.108", "0.3.108") is False
    assert is_newer("0.3.108", "0.3.109") is False
    assert is_newer("1.0.0", "0.9.9") is True


def test_get_local_version_reads_pyproject():
    version = get_local_version()
    assert version is not None
    assert parse_version(version) >= (0, 3, 109)


def test_get_install_command_per_os():
    linux_cmd = get_install_command("Linux")
    mac_cmd = get_install_command("Darwin")
    win_cmd = get_install_command("Windows")

    assert "install_vaila_linux.sh" in linux_cmd
    assert "install_vaila_mac.sh" in mac_cmd
    assert "install_vaila_win.ps1" in win_cmd
    assert "Unblock-File" in win_cmd
    assert "powershell.exe" in win_cmd
    assert get_install_command("PlanNine") == linux_cmd


def test_get_git_pull_command_contains_cd_and_pull():
    cmd = get_git_pull_command()
    assert "git pull" in cmd
    assert "origin main" in cmd


def test_is_git_repository_for_this_checkout():
    root = Path(__file__).resolve().parent.parent
    assert is_git_repository(root) is True


def test_git_status_behind_main_in_sync_when_fetched():
    root = Path(__file__).resolve().parent.parent
    if not is_git_repository(root):
        return
    fetch = subprocess.run(
        ["git", "fetch", "--quiet", "origin", "main"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if fetch.returncode != 0:
        return
    behind, local_short, remote_short, err = git_status_behind_main(root)
    assert err is None
    assert local_short
    assert remote_short
    assert behind >= 0


# -- SSH-vs-HTTPS remote handling ------------------------------------------
# All of these use a throwaway fixture git repo (never this checkout's real
# `origin`), per the windows-install-fix loop's rollback/isolation guardrail.


def _init_fixture_repo(tmp_path: Path, remote_url: str | None = None) -> Path:
    repo = tmp_path / "fixture_repo"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)
    if remote_url:
        subprocess.run(["git", "remote", "add", "origin", remote_url], cwd=repo, check=True)
    return repo


def test_is_ssh_github_remote():
    assert is_ssh_github_remote("git@github.com:vaila-multimodaltoolbox/vaila.git") is True
    assert is_ssh_github_remote("ssh://git@github.com/vaila-multimodaltoolbox/vaila.git") is True
    assert is_ssh_github_remote("https://github.com/vaila-multimodaltoolbox/vaila.git") is False
    assert is_ssh_github_remote(None) is False


def test_sanitize_remote_url_only_rewrites_ssh():
    assert (
        sanitize_remote_url("git@github.com:vaila-multimodaltoolbox/vaila.git") == HTTPS_REMOTE_URL
    )
    assert sanitize_remote_url("https://github.com/vaila-multimodaltoolbox/vaila.git") is None
    assert sanitize_remote_url(None) is None


def test_get_remote_url_reads_configured_origin(tmp_path):
    repo = _init_fixture_repo(tmp_path, "git@github.com:vaila-multimodaltoolbox/vaila.git")
    assert get_remote_url(repo) == "git@github.com:vaila-multimodaltoolbox/vaila.git"


def test_get_remote_url_none_without_remote(tmp_path):
    repo = _init_fixture_repo(tmp_path)
    assert get_remote_url(repo) is None


def test_fix_ssh_remote_rewrites_fixture_repo(tmp_path):
    repo = _init_fixture_repo(tmp_path, "git@github.com:vaila-multimodaltoolbox/vaila.git")
    changed, message = fix_ssh_remote(repo)
    assert changed is True
    assert HTTPS_REMOTE_URL in message
    assert get_remote_url(repo) == HTTPS_REMOTE_URL


def test_fix_ssh_remote_noop_when_already_https(tmp_path):
    repo = _init_fixture_repo(tmp_path, HTTPS_REMOTE_URL)
    changed, message = fix_ssh_remote(repo)
    assert changed is False
    assert get_remote_url(repo) == HTTPS_REMOTE_URL


def test_git_fetch_main_classifies_ssh_permission_denied(tmp_path, monkeypatch):
    from vaila import update_checker as uc

    repo = _init_fixture_repo(tmp_path, "git@github.com:vaila-multimodaltoolbox/vaila.git")

    class FakeDenied:
        returncode = 128
        stdout = ""
        stderr = (
            "git@github.com: Permission denied (publickey).\n"
            "fatal: Could not read from remote repository.\n"
        )

    monkeypatch.setattr(uc, "_run_git", lambda args, root, timeout=uc.GIT_TIMEOUT: FakeDenied())
    ok, err = git_fetch_main(repo)
    assert ok is False
    assert err == "ssh_permission_denied"


def test_describe_update_error_ssh_message_is_actionable():
    text = describe_update_error("ssh_permission_denied")
    assert "SSH" in text
    assert "git remote set-url origin" in text
    assert HTTPS_REMOTE_URL in text


def test_describe_update_error_flags_ssh_origin_even_on_generic_error():
    text = describe_update_error("git_fetch_failed", ssh_origin_detected=True)
    assert "SSH" in text


def test_describe_update_error_generic_and_unknown():
    assert "not a git clone" in describe_update_error("not_a_git_repo")
    assert describe_update_error(None) == "Unknown error."


def test_check_git_updates_falls_back_to_https_on_ssh_denial(monkeypatch):
    """The GUI's 'Check for Updates' must not surface a raw SSH stack trace:
    when origin is SSH and fetch fails, fall back to a read-only HTTPS
    ls-remote instead of reporting checked=False with the raw git error."""
    from vaila import update_checker as uc

    monkeypatch.setattr(uc, "_load_cache", lambda: {})
    monkeypatch.setattr(uc, "_save_cache", lambda data: None)
    monkeypatch.setattr(
        uc,
        "get_remote_url",
        lambda root, remote="origin": "git@github.com:vaila-multimodaltoolbox/vaila.git",
    )
    monkeypatch.setattr(uc, "git_fetch_main", lambda root: (False, "ssh_permission_denied"))
    monkeypatch.setattr(uc, "ls_remote_main_https", lambda: ("abc1234", None))

    class FakeRevParse:
        returncode = 0
        stdout = "deadbee\n"

    monkeypatch.setattr(uc, "_run_git", lambda args, root, timeout=uc.GIT_TIMEOUT: FakeRevParse())

    result = uc._check_git_updates(force=True, local_version="0.3.118")

    assert result.checked is True
    assert result.ssh_origin_detected is True
    assert result.used_https_fallback is True
    assert result.error == "ssh_permission_denied"
    assert result.remote_commit == "abc1234"
    assert result.local_commit == "deadbee"
    assert result.update_available is True


def test_check_git_updates_non_ssh_failure_skips_https_fallback(monkeypatch):
    """A non-SSH fetch failure must stay checked=False, unchanged — the HTTPS
    fallback is only for the specific SSH-auth failure case."""
    from vaila import update_checker as uc

    monkeypatch.setattr(uc, "_load_cache", lambda: {})
    monkeypatch.setattr(uc, "_save_cache", lambda data: None)
    monkeypatch.setattr(
        uc,
        "get_remote_url",
        lambda root, remote="origin": "https://github.com/vaila-multimodaltoolbox/vaila.git",
    )
    monkeypatch.setattr(uc, "git_fetch_main", lambda root: (False, "some_other_git_error"))

    def _unexpected_fallback():
        raise AssertionError("ls_remote_main_https must not be called for a non-SSH failure")

    monkeypatch.setattr(uc, "ls_remote_main_https", _unexpected_fallback)

    result = uc._check_git_updates(force=True, local_version="0.3.118")

    assert result.checked is False
    assert result.used_https_fallback is False
    assert result.ssh_origin_detected is False
    assert result.error == "some_other_git_error"
