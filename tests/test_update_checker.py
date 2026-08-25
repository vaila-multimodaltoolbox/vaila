"""Tests for vaila.update_checker (offline-safe helpers + local git status)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from vaila.update_checker import (
    get_git_pull_command,
    get_install_command,
    get_local_version,
    git_status_behind_main,
    is_git_repository,
    is_newer,
    parse_version,
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
