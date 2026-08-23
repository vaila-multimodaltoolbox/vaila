"""Tests for vaila.update_checker (offline-safe: version parsing/compare + local read).

Network-dependent behavior (fetch_remote_version, check_for_updates_async) is
exercised manually / in CI smoke runs, not here - these tests must pass with
no internet access.
"""

from __future__ import annotations

from vaila.update_checker import (
    get_install_command,
    get_local_version,
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
    # Unknown platform falls back to the Linux one-liner rather than raising.
    assert get_install_command("PlanNine") == linux_cmd
