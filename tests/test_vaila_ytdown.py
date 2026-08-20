"""Unit tests for vaila_ytdown YouTube EJS / yt-dlp option helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from vaila.vaila_ytdown import build_ytdlp_base_opts, detect_js_runtimes, read_urls_from_file


def test_read_urls_from_file_skips_comments_and_blanks(tmp_path: Path) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        "# comment\n\nhttps://www.youtube.com/watch?v=abc\n  \nhttps://youtu.be/xyz\n",
        encoding="utf-8",
    )
    assert read_urls_from_file(url_file) == [
        "https://www.youtube.com/watch?v=abc",
        "https://youtu.be/xyz",
    ]


def test_build_ytdlp_base_opts_includes_retries_and_cert_skip() -> None:
    opts = build_ytdlp_base_opts(format="best", quiet=True)
    assert opts["no_check_certificate"] is True
    assert opts["noplaylist"] is True
    assert opts["retries"] == 10
    assert opts["format"] == "best"
    assert opts["quiet"] is True


def test_detect_js_runtimes_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_which(name: str) -> str | None:
        if name == "node":
            return "/usr/bin/node"
        return None

    monkeypatch.setattr("vaila.vaila_ytdown.shutil.which", fake_which)
    runtimes = detect_js_runtimes()
    assert runtimes == {"node": {"path": "/usr/bin/node"}}
    opts = build_ytdlp_base_opts()
    assert opts["js_runtimes"] == {"node": {"path": "/usr/bin/node"}}


def test_build_ytdlp_base_opts_remote_ejs_when_package_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yt_dlp_ejs":
            raise ImportError("simulated missing ejs")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr("vaila.vaila_ytdown.detect_js_runtimes", lambda: {})
    opts = build_ytdlp_base_opts()
    assert opts.get("remote_components") == {"ejs:github"}
