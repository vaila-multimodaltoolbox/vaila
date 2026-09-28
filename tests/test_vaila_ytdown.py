"""Unit tests for downloader quality selection.

Version: 0.4.5
Update Date: 27 September 2026
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vaila import vaila_ytdown as yt
from vaila.vaila_ytdown import build_ytdlp_base_opts, detect_js_runtimes, read_urls_from_file


def format_info(identifier, width=1920, height=1080, fps=60, **kwargs):
    return {
        "format_id": identifier,
        "width": width,
        "height": height,
        "fps": fps,
        "vcodec": "h264",
        "acodec": "none",
        "ext": "mp4",
        "protocol": "https",
        "url": f"https://example.com/{identifier}",
        **kwargs,
    }


def test_quality_options_preserve_fps_and_prefer_best_variant():
    info = {
        "formats": [
            format_info("unknown", None, None, None),
            format_info("4k", 3840, 2160, 30),
            format_info("30", fps=30),
            format_info("60-low"),
            format_info("60-best"),
            format_info("drm", has_drm=True),
            format_info("audio", vcodec="none"),
            format_info("storyboard", vcodec=None),
        ]
    }
    assert [f["format_id"] for f in yt.quality_options(info)] == ["60-best", "4k", "30", "unknown"]
    assert yt.quality_label(info["formats"][0]) == "?x? · unknown FPS"


@pytest.mark.parametrize("values", [["0=a"], ["3=a"], ["1="], ["abc"], ["1=a", "1=b"]])
def test_invalid_selections(values):
    with pytest.raises(ValueError):
        yt.parse_video_selections(values, 2)


def test_fragment_progress_does_not_claim_completion():
    downloader = yt.YTDownloader()
    events = []
    downloader.event_callback = lambda *args: events.append(args)
    downloader._progress_hook(
        {
            "status": "downloading",
            "downloaded_bytes": 712,
            "total_bytes_estimate": 712,
            "fragment_index": 0,
            "fragment_count": 43,
        }
    )
    assert events[-1] == ("progress", {"percent": 0, "detail": "Fragments: 0/43"})


def test_cli_list_formats_does_not_create_output(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        yt.YTDownloader,
        "get_video_info",
        lambda *args: {
            "title": "Video",
            "formats": [format_info("4k", 3840, 2160, 30), format_info("60")],
        },
    )
    target = tmp_path / "absent"
    assert yt.run_ytdown(["--url", "https://example.com", "--list-formats", "-o", str(target)]) == 0
    assert not target.exists()
    assert "DEFAULT 1920x1080 · 60 FPS | --video-format 1=60" in capsys.readouterr().out


@pytest.mark.parametrize("embedded_audio", [False, True])
def test_real_ytdlp_selection_reuses_preview(monkeypatch, tmp_path, embedded_audio):
    """Exercise actual yt-dlp format processing, stubbing only media transfer."""
    info = {
        "id": "example",
        "title": "Example",
        "extractor": "generic",
        "webpage_url": "https://example.com/video",
        "formats": [
            format_info("audio", vcodec="none", acodec="aac", width=None, height=None, fps=None),
            format_info("4k", 3840, 2160, 30),
            format_info("60", acodec="aac" if embedded_audio else "none"),
        ],
    }
    downloader = yt.YTDownloader()
    downloader.ffmpeg_available = True
    monkeypatch.setattr(
        downloader, "get_video_info", lambda *args: pytest.fail("Unnecessary extraction")
    )
    seen = []

    def process_info(ydl, result):
        seen.append(result)
        path = Path(ydl.prepare_filename(result)).with_suffix(".mp4")
        path.write_bytes(b"finished")
        result["filepath"] = str(path)

    monkeypatch.setattr(yt.yt_dlp.YoutubeDL, "process_info", process_info)
    result = downloader.download_video("https://example.com/video", str(tmp_path), video_info=info)
    assert Path(result).exists()
    assert seen[0]["format_id"] == ("60" if embedded_audio else "60+audio")
    assert seen[0]["fps"] == 60
    assert "Requested quality" in (Path(result).parent / "video_info.txt").read_text()
    with pytest.raises(ValueError, match="unavailable"):
        downloader.download_video(
            "https://example.com/video", str(tmp_path), video_info=info, format_id="missing"
        )


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


@pytest.mark.parametrize("quality_remains", [False, True])
def test_expired_url_refresh_preserves_quality(monkeypatch, tmp_path, quality_remains):
    downloader = yt.YTDownloader()
    downloader.ffmpeg_available = True
    preview = {"title": "Example", "formats": [format_info("old")]}
    refreshes, attempts = [], []

    def refresh(url):
        refreshes.append(url)
        return {
            "title": "Example",
            "formats": [format_info("new", fps=60 if quality_remains else 30)],
        }

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def process_ie_result(self, info, download):
            attempts.append(info)
            if len(attempts) == 1:
                raise yt.yt_dlp.utils.DownloadError("HTTP Error 403: Forbidden")
            path = tmp_path / "Example.mp4"
            path.write_bytes(b"finished")
            return {**info, "filepath": str(path)}

    monkeypatch.setattr(downloader, "get_video_info", refresh)
    monkeypatch.setattr(yt.yt_dlp, "YoutubeDL", FakeYDL)
    if quality_remains:
        downloader.download_video("https://example.com", str(tmp_path), video_info=preview)
        assert len(attempts) == 2
    else:
        with pytest.raises(Exception, match="quality is no longer available"):
            downloader.download_video("https://example.com", str(tmp_path), video_info=preview)
        assert len(attempts) == 1
    assert len(refreshes) == 1


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
