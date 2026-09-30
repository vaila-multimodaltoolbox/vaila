"""Unit tests for downloader quality selection.

Version: 0.4.6
Update Date: 30 September 2026
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vaila import vaila_ytdown as yt
from vaila.vaila_ytdown import (
    build_ytdlp_base_opts,
    detect_js_runtimes,
    format_ytdown_cli_command,
    read_urls_from_file,
)


@pytest.fixture(autouse=True)
def _fake_downloads_are_readable(monkeypatch):
    # Test downloads are placeholder bytes, not videos: skip the OpenCV codec check.
    monkeypatch.setattr(yt, "opencv_reads_video", lambda _path: True)


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
    assert yt.quality_label(info["formats"][0]) == "?x? · unknown FPS · H.264"


def test_quality_options_prefer_opencv_decodable_codec():
    # yt-dlp orders AV1 last (best); OpenCV cannot decode AV1, so H.264 then VP9 win.
    info = {
        "formats": [
            format_info("h264", vcodec="avc1.640028"),
            format_info("vp9", vcodec="vp09.00.40.08"),
            format_info("av1", vcodec="av01.0.09M.08"),
            format_info("4k-vp9", 3840, 2160, 60, vcodec="vp9"),
            format_info("4k-av1", 3840, 2160, 60, vcodec="av01.0.13M.08"),
        ]
    }
    assert [f["format_id"] for f in yt.quality_options(info)] == ["4k-vp9", "h264"]
    # --keep-codec / GUI box unticked: YouTube's own best variant, same qualities.
    kept = yt.quality_options(info, keep_codec=True)
    assert [f["format_id"] for f in kept] == ["4k-av1", "av1"]
    assert [yt.quality_label(f) for f in kept] == [
        "3840x2160 · 60 FPS · AV1",
        "1920x1080 · 60 FPS · AV1",
    ]


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
    assert "DEFAULT 1920x1080 · 60 FPS · H.264 | --video-format 1=60" in capsys.readouterr().out


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


def test_format_ytdown_cli_command_single_url():
    cmd = format_ytdown_cli_command(
        ["https://www.youtube.com/watch?v=abc"],
        "/data/out",
        audio_only=False,
    )
    assert cmd == [
        "uv",
        "run",
        "--no-sync",
        "vaila/vaila_ytdown.py",
        "--output",
        "/data/out",
        "--no-gui",
        "--url",
        "https://www.youtube.com/watch?v=abc",
    ]


def test_format_ytdown_cli_command_batch_and_formats():
    cmd = format_ytdown_cli_command(
        ["https://example/1", "https://example/2"],
        "/data/out",
        audio_only=False,
        selections={1: "137", 2: "299"},
        debug=True,
        url_file=Path("/data/out/urls.txt"),
    )
    assert cmd == [
        "uv",
        "run",
        "--no-sync",
        "vaila/vaila_ytdown.py",
        "--output",
        "/data/out",
        "--no-gui",
        "--file",
        "/data/out/urls.txt",
        "--video-format",
        "1=137",
        "--video-format",
        "2=299",
        "--debug",
    ]


def test_format_ytdown_cli_command_audio_only():
    cmd = format_ytdown_cli_command(
        ["https://example/audio"],
        "/data/music",
        audio_only=True,
    )
    assert "--audio-only" in cmd
    assert "--output" in cmd and "/data/music" in cmd
    assert "--no-gui" in cmd


def test_download_urls_transparency_and_cli_mirror_prints(tmp_path, monkeypatch, capsys):
    downloader = yt.YTDownloader()
    downloader.ffmpeg_available = True

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download=True):
            info = {
                "title": "Test Video",
                "width": 1920,
                "height": 1080,
                "fps": 60,
                "duration": 12,
                "format_id": "137",
                "formats": [format_info("137")],
            }
            output = tmp_path / "Test Video.mp4"
            output.write_bytes(b"dummy video data")
            info["filepath"] = str(output)
            return info

        def process_ie_result(self, info, download=True):
            return self.extract_info(info.get("webpage_url", "https://example.com/test"), download)

    mirrors = []
    monkeypatch.setattr(yt.yt_dlp, "YoutubeDL", FakeYDL)
    monkeypatch.setattr(
        yt, "print_gui_cli_mirror", lambda label, cmd, **kw: mirrors.append((label, cmd, kw))
    )

    out_dir = tmp_path / "downloads"
    result = downloader.download_urls(["https://example.com/test"], output_dir=out_dir)

    captured = capsys.readouterr().out
    assert "Starting YouTube download run (1 item)" in captured
    assert "Destination parent directory:" in captured
    assert "Run output directory:" in captured
    assert "Equivalent CLI:" in captured
    assert "Download successful: Test Video" in captured
    assert "Saved to:" in captured
    assert "Final output directory:" in captured
    assert len(mirrors) >= 2  # printed at run start and run finish
    assert mirrors[0][0] == "vaila/vaila_ytdown"
    assert "vaila/vaila_ytdown.py" in mirrors[0][1]
    assert result.exit_code == 0


def test_download_is_reencoded_in_place_when_opencv_cannot_read_it(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"av1")
    readable = {str(video): False}

    def fake_copy(path):
        copy = tmp_path / "clip_h264.mp4"
        copy.write_bytes(b"h264")
        return copy

    monkeypatch.setattr(yt, "opencv_reads_video", lambda p: readable.get(str(p), True))
    monkeypatch.setattr(yt, "opencv_compatible_copy", fake_copy)
    downloader = yt.YTDownloader()
    assert downloader._make_vaila_readable(str(video)) == str(video)
    assert video.read_bytes() == b"h264"  # same name, now H.264
    assert not (tmp_path / "clip_h264.mp4").exists()
    readable[str(video)] = True
    monkeypatch.setattr(yt, "opencv_compatible_copy", lambda _p: pytest.fail("no re-encode"))
    assert downloader._make_vaila_readable(str(video)) == str(video)


def test_keep_codec_skips_reencode_and_is_mirrored_in_cli():
    downloader = yt.YTDownloader()
    downloader.keep_codec = True
    assert downloader._make_vaila_readable("/no/such/file.mp4") == "/no/such/file.mp4"
    cmd = format_ytdown_cli_command(["https://y/1"], "/out", keep_codec=True, selections={1: "399"})
    assert cmd[-3:] == ["--keep-codec", "--video-format", "1=399"]
    assert "--keep-codec" not in format_ytdown_cli_command(["https://y/1"], "/out")


def test_finished_file_gets_rename_style_name_unless_original_names(tmp_path):
    downloader = yt.YTDownloader()
    video = tmp_path / "001_Australia vs Brazil ｜ Highlights - Friendly.MP4"
    video.write_bytes(b"v")
    renamed = Path(downloader._sanitized_name(str(video)))
    assert renamed.name == "001_australia_vs_brazil_highlights_friendly.mp4"
    assert renamed.read_bytes() == b"v" and not video.exists()
    clash = tmp_path / "Clip A.mp4"
    clash.write_bytes(b"x")
    (tmp_path / "clip_a.mp4").write_bytes(b"old")
    assert Path(downloader._sanitized_name(str(clash))).name == "clip_a_1.mp4"
    downloader.original_names = True
    keep = tmp_path / "Keep Me.mp4"
    keep.write_bytes(b"k")
    assert downloader._sanitized_name(str(keep)) == str(keep)
    cmd = format_ytdown_cli_command(["https://y/1"], "/out", original_names=True)
    assert "--original-names" in cmd
    assert "--original-names" not in format_ytdown_cli_command(["https://y/1"], "/out")
