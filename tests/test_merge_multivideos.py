"""Terminal banner for multi-video merge completion."""

from __future__ import annotations

from vaila.merge_multivideos import format_merge_banner


def test_format_merge_banner_done_includes_output_and_report() -> None:
    message = (
        "/data/merge_accurate_20261004/newrarepoints.mp4\n\n"
        "Frame report: /data/merge_accurate_20261004/newrarepoints_frame_report.txt"
    )
    text = format_merge_banner(True, message)
    assert ">> vaila/merge_multivideos: DONE" in text
    assert ">> Output: /data/merge_accurate_20261004/newrarepoints.mp4" in text
    assert ">> Frame report: /data/merge_accurate_20261004/newrarepoints_frame_report.txt" in text
    assert "FAILED" not in text


def test_format_merge_banner_done_without_report() -> None:
    text = format_merge_banner(True, "/tmp/merged.mp4")
    assert text == ">> vaila/merge_multivideos: DONE\n>> Output: /tmp/merged.mp4"


def test_format_merge_banner_failed() -> None:
    text = format_merge_banner(False, "FFmpeg error: return code 1")
    assert text == ">> vaila/merge_multivideos: FAILED\n>> FFmpeg error: return code 1"
    assert "DONE" not in text


def test_format_merge_banner_failed_empty_reason() -> None:
    text = format_merge_banner(False, "   ")
    assert text == ">> vaila/merge_multivideos: FAILED\n>> unknown error"
