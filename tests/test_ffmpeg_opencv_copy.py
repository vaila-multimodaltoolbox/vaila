"""OpenCV-decodable H.264 copy of videos OpenCV cannot read (e.g. AV1).

Version: 0.4.6
Update Date: 30 September 2026
"""

import shutil
import subprocess

import cv2
import pytest

from vaila import ffmpeg_utils

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def _encode(path, codec_args):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30"]
    subprocess.run([*cmd, "-t", "1", *codec_args, str(path)], check=True)


def test_av1_gets_a_readable_h264_copy_with_every_frame(tmp_path):
    src = tmp_path / "clip.mp4"
    try:
        _encode(src, ["-c:v", "libaom-av1", "-cpu-used", "8"])
    except subprocess.CalledProcessError:
        pytest.skip("ffmpeg has no libaom-av1 encoder")
    if ffmpeg_utils.opencv_reads_video(src):
        pytest.skip("this OpenCV build decodes AV1")
    copy = ffmpeg_utils.opencv_compatible_copy(src)
    assert copy == tmp_path / "clip_h264.mp4" == ffmpeg_utils.opencv_copy_path(src)
    assert ffmpeg_utils.opencv_reads_video(copy)
    assert int(cv2.VideoCapture(str(copy)).get(cv2.CAP_PROP_FRAME_COUNT)) == 30
    assert not list(tmp_path.glob("*.partial.mp4"))
    mtime = copy.stat().st_mtime_ns
    assert ffmpeg_utils.opencv_compatible_copy(src) == copy  # reused, not re-encoded
    assert copy.stat().st_mtime_ns == mtime


def test_readable_video_is_opened_as_is(tmp_path, monkeypatch):
    from vaila import getpixelvideo

    src = tmp_path / "clip.mp4"
    _encode(src, ["-c:v", "libx264", "-pix_fmt", "yuv420p"])
    monkeypatch.setattr(ffmpeg_utils, "opencv_compatible_copy", lambda _p: pytest.fail("no copy"))
    assert getpixelvideo.ensure_decodable_video(str(src)) == str(src)
