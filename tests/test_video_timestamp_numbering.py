"""Frame numbers from video timestamps when the camera dropped frames.

Version: 0.4.15
Update Date: 06 November 2026

A constant-rate video with dropped frames has gaps in its timestamps. OpenCV numbers a
frame reached by seeking by its timestamp but counts frames one by one while reading,
so the two disagree after a gap. ``ffmpeg_utils.TimestampFrameCounter`` and
getpixelvideo's ``VideoFrameSource`` use the timestamp number in both cases.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from vaila.ffmpeg_utils import TimestampFrameCounter, numbering_fps, timestamp_frame_number

# --------------------------------------------------------------------------- #
# counter and frame rate (no video)
# --------------------------------------------------------------------------- #


def _msec(numbers, fps=30.0, start_ms=0.0):
    return [start_ms + 1000.0 * k / fps for k in numbers]


def test_counter_regular_video_counts_like_reading():
    c = TimestampFrameCounter(30.0)
    assert [c.next(t) for t in _msec(range(50))] == list(range(50))
    assert c.dropped == [] and c.irregular == 0 and c.mode == "timestamp"


def test_counter_keeps_numbers_after_dropped_frames():
    kept = [k for k in range(40) if k not in (12, 20, 21)]
    c = TimestampFrameCounter(30.0)
    assert [c.next(t) for t in _msec(kept, start_ms=500.0)] == kept
    assert c.dropped == [12, 20, 21]


def test_counter_jittered_timestamps_of_network_camera():
    # 33.0 / 33.0 / 34.0 ms steps (2970 / 2970 / 3060 at 90 kHz) average 30 fps
    steps = [33.0, 33.0, 34.0] * 20
    t = np.concatenate([[0.0], np.cumsum(steps)])
    c = TimestampFrameCounter(30.0)
    assert [c.next(v) for v in t] == list(range(len(t)))


def test_counter_without_timestamps_counts_in_decode_order():
    c = TimestampFrameCounter(30.0)
    assert [c.next(0.0) for _ in range(5)] == [0, 1, 2, 3, 4]
    assert c.mode == "decode"


def test_counter_repeated_timestamp_gets_next_number():
    c = TimestampFrameCounter(30.0)
    out = [c.next(t) for t in _msec([0, 1, 2, 2, 4, 5])]
    assert out == [0, 1, 2, 3, 4, 5] and c.irregular == 1


def test_numbering_fps_prefers_nominal_rate_only_for_dropped_frames():
    assert numbering_fps(29.0, "30/1", "29/1") == 30.0  # MP4 with drops: count / duration
    assert numbering_fps(30.0, "30/1", "30/1") == 30.0
    assert numbering_fps(29.97, "30000/1001", "30000/1001") == pytest.approx(30000 / 1001)
    assert numbering_fps(24.5, "120/1", "49/2") == 24.5  # variable rate: keep OpenCV's
    assert numbering_fps(25.0, None, "0/0") == 25.0
    assert timestamp_frame_number(1000.0 * 37 / 30, 30.0) == 37


# --------------------------------------------------------------------------- #
# videos with dropped frames (ffmpeg)
# --------------------------------------------------------------------------- #

needs_ffmpeg = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="needs ffmpeg/ffprobe"
)

N_FRAMES, BITS, BLOCK = 90, 7, 32
DROPPED = (35, 37, 38)


def _id_frame(k: int) -> np.ndarray:
    """Frame whose index is written as 7 black/white blocks."""
    img = np.zeros((BLOCK, BITS * BLOCK, 3), np.uint8)
    for b in range(BITS):
        if (k >> b) & 1:
            img[:, b * BLOCK : (b + 1) * BLOCK] = 255
    return img


def _read_id(img: np.ndarray) -> int:
    g = img.mean(axis=2)
    return sum(
        1 << b for b in range(BITS) if g[:, b * BLOCK + 4 : (b + 1) * BLOCK - 4].mean() > 127
    )


def drop_frames(src: Path, dst: Path, dropped) -> None:
    """Copy ``src`` without the frames ``dropped``, keeping the other timestamps."""
    expr = "+".join(f"eq(n\\,{k})" for k in dropped)
    for codec in (["-c:v", "libx264", "-crf", "10"], ["-c:v", "mpeg4", "-q:v", "2"]):
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src)]
        cmd += ["-vf", f"select='not({expr})'", "-fps_mode", "passthrough", "-g", "10"]
        cmd += ["-bf", "0", *codec, "-pix_fmt", "yuv420p", str(dst)]
        if subprocess.run(cmd, capture_output=True).returncode == 0:
            return
    pytest.skip("ffmpeg could not encode the test video")


@pytest.fixture(scope="module")
def dropped_video(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("dropped")
    src = d / "src.avi"
    vw = cv2.VideoWriter(str(src), cv2.VideoWriter.fourcc(*"MJPG"), 30.0, (BITS * BLOCK, BLOCK))
    for k in range(N_FRAMES):
        vw.write(_id_frame(k))
    vw.release()
    dst = d / "dropped.mp4"
    drop_frames(src, dst, DROPPED)
    return dst


def _shown(n: int) -> int:
    """Image expected at frame number n: the last frame not dropped."""
    return max(k for k in range(n + 1) if k not in DROPPED)


@needs_ffmpeg
def test_fixture_reproduces_opencv_disagreement(dropped_video):
    cap = cv2.VideoCapture(str(dropped_video))
    count = 0
    while cap.read()[0]:
        count += 1
    assert count == N_FRAMES - len(DROPPED)
    cap.release()


@needs_ffmpeg
def test_getpixelvideo_numbers_playing_and_paused_alike(dropped_video):
    gpv = pytest.importorskip("vaila.getpixelvideo")
    src = gpv.VideoFrameSource(dropped_video)
    assert src.timestamps and src.numbering_fps == pytest.approx(30.0)
    assert src.frame_span() == N_FRAMES
    played = []
    while True:
        ok, img = src.read()
        if not ok:
            break
        played.append((int(src.get(cv2.CAP_PROP_POS_FRAMES)) - 1, _read_id(img)))
    assert played == [(k, k) for k in range(N_FRAMES) if k not in DROPPED]
    # paused navigation: every number, forwards and backwards (seeks right after a gap
    # are where OpenCV lands late)
    for n in [*range(N_FRAMES), *range(N_FRAMES - 1, -1, -3), 40, 40]:
        assert src.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, img = src.read()
        assert ok
        assert int(src.get(cv2.CAP_PROP_POS_FRAMES)) - 1 == n
        assert _read_id(img) == _shown(n)
    # playing on from a dropped number continues with the next real frame
    src.set(cv2.CAP_PROP_POS_FRAMES, 37)
    src.read()
    ok, img = src.read()
    assert ok and _read_id(img) == 39 and int(src.get(cv2.CAP_PROP_POS_FRAMES)) == 40
    src.set(cv2.CAP_PROP_POS_FRAMES, N_FRAMES + 5)
    assert not src.read()[0]
    src.release()


@needs_ffmpeg
def test_getpixelvideo_metadata_uses_camera_rate(dropped_video):
    gpv = pytest.importorskip("vaila.getpixelvideo")
    src = gpv.VideoFrameSource(dropped_video)
    meta = gpv.align_video_metadata(src, gpv.get_precise_video_metadata(dropped_video))
    assert meta["fps"] == pytest.approx(30.0)
    assert meta["nb_frames"] == N_FRAMES
    src.release()
