"""FreeKiki ``detect`` camera screen: skip videos the network was not trained for.

Version: 0.4.8
Update Date: 15 October 2026
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from vaila import freekiki

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_freekiki_queue_need import H, W, _view  # noqa: E402


def _video(path: Path, frames: int = 30) -> Path:
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))
    for i in range(frames):
        out.write(np.full((H, W, 3), 30 + i, dtype=np.uint8))
    out.release()
    return path


class _Pitch:
    """A network that sees the pitch from a broadcast camera."""

    def predict(self, frame):
        uv, inside = _view()
        return 0.9, uv, np.where(inside, 0.9, 0.0)


class _Zoom:
    """A close-up: the network accepts a handful of points."""

    def predict(self, frame):
        uv, inside = _view()
        kc = np.zeros(49)
        kc[np.flatnonzero(inside)[:3]] = 0.9
        return 0.9, uv, kc


class _Scrambled:
    """Many confident points that lie on no pitch seen from a camera (behind the net)."""

    def predict(self, frame):
        rng = np.random.default_rng(0)
        return 0.9, rng.uniform(0, [W, H], (49, 2)), np.full(49, 0.9)


KW = {"conf": 0.25, "kp_conf": 0.5, "n": 8}


def test_a_broadcast_camera_is_usable(tmp_path):
    rates = freekiki.screen_video(_video(tmp_path / "a.mp4"), _Pitch(), **KW)
    assert rates["camera_ok"] == 1.0 and rates["sampled"] == 8
    assert freekiki.screen_verdict(rates, 0.3) == (True, "")


def test_a_zoom_has_too_few_points(tmp_path):
    rates = freekiki.screen_video(_video(tmp_path / "z.mp4"), _Zoom(), **KW)
    assert rates["camera_ok"] == 0.0 and rates["few_points"] == 1.0
    usable, reason = freekiki.screen_verdict(rates, 0.3)
    assert not usable and "too few keypoints" in reason


def test_points_that_fit_no_camera_are_rejected(tmp_path):
    rates = freekiki.screen_video(_video(tmp_path / "s.mp4"), _Scrambled(), **KW)
    assert rates["camera_ok"] == 0.0 and rates["bad_camera"] + rates["few_points"] == 1.0
    assert not freekiki.screen_verdict(rates, 0.3)[0]


def test_verdict_threshold():
    rates = {"camera_ok": 0.29, "few_points": 0.5, "bad_camera": 0.21}
    assert not freekiki.screen_verdict(rates, 0.3)[0]
    assert freekiki.screen_verdict(rates | {"camera_ok": 0.3}, 0.3)[0]


def test_detect_cli_has_the_screen_options():
    a = freekiki.build_parser().parse_args(["detect", "-w", "W", "--video", "V"])
    assert not a.no_screen and a.min_camera_ok == 0.3
    b = freekiki.build_parser().parse_args(
        ["detect", "-w", "W", "--video", "V", "--no-screen", "--min-camera-ok", "0.5"]
    )
    assert b.no_screen and b.min_camera_ok == 0.5


def test_a_batch_skips_unusable_videos_and_lists_them(tmp_path, monkeypatch):
    ws = freekiki.init_workspace(tmp_path / "ws")
    src = tmp_path / "videos"
    src.mkdir()
    _video(src / "good.mp4")
    _video(src / "zoom.mp4")
    predictors = {"good.mp4": _Pitch(), "zoom.mp4": _Zoom()}

    class Router:
        def __init__(self):
            self.current = None

        def predict(self, frame):
            return predictors[self.current].predict(frame)

    router = Router()
    real_screen = freekiki.screen_video

    def routed(video, predictor, **kw):
        router.current = Path(video).name
        return real_screen(video, router, **kw)

    detected = []
    monkeypatch.setattr(freekiki, "screen_video", routed)
    monkeypatch.setattr(freekiki, "validate_detection_workspace", lambda *a, **k: None)
    monkeypatch.setattr(freekiki, "resolve_model", lambda *a, **k: "model.pt")
    monkeypatch.setattr(freekiki, "load_predictor", lambda *a, **k: object())

    def fake_detect(ws_, video, *, output_dir=None, **kw):
        detected.append(Path(video).name)
        out = Path(output_dir) / f"freekiki_predict_{Path(video).stem}"
        out.mkdir(parents=True)
        (out / "quality.json").write_text("{}")
        return out

    monkeypatch.setattr(freekiki, "detect_video", fake_detect)
    batch = freekiki.detect_videos(ws, src, output_dir=tmp_path / "out")
    assert detected == ["good.mp4"]
    skipped = (batch / "skipped_videos.csv").read_text()
    assert "zoom.mp4" in skipped and "good.mp4" not in skipped
    detected.clear()
    freekiki.detect_videos(ws, src, output_dir=tmp_path / "out2", screen=False)
    assert sorted(detected) == ["good.mp4", "zoom.mp4"]  # --no-screen runs everything
