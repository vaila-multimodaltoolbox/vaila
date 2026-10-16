"""FreeKiki ``select``: windows where the rare keypoints show, cut to a folder.

Version: 0.4.7
Update Date: 08 October 2026
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from vaila import freekiki

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_freekiki_queue_need import H, W, _rows, _view  # noqa: E402


def _series(frames, hit_frames, point=24, conf=0.4):
    """Rows of a stride-5 detect run; ``point`` has raw conf ``conf`` on ``hit_frames``."""
    uv, inside = _view()
    status, raw = [], []
    for f in frames:
        kc = np.where(inside, 0.9, 0.0)
        kc[point] = conf if f in hit_frames else 0.0
        codes = ["D" if inside[i] and i != point else "Rk" for i in range(49)]
        s, r = _rows(f, uv, kc, codes)
        status.append(s)
        raw.append(r)
    return status, raw


def _windows(status, raw, **kw):
    kw = {"low_conf": 0.1, "pad": 10, "join_gap": 30, "min_len": 20, "max_len": 400} | kw
    return freekiki.select_windows(status, raw, (24,), W, H, 1000, **kw)


def test_hits_merge_into_padded_windows():
    frames = range(0, 1000, 5)
    # a camera is fitted on every frame and p24 is in view, so every frame is a hit:
    # use a point the camera does not place in the picture to test the confidence rule
    uv, inside = _view()
    out = [i for i in range(49) if not inside[i]][0]
    status, raw = [], []
    hit = set(range(100, 160, 5)) | set(range(500, 520, 5))
    for f in frames:
        kc = np.where(inside, 0.9, 0.0)
        kc[out] = 0.3 if f in hit else 0.0
        codes = ["D" if inside[i] else "Rk" for i in range(49)]
        s, r = _rows(f, uv, kc, codes)
        status.append(s)
        raw.append(r)
    win = freekiki.select_windows(
        status, raw, (out,), W, H, 1000, low_conf=0.1, pad=10, join_gap=30, min_len=20, max_len=400
    )
    assert [(w["start"], w["end"]) for w in win] == [(90, 165), (490, 525)]
    assert win[0]["hits"] == 12 and win[0]["points"] == [out]


def test_camera_alone_marks_a_frame_and_low_conf_is_the_threshold():
    status, raw = _series(range(0, 100, 5), hit_frames=set())
    win = _windows(status, raw)  # p24 conf 0 but the camera says it is in the picture
    assert len(win) == 1 and win[0]["start"] == 0 and win[0]["end"] == 105
    # a confidence below low_conf and no camera: nothing to select
    uv, inside = _view()
    out = [i for i in range(49) if not inside[i]][0]
    kc = np.where(inside, 0.9, 0.0)
    kc[out] = 0.05
    s, r = _rows(0, uv, kc, ["D" if inside[i] else "Rk" for i in range(49)])
    assert freekiki.select_windows([s], [r], (out,), W, H, 100, low_conf=0.1) == []
    kc[out] = 0.2
    s, r = _rows(0, uv, kc, ["D" if inside[i] else "Rk" for i in range(49)])
    assert len(freekiki.select_windows([s], [r], (out,), W, H, 100, low_conf=0.1, min_len=10)) == 1


def test_window_limits_and_skip():
    status, raw = _series(range(0, 1000, 5), hit_frames=set())
    long = freekiki.select_windows(
        status, raw, (24,), W, H, 1000, pad=20, max_len=200, join_gap=30, per_video=5
    )
    assert len(long) == 1 and long[0]["end"] - long[0]["start"] + 1 <= 200
    labelled = set(range(0, 1000))
    assert freekiki.select_windows(status, raw, (24,), W, H, 1000, skip=labelled) == []


def test_find_videos_walks_subfolders(tmp_path):
    (tmp_path / "001" / "dl").mkdir(parents=True)
    (tmp_path / "001" / "dl" / "a.mp4").write_bytes(b"x")
    (tmp_path / "001" / "dl" / "video_info.txt").write_text("x")
    (tmp_path / "b.MKV").write_bytes(b"x")
    assert [p.name for p in freekiki.find_videos(tmp_path)] == ["a.mp4", "b.MKV"]
    assert [p.name for p in freekiki.find_videos(tmp_path, recursive=False)] == ["b.MKV"]
    assert freekiki.find_videos(tmp_path / "b.MKV")[0].name == "b.MKV"


def test_select_cli_parses():
    a = freekiki.build_parser().parse_args(
        ["select", "-w", "W", "--src", "S", "--out", "O", "--low-conf", "0.05", "--plan-only"]
    )
    assert a.low_conf == 0.05 and a.plan_only and a.need == "p5,p29,p39,p47" and a.stride == 5
    with pytest.raises(SystemExit):
        freekiki.build_parser().parse_args(["select", "-w", "W"])


def _frames_with_camera(heights_rmse):
    """Rows of a run whose camera fit is (height m, rmse px) per frame, via a monkeypatched fit."""
    uv, inside = _view()
    status, raw = [], []
    for f in range(len(heights_rmse)):
        s, r = _rows(
            f * 5, uv, np.where(inside, 0.9, 0.0), ["D" if inside[i] else "Rk" for i in range(49)]
        )
        status.append(s)
        raw.append(r)
    return status, raw


def test_a_camera_that_does_not_fit_the_pitch_makes_no_hit(monkeypatch):
    """The network accepts many points behind the net, but the camera fit is poor / too low."""
    from vaila import freekiki_geom

    real = freekiki_geom.fit_field_camera
    plan = {}

    def fake(xy, ok, w, h, settings=None):
        m = real(xy, ok, w, h, settings)
        height, rmse = plan["current"]
        return m | {"cam_height_m": height, "rmse_px": rmse}

    monkeypatch.setattr(freekiki_geom, "fit_field_camera", fake)
    status, raw = _frames_with_camera([0] * 12)

    def run(height, rmse, **kw):
        plan["current"] = (height, rmse)
        return freekiki.select_windows(status, raw, (24,), W, H, 100, pad=5, min_len=10, **kw)

    assert len(run(12.0, 3.0)) == 1  # a plausible broadcast camera
    assert run(12.0, 8.0) == []  # points do not lie on a plane: rmse too high
    assert run(3.0, 3.0) == []  # a camera at 3 m is behind the net / on the touchline
    assert run(70.0, 3.0) == []  # nor a drone
    assert len(run(12.0, 8.0, max_rmse=10.0)) == 1  # limits are options
    assert len(run(3.0, 3.0, cam_height=(2.0, 45.0))) == 1


def test_invalid_camera_frames_are_listed_on_the_window(monkeypatch):
    from vaila import freekiki_geom

    real = freekiki_geom.fit_field_camera
    calls = {"n": 0}

    def fake(xy, ok, w, h, settings=None):
        calls["n"] += 1
        bad = calls["n"] in (5, 6)  # two frames of the stretch look like a net camera
        return real(xy, ok, w, h, settings) | {"cam_height_m": 3.0 if bad else 15.0, "rmse_px": 2.0}

    monkeypatch.setattr(freekiki_geom, "fit_field_camera", fake)
    status, raw = _frames_with_camera([0] * 12)
    win = freekiki.select_windows(status, raw, (24,), W, H, 100, pad=5, min_len=10)
    assert len(win) == 1 and win[0]["invalid_frames"] == [20, 25]
    assert win[0]["hits"] == 10  # the two invalid frames are not hits


def test_select_cli_has_the_camera_options():
    a = freekiki.build_parser().parse_args(["select", "-w", "W", "--src", "S", "--out", "O"])
    assert a.max_rmse == 5.0 and a.cam_height == "8,45"
    assert freekiki._height_range("8, 45") == (8.0, 45.0)
    with pytest.raises(ValueError):
        freekiki._height_range("45,8")
