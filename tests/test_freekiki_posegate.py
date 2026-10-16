"""FreeKiki close-up gate: pose size from MediaPipe, window splitting, ``select`` integration.

Version: 0.4.8
Update Date: 08 October 2026
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vaila import freekiki
from vaila import freekiki_posegate as pg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_freekiki_queue_need import H, W, _rows, _view  # noqa: E402


def _pose(y0, y1, vis=1.0, n=5):
    ys = np.linspace(y0, y1, n)
    return [SimpleNamespace(x=0.5, y=float(y), visibility=vis) for y in ys]


def test_pose_fraction_is_the_largest_visible_pose():
    assert pg.pose_fraction(SimpleNamespace(pose_landmarks=[])) == 0.0
    assert pg.pose_fraction(SimpleNamespace(pose_landmarks=None)) == 0.0
    one = SimpleNamespace(pose_landmarks=[_pose(0.30, 0.45)])
    assert pg.pose_fraction(one) == pytest.approx(0.15)
    many = SimpleNamespace(pose_landmarks=[_pose(0.30, 0.34), _pose(0.1, 0.7), _pose(0.5, 0.6)])
    assert pg.pose_fraction(many) == pytest.approx(0.6)


def test_hidden_landmarks_do_not_inflate_the_pose():
    pose = _pose(0.40, 0.44) + [SimpleNamespace(x=0.5, y=0.99, visibility=0.05)]
    assert pg.pose_fraction(SimpleNamespace(pose_landmarks=[pose])) == pytest.approx(0.04)


def _samples(frames, bad):
    return [(f, 0.3 if f in bad else 0.02) for f in frames]


def test_split_good_keeps_a_clean_window():
    s = _samples(range(100, 400, 10), set())
    assert pg.split_good(100, 399, s, every=10, min_len=50) == [(100, 399)]


def test_split_good_cuts_out_the_close_up_and_keeps_long_pieces():
    s = _samples(range(100, 400, 10), {250, 260})
    got = pg.split_good(100, 399, s, every=10, min_len=50)
    assert got == [(100, 244), (265, 399)]
    # the close-up stretch itself is not inside any piece
    assert all(not (a <= 250 <= b or a <= 260 <= b) for a, b in got)


def test_split_good_drops_short_pieces_and_all_bad_windows():
    s = _samples(range(100, 200, 10), {130})
    assert pg.split_good(100, 199, s, every=10, min_len=50) == [(135, 199)]
    bad = _samples(range(100, 200, 10), set(range(100, 200, 10)))
    assert pg.split_good(100, 199, bad, every=10, min_len=50) == []


def test_check_window_samples_every_nth_frame(tmp_path):
    video = tmp_path / "v.mp4"
    out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
    for i in range(60):
        out.write(np.full((48, 64, 3), i * 4, dtype=np.uint8))
    out.release()

    class Gate:
        max_pose = 0.1

        def fraction(self, img):
            return 0.5 if img.mean() > 100 else 0.0

    seen = []
    got = pg.check_window(video, 10, 49, Gate(), every=10, on_big=lambda f, img, fr: seen.append(f))
    assert [f for f, _ in got] == [10, 20, 30, 40]
    assert [f for f, fr in got if fr > 0.1] == [30, 40] and seen == [30, 40]


def _bad_frame_trim(bad_frames, every=10, min_len=20):
    def trim(w):
        samples = [
            (f, 0.3 if f in bad_frames else 0.0) for f in range(w["start"], w["end"] + 1, every)
        ]
        pieces = pg.split_good(w["start"], w["end"], samples, every=every, min_len=min_len)
        return pieces, {"max_pose": max(f for _, f in samples), "checked_frames": len(samples)}

    return trim


def test_select_windows_trims_with_the_gate_and_recounts_hits():
    uv, inside = _view()
    out = [i for i in range(49) if not inside[i]][0]
    status, raw = [], []
    for f in range(0, 400, 5):
        kc = np.where(inside, 0.9, 0.0)
        kc[out] = 0.6 if 100 <= f < 300 else 0.0
        codes = ["D" if inside[i] else "Rk" for i in range(49)]
        s, r = _rows(f, uv, kc, codes)
        status.append(s)
        raw.append(r)
    kw = {
        "low_conf": 0.5,
        "pad": 10,
        "join_gap": 30,
        "min_len": 20,
        "max_len": 600,
        "min_accepted": 99,
    }
    plain = freekiki.select_windows(status, raw, (out,), W, H, 400, **kw)
    assert [(w["start"], w["end"]) for w in plain] == [(90, 305)]
    rejected = []
    trimmed = freekiki.select_windows(
        status, raw, (out,), W, H, 400, trim=_bad_frame_trim({200}), rejected=rejected, **kw
    )
    assert [(w["start"], w["end"]) for w in trimmed] == [(90, 194), (205, 305)]
    assert all(w["trimmed_from"] == "91-306" and w["max_pose"] == 0.3 for w in trimmed)
    assert sum(w["hits"] for w in trimmed) < plain[0]["hits"]  # the close-up hits are gone
    assert rejected == []
    dropped = []
    none = freekiki.select_windows(
        status, raw, (out,), W, H, 400,
        trim=_bad_frame_trim(set(range(0, 400))), rejected=dropped, **kw,
    )  # fmt: skip
    assert none == [] and [r["reason"] for r in dropped] == ["close_up"]


def test_a_zoomed_camera_with_few_accepted_points_is_not_trusted():
    uv, inside = _view()
    kc = np.zeros(49)
    few = np.flatnonzero(inside)[:5]
    kc[few] = 0.9
    s, r = _rows(0, uv, kc, ["D" if i in few else "Rk" for i in range(49)])
    kw = {"low_conf": 0.99, "min_len": 10, "pad": 5}
    assert freekiki.select_windows([s], [r], (24,), W, H, 100, min_accepted=6, **kw) == []


def test_select_cli_has_the_gate_options():
    a = freekiki.build_parser().parse_args(
        ["select", "-w", "W", "--src", "S", "--out", "O", "--max-pose", "0.2", "--no-pose-gate"]
    )
    assert a.max_pose == 0.2 and a.no_pose_gate and a.kp_conf is None and a.min_accepted == 6
    b = freekiki.build_parser().parse_args(["select", "-w", "W", "--src", "S", "--out", "O"])
    assert b.max_pose == 0.10 and not b.no_pose_gate and b.pose_complexity == 0
    assert b.check_every == 10 and b.low_conf is None
