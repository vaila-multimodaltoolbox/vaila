"""FreeKiki ``pseudo``: complete frames from the network points + the fitted field camera.

Version: 0.4.8
Update Date: 14 October 2026
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from vaila import freekiki

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_kiki49_build import HEIGHT, WIDTH, left_camera, projected_kiki, render_pitch  # noqa: E402

cv2 = pytest.importorskip("cv2")


def _net(cam, drop=()):
    """Network output of a perfect network: every in-view point accepted except ``drop``."""
    uv, ok = projected_kiki(cam)
    codes = ["D" if ok[i] and i not in drop else "Rk" for i in range(49)]
    return uv, np.where(ok, 0.9, 0.0), codes, ok


def test_a_frame_with_a_good_camera_gets_a_complete_label():
    cam = left_camera()
    uv, kc, codes, ok = _net(cam, drop=(5,))  # the network misses p5 (a rare point)
    points, origin = freekiki.pseudo_label_frame(uv, kc, codes, render_pitch(cam), WIDTH, HEIGHT)
    assert points is not None, origin
    assert all(
        (p is not None) == bool(ok[i]) for i, p in enumerate(points)
    )  # "not in view" = absent
    if ok[5]:
        assert points[5] is not None and np.hypot(*(np.array(points[5]) - uv[5])) < 2.0
    assert len(origin) == 49 and set(origin) <= set("023")


def test_each_gate_rejects_its_frame():
    cam = left_camera()
    uv, kc, codes, ok = _net(cam)
    pitch = render_pitch(cam)
    blank = np.full_like(pitch, (40, 120, 40))
    run = freekiki.pseudo_label_frame
    assert run(uv, kc, codes, blank, WIDTH, HEIGHT)[1] == "line_support"  # nothing painted there
    assert run(uv, kc, ["Rk"] * 49, pitch, WIDTH, HEIGHT)[1] == "few_points"
    assert run(uv, kc, codes, pitch, WIDTH, HEIGHT, cam_height=(30.0, 45.0))[1] == "camera_height"
    noisy = uv + np.random.default_rng(0).normal(0.0, 2.0, uv.shape)
    assert run(noisy, kc, codes, pitch, WIDTH, HEIGHT, max_rmse=0.3)[1] == "camera_rmse"
    off = uv.copy()  # a flag the network accepted far from where the camera puts it
    aux = [i for i in (34, 35, 38, 39, 42, 43, 46, 47) if ok[i] and codes[i] == "D"]
    if aux:
        off[aux[0]] += 60.0
        assert run(off, kc, codes, pitch, WIDTH, HEIGHT, max_rmse=50.0)[1] in (
            "aux_disagree",
            "camera_rmse",
        )


def test_pseudo_cli_parses():
    a = freekiki.build_parser().parse_args(["pseudo", "-w", "W", "--batch", "B", "--every", "5"])
    assert a.batch_dir == "B" and a.every == 5 and a.tau == 0.35 and a.min_accepted == 8


def test_clips_of_one_game_never_share_a_pseudo_name():
    """Long clip names lost their ``_frame_<a>_to_<b>`` tail to a cut and clips overwrote each other."""
    base = "001_vitoria_4_x_0_chapecoense_melhores_momentos_29a_rodada_brasileirao_2026_gegl"
    a, b = f"{base}_frame_1406_to_1470", f"{base}_frame_1481_to_1551"
    assert len(a) > 80 and a[:80] == b[:80]
    na, nb = freekiki._pseudo_name(a, 65), freekiki._pseudo_name(b, 65)
    assert na != nb
    ka, kb = freekiki._pseudo_key(na + ".jpg"), freekiki._pseudo_key(nb + ".jpg")
    assert ka is not None and kb is not None and ka[0] != kb[0] and ka[1] == kb[1] == 65
    assert freekiki._pseudo_key("soccernet__x.jpg") is None


def _review_csv(path, per_point):
    """human_vs_ai.csv rows: per_point = {point: [error_px | 'hid', ...]} (all drawn, conf 0.9)."""
    import csv as _csv

    fields = ["point", "confidence", "error_px", "status"]
    with path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p, errs in per_point.items():
            for e in errs:
                hid = e == "hid"
                w.writerow(
                    {
                        "point": p,
                        "confidence": 0.9,
                        "error_px": "" if hid or e == 0 else e,
                        "status": "human_absent"
                        if hid
                        else ("accepted" if e == 0 else "corrected"),
                    }
                )
            w.writerow({"point": p, "confidence": 0.0, "error_px": "", "status": "human_absent"})


def test_trust_comes_from_how_often_the_reviewer_moved_a_drawn_point(tmp_path):
    csv_path = tmp_path / "human_vs_ai.csv"
    _review_csv(
        csv_path,
        {
            5: [0] * 30,  # always kept
            6: [0] * 25 + [2.0] * 10,  # nudged by 2 px: still fine (tolerance 6 px)
            29: [0] * 2 + [16.0] * 48,  # wrong in almost every frame
            36: [0] * 26 + [13.0] * 94,
            9: [0] * 5,  # too few frames to know
            7: [0] * 22 + ["hid"] * 3,  # a few hidden: 12 % bad, fine
        },
    )
    trust = freekiki.pseudo_point_trust([csv_path])
    assert trust[5]["trusted"] and trust[6]["trusted"] and trust[7]["trusted"]
    assert not trust[29]["trusted"] and trust[29]["bad_rate"] == 0.96
    assert not trust[36]["trusted"]
    assert not trust[9]["trusted"] and trust[9]["n"] == 5  # unknown counts as untrusted
    bad = freekiki.untrusted_points(trust)
    assert {29, 36, 9, 0, 48} <= bad and not ({5, 6, 7} & bad)  # never measured = untrusted


def test_a_frame_showing_an_untrusted_point_is_rejected_not_written_incomplete():
    cam = left_camera()
    uv, kc, codes, ok = _net(cam)
    img = render_pitch(cam)
    seen = int(np.flatnonzero(ok)[0])
    run = freekiki.pseudo_label_frame
    assert run(uv, kc, codes, img, WIDTH, HEIGHT)[0] is not None
    assert (
        run(uv, kc, codes, img, WIDTH, HEIGHT, untrusted=frozenset({seen}))[1] == "untrusted_point"
    )
    out_of_view = int(np.flatnonzero(~ok)[0])  # a point outside the picture does not matter
    assert run(uv, kc, codes, img, WIDTH, HEIGHT, untrusted=frozenset({out_of_view}))[0] is not None


def test_pseudo_cli_takes_trust_tables():
    a = freekiki.build_parser().parse_args(
        ["pseudo", "-w", "W", "--batch", "B", "--trust-csv", "a.csv", "--trust-csv", "b.csv"]
    )
    assert a.trust_csv == ["a.csv", "b.csv"]
