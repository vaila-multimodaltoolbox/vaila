"""kiki49_build (ported from mkvis3d): line support, clicks, SoccerNet lines, splits.

The synthetic broadcast camera and pitch rendering come from mkvis3d
``tests/soccer_field_synth.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vaila"))

from kiki49_build import Options, hashed_split  # noqa: E402
from kiki49_build.camera import Camera, homography_from_camera, in_view, project  # noqa: E402
from kiki49_build.clicks import ARC_TOL_M, camera_agreement, clicks_to_sample  # noqa: E402
from kiki49_build.kiki49 import (  # noqa: E402
    ARC_POINTS,
    LEGACY_GOAL_Y_POINTS_XY,
    N_KPT,
    ground_lines,
    load_kiki49,
)
from kiki49_build.line_support import line_distance_map, support_score  # noqa: E402
from kiki49_build.sample import (  # noqa: E402
    ORIGIN_ANNOTATED,
    ORIGIN_CAMERA,
    ORIGIN_PLANE,
    Reject,
    Sample,
    project_plane,
)
from kiki49_build.soccernet import sample_from_lines  # noqa: E402
from kiki49_build.soccernet_lines import (  # noqa: E402
    CIRCLE_CROSSINGS,
    GROUND_LINES,
    POSTS,
    parse_annotation,
)

OPTS = Options()


WIDTH, HEIGHT = 1280, 720


def look_at(
    centre: tuple[float, float, float],
    target: tuple[float, float, float],
    f: float = 1400.0,
    width: int = WIDTH,
    height: int = HEIGHT,
    dist: np.ndarray | None = None,
) -> Camera:
    """Pinhole camera at ``centre`` looking at ``target`` (z up, image y down)."""
    C = np.asarray(centre, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64) - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    K = np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])
    return Camera(K=K, R=R, t=-R @ C, dist=dist)


def left_camera() -> Camera:
    """Main-stand camera framing the left penalty box and goal."""
    return look_at((-25.0, -55.0, 22.0), (-40.0, 0.0, 0.0))


def render_pitch(cam: Camera, shift_px: float = 0.0) -> np.ndarray:
    """Green image with the painted markings of ``cam`` drawn in white
    (optionally misregistered by ``shift_px`` along both image axes)."""
    img = np.full((HEIGHT, WIDTH, 3), (40, 120, 40), dtype=np.uint8)
    for poly in ground_lines(0.1):
        uv, depth = project(poly, cam)
        uv = uv[depth > 0] + shift_px
        if len(uv) >= 2:
            cv2.polylines(img, [np.round(uv).astype(np.int32)], False, (255, 255, 255), 2)
    return img


def projected_kiki(cam: Camera) -> tuple[np.ndarray, np.ndarray]:
    """(uv (49, 2), in-view mask (49,)) of every kiki point."""
    uv, depth = project(load_kiki49().xyz, cam)
    return uv, in_view(uv, depth, WIDTH, HEIGHT)


def _legacy_uv(cam, k: int) -> np.ndarray:
    return project(np.array([[*LEGACY_GOAL_Y_POINTS_XY[k], 0.0]]), cam)[0][0]


def _clicks(cam) -> np.ndarray:
    uv, ok = projected_kiki(cam)
    clicks = np.zeros((32, 3))
    vis = ok[:32]
    clicks[vis, :2], clicks[vis, 2] = uv[:32][vis], 1.0
    return clicks


def test_line_support_aligned_vs_shifted() -> None:
    cam = left_camera()
    H = homography_from_camera(cam)

    def fn(X: np.ndarray):
        return project_plane(H, X[:, :2], WIDTH, HEIGHT)

    assert support_score(line_distance_map(render_pitch(cam)), fn) > 0.95
    assert support_score(line_distance_map(render_pitch(cam, shift_px=15.0)), fn) < 0.1

    def outside(X: np.ndarray):
        return np.full((len(X), 2), -5.0), np.ones(len(X))

    assert np.isnan(support_score(np.zeros((HEIGHT, WIDTH), np.float32), outside))


def test_camera_agreement_drops_legacy_arc_click() -> None:
    cam = left_camera()
    clicks = _clicks(cam)
    clicks[10, :2] = _legacy_uv(cam, 10)
    ann = np.zeros((N_KPT, 3))
    ann[:32] = clicks
    agreement, cleaned, n_legacy = camera_agreement(ann, cam, WIDTH)
    assert n_legacy == 1 and not cleaned[10].any()
    assert agreement == pytest.approx(0.0, abs=1e-6)


def test_clicks_upgrade_through_click_homography() -> None:
    cam = left_camera()
    img = render_pitch(cam)
    uv, ok = projected_kiki(cam)
    clicks = _clicks(cam)
    clicks[11, :2] = _legacy_uv(cam, 11)  # old pitch32 meaning of point 11
    got = clicks_to_sample("t", "u", "train", "g", clicks, img, Path("x.jpg"), OPTS)
    assert isinstance(got, Sample), got
    assert got.stats["legacy_arc"] == 1.0
    assert got.stats["click_residual"] < 1e-3
    assert got.origin[11] == ORIGIN_PLANE
    np.testing.assert_allclose(got.kps[11, :2], uv[11], atol=1e-3)
    clicked = np.flatnonzero(clicks[:, 2] > 0)
    clicked = clicked[clicked != 11]
    assert (got.origin[clicked] == ORIGIN_ANNOTATED).all()
    assert got.aux3d and (got.kps[ok, 2] == 2).all() and not got.kps[~ok].any()
    np.testing.assert_allclose(got.kps[ok, :2], uv[ok], atol=0.05)


def test_clicks_prefer_an_agreeing_tracked_camera() -> None:
    cam = left_camera()
    uv, ok = projected_kiki(cam)
    clicks = _clicks(cam)
    got = clicks_to_sample(
        "t", "u", "train", "g", clicks, render_pitch(cam), Path("x.jpg"), OPTS, cam
    )
    assert isinstance(got, Sample) and got.H is None and got.camera is cam
    assert (got.origin[32:][ok[32:]] == ORIGIN_CAMERA).all()
    np.testing.assert_allclose(got.kps[ok, :2], uv[ok], atol=1e-6)


def test_clicks_reject_without_line_support() -> None:
    cam = left_camera()
    blank = np.full((HEIGHT, WIDTH, 3), (40, 120, 40), dtype=np.uint8)
    got = clicks_to_sample("t", "u", "train", "g", _clicks(cam), blank, Path("x.jpg"), OPTS)
    assert isinstance(got, Reject) and got.reason == "line_support"


def _soccernet_json(cam) -> dict[str, list[dict[str, float]]]:
    """Normalized polylines of what ``cam`` sees, in calibration-2023 layout."""

    def poly(X: np.ndarray) -> list[dict[str, float]]:
        uv, depth = project(X, cam)
        keep = (
            (depth > 0)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < WIDTH)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < HEIGHT)
        )
        return [{"x": u / WIDTH, "y": v / HEIGHT} for u, v in uv[keep]]

    def seg(a, b, n: int = 40) -> np.ndarray:
        return np.linspace(a, b, n)

    ann = {n: poly(seg([*a, 0.0], [*b, 0.0])) for n, (a, b) in GROUND_LINES.items()}
    ann["Circle left"] = poly(ground_lines(0.1)[12])  # the left penalty arc
    xyz = load_kiki49().xyz
    for name, (base, top) in POSTS.items():
        ann[name] = poly(seg(xyz[base], xyz[top]))
    ann["Goal left crossbar"] = poly(seg(xyz[34], xyz[35]))
    return {k: v for k, v in ann.items() if len(v) >= 2}


def test_soccernet_lines_to_kiki_points() -> None:
    cam = left_camera()
    uv, _ = projected_kiki(cam)
    lab = parse_annotation(_soccernet_json(cam), WIDTH, HEIGHT)
    assert lab.H is not None
    np.testing.assert_allclose(lab.H, homography_from_camera(cam), rtol=1e-5, atol=1e-8)
    got = np.flatnonzero(lab.annotated[:, 2] > 0)
    far, near = CIRCLE_CROSSINGS[("Circle left", "Big rect. left main")]
    assert {1, 2, 3, 4, 6, 7, 9, 12, far, near, 32, 33, 34, 35} <= set(got.tolist())
    np.testing.assert_allclose(lab.annotated[got, :2], uv[got], atol=0.05)
    assert lab.stats["post_named_err"] < 0.05 < lab.stats["post_swap_err"]
    assert np.median(lab.residual_px) < 0.05


def test_hashed_split_is_deterministic_and_balanced() -> None:
    groups = [f"clip:{i}" for i in range(2000)]
    splits = [hashed_split(g) for g in groups]
    assert splits == [hashed_split(g) for g in groups]
    frac = {s: splits.count(s) / len(splits) for s in ("train", "val", "test")}
    assert frac["train"] == pytest.approx(0.70, abs=0.04)
    assert frac["val"] == pytest.approx(0.15, abs=0.03)
    assert frac["test"] == pytest.approx(0.15, abs=0.03)


def test_arc_points_differ_from_legacy_by_more_than_the_tolerance() -> None:
    xyz = load_kiki49().xyz
    for k in ARC_POINTS:
        assert np.hypot(*(xyz[k, :2] - LEGACY_GOAL_Y_POINTS_XY[k])) > 1.5 * ARC_TOL_M


def _clicks49(cam, keep) -> np.ndarray:
    uv, ok = projected_kiki(cam)
    ann = np.zeros((N_KPT, 3))
    sel = np.zeros(N_KPT, dtype=bool)
    sel[list(keep)] = True
    sel &= ok
    ann[sel, :2], ann[sel, 2] = uv[sel], 1.0
    return ann


def test_mapped_49_clicks_keep_off_ground_point_only_when_camera_agrees() -> None:
    cam = left_camera()
    uv, ok = projected_kiki(cam)
    ground = [k for k in range(32) if ok[k]]
    post_top = 35  # left_goal_top_post_top, z = 2.44 m
    assert ok[post_top]
    ann = _clicks49(cam, [*ground, post_top])
    got = clicks_to_sample("t", "u", "train", "g", ann, render_pitch(cam), Path("x.jpg"), OPTS)
    assert isinstance(got, Sample), got
    assert got.origin[post_top] == ORIGIN_ANNOTATED
    np.testing.assert_allclose(got.kps[post_top, :2], uv[post_top], atol=1e-6)

    bad = ann.copy()
    bad[post_top, 1] += 80.0  # a post-top click far from where the camera puts it
    got = clicks_to_sample("t", "u", "train", "g", bad, render_pitch(cam), Path("x.jpg"), OPTS)
    assert isinstance(got, Sample), got
    assert got.origin[post_top] == ORIGIN_CAMERA  # the click was dropped, the camera filled it
    np.testing.assert_allclose(got.kps[post_top, :2], uv[post_top], atol=0.5)


def test_off_ground_clicks_alone_cannot_fit_the_plane() -> None:
    cam = left_camera()
    ann = _clicks49(cam, [34, 35, 38, 2])
    got = clicks_to_sample("t", "u", "train", "g", ann, render_pitch(cam), Path("x.jpg"), OPTS)
    assert isinstance(got, Reject) and got.reason == "few_clicks"


def test_sample_from_lines_completes_the_49_points() -> None:
    cam = left_camera()
    uv, ok = projected_kiki(cam)
    got = sample_from_lines(
        _soccernet_json(cam),
        WIDTH,
        HEIGHT,
        source="t",
        uid="u",
        split="train",
        group="g",
        image=None,
        opts=OPTS,
    )
    assert isinstance(got, Sample), got
    vis = got.kps[:, 2] > 0
    assert vis.sum() >= 15 and not (vis & ~ok).any()
    np.testing.assert_allclose(got.kps[vis, :2], uv[vis], atol=0.1)


def test_degenerate_homography_is_rejected_not_raised() -> None:
    from kiki49_build.soccernet_lines import _Evidence, refine_homography

    ev = _Evidence(pt_world=[np.zeros(2)] * 4, pt_img=[np.zeros(2)] * 4)
    assert refine_homography(np.zeros((3, 3)), ev, 2.0) is None  # singular, H[2,2] = 0
    flat = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert refine_homography(flat, ev, 2.0) is None  # inv(H) does not exist
