"""Quick Measure on a moving camera: points are anchored on the pitch, not on the screen.

With a per-frame ``camera3d`` calibration a point clicked on frame A must be drawn, on frame B,
where that spot of the grass is seen by the camera of frame B (world = ground point of the click
frame, screen = projection with the current frame). The measured values never change. Every
other calibration kind keeps drawing at the clicked pixel.

Synthetic broadcast camera: fixed centre, the pan and the zoom change on every frame.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pytest

from vaila.kick_measure import CameraTrack
from vaila.quickmeasure import (
    CalibrationDraft,
    QuickMeasureCalibration,
    QuickMeasureSession,
    anchored_edge_marker,
    draw_quickmeasure_overlay,
)

W, H = 1920, 1080
CENTRE = np.array([0.0, -75.0, 22.0])
N_FRAMES = 60
ORANGE = (255, 180, 0)  # draft colour of the overlay
MAGENTA = (255, 0, 255)  # finished-result colour of the overlay


def _camera(frame: int) -> np.ndarray:
    """3x4 camera of ``frame``: pans 0.5 m per frame to +x and zooms 10 px per frame."""
    target = np.array([-20.0 + 0.5 * frame, 0.0, 0.0])
    zc = (target - CENTRE) / np.linalg.norm(target - CENTRE)
    xc = np.cross(zc, [0.0, 0.0, 1.0])
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R = np.array([xc, yc, zc])
    f = 2000.0 + 10.0 * frame
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    return K @ np.c_[R, -R @ CENTRE]


@pytest.fixture(scope="module")
def track() -> CameraTrack:
    return CameraTrack.from_matrices({f: _camera(f) for f in range(N_FRAMES)}, n_frames=N_FRAMES)


def _px(frame: int, X: float, Y: float, Z: float = 0.0) -> tuple[float, float]:
    h = _camera(frame) @ np.array([X, Y, Z, 1.0])
    return float(h[0] / h[2]), float(h[1] / h[2])


W1 = (-12.0, -2.0)  # clicked on frame 0
W2 = (10.0, 3.0)  # clicked on frame 40


def _moving_session(track: CameraTrack) -> QuickMeasureSession:
    session = QuickMeasureSession(fps=30.0)
    session.set_calibration(QuickMeasureCalibration.from_camera_track(track))
    return session


def _distance_result_session(track: CameraTrack) -> QuickMeasureSession:
    """A finished distance (point 1 on frame 0, point 2 on frame 40)."""
    session = _moving_session(track)
    session.add_point(0, *_px(0, *W1))
    session.add_point(40, *_px(40, *W2))
    result = session.measure("distance")
    assert result["value"] == pytest.approx(np.hypot(W2[0] - W1[0], W2[1] - W1[1]), abs=1e-6)
    return session


# ---------------------------------------------------------------------------
# (a) (b) the anchor
# ---------------------------------------------------------------------------


def test_anchored_pixel_follows_the_current_camera(track) -> None:
    session = _moving_session(track)
    click0 = _px(0, *W1)
    world = session.world_anchor(0, *click0)
    assert world is not None
    assert world == pytest.approx(W1, abs=1e-6)
    uv40 = session.world_to_pixel_at(40, *world)
    assert uv40 is not None
    expected = track.project(40, np.array([[W1[0], W1[1], 0.0]]))[0]
    assert uv40 == pytest.approx(tuple(expected), abs=1e-6)
    assert np.hypot(uv40[0] - click0[0], uv40[1] - click0[1]) > 5.0  # not where it was clicked


def test_round_trip_on_the_click_frame_is_exact(track) -> None:
    session = _moving_session(track)
    for frame in (0, 17, 40, 59):
        for X, Y in ((-12.0, -2.0), (10.0, 3.0), (-30.0, 8.0)):
            click = _px(frame, X, Y)
            world = session.world_anchor(frame, *click)
            assert world is not None
            back = session.world_to_pixel_at(frame, *world)
            assert back is not None
            assert np.hypot(back[0] - click[0], back[1] - click[1]) < 0.01


def test_anchor_is_none_without_camera3d_or_above_the_horizon(track) -> None:
    session = _moving_session(track)
    assert session.world_anchor(0, 960.0, -5000.0) is None  # the ray never reaches the grass
    assert session.world_to_pixel_at(0, float("nan"), 0.0) is None
    empty = QuickMeasureSession()
    assert empty.world_anchor(0, 10.0, 10.0) is None
    assert empty.world_to_pixel_at(0, 0.0, 0.0) is None


# ---------------------------------------------------------------------------
# (c) (d) off-screen and behind the camera
# ---------------------------------------------------------------------------


def test_point_that_leaves_the_image_is_reported_off_screen_with_its_direction(track) -> None:
    session = _moving_session(track)
    world = session.world_anchor(0, *_px(0, -40.0, -2.0))
    assert world is not None
    uv = session.world_to_pixel_at(40, *world)
    assert uv is not None and uv[0] < 0  # the camera panned right: the spot left the image
    marker = anchored_edge_marker(uv, (0, 0, W, H), margin=14)
    assert marker is not None
    ex, ey, dx, dy = marker
    assert ex == pytest.approx(14.0)  # on the LEFT edge, inside by the margin
    assert 0 <= ey <= H
    assert dx < 0 and abs(dx) > abs(dy) * 0.1  # arrow points to the left
    assert np.hypot(dx, dy) == pytest.approx(1.0)
    # a point on the image has no marker
    assert anchored_edge_marker((100.0, 100.0), (0, 0, W, H), margin=14) is None
    # a point far to the right / below
    right = anchored_edge_marker((W + 500.0, H / 2), (0, 0, W, H), margin=10)
    assert right is not None and right[0] == pytest.approx(W - 10.0) and right[2] > 0
    below = anchored_edge_marker((W / 2, H + 900.0), (0, 0, W, H), margin=10)
    assert below is not None and below[1] == pytest.approx(H - 10.0) and below[3] > 0


def test_point_behind_the_camera_does_not_raise(track) -> None:
    session = _moving_session(track)
    assert session.world_to_pixel_at(0, 0.0, -120.0) is None  # behind the camera centre
    assert session.world_to_pixel_at(0, 0.0, -120.0, 3.0) is None


def test_overlay_draws_a_point_behind_the_camera_as_an_edge_marker(track, monkeypatch) -> None:
    pygame = pytest.importorskip("pygame")
    pygame.font.init()
    font = pygame.font.Font(None, 18)
    session = _distance_result_session(track)
    seen: list[str] = []

    class _Font:
        def render(self, text, *a, **k):
            seen.append(str(text))
            return font.render(text, *a, **k)

    monkeypatch.setattr(session, "world_to_pixel_at", lambda *a, **k: None)
    for lang, word in (("en", "behind camera"), ("pt", "atrás da câmera")):
        seen.clear()
        surface = pygame.Surface((W // 2, H // 2))
        draw_quickmeasure_overlay(surface, session, 0.5, 0, 0, _Font(), frame=20, lang=lang)
        assert any("P1" in t and "q0" in t and word in t for t in seen), seen


# ---------------------------------------------------------------------------
# (e) a fixed camera keeps the old drawing
# ---------------------------------------------------------------------------


def _draw(session, frame, *, lang="en", size=(W // 2, H // 2), zoom=0.5):
    pygame = pytest.importorskip("pygame")
    pygame.font.init()
    surface = pygame.Surface(size)
    surface.fill((0, 0, 0))
    draw_quickmeasure_overlay(
        surface, session, zoom, 0, 0, pygame.font.Font(None, 18), frame=frame, lang=lang
    )
    return surface


@pytest.mark.parametrize("kind", ["line", "plane"])
def test_fixed_camera_calibration_draws_at_the_click_pixels(kind) -> None:
    pygame = pytest.importorskip("pygame")
    if kind == "line":
        calib = QuickMeasureCalibration.from_line_clicks((100, 100), (700, 100), 6.0)
    else:
        draft = CalibrationDraft(mode="plane", unit_label="m")
        for pt in ((200, 200), (800, 200), (800, 700), (200, 700)):
            draft.add_point(*pt)
        calib = draft.build([6.0, 5.0])
    assert calib.kind == kind
    session = QuickMeasureSession(fps=30.0)
    session.set_calibration(calib)
    session.add_point(0, 300.0, 300.0)
    session.add_point(40, 900.0, 500.0)
    session.measure("distance")
    session.set_live_mode("1")
    session.add_live_point(0, 400.0, 400.0)  # a draft point
    # no frame, any frame: the same pixels (the picture does not depend on the frame)
    ref = pygame.image.tobytes(_draw(session, None), "RGB")
    for frame in (0, 20, 40):
        assert pygame.image.tobytes(_draw(session, frame), "RGB") == ref
    surface = _draw(session, 40)
    assert _has_colour(surface, 150, 150, MAGENTA)  # point 1 at (300, 300) * zoom 0.5
    assert _has_colour(surface, 200, 200, ORANGE)  # the draft point at (400, 400) * 0.5


def test_anchored_types_without_camera3d_have_no_anchor() -> None:
    session = QuickMeasureSession()
    session.set_calibration(QuickMeasureCalibration.from_line_clicks((0, 0), (100, 0), 1.0))
    assert session.world_anchor(0, 5.0, 5.0) is None


# ---------------------------------------------------------------------------
# (f) the picture on a headless pygame surface
# ---------------------------------------------------------------------------


def _has_colour(surface, cx: int, cy: int, colour, r: int = 8) -> bool:
    w, h = surface.get_size()
    for x in range(max(0, cx - r), min(w, cx + r + 1)):
        for y in range(max(0, cy - r), min(h, cy + r + 1)):
            if tuple(surface.get_at((x, y)))[:3] == colour:
                return True
    return False


def test_draft_point_is_drawn_where_the_pitch_spot_is_in_the_current_frame(track) -> None:
    pygame = pytest.importorskip("pygame")
    session = _moving_session(track)
    session.set_live_mode("1")
    click0 = _px(0, *W1)
    session.add_live_point(0, *click0)  # draft point 1 of a distance, clicked on frame 0
    zoom = 0.5
    for frame in (0, 20, 40):
        surface = _draw(session, frame)
        ex, ey = _px(frame, *W1)
        assert _has_colour(surface, int(ex * zoom), int(ey * zoom), ORANGE), frame
    old = (int(click0[0] * zoom), int(click0[1] * zoom))
    assert _has_colour(_draw(session, 0), *old, ORANGE)  # click frame: where it was clicked
    assert not _has_colour(_draw(session, 40), *old, ORANGE)  # the camera moved: not there
    assert pygame.image.tobytes(_draw(session, 0), "RGB") != pygame.image.tobytes(
        _draw(session, 40), "RGB"
    )


def test_result_points_and_line_follow_the_pitch(track) -> None:
    session = _distance_result_session(track)
    zoom = 0.5
    click1 = _px(0, *W1)
    surface0 = _draw(session, 0)
    assert _has_colour(surface0, int(click1[0] * zoom), int(click1[1] * zoom), MAGENTA)
    surface40 = _draw(session, 40)
    ax, ay = _px(40, *W1)
    assert _has_colour(surface40, int(ax * zoom), int(ay * zoom), MAGENTA)  # anchored
    p2 = _px(40, *W2)
    assert _has_colour(surface40, int(p2[0] * zoom), int(p2[1] * zoom), MAGENTA)  # click frame
    # the old click pixel of point 1 is empty on frame 40 (line and point moved with the pitch)
    assert not _has_colour(surface40, int(click1[0] * zoom), int(click1[1] * zoom), MAGENTA, r=6)


def test_off_screen_point_gets_an_edge_marker_with_label_and_arrow(track) -> None:
    pygame = pytest.importorskip("pygame")
    pygame.font.init()
    font = pygame.font.Font(None, 18)
    session = _moving_session(track)
    session.add_point(0, *_px(0, -40.0, -2.0))
    session.add_point(40, *_px(40, *W2))
    session.measure("distance")
    seen: list[str] = []

    class _Font:
        def render(self, text, *a, **k):
            seen.append(str(text))
            return font.render(text, *a, **k)

    surface = pygame.Surface((W // 2, H // 2))
    surface.fill((0, 0, 0))
    draw_quickmeasure_overlay(surface, session, 0.5, 0, 0, _Font(), frame=40)
    assert any(t.startswith("P1") and "q0" in t for t in seen), seen
    assert _has_colour(surface, 14, H // 4, MAGENTA, r=H // 4)  # something on the left edge
    # on the click frame there is no marker
    seen.clear()
    surface2 = pygame.Surface((W // 2, H // 2))
    draw_quickmeasure_overlay(surface2, session, 0.5, 0, 0, _Font(), frame=0)
    assert not any(t.startswith("P1") and "q0" in t for t in seen)


def test_height_draws_the_base_on_the_pitch_and_the_top_above_it(track) -> None:
    session = _moving_session(track)
    base_world = (-2.0, 1.0)
    session.add_point(0, *_px(0, *base_world, 0.0))
    session.add_point(0, *_px(0, *base_world, 1.8))
    result = session.measure("height")
    assert result["value"] == pytest.approx(1.8, abs=0.05)
    zoom = 0.5
    surface = _draw(session, 30)
    bx, by = _px(30, *base_world, 0.0)
    tx, ty = _px(30, *base_world, result["value"])
    assert _has_colour(surface, int(bx * zoom), int(by * zoom), MAGENTA)
    assert _has_colour(surface, int(tx * zoom), int(ty * zoom), MAGENTA)


@pytest.mark.parametrize("rtype", ["level", "plane_point", "spin", "height_line", "event"])
def test_image_anchored_types_keep_the_pixels(track, rtype) -> None:
    pygame = pytest.importorskip("pygame")
    session = _moving_session(track)
    session.results.append(
        {
            "type": rtype,
            "value": 1.0,
            "unit": "m",
            "pixels": [(300.0, 300.0), (600.0, 500.0)],
            "frames": [0, 40],
            "point_ids": [1, 2],
        }
    )
    ref = pygame.image.tobytes(_draw(session, 0), "RGB")
    assert pygame.image.tobytes(_draw(session, 40), "RGB") == ref


def test_view_rect_keeps_lines_and_markers_inside_the_image(track) -> None:
    """The host passes the part of the window that shows the video: nothing in the black bars."""
    pygame = pytest.importorskip("pygame")
    pygame.font.init()
    session = _moving_session(track)
    session.add_point(0, *_px(0, -40.0, -2.0))  # leaves the image on the left at frame 40
    session.add_point(40, *_px(40, *W2))
    session.measure("distance")
    surface = pygame.Surface((W // 2, H // 2))
    surface.fill((0, 0, 0))
    view = (100, 0, 600, H // 2)  # image between x = 100 and x = 700, bars on both sides
    draw_quickmeasure_overlay(
        surface, session, 0.5, 0, 0, pygame.font.Font(None, 18), frame=40, view_rect=view
    )
    for x in list(range(0, 100)) + list(range(701, W // 2)):
        for y in range(0, H // 2, 2):
            assert tuple(surface.get_at((x, y)))[:3] != MAGENTA, (x, y)
    # the marker sits just inside the left border of the image (x = 100 + 14)
    assert _has_colour(surface, 114, H // 4, MAGENTA, r=H // 4)
    # no view_rect: the whole window is the view, the line runs on to the window edge
    wide = pygame.Surface((W // 2, H // 2))
    draw_quickmeasure_overlay(wide, session, 0.5, 0, 0, pygame.font.Font(None, 18), frame=40)
    assert any(
        tuple(wide.get_at((x, y)))[:3] == MAGENTA
        for x in range(0, 100)
        for y in range(0, H // 2, 2)
    )


def test_far_off_screen_points_do_not_stack_their_markers(track, monkeypatch) -> None:
    pygame = pytest.importorskip("pygame")
    pygame.font.init()
    session = _moving_session(track)
    session.add_point(40, *_px(40, 25.0, 3.0))
    session.add_point(40, *_px(40, 28.0, 3.0))
    session.measure("distance")
    discs: list[tuple[int, int]] = []
    real_circle = pygame.draw.circle

    def _spy(surf, colour, centre, radius, *a, **k):
        if tuple(colour) == (40, 10, 50) and radius == 6:  # the dark disc of a point marker
            discs.append((int(centre[0]), int(centre[1])))
        return real_circle(surf, colour, centre, radius, *a, **k)

    monkeypatch.setattr(pygame.draw, "circle", _spy)
    surface = pygame.Surface((W // 2, H // 2))
    draw_quickmeasure_overlay(surface, session, 0.5, 0, 0, pygame.font.Font(None, 18), frame=0)
    assert len(discs) == 2  # camera is at x = -20: both spots are off the image on the right
    assert np.hypot(discs[0][0] - discs[1][0], discs[0][1] - discs[1][1]) >= 22
    assert all(abs(x - (W // 2 - 14)) <= 14 for x, _ in discs)  # both on the right border
