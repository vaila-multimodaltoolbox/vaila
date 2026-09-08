"""Guided Pynalty workflow and reusable calibration regression tests.

Update Date: 08 September 2026
Version: 0.3.130
"""

import copy
import os
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import cv2
import numpy as np
import pygame
import pytest

from vaila import pynalty


@pytest.fixture
def app(tmp_path):
    session = pynalty.PynaltyApp(show_wizard=False)
    session.video_path = str(tmp_path / "clip.mp4")
    session.width, session.height, session.total_frames = 640, 480, 100
    return session


def press(app, key):
    return app._handle_keydown(SimpleNamespace(key=key))


def calibrate(app):
    app.seek(5)
    press(app, pygame.K_RETURN)
    for point in ((100, 400), (100, 100), (550, 100), (550, 400)):
        app._mark_point(*point)
    assert app.phase == 0  # Four clicks still require explicit confirmation.
    press(app, pygame.K_RETURN)
    assert app.phase == 1


def mark_frames(app):
    for phase, frame in ((1, 10), (2, 20), (3, 40)):
        assert app.phase == phase
        app.seek(frame)
        app.playing = True
        app._mark_point(150, 200)
        assert not app.current_event.points
        assert app.current_event.frame_idx == -1
        press(app, pygame.K_RETURN)
        assert not app.playing
    assert app.phase == 4
    assert app.current_frame_idx == 20


def mark_points(app):
    for phase, frame in ((4, 20), (5, 40)):
        assert app.phase == phase
        assert app.current_frame_idx == frame
        app.seek(90)
        press(app, pygame.K_SPACE)
        assert app.current_frame_idx == frame and not app.playing
        app._mark_point(300, 250)
        app._mark_point(350, 300)
    assert app.phase == 6
    app._on_button("goal")
    assert app.phase == 7


def test_full_guided_flow_and_review(app):
    calibrate(app)
    mark_frames(app)
    mark_points(app)
    for phase in (8, 9, 10):
        app._on_button("next")
        assert app.phase == phase
    app._on_button("next")
    assert app.phase == 10
    assert app.compute_metrics()
    assert app.step("gk_move").frame_idx < app.step("kick").frame_idx


def test_calibration_pause_bounds_undo(app):
    app._mark_point(100, 100)
    assert not app.calibration_draft
    app.seek(8)
    app.playing = True
    press(app, pygame.K_RETURN)
    app.seek(20)
    assert app.current_frame_idx == 8 and not app.playing
    for point in ((-1, 0), (640, 0), (0, 480), (float("nan"), 1)):
        app._mark_point(*point)
    assert not app.calibration_draft
    app._mark_point(100, 200)
    app._undo_point()
    assert not app.calibration_draft


def test_arrival_order_and_edit_invalidates(app):
    calibrate(app)
    for frame in (10, 20):
        app.seek(frame)
        app.confirm_frame()
    app.seek(20)
    app.confirm_frame()
    assert app.phase == 3 and app.step("goal").frame_idx == -1
    app.seek(40)
    app.confirm_frame()
    mark_points(app)
    app.set_phase(4)
    app._on_button("edit")
    assert app.phase == 2
    app.step("ball_path").points = {"old": [1]}
    app.pose_metrics = {"old": 1}
    app.seek(45)
    app.confirm_frame()
    assert not app.step("kick").points
    assert app.step("goal").frame_idx == -1
    assert not app.step("ball_path").points
    assert not app.last_results and not app.pose_metrics
    assert app.shot_outcome is None


def test_recalibrate_cancel_and_replace(app):
    calibrate(app)
    mark_frames(app)
    mark_points(app)
    old = copy.deepcopy(app.to_data())
    app.redo_calibration()
    app.confirm_calibration()
    app._mark_point(101, 400)
    app.cancel_calibration()
    assert app.phase == 7
    assert app.to_data()["events"] == old["events"]
    app.redo_calibration()
    app.confirm_calibration()
    for point in ((110, 400), (110, 100), (560, 100), (560, 400)):
        app._mark_point(*point)
    app.confirm_calibration()
    assert app.phase == 7
    assert app.calib_points()[0] == [110, 400]
    assert app.step("kick").frame_idx == 20
    assert app.step("kick").points["Ball"] == [300, 250]
    assert app.compute_metrics()


def test_calibration_reuse_and_provenance(app):
    calibrate(app)
    path = app.calibration_path
    fresh = pynalty.PynaltyApp()
    fresh.video_path = app.video_path
    fresh.width, fresh.height, fresh.total_frames = 640, 480, 100
    assert fresh.prepare_calibration()
    assert fresh.phase == 0 and fresh.calibration_stage == "preview"
    assert not fresh.calib_points()
    fresh.confirm_calibration()
    assert fresh.calib_points() == app.calib_points()
    assert fresh.step("calibration").frame_idx == -1
    assert fresh.step("gk_move").frame_idx == -1
    assert fresh.calibration_record["source_frame"] == 5
    assert fresh.calibration_path == path


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(width=999),
        lambda d: d.update(format_version=9),
        lambda d: d.update(points=[[0, 0]] * 4),
        lambda d: d["points"][0].__setitem__(0, float("nan")),
        lambda d: d["points"][0].__setitem__(0, 640),
        lambda d: d.update(points=[[10, 10], [100, 100], [10, 100], [100, 10]]),
        lambda d: d["geometry"].update(width=0),
    ],
)
def test_invalid_calibration(app, mutation):
    calibrate(app)
    data = copy.deepcopy(app.calibration_record)
    mutation(data)
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        app.validate_calibration(data)


def test_bad_file_starts_fresh(app, tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("[broken")
    app.explicit_calibration_path = str(path)
    assert not app.prepare_calibration()
    assert app.phase == 0 and app.calibration_stage == "frame"
    assert app.feedback_msg


def test_write_failure_preserves_file_and_session(app, monkeypatch):
    calibrate(app)
    path = pynalty.Path(app.calibration_path)
    old = path.read_bytes()

    def fail(*args):
        raise PermissionError("read only")

    monkeypatch.setattr(pynalty.os, "replace", fail)
    assert not app.persist_calibration()
    assert path.read_bytes() == old
    assert app.calib_points()


def test_write_failure_offers_destination(app, monkeypatch, tmp_path):
    root = SimpleNamespace(withdraw=lambda: None, destroy=lambda: None)
    monkeypatch.setattr(pynalty.tk, "Tk", lambda: root)
    destination = str(tmp_path / "other.toml")
    monkeypatch.setattr(pynalty.filedialog, "asksaveasfilename", lambda **kwargs: destination)
    writes = []

    def save(path=None):
        writes.append(path)
        return path is not None

    monkeypatch.setattr(app, "persist_calibration", save)
    calibrate(app)
    assert writes == [None, destination]
    assert app.calib_points()


@pytest.mark.parametrize("keyed", [True, False])
def test_session_resume_and_precedence(app, keyed):
    calibrate(app)
    mark_frames(app)
    app._mark_point(300, 250)
    data = app.to_data()
    if not keyed:
        for event in data["events"]:
            event.pop("key")
    if keyed:
        data["events"].reverse()
    other = pynalty.PynaltyApp()
    other.video_path = app.video_path
    other.width, other.height, other.total_frames = 640, 480, 100
    other.load_from_data(data)
    assert other.phase == 4 and other.current_frame_idx == 20
    assert other.step("kick").points == {"Ball": [300, 250]}
    assert other.prepare_calibration()  # Saved calibration overrides discovery.
    assert other.phase == 4
    other.explicit_calibration_path = app.calibration_path
    assert other.prepare_calibration()
    assert other.phase == 0 and other.calibration_stage == "preview"


def test_legacy_flat_and_load_replaces_old_marks(app):
    calibrate(app)
    app.step("goal").points = {"Ball": [20, 30]}
    app.load_from_data(
        {"calibration_pixels": app.calib_points(), "gk_move_frame": 5, "kick_frame": 10}
    )
    assert app.phase == 3
    assert not app.step("goal").points


def test_report_only_never_discovers_or_opens_dialog(app, monkeypatch):
    calibrate(app)
    app.step("calibration").reset()
    app.explicit_calibration_path = None
    assert app.prepare_calibration(report_only=True)
    assert not app.calib_points()
    monkeypatch.setattr(pynalty, "load_video_file_dialog", lambda *a: pytest.fail("dialog opened"))
    assert pynalty.main(["--report-only"]) == 1


def test_cli_language_and_calibration_mirror(capsys):
    args = pynalty.build_parser().parse_args(
        ["--calibration", "/tmp/my goal.toml", "--ui-lang", "en", "--lang", "pt"]
    )
    pynalty._mirror_cli(args, "clip.mp4")
    out = capsys.readouterr().out
    assert "--ui-lang en" in out and "--lang pt" in out and "--calibration" in out
    assert args.calibration == "/tmp/my goal.toml"


def test_render_short_video_and_report(tmp_path, monkeypatch):
    video = tmp_path / "short.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (640, 480))
    assert writer.isOpened()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (550, 400), (230, 230, 230), 4)
    for _ in range(100):
        writer.write(frame)
    writer.release()
    app = pynalty.PynaltyApp(str(video), show_wizard=False)
    app._init_pygame()
    try:
        calibrate(app)
        mark_frames(app)
        mark_points(app)
        marks = copy.deepcopy(app.to_data()["events"])
        for size in ((1280, 800), (760, 600)):
            app.screen = pygame.display.set_mode(size)
            app.fit_view()
            for phase in range(11):
                app.set_phase(phase)
                for language in ("pt", "en"):
                    if app.ui_lang != language:
                        app._on_button("lang")
                    app.draw_content()
                    assert all(app.screen.get_rect().contains(b.rect) for b in app.buttons)
            assert app.to_data()["events"] == marks
        app.set_phase(4)
        app._undo_point()
        app._on_button("lang")
        app.screen = pygame.display.set_mode((900, 650))
        app.fit_view()
        app.draw_content()
        pygame.image.save(app.screen, "/tmp/pynalty-guided-contact.png")
        # Click in letterboxing, inside content panel but outside actual video.
        before = copy.deepcopy(app.step("kick").points)
        app._handle_mousedown(SimpleNamespace(pos=(app.content_rect().x + 1, 1), button=1))
        assert app.step("kick").points == before
        app._mark_point(350, 300)
        app._on_button("next")
        app._set_shot_outcome("goal")
        app.set_phase(10)
        assert app.save_results_package()
        assert app.phase == 10
        config = pynalty.Path(app.get_results_dir()) / "data.toml"
        assert config.exists()
        monkeypatch.setattr(
            pynalty, "load_video_file_dialog", lambda *args: pytest.fail("headless dialog")
        )
        assert pynalty.main(["-i", str(video), "-c", str(config), "--report-only"]) == 0
    finally:
        pygame.quit()
        app.cap.release()


def test_explicit_calibration_headless_overrides_session(app, tmp_path):
    calibrate(app)
    record = copy.deepcopy(app.calibration_record)
    record["geometry"]["width"] = 6.0
    record["source_frame"] = 99
    path = tmp_path / "explicit.toml"
    path.write_text(pynalty.toml.dumps(record))
    app.explicit_calibration_path = str(path)
    assert app.prepare_calibration(report_only=True)
    assert app.goal.width == 6.0
    assert app.step("calibration").frame_idx == -1
    assert app.step("gk_move").frame_idx == -1


def test_edited_reuse_is_saved_as_new_calibration(app):
    calibrate(app)
    app.step("calibration").reset()
    assert app.prepare_calibration()
    app._undo_point()
    app._mark_point(560, 400)
    app.confirm_calibration()
    saved = pynalty.toml.load(app.calibration_path)
    assert saved["points"][-1] == [560, 400]
    assert app.step("calibration").frame_idx == app.current_frame_idx


def test_cancel_load_preserves_phase(app, monkeypatch):
    calibrate(app)
    mark_frames(app)
    before = copy.deepcopy(app.to_data())
    monkeypatch.setattr(app, "load_toml", lambda: None)
    monkeypatch.setattr(
        app, "prepare_calibration", lambda: pytest.fail("cancelled load changed calibration")
    )
    app._on_button("load")
    assert app.to_data() == before


def _write_short_video(path, fps=25, frames=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    assert writer.isOpened()
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    for _ in range(frames):
        writer.write(frame)
    writer.release()


def test_load_video_prefers_precise_fps_over_raw_cv2_reading(tmp_path, monkeypatch):
    video = tmp_path / "clip.avi"
    _write_short_video(video, fps=25)
    # ffprobe/numberframes disagrees with the (often-wrong) raw cv2 reading; the
    # precise value must win, same rationale as numberframes.py's own fallback order.
    monkeypatch.setattr(pynalty, "_detect_fps", lambda path: 59.94)
    session = pynalty.PynaltyApp(str(video), show_wizard=False)
    assert session.fps == 59.94


def test_load_video_falls_back_to_cv2_fps_when_precise_detection_fails(tmp_path, monkeypatch):
    video = tmp_path / "clip.avi"
    _write_short_video(video, fps=25)
    monkeypatch.setattr(pynalty, "_detect_fps", lambda path: None)
    session = pynalty.PynaltyApp(str(video), show_wizard=False)
    assert session.fps == pytest.approx(25.0, abs=0.5)


def test_confirm_fps_on_start_prefills_detected_value_and_applies_correction(app, monkeypatch):
    app.fps = 29.97
    seen = {}

    def fake_text_input(prompt, initial):
        seen["initial"] = initial
        return "50"

    monkeypatch.setattr(app, "_text_input", fake_text_input)
    app.confirm_fps_on_start()
    assert seen["initial"] == "29.970"
    assert app.fps == 50.0


def test_confirm_fps_on_start_plain_enter_keeps_detected_value(app, monkeypatch):
    app.fps = 29.97
    monkeypatch.setattr(app, "_text_input", lambda prompt, initial: initial)
    app.confirm_fps_on_start()
    assert app.fps == pytest.approx(29.97)


def test_confirm_fps_on_start_invalid_text_keeps_previous_fps(app, monkeypatch):
    app.fps = 29.97
    monkeypatch.setattr(app, "_text_input", lambda prompt, initial: "not-a-number")
    app.confirm_fps_on_start()
    assert app.fps == pytest.approx(29.97)


def test_confirm_fps_on_start_cancelled_keeps_previous_fps(app, monkeypatch):
    app.fps = 29.97
    monkeypatch.setattr(app, "_text_input", lambda prompt, initial: None)
    app.confirm_fps_on_start()
    assert app.fps == pytest.approx(29.97)


def test_run_confirms_fps_before_entering_marking_loop(tmp_path, monkeypatch):
    video = tmp_path / "clip.avi"
    _write_short_video(video, fps=25)
    session = pynalty.PynaltyApp(str(video), show_wizard=False)
    calls = []
    monkeypatch.setattr(session, "confirm_fps_on_start", lambda: calls.append(True))
    monkeypatch.setattr(pygame.event, "get", lambda: [SimpleNamespace(type=pygame.QUIT)])
    session.run()
    assert calls == [True]


class _SequentialCapture:
    def __init__(self, position=1):
        self.position = position
        self.set_calls = []

    def get(self, prop):
        assert prop == cv2.CAP_PROP_POS_FRAMES
        return self.position

    def set(self, prop, value):
        assert prop == cv2.CAP_PROP_POS_FRAMES
        self.position = int(value)
        self.set_calls.append(int(value))
        return True

    def read(self):
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        self.position += 1
        return True, frame


def test_playback_reads_sequentially_without_seek(app):
    app.cap = _SequentialCapture(position=1)
    app.current_frame_idx = 0
    app.playing = True

    assert app.advance_playback()

    assert app.current_frame_idx == 1
    assert app.cap.set_calls == []
    assert app.frame_img.get_size() == (3, 2)


def test_playback_repairs_capture_position_only_when_needed(app):
    app.cap = _SequentialCapture(position=40)
    app.current_frame_idx = 7
    app.playing = True

    assert app.advance_playback()

    assert app.current_frame_idx == 8
    assert app.cap.set_calls == [8]


def test_timeline_drag_pauses_before_seeking(app, monkeypatch):
    app.screen = SimpleNamespace(get_size=lambda: (900, 650))
    app.playing = True
    sought = []
    monkeypatch.setattr(app, "_slider_seek", sought.append)

    app._handle_mousedown(SimpleNamespace(pos=(450, 650 - pynalty.BOTTOM_H + 20), button=1))

    assert not app.playing
    assert app.start_drag_slider
    assert sought == [450]
