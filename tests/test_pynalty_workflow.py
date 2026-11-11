"""Guided Pynalty workflow and reusable calibration regression tests.

Update Date: 08 November 2026
Version: 0.4.15
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


def mark_kick(app, frame=20):
    """Macro 1: contact frame + Ball + Keeper."""
    assert app.phase == pynalty.PHASE_KICK_FRAME
    app.seek(frame)
    press(app, pygame.K_RETURN)
    assert app.phase == pynalty.PHASE_KICK_POINTS
    assert app.current_frame_idx == frame
    app._mark_point(300, 250)
    app._mark_point(350, 300)
    # Auto-advances to goal frame (defaults to last frame).
    assert app.phase == pynalty.PHASE_GOAL_FRAME
    assert app.step("kick").points == {"Ball": [300, 250], "GK": [350, 300]}


def calibrate_on_arrival(app, frame=None):
    """Macro 2 calib: confirm arrival frame + 4 corners."""
    assert app.phase == pynalty.PHASE_GOAL_FRAME
    target = max(0, app.total_frames - 1) if frame is None else frame
    app.seek(target)
    press(app, pygame.K_RETURN)
    assert app.is_calibrating
    assert app.calibration_stage == "corners"
    for point in ((100, 400), (100, 100), (550, 100), (550, 400)):
        app._mark_point(*point)
    assert app.calibration_stage == "preview"
    press(app, pygame.K_RETURN)
    assert not app.is_calibrating
    assert app.phase == pynalty.PHASE_GOAL_POINTS
    assert len(app.calib_points()) == 4


def mark_arrival_points_and_outcome(app):
    """Macro 2 points + outcome on the arrival frame."""
    assert app.phase == pynalty.PHASE_GOAL_POINTS
    app._mark_point(400, 280)
    app._mark_point(350, 290)
    assert app.phase == pynalty.PHASE_OUTCOME
    app._on_button("goal")
    assert app.shot_outcome == "goal"


def mark_response(app, frame=10):
    """Macro 3: keeper first-movement frame."""
    assert app.phase == pynalty.PHASE_RESPONSE
    app.seek(frame)
    press(app, pygame.K_RETURN)
    assert app.step("gk_move").frame_idx == frame
    assert app.phase == pynalty.PHASE_REVIEW
    assert app.core_steps_done()


def full_core_flow(app):
    mark_kick(app)
    calibrate_on_arrival(app)
    mark_arrival_points_and_outcome(app)
    mark_response(app)


def test_full_guided_flow_and_review(app):
    full_core_flow(app)
    assert app.phase == pynalty.PHASE_REVIEW
    assert app.compute_metrics()
    assert app.step("gk_move").frame_idx < app.step("kick").frame_idx
    # Optional AI phases are never forced after outcome.
    assert app.phase != pynalty.PHASE_BALL_PATH


def test_calibration_pause_bounds_undo(app):
    mark_kick(app)
    app.seek(90)
    app.playing = True
    press(app, pygame.K_RETURN)
    assert app.is_calibrating and app.calibration_stage == "corners"
    assert app.current_frame_idx == 90 and not app.playing
    for point in ((-1, 0), (640, 0), (0, 480), (float("nan"), 1)):
        app._mark_point(*point)
    assert not app.calibration_draft
    app._mark_point(100, 200)
    app._undo_point()
    assert not app.calibration_draft


def test_arrival_order_and_edit_invalidates(app):
    mark_kick(app, frame=20)
    # Arrival must be after kick.
    app.seek(20)
    press(app, pygame.K_RETURN)
    assert app.phase == pynalty.PHASE_GOAL_FRAME
    assert app.step("goal").frame_idx == -1
    calibrate_on_arrival(app, frame=40)
    mark_arrival_points_and_outcome(app)
    app.set_phase(pynalty.PHASE_KICK_POINTS)
    app._on_button("edit")
    assert app.phase == pynalty.PHASE_KICK_FRAME
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
    full_core_flow(app)
    old = copy.deepcopy(app.to_data())
    app.redo_calibration()
    assert app.is_calibrating
    app.cancel_calibration()
    assert app.phase == pynalty.PHASE_REVIEW
    assert app.to_data()["events"] == old["events"]
    app.redo_calibration()
    for point in ((110, 400), (110, 100), (560, 100), (560, 400)):
        app._mark_point(*point)
    app.confirm_calibration()
    assert app.calib_points()[0] == [110, 400]
    assert app.step("kick").frame_idx == 20
    assert app.step("kick").points["Ball"] == [300, 250]
    assert app.compute_metrics()


def test_calibration_reuse_and_provenance(app):
    mark_kick(app, frame=20)
    calibrate_on_arrival(app, frame=40)
    path = app.calibration_path
    source_frame = app.calibration_record["source_frame"]
    fresh = pynalty.PynaltyApp()
    fresh.video_path = app.video_path
    fresh.width, fresh.height, fresh.total_frames = 640, 480, 100
    assert fresh.prepare_calibration()
    assert fresh.pending_calibration is not None
    assert not fresh.calib_points()
    # Kick first, then accept pending calib on arrival.
    mark_kick(fresh)
    press(fresh, pygame.K_RETURN)
    assert fresh.calibration_stage == "preview"
    fresh.confirm_calibration()
    assert fresh.calib_points() == app.calib_points()
    assert fresh.step("calibration").frame_idx == -1
    assert fresh.step("gk_move").frame_idx == -1
    assert fresh.calibration_record["source_frame"] == source_frame
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
    mark_kick(app)
    calibrate_on_arrival(app)
    data = copy.deepcopy(app.calibration_record)
    mutation(data)
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        app.validate_calibration(data)


def test_bad_file_starts_fresh(app, tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("[broken")
    app.explicit_calibration_path = str(path)
    assert not app.prepare_calibration()
    assert app.calibration_stage == "idle"
    assert app.feedback_msg


def test_write_failure_preserves_file_and_session(app, monkeypatch):
    mark_kick(app)
    calibrate_on_arrival(app)
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
    mark_kick(app)
    calibrate_on_arrival(app)
    assert None in writes and destination in writes
    assert app.calib_points()


@pytest.mark.parametrize("keyed", [True, False])
def test_session_resume_and_precedence(app, keyed):
    mark_kick(app)
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
    assert other.phase == pynalty.PHASE_GOAL_FRAME
    assert other.step("kick").points == {"Ball": [300, 250], "GK": [350, 300]}
    calibrate_on_arrival(app)
    assert other.prepare_calibration()  # Saved calibration overrides discovery.
    assert other.phase == pynalty.PHASE_GOAL_FRAME
    other.explicit_calibration_path = app.calibration_path
    assert other.prepare_calibration()
    assert other.pending_calibration is not None


def test_legacy_flat_and_load_replaces_old_marks(app):
    mark_kick(app)
    calibrate_on_arrival(app)
    app.step("goal").points = {"Ball": [20, 30]}
    app.load_from_data(
        {"calibration_pixels": app.calib_points(), "gk_move_frame": 5, "kick_frame": 10}
    )
    # Kick frame loaded without points → resume at kick points.
    assert app.phase == pynalty.PHASE_KICK_POINTS
    assert not app.step("goal").points


def test_report_only_never_discovers_or_opens_dialog(app, monkeypatch):
    mark_kick(app)
    calibrate_on_arrival(app)
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
        full_core_flow(app)
        marks = copy.deepcopy(app.to_data()["events"])
        for size in ((1280, 800), (760, 600)):
            app.screen = pygame.display.set_mode(size)
            app.fit_view()
            for phase in range(len(app.PHASE_KEYS)):
                app.set_phase(phase)
                for language in ("pt", "en"):
                    if app.ui_lang != language:
                        app._on_button("lang")
                    app.draw_content()
                    assert all(app.screen.get_rect().contains(b.rect) for b in app.buttons)
            assert app.to_data()["events"] == marks
        app._on_button("lang")
        app.screen = pygame.display.set_mode((900, 650))
        app.fit_view()
        app.set_phase(pynalty.PHASE_KICK_POINTS)
        app.draw_content()
        pygame.image.save(app.screen, "/tmp/pynalty-guided-contact.png")
        before = copy.deepcopy(app.step("kick").points)
        app._handle_mousedown(SimpleNamespace(pos=(app.content_rect().x + 1, 1), button=1))
        assert app.step("kick").points == before
        assert app.core_steps_done()
        app.set_phase(pynalty.PHASE_REVIEW)
        assert app.save_results_package()
        assert app.phase == pynalty.PHASE_REVIEW
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
    mark_kick(app)
    calibrate_on_arrival(app)
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
    mark_kick(app)
    calibrate_on_arrival(app)
    app.step("calibration").reset()
    assert app.prepare_calibration()
    mark_kick(app) if app.phase == pynalty.PHASE_KICK_FRAME else None
    # Resume may leave us mid-goal; force preview accept with an edit.
    app.set_phase(pynalty.PHASE_GOAL_FRAME)
    app.seek(99)
    press(app, pygame.K_RETURN)
    if app.calibration_stage == "preview":
        app._undo_point()
        app._mark_point(560, 400)
        app.confirm_calibration()
    else:
        for point in ((100, 400), (100, 100), (550, 100), (560, 400)):
            app._mark_point(*point)
        app.confirm_calibration()
    saved = pynalty.toml.load(app.calibration_path)
    assert saved["points"][-1] == [560, 400]
    assert app.step("calibration").frame_idx == app.current_frame_idx


def test_cancel_load_preserves_phase(app, monkeypatch):
    mark_kick(app)
    before = copy.deepcopy(app.to_data())
    before_phase = app.phase
    monkeypatch.setattr(app, "load_toml", lambda: None)
    monkeypatch.setattr(
        app, "prepare_calibration", lambda: pytest.fail("cancelled load changed calibration")
    )
    app._on_button("load")
    assert app.to_data() == before
    assert app.phase == before_phase


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
    monkeypatch.setattr(pynalty, "_detect_fps", lambda path: 59.94)
    session = pynalty.PynaltyApp(str(video), show_wizard=False)
    assert session.fps == 59.94


def test_load_video_falls_back_to_cv2_fps_when_precise_detection_fails(tmp_path, monkeypatch):
    video = tmp_path / "clip.avi"
    _write_short_video(video, fps=25)
    monkeypatch.setattr(pynalty, "_detect_fps", lambda path: None)
    session = pynalty.PynaltyApp(str(video), show_wizard=False)
    assert session.fps == pytest.approx(25.0, abs=0.5)


def test_confirm_fps_on_start_auto_accepts_detected_value(app):
    app.fps = 29.97
    app.confirm_fps_on_start()
    assert app.fps == pytest.approx(29.97)
    assert app.fps_confirmed
    assert "29.970" in app.feedback_msg or "29.97" in app.feedback_msg


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


def test_macro_steps_navigation_and_completion(app):
    assert app.current_macro_step() == 0
    assert not app.is_macro_step_done(0)

    mark_kick(app)
    assert app.is_macro_step_done(0)
    assert app.current_macro_step() == 1

    calibrate_on_arrival(app)
    mark_arrival_points_and_outcome(app)
    assert app.is_macro_step_done(1)
    assert app.current_macro_step() == 2

    # Jump back to kick without wiping marks
    app.goto_macro_step(0)
    assert app.phase in (pynalty.PHASE_KICK_FRAME, pynalty.PHASE_KICK_POINTS)
    assert app.step("kick").points

    app.goto_macro_step(2)
    assert app.phase == pynalty.PHASE_RESPONSE
    mark_response(app)
    assert app.is_macro_step_done(2)
    assert app.is_macro_step_done(3)
    assert app.current_macro_step() == 3


def test_quick_save_direct_from_outcome(app, tmp_path, monkeypatch):
    out_dir = str(tmp_path / "pynalty_out")
    app.output_dir_override = out_dir

    saved = []
    monkeypatch.setattr(app, "save_results_package", lambda: saved.append(True) or True)

    full_core_flow(app)
    assert app.phase == pynalty.PHASE_REVIEW
    assert app.core_steps_done()
    assert app.shot_outcome == "goal"

    press(app, pygame.K_s)
    assert saved == [True]
    assert app.phase == pynalty.PHASE_REVIEW

    app.set_phase(pynalty.PHASE_OUTCOME)
    app._on_button("save")
    assert len(saved) == 2
    assert app.phase == pynalty.PHASE_REVIEW


def test_outcome_does_not_enter_optional_ai(app):
    mark_kick(app)
    calibrate_on_arrival(app)
    mark_arrival_points_and_outcome(app)
    assert app.phase == pynalty.PHASE_RESPONSE
    mark_response(app)
    assert app.phase == pynalty.PHASE_REVIEW


def test_fast_keys_1_2_3_navigation(app):
    mark_kick(app, frame=30)
    assert app.phase == pynalty.PHASE_GOAL_FRAME

    # Key 1 jumps back to kick
    press(app, pygame.K_1)
    assert app.phase in (pynalty.PHASE_KICK_FRAME, pynalty.PHASE_KICK_POINTS)
    assert app.step("kick").frame_idx == 30

    # Key 2 jumps to goal
    press(app, pygame.K_2)
    assert app.phase == pynalty.PHASE_GOAL_FRAME

    calibrate_on_arrival(app)
    mark_arrival_points_and_outcome(app)
    assert app.phase == pynalty.PHASE_RESPONSE

    # From Save, key 3 jumps back to response
    app.set_phase(pynalty.PHASE_REVIEW)
    press(app, pygame.K_3)
    assert app.phase == pynalty.PHASE_RESPONSE


def test_timeline_snap_to_event_ticks(app, monkeypatch):
    app.screen = SimpleNamespace(get_size=lambda: (900, 650))
    app.total_frames = 101  # frames 0..100
    app.step("kick").frame_idx = 50

    margin = 20
    timeline_w = 900 - margin * 2
    kick_x = margin + int(50 / 100 * timeline_w)

    sought = []
    monkeypatch.setattr(app, "seek", sought.append)

    # Click within 5px of kick marker
    app._handle_mousedown(SimpleNamespace(pos=(kick_x + 3, 650 - pynalty.BOTTOM_H + 20), button=1))
    assert sought == [50]


def test_step_panel_macro_rects_and_quick_save_button(app):
    pygame.init()
    surface = pygame.Surface((1280, 800))
    app.screen = surface
    app.font = pygame.font.SysFont("Arial", 15)
    app.font_big = pygame.font.SysFont("Arial", 22)
    app.font_small = pygame.font.SysFont("Arial", 12)

    # Before completion: 4 macro rects, no quick save button
    app.draw_step_panel()
    assert len(app.macro_step_rects) == 4
    assert app.quick_save_rect is None
    assert len(app.optional_tool_rects) == 3

    full_core_flow(app)

    app.draw_step_panel()
    assert app.core_steps_done()
    assert app.quick_save_rect is not None
    assert app.quick_save_rect.width > 0


def test_action_banner_drawn(app):
    pygame.init()
    app.screen = pygame.display.set_mode((1280, 800))
    app.font = pygame.font.SysFont("Arial", 15)
    app.font_big = pygame.font.SysFont("Arial", 22)
    app.font_small = pygame.font.SysFont("Arial", 12)
    text = app.action_banner_text()
    lowered = text.lower()
    assert any(word in lowered for word in ("contato", "contact", "chute", "kick", "pé", "foot"))
    app.draw_content()
    pygame.display.quit()
