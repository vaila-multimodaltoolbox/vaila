"""Kick Measure in getpixelvideo: ``--kick-run`` opens a kick_analysis folder to measure on.

Update Date: 23 November 2026
Version: 0.4.19

The kick folder is written into ``tmp_path`` (a 6-frame video, ``<stem>.dlt3d`` of a known
camera, a ball track, one SAM person): no GPU, no model. The pure helpers are tested directly;
the window runs in a subprocess with ``SDL_VIDEODRIVER=dummy`` like ``test_getpixelvideo_one_line``
(events are posted by a thread, so real mouse use is NOT covered here).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pygame
import pytest

from vaila import getpixelvideo as gpv
from vaila import kick_measure as km

W, H = 320, 240
FPS = 30.0
N_FRAMES = 6
STEM = "maincam"
TS = "20261123_120000"
CENTRE = np.array([0.0, -40.0, 15.0])


def _camera() -> np.ndarray:
    """A fixed camera 40 m behind the pitch, 15 m high, looking at the centre spot."""
    zc = -CENTRE / np.linalg.norm(CENTRE)
    xc = np.cross(zc, [0.0, 0.0, 1.0])
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R = np.array([xc, yc, zc])
    K = np.array([[300.0, 0, W / 2], [0, 300.0, H / 2], [0, 0, 1.0]])
    return K @ np.c_[R, -R @ CENTRE]


FOOT = np.array([5.0, -4.0, 0.0])


@pytest.fixture
def kick_folder(tmp_path: Path) -> dict:
    clip = tmp_path / "clip"
    clip.mkdir()
    video = clip / f"{STEM}.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), FPS, (W, H))
    for i in range(N_FRAMES):
        writer.write(np.full((H, W, 3), 40 + 20 * i, dtype=np.uint8))
    writer.release()
    run = clip / f"kick_analysis_{TS}"
    predict = run / "work" / STEM / f"freekiki_predict_{STEM}_{TS}"
    predict.mkdir(parents=True)
    P = _camera()
    P = P / P[2, 3]  # the FreeKiki DLT3D has P[2, 3] = 1
    rows = [[f, *P[0], *P[1], *P[2, :3]] for f in range(N_FRAMES)]
    pd.DataFrame(rows, columns=pd.Index(["frame"] + [f"L{i}" for i in range(1, 12)])).to_csv(
        predict / f"{STEM}.dlt3d", index=False
    )
    threed = run / f"{STEM}_3d_{TS}"
    threed.mkdir()
    ball = np.array([[-6.0 + 2.0 * f, -3.0, 0.11 + 0.4 * f] for f in range(N_FRAMES)])
    uv = km.project_points(P, ball)
    pd.DataFrame({"frame": range(N_FRAMES), "cx": uv[:, 0], "cy": uv[:, 1]}).to_csv(
        threed / "ball.csv", index=False
    )
    sam = run / "work" / STEM / "sam3dinov3" / "p1" / STEM / "sam3"
    sam.mkdir(parents=True)
    fuv = km.project_points(P, FOOT)[0]
    pd.DataFrame(
        {
            "frame": [0, 1, 0, 1],
            "obj_id": [3, 3, 4, 4],
            "x_px": [fuv[0] - 10, fuv[0] - 10, fuv[0] - 70, fuv[0] - 70],
            "y_px": [fuv[1] - 50] * 4,
            "w_px": [20.0] * 4,
            "h_px": [50.0] * 4,
        }
    ).to_csv(sam / "sam_tracks.csv", index=False)
    # the minimum an export needs: the 3D pose of person 3 (hips, ankles, nose; 0.1 m per frame)
    kpts = {0: "nose", 9: "left-hip", 10: "right-hip", 13: "left-ankle", 14: "right-ankle"}
    rows3d = [
        (f, 3, idx, name, 5.0 + 0.1 * f, -4.0, {0: 1.7, 9: 0.9, 10: 0.9}.get(idx, 0.1))
        for f in range(N_FRAMES)
        for idx, name in kpts.items()
    ]
    pd.DataFrame(
        rows3d, columns=pd.Index(["frame", "person_id", "kpt_idx", "kpt_name", "x_m", "y_m", "z_m"])
    ).to_csv(threed / "poses_3d_on_field.csv", index=False)
    # a 2D pose of person 3: 70 points on a short vertical line over the foot
    pose2d = run / "work" / STEM / "sam3dinov3" / "processed_sam3dinov3_x" / STEM
    pose2d.mkdir(parents=True)
    cols: dict[str, list[float]] = {"frame": list(range(N_FRAMES))}
    for k in range(70):
        cols[f"p{k}_x"] = [float(fuv[0]) + (k % 5) - 2] * N_FRAMES
        cols[f"p{k}_y"] = [float(fuv[1]) - k * 0.7] * N_FRAMES
    pd.DataFrame(cols).to_csv(pose2d / f"{STEM}_id_03_markers.csv", index=False)
    status = {
        "ts": TS,
        "stages": {
            f"detect:{STEM}": {
                "status": "ok",
                "inputs": {"video": {"path": str(video)}},
                "result": {"predict": str(predict)},
            }
        },
        "options": {"videos": [str(video)], "set_piece": "open_play", "analysis": None, "sync": None},
    }  # fmt: skip
    (run / "pipeline_status.json").write_text(json.dumps(status), encoding="utf-8")
    (run / "README.txt").write_text(f"created: x\nfps: {FPS} | kicker id: 1\n", encoding="utf-8")
    return {"run": run, "video": video, "foot_uv": fuv, "ball_uv": uv, "P": P}


# ----------------------------------------------------------------------------------------------
# pure helpers
# ----------------------------------------------------------------------------------------------


def test_the_camera_of_a_video_is_found_from_the_video(kick_folder: dict) -> None:
    camera, note = gpv.kick_run_pick_camera(kick_folder["run"], kick_folder["video"])
    assert (camera, note) == ("reference", "")
    camera, note = gpv.kick_run_pick_camera(kick_folder["run"], "/elsewhere/other_clip.mp4")
    assert camera == "reference" and "reference camera used" in note
    assert gpv.kick_run_video(kick_folder["run"]) == kick_folder["video"]
    with pytest.raises(ValueError, match="not a kick analysis folder"):
        gpv.kick_run_video(kick_folder["video"].parent)


def test_people_are_drawn_with_the_id_and_the_foot_position(kick_folder: dict) -> None:
    run = km.load_kick_measure_run(kick_folder["run"])
    assert run.cameras.counts()["dlt3d"] == N_FRAMES
    person = run.people_at(0)[0]
    assert gpv.kick_person_caption(run, 0, person) == "ID 3 · (5.0, -4.0) m"
    screen = pygame.Surface((W, H))
    pygame.font.init()
    font = pygame.font.Font(None, 14)
    assert gpv.draw_kick_people(screen, run, 0, 1.0, 0, 0, font) == 2
    assert gpv.draw_kick_people(screen, run, 5, 1.0, 0, 0, font) == 0  # nobody in frame 5
    # the box and the foot point are on the pixels of the person
    fx, fy = kick_folder["foot_uv"]
    assert screen.get_at((int(fx), int(fy)))[:3] != (0, 0, 0)


def test_people_drawing_follows_the_panel_filter_selection_and_labels(kick_folder: dict) -> None:
    run = km.load_kick_measure_run(kick_folder["run"])
    pygame.font.init()
    font = pygame.font.Font(None, 14)
    fx, fy = kick_folder["foot_uv"]
    labelled, bare = pygame.Surface((W, H)), pygame.Surface((W, H))
    only3 = lambda pid: pid == 3  # noqa: E731
    assert gpv.draw_kick_people(labelled, run, 0, 1.0, 0, 0, font, visible=only3) == 1
    assert gpv.draw_kick_people(bare, run, 0, 1.0, 0, 0, font, visible=only3, labels=False) == 1
    assert pygame.image.tobytes(labelled, "RGB") != pygame.image.tobytes(bare, "RGB")  # legend
    thin, thick = pygame.Surface((W, H)), pygame.Surface((W, H))
    gpv.draw_kick_people(thin, run, 0, 1.0, 0, 0, font, visible=only3, labels=False)
    gpv.draw_kick_people(thick, run, 0, 1.0, 0, 0, font, visible=only3, selected={3}, labels=False)
    inside = (int(fx) - 10 + 3, int(fy) - 25)  # 3 px inside the left edge: only the thick box
    assert thin.get_at(inside)[:3] == (0, 0, 0) and thick.get_at(inside)[:3] != (0, 0, 0)
    # nobody of person 4's box is drawn when it is filtered out
    assert bare.get_at((int(fx) - 70, int(fy) - 25))[:3] == (0, 0, 0)


def test_a_click_finds_the_person_under_it(kick_folder: dict) -> None:
    run = km.load_kick_measure_run(kick_folder["run"])
    fx, fy = kick_folder["foot_uv"]
    assert gpv.kick_person_hit(run, 0, fx, fy - 25) == 3
    assert gpv.kick_person_hit(run, 0, fx - 60, fy - 25) == 4
    assert gpv.kick_person_hit(run, 0, fx + 100, fy - 25) is None
    assert gpv.kick_person_hit(run, 0, fx, fy - 25, visible=lambda pid: pid != 3) is None
    pose = np.full((70, 2), np.nan)
    pose[:3] = [[fx + 30, fy - 40], [fx + 34, fy - 20], [fx + 32, fy]]
    assert gpv.kick_person_hit(run, 0, fx + 32, fy - 20, boxes=False, poses_at={9: pose}) == 9
    assert gpv.kick_person_hit(run, 0, fx + 32, fy - 20, boxes=False) is None


def test_kick_panel_and_kick_measure_share_the_metric_kinds() -> None:
    from vaila import kick_panel

    assert kick_panel.METRIC_KINDS == km.METRIC_KINDS


# ----------------------------------------------------------------------------------------------
# window (subprocess, dummy video driver)
# ----------------------------------------------------------------------------------------------

_DRIVER = r"""
import json, os, sys, threading, time
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
import pygame
import vaila.getpixelvideo as gpv

video, run, actions = sys.argv[1], sys.argv[2], sys.argv[3].split()
seen = []
if os.environ.get("KICK_PICK"):  # Tpl -> K: the (faked) Tk folder dialog answers with this one
    import tkinter, tkinter.filedialog

    class FakeTk:
        def withdraw(self): ...
        def attributes(self, *a): ...
        def destroy(self): ...

    tkinter.Tk = FakeTk
    tkinter.filedialog.askdirectory = lambda **kw: os.environ["KICK_PICK"]
    sys.platform = "test"  # not linux: the window asks the Tk dialog, not its own browser

class LogFont:
    # logs the text of the HUD / status bar the window draws, to prove what the user would read
    def __init__(self, font):
        self._font = font
    def render(self, text, *args, **kwargs):
        if any(w in text for w in ("Kick run", "Real:", "Frame ", "MODE ", "Height", "Line plane", "Check the calibration", "Frames by calibration", "DLT3D:", "Field lines", "Speed", "Ball ->", "Export", "now ", "agora ", "Ball track")):
            if text not in seen:
                seen.append(text)
                print("TEXT " + text, flush=True)
        return self._font.render(text, *args, **kwargs)
    def __getattr__(self, name):
        return getattr(self._font, name)

_sysfont = pygame.font.SysFont
pygame.font.SysFont = lambda *a, **k: LogFont(_sysfont(*a, **k))

def key(k):
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=k, mod=0, unicode=""))
    time.sleep(0.4)

def click(x, y, button=1):
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(x, y), button=button))
    time.sleep(0.15)
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(x, y), button=button))
    time.sleep(0.4)

from vaila import kick_panel as kpm

PANELS = []
_panel_draw = kpm.KickPanel.draw
def _draw(self, *a, **k):
    _panel_draw(self, *a, **k)
    PANELS[:] = [self]
kpm.KickPanel.draw = _draw

BOXES = {}  # id -> screen centre of its box, as the window draws it
_draw_people = gpv.draw_kick_people
def _people(screen, run, frame, zoom, cx, cy, font, pad_x=0, pad_y=0, **kw):
    for p in run.people_at(frame):
        bx, by, bw, bh = p.bbox
        BOXES[p.id] = (int((bx + bw / 2) * zoom - cx + pad_x), int((by + bh / 2) * zoom - cy + pad_y))
    return _draw_people(screen, run, frame, zoom, cx, cy, font, pad_x, pad_y, **kw)
gpv.draw_kick_people = _people

POINTER = [-1, -1]
pygame.mouse.get_pos = lambda: tuple(POINTER)  # the window reads the hover from here

def hover(x, y):
    POINTER[:] = [x, y]
    time.sleep(0.6)

KEYS = {"k": pygame.K_k, "q": pygame.K_q, "right": pygame.K_RIGHT, "left": pygame.K_LEFT, "esc": pygame.K_ESCAPE,
        "enter": pygame.K_RETURN, "comma": pygame.K_COMMA, "period": pygame.K_PERIOD}

def script():
    while not pygame.display.get_init() or pygame.display.get_surface() is None:
        time.sleep(0.1)
    time.sleep(2.0)
    for a in actions:
        if a.startswith("L"):
            x, y = a[1:].split(",")
            click(int(x), int(y))
        elif a.startswith("H"):
            x, y = a[1:].split(",")
            hover(int(x), int(y))
        elif a.startswith("B"):  # a toolbar button, by its id in TOOLBAR_HIT_RECTS
            x, y = gpv.TOOLBAR_HIT_RECTS[a[1:]].center
            print(f"BUTTON {a[1:]} {x},{y}", flush=True)
            click(x, y)
        elif a.startswith("P"):  # a control of the Kick panel, by its hit_rects key
            x, y = PANELS[0].hit_rects[a[1:]].center
            print(f"PANELCLICK {a[1:]} {x},{y}", flush=True)
            click(x, y)
        elif a == "Zbox3" or a == "Zbox4":  # the box of a person, as drawn
            x, y = BOXES[int(a[4:])]
            click(x, y)
        elif a in ("Zwheel", "Zwheelvideo"):  # the mouse wheel over the panel / over the video
            r = PANELS[0]._rect
            POINTER[:] = [r.centerx, r.centery] if a == "Zwheel" else [60, 60]
            for _ in range(3):
                pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-1))
                time.sleep(0.2)
            time.sleep(0.3)
        elif a == "Zdump":  # what the panel holds now
            pn = PANELS[0]
            print("STATE " + json.dumps({"step": pn.step, "layers": pn.layers, "selected": sorted(pn.selected),
                "metrics": [m.key() for m in pn.metrics], "last_export": pn.last_export, "message": pn.message,
                "scroll": pn.scroll, "keys": sorted(pn.hit_rects), "boxes": BOXES}), flush=True)
        elif a == "wait2":  # let a started program run
            time.sleep(2.0)
        elif a == "wait":  # the window is being reopened on another video
            time.sleep(5.0)
        elif a == "typek":  # the Tpl dialog: replace its value by "k", then Enter
            for _ in range(4):
                key(pygame.K_BACKSPACE)
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=ord("k"), mod=0, unicode="k"))
            time.sleep(0.3)
            key(pygame.K_RETURN)
            time.sleep(0.5)
        elif a.startswith("d"):  # a digit key
            key(getattr(pygame, "K_" + a[1:]))
        else:
            key(KEYS[a])

threading.Thread(target=script, daemon=True).start()
gpv.run_getpixelvideo(
    initial_media_path=video, initial_source_type="video", initial_kick_run=run or None
)
"""


def _drive(
    video: Path,
    run: Path | None,
    actions: str,
    pick: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "KICK_PICK": str(pick)} if pick else dict(os.environ)
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-c", _DRIVER, str(video), str(run or ""), actions],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        cwd=Path(__file__).resolve().parents[1],
    )


def _texts(done: subprocess.CompletedProcess[str]) -> list[str]:
    return [line[5:] for line in done.stdout.splitlines() if line.startswith("TEXT ")]


def _quickmeasure_dir(video: Path) -> Path:
    found = sorted(video.parent.glob("processed_quickmeasure_*"))
    assert found, f"no processed_quickmeasure_* next to {video}"
    return found[-1]


def test_window_kick_run_loads_the_calibration_only_and_shows_the_panel(
    kick_folder: dict,
) -> None:
    video, run = kick_folder["video"], kick_folder["run"]

    done = _drive(video, run, "Zdump esc")  # open, look at the panel, then Esc = save and leave

    assert done.returncode == 0, done.stderr[-2000:]
    out = " ".join(done.stdout.split())  # rich wraps long lines: compare without line breaks
    assert "Kick run loaded" in out and "(reference camera)" in out
    assert f"calibration camera3d installed for {N_FRAMES} frames" in out
    assert f"ball track {N_FRAMES} frames" in out and "2 people ids" in out
    # the status bar the user reads, and the command that repeats the session
    texts = _texts(done)
    status = next(t for t in texts if t.startswith("Kick run (reference): camera3d for 6 frames"))
    assert "pitch lines" in status and "Kick panel" in status
    assert f">> uv run --no-sync vaila/getpixelvideo.py --kick-run {run} -f {video}" in out
    # the panel is there, on step 1, with only the field lines on, and says what to check
    state = _state(done)
    assert state["step"] == 1
    assert state["layers"] == {
        "field": True, "boxes": False, "pose": False, "ball": False, "labels": False,
    }  # fmt: skip
    assert any(t.startswith("Check the calibration") for t in texts), texts
    assert any(t.startswith("DLT3D: 6/6") for t in texts), texts
    # the ball is NOT written into the marker p0 on load (only the panel does it, on request)
    markers = video.with_name(f"{STEM}_markers.csv")
    assert not markers.exists() or "p0_x" not in pd.read_csv(markers).columns
    # the camera3d calibration is installed as the default of the video and is saved with it
    qm = _quickmeasure_dir(video)
    calib = pd.read_csv(next(qm.glob("*_quickmeasure_calibration.csv")))
    assert set(calib["kind"]) == {"camera3d"}
    assert set(calib["frame"]) == {-1}  # the default of the video, not a per-frame override
    assert (qm / f"{STEM}_quickmeasure_camera3d.csv").is_file()
    for kind in ("points", "calibration", "camera3d", "report"):  # every written file is printed
        assert f"{kind}: {qm}" in done.stdout, kind


def _state(done: subprocess.CompletedProcess[str]) -> dict:
    lines = [ln[6:] for ln in done.stdout.splitlines() if ln.startswith("STATE ")]
    assert lines, done.stdout[-1500:]
    return json.loads(lines[-1])


def test_window_ball_to_p0_only_when_the_panel_is_asked(kick_folder: dict) -> None:
    video, run = kick_folder["video"], kick_folder["run"]

    done = _drive(video, run, "Ptab:2 Pball_to_p0 esc")

    assert done.returncode == 0, done.stderr[-2000:]
    markers = pd.read_csv(video.with_name(f"{STEM}_markers.csv"))
    assert list(markers["frame"]) == list(range(N_FRAMES))
    assert markers["p0_x"].to_numpy() == pytest.approx(kick_folder["ball_uv"][:, 0], abs=1.0)
    assert any(t.startswith("Ball ->") for t in _texts(done))


def test_window_panel_tabs_layers_and_a_click_on_a_box_selects_the_person(
    kick_folder: dict,
) -> None:
    video, run = kick_folder["video"], kick_folder["run"]

    # step 2, boxes on (pose on too), then a click on the box of person 3 and of person 4,
    # then person 3 again (unselect); a click in the video elsewhere keeps its normal behaviour
    done = _drive(
        video, run, "Ptab:2 Zdump Player:boxes Player:pose Zbox3 Zdump Zbox4 Zdump Zbox3 Zdump esc"
    )

    assert done.returncode == 0, done.stderr[-2000:]
    states = [json.loads(ln[6:]) for ln in done.stdout.splitlines() if ln.startswith("STATE ")]
    assert len(states) == 4 and all(st["step"] == 2 for st in states)
    assert states[0]["layers"]["boxes"] is False
    assert states[1]["layers"]["boxes"] and states[1]["layers"]["pose"]
    assert [st["selected"] for st in states[1:]] == [[3], [3, 4], [4]]


def test_window_measure_by_hand_turns_measure_on_and_the_height_answers_in_metres(
    kick_folder: dict,
) -> None:
    video, run = kick_folder["video"], kick_folder["run"]

    # step 3: "Measure by hand" height base+top = mode 7; then base on the grass and the top
    done = _drive(video, run, "Ptab:3 Pmeasure:7 L160,150 L160,100 esc")

    assert done.returncode == 0, done.stderr[-2000:]
    texts = _texts(done)
    assert any(t.startswith("MODE height") for t in texts), texts
    results = pd.read_csv(next(_quickmeasure_dir(video).glob("*_quickmeasure_results.csv")))
    height = results[results["type"] == "height"]["value"].iloc[0]
    assert 0.0 < height < 10.0  # metres


def test_window_metric_shows_value_on_video_and_panel(kick_folder: dict) -> None:
    video, run = kick_folder["video"], kick_folder["run"]

    done = _drive(
        video,
        run,
        "Ptab:2 Pperson:3 Ptab:3 Pkind:speed Padd_metric Zdump esc",
    )

    assert done.returncode == 0, done.stderr[-2000:]
    state = _state(done)
    assert state["metrics"] == ["speed:3"]
    assert any(t.startswith("now ") and "m/s" in t and "max" in t for t in _texts(done)), _texts(
        done
    )


def test_window_export_writes_the_scene_and_prints_the_cli_mirror(kick_folder: dict) -> None:
    video, run = kick_folder["video"], kick_folder["run"]
    fakes = run.parent / "fakes"
    fakes.mkdir()
    mk = fakes / "mkvis3d.py"
    mk.write_text(
        "import pathlib, sys\npathlib.Path(sys.argv[0]).with_name('mkvis3d_args.txt')"
        ".write_text(' '.join(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    blender = fakes / "blender"
    blender.write_text(
        '#!/bin/sh\nif [ "$1" = "--version" ]; then echo \'Blender 4.2.0\'; exit 0; fi\n'
        f'echo "$@" > {fakes}/blender_args.txt\n',
        encoding="utf-8",
    )
    blender.chmod(0o755)
    env = {"VAILA_MKVIS3D": str(mk), "VAILA_BLENDER": str(blender)}

    done = _drive(
        video,
        run,
        "Ptab:2 Pperson:3 Ptab:3 Pkind:speed Padd_metric Ptab:4 Pexport:all "
        "Zdump Popen:mkvis3d Popen:blender wait2 esc",
        extra_env=env,
    )

    assert done.returncode == 0, done.stderr[-2000:]
    exports = sorted(run.glob("processed_kickmeasure_*"))
    assert len(exports) == 1, exports
    out = exports[0]
    for name in (
        "kick_scene.c3d",
        "kick_scene_c3d.csv",
        "metrics.csv",
        "metrics_summary.json",
        "README.txt",
        "blender/person_3.bvh",
        "blender/ball_path.json",
        "blender/kick_scene_blender.py",
    ):
        assert (out / name).is_file(), name
    assert _state(done)["last_export"] == str(out)
    flat = " ".join(done.stdout.split())
    mirror = (
        f">> uv run --no-sync python -m vaila.kick_measure --run {run} --export-scene {out} "
        "--people 3 --metric speed:3"
    )
    assert mirror in flat, flat[-1500:]
    # the mirror reproduces the same files
    again = subprocess.run(
        [sys.executable, "-m", "vaila.kick_measure", "--run", str(run), "--export-scene",
         str(run.parent / "again"), "--people", "3", "--metric", "speed:3"],
        capture_output=True, text=True, timeout=120, cwd=Path(__file__).resolve().parents[1],
    )  # fmt: skip
    assert again.returncode == 0, again.stderr
    mine = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    theirs = sorted(
        p.relative_to(run.parent / "again").as_posix()
        for p in (run.parent / "again").rglob("*")
        if p.is_file()
    )
    assert mine == theirs
    # Open mkvis3d / Open Blender start the (fake) programs without blocking
    assert (fakes / "mkvis3d_args.txt").read_text().strip().endswith("kick_scene.c3d")
    assert "kick_scene_blender.py" in (fakes / "blender_args.txt").read_text()


def test_window_wheel_over_the_panel_does_not_zoom_but_over_the_video_it_does(
    kick_folder: dict,
) -> None:
    video, run = kick_folder["video"], kick_folder["run"]

    done = _drive(video, run, "Ptab:2 Player:boxes Zdump Zwheel Zdump Zwheelvideo Zdump esc")

    assert done.returncode == 0, done.stderr[-2000:]
    states = [json.loads(ln[6:]) for ln in done.stdout.splitlines() if ln.startswith("STATE ")]
    assert len(states) == 3
    assert states[0]["boxes"]["3"] == states[1]["boxes"]["3"]  # wheel over the panel: no zoom
    assert states[2]["boxes"]["3"] != states[1]["boxes"]["3"]  # wheel over the video: zoom
