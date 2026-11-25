"""Per-frame calibration (``<stem>_frames.dlt3d`` / ``.dlt2d``) in getpixelvideo + quickmeasure.

Update Date: 23 November 2026
Version: 0.4.19

The files are written here BY HAND in the frozen format of the SAM 3 output folder (one row per
video frame; ``frame,L1..L11`` for the DLT3D, ``frame,L1..L8`` for the ground homography,
``frame,source,rms_px,n_points,focal_px,from_frame`` for the cameras csv), from one known
camera, so the tests do not depend on how ``kick_measure`` writes them. The window runs in a
subprocess with ``SDL_VIDEODRIVER=dummy`` (events are posted by a thread, like
``test_getpixelvideo_kick_measure``): real mouse use is NOT covered.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pandas as pd
import pytest

from vaila import getpixelvideo as gpv
from vaila import kick_measure as km
from vaila import quickmeasure as qm

W, H = 320, 240
FPS = 30.0
N_FRAMES = 6
STEM = "maincam"
CENTRE = np.array([0.0, -40.0, 15.0])
APPROX_FRAMES = (3, 4)


def _camera(frame: int = 0) -> np.ndarray:
    """A camera 40 m behind the pitch, 15 m high, looking at the centre spot; zoom grows."""
    zc = -CENTRE / np.linalg.norm(CENTRE)
    xc = np.cross(zc, [0.0, 0.0, 1.0])
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R = np.array([xc, yc, zc])
    f = 300.0 + 10.0 * frame
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    return K @ np.c_[R, -R @ CENTRE]


def write_frames_files(
    folder: Path, stem: str = STEM, n: int = N_FRAMES, approx: tuple[int, ...] = APPROX_FRAMES
) -> Path:
    """The frozen per-frame files of one camera in ``folder``; returns the ``.dlt3d`` path."""
    folder.mkdir(parents=True, exist_ok=True)
    rows3, rows2, cams = [], [], []
    for f in range(n):
        P = _camera(f)
        P3 = P / P[2, 3]  # L12 = 1
        rows3.append([f, *P3[0], *P3[1], *P3[2, :3]])
        Hm = P[:, [0, 1, 3]]  # z = 0 columns: pixel ~ H @ [x, y, 1]
        Hm = Hm / Hm[2, 2]
        rows2.append([f, *Hm[0], *Hm[1], Hm[2, 0], Hm[2, 1]])
        src = "approximate" if f in approx else ("dlt3d" if f % 2 == 0 else "tracked")
        cams.append([f, src, 0.5, 5, 300.0 + 10.0 * f, (f - 1) if f in approx else ""])
    pd.DataFrame(rows3, columns=pd.Index(["frame"] + [f"L{i}" for i in range(1, 12)])).to_csv(
        folder / f"{stem}_frames.dlt3d", index=False
    )
    pd.DataFrame(rows2, columns=pd.Index(["frame"] + [f"L{i}" for i in range(1, 9)])).to_csv(
        folder / f"{stem}_frames.dlt2d", index=False
    )
    pd.DataFrame(
        cams, columns=pd.Index(["frame", "source", "rms_px", "n_points", "focal_px", "from_frame"])
    ).to_csv(folder / f"{stem}_frames_cameras.csv", index=False)
    return folder / f"{stem}_frames.dlt3d"


def write_video(folder: Path, stem: str = STEM) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    video = folder / f"{stem}.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), FPS, (W, H))
    for i in range(N_FRAMES):
        writer.write(np.full((H, W, 3), 40 + 20 * i, dtype=np.uint8))
    writer.release()
    return video


def _sam_folder(base: Path, ts: str = "20261123_100000", stem: str = STEM) -> Path:
    return base / "work" / stem / "sam3dinov3" / f"processed_sam3dinov3_{ts}" / stem


GROUND = np.array([5.0, -4.0, 0.0])


# ----------------------------------------------------------------------------------------------
# quickmeasure: loading the files
# ----------------------------------------------------------------------------------------------


def test_frames_dlt3d_loads_as_camera3d_with_the_camera_of_every_frame(tmp_path: Path) -> None:
    path = write_frames_files(tmp_path)
    calib = qm.load_frames_calibration(path, total_frames=N_FRAMES)
    assert isinstance(calib, qm.QuickMeasureCalibration) and calib.kind == "camera3d"
    for f in range(N_FRAMES):
        u, v = km.project_points(_camera(f), GROUND)[0]
        x, y = calib.pixel_to_real_at(f, float(u), float(v))
        assert (x, y) == pytest.approx((GROUND[0], GROUND[1]), abs=1e-3)
    # the calibration of frame 0 is NOT the one of frame 5 (the zoom changed): a click of frame 5
    # read with the camera of frame 0 would be off
    u5, v5 = km.project_points(_camera(5), GROUND)[0]
    assert calib.pixel_to_real_at(0, float(u5), float(v5)) != pytest.approx(
        (GROUND[0], GROUND[1]), abs=0.05
    )
    assert calib.source == str(path)


def test_frames_dlt3d_matches_the_ground_point_of_the_track(tmp_path: Path) -> None:
    path = write_frames_files(tmp_path)
    calib = qm.load_frames_calibration(path, total_frames=N_FRAMES)
    track = km.read_frame_calibration(path)
    assert isinstance(calib, qm.QuickMeasureCalibration) and calib.camera_track is not None
    for f, (u, v) in [(0, (100.0, 180.0)), (3, (210.0, 150.0)), (5, (60.0, 200.0))]:
        expected = track.ground_point(f, u, v, 0.0)
        got = calib.pixel_to_real_at(f, u, v)
        assert got == pytest.approx((float(expected[0]), float(expected[1])), abs=1e-6)


def test_frames_dlt2d_loads_one_homography_per_frame(tmp_path: Path) -> None:
    write_frames_files(tmp_path)
    calibs = qm.load_frames_calibration(tmp_path / f"{STEM}_frames.dlt2d", total_frames=N_FRAMES)
    assert isinstance(calibs, list) and len(calibs) == N_FRAMES
    u, v = km.project_points(_camera(2), GROUND)[0]
    assert calibs[2].pixel_to_real(float(u), float(v)) == pytest.approx(
        (GROUND[0], GROUND[1]), abs=1e-3
    )


def test_a_dlt3d_without_one_row_per_video_frame_is_a_clear_error(tmp_path: Path) -> None:
    path = write_frames_files(tmp_path, n=3)
    with pytest.raises(qm.QuickMeasureError, match="one row per video frame"):
        qm.load_frames_calibration(path, total_frames=N_FRAMES)
    # a sparse FreeKiki-style dlt3d with the wrong number of columns is refused too
    bad = tmp_path / "bad.dlt3d"
    pd.DataFrame([[0, 1, 2, 3]]).to_csv(bad, index=False)
    with pytest.raises(qm.QuickMeasureError, match="11 parameters"):
        qm.load_frames_calibration(bad, total_frames=1)


def test_the_load_dialog_accepts_a_frames_dlt3d(tmp_path: Path, monkeypatch) -> None:
    path = write_frames_files(tmp_path)
    root = Mock()
    monkeypatch.setattr("tkinter.Tk", lambda: root)
    monkeypatch.setattr("tkinter.messagebox.askyesnocancel", lambda *a, **k: True)
    monkeypatch.setattr("tkinter.filedialog.askopenfilename", lambda **k: str(path))
    result, msg = qm._ask_calibration_via_dialog(total_frames=N_FRAMES)
    assert isinstance(result, qm.QuickMeasureCalibration) and result.kind == "camera3d"
    assert "Per-frame DLT3D loaded" in msg
    u, v = km.project_points(_camera(4), GROUND)[0]
    assert result.pixel_to_real_at(4, float(u), float(v)) == pytest.approx(
        (GROUND[0], GROUND[1]), abs=1e-3
    )


def test_the_load_dialog_still_loads_a_frames_dlt2d_and_refuses_a_short_dlt3d(
    tmp_path: Path, monkeypatch
) -> None:
    write_frames_files(tmp_path)
    short = write_frames_files(tmp_path / "short", n=2)
    root, error = Mock(), Mock()
    monkeypatch.setattr("tkinter.Tk", lambda: root)
    monkeypatch.setattr("tkinter.messagebox.askyesnocancel", lambda *a, **k: True)
    monkeypatch.setattr("tkinter.messagebox.showerror", error)
    monkeypatch.setattr(
        "tkinter.filedialog.askopenfilename", lambda **k: str(tmp_path / f"{STEM}_frames.dlt2d")
    )
    result, _ = qm._ask_calibration_via_dialog(total_frames=N_FRAMES)
    assert isinstance(result, list) and len(result) == N_FRAMES
    monkeypatch.setattr("tkinter.filedialog.askopenfilename", lambda **k: str(short))
    result, msg = qm._ask_calibration_via_dialog(total_frames=N_FRAMES)
    assert result is None and "one row per video frame" in msg
    error.assert_called_once()


def test_frame_sources_and_the_approximate_warning_in_both_languages(tmp_path: Path) -> None:
    path = write_frames_files(tmp_path)
    sources = qm.read_frame_sources(path)
    assert [sources[f] for f in range(N_FRAMES)] == [
        "dlt3d", "tracked", "dlt3d", "approximate", "approximate", "tracked",
    ]  # fmt: skip
    assert qm.read_frame_sources(tmp_path / "nothing.dlt3d") == {}
    assert "APPROXIMATE" in qm.approximate_frame_warning("approximate", "en")
    assert "APROXIMADA" in qm.approximate_frame_warning("approximate", "pt")
    for source in ("dlt3d", "solved", "tracked", None):
        assert qm.approximate_frame_warning(source, "en") == ""


# ----------------------------------------------------------------------------------------------
# getpixelvideo: finding the files
# ----------------------------------------------------------------------------------------------


def test_the_frames_file_is_found_from_the_sam3_files_of_the_folder(tmp_path: Path) -> None:
    sam = _sam_folder(tmp_path)
    frames = write_frames_files(sam)
    (sam / "sam3").mkdir()
    track_csv = sam / "sam3" / "sam_tracks.csv"
    track_csv.write_text("frame,obj_id\n", encoding="utf-8")
    assert gpv.find_frames_calibration_near_files(track_csv, STEM) == frames
    assert gpv.find_frames_calibration_near_files(sam / f"{STEM}_id_01_markers.csv") == frames
    # the exact stem wins over another camera's file in the same folder
    other = write_frames_files(sam, stem="othercam")
    assert gpv.find_frames_calibration_near_files(track_csv, STEM) == frames
    assert gpv.find_frames_calibration_near_files(track_csv, "othercam") == other
    assert gpv.find_frames_calibration_near_files(tmp_path / "nowhere" / "x.csv", STEM) is None


def test_opening_a_video_finds_the_newest_kick_runs_frames_file(tmp_path: Path) -> None:
    video = write_video(tmp_path / "clip")
    assert gpv.find_frames_calibration_for_video(video) is None
    old = write_frames_files(_sam_folder(tmp_path / "clip" / "kick_analysis_20261101_100000"))
    new = write_frames_files(_sam_folder(tmp_path / "clip" / "kick_analysis_20261123_120000"))
    (tmp_path / "clip" / "kick_analysis_20261124_000000").mkdir()  # newest run, no file: skipped
    assert gpv.find_frames_calibration_for_video(video) == new
    assert old != new
    # a copy next to the video wins
    near = write_frames_files(tmp_path / "clip")
    assert gpv.find_frames_calibration_for_video(video) == near
    assert gpv.find_frames_calibration_for_video(None) is None
    assert gpv.find_frames_calibration_for_video(tmp_path / "missing.mp4") is None


def test_calib_frames_option_is_parsed() -> None:
    assert gpv.parse_calib_frames_arg(["getpixelvideo.py", "-f", "v.mp4"]) is None
    with pytest.raises(SystemExit, match="requires"):
        gpv.parse_calib_frames_arg(["x", "--calib-frames"])
    with pytest.raises(SystemExit, match="not found"):
        gpv.parse_calib_frames_arg(["x", "--calib-frames", "/nonexistent/a_frames.dlt3d"])


def test_calib_frames_option_returns_the_path(tmp_path: Path) -> None:
    path = write_frames_files(tmp_path)
    assert gpv.parse_calib_frames_arg(["x", "-f", "v.mp4", "--calib-frames", str(path)]) == str(
        path
    )
    csv = tmp_path / f"{STEM}_frames_cameras.csv"
    with pytest.raises(SystemExit, match="expects a .dlt3d or .dlt2d"):
        gpv.parse_calib_frames_arg(["x", "--calib-frames", str(csv)])


def test_the_help_screen_documents_the_frames_calibration() -> None:
    from vaila import getpixelvideo_help as help_

    entries = [e for e in help_.help_entries() if "--calib-frames" in e.en]
    assert entries, "no help entry for --calib-frames"
    assert "--calib-frames" in entries[0].pt and "approximate" in entries[0].en
    assert "aproximada" in entries[0].pt


# ----------------------------------------------------------------------------------------------
# window (subprocess, dummy video driver)
# ----------------------------------------------------------------------------------------------

_DRIVER = r"""
import os, sys, threading, time
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
import pygame
import vaila.getpixelvideo as gpv

video, calib_frames, actions = sys.argv[1], sys.argv[2], sys.argv[3].split()
seen = []

class LogFont:
    # logs the text of the HUD / status bar the window draws, to prove what the user would read
    def __init__(self, font):
        self._font = font
    def render(self, text, *args, **kwargs):
        if any(w in text for w in ("Calibrated per frame", "APPROXIMATE", "APROXIMADA", "Calibrado")):
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

KEYS = {"q": pygame.K_q, "right": pygame.K_RIGHT, "left": pygame.K_LEFT, "esc": pygame.K_ESCAPE}

def script():
    while not pygame.display.get_init() or pygame.display.get_surface() is None:
        time.sleep(0.1)
    time.sleep(2.0)
    for a in actions:
        if a.startswith("L"):
            x, y = a[1:].split(",")
            click(int(x), int(y))
        elif a.startswith("d"):
            key(getattr(pygame, "K_" + a[1:]))
        elif a == "wait":
            time.sleep(1.0)
        else:
            key(KEYS[a])

threading.Thread(target=script, daemon=True).start()
gpv.run_getpixelvideo(
    initial_media_path=video, initial_source_type="video",
    initial_calib_frames=calib_frames or None,
    initial_track_csv=os.environ.get("TRACK_CSV") or None,
)
"""


def _drive(
    video: Path, actions: str, calib_frames: Path | None = None, track_csv: Path | None = None
):
    env = dict(os.environ)
    env.pop("TRACK_CSV", None)
    if track_csv:
        env["TRACK_CSV"] = str(track_csv)
    return subprocess.run(
        [sys.executable, "-c", _DRIVER, str(video), str(calib_frames or ""), actions],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        cwd=Path(__file__).resolve().parents[1],
    )


def _texts(done) -> list[str]:
    return [line[5:] for line in done.stdout.splitlines() if line.startswith("TEXT ")]


def _results(video: Path) -> pd.DataFrame:
    found = sorted(video.parent.glob("processed_quickmeasure_*"))
    assert found, f"no processed_quickmeasure_* next to {video}"
    return pd.read_csv(next(found[-1].glob("*_quickmeasure_results.csv")))


def test_window_opens_calibrated_from_the_kick_run_file_and_a_2_click_distance_is_in_metres(
    tmp_path: Path,
) -> None:
    clip = tmp_path / "clip"
    video = write_video(clip)
    # the SAM 3 folder of a kick run of this video holds the per-frame files (found on opening)
    write_frames_files(_sam_folder(clip / "kick_analysis_20261123_120000"))

    done = _drive(video, "q d1 L100,150 L200,150 wait esc")

    assert done.returncode == 0, done.stderr[-2000:]
    out = " ".join(done.stdout.split())  # rich wraps long lines: compare without line breaks
    assert "per-frame calibration installed (next to the video / kick run)" in out
    assert f"camera3d for {N_FRAMES} frames" in out
    assert f"--calib-frames {clip.name}" not in out  # the mirror carries the absolute path
    assert "--calib-frames" in out and f"{STEM}_frames.dlt3d" in out
    # the HUD names the file, from frame 0
    assert any(t.startswith("Calibrated per frame: maincam_frames.dlt3d") for t in _texts(done))
    results = _results(video)
    dist = results[results["type"] == "distance"]
    assert len(dist) == 1, results
    assert str(dist["unit"].iloc[0]) == "m"
    assert 0.5 < float(dist["value"].iloc[0]) < 80.0  # metres of the pitch, not pixels


def test_window_calib_frames_option_and_the_approximate_frame_warning(tmp_path: Path) -> None:
    clip = tmp_path / "clip"
    video = write_video(clip)
    frames = write_frames_files(tmp_path / "elsewhere")  # not next to the video: explicit option

    done = _drive(video, "wait right right right wait esc", calib_frames=frames)

    assert done.returncode == 0, done.stderr[-2000:]
    out = " ".join(done.stdout.split())
    assert "per-frame calibration installed (--calib-frames)" in out
    assert f"--calib-frames {frames}" in out
    texts = _texts(done)
    assert any(t.startswith("Calibrated per frame") for t in texts), texts
    assert any("APPROXIMATE camera on this frame" in t for t in texts), texts  # frame 3


def test_window_loading_the_sam3_tracking_csv_installs_the_calibration_of_its_folder(
    tmp_path: Path,
) -> None:
    video = write_video(tmp_path / "clip")  # no kick run next to this video
    sam = _sam_folder(tmp_path / "run")
    write_frames_files(sam)
    (sam / "sam3").mkdir()
    points = sam / "sam3" / "sam_points.csv"
    pd.DataFrame({"frame": range(N_FRAMES), "p0_x": 100.0, "p0_y": 150.0}).to_csv(
        points, index=False
    )

    done = _drive(video, "wait esc", track_csv=points)

    assert done.returncode == 0, done.stderr[-2000:]
    out = " ".join(done.stdout.split())
    assert "per-frame calibration installed (SAM 3 folder)" in out
    assert f"--calib-frames {sam / f'{STEM}_frames.dlt3d'}" in out
    assert any(t.startswith("Calibrated per frame") for t in _texts(done))
