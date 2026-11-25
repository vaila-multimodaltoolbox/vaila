"""1-line mode of getpixelvideo: one CSV row of points p0..pN for DLT calibration.

Bug (14 November 2026): a right click did not remove the last point. The click
handler never selected the point it had just added, and the per-frame clamp then
parked the selection on the first point of the frame, so right click removed p0;
the removed point was still drawn, and its number stayed blank in the CSV. Saving
with no points crashed the window (``os.path.basename(None)``).

Also covered: the row can be built over several videos (F8 with 1-line on, or
``--one-line CSV``), each point keeping the video and frame it was clicked in.

Author: Prof. Dr. Paulo R. P. Santiago and Rafael L. M. Monteiro
Update Date: 14 November 2026
Version: 0.4.19
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from vaila.getpixelvideo import (
    one_line_carry,
    one_line_live_indices,
    one_line_marker_in_frame,
    one_line_place,
    one_line_remove,
    one_line_remove_frame,
    one_line_selectable_indices,
    read_one_line_csv,
    sorted_frames_with_visible_markers,
    write_one_line_files,
)


def _three_points() -> tuple[list, set[int]]:
    markers: list = []
    deleted: set[int] = set()
    for frame, x, y in ((0, 10.0, 11.0), (0, 20.0, 21.0), (2, 30.0, 31.0)):
        one_line_place(markers, deleted, -1, frame, x, y)
    return markers, deleted


def test_right_click_removes_the_last_point_and_frees_its_number() -> None:
    markers, deleted = _three_points()

    removed, selected = one_line_remove(markers, deleted, -1)

    assert (removed, selected) == (2, -1)
    assert markers == [(0, 10.0, 11.0), (0, 20.0, 21.0)]
    # The next click is p2 again, not p3 with a blank p2 in the CSV.
    assert one_line_place(markers, deleted, selected, 1, 33.0, 34.0) == 2


def test_right_click_twice_removes_the_last_two_points() -> None:
    markers, deleted = _three_points()

    one_line_remove(markers, deleted, -1)
    one_line_remove(markers, deleted, -1)

    assert markers == [(0, 10.0, 11.0)]
    assert deleted == set()


def test_removing_a_selected_middle_point_keeps_the_other_numbers() -> None:
    markers, deleted = _three_points()

    removed, selected = one_line_remove(markers, deleted, 1)

    assert (removed, selected) == (1, 1)  # stays selected: the next click re-places it
    assert one_line_live_indices(markers, deleted) == [0, 2]
    assert one_line_place(markers, deleted, selected, 4, 25.0, 26.0) == 1
    assert markers[1] == (4, 25.0, 26.0)
    assert one_line_live_indices(markers, deleted) == [0, 1, 2]


def test_removing_the_last_point_also_drops_an_empty_slot_before_it() -> None:
    markers, deleted = _three_points()
    one_line_remove(markers, deleted, 1)  # p1 empty, p2 still there

    one_line_remove(markers, deleted, -1)  # removes p2: p1 is now a trailing empty slot

    assert markers == [(0, 10.0, 11.0)]
    assert deleted == set()


def test_right_click_with_nothing_to_remove_changes_nothing() -> None:
    markers: list = []
    deleted: set[int] = set()

    assert one_line_remove(markers, deleted, -1) == (None, -1)
    assert markers == []


def test_d_key_removes_only_the_points_of_the_frame_on_screen() -> None:
    markers = [(0, 1.0, 1.0, "calib_pos1.mp4"), (0, 2.0, 2.0), (1, 3.0, 3.0), (0, 4.0, 4.0)]
    deleted: set[int] = set()

    assert one_line_remove_frame(markers, deleted, 0) == 2

    # The carried point (frame 0 of ANOTHER video) and the frame-1 point stay.
    assert one_line_live_indices(markers, deleted) == [0, 2]
    assert len(markers) == 3  # the trailing p3 is gone, p1 is an empty slot


def test_selection_is_a_slot_of_the_row_or_none() -> None:
    markers, _ = _three_points()

    assert one_line_selectable_indices(markers, 2) == [2]
    assert one_line_selectable_indices(markers, -1) == []
    assert one_line_selectable_indices(markers, 7) == []


def test_points_from_another_video_are_not_frames_of_this_one() -> None:
    carried = one_line_carry([(0, 10.0, 11.0), (2, 20.0, 21.0)], "/data/calib_pos1.mp4")

    assert carried == [(0, 10.0, 11.0, "calib_pos1.mp4"), (2, 20.0, 21.0, "calib_pos1.mp4")]
    assert not one_line_marker_in_frame(carried[0], 0)
    assert one_line_marker_in_frame((0, 10.0, 11.0), 0)
    # Carried again, a point keeps the video it was really clicked in.
    assert one_line_carry(carried, "/data/calib_pos2.mp4") == carried
    assert sorted_frames_with_visible_markers(
        coordinates=None,
        deleted_positions=None,
        one_line_mode=True,
        one_line_markers=[*carried, (1, 5.0, 6.0)],
        deleted_markers=set(),
        total_frames=3,
    ) == [1]


def test_save_writes_one_row_and_where_each_point_was_clicked(tmp_path: Path) -> None:
    video = tmp_path / "calib_pos2.mp4"
    markers = [
        (0, 100.4, 200.6, "calib_pos1.mp4"),
        (1, 110.0, 90.0, "calib_pos1.mp4"),
        (2, 300.0, 210.0),
        (2, 310.0, 95.0),
    ]

    csv_path = write_one_line_files(str(video), markers, set())

    assert csv_path == str(tmp_path / "calib_pos2_markers_1_line.csv")
    df = pd.read_csv(csv_path)
    assert list(df.columns) == ["frame"] + [f"p{i}_{a}" for i in range(4) for a in "xy"]
    assert df.iloc[0].tolist() == [0, 100, 201, 110, 90, 300, 210, 310, 95]
    with open(tmp_path / "calib_pos2_markers_1_line.toml", "rb") as fh:
        points = tomllib.load(fh)["point"]
    assert [(p["marker"], p["video"], p["frame"]) for p in points] == [
        (0, "calib_pos1.mp4", 0),
        (1, "calib_pos1.mp4", 1),
        (2, "calib_pos2.mp4", 2),
        (3, "calib_pos2.mp4", 2),
    ]
    assert points[0]["x"] == pytest.approx(100.4)


def test_save_with_no_points_writes_nothing(tmp_path: Path) -> None:
    assert write_one_line_files(str(tmp_path / "clip.mp4"), [], set()) is None
    assert list(tmp_path.iterdir()) == []


def test_load_gives_back_frames_videos_and_empty_slots(tmp_path: Path) -> None:
    markers = [
        (0, 100.4, 200.6, "calib_pos1.mp4"),
        (0, None, None),  # p1 left empty: its number must survive the round trip
        (2, 300.0, 210.0),
    ]
    csv_path = write_one_line_files(str(tmp_path / "calib_pos2.mp4"), markers, set())
    assert csv_path is not None

    loaded, deleted = read_one_line_csv(csv_path, str(tmp_path / "calib_pos2.mp4"))

    assert loaded[0] == (0, 100.4, 200.6, "calib_pos1.mp4")
    assert loaded[1][1:3] == (None, None)
    assert loaded[2] == (2, 300.0, 210.0)
    assert one_line_live_indices(loaded, deleted) == [0, 2]


def test_load_of_a_plain_csv_from_another_video_marks_its_points_as_carried(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "calib_pos1_markers_1_line.csv"
    csv_path.write_text("frame,p0_x,p0_y,p1_x,p1_y\n7,10,20,30,40\n", encoding="utf-8")

    other, _ = read_one_line_csv(str(csv_path), str(tmp_path / "calib_pos2.mp4"))
    own, _ = read_one_line_csv(str(csv_path), str(tmp_path / "calib_pos1.mp4"))

    assert other == [(7, 10.0, 20.0, "calib_pos1"), (7, 30.0, 40.0, "calib_pos1")]
    assert own == [(7, 10.0, 20.0), (7, 30.0, 40.0)]


def test_a_toml_that_no_longer_matches_the_csv_is_ignored(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    csv_path = write_one_line_files(str(video), [(3, 100.0, 200.0)], set())
    assert csv_path is not None
    Path(csv_path).write_text("frame,p0_x,p0_y\n0,500,600\n", encoding="utf-8")  # edited by hand

    loaded, _ = read_one_line_csv(csv_path, str(video))

    assert loaded == [(0, 500.0, 600.0)]


# --- the real window, headless: synthetic events on a small synthetic video ---------

_DRIVER = """
import os, sys, threading, time
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
import pygame
import vaila.getpixelvideo as gpv

videos, actions = sys.argv[1].split(os.pathsep), sys.argv[2].split()
next_videos = videos[1:]

class FakeTk:
    def withdraw(self): ...
    def attributes(self, *a): ...
    def destroy(self): ...

import tkinter, tkinter.filedialog
tkinter.Tk = FakeTk
tkinter.filedialog.askopenfilename = lambda **kw: next_videos.pop(0)
sys.platform = "test"  # F8 then asks the (faked) Tk dialog, not the pygame browser
import webbrowser
webbrowser.open = lambda url, *a, **kw: print("BROWSER " + url, flush=True)
gpv.open_in_editor = lambda path: print("EDITOR " + path, flush=True)

def key(k, mods=0):
    pygame.key.set_mods(mods)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=k, mod=mods, unicode=""))
    time.sleep(0.3)
    pygame.key.set_mods(0)
    time.sleep(0.2)

def click(x, y, button):
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(x, y), button=button))
    time.sleep(0.15)
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(x, y), button=button))
    time.sleep(0.35)

def script():
    while not pygame.display.get_init() or pygame.display.get_surface() is None:
        time.sleep(0.1)
    time.sleep(1.5)
    for a in actions:
        if a == "insert":
            key(pygame.K_i, pygame.KMOD_CTRL)
        elif a == "c":
            key(pygame.K_c)
        elif a == "esc":
            key(pygame.K_ESCAPE)
        elif a == "f8":
            key(pygame.K_F8)
            time.sleep(3.0)
        elif a.startswith("L"):
            x, y = a[1:].split(",")
            click(int(x), int(y), 1)
        elif a == "R":
            click(5, 5, 3)
        elif a == "enter":  # accept the value shown in a dialog
            key(pygame.K_RETURN)
            time.sleep(0.3)
        elif a.startswith("text:"):  # replace the value shown in a dialog, then Enter
            for _ in range(8):
                key(pygame.K_BACKSPACE)
            for ch in a[5:]:
                pygame.event.post(
                    pygame.event.Event(pygame.KEYDOWN, key=ord(ch), mod=0, unicode=ch)
                )
                time.sleep(0.1)
            key(pygame.K_RETURN)
            time.sleep(0.3)
        elif a == "ref3d":
            key(pygame.K_r, pygame.KMOD_SHIFT)
        elif a == "f1":
            key(pygame.K_F1)
        elif a.startswith("button:"):  # a toolbar button, by its id in TOOLBAR_HIT_RECTS
            # The render loop drops and re-adds some ids every frame (ref3d_guide):
            # wait until this thread sees it instead of racing the redraw.
            name, rect = a.split(":", 1)[1], None
            for _ in range(100):
                rect = gpv.TOOLBAR_HIT_RECTS.get(name)
                if rect is not None:
                    break
                time.sleep(0.1)
            x, y = rect.center
            click(x, y, 1)
            time.sleep(3.0)

threading.Thread(target=script, daemon=True).start()
gpv.run_getpixelvideo(initial_media_path=videos[0], initial_source_type="video")
"""


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 30.0, (320, 240))
    for i in range(5):
        writer.write(np.full((240, 320, 3), 40 + 20 * i, dtype=np.uint8))
    writer.release()


def _drive(videos: list[Path], actions: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _DRIVER, ":".join(map(str, videos)), actions],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=Path(__file__).resolve().parents[1],
    )


def test_window_right_click_removes_the_last_clicked_point(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _write_video(video)

    done = _drive([video], "insert c L200,150 L300,100 L250,60 R R esc")

    assert done.returncode == 0, done.stderr[-2000:]
    df = pd.read_csv(tmp_path / "clip_markers_1_line.csv")
    assert list(df.columns) == ["frame", "p0_x", "p0_y"]
    assert df.notna().all(axis=None)


def test_window_save_with_no_points_does_not_crash(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _write_video(video)

    done = _drive([video], "insert c R esc")

    assert done.returncode == 0, done.stderr[-2000:]
    assert not (tmp_path / "clip_markers_1_line.csv").exists()


def test_window_f8_in_one_line_mode_takes_the_points_to_the_next_video(tmp_path: Path) -> None:
    first, second = tmp_path / "pos1.mp4", tmp_path / "pos2.mp4"
    _write_video(first)
    _write_video(second)

    done = _drive([first, second], "insert c L200,150 L210,60 f8 L300,150 L310,60 esc")

    assert done.returncode == 0, done.stderr[-2000:]
    df = pd.read_csv(tmp_path / "pos2_markers_1_line.csv")
    assert list(df.columns) == ["frame"] + [f"p{i}_{a}" for i in range(4) for a in "xy"]
    assert df.notna().all(axis=None)
    with open(tmp_path / "pos2_markers_1_line.toml", "rb") as fh:
        videos = [p["video"] for p in tomllib.load(fh)["point"]]
    assert videos == ["pos1.mp4", "pos1.mp4", "pos2.mp4", "pos2.mp4"]


def test_window_next_video_button_takes_the_points_to_the_next_video(tmp_path: Path) -> None:
    first, second = tmp_path / "pos1.mp4", tmp_path / "pos2.mp4"
    _write_video(first)
    _write_video(second)

    # With 1-line points the Open button reads "Add video" (id open_next).
    done = _drive(
        [first, second], "insert c L200,150 L210,60 button:open_next L300,150 L310,60 esc"
    )

    assert done.returncode == 0, done.stderr[-2000:]
    df = pd.read_csv(tmp_path / "pos2_markers_1_line.csv")
    assert list(df.columns) == ["frame"] + [f"p{i}_{a}" for i in range(4) for a in "xy"]
    assert df.notna().all(axis=None)


def test_window_ref3d_pole_over_two_videos_writes_ref3d_and_dlt3d(tmp_path: Path) -> None:
    """REF3D button (Shift+R), pole template with the default answers (marks 0, 1 at
    the corners of a 1 x 5 rectangle = 8 points), 4 clicks in each of two videos."""
    from vaila import ref3d_builder
    from vaila.dlt3d import normalize_ref3d_to_format1

    first, second = tmp_path / "pos1.mp4", tmp_path / "pos2.mp4"
    _write_video(first)
    _write_video(second)
    clicks_1 = "L60,200 L62,120 L140,190 L141,125"
    clicks_2 = "L210,170 L211,110 L270,165 L272,112"

    done = _drive(
        [first, second],
        f"insert ref3d enter enter enter {clicks_1} f8 {clicks_2} esc",
    )

    assert done.returncode == 0, done.stderr[-2000:]
    stem = tmp_path / "pos2_markers_1_line"
    pixels = pd.read_csv(f"{stem}.csv")
    assert list(pixels.columns) == ["frame"] + [f"p{i}_{a}" for i in range(8) for a in "xy"]
    assert pixels.notna().all(axis=None)
    ref = normalize_ref3d_to_format1(f"{stem}.ref3d")
    assert ref is not None
    assert ref.iloc[0, 1:].to_numpy(dtype=float).reshape(-1, 3)[-1].tolist() == [1, 5, 1]
    assert ref3d_builder.read_template(f"{stem}.template.toml").kind == "pole"
    dlt = pd.read_csv(f"{stem}.dlt3d")
    assert list(dlt.columns) == ["frame"] + [f"L{i}" for i in range(1, 12)]
    assert "DLT3D from 8 points" in done.stdout
    # These clicks are synthetic (fx 129 vs fy 1202): not a real camera, so the terminal must
    # not claim that the .dlt3d is final. The positive case is in test_ref3d_builder.py.
    assert "NOT A REAL CAMERA" in done.stdout
    assert "no need to run it again" not in done.stdout


def test_window_ref3d_guide_button_and_f1_open_the_illustrated_guide(tmp_path: Path) -> None:
    """With REF3D on, F1 and the Guide button of the panel open ref3d_guide.html;
    without it, F1 opens the getpixelvideo manual. No browser opens (faked)."""
    video = tmp_path / "clip.mp4"
    _write_video(video)

    done = _drive([video], "f1 insert ref3d enter enter enter button:ref3d_guide f1 esc")

    assert done.returncode == 0, done.stderr[-2000:]
    opened = [ln.split()[-1] for ln in done.stdout.splitlines() if ln.startswith("BROWSER ")]
    assert len(opened) == 3, done.stdout[-2000:]
    assert opened[0].endswith("/help/getpixelvideo.html")
    assert all(url.endswith("/help/ref3d_guide.html") for url in opened[1:])
    assert all(Path(url.removeprefix("file://")).is_file() for url in opened)


def test_window_ref3d_new_template_then_load_it_with_enter(tmp_path: Path) -> None:
    """REF3D N writes ref3d_template.toml next to the video and opens it in the editor
    (faked); after it is filled in, REF3D + Enter (T) loads it: no measures typed."""
    from vaila import ref3d_builder
    from vaila.dlt3d import normalize_ref3d_to_format1

    video = tmp_path / "pos1.mp4"
    _write_video(video)
    template = tmp_path / "ref3d_template.toml"

    done = _drive([video], "insert ref3d text:n esc")

    assert done.returncode == 0, done.stderr[-2000:]
    assert f"EDITOR {template}" in done.stdout
    assert ref3d_builder.read_template(template).kind == "pole"
    text = template.read_text(encoding="utf-8")
    assert "rect = [1.4, 5]" in text
    # two marks x 4 corners = 8 points, as the clicks below expect
    edited = text.replace("marks = [0, 0.25, 0.5, 0.75, 1]", "marks = [0, 1]")
    template.write_text(edited.replace("rect = [1.4, 5]", "rect = [1.38, 5]"), encoding="utf-8")

    clicks = "L60,200 L62,120 L140,190 L141,125 L210,170 L211,110 L270,165 L272,112"
    done = _drive([video], f"insert ref3d enter {clicks} esc")

    assert done.returncode == 0, done.stderr[-2000:]
    assert "--template-file" in done.stdout
    stem = tmp_path / "pos1_markers_1_line"
    ref = normalize_ref3d_to_format1(f"{stem}.ref3d")
    assert ref is not None
    xyz = ref.iloc[0, 1:].to_numpy(dtype=float).reshape(-1, 3)
    assert len(xyz) == 8
    assert xyz[-1].tolist() == [1.38, 5, 1]
    assert (tmp_path / "pos1_markers_1_line.dlt3d").is_file()


def test_window_ref3d_template_videos_send_clicks_to_their_own_points(tmp_path: Path) -> None:
    """A template whose [videos] puts pos2.mp4 at (1.38, 0): with REF3D + Enter (T) the two
    clicks made in that video go to its own points p2 and p3, not to p0 and p1."""
    video = tmp_path / "pos2.mp4"
    _write_video(video)
    (tmp_path / "ref3d_template.toml").write_text(
        'kind = "pole"\n[pole]\nmarks = [0, 1]\nrect = [1.38, 5]\n'
        '[videos]\n"pos2.mp4" = [1.38, 0]\n',
        encoding="utf-8",
    )

    done = _drive([video], "insert ref3d enter L60,200 L62,120 esc")

    assert done.returncode == 0, done.stderr[-2000:]
    assert "--template-file" in done.stdout
    import tomllib as _tomllib

    saved = _tomllib.loads((tmp_path / "pos2_markers_1_line.toml").read_text(encoding="utf-8"))
    assert [p["marker"] for p in saved["point"]] == [2, 3]
