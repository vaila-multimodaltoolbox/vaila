"""AI Track network in getpixelvideo: where it is saved, when, and the Net button.

Update Date: 24 November 2026
Version: 0.4.19

Defect 2 of the AI Track plan: the discriminator was saved only when AI Track was switched
OFF, so closing the window (or switching video) with it ON lost the session. These tests run
the real window in a subprocess (``SDL_VIDEODRIVER=dummy``, events posted by a thread, like
``test_getpixelvideo_one_line``) on a small synthetic video with a moving white disc; Deep is
OFF, so no ResNet is needed. ``VAILA_AI_MODELS_HOME`` points every subprocess at a temporary
folder and nothing here calls ``ai_tracker_store.set_home`` (no ``~/.vaila`` file is written).

Real mouse use is NOT covered: the clicks are synthetic pygame events.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from vaila import ai_tracker_store

REPO = Path(__file__).resolve().parents[1]
N_FRAMES = 40

# ``mode``: "direct" calls run_getpixelvideo (the window module is ``gpv``); "cli" runs the file
# as ``__main__`` with sys.argv, so ``--ai-models`` goes through the real command-line parser.
_DRIVER = r"""
import json, os, runpy, subprocess, sys, threading, time
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
import pygame

video, actions, mode = sys.argv[1], sys.argv[2].split(), sys.argv[3]
extra = json.loads(sys.argv[4])  # extra command-line words for mode "cli"
repo = os.getcwd()


class FakeTk:
    def withdraw(self): ...
    def attributes(self, *a): ...
    def destroy(self): ...


import tkinter, tkinter.filedialog, tkinter.messagebox
tkinter.Tk = FakeTk
tkinter.messagebox.showinfo = lambda *a, **kw: print("TKINFO " + str(a[:2]), flush=True)
tkinter.filedialog.askdirectory = lambda **kw: print("ASKDIR", flush=True) or os.environ.get(
    "AIM_CHOSEN", ""
)

if os.environ.get("AIM_SANDBOX"):  # first-run tests: never touch ~/.vaila or the real legacy folder
    from vaila import ai_tracker_store as _st

    _st.CONFIG_FILE = os.path.join(os.environ["AIM_SANDBOX"], "cfg", "ai_models.json")
    _st.LEGACY_DIR = os.path.join(os.environ["AIM_SANDBOX"], "legacy")


class FakeProc:
    pid = 0


_real_popen = subprocess.Popen


def fake_popen(argv, *a, **kw):
    # Only the hub is recorded instead of started; any other process (ldconfig, ffmpeg) is real.
    if isinstance(argv, (list, tuple)) and "vaila.ai_tracker_hub" in map(str, argv):
        print("POPEN " + json.dumps([str(x) for x in argv]) + " CWD " + str(kw.get("cwd")), flush=True)
        return FakeProc()
    return _real_popen(argv, *a, **kw)


subprocess.Popen = fake_popen


def window_module():
    return sys.modules["__main__"] if mode == "cli" else sys.modules["vaila.getpixelvideo"]


def key(k, mods=0):
    pygame.key.set_mods(mods)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=k, mod=mods, unicode=""))
    time.sleep(0.3)
    pygame.key.set_mods(0)
    time.sleep(0.2)


def click(x, y, button=1):
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(x, y), button=button))
    time.sleep(0.15)
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(x, y), button=button))
    time.sleep(0.35)


def script():
    while not pygame.display.get_init() or pygame.display.get_surface() is None:
        time.sleep(0.1)
    time.sleep(2.0)
    print("WINDOW " + str(pygame.display.get_surface().get_size()), flush=True)
    for a in actions:
        if a == "insert":
            key(pygame.K_i, pygame.KMOD_CTRL)
        elif a == "t":  # the AI Track hotkey
            key(pygame.K_t)
        elif a == "right":
            key(pygame.K_RIGHT)
        elif a == "left":
            key(pygame.K_LEFT)
        elif a == "esc":
            key(pygame.K_ESCAPE)
        elif a == "quit":  # the window close button
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            time.sleep(0.5)
        elif a == "ctrl+s":  # Save hub, then Esc cancels its dialog
            key(pygame.K_s, pygame.KMOD_CTRL)
            time.sleep(0.5)
            key(pygame.K_RETURN)
            time.sleep(2.0)
        elif a.startswith("L"):
            x, y = a[1:].split(",")
            click(int(x), int(y))
        elif a.startswith("button:"):
            name, rect = a.split(":", 1)[1], None
            for _ in range(100):
                rect = window_module().TOOLBAR_HIT_RECTS.get(name)
                if rect is not None:
                    break
                time.sleep(0.1)
            x, y = rect.center
            click(x, y)
            time.sleep(1.0)
        elif a == "wait":
            time.sleep(1.0)


threading.Thread(target=script, daemon=True).start()
if mode == "cli":
    sys.argv = ["getpixelvideo.py", "-f", video, *extra]
    sys.path.insert(0, os.path.join(repo, "vaila"))
    runpy.run_path(os.path.join(repo, "vaila", "getpixelvideo.py"), run_name="__main__")
else:
    import vaila.getpixelvideo as gpv
    gpv.run_getpixelvideo(initial_media_path=video, initial_source_type="video")
"""


def _write_video(path: Path) -> None:
    """40 frames, 320x240: a textured background and a white disc moving 5 px per frame."""
    rng = np.random.default_rng(0)
    background = rng.integers(30, 70, size=(240, 320, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 30.0, (320, 240))
    for i in range(N_FRAMES):
        frame = background.copy()
        cv2.circle(frame, (60 + 5 * i, 120), 14, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def _drive(
    video: Path,
    actions: str,
    home: Path,
    *,
    mode: str = "direct",
    extra: list[str] | None = None,
    env_home: bool = True,
    sandbox: Path | None = None,
    chosen: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("VAILA_AI_MODELS_HOME", None)
    if sandbox is not None:
        env["AIM_SANDBOX"] = str(sandbox)
    if chosen is not None:
        env["AIM_CHOSEN"] = str(chosen)
    if env_home:
        env["VAILA_AI_MODELS_HOME"] = str(home)
    return subprocess.run(
        [sys.executable, "-c", _DRIVER, str(video), actions, mode, json.dumps(extra or [])],
        capture_output=True,
        text=True,
        timeout=240,
        cwd=REPO,
        env=env,
    )


# The disc is at (60, 120) in video pixels at frame 0; the window maps the 320x240 video onto
# the screen, so the clicks below are in SCREEN pixels. The anchor is a real click on the disc;
# the next ones are corrections a few frames later (the human anchors the tracker learns from).
ANCHOR_AND_CORRECTIONS = "insert t L{x0},{y0} right right right L{x1},{y1} right right L{x2},{y2}"


def _clicks(scale: float = 1.0, ox: float = 0.0, oy: float = 0.0) -> str:
    def px(i: int) -> tuple[int, int]:
        return round(ox + (60 + 5 * i) * scale), round(oy + 120 * scale)

    (x0, y0), (x1, y1), (x2, y2) = px(0), px(3), px(5)
    return ANCHOR_AND_CORRECTIONS.format(x0=x0, y0=y0, x1=x1, y1=y1, x2=x2, y2=y2)


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "disc.mp4"
    _write_video(path)
    return path


def _saved(home: Path) -> tuple[list[Path], list[Path]]:
    root = home / "ai_tracker"
    return (
        sorted((root / "discriminators").glob("discriminator_default__classic.npz")),
        sorted((root / "samples").glob("*.npz")),
    )


def _window_scale(done: subprocess.CompletedProcess[str]) -> str:
    line = next((ln for ln in done.stdout.splitlines() if ln.startswith("WINDOW ")), "")
    return line


def test_closing_with_ai_track_on_saves_the_network_and_the_samples(
    tmp_path: Path, video: Path
) -> None:
    """Defect 2: Esc with AI Track still ON must write the checkpoint and the human samples."""
    home = tmp_path / "models_home"

    done = _drive(video, f"{_clicks()} esc", home)

    assert done.returncode == 0, done.stderr[-2000:]
    checkpoints, samples = _saved(home)
    assert checkpoints, f"no checkpoint after closing with AI Track ON\n{done.stdout[-3000:]}"
    assert samples, "no human samples saved"
    assert "AI Track saved" in done.stdout


def test_window_close_button_with_ai_track_on_also_saves(tmp_path: Path, video: Path) -> None:
    home = tmp_path / "models_home"

    done = _drive(video, f"{_clicks()} quit", home)

    assert done.returncode == 0, done.stderr[-2000:]
    checkpoints, samples = _saved(home)
    assert checkpoints and samples, done.stdout[-3000:]


def test_turning_ai_track_off_saves_under_the_home(tmp_path: Path, video: Path) -> None:
    home = tmp_path / "models_home"

    done = _drive(video, f"{_clicks()} t esc", home)

    assert done.returncode == 0, done.stderr[-2000:]
    checkpoints, samples = _saved(home)
    assert checkpoints and samples, done.stdout[-3000:]
    assert "(ai_track_off)" in done.stdout


def test_save_hub_saves_the_network_before_the_files(tmp_path: Path, video: Path) -> None:
    home = tmp_path / "models_home"

    done = _drive(video, f"{_clicks()} ctrl+s esc", home)

    assert "(save)" in done.stdout, done.stdout[-3000:]
    checkpoints, _ = _saved(home)
    assert checkpoints


def test_nothing_learned_writes_nothing_and_never_asks_for_a_folder(
    tmp_path: Path, video: Path
) -> None:
    """AI Track never switched on: no checkpoint, no samples, no folder dialog."""
    home = tmp_path / "models_home"

    done = _drive(video, "insert esc", home, env_home=False)

    assert done.returncode == 0, done.stderr[-2000:]
    assert "ASKDIR" not in done.stdout and "TKINFO" not in done.stdout
    assert not (home / "ai_tracker").exists()


def test_first_save_asks_for_the_folder_once_and_uses_the_answer(
    tmp_path: Path, video: Path
) -> None:
    sandbox, chosen = tmp_path / "sandbox", tmp_path / "chosen"

    done = _drive(
        video, f"{_clicks()} ctrl+s esc", tmp_path, env_home=False, sandbox=sandbox, chosen=chosen
    )

    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.count("ASKDIR") == 1 and "TKINFO" in done.stdout
    checkpoints, samples = _saved(chosen)
    assert checkpoints and samples, done.stdout[-3000:]
    cfg = json.loads((sandbox / "cfg" / "ai_models.json").read_text(encoding="utf-8"))
    assert Path(cfg["home"]) == chosen


def test_cancelling_the_folder_question_keeps_the_legacy_folder_and_is_not_asked_again(
    tmp_path: Path, video: Path
) -> None:
    sandbox = tmp_path / "sandbox"

    done = _drive(
        video, f"{_clicks()} ctrl+s esc", tmp_path, env_home=False, sandbox=sandbox, chosen=None
    )

    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.count("ASKDIR") == 1, done.stdout[-3000:]  # Save and exit saved twice
    assert list((sandbox / "legacy" / "discriminators").glob("*.npz"))
    assert not (sandbox / "cfg").exists()  # nothing was chosen, nothing remembered


def test_cli_ai_models_beats_the_environment_home(tmp_path: Path, video: Path) -> None:
    env_home, cli_home = tmp_path / "from_env", tmp_path / "from_cli"

    done = _drive(
        video, f"{_clicks()} esc", env_home, mode="cli", extra=["--ai-models", str(cli_home)]
    )

    assert done.returncode == 0, done.stderr[-2000:]
    checkpoints, samples = _saved(cli_home)
    assert checkpoints and samples, done.stdout[-3000:]
    assert not (env_home / "ai_tracker" / "discriminators").exists()


def test_cli_ai_models_that_cannot_be_created_stops_with_a_message(
    tmp_path: Path, video: Path
) -> None:
    blocker = tmp_path / "a_file"
    blocker.write_text("not a folder", encoding="utf-8")

    done = _drive(
        video,
        "wait",
        tmp_path / "unused",
        mode="cli",
        extra=["--ai-models", str(blocker / "sub")],
    )

    assert done.returncode != 0
    assert "--ai-models" in (done.stdout + done.stderr)


def test_net_button_opens_the_hub_in_another_process(tmp_path: Path, video: Path) -> None:
    home = tmp_path / "models_home"

    done = _drive(video, f"{_clicks()} button:ai_net esc", home)

    assert done.returncode == 0, done.stderr[-2000:]
    popen = [ln for ln in done.stdout.splitlines() if ln.startswith("POPEN ")]
    assert len(popen) == 1, done.stdout[-3000:]
    argv = json.loads(popen[0][len("POPEN ") :].split(" CWD ")[0])
    assert argv[1:] == ["-m", "vaila.ai_tracker_hub", "gui"]
    assert popen[0].endswith(f"CWD {REPO}")
    assert ">> vaila/ai_tracker_hub: uv run --no-sync python -m vaila.ai_tracker_hub gui" in (
        done.stdout
    )
    # The network is saved BEFORE the window opens, so it shows fresh data.
    assert "(net)" in done.stdout
    assert done.stdout.index("(net)") < done.stdout.index("POPEN ")
    assert _saved(home)[0]


def test_net_button_passes_the_cli_home_to_the_hub(tmp_path: Path, video: Path) -> None:
    cli_home = tmp_path / "from_cli"

    done = _drive(
        video,
        "button:ai_net esc",
        tmp_path / "unused",
        mode="cli",
        extra=["--ai-models", str(cli_home)],
    )

    popen = [ln for ln in done.stdout.splitlines() if ln.startswith("POPEN ")]
    assert len(popen) == 1, done.stdout[-3000:]
    argv = json.loads(popen[0][len("POPEN ") :].split(" CWD ")[0])
    assert argv[1:] == ["-m", "vaila.ai_tracker_hub", "--ai-models", str(cli_home), "gui"]
    assert (
        f">> vaila/ai_tracker_hub: uv run --no-sync python -m vaila.ai_tracker_hub "
        f"--ai-models {cli_home} gui"
    ) in done.stdout


# --- Sample bank: only points a person clicked in this session ---------------------------------


# In the 860x386 dummy window the 320x240 video is centred: video x = screen x - 245 (zoom 1).
SCREEN_OX = 245


def _bank_frames(home: Path) -> tuple[list[Path], set[int]]:
    """Files in the sample bank and the frames of its positive (marker) samples."""
    files = _saved(home)[1]
    frames: set[int] = set()
    for path in files:
        data = ai_tracker_store.load_samples(path)
        frames |= {int(f) for f, lab in zip(data["frames"], data["labels"], strict=True) if lab > 0}
    return files, frames


def test_switching_ai_track_on_over_existing_points_logs_no_samples(
    tmp_path: Path, video: Path
) -> None:
    """Existing coordinates fed to the tracker are not human clicks in this session."""
    home = tmp_path / "models_home"
    p0, p3, p5 = (SCREEN_OX + 60 + 5 * i for i in (0, 3, 5))
    actions = (
        f"insert L{p0},120 right right right L{p3},120 right right L{p5},120 t wait right right esc"
    )

    done = _drive(video, actions, home)

    assert done.returncode == 0, done.stderr[-2000:]
    assert "AI Track" in done.stdout, done.stdout[-3000:]
    files, frames = _bank_frames(home)
    assert not files and not frames, (
        f"samples saved without a click: {frames}\n{done.stdout[-3000:]}"
    )


def test_frames_written_by_the_tracker_never_become_samples(tmp_path: Path, video: Path) -> None:
    """One click, the tracker writes the next frames, stepping back over them re-anchors on its
    own output: the bank keeps only the clicked frame."""
    home = tmp_path / "models_home"
    x0 = SCREEN_OX + 60
    actions = f"insert t L{x0},120 right right right right left left left left left esc"

    done = _drive(video, actions, home)

    assert done.returncode == 0, done.stderr[-2000:]
    files, frames = _bank_frames(home)
    assert files, f"the click was not saved\n{done.stdout[-3000:]}"
    assert frames == {0}, f"samples at frames {sorted(frames)}, only frame 0 was clicked"
