"""The '?' help button (vaila/gui_help.py): the window opens with the text, and the hint is shown on hover.

The tests need a display; without one they are skipped, and the GUI is then reported as not tested.
"""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk

import pytest

from vaila import gui_help


@pytest.fixture
def root():
    try:
        window = tk.Tk()
    except tk.TclError:
        pytest.skip("no display: the help button was not tested in the GUI")
    window.withdraw()
    yield window
    window.destroy()


def test_the_button_is_a_question_mark_and_opens_the_text(root: tk.Tk) -> None:
    frame = ttk.Frame(root)
    frame.pack()
    text = "Retrain = Base model 'active' (default slot)."
    button = gui_help.add_help(frame, text, title="3. Train / Retrain")
    assert button.cget("text") == "?"
    button.invoke()
    root.update()
    windows = [w for w in frame.winfo_children() if isinstance(w, tk.Toplevel)]
    assert len(windows) == 1
    assert windows[0].title() == "3. Train / Retrain"
    labels = [
        w for w in windows[0].winfo_children()[0].winfo_children() if isinstance(w, ttk.Label)
    ]
    assert labels and labels[0].cget("text") == text
    windows[0].destroy()


def test_ok_closes_the_help_window_and_escape_is_bound(root: tk.Tk) -> None:
    # a key event only reaches a window that has the focus, and a display without a window manager
    # gives none: Escape is checked as bound, and the OK button is pressed
    frame = ttk.Frame(root)
    frame.pack()
    gui_help.add_help(frame, "text").invoke()
    root.update()
    window = [w for w in frame.winfo_children() if isinstance(w, tk.Toplevel)][0]
    assert window.bind("<Escape>")
    body = window.winfo_children()[0]
    ok = [w for w in body.winfo_children() if isinstance(w, ttk.Button) and w.cget("text") == "OK"]
    assert len(ok) == 1
    ok[0].invoke()
    root.update()
    assert not window.winfo_exists()


def test_hovering_shows_the_hint_and_leaving_hides_it(root: tk.Tk, monkeypatch) -> None:
    monkeypatch.setattr(gui_help, "HOVER_DELAY_MS", 1)
    frame = ttk.Frame(root)
    frame.pack()
    button = gui_help.add_help(frame, "Detect: field keypoints only.")
    button.pack()
    root.update()
    button.event_generate("<Enter>")
    for _ in range(20):
        root.update()
        time.sleep(0.005)
    tips = [w for w in button.winfo_children() if isinstance(w, tk.Toplevel)]
    assert len(tips) == 1
    button.event_generate("<Leave>")
    root.update()
    assert [w for w in button.winfo_children() if isinstance(w, tk.Toplevel)] == []
