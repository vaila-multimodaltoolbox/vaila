"""
================================================================================
Script: gui_help.py - the '?' button of a Tk window: a hint on hover, a window on click
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.12
Created: 28 October 2026
Update Date: 28 October 2026

Description:
    :func:`add_help` puts a small '?' button next to a control or a section title. Moving the mouse
    over the button shows the text as a hint after a short delay; clicking it opens a window with the
    full text (Esc or OK closes it). Long explanatory labels in a window become a '?' this way.

    Only tkinter and ttk are used: no second ``Tk()`` is created, and the button runs in the window
    that already exists. The text is given by the caller (keep it in a dictionary, so it can be read
    and tested without a display).
"""

from __future__ import annotations

import contextlib
import tkinter as tk
from tkinter import ttk

HOVER_DELAY_MS = 400
HINT_WRAP_PX = 420
WINDOW_WRAP_PX = 520


class _Hint:
    """The hint of one button: a borderless label under the pointer, shown after a short delay."""

    def __init__(self, widget: tk.Misc, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip: tk.Toplevel | None = None
        self.after_id: str | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self.after_id = self.widget.after(HOVER_DELAY_MS, self._show)

    def _cancel(self) -> None:
        if self.after_id is not None:
            with contextlib.suppress(tk.TclError):
                self.widget.after_cancel(self.after_id)
            self.after_id = None

    def _show(self) -> None:
        self.after_id = None
        if self.tip is not None:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self.tip,
            text=self.text,
            justify="left",
            wraplength=HINT_WRAP_PX,
            background="#fffde0",
            relief="solid",
            borderwidth=1,
            padx=6,
            pady=4,
        ).pack()

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


def open_help_window(parent: tk.Misc, title: str, text: str) -> tk.Toplevel:
    """A window with the full text; Esc and OK close it."""
    win = tk.Toplevel(parent)
    win.title(title)
    win.transient(parent.winfo_toplevel())
    frame = ttk.Frame(win, padding=12)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text=text, wraplength=WINDOW_WRAP_PX, justify="left").pack(anchor="w")
    ttk.Button(frame, text="OK", command=win.destroy).pack(anchor="e", pady=(10, 0))
    win.bind("<Escape>", lambda _event: win.destroy())
    win.focus_set()
    return win


def add_help(parent: tk.Misc, text: str, *, title: str = "Help") -> ttk.Button:
    """A '?' button in ``parent``: hover for the hint, click for the window with ``text``."""
    button = ttk.Button(
        parent, text="?", width=2, command=lambda: open_help_window(parent, title, text)
    )
    _Hint(button, text)
    return button
