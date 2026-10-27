"""Layout of the FreeKiki window: it must not ask for a window wider than a laptop screen.

The window is built for real (Tk) and the main loop is replaced by a function that reads the
requested size and closes the window. Skipped when there is no display.
"""

from __future__ import annotations

import os
import tkinter as tk

import pytest

from vaila import freekiki

pytestmark = pytest.mark.skipif(
    not os.environ.get("DISPLAY") and os.name != "nt",
    reason="needs a display to build the Tk window",
)

# A laptop screen is about 1366 px wide; the window must fit with room for the title bar.
MAX_REQUESTED_WIDTH = 900


def test_freekiki_window_requests_no_more_than_a_laptop_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measured: dict[str, int] = {}

    def read_size_and_close(self: tk.Tk, n: int = 0) -> None:
        self.update_idletasks()
        self.update()
        measured["width"] = self.winfo_reqwidth()
        self.destroy()

    monkeypatch.setattr(tk.Tk, "mainloop", read_size_and_close)
    freekiki.run_freekiki()
    assert "width" in measured, "the window was not built"
    assert measured["width"] <= MAX_REQUESTED_WIDTH, (
        f"the FreeKiki window asks for {measured['width']} px; the long lines must wrap"
    )
