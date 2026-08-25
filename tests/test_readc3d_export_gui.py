"""
Headless GUI test for `vaila.readc3d_export.open_menu()`'s button grouping.

`open_menu()` used to be a nested closure inside `if __name__ == "__main__"`
(untestable); it now lives at module level. `root.mainloop()` is patched to
a no-op so the test doesn't block.
"""

from __future__ import annotations

import pytest

EXPECTED_BUTTONS = (
    "Batch Convert (Directory)",
    "Single Convert (File)",
    "Inspect C3D File",
    "Exit",
)


def _walk(widget):
    yield widget
    for child in widget.children.values():
        yield from _walk(child)


def test_open_menu_groups_buttons_into_actions_labelframe(monkeypatch: pytest.MonkeyPatch) -> None:
    tkinter = pytest.importorskip("tkinter")
    try:
        probe = tkinter.Tk()
        probe.destroy()
    except tkinter.TclError as exc:
        pytest.skip(f"no display available for Tk: {exc}")

    from vaila.readc3d_export import open_menu

    created_roots = []
    real_tk_init = tkinter.Tk.__init__

    def _tracking_init(self, *a, **k):
        real_tk_init(self, *a, **k)
        created_roots.append(self)

    monkeypatch.setattr(tkinter.Tk, "__init__", _tracking_init)
    monkeypatch.setattr(tkinter.Tk, "mainloop", lambda self: None)

    open_menu()
    assert created_roots, "open_menu() did not create a Tk root"
    root = created_roots[0]
    try:
        widgets = list(_walk(root))

        section_titles = {
            str(w.cget("text")) for w in widgets if w.winfo_class() == "TLabelframe"
        }
        assert "Actions" in section_titles

        buttons = {w for w in widgets if w.winfo_class() == "TButton"}
        button_texts = {str(w.cget("text")) for w in buttons}
        missing = set(EXPECTED_BUTTONS) - button_texts
        assert not missing, f"missing buttons: {missing}"

        actions_frame = next(w for w in widgets if w.winfo_class() == "TLabelframe")
        for w in buttons:
            text = str(w.cget("text"))
            if text == "Exit":
                # Exit sits outside the Actions group (own bottom row).
                continue
            assert w.master is actions_frame, f"button {text!r} not parented under Actions"
    finally:
        root.destroy()
