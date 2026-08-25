"""
Headless GUI test for `vaila.rearrange_data.ColumnReorderGUI`'s button
grouping (see readc3d_export.py's sibling test for the same pattern).

Follows the existing headless-Tk convention from
`tests/test_vaila_sam.py::test_sam_video_dialog_has_help_button`: try to
create a real `tkinter.Tk()` root under Xvfb/whatever display is available,
skip the test outright if none is available, and walk the live widget tree
rather than mocking Tkinter.
"""

from __future__ import annotations

import pytest

EXPECTED_SECTIONS = ("Columns", "Combine Files", "Import to vailá", "Advanced")

EXPECTED_BUTTONS = (
    "Convert Units",
    "Modify Lab Ref System",
    "Reset Index Col 0",
    "Merge CSV",
    "Stack/Append CSV",
    "Save 2nd Half CSV",
    "Convert YOLO Tracker to vailá",
    "Convert MediaPipe to vailá",
    "Convert Dvideo to vailá",
    "Convert DLC to vailá",
    "Convert Kinovea to vailá",
    "Standardize Header",
    "Custom Math Operation",
)


def _walk(widget):
    yield widget
    for child in widget.children.values():
        yield from _walk(child)


def test_column_reorder_gui_groups_buttons_into_labelframes(tmp_path) -> None:
    tkinter = pytest.importorskip("tkinter")
    try:
        probe = tkinter.Tk()
        probe.destroy()
    except tkinter.TclError as exc:
        pytest.skip(f"no display available for Tk: {exc}")

    from vaila.rearrange_data import ColumnReorderGUI

    dlg = ColumnReorderGUI(
        original_headers=["Column1", "Column2", "Column3"],
        file_names=["Empty"],
        directory_path=str(tmp_path),
    )
    dlg.withdraw()
    try:
        widgets = list(_walk(dlg))

        section_titles = {
            str(w.cget("text")) for w in widgets if w.winfo_class() == "TLabelframe"
        }
        missing_sections = set(EXPECTED_SECTIONS) - section_titles
        assert not missing_sections, f"missing button sections: {missing_sections}"

        button_texts = {str(w.cget("text")) for w in widgets if w.winfo_class() == "TButton"}
        missing_buttons = set(EXPECTED_BUTTONS) - button_texts
        assert not missing_buttons, f"missing buttons: {missing_buttons}"

        # Every button must live inside one of the labelframe sections, not
        # directly on the old flat button_frame.
        labelframes = {w for w in widgets if w.winfo_class() == "TLabelframe"}
        for w in widgets:
            if w.winfo_class() == "TButton" and str(w.cget("text")) in EXPECTED_BUTTONS:
                assert w.master in labelframes, (
                    f"button {w.cget('text')!r} is not parented under a LabelFrame section"
                )
    finally:
        dlg.destroy()
