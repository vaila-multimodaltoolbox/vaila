"""Output-directory defaults: output follows the input folder unless the user picks another."""

import tkinter as tk
from unittest import mock

import pytest

from vaila import dialogsuser
from vaila.dialogsuser import ask_output_directory, default_output_dir, link_output_to_input


def test_default_output_dir_file_dir_list_and_empty(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    assert default_output_dir(str(video)) == str(tmp_path)
    assert default_output_dir(tmp_path) == str(tmp_path)
    assert default_output_dir(("", str(video))) == str(tmp_path)
    assert default_output_dir("") == ""
    assert default_output_dir(None) == ""
    assert default_output_dir([]) == ""
    assert default_output_dir(str(tmp_path / "missing" / "x.csv")) == ""


def test_ask_output_directory_preselects_input_folder(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("a\n1\n")
    with mock.patch.object(dialogsuser.filedialog, "askdirectory", return_value="") as ask:
        assert ask_output_directory(str(csv), title="Out") == ""
    kwargs = ask.call_args.kwargs
    assert kwargs["initialdir"] == str(tmp_path)
    assert kwargs["title"] == "Out (default: input folder)"

    with mock.patch.object(dialogsuser.filedialog, "askdirectory", return_value="/x") as ask:
        assert ask_output_directory(None, title="Out") == "/x"
    assert "initialdir" not in ask.call_args.kwargs
    assert ask.call_args.kwargs["title"] == "Out"


@pytest.fixture
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available for Tk")
    root.withdraw()
    yield root
    root.destroy()


def test_link_output_to_input_follows_until_user_overrides(tmp_path, tk_root):
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    in_var = tk.StringVar(tk_root, value="")
    out_var = tk.StringVar(tk_root, value="")
    link_output_to_input(in_var, out_var)

    in_var.set(str(first / "v.mp4"))
    assert out_var.get() == str(first)
    in_var.set(str(second))
    assert out_var.get() == str(second)

    out_var.set("/custom/out")
    in_var.set(str(first))
    assert out_var.get() == "/custom/out"
