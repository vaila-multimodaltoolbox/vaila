"""Downloader backend and batch regression tests (no network).

Version: 0.3.137
Update Date: 11 September 2026
"""

import contextlib
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from vaila import vaila_ytdown as yt
from vaila.task_feedback import Feedback, redact


@pytest.fixture
def backend(monkeypatch):
    state = {"calls": [], "options": [], "during_processing": None, "missing_output": False}

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts
            state["options"].append(opts)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def prepare_filename(self, info):
            return (
                self.opts["outtmpl"].replace("%(title)s", info["title"]).replace("%(ext)s", "webm")
            )

        def extract_info(self, url, download):
            if not download:
                return {"title": "Example", "formats": []}
            state["calls"].append(url)
            if "fail" in url:
                raise OSError("simulated transfer failure token=hidden-token")
            info = {"title": "Example", "width": 1920, "height": 1080, "fps": 60, "duration": 5}
            for hook in self.opts.get("progress_hooks", []):
                hook({"status": "downloading", "downloaded_bytes": 50, "total_bytes": 100})
                hook({"status": "finished"})
            for hook in self.opts.get("postprocessor_hooks", []):
                hook({"status": "started", "postprocessor": "FFmpeg", "info_dict": info})
            if state["during_processing"]:
                state["during_processing"]()
            suffix = ".mp4" if self.opts["format"] == "bestvideo+bestaudio/best" else ".mp3"
            output = Path(self.prepare_filename(info)).with_suffix(suffix)
            if not state["missing_output"]:
                output.write_bytes(b"fake finished media")
            info["filepath"] = str(output)
            for hook in self.opts.get("postprocessor_hooks", []):
                hook({"status": "finished", "postprocessor": "FFmpeg", "info_dict": info})
            return info

    monkeypatch.setattr(yt.yt_dlp, "YoutubeDL", FakeYDL)
    monkeypatch.setattr(yt.YTDownloader, "_check_ffmpeg", lambda self: True)
    return state


@pytest.mark.parametrize("audio", [False, True])
def test_single_download_waits_for_postprocessing(tmp_path, backend, audio):
    downloader = yt.YTDownloader()
    events = []
    downloader.event_callback = lambda kind, value: events.append((kind, value))

    def processing():
        assert not downloader.last_result.files
        assert not any(kind == "summary" for kind, _ in events)

    backend["during_processing"] = processing
    result = downloader.download_urls(["https://example/video"], tmp_path, audio)
    assert result.exit_code == 0 and len(result.files) == 1
    assert Path(result.files[0]).suffix == (".mp3" if audio else ".mp4")
    assert Path(result.files[0]).is_file()
    assert any(kind == "progress" and value["percent"] == 50 for kind, value in events)
    assert any(kind == "phase" for kind, _ in events)
    if audio:
        opts = backend["options"][-1]
        assert opts["postprocessors"][0]["preferredquality"] == "192"
    else:
        assert (Path(result.directory) / "video_info.txt").exists()


def test_batch_txt_failure_codes_log_and_replay(tmp_path, backend, capsys):
    url_file = tmp_path / "my URLs.txt"
    url_file.write_text(
        "# review\nhttps://example/one\nhttps://example/fail\nhttps://example/three\n"
    )
    downloader = yt.YTDownloader()
    directory = downloader.download_from_file(url_file, tmp_path / "my output", True)
    result = downloader.last_result
    assert result.exit_code == 1 and len(result.files) == 2 and len(result.errors) == 1
    assert [Path(p).suffix for p in result.files] == [".mp3", ".mp3"]
    assert (Path(directory) / "urls.txt").read_text().splitlines() == yt.read_urls_from_file(
        url_file
    )
    log = (Path(directory) / "download_log.txt").read_text()
    assert "hidden-token" not in log
    assert "SUCCESS" in log and "Finished with failures" in log
    output = capsys.readouterr().out
    command = next(
        line.split("Equivalent CLI: ", 1)[1]
        for line in output.splitlines()
        if "Equivalent CLI:" in line
    )
    if os.name != "nt":
        argv = shlex.split(command)
        assert argv[argv.index("--file") + 1] == str(Path(directory) / "urls.txt")
    assert (
        yt.run_ytdown(
            ["--file", str(url_file), "--output", str(tmp_path / "cli"), "--audio-only", "--no-gui"]
        )
        == 1
    )


def test_cancel_during_postprocessing_preserves_completed_item(tmp_path, backend):
    downloader = yt.YTDownloader()
    backend["during_processing"] = downloader.cancel_event.set
    result = downloader.download_urls(["https://example/one", "https://example/two"], tmp_path)
    assert result.cancelled and result.exit_code == 130
    assert len(backend["calls"]) == 1 and len(result.files) == 1
    assert Path(result.files[0]).exists()


def test_cancel_before_download_and_progress_hook(tmp_path, backend):
    downloader = yt.YTDownloader()
    downloader.cancel_event.set()
    result = downloader.download_urls(["https://example/one"], tmp_path)
    assert result.cancelled and not backend["calls"]
    with pytest.raises(yt.DownloadCancelledError):
        downloader._progress_hook({"status": "downloading"})


def test_missing_final_file_is_failure(tmp_path, backend):
    backend["missing_output"] = True
    result = yt.YTDownloader().download_urls(["https://example/video"], tmp_path, True)
    assert result.exit_code == 1 and not result.files


def test_duplicate_batch_rejected(tmp_path, backend):
    downloader = yt.YTDownloader()

    def duplicate():
        with pytest.raises(RuntimeError, match="already running"):
            downloader.download_urls(["https://example/duplicate"], tmp_path)

    backend["during_processing"] = duplicate
    assert downloader.download_urls(["https://example/one"], tmp_path).exit_code == 0
    assert len(backend["calls"]) == 1


def test_interactive_cli_respects_audio(tmp_path, backend, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda: "https://example/audio")
    assert yt.run_ytdown(["--no-gui", "--audio-only", "--output", str(tmp_path)]) == 0
    assert backend["options"][-1]["format"] == "bestaudio/best"


def test_ffmpeg_requirement_and_no_display(tmp_path, monkeypatch):
    monkeypatch.setattr(yt.YTDownloader, "_check_ffmpeg", lambda self: False)
    result = yt.YTDownloader().download_urls(["https://example/no-network"], tmp_path)
    assert result.exit_code == 1
    env = dict(os.environ, DISPLAY="")
    for args in (["--help"], ["--file", str(tmp_path / "missing.txt"), "--no-gui"]):
        proc = subprocess.run(
            [sys.executable, "-m", "vaila.vaila_ytdown", *args],
            env=env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == (0 if "--help" in args else 1), proc.stderr
        assert "TclError" not in proc.stderr


def test_feedback_redacts_and_debug_traceback_only(tmp_path, capsys):
    feedback = Feedback("test", log_file=tmp_path / "log.txt")
    try:
        raise ValueError("password=secret cookie=session token=token-value")
    except ValueError as error:
        feedback.error(error)
    output = capsys.readouterr().out
    assert "secret" not in output and "session" not in output and "token-value" not in output
    assert "Traceback" not in output
    assert "<redacted>" in redact("https://user:pass@example.com?token=secret")
    feedback.debug_enabled = True
    try:
        raise ValueError("diagnostic")
    except ValueError as error:
        feedback.error(error)
    assert "Traceback" in capsys.readouterr().out


def pump(root, condition=lambda: False, timeout=3):
    import time

    deadline = time.monotonic() + timeout

    def tick():
        if condition() or time.monotonic() > deadline:
            root.quit()
        else:
            root.after(20, tick)

    root.after(20, tick)
    root.mainloop()


@pytest.fixture(params=["standalone", "integrated"])
def gui_window(request, monkeypatch):
    import importlib.util
    import tkinter as tk

    try:
        if request.param == "integrated":
            spec = importlib.util.spec_from_file_location(
                "vaila_main_smoke", Path(__file__).parents[1] / "vaila.py"
            )
            main = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(main)
            monkeypatch.setattr(main.Vaila, "_start_update_check", lambda *args, **kwargs: None)
            parent = main.Vaila()
            window = tk.Toplevel(parent)
            window.transient(parent)
        else:
            parent = window = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f"Display unavailable: {error}")
    callback_errors = []
    parent.report_callback_exception = lambda *args: callback_errors.append(args)
    yield parent, window, request.param
    with contextlib.suppress(tk.TclError):
        parent.destroy()
    assert not callback_errors, callback_errors


def capture_window(window, name):
    artifact_dir = os.environ.get("VAILA_GUI_ARTIFACTS")
    if not artifact_dir:
        return
    from PIL import ImageGrab

    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    x, y = window.winfo_rootx(), window.winfo_rooty()
    ImageGrab.grab(bbox=(x, y, x + window.winfo_width(), y + window.winfo_height())).save(
        directory / f"{name}.png"
    )


def test_downloader_gui_responsive_cancel_and_help(gui_window, tmp_path, backend, monkeypatch):
    import threading

    parent, window, mode = gui_window
    app = yt.DownloaderGUI(window)
    entered, release = threading.Event(), threading.Event()

    def processing():
        entered.set()
        release.wait(5)

    backend["during_processing"] = processing
    url_file = tmp_path / "review.txt"
    url_file.write_text("https://example/one\nhttps://example/two\n")
    monkeypatch.setattr(yt.filedialog, "askopenfilename", lambda **kwargs: str(url_file))
    app.load_txt()
    assert len(yt.parse_urls(app.urls_text.get("1.0", "end"))) == 2
    assert not backend["calls"]  # Loading only fills the editable list.
    app.output_dir_var.set(str(tmp_path / "outputs"))
    app.audio_only.set(True)
    app.details_visible.set(True)
    app.toggle_details()
    window.geometry("900x750")
    window.lift()
    app.urls_text.focus_force()
    pump(parent, timeout=0.15)
    assert window.focus_get() == app.urls_text
    opened = []
    monkeypatch.setattr(yt.webbrowser, "open_new_tab", lambda value: opened.append(value))
    app.show_help()
    assert opened[0].endswith("/help/vaila_ytdown.html") and opened[0].startswith("file:")
    assert app.start_download()
    assert app.start_download() is False
    pump(parent, entered.is_set)
    assert entered.is_set()
    heartbeat = []
    parent.after(25, lambda: heartbeat.append(True))
    pump(parent, lambda: bool(heartbeat))
    assert heartbeat and app.task.busy
    assert "Success: 0" in app.counts.cget("text")
    app.cancel()
    assert "Cancellation requested" in app.status.cget("text")
    pump(parent, timeout=0.15)
    capture_window(window, f"downloader-{mode}")
    release.set()
    pump(parent, lambda: not app.task.busy)
    assert not app.task.busy
    assert "Cancelled" in app.status.cget("text")
    assert "Success: 1" in app.counts.cget("text")
    assert len(backend["calls"]) == 1
    app.close()


def test_filemanager_gui_same_operations_as_cli(gui_window, tmp_path, monkeypatch):
    from vaila import filemanager as fm
    from vaila.filemanager_gui import FileManagerGUI

    parent, window, mode = gui_window
    app = FileManagerGUI(window, "copy")
    source = tmp_path / "source with spaces"
    source.mkdir()
    (source / "trial.csv").write_text("identical content")
    destination = tmp_path / "GUI output"
    for key, value in (
        ("source", str(source)),
        ("destination", str(destination)),
        ("extension", ".csv"),
    ):
        app.fields[key].insert(0, value)
    window.geometry("900x700")
    window.lift()
    app.fields["source"].focus_force()
    pump(parent, timeout=0.15)
    assert window.focus_get() == app.fields["source"]
    opened = []
    monkeypatch.setattr(
        "vaila.filemanager_gui.webbrowser.open_new_tab", lambda value: opened.append(value)
    )
    app.help()
    assert opened[0].endswith("/help/filemanager.html")
    app.preview()
    pump(parent, lambda: not app.task.busy and not app.rendering and app.plan is not None)
    assert app.plan is not None and len(app.plan.targets) == 1
    capture_window(window, f"filemanager-{mode}")
    app.apply()
    pump(parent, lambda: not app.task.busy)
    assert "Completed: 1" in app.status.cget("text")
    cli_dest = tmp_path / "CLI output"
    assert (
        fm.main(
            ["copy", "--source", str(source), "--destination", str(cli_dest), "--extension", ".csv"]
        )
        == 0
    )
    assert [p.read_text() for p in destination.rglob("*.csv")] == [
        p.read_text() for p in cli_dest.rglob("*.csv")
    ]
    app.close()


def test_feedback_redacts_bearer_and_multi_cookie_headers():
    assert "bearer-value" not in redact("Authorization: Bearer bearer-value")
    assert "two" not in redact("Cookie: first=one; second=two")
    assert "two" not in redact('password="one two"')


def test_transfer_gui_waits_for_actual_status(gui_window, tmp_path, monkeypatch):
    if sys.platform != "linux":
        pytest.skip("Linux terminal launch branch")
    from vaila import filemanager as fm
    from vaila.filemanager_gui import FileManagerGUI

    parent, window, mode = gui_window
    app = FileManagerGUI(window, "transfer")
    local = tmp_path / "my local data"
    local.mkdir()
    for key, value in (
        ("local", str(local)),
        ("host", "server.example"),
        ("user", "analyst"),
        ("remote", "/data/remote folder"),
    ):
        app.fields[key].insert(0, value)
    calls = []
    monkeypatch.setattr(
        fm.shutil,
        "which",
        lambda name: "/usr/bin/" + name if name in ("rsync", "gnome-terminal") else None,
    )
    monkeypatch.setattr(
        "vaila.filemanager_gui.subprocess.Popen", lambda argv, **kwargs: calls.append(argv)
    )
    app.preview()
    assert len(calls) == 1
    assert app.transfer_status is not None
    assert "waiting for transfer result" in app.status.cget("text")
    script = (app.transfer_dir / "transfer.sh").read_text()
    assert "vaila.filemanager transfer" in script and "--status-file" in script
    window.lift()
    pump(parent, timeout=0.15)
    capture_window(window, f"transfer-{mode}")
    app.transfer_status.write_text("23")
    pump(parent, lambda: app.transfer_status is None)
    assert "failed (exit 23)" in app.status.cget("text")
    app.preview()
    app.transfer_status.write_text("0")
    pump(parent, lambda: app.transfer_status is None)
    assert app.status.cget("text") == "Transfer completed"
    app.close()


def test_main_filemanager_buttons_dispatch_to_shared_operations(monkeypatch):
    import importlib.util

    from vaila import filemanager as fm

    spec = importlib.util.spec_from_file_location(
        "vaila_main_dispatch", Path(__file__).parents[1] / "vaila.py"
    )
    main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main)
    actions = []
    monkeypatch.setattr(fm, "show_action", actions.append)
    for method in (
        "copy_file",
        "move_file",
        "remove_file",
        "rename_files",
        "import_file",
        "export_file",
        "tree_file",
        "find_file",
        "transfer_file",
    ):
        getattr(main.Vaila, method)(object())
    assert actions == [
        "copy",
        "move",
        "remove",
        "normalize",
        "import-vicon",
        "export",
        "tree",
        "find",
        "transfer",
    ]
