"""Terminal feedback and worker events for file/download tools.

Version: 0.3.137
Update Date: 11 September 2026
"""

import queue
import re
import shlex
import subprocess
import sys
import threading
import traceback
from pathlib import Path


def command_text(argv):
    """Quote for the native command shell (POSIX shell or Windows cmd)."""
    return subprocess.list2cmdline(argv) if sys.platform == "win32" else shlex.join(argv)


def redact(text):
    text = str(text)
    text = re.sub(r"(https?://)[^/@\s]+:[^/@\s]+@", r"\1<redacted>@", text)
    text = re.sub(r"(?i)(authorization[=:\s]+)(?:Bearer|Basic)\s+[^\s]+", r"\1<redacted>", text)
    text = re.sub(r"(?i)(cookies?\s*[:=]\s*)[^\n]+", r"\1<redacted>", text)
    return re.sub(
        r"""(?i)(password|token|secret|signature|sig)([=:\s]+)("[^"]*"|'[^']*'|[^\s&]+)""",
        r"\1\2<redacted>",
        text,
    )


class Feedback:
    def __init__(self, module, debug=False, log_file=None, callback=None):
        self.module = module
        self.debug_enabled = debug
        self.log_file = Path(log_file) if log_file else None
        self.callback = callback

    def __call__(self, message):
        line = f">> vaila/{self.module}: {redact(message)}"
        print(line, flush=True)
        if self.log_file:
            try:
                with self.log_file.open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
            except OSError:
                self.log_file = None
                raise
        if self.callback:
            self.callback(line)

    def debug(self, message):
        if self.debug_enabled:
            self(message)

    def error(self, error):
        self(f"Failed: {error}")
        self.debug("".join(traceback.format_exception(error)))


class WorkerTask:
    """Workers produce events only; a Tk after callback consumes them."""

    def __init__(self):
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.thread = None
        self.busy = False

    def emit(self, kind, payload):
        self.events.put((kind, payload))

    def start(self, function):
        if self.busy:
            return False
        self.busy = True
        self.cancel.clear()

        def run():
            try:
                self.emit("result", function())
            except Exception as error:
                self.emit("error", error)
            finally:
                self.emit("done", None)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return True

    def drain(self):
        # Bound each GUI polling turn so input and repaint events keep running.
        for _ in range(100):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                return
            if event[0] == "done":
                self.busy = False
            yield event
