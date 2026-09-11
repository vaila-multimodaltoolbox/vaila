"""Tkinter parameter collection for File Manager.

Version: 0.3.137
Update Date: 11 September 2026
"""

import json
import subprocess
import sys
import tempfile
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

try:
    from . import filemanager as operations
    from .task_feedback import Feedback, WorkerTask, command_text
except ImportError:
    import filemanager as operations
    from task_feedback import Feedback, WorkerTask, command_text


class FileManagerGUI:
    def __init__(self, root, action):
        self.root, self.action = root, action
        self.task, self.plan, self.pending_args = WorkerTask(), None, None
        self.feedback = Feedback("filemanager", callback=lambda msg: self.task.emit("log", msg))
        self.fields = {}
        self.rendering = False
        root.title(f"vailá File Manager - {action}")
        root.geometry("860x680")
        root.minsize(660, 500)
        frame = ttk.Frame(root, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text=f"{action.upper()} - choose parameters, then preview",
            font=("TkDefaultFont", 13, "bold"),
        ).pack(anchor="w")
        if action == "import-vicon":
            ttk.Label(
                frame,
                text="VICON Nexus CSV: available (first level only).\n"
                "Qualysis, MATLAB, HTML, XML, XLSX, BVH, FBX and TRC: unavailable.",
            ).pack(anchor="w")
        fields = [("source", "Source file" if action == "export" else "Source directory", "")]
        if action in ("copy", "move", "find", "tree", "export", "import-vicon"):
            fields.append(("destination", "Destination directory", ""))
        if action in ("copy", "move", "rename", "find", "tree"):
            fields.append(("extension", "Extension (.csv; blank = all)", ""))
        if action in ("copy", "move", "find", "remove"):
            fields.append(("pattern", "Pattern (one per line)", ""))
        if action == "remove":
            fields.append(("removal-type", "Match type: ext / name / dir", "ext"))
        if action == "rename":
            fields.extend(
                [("text", "Text to replace", ""), ("replacement", "Replacement (may be empty)", "")]
            )
        if action == "transfer":
            fields = [
                ("local", "Local directory", ""),
                ("host", "SSH host", ""),
                ("user", "SSH user", ""),
                ("remote", "Remote directory (absolute)", ""),
                ("port", "SSH port", "22"),
                ("mode", "Direction: upload / download", "upload"),
            ]
        for name, label, value in fields:
            row = ttk.Frame(frame)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=label, width=34).pack(side="left")
            if name == "pattern":
                widget = tk.Text(row, height=2, width=30)
                widget.insert("1.0", value)
            elif name in ("mode", "removal-type"):
                widget = ttk.Combobox(
                    row,
                    values=("upload", "download") if name == "mode" else ("ext", "name", "dir"),
                    state="readonly",
                )
                widget.set(value)
            else:
                widget = ttk.Entry(row)
                widget.insert(0, value)
            widget.pack(side="left", fill="x", expand=True)
            self.fields[name] = widget
            if name in ("source", "destination", "local"):
                ttk.Button(row, text="Browse...", command=lambda key=name: self.browse(key)).pack(
                    side="left"
                )
        explanation = {
            "copy": "Recursive, case-sensitive suffix + literal substring. No wildcards. Blank = all.",
            "move": "Recursive suffix + literal substring. First matching pattern owns each file.",
            "remove": "Permanent removal. ext: suffix (.csv); name: glob (*backup*); dir: substring (backup).",
            "find": "Recursive glob fragments: trial* with extension .csv. Includes matching directories.",
            "normalize": "Files and folders: lowercase, remove accents/special characters, spaces -> underscores.",
            "transfer": "SSH authentication and live progress open in an interactive terminal.",
        }.get(
            action, "Review targets before applying. Existing output files are never overwritten."
        )
        ttk.Label(frame, text=explanation, wraplength=780).pack(anchor="w", pady=6)
        self.debug = tk.BooleanVar(root, value=False)
        ttk.Checkbutton(frame, text="Diagnostic details (--debug)", variable=self.debug).pack(
            anchor="w"
        )
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=6)
        self.preview_button = ttk.Button(
            buttons,
            text="Open terminal" if action == "transfer" else "Preview",
            command=self.preview,
        )
        self.preview_button.pack(side="left")
        self.run_button = ttk.Button(
            buttons, text="Apply preview", command=self.apply, state="disabled"
        )
        if action != "transfer":
            self.run_button.pack(side="left", padx=6)
        ttk.Button(buttons, text="Cancel / Close", command=self.close).pack(side="left")
        ttk.Button(buttons, text="Help", command=self.help).pack(side="right")
        self.status = ttk.Label(frame, text="Waiting for input")
        self.status.pack(anchor="w")
        self.output = ScrolledText(frame, height=10, state="disabled", wrap="word")
        self.output.pack(fill="both", expand=True)
        self.transfer_status = self.transfer_dir = None
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda event: self.close())
        next(iter(self.fields.values())).focus_set()
        self.feedback(f"Opened action: {action}. Waiting for input.")
        self.poll_id = root.after(75, self.poll)

    def help(self):
        webbrowser.open_new_tab((Path(__file__).parent / "help/filemanager.html").as_uri())

    def browse(self, key):
        choose = (
            filedialog.askopenfilename
            if key == "source" and self.action == "export"
            else filedialog.askdirectory
        )
        value = choose(parent=self.root, title=f"Select {key}")
        if value:
            self.fields[key].delete(0, "end")
            self.fields[key].insert(0, value)
        else:
            self.feedback("Selection cancelled; waiting for input.")

    def arguments(self):
        argv = [self.action]
        for key, widget in self.fields.items():
            if key == "pattern":
                for value in widget.get("1.0", "end").splitlines():
                    if value.strip():
                        argv.extend(["--pattern", value.strip()])
            else:
                value = widget.get()
                if key in ("source", "destination", "local") and not value.strip():
                    raise ValueError(f"{key.capitalize()} is required")
                # --key=value also safely handles replacement text starting with '-'.
                argv.append(f"--{key}={value}")
        if self.debug.get():
            argv.append("--debug")
        return argv

    def preview(self):
        if self.task.busy or self.transfer_status or self.rendering:
            return
        self.plan = None
        self.run_button.configure(state="disabled")
        self.feedback.debug_enabled = self.debug.get()
        try:
            argv = self.arguments()
            args = operations.build_parser().parse_args(argv)
            self.pending_args = argv
            if self.action == "transfer":
                operations.build_transfer_command(
                    args.local, args.host, args.user, args.remote, args.port, args.mode
                )
                self.open_terminal(argv)
                return
            self.feedback(
                "Equivalent CLI: "
                + command_text([sys.executable, "-m", "vaila.filemanager", *argv, "--dry-run"])
            )
            self.preview_button.configure(state="disabled")
            self.status.configure(text="Searching...")
            self.task.start(
                lambda: operations.build_plan(
                    **operations._plan_args(args), feedback=self.feedback, cancel=self.task.cancel
                )
            )
        except (Exception, SystemExit) as error:
            self.feedback.error(error)
            self.status.configure(text="Invalid parameters; see details below.")

    def apply(self):
        if self.task.busy or self.rendering or not self.plan:
            return
        try:
            unchanged = self.arguments() == self.pending_args
        except ValueError:
            unchanged = False
        if not unchanged:
            self.feedback("Parameters changed. Preview again before applying.")
            self.run_button.configure(state="disabled")
            return
        if self.action in ("remove", "move", "rename", "normalize"):
            label = "PERMANENT removal" if self.action == "remove" else self.action
            self.feedback("Waiting for confirmation of the displayed preview.")
            if not messagebox.askyesno(
                "Confirm preview",
                f"Apply {label} to exactly {len(self.plan.targets)} displayed targets?",
                parent=self.root,
            ):
                self.feedback("Cancelled; no files changed.")
                return
        argv = [sys.executable, "-m", "vaila.filemanager", *self.pending_args]
        if self.action == "remove":
            argv.append("--dry-run")
        self.feedback("Equivalent CLI: " + command_text(argv))
        self.run_button.configure(state="disabled")
        self.preview_button.configure(state="disabled")
        self.status.configure(text="Working...")
        plan, self.plan = self.plan, None
        self.task.start(
            lambda: operations.execute_plan(plan, cancel=self.task.cancel, feedback=self.feedback)
        )

    def open_terminal(self, argv):
        command = [sys.executable, "-m", "vaila.filemanager", *argv]
        self.feedback("Equivalent CLI: " + command_text(command))
        directory = Path(tempfile.mkdtemp(prefix="vaila_transfer_"))
        status = directory / "status.txt"
        command.extend(["--status-file", str(status)])
        project = Path(__file__).resolve().parent.parent
        try:
            if sys.platform == "win32":
                subprocess.Popen(command, cwd=project, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                script = directory / "transfer.sh"
                script.write_text(
                    "#!/bin/sh\ncd "
                    + command_text([str(project)])
                    + "\n"
                    + command_text(command)
                    + '\nprintf "Press Enter to close..."\nread answer\n',
                    encoding="utf-8",
                )
                if sys.platform == "darwin":
                    subprocess.run(
                        [
                            "osascript",
                            "-e",
                            'tell application "Terminal" to do script '
                            + json.dumps(command_text(["sh", str(script)])),
                        ],
                        check=True,
                    )
                else:
                    terminal = next(
                        (
                            name
                            for name in (
                                "gnome-terminal",
                                "konsole",
                                "xfce4-terminal",
                                "x-terminal-emulator",
                                "xterm",
                            )
                            if operations.shutil.which(name)
                        ),
                        None,
                    )
                    if not terminal:
                        raise OSError("No terminal emulator found; run the printed CLI command")
                    args = (
                        [terminal, "--", "sh", str(script)]
                        if terminal == "gnome-terminal"
                        else [terminal, "-e", "sh", str(script)]
                    )
                    if terminal == "xfce4-terminal":
                        args = [terminal, "-e", command_text(["sh", str(script)])]
                    subprocess.Popen(args)
            self.transfer_dir, self.transfer_status = directory, status
            self.preview_button.configure(state="disabled")
            self.feedback(
                "Terminal opened; transfer completion is not yet known. Follow progress there."
            )
            self.status.configure(text="Terminal opened - waiting for transfer result")
        except Exception:
            operations.shutil.rmtree(directory)
            raise

    def poll(self):
        for kind, payload in self.task.drain():
            if kind == "log":
                self.output.configure(state="normal")
                self.output.insert("end", payload + "\n")
                self.output.see("end")
                self.output.configure(state="disabled")
            elif kind == "result":
                if isinstance(payload, operations.OperationPlan):
                    self.plan = payload
                    self.preview_rows = iter(payload.targets)
                    self.rendering = True
                    self.status.configure(text=f"Rendering preview: {len(payload.targets)} targets")
                    self.root.after(1, self.render_preview)
                else:
                    self.status.configure(text=payload.summary())
            elif kind == "error":
                self.feedback.error(payload)
                self.status.configure(text=str(payload))
            elif kind == "done" and not self.rendering:
                self.preview_button.configure(state="normal")
        if self.transfer_status and self.transfer_status.exists():
            value = self.transfer_status.read_text(encoding="utf-8").strip()
            if value:
                message = (
                    "Transfer completed" if value == "0" else f"Transfer failed (exit {value})"
                )
                self.status.configure(text=message)
                self.feedback(message)
                operations.shutil.rmtree(self.transfer_dir)
                self.transfer_dir = self.transfer_status = None
                self.preview_button.configure(state="normal")
        self.poll_id = self.root.after(75, self.poll)

    def render_preview(self):
        if self.task.cancel.is_set():
            self.plan = None
            self.rendering = False
            self.preview_button.configure(state="normal")
            self.status.configure(text="Preview cancelled; no files changed")
            return
        self.output.configure(state="normal")
        for _ in range(100):
            target = next(self.preview_rows, None)
            if target is None:
                self.output.configure(state="disabled")
                self.rendering = False
                self.preview_button.configure(state="normal")
                self.run_button.configure(
                    state="normal" if self.plan.targets or self.plan.report else "disabled"
                )
                self.status.configure(text=f"Preview ready: {len(self.plan.targets)} targets")
                return
            self.output.insert(
                "end",
                f"{target.source}"
                + (f" -> {target.destination}" if target.destination else "")
                + "\n",
            )
            # Full target preview also goes to the terminal.
            print(
                f">> vaila/filemanager: Preview: {target.source}"
                + (f" -> {target.destination}" if target.destination else ""),
                flush=True,
            )
        self.output.configure(state="disabled")
        self.root.after(1, self.render_preview)

    def close(self):
        if self.task.busy or self.rendering:
            self.task.cancel.set()
            self.feedback("Cancellation requested; waiting for the current step to finish.")
            self.status.configure(text="Cancellation requested; close after the worker stops")
            return
        if self.transfer_status:
            self.feedback("Transfer result is unknown. Check its terminal.")
            if not messagebox.askyesno(
                "Transfer status unknown",
                "Close this window? The terminal may still be transferring. Its final result remains in the terminal.",
                parent=self.root,
            ):
                return
            self.feedback(f"Transfer status file retained: {self.transfer_status}")
        self.feedback("Closed / cancelled. No further operations scheduled.")
        self.root.after_cancel(self.poll_id)
        self.root.destroy()


def show_action(action):
    parent = tk._default_root
    root = tk.Toplevel(parent) if parent else tk.Tk()
    if parent:
        root.transient(parent)
    root.app = FileManagerGUI(root, action)
    if parent:
        parent.wait_window(root)
    else:
        root.mainloop()
