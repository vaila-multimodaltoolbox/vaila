# File Manager Buttons & Hybrid SSH Transfer GUI-Terminal Integration

This skill documents the patterns and fixes for maintaining cross-platform compatibility, process safety, and proper interactive behavior for the File Manager utilities (Rename, Import, Export, Copy, Move, Remove, Tree, Find, and SSH Transfer).

## vailá maintenance rule (version/date)

When you edit `vaila/filemanager.py` or any `*.py` in this repo, also update:

- Edited script header **Update Date** (today) + **Version** (global, from `vaila.py` header/banner)
- Root `README.md` line `Last updated: YYYY-MM-DD`
- Help docs: matching `vaila/help/<module>.md` + `.html`, plus `vaila/help/index.md` + `.html` (“Generated on”)
- Installers / `vaila.py` if change impacts install/run UX

Reference checklist: `AGENTS.md` (“Mandatory: Update metadata on any script change”).

---

## Technical Challenges & Standard Patterns

### 1. The Duplicate Root Window Bug (`tk.Tk()` vs `tk.Toplevel`)
**Issue**: Creating a new root window using `tk.Tk()` and calling `mainloop()` in a lazily-imported module when the main `vaila.py` GUI is already running causes thread/state conflicts in Tcl/Tk. This leads to hangs, focus issues, or Segmentation Faults.
**Solution**: Implement the hybrid/dual window management pattern. Detect if a root window already exists in the process, and if so, spawn a transient modal `Toplevel` dialog and block using `wait_window()`. If no root exists (standalone execution), fall back to `tk.Tk()` and `mainloop()`.

```python
# Get existing root window
parent_root = None
if hasattr(tk, "_default_root") and tk._default_root is not None:
    parent_root = tk._default_root

# Create the window
if parent_root:
    dialog = tk.Toplevel(parent_root)
    dialog.transient(parent_root)
    dialog.grab_set()  # Make it modal
else:
    dialog = tk.Tk()

# ... construct widgets packed into dialog ...

# Block execution until closed
if parent_root:
    parent_root.wait_window(dialog)
else:
    dialog.mainloop()
```
*Applied to: `copy_file()`, `move_file()`, and `import_file()` inside `vaila/filemanager.py`.*

### 2. Tcl/Tk Segmentation Fault on Function Return (Exit Code 139)
**Issue**: When a function creating a non-blocking `Toplevel` dialog returns, Python garbage-collects all local variables in its scope (including `StringVar` instances). The `StringVar.__del__` hook unregisters Tcl variables. However, the UI widgets (like `tk.Entry` fields) are still visible on-screen and bound to those Tcl variables. In the next idle cycle, the Tcl/Tk event loop attempts to render the entry fields, encounters a NULL pointer reference to the deleted Tcl variables, and triggers a Segmentation Fault (exit code 139).
**Solution**:
- Block execution using `parent_root.wait_window(dialog)`. This keeps the local call stack frame active, preventing variables from going out of scope while the window is open.
- Explicitly bind `StringVar` and widget references to the `Toplevel` window object as attributes to ensure their lifecycle matches the window.

```python
# Prevent garbage collection of variables by binding them to the window object
dialog.local_dir_var = local_dir_var
dialog.remote_host_var = remote_host_var
# ...
```

### 3. SSH Password Prompt & Terminal Integration
**Issue**: Commands like `rsync` and `scp` require a real interactive terminal (TTY) to prompt for and read the SSH password. They cannot receive password input programmatically through GUI text boxes easily without introducing insecure dependencies like raw password entry boxes (which are prohibited or unsafe).
**Solution**: Use a hybrid GUI-Terminal design:
1. Build a clean Tkinter `Toplevel` dialog to collect all parameter variables (paths, username, host, port).
2. Validate user inputs in Python (e.g. verify local paths exist).
3. Generate a temporary executable bash script containing the pre-assembled `rsync` or `scp` command.
4. Launch the script in a new terminal emulator window (`gnome-terminal`, `konsole`, `xfce4-terminal`, or `xterm` on Linux; `Terminal.app` on macOS; `cmd` on Windows) and auto-close the GUI dialog. The user only needs to type their password in the terminal.

```python
# Launch script in a terminal window (Linux example)
terminals = [
    ("gnome-terminal", ["--", "bash", script_path]),
    ("konsole", ["-e", "bash", script_path]),
    ("xfce4-terminal", ["-e", f"bash {script_path}"]),
    ("x-terminal-emulator", ["-e", f"bash {script_path}"]),
    ("xterm", ["-hold", "-e", "bash", script_path]),
]
for tname, targs in terminals:
    if shutil.which(tname):
        subprocess.Popen([tname, *targs])
        break
```

### 4. Whitespace Trimming on Inputs
**Issue**: Copy-pasting paths or usernames frequently introduces trailing whitespace (e.g. `/mnt/disco2tb1/Downloads   `), causing `rsync`/`scp` to fail with "No such file or directory" or "Permission denied".
**Solution**: Trim whitespace on all collected paths and parameters.
- In Python: `.strip()` on string variables.
- In Shell Scripts: use `xargs` to trim leading/trailing spaces.
```bash
LOCAL_DIR=$(echo "$LOCAL_DIR" | xargs)
```
