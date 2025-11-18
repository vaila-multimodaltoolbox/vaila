"""
================================================================================
Native File Dialog Module - native_file_dialog.py
================================================================================
vailá - Multimodal Toolbox
Author: Prof. Dr. Paulo R. P. Santiago
https://github.com/paulopreto/vaila-multimodaltoolbox
Created: 17 November 2025
Updated: 17 November 2025
Version: 0.2.0
Python Version: 3.12.12

Description:
------------
This module provides native file dialogs that work without conflicts with Pygame.
It uses system-native dialogs via subprocess to avoid event loop conflicts.

Usage:
------
from native_file_dialog import open_native_file_dialog, open_yes_no_dialog

# Open file dialog
file_path = open_native_file_dialog(
    title="Select File",
    file_types=[("*.csv", "CSV Files"), ("*.txt", "Text Files")]
)

# Open yes/no dialog
result = open_yes_no_dialog(
    title="Question",
    message="Do you want to continue?"
)

================================================================================
"""

import os
import subprocess
import sys
from pathlib import Path


def terminal_file_browser(initial_dir=None, file_types=None):
    """
    Minimal terminal-based file browser used as a last-resort fallback.

    Args:
        initial_dir (str | Path): Starting directory.
        file_types (list[tuple[str, str]]): Patterns such as [("*.csv", "CSV Files")].

    Returns:
        str | None: Selected file path or None if cancelled.
    """
    file_types = file_types or [("*.*", "All Files")]

    def matches(path: Path) -> bool:
        if not file_types:
            return True
        name = path.name.lower()
        for pattern, _ in file_types:
            pattern = pattern.strip()
            if pattern == "*.*" or pattern == "*":
                return True
            if pattern.startswith("*."):
                ext = pattern[1:].lower()
                if name.endswith(ext):
                    return True
            elif pattern == name:
                return True
        return False

    current_dir = Path(initial_dir or Path.home()).expanduser().resolve()

    while True:
        try:
            entries = []
            dir_entries = []
            file_entries = []

            for item in sorted(
                current_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
            ):
                if item.is_dir():
                    dir_entries.append(item)
                elif matches(item):
                    file_entries.append(item)

            entries.extend(dir_entries)
            entries.extend(file_entries)
        except PermissionError:
            print(f"[native_file_dialog] Permission denied: {current_dir}")
            current_dir = current_dir.parent
            continue
        except FileNotFoundError:
            print(f"[native_file_dialog] Directory not found: {current_dir}")
            current_dir = Path.home()
            continue

        print("\n" + "=" * 80)
        print(f"Directory: {current_dir}")
        print("=" * 80)
        print("Use the index to select items. Options:")
        print("  [number] -> open directory / select file")
        print("  ..       -> go to parent directory")
        print("  /path    -> jump to absolute path")
        print("  q        -> cancel\n")

        if not entries:
            print("(Empty directory)")
        else:
            for idx, entry in enumerate(entries, start=1):
                marker = "/" if entry.is_dir() else ""
                print(f"{idx:3d}. {entry.name}{marker}")

        choice = input("\nSelection: ").strip()

        if not choice:
            continue
        if choice.lower() in {"q", "quit", "exit"}:
            return None
        if choice == "..":
            parent = current_dir.parent
            if parent != current_dir:
                current_dir = parent
            continue
        if choice.startswith("~") or choice.startswith("/"):
            target = Path(choice).expanduser()
            if target.is_dir():
                current_dir = target.resolve()
                continue
            if target.is_file() and matches(target):
                return str(target.resolve())
            print("Invalid path or file does not match required extensions.")
            continue
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(entries):
                selected = entries[idx]
                if selected.is_dir():
                    current_dir = selected.resolve()
                elif selected.is_file():
                    return str(selected.resolve())
            else:
                print("Invalid index.")
            continue

        print("Invalid selection. Please use the provided commands or indexes.")


def open_native_file_dialog(
    title="Select File",
    file_types=None,
    initial_dir=None,
    prefer_terminal=False,
):
    """
    Open native file dialog without blocking Pygame.

    Args:
        title (str): Dialog title
        file_types (list): List of tuples like [("*.csv", "CSV Files")]

    Returns:
        str: Selected file path or None if cancelled
    """
    if file_types is None:
        file_types = [("*.*", "All Files")]

    try:
        env_force_terminal = os.environ.get("VAILA_FORCE_TERMINAL_DIALOG", "").lower() in {
            "1",
            "true",
            "yes",
        }
        if prefer_terminal or env_force_terminal:
            return terminal_file_browser(initial_dir, file_types)

        if sys.platform == "win32":
            # Windows - PowerShell
            filters = "|".join([f"{desc} ({pat})|{pat}" for pat, desc in file_types])
            command = (
                'powershell -Command "'
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$dlg = New-Object System.Windows.Forms.OpenFileDialog; "
                f"$dlg.Title = '{title}'; "
                f"$dlg.Filter = '{filters}'; "
                "if ($dlg.ShowDialog() -eq 'OK') { $dlg.FileName }"
                '"'
            )
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            path = result.stdout.strip()
            return path if path else None

        elif sys.platform == "darwin":
            # macOS - osascript
            extensions = ",".join([pat.replace("*.", "") for pat, _ in file_types])
            script = f"""
            set theFile to choose file with prompt "{title}" of type {{{extensions}}}
            return POSIX path of theFile
            """
            result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
            path = result.stdout.strip()
            return path if path else None

        else:
            # Linux - zenity (needs to be installed)
            # Check if zenity is available
            try:
                subprocess.run(["zenity", "--version"], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                # zenity not available, return None to trigger tkinter fallback
                return terminal_file_browser(initial_dir, file_types)

            # Build file filter arguments for zenity
            # zenity format: --file-filter=Description|*.ext1 *.ext2
            filter_args = []
            if file_types:
                for pattern, desc in file_types:
                    ext = pattern.replace("*.", "").replace("*", "")
                    if not ext:
                        continue
                    variants = [f"*.{ext.lower()}"]
                    if ext.lower() != ext.upper():
                        variants.append(f"*.{ext.upper()}")
                    filter_args.append(f"{desc}|{' '.join(variants)}")
                # Always append an "All Files" option for safety
                filter_args.append("All Files|*")

            command = ["zenity", "--file-selection", "--title", title]
            for filt in filter_args:
                command.append(f"--file-filter={filt}")

            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    path = result.stdout.strip()
                    return path if path else None
                else:
                    # User cancelled or error occurred
                    return None
            except subprocess.TimeoutExpired:
                print("File dialog timed out")
                return terminal_file_browser(initial_dir, file_types)
            except Exception as e:
                print(f"Error with zenity: {e}")
                return terminal_file_browser(initial_dir, file_types)

    except Exception as e:
        print(f"Error opening dialog: {e}")
        return None


def open_yes_no_dialog(title="Question", message="Continue?"):
    """
    Open native yes/no dialog.

    Args:
        title (str): Dialog title
        message (str): Dialog message

    Returns:
        bool: True for Yes, False for No
    """
    try:
        if sys.platform == "win32":
            command = (
                'powershell -Command "'
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$result = [System.Windows.Forms.MessageBox]::Show("
                f"'{message}', '{title}', 'YesNo', 'Question'); "
                "if ($result -eq 'Yes') { 'yes' }"
                '"'
            )
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            return "yes" in result.stdout.strip().lower()

        elif sys.platform == "darwin":
            script = f"""
            display dialog "{message}" with title "{title}" buttons {{"No", "Yes"}} default button "Yes"
            if button returned of result is "Yes" then return "yes" else return "no"
            """
            result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
            return "yes" in result.stdout.strip().lower()

        else:
            command = f'zenity --question --title="{title}" --text="{message}"'
            result = subprocess.run(command, shell=True)
            return result.returncode == 0

    except Exception as e:
        print(f"Error in dialog: {e}")
        return False


def open_save_file_dialog(title="Save File", file_types=None, default_name=""):
    """
    Open native save file dialog.

    Args:
        title (str): Dialog title
        file_types (list): List of tuples like [("*.csv", "CSV Files")]
        default_name (str): Default filename

    Returns:
        str: Selected file path or None if cancelled
    """
    if file_types is None:
        file_types = [("*.*", "All Files")]

    try:
        if sys.platform == "win32":
            # Windows - PowerShell
            filters = "|".join([f"{desc} ({pat})|{pat}" for pat, desc in file_types])
            command = (
                'powershell -Command "'
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$dlg = New-Object System.Windows.Forms.SaveFileDialog; "
                f"$dlg.Title = '{title}'; "
                f"$dlg.Filter = '{filters}'; "
                f"$dlg.FileName = '{default_name}'; "
                "if ($dlg.ShowDialog() -eq 'OK') { $dlg.FileName }"
                '"'
            )
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            path = result.stdout.strip()
            return path if path else None

        elif sys.platform == "darwin":
            # macOS - osascript
            extensions = ",".join([pat.replace("*.", "") for pat, _ in file_types])
            script = f"""
            set theFile to choose file name with prompt "{title}" default name "{default_name}"
            return POSIX path of theFile
            """
            result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
            path = result.stdout.strip()
            return path if path else None

        else:
            # Linux - zenity
            filters = " ".join([f'--file-filter="{desc}|{pat}"' for pat, desc in file_types])
            command = f'zenity --file-selection --save --title="{title}" --filename="{default_name}" {filters}'
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            path = result.stdout.strip()
            return path if path else None

    except Exception as e:
        print(f"Error opening save dialog: {e}")
        return None
