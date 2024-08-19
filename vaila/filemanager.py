"""
File: filemanager.py

Description:
This script, named `filemanager.py`, is designed to manage files and directories efficiently. It supports various operations, including importing, exporting, copying, moving, and removing files. The script leverages the Tkinter graphical interface to facilitate user interaction, enabling the selection of files and directories through an easy-to-use GUI.

Version: 1.0
Last Updated: August 16, 2024
Author: Prof. Paulo Santiago

Main Features:
- Import specific files from a selected directory into a predefined structure.
- Export files matching specific patterns and extensions into a newly created directory.
- Copy files from one location to another within the allowed directories.
- Move files between predefined directories.
- Remove files with specific extensions from selected directories.

Usage Notes:
- A directory named 'vaila_export' will be automatically created within the chosen destination directory for exporting files.
- The script ensures that essential directories ('data', 'import', 'export', 'results') exist before performing operations.

Dependencies:
- Python 3.x
- Tkinter (for the graphical user interface)
- shutil, os, time (standard Python libraries)

How to Run:
- Execute the script in a Python environment that supports Tkinter.
- Follow on-screen prompts to perform the desired file operations.

"""

import shutil
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import time

# Ensure the 'data', 'import', 'export', and 'results' directories exist
for directory in ["data", "import", "export", "results"]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Create the directories list
directories = ["import", "export", "results"] + [
    os.path.join("data", d)
    for d in os.listdir("data")
    if os.path.isdir(os.path.join("data", d))
]


def import_file():
    allowed_extensions = {
        ".avi",
        ".AVI",
        ".mp4",
        ".MP4",
        ".mov",
        ".MOV",
        ".mkv",
        ".MKV",
        ".c3d",
        ".csv",
        ".txt",
        ".ref2d",
        ".dlt2d",
        ".2d",
        ".ref3d",
        ".3d",
        ".dlt3d",
    }

    def select_directory(title):
        directory_path = filedialog.askdirectory(title=title)
        return directory_path

    # Select source directory
    src = select_directory("Select the source directory")
    if not src:
        messagebox.showerror("Error", "No source directory selected.")
        return

    # Destination directory is always the 'import' directory in the current working directory
    dest = os.path.join(os.getcwd(), "import")

    # Ensure the destination directory exists
    os.makedirs(dest, exist_ok=True)

    # Copy all allowed files from source to destination
    for item in os.listdir(src):
        src_path = os.path.join(src, item)
        dest_path = os.path.join(dest, item)
        if os.path.isdir(src_path):
            for root_dir, dirs, files in os.walk(src_path):
                for file in files:
                    if os.path.splitext(file)[1].lower() in allowed_extensions:
                        shutil.copy2(
                            os.path.join(root_dir, file), os.path.join(dest, file)
                        )
        elif os.path.splitext(item)[1].lower() in allowed_extensions:
            shutil.copy2(src_path, dest_path)

    messagebox.showinfo(
        "Success", f"All allowed files from {src} have been imported to {dest}."
    )


def copy_file():
    # Prompt the user to select the main path directory for recursive search
    src_directory = filedialog.askdirectory(title="Select Source Directory")

    # Check if a source directory was selected; if not, show an error message
    if not src_directory:
        messagebox.showerror("Error", "No source directory selected.")
        return

    # Prompt the user to enter the file extension to search for
    file_extension = simpledialog.askstring(
        "File Extension", "Enter the file extension to export (e.g., .csv, .mp4):"
    )
    if not file_extension:
        messagebox.showerror("Error", "No file extension provided.")
        return

    # Create a new window for pattern entry
    pattern_window = tk.Tk()
    pattern_window.title("Enter File Patterns")

    # Text box for entering multiple patterns, one per line
    pattern_label = tk.Label(pattern_window, text="Enter file patterns (one per line):")
    pattern_label.pack()

    pattern_text = tk.Text(pattern_window, height=10, width=50)
    pattern_text.pack()

    def on_submit():
        patterns = pattern_text.get("1.0", "end").strip().splitlines()
        pattern_window.destroy()
        process_copy(src_directory, file_extension, patterns)

    submit_button = tk.Button(pattern_window, text="Submit", command=on_submit)
    submit_button.pack()

    pattern_window.mainloop()


def process_copy(src_directory, file_extension, patterns):
    # Prompt the user to select the destination directory where the new directories will be created
    base_dest_directory = filedialog.askdirectory(title="Select Destination Directory")
    if not base_dest_directory:
        messagebox.showerror("Error", "No destination directory selected.")
        return

    # Ensure the 'vaila_copy' directory exists within the selected destination directory
    base_dest_directory = os.path.join(base_dest_directory, "vaila_copy")
    os.makedirs(base_dest_directory, exist_ok=True)

    try:
        for file_pattern in patterns:
            # Generate a timestamp to create a unique directory name for each pattern
            timestamp = time.strftime("%Y%m%d%H%M%S")
            copy_directory = os.path.join(
                base_dest_directory,
                f"vaila_copy_{file_pattern.strip('_')}_{timestamp}",
            )
            os.makedirs(copy_directory, exist_ok=True)  # Create the export directory

            # Walk through the source directory and copy matching files to the new directory structure
            for root, dirs, files in os.walk(src_directory):
                for file in files:
                    # Check if the file matches the specified extension and pattern
                    if file.endswith(file_extension) and file_pattern in file:
                        # Copy the file to the appropriate subdirectory
                        src_path = os.path.join(root, file)
                        dest_path = os.path.join(copy_directory, file)
                        shutil.copy2(src_path, dest_path)

        # Show a success message after the operation is complete
        messagebox.showinfo(
            "Success",
            f"Files matching the specified patterns and extension {file_extension} have been exported successfully.",
        )
    except Exception as e:
        # Show an error message if something goes wrong
        messagebox.showerror("Error", f"Error copy files: {e}")


def export_file():
    src = filedialog.askopenfilename(title="Select the source file")
    if not src:
        messagebox.showerror("Error", "No source file selected.")
        return

    dest = filedialog.askdirectory(title="Select the destination directory")
    if not dest:
        messagebox.showerror("Error", "No destination directory selected.")
        return

    try:
        shutil.copy(src, dest)
        messagebox.showinfo("Success", f"File copied from {src} to {dest}")
    except Exception as e:
        messagebox.showerror("Error", f"Error copying file: {e}")


def move_file():
    allowed_directories = ["data", "results", "import", "export"]

    def select_directory(title):
        directory_path = filedialog.askdirectory(title=title)
        return directory_path

    src = select_directory("Select the source directory")
    if not src or not any(
        src.startswith(os.path.join(os.getcwd(), d)) for d in allowed_directories
    ):
        messagebox.showerror("Error", "No valid source directory selected.")
        return

    dest = select_directory("Select the destination directory")
    if not dest or not any(
        dest.startswith(os.path.join(os.getcwd(), d)) for d in allowed_directories
    ):
        messagebox.showerror("Error", "No valid destination directory selected.")
        return

    try:
        for item in os.listdir(src):
            src_path = os.path.join(src, item)
            dest_path = os.path.join(dest, item)
            if os.path.isfile(src_path):
                shutil.move(src_path, dest_path)
        messagebox.showinfo("Success", f"Files moved from {src} to {dest}")
    except Exception as e:
        messagebox.showerror("Error", f"Error moving files from {src} to {dest}: {e}")


def remove_file():
    # Lista de padrões perigosos e arquivos de sistema a serem protegidos
    forbidden_patterns = ["*", ".", "/", "\\"]
    system_files = [
        "boot.ini",
        "ntldr",
        "ntdetect.com",
        "autoexec.bat",
        "config.sys",
        "System",
        "System32",  # Windows
        ".bashrc",
        ".profile",
        ".bash_profile",
        ".bash_logout",
        "/etc/passwd",
        "/etc/shadow",  # Linux
        ".DS_Store",
        "/System",
        "/Applications",
        "/Users",
        "/Library",  # macOS
    ]

    # Prompt the user to select the root directory for the removal process
    root_directory = filedialog.askdirectory(title="Select Root Directory for Removal")
    if not root_directory:
        messagebox.showerror("Error", "No root directory selected.")
        return

    # Prompt the user to choose between removing by file extension or by directory/folder name
    removal_type = simpledialog.askstring(
        "Removal Type",
        "Enter 'ext' to remove by file extension or 'dir' to remove by directory/folder name:",
    )
    if not removal_type or removal_type not in ["ext", "dir"]:
        messagebox.showerror("Error", "Invalid removal type provided.")
        return

    # Prompt the user to enter the file extension or directory name pattern
    pattern = simpledialog.askstring(
        "Pattern",
        "Enter the file extension (e.g., .csv) or directory/folder name pattern to remove:",
    )
    if not pattern:
        messagebox.showerror("Error", "No pattern provided.")
        return

    # Verificar se o padrão está na lista de padrões proibidos
    if pattern in forbidden_patterns or pattern in system_files:
        messagebox.showerror("Error", "This pattern is forbidden for removal.")
        return

    # Verificar se o padrão pode causar remoção de arquivos de sistema operacional
    if removal_type == "dir" and any(sys_file in pattern for sys_file in system_files):
        messagebox.showerror(
            "Error", "Attempting to remove a system directory is not allowed."
        )
        return

    # Confirmation step - user must re-enter the pattern to confirm
    confirm_pattern = simpledialog.askstring(
        "Confirm Removal",
        f"To confirm, please re-enter the {('extension' if removal_type == 'ext' else 'directory/folder name')} you want to remove:",
    )
    if confirm_pattern != pattern:
        messagebox.showerror(
            "Error", "The confirmation pattern does not match. Operation canceled."
        )
        return

    # Final confirmation dialog
    confirmation = messagebox.askyesno(
        "Final Confirmation",
        f"Are you absolutely sure you want to remove all items matching '{pattern}' in '{root_directory}'?",
    )
    if not confirmation:
        return

    try:
        if removal_type == "ext":
            # Walk through the directories and remove files matching the extension
            for root, dirs, files in os.walk(root_directory):
                for file in files:
                    if file.endswith(pattern):
                        os.remove(os.path.join(root, file))

        elif removal_type == "dir":
            # Walk through the directories and remove folders matching the name pattern
            for root, dirs, files in os.walk(root_directory):
                for dir_name in dirs:
                    if pattern in dir_name:
                        shutil.rmtree(os.path.join(root, dir_name))

        messagebox.showinfo("Success", f"Items matching '{pattern}' have been removed.")
    except Exception as e:
        messagebox.showerror("Error", f"Error removing items: {e}")
