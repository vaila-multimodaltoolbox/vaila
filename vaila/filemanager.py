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


def export_file():
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
        process_export(src_directory, file_extension, patterns)

    submit_button = tk.Button(pattern_window, text="Submit", command=on_submit)
    submit_button.pack()

    pattern_window.mainloop()


def process_export(src_directory, file_extension, patterns):
    # Prompt the user to select the destination directory where the new directories will be created
    base_dest_directory = filedialog.askdirectory(title="Select Destination Directory")
    if not base_dest_directory:
        messagebox.showerror("Error", "No destination directory selected.")
        return

    # Ensure the 'vaila_export' directory exists within the selected destination directory
    base_dest_directory = os.path.join(base_dest_directory, "vaila_export")
    os.makedirs(base_dest_directory, exist_ok=True)

    try:
        for file_pattern in patterns:
            # Generate a timestamp to create a unique directory name for each pattern
            timestamp = time.strftime("%Y%m%d%H%M%S")
            export_directory = os.path.join(
                base_dest_directory,
                f"vaila_export_{file_pattern.strip('_')}_{timestamp}",
            )
            os.makedirs(export_directory, exist_ok=True)  # Create the export directory

            # Walk through the source directory and copy matching files to the new directory structure
            for root, dirs, files in os.walk(src_directory):
                for file in files:
                    # Check if the file matches the specified extension and pattern
                    if file.endswith(file_extension) and file_pattern in file:
                        # Copy the file to the appropriate subdirectory
                        src_path = os.path.join(root, file)
                        dest_path = os.path.join(export_directory, file)
                        shutil.copy2(src_path, dest_path)

        # Show a success message after the operation is complete
        messagebox.showinfo(
            "Success",
            f"Files matching the specified patterns and extension {file_extension} have been exported successfully.",
        )
    except Exception as e:
        # Show an error message if something goes wrong
        messagebox.showerror("Error", f"Error exporting files: {e}")


def copy_file():
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
    allowed_extensions = {
        ".avi",
        ".AVI",
        ".mp4",
        ".MP4",
        ".mov",
        ".MOV",
        ".mkv",
        ".MKV",
    }
    allowed_directories = ["data", "results", "import", "export"]

    def select_directory(title):
        directory_path = filedialog.askdirectory(title=title)
        return directory_path

    selected_directory = select_directory("Select the directory to remove files from")
    if not selected_directory or not any(
        selected_directory.startswith(os.path.join(os.getcwd(), d))
        for d in allowed_directories
    ):
        messagebox.showerror("Error", "No valid directory selected.")
        return

    confirmation = messagebox.askyesno(
        "Confirm", f"Are you sure you want to remove files from {selected_directory}?"
    )
    if confirmation:
        for item in os.listdir(selected_directory):
            item_path = os.path.join(selected_directory, item)
            if (
                os.path.isfile(item_path)
                and os.path.splitext(item_path)[1].lower() in allowed_extensions
            ):
                os.remove(item_path)
        messagebox.showinfo(
            "Success", f"Specified files in {selected_directory} have been removed."
        )
