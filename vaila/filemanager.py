"""
================================================================================
Project: vailá Multimodal Toolbox
Script: filemanager.py - File Manager
================================================================================
Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 29 July 2024
Update Date: 5 February 2026
Version: 0.3.16

Description:
This script is designed to manage files and directories through a graphical user
interface (GUI) using Tkinter. It supports various operations such as copying,
moving, removing, and converting files, along with advanced features like
pattern matching and batch processing. The tool is particularly useful for
organizing large datasets and automating repetitive file operations.

    File Copy/Move:
        Allows copying or moving files based on file extensions and pattern matching.
        Streamlines file management across directories, minimizing manual effort.

    File Removal:
        Removes files matching specific extensions or directories.
        Safeguards critical system files from accidental deletion by recognizing
        forbidden patterns and offering multiple user confirmations.

    File Transfer:
        Provides a GUI for transferring files via SSH (rsync/scp).
        Supports both Upload (Send) and Download (Receive) modes.
        Includes cross-platform support (Linux, macOS, Windows) with password
        (paramiko) or key-based authentication.

Changelog for Version 0.3.16:
    - Updated Stroboscopic script help and integration.
    - Updated documentation headers.

License:
This script is distributed under the AGPL3 License
================================================================================
"""

import contextlib
import fnmatch
import os
import platform  # Add this import at the top with other imports
import re
import shutil
import subprocess
import threading
import time
import tkinter as tk
import unicodedata
from tkinter import filedialog, messagebox, simpledialog


def ask_directory(title="Select Directory", initialdir=None, parent=None):
    """
    Custom directory selection function to handle Linux quirks.
    Uses zenity if available on Linux, otherwise falls back to tkinter.
    """
    if initialdir:
        initialdir = os.path.normpath(initialdir)

    if platform.system() == "Linux" and shutil.which("zenity"):
        try:
            cmd = ["zenity", "--file-selection", "--directory", f"--title={title}"]
            if initialdir and os.path.exists(initialdir):
                cmd.append(f"--filename={initialdir}")

            # zenity returns the path with a newline at the end
            result = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
            return result
        except subprocess.CalledProcessError:
            # User cancelled
            return ""
        except Exception as e:
            print(f"Zenity error: {e}, falling back to tkinter")

    # Fallback to tkinter
    kwargs = {"title": title}
    if initialdir:
        kwargs["initialdir"] = initialdir
    if parent:
        kwargs["parent"] = parent

    path = filedialog.askdirectory(**kwargs)
    if path:
        return os.path.normpath(path)
    return path


def copy_file():
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("[Action] Copy files")

    # Prompt the user to select the main path directory for recursive search
    src_directory = ask_directory(title="Select Source Directory")

    # Check if a source directory was selected; if not, show an error message
    if not src_directory:
        messagebox.showerror("Error", "No source directory selected.")
        return

    # Prompt the user to enter the file extension to search for
    file_extension = simpledialog.askstring(
        "File Extension", "Enter the file extension to copy (e.g., .csv, .mp4):"
    )
    # Allow blank to mean "all extensions"
    if not file_extension:
        file_extension = None

    # Create a new window for pattern entry
    pattern_window = tk.Tk()
    pattern_window.title("Enter File Patterns")

    # Text box for entering multiple patterns, one per line
    pattern_label = tk.Label(pattern_window, text="Enter file patterns (one per line):")
    pattern_label.pack()

    pattern_text = tk.Text(pattern_window, height=44, width=60)
    pattern_text.pack()

    def on_submit():
        patterns = [
            p for p in pattern_text.get("1.0", "end").strip().splitlines() if p.strip() != ""
        ]
        pattern_window.destroy()
        process_copy(src_directory, file_extension, patterns)

    submit_button = tk.Button(pattern_window, text="Submit", command=on_submit)
    submit_button.pack()

    pattern_window.mainloop()


def process_copy(src_directory, file_extension, patterns):
    print(
        f"[Action] Copy processing | src={src_directory} | ext={file_extension or 'ALL'} | patterns={patterns or ['ALL']}"
    )

    # If no patterns provided, match all
    if not patterns:
        patterns = [""]

    # Allow copying all extensions when None
    ext_filter = file_extension

    # Prompt the user to select the destination directory where the new directories will be created
    base_dest_directory = ask_directory(title="Select Destination Directory")
    if not base_dest_directory:
        messagebox.showerror("Error", "No destination directory selected.")
        return

    # Ensure the 'vaila_copy' directory exists within the selected destination directory
    base_dest_directory = os.path.join(base_dest_directory, "vaila_copy")
    os.makedirs(base_dest_directory, exist_ok=True)

    def matches(file_name: str, pattern: str) -> bool:
        if ext_filter and not file_name.endswith(ext_filter):
            return False
        return not (pattern and pattern not in file_name)

    copied = 0
    errors = []
    try:
        for file_pattern in patterns:
            pattern_label = file_pattern.strip("_") or "all"
            timestamp = time.strftime("%Y%m%d%H%M%S")
            copy_directory = os.path.join(
                base_dest_directory,
                f"vaila_copy_{pattern_label}_{timestamp}",
            )
            os.makedirs(copy_directory, exist_ok=True)

            for root, _dirs, files in os.walk(src_directory):
                for file in files:
                    if matches(file, file_pattern):
                        src_path = os.path.join(root, file)
                        dest_path = os.path.join(copy_directory, file)

                        # handle collisions by suffixing
                        counter = 1
                        base_name, ext = os.path.splitext(file)
                        while os.path.exists(dest_path):
                            dest_path = os.path.join(copy_directory, f"{base_name}_{counter}{ext}")
                            counter += 1
                        try:
                            shutil.copy2(src_path, dest_path)
                            copied += 1
                            print(f"[Copied] {src_path} -> {dest_path}")
                        except Exception as e:
                            errors.append(f"{src_path}: {e}")

        msg = f"Copied files: {copied}"
        if errors:
            msg += f"\nErrors: {len(errors)} (see console)"
        messagebox.showinfo("Success", msg)
    except Exception as e:
        messagebox.showerror("Error", f"Error copying files: {e}")


def export_file():
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    src = filedialog.askopenfilename(title="Select the source file")
    if not src:
        messagebox.showerror("Error", "No source file selected.")
        return

    dest = ask_directory(title="Select the destination directory")
    if not dest:
        messagebox.showerror("Error", "No destination directory selected.")
        return

    try:
        shutil.copy(src, dest)
        messagebox.showinfo("Success", f"File copied from {src} to {dest}")
    except Exception as e:
        messagebox.showerror("Error", f"Error copying file: {e}")


def move_file():
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("[Action] Move files")

    # Prompt the user to select the main path directory for recursive search
    src_directory = ask_directory(title="Select Source Directory")

    # Check if a source directory was selected; if not, show an error message
    if not src_directory:
        messagebox.showerror("Error", "No source directory selected.")
        return

    # Prompt the user to enter the file extension to search for
    file_extension = simpledialog.askstring(
        "File Extension", "Enter the file extension to move (e.g., .csv, .mp4):"
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
        process_move(src_directory, file_extension, patterns)

    submit_button = tk.Button(pattern_window, text="Submit", command=on_submit)
    submit_button.pack()

    pattern_window.mainloop()


def process_move(src_directory, file_extension, patterns):
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print(
        f"[Action] Move processing | src={src_directory} | ext={file_extension} | patterns={patterns}"
    )

    # Prompt the user to select the destination directory where the new directories will be created
    base_dest_directory = ask_directory(title="Select Destination Directory")
    if not base_dest_directory:
        messagebox.showerror("Error", "No destination directory selected.")
        return

    # Ensure the 'vaila_move' directory exists within the selected destination directory
    base_dest_directory = os.path.join(base_dest_directory, "vaila_move")
    os.makedirs(base_dest_directory, exist_ok=True)

    try:
        for file_pattern in patterns:
            # Generate a timestamp to create a unique directory name for each pattern
            timestamp = time.strftime("%Y%m%d%H%M%S")
            move_directory = os.path.join(
                base_dest_directory,
                f"vaila_move_{file_pattern.strip('_')}_{timestamp}",
            )
            os.makedirs(move_directory, exist_ok=True)  # Create the move directory

            # Walk through the source directory and move matching files to the new directory structure
            for root, _dirs, files in os.walk(src_directory):
                for file in files:
                    # Check if the file matches the specified extension and pattern
                    if file.endswith(file_extension) and file_pattern in file:
                        # Move the file to the appropriate subdirectory
                        src_path = os.path.join(root, file)
                        dest_path = os.path.join(move_directory, file)
                        shutil.move(src_path, dest_path)

        # Show a success message after the operation is complete
        messagebox.showinfo(
            "Success",
            f"Files matching the specified patterns and extension {file_extension} have been moved successfully.",
        )
    except Exception as e:
        # Show an error message if something goes wrong
        messagebox.showerror("Error", f"Error moving files: {e}")


def remove_file():
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("[Action] Remove files")

    # List of dangerous patterns and system files to protect
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
    root_directory = ask_directory(title="Select Root Directory for Removal")
    if not root_directory:
        messagebox.showerror("Error", "No root directory selected.")
        return

    # Prompt the user to choose the removal type: by file extension, directory name, or file name pattern
    removal_type = simpledialog.askstring(
        "Removal Type",
        "Enter 'ext' to remove by file extension, 'dir' to remove by directory name, or 'name' to remove by file name pattern:",
    )
    if not removal_type or removal_type not in ["ext", "dir", "name"]:
        messagebox.showerror("Error", "Invalid removal type provided.")
        return

    # Prompt the user to enter the file extension, directory name pattern, or file name pattern
    pattern = simpledialog.askstring(
        "Pattern",
        "Enter the file extension (e.g., .csv), directory name pattern, or file name pattern (e.g., *backup*):",
    )
    if not pattern:
        messagebox.showerror("Error", "No pattern provided.")
        return

    # Check if the pattern is in the list of forbidden patterns or system files
    if pattern in forbidden_patterns or pattern in system_files:
        messagebox.showerror("Error", "This pattern is forbidden for removal.")
        return

    # Check if the pattern might cause removal of critical system files or directories
    if removal_type == "dir" and any(sys_file in pattern for sys_file in system_files):
        messagebox.showerror("Error", "Attempting to remove a system directory is not allowed.")
        return

    # Confirmation step - user must re-enter the pattern to confirm
    confirm_pattern = simpledialog.askstring(
        "Confirm Removal",
        f"To confirm, please re-enter the {('extension' if removal_type == 'ext' else 'directory/folder name' if removal_type == 'dir' else 'file name pattern')} you want to remove:",
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
            # Walk through directories and remove files matching the extension
            for root, _dirs, files in os.walk(root_directory):
                for file in files:
                    if file.endswith(pattern):
                        os.remove(os.path.join(root, file))

        elif removal_type == "dir":
            # Walk through directories and remove folders matching the name pattern
            for root, dirs, _files in os.walk(root_directory):
                for dir_name in dirs:
                    if pattern in dir_name:
                        shutil.rmtree(os.path.join(root, dir_name))

        elif removal_type == "name":
            # Walk through directories and remove files matching the file name pattern
            for root, _dirs, files in os.walk(root_directory):
                for file in files:
                    if fnmatch.fnmatch(file, pattern):
                        os.remove(os.path.join(root, file))

        messagebox.showinfo("Success", f"Items matching '{pattern}' have been removed.")
    except Exception as e:
        messagebox.showerror("Error", f"Error removing items: {e}")


def import_file():
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("[Action] Import/Export menu")

    """
    Create a GUI with multiple buttons, each calling an external script or function.
    """

    def data_import_tsv_qualysis():
        messagebox.showinfo("Data Import", "You selected to import a .tsv file from Qualysis.")
        # Add your .tsv processing logic here

    def data_import_mat_matlab():
        messagebox.showinfo("Data Import", "You selected to import a .mat file from MATLAB.")
        # Add your .mat processing logic here

    def data_import_csv_vicon_nexus():
        """
        Executes the VICON Nexus CSV batch processing from load_vicon_csv_split_batch.py
        """
        try:
            # Import the batch processing module (adjust the path as necessary)
            from vaila import load_vicon_csv_split_batch

            # Confirm if the 'process_csv_files_first_level' exists in the module
            if not hasattr(load_vicon_csv_split_batch, "process_csv_files_first_level"):
                raise AttributeError(
                    "The module 'load_vicon_csv_split_batch' does not have 'process_csv_files_first_level' function"
                )

            # Ask the user to select the source and output directories
            src_directory, output_directory = load_vicon_csv_split_batch.select_directory()

            # Run the batch processing function from load_vicon_csv_split_batch.py
            load_vicon_csv_split_batch.process_csv_files_first_level(
                src_directory, output_directory
            )

            # Show success message once processing is completed
            messagebox.showinfo(
                "Success",
                "Batch processing of VICON Nexus CSV files completed successfully.",
            )

        except AttributeError as e:
            # Handle the case where the function is missing
            messagebox.showerror("Error", f"An error occurred: {e}")

        except Exception as e:
            # General error handling
            messagebox.showerror("Error", f"An error occurred during batch processing: {e}")

    def data_import_html():
        messagebox.showinfo("Data Import", "You selected to import a .html file.")
        # Add your .html processing logic here

    def data_import_xml():
        messagebox.showinfo("Data Import", "You selected to import a .xml file.")
        # Add your .xml processing logic here

    def data_import_xlsx():
        messagebox.showinfo("Data Import", "You selected to import an .xlsx file.")
        # Add your .xlsx processing logic here

    def data_import_bvh():
        messagebox.showinfo("Data Import", "You selected to import a BVH file.")
        # Add your .bvh processing logic here

    def data_export_csv():
        messagebox.showinfo("Data Export", "You selected to export to CSV.")
        # Add your CSV export logic here

    def data_export_fbx():
        messagebox.showinfo("Data Export", "You selected to export to FBX.")
        # Add your FBX export logic here

    def data_export_trc():
        messagebox.showinfo("Data Export", "You selected to export to TRC.")
        # Add your TRC export logic here

    # Create the main GUI window
    root = tk.Tk()
    root.title("File Manager - Import/Export Options")

    # Create buttons for each file type or operation
    buttons = [
        ("Data Import: .tsv QUALYSIS", data_import_tsv_qualysis),
        ("Data Import: .mat MATLAB", data_import_mat_matlab),
        ("Data Import: .csv VICON NEXUS", data_import_csv_vicon_nexus),
        ("Data Import: HTML", data_import_html),  # New HTML button
        ("Data Import: XML", data_import_xml),  # New XML button
        ("Data Import: XLSX", data_import_xlsx),  # New XLSX button
        ("Data Import: BVH", data_import_bvh),
        ("Data Export: CSV", data_export_csv),
        ("Data Export: FBX", data_export_fbx),
        ("Data Export: TRC", data_export_trc),
    ]

    # Create a grid of buttons
    for i, (label, command) in enumerate(buttons):
        btn = tk.Button(root, text=label, command=command, width=40)
        btn.grid(row=i // 2, column=i % 2, padx=10, pady=10)

    # Start the GUI event loop
    root.mainloop()


def rename_files():
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("[Action] Rename files")

    # Prompt the user to select the directory containing the files to rename
    directory = ask_directory(title="Select Directory with Files to Rename")

    if not directory:
        messagebox.showerror("Error", "No directory selected.")
        return

    # Prompt the user to enter the text to be replaced and the text to replace it with
    text_to_replace = simpledialog.askstring("Text to Replace", "Enter the text to replace:")
    replacement_text = simpledialog.askstring(
        "Replacement Text",
        "Enter the replacement text (leave empty to remove the text):",
    )

    if text_to_replace is None or replacement_text is None:
        messagebox.showerror("Error", "No text provided.")
        return

    # Prompt the user to enter the file extension
    file_extension = simpledialog.askstring(
        "File Extension", "Enter the file extension (e.g., .png, .txt):"
    )

    if not file_extension:
        messagebox.showerror("Error", "No file extension provided.")
        return

    try:
        # Walk through the directory and its subdirectories
        for root, _dirs, files in os.walk(directory):
            for filename in files:
                # Check if the file has the specified extension and contains the text to replace
                if filename.endswith(file_extension) and text_to_replace in filename:
                    # Create the new filename by replacing the text
                    new_filename = filename.replace(text_to_replace, replacement_text)
                    # Rename the file
                    os.rename(os.path.join(root, filename), os.path.join(root, new_filename))

        # Show a success message after renaming is complete
        messagebox.showinfo("Success", "Files have been renamed successfully.")

    except Exception as e:
        # Show an error message if something goes wrong
        messagebox.showerror("Error", f"Error renaming files: {e}")


def _clean_text(text):
    """
    Auxiliary function to normalize strings:
    - Remove accents
    - Lowercase
    - Replace ç with c
    - Replace spaces and hyphens with _
    - Remove special characters
    """
    # 1. Convert to string and lowercase
    text = str(text).lower()

    # 2. Manual substitutions
    text = text.replace("ç", "c")

    # 3. Unicode normalization (remove accents)
    nfkd_form = unicodedata.normalize("NFKD", text)
    text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])

    # 4. Replace spaces and hyphens with underscore
    text = text.replace(" ", "_").replace("-", "_")

    # 5. Remove non-alphanumeric characters (keep only letters, numbers, _ and .)
    # This preserves file extensions (e.g., .mp4)
    text = re.sub(r"[^a-z0-9_.]", "", text)

    # 6. Remove duplicate underscores
    text = re.sub(r"_+", "_", text)

    # 7. Remove underscore at edges (aesthetic)
    text = text.strip("_")

    return text


def normalize_names():
    """
    Normalize file and folder names in a directory.

    This function will:
    - Convert to lowercase
    - Remove accents
    - Replace spaces and hyphens with underscores
    - Remove special characters
    - Preserve file extensions

    WARNING: This action is irreversible!
    """
    # Print header info consistent with other functions
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("[Action] Normalize names")

    # Prompt user for directory
    directory = ask_directory(title="Select Directory to Normalize (Files & Folders)")

    if not directory:
        messagebox.showerror("Error", "No directory selected.")
        return

    # Warning Dialog
    confirm = messagebox.askyesno(
        "WARNING: Irreversible Action",
        f"This will rename ALL files and folders inside:\n{directory}\n\n"
        "Changes applied:\n"
        "- Lowercase\n- Remove accents\n- Spaces to underscores\n- Remove special chars\n\n"
        "Do you want to proceed?",
    )

    if not confirm:
        return

    count_files = 0
    count_dirs = 0
    errors = []

    # topdown=False is crucial: rename children before parents
    for root, dirs, files in os.walk(directory, topdown=False):
        # 1. Normalize Files
        for filename in files:
            original_path = os.path.join(root, filename)
            new_filename = _clean_text(filename)
            new_path = os.path.join(root, new_filename)

            if original_path != new_path:
                # Check for collision
                if os.path.exists(new_path):
                    print(f"Skipped (exists): {filename} -> {new_filename}")
                    continue

                try:
                    os.rename(original_path, new_path)
                    print(f"File Renamed: {filename} -> {new_filename}")
                    count_files += 1
                except Exception as e:
                    errors.append(f"File {filename}: {str(e)}")

        # 2. Normalize Directories
        for dirname in dirs:
            original_path = os.path.join(root, dirname)
            new_dirname = _clean_text(dirname)
            new_path = os.path.join(root, new_dirname)

            if original_path != new_path:
                # Check for collision
                if os.path.exists(new_path):
                    print(f"Skipped (exists): {dirname} -> {new_dirname}")
                    continue

                try:
                    os.rename(original_path, new_path)
                    print(f"Folder Renamed: {dirname} -> {new_dirname}")
                    count_dirs += 1
                except Exception as e:
                    errors.append(f"Dir {dirname}: {str(e)}")

    # Final Report
    msg = f"Normalization Complete!\n\nFiles renamed: {count_files}\nFolders renamed: {count_dirs}"
    if errors:
        msg += f"\n\nErrors encountered: {len(errors)}\n(Check console for details)"

    messagebox.showinfo("Success", msg)


def tree_file():
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("[Action] Tree file")

    # Prompt the user to select the main path directory for recursive search
    src_directory = ask_directory(title="Select Source Directory")

    # Check if a source directory was selected; if not, show an error message
    if not src_directory:
        messagebox.showerror("Error", "No source directory selected.")
        return

    # Prompt the user to enter the file extension to search for
    file_extension = simpledialog.askstring(
        "File Extension", "Enter the file extension to search for (e.g., .csv, .mp4):"
    )
    if not file_extension:
        messagebox.showerror("Error", "No file extension provided.")
        return

    # Prompt the user to select the destination directory where the .txt file will be saved
    dest_directory = ask_directory(title="Select Destination Directory")
    if not dest_directory:
        messagebox.showerror("Error", "No destination directory selected.")
        return

    # Generate the output file path with the timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")
    output_file_path = os.path.join(dest_directory, f"vaila_tree_{timestamp}.txt")

    try:
        with open(output_file_path, "w") as output_file:
            # Walk through the source directory and list matching files
            for root, _dirs, files in os.walk(src_directory):
                for file in files:
                    if file.endswith(file_extension):
                        # Write the relative file path to the output file
                        relative_path = os.path.relpath(os.path.join(root, file), src_directory)
                        output_file.write(f"{relative_path}\n")

        # Show a success message after the operation is complete
        messagebox.showinfo(
            "Success",
            f"File tree saved successfully to {output_file_path}.",
        )
    except Exception as e:
        # Show an error message if something goes wrong
        messagebox.showerror("Error", f"Error saving file tree: {e}")


def find_file():
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("[Action] Find file")

    # Prompt the user to select the main path directory for recursive search
    src_directory = ask_directory(title="Select Source Directory")

    # Check if a source directory was selected; if not, show an error message
    if not src_directory:
        messagebox.showerror("Error", "No source directory selected.")
        return

    # Prompt the user to enter the file name pattern and extension separately
    search_pattern = simpledialog.askstring(
        "Search Pattern",
        "Enter the name pattern (optional) and file extension (e.g., vailá .mp4 or just .mp4):",
    )
    if not search_pattern:
        messagebox.showerror("Error", "No search pattern or extension provided.")
        return

    # Split the input to get the name pattern and extension
    search_parts = search_pattern.split()
    if len(search_parts) == 1:
        # If only the extension is provided
        pattern = ""
        file_extension = search_parts[0]
    else:
        # If both pattern and extension are provided
        pattern = search_parts[0]
        file_extension = search_parts[1]

    # Ensure the extension starts with an asterisk for the search
    if not file_extension.startswith("*"):
        file_extension = f"*{file_extension}"

    # Combine pattern and extension into a complete search pattern
    if pattern:
        full_pattern = f"*{pattern}*{file_extension}"
    else:
        full_pattern = file_extension  # Use extension only if no pattern is provided

    # Prompt the user to select the destination directory where the .txt file will be saved
    dest_directory = ask_directory(title="Select Destination Directory")
    if not dest_directory:
        messagebox.showerror("Error", "No destination directory selected.")
        return

    # Generate the output file path with the timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")
    output_file_path = os.path.join(dest_directory, f"vaila_find_{timestamp}.txt")

    # Variables to count files and total size
    extension_count = 0
    total_size_bytes = 0
    found_files = []

    try:
        # Walk through the source directory and find matching files or directories
        for root, dirs, files in os.walk(src_directory):
            # Match files and directories based on the search pattern
            for name in dirs + files:
                if fnmatch.fnmatch(name, full_pattern):
                    # Write the relative file path to the list
                    relative_path = os.path.relpath(os.path.join(root, name), src_directory)
                    found_files.append(relative_path)

            # Count files with the specified extension and calculate their sizes
            for file in files:
                if fnmatch.fnmatch(file, full_pattern):
                    extension_count += 1
                    total_size_bytes += os.path.getsize(os.path.join(root, file))

        # Convert total size to megabytes
        total_size_mb = total_size_bytes / (1024 * 1024)

        # Write the header with summary information to the file
        with open(output_file_path, "w") as output_file:
            output_file.write("Summary of Search Results\n")
            output_file.write(f"Pattern Searched: {full_pattern}\n")
            output_file.write(f"Number of Files Found: {extension_count}\n")
            output_file.write(f"Total Size: {total_size_mb:.2f} MB\n\n")
            output_file.write("File Tree:\n")
            output_file.write("\n".join(found_files))

        # Print the results in the terminal
        print("Summary of Search Results")
        print(f"Pattern Searched: {full_pattern}")
        print(f"Number of Files Found: {extension_count}")
        print(f"Total Size: {total_size_mb:.2f} MB\n")
        print("File Tree:")
        for file in found_files:
            print(file)

        # Show a success message after the operation is complete
        messagebox.showinfo(
            "Success",
            f"Find results saved successfully to {output_file_path}.\n"
            f"Number of files with pattern '{pattern}' and extension {file_extension}: {extension_count}.",
        )
    except Exception as e:
        # Show an error message if something goes wrong
        messagebox.showerror("Error", f"Error finding files: {e}")


def transfer_file():
    """Transfer files to remote server using rsync or scp."""
    print("Run def transfer_file")
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("[Action] Transfer file")

    # Call the cross-platform GUI
    try:
        _transfer_file_gui()
    except Exception as e:
        print(f"Error launching transfer GUI: {e}")
        messagebox.showerror("Error", f"Error launching transfer GUI: {e}")


def _transfer_file_gui():
    """Cross-platform file transfer implementation using Python with GUI."""
    import shutil

    # Check for rsync or scp
    rsync_path = shutil.which("rsync")
    scp_path = shutil.which("scp")

    # Create GUI window first (needed for messagebox parent)
    # Get or create root window
    root = None
    try:
        # Try to get existing root window
        if hasattr(tk, "_default_root") and tk._default_root is not None:
            root = tk._default_root
        else:
            # Create a temporary root window if none exists
            root = tk.Tk()
            root.withdraw()  # Hide it
    except Exception:
        # Fallback: create new root
        root = tk.Tk()
        root.withdraw()

    if not rsync_path and not scp_path:
        error_msg = (
            "Neither rsync nor scp is found in PATH.\n\n"
            "To use file transfer, please install rsync or ensure OpenSSH is installed."
        )
        if platform.system() == "Windows":
            error_msg += (
                "\n\nOption 1: SCP (Recommended for Windows)\n"
                "  - Enable OpenSSH Client: Settings > Apps > Optional Features > Add OpenSSH Client\n"
                "  - Or run as Administrator: dism /online /Add-Capability /CapabilityName:OpenSSH.Client~~~~0.0.1.0\n\n"
                "Option 2: rsync\n"
                "  - Install via Git for Windows (includes rsync)\n"
                "  - Or install via Chocolatey: choco install rsync\n"
                "  - Or install via WSL/Cygwin"
            )
        else:
            error_msg += (
                "\n\nLinux: sudo apt install rsync openssh-client\nmacOS: brew install rsync"
            )

        # Use print if root is not available to avoid the error
        if root is None:
            print(f"ERROR: {error_msg}")
            return
        try:
            messagebox.showerror("Transfer Tool Not Found", error_msg, parent=root)
        except Exception:
            print(f"ERROR: {error_msg}")
        return

    # Determine which tool to use
    use_rsync = rsync_path is not None
    transfer_method = "RSYNC" if use_rsync else "SCP"

    # Ensure root is properly initialized before creating Toplevel
    if root is None:
        print("ERROR: Cannot create transfer window - no root window available")
        return

    # Update root window to ensure it's ready
    with contextlib.suppress(BaseException):
        root.update_idletasks()

    # Create GUI window for transfer configuration
    try:
        transfer_window = tk.Toplevel(root)
        transfer_window.title("File Transfer Configuration")
        transfer_window.geometry("600x500")
        transfer_window.resizable(True, True)
    except Exception as toplevel_error:
        print(f"ERROR: Failed to create transfer window: {toplevel_error}")
        # Try to show error via print instead of messagebox to avoid recursion
        return

    # Default local directory
    default_local_dir = os.path.join(os.path.expanduser("~"), "Downloads")

    # Variables
    local_dir_var = tk.StringVar(value=default_local_dir)
    remote_user_var = tk.StringVar()
    remote_host_var = tk.StringVar()
    remote_port_var = tk.StringVar(value="22")
    remote_dir_var = tk.StringVar()
    ssh_password_var = tk.StringVar()  # Password field
    mode_var = tk.StringVar(value="upload")  # Transfer mode: upload or download
    local_label_var = tk.StringVar(value="Local Directory (Source):")
    remote_label_var = tk.StringVar(value="Remote Directory (Destination):")

    # Main frame
    main_frame = tk.Frame(transfer_window, padx=10, pady=10)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Title
    title_label = tk.Label(
        main_frame, text=f"File Transfer Tool ({transfer_method})", font=("Arial", 12, "bold")
    )
    title_label.pack(pady=(0, 10))

    if not use_rsync:
        info_label = tk.Label(
            main_frame,
            text="Note: Using SCP (rsync not available). SCP transfers entire directories recursively.",
            fg="orange",
            font=("Arial", 9),
        )
        info_label.pack(pady=(0, 10))

    # Mode selection
    mode_frame = tk.LabelFrame(main_frame, text="Transfer Mode", padx=5, pady=5)
    mode_frame.pack(fill=tk.X, pady=(0, 10))

    def update_labels():
        if mode_var.get() == "upload":
            local_label_var.set("Local Directory (Source):")
            remote_label_var.set("Remote Directory (Destination):")
        else:
            local_label_var.set("Local Directory (Destination):")
            remote_label_var.set("Remote Directory (Source):")

    tk.Radiobutton(
        mode_frame,
        text="Upload (Send to Remote)",
        variable=mode_var,
        value="upload",
        command=update_labels,
    ).pack(side=tk.LEFT, padx=10)
    tk.Radiobutton(
        mode_frame,
        text="Download (Receive from Remote)",
        variable=mode_var,
        value="download",
        command=update_labels,
    ).pack(side=tk.LEFT, padx=10)

    # Local directory
    tk.Label(main_frame, textvariable=local_label_var, anchor="w").pack(fill=tk.X, pady=(5, 2))
    local_frame = tk.Frame(main_frame)
    local_frame.pack(fill=tk.X, pady=(0, 10))
    local_entry = tk.Entry(local_frame, textvariable=local_dir_var, width=50)
    local_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

    def browse_local():
        dir_path = ask_directory(
            title="Select Local Directory to Transfer",
            initialdir=local_dir_var.get() or default_local_dir,
            parent=transfer_window,
        )
        if dir_path:
            local_dir_var.set(dir_path)

    tk.Button(local_frame, text="Browse...", command=browse_local).pack(side=tk.LEFT)

    # Remote username
    tk.Label(main_frame, text="Remote Username:", anchor="w").pack(fill=tk.X, pady=(5, 2))
    tk.Entry(main_frame, textvariable=remote_user_var, width=50).pack(fill=tk.X, pady=(0, 10))

    # Remote host
    tk.Label(main_frame, text="Remote Host (IP or hostname):", anchor="w").pack(
        fill=tk.X, pady=(5, 2)
    )
    tk.Entry(main_frame, textvariable=remote_host_var, width=50).pack(fill=tk.X, pady=(0, 10))

    # SSH port
    tk.Label(main_frame, text="SSH Port:", anchor="w").pack(fill=tk.X, pady=(5, 2))
    tk.Entry(main_frame, textvariable=remote_port_var, width=50).pack(fill=tk.X, pady=(0, 10))

    # Remote directory
    tk.Label(main_frame, textvariable=remote_label_var, anchor="w").pack(fill=tk.X, pady=(5, 2))
    tk.Entry(main_frame, textvariable=remote_dir_var, width=50).pack(fill=tk.X, pady=(0, 10))

    # SSH Password (REMOVED/DISABLED)
    # tk.Label(main_frame, text="SSH Password:", anchor="w").pack(fill=tk.X, pady=(5, 2))
    # password_entry = tk.Entry(main_frame, textvariable=ssh_password_var, width=50, show="*")
    # password_entry.pack(fill=tk.X, pady=(0, 10))
    tk.Label(
        main_frame,
        text="Note: You will be prompted for the password in the terminal window.",
        fg="blue",
        font=("Arial", 9, "italic"),
    ).pack(fill=tk.X, pady=(5, 10))

    # Progress text area (initially hidden)
    progress_frame = tk.Frame(main_frame)
    progress_label = tk.Label(progress_frame, text="Transfer Progress:", font=("Arial", 10, "bold"))
    progress_text = tk.Text(progress_frame, height=10, width=70, wrap=tk.WORD, state=tk.DISABLED)
    progress_scroll = tk.Scrollbar(progress_frame, orient=tk.VERTICAL, command=progress_text.yview)
    progress_text.config(yscrollcommand=progress_scroll.set)

    # Buttons frame
    buttons_frame = tk.Frame(main_frame)
    buttons_frame.pack(fill=tk.X, pady=(10, 0))

    # Transfer process variable (needs to be accessible from both functions)
    transfer_process_ref = {"process": None}

    def start_transfer():
        # Validate inputs
        local_dir = local_dir_var.get().strip()
        remote_user = remote_user_var.get().strip()
        remote_host = remote_host_var.get().strip()
        remote_port = remote_port_var.get().strip() or "22"
        remote_dir = remote_dir_var.get().strip()
        ssh_password_var.get()  # Get password from GUI

        if not local_dir:
            messagebox.showerror("Error", "Please specify local directory.", parent=transfer_window)
            return

        if not os.path.exists(local_dir):
            messagebox.showerror(
                "Error", f"Local directory not found: {local_dir}", parent=transfer_window
            )
            return

        if not remote_user:
            messagebox.showerror("Error", "Please enter remote username.", parent=transfer_window)
            return

        if not remote_host:
            messagebox.showerror("Error", "Please enter remote host.", parent=transfer_window)
            return

        if not remote_dir:
            messagebox.showerror("Error", "Please enter remote directory.", parent=transfer_window)
            return

        # Always force terminal prompt - no password checking needed here
        # if not ssh_password: ... logic removed

        mode = mode_var.get()

        # Show progress area
        progress_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        progress_label.pack(anchor="w")
        progress_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        progress_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        progress_text.config(state=tk.NORMAL)
        progress_text.delete(1.0, tk.END)
        progress_text.insert(tk.END, f"Starting {mode}...\n")

        if mode == "upload":
            source_desc = local_dir
            dest_desc = f"{remote_user}@{remote_host}:{remote_dir}"
        else:
            source_desc = f"{remote_user}@{remote_host}:{remote_dir}"
            dest_desc = local_dir

        progress_text.insert(tk.END, f"From: {source_desc}\n")
        progress_text.insert(tk.END, f"To: {dest_desc}\n")
        progress_text.insert(tk.END, f"SSH Port: {remote_port}\n")
        progress_text.insert(tk.END, f"Method: {transfer_method}\n")
        progress_text.insert(tk.END, "-" * 60 + "\n\n")
        progress_text.config(state=tk.DISABLED)
        progress_text.see(tk.END)

        # Disable start button
        start_btn.config(state=tk.DISABLED)
        cancel_btn.config(state=tk.NORMAL)

        # Build command
        # Normalize paths (Universal format for Windows compatibility)
        local_dir_normalized = local_dir.replace("\\", "/").rstrip("/")

        # Context: Remote path likely doesn't need normalization if it's Linux, but good to be safe w.r.t slashes
        if (
            not remote_dir.startswith("/") and ":" not in remote_dir
        ):  # Simple check assuming Linux/Unix remote
            remote_dir_normalized = f"/{remote_dir}".replace("//", "/")
        else:
            remote_dir_normalized = remote_dir

        remote_path_trimmed = remote_dir_normalized.rstrip("/")

        # Build command
        if use_rsync:
            # For rsync, the -e option needs the ssh command as a single string
            ssh_cmd = f"ssh -p {remote_port}"

            if mode == "upload":
                # Local -> Remote
                cmd = [
                    rsync_path,
                    "-avzhP",
                    "-e",
                    ssh_cmd,
                    f"{local_dir_normalized}",  # Source: send directory itself
                    f"{remote_user}@{remote_host}:{remote_path_trimmed}/",  # Dest: into this dir
                ]
            else:
                # Remote -> Local
                cmd = [
                    rsync_path,
                    "-avzhP",
                    "-e",
                    ssh_cmd,
                    f"{remote_user}@{remote_host}:{remote_path_trimmed}",  # Source: send directory itself
                    f"{local_dir_normalized}/",  # Dest: into this dir
                ]

        else:  # SCP
            # SCP treats recursive copy of 'dir' to 'dest' as 'dest/dir'.
            # Similar to rsync with trailing slash on dest, or no trailing slash on src.
            # Using normalized paths with forward slashes is safer on Windows OpenSSH/Git Bash.
            if mode == "upload":
                cmd = [
                    scp_path,
                    "-r",
                    "-P",
                    remote_port,
                    "-v",
                    "-C",
                    local_dir_normalized,
                    f"{remote_user}@{remote_host}:{remote_dir}",
                ]
            else:
                cmd = [
                    scp_path,
                    "-r",
                    "-P",
                    remote_port,
                    "-v",
                    "-C",
                    f"{remote_user}@{remote_host}:{remote_dir}",
                    local_dir_normalized,
                ]

        def run_transfer():
            try:
                progress_text.config(state=tk.NORMAL)
                # Try to run in terminal if we can't be sure

                # Try to execute directly first and capture output (similar to Windows "wait" approach but without new window)
                # BUT updating the GUI in real-time

                # Note: If rsync prompts for password, it reads from /dev/tty.
                # Without a proper TTY, it might fail or we can't feed it password.

                # On Linux/Mac, to allow password input, we should spawn a terminal
                terminal_cmd = None
                script_cmd = " ".join([f"'{arg}'" if " " in arg else arg for arg in cmd])

                # Detect terminal emulator
                if shutil.which("gnome-terminal"):
                    # gnome-terminal requires -- bash -c "cmd; exec bash" to keep open or just run
                    terminal_cmd = [
                        "gnome-terminal",
                        "--",
                        "bash",
                        "-c",
                        f"{script_cmd}; echo 'Press Enter to exit...'; read",
                    ]
                elif shutil.which("xterm"):
                    terminal_cmd = [
                        "xterm",
                        "-e",
                        f"{script_cmd}; echo 'Press Enter to exit...'; read",
                    ]
                elif shutil.which("konsole"):
                    terminal_cmd = [
                        "konsole",
                        "-e",
                        "bash",
                        "-c",
                        f"{script_cmd}; echo 'Press Enter to exit...'; read",
                    ]
                elif shutil.which("xfce4-terminal"):
                    terminal_cmd = [
                        "xfce4-terminal",
                        "-e",
                        f"bash -c \"{script_cmd}; echo 'Press Enter to exit...'; read\"",
                    ]
                elif platform.system() == "Darwin":  # macOS
                    # Use AppleScript to tell Terminal to do script
                    escaped_script = script_cmd.replace('"', '\\"')
                    subprocess.run(
                        [
                            "osascript",
                            "-e",
                            f'tell application "Terminal" to do script "{escaped_script}"',
                        ]
                    )
                    terminal_cmd = []  # Handled

                if terminal_cmd:
                    subprocess.Popen(terminal_cmd)
                    progress_text.config(state=tk.NORMAL)
                    progress_text.insert(
                        tk.END, "\n Launched external terminal for password input.\n"
                    )
                    progress_text.config(state=tk.DISABLED)
                elif platform.system() == "Darwin" and not terminal_cmd:
                    # Handled by osascript above
                    progress_text.config(state=tk.NORMAL)
                    progress_text.insert(tk.END, "\n Launched macOS Terminal for password input.\n")
                    progress_text.config(state=tk.DISABLED)
                else:
                    # Fallback for when no known terminal found - try running in current pty if possible or error
                    msg = (
                        "Could not detect a supported terminal emulator (gnome-terminal, xterm, konsole, xfce4-terminal).\n"
                        "Please run the following command manually in your terminal:\n\n"
                        + script_cmd
                    )
                    messagebox.showinfo("Manual Run Required", msg)
                    progress_text.config(state=tk.NORMAL)
                    progress_text.insert(tk.END, f"\n{msg}\n")
                    progress_text.insert(tk.END, f"Command: {script_cmd}\n")
                    progress_text.config(state=tk.DISABLED)

                    # Since it runs externally, we re-enable buttons immediately (or we could wait if we tracked PID)
                    start_btn.config(state=tk.NORMAL)
                    cancel_btn.config(state=tk.DISABLED)

            except Exception as e:
                progress_text.config(state=tk.NORMAL)
                progress_text.insert(tk.END, f"\n✗ Error: {e}\n", "error")
                progress_text.config(state=tk.DISABLED)
                progress_text.see(tk.END)
                start_btn.config(state=tk.NORMAL)
                cancel_btn.config(state=tk.DISABLED)

        # Configure text tags for colors
        progress_text.tag_config("success", foreground="green")
        progress_text.tag_config("error", foreground="red")

        # Run transfer in separate thread
        transfer_thread = threading.Thread(target=run_transfer, daemon=True)
        transfer_thread.start()

    def cancel_transfer():
        if transfer_process_ref["process"] and transfer_process_ref["process"].poll() is None:
            transfer_process_ref["process"].terminate()
            progress_text.config(state=tk.NORMAL)
            progress_text.insert(tk.END, "\n\nTransfer cancelled by user.\n")
            progress_text.config(state=tk.DISABLED)
            start_btn.config(state=tk.NORMAL)
            cancel_btn.config(state=tk.DISABLED)

    start_btn = tk.Button(
        buttons_frame,
        text="Start Transfer",
        command=start_transfer,
        bg="#4CAF50",
        fg="white",
        font=("Arial", 10, "bold"),
    )
    start_btn.pack(side=tk.LEFT, padx=(0, 5))

    cancel_btn = tk.Button(buttons_frame, text="Cancel", command=cancel_transfer, state=tk.DISABLED)
    cancel_btn.pack(side=tk.LEFT, padx=(0, 5))

    close_btn = tk.Button(buttons_frame, text="Close", command=transfer_window.destroy)
    close_btn.pack(side=tk.RIGHT)

    # Center window
    transfer_window.update_idletasks()
    width = transfer_window.winfo_width()
    height = transfer_window.winfo_height()
    x = (transfer_window.winfo_screenwidth() // 2) - (width // 2)
    y = (transfer_window.winfo_screenheight() // 2) - (height // 2)
    transfer_window.geometry(f"{width}x{height}+{x}+{y}")
