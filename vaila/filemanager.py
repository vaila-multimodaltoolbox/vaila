"""
================================================================================
Project: vailá Multimodal Toolbox
Script: filemanager.py - File Manager
================================================================================
Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 29 July 2024
Update Date: 22 January 2026
Version: 0.3.14

Description:
This script is designed to manage files and directories through a graphical user interface (GUI) using Tkinter. It supports various operations such as copying, moving, removing, and converting files, along with advanced features like pattern matching and batch processing. The tool is particularly useful for organizing large datasets and automating repetitive file operations.

    File Export:
        Exports files to formats such as CSV, FBX, and TRC.
        Enables the user to prepare data for specific tools or further analysis.

    File Copy/Move:
        Allows copying or moving files based on file extensions and pattern matching.
        Streamlines file management across directories, minimizing manual effort.

    File Removal:
        Removes files matching specific extensions or directories.
        Safeguards critical system files from accidental deletion by recognizing forbidden patterns and offering multiple user confirmations.

Changelog for Version 0.1.1:

    Added support for file export to CSV, FBX, and TRC formats.
    Added support for file copy/move based on file extensions and pattern matching.
    Added support for file removal based on file extensions or directory names.
    Added support for file tree generation.
    Added support for file find based on file name pattern and extension.
    Added support for file transfer using SSH.
    Added support for file import from CSV, FBX, and TRC formats.
    Added support for file rename based on file name pattern.
    Added support for file tree generation.
    Added support for file find based on file name pattern and extension.
    Added support for file transfer using SSH.
    Added support for file import from CSV, FBX, and TRC formats.
    Added support for file rename based on file name pattern.
    Added support for file tree generation.
    Added support for file find based on file name pattern and extension.

    Initial release of the filemanager.py script.

License:
This script is distributed under the AGPL3 License
================================================================================
"""

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


def copy_file():
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("[Action] Copy files")

    # Prompt the user to select the main path directory for recursive search
    src_directory = filedialog.askdirectory(title="Select Source Directory")

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
    base_dest_directory = filedialog.askdirectory(title="Select Destination Directory")
    if not base_dest_directory:
        messagebox.showerror("Error", "No destination directory selected.")
        return

    # Ensure the 'vaila_copy' directory exists within the selected destination directory
    base_dest_directory = os.path.join(base_dest_directory, "vaila_copy")
    os.makedirs(base_dest_directory, exist_ok=True)

    def matches(file_name: str, pattern: str) -> bool:
        if ext_filter and not file_name.endswith(ext_filter):
            return False
        if pattern and pattern not in file_name:
            return False
        return True

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

            for root, dirs, files in os.walk(src_directory):
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
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("[Action] Move files")

    # Prompt the user to select the main path directory for recursive search
    src_directory = filedialog.askdirectory(title="Select Source Directory")

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
    base_dest_directory = filedialog.askdirectory(title="Select Destination Directory")
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
            for root, dirs, files in os.walk(src_directory):
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
    root_directory = filedialog.askdirectory(title="Select Root Directory for Removal")
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
            for root, dirs, files in os.walk(root_directory):
                for file in files:
                    if file.endswith(pattern):
                        os.remove(os.path.join(root, file))

        elif removal_type == "dir":
            # Walk through directories and remove folders matching the name pattern
            for root, dirs, files in os.walk(root_directory):
                for dir_name in dirs:
                    if pattern in dir_name:
                        shutil.rmtree(os.path.join(root, dir_name))

        elif removal_type == "name":
            # Walk through directories and remove files matching the file name pattern
            for root, dirs, files in os.walk(root_directory):
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
    directory = filedialog.askdirectory(title="Select Directory with Files to Rename")

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
        for root, dirs, files in os.walk(directory):
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
    directory = filedialog.askdirectory(title="Select Directory to Normalize (Files & Folders)")

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
    src_directory = filedialog.askdirectory(title="Select Source Directory")

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
    dest_directory = filedialog.askdirectory(title="Select Destination Directory")
    if not dest_directory:
        messagebox.showerror("Error", "No destination directory selected.")
        return

    # Generate the output file path with the timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")
    output_file_path = os.path.join(dest_directory, f"vaila_tree_{timestamp}.txt")

    try:
        with open(output_file_path, "w") as output_file:
            # Walk through the source directory and list matching files
            for root, dirs, files in os.walk(src_directory):
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
    src_directory = filedialog.askdirectory(title="Select Source Directory")

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
    dest_directory = filedialog.askdirectory(title="Select Destination Directory")
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

    # Ensure we have a root window before any messagebox calls
    root_for_error = None
    try:
        if hasattr(tk, '_default_root') and tk._default_root is not None:
            root_for_error = tk._default_root
        else:
            # Create a hidden root window for error dialogs
            root_for_error = tk.Tk()
            root_for_error.withdraw()
    except Exception as root_err:
        root_for_error = None

    try:
        if platform.system() == "Windows":
            # Windows: Use Python implementation
            _transfer_file_windows()
        else:
            # Linux/macOS: Use shell script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            script_name = "transfer.sh"
            script_path = os.path.join(script_dir, script_name)

            if not os.path.exists(script_path):
                # Try to get root for parent
                try:
                    root = tk._default_root if hasattr(tk,'_default_root') and tk._default_root is not None else None
                except:
                    root = None
                messagebox.showerror("Error", f"Transfer script not found: {script_path}", parent=root)
                return

            # Make the script executable
            os.chmod(script_path, 0o755)

            print(f"Executing transfer script: {script_path}")

            # Execute the shell script in a new terminal
            if platform.system() == "Darwin":  # macOS
                subprocess.Popen(["open", "-a", "Terminal", script_path])
            else:  # Linux
                # Try different terminal emulators
                terminals = ["gnome-terminal", "xterm", "konsole", "lxterminal"]
                for terminal in terminals:
                    try:
                        subprocess.Popen([terminal, "-e", f"bash {script_path}"], cwd=script_dir)
                        break
                    except FileNotFoundError:
                        continue
                else:
                    # Fallback: try to run directly
                    subprocess.Popen(["bash", script_path], cwd=script_dir)

    except FileNotFoundError as e:
        messagebox.showerror(
            "Error",
            f"Transfer script not found: {e.filename}\n"
            f"Please ensure {script_name} is in the same directory as filemanager.py",
        )
        print(f"Script not found: {e}")

    except Exception as e:
        # Always use print to avoid Tkinter messagebox errors
        error_msg = f"Unexpected error during transfer: {e}"
        print(error_msg)
        # DO NOT use messagebox here - it causes the "-default value" error
        # The error will be visible in the console/terminal


def _transfer_file_windows():
    """Windows-specific file transfer implementation using Python with GUI."""
    import shutil
    import threading
    
    # Check for rsync or scp
    rsync_path = shutil.which("rsync")
    scp_path = shutil.which("scp")
    
    # Create GUI window first (needed for messagebox parent)
    # Get or create root window
    root = None
    try:
        # Try to get existing root window
        if hasattr(tk, '_default_root') and tk._default_root is not None:
            root = tk._default_root
        else:
            # Create a temporary root window if none exists
            root = tk.Tk()
            root.withdraw()  # Hide it
    except Exception as root_error:
        # Fallback: create new root
        root = tk.Tk()
        root.withdraw()
    
    if not rsync_path and not scp_path:
        error_msg = (
            "Neither rsync nor scp is found in PATH.\n\n"
            "To use file transfer, install one of:\n\n"
            "Option 1: SCP (Recommended for Windows)\n"
            "  - Enable OpenSSH Client: Settings > Apps > Optional Features > Add OpenSSH Client\n"
            "  - Or run as Administrator: dism /online /Add-Capability /CapabilityName:OpenSSH.Client~~~~0.0.1.0\n\n"
            "Option 2: rsync\n"
            "  - Install via Git for Windows (includes rsync)\n"
            "  - Or install via Chocolatey: choco install rsync\n"
            "  - Or install via WSL/Cygwin"
        )
        # Use print if root is not available to avoid the error
        if root is None:
            print(f"ERROR: {error_msg}")
            return
        try:
            messagebox.showerror("Transfer Tool Not Found", error_msg, parent=root)
        except Exception as msg_err:
            print(f"ERROR: {error_msg}")
        return
    
    # Determine which tool to use
    use_rsync = rsync_path is not None
    transfer_method = "RSYNC" if use_rsync else "SCP"
    tool_path = rsync_path if use_rsync else scp_path
    
    # Ensure root is properly initialized before creating Toplevel
    if root is None:
        print("ERROR: Cannot create transfer window - no root window available")
        return
    
    # Update root window to ensure it's ready
    try:
        root.update_idletasks()
    except:
        pass
    
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
    
    # Main frame
    main_frame = tk.Frame(transfer_window, padx=10, pady=10)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Title
    title_label = tk.Label(
        main_frame,
        text=f"File Transfer Tool ({transfer_method})",
        font=("Arial", 12, "bold")
    )
    title_label.pack(pady=(0, 10))
    
    if not use_rsync:
        info_label = tk.Label(
            main_frame,
            text="Note: Using SCP (rsync not available). SCP transfers entire directories recursively.",
            fg="orange",
            font=("Arial", 9)
        )
        info_label.pack(pady=(0, 10))
    
    # Local directory
    tk.Label(main_frame, text="Local Directory:", anchor="w").pack(fill=tk.X, pady=(5, 2))
    local_frame = tk.Frame(main_frame)
    local_frame.pack(fill=tk.X, pady=(0, 10))
    local_entry = tk.Entry(local_frame, textvariable=local_dir_var, width=50)
    local_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
    
    def browse_local():
        dir_path = filedialog.askdirectory(
            title="Select Local Directory to Transfer",
            initialdir=local_dir_var.get() or default_local_dir,
            parent=transfer_window
        )
        if dir_path:
            local_dir_var.set(dir_path)
    
    tk.Button(local_frame, text="Browse...", command=browse_local).pack(side=tk.LEFT)
    
    # Remote username
    tk.Label(main_frame, text="Remote Username:", anchor="w").pack(fill=tk.X, pady=(5, 2))
    tk.Entry(main_frame, textvariable=remote_user_var, width=50).pack(fill=tk.X, pady=(0, 10))
    
    # Remote host
    tk.Label(main_frame, text="Remote Host (IP or hostname):", anchor="w").pack(fill=tk.X, pady=(5, 2))
    tk.Entry(main_frame, textvariable=remote_host_var, width=50).pack(fill=tk.X, pady=(0, 10))
    
    # SSH port
    tk.Label(main_frame, text="SSH Port:", anchor="w").pack(fill=tk.X, pady=(5, 2))
    tk.Entry(main_frame, textvariable=remote_port_var, width=50).pack(fill=tk.X, pady=(0, 10))
    
    # Remote directory
    tk.Label(main_frame, text="Remote Directory (full path):", anchor="w").pack(fill=tk.X, pady=(5, 2))
    tk.Entry(main_frame, textvariable=remote_dir_var, width=50).pack(fill=tk.X, pady=(0, 10))
    
    # SSH Password
    tk.Label(main_frame, text="SSH Password:", anchor="w").pack(fill=tk.X, pady=(5, 2))
    password_entry = tk.Entry(main_frame, textvariable=ssh_password_var, width=50, show="*")
    password_entry.pack(fill=tk.X, pady=(0, 10))
    
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
        ssh_password = ssh_password_var.get()  # Get password from GUI
        
        if not local_dir:
            messagebox.showerror("Error", "Please specify local directory.", parent=transfer_window)
            return
        
        if not os.path.exists(local_dir):
            messagebox.showerror("Error", f"Local directory not found: {local_dir}", parent=transfer_window)
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
        
        if not ssh_password:
            # Password is optional - user can choose to enter it in terminal if not provided
            use_password_in_gui = messagebox.askyesno(
                "SSH Password",
                "No password entered. Do you want to enter it now?\n\n"
                "Click 'Yes' to enter password in GUI, or 'No' to enter it in the terminal window.",
                parent=transfer_window
            )
            if use_password_in_gui:
                return  # User will enter password and try again
            # If No, continue without password (will prompt in terminal)
        
        # Show progress area
        progress_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        progress_label.pack(anchor="w")
        progress_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        progress_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        progress_text.config(state=tk.NORMAL)
        progress_text.delete(1.0, tk.END)
        progress_text.insert(tk.END, f"Starting transfer...\n")
        progress_text.insert(tk.END, f"From: {local_dir}\n")
        progress_text.insert(tk.END, f"To: {remote_user}@{remote_host}:{remote_dir}\n")
        progress_text.insert(tk.END, f"SSH Port: {remote_port}\n")
        progress_text.insert(tk.END, f"Method: {transfer_method}\n")
        progress_text.insert(tk.END, "-" * 60 + "\n\n")
        progress_text.config(state=tk.DISABLED)
        progress_text.see(tk.END)
        
        # Disable start button
        start_btn.config(state=tk.DISABLED)
        cancel_btn.config(state=tk.NORMAL)
        
        # Build command
        if use_rsync:
            # For rsync, the -e option needs the ssh command as a single string
            # The ssh command with port must be properly quoted
            # Normalize Windows paths (use forward slashes for rsync)
            local_dir_normalized = local_dir.replace("\\", "/")
            # Ensure remote_dir has proper format
            if not remote_dir.startswith("/"):
                remote_dir_normalized = f"/{remote_dir}".replace("//", "/")
            else:
                remote_dir_normalized = remote_dir
            
            # Build ssh command - must be a single string for -e option
            # For Windows rsync, we need to ensure the path format is correct
            ssh_cmd = f"ssh -p {remote_port}"
            # Remove trailing slash from local path - without it, rsync transfers the directory itself
            # With trailing slash, rsync transfers only the contents
            local_path_for_rsync = local_dir_normalized.rstrip('/')
            # Ensure remote path has proper format
            remote_path_for_rsync = remote_dir_normalized.rstrip('/')
            cmd = [
                rsync_path,
                "-avhP",
                "-e", ssh_cmd,
                f"{local_path_for_rsync}",  # No trailing slash - transfers the directory itself
                f"{remote_user}@{remote_host}:{remote_path_for_rsync}/"
            ]
        else:
            cmd = [
                scp_path,
                "-r",
                "-P", remote_port,
                "-v",
                "-C",
                local_dir,
                f"{remote_user}@{remote_host}:{remote_dir}"
            ]
        
        def run_transfer():
            try:
                progress_text.config(state=tk.NORMAL)
                progress_text.insert(tk.END, f"Command: {' '.join(cmd)}\n\n")
                progress_text.config(state=tk.DISABLED)
                progress_text.see(tk.END)
                transfer_window.update_idletasks()
                
                # Try to use paramiko if password is provided
                use_paramiko = False
                if ssh_password:
                    try:
                        import paramiko
                        use_paramiko = True
                        progress_text.config(state=tk.NORMAL)
                        progress_text.insert(tk.END, "Using paramiko for password authentication...\n")
                        progress_text.config(state=tk.DISABLED)
                        progress_text.see(tk.END)
                        transfer_window.update_idletasks()
                        
                        # Use paramiko to transfer files via SFTP
                        def transfer_with_paramiko():
                            try:
                                progress_text.config(state=tk.NORMAL)
                                progress_text.insert(tk.END, "Connecting to server...\n")
                                progress_text.config(state=tk.DISABLED)
                                progress_text.see(tk.END)
                                transfer_window.update_idletasks()
                                
                                # Create SSH client
                                ssh = paramiko.SSHClient()
                                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                                
                                # Connect with password
                                ssh.connect(
                                    hostname=remote_host,
                                    port=int(remote_port),
                                    username=remote_user,
                                    password=ssh_password,
                                    timeout=30
                                )
                                
                                progress_text.config(state=tk.NORMAL)
                                progress_text.insert(tk.END, "Connected! Starting file transfer...\n")
                                progress_text.config(state=tk.DISABLED)
                                progress_text.see(tk.END)
                                transfer_window.update_idletasks()
                                
                                # Use SFTP to transfer files
                                sftp = ssh.open_sftp()
                                
                                # Ensure remote directory exists
                                remote_dir_parts = remote_dir.strip('/').split('/')
                                current_remote_path = ''
                                for part in remote_dir_parts:
                                    if part:
                                        current_remote_path = f"{current_remote_path}/{part}" if current_remote_path else f"/{part}"
                                        try:
                                            sftp.stat(current_remote_path)
                                        except IOError:
                                            sftp.mkdir(current_remote_path)
                                
                                # Transfer directory recursively
                                # First, create the directory on remote (using the basename of local_dir)
                                local_dir_name = os.path.basename(local_dir.rstrip('/\\'))
                                remote_target_dir = f"{remote_dir.rstrip('/')}/{local_dir_name}"
                                
                                # Ensure remote target directory exists
                                try:
                                    sftp.stat(remote_target_dir)
                                except IOError:
                                    sftp.mkdir(remote_target_dir)
                                
                                # Transfer files recursively
                                def upload_directory(local_path, remote_path):
                                    for item in os.listdir(local_path):
                                        local_item = os.path.join(local_path, item)
                                        remote_item = f"{remote_path.rstrip('/')}/{item}"
                                        
                                        if os.path.isdir(local_item):
                                            try:
                                                sftp.mkdir(remote_item)
                                            except IOError:
                                                pass  # Directory may already exist
                                            upload_directory(local_item, remote_item)
                                        else:
                                            progress_text.config(state=tk.NORMAL)
                                            progress_text.insert(tk.END, f"Transferring: {os.path.basename(local_item)}\n")
                                            progress_text.config(state=tk.DISABLED)
                                            progress_text.see(tk.END)
                                            transfer_window.update_idletasks()
                                            sftp.put(local_item, remote_item)
                                
                                # Upload the entire directory structure
                                upload_directory(local_dir, remote_target_dir)
                                
                                sftp.close()
                                ssh.close()
                                
                                progress_text.config(state=tk.NORMAL)
                                progress_text.insert(tk.END, "\n" + "-" * 60 + "\n")
                                progress_text.insert(tk.END, "✓ Transfer completed successfully!\n", "success")
                                progress_text.config(state=tk.DISABLED)
                                progress_text.see(tk.END)
                                
                                start_btn.config(state=tk.NORMAL)
                                cancel_btn.config(state=tk.DISABLED)
                                
                            except Exception as e:
                                progress_text.config(state=tk.NORMAL)
                                progress_text.insert(tk.END, f"\n✗ Error: {e}\n", "error")
                                progress_text.config(state=tk.DISABLED)
                                progress_text.see(tk.END)
                                start_btn.config(state=tk.NORMAL)
                                cancel_btn.config(state=tk.DISABLED)
                        
                        # Run paramiko transfer in thread
                        transfer_thread = threading.Thread(target=transfer_with_paramiko, daemon=True)
                        transfer_thread.start()
                        return  # Exit early, paramiko handles everything
                        
                    except ImportError:
                        # paramiko not available, fall back to terminal approach
                        use_paramiko = False
                        progress_text.config(state=tk.NORMAL)
                        progress_text.insert(tk.END, "Note: paramiko not available. Password will be entered in terminal.\n")
                        progress_text.insert(tk.END, "To use password from GUI, install: pip install paramiko\n\n")
                        progress_text.config(state=tk.DISABLED)
                        progress_text.see(tk.END)
                
                # On Windows, execute in a separate terminal to allow password input
                if platform.system() == "Windows":
                    # Create a PowerShell script to run the command
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    ps_script = os.path.join(script_dir, "transfer_run.ps1")
                    
                    # Build PowerShell script with proper command execution
                    # For rsync, we need to ensure the -e option is passed correctly
                    # The issue is that PowerShell array syntax may not work correctly with rsync
                    # So we'll build the command as a string instead
                    
                    def escape_ps_string(s):
                        # Escape for PowerShell string
                        return s.replace('`', '``').replace('$', '`$').replace('"', '`"').replace("'", "''")
                    
                    # Build command for PowerShell execution
                    # Use Start-Process or direct execution with proper argument array
                    if use_rsync:
                        # For rsync, build arguments array for PowerShell
                        # PowerShell needs arguments as an array, not a string
                        exe_path = cmd[0]
                        args_list = []
                        i = 1
                        while i < len(cmd):
                            if cmd[i] == "-e" and i + 1 < len(cmd):
                                # -e and its value must be separate array elements
                                args_list.append('-e')
                                args_list.append(cmd[i+1])  # "ssh -p PORT" as single string
                                i += 2
                            else:
                                args_list.append(cmd[i])
                                i += 1
                        
                        # Build PowerShell command using array syntax
                        args_str = ', '.join([f'"{escape_ps_string(arg)}"' for arg in args_list])
                        exe_escaped = escape_ps_string(exe_path)
                        cmd_str = f'& "{exe_escaped}" {args_str}'
                    else:
                        # For scp, build command string
                        cmd_parts = [escape_ps_string(cmd[0])]
                        for arg in cmd[1:]:
                            if ' ' in arg or ':' in arg or '@' in arg:
                                cmd_parts.append(f'"{escape_ps_string(arg)}"')
                            else:
                                cmd_parts.append(escape_ps_string(arg))
                        cmd_str = ' '.join(cmd_parts)
                    
                    # If password provided, we'll need to use a different approach
                    # For now, we'll create a script that can use the password
                    password_note = ""
                    if ssh_password:
                        password_note = f"""
Write-Host "Password provided in GUI - using it for authentication..." -ForegroundColor Green
# Note: rsync/scp don't accept password via command line for security
# The password will be used via SSH_ASKPASS or similar mechanism if available
# Otherwise, you may still be prompted
"""
                    else:
                        password_note = """
Write-Host "You will be prompted for your SSH password." -ForegroundColor Green
Write-Host "Enter your password when prompted (it will not be visible)." -ForegroundColor Green
"""
                    
                    # Write PowerShell script
                    # Build PowerShell script with proper argument array
                    if use_rsync:
                        # Build arguments array for PowerShell
                        ps_args_array = []
                        for i, arg in enumerate(cmd[1:]):  # Skip executable
                            escaped = escape_ps_string(arg)
                            ps_args_array.append(f'    "{escaped}"')
                        ps_args_str = ',\n'.join(ps_args_array)
                        exe_escaped = escape_ps_string(cmd[0])
                        ps_exec_cmd = f"""$args = @(
{ps_args_str}
)
& "{exe_escaped}" $args"""
                    else:
                        # For scp
                        ps_args_array = []
                        for arg in cmd[1:]:
                            escaped = escape_ps_string(arg)
                            ps_args_array.append(f'    "{escaped}"')
                        ps_args_str = ',\n'.join(ps_args_array)
                        exe_escaped = escape_ps_string(cmd[0])
                        ps_exec_cmd = f"""$args = @(
{ps_args_str}
)
& "{exe_escaped}" $args"""
                    
                    ps_content = f"""# Transfer script - allows interactive password input
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "File Transfer ({transfer_method})" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Transferring: {local_dir}" -ForegroundColor Yellow
Write-Host "To: {remote_user}@{remote_host}:{remote_dir}" -ForegroundColor Yellow
Write-Host "SSH Port: {remote_port}" -ForegroundColor Yellow
Write-Host ""
{password_note}
Write-Host "Starting transfer..." -ForegroundColor Cyan
Write-Host ""

# Execute the command with proper argument array handling
{ps_exec_cmd}
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {{
    Write-Host ""
    Write-Host "✓ Transfer completed successfully!" -ForegroundColor Green
}} else {{
    Write-Host ""
    Write-Host "✗ Transfer failed! (Exit code: $exitCode)" -ForegroundColor Red
    Write-Host "Please check your connection and credentials." -ForegroundColor Yellow
}}

Write-Host ""
Write-Host "Press any key to close this window..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
"""
                    
                    with open(ps_script, "w", encoding="utf-8") as f:
                        f.write(ps_content)
                    
                    # Execute PowerShell script in a new window
                    subprocess.Popen(
                        [
                            "powershell.exe",
                            "-ExecutionPolicy", "Bypass",
                            "-NoExit",
                            "-File", ps_script
                        ],
                        cwd=script_dir
                    )
                    
                    progress_text.config(state=tk.NORMAL)
                    progress_text.insert(tk.END, "\n" + "-" * 60 + "\n")
                    progress_text.insert(tk.END, "✓ Transfer window opened.\n", "success")
                    progress_text.insert(tk.END, "Please enter your SSH password in the PowerShell window.\n")
                    progress_text.insert(tk.END, "Monitor the progress in that window.\n")
                    progress_text.config(state=tk.DISABLED)
                    progress_text.see(tk.END)
                    
                    # Re-enable start button
                    start_btn.config(state=tk.NORMAL)
                    cancel_btn.config(state=tk.DISABLED)
                else:
                    # Linux/macOS: Use subprocess with stdin for password
                    # This is more complex and may not work well, so we'll use a terminal too
                    import shlex
                    cmd_str = ' '.join([shlex.quote(arg) for arg in cmd])
                    subprocess.Popen(
                        ["xterm", "-e", f"bash -c '{cmd_str}; read -p \"Press Enter to close...\"'"],
                        cwd=os.path.dirname(os.path.abspath(__file__))
                    )
                    
                    progress_text.config(state=tk.NORMAL)
                    progress_text.insert(tk.END, "\n✓ Transfer window opened. Enter password in terminal.\n")
                    progress_text.config(state=tk.DISABLED)
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
    
    start_btn = tk.Button(buttons_frame, text="Start Transfer", command=start_transfer, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
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
