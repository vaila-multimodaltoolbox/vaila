"""
================================================================================
File Manager - Comprehensive File and Directory Management Tool
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Date: 2024-08-29
Version: 1.2

Description:
------------
This Python script provides a comprehensive tool for managing files and directories. 
It supports importing, converting, exporting, copying, moving, removing, and finding 
files based on user-defined patterns. The script includes a user-friendly GUI, built 
with Tkinter, that allows users to perform operations on files and directories easily.

Main Features:
--------------
1. **File Import**: Import specific file types from selected directories into a predefined 
   structure. Supported file types include `.csv`, `.mat`, `.tsv`, `.html`, `.xml`, `.xlsx`, etc.
2. **File Conversion**: Convert various file types (e.g., `.c3d`, `.yaml`, `.xml`, `.html`, `.h5`) 
   into CSV format.
3. **File Export**: Export files to various formats such as CSV, FBX, and TRC.
4. **File Copy/Move**: Copy or move files based on specified patterns and file extensions.
5. **File Removal**: Remove files with specific extensions or directory names from selected 
   directories, with safeguards to prevent accidental deletion of critical files.
6. **File Search**: Find files or directories based on user-defined patterns and count files 
   with a specified extension.
7. **Batch Processing**: Includes batch processing of files, such as CSV file splitting by 
   devices using a VICON Nexus CSV processing module.

Usage Notes:
------------
- The GUI enables the selection of files and directories for each operation.
- Directories like 'vaila_export', 'vaila_copy', 'vaila_move', or 'vaila_import' are automatically 
  created within the destination directory for respective operations.
- The script ensures essential directories (e.g., 'data', 'import', 'export', 'results') are created 
  before performing operations.

Dependencies:
-------------
- Python 3.x
- Required Libraries: pandas, ezc3d, yaml, toml, lxml, BeautifulSoup4, pickle5, hdf5plugin, paramiko, scp

How to Run:
-----------
- Run the script in a Python environment managed by Conda or with the required packages installed.
- Follow the on-screen prompts to perform the desired file operations.

Changelog:
----------
Version 1.2 - 2024-08-29:
    - Added batch processing for VICON Nexus CSV files.
    - Expanded import options to include `.html`, `.xml`, and `.xlsx`.
    - Improved file copy and move functionality with pattern matching.
    - Updated file removal functionality with safeguards to prevent critical file deletion.

License:
--------
This script is licensed under the MIT License.

Disclaimer:
-----------
This script is provided "as is" without warranty of any kind. The author is not responsible for any 
damage or loss resulting from the use of this script.
================================================================================
"""


import shutil
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Tk
import time
import pandas as pd
import ezc3d
import pickle  # pickle native in Python 3.11
import yaml
import toml
from lxml import etree
from bs4 import BeautifulSoup
from datetime import datetime
import h5py
import json
import fnmatch
import paramiko
from scp import SCPClient


def copy_file():
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

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

    pattern_text = tk.Text(pattern_window, height=44, width=60)
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
        messagebox.showerror(
            "Error", "Attempting to remove a system directory is not allowed."
        )
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
            if not hasattr(load_vicon_csv_split_batch, 'process_csv_files_first_level'):
                raise AttributeError("The module 'load_vicon_csv_split_batch' does not have 'process_csv_files_first_level' function")
    
            # Ask the user to select the source and output directories
            src_directory, output_directory = load_vicon_csv_split_batch.select_directory()
    
            # Run the batch processing function from load_vicon_csv_split_batch.py
            load_vicon_csv_split_batch.process_csv_files_first_level(src_directory, output_directory)
    
            # Show success message once processing is completed
            messagebox.showinfo("Success", "Batch processing of VICON Nexus CSV files completed successfully.")
    
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
        ("Data Import: HTML", data_import_html),          # New HTML button
        ("Data Import: XML", data_import_xml),            # New XML button
        ("Data Import: XLSX", data_import_xlsx),          # New XLSX button
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

    # Prompt the user to select the directory containing the files to rename
    directory = filedialog.askdirectory(title="Select Directory with Files to Rename")

    if not directory:
        messagebox.showerror("Error", "No directory selected.")
        return

    # Prompt the user to enter the text to be replaced and the text to replace it with
    text_to_replace = simpledialog.askstring(
        "Text to Replace", "Enter the text to replace:"
    )
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
                    os.rename(
                        os.path.join(root, filename), os.path.join(root, new_filename)
                    )

        # Show a success message after renaming is complete
        messagebox.showinfo("Success", "Files have been renamed successfully.")

    except Exception as e:
        # Show an error message if something goes wrong
        messagebox.showerror("Error", f"Error renaming files: {e}")


def tree_file():
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

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
                        relative_path = os.path.relpath(
                            os.path.join(root, file), src_directory
                        )
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
                    relative_path = os.path.relpath(
                        os.path.join(root, name), src_directory
                    )
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
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    # Initialize Tkinter root
    root = Tk()
    root.withdraw()  # Hide the root window
    root.tk.call('wm', 'attributes', '.', '-topmost', '1')  # Keep window on top
    
    # Prompt the user to select Upload or Download
    transfer_type = simpledialog.askstring(
        "Transfer Type", "Enter 'upload' to send files or 'download' to receive files:"
    )

    if transfer_type not in ["upload", "download"]:
        messagebox.showerror("Error", "Invalid transfer type provided.")
        return

    # Select file or directory for upload or specify destination directory for download
    if transfer_type == "upload":
        src_path = filedialog.askopenfilename(
            title="Select the file or directory to transfer"
        )
        if not src_path:
            messagebox.showerror("Error", "No file or directory selected.")
            return
    else:  # download
        remote_file = simpledialog.askstring(
            "Remote File", "Enter the remote file or directory path:"
        )
        if not remote_file:
            messagebox.showerror("Error", "No remote file or directory path provided.")
            return
        dest_path = filedialog.askdirectory(title="Select the destination directory")
        if not dest_path:
            messagebox.showerror("Error", "No destination directory selected.")
            return

    # Remote server details
    remote_host = simpledialog.askstring(
        "Remote Host", "Enter the remote host (e.g., example.com):"
    )
    if not remote_host:
        messagebox.showerror("Error", "No remote host provided.")
        return

    remote_port = simpledialog.askinteger(
        "Remote Port", "Enter the SSH port (default is 22):", initialvalue=22
    )
    if not remote_port:
        remote_port = 22

    remote_user = simpledialog.askstring("Remote User", "Enter the SSH username:")
    if not remote_user:
        messagebox.showerror("Error", "No SSH username provided.")
        return

    remote_password = simpledialog.askstring(
        "Remote Password", "Enter the SSH password:", show="*"
    )
    if remote_password is None:
        messagebox.showerror("Error", "No SSH password provided.")
        return

    try:
        # Create an SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Debugging output
        print(f"Connecting to {remote_host}:{remote_port} as {remote_user}")

        ssh.connect(
            hostname=remote_host,
            port=remote_port,
            username=remote_user,
            password=remote_password,
        )

        # Create an SCP client
        with SCPClient(
            ssh.get_transport(), compress=True
        ) as scp:  # Compression enabled

            if transfer_type == "upload":
                # Upload the file or directory
                print(f"Uploading {src_path} to {remote_host}:{remote_file}")
                scp.put(
                    src_path, remote_path=remote_file, recursive=os.path.isdir(src_path)
                )
                messagebox.showinfo(
                    "Success",
                    f"File or directory successfully uploaded to {remote_host}:{remote_file}",
                )
            else:
                # Download the file or directory
                print(f"Downloading {remote_file} from {remote_host} to {dest_path}")
                scp.get(remote_file, local_path=dest_path, recursive=True)
                messagebox.showinfo(
                    "Success",
                    f"File or directory successfully downloaded from {remote_host}:{remote_file} to {dest_path}",
                )

    except paramiko.AuthenticationException as auth_error:
        messagebox.showerror(
            "Authentication Error", f"Authentication failed: {auth_error}"
        )
        print(f"Authentication failed: {auth_error}")

    except paramiko.SSHException as ssh_error:
        messagebox.showerror("SSH Error", f"SSH connection error: {ssh_error}")
        print(f"SSH connection error: {ssh_error}")

    except Exception as e:
        # Show an error message if something goes wrong
        messagebox.showerror("Error", f"Error during file transfer: {e}")
        print(f"Error during file transfer: {e}")

    finally:
        # Close the SSH connection
        ssh.close()
