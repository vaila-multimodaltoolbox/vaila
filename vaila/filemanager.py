"""
File: filemanager.py

Description:
This script, named filemanager.py, is designed to manage files and directories efficiently. It supports various operations, including importing, converting, exporting, copying, moving, removing, and finding files with specific patterns. The script leverages the Tkinter graphical interface to facilitate user interaction, enabling the selection of files and directories through an easy-to-use GUI.

Version: 1.2
Last Updated: 2024-08-29
Author: Prof. Paulo Santiago

Main Features:
- Import specific files from a selected directory into a predefined structure.
- Convert various file types (e.g., c3d, yaml, xml, html, h5, etc.) to CSV format.
- Copy files from one location to another within the allowed directories.
- Move files between predefined directories.
- Remove files with specific extensions or directory names from selected directories.
- Find files or directories based on user-defined patterns and count the number of files with a specified extension.

Usage Notes:
- A directory named 'vaila_export', 'vaila_copy', 'vaila_move', or 'vaila_import' will be automatically created within the chosen destination directory for the respective operations.
- The script ensures that essential directories ('data', 'import', 'export', 'results') exist before performing operations.

Dependencies:
- Python 3.x
- Conda environment with the following packages: pandas, ezc3d, yaml, toml, lxml, BeautifulSoup4, pickle5, hdf5plugin

How to Run:
- Execute the script in a Python environment managed by Conda that includes the necessary packages.
- Follow on-screen prompts to perform the desired file operations.
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
        messagebox.showerror("Error", f"Error copying files: {e}")


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
        process_move(src_directory, file_extension, patterns)

    submit_button = tk.Button(pattern_window, text="Submit", command=on_submit)
    submit_button.pack()

    pattern_window.mainloop()


def process_move(src_directory, file_extension, patterns):
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


def import_file():
    root = tk.Tk()
    root.withdraw()

    # Seleção do diretório fonte
    src_directory = filedialog.askdirectory(title="Select Source Directory")
    if not src_directory:
        messagebox.showerror("Error", "No source directory selected.")
        return

    # Seleção do diretório de destino
    dest_directory = filedialog.askdirectory(title="Select Destination Directory")
    if not dest_directory:
        messagebox.showerror("Error", "No destination directory selected.")
        return

    dest_directory = os.path.join(dest_directory, "vaila_import")
    os.makedirs(dest_directory, exist_ok=True)

    # Solicitar a extensão do arquivo ao usuário
    file_extension = simpledialog.askstring(
        "File Extension",
        "Enter the file extension to process (e.g., .csv, .json, .xml):",
    )
    if not file_extension:
        messagebox.showerror("Error", "No file extension provided.")
        return

    # Filtrar arquivos pela extensão fornecida
    files = sorted(
        [
            f
            for f in os.listdir(src_directory)
            if f.endswith(file_extension)
            and os.path.isfile(os.path.join(src_directory, f))
        ]
    )
    if not files:
        messagebox.showerror(
            "Error",
            f"No files with extension {file_extension} found in the source directory.",
        )
        return

    for file in files:
        src_file_path = os.path.join(src_directory, file)
        file_name, file_extension = os.path.splitext(file)

        print(f"Processing file: {src_file_path}")

        def save_csv(dataframe, base_name, suffix):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest_file = os.path.join(
                dest_directory, f"{base_name}_{suffix}_{timestamp}.csv"
            )
            dataframe.to_csv(dest_file, index=False)
            print(f"Saved: {dest_file}")

        try:
            # Processamento de arquivos CSV, TSV, TXT
            if file_extension in [".csv", ".tsv", ".txt"]:
                df = pd.read_csv(src_file_path, sep=None, engine="python")
                save_csv(df, file_name, "data")

            # Processamento de arquivos Excel
            elif file_extension in [".xlsx", ".xls", ".ods"]:
                df = pd.read_excel(src_file_path)
                save_csv(df, file_name, "data")

            # Processamento de arquivos HDF5
            elif file_extension == ".h5":
                with h5py.File(src_file_path, "r") as hdf:
                    for key in hdf.keys():
                        data = hdf[key][:]
                        df = pd.DataFrame(data)
                        save_csv(df, file_name, key)

            # Processamento de arquivos Pickle
            elif file_extension == ".pickle":
                with open(src_file_path, "rb") as f:
                    data = pickle.load(f)
                df = pd.DataFrame(data)
                save_csv(df, file_name, "data")

            # Processamento de arquivos YAML
            elif file_extension in [".yml", ".yaml"]:
                with open(src_file_path, "r") as f:
                    data = yaml.safe_load(f)
                df = pd.DataFrame(data)
                save_csv(df, file_name, "data")

            # Processamento de arquivos TOML
            elif file_extension == ".toml":
                data = toml.load(src_file_path)
                df = pd.DataFrame(data)
                save_csv(df, file_name, "data")

            # Processamento de arquivos HTML
            elif file_extension == ".html":
                try:
                    with open(src_file_path, "r") as f:
                        soup = BeautifulSoup(f, "html.parser")
                    tables = pd.read_html(str(soup))
                    for i, table in enumerate(tables):
                        table_name = (
                            table.columns[0] if len(table.columns) > 0 else f"table_{i}"
                        )
                        save_csv(table, file_name, table_name)
                except Exception as e:
                    print(f"Error processing HTML file {src_file_path}: {e}")
                    messagebox.showerror("Error", f"Failed to process HTML file: {e}")

            # Processamento de arquivos XML
            elif file_extension == ".xml":
                try:
                    with open(src_file_path, "r") as f:
                        tree = etree.parse(f)
                    root = tree.getroot()
                    data = []
                    columns = []
                    for element in root.iter():
                        if element.tag not in columns:
                            columns.append(element.tag)
                        data.append({element.tag: element.text})
                    df = pd.DataFrame(data)
                    save_csv(df, file_name, "data")
                except Exception as e:
                    print(f"Error processing XML file {src_file_path}: {e}")
                    messagebox.showerror("Error", f"Failed to process XML file: {e}")

            # Processamento de arquivos C3D
            elif file_extension == ".c3d":
                try:
                    c3d_data = ezc3d.c3d(src_file_path)
                    point_data = c3d_data["data"]["points"]
                    marker_labels = c3d_data["parameters"]["POINT"]["LABELS"]["value"]
                    markers = point_data[0:3, :, :].T.reshape(
                        -1, len(marker_labels) * 3
                    )
                    marker_freq = c3d_data["header"]["points"]["frame_rate"]

                    marker_columns = [
                        f"{label}_{axis}"
                        for label in marker_labels
                        for axis in ["X", "Y", "Z"]
                    ]
                    markers_df = pd.DataFrame(markers, columns=marker_columns)

                    num_marker_frames = markers_df.shape[0]
                    marker_time_column = pd.Series(
                        [f"{i / marker_freq:.3f}" for i in range(num_marker_frames)],
                        name="Time",
                    )
                    markers_df.insert(0, "Time", marker_time_column)
                    save_csv(markers_df, file_name, "markers")

                    analogs = c3d_data["data"]["analogs"].squeeze(axis=0).T
                    analog_labels = c3d_data["parameters"]["ANALOG"]["LABELS"]["value"]
                    analogs_df = pd.DataFrame(analogs, columns=analog_labels)

                    analog_freq = c3d_data["header"]["analogs"]["frame_rate"]
                    num_analog_frames = analogs_df.shape[0]
                    analog_time_column = pd.Series(
                        [f"{i / analog_freq:.3f}" for i in range(num_analog_frames)],
                        name="Time",
                    )
                    analogs_df.insert(0, "Time", analog_time_column)
                    save_csv(analogs_df, file_name, "analogs")
                except Exception as e:
                    print(f"Error processing C3D file {src_file_path}: {e}")
                    messagebox.showerror("Error", f"Failed to process C3D file: {e}")

            # Processamento de arquivos JSON
            elif file_extension == ".json":
                try:
                    with open(src_file_path, "r") as f:
                        data = json.load(f)

                    print("JSON data loaded successfully.")

                    timeseries = data.get("data", {}).get("timeseries", [])
                    print(f"Found {len(timeseries)} timeseries.")

                    for series in timeseries:
                        time = series.get("time", [])
                        positions = series.get("data", {}).get("0", [])

                        if not time or not positions:
                            print(
                                f"No valid time or position data found in series: {series.get('name', 'Unnamed Series')}"
                            )
                            continue

                        rows = []
                        for t, pos in zip(time, positions):
                            row = {"Time": t}
                            if isinstance(pos, list) and len(pos) == 2:
                                row["p1_x"] = pos[0]
                                row["p1_y"] = pos[1]
                            else:
                                print(f"Invalid position data: {pos} for time {t}")
                            rows.append(row)

                        if rows:
                            df = pd.DataFrame(rows)
                            series_name = series.get("name", "data").replace(" ", "_")
                            save_csv(df, file_name, series_name)
                        else:
                            print(
                                f"No data to save for series: {series.get('name', 'Unnamed Series')}"
                            )

                except Exception as e:
                    print(f"Error processing JSON file {src_file_path}: {e}")
                    messagebox.showerror("Error", f"Failed to process JSON file: {e}")

            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

        except Exception as e:
            print(f"Error processing {src_file_path}: {e}")
            messagebox.showerror("Error", f"Failed to process {src_file_path}: {e}")

    messagebox.showinfo(
        "Success", f"Files have been processed and saved to {dest_directory}."
    )


def rename_files():
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
    # Initialize Tkinter root
    root = Tk()
    root.withdraw()  # Hide the root window

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
        ssh.connect(
            remote_host,
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
                scp.put(
                    src_path, remote_path=remote_file, recursive=os.path.isdir(src_path)
                )
                messagebox.showinfo(
                    "Success",
                    f"File or directory successfully uploaded to {remote_host}:{remote_file}",
                )
            else:
                # Download the file or directory
                scp.get(remote_file, local_path=dest_path, recursive=True)
                messagebox.showinfo(
                    "Success",
                    f"File or directory successfully downloaded from {remote_host}:{remote_file} to {dest_path}",
                )

    except Exception as e:
        # Show an error message if something goes wrong
        messagebox.showerror("Error", f"Error during file transfer: {e}")

    finally:
        # Close the SSH connection
        ssh.close()
