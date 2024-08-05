import shutil
import os
import tkinter as tk
from tkinter import filedialog, messagebox

# Ensure the 'data', 'import', 'export', and 'results' directories exist
for directory in ['data', 'import', 'export', 'results']:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Create the directories list
directories = ['import', 'export', 'results'] + [os.path.join('data', d) for d in os.listdir('data') if os.path.isdir(os.path.join('data', d))]

def import_file():
    allowed_extensions = {'.avi', '.AVI', '.mp4', '.MP4', '.mov', '.MOV', '.mkv', '.MKV',
                          '.c3d', '.csv', '.txt', '.ref2d', '.dlt2d', '.2d', '.ref3d', '.3d', '.dlt3d'}
    
    def select_directory(title):
        directory_path = filedialog.askdirectory(title=title)
        return directory_path

    # Select source directory
    src = select_directory("Select the source directory")
    if not src:
        messagebox.showerror("Error", "No source directory selected.")
        return

    # Destination directory is always the 'import' directory in the current working directory
    dest = os.path.join(os.getcwd(), 'import')
    
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
                        shutil.copy2(os.path.join(root_dir, file), os.path.join(dest, file))
        elif os.path.splitext(item)[1].lower() in allowed_extensions:
            shutil.copy2(src_path, dest_path)

    messagebox.showinfo("Success", f"All allowed files from {src} have been imported to {dest}.")

def export_file():
    # Allow exporting from 'import', 'export', 'results', and 'data' directories
    allowed_directories = ['import', 'export', 'results', 'data']
    src_directory = filedialog.askdirectory(title="Select the source directory", initialdir=os.getcwd())
    
    # Check if the selected directory is in one of the allowed directories
    if not src_directory or not any(src_directory.startswith(os.path.join(os.getcwd(), d)) for d in allowed_directories):
        messagebox.showerror("Error", "No valid source directory selected.")
        return

    # Let the user choose the file extension to export
    file_type = filedialog.askstring("Input", "Enter the file extension to export (e.g., .csv):")
    if not file_type:
        messagebox.showerror("Error", "No file extension provided.")
        return

    # Select destination directory
    dest_directory = filedialog.askdirectory(title="Select the destination directory")
    if not dest_directory:
        messagebox.showerror("Error", "No destination directory selected.")
        return

    # Copy all files with the chosen extension from the source to the destination
    try:
        for item in os.listdir(src_directory):
            if item.endswith(file_type):
                src_path = os.path.join(src_directory, item)
                dest_path = os.path.join(dest_directory, item)
                shutil.copy2(src_path, dest_path)
        messagebox.showinfo("Success", f"All {file_type} files from {src_directory} have been exported to {dest_directory}")
    except Exception as e:
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
    allowed_directories = ['data', 'results', 'import', 'export']

    def select_directory(title):
        directory_path = filedialog.askdirectory(title=title)
        return directory_path

    src = select_directory("Select the source directory")
    if not src or not any(src.startswith(os.path.join(os.getcwd(), d)) for d in allowed_directories):
        messagebox.showerror("Error", "No valid source directory selected.")
        return

    dest = select_directory("Select the destination directory")
    if not dest or not any(dest.startswith(os.path.join(os.getcwd(), d)) for d in allowed_directories):
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
    allowed_extensions = {'.avi', '.AVI', '.mp4', '.MP4', '.mov', '.MOV', '.mkv', '.MKV'}
    allowed_directories = ['data', 'results', 'import', 'export']

    def select_directory(title):
        directory_path = filedialog.askdirectory(title=title)
        return directory_path

    selected_directory = select_directory("Select the directory to remove files from")
    if not selected_directory or not any(selected_directory.startswith(os.path.join(os.getcwd(), d)) for d in allowed_directories):
        messagebox.showerror("Error", "No valid directory selected.")
        return

    confirmation = messagebox.askyesno("Confirm", f"Are you sure you want to remove files from {selected_directory}?")
    if confirmation:
        for item in os.listdir(selected_directory):
            item_path = os.path.join(selected_directory, item)
            if os.path.isfile(item_path) and os.path.splitext(item_path)[1].lower() in allowed_extensions:
                os.remove(item_path)
        messagebox.showinfo("Success", f"Specified files in {selected_directory} have been removed.")


