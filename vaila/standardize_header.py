"""
Script Name: standardize_header.py
Version: 1.1
Author: [Your Name]

Description:
-------------
This script provides a tool to standardize headers across multiple CSV files in a selected directory.
It allows the user to select a specific line to use as the header and remove unwanted rows from each file.

Features:
---------
- Batch processing of CSV files in a selected directory.
- User-friendly GUI to select header lines and rows to delete.
- Handles parsing errors and ensures consistent formatting across files.
- Saves the processed files in a new directory with a timestamp.

Changelog:
----------
- Version 1.1: Enhanced error handling and user prompts for selecting header lines and rows to delete.
- Version 1.0: Initial release with basic functionality for header standardization.

Usage:
------
- Run the script to launch a GUI for selecting the header line and rows to delete.
- Select the directory containing CSV files and follow the prompts to process the files.

Requirements:
-------------
- Python 3.x
- pandas
- tkinter
"""

import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from datetime import datetime


def standardize_header():
    """
    Standardize headers and clean rows in all CSV files within a selected directory.
    Allows the user to select the header line and specify rows to delete.
    """
    # Open a window to select the directory containing CSV files
    directory_path = filedialog.askdirectory(title="Select Directory Containing CSV Files")
    if not directory_path:
        return

    # Find all CSV files in the directory, sorted alphabetically
    csv_files = sorted([f for f in os.listdir(directory_path) if f.endswith(".csv")])
    if not csv_files:
        messagebox.showerror("Error", "No CSV files found in the selected directory.")
        return

    # Create a new directory with a timestamp to save the standardized files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_directory = os.path.join(directory_path, f"vaila_stand_header_{timestamp}")
    os.makedirs(save_directory, exist_ok=True)

    # Read the first file to display the first few lines
    first_file_path = os.path.join(directory_path, csv_files[0])
    try:
        with open(first_file_path, "r") as file:
            lines = file.readlines()
    except Exception as e:
        messagebox.showerror("Error", f"Unable to read the file: {e}")
        return

    # Function to choose the header line and row deletion range
    def choose_header_line():
        # Create a window to select the header line and rows to delete
        header_window = tk.Toplevel()
        header_window.title("Select Header Line and Rows to Delete")

        # Show the first lines in a text box without line wrapping
        header_text = tk.Text(header_window, width=220, height=50, wrap=tk.NONE)
        header_text.pack()
        for i, line in enumerate(lines):
            header_text.insert(tk.END, f"{i}: {line}")

        # Entry to select the header line number
        header_line_label = tk.Label(header_window, text="Enter the line number to use as the header (e.g., 3):")
        header_line_label.pack()
        header_line_entry = tk.Entry(header_window)
        header_line_entry.pack()

        # Entry to select the range or single row to delete
        delete_rows_label = tk.Label(header_window, text="Enter a range (start:end) of rows to delete (e.g., 4:5) or a single row number (e.g., 4):")
        delete_rows_label.pack()
        delete_rows_entry = tk.Entry(header_window)
        delete_rows_entry.pack()

        def confirm_selection():
            """
            Confirm the user's selection of the header line and rows to delete, and process all CSV files accordingly.
            """
            # Get the line selected by the user for the header
            try:
                selected_line = int(header_line_entry.get())
                if selected_line < 0 or selected_line >= len(lines):
                    messagebox.showerror("Error", "Invalid line number.")
                    return
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid line number for the header.")
                return

            # Get the range or single row to delete
            delete_rows = delete_rows_entry.get()
            rows_to_delete = []

            if delete_rows:
                try:
                    # Check if it's a range of rows
                    if ":" in delete_rows:
                        start, end = map(int, delete_rows.split(":"))
                        if start < 0 or end < 0 or start > end:
                            raise ValueError
                        rows_to_delete.extend(range(start, end + 1))
                    else:
                        # Assume it's a single line
                        single_row = int(delete_rows)
                        if single_row < 0:
                            raise ValueError
                        rows_to_delete.append(single_row)
                except ValueError:
                    messagebox.showerror("Error", "Invalid input for rows to delete. Use 'start:end' or a single row number.")
                    return

            # Process all CSV files in the directory
            for file_name in csv_files:
                file_path = os.path.join(directory_path, file_name)
                try:
                    # Read the file, skipping problematic lines
                    with open(file_path, "r") as file:
                        all_lines = file.readlines()

                    # Try reading the file with the selected header line and cleaning rows
                    try:
                        df = pd.read_csv(file_path, header=selected_line, skip_blank_lines=False)
                    except pd.errors.ParserError as pe:
                        # Handle tokenizing errors by cleaning lines manually
                        cleaned_lines = []
                        for line in all_lines:
                            if len(line.split(",")) == len(all_lines[selected_line].split(",")):
                                cleaned_lines.append(line)
                        df = pd.read_csv(pd.compat.StringIO("".join(cleaned_lines)), header=selected_line)

                    # Calculate the actual row indices to delete in the DataFrame
                    rows_to_delete_after_header = [row - (selected_line + 1) for row in rows_to_delete if row > selected_line]

                    # Delete the specified rows if they exist
                    if rows_to_delete_after_header:
                        df.drop(rows_to_delete_after_header, inplace=True)

                    # Remove any trailing empty rows
                    df = df.dropna(how='all')

                    # Save the standardized file in the new directory
                    base_name = os.path.splitext(file_name)[0]
                    new_file_name = f"{base_name}_vaila_stand_{timestamp}.csv"
                    new_file_path = os.path.join(save_directory, new_file_name)
                    df.to_csv(new_file_path, index=False)

                except Exception as e:
                    messagebox.showerror("Error", f"Error processing {file_name}: {e}")

            messagebox.showinfo("Success", f"Files have been standardized and saved in {save_directory}")
            header_window.destroy()

        confirm_button = tk.Button(header_window, text="Confirm", command=confirm_selection)
        confirm_button.pack()

    choose_header_line()

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window
    standardize_header()
