"""
fixnoise_batch.py
version 0.02

This script is designed to batch process CSV files, applying a noise filter (fixnoise) interactively.
The user selects a directory containing CSV files and, from the first file, chooses the headers of interest.
The script applies the same header selection for all subsequent files and allows the user to mark start and end points
for specific segment replacements. If no points are selected, the file is saved without changes.

Author: Prof. Dr. Paulo R. P. Santiago
Date: September 13, 2024

Modifications:
- Added batch processing for multiple CSV files.
- Unified header selection based on the first file.
- Saved processed files in a new directory with a timestamp.

This script was created to facilitate the processing of large volumes of experimental data, allowing fine adjustments
through a user-friendly graphical interface.

Instructions:
1. Run the script.
2. Select the directory containing the CSV files to be processed.
3. Follow the instructions in the graphical interface to choose headers and adjustment points.

"""

import pandas as pd
import matplotlib.pyplot as plt
from tkinter import (
    Tk,
    Toplevel,
    Canvas,
    Scrollbar,
    Frame,
    Button,
    Checkbutton,
    BooleanVar,
    messagebox,
    filedialog,
)
import os
import glob
from datetime import datetime

# Print the directory and name of the script being executed
print(f"Running script: {os.path.basename(__file__)}")
print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")


def read_csv_full(filename):
    try:
        return pd.read_csv(filename, delimiter=",")
    except Exception as e:
        raise Exception(f"Error reading the CSV file: {str(e)}")


def select_headers_and_load_data(file_path):
    def get_csv_headers(file_path):
        df = pd.read_csv(file_path)
        return list(df.columns), df

    headers, df = get_csv_headers(file_path)
    selected_headers = []

    def on_select():
        nonlocal selected_headers
        selected_headers = [
            header for header, var in zip(headers, header_vars) if var.get()
        ]
        selection_window.quit()
        selection_window.destroy()

    def select_all():
        for var in header_vars:
            var.set(True)

    def unselect_all():
        for var in header_vars:
            var.set(False)

    selection_window = Toplevel()
    selection_window.title("Select Headers")
    selection_window.geometry(
        f"{selection_window.winfo_screenwidth()}x{int(selection_window.winfo_screenheight()*0.9)}"
    )

    canvas = Canvas(selection_window)
    scrollbar = Scrollbar(selection_window, orient="vertical", command=canvas.yview)
    scrollable_frame = Frame(canvas)

    scrollable_frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    header_vars = [BooleanVar() for _ in headers]

    num_columns = 7  # Number of columns for header labels

    for i, label in enumerate(headers):
        chk = Checkbutton(scrollable_frame, text=label, variable=header_vars[i])
        chk.grid(row=i // num_columns, column=i % num_columns, sticky="w")

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    btn_frame = Frame(selection_window)
    btn_frame.pack(side="right", padx=10, pady=10, fill="y", anchor="center")

    btn_select_all = Button(btn_frame, text="Select All", command=select_all)
    btn_select_all.pack(side="top", pady=5)

    btn_unselect_all = Button(btn_frame, text="Unselect All", command=unselect_all)
    btn_unselect_all.pack(side="top", pady=5)

    btn_select = Button(btn_frame, text="Confirm", command=on_select)
    btn_select.pack(side="top", pady=5)

    selection_window.mainloop()

    if not selected_headers:
        messagebox.showinfo("Info", "No headers were selected.")
        return None, None

    selected_data = df[selected_headers]
    return selected_headers, selected_data


def makefig1(data):
    fig1, ax1 = plt.subplots()
    ax1.plot(data * -1)
    ax1.set_title(
        "Select range to remove noise. Hold Space + Left Click to mark, Right Click to remove, 'Enter' to confirm."
    )
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Force Value")
    ax1.grid(True)

    points = []
    space_held = False

    def on_key_press(event):
        nonlocal space_held
        if event.key == " ":
            space_held = True
        elif event.key == "enter":
            plt.close(fig1)

    def on_key_release(event):
        nonlocal space_held
        if event.key == " ":
            space_held = False

    def onclick(event):
        if space_held and event.button == 1:
            x_value = event.xdata
            if x_value is not None:
                points.append((x_value, event.ydata))
                ax1.axvline(x=x_value, color="red", linestyle="--")
                fig1.canvas.draw()
        elif event.button == 3:
            if points:
                points.pop()
                ax1.cla()
                ax1.plot(data)
                ax1.grid(True)
                for point in points:
                    ax1.axvline(x=point[0], color="red", linestyle="--")
                fig1.canvas.draw()

    fig1.canvas.mpl_connect("button_press_event", onclick)
    fig1.canvas.mpl_connect("key_press_event", on_key_press)
    fig1.canvas.mpl_connect("key_release_event", on_key_release)

    plt.show(block=True)

    if len(points) < 2:
        print("No points selected. Saving without changes.")
        return []

    indices = sorted([int(point[0]) for point in points])
    return indices


def replace_segments(data, indices, column_index):
    for i in range(0, len(indices), 2):  # Step of 2 to process pairs
        start, end = indices[i], indices[i + 1]
        data.iloc[start:end, column_index] = 0  # Only modify the specific column
    return data


def process_files_in_directory(directory):
    csv_files = sorted(glob.glob(os.path.join(directory, "*.csv")))

    if not csv_files:
        print("No CSV files found in the directory.")
        return

    # Create a new directory to save the processed files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = os.path.join(directory, f"vaila_fixnoise_{timestamp}")
    os.makedirs(output_directory, exist_ok=True)

    selected_headers = None
    for idx, file_path in enumerate(csv_files):
        print(f"Processing {file_path}")

        if idx == 0:
            # For the first file, open the dialog to select headers
            selected_headers, _ = select_headers_and_load_data(file_path)
            if selected_headers is None or len(selected_headers) == 0:
                messagebox.showerror("Error", "No headers were selected.")
                return  # Exit the function if no headers are selected
        else:
            print(f"Using previously selected headers: {selected_headers}")

        if selected_headers is None or len(selected_headers) == 0:
            messagebox.showerror("Error", "No headers were selected.")
            return  # Exit the function if no headers are selected

        selected_column = selected_headers[0]  # Assume the first column is selected

        data = read_csv_full(file_path)
        if selected_column not in data.columns:
            messagebox.showerror(
                "Error", f"Column '{selected_column}' not found in the data."
            )
            return  # Exit if the column is not found

        target_column_index = data.columns.get_loc(
            selected_column
        )  # Get the index of the selected column
        indices = makefig1(
            data.iloc[:, target_column_index]
        )  # Access the column safely
        if indices:  # Only if points were selected
            modified_data = replace_segments(data, indices, target_column_index)
            new_filename = os.path.join(
                output_directory,
                os.path.basename(file_path).replace(".csv", "_fixnoise.csv"),
            )
            modified_data.to_csv(new_filename, index=False)
            print(f"File saved as {new_filename}")
        else:
            # Save the file without changes
            new_filename = os.path.join(
                output_directory,
                os.path.basename(file_path).replace(".csv", "_nochange.csv"),
            )
            data.to_csv(new_filename, index=False)
            print(f"No changes made. File saved as {new_filename}")


def main():
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window
    directory_path = filedialog.askdirectory(
        title="Select Directory Containing CSV Files"
    )

    if not directory_path:
        messagebox.showerror("Error", "No directory selected.")
        return

    process_files_in_directory(directory_path)


if __name__ == "__main__":
    main()
