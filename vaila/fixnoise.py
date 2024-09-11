# fixnoise.py

import pandas as pd
import matplotlib.pyplot as plt
import sys
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


def read_csv_full(filename):
    try:
        return pd.read_csv(filename, delimiter=",")
    except Exception as e:
        raise Exception(f"Error reading the CSV file: {str(e)}")


def select_headers_and_load_data(file_path):
    """
    Displays a GUI to select the desired headers with Select All and Unselect All options
    and loads the corresponding data for the selected headers.
    """

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
        "Space + Left Click to select, Right Click to remove, press 'Enter' to finish"
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
        print("At least two points required.")
        sys.exit(1)

    indices = sorted([int(point[0]) for point in points])
    return indices


def replace_segments(data, indices, column_index):
    for i in range(0, len(indices), 2):  # Step of 2 to process pairs
        start, end = indices[i], indices[i + 1]
        data.iloc[start:end, column_index] = 0  # Only modify the specific column
    return data


def main():
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window
    file_path = filedialog.askopenfilename(
        title="Select CSV File", filetypes=[("CSV files", "*.csv")]
    )

    if not file_path:
        messagebox.showerror("Error", "No file selected.")
        return

    selected_headers, _ = select_headers_and_load_data(file_path)
    if not selected_headers:
        print("No headers selected.")
        return

    selected_column = selected_headers[0]  # Assume first column is selected

    data = read_csv_full(file_path)
    target_column_index = data.columns.get_loc(
        selected_column
    )  # Get the index of the selected column

    indices = makefig1(data.iloc[:, target_column_index])
    modified_data = replace_segments(data, indices, target_column_index)
    new_filename = file_path.replace(".csv", "_fixnoise.csv")
    modified_data.to_csv(new_filename, index=False)
    print(f"File saved as {new_filename}")


if __name__ == "__main__":
    main()
