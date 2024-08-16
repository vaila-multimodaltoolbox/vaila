import tkinter as tk
from tkinter import ttk


def get_user_inputs():
    def confirm():
        try:
            sample_rate = float(sample_rate_entry.get())
            file_type = file_type_entry.get().strip().lower()
            if file_type not in ["csv", "c3d"]:
                error_label.config(text="Please enter 'csv' or 'c3d' for file type.")
                return
            user_inputs["sample_rate"] = sample_rate
            user_inputs["file_type"] = file_type
            app.quit()  # Quit the main loop
        except ValueError:
            error_label.config(text="Please enter a valid float for sample rate.")

    app = tk.Tk()
    app.title("Select File Format and Sample Rate")

    user_inputs = {}  # Dictionary to hold user inputs

    # Create a LabelFrame for entering sample rate
    sample_rate_frame = ttk.LabelFrame(app, text="Enter sample rate:")
    sample_rate_frame.pack(padx=10, pady=10)

    # Create an Entry widget for sample rate
    sample_rate_entry = ttk.Entry(sample_rate_frame)
    sample_rate_entry.pack(padx=10, pady=5)

    # Create a LabelFrame for entering file type
    file_type_frame = ttk.LabelFrame(app, text="Enter file type (csv or c3d):")
    file_type_frame.pack(padx=10, pady=10)

    # Create an Entry widget for file type
    file_type_entry = ttk.Entry(file_type_frame)
    file_type_entry.pack(padx=10, pady=5)

    # Create a confirm button to collect the sample rate and file type
    confirm_button = ttk.Button(app, text="Confirm", command=confirm)
    confirm_button.pack(pady=10)

    # Create a label to display errors
    error_label = tk.Label(app, text="", fg="red")
    error_label.pack()

    app.mainloop()
    app.destroy()  # Close the GUI window after the main loop ends

    return user_inputs
