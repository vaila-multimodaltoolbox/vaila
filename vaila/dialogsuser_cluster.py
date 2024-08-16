"""
Nome: dialogsuser_cluster.py
Data: 2024-07-29
Versão: 1.1
Descrição: Este módulo solicita ao usuário detalhes de configuração, incluindo taxa de amostragem, configuração e nomes dos clusters.
"""

import tkinter as tk
from tkinter import ttk


def get_user_inputs():
    def confirm(event=None):
        try:
            sample_rate = float(sample_rate_entry.get())
            cluster1_config = cluster1_config_entry.get().strip().upper()
            cluster2_config = cluster2_config_entry.get().strip().upper()
            cluster1_name = cluster1_name_entry.get().strip()
            cluster2_name = cluster2_name_entry.get().strip()

            if not cluster1_name or not cluster2_name:
                error_label.config(text="Cluster names cannot be empty.")
                return

            user_inputs["sample_rate"] = sample_rate
            user_inputs["cluster1_config"] = cluster1_config
            user_inputs["cluster2_config"] = cluster2_config
            user_inputs["cluster1_name"] = cluster1_name
            user_inputs["cluster2_name"] = cluster2_name

            app.quit()  # Quit the main loop
        except ValueError:
            error_label.config(text="Please enter a valid float for sample rate.")

    app = tk.Tk()
    app.title("Enter Configuration Details")

    user_inputs = {}  # Dictionary to hold user inputs

    # Create a frame to hold the inputs and the image
    main_frame = ttk.Frame(app)
    main_frame.pack(padx=10, pady=10)

    # Create a frame for inputs
    input_frame = ttk.Frame(main_frame)
    input_frame.grid(row=0, column=0, padx=10, pady=10)

    # Create a LabelFrame for entering sample rate
    sample_rate_frame = ttk.LabelFrame(input_frame, text="Enter sample rate:")
    sample_rate_frame.pack(padx=10, pady=10)

    # Create an Entry widget for sample rate
    sample_rate_entry = ttk.Entry(sample_rate_frame)
    sample_rate_entry.pack(padx=10, pady=5)

    # Create a LabelFrame for entering Cluster 1 configuration
    cluster1_config_frame = ttk.LabelFrame(
        input_frame, text="Cluster 1 Configuration (A/B/C/D):"
    )
    cluster1_config_frame.pack(padx=10, pady=10)

    # Create an Entry widget for Cluster 1 configuration
    cluster1_config_entry = ttk.Entry(cluster1_config_frame)
    cluster1_config_entry.pack(padx=10, pady=5)

    # Create a LabelFrame for entering Cluster 1 name
    cluster1_name_frame = ttk.LabelFrame(input_frame, text="Cluster 1 Name:")
    cluster1_name_frame.pack(padx=10, pady=10)

    # Create an Entry widget for Cluster 1 name
    cluster1_name_entry = ttk.Entry(cluster1_name_frame)
    cluster1_name_entry.pack(padx=10, pady=5)

    # Create a LabelFrame for entering Cluster 2 configuration
    cluster2_config_frame = ttk.LabelFrame(
        input_frame, text="Cluster 2 Configuration (A/B/C/D):"
    )
    cluster2_config_frame.pack(padx=10, pady=10)

    # Create an Entry widget for Cluster 2 configuration
    cluster2_config_entry = ttk.Entry(cluster2_config_frame)
    cluster2_config_entry.pack(padx=10, pady=5)

    # Create a LabelFrame for entering Cluster 2 name
    cluster2_name_frame = ttk.LabelFrame(input_frame, text="Cluster 2 Name:")
    cluster2_name_frame.pack(padx=10, pady=10)

    # Create an Entry widget for Cluster 2 name
    cluster2_name_entry = ttk.Entry(cluster2_name_frame)
    cluster2_name_entry.pack(padx=10, pady=5)

    # Create a confirm button to collect the inputs
    confirm_button = ttk.Button(input_frame, text="Confirm", command=confirm)
    confirm_button.pack(pady=10)

    # Create a label to display errors
    error_label = tk.Label(input_frame, text="", fg="red")
    error_label.pack()

    # Bind the Return key (Enter) to the confirm function
    app.bind("<Return>", confirm)

    app.mainloop()
    app.destroy()  # Close the GUI window after the main loop ends

    return user_inputs
