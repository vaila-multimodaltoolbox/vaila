"""
vaila.py

Name: Paulo Santiago
Date: 29/07/2024

Description:
Main application script for the vailá Multimodal Toolbox. This script provides
a Tkinter-based GUI to access various tools and functionalities for multimodal
analysis, including file management, data analysis, and visualization.

Version: 0.2

Changelog:
v0.2 - Added Plot 2D button for 2D plotting of CSV or C3D files using Matplotlib.
       Improved exit functionality to ensure proper shutdown of the application.
       Refactored code for better readability and maintainability.
v0.1 - Initial version with basic GUI layout and functionality.
"""

import os
import signal
import sys
import platform
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk
from PIL import Image, ImageTk
from multimodal_mocap_coord_toolbox import (
    cluster_analysis,
    imu_analysis,
    markerless_2D_analysis,
    markerless_3D_analysis,
    mocap_analysis,
    convert_c3d_to_csv,
    convert_csv_to_c3d,
    rearrange_data_in_directory,
    run_drawboxe,
    count_frames_in_videos,
    export_file, copy_file, move_file, remove_file, import_file,
    show_c3d,
    vector_coding,
    sync_videos,
    extract_png_from_videos,
    compress_videos_h264_gui,
    compress_videos_h265_gui,
    batchcut, cut_videos,
    show_csv,
    getpixelvideo,
    dlt2d,
    rec2d,
    rec2d_one_dlt2d,
    show_vaila_message,
    emg_labiocom,
    plot_2d
)

if platform.system() == 'Darwin':
    import AppKit

class Vaila(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("vailá - 0.1.0")
        self.geometry("1280x720")

        # Set window icon based on OS
        icon_path_ico = os.path.join(os.path.dirname(__file__), "multimodal_mocap_coord_toolbox", "images", "vaila.ico")
        icon_path_png = os.path.join(os.path.dirname(__file__), "multimodal_mocap_coord_toolbox", "images", "vaila_ico_mac.png")

        if platform.system() == 'Windows':
            self.iconbitmap(icon_path_ico)
        else:
            img = Image.open(icon_path_png)
            img = ImageTk.PhotoImage(img)
            self.iconphoto(True, img)

        # Set application name for macOS dock
        if platform.system() == 'Darwin':
            AppKit.NSBundle.mainBundle().infoDictionary()['CFBundleName'] = 'Vaila'

        self.create_widgets()

    def create_widgets(self):
        button_width = 12  # Define a largura dos botões

        # Header with program name and description
        header_frame = tk.Frame(self)
        header_frame.pack(pady=10)

        # Load and place the image
        image_path_preto = os.path.join(os.path.dirname(__file__), "multimodal_mocap_coord_toolbox", "images", "vaila_logo.png")
        preto_image = Image.open(image_path_preto)
        preto_image = preto_image.resize((87, 87), Image.LANCZOS)
        preto_photo = ImageTk.PhotoImage(preto_image)

        preto_label = tk.Label(header_frame, image=preto_photo)
        preto_label.image = preto_photo
        preto_label.pack(side="left", padx=10)

        header_label = tk.Label(header_frame, text="vailá - Multimodal Toolbox", font=("Courier", 29, "bold"), anchor="center")
        header_label.pack(side="left")
        
        # Subheader with hyperlink for "vailá"
        subheader_frame = tk.Frame(self)
        subheader_frame.pack(pady=5)

        subheader_label1 = tk.Label(subheader_frame, text="Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox", font=("Arial", 15), anchor="center")
        subheader_label1.pack()

        subheader_label2_frame = tk.Frame(subheader_frame)
        subheader_label2_frame.pack()

        vaila_link = tk.Label(subheader_label2_frame, text="vailá", font=("Arial", 17, "italic"), fg="blue", cursor="hand2")
        vaila_link.pack(side="left")
        vaila_link.bind("<Button-1>", lambda e: self.open_link())

        unleash_label = tk.Label(subheader_label2_frame, text=" and unleash your imagination!", font=("Arial", 17), anchor="center")
        unleash_label.pack(side="left")


        # Create a canvas to add scrollbar
        canvas = tk.Canvas(self)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add a scrollbar to the canvas
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill="y")

        # Create a frame inside the canvas to hold all content
        scrollable_frame = tk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        # Add the scrollable frame to the canvas
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # File Manager Frame
        file_manager_frame = tk.LabelFrame(scrollable_frame, text="File Manager", padx=5, pady=5, font=("Arial", 17), labelanchor="n")
        file_manager_frame.pack(pady=10, fill="x")

        file_manager_btn_frame = tk.Frame(file_manager_frame)
        file_manager_btn_frame.pack(pady=5)

        import_btn = tk.Button(file_manager_btn_frame, text="Import", command=self.import_file, width=button_width)
        export_btn = tk.Button(file_manager_btn_frame, text="Export", command=self.export_file, width=button_width)
        copy_btn = tk.Button(file_manager_btn_frame, text="Copy", command=self.copy_file, width=button_width)
        move_btn = tk.Button(file_manager_btn_frame, text="Move", command=self.move_file, width=button_width)
        remove_btn = tk.Button(file_manager_btn_frame, text="Remove", command=self.remove_file, width=button_width)

        import_btn.pack(side="left", padx=2, pady=2)
        export_btn.pack(side="left", padx=2, pady=2)
        copy_btn.pack(side="left", padx=2, pady=2)
        move_btn.pack(side="left", padx=2, pady=2)
        remove_btn.pack(side="left", padx=2, pady=2)

        # Analysis Frame
        analysis_frame = tk.LabelFrame(scrollable_frame, text="Multimodal Analysis", padx=5, pady=5, font=("Arial", 17), labelanchor="n")
        analysis_frame.pack(pady=10, fill="x")

        #VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
        ## Insert the buttons for each analysis here
        # Buttons for each Multimodal Toolbox Analysis
        # First row of buttons
        row1_frame = tk.Frame(analysis_frame)
        row1_frame.pack(fill="x")
        imu_analysis_btn = tk.Button(row1_frame, text="IMU", width=button_width, command=self.imu_analysis)
        cluster_analysis_btn = tk.Button(row1_frame, text="Motion Capture Cluster", width=button_width, command=self.cluster_analysis)
        mocap_analysis_btn = tk.Button(row1_frame, text="Motion Capture Full Body", width=button_width, command=self.mocap_analysis)
        markerless_2d_analysis_btn = tk.Button(row1_frame, text="Markerless 2D", width=button_width, command=self.markerless_2d_analysis)
        markerless_3d_analysis_btn = tk.Button(row1_frame, text="Markerless 3D", width=button_width, command=self.markerless_3d_analysis)

        imu_analysis_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        cluster_analysis_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        mocap_analysis_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        markerless_2d_analysis_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        markerless_3d_analysis_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)

        # Second row of buttons
        row2_frame = tk.Frame(analysis_frame)
        row2_frame.pack(fill="x")
        vector_coding_btn = tk.Button(row2_frame, text="Vector Coding", width=button_width, command=self.vector_coding)
        emg_analysis_btn = tk.Button(row2_frame, text="EMG", width=button_width, command=self.emg_analysis)
        vaila_btn1 = tk.Button(row2_frame, text="vailá", width=button_width, command=self.show_vaila_message)
        vaila_btn2 = tk.Button(row2_frame, text="vailá", width=button_width, command=self.show_vaila_message)
        vaila_btn3 = tk.Button(row2_frame, text="vailá", width=button_width, command=self.show_vaila_message)

        vector_coding_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        emg_analysis_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        vaila_btn1.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        vaila_btn2.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        vaila_btn3.pack(side="left", expand=True, fill="x", padx=2, pady=2)

        # Third row of buttons (if needed)
        row3_frame = tk.Frame(analysis_frame)
        row3_frame.pack(fill="x")
        vaila_btn4 = tk.Button(row3_frame, text="vailá", width=button_width, command=self.show_vaila_message)
        vaila_btn5 = tk.Button(row3_frame, text="vailá", width=button_width, command=self.show_vaila_message)
        vaila_btn6 = tk.Button(row3_frame, text="vailá", width=button_width, command=self.show_vaila_message)
        vaila_btn7 = tk.Button(row3_frame, text="vailá", width=button_width, command=self.show_vaila_message)
        vaila_btn8 = tk.Button(row3_frame, text="vailá", width=button_width, command=self.show_vaila_message)

        vaila_btn4.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        vaila_btn5.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        vaila_btn6.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        vaila_btn7.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        vaila_btn8.pack(side="left", expand=True, fill="x", padx=2, pady=2)


        #VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
        # Tools Frame
        # Create a frame for the tools
        tools_frame = tk.LabelFrame(scrollable_frame, text="Available Tools", padx=5, pady=5, font=("Arial", 17), labelanchor="n")
        tools_frame.pack(pady=10, fill="x")

        tools_col1 = tk.LabelFrame(tools_frame, text="Data Files", padx=5, pady=5, font=("Arial", 14))
        tools_col2 = tk.LabelFrame(tools_frame, text="Video", padx=5, pady=5, font=("Arial", 14))
        tools_col3 = tk.LabelFrame(tools_frame, text="Visualization", padx=5, pady=5, font=("Arial", 14))

        # Data Files sub-columns
        reorder_csv_btn = tk.Button(tools_col1, text="Edit CSV", command=self.reorder_csv_data, width=button_width)
        convert_c3d_btn = tk.Button(tools_col1, text="C3D to CSV", command=self.convert_c3d_data, width=button_width) 
        create_c3d_btn = tk.Button(tools_col1, text="CSV to C3D", command=self.convert_csv_to_c3d, width=button_width)
        dlt2d_btn = tk.Button(tools_col1, text="DLT 2D", command=self.dlt2d, width=button_width)
        rec2d_multiple_btn = tk.Button(tools_col1, text="Rec2D Multi DLTs", command=self.rec2d, width=button_width)
        rec2d_one_btn = tk.Button(tools_col1, text="Rec2D One DLT", command=self.rec2d_one_dlt2d, width=button_width)
        vaila_btn1 = tk.Button(tools_col1, text="vailá", command=self.show_vaila_message, width=button_width)
        vaila_btn2 = tk.Button(tools_col1, text="vailá", command=self.show_vaila_message, width=button_width)
        vaila_btn3 = tk.Button(tools_col1, text="vailá", command=self.show_vaila_message, width=button_width)
        vaila_btn4 = tk.Button(tools_col1, text="vailá", command=self.show_vaila_message, width=button_width)
        vaila_btn5 = tk.Button(tools_col1, text="vailá", command=self.show_vaila_message, width=button_width)
        vaila_btn6 = tk.Button(tools_col1, text="vailá", command=self.show_vaila_message, width=button_width)
        # For more buttons, add more vaila_btns in the Visualization sub-columns or edit the code to add more columns
        # Example:
        # vaila_btn7 = tk.Button(tools_col1, text="vailá", command=self.show_vaila_message, width=button_width)

        # Packing Data Files buttons
        reorder_csv_btn.grid(row=0, column=0, padx=2, pady=2)
        convert_c3d_btn.grid(row=0, column=1, padx=2, pady=2)
        create_c3d_btn.grid(row=0, column=2, padx=2, pady=2)
        dlt2d_btn.grid(row=1, column=0, padx=2, pady=2)
        rec2d_multiple_btn.grid(row=1, column=1, padx=2, pady=2)
        rec2d_one_btn.grid(row=1, column=2, padx=2, pady=2)
        vaila_btn1.grid(row=2, column=0, padx=2, pady=2)
        vaila_btn2.grid(row=2, column=1, padx=2, pady=2)
        vaila_btn3.grid(row=2, column=2, padx=2, pady=2)
        vaila_btn4.grid(row=3, column=0, padx=2, pady=2)
        vaila_btn5.grid(row=3, column=1, padx=2, pady=2)
        vaila_btn6.grid(row=3, column=2, padx=2, pady=2)
        # For more buttons, add more vaila_btns in the Visualization sub-columns or edit the code to add more columns
        # Example:
        # vaila_btn7.grid(row=4, column=0, padx=2, pady=2)

        tools_col1.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        # Video sub-columns
        extract_png_btn = tk.Button(tools_col2, text="Extract PNG", command=self.extract_png_from_videos, width=button_width)
        cut_videos_btn = tk.Button(tools_col2, text="Cut Videos", command=self.cut_videos, width=button_width)
        draw_box_btn = tk.Button(tools_col2, text="Draw Box", command=self.draw_box, width=button_width)
        compress_videos_h264_btn = tk.Button(tools_col2, text="Compress H264", command=self.compress_videos_h264_gui, width=button_width)
        compress_videos_h265_btn = tk.Button(tools_col2, text="Compress H265", command=self.compress_videos_h265_gui, width=button_width)
        sync_videos_btn = tk.Button(tools_col2, text="Make Sync file", command=self.sync_videos, width=button_width)
        getpixelvideo_btn = tk.Button(tools_col2, text="Get Pixel Coords", command=self.getpixelvideo, width=button_width)
        count_frames_btn = tk.Button(tools_col2, text="Metadata info", command=self.count_frames_in_videos, width=button_width)
        vaila_btn7 = tk.Button(tools_col2, text="vailá", command=self.show_vaila_message, width=button_width)
        vaila_btn8 = tk.Button(tools_col2, text="vailá", command=self.show_vaila_message, width=button_width)
        vaila_btn9 = tk.Button(tools_col2, text="vailá", command=self.show_vaila_message, width=button_width)
        vaila_btn10 = tk.Button(tools_col2, text="vailá", command=self.show_vaila_message, width=button_width)
        # For more buttons, add more vaila_btns in the Visualization sub-columns or edit the code to add more columns
        # Example:
        # vaila_btn11 = tk.Button(tools_col2, text="vailá", command=self.show_vaila_message, width=button_width)

        # Packing Video buttons
        extract_png_btn.grid(row=0, column=0, padx=2, pady=2)
        cut_videos_btn.grid(row=0, column=1, padx=2, pady=2)
        draw_box_btn.grid(row=0, column=2, padx=2, pady=2)
        compress_videos_h264_btn.grid(row=1, column=0, padx=2, pady=2)
        compress_videos_h265_btn.grid(row=1, column=1, padx=2, pady=2)
        sync_videos_btn.grid(row=1, column=2, padx=2, pady=2)
        getpixelvideo_btn.grid(row=2, column=0, padx=2, pady=2)
        count_frames_btn.grid(row=2, column=1, padx=2, pady=2)
        vaila_btn7.grid(row=2, column=2, padx=2, pady=2)
        vaila_btn8.grid(row=3, column=0, padx=2, pady=2)
        vaila_btn9.grid(row=3, column=1, padx=2, pady=2)
        vaila_btn10.grid(row=3, column=2, padx=2, pady=2)
        # For more buttons, add more vaila_btns in the Visualization sub-columns ou edite o código para adicionar mais colunas
        # Example:
        # vaila_btn11.grid(row=4, column=0, padx=2, pady=2)
        
        tools_col2.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        # Visualization sub-columns
        show_c3d_btn = tk.Button(tools_col3, text="Show C3D", command=self.show_c3d_data, width=button_width)
        show_csv_btn = tk.Button(tools_col3, text="Show CSV", command=self.show_csv_file, width=button_width)
        plot_2d_btn = tk.Button(tools_col3, text="Plot 2D", command=self.plot_2d_data, width=button_width)
        vaila_btn12 = tk.Button(tools_col3, text="vailá", command=self.show_vaila_message, width=button_width)
        vaila_btn13 = tk.Button(tools_col3, text="vailá", command=self.show_vaila_message, width=button_width)
        vaila_btn14 = tk.Button(tools_col3, text="vailá", command=self.show_vaila_message, width=button_width)
        vaila_btn15 = tk.Button(tools_col3, text="vailá", command=self.show_vaila_message, width=button_width)
        vaila_btn16 = tk.Button(tools_col3, text="vailá", command=self.show_vaila_message, width=button_width)
        # For more buttons, add more vaila_btns in the Visualization sub-columns ou edite o código para adicionar mais colunas
        # Example:
        # vaila_btn17 = tk.Button(tools_col3, text="vailá", command=self.show_vaila_message, width=button_width)
        
        # Packing Visualization buttons
        show_c3d_btn.grid(row=0, column=0, padx=2, pady=2)
        show_csv_btn.grid(row=0, column=1, padx=2, pady=2)
        plot_2d_btn.grid(row=1, column=0, padx=2, pady=2)
        vaila_btn12.grid(row=1, column=1, padx=2, pady=2)
        vaila_btn13.grid(row=2, column=0, padx=2, pady=2)
        vaila_btn14.grid(row=2, column=1, padx=2, pady=2)
        vaila_btn15.grid(row=3, column=0, padx=2, pady=2)
        vaila_btn16.grid(row=3, column=1, padx=2, pady=2)
        # For more buttons, add more vaila_btns in the Visualization sub-columns ou edite o código para adicionar mais colunas
        # Example:
        # vaila_btn17.grid(row=4, column=0, padx=2, pady=2)
        
        tools_col3.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        # Help and Exit Buttons Frame
        bottom_frame = tk.Frame(scrollable_frame)
        bottom_frame.pack(pady=10)

        help_btn = tk.Button(bottom_frame, text="Help", command=self.display_help)
        exit_btn = tk.Button(bottom_frame, text="Exit", command=self.quit_app)

        help_btn.pack(side="left", padx=5)
        exit_btn.pack(side="left", padx=5)

        license_label = tk.Label(scrollable_frame, text="© 2024 vailá - Multimodal Toolbox. Licensed under the GNU Lesser General Public License v3.0.", font=("Arial", 11), anchor="center")
        license_label.pack(pady=5)

    def import_file(self):
        import_file()

    def export_file(self):
        export_file()

    def copy_file(self):
        copy_file()

    def move_file(self):
        move_file()

    def remove_file(self):
        remove_file()

    def imu_analysis(self):
        imu_analysis.analyze_imu_data()

    def cluster_analysis(self):
        cluster_analysis.analyze_cluster_data()

    def mocap_analysis(self):
        mocap_analysis.analyze_mocap_fullbody_data()

    def markerless_2d_analysis(self):
        selected_path = filedialog.askdirectory()
        if selected_path:
            markerless_2D_analysis.process_videos_in_directory(selected_path, os.path.join(selected_path, 'working'), pose_config={
                'min_detection_confidence': 0.1,
                'min_tracking_confidence': 0.1
            })

    def markerless_3d_analysis(self):
        selected_path = filedialog.askdirectory()
        if selected_path:
            markerless_3D_analysis.analyze_markerless_3D_data(selected_path)

    def vector_coding(self):
        pass

    def emg_analysis(self):
        emg_labiocom.run_emg_gui()

    def reorder_csv_data(self):
        rearrange_data_in_directory()

    def convert_c3d_data(self):
        convert_c3d_to_csv()

    def convert_csv_to_c3d(self):
        convert_csv_to_c3d()

    def count_frames_in_videos(self):
        selected_path = filedialog.askdirectory()
        if selected_path:
            count_frames_in_videos(selected_path)

    def cut_videos(self):
        cut_videos()

    def draw_box(self):
        run_drawboxe()

    def compress_videos_h264_gui(self):
        compress_videos_h264_gui()

    def compress_videos_h265_gui(self):
        compress_videos_h265_gui()

    def show_c3d_data(self):
        show_c3d()

    def sync_videos(self):
        sync_videos()

    def extract_png_from_videos(self):
        extract_png_from_videos()

    def getpixelvideo(self):
        getpixelvideo()

    def dlt2d(self):
        dlt2d()

    def rec2d(self):
        rec2d()

    def rec2d_one_dlt2d(self):
        rec2d_one_dlt2d()

    def show_csv_file(self):
        show_csv()

    def plot_2d_data(self):
        plot_2d()

    def display_help(self):
        help_file_path = os.path.join(os.path.dirname(__file__), 'docs', 'help.html')
        if os.path.exists(help_file_path):
            os.system(f'start {help_file_path}' if os.name == 'nt' else f'open {help_file_path}')
        else:
            messagebox.showerror("Error", "Help file not found.")

    def show_vaila_message(self):
        show_vaila_message()

    def open_link(self, event=None):
        import webbrowser
        webbrowser.open("https://github.com/paulopreto/vaila-multimodaltoolbox")

    def quit_app(self):
        self.destroy()
        os.kill(os.getpid(), signal.SIGTERM)

if __name__ == "__main__":
    app = Vaila()
    app.mainloop()
