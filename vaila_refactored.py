"""
===============================================================================
vaila_refactored.py - Refactored Version with Improved Architecture
===============================================================================
Author: Prof. Paulo R. P. Santiago
Date: 7 October 2024
Update: 20 January 2025
Version: 0.9.9
Python Version: 3.12.11

Key Improvements:
- Tab-based interface using ttk.Notebook
- Data-driven UI construction
- Component-based architecture
- Better error handling and user feedback
- Modern ttk widgets
- Centralized command management
===============================================================================
"""

import os
import signal
import platform
import subprocess
import sys
import webbrowser
from typing import Optional, Dict, List, Callable

import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from PIL import Image, ImageTk

# UI Structure Definition - Data-driven approach
UI_STRUCTURE = {
    "File Manager": {
        "buttons": [
            {"text": "Rename", "command": "rename_files", "tooltip": "Rename files in batch"},
            {"text": "Import", "command": "import_file", "tooltip": "Import files to current directory"},
            {"text": "Export", "command": "export_file", "tooltip": "Export files to another directory"},
            {"text": "Copy", "command": "copy_file", "tooltip": "Copy files between directories"},
            {"text": "Move", "command": "move_file", "tooltip": "Move files between directories"},
            {"text": "Remove", "command": "remove_file", "tooltip": "Remove selected files"},
            {"text": "Tree", "command": "tree_file", "tooltip": "Show directory tree structure"},
            {"text": "Find", "command": "find_file", "tooltip": "Find files by name or pattern"},
            {"text": "Transfer", "command": "transfer_file", "tooltip": "Transfer files via SSH"},
        ]
    },
    "Multimodal Analysis": {
        "sections": {
            "Motion Analysis": [
                {"text": "IMU", "command": "imu_analysis", "tooltip": "Analyze IMU sensor data"},
                {"text": "MoCap Cluster", "command": "cluster_analysis", "tooltip": "Motion capture cluster analysis"},
                {"text": "MoCap Full Body", "command": "mocap_analysis", "tooltip": "Full body motion capture analysis"},
                {"text": "Markerless 2D", "command": "markerless_2d_analysis", "tooltip": "2D markerless motion analysis"},
                {"text": "Markerless 3D", "command": "markerless_3d_analysis", "tooltip": "3D markerless motion analysis"},
            ],
            "Force & Physiology": [
                {"text": "Vector Coding", "command": "vector_coding", "tooltip": "Vector coding analysis"},
                {"text": "EMG", "command": "emg_analysis", "tooltip": "Electromyography analysis"},
                {"text": "Force Plate", "command": "force_analysis", "tooltip": "Force plate data analysis"},
                {"text": "GNSS/GPS", "command": "gnss_analysis", "tooltip": "GPS/GNSS trajectory analysis"},
                {"text": "MEG/EEG", "command": "eeg_analysis", "tooltip": "Neurophysiology data analysis"},
            ],
            "Specialized Analysis": [
                {"text": "HR/ECG", "command": "hr_analysis", "tooltip": "Heart rate and ECG analysis"},
                {"text": "Yolo + MP", "command": "markerless2d_mpyolo", "tooltip": "YOLO + MediaPipe analysis"},
                {"text": "Vertical Jump", "command": "vailajump", "tooltip": "Jump performance analysis"},
                {"text": "Cube2D", "command": "cube2d_kinematics", "tooltip": "2D cube kinematics"},
                {"text": "Animal Open Field", "command": "animal_open_field", "tooltip": "Animal behavior analysis"},
            ],
            "Advanced Tools": [
                {"text": "Tracker", "command": "tracker", "tooltip": "Object tracking with YOLO"},
                {"text": "ML Walkway", "command": "ml_walkway", "tooltip": "Machine learning gait analysis"},
                {"text": "Markerless Hands", "command": "markerless_hands", "tooltip": "Hand tracking analysis"},
                {"text": "MP Angles", "command": "mp_angles_calculation", "tooltip": "MediaPipe angle calculations"},
                {"text": "Markerless Live", "command": "markerless_live", "tooltip": "Real-time markerless tracking"},
            ],
            "Research Tools": [
                {"text": "Ultrasound", "command": "ultrasound", "tooltip": "Ultrasound image analysis"},
                {"text": "Brainstorm", "command": "brainstorm", "tooltip": "Voice-to-AI creative assistant"},
                {"text": "Future Tool 1", "command": "show_vaila_message", "tooltip": "Coming soon..."},
                {"text": "Future Tool 2", "command": "show_vaila_message", "tooltip": "Coming soon..."},
                {"text": "Future Tool 3", "command": "show_vaila_message", "tooltip": "Coming soon..."},
            ]
        }
    },
    "Data Tools": {
        "sections": {
            "Data Processing": [
                {"text": "Edit CSV", "command": "reorder_csv_data", "tooltip": "Edit and reorder CSV data"},
                {"text": "C3D ↔ CSV", "command": "convert_c3d_csv", "tooltip": "Convert between C3D and CSV formats"},
                {"text": "Smooth/Fill/Split", "command": "gapfill_split", "tooltip": "Data smoothing and gap filling"},
            ],
            "2D Calibration": [
                {"text": "Make DLT2D", "command": "dlt2d", "tooltip": "Create 2D DLT calibration"},
                {"text": "Rec2D 1DLT", "command": "rec2d_one_dlt2d", "tooltip": "2D reconstruction with single DLT"},
                {"text": "Rec2D MultiDLT", "command": "rec2d", "tooltip": "2D reconstruction with multiple DLTs"},
            ],
            "3D Calibration": [
                {"text": "Make DLT3D", "command": "run_dlt3d", "tooltip": "Create 3D DLT calibration"},
                {"text": "Rec3D 1DLT", "command": "rec3d_one_dlt3d", "tooltip": "3D reconstruction with single DLT"},
                {"text": "Rec3D MultiDLT", "command": "rec3d", "tooltip": "3D reconstruction with multiple DLTs"},
            ],
            "Advanced Processing": [
                {"text": "ReID Marker", "command": "reid_marker", "tooltip": "Marker re-identification"},
                {"text": "Future Tool", "command": "show_vaila_message", "tooltip": "Coming soon..."},
                {"text": "Future Tool", "command": "show_vaila_message", "tooltip": "Coming soon..."},
            ]
        }
    },
    "Video Tools": {
        "sections": {
            "Video Processing": [
                {"text": "Video ↔ PNG", "command": "extract_png_from_videos", "tooltip": "Extract/create video frames"},
                {"text": "Sync Video", "command": "cut_videos", "tooltip": "Synchronize video files"},
                {"text": "Draw Box", "command": "draw_box", "tooltip": "Draw boxes on video frames"},
            ],
            "Video Compression": [
                {"text": "Compress H264", "command": "compress_videos_h264_gui", "tooltip": "H.264 video compression"},
                {"text": "Compress H265", "command": "compress_videos_h265_gui", "tooltip": "H.265 video compression"},
                {"text": "Make Sync File", "command": "sync_videos", "tooltip": "Create video sync files"},
            ],
            "Video Analysis": [
                {"text": "Get Pixel Coord", "command": "getpixelvideo", "tooltip": "Extract pixel coordinates"},
                {"text": "Metadata Info", "command": "count_frames_in_videos", "tooltip": "Video metadata analysis"},
                {"text": "Merge/Split Video", "command": "process_videos_gui", "tooltip": "Merge or split videos"},
            ],
            "Video Effects": [
                {"text": "Distort Video/Data", "command": "run_distortvideo", "tooltip": "Apply distortion effects"},
                {"text": "Cut Video", "command": "cut_video", "tooltip": "Cut video segments"},
                {"text": "Resize Video", "command": "resize_video", "tooltip": "Resize video dimensions"},
            ],
            "Audio & Utils": [
                {"text": "YT Downloader", "command": "ytdownloader", "tooltip": "Download YouTube videos"},
                {"text": "Insert Audio", "command": "run_iaudiovid", "tooltip": "Add audio to videos"},
                {"text": "Remove Dup PNG", "command": "remove_duplicate_frames", "tooltip": "Remove duplicate frames"},
            ]
        }
    },
    "Visualization": {
        "sections": {
            "Data Viewers": [
                {"text": "Show C3D", "command": "show_c3d_data", "tooltip": "3D C3D file viewer"},
                {"text": "Show CSV 3D", "command": "show_csv_file", "tooltip": "3D CSV data viewer"},
            ],
            "2D/3D Plotting": [
                {"text": "Plot 2D", "command": "plot_2d_data", "tooltip": "2D data plotting"},
                {"text": "Plot 3D", "command": "plot_3d_data", "tooltip": "3D data plotting"},
            ],
            "Specialized Views": [
                {"text": "Soccer Field", "command": "draw_soccerfield", "tooltip": "Draw soccer field visualization"},
                {"text": "Future Tool", "command": "show_vaila_message", "tooltip": "Coming soon..."},
            ]
        }
    }
}

class StatusBar(ttk.Frame):
    """Status bar component for providing user feedback"""
    def __init__(self, parent):
        super().__init__(parent)
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        
        self.status_label = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.progress = ttk.Progressbar(self, mode='indeterminate')
        self.progress.pack(side=tk.RIGHT, padx=(5, 0))
    
    def set_status(self, message: str):
        """Set status message"""
        self.status_var.set(message)
        self.update_idletasks()
    
    def show_progress(self):
        """Show progress indicator"""
        self.progress.start()
    
    def hide_progress(self):
        """Hide progress indicator"""
        self.progress.stop()

class FileManagerTab(ttk.Frame):
    """File Manager tab component"""
    def __init__(self, parent, commands: Dict[str, Callable], button_width: int):
        super().__init__(parent)
        self.commands = commands
        self.button_width = button_width
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create file manager widgets"""
        # Main frame
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Description
        desc_label = ttk.Label(main_frame, text="File management operations for batch processing", 
                              font=("Arial", 11))
        desc_label.pack(pady=(0, 15))
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create buttons from UI structure
        buttons_config = UI_STRUCTURE["File Manager"]["buttons"]
        buttons_per_row = 3
        
        for i, button_config in enumerate(buttons_config):
            row = i // buttons_per_row
            col = i % buttons_per_row
            
            btn = ttk.Button(
                button_frame,
                text=button_config["text"],
                command=lambda cmd=button_config["command"]: self.execute_command(cmd),
                width=self.button_width
            )
            btn.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
            
            # Add tooltip if available
            if "tooltip" in button_config:
                self.create_tooltip(btn, button_config["tooltip"])
        
        # Configure grid weights
        for i in range(buttons_per_row):
            button_frame.columnconfigure(i, weight=1)
    
    def execute_command(self, command_name: str):
        """Execute a command by name"""
        if command_name in self.commands:
            self.commands[command_name]()
        else:
            messagebox.showerror("Error", f"Command '{command_name}' not found")
    
    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = ttk.Label(tooltip, text=text, background="#ffffe0", 
                            relief=tk.SOLID, borderwidth=1)
            label.pack()
            widget.tooltip = tooltip
        
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip
        
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

class MultimodalAnalysisTab(ttk.Frame):
    """Multimodal Analysis tab component"""
    def __init__(self, parent, commands: Dict[str, Callable], button_width: int):
        super().__init__(parent)
        self.commands = commands
        self.button_width = button_width
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create multimodal analysis widgets"""
        # Main frame with scrollable content
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Main content frame
        main_frame = ttk.Frame(scrollable_frame, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Description
        desc_label = ttk.Label(main_frame, text="Advanced biomechanical and multimodal analysis tools", 
                              font=("Arial", 11))
        desc_label.pack(pady=(0, 15))
        
        # Create sections
        sections = UI_STRUCTURE["Multimodal Analysis"]["sections"]
        for section_name, buttons in sections.items():
            self.create_section(main_frame, section_name, buttons)
    
    def create_section(self, parent, section_name: str, buttons: List[Dict]):
        """Create a section with buttons"""
        # Section frame
        section_frame = ttk.LabelFrame(parent, text=section_name, padding=10)
        section_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Buttons
        buttons_per_row = 5
        for i, button_config in enumerate(buttons):
            row = i // buttons_per_row
            col = i % buttons_per_row
            
            btn = ttk.Button(
                section_frame,
                text=button_config["text"],
                command=lambda cmd=button_config["command"]: self.execute_command(cmd),
                width=self.button_width
            )
            btn.grid(row=row, column=col, padx=2, pady=2, sticky="ew")
            
            # Add tooltip
            if "tooltip" in button_config:
                self.create_tooltip(btn, button_config["tooltip"])
        
        # Configure grid weights
        for i in range(buttons_per_row):
            section_frame.columnconfigure(i, weight=1)
    
    def execute_command(self, command_name: str):
        """Execute a command by name"""
        if command_name in self.commands:
            self.commands[command_name]()
        else:
            messagebox.showerror("Error", f"Command '{command_name}' not found")
    
    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = ttk.Label(tooltip, text=text, background="#ffffe0", 
                            relief=tk.SOLID, borderwidth=1)
            label.pack()
            widget.tooltip = tooltip
        
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip
        
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

class ToolsTab(ttk.Frame):
    """Tools tab component (for Data, Video, Visualization tools)"""
    def __init__(self, parent, tab_name: str, commands: Dict[str, Callable], button_width: int):
        super().__init__(parent)
        self.tab_name = tab_name
        self.commands = commands
        self.button_width = button_width
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create tools widgets"""
        # Main frame
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Description
        descriptions = {
            "Data Tools": "Data processing, calibration, and format conversion tools",
            "Video Tools": "Video processing, compression, and analysis utilities",
            "Visualization": "Data visualization and 3D viewing tools"
        }
        
        desc_label = ttk.Label(main_frame, text=descriptions.get(self.tab_name, ""), 
                              font=("Arial", 11))
        desc_label.pack(pady=(0, 15))
        
        # Create sections
        sections = UI_STRUCTURE[self.tab_name]["sections"]
        for section_name, buttons in sections.items():
            self.create_section(main_frame, section_name, buttons)
    
    def create_section(self, parent, section_name: str, buttons: List[Dict]):
        """Create a section with buttons"""
        # Section frame
        section_frame = ttk.LabelFrame(parent, text=section_name, padding=10)
        section_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Buttons
        buttons_per_row = 3
        for i, button_config in enumerate(buttons):
            row = i // buttons_per_row
            col = i % buttons_per_row
            
            btn = ttk.Button(
                section_frame,
                text=button_config["text"],
                command=lambda cmd=button_config["command"]: self.execute_command(cmd),
                width=self.button_width
            )
            btn.grid(row=row, column=col, padx=2, pady=2, sticky="ew")
            
            # Add tooltip
            if "tooltip" in button_config:
                self.create_tooltip(btn, button_config["tooltip"])
        
        # Configure grid weights
        for i in range(buttons_per_row):
            section_frame.columnconfigure(i, weight=1)
    
    def execute_command(self, command_name: str):
        """Execute a command by name"""
        if command_name in self.commands:
            self.commands[command_name]()
        else:
            messagebox.showerror("Error", f"Command '{command_name}' not found")
    
    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = ttk.Label(tooltip, text=text, background="#ffffe0", 
                            relief=tk.SOLID, borderwidth=1)
            label.pack()
            widget.tooltip = tooltip
        
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip
        
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

class CommandManager:
    """Centralized command management"""
    def __init__(self, status_bar: StatusBar):
        self.status_bar = status_bar
        self.commands = {}
        self.setup_commands()
    
    def setup_commands(self):
        """Setup all available commands"""
        # Import commands here to avoid early loading issues
        self.commands = {
            # File Manager Commands
            "rename_files": self.rename_files,
            "import_file": self.import_file,
            "export_file": self.export_file,
            "copy_file": self.copy_file,
            "move_file": self.move_file,
            "remove_file": self.remove_file,
            "tree_file": self.tree_file,
            "find_file": self.find_file,
            "transfer_file": self.transfer_file,
            
            # Multimodal Analysis Commands
            "imu_analysis": self.imu_analysis,
            "cluster_analysis": self.cluster_analysis,
            "mocap_analysis": self.mocap_analysis,
            "markerless_2d_analysis": self.markerless_2d_analysis,
            "markerless_3d_analysis": self.markerless_3d_analysis,
            "vector_coding": self.vector_coding,
            "emg_analysis": self.emg_analysis,
            "force_analysis": self.force_analysis,
            "gnss_analysis": self.gnss_analysis,
            "eeg_analysis": self.eeg_analysis,
            "hr_analysis": self.hr_analysis,
            "markerless2d_mpyolo": self.markerless2d_mpyolo,
            "vailajump": self.vailajump,
            "cube2d_kinematics": self.cube2d_kinematics,
            "animal_open_field": self.animal_open_field,
            "tracker": self.tracker,
            "ml_walkway": self.ml_walkway,
            "markerless_hands": self.markerless_hands,
            "mp_angles_calculation": self.mp_angles_calculation,
            "markerless_live": self.markerless_live,
            "ultrasound": self.ultrasound,
            "brainstorm": self.brainstorm,
            
            # Data Tools Commands
            "reorder_csv_data": self.reorder_csv_data,
            "convert_c3d_csv": self.convert_c3d_csv,
            "gapfill_split": self.gapfill_split,
            "dlt2d": self.dlt2d,
            "rec2d_one_dlt2d": self.rec2d_one_dlt2d,
            "rec2d": self.rec2d,
            "run_dlt3d": self.run_dlt3d,
            "rec3d_one_dlt3d": self.rec3d_one_dlt3d,
            "rec3d": self.rec3d,
            "reid_marker": self.reid_marker,
            
            # Video Tools Commands
            "extract_png_from_videos": self.extract_png_from_videos,
            "cut_videos": self.cut_videos,
            "draw_box": self.draw_box,
            "compress_videos_h264_gui": self.compress_videos_h264_gui,
            "compress_videos_h265_gui": self.compress_videos_h265_gui,
            "sync_videos": self.sync_videos,
            "getpixelvideo": self.getpixelvideo,
            "count_frames_in_videos": self.count_frames_in_videos,
            "process_videos_gui": self.process_videos_gui,
            "run_distortvideo": self.run_distortvideo,
            "cut_video": self.cut_video,
            "resize_video": self.resize_video,
            "ytdownloader": self.ytdownloader,
            "run_iaudiovid": self.run_iaudiovid,
            "remove_duplicate_frames": self.remove_duplicate_frames,
            
            # Visualization Commands
            "show_c3d_data": self.show_c3d_data,
            "show_csv_file": self.show_csv_file,
            "plot_2d_data": self.plot_2d_data,
            "plot_3d_data": self.plot_3d_data,
            "draw_soccerfield": self.draw_soccerfield,
            
            # Placeholder command
            "show_vaila_message": self.show_vaila_message,
        }
    
    def execute_command(self, command_name: str):
        """Execute a command with error handling and status feedback"""
        if command_name not in self.commands:
            messagebox.showerror("Error", f"Command '{command_name}' not found")
            return
        
        try:
            self.status_bar.set_status(f"Executing {command_name}...")
            self.status_bar.show_progress()
            
            # Execute the command
            self.commands[command_name]()
            
            self.status_bar.set_status("Ready")
            
        except ImportError as e:
            messagebox.showerror("Import Error", f"Required module not found: {e}")
            self.status_bar.set_status("Error occurred")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            self.status_bar.set_status("Error occurred")
        finally:
            self.status_bar.hide_progress()
    
    # Command implementations (same as original, but with better error handling)
    def rename_files(self):
        from vaila.filemanager import rename_files
        rename_files()
    
    def import_file(self):
        from vaila.filemanager import import_file
        import_file()
    
    def export_file(self):
        from vaila.filemanager import export_file
        export_file()
    
    def copy_file(self):
        from vaila.filemanager import copy_file
        copy_file()
    
    def move_file(self):
        from vaila.filemanager import move_file
        move_file()
    
    def remove_file(self):
        from vaila.filemanager import remove_file
        remove_file()
    
    def tree_file(self):
        from vaila.filemanager import tree_file
        tree_file()
    
    def find_file(self):
        from vaila.filemanager import find_file
        find_file()
    
    def transfer_file(self):
        from vaila.filemanager import transfer_file
        transfer_file()
    
    def imu_analysis(self):
        from vaila import imu_analysis
        imu_analysis.analyze_imu_data()
    
    def cluster_analysis(self):
        from vaila import cluster_analysis
        cluster_analysis.analyze_cluster_data()
    
    def mocap_analysis(self):
        from vaila import mocap_analysis
        mocap_analysis.analyze_mocap_fullbody_data()
    
    def markerless_2d_analysis(self):
        # Implementation with version choice dialog
        import tkinter as tk
        from tkinter import messagebox, simpledialog

        root = tk.Tk()
        root.withdraw()

        choice = simpledialog.askstring(
            "Markerless 2D Analysis Version",
            "Select version:\n\n1: Standard (Faster, single-person)\n2: Advanced (Slower, multi-person with YOLO)",
            initialvalue="1",
        )

        if not choice or choice not in ["1", "2"]:
            return

        if choice == "1":
            from vaila.markerless_2D_analysis import process_videos_in_directory
        else:
            from vaila.markerless2d_analysis_v2 import process_videos_in_directory

        process_videos_in_directory()
    
    def markerless_3d_analysis(self):
        # Implementation with version choice dialog
        import tkinter as tk
        from tkinter import messagebox, simpledialog

        root = tk.Tk()
        root.withdraw()

        choice = simpledialog.askstring(
            "Markerless 3D Analysis Version",
            "Select version:\n\n1: Standard (Faster, single-person)\n2: Advanced (Slower, multi-person with YOLO)",
            initialvalue="1",
        )

        if not choice or choice not in ["1", "2"]:
            return

        if choice == "1":
            from vaila.markerless_3D_analysis import process_videos_in_directory
        else:
            from vaila.markerless3d_analysis_v2 import process_videos_in_directory

        process_videos_in_directory()
    
    def vector_coding(self):
        from vaila.run_vector_coding import run_vector_coding
        run_vector_coding()
    
    def emg_analysis(self):
        from vaila import emg_labiocom
        emg_labiocom.run_emg_gui()
    
    def force_analysis(self):
        from vaila import forceplate_analysis
        forceplate_analysis.run_force_analysis()
    
    def gnss_analysis(self):
        from vaila.gnss_analysis import run_gnss_analysis_gui
        run_gnss_analysis_gui()
    
    def eeg_analysis(self):
        webbrowser.open("https://mne.tools/dev/auto_tutorials/intro/10_overview.html")
    
    def hr_analysis(self):
        webbrowser.open("https://github.com/paulvangentcom/heartrate_analysis_python")
    
    def markerless2d_mpyolo(self):
        from vaila import markerless2d_mpyolo
        markerless2d_mpyolo.run_markerless2d_mpyolo()
    
    def vailajump(self):
        from vaila.vaila_and_jump import vaila_and_jump
        vaila_and_jump()
    
    def cube2d_kinematics(self):
        from vaila import cube2d_kinematics
        cube2d_kinematics.run_cube2d_kinematics()
    
    def animal_open_field(self):
        from vaila import animal_open_field
        animal_open_field.run_animal_open_field()
    
    def tracker(self):
        # Create version selection dialog
        dialog = tk.Toplevel()
        dialog.title("Select YOLO Version")
        dialog.geometry("400x320")
        dialog.transient()
        dialog.grab_set()

        tk.Label(dialog, text="Select YOLO tracker version to use:", pady=15).pack()

        def use_yolov12():
            dialog.destroy()
            try:
                from vaila import yolov12track
                yolov12track.run_yolov12track()
            except Exception as e:
                messagebox.showerror("Error Running YOLOv12", f"Error: {str(e)}")

        def use_yolov11():
            dialog.destroy()
            try:
                from vaila import yolov11track
                yolov11track.run_yolov11track()
            except Exception as e:
                messagebox.showerror("Error Running YOLOv11", f"Error: {str(e)}")

        def use_train_yolov11():
            dialog.destroy()
            try:
                from vaila import yolotrain
                yolotrain.run_yolotrain_gui()
            except Exception as e:
                messagebox.showerror("Error in YOLO Training", f"Error: {str(e)}")

        tk.Button(dialog, text="YOLOv12 Tracker", command=use_yolov12, width=20).pack(pady=10)
        tk.Button(dialog, text="YOLOv11 Tracker", command=use_yolov11, width=20).pack(pady=10)
        tk.Button(dialog, text="Train YOLO", command=use_train_yolov11, width=20).pack(pady=10)
        tk.Button(dialog, text="Cancel", command=dialog.destroy, width=10).pack(pady=10)
    
    def ml_walkway(self):
        from vaila import vaila_mlwalkway
        vaila_mlwalkway.run_vaila_mlwalkway_gui()
    
    def markerless_hands(self):
        from vaila import mphands
        mphands.run_mphands()
    
    def mp_angles_calculation(self):
        from vaila import mpangles
        mpangles.run_mp_angles()
    
    def markerless_live(self):
        from vaila import markerless_live
        markerless_live.run_markerless_live()
    
    def ultrasound(self):
        from vaila import usound_biomec1
        usound_biomec1.run_usound()
    
    def brainstorm(self):
        from vaila import brainstorm
        brainstorm.run_brainstorm()
    
    def reorder_csv_data(self):
        from vaila import rearrange_data
        rearrange_data.rearrange_data_in_directory()
    
    def convert_c3d_csv(self):
        # Create conversion choice dialog
        from vaila import convert_c3d_to_csv, convert_csv_to_c3d
        
        window = tk.Toplevel()
        window.title("Choose Action")
        window.geometry("300x150")
        window.transient()
        window.grab_set()
        
        tk.Label(window, text="Which conversion would you like to perform?").pack(pady=10)
        
        tk.Button(window, text="C3D → CSV", 
                 command=lambda: [convert_c3d_to_csv(), window.destroy()]).pack(side="left", padx=20, pady=20)
        
        tk.Button(window, text="CSV → C3D", 
                 command=lambda: [convert_csv_to_c3d(), window.destroy()]).pack(side="right", padx=20, pady=20)
    
    def gapfill_split(self):
        from vaila.interp_smooth_split import run_fill_split_dialog
        run_fill_split_dialog()
    
    def dlt2d(self):
        from vaila import dlt2d
        dlt2d.run_dlt2d()
    
    def rec2d_one_dlt2d(self):
        from vaila import rec2d_one_dlt2d
        rec2d_one_dlt2d.run_rec2d_one_dlt2d()
    
    def rec2d(self):
        from vaila import rec2d
        rec2d.run_rec2d()
    
    def run_dlt3d(self):
        try:
            script_path = os.path.join("vaila", "dlt3d.py")
            subprocess.Popen([sys.executable, script_path])
        except Exception as e:
            messagebox.showerror("Error", f"Error running dlt3d.py: {e}")
    
    def rec3d_one_dlt3d(self):
        from vaila import rec3d_one_dlt3d
        rec3d_one_dlt3d.run_rec3d_one_dlt3d()
    
    def rec3d(self):
        # Placeholder - implement as needed
        messagebox.showinfo("Info", "3D reconstruction with multiple DLTs - Coming soon")
    
    def reid_marker(self):
        from vaila import reid_markers
        reid_markers.create_gui_menu()
    
    def extract_png_from_videos(self):
        from vaila.extractpng import VideoProcessor
        processor = VideoProcessor()
        processor.run()
    
    def cut_videos(self):
        from vaila import cutvideo
        cutvideo.run_cutvideo()
    
    def draw_box(self):
        from vaila import drawboxe
        drawboxe.run_drawboxe()
    
    def compress_videos_h264_gui(self):
        from vaila import compress_videos_h264
        compress_videos_h264.compress_videos_h264_gui()
    
    def compress_videos_h265_gui(self):
        from vaila import compress_videos_h265
        compress_videos_h265.compress_videos_h265_gui()
    
    def sync_videos(self):
        from vaila import syncvid
        syncvid.sync_videos()
    
    def getpixelvideo(self):
        from vaila import getpixelvideo
        getpixelvideo.run_getpixelvideo()
    
    def count_frames_in_videos(self):
        from vaila import numberframes
        numberframes.count_frames_in_videos()
    
    def process_videos_gui(self):
        from vaila import videoprocessor
        videoprocessor.process_videos_gui()
    
    def run_distortvideo(self):
        # Create distortion type choice dialog
        from vaila import vaila_lensdistortvideo, vaila_distortvideo_gui, vaila_datdistort
        
        dialog = tk.Toplevel()
        dialog.title("Choose Distortion Correction Type")
        dialog.geometry("300x200")
        dialog.transient()
        dialog.grab_set()
        
        tk.Label(dialog, text="Select the type of distortion correction:", pady=10).pack()
        
        tk.Button(dialog, text="Video Correction",
                 command=lambda: (vaila_lensdistortvideo.run_distortvideo(), dialog.destroy())).pack(pady=5)
        
        tk.Button(dialog, text="Interactive Distortion Correction",
                 command=lambda: (vaila_distortvideo_gui.run_distortvideo_gui(), dialog.destroy())).pack(pady=5)
        
        tk.Button(dialog, text="CSV/DAT Coordinates Correction",
                 command=lambda: (vaila_datdistort.run_datdistort(), dialog.destroy())).pack(pady=5)
        
        tk.Button(dialog, text="Cancel", command=dialog.destroy).pack(pady=10)
    
    def cut_video(self):
        from vaila import cutvideo
        cutvideo.run_cutvideo()
    
    def resize_video(self):
        from vaila import resize_video
        resize_video.run_resize_video()
    
    def ytdownloader(self):
        from vaila import vaila_ytdown
        vaila_ytdown.run_ytdown()
    
    def run_iaudiovid(self):
        from vaila import vaila_iaudiovid
        vaila_iaudiovid.run_iaudiovid()
    
    def remove_duplicate_frames(self):
        from vaila import rm_duplicateframes
        rm_duplicateframes.run_rm_duplicateframes()
    
    def show_c3d_data(self):
        from vaila.viewc3d import run_viewc3d
        run_viewc3d()
    
    def show_csv_file(self):
        from vaila.readcsv import show_csv
        show_csv()
    
    def plot_2d_data(self):
        from vaila import vailaplot2d
        vailaplot2d.run_plot_2d()
    
    def plot_3d_data(self):
        from vaila import vailaplot3d
        vailaplot3d.run_plot_3d()
    
    def draw_soccerfield(self):
        from vaila import soccerfield
        soccerfield.run_soccerfield()
    
    def show_vaila_message(self):
        from vaila.vaila_manifest import show_vaila_message
        show_vaila_message()

class VailaRefactored(tk.Tk):
    """
    Refactored Vaila application with improved architecture
    """
    def __init__(self):
        super().__init__()
        
        # Set title and configure window
        self.title("vailá - 20.January.2025 v0.9.9 (Python 3.12.11) - Refactored")
        
        # Configure dimensions based on OS
        self.set_dimensions_based_on_os()
        
        # Configure icon
        self.configure_icon()
        
        # Configure macOS app name if available
        self.configure_macos_app()
        
        # Create widgets
        self.create_widgets()
    
    def set_dimensions_based_on_os(self):
        """Set window dimensions based on OS"""
        if platform.system() == "Darwin":  # macOS
            self.geometry("1920x1080")
            self.button_width = 12
            self.font_size = 11
        elif platform.system() == "Windows":  # Windows
            self.geometry("1200x800")
            self.button_width = 13
            self.font_size = 11
        elif platform.system() == "Linux":  # Linux
            self.geometry("1400x900")
            self.button_width = 15
            self.font_size = 11
        else:  # Default
            self.geometry("1200x800")
            self.button_width = 15
            self.font_size = 11
    
    def configure_icon(self):
        """Configure application icon"""
        icon_path_ico = os.path.join(os.path.dirname(__file__), "vaila", "images", "vaila.ico")
        icon_path_png = os.path.join(os.path.dirname(__file__), "vaila", "images", "vaila_ico_mac.png")
        
        if platform.system() == "Windows":
            try:
                self.iconbitmap(icon_path_ico)
            except Exception as e:
                print(f"Could not set Windows icon: {e}")
        else:
            try:
                icon = tk.PhotoImage(file=icon_path_png)
                self.iconphoto(True, icon)
            except Exception as e:
                print(f"Could not set icon: {e}")
    
    def configure_macos_app(self):
        """Configure macOS application name"""
        if platform.system() == "Darwin":
            try:
                import AppKit
                AppKit.NSBundle.mainBundle().infoDictionary()["CFBundleName"] = "vailá"
            except (ImportError, Exception) as e:
                print(f"Could not set macOS application name: {e}")
    
    def create_widgets(self):
        """Create the main interface widgets"""
        # Main container
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header
        self.create_header(main_container)
        
        # Status bar (create first so other components can use it)
        self.status_bar = StatusBar(main_container)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Command manager
        self.command_manager = CommandManager(self.status_bar)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_tabs()
        
        # Bottom frame with help and exit
        self.create_bottom_frame(main_container)
    
    def create_header(self, parent):
        """Create the header section"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(pady=10)
        
        # Load and display logo
        try:
            image_path = os.path.join(os.path.dirname(__file__), "vaila", "images", "vaila_logo.png")
            logo_image = Image.open(image_path)
            logo_image = logo_image.resize((87, 87))
            self.logo_photo = ImageTk.PhotoImage(logo_image)
            
            logo_label = ttk.Label(header_frame, image=self.logo_photo)
            logo_label.pack(side="left", padx=10)
        except Exception as e:
            print(f"Could not load logo: {e}")
        
        # Title section
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(side="left")
        
        # Clickable vailá label
        vaila_label = ttk.Label(title_frame, text="vailá", 
                               font=("default", self.font_size, "italic"))
        vaila_label.pack(side="left")
        vaila_label.bind("<Button-1>", self.open_project_directory)
        
        # Toolbox label
        toolbox_label = ttk.Label(title_frame, text=" - Multimodal Toolbox", 
                                 font=("default", self.font_size))
        toolbox_label.pack(side="left")
        
        # Subheader
        subheader_frame = ttk.Frame(header_frame)
        subheader_frame.pack(side="left", padx=(20, 0))
        
        # Description
        desc_label = ttk.Label(subheader_frame, 
                              text="Versatile Anarcho Integrated Liberation Ánalysis",
                              font=("default", self.font_size))
        desc_label.pack()
        
        # Action frame
        action_frame = ttk.Frame(subheader_frame)
        action_frame.pack()
        
        # Project link
        project_link = ttk.Label(action_frame, text="vailá", 
                                font=("default", self.font_size, "italic"))
        project_link.pack(side="left")
        project_link.bind("<Button-1>", lambda e: self.open_link())
        
        # Unleash text
        unleash_label = ttk.Label(action_frame, text=" and unleash your ", 
                                 font=("default", self.font_size))
        unleash_label.pack(side="left")
        
        # Imagination button
        imagination_btn = ttk.Button(action_frame, text="imagination!", 
                                   command=self.open_terminal_shell)
        imagination_btn.pack(side="left")
    
    def create_tabs(self):
        """Create all tabs"""
        # File Manager tab
        file_manager_tab = FileManagerTab(self.notebook, self.command_manager.commands, self.button_width)
        self.notebook.add(file_manager_tab, text="File Manager")
        
        # Multimodal Analysis tab
        analysis_tab = MultimodalAnalysisTab(self.notebook, self.command_manager.commands, self.button_width)
        self.notebook.add(analysis_tab, text="Multimodal Analysis")
        
        # Data Tools tab
        data_tools_tab = ToolsTab(self.notebook, "Data Tools", self.command_manager.commands, self.button_width)
        self.notebook.add(data_tools_tab, text="Data Tools")
        
        # Video Tools tab
        video_tools_tab = ToolsTab(self.notebook, "Video Tools", self.command_manager.commands, self.button_width)
        self.notebook.add(video_tools_tab, text="Video Tools")
        
        # Visualization tab
        visualization_tab = ToolsTab(self.notebook, "Visualization", self.command_manager.commands, self.button_width)
        self.notebook.add(visualization_tab, text="Visualization")
    
    def create_bottom_frame(self, parent):
        """Create bottom frame with help and exit buttons"""
        bottom_frame = ttk.Frame(parent)
        bottom_frame.pack(pady=10)
        
        # Help and Exit buttons
        help_btn = ttk.Button(bottom_frame, text="Help", command=self.display_help)
        help_btn.pack(side="left", padx=5)
        
        exit_btn = ttk.Button(bottom_frame, text="Exit", command=self.quit_app)
        exit_btn.pack(side="left", padx=5)
        
        # License frame
        license_frame = ttk.Frame(bottom_frame)
        license_frame.pack(side="left", padx=(20, 0))
        
        # License text
        license_text = ttk.Label(license_frame, text="© 2025 ", font=("default", 11))
        license_text.pack(side="left")
        
        # License link
        license_link = ttk.Label(license_frame, text="vailá", 
                                font=("default", 11, "italic"))
        license_link.pack(side="left")
        license_link.bind("<Button-1>", lambda e: webbrowser.open("https://doi.org/10.48550/arXiv.2410.07238"))
        
        # License continuation
        license_cont = ttk.Label(license_frame, 
                                text=" - Multimodal Toolbox. Licensed under the GNU Lesser General Public License v3.0.",
                                font=("default", 11))
        license_cont.pack(side="left")
    
    def open_project_directory(self, event=None):
        """Open project directory"""
        if platform.system() == "Windows":
            os.startfile(os.getcwd())
        else:
            subprocess.Popen(["xdg-open", os.getcwd()])
    
    def open_link(self):
        """Open project GitHub link"""
        webbrowser.open("https://github.com/vaila-multimodaltoolbox/vaila")
    
    def open_terminal_shell(self):
        """Open terminal with conda environment"""
        if platform.system() == "Darwin":  # macOS
            subprocess.Popen([
                "osascript", "-e",
                'tell application "Terminal" to do script "source ~/anaconda3/etc/profile.d/conda.sh && conda activate vaila && xonsh"'
            ])
        elif platform.system() == "Windows":  # Windows
            subprocess.Popen(
                "start pwsh -NoExit -Command \"& 'C:\\ProgramData\\anaconda3\\shell\\condabin\\conda-hook.ps1'; conda activate vaila; xonsh\"",
                shell=True,
            )
        elif platform.system() == "Linux":  # Linux
            subprocess.Popen([
                "x-terminal-emulator", "-e", "bash", "-c",
                "source ~/anaconda3/etc/profile.d/conda.sh && conda activate vaila && xonsh"
            ], start_new_session=True)
    
    def display_help(self):
        """Display help file"""
        help_file_path = os.path.join(os.path.dirname(__file__), "docs", "help.html")
        if os.path.exists(help_file_path):
            if platform.system() == "Windows":
                os.system(f"start {help_file_path}")
            else:
                os.system(f"open {help_file_path}")
        else:
            messagebox.showerror("Error", "Help file not found.")
    
    def quit_app(self):
        """Quit the application cleanly"""
        self.destroy()

if __name__ == "__main__":
    # Print startup info
    print("""
vailá - 20.January.2025 v0.9.9 (Python 3.12.11) - Refactored Version
================================================================================
                                             o
                                _,  o |\  _,/
                          |  |_/ |  | |/ / |
                           \/  \/|_/|/|_/\/|_/                   
##########################################################################
                    Improved Architecture & User Experience
================================================================================
""")
    
    app = VailaRefactored()
    app.mainloop() 