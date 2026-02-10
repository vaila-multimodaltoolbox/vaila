"""
================================================================================
Combined Marker Re-identification and Video Tool - reidvideogui.py
================================================================================
Author: Paulo R. P. Santiago
Date: 2025-03-27
Updated: 2025-03-27
Version: 0.1.0
Python Version: 3.12.9

Description:
------------
This tool combines the functionalities of reid_markers.py and getpixelvideo.py
to provide a synchronized view of marker trajectories alongside the video.
It allows for:
1. Visualization of marker positions in the video
2. Plotting of marker trajectories (x,y coordinates)
3. Editing tools for marker correction (merging, gap filling, swapping)
4. Synchronized frame selection between video and plots

================================================================================
"""

import os
import sys
from tkinter import Tk, filedialog

import cv2
import pandas as pd

# Garantir que o Qt encontre seus plugins
from PySide6 import QtCore

# Exibir diretório de plugins do Qt para diagnóstico
qt_plugin_path = QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.PluginsPath)
print(f"Qt plugins directory: {qt_plugin_path}")

# Se estiver em um ambiente virtual, adicione o caminho dos plugins
if hasattr(sys, "frozen"):
    os.environ["QT_PLUGIN_PATH"] = os.path.join(
        os.path.dirname(sys.executable), "plugins", "platforms"
    )
elif "VIRTUAL_ENV" in os.environ:
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
        os.environ["VIRTUAL_ENV"],
        "Lib",
        "site-packages",
        "PySide6",
        "plugins",
        "platforms",
    )

# Matplotlib setup for PySide6
import matplotlib

matplotlib.use("QtAgg")  # Usar backend genérico QtAgg

# Import Figure and FigureCanvas
from getpixelvideo import get_color_for_id
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

# PySide6 imports
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# Import functions from both source files
from reid_markers import (
    detect_markers,
    fill_gaps,
    get_marker_coords,
    load_markers_file,
    merge_markers,
    save_markers_file,
    swap_markers,
)


class MplCanvas(FigureCanvas):
    """Matplotlib canvas for plotting trajectories"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax_x = self.fig.add_subplot(211)  # X coordinates plot
        self.ax_y = self.fig.add_subplot(212)  # Y coordinates plot
        super().__init__(self.fig)
        self.fig.tight_layout()


class CombinedMarkerGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # Data variables
        self.df = None
        self.file_path = None
        self.video_path = None
        self.markers = []
        self.current_frame = 0
        self.total_frames = 0
        self.frame_width = 0
        self.frame_height = 0
        self.video_cap = None
        self.playing = False
        self.selected_markers = []
        self.frame_range = (0, 0)

        # History for undo
        self.history = []

        # Set up the UI
        self.initUI()

        # Timer for video playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

    def initUI(self):
        """Initialize the user interface"""
        self.setWindowTitle("Combined Marker Re-identification and Video Tool")
        self.setGeometry(100, 100, 1200, 800)

        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # Create splitter for video and plots
        splitter = QSplitter(Qt.Horizontal)

        # Video panel on the left
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)

        # Video display
        self.video_label = QLabel("Load a video file")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(400, 300)
        self.video_label.setStyleSheet("background-color: black;")
        video_layout.addWidget(self.video_label)

        # Video controls
        video_controls = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        self.prev_button = QPushButton("< Prev")
        self.prev_button.clicked.connect(self.prev_frame)
        self.next_button = QPushButton("Next >")
        self.next_button.clicked.connect(self.next_frame)

        video_controls.addWidget(self.prev_button)
        video_controls.addWidget(self.play_button)
        video_controls.addWidget(self.next_button)

        video_layout.addLayout(video_controls)

        # Frame slider
        slider_layout = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)  # Will be updated when video is loaded
        self.frame_slider.valueChanged.connect(self.slider_changed)
        self.frame_label = QLabel("Frame: 0/0")

        slider_layout.addWidget(self.frame_slider)
        slider_layout.addWidget(self.frame_label)

        video_layout.addLayout(slider_layout)

        # Add marker selection section
        marker_group = QGroupBox("Marker Selection")
        marker_layout = QVBoxLayout()
        self.marker_checkboxes = []
        self.marker_layout_widget = QWidget()
        self.marker_checkbox_layout = QVBoxLayout(self.marker_layout_widget)

        # Initially empty, will be populated when data is loaded
        marker_layout.addWidget(self.marker_layout_widget)

        # Select All/None buttons
        select_buttons = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_markers)
        select_none_btn = QPushButton("Select None")
        select_none_btn.clicked.connect(self.select_no_markers)

        select_buttons.addWidget(select_all_btn)
        select_buttons.addWidget(select_none_btn)
        marker_layout.addLayout(select_buttons)

        marker_group.setLayout(marker_layout)
        video_layout.addWidget(marker_group)

        # Plots panel on the right
        plots_widget = QWidget()
        plots_layout = QVBoxLayout(plots_widget)

        # Create matplotlib canvas
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        plots_layout.addWidget(self.canvas)

        # Frame range selection
        range_group = QGroupBox("Frame Range")
        range_layout = QVBoxLayout()

        self.range_slider = QSlider(Qt.Horizontal)
        self.range_slider.setMinimum(0)
        self.range_slider.setMaximum(100)  # Will be updated when data is loaded
        self.range_slider.valueChanged.connect(self.range_slider_changed)
        self.range_label = QLabel("Start: 0, End: 0")

        range_layout.addWidget(self.range_slider)
        range_layout.addWidget(self.range_label)
        range_group.setLayout(range_layout)
        plots_layout.addWidget(range_group)

        # Editing tools
        tools_group = QGroupBox("Editing Tools")
        tools_layout = QGridLayout()

        self.fill_button = QPushButton("Fill Gaps")
        self.fill_button.clicked.connect(self.fill_gaps)
        self.merge_button = QPushButton("Merge Markers")
        self.merge_button.clicked.connect(self.merge_markers)
        self.swap_button = QPushButton("Swap Markers")
        self.swap_button.clicked.connect(self.swap_markers)

        tools_layout.addWidget(self.fill_button, 0, 0)
        tools_layout.addWidget(self.merge_button, 0, 1)
        tools_layout.addWidget(self.swap_button, 0, 2)

        tools_group.setLayout(tools_layout)
        plots_layout.addWidget(tools_group)

        # Add widgets to splitter
        splitter.addWidget(video_widget)
        splitter.addWidget(plots_widget)
        splitter.setSizes([400, 800])

        # Add splitter to main layout
        main_layout.addWidget(splitter)

        # File operations
        file_layout = QHBoxLayout()
        load_button = QPushButton("Load Data")
        load_button.clicked.connect(self.load_data)
        save_button = QPushButton("Save Changes")
        save_button.clicked.connect(self.save_changes)

        file_layout.addWidget(load_button)
        file_layout.addWidget(save_button)
        main_layout.addLayout(file_layout)

        self.setCentralWidget(main_widget)

        # Initialize UI as disabled until data is loaded
        self.enable_controls(False)

    def enable_controls(self, enabled=True):
        """Enable or disable UI controls"""
        self.play_button.setEnabled(enabled)
        self.prev_button.setEnabled(enabled)
        self.next_button.setEnabled(enabled)
        self.frame_slider.setEnabled(enabled)
        self.range_slider.setEnabled(enabled)
        self.fill_button.setEnabled(enabled)
        self.merge_button.setEnabled(enabled)
        self.swap_button.setEnabled(enabled)

    def load_data(self):
        """Load marker data and associated video"""
        # Load marker file
        self.df, self.file_path = load_markers_file()
        if self.df is None:
            return

        # Extract markers and prepare data
        self.markers = detect_markers(self.df)

        # Create marker checkboxes
        self.create_marker_checkboxes()

        # Try to find video
        self.find_video()

        # Initialize video if found
        if self.video_path and os.path.exists(self.video_path):
            self.init_video()
        else:
            QMessageBox.warning(
                self,
                "Video Not Found",
                "No corresponding video found. Only trajectory plotting will be available.",
            )

        # Update frame range slider
        self.total_frames = len(self.df)
        self.frame_slider.setMaximum(self.total_frames - 1)
        self.range_slider.setMaximum(self.total_frames - 1)
        self.frame_range = (0, self.total_frames - 1)
        self.range_label.setText(f"Start: 0, End: {self.total_frames - 1}")

        # Update plot
        self.update_plot()

        # Enable controls
        self.enable_controls(True)

    def find_video(self):
        """Try to find the video associated with the marker file."""
        if not self.file_path:
            return

        # Try different strategies to find the video
        base_dir = os.path.dirname(self.file_path)
        base_name = os.path.basename(self.file_path).split(".")[0]

        # Common video extensions
        extensions = [".mp4", ".avi", ".mov", ".mkv"]

        # Check for videos with similar names in the same directory
        for ext in extensions:
            possible_path = os.path.join(base_dir, f"{base_name}{ext}")
            if os.path.exists(possible_path):
                self.video_path = possible_path
                return

        # If not found, let the user select
        root = Tk()
        root.withdraw()
        self.video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")],
            initialdir=base_dir,
        )
        if not self.video_path:
            self.video_path = None

    def init_video(self):
        """Initialize video capture and parameters."""
        if not self.video_path:
            return

        # Open video file
        self.video_cap = cv2.VideoCapture(self.video_path)
        if not self.video_cap.isOpened():
            QMessageBox.critical(self, "Error", f"Could not open video file: {self.video_path}")
            self.video_cap = None
            return

        # Get video properties
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Show first frame
        self.update_video_frame(0)

    def create_marker_checkboxes(self):
        """Create checkboxes for marker selection"""
        # Clear existing checkboxes
        for cb in self.marker_checkboxes:
            self.marker_checkbox_layout.removeWidget(cb)
            cb.deleteLater()

        self.marker_checkboxes = []

        # Create new checkboxes
        for marker_id in self.markers:
            cb = QCheckBox(f"Marker {marker_id}")
            cb.setChecked(True)  # All markers selected by default
            cb.stateChanged.connect(self.marker_selection_changed)
            self.marker_checkbox_layout.addWidget(cb)
            self.marker_checkboxes.append(cb)

        # Update selected markers list
        self.update_selected_markers()

    def update_selected_markers(self):
        """Update the list of selected markers based on checkboxes"""
        self.selected_markers = []
        for i, cb in enumerate(self.marker_checkboxes):
            if cb.isChecked():
                self.selected_markers.append(self.markers[i])

    def marker_selection_changed(self):
        """Handle marker selection change"""
        self.update_selected_markers()
        self.update_plot()
        if self.video_cap:
            self.update_video_frame(self.current_frame)

    def select_all_markers(self):
        """Select all markers"""
        for cb in self.marker_checkboxes:
            cb.setChecked(True)

    def select_no_markers(self):
        """Deselect all markers"""
        for cb in self.marker_checkboxes:
            cb.setChecked(False)

    def update_plot(self):
        """Update the trajectory plots based on marker selection"""
        # Clear axes
        self.canvas.ax_x.clear()
        self.canvas.ax_y.clear()

        # Set up axes labels
        self.canvas.ax_x.set_title(f"X Coordinates - Frame: {self.current_frame}")
        self.canvas.ax_x.set_xlabel("Frame")
        self.canvas.ax_x.set_ylabel("X Position")
        self.canvas.ax_x.grid(True)

        self.canvas.ax_y.set_title("Y Coordinates of Markers")
        self.canvas.ax_y.set_xlabel("Frame")
        self.canvas.ax_y.set_ylabel("Y Position")
        self.canvas.ax_y.grid(True)

        # Plot markers
        if self.df is not None:
            frames = self.df["frame"].values

            for marker_id in self.selected_markers:
                x_values, y_values = get_marker_coords(self.df, marker_id)
                if x_values is not None and y_values is not None:
                    # Get the color for this marker
                    color_rgb = get_color_for_id(marker_id)
                    # Normalize to 0-1 for matplotlib
                    normalized_color = (
                        color_rgb[0] / 255,
                        color_rgb[1] / 255,
                        color_rgb[2] / 255,
                    )

                    self.canvas.ax_x.plot(
                        frames,
                        x_values,
                        label=f"Marker {marker_id}",
                        color=normalized_color,
                    )
                    self.canvas.ax_y.plot(
                        frames,
                        y_values,
                        label=f"Marker {marker_id}",
                        color=normalized_color,
                    )

            # Show frame range
            start_frame, end_frame = self.frame_range
            self.canvas.ax_x.axvline(start_frame, color="r", linestyle="--", alpha=0.5)
            self.canvas.ax_x.axvline(end_frame, color="r", linestyle="--", alpha=0.5)
            self.canvas.ax_y.axvline(start_frame, color="r", linestyle="--", alpha=0.5)
            self.canvas.ax_y.axvline(end_frame, color="r", linestyle="--", alpha=0.5)

            # Add current frame indicator
            self.canvas.ax_x.axvline(self.current_frame, color="g", linestyle="-", linewidth=2)
            self.canvas.ax_y.axvline(self.current_frame, color="g", linestyle="-", linewidth=2)

            # Set limits
            self.canvas.ax_x.set_xlim(0, len(self.df) - 1)
            self.canvas.ax_y.set_xlim(0, len(self.df) - 1)

            # Add legend if there are markers to show
            if self.selected_markers:
                self.canvas.ax_x.legend(loc="upper right")
                self.canvas.ax_y.legend(loc="upper right")

        # Draw the updated canvas
        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def update_video_frame(self, frame_num):
        """Update the video display with the given frame"""
        if not self.video_cap:
            return

        # Set the video position to the current frame
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        # Read the frame
        ret, frame = self.video_cap.read()
        if not ret:
            return

        # Draw markers on frame
        for marker_id in self.selected_markers:
            x_col = f"p{marker_id}_x"
            y_col = f"p{marker_id}_y"

            if x_col in self.df.columns and y_col in self.df.columns:
                if frame_num < len(self.df):
                    x = self.df.at[frame_num, x_col]
                    y = self.df.at[frame_num, y_col]

                    if not pd.isna(x) and not pd.isna(y):
                        # Get color for this marker
                        color = get_color_for_id(marker_id)
                        # Convert RGB to BGR for OpenCV
                        color_bgr = (color[2], color[1], color[0])

                        # Draw circle for marker
                        cv2.circle(frame, (int(x), int(y)), 5, color_bgr, -1)

                        # Draw text for marker ID
                        cv2.putText(
                            frame,
                            str(marker_id),
                            (int(x) + 10, int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color_bgr,
                            2,
                            cv2.LINE_AA,
                        )

        # Add frame number to the frame
        cv2.putText(
            frame,
            f"Frame: {frame_num}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Convert to QImage for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Scale to fit video label
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(
            pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
        )

        # Update frame label
        self.frame_label.setText(f"Frame: {frame_num}/{self.total_frames - 1}")

    def toggle_play(self):
        """Toggle video playback"""
        if not self.video_cap:
            return

        if self.playing:
            self.timer.stop()
            self.playing = False
            self.play_button.setText("Play")
        else:
            self.timer.start(33)  # ~30 fps
            self.playing = True
            self.play_button.setText("Pause")

    def next_frame(self):
        """Go to next frame"""
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.frame_slider.setValue(self.current_frame)
            self.update_video_frame(self.current_frame)
            self.update_plot()
        elif self.playing:
            # Reached the end, stop playing
            self.toggle_play()

    def prev_frame(self):
        """Go to previous frame"""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.frame_slider.setValue(self.current_frame)
            self.update_video_frame(self.current_frame)
            self.update_plot()

    def slider_changed(self):
        """Handle frame slider value change"""
        self.current_frame = self.frame_slider.value()
        self.update_video_frame(self.current_frame)
        self.update_plot()

    def range_slider_changed(self):
        """Handle range slider value change"""
        value = self.range_slider.value()
        # Simple solution: use slider value as the start frame
        # and slider value + 20% of total as end frame
        start_frame = value
        end_frame = min(self.total_frames - 1, int(value + self.total_frames * 0.2))
        self.frame_range = (start_frame, end_frame)
        self.range_label.setText(f"Start: {start_frame}, End: {end_frame}")
        self.update_plot()

    def fill_gaps(self):
        """Fill gaps in selected markers"""
        if not self.selected_markers:
            QMessageBox.information(
                self,
                "Selection Needed",
                "Please select at least one marker to fill gaps.",
            )
            return

        # Backup for undo
        self.history.append(self.df.copy())

        # Fill gaps for each selected marker
        for marker_id in self.selected_markers:
            self.df = fill_gaps(self.df, marker_id)

        # Update display
        self.update_plot()
        if self.video_cap:
            self.update_video_frame(self.current_frame)

        QMessageBox.information(
            self, "Success", f"Filled gaps for {len(self.selected_markers)} markers."
        )

    def merge_markers(self):
        """Merge selected markers"""
        if len(self.selected_markers) < 2:
            QMessageBox.information(
                self, "Selection Needed", "Please select at least 2 markers to merge."
            )
            return

        # Backup for undo
        self.history.append(self.df.copy())

        # First marker is target, others are sources
        target_id = self.selected_markers[0]
        source_ids = self.selected_markers[1:]

        start_frame, end_frame = self.frame_range

        # Merge each source into the target
        for source_id in source_ids:
            self.df = merge_markers(self.df, source_id, target_id, (start_frame, end_frame))

        # Update display
        self.update_plot()
        if self.video_cap:
            self.update_video_frame(self.current_frame)

        QMessageBox.information(
            self,
            "Success",
            f"Merged markers {', '.join(map(str, source_ids))} into marker {target_id}.",
        )

    def swap_markers(self):
        """Swap selected markers"""
        if len(self.selected_markers) != 2:
            QMessageBox.information(
                self, "Selection Needed", "Please select exactly 2 markers to swap."
            )
            return

        # Backup for undo
        self.history.append(self.df.copy())

        # Get markers and frame range
        marker_id1, marker_id2 = self.selected_markers
        start_frame, end_frame = self.frame_range

        # Swap markers
        self.df = swap_markers(self.df, marker_id1, marker_id2, (start_frame, end_frame))

        # Update display
        self.update_plot()
        if self.video_cap:
            self.update_video_frame(self.current_frame)

        QMessageBox.information(
            self,
            "Success",
            f"Swapped markers {marker_id1} and {marker_id2} in frames {start_frame}-{end_frame}.",
        )

    def save_changes(self):
        """Save changes to markers file"""
        if self.df is None or self.file_path is None:
            return

        # Save to a new file with _edited suffix
        save_path = save_markers_file(self.df, self.file_path, suffix="_edited")
        QMessageBox.information(self, "Saved", f"Markers saved to: {save_path}")

    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        if hasattr(self, "video_label") and self.video_cap:
            # Refresh the video frame to fit the new size
            self.update_video_frame(self.current_frame)


def main():
    app = QApplication(sys.argv)
    window = CombinedMarkerGUI()
    window.show()
    sys.exit(app.exec())  # No PySide6, é exec() sem underscore


if __name__ == "__main__":
    main()
