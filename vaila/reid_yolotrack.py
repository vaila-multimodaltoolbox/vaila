"""
Project: vailá
Script: reid_yolotrack.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 20 March 2025
Version: 0.01

Description:
    This script performs post-processing on tracker CSV files to correct ID inconsistencies:
    - Merges multiple IDs that belong to the same person
    - Corrects ID switches between different people
    - Uses ReID (re-identification) features from boxmot for identity matching
    - Creates corrected CSV files and an updated visualization video

Usage:
    Run the script from the command line:
        python reid_yolotrack.py

Requirements:
    - Python 3.x
    - OpenCV
    - PyTorch
    - boxmot (for ReID feature extraction)
    - numpy, pandas (for data processing)
    - Additional dependencies as imported
"""

import os
from rich import print
import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import glob
from boxmot.deep import StrongSORT
from boxmot.reid_models.frameworks.torch import ReIdentifier
import colorsys
import csv

# Configuration to avoid library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_num_threads(1)

# First, install boxmot if not already installed
try:
    import boxmot
except ImportError:
    import subprocess

    print("Installing boxmot package...")
    subprocess.check_call(["pip", "install", "boxmot"])

# Now use the correct imports based on the installed boxmot version
try:
    # Try current structure (newer versions)
    from boxmot.trackers.strongsort.strong_sort import StrongSORT
    from boxmot.appearance.reid_model_factory import ReIDFactory

    def get_reid_model(weights="osnet_x0_25_msmt17.pt", device="cpu"):
        reid_factory = ReIDFactory()
        return reid_factory.get_model(weights, device)

except ImportError:
    # Fall back to older structure if needed
    try:
        from boxmot.deep import StrongSORT
        from boxmot.reid_models.frameworks.torch import ReIdentifier

        def get_reid_model(weights="osnet_x0_25_msmt17.pt", device="cpu"):
            return ReIdentifier(model_weights=weights, device=device)

    except ImportError:
        print(
            "Could not import required modules from boxmot. Please ensure it's installed correctly."
        )
        print("Try: pip install -U boxmot")
        raise


def get_color_for_id(tracker_id):
    """Generate a distinct color for each tracker ID using a combination of techniques."""
    # Predefined distinct colors for the first 10 IDs (in BGR)
    distinct_colors = [
        (255, 0, 0),  # Blue
        (0, 0, 255),  # Red
        (0, 255, 0),  # Green
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (255, 255, 0),  # Cyan
        (128, 0, 255),  # Purple
        (0, 165, 255),  # Orange
        (255, 255, 255),  # White
        (0, 0, 0),  # Black
    ]

    # Use predefined colors for first few IDs
    if 0 <= tracker_id < len(distinct_colors):
        return distinct_colors[tracker_id]

    # For higher IDs, use more sophisticated color generation to ensure uniqueness
    # Using prime numbers and golden ratio for better distribution
    phi = 0.618033988749895  # Golden ratio conjugate

    # Use different methods based on ID ranges for more variety
    if tracker_id < 50:
        h = ((tracker_id * 11) % 32) / 32.0
        s = 0.9
        v = 0.95
    elif tracker_id < 100:
        h = ((tracker_id * 7) % 32) / 32.0
        s = 0.85
        v = 0.9
    else:
        # Golden ratio method for very high IDs
        h = (tracker_id * phi) % 1.0
        s = 0.8 + (((tracker_id * 13) % 20) / 100.0)  # 0.8-1.0
        v = 0.9 + (((tracker_id * 17) % 10) / 100.0)  # 0.9-1.0

    rgb = colorsys.hsv_to_rgb(h, s, v)
    # Convert to BGR (OpenCV format) with values 0-255
    return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))


def get_rgb_color_string(color):
    """Convert BGR color tuple to RGB string format for CSV storage."""
    b, g, r = color  # OpenCV uses BGR order
    return f"{r},{g},{b}"


class ReidProcessor:
    def __init__(self, input_dir, reid_threshold=0.6):
        """
        Initialize the ReID processor.

        Args:
            input_dir: Directory containing CSV files and original video
            reid_threshold: Similarity threshold for considering two IDs as the same person
        """
        self.input_dir = input_dir
        self.reid_threshold = reid_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reid_weights = "osnet_x0_25_msmt17.pt"  # Default ReID model

        # Initialize ReID model
        self.reid_model = get_reid_model(self.reid_weights, self.device)

        # Identify the original video file
        self.video_file = self._find_original_video()
        if not self.video_file:
            raise FileNotFoundError("No video file found in the input directory")

        # Load all CSV files
        self.csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
        if not self.csv_files:
            raise FileNotFoundError("No CSV files found in the input directory")

        # Create output directory
        self.output_dir = os.path.join(input_dir, "reid_corrected")
        os.makedirs(self.output_dir, exist_ok=True)

        # Store features for each ID
        self.id_features = {}
        # Store ID mapping (old_id -> new_id)
        self.id_mapping = {}
        # Store each tracking entry by frame
        self.tracks_by_frame = {}

    def _find_original_video(self):
        """Find the original video file in the parent directory."""
        video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
        for ext in video_extensions:
            parent_dir = os.path.dirname(self.input_dir)
            # Try to find the original video in the parent directory
            video_files = glob.glob(os.path.join(parent_dir, f"*{ext}"))
            if video_files:
                return video_files[0]

            # Also check for a processed video in the input directory
            video_files = glob.glob(os.path.join(self.input_dir, f"*{ext}"))
            if video_files:
                return video_files[0]

        return None

    def load_tracking_data(self):
        """Load all tracking data from CSV files."""
        all_data = []

        for csv_file in self.csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Extract tracker ID and label from filename
                filename = os.path.basename(csv_file)
                if "_id" in filename:
                    # Format is typically "{label}_id{tracker_id}.csv"
                    label, id_part = filename.split("_id")
                    tracker_id = int(id_part.split(".")[0])

                    # Add columns if they don't exist
                    if "Tracker ID" not in df.columns:
                        df["Tracker ID"] = tracker_id
                    if "Label" not in df.columns:
                        df["Label"] = label

                # Keep only rows with valid bounding box data
                df = df.dropna(subset=["X_min", "Y_min", "X_max", "Y_max"])

                all_data.append(df)
                print(f"Loaded {len(df)} tracks from {csv_file}")

            except Exception as e:
                print(f"Error loading {csv_file}: {e}")

        # Combine all dataframes
        if all_data:
            self.tracking_df = pd.concat(all_data, ignore_index=True)
            print(f"Total loaded tracks: {len(self.tracking_df)}")

            # Organize tracks by frame for easier processing
            for _, row in self.tracking_df.iterrows():
                frame = int(row["Frame"])
                if frame not in self.tracks_by_frame:
                    self.tracks_by_frame[frame] = []
                self.tracks_by_frame[frame].append(row)
        else:
            raise ValueError("No valid tracking data found in CSV files")

    def extract_reid_features(self):
        """Extract ReID features for each tracked person."""
        print("Extracting ReID features from video...")

        cap = cv2.VideoCapture(self.video_file)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_file}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 100 == 0:
                print(f"Processing frame {frame_idx}/{total_frames}")

            # Check if we have tracking data for this frame
            if frame_idx in self.tracks_by_frame:
                for track in self.tracks_by_frame[frame_idx]:
                    tracker_id = int(track["Tracker ID"])
                    label = track["Label"]

                    # Extract bounding box
                    x_min, y_min = int(track["X_min"]), int(track["Y_min"])
                    x_max, y_max = int(track["X_max"]), int(track["Y_max"])

                    # Ensure valid bounding box
                    if x_min < 0 or y_min < 0 or x_max <= x_min or y_max <= y_min:
                        continue

                    # Extract person image
                    person_img = frame[y_min:y_max, x_min:x_max]

                    # Extract ReID features
                    feature = self.reid_model(person_img)

                    # Store feature for this ID
                    key = (tracker_id, label)
                    if key not in self.id_features:
                        self.id_features[key] = []
                    self.id_features[key].append(feature)

            frame_idx += 1

        cap.release()
        print(f"Extracted features for {len(self.id_features)} unique IDs")

    def compute_average_features(self):
        """Compute average feature for each ID."""
        for key in self.id_features:
            if self.id_features[key]:
                # Stack all features and compute mean
                features = torch.stack(self.id_features[key])
                self.id_features[key] = torch.mean(features, dim=0)

    def cluster_similar_ids(self):
        """Cluster similar IDs based on feature similarity."""
        print("Clustering similar IDs...")

        # Sort keys for deterministic processing
        keys = sorted(list(self.id_features.keys()))
        processed = set()
        next_new_id = 0

        # First pass: assign new IDs to each original ID
        for key in keys:
            if key in processed:
                continue

            tracker_id, label = key
            cluster = [key]
            processed.add(key)

            # Find similar IDs
            for other_key in keys:
                if other_key in processed or other_key == key:
                    continue

                other_id, other_label = other_key

                # Only compare IDs with the same label
                if label != other_label:
                    continue

                # Compute similarity
                similarity = torch.cosine_similarity(
                    self.id_features[key].unsqueeze(0),
                    self.id_features[other_key].unsqueeze(0),
                )

                if similarity > self.reid_threshold:
                    cluster.append(other_key)
                    processed.add(other_key)

            # Assign new ID to all IDs in this cluster
            for c_key in cluster:
                self.id_mapping[c_key] = next_new_id

            next_new_id += 1

        print(f"Reduced {len(keys)} IDs to {next_new_id} unique identities")

    def create_corrected_csvs(self):
        """Create corrected CSV files with new IDs."""
        print("Creating corrected CSV files...")

        # Group by label and new ID
        new_tracks = {}
        id_colors = {}  # Store color for each ID

        # First, assign a color to each unique new ID
        for key in self.id_mapping.values():
            if key not in id_colors:
                color = get_color_for_id(key)
                id_colors[key] = get_rgb_color_string(color)

        for _, row in self.tracking_df.iterrows():
            tracker_id = int(row["Tracker ID"])
            label = row["Label"]
            key = (tracker_id, label)

            # Get new ID (if exists, otherwise keep the original)
            new_id = self.id_mapping.get(key, tracker_id)

            # Make sure we have a color for this ID
            if new_id not in id_colors:
                color = get_color_for_id(new_id)
                id_colors[new_id] = get_rgb_color_string(color)

            new_key = (new_id, label)
            if new_key not in new_tracks:
                new_tracks[new_key] = []

            # Create a copy of the row with the new ID
            row_data = row.copy()
            row_data["Tracker ID"] = new_id
            # Add color information
            row_data["Color_RGB"] = id_colors[new_id]
            new_tracks[new_key].append(row_data)

        # Create new CSV files
        for (new_id, label), tracks in new_tracks.items():
            df = pd.DataFrame(tracks)

            # Sort by frame
            df = df.sort_values("Frame")

            # Save to CSV
            output_file = os.path.join(self.output_dir, f"{label}_id{new_id}.csv")
            df.to_csv(output_file, index=False)
            print(f"Created {output_file}")

    def create_visualization_video(self):
        """Create a visualization video with corrected IDs."""
        print("Creating visualization video...")

        cap = cv2.VideoCapture(self.video_file)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_file}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create output video
        output_video = os.path.join(self.output_dir, "reid_corrected_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        # Convert tracking data to frame-based structure
        corrected_tracks_by_frame = {}
        id_colors = {}  # Cache for colors

        for _, row in self.tracking_df.iterrows():
            frame = int(row["Frame"])
            tracker_id = int(row["Tracker ID"])
            label = row["Label"]
            key = (tracker_id, label)

            # Get new ID
            new_id = self.id_mapping.get(key, tracker_id)

            if frame not in corrected_tracks_by_frame:
                corrected_tracks_by_frame[frame] = []

            # Update the track with the new ID
            track_data = row.copy()
            track_data["Tracker ID"] = new_id

            # Assign a persistent color for this ID if not already done
            if new_id not in id_colors:
                color = get_color_for_id(new_id)
                id_colors[new_id] = color
                color_rgb_str = get_rgb_color_string(color)
                track_data["Color_RGB"] = color_rgb_str
            else:
                track_data["Color_RGB"] = get_rgb_color_string(id_colors[new_id])

            corrected_tracks_by_frame[frame].append(track_data)

        # Process each frame
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 100 == 0:
                print(f"Processing visualization frame {frame_idx}/{total_frames}")

            # Draw tracks for this frame
            if frame_idx in corrected_tracks_by_frame:
                for track in corrected_tracks_by_frame[frame_idx]:
                    new_id = int(track["Tracker ID"])
                    label = track["Label"]

                    # Extract bounding box
                    x_min, y_min = int(track["X_min"]), int(track["Y_min"])
                    x_max, y_max = int(track["X_max"]), int(track["Y_max"])

                    # Get color from track data or cache
                    if "Color_RGB" in track:
                        # Parse from RGB string to BGR tuple
                        rgb_parts = track["Color_RGB"].split(",")
                        if len(rgb_parts) == 3:
                            r, g, b = map(int, rgb_parts)
                            color = (b, g, r)  # Convert to BGR for OpenCV
                        else:
                            color = id_colors.get(new_id, get_color_for_id(new_id))
                    else:
                        color = id_colors.get(new_id, get_color_for_id(new_id))

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

                    # Display label with new ID
                    label_text = f"{label}_id{new_id}"
                    cv2.putText(
                        frame,
                        label_text,
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

            # Write the frame
            out.write(frame)
            frame_idx += 1

        # Release resources
        cap.release()
        out.release()
        print(f"Created visualization video: {output_video}")

    def process(self):
        """Run the full ReID processing pipeline."""
        print(f"Processing tracking data in: {self.input_dir}")
        print(f"Using ReID model: {self.reid_weights}")
        print(f"Similarity threshold: {self.reid_threshold}")

        self.load_tracking_data()
        self.extract_reid_features()
        self.compute_average_features()
        self.cluster_similar_ids()
        self.create_corrected_csvs()
        self.create_visualization_video()

        print(f"ReID processing complete. Results saved in: {self.output_dir}")


def run_reid_yolotrack():
    """Main function to run the ReID processor."""
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    root = tk.Tk()
    root.withdraw()

    # Select input directory (where CSV files are located)
    input_dir = filedialog.askdirectory(
        title="Select directory with tracking CSV files"
    )
    if not input_dir:
        print("No directory selected. Exiting.")
        return

    # Get ReID threshold from user
    threshold = 0.6  # Default threshold
    threshold_str = tk.simpledialog.askstring(
        "ReID Threshold",
        "Enter similarity threshold (0-1):\n"
        "Higher values (e.g., 0.8): More strict matching, fewer merges\n"
        "Lower values (e.g., 0.5): More lenient matching, more merges\n"
        "Recommended: 0.6-0.7",
        initialvalue="0.6",
    )

    if threshold_str:
        try:
            threshold = float(threshold_str)
            if threshold <= 0 or threshold >= 1:
                messagebox.showwarning(
                    "Warning", "Threshold must be between 0 and 1. Using default (0.6)."
                )
                threshold = 0.6
        except ValueError:
            messagebox.showwarning("Warning", "Invalid threshold. Using default (0.6).")

    try:
        processor = ReidProcessor(input_dir, reid_threshold=threshold)
        processor.process()
        messagebox.showinfo(
            "Success",
            f"ReID processing complete.\nResults saved in: {processor.output_dir}",
        )
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        raise

    root.destroy()


if __name__ == "__main__":
    run_reid_yolotrack()
