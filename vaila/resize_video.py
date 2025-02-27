"""
resize_video.py

Description:
-----------
This script provides tools for improving pose detection in videos:
1. Batch resize videos to higher resolutions (2x-8x)
2. Crop specific regions of interest and resize them
3. Convert MediaPipe pixel coordinates back to original video coordinates

Version:
--------
0.3.0
date: 2025-02-27

Author:
-------
Prof. PhD. Paulo Santiago

License:
--------
This code is licensed under the MIT License.

Dependencies:
-------------
- Python 3.12.9
- opencv-python
- tkinter
- pandas (for coordinates conversion)
"""

import os
import cv2
import tkinter as tk
from tkinter import filedialog, Button, Label, Frame, StringVar, messagebox, Radiobutton
import threading
from datetime import datetime
import numpy as np
import json
import glob
import pandas as pd

def get_video_info(video_path):
    """Get video information using OpenCV."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration
        }
    except Exception as e:
        print(f"Error getting video info: {e}")
        raise e

def resize_with_opencv(input_file, output_file, scale_factor, roi=None, progress_callback=None):
    """
    Resize video using OpenCV, optionally cropping to a region of interest.
    
    Args:
        input_file (str): Path to input video file
        output_file (str): Path to output video file
        scale_factor (int): Factor by which to scale the video resolution
        roi (tuple, optional): Region of interest as (x, y, width, height)
        progress_callback (function, optional): Function to call with progress updates
    
    Returns:
        dict: Processing metadata including original and new dimensions, crop info, etc.
    """
    try:
        print(f"Processing video: {input_file}")
        
        # Open input video
        cap = cv2.VideoCapture(input_file)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_file}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Store video metadata
        metadata = {
            "original_video": os.path.basename(input_file),
            "original_width": width,
            "original_height": height,
            "original_fps": fps,
            "original_frames": total_frames,
            "scale_factor": scale_factor,
        }
        
        # Determine output dimensions
        if roi:
            # If ROI is provided, use it
            x, y, w, h = roi
            new_width = int(w * scale_factor)
            new_height = int(h * scale_factor)
            message = f"Cropping to {w}x{h} and resizing to {new_width}x{new_height}"
            
            # Add crop info to metadata
            metadata["crop"] = {
                "x": x,
                "y": y,
                "width": w,
                "height": h
            }
            metadata["crop_applied"] = True
        else:
            # Full frame resize
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            message = f"Resizing from {width}x{height} to {new_width}x{new_height}"
            metadata["crop_applied"] = False
        
        metadata["output_width"] = new_width
        metadata["output_height"] = new_height
        metadata["output_video"] = os.path.basename(output_file)
        
        if progress_callback:
            progress_callback(message)
        print(message)
        
        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID is more reliable across platforms
        out = cv2.VideoWriter(output_file, fourcc, fps, (new_width, new_height))
        
        # Process the video frame by frame
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply ROI if specified
            if roi:
                x, y, w, h = roi
                # Ensure ROI is within frame boundaries
                x = max(0, min(x, width-1))
                y = max(0, min(y, height-1))
                w = min(w, width-x)
                h = min(h, height-y)
                
                # Crop the frame
                frame = frame[y:y+h, x:x+w]
            
            # Resize the frame
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Write the frame to the output video
            out.write(resized_frame)
            
            # Update progress every 10 frames
            frame_count += 1
            if frame_count % 10 == 0:
                progress = (frame_count / total_frames) * 100
                status = f"Processing: {progress:.1f}% ({frame_count}/{total_frames})"
                print(status)
                if progress_callback:
                    progress_callback(status)
        
        # Release resources
        cap.release()
        out.release()
        
        # Save metadata to a JSON file with the same name as the output video
        metadata_file = os.path.splitext(output_file)[0] + "_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print(f"Metadata saved to: {metadata_file}")
        if progress_callback:
            progress_callback(f"Metadata saved to: {os.path.basename(metadata_file)}")
        
        print(f"Video processed successfully: {output_file}")
        if progress_callback:
            progress_callback(f"Completed! Output: {output_file}")
        
        # Return processing metadata
        return metadata
        
    except Exception as e:
        error_msg = f"Error during video processing: {str(e)}"
        print(error_msg)
        if progress_callback:
            progress_callback(f"Error: {str(e)}")
        return None

def convert_coordinates(x, y, metadata):
    """
    Convert coordinates from processed video back to original video.
    
    Args:
        x (float): X coordinate in the processed video
        y (float): Y coordinate in the processed video
        metadata (dict): Video processing metadata from JSON file
    
    Returns:
        tuple: (original_x, original_y) coordinates in the original video
    """
    if not metadata["crop_applied"]:
        # If no crop was applied, just divide by scale factor
        original_x = x / metadata["scale_factor"]
        original_y = y / metadata["scale_factor"]
    else:
        # If crop was applied, first divide by scale factor, then add crop offset
        crop = metadata["crop"]
        original_x = (x / metadata["scale_factor"]) + crop["x"]
        original_y = (y / metadata["scale_factor"]) + crop["y"]
    
    return original_x, original_y

def convert_mediapipe_coordinates(pixel_csv_path, metadata_path, output_csv_path, progress_callback=None):
    """
    Convert MediaPipe pixel coordinates from processed video back to original video.
    
    Args:
        pixel_csv_path (str): Path to MediaPipe pixel coordinates CSV file
        metadata_path (str): Path to video processing metadata JSON file
        output_csv_path (str): Path to save the converted coordinates CSV file
        progress_callback (function, optional): Function to call with progress updates
    
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    try:
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if progress_callback:
            progress_callback(f"Loaded metadata from: {os.path.basename(metadata_path)}")
        
        # Load MediaPipe CSV file
        df = pd.read_csv(pixel_csv_path)
        
        if progress_callback:
            progress_callback(f"Loaded MediaPipe data: {os.path.basename(pixel_csv_path)}")
            progress_callback(f"Found {len(df)} frames with landmarks")
        
        # Make a copy of the DataFrame for the converted coordinates
        converted_df = df.copy()
        
        # Detect coordinate columns (should be pairs of x,y columns)
        # Skip the first column which is usually 'frame'
        columns = df.columns[1:]
        coord_columns = []
        
        for col in columns:
            if col.endswith('_x') or col.endswith('_y'):
                coord_columns.append(col)
        
        if progress_callback:
            progress_callback(f"Found {len(coord_columns)} coordinate columns")
        
        # Convert each coordinate column
        for col in coord_columns:
            if col.endswith('_x'):
                x_col = col
                y_col = col.replace('_x', '_y')
                
                # Process each row
                for idx, row in df.iterrows():
                    if pd.notna(row[x_col]) and pd.notna(row[y_col]):
                        # Convert coordinates
                        orig_x, orig_y = convert_coordinates(row[x_col], row[y_col], metadata)
                        
                        # Store in converted DataFrame
                        converted_df.at[idx, x_col] = orig_x
                        converted_df.at[idx, y_col] = orig_y
        
        # Save converted DataFrame
        converted_df.to_csv(output_csv_path, index=False)
        
        if progress_callback:
            progress_callback(f"Converted coordinates saved to: {os.path.basename(output_csv_path)}")
        
        return True
        
    except Exception as e:
        error_msg = f"Error converting coordinates: {str(e)}"
        print(error_msg)
        if progress_callback:
            progress_callback(f"Error: {str(e)}")
        return False

class ROISelector:
    """Class to select a region of interest on a video frame"""
    def __init__(self, video_path):
        self.video_path = video_path
        self.roi = None
        self.roi_selected = False
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.frame = None
        self.scale = 1.0
        
    def select_roi(self):
        """Open a window to select ROI on the first frame"""
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error opening video: {self.video_path}")
            return None
        
        # Read first frame
        ret, self.frame = cap.read()
        if not ret:
            print("Failed to read frame from video")
            cap.release()
            return None
        
        # Get original dimensions
        height, width = self.frame.shape[:2]
        
        # Scale down if too large for display
        max_display_width = 1280
        max_display_height = 720
        
        if width > max_display_width or height > max_display_height:
            # Calculate scale factor
            scale_w = max_display_width / width
            scale_h = max_display_height / height
            self.scale = min(scale_w, scale_h)
            
            # Resize frame for display
            display_width = int(width * self.scale)
            display_height = int(height * self.scale)
            display_frame = cv2.resize(self.frame, (display_width, display_height))
        else:
            self.scale = 1.0
            display_frame = self.frame.copy()
        
        # Create window and set mouse callback
        window_name = "Select Region of Interest (ROI) - Press Enter when done, ESC to cancel"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.draw_rectangle, param=display_frame)
        
        # Display instructions
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display_frame, "Click and drag to select ROI", (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press Enter to confirm, ESC to cancel", (10, 60), font, 0.7, (0, 255, 0), 2)
        
        while True:
            # Display the image
            cv2.imshow(window_name, display_frame)
            
            # Wait for key press
            key = cv2.waitKey(1) & 0xFF
            
            # Enter key - confirm selection
            if key == 13 and self.roi_selected:
                # Convert the ROI coordinates back to original scale
                x, y, w, h = self.roi
                x = int(x / self.scale)
                y = int(y / self.scale)
                w = int(w / self.scale)
                h = int(h / self.scale)
                self.roi = (x, y, w, h)
                break
            
            # Escape key - cancel
            if key == 27:
                self.roi = None
                break
        
        # Clean up
        cv2.destroyAllWindows()
        cap.release()
        
        return self.roi
    
    def draw_rectangle(self, event, x, y, flags, param):
        """Mouse callback function for ROI selection"""
        img = param.copy()
        
        # Mouse left button down - start drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            
        # Mouse move while button down - update rectangle
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Draw rectangle on display image
            cv2.rectangle(img, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Select Region of Interest (ROI) - Press Enter when done, ESC to cancel", img)
            
        # Mouse left button up - finish drawing
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            # Ensure we have a valid rectangle (width and height > 0)
            roi_x = min(self.ix, x)
            roi_y = min(self.iy, y)
            roi_w = abs(x - self.ix)
            roi_h = abs(y - self.iy)
            
            if roi_w > 0 and roi_h > 0:
                self.roi = (roi_x, roi_y, roi_w, roi_h)
                self.roi_selected = True
                
                # Draw final rectangle
                cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
                
                # Add ROI dimensions
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f"ROI: {roi_w}x{roi_h}"
                cv2.putText(img, text, (roi_x, roi_y-10), font, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Select Region of Interest (ROI) - Press Enter when done, ESC to cancel", img)

def batch_resize_videos():
    """Function to batch process multiple videos"""
    root = tk.Tk()
    root.title("Batch Video Processor")
    root.geometry("550x420")
    
    # Variables to store paths and settings
    input_dir_var = StringVar(value="No directory selected")
    output_dir_var = StringVar(value="No directory selected")
    scale_var = tk.IntVar(value=2)
    
    # Frame for directory selection
    dir_frame = Frame(root, padx=10, pady=10)
    dir_frame.pack(fill=tk.X)
    
    # Input directory selection
    Label(dir_frame, text="Input Directory:").grid(row=0, column=0, sticky=tk.W)
    Label(dir_frame, textvariable=input_dir_var, width=40).grid(row=0, column=1, padx=5)
    Button(dir_frame, text="Browse", command=lambda: select_input_dir(input_dir_var)).grid(row=0, column=2, padx=5)
    
    # Output directory selection
    Label(dir_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W)
    Label(dir_frame, textvariable=output_dir_var, width=40).grid(row=1, column=1, padx=5)
    Button(dir_frame, text="Browse", command=lambda: select_output_dir(output_dir_var)).grid(row=1, column=2, padx=5)
    
    # Scale factor selection
    scale_frame = Frame(root, padx=10, pady=10)
    scale_frame.pack(fill=tk.X)
    
    Label(scale_frame, text="Select scale factor:").pack(anchor=tk.W)
    
    # Create radio buttons for each scale factor
    factors_frame = Frame(scale_frame)
    factors_frame.pack(fill=tk.X, pady=5)
    
    for i, factor in enumerate([2, 3, 4, 5, 6, 7, 8]):
        Radiobutton(
            factors_frame, 
            text=f"{factor}x", 
            variable=scale_var, 
            value=factor
        ).grid(row=0, column=i, padx=10)
    
    # Help text frame
    help_frame = Frame(root, padx=10, pady=5)
    help_frame.pack(fill=tk.X)
    
    help_text = """Note: For cropped videos, a metadata file is saved that allows 
converting MediaPipe coordinates back to the original video dimensions."""
    
    Label(help_frame, text=help_text, justify=tk.LEFT, wraplength=500).pack(fill=tk.X)
    
    # Status display
    status_var = StringVar(value="Ready to process videos")
    status_label = Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
    status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
    
    # Buttons frame
    buttons_frame = Frame(root, padx=20, pady=10)
    buttons_frame.pack(fill=tk.X)
    
    # Process button for full resize
    Button(
        buttons_frame, 
        text="Full Resize", 
        command=lambda: start_batch_processing(
            input_dir_var.get(), 
            output_dir_var.get(), 
            scale_var.get(),
            False,  # No ROI
            status_var,
            root
        ),
        bg="#4CAF50",
        fg="white",
        font=("Arial", 11, "bold"),
        width=20,
        height=2
    ).pack(pady=5)
    
    # Process button for crop and resize
    Button(
        buttons_frame, 
        text="Crop and Resize", 
        command=lambda: crop_and_resize_single(
            input_dir_var.get(),
            output_dir_var.get(),
            scale_var.get(),
            status_var,
            root
        ),
        bg="#2196F3",
        fg="white",
        font=("Arial", 11, "bold"),
        width=20,
        height=2
    ).pack(pady=5)
    
    # Convert MediaPipe coordinates button
    Button(
        buttons_frame, 
        text="Convert MediaPipe Coordinates", 
        command=lambda: convert_mediapipe_coordinates_gui(root, status_var),
        bg="#FFC107",
        fg="black",
        font=("Arial", 11),
        width=20,
        height=2
    ).pack(pady=5)
    
    def select_input_dir(var):
        directory = filedialog.askdirectory(title="Select Directory with Videos")
        if directory:
            var.set(directory)
    
    def select_output_dir(var):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            var.set(directory)
            
    def convert_mediapipe_coordinates_gui(root, status_var):
        """GUI for converting MediaPipe coordinates to original video space"""
        # Create new window for conversion
        convert_window = tk.Toplevel(root)
        convert_window.title("Convert MediaPipe Coordinates")
        convert_window.geometry("600x450")
        
        # Variables for file paths
        metadata_path_var = StringVar(value="No file selected")
        pixel_csv_path_var = StringVar(value="No file selected")
        output_path_var = StringVar(value="No file selected")
        
        # Create frame for file selection
        file_frame = Frame(convert_window, padx=10, pady=10)
        file_frame.pack(fill=tk.X)
        
        # Metadata JSON file selection
        Label(file_frame, text="1. Select Metadata JSON:").grid(row=0, column=0, sticky=tk.W)
        Label(file_frame, textvariable=metadata_path_var, width=40).grid(row=0, column=1, padx=5)
        Button(
            file_frame, 
            text="Browse", 
            command=lambda: select_metadata_file(metadata_path_var)
        ).grid(row=0, column=2, padx=5)
        
        # MediaPipe CSV file selection
        Label(file_frame, text="2. Select MediaPipe Pixel CSV:").grid(row=1, column=0, sticky=tk.W)
        Label(file_frame, textvariable=pixel_csv_path_var, width=40).grid(row=1, column=1, padx=5)
        Button(
            file_frame, 
            text="Browse", 
            command=lambda: select_pixel_csv_file(pixel_csv_path_var)
        ).grid(row=1, column=2, padx=5)
        
        # Output CSV file selection
        Label(file_frame, text="3. Output Converted CSV:").grid(row=2, column=0, sticky=tk.W)
        Label(file_frame, textvariable=output_path_var, width=40).grid(row=2, column=1, padx=5)
        Button(
            file_frame, 
            text="Browse", 
            command=lambda: select_output_csv_file(output_path_var)
        ).grid(row=2, column=2, padx=5)
        
        # Progress text
        progress_frame = Frame(convert_window, padx=10, pady=10)
        progress_frame.pack(fill=tk.BOTH, expand=True)
        
        progress_text = tk.Text(progress_frame, height=15, width=70)
        progress_text.pack(fill=tk.BOTH, expand=True)
        
        # Function to update progress text
        def update_progress(message):
            progress_text.insert(tk.END, message + "\n")
            progress_text.see(tk.END)
            convert_window.update()
        
        # Convert button
        Button(
            convert_window,
            text="Convert Coordinates",
            command=lambda: start_conversion(
                metadata_path_var.get(),
                pixel_csv_path_var.get(),
                output_path_var.get(),
                update_progress,
                status_var
            ),
            bg="#FF5722",
            fg="white",
            font=("Arial", 11, "bold"),
            width=15,
            height=2
        ).pack(pady=10)
        
        def select_metadata_file(var):
            file_path = filedialog.askopenfilename(
                title="Select Metadata JSON File",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if file_path:
                var.set(file_path)
                update_progress(f"Selected metadata file: {os.path.basename(file_path)}")
        
        def select_pixel_csv_file(var):
            file_path = filedialog.askopenfilename(
                title="Select MediaPipe Pixel CSV File",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if file_path:
                var.set(file_path)
                update_progress(f"Selected MediaPipe CSV file: {os.path.basename(file_path)}")
        
        def select_output_csv_file(var):
            file_path = filedialog.asksaveasfilename(
                title="Save Converted Coordinates CSV",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if file_path:
                var.set(file_path)
                update_progress(f"Output will be saved to: {os.path.basename(file_path)}")
        
        def start_conversion(metadata_path, pixel_csv_path, output_path, progress_callback, status_var):
            """Start the coordinate conversion process"""
            if metadata_path == "No file selected":
                messagebox.showerror("Error", "Please select a metadata JSON file")
                return
            
            if pixel_csv_path == "No file selected":
                messagebox.showerror("Error", "Please select a MediaPipe pixel CSV file")
                return
            
            if output_path == "No file selected":
                messagebox.showerror("Error", "Please specify an output CSV file")
                return
            
            # Create a thread to run the conversion
            def conversion_thread():
                try:
                    progress_callback("Starting coordinate conversion...")
                    status_var.set("Converting coordinates...")
                    
                    # Run the conversion
                    result = convert_mediapipe_coordinates(
                        pixel_csv_path, 
                        metadata_path, 
                        output_path,
                        progress_callback
                    )
                    
                    if result:
                        progress_callback("\nConversion completed successfully!")
                        progress_callback(f"Original coordinates saved to: {os.path.basename(output_path)}")
                        status_var.set("Coordinate conversion completed")
                        
                        # Show success message
                        convert_window.after(0, lambda: messagebox.showinfo(
                            "Success", 
                            f"Coordinates successfully converted and saved to:\n{output_path}"
                        ))
                    else:
                        progress_callback("\nConversion failed. See error messages above.")
                        status_var.set("Coordinate conversion failed")
                
                except Exception as e:
                    error_msg = f"Error during conversion: {str(e)}"
                    progress_callback(error_msg)
                    status_var.set("Error during conversion")
            
            # Start the thread
            thread = threading.Thread(target=conversion_thread)
            thread.daemon = True
            thread.start()
    
    def crop_and_resize_single(input_dir, output_dir, scale_factor, status_var, root):
        """Handle crop and resize operation for a single video"""
        if input_dir == "No directory selected" or not os.path.exists(input_dir):
            messagebox.showerror("Error", "Please select a valid input directory")
            return
        
        if output_dir == "No directory selected" or not os.path.exists(output_dir):
            messagebox.showerror("Error", "Please select a valid output directory")
            return
        
        # Get video file from user
        video_file = filedialog.askopenfilename(
            title="Select Video to Crop and Resize",
            initialdir=input_dir,
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        
        if not video_file:
            return
        
        # Let user select ROI
        status_var.set("Please select a region of interest (ROI) in the video...")
        
        roi_selector = ROISelector(video_path=video_file)
        roi = roi_selector.select_roi()
        
        if roi is None:
            status_var.set("ROI selection canceled")
            return
        
        # Create timestamp directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_output_dir = os.path.join(output_dir, f"cropped_resized_{timestamp}")
        os.makedirs(batch_output_dir, exist_ok=True)
        
        # Setup output file path
        input_filename = os.path.basename(video_file)
        name, ext = os.path.splitext(input_filename)
        x, y, w, h = roi
        output_filename = f"{name}_crop_{x}_{y}_{w}_{h}_{scale_factor}x{ext}"
        output_path = os.path.join(batch_output_dir, output_filename)
        
        # Progress window
        progress_window = tk.Toplevel(root)
        progress_window.title("Processing Video")
        progress_window.geometry("600x300")
        
        progress_text = tk.Text(progress_window, height=15, width=70)
        progress_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Function to update progress text
        def update_progress(message):
            progress_text.insert(tk.END, message + "\n")
            progress_text.see(tk.END)
            progress_window.update()
        
        # Start processing in a thread
        def process_thread():
            try:
                update_progress(f"Processing video: {input_filename}")
                update_progress(f"ROI: x={x}, y={y}, width={w}, height={h}")
                update_progress(f"Scale: {scale_factor}x")
                
                status_var.set(f"Processing video with ROI...")
                
                # Process the video
                metadata = resize_with_opencv(
                    video_file,
                    output_path,
                    scale_factor,
                    roi,
                    update_progress
                )
                
                if metadata:
                    update_progress(f"Success! Output saved to: {output_path}")
                    metadata_file = os.path.splitext(output_path)[0] + "_metadata.json"
                    update_progress(f"Metadata saved to: {os.path.basename(metadata_file)}")
                    
                    update_progress("\nTo convert MediaPipe coordinates back to original video:")
                    update_progress("Click 'Convert MediaPipe Coordinates' button after processing")
                    
                    status_var.set("Crop and resize completed successfully")
                else:
                    update_progress("Failed to process video")
                    status_var.set("Failed to process video")
                
                # Add close button
                Button(
                    progress_window,
                    text="Close",
                    command=progress_window.destroy
                ).pack(pady=10)
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                update_progress(error_msg)
                status_var.set(error_msg)
        
        # Start thread
        thread = threading.Thread(target=process_thread)
        thread.daemon = True
        thread.start()
    
    def start_batch_processing(input_dir, output_dir, scale_factor, use_roi, status_var, root):
        """Start batch processing videos"""
        if input_dir == "No directory selected" or not os.path.exists(input_dir):
            messagebox.showerror("Error", "Please select a valid input directory")
            return
        
        if output_dir == "No directory selected" or not os.path.exists(output_dir):
            messagebox.showerror("Error", "Please select a valid output directory")
            return
        
        # Create progress window
        progress_window = tk.Toplevel(root)
        progress_window.title("Processing Videos")
        progress_window.geometry("600x400")
        
        progress_text = tk.Text(progress_window, height=20, width=70)
        progress_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Function to update progress text
        def update_progress(message):
            progress_text.insert(tk.END, message + "\n")
            progress_text.see(tk.END)
            progress_window.update()
        
        # Start processing in a new thread
        def process_thread():
            try:
                # Create timestamp directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_output_dir = os.path.join(output_dir, f"resized_videos_{timestamp}")
                os.makedirs(batch_output_dir, exist_ok=True)
                
                # Find video files
                video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")
                video_files = [
                    os.path.join(input_dir, f) for f in os.listdir(input_dir)
                    if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(video_extensions)
                ]
                
                if not video_files:
                    update_progress("No video files found in the selected directory.")
                    return
                
                update_progress(f"Found {len(video_files)} videos to process with {scale_factor}x scaling")
                status_var.set(f"Processing {len(video_files)} videos...")
                
                # Process each video
                for i, video_file in enumerate(video_files, 1):
                    try:
                        input_filename = os.path.basename(video_file)
                        name, ext = os.path.splitext(input_filename)
                        output_filename = f"{name}_{scale_factor}x{ext}"
                        output_path = os.path.join(batch_output_dir, output_filename)
                        
                        update_progress(f"\n[{i}/{len(video_files)}] Processing: {input_filename}")
                        
                        # Process with OpenCV (no ROI for batch mode)
                        metadata = resize_with_opencv(
                            video_file, 
                            output_path, 
                            scale_factor,
                            None,  # No ROI in batch mode
                            lambda msg: update_progress(f"  {msg}")
                        )
                        
                        if metadata:
                            update_progress(f"  Completed: {output_filename}")
                            metadata_file = os.path.splitext(output_path)[0] + "_metadata.json"
                            update_progress(f"  Metadata saved to: {os.path.basename(metadata_file)}")
                        else:
                            update_progress(f"  Failed to process: {input_filename}")
                        
                    except Exception as e:
                        update_progress(f"  Error processing {video_file}: {str(e)}")
                
                update_progress("\nBatch processing complete!")
                status_var.set("Processing complete!")
                
                # Add close button
                Button(
                    progress_window, 
                    text="Close", 
                    command=progress_window.destroy
                ).pack(pady=10)
                
            except Exception as e:
                update_progress(f"Error in batch processing: {str(e)}")
                status_var.set(f"Error: {str(e)}")
        
        # Start processing thread
        thread = threading.Thread(target=process_thread)
        thread.daemon = True
        thread.start()
    
    root.mainloop()

def run_resize_video():
    """Main function to run the video resizer application"""
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    
    # Start batch processing
    batch_resize_videos()

if __name__ == "__main__":
    run_resize_video()



