"""
===============================================================================
Project: vailá Multimodal Toolbox
Script: vaila_stroboscopic.py
===============================================================================
Author: Paulo R. P. Santiago & Antigravity (Google Deepmind)
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 5 February 2026
Last Update: 5 February 2026
Version: 0.1.0

Description:
------------
Generates a stroboscopic (chronophotography) image from video and pose data.
It creates a single image with multiple skeleton instances overlaid, using a 
temporal color gradient (Blue -> Red) to visualize motion direction.

Key Features:
- Uses MediaPipe pixel coordinates from CSV.
- "Enhanced Skeleton" visualization (midpoints, spine, hands).
- Temporal color gradient.
- High-quality Anti-Aliased drawing.
- Support for both CLI and GUI execution.

Usage:
------
python vaila_stroboscopic.py -v video.mp4 [-c data.csv] [-i 10]
or run without arguments to use GUI file picker.
===============================================================================
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import os
import sys
from pathlib import Path

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================

# Landmark names (Standard MediaPipe Pose model)
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

def generate_stroboscopic_image(video_path, csv_path=None, output_path=None, strobe_interval=10, decay=False):
    """
    Generates the stroboscopic image.
    
    Args:
        video_path (str): Path to input video.
        csv_path (str, optional): Path to pixel coordinates CSV. Auto-detected if None.
        output_path (str, optional): Path to save PNG. Auto-generated if None.
        strobe_interval (int): Frame interval for drawing.
        decay (bool): If True, older skeletons are more transparent (not implemented yet for simple drawing).
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return

    # 1. Auto-detect CSV if missing
    if csv_path is None:
        # Check standard vailá naming patterns
        candidates = [
            video_path.parent / f"{video_path.stem}_pixel_vaila.csv",
            video_path.parent / f"{video_path.stem}_mp_pixel.csv",
            video_path.parent / f"{video_path.stem}_mp_norm.csv", # Note: norm needs scaling, we handle it
        ]
        
        found = False
        for p in candidates:
            if p.exists():
                csv_path = p
                print(f"Auto-detected CSV: {csv_path}")
                found = True
                break
        
        if not found:
            # Fallback: check any CSV starting with video stem
            for f in video_path.parent.glob(f"{video_path.stem}*.csv"):
                 # exclude known other types if needed, but for now take first match
                 csv_path = f
                 print(f"Auto-detected CSV (fuzzy match): {csv_path}")
                 found = True
                 break

            print("Error: Could not find corresponding CSV file.")
            return False # Signal failure to caller
            
    # 2. Load Data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Check/Fix Column format (mp_norm vs pixel)
    # Ideally we expect pixel coordinates (pX_x, pX_y) or (name_x, name_y)
    # We will normalize reading into a dictionary structure later.
    
    # 3. Setup Canvas (Background)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error opening video.")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Check simple normalization heuristic (values <= 1.2)
    # We check a sample column
    is_normalized = False
    
    # Try to find a coordinate column
    sample_col = None
    cols = [str(c) for c in df.columns]
    
    pixel_vaila_format = 'p1_x' in cols # check for p1_x
    mp_format = 'nose_x' in cols or (len(cols) > 33 and 'x' in cols[1]) # basic check

    if pixel_vaila_format:
        sample_val = df['p11_x'].max() # shoulder approx
    elif mp_format:
        # assume columns 1,2 are nose x,y
        # If headers are pose names
        if 'left_shoulder_x' in cols:
            sample_val = df['left_shoulder_x'].max()
        else:
             # numeric columns potentially
             sample_val = df.iloc[:, 1].max() if len(df.columns) > 1 else 0
    else:
        sample_val = 1000 # Assume pixel if we can't tell, or crash later safely

    if sample_val <= 1.5:
        print("Detected Normalized Coordinates. Will scale to video dimensions.")
        is_normalized = True
    else:
        print("Detected Pixel Coordinates.")
        is_normalized = False


    # Read first frame for background
    ret, background = cap.read()
    
    # We keep cap open for random access in the loop
    if not ret:
        # Create black background if video fails (unlikely if opened)
        background = np.zeros((height, width, 3), dtype=np.uint8)
        print("Warning: Could not read first frame using black background.")

    # Create Composite Canvas
    # To make the skeleton pop, we can darken the background slightly
    canvas = cv2.addWeighted(background, 0.4, np.zeros(background.shape, background.dtype), 0, 0)
    
    # 4. Processing Loop
    
    # Determine frame range from CSV
    # Usually CSV has a 'frame' or 'frame_index' column
    frame_col = None
    for c in ['frame', 'frame_index', 'Frame', 'index']:
        if c in df.columns:
            frame_col = c
            break
            
    total_frames_data = len(df)
    max_frame_idx = df[frame_col].max() if frame_col else total_frames_data
    
    print(f"Generating Stroboscopic Image...")
    print(f"Video Resolution: {width}x{height}")
    print(f"Data Frames: {total_frames_data}")
    print(f"Strobe Interval: {strobe_interval}")

    # Helper to get point from row
    def get_point(row_data, name_idx, col_names, is_norm):
        # Determine x, y columns
        # Case 1: p1_x, p1_y (vaila format, 1-based index)
        if pixel_vaila_format:
            xc = f"p{name_idx+1}_x"
            yc = f"p{name_idx+1}_y"
        # Case 2: nose_x, nose_y (mp format)
        elif 'nose_x' in col_names:
            name = LANDMARK_NAMES[name_idx]
            xc = f"{name}_x"
            yc = f"{name}_y"
        # Case 3: Raw numeric columns (frame, x0, y0, z0, x1...)
        else:
            # Assume col 0 is frame
            # idx 0 -> cols 1, 2
            # idx n -> cols 1 + n*3 (if z exists) or 1 + n*2
            # Heuristic: verify column count
            has_z = (len(col_names) - 1) >= 99 # 33 * 3
            step = 3 if has_z else 2
            start = 1
            if frame_col is None: start = 0 # no frame col
            
            xi = start + name_idx * step
            yi = xi + 1
            xc = col_names[xi]
            yc = col_names[yi]

        try:
            x = float(row_data[xc])
            y = float(row_data[yc])
            
            if np.isnan(x) or np.isnan(y):
                return np.array([np.nan, np.nan])
                
            if is_norm:
                x *= width
                y *= height
                
            return np.array([x, y])
        except KeyError:
            return np.array([np.nan, np.nan])

    # Drawing helpers
    def dline(p1, p2, color, thick=2):
        if np.isnan(p1).any() or np.isnan(p2).any(): return
        pt1 = (int(p1[0]), int(p1[1]))
        pt2 = (int(p2[0]), int(p2[1]))
        cv2.line(canvas, pt1, pt2, color, thick, cv2.LINE_AA)

    def dcircle(p, color, radius=4):
        if np.isnan(p).any(): return
        pt = (int(p[0]), int(p[1]))
        cv2.circle(canvas, pt, radius, (255,255,255), -1, cv2.LINE_AA)
        cv2.circle(canvas, pt, radius-2, color, -1, cv2.LINE_AA)

    def compute_mid(p1, p2):
        if np.isnan(p1).any() or np.isnan(p2).any(): return np.array([np.nan, np.nan])
        return (p1 + p2) / 2

    # Helper for BBox
    def get_bbox(points_dict, img_w, img_h, padding=30):
        xs = [p[0] for p in points_dict.values() if not np.isnan(p[0])]
        ys = [p[1] for p in points_dict.values() if not np.isnan(p[1])]
        
        if not xs or not ys:
            return None
            
        x1 = max(0, int(min(xs) - padding))
        y1 = max(0, int(min(ys) - padding))
        x2 = min(img_w, int(max(xs) + padding))
        y2 = min(img_h, int(max(ys) + padding))
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        return (x1, y1, x2, y2)

    # Iterate
    # We iterate based on row index for simplicity, assuming sorted by frame
    for i in range(0, total_frames_data, strobe_interval):
        row = df.iloc[i]
        
        # Calculate Progress (0.0 to 1.0)
        progress = i / total_frames_data
        
        # Color Gradient: Blue -> Red
        # Blue: (255, 0, 0) -> Red: (0, 0, 255) in BGR
        # Interpolate
        b = int(255 * (1 - progress))
        r = int(255 * progress)
        g = 0 # Keep it clean
        
        # Let's use the nice Sky Blue to Coral transition from the user prompt history advice
        # Start (Blueish): (255, 191, 0) -> End (Reddish): (80, 80, 255) ?? No those are static side colors.
        # User suggested: "Blue -> Red" for time.
        # Let's stick to standard Blue to Red for time.
        color_time = (b, g, r)

        # Extract Points
        pts = {}
        for idx, name in enumerate(LANDMARK_NAMES):
            pts[name] = get_point(row, idx, df.columns, is_normalized)

        # Compute Midpoints (Enhanced Skeleton)
        pts['mid_shoulder'] = compute_mid(pts['left_shoulder'], pts['right_shoulder'])
        pts['mid_hip'] = compute_mid(pts['left_hip'], pts['right_hip'])
        pts['mid_ear'] = compute_mid(pts['left_ear'], pts['right_ear'])
        pts['left_mid_hand'] = compute_mid(pts['left_pinky'], pts['left_index'])
        pts['right_mid_hand'] = compute_mid(pts['right_pinky'], pts['right_index'])

        # --- BBOX CROP & OVERLAY ---
        # 1. Get BBox
        bbox = get_bbox(pts, width, height, padding=30)
        
        # 2. Read Frame
        # Ideally we seek to the specific frame index
        # Note: 'i' in the loop is the CSV index. We need the corresponding VIDEO frame index.
        # If the CSV has a 'frame' column, use that.
        current_frame_idx = i
        if frame_col:
            current_frame_idx = int(row[frame_col])
        
        # Only seek/read if we have a valid bbox to draw
        if bbox:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret_f, frame_img = cap.read()
            
            if ret_f:
                x1, y1, x2, y2 = bbox
                
                # Extract ROI from video frame
                roi = frame_img[y1:y2, x1:x2]
                
                # Overlay ROI onto canvas
                # We can do a simple copy, or a weighted addition if we want some transparency
                # User asked for "crop of bbox... to create new png image that has the other bbox drawing of pose"
                # This implies seeing the person. Simple copy is best for visibility.
                
                # Optional: Feather edges? Too complex for now.
                canvas[y1:y2, x1:x2] = roi

        # --- DRAW SKELETON ---

        # Draw Segments
        # We use the time-color for all segments to show "ghost" effect
        # Or we could preserve Left/Right colors but fade them?
        # Standard stroboscopic usually uses solid color per frame to show time.
        # Let's use the Time Color.
        
        c_skel = color_time
        thick = 2
        
        # Arms
        dline(pts['right_shoulder'], pts['right_elbow'], c_skel, thick)
        dline(pts['right_elbow'], pts['right_wrist'], c_skel, thick)
        dline(pts['right_wrist'], pts['right_mid_hand'], c_skel, thick)
        
        dline(pts['left_shoulder'], pts['left_elbow'], c_skel, thick)
        dline(pts['left_elbow'], pts['left_wrist'], c_skel, thick)
        dline(pts['left_wrist'], pts['left_mid_hand'], c_skel, thick)
        
        # Legs
        dline(pts['right_hip'], pts['right_knee'], c_skel, thick)
        dline(pts['right_knee'], pts['right_ankle'], c_skel, thick)
        dline(pts['right_ankle'], pts['right_heel'], c_skel, thick)
        dline(pts['right_heel'], pts['right_foot_index'], c_skel, thick) # Foot
        
        dline(pts['left_hip'], pts['left_knee'], c_skel, thick)
        dline(pts['left_knee'], pts['left_ankle'], c_skel, thick)
        dline(pts['left_ankle'], pts['left_heel'], c_skel, thick)
        dline(pts['left_heel'], pts['left_foot_index'], c_skel, thick)

        # Center/Spine
        dline(pts['mid_shoulder'], pts['mid_hip'], c_skel, thick)
        dline(pts['mid_ear'], pts['mid_shoulder'], c_skel, thick)
        
        # Girdles
        dline(pts['right_shoulder'], pts['left_shoulder'], c_skel, thick)
        dline(pts['right_hip'], pts['left_hip'], c_skel, thick)

        # Joints (White or Green? Let's use White for clean high contrast)
        C_JOINT = (255, 255, 255)
        for name, pt in pts.items():
            if 'mid' in name: continue
            if name == 'nose': continue
            if 'eye' in name: continue
            # dcircle(pt, C_JOINT, 3) 
            # Ideally joints might clutter a strobe image. Let's draw small dots.
            if not np.isnan(pt).any():
                cv2.circle(canvas, (int(pt[0]), int(pt[1])), 3, c_skel, -1, cv2.LINE_AA)

    cap.release() # Close video file

    # 5. Save Output
    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_stroboscopic.png"
    
    cv2.imwrite(str(output_path), canvas)
    print(f"\nSuccess! Stroboscopic image saved to: {output_path}")

def main():
    print("Generating Stroboscopic Effect Image from Video + CSV")
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent.resolve()}")
    parser = argparse.ArgumentParser(description="Generate Stroboscopic Effect Image from Video + CSV")
    parser.add_argument("-v", "--video", required=False, help="Path to input video")
    parser.add_argument("-c", "--csv", help="Path to pixel coordinates CSV (optional, auto-detected)")
    parser.add_argument("-o", "--output", help="Path to output PNG (optional)")
    parser.add_argument("-i", "--interval", type=int, default=10, help="Strobe interval (frames)")
    
    args = parser.parse_args()
    
    video_path = args.video
    
    # GUI Fallback if no video provided
    if video_path is None:
        try:
            import tkinter as tk
            from tkinter import filedialog, simpledialog
            
            root = tk.Tk()
            root.withdraw() # Hide main window
            
            video_path = filedialog.askopenfilename(
                title="Select Video for Stroboscopic Analysis",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
            )
            
            if not video_path:
                print("No file selected.")
                return

            # Ask for interval optional
            user_interval = simpledialog.askinteger("Stroboscopic Interval", "Enter frame interval (default 10):", minvalue=1, initialvalue=10)
            if user_interval:
                args.interval = user_interval
                
            root.destroy()
        except ImportError:
            print("Error: Tkinter not installed and no video argument provided.")
            return
    
    success = generate_stroboscopic_image(
        video_path,
        csv_path=args.csv,
        output_path=args.output,
        strobe_interval=args.interval
    )

    # If failed (likely due to missing CSV) and we are in GUI mode (video_path was None originally, 
    # but we can't easily track that here without passing a flag or checking args.video is None).
    # Actually, we can check if success is False and args.video was None.
    
    if success is False and args.video is None:
        print("Auto-detection of CSV failed. Asking user...")
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            # Root might be destroyed, need new one or ensure we didn't destroy if we planned to reuse.
            # But we destroyed it. Let's create a temp one just for this dialog.
            root = tk.Tk()
            root.withdraw()

            csv_path = filedialog.askopenfilename(
                title="Select Pixel Coordinates CSV",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            root.destroy()
            
            if csv_path:
                generate_stroboscopic_image(
                    video_path,
                    csv_path=csv_path,
                    output_path=args.output,
                    strobe_interval=args.interval
                )
            else:
                print("No CSV selected. Aborting.")

        except ImportError:
            pass

if __name__ == "__main__":
    main()