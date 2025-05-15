"""
usound_biomec1.py

Module to analyze ultrasound data from images.

Updated by: Prof. Paulo R. P. Santiago
Updated: 15 May 2025
Version: 0.0.7

- Crop all images using first image as template
- Original image display for measurements
- Enhanced edge detection with adjustable parameters
- Additional edge-only comparison visualization
"""

import cv2
import pandas as pd
import numpy as np
import math
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog
import os
import datetime
from rich import print

def list_images(directory):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    files = []
    for ext in exts:
        files.extend(Path(directory).glob(ext))
    return [str(f) for f in sorted(files)]

def redraw_annotations(state):
    """Draw only the lines and points on base_img."""
    ann = state["base_img"].copy()
    # Calibration
    if len(state["calib_points"]) == 1 and not state["calibrated"]:
        cv2.circle(ann, state["calib_points"][0], 5, (255,255,0), -1)
    if state["calibrated"] and len(state["calib_points"]) == 2:
        p1, p2 = state["calib_points"]
        cv2.line(ann, p1, p2, (255,255,0), 2)
        dist_cm = state["scale"] * state["calib_dist_px"]
        mx, my = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2
        cv2.putText(ann, f"{dist_cm:.2f} cm", (mx+5,my-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    # Measurements
    for p1, p2, _, dist_cm in state["measurements"]:
        cv2.line(ann, p1, p2, (0,255,0), 2)
        mx, my = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2
        cv2.putText(ann, f"{dist_cm:.2f} cm", (mx+5,my-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    # First point marker
    if state["calibrated"] and state["first_point"]:
        cv2.circle(ann, state["first_point"], 5, (0,255,0), -1)
    return ann

def mouse_event(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN:
        if not state["calibrated"]:
            state["calib_points"].append((x,y))
            if len(state["calib_points"]) == 2:
                p1, p2 = state["calib_points"]
                dx, dy = p2[0]-p1[0], p2[1]-p1[1]
                state["calib_dist_px"] = math.hypot(dx,dy)
                real_cm = simpledialog.askfloat(
                    "Calibration",
                    f"VERTICAL height in cm for {int(state['calib_dist_px'])} px:"
                )
                if real_cm:
                    state["scale"] = real_cm / state["calib_dist_px"]
                    state["calibrated"] = True
            state["display_img"] = redraw_annotations(state)
        else:
            if state["first_point"] is None:
                state["first_point"] = (x,y)
                state["display_img"] = redraw_annotations(state)
            else:
                p1 = state["first_point"]; p2 = (x,y)
                dx, dy = p2[0]-p1[0], p2[1]-p1[1]
                dist_px = math.hypot(dx,dy)
                dist_cm = dist_px * state["scale"]
                state["measurements"].append((p1,p2,dist_px,dist_cm))
                # Save to CSV
                with open(state["output_csv"], "a") as f:
                    f.write(f"{state['img_name']},{p1[0]},{p1[1]},{p2[0]},{p2[1]},"
                            f"{dist_px:.2f},{dist_cm:.2f}\n")
                print(f"Saved: {p1}->{p2} = {dist_cm:.2f} cm")
                state["first_point"] = None
                state["display_img"] = redraw_annotations(state)

def process_images(input_dir, output_csv, scale=None):
    files = list_images(input_dir)
    if not files:
        print("No images found.")
        return

    # CSV header
    with open(output_csv, "w") as f:
        f.write("File,p1_x,p1_y,p2_x,p2_y,Dist_px,Width_cm\n")

    # Main window with instructions in title
    title = (
        "Measure width: click left then right edge | "
        "u=undo r=reset n=next image q=quit"
    )
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    # Dictionary to store all measurements by image
    all_measurements = {}

    for path in files:
        img = cv2.imread(path)
        if img is None:
            print(f"Error loading {path}")
            continue

        img_name = Path(path).name
        all_measurements[img_name] = []
        
        state = {
            "img_name": img_name,
            "base_img": img.copy(),
            "calib_points": [], 
            "calibrated": scale is not None,
            "calib_dist_px": None, 
            "scale": scale,
            "first_point": None, 
            "measurements": [],
            "output_csv": output_csv
        }
        state["display_img"] = img.copy()
        cv2.setMouseCallback(title, mouse_event, state)

        print(f"\n=== Measuring width in {state['img_name']} ===")
        print(f"→ Calibration: 1/scale = {1/scale:.2f} px/cm")
        print("→ Click at LEFT edge, then RIGHT edge to measure width")
        print("→ Press 'n' for next image, 'q' to finish")

        while True:
            # Show only the original with annotations
            cv2.imshow(title, state["display_img"])
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), 27):
                cv2.destroyAllWindows()
                return all_measurements
            elif key == ord('n'):
                # Store measurements for this image
                all_measurements[img_name] = state["measurements"]
                break
            elif key == ord('u'):
                # Undo
                if state["first_point"]:
                    state["first_point"] = None
                elif state["measurements"]:
                    state["measurements"].pop()
                state["display_img"] = redraw_annotations(state)
            elif key == ord('r'):
                # Full reset
                state.update({
                    "first_point": None, 
                    "measurements": []
                })
                state["display_img"] = redraw_annotations(state)

    cv2.destroyAllWindows()
    print("Measurement process complete.")
    return all_measurements

def crop_images_batch(input_dir, output_dir, timestamp):
    files = list_images(input_dir)
    if not files:
        print("No images found for cropping.")
        return None, None, None

    # Use first image to define crop
    first_img_path = files[0]
    first_img = cv2.imread(first_img_path)
    if first_img is None:
        print(f"Error loading first image {first_img_path}")
        return None, None, None

    # Define crop region using first image
    pts = []
    disp = first_img.copy()
    win = "Define crop region on first image (will be applied to all)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    
    def on_crop(e, x, y, f, s):
        if e == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))
            cv2.circle(disp, (x, y), 5, (0, 255, 0), -1)
            if len(pts) == 2:
                cv2.rectangle(disp, pts[0], pts[1], (0, 255, 0), 2)
            cv2.imshow(win, disp)
    
    cv2.setMouseCallback(win, on_crop)
    print(f"Using {Path(first_img_path).name} to define crop region")
    print("Click top-left and bottom-right to define crop region.")
    cv2.imshow(win, disp)
    
    while len(pts) < 2:
        if cv2.waitKey(20) == 27:
            cv2.destroyWindow(win)
            return None, None, None
    
    cv2.destroyWindow(win)

    # Extract crop coordinates
    (x1, y1), (x2, y2) = pts
    y_min, y_max = sorted([y1, y2])
    x_min, x_max = sorted([x1, x2])
    
    # Get calibration for vertical height
    height_px = y_max - y_min
    calib_cm = simpledialog.askfloat(
        "Crop Calibration",
        f"VERTICAL height in cm for {height_px} px:"
    )
    if calib_cm is None:
        print("Calibration not provided. Aborting.")
        return None, None, None

    scale = calib_cm / height_px

    # Create output directory with timestamp
    crop_dir = os.path.join(output_dir, f"crop_{timestamp}")
    os.makedirs(crop_dir, exist_ok=True)
    
    # Apply same crop to all images
    cropped_paths = []
    for path in files:
        img = cv2.imread(path)
        if img is None:
            print(f"Error loading {path}")
            continue
            
        crop = img[y_min:y_max, x_min:x_max]
        out_path = os.path.join(crop_dir, Path(path).name)
        cv2.imwrite(out_path, crop)
        cropped_paths.append(out_path)
    
    print(f"All images cropped using same region, saved in: {crop_dir}")
    return crop_dir, scale, cropped_paths

def adjust_edge_parameters(img1, img2):
    """Interactive UI to adjust edge detection parameters with single image focus."""
    params = {
        "threshold1": 30,  # Initial values
        "threshold2": 100,
        "blur": 5
    }
    
    win = "Edge Detection Parameters (press SPACE when satisfied)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    
    # Create trackbars
    def on_threshold1(val):
        params["threshold1"] = val
        update_display()
    
    def on_threshold2(val):
        params["threshold2"] = val
        update_display()
    
    def on_blur(val):
        params["blur"] = val * 2 + 1  # Ensure odd (3, 5, 7, etc.)
        update_display()
    
    cv2.createTrackbar("Lower Threshold", win, params["threshold1"], 255, on_threshold1)
    cv2.createTrackbar("Upper Threshold", win, params["threshold2"], 255, on_threshold2)
    cv2.createTrackbar("Blur Size", win, params["blur"] // 2, 10, on_blur)
    
    # Use only the first image for edge detection tuning
    # Resize for display
    h, w = img1.shape[:2]
    max_height = 600  # Maximum display height
    
    if h > max_height:
        scale_factor = max_height / h
        w = int(w * scale_factor)
        h = max_height
        img1_display = cv2.resize(img1, (w, h))
    else:
        img1_display = img1.copy()
    
    def update_display():
        # Process with current parameters
        gray1 = cv2.cvtColor(img1_display, cv2.COLOR_BGR2GRAY)
        
        if params["blur"] > 1:
            gray1 = cv2.GaussianBlur(gray1, (params["blur"], params["blur"]), 0)
        
        edges1 = cv2.Canny(gray1, params["threshold1"], params["threshold2"])
        edges1_bgr = cv2.cvtColor(edges1, cv2.COLOR_GRAY2BGR)
        
        # Create a composite image for display (edge result on left, original on right)
        separator = np.zeros((h, 5, 3), dtype=np.uint8)
        separator[:, :] = [255, 255, 255]  # White separator
        
        # Stack side by side: edges on left, original on right
        display = np.hstack((edges1_bgr, separator, img1_display))
        
        # Add parameter text at bottom
        text = f"Lower: {params['threshold1']} Upper: {params['threshold2']} Blur: {params['blur']}"
        text_y = h + 30
        
        # Create a text area below the images
        text_area = np.zeros((40, display.shape[1], 3), dtype=np.uint8)
        display_with_text = np.vstack((display, text_area))
        
        cv2.putText(display_with_text, text, (10, h + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add labels to each side
        cv2.putText(display_with_text, "Edge Detection", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_with_text, "Original Image", (edges1_bgr.shape[1] + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow(win, display_with_text)
    
    update_display()
    
    # Wait for space key
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == 32:  # SPACE
            break
    
    cv2.destroyWindow(win)
    return params

def create_comparison_images(cropped_paths, output_dir):
    """Create side-by-side and overlay comparison images for all cropped files"""
    if len(cropped_paths) < 2:
        print("Need at least 2 images for comparisons.")
        return
    
    # Directories for each type of comparison
    sidebyside_dir = os.path.join(output_dir, "side_by_side")
    overlay_dir = os.path.join(output_dir, "overlay")
    edges_dir = os.path.join(output_dir, "edge_overlay")
    edges_only_dir = os.path.join(output_dir, "edges_only")
    
    os.makedirs(sidebyside_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(edges_dir, exist_ok=True)
    os.makedirs(edges_only_dir, exist_ok=True)
    
    # Get first two images to adjust edge parameters
    img1 = cv2.imread(cropped_paths[0])
    img2 = cv2.imread(cropped_paths[1])
    
    # Adjust edge parameters interactively
    print("Adjust edge detection parameters using the sliders...")
    edge_params = adjust_edge_parameters(img1, img2)
    
    print("Creating comparison images with selected parameters...")
    for i in range(len(cropped_paths) - 1):
        for j in range(i + 1, len(cropped_paths)):
            # File names for reference
            name_i = Path(cropped_paths[i]).stem
            name_j = Path(cropped_paths[j]).stem
            
            # Side by side
            side_path = os.path.join(sidebyside_dir, f"{name_i}_vs_{name_j}.jpg")
            side_by_side_images(cropped_paths[i], cropped_paths[j], side_path)
            
            # Overlay with clear markers
            overlay_path = os.path.join(overlay_dir, f"{name_i}_over_{name_j}.jpg")
            overlay_images(cropped_paths[i], cropped_paths[j], overlay_path, 
                           alpha=0.5, add_labels=True)
            
            # Edge-enhanced overlay
            edge_path = os.path.join(edges_dir, f"{name_i}_edge_{name_j}.jpg")
            overlay_with_edges(cropped_paths[i], cropped_paths[j], edge_path, edge_params)
            
            # Edges only (black and red)
            edges_only_path = os.path.join(edges_only_dir, f"{name_i}_edges_{name_j}.jpg")
            edges_only_comparison(cropped_paths[i], cropped_paths[j], edges_only_path, edge_params)
    
    print(f"Side by side comparisons saved in: {sidebyside_dir}")
    print(f"Overlay comparisons saved in: {overlay_dir}")
    print(f"Edge-enhanced overlays saved in: {edges_dir}")
    print(f"Edge-only comparisons saved in: {edges_only_dir}")

def side_by_side_images(img_path1, img_path2, output_path):
    """Create a side by side comparison with clear labels."""
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    
    # Add image labels
    cv2.putText(img1, "Image 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 255), 2)
    cv2.putText(img2, "Image 2", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 255), 2)
    
    # Resize to same height
    h = min(img1.shape[0], img2.shape[0])
    img1 = cv2.resize(img1, (int(img1.shape[1] * h / img1.shape[0]), h))
    img2 = cv2.resize(img2, (int(img2.shape[1] * h / img2.shape[0]), h))
    
    # Add a separator line
    separator = np.zeros((h, 5, 3), dtype=np.uint8)
    separator[:, :] = [255, 255, 255]  # White line
    
    side = np.hstack((img1, separator, img2))
    cv2.imwrite(output_path, side)

def overlay_images(img_path1, img_path2, output_path, alpha=0.5, add_labels=True):
    """Create an overlay with clearly labeled images."""
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    
    # Resize to same size
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = cv2.resize(img1, (w, h))
    img2 = cv2.resize(img2, (w, h))
    
    # Add colored box in corner to identify images
    if add_labels:
        # Image 1: Green box + label
        cv2.rectangle(img1, (10, 10), (120, 40), (0, 200, 0), -1)
        cv2.putText(img1, "Image 1", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
        
        # Image 2: Blue box + label
        cv2.rectangle(img2, (10, 10), (120, 40), (200, 0, 0), -1)  
        cv2.putText(img2, "Image 2", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
    
    # Create blend
    overlay = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
    cv2.imwrite(output_path, overlay)

def overlay_with_edges(img_path1, img_path2, output_path, edge_params):
    """Create an overlay with edge enhancement on second image using specified parameters."""
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    
    # Resize to same size
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = cv2.resize(img1, (w, h))
    img2 = cv2.resize(img2, (w, h))
    
    # Extract edges from second image with adjusted parameters
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    if edge_params["blur"] > 1:
        img2_gray = cv2.GaussianBlur(img2_gray, (edge_params["blur"], edge_params["blur"]), 0)
    img2_edges = cv2.Canny(img2_gray, edge_params["threshold1"], edge_params["threshold2"])
    img2_edges = cv2.cvtColor(img2_edges, cv2.COLOR_GRAY2BGR)
    
    # Red edges
    img2_edges[np.where((img2_edges == [255, 255, 255]).all(axis=2))] = [0, 0, 255]
    
    # Background (first image)
    cv2.putText(img1, "Image 1 (background)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Overlay edges on first image
    result = img1.copy()
    mask = (img2_edges != [0, 0, 0]).any(axis=2)
    result[mask] = img2_edges[mask]
    
    # Add label for edge source
    cv2.putText(result, "Image 2 (red edges)", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imwrite(output_path, result)

def edges_only_comparison(img_path1, img_path2, output_path, edge_params):
    """Create an image showing only edges: Image 1 in black, Image 2 in red."""
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    
    # Resize to same size
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = cv2.resize(img1, (w, h))
    img2 = cv2.resize(img2, (w, h))
    
    # Create blank canvas
    result = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background
    
    # Extract edges with adjusted parameters
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    if edge_params["blur"] > 1:
        gray1 = cv2.GaussianBlur(gray1, (edge_params["blur"], edge_params["blur"]), 0)
        gray2 = cv2.GaussianBlur(gray2, (edge_params["blur"], edge_params["blur"]), 0)
    
    edges1 = cv2.Canny(gray1, edge_params["threshold1"], edge_params["threshold2"])
    edges2 = cv2.Canny(gray2, edge_params["threshold1"], edge_params["threshold2"])
    
    # Draw Image 1 edges in black
    result[edges1 == 255] = [0, 0, 0]  # Black
    
    # Draw Image 2 edges in red (on top)
    result[edges2 == 255] = [0, 0, 255]  # Red
    
    # Add labels
    cv2.putText(result, "Image 1 (black edges)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(result, "Image 2 (red edges)", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imwrite(output_path, result)

def run_usound():
    root = tk.Tk(); root.withdraw()
    
    # Get input directory with images
    inp = filedialog.askdirectory(title="Select folder with images")
    if not inp: 
        return
    
    # Get output parent directory
    outp = filedialog.askdirectory(title="Select output parent directory")
    if not outp: 
        return

    # Create timestamp for unique folder names
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(outp, f"vaila_usound_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: Crop all images using first image as template
    crop_dir, scale, cropped_paths = crop_images_batch(inp, out_dir, ts)
    if not crop_dir:
        return

    # Step 2: Measure widths in each cropped image
    csv_path = os.path.join(out_dir, f"usound_{ts}.csv")
    measurements = process_images(crop_dir, csv_path, scale)
    
    # Step 3: Create comparison images after all measurements
    create_comparison_images(cropped_paths, out_dir)

    print(f"\nAll results saved in: {out_dir}")

if __name__ == "__main__":
    run_usound()
