"""
usound_biomec1.py

Module to analyze ultrasound images with manual thickness measurements and
smarter before/after batch comparison support.

Updated by: Prof. Paulo R. P. Santiago
Updated: 12 May 2026
Version: 0.3.44

- Separate BEFORE and AFTER directory selection for comparison workflows
- Parent-folder batch mode for muscle/before and muscle/after structures
- Automatic ROI detection with manual fallback for ultrasound content
- Condition-aware measurement CSV and summary reports
- Smarter visual comparisons based on best cross-condition image matches
"""

import csv
import datetime
import math
import os
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog

import cv2
import numpy as np
from rich import print


@dataclass(frozen=True)
class ImageRecord:
    """Simple container for condition-aware image processing."""

    path: str
    condition: str
    source_path: str | None = None

    @property
    def file_name(self) -> str:
        return Path(self.path).name

    @property
    def source_name(self) -> str:
        return Path(self.source_path or self.path).name

    @property
    def display_name(self) -> str:
        return f"{self.condition}_{self.file_name}"


def list_images(directory):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    files = []
    for ext in exts:
        files.extend(Path(directory).glob(ext))
    return [str(f) for f in sorted(files)]


def sanitize_label(text):
    """Convert free text into a filesystem-friendly label."""
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in text.strip())
    compact = "_".join(part for part in cleaned.split("_") if part)
    return compact or "comparison"


def load_image_or_raise(image_path, flags=cv2.IMREAD_COLOR):
    """Load an image and fail early with a clear message when missing."""
    img = cv2.imread(image_path, flags)
    if img is None:
        raise ValueError(f"Unable to load image: {image_path}")
    return img


def infer_comparison_label(before_dir, after_dir):
    before_path = Path(before_dir)
    after_path = Path(after_dir)
    if before_path.parent == after_path.parent:
        return before_path.parent.name
    return f"{before_path.name}_vs_{after_path.name}"


def discover_before_after_groups(parent_dir):
    """Discover muscle groups laid out as parent/muscle/before and parent/muscle/after."""
    groups = []
    skipped = []
    parent_path = Path(parent_dir)

    for muscle_dir in sorted(parent_path.iterdir()):
        if not muscle_dir.is_dir():
            continue

        before_dir = muscle_dir / "before"
        after_dir = muscle_dir / "after"
        before_exists = before_dir.is_dir()
        after_exists = after_dir.is_dir()
        before_count = len(list_images(before_dir)) if before_exists else 0
        after_count = len(list_images(after_dir)) if after_exists else 0

        if before_count and after_count:
            groups.append(
                {
                    "muscle": muscle_dir.name,
                    "before_dir": str(before_dir),
                    "after_dir": str(after_dir),
                    "before_count": before_count,
                    "after_count": after_count,
                }
            )
            continue

        missing_parts = []
        if not before_exists:
            missing_parts.append("missing before/")
        elif before_count == 0:
            missing_parts.append("empty before/")
        if not after_exists:
            missing_parts.append("missing after/")
        elif after_count == 0:
            missing_parts.append("empty after/")
        skipped.append({"muscle": muscle_dir.name, "reason": ", ".join(missing_parts)})

    return groups, skipped


def write_batch_summary(batch_rows, skipped_rows, output_dir, timestamp):
    """Save a master CSV for batch parent-folder processing."""
    summary_path = os.path.join(output_dir, f"usound_batch_summary_{timestamp}.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "Muscle",
                "Status",
                "Reason",
                "Before_Images",
                "After_Images",
                "Output_Directory",
            ]
        )
        for row in batch_rows:
            writer.writerow(
                [
                    row["muscle"],
                    row["status"],
                    row["reason"],
                    row["before_count"],
                    row["after_count"],
                    row["output_dir"],
                ]
            )
        for row in skipped_rows:
            writer.writerow([row["muscle"], "skipped", row["reason"], 0, 0, ""])
    print(f"Batch summary saved in: {summary_path}")
    return summary_path


def build_condition_records(before_dir, after_dir):
    before_files = list_images(before_dir)
    after_files = list_images(after_dir)
    before_records = [ImageRecord(path=path, condition="before") for path in before_files]
    after_records = [ImageRecord(path=path, condition="after") for path in after_files]
    return before_records + after_records, before_records, after_records


def detect_ultrasound_roi(img, padding=20, min_area_ratio=0.01):
    """Detect the main ultrasound content area from a B-mode frame."""
    if img is None:
        return None

    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_threshold, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_value = max(int(otsu_threshold * 0.6), 20)
    _, binary = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = int(gray.shape[0] * gray.shape[1] * min_area_ratio)
    candidates = []
    for label in range(1, n_labels):
        x, y, w, h, area = stats[label]
        if area >= min_area:
            candidates.append((int(area), int(x), int(y), int(w), int(h)))

    if not candidates:
        return None

    _, x, y, w, h = max(candidates, key=lambda item: item[0])
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(gray.shape[1], x + w + padding)
    y2 = min(gray.shape[0], y + h + padding)
    return x1, y1, x2 - x1, y2 - y1


def combine_rois(rois, image_shape):
    """Combine multiple ROI boxes into one shared bounding box."""
    if not rois:
        return None

    height, width = image_shape[:2]
    x1 = min(roi[0] for roi in rois)
    y1 = min(roi[1] for roi in rois)
    x2 = max(roi[0] + roi[2] for roi in rois)
    y2 = max(roi[1] + roi[3] for roi in rois)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    return x1, y1, x2 - x1, y2 - y1


def compute_shared_roi(image_records):
    """Estimate one ROI that covers the useful ultrasound content for all images."""
    rois = []
    image_shape = None
    for record in image_records:
        img = cv2.imread(record.path)
        if img is None:
            continue
        image_shape = img.shape[:2]
        roi = detect_ultrasound_roi(img)
        if roi is not None:
            rois.append(roi)

    if not rois or image_shape is None:
        return None
    return combine_rois(rois, image_shape)


def preview_roi(image_path, roi):
    """Display the automatically detected ROI before confirmation."""
    img = cv2.imread(image_path)
    if img is None:
        return

    x, y, w, h = roi
    preview = img.copy()
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(
        preview,
        "Automatic ROI preview",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        preview,
        "Check box position, then confirm in dialog",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    window_name = "Automatic ROI Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, preview)
    cv2.waitKey(1)


def select_roi_manually(image_path):
    """Fallback manual ROI selection on a sample image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading {image_path}")
        return None

    points = []
    display = img.copy()
    window_name = "Manual ROI: click top-left and bottom-right"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def on_click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
            if len(points) == 2:
                cv2.rectangle(display, points[0], points[1], (0, 255, 0), 2)
            cv2.imshow(window_name, display)

    cv2.setMouseCallback(window_name, on_click)
    cv2.imshow(window_name, display)
    print("Manual ROI selection: click top-left and bottom-right. Press ESC to cancel.")

    while len(points) < 2:
        if cv2.waitKey(20) == 27:
            cv2.destroyWindow(window_name)
            return None

    cv2.destroyWindow(window_name)
    (x1, y1), (x2, y2) = points
    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])
    return x_min, y_min, x_max - x_min, y_max - y_min


def choose_roi_and_scale(image_records):
    """Choose an automatic ROI when possible, then calibrate its real height."""
    if not image_records:
        return None, None

    sample_path = image_records[0].path
    roi = compute_shared_roi(image_records)

    if roi is not None:
        preview_roi(sample_path, roi)
        use_auto = messagebox.askyesno(
            "Automatic ROI",
            "Automatic ROI detected for the ultrasound content.\n\n"
            "Use it for both BEFORE and AFTER groups?\n\n"
            "Yes = use automatic ROI\nNo = draw ROI manually",
        )
        cv2.destroyWindow("Automatic ROI Preview")
    else:
        messagebox.showwarning(
            "Automatic ROI",
            "Could not detect the ultrasound ROI automatically.\n"
            "You will define it manually on a sample image.",
        )
        use_auto = False

    if not use_auto:
        roi = select_roi_manually(sample_path)

    if roi is None:
        return None, None

    _, _, _, roi_height = roi
    calib_cm = simpledialog.askfloat(
        "ROI Calibration",
        f"VERTICAL height in cm for {roi_height} px:",
        minvalue=0.001,
    )
    if calib_cm is None:
        print("Calibration not provided. Aborting.")
        return None, None

    scale = calib_cm / roi_height
    return roi, scale


def crop_image_records(image_records, output_dir, timestamp, roi):
    """Crop condition-aware image records using one shared ROI."""
    x, y, w, h = roi
    crop_root = os.path.join(output_dir, f"crop_{timestamp}")
    os.makedirs(crop_root, exist_ok=True)

    cropped_records = []
    for record in image_records:
        img = cv2.imread(record.path)
        if img is None:
            print(f"Error loading {record.path}")
            continue

        condition_dir = os.path.join(crop_root, record.condition)
        os.makedirs(condition_dir, exist_ok=True)
        cropped = img[y : y + h, x : x + w]
        output_path = os.path.join(condition_dir, Path(record.path).name)
        cv2.imwrite(output_path, cropped)
        cropped_records.append(
            ImageRecord(path=output_path, condition=record.condition, source_path=record.path)
        )

    print(f"Automatic crop saved in: {crop_root}")
    return crop_root, cropped_records


def redraw_annotations(state):
    """Draw only the lines and points on base_img."""
    ann = state["base_img"].copy()
    # Calibration
    if len(state["calib_points"]) == 1 and not state["calibrated"]:
        cv2.circle(ann, state["calib_points"][0], 5, (255, 255, 0), -1)
    if state["calibrated"] and len(state["calib_points"]) == 2:
        p1, p2 = state["calib_points"]
        cv2.line(ann, p1, p2, (255, 255, 0), 2)
        dist_cm = state["scale"] * state["calib_dist_px"]
        mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        cv2.putText(
            ann,
            f"{dist_cm:.2f} cm",
            (mx + 5, my - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
    # Measurements
    for p1, p2, _, dist_cm in state["measurements"]:
        cv2.line(ann, p1, p2, (0, 255, 0), 2)
        mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        cv2.putText(
            ann,
            f"{dist_cm:.2f} cm",
            (mx + 5, my - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    # First point marker
    if state["calibrated"] and state["first_point"]:
        cv2.circle(ann, state["first_point"], 5, (0, 255, 0), -1)
    return ann


def mouse_event(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN:
        if not state["calibrated"]:
            state["calib_points"].append((x, y))
            if len(state["calib_points"]) == 2:
                p1, p2 = state["calib_points"]
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                state["calib_dist_px"] = math.hypot(dx, dy)
                real_cm = simpledialog.askfloat(
                    "Calibration",
                    f"VERTICAL height in cm for {int(state['calib_dist_px'])} px:",
                )
                if real_cm:
                    state["scale"] = real_cm / state["calib_dist_px"]
                    state["calibrated"] = True
            state["display_img"] = redraw_annotations(state)
        else:
            if state["first_point"] is None:
                state["first_point"] = (x, y)
                state["display_img"] = redraw_annotations(state)
            else:
                p1 = state["first_point"]
                p2 = (x, y)
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                dist_px = math.hypot(dx, dy)
                dist_cm = dist_px * state["scale"]
                state["measurements"].append((p1, p2, dist_px, dist_cm))
                write_measurement_row(state, p1, p2, dist_px, dist_cm)
                print(f"Saved: {p1}->{p2} = {dist_cm:.2f} cm")
                state["first_point"] = None
                state["display_img"] = redraw_annotations(state)


def write_measurement_row(state, p1, p2, dist_px, dist_cm):
    """Write a measurement row using legacy or condition-aware CSV format."""
    with open(state["output_csv"], "a", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        if state.get("condition"):
            writer.writerow(
                [
                    state["condition"],
                    state["img_name"],
                    state.get("source_name", state["img_name"]),
                    p1[0],
                    p1[1],
                    p2[0],
                    p2[1],
                    f"{dist_px:.2f}",
                    f"{dist_cm:.2f}",
                ]
            )
        else:
            writer.writerow(
                [
                    state["img_name"],
                    p1[0],
                    p1[1],
                    p2[0],
                    p2[1],
                    f"{dist_px:.2f}",
                    f"{dist_cm:.2f}",
                ]
            )


def process_images(input_dir, output_csv, scale=None):
    files = list_images(input_dir)
    if not files:
        print("No images found.")
        return

    # CSV header
    with open(output_csv, "w") as f:
        f.write("File,p1_x,p1_y,p2_x,p2_y,Dist_px,Width_cm\n")

    # Main window with instructions in title
    title = "Measure width: click left then right edge | u=undo r=reset n=next image q=quit"
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
            "output_csv": output_csv,
        }
        state["display_img"] = img.copy()
        cv2.setMouseCallback(title, mouse_event, state)

        print(f"\n=== Measuring width in {state['img_name']} ===")
        if scale is not None:
            print(f"→ Calibration: 1/scale = {1 / scale:.2f} px/cm")
        else:
            print("→ Calibration will be requested on the first image")
        print("→ Click at LEFT edge, then RIGHT edge to measure width")
        print("→ Press 'n' for next image, 'q' to finish")

        while True:
            # Show only the original with annotations
            cv2.imshow(title, state["display_img"])
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return all_measurements
            elif key == ord("n"):
                # Store measurements for this image
                all_measurements[img_name] = state["measurements"]
                break
            elif key == ord("u"):
                # Undo
                if state["first_point"]:
                    state["first_point"] = None
                elif state["measurements"]:
                    state["measurements"].pop()
                state["display_img"] = redraw_annotations(state)
            elif key == ord("r"):
                # Full reset
                state.update({"first_point": None, "measurements": []})
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
        "Crop Calibration", f"VERTICAL height in cm for {height_px} px:"
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
    params = {"threshold1": 30, "threshold2": 100, "blur": 5}  # Initial values

    win = "Edge Detection Parameters (press SPACE when satisfied)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

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
        h + 30

        # Create a text area below the images
        text_area = np.zeros((40, display.shape[1], 3), dtype=np.uint8)
        display_with_text = np.vstack((display, text_area))

        cv2.putText(
            display_with_text,
            text,
            (10, h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Add labels to each side
        cv2.putText(
            display_with_text,
            "Edge Detection",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display_with_text,
            "Original Image",
            (edges1_bgr.shape[1] + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow(win, display_with_text)

    # Trackbar callbacks are defined only after update_display exists because
    # some OpenCV builds invoke the callback immediately on trackbar creation.
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
            overlay_images(
                cropped_paths[i],
                cropped_paths[j],
                overlay_path,
                alpha=0.5,
                add_labels=True,
            )

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


def _fit_label(label: str, max_chars: int = 28) -> str:
    return label if len(label) <= max_chars else f"{label[: max_chars - 3]}..."


def side_by_side_images(img_path1, img_path2, output_path, label1="Image 1", label2="Image 2"):
    """Create a side by side comparison with clear labels."""
    img1 = load_image_or_raise(img_path1)
    img2 = load_image_or_raise(img_path2)

    # Add image labels
    cv2.putText(
        img1,
        _fit_label(label1),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        img2,
        _fit_label(label2),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    # Resize to same height
    h = min(img1.shape[0], img2.shape[0])
    img1 = cv2.resize(img1, (int(img1.shape[1] * h / img1.shape[0]), h))
    img2 = cv2.resize(img2, (int(img2.shape[1] * h / img2.shape[0]), h))

    # Add a separator line
    separator = np.zeros((h, 5, 3), dtype=np.uint8)
    separator[:, :] = [255, 255, 255]  # White line

    side = np.hstack((img1, separator, img2))
    cv2.imwrite(output_path, side)


def overlay_images(
    img_path1,
    img_path2,
    output_path,
    alpha=0.5,
    add_labels=True,
    label1="Image 1",
    label2="Image 2",
):
    """Create an overlay with clearly labeled images."""
    img1 = load_image_or_raise(img_path1)
    img2 = load_image_or_raise(img_path2)

    # Resize to same size
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = cv2.resize(img1, (w, h))
    img2 = cv2.resize(img2, (w, h))

    # Add colored box in corner to identify images
    if add_labels:
        # Image 1: Green box + label
        cv2.rectangle(img1, (10, 10), (210, 40), (0, 200, 0), -1)
        cv2.putText(
            img1,
            _fit_label(label1, max_chars=24),
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Image 2: Blue box + label
        cv2.rectangle(img2, (10, 10), (210, 40), (200, 0, 0), -1)
        cv2.putText(
            img2,
            _fit_label(label2, max_chars=24),
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    # Create blend
    overlay = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
    cv2.imwrite(output_path, overlay)


def overlay_with_edges(
    img_path1,
    img_path2,
    output_path,
    edge_params,
    label1="Image 1",
    label2="Image 2",
):
    """Create an overlay with edge enhancement on second image using specified parameters."""
    img1 = load_image_or_raise(img_path1)
    img2 = load_image_or_raise(img_path2)

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
    cv2.putText(
        img1,
        f"{_fit_label(label1, max_chars=20)} (background)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    # Overlay edges on first image
    result = img1.copy()
    mask = (img2_edges != [0, 0, 0]).any(axis=2)
    result[mask] = img2_edges[mask]

    # Add label for edge source
    cv2.putText(
        result,
        f"{_fit_label(label2, max_chars=20)} (red edges)",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )

    cv2.imwrite(output_path, result)


def edges_only_comparison(
    img_path1,
    img_path2,
    output_path,
    edge_params,
    label1="Image 1",
    label2="Image 2",
):
    """Create an image showing only edges: Image 1 in black, Image 2 in red."""
    img1 = load_image_or_raise(img_path1)
    img2 = load_image_or_raise(img_path2)

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
    cv2.putText(
        result,
        f"{_fit_label(label1, max_chars=20)} (black)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
    )
    cv2.putText(
        result,
        f"{_fit_label(label2, max_chars=20)} (red)",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )

    cv2.imwrite(output_path, result)


def process_condition_image_records(image_records, output_csv, scale):
    """Measure cropped BEFORE/AFTER images while keeping condition metadata."""
    if not image_records:
        print("No images found.")
        return {}

    with open(output_csv, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "Condition",
                "File",
                "Source_File",
                "p1_x",
                "p1_y",
                "p2_x",
                "p2_y",
                "Dist_px",
                "Width_cm",
            ]
        )

    title = "Measure thickness: click first then opposite edge | u=undo r=reset n=next q=quit"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    all_measurements = {}

    for record in image_records:
        img = cv2.imread(record.path)
        if img is None:
            print(f"Error loading {record.path}")
            continue

        state = {
            "record_key": record.display_name,
            "condition": record.condition,
            "img_name": record.file_name,
            "source_name": record.source_name,
            "base_img": img.copy(),
            "calib_points": [],
            "calibrated": True,
            "calib_dist_px": None,
            "scale": scale,
            "first_point": None,
            "measurements": [],
            "output_csv": output_csv,
        }
        state["display_img"] = img.copy()
        cv2.setMouseCallback(title, mouse_event, state)

        print(f"\n=== Measuring {record.condition.upper()} | {record.file_name} ===")
        print(f"→ Source image: {record.source_name}")
        print(f"→ Calibration: 1/scale = {1 / scale:.2f} px/cm")
        print("→ Click first edge, then opposite edge")
        print("→ Press 'n' for next image, 'q' to finish")

        while True:
            cv2.imshow(title, state["display_img"])
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                all_measurements[state["record_key"]] = {
                    "condition": record.condition,
                    "file": record.file_name,
                    "source_file": record.source_name,
                    "measurements": list(state["measurements"]),
                }
                cv2.destroyAllWindows()
                return all_measurements
            elif key == ord("n"):
                all_measurements[state["record_key"]] = {
                    "condition": record.condition,
                    "file": record.file_name,
                    "source_file": record.source_name,
                    "measurements": list(state["measurements"]),
                }
                break
            elif key == ord("u"):
                if state["first_point"]:
                    state["first_point"] = None
                elif state["measurements"]:
                    state["measurements"].pop()
                state["display_img"] = redraw_annotations(state)
            elif key == ord("r"):
                state.update({"first_point": None, "measurements": []})
                state["display_img"] = redraw_annotations(state)

    cv2.destroyAllWindows()
    print("Condition-aware measurement process complete.")
    return all_measurements


def _measurement_stats(values):
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def write_condition_summaries(all_measurements, output_dir, timestamp):
    """Write per-image, per-condition, and before-vs-after summary CSVs."""
    image_summary_path = os.path.join(output_dir, f"usound_image_summary_{timestamp}.csv")
    condition_summary_path = os.path.join(output_dir, f"usound_condition_summary_{timestamp}.csv")
    comparison_summary_path = os.path.join(
        output_dir, f"usound_before_after_summary_{timestamp}.csv"
    )

    image_rows = []
    per_condition_image_means = {}
    per_condition_all_values = {}
    per_condition_total_images = {}

    for record in all_measurements.values():
        condition = record["condition"]
        values = [measurement[3] for measurement in record["measurements"]]
        stats = _measurement_stats(values)
        per_condition_total_images[condition] = per_condition_total_images.get(condition, 0) + 1

        if stats is not None:
            per_condition_image_means.setdefault(condition, []).append(stats["mean"])
            per_condition_all_values.setdefault(condition, []).extend(values)

        image_rows.append(
            [
                condition,
                record["file"],
                record["source_file"],
                len(values),
                "" if stats is None else f"{stats['mean']:.4f}",
                "" if stats is None else f"{stats['median']:.4f}",
                "" if stats is None else f"{stats['std']:.4f}",
                "" if stats is None else f"{stats['min']:.4f}",
                "" if stats is None else f"{stats['max']:.4f}",
            ]
        )

    with open(image_summary_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "Condition",
                "File",
                "Source_File",
                "N_Measurements",
                "Mean_cm",
                "Median_cm",
                "Std_cm",
                "Min_cm",
                "Max_cm",
            ]
        )
        writer.writerows(image_rows)

    condition_rows = []
    for condition in sorted(per_condition_total_images):
        image_means = per_condition_image_means.get(condition, [])
        all_values = per_condition_all_values.get(condition, [])
        image_stats = _measurement_stats(image_means)
        value_stats = _measurement_stats(all_values)
        condition_rows.append(
            [
                condition,
                per_condition_total_images.get(condition, 0),
                len(image_means),
                len(all_values),
                "" if image_stats is None else f"{image_stats['mean']:.4f}",
                "" if image_stats is None else f"{image_stats['median']:.4f}",
                "" if image_stats is None else f"{image_stats['std']:.4f}",
                "" if value_stats is None else f"{value_stats['mean']:.4f}",
            ]
        )

    with open(condition_summary_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "Condition",
                "N_Images_Total",
                "N_Images_Measured",
                "N_Measurements_Total",
                "Mean_Image_Thickness_cm",
                "Median_Image_Thickness_cm",
                "Std_Image_Thickness_cm",
                "Mean_All_Measurements_cm",
            ]
        )
        writer.writerows(condition_rows)

    before_means = per_condition_image_means.get("before", [])
    after_means = per_condition_image_means.get("after", [])
    comparison_rows = []
    if before_means and after_means:
        before_mean = float(np.mean(before_means))
        after_mean = float(np.mean(after_means))
        delta_cm = after_mean - before_mean
        delta_pct = (delta_cm / before_mean * 100.0) if before_mean else float("nan")
        comparison_rows.append(
            [
                f"{before_mean:.4f}",
                f"{after_mean:.4f}",
                f"{delta_cm:.4f}",
                "" if math.isnan(delta_pct) else f"{delta_pct:.2f}",
            ]
        )
        print(
            "Before/after summary: "
            f"before={before_mean:.4f} cm | after={after_mean:.4f} cm | "
            f"delta={delta_cm:.4f} cm"
        )

    with open(comparison_summary_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "Before_Mean_Image_Thickness_cm",
                "After_Mean_Image_Thickness_cm",
                "Delta_cm",
                "Delta_percent",
            ]
        )
        writer.writerows(comparison_rows)

    print(f"Image summary saved in: {image_summary_path}")
    print(f"Condition summary saved in: {condition_summary_path}")
    print(f"Before/after summary saved in: {comparison_summary_path}")


def prepare_similarity_image(image_path, size=(256, 256)):
    img = load_image_or_raise(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    return cv2.equalizeHist(img)


def calculate_image_similarity(img_path1, img_path2):
    img1 = prepare_similarity_image(img_path1)
    img2 = prepare_similarity_image(img_path2)

    corr = np.corrcoef(img1.flatten(), img2.flatten())[0, 1]
    if np.isnan(corr):
        corr = 0.0
    shift, _ = cv2.phaseCorrelate(img1.astype(np.float32), img2.astype(np.float32))
    return float(corr), float(shift[0]), float(shift[1])


def build_comparison_plan(before_records, after_records):
    """Rank all BEFORE/AFTER pairs and keep the best unique matches."""
    plan = []
    for before_record in before_records:
        for after_record in after_records:
            correlation, shift_x, shift_y = calculate_image_similarity(
                before_record.path, after_record.path
            )
            plan.append(
                {
                    "before_path": before_record.path,
                    "after_path": after_record.path,
                    "before_file": before_record.file_name,
                    "after_file": after_record.file_name,
                    "correlation": correlation,
                    "shift_x": shift_x,
                    "shift_y": shift_y,
                }
            )

    plan.sort(key=lambda row: row["correlation"], reverse=True)
    selected_pairs = select_unique_best_pairs(plan)
    selected_keys = {(row["before_path"], row["after_path"]) for row in selected_pairs}
    for row in plan:
        row["selected_pair"] = (row["before_path"], row["after_path"]) in selected_keys
    return plan


def select_unique_best_pairs(plan):
    """Greedy unique matching so each image is used at most once in visual reports."""
    used_before = set()
    used_after = set()
    selected = []
    for row in plan:
        before_path = row["before_path"]
        after_path = row["after_path"]
        if before_path in used_before or after_path in used_after:
            continue
        selected.append(row)
        used_before.add(before_path)
        used_after.add(after_path)
    return selected


def save_comparison_plan(plan, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "Before_File",
                "After_File",
                "Correlation",
                "Shift_X_px",
                "Shift_Y_px",
                "Selected_For_Visual_Report",
            ]
        )
        for row in plan:
            writer.writerow(
                [
                    row["before_file"],
                    row["after_file"],
                    f"{row['correlation']:.6f}",
                    f"{row['shift_x']:.3f}",
                    f"{row['shift_y']:.3f}",
                    int(row["selected_pair"]),
                ]
            )


def create_before_after_comparison_images(before_records, after_records, output_dir):
    """Create visual comparisons only for the best unique BEFORE/AFTER matches."""
    if not before_records or not after_records:
        print("Need at least one BEFORE and one AFTER image for comparisons.")
        return

    plan = build_comparison_plan(before_records, after_records)
    comparison_plan_path = os.path.join(output_dir, "before_after_comparison_plan.csv")
    save_comparison_plan(plan, comparison_plan_path)
    selected_pairs = [row for row in plan if row["selected_pair"]]

    if not selected_pairs:
        print("No valid BEFORE/AFTER pair found for visual comparisons.")
        return

    sidebyside_dir = os.path.join(output_dir, "side_by_side")
    overlay_dir = os.path.join(output_dir, "overlay")
    edges_dir = os.path.join(output_dir, "edge_overlay")
    edges_only_dir = os.path.join(output_dir, "edges_only")

    os.makedirs(sidebyside_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(edges_dir, exist_ok=True)
    os.makedirs(edges_only_dir, exist_ok=True)

    edge_pair = selected_pairs[0]
    img1 = cv2.imread(edge_pair["before_path"])
    img2 = cv2.imread(edge_pair["after_path"])
    print("Adjust edge detection parameters using the best BEFORE/AFTER pair...")
    edge_params = adjust_edge_parameters(img1, img2)

    for row in selected_pairs:
        before_label = f"Before: {Path(row['before_path']).stem}"
        after_label = f"After: {Path(row['after_path']).stem}"
        pair_stem = f"{Path(row['before_path']).stem}_vs_{Path(row['after_path']).stem}"

        side_by_side_images(
            row["before_path"],
            row["after_path"],
            os.path.join(sidebyside_dir, f"{pair_stem}.jpg"),
            label1=before_label,
            label2=after_label,
        )
        overlay_images(
            row["before_path"],
            row["after_path"],
            os.path.join(overlay_dir, f"{pair_stem}.jpg"),
            alpha=0.5,
            add_labels=True,
            label1=before_label,
            label2=after_label,
        )
        overlay_with_edges(
            row["before_path"],
            row["after_path"],
            os.path.join(edges_dir, f"{pair_stem}.jpg"),
            edge_params,
            label1=before_label,
            label2=after_label,
        )
        edges_only_comparison(
            row["before_path"],
            row["after_path"],
            os.path.join(edges_only_dir, f"{pair_stem}.jpg"),
            edge_params,
            label1=before_label,
            label2=after_label,
        )

    print(f"Comparison plan saved in: {comparison_plan_path}")
    print(f"Smart visual comparisons created for {len(selected_pairs)} pair(s).")


def run_before_after_workflow(before_dir, after_dir, output_dir, timestamp):
    """Run the smarter before/after workflow with automatic ROI and summaries."""
    image_records, before_records, after_records = build_condition_records(before_dir, after_dir)
    if not before_records:
        print("No images found in BEFORE directory.")
        return {
            "success": False,
            "reason": "No images found in BEFORE directory.",
            "before_count": 0,
            "after_count": len(after_records),
        }
    if not after_records:
        print("No images found in AFTER directory.")
        return {
            "success": False,
            "reason": "No images found in AFTER directory.",
            "before_count": len(before_records),
            "after_count": 0,
        }

    roi, scale = choose_roi_and_scale(image_records)
    if roi is None:
        return {
            "success": False,
            "reason": "ROI selection or calibration cancelled.",
            "before_count": len(before_records),
            "after_count": len(after_records),
        }

    _, cropped_records = crop_image_records(image_records, output_dir, timestamp, roi)
    csv_path = os.path.join(output_dir, f"usound_before_after_{timestamp}.csv")
    all_measurements = process_condition_image_records(cropped_records, csv_path, scale)
    write_condition_summaries(all_measurements, output_dir, timestamp)

    before_cropped = [record for record in cropped_records if record.condition == "before"]
    after_cropped = [record for record in cropped_records if record.condition == "after"]
    create_before_after_comparison_images(before_cropped, after_cropped, output_dir)
    return {
        "success": True,
        "reason": "",
        "before_count": len(before_records),
        "after_count": len(after_records),
    }


def run_parent_batch_workflow(parent_dir, output_dir, timestamp):
    """Run one before/after workflow per muscle under a parent directory."""
    groups, skipped = discover_before_after_groups(parent_dir)
    if not groups:
        messagebox.showwarning(
            "Ultrasound batch mode",
            "No valid muscle groups were found.\n\n"
            "Expected layout:\n"
            "parent/muscle_name/before\n"
            "parent/muscle_name/after",
        )
        write_batch_summary([], skipped, output_dir, timestamp)
        return

    print(f"Discovered {len(groups)} muscle group(s) for batch processing.")
    batch_rows = []
    for index, group in enumerate(groups, start=1):
        muscle_label = sanitize_label(group["muscle"])
        muscle_output_dir = os.path.join(output_dir, muscle_label)
        os.makedirs(muscle_output_dir, exist_ok=True)

        print(
            f"\n=== Batch muscle {index}/{len(groups)}: {group['muscle']} ===\n"
            f"BEFORE images: {group['before_count']} | AFTER images: {group['after_count']}"
        )
        result = run_before_after_workflow(
            group["before_dir"], group["after_dir"], muscle_output_dir, timestamp
        )
        batch_rows.append(
            {
                "muscle": group["muscle"],
                "status": "completed" if result["success"] else "stopped",
                "reason": result["reason"],
                "before_count": result["before_count"],
                "after_count": result["after_count"],
                "output_dir": muscle_output_dir,
            }
        )
        write_batch_summary(batch_rows, skipped, output_dir, timestamp)

        if not result["success"]:
            print(f"Batch stopped at muscle '{group['muscle']}': {result['reason']}")
            return

    write_batch_summary(batch_rows, skipped, output_dir, timestamp)


def choose_usound_workflow_mode():
    """Ask whether to run batch-parent mode or direct folder mode."""
    return messagebox.askyesnocancel(
        "Ultrasound workflow",
        "Choose workflow mode:\n\n"
        "Yes = parent folder batch mode\n"
        "  Expected structure: muscle_name/before and muscle_name/after\n\n"
        "No = choose BEFORE and AFTER folders directly\n"
        "  Cancel AFTER selection later to use the legacy single-folder mode\n\n"
        "Cancel = abort",
    )


def run_usound():
    root = tk.Tk()
    root.withdraw()

    workflow_mode = choose_usound_workflow_mode()
    if workflow_mode is None:
        return

    if workflow_mode:
        parent_dir = filedialog.askdirectory(
            title="Select parent folder with muscle_name/before and muscle_name/after"
        )
        if not parent_dir:
            return

        outp = filedialog.askdirectory(title="Select output parent directory")
        if not outp:
            return

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        parent_label = sanitize_label(Path(parent_dir).name)
        out_dir = os.path.join(outp, f"vaila_usound_batch_{parent_label}_{ts}")
        os.makedirs(out_dir, exist_ok=True)
        run_parent_batch_workflow(parent_dir, out_dir, ts)
    else:
        # Get BEFORE directory (also used as legacy single-folder input if AFTER is skipped)
        before_dir = filedialog.askdirectory(
            title="Select BEFORE folder (Cancel AFTER selection to use single-folder mode)"
        )
        if not before_dir:
            return

        # AFTER is optional to preserve the legacy single-folder workflow
        after_dir = filedialog.askdirectory(
            title="Select AFTER folder (Cancel to keep legacy single-folder workflow)"
        )

        outp = filedialog.askdirectory(title="Select output parent directory")
        if not outp:
            return

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if after_dir:
            comparison_label = sanitize_label(infer_comparison_label(before_dir, after_dir))
            out_dir = os.path.join(outp, f"vaila_usound_{comparison_label}_{ts}")
        else:
            out_dir = os.path.join(outp, f"vaila_usound_{ts}")
        os.makedirs(out_dir, exist_ok=True)

        if after_dir:
            run_before_after_workflow(before_dir, after_dir, out_dir, ts)
        else:
            print("Running legacy single-folder workflow.")
            crop_dir, scale, cropped_paths = crop_images_batch(before_dir, out_dir, ts)
            if not crop_dir:
                return

            csv_path = os.path.join(out_dir, f"usound_{ts}.csv")
            process_images(crop_dir, csv_path, scale)
            create_comparison_images(cropped_paths, out_dir)

    print(f"\nAll results saved in: {out_dir}")


if __name__ == "__main__":
    run_usound()
