"""
common_utils.py
Version: 2024-07-15 20:00:00
"""

import os
import pandas as pd


def determine_header_lines(file_path):
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            first_element = line.split(",")[0].strip()
            if first_element.replace(".", "", 1).isdigit():
                return i
    return 0


def headersidx(file_path):
    try:
        header_lines = determine_header_lines(file_path)
        df = pd.read_csv(file_path, header=list(range(header_lines)))

        print("Headers with indices:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i}: {col}")

        print("\nExample of new order:")
        new_order = ["Time"]
        for i in range(1, len(df.columns), 3):
            new_order.append(df.columns[i][0])
        print(new_order)

        return new_order

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def reshapedata(file_path, new_order, save_directory):
    try:
        header_lines = determine_header_lines(file_path)
        df = pd.read_csv(file_path, skiprows=header_lines, header=header_lines)

        new_order_indices = [0]
        for header in new_order[1:]:
            base_idx = [i for i, col in enumerate(df.columns) if col == header][0]
            new_order_indices.extend([base_idx, base_idx + 1, base_idx + 2])

        df_reordered = df.iloc[:, new_order_indices]
        new_file_path = os.path.join(save_directory, os.path.basename(file_path))
        df_reordered.to_csv(new_file_path, index=False)

        print(f"Reordered data saved to {new_file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
