#!/usr/bin/env python3
"""
===============================================================================
example_batch_usage.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Version: 03 September 2025
Version updated: 0.1.0
Python Version: 3.12.11

Description:
Example script demonstrating how to use the batch processing functionality
in readcsv_export.py for converting multiple CSV files to C3D format.

This script shows how to:
1. Import and use the batch processing function directly
2. Set up parameters for batch conversion
3. Handle the conversion process programmatically

Usage:
python example_batch_usage.py
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path to import readcsv_export
sys.path.append(str(Path(__file__).parent))

from readcsv_export import batch_convert_csv_to_c3d, auto_create_c3d_from_csv
import pandas as pd
import numpy as np

def create_sample_csv_files(output_dir, num_files=3):
    """
    Create sample CSV files for testing batch processing.
    
    Args:
        output_dir (str): Directory to create sample files in
        num_files (int): Number of sample files to create
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_files):
        # Create sample data with 3 markers (9 columns: frame + 3 markers × 3 coordinates)
        frames = 100
        data = {
            'frame': range(frames),
            'P1_X': np.random.randn(frames) * 0.1,
            'P1_Y': np.random.randn(frames) * 0.1,
            'P1_Z': np.random.randn(frames) * 0.1,
            'P2_X': np.random.randn(frames) * 0.1 + 1.0,
            'P2_Y': np.random.randn(frames) * 0.1 + 1.0,
            'P2_Z': np.random.randn(frames) * 0.1 + 1.0,
            'P3_X': np.random.randn(frames) * 0.1 + 2.0,
            'P3_Y': np.random.randn(frames) * 0.1 + 2.0,
            'P3_Z': np.random.randn(frames) * 0.1 + 2.0,
        }
        
        df = pd.DataFrame(data)
        filename = f"sample_data_{i+1:02d}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Created sample file: {filepath}")
    
    print(f"Created {num_files} sample CSV files in {output_dir}")

def demonstrate_batch_processing():
    """
    Demonstrate the batch processing functionality.
    """
    print("=== vailá CSV to C3D Batch Processing Demo ===\n")
    
    # Create sample data directory
    sample_dir = "sample_csv_files"
    create_sample_csv_files(sample_dir)
    
    print(f"\nSample files created in: {sample_dir}")
    print("You can now run the batch processing script to convert these files.")
    print("\nTo use batch processing:")
    print("1. Run: python readcsv_export.py")
    print("2. Choose 'Yes' for batch processing")
    print("3. Select input directory: {sample_dir}")
    print("4. Select output directory for C3D files")
    print("5. Enter parameters (point rate, analog data, etc.)")
    print("6. Wait for processing to complete")
    
    print(f"\nOr run directly: python -c \"from readcsv_export import batch_convert_csv_to_c3d; batch_convert_csv_to_c3d()\"")

if __name__ == "__main__":
    demonstrate_batch_processing()
