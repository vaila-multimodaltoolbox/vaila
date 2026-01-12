# join2dataset

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila\join2dataset.py`
- **Lines:** 141
- **Size:** 5605 characters
- **Version:** 1.0
- **Author:** Prof. Paulo Santiago
- **GUI Interface:** ❌ No

## 📖 Description


================================================================================
join2dataset.py
================================================================================
Author: Prof. Paulo Santiago
Date: 05 June 2024
Version: 1.0

Description:
------------
This script consolidates multiple CSV files located in subdirectories of a given
base directory into a single unified dataset. It scans for files ending with
"_result.csv" in each subdirectory, extracts relevant data, and adds a "Subject"
column derived from the filename. The resulting combined dataset is saved as a
CSV file in a newly created directory, timestamped with the current date and time.

Key Functionalities:
---------------------
1. Scans all subdirectories in a specified base directory for CSV files that
   end with "_result.csv".
2. Adds a "Subject" column based on the filename, to identify data by subject.
3. Reorders columns to place "Subject" between "TimeStamp" and "Trial".
4. Concatenates all found dataset...

## 🔧 Main Functions

**Total functions found:** 1

- `join_datasets`




---

📅 **Generated automatically on:** 15/10/2025 08:04:44
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
