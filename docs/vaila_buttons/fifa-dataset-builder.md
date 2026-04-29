# FIFA Dataset Builder Button

The **FIFA Dataset Builder** button launches the `fifa_dataset_builder.py` module for aggregating and standardizing soccer pitch datasets.

## Overview

This tool is used to build a unified 32-keypoint soccer pitch dataset from multiple sources (e.g., SoccerNet, human-labeled frames). It handles coordinate mapping, deduplication, and prepares the dataset for YOLO-pose training.

## Key Features

- **Unified Schema:** Maps various keypoint schemas to the canonical FIFA 32-point model.
- **Dataset Splitting:** Automatically creates train, validation, and test splits.
- **Deduplication:** Removes redundant samples to improve training efficiency.
- **Ultralytics Ready:** Generates `data.yaml` and formatted label files for direct use with YOLO.

## Usage

1. Click **FIFA Dataset Builder** in Frame B.
2. Configure the source directories and output root.
3. Run the builder to generate the `unified/` dataset tree.

---
See also: [FIFA Workflow](../../docs/fifa_workflow.md), [FIFA Dataset Help](../../vaila/help/fifa_dataset_builder.html)
