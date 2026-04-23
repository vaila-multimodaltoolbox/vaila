#!/usr/bin/env python3
"""
Recursive Video Cutting Script
Author: Antigravity AI
Date: 04 March 2026

Description:
This script recursively walks through a root directory, finds .txt sync files,
and processes videos in each directory using the logic from vaila/cutvideo.py.
"""

import argparse
import os
import sys
from pathlib import Path

from rich import print

# Add the root directory to sys.path to ensure vaila modules can be found
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Import needed functions from cutvideo
try:
    from vaila.cutvideo import batch_process_sync_videos, parse_sync_file_content
except ImportError as e:
    print(f"[red]Error importing vaila.cutvideo: {e}[/]")
    print(
        "Make sure you are running this from the vaila root directory or it is in your PYTHONPATH."
    )
    sys.exit(1)


def recursive_batch_cut(target_root, dry_run=False):
    target_root = Path(target_root).resolve()
    if not target_root.exists():
        print(f"[red]Error: Directory {target_root} does not exist.[/]")
        return

    print(f"[bold blue]Starting recursive batch cut in:[/] {target_root}")
    if dry_run:
        print("[yellow]DRY RUN MODE - No videos will be cut.[/]")

    sync_files_found = []

    # Walk through the directory tree
    for root, dirs, files in os.walk(target_root):
        # Look for .txt files that might be sync files
        for file in files:
            if file.lower().endswith(".txt") and file.lower() not in [
                "requirements.txt",
                "readme.txt",
                "license.txt",
            ]:
                sync_file_path = Path(root) / file
                sync_files_found.append(sync_file_path)

    if not sync_files_found:
        print("[yellow]No potential sync files (.txt) found.[/]")
        return

    print(f"Found {len(sync_files_found)} potential sync files.")

    for sync_file in sync_files_found:
        print(f"\n[bold green]Processing directory:[/] {sync_file.parent}")
        print(f"Using sync file: {sync_file.name}")

        # Try to find at least one video file in the same directory to "anchor" the batch process
        # cutvideo.batch_process_sync_videos uses Path(video_path).parent to find the directory
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"]
        local_videos = [
            f
            for f in os.listdir(sync_file.parent)
            if any(f.endswith(ext) for ext in video_extensions)
        ]

        if not local_videos:
            print(f"  [yellow]No video files found in {sync_file.parent}. Skipping.[/]")
            continue

        # Use the first video found as the anchor path
        anchor_video_path = sync_file.parent / local_videos[0]

        # Parse the sync file
        cuts, is_sync, sync_data = parse_sync_file_content(sync_file, anchor_video_path)

        if not is_sync or not sync_data:
            print(
                f"  [yellow]Sync file {sync_file.name} does not seem to follow the expected format. Skipping.[/]"
            )
            continue

        print(f"  [cyan]Found sync data for {len(sync_data)} videos.[/]")

        if dry_run:
            for vid, info in sync_data.items():
                if (sync_file.parent / vid).exists():
                    print(
                        f"    [white]- Would process: {vid} -> {info['new_name']} (Frames: {info['initial_frame']} to {info['final_frame']})[/]"
                    )
                else:
                    print(f"    [red]- Video not found: {vid}[/]")
            continue

        # Execute batch processing
        try:
            success = batch_process_sync_videos(anchor_video_path, sync_data)
            if success:
                print(f"  [green]Successfully processed directory: {sync_file.parent}[/]")
            else:
                print(f"  [red]Failed to process some or all videos in: {sync_file.parent}[/]")
        except Exception as e:
            print(f"  [red]Exception occurred while processing {sync_file.parent}: {e}[/]")


def main():
    parser = argparse.ArgumentParser(
        description="Recursively apply vaila/cutvideo.py batch processing."
    )
    parser.add_argument("directory", help="The root directory to search for videos and sync files.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Find sync files and show what would be processed without cutting.",
    )

    args = parser.parse_args()

    recursive_batch_cut(args.directory, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
