"""
================================================================================
Codec Benchmark Tool - codec_benchmark.py
================================================================================
vailá - Multimodal Toolbox
Authors: Prof. Dr. Paulo R. P. Santiago
https://github.com/paulopreto/vaila-multimodaltoolbox
Date: 05 March 2026
Version: 0.1.0

Description:
------------
This script benchmarks H.264, H.265, and H.266/VVC codecs on a set of videos.
It records input size, output size, compression ratio, and execution time for each codec.
Results are saved to a JSON file for analysis.

Usage:
------
python codec_benchmark.py --dir /path/to/videos --workers 2 --output results_benchmark
"""

import argparse
import os
import time
import json
import subprocess
import concurrent.futures
from datetime import datetime
from rich import print
from vaila.ffmpeg_utils import get_ffmpeg_path

FFMPEG = get_ffmpeg_path()

def run_benchmark_for_video(video_info):
    """Run H.264, H.265, and H.266 benchmarks for a single video."""
    video_path = video_info["video_path"]
    output_base_dir = video_info["output_dir"]
    basename = os.path.basename(video_path)
    original_size = os.path.getsize(video_path)
    
    video_results = {
        "video": basename,
        "input_size_bytes": original_size,
        "results": []
    }
    
    codecs = [
        {"name": "H.264", "ext": "_h264.mp4", "params": ["-c:v", "libx264", "-preset", "medium", "-crf", "23", "-bf", "0"]},
        {"name": "H.265", "ext": "_h265.mp4", "params": ["-c:v", "libx265", "-preset", "medium", "-crf", "28"]},
        {"name": "H.266", "ext": "_h266.mp4", "params": ["-c:v", "libvvenc", "-vvenc-params", "preset=medium:qp=32"]},
    ]
    
    for codec in codecs:
        output_path = os.path.join(output_base_dir, f"{os.path.splitext(basename)[0]}{codec['ext']}")
        
        start_time = time.time()
        success = False
        error = None
        output_size = 0
        
        try:
            cmd = [FFMPEG, "-y", "-i", video_path] + codec["params"] + ["-c:a", "copy", "-hide_banner", "-nostats", output_path]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            if os.path.exists(output_path):
                output_size = os.path.getsize(output_path)
                success = True
        except subprocess.CalledProcessError as e:
            error = f"FFmpeg failed: {e.stderr[:100]}"
        except Exception as e:
            error = str(e)
            
        elapsed_time = time.time() - start_time
        
        codec_res = {
            "codec": codec["name"],
            "success": success,
            "error": error,
            "output_size_bytes": output_size,
            "compression_ratio": original_size / output_size if output_size > 0 else 0,
            "time_seconds": elapsed_time
        }
        video_results["results"].append(codec_res)
        
    return video_results

def main():
    parser = argparse.ArgumentParser(description="Benchmark video codecs H.264, H.265, H.266.")
    parser.add_argument("--dir", type=str, required=True, help="Directory with source videos.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel video tasks.")
    parser.add_argument("--output", type=str, default="benchmark_results", help="Output directory.")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.dir):
        print(f"[red]Error: {args.dir} is not a directory.[/red]")
        return

    os.makedirs(args.output, exist_ok=True)
    
    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".wmv")
    video_files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.lower().endswith(video_extensions)]
    
    if not video_files:
        print("[red]No videos found in directory.[/red]")
        return

    print(f"[bold cyan]Starting Benchmark for {len(video_files)} videos using {args.workers} workers...[/bold cyan]")
    
    video_infos = [{"video_path": v, "output_dir": args.output} for v in video_files]
    
    all_results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_benchmark_for_video, info): info for info in video_infos}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            res = future.result()
            all_results.append(res)
            print(f"[{i}/{len(video_files)}] Finished benchmarking: {res['video']}")

    # Save to JSON
    json_path = os.path.join(args.output, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\n[bold green]Benchmark completed! Results saved to {json_path}[/bold green]")

if __name__ == "__main__":
    main()
