#!/bin/bash

# vailá - Multimodal Toolbox
# © Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
# https://github.com/paulopreto/vaila-multimodaltoolbox
# Please see AUTHORS for contributors.
#
# Licensed under GNU Lesser General Public License v3.0
#
# compress_videos_h265.sh
# This script compresses a video file to H.265/HEVC format using FFmpeg.
# It is designed to be called from a Python script that handles
# video compression on Linux/macOS systems.
#
# Usage:
# ./compress_videos_h265.sh input_file output_file
#
# Requirements:
# - FFmpeg must be installed and accessible in the system PATH.
#
# Note:
# The compression process might take several hours depending on the size
# of the video and the performance of your computer.
#

# Check if the required number of arguments is passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input_file output_file"
    exit 1
fi

# Assign input and output file paths
INPUT_FILE=$1
OUTPUT_FILE=$2

# Compress the video using FFmpeg with H.265 codec
ffmpeg -y -i "$INPUT_FILE" -c:v libx265 -preset medium -crf 23 "$OUTPUT_FILE"

# Check if ffmpeg succeeded
if [ $? -ne 0 ]; then
    echo "Error: Failed to compress video"
    exit 1
fi

echo "Compression completed successfully: $OUTPUT_FILE"
exit 0
