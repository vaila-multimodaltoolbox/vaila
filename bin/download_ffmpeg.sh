#!/usr/bin/env bash
# ============================================================================
# download_ffmpeg.sh
#
# Downloads a static FFmpeg build into bin/ffmpeg/ within the vaila project.
# This provides NVENC, libx265, libvvenc, and other encoders that the
# system-packaged FFmpeg typically lacks.
#
# Usage:
#   bash bin/download_ffmpeg.sh          # Download latest release
#   bash bin/download_ffmpeg.sh --force  # Re-download even if already present
#
# Source: https://johnvansickle.com/ffmpeg/
# ============================================================================

set -euo pipefail

# Resolve the project root (this script lives in bin/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FFMPEG_DIR="$PROJECT_ROOT/bin/ffmpeg"

# Architecture detection
ARCH=$(uname -m)
case "$ARCH" in
    x86_64)
        ARCH_NAME="amd64"
        ;;
    aarch64|arm64)
        ARCH_NAME="arm64"
        ;;
    *)
        echo "Error: Unsupported architecture: $ARCH"
        echo "Only x86_64 (amd64) and aarch64 (arm64) are supported."
        exit 1
        ;;
esac

# OS detection
OS=$(uname -s)
if [[ "$OS" != "Linux" ]]; then
    echo "Error: This script only supports Linux."
    echo "For macOS, install FFmpeg via: brew install ffmpeg"
    echo "For Windows, download from: https://www.gyan.dev/ffmpeg/builds/"
    exit 1
fi

# URL for static build
DOWNLOAD_URL="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-${ARCH_NAME}-static.tar.xz"

# Check if already installed
FORCE=false
if [[ "${1:-}" == "--force" ]]; then
    FORCE=true
fi

if [[ -f "$FFMPEG_DIR/ffmpeg" && "$FORCE" != true ]]; then
    echo "FFmpeg is already installed at: $FFMPEG_DIR/ffmpeg"
    "$FFMPEG_DIR/ffmpeg" -version 2>/dev/null | head -1 || true
    echo ""
    echo "To re-download, run: bash bin/download_ffmpeg.sh --force"
    exit 0
fi

# Create temp directory for download
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "============================================================"
echo "Downloading static FFmpeg for Linux ($ARCH_NAME)..."
echo "Source: $DOWNLOAD_URL"
echo "============================================================"
echo ""

# Download
echo "Downloading... (this may take a minute, ~80MB)"
if command -v wget &> /dev/null; then
    wget -q --show-progress -O "$TMPDIR/ffmpeg.tar.xz" "$DOWNLOAD_URL"
elif command -v curl &> /dev/null; then
    curl -L --progress-bar -o "$TMPDIR/ffmpeg.tar.xz" "$DOWNLOAD_URL"
else
    echo "Error: Neither wget nor curl is available."
    exit 1
fi

echo ""
echo "Extracting..."
tar xf "$TMPDIR/ffmpeg.tar.xz" -C "$TMPDIR"

# Find the extracted directory (name varies with version)
EXTRACTED_DIR=$(find "$TMPDIR" -maxdepth 1 -type d -name "ffmpeg-*-static" | head -1)
if [[ -z "$EXTRACTED_DIR" ]]; then
    # Try alternative naming
    EXTRACTED_DIR=$(find "$TMPDIR" -maxdepth 1 -type d -name "ffmpeg-*" | head -1)
fi

if [[ -z "$EXTRACTED_DIR" ]]; then
    echo "Error: Could not find extracted FFmpeg directory."
    ls -la "$TMPDIR"
    exit 1
fi

# Create target directory and copy binaries
mkdir -p "$FFMPEG_DIR"
cp "$EXTRACTED_DIR/ffmpeg" "$FFMPEG_DIR/ffmpeg"
cp "$EXTRACTED_DIR/ffprobe" "$FFMPEG_DIR/ffprobe"
chmod +x "$FFMPEG_DIR/ffmpeg" "$FFMPEG_DIR/ffprobe"

echo ""
echo "============================================================"
echo "FFmpeg installed successfully!"
echo "============================================================"
echo ""
echo "Location: $FFMPEG_DIR/"
echo ""

# Show version
echo "Version:"
"$FFMPEG_DIR/ffmpeg" -version 2>/dev/null | head -1
echo ""

# Check for key encoders
echo "Encoder support:"
ENCODERS_OUTPUT=$("$FFMPEG_DIR/ffmpeg" -encoders 2>/dev/null)
for encoder in libx264 libx265 h264_nvenc hevc_nvenc libvvenc; do
    if echo "$ENCODERS_OUTPUT" | grep -qw "$encoder"; then
        echo "  [OK] $encoder"
    else
        echo "  [FAIL] $encoder (not available)"
    fi
done

echo ""
echo "The vaila compression scripts will automatically use this FFmpeg."
echo "To update, run: bash bin/download_ffmpeg.sh --force"

# Create symlinks in .venv/bin/ so the terminal also uses this FFmpeg when venv is active
VENV_BIN="$PROJECT_ROOT/.venv/bin"
if [ -d "$VENV_BIN" ]; then
    ln -sf "$FFMPEG_DIR/ffmpeg" "$VENV_BIN/ffmpeg"
    ln -sf "$FFMPEG_DIR/ffprobe" "$VENV_BIN/ffprobe"
    echo ""
    echo "Symlinks created in .venv/bin/ — 'ffmpeg' in terminal will use"
    echo "the static version when the virtual environment is active."
fi
