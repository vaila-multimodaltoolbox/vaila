#!/bin/bash

#########################################################################################
#                                                                                       #
# Script: install_vaila_mac.sh                                                          #
# Description: Installs the vaila - Multimodal Toolbox on macOS using uv (Astral).      #
#              Conda is no longer supported; uv is the single install method.           #
#                                                                                       #
# Usage:                                                                                #
#   1. Download/clone the repository.                                                   #
#   2. Make the script executable:                                                      #
#      chmod +x install_vaila_mac.sh                                                    #
#   3. Run the script from the root directory of the repository:                        #
#      ./install_vaila_mac.sh                                                           #
#                                                                                       #
# Notes:                                                                                #
#   - Requires Homebrew (https://brew.sh/) for system dependencies (auto-installed).    #
#   - uv will be automatically installed if not present.                                #
#   - Python 3.12.13 will be installed via `uv python install`.                         #
#                                                                                       #
# Author: Prof. Dr. Paulo R. P. Santiago                                                #
# Creation: 20 November 2025                                                            #
# Update: 29 June 2026
# Version: 0.3.67
# OS: macOS (Apple Silicon or Intel)                                                    #
#########################################################################################

set -e  # Exit on error

echo "============================================================"
echo "vaila - Multimodal Toolbox Installation/Update (uv)"
echo "============================================================"
echo ""
echo "This script will install or update vaila in: ~/vaila"
echo "If vaila is already installed, it will be updated with the latest code."
echo ""

# Define paths
USER_HOME="$HOME"

# ============================================================================
# INSTALL LOCATION
# ============================================================================

echo "---------------------------------------------"
echo "Install Location Selection"
echo "  [1] Default (~/vaila) - Recommended"
echo "  [2] Current Directory ($(pwd)) - Local/Portable"
echo "---------------------------------------------"
printf "Choose an option [1-2] (default: 1): "
read INSTALL_LOC_OPTION
INSTALL_LOC_OPTION=${INSTALL_LOC_OPTION:-1}

if [[ "$INSTALL_LOC_OPTION" == "2" ]]; then
    VAILA_HOME="$(pwd)"
    echo "Installing in current directory: $VAILA_HOME"
else
    VAILA_HOME="$USER_HOME/vaila"
    echo "Installing in default location: $VAILA_HOME"
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bootstrap: clone repo if pyproject.toml is missing locally
if [ ! -f "$PROJECT_DIR/pyproject.toml" ]; then
    echo "Bootstrap Mode: vaila source not found locally."
    echo "Cloning vaila repository from GitHub..."

    if ! command -v git &> /dev/null; then
        echo "Error: git is not installed. Please install git first."
        exit 1
    fi

    TEMP_DIR=$(mktemp -d)
    echo "Downloading to temporary directory: $TEMP_DIR"
    git clone --depth 1 https://github.com/vaila-multimodaltoolbox/vaila.git "$TEMP_DIR/vaila"

    echo "Running installer from downloaded source..."
    chmod +x "$TEMP_DIR/vaila/install_vaila_mac.sh"
    "$TEMP_DIR/vaila/install_vaila_mac.sh" "$@"
    EXIT_CODE=$?

    echo "Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
    exit $EXIT_CODE
fi

# ============================================================================
# UV INSTALLATION
# ============================================================================

echo ""
echo "============================================================"
echo "Installing vaila using uv"
echo "============================================================"
echo ""

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. It is required for system dependencies."
    echo "Attempting to install Homebrew automatically..."

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Configure Homebrew in current environment
    echo "Configuring Homebrew environment..."
    if [[ "$(uname -m)" == "arm64" ]]; then
        if [ -f "/opt/homebrew/bin/brew" ]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
    else
        if [ -f "/usr/local/bin/brew" ]; then
            eval "$(/usr/local/bin/brew shellenv)"
        fi
    fi

    if ! command -v brew &> /dev/null; then
        echo "Error: Homebrew installation failed or 'brew' is not in PATH."
        echo "Please install Homebrew manually: https://brew.sh/"
        exit 1
    fi
    echo "Homebrew installed successfully."
fi

# Check for missing system dependencies
echo "Verifying system dependencies..."
MISSING_DEPS=()
for pkg in git curl wget ffmpeg pkg-config cairo; do
    if ! brew list --formula | grep -q "^$pkg\$"; then
        MISSING_DEPS+=("$pkg")
    fi
done

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo "Installing missing system dependencies: ${MISSING_DEPS[*]}"
    brew install "${MISSING_DEPS[@]}"
fi

# Install python-tk for Python 3.12 (required by Tkinter)
echo ""
echo "Checking for python-tk@3.12 (required for tkinter/GUI)..."
if ! brew list --formula | grep -q "^python-tk@3.12\$"; then
    echo "Installing python-tk@3.12..."
    brew install python-tk@3.12
else
    echo "python-tk@3.12 is already installed."
fi

# Install uv if not present
echo ""
echo "Checking for uv installation..."
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    echo "Using official installer: curl -LsSf https://astral.sh/uv/install.sh | sh"
    curl -LsSf https://astral.sh/uv/install.sh | sh

    if [ -d "$HOME/.local/bin" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi
    if [ -d "$HOME/.cargo/bin" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    fi

    if [ -f "$HOME/.zshrc" ]; then
        source "$HOME/.zshrc" 2>/dev/null || true
    elif [ -f "$HOME/.bash_profile" ]; then
        source "$HOME/.bash_profile" 2>/dev/null || true
    fi

    if ! command -v uv &> /dev/null; then
        echo "Error: uv installation failed. Please install manually:"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    echo "uv installed successfully!"
else
    echo "uv is already installed."
    echo "Updating uv..."
    uv self update || echo "Warning: Failed to update uv. Continuing with current version."
fi

UV_VERSION=$(uv --version 2>/dev/null || echo "unknown")
echo "uv version: $UV_VERSION"

# Configure shell autocompletion
echo "Configuring shell autocompletion for uv..."
SHELL_NAME=$(basename "$SHELL")
if [ "$SHELL_NAME" = "zsh" ] || [ -f "$HOME/.zshrc" ]; then
    if ! grep -q "uv generate-shell-completion zsh" "$HOME/.zshrc" 2>/dev/null; then
        echo 'eval "$(uv generate-shell-completion zsh)"' >> "$HOME/.zshrc"
        echo "Added uv autocompletion to ~/.zshrc"
    fi
fi
if [ "$SHELL_NAME" = "bash" ] || [ -f "$HOME/.bash_profile" ]; then
    TARGET_FILE="$HOME/.bash_profile"
    [ -f "$HOME/.bashrc" ] && TARGET_FILE="$HOME/.bashrc"

    if ! grep -q "uv generate-shell-completion bash" "$TARGET_FILE" 2>/dev/null; then
        echo 'eval "$(uv generate-shell-completion bash)"' >> "$TARGET_FILE"
        echo "Added uv autocompletion to $TARGET_FILE"
    fi
fi
echo ""

# Install Python 3.12.13 via uv if needed
echo "Checking Python version..."
if ! uv python list | grep -q "3.12.13"; then
    echo "Python 3.12.13 not found. Installing via uv..."
    uv python install 3.12.13
else
    echo "Python 3.12.13 found."
fi

echo "Preparing destination directory..."

if [[ "$INSTALL_LOC_OPTION" == "2" ]]; then
    echo "Local install selected. Skipping rsync file copy."
    echo "Using current directory as VAILA_HOME."
else
    echo ""
    if [ -d "$VAILA_HOME" ]; then
        echo "Updating existing vaila installation in $VAILA_HOME..."
        echo "Removing old files (keeping .venv to be recreated)..."
        find "$VAILA_HOME" -mindepth 1 -maxdepth 1 ! -name '.venv' -exec rm -rf {} +
    else
        echo "Installing vaila to $VAILA_HOME..."
        mkdir -p "$VAILA_HOME"
    fi

    echo "Copying files..."
    rsync -av --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' --exclude='uv.lock' --exclude='.python-version' "$PROJECT_DIR/" "$VAILA_HOME/"
fi

if [ -f "$VAILA_HOME/uv.lock" ]; then
    rm -f "$VAILA_HOME/uv.lock"
fi

cd "$VAILA_HOME"

# Select appropriate pyproject.toml template based on architecture and user choice
echo ""
echo "Selecting pyproject.toml configuration..."

ARCH=$(uname -m)
USE_METAL=false

if [[ "$ARCH" == "arm64" ]]; then
    echo "Apple Silicon detected. Use Metal/MPS acceleration? [Y/n]"
    read metal_choice
    USE_METAL=$([[ "$metal_choice" != "n" && "$metal_choice" != "N" ]])
else
    echo "Intel Mac detected. Using CPU-only configuration."
fi

USE_SAM_EXTRA=false
echo ""
echo "Install optional SAM 3 (Meta) segmentation stack (extra 'sam')?"
echo "Note: on macOS this is install-only. SAM 3 inference requires NVIDIA CUDA at runtime."
echo "[y/N]"
read -r sam_choice
if [[ "$sam_choice" == "y" || "$sam_choice" == "Y" ]]; then
    USE_SAM_EXTRA=true
fi

# Choose template
if [[ "$USE_METAL" == true ]]; then
    if [ -f "$VAILA_HOME/pyproject_macos.toml" ]; then
        cp "$VAILA_HOME/pyproject_macos.toml" "$VAILA_HOME/pyproject.toml"
        echo "Using macOS Metal/MPS configuration."
    else
        echo "Warning: pyproject_macos.toml not found. Using CPU-only."
        cp "$VAILA_HOME/pyproject_universal_cpu.toml" "$VAILA_HOME/pyproject.toml"
        USE_METAL=false
    fi
else
    cp "$VAILA_HOME/pyproject_universal_cpu.toml" "$VAILA_HOME/pyproject.toml"
    echo "Using CPU-only configuration."
fi

# Initialize uv project
echo ""
echo "Initializing uv project..."
if [ ! -f ".python-version" ]; then
    uv python pin 3.12.13
fi

# Create virtual environment
echo ""
echo "Creating/updating virtual environment (.venv)..."
if [ ! -d ".venv" ]; then
    echo "Creating new virtual environment..."
    uv venv --python 3.12.13
else
    echo "Virtual environment already exists. uv sync will update it as needed."
fi

# Sync dependencies
echo ""
echo "Installing vaila dependencies with uv..."
echo "This may take a few minutes on first run..."

UV_SYNC_CMD=(uv sync)
if [[ "$USE_SAM_EXTRA" == true ]]; then
    UV_SYNC_CMD+=(--extra sam)
fi
if ! "${UV_SYNC_CMD[@]}"; then
    echo "Error: uv sync failed."
    if [[ "$USE_SAM_EXTRA" == true ]]; then
        echo "Retrying without SAM extra..."
        if uv sync; then
            echo "Base install succeeded (without --extra sam)."
            echo "You can retry SAM later on a CUDA host with: uv sync --extra sam"
        else
            echo "Restoring universal CPU configuration..."
            cp "$VAILA_HOME/pyproject_universal_cpu.toml" "$VAILA_HOME/pyproject.toml"
            echo "Installation failed. Please check the error messages above."
            exit 1
        fi
    else
        echo "Restoring universal CPU configuration..."
        cp "$VAILA_HOME/pyproject_universal_cpu.toml" "$VAILA_HOME/pyproject.toml"
        echo "Installation failed. Please check the error messages above."
        exit 1
    fi
fi
echo "Dependencies installed successfully."
echo ""
echo "PyTorch, torchvision, torchaudio, ultralytics, and boxmot are installed via uv sync from pyproject.toml."

if [[ "$USE_SAM_EXTRA" == true ]]; then
    echo ""
    echo "------------------------------------------------------------"
    echo "SAM 3: accept https://huggingface.co/facebook/sam3 then run:"
    echo "  cd \"$VAILA_HOME\" && uv run hf auth login"
    echo "------------------------------------------------------------"
    read -r -p "Run 'uv run hf auth login' now? [y/N] " hf_login_now
    if [[ "$hf_login_now" == "y" || "$hf_login_now" == "Y" ]]; then
        (cd "$VAILA_HOME" && uv run hf auth login) || {
            echo "Warning: hf auth login failed or was cancelled."
        }
    fi
fi

if [[ "$ARCH" == "arm64" ]]; then
    echo "Apple Silicon: PyTorch can use MPS (Metal) when available; otherwise it falls back to CPU."
fi

# Install pycairo
echo ""
echo "Installing pycairo..."
if ! uv pip install pycairo; then
    echo "Warning: pycairo installation failed. Trying with force-reinstall..."
    uv pip install --force-reinstall --no-cache-dir pycairo || {
        echo "Warning: pycairo installation failed. This may cause issues with the application."
    }
fi

# Verify tkinter availability
echo ""
echo "Verifying tkinter availability..."
if uv run python -c "import tkinter" 2>/dev/null; then
    echo "tkinter is available."
else
    echo ""
    echo "Warning: tkinter is not available in the Python environment."
    echo "This is required for the GUI."
    echo ""
    echo "Note: On macOS, tkinter requires python-tk from Homebrew."
    echo "If tkinter is still not available after installation, you may need to:"
    echo "  1. Ensure python-tk is installed: brew install python-tk@3.12"
    echo "  2. Or use the system Python instead of uv's Python"
fi

# Use generic run_vaila.sh from bin/
RUN_SCRIPT="$VAILA_HOME/bin/run_vaila.sh"
echo ""
echo "Setting up run script..."
if [ -f "$RUN_SCRIPT" ]; then
    chmod +x "$RUN_SCRIPT"
    echo "Using existing run script: $RUN_SCRIPT"
else
    echo "Warning: $RUN_SCRIPT not found. Creating a fallback..."
    RUN_SCRIPT="$VAILA_HOME/run_vaila.sh"
    cat <<EOF > "$RUN_SCRIPT"
#!/bin/bash
cd "$VAILA_HOME" || exit 1
# Use uv run with --no-extra gpu to exclude TensorRT on macOS
uv run --no-sync --no-extra gpu vaila.py
EOF
    chmod +x "$RUN_SCRIPT"
fi

# Create a convenient 'vaila' command in venv bin directory
if [ -d "$VAILA_HOME/.venv/bin" ]; then
    cat << 'VAILA_WRAPPER' > "$VAILA_HOME/.venv/bin/vaila"
#!/bin/bash
# Wrapper script to run vaila.py using venv Python directly
# This avoids uv run resolving project dependencies (including gpu extra on macOS)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SCRIPT_DIR" || exit 1
exec "$(dirname "${BASH_SOURCE[0]}")/python" vaila.py "$@"
VAILA_WRAPPER
    chmod +x "$VAILA_HOME/.venv/bin/vaila"

    cat << 'UV_RUN_WRAPPER' > "$VAILA_HOME/.venv/bin/uv-run-vaila"
#!/bin/bash
# Wrapper for uv run that excludes gpu extra on macOS
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SCRIPT_DIR" || exit 1
if command -v uv &> /dev/null; then
    uv run --no-sync --no-extra gpu python vaila.py "$@"
else
    echo "Error: uv not found in PATH"
    exit 1
fi
UV_RUN_WRAPPER
    chmod +x "$VAILA_HOME/.venv/bin/uv-run-vaila"

    echo "Created convenient commands:"
    echo "  - .venv/bin/vaila (recommended - uses venv Python directly)"
    echo "  - .venv/bin/uv-run-vaila (wrapper for 'uv run' that excludes gpu extra)"
fi

# ============================================================================
# APP BUNDLE
# ============================================================================

create_app_bundle() {
    APP_NAME="vaila"
    APP_DIR="$USER_HOME/Applications/$APP_NAME.app"
    mkdir -p "$USER_HOME/Applications"

    echo "Creating macOS Application Bundle at $APP_DIR..."

    mkdir -p "$APP_DIR/Contents/MacOS"
    mkdir -p "$APP_DIR/Contents/Resources"

    cat <<EOF > "$APP_DIR/Contents/Info.plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>vaila</string>
    <key>CFBundleIconFile</key>
    <string>vaila.icns</string>
    <key>CFBundleIconName</key>
    <string>vaila</string>
    <key>CFBundleIdentifier</key>
    <string>com.vaila.toolbox</string>
    <key>CFBundleName</key>
    <string>vaila</string>
    <key>CFBundleDisplayName</key>
    <string>vaila</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>0.3.51</string>
    <key>CFBundleVersion</key>
    <string>0.3.51</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.utilities</string>
    <key>Terminal</key>
    <true/>
</dict>
</plist>
EOF

    if [ -z "$RUN_SCRIPT" ]; then
        RUN_SCRIPT="$VAILA_HOME/run_vaila.sh"
    fi

    cat <<EOF > "$APP_DIR/Contents/MacOS/vaila"
#!/bin/bash
# Open Terminal by bundle ID (works in any system language: English, Portuguese, etc.)
open -b com.apple.Terminal "$RUN_SCRIPT"
EOF

    chmod +x "$APP_DIR/Contents/MacOS/vaila"

    # Handle icon
    ICON_CREATED=false
    ICON_SRC=""

    for icon_path in "$PROJECT_DIR/docs/images/vaila.icns" "$VAILA_HOME/docs/images/vaila.icns" "$PROJECT_DIR/vaila/images/vaila.icns" "$VAILA_HOME/vaila/images/vaila.icns"; do
        if [ -f "$icon_path" ]; then
            ICON_SRC="$icon_path"
            echo "Found icon at: $ICON_SRC"
            break
        fi
    done

    if [ -z "$ICON_SRC" ]; then
        ICONSET_DIR=""
        for iconset_path in "$VAILA_HOME/docs/images/vaila.iconset" "$PROJECT_DIR/docs/images/vaila.iconset" "$VAILA_HOME/vaila/images/vaila.iconset" "$PROJECT_DIR/vaila/images/vaila.iconset"; do
            if [ -d "$iconset_path" ]; then
                ICONSET_DIR="$iconset_path"
                break
            fi
        done

        if [ -n "$ICONSET_DIR" ]; then
            echo "Found iconset directory at $ICONSET_DIR"
            echo "Converting iconset to .icns format using iconutil..."

            TEMP_ICNS="/tmp/vaila.icns"
            if command -v iconutil &> /dev/null; then
                if iconutil -c icns "$ICONSET_DIR" -o "$TEMP_ICNS" 2>/dev/null; then
                    ICON_SRC="$TEMP_ICNS"
                    ICON_CREATED=true
                    echo "Icon converted successfully from iconset using iconutil."
                fi
            fi
        fi
    fi

    if [ -z "$ICON_SRC" ]; then
        PNG_SRC=""
        for png_path in "$VAILA_HOME/docs/images/vaila_logo.png" "$PROJECT_DIR/docs/images/vaila_logo.png" \
                     "$VAILA_HOME/docs/images/vaila_ico.png" "$PROJECT_DIR/docs/images/vaila_ico.png" \
                     "$VAILA_HOME/vaila/images/vaila_logo.png" "$PROJECT_DIR/vaila/images/vaila_logo.png" \
                     "$VAILA_HOME/vaila/images/vaila_ico_mac.png" "$PROJECT_DIR/vaila/images/vaila_ico_mac.png"; do
            if [ -f "$png_path" ]; then
                PNG_SRC="$png_path"
                break
            fi
        done

        if [ -n "$PNG_SRC" ]; then
            echo "Found PNG image at $PNG_SRC"
            echo "Creating .icns from PNG using Pillow..."

            TEMP_ICNS="/tmp/vaila.icns"

            cat > /tmp/create_icns.py << 'PYTHON_SCRIPT'
import sys
from PIL import Image

def create_icns_from_png(png_path, icns_path):
    try:
        img = Image.open(png_path)
        if img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            rgb_img.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
            img = rgb_img
        img.save(icns_path, format='ICNS')
        return True
    except Exception as e:
        print(f"Error creating ICNS: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)
    png_path = sys.argv[1]
    icns_path = sys.argv[2]
    success = create_icns_from_png(png_path, icns_path)
    sys.exit(0 if success else 1)
PYTHON_SCRIPT

            # Use venv Python directly (avoids uv run resolution issues on macOS)
            if [ -f ".venv/bin/python" ]; then
                if .venv/bin/python /tmp/create_icns.py "$PNG_SRC" "$TEMP_ICNS" 2>/dev/null; then
                    if [ -f "$TEMP_ICNS" ]; then
                        ICON_SRC="$TEMP_ICNS"
                        ICON_CREATED=true
                        echo "Icon created successfully from PNG using Pillow."
                    fi
                fi
            elif uv run python /tmp/create_icns.py "$PNG_SRC" "$TEMP_ICNS" 2>/dev/null; then
                if [ -f "$TEMP_ICNS" ]; then
                    ICON_SRC="$TEMP_ICNS"
                    ICON_CREATED=true
                    echo "Icon created successfully from PNG using Pillow."
                fi
            fi

            rm -f /tmp/create_icns.py
        fi
    fi

    if [ -n "$ICON_SRC" ]; then
        echo "Copying icon to application bundle..."
        cp "$ICON_SRC" "$APP_DIR/Contents/Resources/vaila.icns"
        echo "Icon copied successfully."

        if [ "$ICON_CREATED" = true ] && [ "$ICON_SRC" = "/tmp/vaila.icns" ]; then
            rm -f "$TEMP_ICNS"
        fi

        touch "$APP_DIR"
        /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP_DIR" 2>/dev/null || true
        killall Finder 2>/dev/null || true
    else
        echo "Warning: Icon file (vaila.icns or vaila.iconset) not found. Application will use default icon."
    fi

    echo "Application Bundle created at $APP_DIR."

    # Apply icon using Python script (AppKit)
    if [ -f "$APP_DIR/Contents/Resources/vaila.icns" ]; then
        echo ""
        echo "Setting application icon..."
        cd "$VAILA_HOME"
        if [ -f "$PROJECT_DIR/set_mac_icon.py" ]; then
            if [ -f ".venv/bin/python" ]; then
                if .venv/bin/python "$PROJECT_DIR/set_mac_icon.py" "$APP_DIR" "$APP_DIR/Contents/Resources/vaila.icns" 2>&1 | grep -v -i "tensorrt" || true; then
                    echo "Icon applied successfully to App Bundle."
                fi
                if .venv/bin/python "$PROJECT_DIR/set_mac_icon.py" "$VAILA_HOME" "$APP_DIR/Contents/Resources/vaila.icns" 2>&1 | grep -v -i "tensorrt" || true; then
                    echo "Icon applied successfully to installation directory."
                fi
            else
                if uv run python "$PROJECT_DIR/set_mac_icon.py" "$APP_DIR" "$APP_DIR/Contents/Resources/vaila.icns" 2>&1 | grep -v -i "tensorrt" || true; then
                    echo "Icon applied successfully to App Bundle."
                fi
                if uv run python "$PROJECT_DIR/set_mac_icon.py" "$VAILA_HOME" "$APP_DIR/Contents/Resources/vaila.icns" 2>&1 | grep -v -i "tensorrt" || true; then
                    echo "Icon applied successfully to installation directory."
                fi
            fi
        fi

        touch "$APP_DIR"
        touch "$VAILA_HOME"
    fi

    # Create symbolic link in /Applications
    echo ""
    echo "Creating symbolic link in /Applications..."
    if [ -e "/Applications/vaila.app" ]; then
        echo "Removing existing symlink in /Applications..."
        sudo rm -rf "/Applications/vaila.app"
    fi

    echo "Creating symlink from /Applications/vaila.app to $APP_DIR..."
    sudo ln -s "$APP_DIR" "/Applications/vaila.app"

    if [ -L "/Applications/vaila.app" ]; then
        sudo chown -h "${USER}:admin" "/Applications/vaila.app"
        echo "Symlink created successfully in /Applications."

        /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "/Applications/vaila.app" 2>/dev/null || true
        touch "/Applications/vaila.app" 2>/dev/null || true
    else
        echo "Warning: Failed to create symlink in /Applications."
    fi

    # Rebuild Launch Services database
    echo ""
    echo "Rebuilding Launch Services cache..."
    /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -kill -r -domain local -domain system -domain user 2>/dev/null || true

    echo "Restarting Finder and Dock..."
    killall Finder 2>/dev/null || true
    killall Dock 2>/dev/null || true

    echo ""
    echo "Icon cache refreshed. The vaila logo should appear correctly now."
}

create_app_bundle

# Ensure correct ownership and permissions
echo "Ensuring correct ownership and permissions..."
chown -R "$USER" "$VAILA_HOME"
chmod -R u+rwX "$VAILA_HOME"

echo ""
echo "============================================================"
echo "vaila installation completed successfully!"
echo "============================================================"
echo ""
echo "Ways to run vaila:"
echo "1. Recommended: $RUN_SCRIPT"
echo "2. Or: cd \"$VAILA_HOME\" && .venv/bin/python vaila.py"
echo "3. Or: cd \"$VAILA_HOME\" && uv run vaila.py"
echo "4. App bundle: Launchpad, /Applications/vaila.app, or ~/Applications/vaila.app"
echo ""
