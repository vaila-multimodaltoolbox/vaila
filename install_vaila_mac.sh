#!/bin/bash

#########################################################################################
#                                                                                       #
# Script: install_vaila_mac.sh                                                          #
# Description: Installs the vaila - Multimodal Toolbox on macOS                       #
#              Supports both uv (recommended) and Conda (legacy) installation methods #
#                                                                                       #
# Usage:                                                                                #
#   1. Download the repository from GitHub manually and extract it.                     #
#   2. Make the script executable:                                                      #
#      chmod +x install_vaila_mac.sh                                                   #
#   3. Run the script from the root directory of the extracted repository:              #
#      ./install_vaila_mac.sh                                                          #
#                                                                                       #
# Notes:                                                                                #
#   - uv method: Requires Homebrew (https://brew.sh/) for system dependencies          #
#   - conda method: Requires Conda (Anaconda or Miniconda) to be installed             #
#   - uv will be automatically installed if not present (uv method only)              #
#   - Python 3.12.12 will be installed via uv or conda depending on method chosen     #
#                                                                                       #
# Author: Prof. Dr. Paulo R. P. Santiago                                                #
# Creation: 20 November 2025                                                          #
# Update: 27 January 2026                                                              #
# Version: 0.3.16                                                                        #
# OS: macOS (Apple Silicon or Intel)                                                    #
#########################################################################################

set -e  # Exit on error

echo "============================================================"
echo "vaila - Multimodal Toolbox Installation/Update"
echo "============================================================"
echo ""
echo "This script will install or update vaila in: ~/vaila"
echo "If vaila is already installed, it will be updated with the latest code."
echo ""

# Prompt user to choose installation method
echo "---------------------------------------------"
echo "Installation Method Selection"
echo "  [1] uv (recommended - modern, fast)"
echo "  [2] Conda (legacy - for compatibility)"
echo "---------------------------------------------"
printf "Choose an option [1-2] (default: 1): "
read INSTALL_METHOD
INSTALL_METHOD=${INSTALL_METHOD:-1}

if [[ "$INSTALL_METHOD" != "1" && "$INSTALL_METHOD" != "2" ]]; then
    echo "Invalid option. Defaulting to uv (option 1)."
    INSTALL_METHOD=1
fi

# Define paths (common to both methods)
USER_HOME="$HOME"
VAILA_HOME="$USER_HOME/vaila"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if pyproject.toml exists
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
# COMMON FUNCTIONS (used by both methods)
# ============================================================================

create_app_bundle() {
    # Create a simple macOS Application Bundle
    APP_NAME="vaila"
    APP_DIR="$USER_HOME/Applications/$APP_NAME.app"
    mkdir -p "$USER_HOME/Applications"

    echo "Creating macOS Application Bundle at $APP_DIR..."

    # Create directory structure
    mkdir -p "$APP_DIR/Contents/MacOS"
    mkdir -p "$APP_DIR/Contents/Resources"

    # Create Info.plist
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
    <string>0.2.0</string>
    <key>CFBundleVersion</key>
    <string>0.2.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.utilities</string>
    <key>Terminal</key>
    <true/>
</dict>
</plist>
EOF

    # Create the launcher script inside the App
    # RUN_SCRIPT should be defined by the calling function (install_with_uv or install_with_conda)
    if [ -z "$RUN_SCRIPT" ]; then
        RUN_SCRIPT="$VAILA_HOME/run_vaila.sh"
    fi
    
    cat <<EOF > "$APP_DIR/Contents/MacOS/vaila"
#!/bin/bash
# Open a terminal window to run the script
open -a Terminal "$RUN_SCRIPT"
EOF

    chmod +x "$APP_DIR/Contents/MacOS/vaila"

    # Handle icon (same logic for both methods)
    ICON_CREATED=false
    ICON_SRC=""
    
    # Try to find existing .icns file
    for icon_path in "$PROJECT_DIR/docs/images/vaila.icns" "$VAILA_HOME/docs/images/vaila.icns" "$PROJECT_DIR/vaila/images/vaila.icns" "$VAILA_HOME/vaila/images/vaila.icns"; do
        if [ -f "$icon_path" ]; then
            ICON_SRC="$icon_path"
            echo "Found icon at: $ICON_SRC"
            break
        fi
    done

    # If .icns not found, try to create it from .iconset directory
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

    # If still no .icns, try to create it from PNG using Pillow
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
            
            # Use Python with Pillow to convert PNG to ICNS
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

            # Use the appropriate Python command based on installation method
            if [[ "$INSTALL_METHOD" == "1" ]]; then
                # uv method
                if uv run python /tmp/create_icns.py "$PNG_SRC" "$TEMP_ICNS" 2>/dev/null; then
                    if [ -f "$TEMP_ICNS" ]; then
                        ICON_SRC="$TEMP_ICNS"
                        ICON_CREATED=true
                        echo "Icon created successfully from PNG using Pillow."
                    fi
                fi
            else
                # conda method
                if conda run -n vaila python /tmp/create_icns.py "$PNG_SRC" "$TEMP_ICNS" 2>/dev/null; then
                    if [ -f "$TEMP_ICNS" ]; then
                        ICON_SRC="$TEMP_ICNS"
                        ICON_CREATED=true
                        echo "Icon created successfully from PNG using Pillow."
                    fi
                fi
            fi
            
            rm -f /tmp/create_icns.py
        fi
    fi

    # Copy icon to app bundle if found or created
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
        
        if [[ "$INSTALL_METHOD" == "1" ]]; then
            # uv method
            # pyobjc-framework-Cocoa should already be installed via uv sync (done before restoring pyproject.toml)
            # Skip installation check here to avoid TensorRT resolution errors
            # The package is optional (only needed for icon setting), so we continue even if not installed
            # No need to check or install - already handled during uv sync
            cd "$VAILA_HOME"
            if [ -f "$PROJECT_DIR/set_mac_icon.py" ]; then
                # Use venv Python directly instead of uv run to avoid project resolution
                # This prevents TensorRT resolution errors
                if [ -f ".venv/bin/python" ]; then
                    if .venv/bin/python "$PROJECT_DIR/set_mac_icon.py" "$APP_DIR" "$APP_DIR/Contents/Resources/vaila.icns" 2>&1 | grep -v -i "tensorrt" || true; then
                        echo "Icon applied successfully to App Bundle."
                    fi
                    if .venv/bin/python "$PROJECT_DIR/set_mac_icon.py" "$VAILA_HOME" "$APP_DIR/Contents/Resources/vaila.icns" 2>&1 | grep -v -i "tensorrt" || true; then
                        echo "Icon applied successfully to installation directory."
                    fi
                else
                    # Fallback to uv run if venv python not available
                    if uv run python "$PROJECT_DIR/set_mac_icon.py" "$APP_DIR" "$APP_DIR/Contents/Resources/vaila.icns" 2>&1 | grep -v -i "tensorrt" || true; then
                        echo "Icon applied successfully to App Bundle."
                    fi
                    if uv run python "$PROJECT_DIR/set_mac_icon.py" "$VAILA_HOME" "$APP_DIR/Contents/Resources/vaila.icns" 2>&1 | grep -v -i "tensorrt" || true; then
                        echo "Icon applied successfully to installation directory."
                    fi
                fi
            fi
        else
            # conda method
            if ! conda list -n vaila | grep -q "pyobjc-framework-Cocoa"; then
                echo "Installing pyobjc-framework-Cocoa..."
                conda install -n vaila -c conda-forge pyobjc-framework-Cocoa -y || true
            fi
            
            cd "$VAILA_HOME"
            if [ -f "$PROJECT_DIR/set_mac_icon.py" ]; then
                if conda run -n vaila python "$PROJECT_DIR/set_mac_icon.py" "$APP_DIR" "$APP_DIR/Contents/Resources/vaila.icns"; then
                    echo "Icon applied successfully to App Bundle."
                fi
                if conda run -n vaila python "$PROJECT_DIR/set_mac_icon.py" "$VAILA_HOME" "$APP_DIR/Contents/Resources/vaila.icns"; then
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

# ============================================================================
# UV INSTALLATION METHOD
# ============================================================================

install_with_uv() {
    echo ""
    echo "============================================================"
    echo "Installing vaila using uv (recommended method)"
    echo "============================================================"
    echo ""

    # Check for Homebrew
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
        
        # Verify installation
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

    # Install python-tk for Python 3.12
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

    # Install Python 3.12.12 via uv if needed
    echo "Checking Python version..."
    if ! uv python list | grep -q "3.12.12"; then
        echo "Python 3.12.12 not found. Installing via uv..."
        uv python install 3.12.12
    else
        echo "Python 3.12.12 found."
    fi

    # Clean destination directory and copy files
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

    if [ -f "$VAILA_HOME/uv.lock" ]; then
        rm -f "$VAILA_HOME/uv.lock"
    fi

    cd "$VAILA_HOME"

    # Select appropriate pyproject.toml template based on architecture and user choice
    echo ""
    echo "Selecting pyproject.toml configuration..."
    
    # Detect architecture
    ARCH=$(uname -m)
    USE_METAL=false
    
    if [[ "$ARCH" == "arm64" ]]; then
        echo "Apple Silicon detected. Use Metal/MPS acceleration? [Y/n]"
        read metal_choice
        USE_METAL=$([[ "$metal_choice" != "n" && "$metal_choice" != "N" ]])
    else
        echo "Intel Mac detected. Using CPU-only configuration."
    fi
    
    # Backup current pyproject.toml
    if [ -f "$VAILA_HOME/pyproject.toml" ]; then
        cp "$VAILA_HOME/pyproject.toml" "$VAILA_HOME/pyproject_universal_cpu.toml"
        echo "Backed up pyproject.toml to pyproject_universal_cpu.toml"
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
        uv python pin 3.12.12
    fi

    # Create virtual environment
    echo ""
    echo "Creating/updating virtual environment (.venv)..."
    if [ ! -d ".venv" ]; then
        echo "Creating new virtual environment..."
        uv venv --python 3.12.12
    else
        echo "Virtual environment already exists. uv sync will update it as needed."
    fi

    # Sync dependencies
    echo ""
    echo "Installing vaila dependencies with uv..."
    echo "This may take a few minutes on first run..."
    
    if ! uv sync; then
        echo "Error: uv sync failed. Restoring universal CPU configuration..."
        cp "$VAILA_HOME/pyproject_universal_cpu.toml" "$VAILA_HOME/pyproject.toml"
        echo "Installation failed. Please check the error messages above."
        exit 1
    fi
    echo "Dependencies installed successfully."

    # Detect architecture for PyTorch
    echo ""
    echo "Checking system architecture..."
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]]; then
        echo "Apple Silicon (ARM64) detected."
        PYTORCH_CONFIG="PyTorch with MPS (GPU acceleration via Metal Performance Shaders)"
        PYTORCH_NOTE="PyTorch will automatically use MPS (GPU) on Apple Silicon."
        DEFAULT_PYTORCH_OPTION=2
        DEFAULT_TEXT="(default: 2 - recommended for Apple Silicon)"
    elif [[ "$ARCH" == "x86_64" ]]; then
        echo "Intel Mac (x86_64) detected."
        PYTORCH_CONFIG="PyTorch CPU-only (macOS Intel doesn't support MPS)"
        PYTORCH_NOTE="PyTorch will run on CPU."
        DEFAULT_PYTORCH_OPTION=1
        DEFAULT_TEXT="(default: 1)"
    else
        echo "Architecture $ARCH detected."
        PYTORCH_CONFIG="PyTorch (architecture: $ARCH)"
        PYTORCH_NOTE="PyTorch will be installed for this architecture."
        DEFAULT_PYTORCH_OPTION=1
        DEFAULT_TEXT="(default: 1)"
    fi
    echo ""

    # Prompt user about installing PyTorch/YOLO stack
    echo "---------------------------------------------"
    echo "PyTorch / YOLO installation options"
    echo "  [1] Skip"
    echo "  [2] Install $PYTORCH_CONFIG"
    echo "      + YOLO (ultralytics, boxmot)"
    echo "---------------------------------------------"
    printf "Choose an option [1-2] $DEFAULT_TEXT: "
    read INSTALL_OPTION
    INSTALL_OPTION=${INSTALL_OPTION:-$DEFAULT_PYTORCH_OPTION}

    if [[ "$INSTALL_OPTION" == "2" ]]; then
        echo ""
        echo "Installing PyTorch..."
        if uv pip install torch torchvision torchaudio; then
            echo "PyTorch installed successfully."
            echo ""
            echo "Note: $PYTORCH_NOTE"
            if [[ "$ARCH" == "arm64" ]]; then
                echo "      If MPS is unavailable, it will fall back to CPU."
            fi
            echo ""
            echo "Installing YOLO dependencies (ultralytics, boxmot)..."
            if ! uv pip install ultralytics boxmot; then
                echo "Warning: Failed to install YOLO dependencies."
            fi
        else
            echo "Warning: Failed to install PyTorch."
        fi
    else
        echo ""
        echo "Skipping PyTorch/YOLO installation. You can install later using:"
        echo "  uv pip install torch torchvision torchaudio"
        echo "  uv pip install ultralytics boxmot"
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
        echo "This is required for the GUI. Attempting to fix..."
        echo ""
        echo "Note: On macOS, tkinter requires python-tk from Homebrew."
        echo "If tkinter is still not available after installation, you may need to:"
        echo "  1. Ensure python-tk is installed: brew install python-tk"
        echo "  2. Or use the system Python instead of uv's Python"
    fi

    # Create run_vaila.sh script
    RUN_SCRIPT="$VAILA_HOME/run_vaila.sh"
    echo ""
    echo "Creating run_vaila.sh script..."
    cat <<EOF > "$RUN_SCRIPT"
#!/bin/bash
cd "$VAILA_HOME" || {
    echo "Error: Cannot change to directory $VAILA_HOME"
    exit 1
}

export PATH="\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH"

if [ -d "/opt/homebrew/bin" ]; then
    export PATH="/opt/homebrew/bin:\$PATH"
fi
if [ -d "/usr/local/bin" ]; then
    export PATH="/usr/local/bin:\$PATH"
fi

if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found in $VAILA_HOME"
    exit 1
fi

    # Use venv Python directly to avoid uv run resolving project dependencies
    # This prevents TensorRT resolution errors on macOS when pyproject.toml has gpu extra
    if [ -f ".venv/bin/python" ]; then
        # Activate venv and run directly - avoids project resolution
        source .venv/bin/activate 2>/dev/null || true
        .venv/bin/python vaila.py
    else
        # Fallback to uv run with --no-extra gpu to exclude TensorRT on macOS
        # This prevents the extra gpu from being resolved even though it's in pyproject.toml
        uv run --no-sync --no-extra gpu python vaila.py
    fi

echo
echo "Program finished. Press Enter to close this window..."
read
EOF

    chmod +x "$RUN_SCRIPT"
    
    # Create a convenient 'vaila' command in venv bin directory
    # This allows running 'vaila' directly without 'uv run' which tries to resolve gpu extra
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
        
        # Create a uv-run wrapper that excludes gpu extra on macOS
        # This allows users to use 'uv run vaila.py' without errors
        cat << 'UV_RUN_WRAPPER' > "$VAILA_HOME/.venv/bin/uv-run-vaila"
#!/bin/bash
# Wrapper for uv run that excludes gpu extra on macOS
# This prevents TensorRT resolution errors when using 'uv run vaila.py'
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SCRIPT_DIR" || exit 1
# Use uv run with --no-extra gpu to avoid TensorRT on macOS
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
}

# ============================================================================
# CONDA INSTALLATION METHOD
# ============================================================================

install_with_conda() {
    echo ""
    echo "============================================================"
    echo "Installing vaila using Conda (legacy method)"
    echo "============================================================"
    echo ""

    # Check if Conda is installed
    if ! command -v conda &> /dev/null; then
        echo "Error: Conda is not installed."
        echo "Please install Conda (Anaconda or Miniconda) first:"
        echo "  https://www.anaconda.com/products/individual"
        exit 1
    fi

    # Get Conda base path
    CONDA_BASE=$(conda info --base)

    # Check for Homebrew (for system dependencies)
    if ! command -v brew &> /dev/null; then
        echo "Warning: Homebrew is not installed."
        echo "Some system dependencies may need to be installed manually."
        echo "Consider installing Homebrew: https://brew.sh/"
    else
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
    fi

    # Check if the "vaila" environment already exists
    echo "Checking for existing 'vaila' environment..."
    if conda env list | grep -q "^vaila"; then
        echo ""
        echo "============================================================"
        echo "vaila environment already exists!"
        echo "============================================================"
        echo ""
        echo "Choose installation type:"
        echo "1. UPDATE - Keep existing environment and update vaila files only"
        echo "   (Preserves custom packages and configurations)"
        echo "2. RESET - Remove existing environment and create fresh installation"
        echo "   (Will require reinstalling all packages)"
        echo ""
        
        while true; do
            read -p "Enter your choice (1 for UPDATE, 2 for RESET): " choice
            case $choice in
                1)
                    echo ""
                    echo "Selected: UPDATE - Keeping existing environment"
                    echo "Updating existing 'vaila' environment..."
                    conda env update -n vaila -f "$PROJECT_DIR/yaml_for_conda_env/vaila_mac.yaml" --prune
                    if [ $? -eq 0 ]; then
                        echo "'vaila' environment updated successfully."
                    else
                        echo "Failed to update 'vaila' environment."
                        exit 1
                    fi
                    break
                    ;;
                2)
                    echo ""
                    echo "Selected: RESET - Creating fresh environment"
                    echo "Removing old 'vaila' environment..."
                    conda env remove -n vaila -y
                    if [ $? -eq 0 ]; then
                        echo "Old 'vaila' environment removed successfully."
                    else
                        echo "Warning: Could not remove old environment. Continuing anyway."
                    fi
                    
                    echo "Cleaning conda cache..."
                    conda clean --all -y
                    
                    echo "Creating Conda environment from vaila_mac.yaml..."
                    conda env create -f "$PROJECT_DIR/yaml_for_conda_env/vaila_mac.yaml"
                    if [ $? -eq 0 ]; then
                        echo "'vaila' environment created successfully on macOS."
                    else
                        echo "Failed to create 'vaila' environment."
                        exit 1
                    fi
                    break
                    ;;
                *)
                    echo "Invalid choice. Please enter 1 or 2."
                    ;;
            esac
        done
    else
        echo "'vaila' environment does not exist. Creating new environment..."
        
        echo "Cleaning conda cache..."
        conda clean --all -y
        
        echo "Creating Conda environment from vaila_mac.yaml..."
        conda env create -f "$PROJECT_DIR/yaml_for_conda_env/vaila_mac.yaml"
        if [ $? -eq 0 ]; then
            echo "'vaila' environment created successfully on macOS."
        else
            echo "Failed to create 'vaila' environment."
            exit 1
        fi
    fi

    # Clean destination directory and copy files
    echo ""
    echo "Cleaning destination directory and copying vaila program to the user's home directory..."
    if [ -d "$VAILA_HOME" ]; then
        echo "Removing existing files from destination directory..."
        rm -rf "$VAILA_HOME"/*
    else
        echo "Installing vaila to $VAILA_HOME..."
        mkdir -p "$VAILA_HOME"
    fi

    echo "Copying files..."
    rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv' --exclude='uv.lock' --exclude='.python-version' "$PROJECT_DIR/" "$VAILA_HOME/"

    # Create run_vaila.sh script
    RUN_SCRIPT="$VAILA_HOME/run_vaila.sh"
    echo ""
    echo "Creating run_vaila.sh script..."
    cat <<EOF > "$RUN_SCRIPT"
#!/bin/bash
# Detect Conda installation
if [ -d "\$HOME/anaconda3" ]; then
    source "\$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -d "\$HOME/miniconda3" ]; then
    source "\$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -d "/opt/homebrew/anaconda3" ]; then
    source "/opt/homebrew/anaconda3/etc/profile.d/conda.sh"
elif [ -d "/opt/homebrew/miniconda3" ]; then
    source "/opt/homebrew/miniconda3/etc/profile.d/conda.sh"
else
    echo "Conda not found. Please ensure Conda is installed and in your PATH."
    exit 1
fi

cd "$VAILA_HOME" || {
    echo "Error: Cannot change to directory $VAILA_HOME"
    exit 1
}

conda activate vaila
python3 vaila.py

echo
echo "Program finished. Press Enter to close this window..."
read
EOF

    chmod +x "$RUN_SCRIPT"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Initialize RUN_SCRIPT variable (will be set by install functions)
RUN_SCRIPT="$VAILA_HOME/run_vaila.sh"

# Execute installation based on chosen method
if [[ "$INSTALL_METHOD" == "1" ]]; then
    install_with_uv
else
    install_with_conda
fi

# Ensure RUN_SCRIPT is set (should be set by install functions, but ensure it exists)
if [ -z "$RUN_SCRIPT" ]; then
    RUN_SCRIPT="$VAILA_HOME/run_vaila.sh"
fi

# Create app bundle (common to both methods)
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
echo "IMPORTANT: On macOS, do NOT use 'uv run vaila.py' directly!"
echo "The 'gpu' extra in pyproject.toml includes TensorRT which doesn't support macOS."
echo ""
echo "You can run vaila by:"
echo "1. Recommended: $RUN_SCRIPT"
echo "2. Recommended: cd ~/vaila && .venv/bin/vaila"
echo "3. Alternative: cd ~/vaila && .venv/bin/python vaila.py"
echo "4. Alternative: cd ~/vaila && .venv/bin/uv-run-vaila (wrapper that excludes gpu extra)"
echo "5. Opening 'vaila' from your Applications folder"
echo "   - Launchpad or /Applications/vaila.app (recommended)"
echo "   - Or ~/Applications/vaila.app"
echo ""
echo "If you really need to use 'uv run', use:"
echo "  uv run --no-sync --no-extra gpu python vaila.py"
echo ""