#!/bin/bash

#########################################################################################
#                                                                                       #
# Script: install_vaila_mac_uv.sh                                                       #
# Description: Installs the vaila - Multimodal Toolbox on macOS using uv                #
#              (ultra-fast Python package manager).                                     #
#                                                                                       #
# Usage:                                                                                #
#   1. Download the repository from GitHub manually and extract it.                     #
#   2. Make the script executable:                                                      #
#      chmod +x install_vaila_mac_uv.sh                                                 #
#   3. Run the script from the root directory of the extracted repository:              #
#      ./install_vaila_mac_uv.sh                                                      #
#                                                                                       #
# Notes:                                                                                #
#   - Requires Homebrew (https://brew.sh/) for system dependencies                      #
#   - uv will be automatically installed if not present                                 #
#   - Python 3.12.12 will be installed via uv if needed                                 #
#                                                                                       #
# Author: Prof. Dr. Paulo R. P. Santiago                                                #
# Creation: 20 November 2025      
# Update: 17 December 2025                                                              #
# Version: 0.2.1                                                                        #
# OS: macOS (Apple Silicon or Intel)                                                    #
#########################################################################################

set -e  # Exit on error

echo "============================================================"
echo "vaila - Multimodal Toolbox Installation/Update (uv-based)"
echo "============================================================"
echo ""
echo "This script will install or update vaila in: ~/vaila"
echo "If vaila is already installed, it will be updated with the latest code."
echo ""

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "Error: Homebrew is not installed."
    echo "Please install Homebrew first: https://brew.sh/"
    echo "Command: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
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

# Install python-tk for Python 3.12 (required for tkinter/GUI)
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
    
    # Add uv to PATH for current session
    if [ -d "$HOME/.local/bin" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi
    if [ -d "$HOME/.cargo/bin" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    
    # Source .zshrc or .bash_profile if they exist to ensure path persistence
    if [ -f "$HOME/.zshrc" ]; then
        source "$HOME/.zshrc" 2>/dev/null || true
    elif [ -f "$HOME/.bash_profile" ]; then
        source "$HOME/.bash_profile" 2>/dev/null || true
    fi
    
    # Verify installation
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

# Get uv version
UV_VERSION=$(uv --version 2>/dev/null || echo "unknown")
echo "uv version: $UV_VERSION"

# Configure shell autocompletion
echo "Configuring shell autocompletion for uv..."
SHELL_NAME=$(basename "$SHELL")
if [ "$SHELL_NAME" = "zsh" ] || [ -f "$HOME/.zshrc" ]; then
    if ! grep -q "uv generate-shell-completion zsh" "$HOME/.zshrc" 2>/dev/null; then
        echo 'eval "$(uv generate-shell-completion zsh)"' >> "$HOME/.zshrc"
        echo "Added uv autocompletion to ~/.zshrc"
    else
        echo "uv autocompletion already configured in ~/.zshrc"
    fi
fi
if [ "$SHELL_NAME" = "bash" ] || [ -f "$HOME/.bash_profile" ]; then
    TARGET_FILE="$HOME/.bash_profile"
    [ -f "$HOME/.bashrc" ] && TARGET_FILE="$HOME/.bashrc"
    
    if ! grep -q "uv generate-shell-completion bash" "$TARGET_FILE" 2>/dev/null; then
        echo 'eval "$(uv generate-shell-completion bash)"' >> "$TARGET_FILE"
        echo "Added uv autocompletion to $TARGET_FILE"
    else
        echo "uv autocompletion already configured in $TARGET_FILE"
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

# Define paths
USER_HOME="$HOME"
VAILA_HOME="$USER_HOME/vaila"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if pyproject.toml exists
if [ ! -f "$PROJECT_DIR/pyproject.toml" ]; then
    echo "Error: pyproject.toml not found in $PROJECT_DIR"
    echo "Please ensure you're running this script from the vaila project root."
    exit 1
fi

# Clean destination directory and copy the entire vaila program to the user's home directory
echo ""
if [ -d "$VAILA_HOME" ]; then
    echo "Updating existing vaila installation in $VAILA_HOME..."
    echo "Removing old files (keeping .venv to be recreated)..."
    # Use rsync for safer deletion/update or just rm -rf content
    # To match linux script logic:
    find "$VAILA_HOME" -mindepth 1 -maxdepth 1 ! -name '.venv' -exec rm -rf {} +
else
    echo "Installing vaila to $VAILA_HOME..."
    mkdir -p "$VAILA_HOME"
fi

# Copy all files and directories except .venv, __pycache__, and build artifacts
echo "Copying files..."
rsync -av --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' --exclude='uv.lock' --exclude='.python-version' "$PROJECT_DIR/" "$VAILA_HOME/"

# Ensure no stale uv.lock is copied over (rsync might have excluded it, but double check)
if [ -f "$VAILA_HOME/uv.lock" ]; then
    rm -f "$VAILA_HOME/uv.lock"
fi

# Change to vaila home directory for uv operations
cd "$VAILA_HOME"

# Initialize uv project (if not already initialized)
echo ""
echo "Initializing uv project..."
if [ ! -f ".python-version" ]; then
    uv python pin 3.12.12
fi

# Create virtual environment explicitly
echo ""
echo "Creating virtual environment (.venv)..."
if [ -d ".venv" ]; then
    echo "Virtual environment already exists. Removing old one..."
    rm -rf .venv
fi
uv venv --python 3.12.12

# Generate lock file
echo ""
echo "Generating lock file (uv.lock)..."
uv lock

# Sync dependencies (install all packages from pyproject.toml)
echo ""
echo "Installing vaila dependencies with uv..."
echo "This may take a few minutes on first run..."
uv sync

# Detect architecture before showing options
echo ""
echo "Checking system architecture..."
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    echo "Apple Silicon (ARM64) detected."
    PYTORCH_CONFIG="PyTorch with MPS (GPU acceleration via Metal Performance Shaders)"
    PYTORCH_NOTE="PyTorch will automatically use MPS (GPU) on Apple Silicon."
elif [[ "$ARCH" == "x86_64" ]]; then
    echo "Intel Mac (x86_64) detected."
    PYTORCH_CONFIG="PyTorch CPU-only (macOS Intel doesn't support MPS)"
    PYTORCH_NOTE="PyTorch will run on CPU."
else
    echo "Architecture $ARCH detected."
    PYTORCH_CONFIG="PyTorch (architecture: $ARCH)"
    PYTORCH_NOTE="PyTorch will be installed for this architecture."
fi
echo ""

# Set default option based on architecture
# On ARM64 (Apple Silicon), default to installing PyTorch since MPS GPU acceleration is available
if [[ "$ARCH" == "arm64" ]]; then
    DEFAULT_PYTORCH_OPTION=2
    DEFAULT_TEXT="(default: 2 - recommended for Apple Silicon)"
else
    DEFAULT_PYTORCH_OPTION=1
    DEFAULT_TEXT="(default: 1)"
fi

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
# pycairo often needs pkg-config to find cairo
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
    echo ""
    echo "For now, continuing installation. If GUI doesn't work, run:"
    echo "  brew install python-tk"
    echo "  Then reinstall vaila or manually link tkinter to your Python environment."
fi

# Create a run_vaila.sh script using uv
RUN_SCRIPT="$VAILA_HOME/run_vaila.sh"
echo ""
echo "Creating run_vaila.sh script..."
cat <<EOF > "$RUN_SCRIPT"
#!/bin/bash
# Change to vaila directory (where pyproject.toml and .venv are located)
cd "$VAILA_HOME" || {
    echo "Error: Cannot change to directory $VAILA_HOME"
    exit 1
}

# Ensure uv is in PATH
export PATH="\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH"

# Verify we're in the correct directory (must have pyproject.toml and .venv)
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found in $VAILA_HOME"
    exit 1
fi

# Run vaila using uv run from the project directory
# uv run must be executed from the directory containing pyproject.toml
uv run python vaila.py

# Keep terminal open after execution
echo
echo "Program finished. Press Enter to close this window..."
read
EOF

# Make the run_vaila.sh script executable
chmod +x "$RUN_SCRIPT"

# Create a simple macOS Application Bundle (optional but nice)
APP_NAME="vaila"
APP_DIR="$USER_HOME/Applications/$APP_NAME.app"
# Ensure user Applications directory exists
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
cat <<EOF > "$APP_DIR/Contents/MacOS/vaila"
#!/bin/bash
# Open a terminal window to run the script
open -a Terminal "$RUN_SCRIPT"
EOF

chmod +x "$APP_DIR/Contents/MacOS/vaila"

# Try to find or create the icon
ICON_CREATED=false

# First, try to find existing .icns file - prioritize docs/images/vaila.icns
ICON_SRC=""
for icon_path in "$PROJECT_DIR/docs/images/vaila.icns" "$VAILA_HOME/docs/images/vaila.icns" "$PROJECT_DIR/vaila/images/vaila.icns" "$VAILA_HOME/vaila/images/vaila.icns"; do
    if [ -f "$icon_path" ]; then
        ICON_SRC="$icon_path"
        echo "Found icon at: $ICON_SRC"
        break
    fi
done

# If .icns not found, try to create it from .iconset directory using iconutil
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
        
        # Create temporary .icns file
        TEMP_ICNS="/tmp/vaila.icns"
        if command -v iconutil &> /dev/null; then
            # Use iconutil to convert .iconset to .icns
            if iconutil -c icns "$ICONSET_DIR" -o "$TEMP_ICNS" 2>/dev/null; then
                ICON_SRC="$TEMP_ICNS"
                ICON_CREATED=true
                echo "Icon converted successfully from iconset using iconutil."
            else
                echo "Warning: Failed to convert iconset to .icns using iconutil."
            fi
        else
            echo "Warning: iconutil not found. Trying Pillow as fallback..."
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
        
        # Create temporary .icns file
        TEMP_ICNS="/tmp/vaila.icns"
        
        # Use Python with Pillow to convert PNG to ICNS
        cat > /tmp/create_icns.py << 'PYTHON_SCRIPT'
import sys
from PIL import Image

def create_icns_from_png(png_path, icns_path):
    try:
        # Open the PNG image
        img = Image.open(png_path)
        
        # Convert RGBA to RGB if needed (ICNS format doesn't support alpha well)
        if img.mode in ('RGBA', 'LA', 'P'):
            # Create a white background
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            rgb_img.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
            img = rgb_img
        
        # Save as ICNS
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

        # Run the Python script using uv run python
        if uv run python /tmp/create_icns.py "$PNG_SRC" "$TEMP_ICNS" 2>/dev/null; then
            if [ -f "$TEMP_ICNS" ]; then
                ICON_SRC="$TEMP_ICNS"
                ICON_CREATED=true
                echo "Icon created successfully from PNG using Pillow."
            fi
        else
            echo "Warning: Failed to create .icns from PNG using Pillow."
        fi
        
        # Clean up temporary Python script
        rm -f /tmp/create_icns.py
    fi
fi

# Copy icon to app bundle if found or created
if [ -n "$ICON_SRC" ]; then
    echo "Copying icon to application bundle..."
    cp "$ICON_SRC" "$APP_DIR/Contents/Resources/vaila.icns"
    echo "Icon copied successfully."
    
    # Clean up temporary file if we created it
    if [ "$ICON_CREATED" = true ] && [ "$ICON_SRC" = "/tmp/vaila.icns" ]; then
        rm -f "$TEMP_ICNS"
    fi
    
    # Update icon cache for macOS Finder
    echo "Updating macOS icon cache..."
    touch "$APP_DIR"
    # Register with LaunchServices
    /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP_DIR" 2>/dev/null || true
    # Restart Finder to show updated icon
    killall Finder 2>/dev/null || true
else
    echo "Warning: Icon file (vaila.icns or vaila.iconset) not found. Application will use default icon."
    echo "Expected locations:"
    echo "  - $VAILA_HOME/docs/images/vaila.icns"
    echo "  - $VAILA_HOME/docs/images/vaila.iconset/"
fi

echo "Application Bundle created at $APP_DIR."

# Force macOS to recognize the icon properly BEFORE creating symlink
echo ""
echo "Setting application icon..."

# Apply icon using Python script (AppKit)
if [ -f "$APP_DIR/Contents/Resources/vaila.icns" ]; then
    echo "Applying icon to app bundle using Python script..."
    
    # Ensure we're in the project directory for uv to work correctly
    cd "$VAILA_HOME"
    
    # Run the Python script using uv run python
    if uv run python "$PROJECT_DIR/set_mac_icon.py" "$APP_DIR" "$APP_DIR/Contents/Resources/vaila.icns"; then
        echo "Icon applied successfully to App Bundle."
    else
        echo "Warning: Failed to apply icon to App Bundle using Python script."
    fi
    
    # Also try to apply to the installation directory
    echo "Applying icon to installation directory..."
    if uv run python "$PROJECT_DIR/set_mac_icon.py" "$VAILA_HOME" "$APP_DIR/Contents/Resources/vaila.icns"; then
        echo "Icon applied successfully to installation directory."
    else
        echo "Warning: Failed to apply icon to installation directory."
    fi

    # Force Finder refresh
    touch "$APP_DIR"
    touch "$VAILA_HOME"
fi

# Create a symbolic link in /Applications (like the Conda script does)
# This is important for macOS to properly recognize the app icon
echo ""
echo "Creating symbolic link in /Applications..."
if [ -e "/Applications/vaila.app" ]; then
    echo "Removing existing symlink in /Applications..."
    sudo rm -rf "/Applications/vaila.app"
fi

echo "Creating symlink from /Applications/vaila.app to $APP_DIR..."
sudo ln -s "$APP_DIR" "/Applications/vaila.app"

# Ensure the symbolic link has the correct permissions
if [ -L "/Applications/vaila.app" ]; then
    sudo chown -h "${USER}:admin" "/Applications/vaila.app"
    echo "Symlink created successfully in /Applications."
    
    # Register symlink with Launch Services
    echo "Registering symlink with Launch Services..."
    /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "/Applications/vaila.app" 2>/dev/null || true
    touch "/Applications/vaila.app" 2>/dev/null || true
else
    echo "Warning: Failed to create symlink in /Applications."
fi

# Rebuild Launch Services database for Applications folder
echo ""
echo "Rebuilding Launch Services cache..."
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -kill -r -domain local -domain system -domain user 2>/dev/null || true

# Restart Finder and Dock to show updated icon
echo "Restarting Finder and Dock..."
killall Finder 2>/dev/null || true
killall Dock 2>/dev/null || true

echo ""
echo "Icon cache refreshed. The vaila logo should appear correctly now."

# Ensure the application directory is owned by the user and has the correct permissions
echo "Ensuring correct ownership and permissions..."
chown -R "$USER" "$VAILA_HOME"
chmod -R u+rwX "$VAILA_HOME"

echo ""
echo "============================================================"
echo "vaila installation completed successfully!"
echo "============================================================"
echo "You can run vaila by:"
echo "1. Running: $RUN_SCRIPT"
echo "2. Opening 'vaila' from your Applications folder"
echo "   - Launchpad or /Applications/vaila.app (recommended)"
echo "   - Or ~/Applications/vaila.app"
echo ""
