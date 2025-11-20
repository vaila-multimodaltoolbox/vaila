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
# Update: 20 November 2025                                                              #
# Version: 0.2.0                                                                        #
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

# Prompt user about installing PyTorch/YOLO stack
echo ""
echo "---------------------------------------------"
echo "PyTorch / YOLO installation options"
echo "  [1] Skip (default)"
echo "  [2] Install PyTorch + YOLO (ultralytics/boxmot)"
echo "---------------------------------------------"
printf "Choose an option [1-2]: "
read INSTALL_OPTION
INSTALL_OPTION=${INSTALL_OPTION:-1}

if [[ "$INSTALL_OPTION" == "2" ]]; then
    echo ""
    echo "Installing PyTorch (macOS uses MPS acceleration automatically if available)..."
    # On macOS, standard torch usually supports MPS.
    if uv pip install torch torchvision torchaudio; then
        echo "PyTorch installed successfully."
        
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

# Create a run_vaila.sh script using uv
RUN_SCRIPT="$VAILA_HOME/run_vaila.sh"
echo ""
echo "Creating run_vaila.sh script..."
cat <<EOF > "$RUN_SCRIPT"
#!/bin/bash
cd "$VAILA_HOME"
uv run "$VAILA_HOME/vaila.py"
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
    <key>CFBundleIdentifier</key>
    <string>com.vaila.toolbox</string>
    <key>CFBundleName</key>
    <string>vaila</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>0.2.0</string>
    <key>CFBundleVersion</key>
    <string>0.2.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
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

# Try to copy icon if it exists
ICON_SRC="$VAILA_HOME/vaila/images/vaila.icns"
if [ -f "$ICON_SRC" ]; then
    cp "$ICON_SRC" "$APP_DIR/Contents/Resources/"
fi

echo "Application Bundle created. You can find 'vaila' in your Applications folder."

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
echo "2. Opening 'vaila' from your Applications folder (~/Applications/vaila.app)"
echo ""
