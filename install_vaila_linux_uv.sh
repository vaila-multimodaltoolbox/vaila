#!/bin/bash

#########################################################################################
#                                                                                       #
# Script: install_vaila_linux_uv.sh                                                     #
# Description: Installs the vaila - Multimodal Toolbox on Ubuntu Linux using uv         #
#              (ultra-fast Python package manager). This replaces the Conda-based       #
#              installation with a modern uv-based setup.                               #
#                                                                                       #
# Usage:                                                                                #
#   1. Download the repository from GitHub manually and extract it.                     #
#   2. Make the script executable:                                                      #
#      chmod +x install_vaila_linux_uv.sh                                               #
#   3. Run the script from the root directory of the extracted repository:              #
#      ./install_vaila_linux_uv.sh                                                      #
#                                                                                       #
# Notes:                                                                                #
#   - uv will be automatically installed if not present                                 #
#   - Python 3.12.12 will be installed via uv if needed                                 #
#   - This script is 10-100x faster than Conda-based installation                       #
#                                                                                       #
# Author: Prof. Dr. Paulo R. P. Santiago                                                #
# Creation: 17 November 2025                                                            #
# Updated: 17 November 2025                                                             #
# Version: 0.2.0                                                                        #
# OS: Linux in Distro: Base Debian .deb via apt package manager, Gnome, KDE, XFCE, etc. #
# Install dependencies from system repositories and uv An extremely fast Python package #
# and project manager, written in Rust.                                                 #
# Reference: https://docs.astral.sh/uv/                                               #
#########################################################################################

set -e  # Exit on error

echo "============================================================"
echo "vaila - Multimodal Toolbox Installation/Update (uv-based)"
echo "============================================================"
echo ""
echo "This script will install or update vaila in: ~/vaila"
echo "If vaila is already installed, it will be updated with the latest code."
echo ""

# Check for missing system dependencies
echo "Verifying system dependencies..."
MISSING_DEPS=()
for pkg in python3 git curl wget ffmpeg rsync pkg-config libcairo2-dev python3-dev build-essential zenity; do
    if ! dpkg -l | grep -q "^ii  $pkg "; then
        MISSING_DEPS+=("$pkg")
    fi
done

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo "Installing missing system dependencies: ${MISSING_DEPS[*]}"
    sudo apt update
    sudo apt install -y "${MISSING_DEPS[@]}"
fi

# Install uv if not present
echo ""
echo "Checking for uv installation..."
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    echo "Using official installer: curl -LsSf https://astral.sh/uv/install.sh | sh"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for current session
    # Note: uv installer may install to ~/.local/bin or ~/.cargo/bin
    if [ -d "$HOME/.local/bin" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi
    if [ -d "$HOME/.cargo/bin" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    
    # Also add to .bashrc for future sessions
    if [ -d "$HOME/.local/bin" ] && ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc 2>/dev/null; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    fi
    if [ -d "$HOME/.cargo/bin" ] && ! grep -q 'export PATH="$HOME/.cargo/bin:$PATH"' ~/.bashrc 2>/dev/null; then
        echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
    fi
    
    # Source .bashrc to get uv in current session
    source ~/.bashrc 2>/dev/null || true
    
    # Verify installation
    if ! command -v uv &> /dev/null; then
        echo "Error: uv installation failed. Please install manually:"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo "  Or visit: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    echo "uv installed successfully!"
else
    echo "uv is already installed."
fi

# Get uv version
UV_VERSION=$(uv --version 2>/dev/null || echo "unknown")
echo "uv version: $UV_VERSION"
echo ""

# Install Python 3.12.12 via uv if needed
echo "Checking Python version..."
if ! command -v python3.12.12 &> /dev/null; then
    echo "Python 3.12.12 not found. Installing via uv..."
    uv python install 3.12.12  # Install Python 3.12.12 via uv
else
    PYTHON_VERSION=$(python3.12.12 --version 2>/dev/null | cut -d' ' -f2 || echo "unknown")
    echo "Python 3.12.12 found: $PYTHON_VERSION"
fi

# Define paths
USER_HOME="$HOME"
VAILA_HOME="$USER_HOME/vaila"
DESKTOP_ENTRY_PATH="$HOME/.local/share/applications/vaila.desktop"
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
    rm -rf "$VAILA_HOME"/*
else
    echo "Installing vaila to $VAILA_HOME..."
fi
mkdir -p "$VAILA_HOME"
# Copy all files and directories except .venv, __pycache__, and build artifacts
# This ensures updates from git pull are properly copied to the user's installation
find "$PROJECT_DIR" -mindepth 1 -maxdepth 1 ! -name '.venv' ! -name '__pycache__' ! -name '*.pyc' ! -name '.git' ! -name 'uv.lock' ! -name '.python-version' -exec cp -Rfa {} "$VAILA_HOME/" \;

# Ensure no stale uv.lock is copied over
if [ -f "$VAILA_HOME/uv.lock" ]; then
    echo "Removing existing uv.lock to avoid stale dependency locks..."
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

# Install Cairo dependencies BEFORE uv sync (needed for pycairo compilation)
echo ""
echo "Installing Cairo system dependencies (required for pycairo)..."
sudo apt install -y libcairo2-dev pkg-config python3-dev build-essential || {
    echo "Warning: Failed to install Cairo dependencies. pycairo may fail to build."
}

# Sync dependencies (install all packages from pyproject.toml)
echo ""
echo "Installing vaila dependencies with uv..."
echo "This may take a few minutes on first run..."
uv sync

# Detect NVIDIA GPU
echo ""
echo "Checking for NVIDIA GPU..."
NVIDIA_GPU_DETECTED=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        NVIDIA_GPU_DETECTED=true
        echo "NVIDIA GPU detected!"
        nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/^/  GPU: /'
    else
        echo "nvidia-smi found but no GPU detected (drivers may not be installed)."
    fi
else
    echo "No NVIDIA GPU detected (nvidia-smi not found)."
fi
echo ""

# Prepare GPU info for prompt
if [[ "$NVIDIA_GPU_DETECTED" == true ]]; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    PYTORCH_DESC="PyTorch (CPU or CUDA/GPU) + YOLO"
    PYTORCH_NOTE="NVIDIA GPU detected: $GPU_NAME"
else
    PYTORCH_DESC="PyTorch (CPU or CUDA/GPU) + YOLO"
    PYTORCH_NOTE="No NVIDIA GPU detected (CUDA option requires GPU)"
fi

# Prompt user about installing PyTorch/YOLO stack
echo "---------------------------------------------"
echo "PyTorch / YOLO installation options"
echo "  [1] Skip (default)"
echo "  [2] Install $PYTORCH_DESC"
if [[ "$NVIDIA_GPU_DETECTED" == true ]]; then
    echo "      ($PYTORCH_NOTE)"
else
    echo "      Note: $PYTORCH_NOTE"
fi
echo "---------------------------------------------"
printf "Choose an option [1-2]: "
read INSTALL_OPTION
INSTALL_OPTION=${INSTALL_OPTION:-1}

if [[ "$INSTALL_OPTION" == "2" ]]; then
    PYTORCH_INSTALLED=false
    echo ""
    
    # Show options based on GPU detection (informative only)
    if [[ "$NVIDIA_GPU_DETECTED" == true ]]; then
        echo "Select PyTorch build:"
        echo "  [1] CPU-only"
        echo "  [2] CUDA/GPU (NVIDIA GPU detected: $GPU_NAME)"
    else
        echo "Select PyTorch build:"
        echo "  [1] CPU-only"
        echo "  [2] CUDA/GPU (requires NVIDIA GPU + drivers)"
        echo "     Note: No NVIDIA GPU detected. CUDA will not work without GPU."
    fi
    
    printf "Choose an option [1-2]: "
    read PYTORCH_OPTION
    PYTORCH_OPTION=${PYTORCH_OPTION:-1}

    if [[ "$PYTORCH_OPTION" == "2" ]]; then
        echo ""
        echo "Installing PyTorch with CUDA support..."
        if uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; then
            PYTORCH_INSTALLED=true
            echo "PyTorch with CUDA installed successfully."
        else
            echo "Warning: Failed to install CUDA-enabled PyTorch."
            echo "This may be due to missing NVIDIA drivers or incompatible CUDA version."
            echo "You can retry later with:"
            echo "  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        fi
    elif [[ "$PYTORCH_OPTION" == "1" ]]; then
        echo ""
        echo "Installing CPU-only PyTorch..."
        if uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; then
            PYTORCH_INSTALLED=true
            echo "CPU-only PyTorch installed successfully."
        else
            echo "Warning: Failed to install CPU-only PyTorch. You can retry later with:"
            echo "  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        fi
    else
        echo "Invalid option. Skipping PyTorch installation."
    fi

    if [ "$PYTORCH_INSTALLED" = true ]; then
        echo ""
        echo "Installing YOLO dependencies (ultralytics, boxmot)..."
        if ! uv pip install ultralytics boxmot; then
            echo "Warning: Failed to install YOLO dependencies. You can install later with:"
            echo "  uv pip install ultralytics boxmot"
        fi
    else
        echo "Skipping YOLO packages because PyTorch installation failed."
    fi
else
    echo ""
    echo "Skipping PyTorch/YOLO installation. You can install later using:"
    echo "  CUDA PyTorch: uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    echo "  CPU PyTorch : uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    echo "  YOLO stack  : uv pip install ultralytics boxmot"
fi

# Verify pycairo installation (should already be installed via uv sync)
echo ""
echo "Verifying pycairo installation..."
if ! uv pip show pycairo &>/dev/null; then
    echo "pycairo not found. Attempting to install..."
    if ! uv pip install pycairo; then
        echo "Warning: pycairo installation failed. Trying with force-reinstall..."
        uv pip install --force-reinstall --no-cache-dir pycairo || {
            echo "Warning: pycairo installation failed. This may cause issues with the application."
            echo "Make sure libcairo2-dev is installed: sudo apt install -y libcairo2-dev pkg-config python3-dev"
        }
    fi
else
    echo "pycairo is already installed."
fi

# Install and configure SSH
echo ""
echo "Installing and configuring OpenSSH Server..."
sudo apt install -y openssh-server || true

echo "Ensuring SSH service is enabled and running..."
sudo systemctl enable ssh 2>/dev/null || true
sudo systemctl start ssh 2>/dev/null || true

# Configure SSH for better security (optional but recommended)
echo "Configuring SSH security settings..."
sudo sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin no/' /etc/ssh/sshd_config 2>/dev/null || true
sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config 2>/dev/null || true
sudo systemctl restart ssh 2>/dev/null || true

# Configure firewall to allow SSH connections
echo "Configuring firewall for SSH..."
if command -v ufw &> /dev/null; then
    sudo ufw allow ssh 2>/dev/null || true
    sudo ufw status 2>/dev/null || true
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

# Create a desktop entry for the application
echo "Creating a desktop entry for vaila..."
cat <<EOF > "$DESKTOP_ENTRY_PATH"
[Desktop Entry]
Version=0.2.0
Name=vaila
GenericName=Multimodal Toolbox
Comment=Multimodal Toolbox for Biomechanics and Motion Analysis
Exec=$RUN_SCRIPT
Icon=$VAILA_HOME/vaila/images/vaila_ico.png
Terminal=true
Type=Application
Categories=Science;Education;Utility;
Keywords=biomechanics;motion;analysis;multimodal;
StartupNotify=true
StartupWMClass=vaila
EOF

# Update desktop database for all desktop environments
echo "Updating desktop database..."
if command -v update-desktop-database &> /dev/null; then
    update-desktop-database "$HOME/.local/share/applications"
fi

# For KDE Plasma, also create a .desktop file in the system applications directory
if [ -d "/usr/share/applications" ]; then
    echo "Creating system-wide desktop entry for KDE compatibility..."
    sudo tee "/usr/share/applications/vaila.desktop" > /dev/null <<EOF
[Desktop Entry]
Version=0.2.0
Name=vaila
GenericName=Multimodal Toolbox
Comment=Multimodal Toolbox for Biomechanics and Motion Analysis
Exec=$RUN_SCRIPT
Icon=$VAILA_HOME/vaila/images/vaila_ico.png
Terminal=true
Type=Application
Categories=Science;Education;Utility;
Keywords=biomechanics;motion;analysis;multimodal;
StartupNotify=true
StartupWMClass=vaila
EOF
    sudo update-desktop-database
fi

# For XFCE, ensure the desktop entry is properly registered
if command -v xfce4-appfinder &> /dev/null; then
    echo "XFCE detected. Ensuring desktop entry is properly registered..."
    if command -v xfce4-panel &> /dev/null; then
        echo "Refreshing XFCE panel..."
        xfce4-panel --restart 2>/dev/null || true
    fi
fi

# For KDE Plasma, refresh the application menu
if command -v kbuildsycoca5 &> /dev/null; then
    echo "KDE Plasma detected. Refreshing application menu..."
    kbuildsycoca5 --noincremental 2>/dev/null || true
elif command -v kbuildsycoca6 &> /dev/null; then
    echo "KDE Plasma 6 detected. Refreshing application menu..."
    kbuildsycoca6 --noincremental 2>/dev/null || true
fi

# For GNOME, refresh the application menu
if command -v gtk-update-icon-cache &> /dev/null; then
    echo "GNOME detected. Updating icon cache..."
    gtk-update-icon-cache -f -t "$HOME/.local/share/icons" 2>/dev/null || true
fi

# Ensure the application directory is owned by the user and has the correct permissions
echo "Ensuring correct ownership and permissions for the application..."
chown -R "$USER:$USER" "$VAILA_HOME"
chmod -R u+rwX "$VAILA_HOME"
chmod +x "$RUN_SCRIPT"

# Verify and install x-terminal-emulator if necessary
if ! command -v x-terminal-emulator &> /dev/null; then
    echo "Installing terminal utility..."
    sudo apt install -y x-terminal-emulator || true
fi
