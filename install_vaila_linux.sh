#!/bin/bash

#########################################################################################
#                                                                                       #
# Script: install_vaila_linux.sh                                                        #
# Description: Installs the vaila - Multimodal Toolbox on Linux                        #
#              Supports both uv (recommended) and Conda (legacy) installation methods #
#                                                                                       #
# Usage:                                                                                #
#   1. Download the repository from GitHub manually and extract it.                     #
#   2. Make the script executable:                                                      #
#      chmod +x install_vaila_linux.sh                                                 #
#   3. Run the script from the root directory of the extracted repository:              #
#      ./install_vaila_linux.sh                                                        #
#                                                                                       #
# Notes:                                                                                #
#   - uv method: uv will be automatically installed if not present                     #
#   - conda method: Requires Conda (Anaconda or Miniconda) to be installed             #
#   - uv will be automatically installed if not present (uv method only)              #
#   - Python 3.12.13 will be installed via uv or conda depending on method chosen     #
#                                                                                       #
# Author: Prof. Dr. Paulo R. P. Santiago                                                #
# Creation: September 17, 2024                                                          #
# Updated: 20 March 2026                                                                #
# Version: 0.3.31                                                                        #
# OS: Ubuntu, Kubuntu, Linux Mint, Pop_OS!, Zorin OS, etc. (Debian-based)             #
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
DESKTOP_ENTRY_PATH="$HOME/.local/share/applications/vaila.desktop"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if running remotely (bootstrap mode)
# If files are missing, download them first
if [ ! -f "$PROJECT_DIR/pyproject.toml" ]; then
    echo "Bootstrap Mode: vaila source not found locally."
    echo "Cloning vaila repository from GitHub..."
    
    if ! command -v git &> /dev/null; then
        echo "Error: git is not installed. Please install git first."
        echo "sudo apt install git"
        exit 1
    fi

    TEMP_DIR=$(mktemp -d)
    echo "Downloading to temporary directory: $TEMP_DIR"
    git clone --depth 1 https://github.com/vaila-multimodaltoolbox/vaila.git "$TEMP_DIR/vaila"
    
    echo "Running installer from downloaded source..."
    chmod +x "$TEMP_DIR/vaila/install_vaila_linux.sh"
    "$TEMP_DIR/vaila/install_vaila_linux.sh" "$@"
    EXIT_CODE=$?
    
    echo "Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
    exit $EXIT_CODE
fi

# ============================================================================
# INSTALL LOCATION
# ============================================================================

echo ""
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

DESKTOP_ENTRY_PATH="$HOME/.local/share/applications/vaila.desktop"

# ============================================================================
# COMMON FUNCTIONS (used by both methods)
# ============================================================================

create_desktop_entry() {
    # Create a desktop entry for the application
    echo "Creating a desktop entry for vaila..."
    
    # Find the best available icon
    ICON_SOURCE=""
    
    # First, try to find icons in the iconset directory (preferred location)
    ICONSET_DIRS=(
        "$PROJECT_DIR/docs/images/vaila.iconset"
        "$VAILA_HOME/docs/images/vaila.iconset"
        "$PROJECT_DIR/vaila/images/vaila.iconset"
        "$VAILA_HOME/vaila/images/vaila.iconset"
    )
    
    ICONSET_DIR=""
    for iconset_dir in "${ICONSET_DIRS[@]}"; do
        if [ -d "$iconset_dir" ]; then
            ICONSET_DIR="$iconset_dir"
            echo "Found iconset directory at: $ICONSET_DIR"
            break
        fi
    done
    
    # If iconset directory found, use appropriate size for Linux (256x256 or 512x512)
    if [ -n "$ICONSET_DIR" ]; then
        # Try 256x256 first (good balance for dock icons)
        if [ -f "$ICONSET_DIR/icon_256x256.png" ]; then
            ICON_SOURCE="$ICONSET_DIR/icon_256x256.png"
            echo "Using icon from iconset: $ICON_SOURCE"
        # Fallback to 512x512 if 256x256 not available
        elif [ -f "$ICONSET_DIR/icon_512x512.png" ]; then
            ICON_SOURCE="$ICONSET_DIR/icon_512x512.png"
            echo "Using icon from iconset: $ICON_SOURCE"
        # Fallback to 128x128
        elif [ -f "$ICONSET_DIR/icon_128x128.png" ]; then
            ICON_SOURCE="$ICONSET_DIR/icon_128x128.png"
            echo "Using icon from iconset: $ICON_SOURCE"
        fi
    fi
    
    # If no icon found in iconset, try other locations
    if [ -z "$ICON_SOURCE" ]; then
        POSSIBLE_ICON_PATHS=(
            "$VAILA_HOME/vaila/images/vaila_ico.png"
            "$VAILA_HOME/vaila/images/vaila_ico_trans.ico"
            "$VAILA_HOME/vaila/images/vaila_logo.png"
            "$VAILA_HOME/docs/images/vaila_ico.png"
            "$VAILA_HOME/docs/images/vaila_logo.png"
            "$PROJECT_DIR/vaila/images/vaila_ico.png"
            "$PROJECT_DIR/docs/images/vaila_ico.png"
        )
        
        for icon_path in "${POSSIBLE_ICON_PATHS[@]}"; do
            if [ -f "$icon_path" ]; then
                ICON_SOURCE="$icon_path"
                echo "Found icon at: $ICON_SOURCE"
                break
            fi
        done
    fi
    
    # If still no icon found, use a default path
    if [ -z "$ICON_SOURCE" ]; then
        ICON_SOURCE="$VAILA_HOME/vaila/images/vaila_ico.png"
        echo "Warning: Icon not found. Using default path: $ICON_SOURCE"
    fi
    
    # Create icons directory in user's local share
    ICONS_DIR="$HOME/.local/share/icons"
    PIXMAPS_DIR="$HOME/.local/share/pixmaps"
    mkdir -p "$ICONS_DIR"
    mkdir -p "$PIXMAPS_DIR"
    
    # Copy icon to user's icons directory with a standard name
    ICON_NAME="vaila"
    ICON_DEST="$ICONS_DIR/${ICON_NAME}.png"
    ICON_DEST_PIX="$PIXMAPS_DIR/${ICON_NAME}.png"
    
    # If source is PNG, copy directly; if ICO, try to convert or use as-is
    if [ -f "$ICON_SOURCE" ]; then
        if [[ "$ICON_SOURCE" == *.png ]]; then
            cp "$ICON_SOURCE" "$ICON_DEST" 2>/dev/null || true
            cp "$ICON_SOURCE" "$ICON_DEST_PIX" 2>/dev/null || true
            echo "Icon copied to: $ICON_DEST and $ICON_DEST_PIX"
        elif [[ "$ICON_SOURCE" == *.ico ]]; then
            # Try to convert ICO to PNG using ImageMagick if available
            if command -v convert &> /dev/null; then
                convert "$ICON_SOURCE" "$ICON_DEST" 2>/dev/null && cp "$ICON_DEST" "$ICON_DEST_PIX" && echo "Icon converted and copied to: $ICON_DEST" || cp "$ICON_SOURCE" "$ICON_DEST" 2>/dev/null || true
            else
                # If no convert, try to use the ICO directly (some systems support it)
                cp "$ICON_SOURCE" "$ICON_DEST" 2>/dev/null || true
                cp "$ICON_SOURCE" "$ICON_DEST_PIX" 2>/dev/null || true
            fi
        else
            # For other formats, try to copy or convert
            cp "$ICON_SOURCE" "$ICON_DEST" 2>/dev/null || true
            cp "$ICON_SOURCE" "$ICON_DEST_PIX" 2>/dev/null || true
        fi
    fi
    
    # Use generic name "vaila" which should be picked up from standard paths
    ICON_NAME_FOR_DESKTOP="vaila"
    # Also define absolute path for desktop shortcut copying
    ICON_PATH_FOR_SHORTCUT="$ICON_DEST"
    
    cat <<EOF > "$DESKTOP_ENTRY_PATH"
[Desktop Entry]
Version=1.0
Type=Application
Name=vaila
GenericName=Multimodal Toolbox
Comment=Multimodal Toolbox for Biomechanics and Motion Analysis
Exec=$RUN_SCRIPT
Icon=$ICON_DEST
Terminal=true
Categories=Science;Education;Utility;
Keywords=biomechanics;motion;analysis;multimodal;
StartupNotify=true
StartupWMClass=Vaila
MimeType=
EOF

    # Update desktop database for all desktop environments
    echo "Updating desktop database..."
    if command -v update-desktop-database &> /dev/null; then
        update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
    fi
    
    # Update icon cache
    if command -v gtk-update-icon-cache &> /dev/null; then
        echo "Updating icon cache..."
        gtk-update-icon-cache -f -t "$ICONS_DIR" 2>/dev/null || true
    fi
    
    # Make desktop entry executable (required for some desktop environments)
    chmod +x "$DESKTOP_ENTRY_PATH" 2>/dev/null || true


    # For KDE Plasma, also create a .desktop file in the system applications directory
    if [ -d "/usr/share/applications" ]; then
        echo "Creating system-wide desktop entry for KDE compatibility..."
        sudo tee "/usr/share/applications/vaila.desktop" > /dev/null <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=vaila
GenericName=Multimodal Toolbox
Comment=Multimodal Toolbox for Biomechanics and Motion Analysis
Exec=$RUN_SCRIPT
Icon=$ICON_DEST
Terminal=true
Categories=Science;Education;Utility;
Keywords=biomechanics;motion;analysis;multimodal;
StartupNotify=true
StartupWMClass=Vaila
MimeType=
EOF
        sudo update-desktop-database 2>/dev/null || true
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
    
    # For Ubuntu/GNOME Shell, try to restart the dock (if running)
    if command -v gnome-shell &> /dev/null; then
        echo "GNOME Shell detected. Refreshing application menu..."
        # Try to reload the shell (non-destructive)
        dbus-send --type=method_call --dest=org.gnome.Shell /org/gnome/Shell org.gnome.Shell.Eval string:'global.reexec_self()' 2>/dev/null || true
    fi
    
    echo ""
    echo "Desktop entry created successfully!"
    echo "Location: $DESKTOP_ENTRY_PATH"
    echo "Icon location: $ICON_DEST"
    echo ""
    echo "Note: You may need to log out and log back in, or restart your desktop"
    echo "      environment for the icon to appear in the dock/launcher."
    echo "      Alternatively, you can search for 'vaila' in the application menu."
}

setup_ssh() {
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
        echo "Updating uv..."
        uv self update || echo "Warning: Failed to update uv. Continuing with current version."
    fi

    # Get uv version
    UV_VERSION=$(uv --version 2>/dev/null || echo "unknown")
    echo "uv version: $UV_VERSION"
    echo ""

    # Install Python 3.12.13 via uv if needed
    echo "Checking Python version..."
    if ! uv python list | grep -q "3.12.13"; then
        echo "Python 3.12.13 not found. Installing via uv..."
        uv python install 3.12.13
    else
        echo "Python 3.12.13 found."
    fi

    echo "Clean destination directory and copy files..."
    
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

    # Select appropriate pyproject.toml template based on GPU detection and user choice
    echo ""
    echo "Selecting pyproject.toml configuration..."
    
    # Detect NVIDIA GPU
    HAS_NVIDIA_GPU=false
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        HAS_NVIDIA_GPU=true
    fi
    
    USE_GPU=false
    if [[ "$HAS_NVIDIA_GPU" == true ]]; then
        echo "NVIDIA GPU detected. Install with GPU support (CUDA 12.8)? [Y/n]"
        read -r gpu_choice
        if [[ "$gpu_choice" != "n" && "$gpu_choice" != "N" ]]; then
            USE_GPU=true
        fi
    else
        echo "No NVIDIA GPU detected. Using CPU-only configuration."
    fi

    USE_SAM_EXTRA=false
    echo ""
    if [[ "$USE_GPU" != true ]]; then
        echo "Note: SAM 3 video in vailá uses NVIDIA CUDA at runtime. On this CPU-only profile you can skip (N)"
        echo "      or install the extra for use after switching to a CUDA pyproject (see AGENTS.md, Hybrid)."
    fi
    echo "Install optional SAM 3 (Meta) segmentation stack (extra 'sam', CUDA-oriented)? [y/N]"
    read -r sam_choice
    if [[ "$sam_choice" == "y" || "$sam_choice" == "Y" ]]; then
        USE_SAM_EXTRA=true
    fi
    
    # Choose template
    if [[ "$USE_GPU" == true ]]; then
        if [ -f "$VAILA_HOME/pyproject_linux_cuda12.toml" ]; then
            cp "$VAILA_HOME/pyproject_linux_cuda12.toml" "$VAILA_HOME/pyproject.toml"
            echo "Using Linux CUDA 12.8 configuration."
        else
            echo "Warning: pyproject_linux_cuda12.toml not found. Using CPU-only."
            cp "$VAILA_HOME/pyproject_universal_cpu.toml" "$VAILA_HOME/pyproject.toml"
            USE_GPU=false
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

    # Generate lock file
    echo ""
    echo "Generating lock file (uv.lock)..."
    uv lock --upgrade

    # Install Cairo dependencies BEFORE uv sync (needed for pycairo compilation)
    echo ""
    echo "Installing Cairo system dependencies (required for pycairo)..."
    echo "This ensures pycairo can be compiled successfully during uv sync."
    sudo apt update
    sudo apt install -y libcairo2-dev pkg-config python3-dev build-essential || {
        echo "Warning: Failed to install Cairo dependencies. pycairo may fail to build."
    }

    # Sync dependencies
    echo ""
    echo "Installing vaila dependencies with uv..."
    echo "This may take a few minutes on first run..."
    
    UV_SYNC_CMD=(uv sync)
    if [[ "$USE_GPU" == true ]]; then
        UV_SYNC_CMD+=(--extra gpu)
    fi
    if [[ "$USE_SAM_EXTRA" == true ]]; then
        UV_SYNC_CMD+=(--extra sam)
    fi
    if ! "${UV_SYNC_CMD[@]}"; then
        echo "Error: uv sync failed. Restoring universal CPU configuration..."
        cp "$VAILA_HOME/pyproject_universal_cpu.toml" "$VAILA_HOME/pyproject.toml"
        echo "Installation failed. Please check the error messages above."
        exit 1
    fi
    echo "Dependencies installed successfully."
    echo ""
    echo "PyTorch, torchvision, torchaudio, ultralytics, and boxmot are already installed via uv sync from pyproject.toml."

    if [[ "$USE_SAM_EXTRA" == true ]]; then
        echo ""
        echo "------------------------------------------------------------"
        echo "SAM 3 (optional): weights are gated on Hugging Face (facebook/sam3)."
        echo "  1) In a browser, log in and accept the model license:"
        echo "       https://huggingface.co/facebook/sam3"
        echo "  2) Store a token on this machine (Read access):"
        echo "       cd \"$VAILA_HOME\" && uv run hf auth login"
        echo "     Use --force if a different HF account is already cached."
        echo "  3) Optional: download weights into the repo"
        echo "       uv run vaila/vaila_sam.py --download-weights"
        echo "------------------------------------------------------------"
        read -r -p "Run 'uv run hf auth login' now from $VAILA_HOME? [y/N] " hf_login_now
        if [[ "$hf_login_now" == "y" || "$hf_login_now" == "Y" ]]; then
            (cd "$VAILA_HOME" && uv run hf auth login) || {
                echo "Warning: hf auth login failed or was cancelled. You can run it later."
            }
        fi
    fi

    # Verify pycairo installation
    echo ""
    echo "Verifying pycairo installation..."
    if ! uv pip show pycairo &>/dev/null; then
        echo "pycairo not found. Attempting to install..."
        uv pip install pycairo || {
            echo "Warning: pycairo installation failed. Trying with force-reinstall..."
            uv pip install --force-reinstall --no-cache-dir pycairo || {
                echo "Warning: pycairo installation failed. This may cause issues with the application."
            }
        }
    else
        echo "pycairo is already installed."
    fi

    # Verify environment is properly set up by checking for PIL (Pillow)
    echo ""
    echo "Verifying environment setup..."
    if ! uv run python -c "import PIL; print('PIL OK')" 2>&1 | grep -q "PIL OK"; then
        echo "Warning: Environment verification failed. PIL module not found. Running uv sync again..."
        if ! "${UV_SYNC_CMD[@]}"; then
            echo "Error: Failed to sync dependencies during verification."
            exit 1
        fi
    else
        echo "Environment verification successful."
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
        # Fallback creation if bin/run_vaila.sh is missing for some reason
        RUN_SCRIPT="$VAILA_HOME/run_vaila.sh"
        cat <<EOF > "$RUN_SCRIPT"
#!/bin/bash
cd "$VAILA_HOME"
uv run --no-sync "$VAILA_HOME/vaila.py"
# Keep terminal open after execution
echo
echo "Program finished. Press Enter to close this window..."
read
EOF
        chmod +x "$RUN_SCRIPT"
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

    # Check for missing dependencies
    echo "Verifying system dependencies..."
    MISSING_DEPS=()
    for pkg in python3 pip git curl wget ffmpeg rsync pkg-config libcairo2-dev python3-dev; do
        if ! dpkg -l | grep -q "^ii  $pkg "; then
            MISSING_DEPS+=("$pkg")
        fi
    done

    if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
        echo "Installing missing system dependencies: ${MISSING_DEPS[*]}"
        sudo apt update
        sudo apt install -y "${MISSING_DEPS[@]}"
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
        echo "   (Preserves NVIDIA CUDA installations and other custom packages)"
        echo "2. RESET - Remove existing environment and create fresh installation"
        echo "   (Will require reinstalling NVIDIA CUDA and other custom packages)"
        echo ""
        
        while true; do
            read -p "Enter your choice (1 for UPDATE, 2 for RESET): " choice
            case $choice in
                1)
                    echo ""
                    echo "Selected: UPDATE - Keeping existing environment"
                    echo "Updating existing 'vaila' environment..."
                    conda env update -n vaila -f "$PROJECT_DIR/yaml_for_conda_env/vaila_linux.yaml" --prune
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
                    
                    echo "Creating Conda environment from vaila_linux.yaml..."
                    conda env create -f "$PROJECT_DIR/yaml_for_conda_env/vaila_linux.yaml"
                    if [ $? -eq 0 ]; then
                        echo "'vaila' environment created successfully on Linux."
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
        
        echo "Creating Conda environment from vaila_linux.yaml..."
        conda env create -f "$PROJECT_DIR/yaml_for_conda_env/vaila_linux.yaml"
        if [ $? -eq 0 ]; then
            echo "'vaila' environment created successfully on Linux."
        else
            echo "Failed to create 'vaila' environment."
            exit 1
        fi
    fi

    # Check for NVIDIA GPU and install PyTorch with CUDA if available
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected. Installing PyTorch with CUDA support..."
        conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia -n vaila -y
        if [ $? -eq 0 ]; then
            echo "PyTorch with CUDA support installed successfully."
        else
            echo "Failed to install PyTorch with CUDA support."
        fi
    else
        echo "No NVIDIA GPU detected. Skipping PyTorch with CUDA installation."
    fi

    # Clean destination directory and copy files
    echo ""
    
    if [[ "$INSTALL_LOC_OPTION" == "2" ]]; then
        echo "Local install selected. Skipping file copy."
        echo "Using current directory as VAILA_HOME."
    else
        echo "Cleaning destination directory and copying vaila program to the user's home directory..."
        if [ -d "$VAILA_HOME" ]; then
            echo "Removing existing files from destination directory..."
            rm -rf "$VAILA_HOME"/*
        else
            echo "Installing vaila to $VAILA_HOME..."
            mkdir -p "$VAILA_HOME"
        fi
        cp -Rfa "$PROJECT_DIR/." "$VAILA_HOME/"
    fi

    # Remove ffmpeg from the Conda environment if installed
    echo "Removing ffmpeg installed via Conda..."
    conda remove -n vaila ffmpeg -y

    # Install the system version of ffmpeg
    echo "Installing ffmpeg from system repositories..."
    sudo apt update
    sudo apt install ffmpeg -y

    # Activate the Conda environment
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate vaila

    # Install Cairo dependencies and pycairo
    echo "Installing Cairo dependencies..."
    sudo apt install libcairo2-dev pkg-config python3-dev -y

    # First try normal installation of pycairo
    echo "Trying normal installation of pycairo..."
    if pip install pycairo; then
        echo "Normal pycairo installation succeeded."
    else
        echo "Normal pycairo installation failed. Trying with force-reinstall option..."
        pip install --force-reinstall --no-cache-dir pycairo
        
        if [ $? -eq 0 ]; then
            echo "Forced pycairo installation succeeded."
        else
            echo "Warning: pycairo installation failed. This may cause issues with the application."
        fi
    fi

    # Install moviepy using pip
    echo "Installing moviepy..."
    pip install moviepy

    # Grant permissions to the Conda environment directory
    echo "Setting permissions for Conda environment..."
    VAILA_ENV_DIR="${CONDA_BASE}/envs/vaila"
    if [ -d "$VAILA_ENV_DIR" ]; then
        chmod -R u+rwX "$VAILA_ENV_DIR"
        echo "Permissions set for Conda environment directory."
    else
        echo "Conda environment directory not found at $VAILA_ENV_DIR."
    fi

    # Create run_vaila.sh script
    RUN_SCRIPT="$VAILA_HOME/run_vaila.sh"
    echo ""
    echo "Creating run_vaila.sh script..."
    cat <<EOF > "$RUN_SCRIPT"
#!/bin/bash
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate vaila
python3 "$VAILA_HOME/vaila.py"
# Keep terminal open after execution
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

# Common setup tasks (for both methods)
setup_ssh
create_desktop_entry

# Ensure correct ownership and permissions
echo "Ensuring correct ownership and permissions for the application..."
chown -R "$USER:$USER" "$VAILA_HOME"
chmod -R u+rwX "$VAILA_HOME"
chmod +x "$RUN_SCRIPT"

# Verify and install x-terminal-emulator if necessary
if ! command -v x-terminal-emulator &> /dev/null; then
    echo "Installing terminal utility..."
    sudo apt install -y x-terminal-emulator || true
fi


echo "=================================================================="
echo "vaila installation completed successfully!"
echo "Instalação do vaila concluída com sucesso!"
echo "=================================================================="
echo ""
echo "If the application doesn't appear in your application menu, try:"
echo "Se o aplicativo não aparecer no menu de aplicações, tente:"
echo "1. Log out and log back in"
echo "   1. Fazer logout e login novamente"
echo "2. Restart your desktop environment"
echo "   2. Reiniciar seu ambiente de desktop"
echo "3. Run the installation script again"
echo "   3. Executar o script de instalação novamente"
echo ""
