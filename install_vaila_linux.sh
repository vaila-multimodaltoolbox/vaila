#!/bin/bash

#########################################################################################
#                                                                                       #
# Script: install_vaila_linux.sh                                                        #
# Description: Installs the vaila - Multimodal Toolbox on Linux using uv (Astral).      #
#              Official install method: uv (Astral).                                      #
#                                                                                       #
# Usage:                                                                                #
#   1. Download/clone the repository.                                                   #
#   2. Make the script executable:                                                      #
#      chmod +x install_vaila_linux.sh                                                  #
#   3. Run the script from the root directory of the repository:                        #
#      ./install_vaila_linux.sh                                                         #
#                                                                                       #
# Notes:                                                                                #
#   - uv will be automatically installed if not present.                                #
#   - Python 3.12.13 will be installed via `uv python install`.                         #
#   - Portable/git installs keep the committed uv.lock (so git pull keeps working).     #
#   - Profile copies (~/vaila) may regenerate uv.lock; no git tree there.               #
#                                                                                       #
# Author: Prof. Dr. Paulo R. P. Santiago                                                #
# Creation: September 17, 2024                                                          #
# Updated: 24 August 2026
# Version: 0.3.113
# OS: Ubuntu, Kubuntu, Linux Mint, Pop_OS!, Zorin OS, etc. (Debian-based)               #
#########################################################################################

set -e  # Exit on error

echo "============================================================"
echo "vaila - Multimodal Toolbox Installation/Update (uv)"
echo "============================================================"
echo ""
echo "This script will install or update vaila (default: current directory, portable)."
echo "If vaila is already installed, it will be updated with the latest code."
echo ""

# Define paths
USER_HOME="$HOME"
VAILA_HOME="$(pwd)"
DESKTOP_ENTRY_PATH="$HOME/.local/share/applications/vaila.desktop"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bootstrap: clone repo if pyproject.toml is missing locally
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
echo "  [1] Current Directory ($(pwd)) - Local/Portable - Recommended"
echo "  [2] User Profile (~/vaila)"
echo "---------------------------------------------"
printf "Choose an option [1-2] (default: 1): "
read INSTALL_LOC_OPTION
INSTALL_LOC_OPTION=${INSTALL_LOC_OPTION:-1}

if [[ "$INSTALL_LOC_OPTION" == "2" ]]; then
    VAILA_HOME="$USER_HOME/vaila"
    echo "Installing in user profile: $VAILA_HOME"
else
    VAILA_HOME="$(pwd)"
    echo "Installing in current directory (portable): $VAILA_HOME"
fi

DESKTOP_ENTRY_PATH="$HOME/.local/share/applications/vaila.desktop"

# ============================================================================
# DESKTOP ENTRY (called after RUN_SCRIPT is set)
# ============================================================================

create_desktop_entry() {
    echo "Creating a desktop entry for vaila..."

    # Find the best available icon
    ICON_SOURCE=""

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

    if [ -n "$ICONSET_DIR" ]; then
        if [ -f "$ICONSET_DIR/icon_256x256.png" ]; then
            ICON_SOURCE="$ICONSET_DIR/icon_256x256.png"
        elif [ -f "$ICONSET_DIR/icon_512x512.png" ]; then
            ICON_SOURCE="$ICONSET_DIR/icon_512x512.png"
        elif [ -f "$ICONSET_DIR/icon_128x128.png" ]; then
            ICON_SOURCE="$ICONSET_DIR/icon_128x128.png"
        fi
        [ -n "$ICON_SOURCE" ] && echo "Using icon from iconset: $ICON_SOURCE"
    fi

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

    if [ -z "$ICON_SOURCE" ]; then
        ICON_SOURCE="$VAILA_HOME/vaila/images/vaila_ico.png"
        echo "Warning: Icon not found. Using default path: $ICON_SOURCE"
    fi

    ICONS_DIR="$HOME/.local/share/icons"
    PIXMAPS_DIR="$HOME/.local/share/pixmaps"
    mkdir -p "$ICONS_DIR"
    mkdir -p "$PIXMAPS_DIR"

    ICON_NAME="vaila"
    ICON_DEST="$ICONS_DIR/${ICON_NAME}.png"
    ICON_DEST_PIX="$PIXMAPS_DIR/${ICON_NAME}.png"

    if [ -f "$ICON_SOURCE" ]; then
        if [[ "$ICON_SOURCE" == *.png ]]; then
            cp "$ICON_SOURCE" "$ICON_DEST" 2>/dev/null || true
            cp "$ICON_SOURCE" "$ICON_DEST_PIX" 2>/dev/null || true
            echo "Icon copied to: $ICON_DEST and $ICON_DEST_PIX"
        elif [[ "$ICON_SOURCE" == *.ico ]]; then
            if command -v convert &> /dev/null; then
                convert "$ICON_SOURCE" "$ICON_DEST" 2>/dev/null && cp "$ICON_DEST" "$ICON_DEST_PIX" || cp "$ICON_SOURCE" "$ICON_DEST" 2>/dev/null || true
            else
                cp "$ICON_SOURCE" "$ICON_DEST" 2>/dev/null || true
                cp "$ICON_SOURCE" "$ICON_DEST_PIX" 2>/dev/null || true
            fi
        else
            cp "$ICON_SOURCE" "$ICON_DEST" 2>/dev/null || true
            cp "$ICON_SOURCE" "$ICON_DEST_PIX" 2>/dev/null || true
        fi
    fi

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

    echo "Updating desktop database..."
    if command -v update-desktop-database &> /dev/null; then
        update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
    fi

    if command -v gtk-update-icon-cache &> /dev/null; then
        echo "Updating icon cache..."
        gtk-update-icon-cache -f -t "$ICONS_DIR" 2>/dev/null || true
    fi

    chmod +x "$DESKTOP_ENTRY_PATH" 2>/dev/null || true

    # System-wide entry for KDE Plasma
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

    if command -v xfce4-appfinder &> /dev/null; then
        echo "XFCE detected. Refreshing panel..."
        xfce4-panel --restart 2>/dev/null || true
    fi

    if command -v kbuildsycoca5 &> /dev/null; then
        kbuildsycoca5 --noincremental 2>/dev/null || true
    elif command -v kbuildsycoca6 &> /dev/null; then
        kbuildsycoca6 --noincremental 2>/dev/null || true
    fi

    if command -v gtk-update-icon-cache &> /dev/null; then
        gtk-update-icon-cache -f -t "$HOME/.local/share/icons" 2>/dev/null || true
    fi

    if command -v gnome-shell &> /dev/null; then
        dbus-send --type=method_call --dest=org.gnome.Shell /org/gnome/Shell org.gnome.Shell.Eval string:'global.reexec_self()' 2>/dev/null || true
    fi

    echo ""
    echo "Desktop entry created successfully!"
    echo "Location: $DESKTOP_ENTRY_PATH"
    echo "Icon location: $ICON_DEST"
    echo ""
    echo "Note: You may need to log out and log back in, or restart your desktop"
    echo "      environment for the icon to appear in the dock/launcher."
}

setup_ssh() {
    echo ""
    echo "Installing and configuring OpenSSH Server..."
    sudo apt install -y openssh-server || true

    echo "Ensuring SSH service is enabled and running..."
    sudo systemctl enable ssh 2>/dev/null || true
    sudo systemctl start ssh 2>/dev/null || true

    echo "Configuring SSH security settings..."
    sudo sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin no/' /etc/ssh/sshd_config 2>/dev/null || true
    sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config 2>/dev/null || true
    sudo systemctl restart ssh 2>/dev/null || true

    echo "Configuring firewall for SSH..."
    if command -v ufw &> /dev/null; then
        sudo ufw allow ssh 2>/dev/null || true
        sudo ufw status 2>/dev/null || true
    fi
}

# ============================================================================
# UV INSTALLATION
# ============================================================================

echo ""
echo "============================================================"
echo "Installing vaila using uv"
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

    if [ -d "$HOME/.local/bin" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi
    if [ -d "$HOME/.cargo/bin" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    fi

    if [ -d "$HOME/.local/bin" ] && ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc 2>/dev/null; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    fi
    if [ -d "$HOME/.cargo/bin" ] && ! grep -q 'export PATH="$HOME/.cargo/bin:$PATH"' ~/.bashrc 2>/dev/null; then
        echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
    fi

    source ~/.bashrc 2>/dev/null || true

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

echo "Preparing destination directory..."

if [[ "$INSTALL_LOC_OPTION" != "2" ]]; then
    echo "Local/portable install selected. Skipping rsync file copy."
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

# Portable install into a git clone: keep committed uv.lock so `git pull` works.
# Profile copy (~/vaila, no .git): regenerating the lock is fine.
IS_GIT_TREE=false
if [ -d "$VAILA_HOME/.git" ]; then
    IS_GIT_TREE=true
fi

if [[ "$IS_GIT_TREE" == true ]]; then
    echo "Git working tree detected at $VAILA_HOME — keeping committed uv.lock."
else
    if [ -f "$VAILA_HOME/uv.lock" ]; then
        echo "Removing uv.lock in profile install (no git tree)..."
        rm -f "$VAILA_HOME/uv.lock"
    fi
fi

cd "$VAILA_HOME"

# Select appropriate pyproject.toml template based on GPU detection and user choice
echo ""
echo "Selecting pyproject.toml configuration..."

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
    echo "      or install the extra for later use after switching to a CUDA pyproject (see AGENTS.md)."
fi
echo "Install optional SAM 3 (Meta) segmentation stack (extra 'sam', CUDA-oriented)? [y/N]"
read -r sam_choice
if [[ "$sam_choice" == "y" || "$sam_choice" == "Y" ]]; then
    USE_SAM_EXTRA=true
fi

USE_SAPIENS_EXTRA=false
if [[ "$USE_GPU" == true ]]; then
    echo ""
    echo "Install optional Sapiens2 Pose (Meta 308-keypoint pose, extra 'sapiens', CUDA)? [y/N]"
    read -r sapiens_choice
    if [[ "$sapiens_choice" == "y" || "$sapiens_choice" == "Y" ]]; then
        USE_SAPIENS_EXTRA=true
    fi
fi

USE_FIFA_EXTRA=false
if [[ "$USE_GPU" == true ]]; then
    echo ""
    echo "Install optional FIFA Skeletal Tracking Light / SAM 3D Body stack (markerless 3D mesh, extra 'fifa', CUDA)? [y/N]"
    read -r fifa_choice
    if [[ "$fifa_choice" == "y" || "$fifa_choice" == "Y" ]]; then
        USE_FIFA_EXTRA=true
    fi
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

# Lock + sync:
# - Git clone (portable): prefer committed uv.lock via `uv sync --frozen` so pull stays clean.
# - GPU template on a git clone: must relock for CUDA indexes; warn before pull.
# - Profile copy (no .git): `uv lock --upgrade` then sync (legacy behaviour).
echo ""
LOCK_WAS_REGENERATED=false
if [[ "$IS_GIT_TREE" == true ]]; then
    if [[ "$USE_GPU" == true ]]; then
        echo "CUDA template selected in a git clone — regenerating uv.lock for this machine..."
        uv lock
        LOCK_WAS_REGENERATED=true
    else
        echo "Using committed uv.lock (no uv lock --upgrade) so git pull is not blocked."
    fi
else
    echo "Generating lock file (uv.lock)..."
    uv lock --upgrade
    LOCK_WAS_REGENERATED=true
fi

# Install Cairo dependencies BEFORE uv sync (needed for pycairo compilation)
echo ""
echo "Installing Cairo system dependencies (required for pycairo)..."
sudo apt update
sudo apt install -y libcairo2-dev pkg-config python3-dev build-essential || {
    echo "Warning: Failed to install Cairo dependencies. pycairo may fail to build."
}

# Sync dependencies
echo ""
echo "Installing vaila dependencies with uv..."
echo "This may take a few minutes on first run..."

UV_SYNC_CMD=(uv sync)
if [[ "$IS_GIT_TREE" == true && "$LOCK_WAS_REGENERATED" != true ]]; then
    UV_SYNC_CMD+=(--frozen)
fi
if [[ "$USE_GPU" == true ]]; then
    UV_SYNC_CMD+=(--extra gpu)
fi
if [[ "$USE_SAM_EXTRA" == true ]]; then
    UV_SYNC_CMD+=(--extra sam)
fi
if [[ "$USE_SAPIENS_EXTRA" == true ]]; then
    UV_SYNC_CMD+=(--extra sapiens)
fi
if [[ "$USE_FIFA_EXTRA" == true ]]; then
    UV_SYNC_CMD+=(--extra fifa)
fi
if ! "${UV_SYNC_CMD[@]}"; then
    if [[ "$IS_GIT_TREE" == true && "$LOCK_WAS_REGENERATED" != true ]]; then
        echo "Frozen sync failed — retrying without --frozen (may update uv.lock)..."
        UV_SYNC_CMD=(uv sync)
        if [[ "$USE_GPU" == true ]]; then
            UV_SYNC_CMD+=(--extra gpu)
        fi
        if [[ "$USE_SAM_EXTRA" == true ]]; then
            UV_SYNC_CMD+=(--extra sam)
        fi
        if [[ "$USE_SAPIENS_EXTRA" == true ]]; then
            UV_SYNC_CMD+=(--extra sapiens)
        fi
        if [[ "$USE_FIFA_EXTRA" == true ]]; then
            UV_SYNC_CMD+=(--extra fifa)
        fi
        if "${UV_SYNC_CMD[@]}"; then
            LOCK_WAS_REGENERATED=true
        else
            echo "Error: uv sync failed. Restoring universal CPU configuration..."
            cp "$VAILA_HOME/pyproject_universal_cpu.toml" "$VAILA_HOME/pyproject.toml"
            echo "Installation failed. Please check the error messages above."
            exit 1
        fi
    else
        echo "Error: uv sync failed. Restoring universal CPU configuration..."
        cp "$VAILA_HOME/pyproject_universal_cpu.toml" "$VAILA_HOME/pyproject.toml"
        echo "Installation failed. Please check the error messages above."
        exit 1
    fi
fi
echo "Dependencies installed successfully."
echo ""
echo "PyTorch, torchvision, torchaudio, ultralytics, and boxmot are installed via uv sync from pyproject.toml."

# Verify + repair CUDA wheel integrity (GPU template only).
# Real bug hit in production: uv sync can report "nothing to do" for an
# nvidia-*-cu12 package whose dist-info is present but whose actual .so
# payload is missing on disk (broken hardlink / interrupted extraction /
# disk full) -- `import torch` then fails with e.g.
# "ImportError: libcusparseLt.so.0: cannot open shared object file",
# invisible to uv's own "already satisfied" bookkeeping.
if [[ "$USE_GPU" == true ]]; then
    echo ""
    echo "Verifying NVIDIA/PyTorch CUDA wheel integrity..."
    BROKEN="$(uv run python bin/verify_cuda_libs.py --quiet 2>/dev/null || true)"
    if [[ -n "$BROKEN" ]]; then
        echo "Warning: corrupted CUDA wheels detected (metadata present, files missing): $(echo "$BROKEN" | tr '\n' ' ')"
        echo "Reinstalling only the broken packages..."
        REPAIR_CMD=(uv sync)
        for pkg in $BROKEN; do REPAIR_CMD+=(--reinstall-package "$pkg"); done
        if [[ "$USE_GPU" == true ]]; then REPAIR_CMD+=(--extra gpu); fi
        if [[ "$USE_SAM_EXTRA" == true ]]; then REPAIR_CMD+=(--extra sam); fi
        if [[ "$USE_SAPIENS_EXTRA" == true ]]; then REPAIR_CMD+=(--extra sapiens); fi
        if [[ "$USE_FIFA_EXTRA" == true ]]; then REPAIR_CMD+=(--extra fifa); fi
        "${REPAIR_CMD[@]}"
        STILL_BROKEN="$(uv run python bin/verify_cuda_libs.py --quiet 2>/dev/null || true)"
        if [[ -n "$STILL_BROKEN" ]]; then
            echo "Warning: still broken after reinstall: $(echo "$STILL_BROKEN" | tr '\n' ' ')"
            echo "Check disk space (df -h) and, if the uv cache and .venv are on different"
            echo "filesystems, try: export UV_LINK_MODE=copy   then re-run this installer."
        else
            echo "CUDA wheel integrity repaired."
        fi
    else
        echo "CUDA wheel integrity verified."
    fi
    uv run python -c "import torch; print('torch', torch.__version__, '- CUDA available:', torch.cuda.is_available())" \
        || echo "Warning: torch import still failing after CUDA wheel repair -- see errors above."
fi

# Verify + repair the Sapiens2 editable install.
# Real bug hit in production: `uv sync` (even with --extra sapiens) does not
# know about the local editable checkout at .local/third_party/sapiens2 --
# a plain sync can silently drop it. Re-register it if the checkout exists
# on disk but the package no longer imports (cheap, no network).
if [[ "$USE_SAPIENS_EXTRA" == true ]]; then
    if ! uv run python -c "import sapiens" >/dev/null 2>&1; then
        if [[ -d "$VAILA_HOME/.local/third_party/sapiens2" ]]; then
            echo "Warning: sapiens checkout exists but is not importable -- re-registering editable install..."
            uv pip install -e "$VAILA_HOME/.local/third_party/sapiens2" \
                && echo "sapiens editable install repaired." \
                || echo "Warning: failed to repair sapiens editable install; run: bash bin/setup_sapiens2.sh"
        fi
    fi
fi

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

if [[ "$USE_SAPIENS_EXTRA" == true ]]; then
    echo ""
    echo "------------------------------------------------------------"
    echo "Sapiens2 Pose (optional): clone + weights via bin/setup_sapiens2.sh"
    echo "  - Clones facebookresearch/sapiens2 into .local/third_party/sapiens2 (editable install)"
    echo "  - Downloads pose (1B default) + DETR detector to vaila/models/sapiens2/"
    echo "  - GUI: Frame B -> YOLO + FB -> Sapiens2 Pose"
    echo "  - Test: uv run vaila/vaila_sapiens.py -i tests/markerless_2d_analysis/ -o /tmp/out --dry-run"
    echo "  - License: Meta Sapiens2 License (not AGPL) — see vaila/help/vaila_sapiens.md"
    echo "------------------------------------------------------------"
    read -r -p "Run 'bash bin/setup_sapiens2.sh' now from $VAILA_HOME? [y/N] " sapiens_setup_now
    if [[ "$sapiens_setup_now" == "y" || "$sapiens_setup_now" == "Y" ]]; then
        if [[ -x "$VAILA_HOME/bin/setup_sapiens2.sh" ]]; then
            (cd "$VAILA_HOME" && bash bin/setup_sapiens2.sh) || {
                echo "Warning: setup_sapiens2.sh failed or was cancelled. You can run it later:"
                echo "  cd \"$VAILA_HOME\" && bash bin/setup_sapiens2.sh"
            }
        else
            echo "Warning: bin/setup_sapiens2.sh not found. Run manually after updating the repo."
        fi
    fi
fi

if [[ "$USE_FIFA_EXTRA" == true ]]; then
    echo ""
    echo "------------------------------------------------------------"
    echo "FIFA Skeletal Tracking Light / SAM 3D Body (optional): clone + weights via bin/setup_fifa_sam3d.sh"
    echo "  - Clones facebookresearch/sam-3d-body into sam_3d_body/ (NOT pip-installable; runtime deps only)"
    echo "  - Downloads gated facebook/sam-3d-body-dinov3 weights into vaila/models/sam-3d-dinov3/"
    echo "  - GUI: Frame B -> YOLO + FB -> SAM3+DINOv3 3D (markerless 3D mesh + metric joints)"
    echo "  - Test: uv run python -u vaila/sam3dinov3.py -i tests/markerless_2d_analysis/ -o /tmp/out --dry-run"
    echo "  - License: SAM 3D Body keeps its Meta license (not AGPL) — see vaila/help/sam3dinov3.md"
    echo "------------------------------------------------------------"
    read -r -p "Run 'bash bin/setup_fifa_sam3d.sh' now from $VAILA_HOME? [y/N] " fifa_setup_now
    if [[ "$fifa_setup_now" == "y" || "$fifa_setup_now" == "Y" ]]; then
        if [[ -x "$VAILA_HOME/bin/setup_fifa_sam3d.sh" ]]; then
            (cd "$VAILA_HOME" && bash bin/setup_fifa_sam3d.sh) || {
                echo "Warning: setup_fifa_sam3d.sh failed or was cancelled. You can run it later:"
                echo "  cd \"$VAILA_HOME\" && bash bin/setup_fifa_sam3d.sh"
            }
        else
            echo "Warning: bin/setup_fifa_sam3d.sh not found. Run manually after updating the repo."
        fi
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

# ============================================================================
# MAIN EXECUTION
# ============================================================================

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
if [[ "$IS_GIT_TREE" == true ]]; then
    if [[ "$LOCK_WAS_REGENERATED" == true ]] || ! git -C "$VAILA_HOME" diff --quiet -- uv.lock pyproject.toml 2>/dev/null; then
        echo "NOTE (git pull): local pyproject.toml / uv.lock differ from the repo"
        echo "(common after a CUDA template switch). Before pulling:"
        echo "  git -C \"$VAILA_HOME\" restore uv.lock pyproject.toml"
        echo "  git -C \"$VAILA_HOME\" pull"
        echo "Then re-apply the platform template if needed:"
        echo "  bash bin/setup_pyproject.sh --target=linux-cuda --yes"
        echo ""
    else
        echo "Git tree kept clean for uv.lock — you can 'git pull' normally."
        echo ""
    fi
fi
echo "If the application doesn't appear in your application menu, try:"
echo "Se o aplicativo não aparecer no menu de aplicações, tente:"
echo "1. Log out and log back in"
echo "   1. Fazer logout e login novamente"
echo "2. Restart your desktop environment"
echo "   2. Reiniciar seu ambiente de desktop"
echo "3. Run the installation script again"
echo "   3. Executar o script de instalação novamente"
echo ""
