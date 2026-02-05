#!/bin/bash

#########################################################################################
#                                                                                       #
# Script: create_linux_packages.sh                                                      #
# Description: Creates .deb and .rpm packages for vaila using FPM                       #
#                                                                                       #
# Usage:                                                                                #
#   ./create_linux_packages.sh                                                          #
#                                                                                       #
# Prerequisites:                                                                        #
#   - ruby and gem installed                                                            #
#   - fpm installed (gem install fpm)                                                   #
#   - rpmbuild installed (for rpm packages)                                             #
#                                                                                       #
#########################################################################################

set -e

APP_NAME="vaila"
VERSION="0.3.19" # Should match version in scripts
ARCH="all"
MAINTAINER="Prof. Dr. Paulo R. P. Santiago <paulosantiago@usp.br>"
URL="https://github.com/vaila-multimodaltoolbox/vaila"
DESCRIPTION="Multimodal Toolbox for Biomechanics and Motion Analysis"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build_packages"
STAGE_DIR="$BUILD_DIR/stage"
OPT_DIR="$STAGE_DIR/opt/$APP_NAME"
BIN_DIR="$STAGE_DIR/usr/bin"
DESKTOP_DIR="$STAGE_DIR/usr/share/applications"
ICON_DIR="$STAGE_DIR/usr/share/icons/hicolor/256x256/apps"

echo "============================================================"
echo "vaila - Creating Linux Packages (.deb, .rpm)"
echo "============================================================"
echo ""

# Check for fpm
if ! command -v fpm &> /dev/null; then
    echo "Error: fpm is not installed."
    echo "Please install it using RubyGems:"
    echo "  sudo apt install ruby ruby-dev build-essential rpm"
    echo "  sudo gem install fpm"
    exit 1
fi

# Clean build directory
rm -rf "$BUILD_DIR"
mkdir -p "$OPT_DIR"
mkdir -p "$BIN_DIR"
mkdir -p "$DESKTOP_DIR"
mkdir -p "$ICON_DIR"

echo "Preparing files..."

# Copy project files to /opt/vaila
# Exhaustive exclude list to keep package clean
rsync -av \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='build_packages' \
    --exclude='create_linux_packages.sh' \
    --exclude='.cursor' \
    --exclude='.gemini' \
    --exclude='tmp' \
    "$PROJECT_DIR/" "$OPT_DIR/"

# Create launcher script wrapper
cat > "$BIN_DIR/$APP_NAME" <<EOF
#!/bin/bash
export VAILA_HOME="/opt/$APP_NAME"
# Attempt to use the venv python if it exists, otherwise system python or error
if [ -f "\$VAILA_HOME/.venv/bin/python" ]; then
    exec "\$VAILA_HOME/.venv/bin/python" "\$VAILA_HOME/vaila.py" "\$@"
else
    echo "Error: vaila virtual environment not found in \$VAILA_HOME/.venv"
    echo "Please ensure the package was installed correctly or run the setup manually."
    exit 1
fi
EOF
chmod +x "$BIN_DIR/$APP_NAME"

# Copy Icon
if [ -f "$PROJECT_DIR/docs/images/vaila.iconset/icon_256x256.png" ]; then
    cp "$PROJECT_DIR/docs/images/vaila.iconset/icon_256x256.png" "$ICON_DIR/$APP_NAME.png"
elif [ -f "$PROJECT_DIR/vaila/images/vaila_ico.png" ]; then
    cp "$PROJECT_DIR/vaila/images/vaila_ico.png" "$ICON_DIR/$APP_NAME.png"
fi

# Create Desktop Entry
cat > "$DESKTOP_DIR/$APP_NAME.desktop" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=vaila
GenericName=Multimodal Toolbox
Comment=$DESCRIPTION
Exec=/usr/bin/$APP_NAME
Icon=$APP_NAME
Terminal=true
Categories=Science;Education;Utility;
Keywords=biomechanics;motion;analysis;multimodal;
EOF

# Define Packaging Scripts
SCRIPTS_DIR="$BUILD_DIR/scripts"
mkdir -p "$SCRIPTS_DIR"

# Post-Install Script
# This script runs AFTER files are unpacked.
# It attempts to set up the environment.
cat > "$SCRIPTS_DIR/postinst.sh" <<EOF
#!/bin/bash
set -e

VAILA_HOME="/opt/$APP_NAME"

echo "vaila installed to \$VAILA_HOME"
echo "Setting up environment..."

# Detect if we have internet to install uv/deps?
# Ideally, we should use system uv if available, or download it.
# Check for uv
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="\$HOME/.local/bin:\$PATH"
fi

cd "\$VAILA_HOME"

# Permissions: Ensure root owns the files but maybe users can run? 
# Usually in /opt, root owns everything.
# We need to run uv sync as root to create .venv owned by root.
# Users will read from it.

echo "Creating virtual environment and syncing dependencies..."
# This requires internet access. If offline, this will fail.
if uv sync; then
    echo "Environment setup complete."
else
    echo "Warning: Failed to setup environment (uv sync failed)."
    echo "You may need to run 'cd /opt/$APP_NAME && sudo uv sync' manually."
fi

# Update desktop database
if command -v update-desktop-database &> /dev/null; then
    update-desktop-database /usr/share/applications || true
fi
EOF
chmod +x "$SCRIPTS_DIR/postinst.sh"

# Pre-Remove Script
cat > "$SCRIPTS_DIR/prerm.sh" <<EOF
#!/bin/bash
# Clean up the virtual environment which might have generated files not tracked by package manager
rm -rf /opt/$APP_NAME/.venv
rm -f /opt/$APP_NAME/uv.lock
EOF
chmod +x "$SCRIPTS_DIR/prerm.sh"

# Dependencies (Debian)
DEB_DEPENDS="python3, git, curl, wget, ffmpeg, rsync, pkg-config, libcairo2-dev, python3-dev, build-essential, zenity"

# Dependencies (RPM) - Names might differ slightly, using common ones
RPM_DEPENDS="python3, git, curl, wget, ffmpeg, rsync, pkgconfig, cairo-devel, python3-devel, zenity"

echo "Generating .deb package..."
fpm -s dir -t deb \
    -n "$APP_NAME" \
    -v "$VERSION" \
    --architecture "$ARCH" \
    --maintainer "$MAINTAINER" \
    --description "$DESCRIPTION" \
    --url "$URL" \
    --license "MIT" \
    --vendor "vaila-multimodaltoolbox" \
    --prefix / \
    --after-install "$SCRIPTS_DIR/postinst.sh" \
    --before-remove "$SCRIPTS_DIR/prerm.sh" \
    $(echo "$DEB_DEPENDS" | tr ',' '\n' | sed 's/^/--depends /' | tr '\n' ' ') \
    -C "$STAGE_DIR" .

echo "Generating .rpm package..."
# Check for rpmbuild
if command -v rpmbuild &> /dev/null; then
    fpm -s dir -t rpm \
        -n "$APP_NAME" \
        -v "$VERSION" \
        --architecture "$ARCH" \
        --maintainer "$MAINTAINER" \
        --description "$DESCRIPTION" \
        --url "$URL" \
        --license "MIT" \
        --vendor "vaila-multimodaltoolbox" \
        --prefix / \
        --after-install "$SCRIPTS_DIR/postinst.sh" \
        --before-remove "$SCRIPTS_DIR/prerm.sh" \
        $(echo "$RPM_DEPENDS" | tr ',' '\n' | sed 's/^/--depends /' | tr '\n' ' ') \
        -C "$STAGE_DIR" .
else
    echo "Warning: rpmbuild not found. Skipping .rpm generation."
fi

echo ""
echo "============================================================"
echo "Build Complete!"
ls -lh *.deb *.rpm 2>/dev/null
echo "============================================================"
