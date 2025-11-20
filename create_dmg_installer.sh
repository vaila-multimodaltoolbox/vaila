#!/bin/bash

#########################################################################################
#                                                                                       #
# Script: create_dmg_installer.sh                                                       #
# Description: Creates a .dmg disk image installer for vaila on macOS                   #
#              Similar to .exe installer on Windows                                     #
#                                                                                       #
# Usage:                                                                                #
#   1. First run the installation script to create ~/vaila:                             #
#      ./install_vaila_mac_uv.sh                                                        #
#   2. Then run this script to create the .dmg installer:                               #
#      ./create_dmg_installer.sh                                                        #
#                                                                                       #
# Notes:                                                                                #
#   - Requires hdiutil (built-in macOS tool)                                           #
#   - Creates a .dmg file in the current directory                                      #
#   - The .dmg will contain vaila.app ready to drag to Applications                     #
#                                                                                       #
# Author: Prof. Dr. Paulo R. P. Santiago                                                #
# Creation: 20 November 2025                                                           #
# Version: 0.1.0                                                                        #
# OS: macOS                                                                             #
#########################################################################################

set -e  # Exit on error

echo "============================================================"
echo "vaila - Creating macOS DMG Installer"
echo "============================================================"
echo ""

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This script can only be run on macOS."
    exit 1
fi

# Check if hdiutil is available
if ! command -v hdiutil &> /dev/null; then
    echo "Error: hdiutil is not available. This script requires macOS."
    exit 1
fi

# Define paths
USER_HOME="$HOME"
VAILA_HOME="$USER_HOME/vaila"
APP_PATH="$USER_HOME/Applications/vaila.app"  # App is created in ~/Applications/ by install script
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DMG_NAME="vaila_installer"
DMG_TEMP_DIR="/tmp/vaila_dmg"
DMG_OUTPUT="$PROJECT_DIR/${DMG_NAME}.dmg"

# Check if we're in the project directory (must have pyproject.toml and install script)
if [ ! -f "$PROJECT_DIR/pyproject.toml" ]; then
    echo "Error: pyproject.toml not found in $PROJECT_DIR"
    echo "Please run this script from the vaila project root directory."
    exit 1
fi

if [ ! -f "$PROJECT_DIR/install_vaila_mac_uv.sh" ]; then
    echo "Error: install_vaila_mac_uv.sh not found in $PROJECT_DIR"
    echo "Please run this script from the vaila project root directory."
    exit 1
fi

echo "Preparing DMG installer with all necessary files..."

echo "Preparing DMG installer..."
echo "Source: $APP_PATH"
echo "Output: $DMG_OUTPUT"
echo ""

# Clean up any existing DMG files
if [ -f "$DMG_OUTPUT" ]; then
    echo "Removing existing DMG file..."
    rm -f "$DMG_OUTPUT"
fi

# Clean up temp directory
if [ -d "$DMG_TEMP_DIR" ]; then
    echo "Cleaning up temporary directory..."
    rm -rf "$DMG_TEMP_DIR"
fi

# Create temporary directory for DMG contents
echo "Creating temporary DMG directory..."
mkdir -p "$DMG_TEMP_DIR"

# Copy all necessary files from project to DMG (excluding build artifacts and git)
echo "Copying installation files to DMG..."
# Create a directory for the installer package
INSTALLER_DIR="$DMG_TEMP_DIR/vaila_installer"
mkdir -p "$INSTALLER_DIR"

# Copy all project files needed for installation (similar to what install script does)
echo "  Copying project files..."
rsync -av \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='uv.lock' \
    --exclude='.python-version' \
    --exclude='*.dmg' \
    --exclude='vaila_installer.dmg' \
    --exclude='*.temp.dmg' \
    "$PROJECT_DIR/" "$INSTALLER_DIR/"

# Make install script executable
chmod +x "$INSTALLER_DIR/install_vaila_mac_uv.sh"

# Create Applications symlink (for drag-and-drop install)
echo "Creating Applications symlink..."
ln -s /Applications "$DMG_TEMP_DIR/Applications"

# Create a README file with installation instructions
cat > "$DMG_TEMP_DIR/README.txt" << 'EOF'
=====================================
vaila - Installation Instructions
=====================================

To install vaila:

1. Double-click "vaila_installer" folder
2. Open Terminal and run:
   cd vaila_installer
   ./install_vaila_mac_uv.sh

   OR simply drag "vaila_installer" folder to your Desktop,
   open Terminal, and run:
   cd ~/Desktop/vaila_installer
   ./install_vaila_mac_uv.sh

3. The installer will:
   - Install uv (if needed)
   - Install Python 3.12.12
   - Create virtual environment
   - Install all dependencies
   - Create vaila.app in Applications

4. After installation, you can find vaila in:
   - Launchpad
   - /Applications/vaila.app
   - Spotlight (search for "vaila")

=====================================
For more information, visit:
https://github.com/vaila-multimodaltoolbox/vaila
=====================================
EOF

# Set up DMG background and window properties using AppleScript
echo "Configuring DMG appearance..."

# Create background image directory
mkdir -p "$DMG_TEMP_DIR/.background"

# Try to copy logo as background (optional)
if [ -f "$PROJECT_DIR/docs/images/vaila_logo.png" ]; then
    # Resize logo for background (512x384 is good for DMG background)
    if command -v sips &> /dev/null; then
        sips -z 384 512 "$PROJECT_DIR/docs/images/vaila_logo.png" -o "$DMG_TEMP_DIR/.background/background.png" 2>/dev/null || true
    fi
fi

# Copy volume icon to DMG temp directory (must be done BEFORE creating DMG)
echo "Setting up volume icon..."
DMG_ICON=""
# Try to find vaila.icns - prioritize docs/images/vaila.icns
for icon_path in "$PROJECT_DIR/docs/images/vaila.icns" "$PROJECT_DIR/vaila/images/vaila.icns" "$VAILA_HOME/docs/images/vaila.icns"; do
    if [ -f "$icon_path" ]; then
        DMG_ICON="$icon_path"
        break
    fi
done

if [ -n "$DMG_ICON" ]; then
    echo "Found volume icon at: $DMG_ICON"
    # Copy icon to temp directory as .VolumeIcon.icns (this will become the volume icon)
    cp "$DMG_ICON" "$DMG_TEMP_DIR/.VolumeIcon.icns"
    # Mark the volume icon file with the 'V' attribute (invisible) and 'C' (custom icon)
    # This must be done BEFORE creating the DMG
    if command -v SetFile &> /dev/null; then
        SetFile -a V "$DMG_TEMP_DIR/.VolumeIcon.icns" 2>/dev/null || true
        SetFile -a C "$DMG_TEMP_DIR/.VolumeIcon.icns" 2>/dev/null || true
    fi
    echo "Volume icon configured."
else
    echo "Warning: Volume icon (vaila.icns) not found. DMG volume will use default icon."
fi

# Calculate DMG size (app size + 50MB overhead)
DMG_SIZE=$(du -sk "$DMG_TEMP_DIR" | awk '{print $1}')
DMG_SIZE=$((DMG_SIZE + 51200))  # Add 50MB overhead
DMG_SIZE=$((DMG_SIZE / 1024 + 1))  # Convert to MB and add 1

echo "Creating DMG file (size: ${DMG_SIZE}MB)..."
hdiutil create -srcfolder "$DMG_TEMP_DIR" -volname "vaila Installer" -fs HFS+ -fsargs "-c c=64,a=16,e=16" -format UDRW -size ${DMG_SIZE}m "$DMG_OUTPUT.temp.dmg"

# Mount the DMG
echo "Mounting DMG to configure appearance..."
MOUNT_DIR="/Volumes/vaila Installer"
DEVICE=$(hdiutil attach -readwrite -noverify -noautoopen "$DMG_OUTPUT.temp.dmg" | grep -E '^/dev/' | sed 1q | awk '{print $1}')

# Wait for mount to complete
echo "Waiting for DMG to mount..."
sleep 3

# Configure DMG window appearance using AppleScript
echo "Configuring DMG window appearance..."
if [ ! -f "$DMG_TEMP_DIR/vaila.app/Contents/Resources/vaila.icns" ]; then
    echo "Warning: vaila.app icon not found in DMG temp directory. Copying from source..."
    # Ensure Resources directory exists
    mkdir -p "$DMG_TEMP_DIR/vaila.app/Contents/Resources"
    # Use DMG_ICON if available, otherwise try to find it again
    if [ -z "$DMG_ICON" ] || [ ! -f "$DMG_ICON" ]; then
        for icon_path in "$PROJECT_DIR/docs/images/vaila.icns" "$PROJECT_DIR/vaila/images/vaila.icns" "$VAILA_HOME/docs/images/vaila.icns"; do
            if [ -f "$icon_path" ]; then
                DMG_ICON="$icon_path"
                break
            fi
        done
    fi
    if [ -n "$DMG_ICON" ] && [ -f "$DMG_ICON" ]; then
        cp "$DMG_ICON" "$DMG_TEMP_DIR/vaila.app/Contents/Resources/vaila.icns"
        echo "Icon copied to vaila.app in DMG temp directory."
        
        # Apply icon attributes to vaila.app in DMG (before creating DMG)
        if command -v sips &> /dev/null; then
            sips -i "$DMG_TEMP_DIR/vaila.app/Contents/Resources/vaila.icns" &>/dev/null || true
        fi
        if command -v SetFile &> /dev/null; then
            SetFile -a C "$DMG_TEMP_DIR/vaila.app" 2>/dev/null || true
            SetFile -a B "$DMG_TEMP_DIR/vaila.app" 2>/dev/null || true
        fi
    else
        echo "Error: Could not find icon source to copy to vaila.app in DMG."
    fi
else
    echo "vaila.app icon found in DMG temp directory."
fi

# Configure DMG window appearance using AppleScript
echo "Configuring DMG window appearance..."
osascript << 'APPLESCRIPT'
tell application "Finder"
    activate
    try
        set dmg to disk "vaila Installer"
        open dmg
        delay 1
        
        set theWindow to container window of dmg
        set theViewOptions to icon view options of theWindow
        
        -- Configure window
        set toolbar visible of theWindow to false
        set statusbar visible of theWindow to false
        set bounds of theWindow to {400, 100, 920, 420}
        set current view of theWindow to icon view
        
        -- Configure icon view
        set icon size of theViewOptions to 96
        set arrangement of theViewOptions to not arranged
        
        -- Position icons
        try
            set position of item "vaila_installer" of theWindow to {160, 205}
        end try
        try
            set position of item "Applications" of theWindow to {360, 205}
        end try
        try
            set position of item "README.txt" of theWindow to {560, 205}
        end try
        
        -- Set background if available
        try
            if exists file ".background:background.png" of dmg then
                set background picture of theViewOptions to file ".background:background.png" of dmg
            end if
        end try
        
        close theWindow
        update dmg without registering applications
        
    on error errMsg
        display notification "DMG configuration warning: " & errMsg
    end try
end tell
APPLESCRIPT

# Wait a bit for changes to be saved
sleep 2

# Unmount the DMG
echo "Unmounting DMG..."
hdiutil detach "$DEVICE"

# Convert to read-only/compressed DMG
echo "Converting to final DMG format..."
hdiutil convert "$DMG_OUTPUT.temp.dmg" -format UDZO -imagekey zlib-level=9 -o "$DMG_OUTPUT"

# Clean up temporary DMG
rm -f "$DMG_OUTPUT.temp.dmg"

# Set custom icon for the DMG file itself (the .dmg file icon in Finder)
echo "Setting DMG file icon..."
DMG_ICON=""
# Try to find vaila.icns - prioritize docs/images/vaila.icns
for icon_path in "$PROJECT_DIR/docs/images/vaila.icns" "$PROJECT_DIR/vaila/images/vaila.icns" "$VAILA_HOME/docs/images/vaila.icns"; do
    if [ -f "$icon_path" ]; then
        DMG_ICON="$icon_path"
        break
    fi
done

if [ -n "$DMG_ICON" ]; then
    echo "Found icon at: $DMG_ICON"
    # Use sips to attach icon resource to DMG file
    if command -v sips &> /dev/null; then
        sips -i "$DMG_ICON" &>/dev/null || true
    fi
    # Use DeRez/Rez to copy icon resource if available (more reliable)
    if command -v DeRez &> /dev/null && command -v Rez &> /dev/null; then
        TEMP_RSRC="/tmp/vaila_dmg_icon.rsrc"
        # Extract icon resource
        DeRez -only icns "$DMG_ICON" > "$TEMP_RSRC" 2>/dev/null || true
        if [ -f "$TEMP_RSRC" ]; then
            # Append icon resource to DMG file
            Rez -append "$TEMP_RSRC" -o "$DMG_OUTPUT" 2>/dev/null || true
            # Mark DMG file with custom icon attribute
            if command -v SetFile &> /dev/null; then
                SetFile -a C "$DMG_OUTPUT" 2>/dev/null || true
            fi
            rm -f "$TEMP_RSRC"
        fi
    fi
    # Update Finder cache
    touch "$DMG_OUTPUT" 2>/dev/null || true
    killall Finder 2>/dev/null || true
    echo "DMG file icon set successfully."
else
    echo "Warning: DMG file icon (vaila.icns) not found. DMG file will use default icon."
fi

# Clean up temp directory
rm -rf "$DMG_TEMP_DIR"

# Get DMG size
DMG_SIZE_MB=$(du -h "$DMG_OUTPUT" | awk '{print $1}')

echo ""
echo "============================================================"
echo "DMG installer created successfully!"
echo "============================================================"
echo "File: $DMG_OUTPUT"
echo "Size: $DMG_SIZE_MB"
echo ""
echo "The DMG installer is ready for distribution!"
echo "Users can double-click the .dmg file and drag vaila.app"
echo "to Applications to install."
echo ""

