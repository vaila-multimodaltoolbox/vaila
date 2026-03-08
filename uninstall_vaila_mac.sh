#!/bin/bash

#########################################################################################
#                                                                                       #
# Script: uninstall_vaila_mac.sh                                                        #
# Description: Uninstalls the vaila - Multimodal Toolbox from macOS, including the      #
#              uv virtual environment removal, deletion of program files from the       #
#              user's home directory, and removal of the application from Applications. #
#              Also refreshes Launchpad to remove the icon.                             #
#                                                                                       #
# Usage:                                                                                #
#   1. Make the script executable:                                                      #
#      chmod +x uninstall_vaila_mac.sh                                                  #
#   2. Run the script:                                                                  #
#      ./uninstall_vaila_mac.sh                                                         #
#      (You will be prompted for your password when sudo is required)                   #
#                                                                                       #
# Author: Prof. Dr. Paulo R. P. Santiago                                                #
# Date: September 17, 2024                                                              #
# Updated Date: 30 December 2025                                                        #
# Update: 27 January 2026                                                               #
# Version: 0.3.16                                                                        #
# OS: macOS                                                                             #
#########################################################################################

# Note: We don't use 'set -e' here because some commands (like killall) may fail
# normally if processes aren't running, which is OK.

echo "============================================================"
echo "vaila - Multimodal Toolbox Uninstallation"
echo "============================================================"
echo ""

# Define paths
USER_HOME="$HOME"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect if running from the installation directory (Local Install or Default)
if [ -f "$SCRIPT_DIR/vaila.py" ]; then
    VAILA_HOME="$SCRIPT_DIR"
    echo "Detected vaila installation in current directory: $VAILA_HOME"
else
    # Fallback to default location
    VAILA_HOME="$USER_HOME/vaila"
    echo "Using default vaila location: $VAILA_HOME"
fi
USER_APP_PATH="$USER_HOME/Applications/vaila.app"
SYSTEM_APP_PATH="/Applications/vaila.app"
RUN_SCRIPT="$VAILA_HOME/run_vaila.sh"
LOG_FILE="$USER_HOME/vaila_app.log"

# Ask for confirmation
echo "This script will remove vaila from your system:"
echo "  - Application: $USER_APP_PATH and $SYSTEM_APP_PATH"
echo "  - Installation directory: $VAILA_HOME"
echo "  - Virtual environment (.venv)"
echo "  - All related files and configurations"
echo ""
printf "Do you want to continue? (yes/no): "
read CONFIRM

if [ "$CONFIRM" != "yes" ] && [ "$CONFIRM" != "y" ]; then
    echo "Uninstallation cancelled."
    exit 0
fi

echo ""
echo "Starting uninstallation..."

# Check for legacy Conda installation (backward compatibility)
if command -v conda &> /dev/null; then
    if conda info --envs 2>/dev/null | grep -qw "^vaila"; then
        echo "Found legacy Conda environment 'vaila'. Removing..."
        conda remove --name vaila --all -y 2>/dev/null || {
            echo "Warning: Failed to remove Conda environment. Continuing anyway..."
        }
        echo "Legacy Conda environment removed."
    fi
fi

# Remove vaila from Applications first (before removing VAILA_HOME).
# Try both system (/Applications) and user (~/Applications) locations.

# 1. Remove /Applications/vaila.app (symlink or real bundle)
if [ -e "$SYSTEM_APP_PATH" ] || [ -L "$SYSTEM_APP_PATH" ]; then
    echo "Removing vaila app from /Applications..."
    if [ -L "$SYSTEM_APP_PATH" ]; then
        sudo rm -f "$SYSTEM_APP_PATH"
    else
        sudo rm -rf "$SYSTEM_APP_PATH"
    fi
    if [ $? -eq 0 ]; then
        echo "[OK] /Applications/vaila.app removed successfully."
    else
        echo "Warning: Could not remove /Applications/vaila.app (sudo may have failed). You can remove it manually:"
        echo "  sudo rm -rf $SYSTEM_APP_PATH"
    fi
else
    echo "/Applications/vaila.app not found. Skipping."
fi

# 2. Remove ~/Applications/vaila.app (actual app bundle when install used symlink in /Applications)
if [ -e "$USER_APP_PATH" ] || [ -L "$USER_APP_PATH" ]; then
    echo "Removing vaila app from ~/Applications..."
    rm -rf "$USER_APP_PATH"
    if [ $? -eq 0 ]; then
        echo "[OK] ~/Applications/vaila.app removed successfully."
    else
        echo "Warning: Could not remove ~/Applications/vaila.app. You can remove it manually:"
        echo "  rm -rf $USER_APP_PATH"
    fi
else
    echo "~/Applications/vaila.app not found. Skipping."
fi

# Remove the vaila installation directory (includes .venv, project files, run_vaila.sh)
if [ -d "$VAILA_HOME" ]; then
    echo "Removing vaila installation directory (including .venv)..."
    rm -rf "$VAILA_HOME"
    if [ $? -eq 0 ]; then
        echo "[OK] Installation directory removed successfully."
    else
        echo "Error: Failed to remove installation directory."
        echo "You may need to remove it manually: rm -rf $VAILA_HOME"
        exit 1
    fi
else
    echo "Installation directory not found. Skipping."
fi

# Remove the log file if it exists
if [ -f "$LOG_FILE" ]; then
    echo "Removing vaila log file..."
    rm -f "$LOG_FILE" && echo "[OK] Log file removed." || echo "Warning: Failed to remove log file."
else
    echo "Log file not found. Skipping."
fi

# Verify Applications are cleared; if not, tell user to remove manually
if [ -e "$SYSTEM_APP_PATH" ] || [ -L "$SYSTEM_APP_PATH" ] || [ -e "$USER_APP_PATH" ] || [ -L "$USER_APP_PATH" ]; then
    echo ""
    echo "Note: vaila.app is still present in Applications. Remove it manually:"
    [ -e "$SYSTEM_APP_PATH" ] || [ -L "$SYSTEM_APP_PATH" ] && echo "  sudo rm -rf $SYSTEM_APP_PATH"
    [ -e "$USER_APP_PATH" ] || [ -L "$USER_APP_PATH" ] && echo "  rm -rf $USER_APP_PATH"
    echo ""
fi

# Clean up Launch Services database to remove cached app references
echo ""
echo "Cleaning up Launch Services database..."
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister \
    -kill -r -domain local -domain system -domain user 2>/dev/null || {
    echo "Warning: Failed to rebuild Launch Services database."
}

# Refresh Launchpad and Dock to remove any cached icons
echo "Refreshing Dock and Launchpad..."
killall Dock 2>/dev/null || true
echo "[OK] Dock refreshed."

# Force Finder to refresh
echo "Refreshing Finder..."
killall Finder 2>/dev/null || true

echo ""
echo "============================================================"
echo "Uninstallation completed successfully!"
echo "============================================================"
echo "vaila has been removed from your system."
echo ""
echo "Removed:"
echo "  - Application bundles (user and system)"
echo "  - Installation directory: $VAILA_HOME"
echo "  - Virtual environment (.venv)"
echo "  - All related files"
echo ""
echo "The application icon should no longer appear in Launchpad."
echo "If the icon still appears, restart your Mac or log out/in."
echo ""
