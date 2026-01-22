#!/bin/bash

#########################################################################################
#                                                                                       #
# Script: uninstall_vaila_linux.sh                                                      #
# Description: Uninstalls the vaila - Multimodal Toolbox from Ubuntu Linux, including   #
#              removing the Conda environment, deleting program files from the user's   #
#              home directory, and removing the desktop entry.                          #
#                                                                                       #
# Usage:                                                                                #
#   1. Make the script executable:                                                      #
#      chmod +x uninstall_vaila_linux.sh                                                #
#   2. Run the script:                                                                  #
#      ./uninstall_vaila_linux.sh                                                       #
#                                                                                       #
# Author: Prof. Dr. Paulo R. P. Santiago                                                #
# Date: September 17, 2024                                                              #
# Updated Date: 22 January 2026                                                            #
# Version: 0.3.14                                                                        #
# OS: Ubuntu, Kubuntu, Linux Mint, Pop_OS!, Zorin OS, etc.                              #
#########################################################################################

echo "Starting uninstallation of vaila - Multimodal Toolbox on Linux..."

# Get Conda base path
CONDA_BASE=$(conda info --base 2>/dev/null)

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Cannot proceed with environment removal."
else
    # Remove the 'vaila' Conda environment if it exists
    if conda env list | grep -q "^vaila"; then
        echo "Removing 'vaila' Conda environment..."
        conda remove --name vaila --all -y
        if [ $? -eq 0 ]; then
            echo "'vaila' environment removed successfully."
        else
            echo "Failed to remove 'vaila' environment."
        fi
    else
        echo "'vaila' environment does not exist. Skipping environment removal."
    fi
fi

# Define paths
USER_HOME="$HOME"
VAILA_HOME="$USER_HOME/vaila"
DESKTOP_ENTRY_PATH="$HOME/.local/share/applications/vaila.desktop"
SYSTEM_DESKTOP_ENTRY_PATH="/usr/share/applications/vaila.desktop"
RUN_SCRIPT="$VAILA_HOME/run_vaila.sh"

# Remove the vaila directory from the user's home
if [ -d "$VAILA_HOME" ]; then
    echo "Removing vaila directory from the user's home..."
    rm -rf "$VAILA_HOME"
    if [ $? -eq 0 ]; then
        echo "vaila directory removed successfully."
    else
        echo "Failed to remove vaila directory."
    fi
else
    echo "vaila directory not found in the user's home. Skipping removal."
fi

# Remove the user desktop entry
if [ -f "$DESKTOP_ENTRY_PATH" ]; then
    echo "Removing user desktop entry..."
    rm -f "$DESKTOP_ENTRY_PATH"
    if [ $? -eq 0 ]; then
        echo "User desktop entry removed successfully."
    else
        echo "Failed to remove user desktop entry."
    fi
else
    echo "User desktop entry not found. Skipping removal."
fi

# Remove the system desktop entry
if [ -f "$SYSTEM_DESKTOP_ENTRY_PATH" ]; then
    echo "Removing system desktop entry..."
    sudo rm -f "$SYSTEM_DESKTOP_ENTRY_PATH"
    if [ $? -eq 0 ]; then
        echo "System desktop entry removed successfully."
    else
        echo "Failed to remove system desktop entry."
    fi
else
    echo "System desktop entry not found. Skipping removal."
fi

# Remove the run_vaila.sh script if it exists
if [ -f "$RUN_SCRIPT" ]; then
    echo "Removing run_vaila.sh script..."
    rm -f "$RUN_SCRIPT"
    if [ $? -eq 0 ]; then
        echo "run_vaila.sh script removed successfully."
    else
        echo "Failed to remove run_vaila.sh script."
    fi
fi

# Update desktop database for all desktop environments
echo "Updating desktop database..."
if command -v update-desktop-database &> /dev/null; then
    update-desktop-database "$HOME/.local/share/applications"
    sudo update-desktop-database
fi

# Clean desktop environment specific caches
echo "Cleaning desktop environment caches..."

# KDE Plasma cache cleanup
if command -v plasmashell &> /dev/null; then
    echo "Cleaning KDE Plasma cache..."
    rm -rf "$HOME/.cache/plasmashell" 2>/dev/null || true
    rm -rf "$HOME/.cache/kbuildsycoca5*" 2>/dev/null || true
    rm -rf "$HOME/.cache/kbuildsycoca6*" 2>/dev/null || true
    
    # Refresh KDE application menu
    if command -v kbuildsycoca5 &> /dev/null; then
        kbuildsycoca5 --noincremental 2>/dev/null || true
    elif command -v kbuildsycoca6 &> /dev/null; then
        kbuildsycoca6 --noincremental 2>/dev/null || true
    fi
fi

# XFCE cache cleanup
if command -v xfce4-appfinder &> /dev/null; then
    echo "Cleaning XFCE cache..."
    rm -rf "$HOME/.cache/xfce4/desktop" 2>/dev/null || true
    rm -rf "$HOME/.cache/xfce4/panel" 2>/dev/null || true
fi

# GNOME cache cleanup
if command -v gnome-shell &> /dev/null; then
    echo "Cleaning GNOME cache..."
    if command -v gtk-update-icon-cache &> /dev/null; then
        gtk-update-icon-cache -f -t "$HOME/.local/share/icons" 2>/dev/null || true
    fi
fi

# Cinnamon cache cleanup
if command -v cinnamon-session &> /dev/null; then
    echo "Cleaning Cinnamon cache..."
    rm -rf "$HOME/.cache/cinnamon" 2>/dev/null || true
fi

# MATE cache cleanup
if command -v mate-session &> /dev/null; then
    echo "Cleaning MATE cache..."
    rm -rf "$HOME/.cache/mate" 2>/dev/null || true
fi

# LXDE/LXQt cache cleanup
if command -v lxsession &> /dev/null; then
    echo "Cleaning LXDE cache..."
    rm -rf "$HOME/.cache/lxde" 2>/dev/null || true
fi

if command -v lxqt-session &> /dev/null; then
    echo "Cleaning LXQt cache..."
    rm -rf "$HOME/.cache/lxqt" 2>/dev/null || true
fi

echo "Uninstallation completed. vaila has been removed from your system."
echo ""
echo "Note: You may need to log out and log back in for all changes to take effect."
echo "Nota: Você pode precisar fazer logout e login novamente para que todas as mudanças tenham efeito."

