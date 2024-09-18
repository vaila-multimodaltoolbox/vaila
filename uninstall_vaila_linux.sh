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
# Version: 1.0                                                                          #
# OS: Ubuntu Linux                                                                      #
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

# Remove the desktop entry
if [ -f "$DESKTOP_ENTRY_PATH" ]; then
    echo "Removing desktop entry..."
    rm -f "$DESKTOP_ENTRY_PATH"
    if [ $? -eq 0 ]; then
        echo "Desktop entry removed successfully."
    else
        echo "Failed to remove desktop entry."
    fi
else
    echo "Desktop entry not found. Skipping removal."
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

echo "Uninstallation completed. vaila has been removed from your system."

