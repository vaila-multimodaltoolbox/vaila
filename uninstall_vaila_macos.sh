#!/bin/bash

#########################################################################################
#                                                                                       #
# Script: uninstall_vaila.mac.sh                                                        #
# Description: Uninstalls the vaila - Multimodal Toolbox from macOS, including the      #
#              Conda environment removal, deletion of program files from the user's     #
#              home directory, and removal of the application from /Applications.       #
#              Also refreshes the Launchpad to remove the icon.                         #
#                                                                                       #
# Usage:                                                                                #
#   1. Make the script executable:                                                      #
#      chmod +x uninstall_vaila.mac.sh                                                  #
#   2. Run the script with sudo:                                                        #
#      sudo ./uninstall_vaila.mac.sh                                                    #
#                                                                                       #
# Author: Prof. Dr. Paulo R. P. Santiago                                                #
# Date: September 17, 2024                                                              #
# Version: 1.5                                                                          #
# OS: macOS                                                                             #
#########################################################################################

echo "Starting uninstallation of vaila - Multimodal Toolbox on macOS..."

# Define paths
USER_HOME="$HOME"
VAILA_HOME="$USER_HOME/vaila"
APP_PATH="/Applications/vaila.app"
LOG_FILE="$USER_HOME/vaila_app.log"

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Cannot proceed with environment removal."
else
    # Remove the 'vaila' Conda environment if it exists
    if conda info --envs | grep -qw "vaila"; then
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

# Remove the vaila.app from /Applications
if [ -d "$APP_PATH" ]; then
    echo "Removing vaila app from /Applications..."
    rm -rf "$APP_PATH"
    if [ $? -eq 0 ]; then
        echo "vaila app removed successfully."
    else
        echo "Failed to remove vaila app."
    fi
else
    echo "vaila app not found in /Applications. Skipping removal."
fi

# Remove the log file if it exists
if [ -f "$LOG_FILE" ]; then
    echo "Removing vaila log file..."
    rm -f "$LOG_FILE"
    if [ $? -eq 0 ]; then
        echo "vaila log file removed successfully."
    else
        echo "Failed to remove vaila log file."
    fi
else
    echo "vaila log file not found. Skipping removal."
fi

# Refresh Launchpad to remove any cached icons
echo "Refreshing Launchpad..."
killall Dock
echo "Launchpad refreshed."

echo "Uninstallation completed. vaila has been removed from your system."
