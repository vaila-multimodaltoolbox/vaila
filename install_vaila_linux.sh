#!/bin/bash

#########################################################################################
#                                                                                       #
# Script: install_vaila.linux.sh                                                        #
# Description: Installs the vaila - Multimodal Toolbox on Ubuntu Linux, including the   #
#              Conda environment setup, copying program files to the user's home        #
#              directory, creating a desktop entry, and creating a symlink in /usr/local/bin. #
#                                                                                       #
# Usage:                                                                                #
#   1. Download the repository from GitHub manually and extract it.                     #
#   2. Make the script executable:                                                      #
#      chmod +x install_vaila.linux.sh                                                  #
#   3. Run the script from the root directory of the extracted repository:              #
#      ./install_vaila.linux.sh                                                         #
#                                                                                       #
# Notes:                                                                                #
#   - Ensure Conda is installed and accessible from the command line before running.    #
#                                                                                       #
# Author: Prof. Dr. Paulo R. P. Santiago                                                #
# Date: September 17, 2024                                                              #
# Version: 1.0                                                                          #
# OS: Ubuntu Linux                                                                      #
#########################################################################################

echo "Starting installation of vailá - Multimodal Toolbox on Linux..."

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Conda first."
    exit 1
fi

# Check if the "vaila" environment already exists
if conda info --envs | grep -q "^vaila"; then
    echo "Conda environment 'vaila' already exists. Updating it..."
    # Update the existing environment
    conda env update -f yaml_for_conda_env/vaila_linux.yaml --prune
    if [ $? -eq 0 ]; then
        echo "'vaila' environment updated successfully."
    else
        echo "Failed to update 'vaila' environment."
        exit 1
    fi
else
    # Create the environment if it does not exist
    echo "Creating Conda environment from vaila_linux.yaml..."
    conda env create -f yaml_for_conda_env/vaila_linux.yaml
    if [ $? -eq 0 ]; then
        echo "'vaila' environment created successfully on Linux."
    else
        echo "Failed to create 'vaila' environment."
        exit 1
    fi
fi

# Define paths
USER_HOME="$HOME"
VAILA_HOME="$USER_HOME/vaila"
APP_PATH="$VAILA_HOME/vaila"
DESKTOP_ENTRY_PATH="$HOME/.local/share/applications/vaila.desktop"

# Copy the entire vaila program to the user's home directory
echo "Copying vaila program to the user's home directory..."
mkdir -p "$VAILA_HOME"
cp -R "$(pwd)/"* "$VAILA_HOME/"

# Create a desktop entry for the application
echo "Creating a desktop entry for vailá..."
cat <<EOF > "$DESKTOP_ENTRY_PATH"
[Desktop Entry]
Version=1.0
Name=vailá
Comment=Multimodal Toolbox
Exec=bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate vaila && python3 $VAILA_HOME/vaila.py"
Icon=$VAILA_HOME/docs/images/vaila.png
Terminal=false
Type=Application
Categories=Utility;
EOF

# Make the desktop entry executable
chmod +x "$DESKTOP_ENTRY_PATH"

# Create a symbolic link to make the application globally accessible
echo "Creating a symbolic link in /usr/local/bin for global access..."
sudo ln -sf "$DESKTOP_ENTRY_PATH" /usr/local/bin/vaila

# Ensure the application directory is owned by the user and has the correct permissions
echo "Ensuring correct ownership and permissions for the application..."
chown -R "$USER:$USER" "$VAILA_HOME"
chmod -R +x "$VAILA_HOME"

# Remove ffmpeg from the Conda environment if installed
echo "Removing ffmpeg installed via Conda..."
conda remove -n vaila ffmpeg -y

# Install the system version of ffmpeg
echo "Installing ffmpeg from system repositories..."
sudo apt update
sudo apt install ffmpeg -y

# Install moviepy using pip
echo "Installing moviepy..."
pip install moviepy

echo "vailá Launcher created and available in the Applications menu!"
echo "Installation and setup completed."
echo " "
