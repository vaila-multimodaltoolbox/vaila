#!/bin/bash

#########################################################################################
#                                                                                       #
# Script: install_vaila.mac.sh                                                          #
# Description: Installs the vaila - Multimodal Toolbox on macOS, including the Conda    #
#              environment setup, copying program files to the user's home directory,   #
#              configuring the macOS app icon, and creating a symlink in ~/Applications.#
#                                                                                       #
# Usage:                                                                                #
#   1. Make the script executable:                                                      #
#      chmod +x install_vaila.mac.sh                                                    #
#   2. Run the script:                                                                  #
#      ./install_vaila.mac.sh                                                           #
#                                                                                       #
# Notes:                                                                                #
#   - This script does not require sudo access.                                         #
#   - Ensure Conda is installed and accessible from the command line before running.    #
#                                                                                       #
# Author: Prof. Dr. Paulo R. P. Santiago                                                #
# Date: September 17, 2024                                                              #
# Version: 1.3                                                                          #
# OS: macOS                                                                             #
#########################################################################################

# Start of the installation process
echo "Starting installation of vaila - Multimodal Toolbox on macOS..."

# Display ASCII Art
echo " "
echo "                        ___             __"
echo "                    __ /\_ \           /\ \\"
echo "   __  __     __   /\_\\//\ \      __  \ \/"
echo "  /\ \/\ \  /'__\` \/\ \ \ \ \   /'__\` \/"
echo "  \ \ \_/ |/\ \L\.\_\ \ \ \_\ \_/\ \L\.\_"
echo "   \ \___/ \ \__/.\_\\ \_\/\____\ \__/.\_\\"
echo "    \/__/   \/__/\/_/ \/_/\/____/\/__/\/_/"
echo " "

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Conda first."
    exit 1
fi

# Check if the "vaila" environment already exists
if conda info --envs | grep -q "^vaila"; then
    echo "Conda environment 'vaila' already exists. Updating it..."
    # Update the existing environment
    conda env update -f yaml_for_conda_env/vaila_mac.yaml --prune
    if [ $? -eq 0 ]; then
        echo "'vaila' environment updated successfully."
    else
        echo "Failed to update 'vaila' environment."
        exit 1
    fi
else
    # Create the environment if it does not exist
    echo "Creating Conda environment from vaila_mac.yaml..."
    conda env create -f yaml_for_conda_env/vaila_mac.yaml
    if [ $? -eq 0 ]; then
        echo "'vaila' environment created successfully on macOS."
    else
        echo "Failed to create 'vaila' environment."
        exit 1
    fi
fi

# Define paths
USER_HOME="$HOME"
VAILA_HOME="$USER_HOME/vaila"
APP_PATH="$USER_HOME/Applications/vaila.app"
ICON_PATH="$(dirname "$0")/docs/images/vaila.icns"

# Copy the entire vaila program to the user's home directory
echo "Copying vaila program to the user's home directory..."
mkdir -p "$VAILA_HOME"
cp -R "$(dirname "$0")/"* "$VAILA_HOME/"

# Configure macOS app icon and executable
echo "Configuring macOS app icon for vaila..."

# Ensure the application directory exists
if [ ! -d "$APP_PATH" ]; then
    echo "Application not found at $APP_PATH, creating a new app directory..."
    mkdir -p "$APP_PATH/Contents/MacOS"
    mkdir -p "$APP_PATH/Contents/Resources"
fi

# Copy the .icns file to the app resources
if [ -f "$ICON_PATH" ]; then
    cp "$ICON_PATH" "$APP_PATH/Contents/Resources/vaila.icns"
else
    echo "Icon file not found at $ICON_PATH. Please check the path."
    exit 1
fi

# Create the Info.plist file
cat <<EOF > "$APP_PATH/Contents/Info.plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDisplayName</key>
    <string>vaila</string>
    <key>CFBundleName</key>
    <string>vaila</string>
    <key>CFBundleIdentifier</key>
    <string>com.example.vaila</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleIconFile</key>
    <string>vaila.icns</string>
    <key>CFBundleExecutable</key>
    <string>run_vaila</string>
    <key>LSRequiresICloud</key>
    <false/>
    <key>LSMinimumSystemVersion</key>
    <string>10.10</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.utilities</string>
</dict>
</plist>
EOF

# Create the executable script for the app
cat <<EOF > "$APP_PATH/Contents/MacOS/run_vaila"
#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate vaila
python3 "$VAILA_HOME/vaila.py"
EOF

# Make the executable script runnable
chmod +x "$APP_PATH/Contents/MacOS/run_vaila"
chmod -R +x "$VAILA_HOME"

echo "vaila Launcher created and configured in ~/Applications! Check the Applications folder."
echo "Installation and setup completed."
echo " "
echo "                   _         o   "
echo "                o  | |       /   "
echo "           __,     | |  __,      "
echo "     |  |_/  |  |  |/  /  |      "
echo "      \/  \_/|_/|_/|__/\_/|_/    "
echo " "
echo " "
