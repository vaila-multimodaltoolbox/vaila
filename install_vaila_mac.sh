#!/bin/bash

#########################################################################################
#                                                                                       #
# Script: install_vaila_mac.sh                                                          #
# Description: Installs the vaila - Multimodal Toolbox on macOS, sets up the Conda      #
#              environment, copies program files to the user's home directory,          #
#              configures the macOS app icon, and creates a symlink in /Applications.   #
#                                                                                       #
# Usage:                                                                                #
#   1. Download the repository from GitHub manualmente ou usando git clone.             #
#   2. Make the script executable:                                                      #
#      chmod +x install_vaila_mac.sh                                                    #
#   3. Run the script from the root directory of the extracted repository:              #
#      ./install_vaila_mac.sh                                                           #
#                                                                                       #
# Notes:                                                                                #
#   - Ensure Conda is installed and accessible from the command line before running.    #
#                                                                                       #
# Author: Prof. Dr. Paulo R. P. Santiago                                                #
# Date: September 22, 2024                                                              #
# Version: 2.0                                                                          #
# OS: macOS                                                                             #
#########################################################################################

echo "Starting installation of vaila - Multimodal Toolbox on macOS..."

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Conda first."
    exit 1
fi

# Check if the "vaila" environment already exists
if conda info --envs | grep -q "^vaila"; then
    echo "Conda environment 'vaila' already exists. Updating..."
    conda env update -f yaml_for_conda_env/vaila_mac.yaml --prune
    if [ $? -eq 0 ]; then
        echo "'vaila' environment updated successfully."
    else
        echo "Failed to update 'vaila' environment."
        exit 1
    fi
else
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
USER_HOME="${HOME}"
VAILA_HOME="${USER_HOME}/vaila"
APP_PATH="${VAILA_HOME}/vaila.app"
ICON_PATH="$(pwd)/docs/images/vaila.icns"

# Copy the entire vaila program to the user's home directory, including the .git folder
echo "Copying vaila program to the user's home directory..."
mkdir -p "${VAILA_HOME}"
cp -Rfa "$(pwd)/." "${VAILA_HOME}/"

# Ensure the application directory exists in the user's home directory
echo "Configuring macOS app icon for vaila..."
if [ ! -d "${APP_PATH}" ]; then
    echo "Application not found at ${APP_PATH}, creating a new app directory..."
    mkdir -p "${APP_PATH}/Contents/MacOS"
    mkdir -p "${APP_PATH}/Contents/Resources"
fi

# Copy the .icns file to the app resources
if [ -f "${ICON_PATH}" ]; then
    cp "${ICON_PATH}" "${APP_PATH}/Contents/Resources/vaila.icns"
else
    echo "Icon file not found at ${ICON_PATH}. Please check the path."
    exit 1
fi

# Create the Info.plist file
cat <<EOF > "${APP_PATH}/Contents/Info.plist"
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
    <string>run_vaila.sh</string>
    <key>LSRequiresICloud</key>
    <false/>
    <key>LSMinimumSystemVersion</key>
    <string>10.10</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.utilities</string>
</dict>
</plist>
EOF

# Create the executable script for the app, using osascript to open in terminal
CONDA_BASE=$(conda info --base)
cat <<EOF > "${APP_PATH}/Contents/MacOS/run_vaila.sh"
#!/bin/zsh
osascript -e "tell application \"Terminal\" to do script \"source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate vaila && python3 ${HOME}/vaila/vaila.py\""
EOF

# Make the executable script runnable
chmod +x "${APP_PATH}/Contents/MacOS/run_vaila.sh"

# Create a symbolic link in /Applications
echo "Creating a symbolic link in /Applications to the app in the user's home directory..."
if [ -e "/Applications/vaila.app" ]; then
    sudo rm -rf "/Applications/vaila.app"
fi
sudo ln -s "${APP_PATH}" "/Applications/vaila.app"

# Ensure the symbolic link has the correct permissions
echo "Ensuring correct permissions for the application link..."
sudo chown -h "${USER}:admin" "/Applications/vaila.app"

# Ensure the application directory is owned by the user and has the correct permissions
echo "Ensuring correct ownership and permissions for the application..."
sudo chown -R "${USER}:admin" "${APP_PATH}"
chmod -R +x "${APP_PATH}"

# Check if Homebrew is installed, install it if not
echo "Checking if Homebrew is installed..."
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "Homebrew is already installed."
    brew update
fi

# Check if rsync is available (it should be built into macOS)
echo "Verifying rsync availability..."
if ! command -v rsync &> /dev/null; then
    echo "Installing rsync via Homebrew..."
    brew install rsync
else
    echo "rsync is already available on this system."
fi

# Enable SSH (Remote Login) service
echo "Configuring SSH (Remote Login) service..."
echo "You may be prompted for your password to enable SSH."
sudo systemsetup -setremotelogin on

# Remove ffmpeg from Conda environment if installed
echo "Removing ffmpeg installed via Conda..."
conda remove -n vaila ffmpeg -y

# Install ffmpeg via Homebrew
echo "Installing ffmpeg via Homebrew..."
brew install ffmpeg

# Set permissions for the Conda environment
echo "Setting permissions for Conda environment..."
CONDA_BASE=$(conda info --base)
VAILA_ENV_DIR="${CONDA_BASE}/envs/vaila"
if [ -d "$VAILA_ENV_DIR" ]; then
    chmod -R u+rwX "$VAILA_ENV_DIR"
    echo "Permissions set for Conda environment directory."
else
    echo "Conda environment directory not found at $VAILA_ENV_DIR."
fi

echo "vaila Launcher created and configured in /Applications as a symbolic link! Check the Applications folder."
echo "Installation and setup completed."
