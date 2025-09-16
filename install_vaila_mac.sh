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
# Updated Date: 11 September 2025                                                            #
# Version: 0.0.11                                                                        #
# OS: macOS                                                                             #
#########################################################################################

echo "Starting installation of vaila - Multimodal Toolbox on macOS..."

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Conda first."
    exit 1
fi

# Check if the "vaila" environment already exists and ask user for installation type
echo "Checking for existing 'vaila' environment..."
if conda info --envs | grep -q "^vaila"; then
    echo ""
    echo "============================================================"
    echo "vaila environment already exists!"
    echo "============================================================"
    echo ""
    echo "Choose installation type:"
    echo "1. UPDATE - Keep existing environment and update vaila files only"
    echo "   (Preserves NVIDIA CUDA installations and other custom packages)"
    echo "2. RESET - Remove existing environment and create fresh installation"
    echo "   (Will require reinstalling NVIDIA CUDA and other custom packages)"
    echo ""
    
    while true; do
        read -p "Enter your choice (1 for UPDATE, 2 for RESET): " choice
        case $choice in
            1)
                echo ""
                echo "Selected: UPDATE - Keeping existing environment"
                echo "Updating existing 'vaila' environment..."
                conda env update -f yaml_for_conda_env/vaila_mac.yaml --prune
                if [ $? -eq 0 ]; then
                    echo "'vaila' environment updated successfully."
                else
                    echo "Failed to update 'vaila' environment."
                    exit 1
                fi
                break
                ;;
            2)
                echo ""
                echo "Selected: RESET - Creating fresh environment"
                echo "Removing old 'vaila' environment..."
                conda env remove -n vaila -y
                if [ $? -eq 0 ]; then
                    echo "Old 'vaila' environment removed successfully."
                else
                    echo "Warning: Could not remove old environment. Continuing anyway."
                fi
                
                # Clean conda cache
                echo "Cleaning conda cache..."
                conda clean --all -y
                
                # Create the environment
                echo "Creating Conda environment from vaila_mac.yaml..."
                conda env create -f yaml_for_conda_env/vaila_mac.yaml
                if [ $? -eq 0 ]; then
                    echo "'vaila' environment created successfully on macOS."
                else
                    echo "Failed to create 'vaila' environment."
                    exit 1
                fi
                break
                ;;
            *)
                echo "Invalid choice. Please enter 1 or 2."
                ;;
        esac
    done
else
    echo "'vaila' environment does not exist. Creating new environment..."
    
    # Clean conda cache
    echo "Cleaning conda cache..."
    conda clean --all -y
    
    # Create the environment
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

# Clean destination directory and copy the entire vaila program to the user's home directory, including the .git folder
echo "Cleaning destination directory and copying vaila program to the user's home directory..."
if [ -d "${VAILA_HOME}" ]; then
    echo "Removing existing files from destination directory..."
    rm -rf "${VAILA_HOME}"/*
fi
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

# --------- AnyLabeling Download Information ---------
echo ""
echo "=================================================================="
echo "IMPORTANT INFORMATION FOR YOLO TRAINING / INFORMAÇÃO IMPORTANTE PARA TREINO YOLO"
echo "=================================================================="
echo ""
echo "To use YOLO training resources in vaila, you need AnyLabeling."
echo "Para usar recursos de treino YOLO no vaila, você precisa do AnyLabeling."
echo ""
echo "AnyLabeling is a free tool for training data annotation."
echo "O AnyLabeling é uma ferramenta gratuita para anotação de dados de treino."
echo ""
echo "Opening AnyLabeling download page in your default browser..."
echo "Abrindo página de download do AnyLabeling no seu navegador padrão..."
echo ""

# Open AnyLabeling download page in default browser
if command -v open &> /dev/null; then
    open "https://github.com/vietanhdev/anylabeling/releases"
    echo "AnyLabeling download page opened in your browser."
    echo "Página do AnyLabeling aberta no navegador."
else
    echo "Could not automatically open browser. Please visit manually:"
    echo "Não foi possível abrir o navegador automaticamente. Por favor, acesse manualmente:"
    echo "https://github.com/vietanhdev/anylabeling/releases"
fi

echo ""
echo "INSTRUCTIONS FOR macOS / INSTRUÇÕES PARA macOS:"
echo "1. Download the appropriate AnyLabeling-Folder.zip (CPU) or AnyLabeling-Folder-GPU.zip (GPU)"
echo "   1. Baixe o AnyLabeling-Folder.zip apropriado (CPU) ou AnyLabeling-Folder-GPU.zip (GPU)"
echo "2. Extract the downloaded ZIP file:"
echo "   2. Extraia o arquivo ZIP baixado:"
echo "   unzip AnyLabeling-Folder.zip"
echo "3. Navigate to the extracted folder:"
echo "   3. Navegue até a pasta extraída:"
echo "   cd AnyLabeling-Folder"
echo "4. Run the application:"
echo "   4. Execute a aplicação:"
echo "   ./anylabeling"
echo "5. Use AnyLabeling to create training annotations"
echo "   5. Use o AnyLabeling para criar anotações de treino"
echo "6. Import the annotations into vaila to train YOLO networks"
echo "   6. Importe as anotações no vaila para treinar redes YOLO"
echo ""
echo "Note: The macOS build is provided as a directory structure rather than a bundled .app file."
echo "Nota: A versão macOS é fornecida como uma estrutura de diretório em vez de um arquivo .app empacotado."
echo "This approach offers better compatibility across different macOS versions."
echo "Esta abordagem oferece melhor compatibilidade entre diferentes versões do macOS."
echo ""

echo "=================================================================="
echo "vaila installation completed successfully!"
echo "Instalação do vaila concluída com sucesso!"
echo "=================================================================="
