#!/bin/bash

#########################################################################################
#                                                                                       #
# Script: install_vaila.linux.sh                                                        #
# Description: Installs the vaila - Multimodal Toolbox on Ubuntu Linux, including the   #
#              Conda environment setup, copying program files to the user's home        #
#              directory, creating a desktop entry, and setting up the application.     #
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
# Updated Date: 16 August 2025                                                            #
# Version: 0.0.10                                                                        #
# OS: Ubuntu, Kubuntu, Linux Mint, Pop_OS!, Zorin OS, etc.                              #
#########################################################################################

echo "Starting installation of vaila - Multimodal Toolbox on Linux..."

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Conda first."
    exit 1
fi

# Get Conda base path
CONDA_BASE=$(conda info --base)

# Check for missing dependencies
echo "Verifying system dependencies..."
for pkg in python3 pip git curl wget ffmpeg rsync pkg-config libcairo2-dev python3-dev; do
    if ! dpkg -l | grep -q " $pkg "; then
        echo "Installing $pkg..."
        sudo apt install -y $pkg
    fi
done

# Check if the "vaila" environment already exists
if conda env list | grep -q "^vaila"; then
    echo "Conda environment 'vaila' already exists. Updating it..."
    # Update the existing environment
    conda env update -n vaila -f yaml_for_conda_env/vaila_linux.yaml --prune
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

# Check for NVIDIA GPU and install PyTorch with CUDA if available
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Installing PyTorch with CUDA support..."
    conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia -n vaila
    if [ $? -eq 0 ]; then
        echo "PyTorch with CUDA support installed successfully."
    else
        echo "Failed to install PyTorch with CUDA support."
    fi
else
    echo "No NVIDIA GPU detected. Skipping PyTorch with CUDA installation."
fi

# Define paths
USER_HOME="$HOME"
VAILA_HOME="$USER_HOME/vaila"
DESKTOP_ENTRY_PATH="$HOME/.local/share/applications/vaila.desktop"

# Copy the entire vaila program to the user's home directory using cp -Rfa
echo "Copying vaila program to the user's home directory..."
mkdir -p "$VAILA_HOME"
cp -Rfa "$(pwd)/." "$VAILA_HOME/"

# Create a run_vaila.sh script
RUN_SCRIPT="$VAILA_HOME/run_vaila.sh"
echo "Creating run_vaila.sh script..."
cat <<EOF > "$RUN_SCRIPT"
#!/bin/bash
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate vaila
python3 "$VAILA_HOME/vaila.py"
# Keep terminal open after execution
echo
echo "Program finished. Press Enter to close this window..."
read
EOF

# Make the run_vaila.sh script executable
chmod +x "$RUN_SCRIPT"

# Create a desktop entry for the application
echo "Creating a desktop entry for vaila..."
cat <<EOF > "$DESKTOP_ENTRY_PATH"
[Desktop Entry]
Version=1.0
Name=vaila
GenericName=Multimodal Toolbox
Comment=Multimodal Toolbox for Biomechanics and Motion Analysis
Exec=$RUN_SCRIPT
Icon=$VAILA_HOME/vaila/images/vaila_ico.png
Terminal=true
Type=Application
Categories=Science;Education;Utility;
Keywords=biomechanics;motion;analysis;multimodal;
StartupNotify=true
StartupWMClass=vaila
EOF

# Update desktop database for all desktop environments
echo "Updating desktop database..."
if command -v update-desktop-database &> /dev/null; then
    update-desktop-database "$HOME/.local/share/applications"
fi

# For KDE Plasma, also create a .desktop file in the system applications directory
if [ -d "/usr/share/applications" ]; then
    echo "Creating system-wide desktop entry for KDE compatibility..."
    sudo tee "/usr/share/applications/vaila.desktop" > /dev/null <<EOF
[Desktop Entry]
Version=1.0
Name=vaila
GenericName=Multimodal Toolbox
Comment=Multimodal Toolbox for Biomechanics and Motion Analysis
Exec=$RUN_SCRIPT
Icon=$VAILA_HOME/vaila/images/vaila_ico.png
Terminal=true
Type=Application
Categories=Science;Education;Utility;
Keywords=biomechanics;motion;analysis;multimodal;
StartupNotify=true
StartupWMClass=vaila
EOF
    sudo update-desktop-database
fi

# For XFCE, ensure the desktop entry is properly registered
if command -v xfce4-appfinder &> /dev/null; then
    echo "XFCE detected. Ensuring desktop entry is properly registered..."
    # XFCE may need a refresh of the application menu
    if command -v xfce4-panel &> /dev/null; then
        echo "Refreshing XFCE panel..."
        xfce4-panel --restart
    fi
fi

# For KDE Plasma, refresh the application menu
if command -v kbuildsycoca5 &> /dev/null; then
    echo "KDE Plasma detected. Refreshing application menu..."
    kbuildsycoca5 --noincremental
elif command -v kbuildsycoca6 &> /dev/null; then
    echo "KDE Plasma 6 detected. Refreshing application menu..."
    kbuildsycoca6 --noincremental
fi

# For GNOME, refresh the application menu
if command -v gtk-update-icon-cache &> /dev/null; then
    echo "GNOME detected. Updating icon cache..."
    gtk-update-icon-cache -f -t "$HOME/.local/share/icons"
fi

# Ensure the application directory is owned by the user and has the correct permissions
echo "Ensuring correct ownership and permissions for the application..."
chown -R "$USER:$USER" "$VAILA_HOME"
chmod -R u+rwX "$VAILA_HOME"
chmod +x "$RUN_SCRIPT"

# Remove ffmpeg from the Conda environment if installed
echo "Removing ffmpeg installed via Conda..."
conda remove -n vaila ffmpeg -y

# Install the system version of ffmpeg
echo "Installing ffmpeg from system repositories..."
sudo apt update
sudo apt install ffmpeg -y

# Install rsync
echo "Installing rsync..."
sudo apt install rsync -y

# Install and configure SSH
echo "Installing and configuring OpenSSH Server..."
sudo apt install openssh-server -y

# Check if SSH server is running and enable it
echo "Ensuring SSH service is enabled and running..."
sudo systemctl enable ssh
sudo systemctl start ssh

# Configure SSH for better security (optional but recommended)
echo "Configuring SSH security settings..."
sudo sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin no/' /etc/ssh/sshd_config
sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config
sudo systemctl restart ssh

# Configure firewall to allow SSH connections
echo "Configuring firewall for SSH..."
if command -v ufw &> /dev/null; then
    sudo ufw allow ssh
    sudo ufw status
fi

# Activate the Conda environment
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate vaila

# Install Cairo dependencies and pycairo
echo "Installing Cairo dependencies..."
sudo apt install libcairo2-dev pkg-config python3-dev -y

# First try normal installation of pycairo
echo "Trying normal installation of pycairo..."
if pip install pycairo; then
    echo "Normal pycairo installation succeeded."
else
    echo "Normal pycairo installation failed. Trying with force-reinstall option..."
    pip install --force-reinstall --no-cache-dir pycairo
    
    if [ $? -eq 0 ]; then
        echo "Forced pycairo installation succeeded."
    else
        echo "Warning: pycairo installation failed. This may cause issues with the application."
    fi
fi

# Install moviepy using pip
echo "Installing moviepy..."
pip install moviepy

# Grant permissions to the Conda environment directory
echo "Setting permissions for Conda environment..."
VAILA_ENV_DIR="${CONDA_BASE}/envs/vaila"
if [ -d "$VAILA_ENV_DIR" ]; then
    chmod -R u+rwX "$VAILA_ENV_DIR"
    echo "Permissions set for Conda environment directory."
else
    echo "Conda environment directory not found at $VAILA_ENV_DIR."
fi

echo "vaila Launcher created and available in the Applications menu!"
echo "Installation and setup completed."
echo " "

# Verify and install x-terminal-emulator if necessary
if ! command -v x-terminal-emulator &> /dev/null; then
    echo "Installing terminal utility..."
    sudo apt install -y x-terminal-emulator
fi

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
if command -v xdg-open &> /dev/null; then
    xdg-open "https://github.com/vietanhdev/anylabeling/releases"
    echo "AnyLabeling download page opened in your browser."
    echo "Página do AnyLabeling aberta no navegador."
elif command -v gnome-open &> /dev/null; then
    gnome-open "https://github.com/vietanhdev/anylabeling/releases"
    echo "AnyLabeling download page opened in your browser."
    echo "Página do AnyLabeling aberta no navegador."
else
    echo "Could not automatically open browser. Please visit manually:"
    echo "Não foi possível abrir o navegador automaticamente. Por favor, acesse manualmente:"
    echo "https://github.com/vietanhdev/anylabeling/releases"
fi

echo ""
echo "INSTRUCTIONS FOR LINUX / INSTRUÇÕES PARA LINUX:"
echo "1. Download the latest AnyLabeling for Linux"
echo "   1. Baixe a versão mais recente do AnyLabeling para Linux"
echo "2. Extract the downloaded file"
echo "   2. Extraia o arquivo baixado"
echo "3. Make the anylabeling binary executable:"
echo "   3. Torne o binário anylabeling executável:"
echo "   sudo chmod +x anylabeling"
echo "4. Run AnyLabeling: ./anylabeling"
echo "   4. Execute o AnyLabeling: ./anylabeling"
echo "5. Use AnyLabeling to create training annotations"
echo "   5. Use o AnyLabeling para criar anotações de treino"
echo "6. Import the annotations into vaila to train YOLO networks"
echo "   6. Importe as anotações no vaila para treinar redes YOLO"
echo ""

echo "=================================================================="
echo "vaila installation completed successfully!"
echo "Instalação do vaila concluída com sucesso!"
echo "=================================================================="
echo ""
echo "If the application doesn't appear in your application menu, try:"
echo "Se o aplicativo não aparecer no menu de aplicações, tente:"
echo "1. Log out and log back in"
echo "   1. Fazer logout e login novamente"
echo "2. Restart your desktop environment"
echo "   2. Reiniciar seu ambiente de desktop"
echo "3. Run the installation script again"
echo "   3. Executar o script de instalação novamente"
echo ""
