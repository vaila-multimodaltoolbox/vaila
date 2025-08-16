#!/bin/bash

#########################################################################################
#                                                                                       #
# Script: fix_kde_integration.sh                                                        #
# Description: Specific fix for KDE Plasma desktop integration issues with vaila.       #
#              This script addresses common problems with KDE application menu          #
#              registration and icon display.                                           #
#                                                                                       #
# Usage:                                                                                #
#   ./fix_kde_integration.sh                                                            #
#                                                                                       #
# Author: Prof. Dr. Paulo R. P. Santiago                                                #
# Date: January 2025                                                                    #
# Version: 1.0.0                                                                        #
# OS: Kubuntu, KDE Neon, openSUSE KDE, etc.                                            #
#########################################################################################

echo "Fixing vaila integration specifically for KDE Plasma..."

# Define paths
USER_HOME="$HOME"
VAILA_HOME="$USER_HOME/vaila"
DESKTOP_ENTRY_PATH="$HOME/.local/share/applications/vaila.desktop"
SYSTEM_DESKTOP_ENTRY_PATH="/usr/share/applications/vaila.desktop"

# Check if vaila is installed
if [ ! -d "$VAILA_HOME" ]; then
    echo "Error: vaila is not installed in $VAILA_HOME"
    echo "Please run install_vaila_linux.sh first."
    exit 1
fi

# Check if we're running KDE
if ! command -v plasmashell &> /dev/null; then
    echo "Warning: KDE Plasma not detected. This script is designed for KDE environments."
    echo "Continuing anyway..."
fi

# Get Conda base path
if [ -d "$HOME/anaconda3" ]; then
    CONDA_BASE="$HOME/anaconda3"
elif [ -d "$HOME/miniconda3" ]; then
    CONDA_BASE="$HOME/miniconda3"
else
    echo "Error: Conda not found. Please install Conda first."
    exit 1
fi

# Create the run script if it doesn't exist
RUN_SCRIPT="$VAILA_HOME/run_vaila.sh"
if [ ! -f "$RUN_SCRIPT" ]; then
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
    chmod +x "$RUN_SCRIPT"
fi

# Create optimized desktop entry for KDE
echo "Creating KDE-optimized desktop entry..."
mkdir -p "$(dirname "$DESKTOP_ENTRY_PATH")"
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
X-KDE-StartupNotify=true
X-KDE-SubstituteUID=false
X-KDE-Username=
EOF

# Create system-wide desktop entry
echo "Creating system-wide desktop entry..."
sudo tee "$SYSTEM_DESKTOP_ENTRY_PATH" > /dev/null <<EOF
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
X-KDE-StartupNotify=true
X-KDE-SubstituteUID=false
X-KDE-Username=
EOF

# Update desktop database
echo "Updating desktop database..."
if command -v update-desktop-database &> /dev/null; then
    update-desktop-database "$HOME/.local/share/applications"
    sudo update-desktop-database
fi

# KDE-specific fixes
echo "Applying KDE-specific fixes..."

# Clear KDE application cache
echo "Clearing KDE application cache..."
rm -rf "$HOME/.cache/plasmashell"
rm -rf "$HOME/.cache/plasma-svgelements"
rm -rf "$HOME/.cache/plasma_engine_plasma"
rm -rf "$HOME/.cache/kioexec"
rm -rf "$HOME/.cache/kbuildsycoca5*"
rm -rf "$HOME/.cache/kbuildsycoca6*"

# Refresh KDE application menu
echo "Refreshing KDE application menu..."
if command -v kbuildsycoca5 &> /dev/null; then
    echo "Using kbuildsycoca5..."
    kbuildsycoca5 --noincremental
elif command -v kbuildsycoca6 &> /dev/null; then
    echo "Using kbuildsycoca6..."
    kbuildsycoca6 --noincremental
else
    echo "kbuildsycoca not found, trying alternative methods..."
fi

# Restart Plasma shell if possible
echo "Attempting to restart Plasma shell..."
if command -v kquitapp5 &> /dev/null; then
    echo "Restarting Plasma shell with kquitapp5..."
    kquitapp5 plasmashell && kstart5 plasmashell
elif command -v kquitapp &> /dev/null; then
    echo "Restarting Plasma shell with kquitapp..."
    kquitapp plasmashell && kstart plasmashell
else
    echo "kquitapp not found. You may need to restart Plasma manually."
fi

# Update icon cache
echo "Updating icon cache..."
if command -v gtk-update-icon-cache &> /dev/null; then
    gtk-update-icon-cache -f -t "$HOME/.local/share/icons" 2>/dev/null || true
    sudo gtk-update-icon-cache -f -t /usr/share/icons 2>/dev/null || true
fi

# Set proper permissions
echo "Setting proper permissions..."
chmod +x "$DESKTOP_ENTRY_PATH"
sudo chmod +x "$SYSTEM_DESKTOP_ENTRY_PATH"

# Create a symbolic link to ensure the icon is found
echo "Creating symbolic link for icon..."
mkdir -p "$HOME/.local/share/icons/hicolor/256x256/apps"
ln -sf "$VAILA_HOME/vaila/images/vaila_ico.png" "$HOME/.local/share/icons/hicolor/256x256/apps/vaila.png"

# Alternative: Copy icon to system directory
echo "Copying icon to system directory..."
sudo mkdir -p "/usr/share/icons/hicolor/256x256/apps"
sudo cp "$VAILA_HOME/vaila/images/vaila_ico.png" "/usr/share/icons/hicolor/256x256/apps/vaila.png"

# Update icon cache again
echo "Updating icon cache after copying..."
if command -v gtk-update-icon-cache &> /dev/null; then
    gtk-update-icon-cache -f -t "$HOME/.local/share/icons" 2>/dev/null || true
    sudo gtk-update-icon-cache -f -t /usr/share/icons 2>/dev/null || true
fi

echo ""
echo "=================================================================="
echo "KDE integration fix completed!"
echo "Correção da integração com KDE concluída!"
echo "=================================================================="
echo ""
echo "The vaila application should now appear in your KDE application menu."
echo "O aplicativo vaila deve agora aparecer no menu de aplicações do KDE."
echo ""
echo "If the application still doesn't appear, try:"
echo "Se o aplicativo ainda não aparecer, tente:"
echo "1. Log out and log back in"
echo "   1. Fazer logout e login novamente"
echo "2. Restart Plasma: kquitapp5 plasmashell && kstart5 plasmashell"
echo "   2. Reiniciar o Plasma: kquitapp5 plasmashell && kstart5 plasmashell"
echo "3. Run this script again"
echo "   3. Executar este script novamente"
echo ""
echo "You can also run vaila directly from terminal:"
echo "Você também pode executar o vaila diretamente do terminal:"
echo "cd ~/vaila && ./run_vaila.sh"
echo ""
echo "Or search for 'vaila' in the KDE application launcher (Alt+F2)"
echo "Ou procure por 'vaila' no lançador de aplicações do KDE (Alt+F2)"
echo ""
