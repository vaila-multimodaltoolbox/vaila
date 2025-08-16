#!/bin/bash

#########################################################################################
#                                                                                       #
# Script: fix_desktop_integration.sh                                                    #
# Description: Fixes desktop integration issues for vaila on different Linux desktop   #
#              environments (KDE Plasma, XFCE, GNOME, etc.) by properly registering     #
#              the application and refreshing the desktop database.                     #
#                                                                                       #
# Usage:                                                                                #
#   ./fix_desktop_integration.sh                                                        #
#                                                                                       #
# Author: Prof. Dr. Paulo R. P. Santiago                                                #
# Date: January 2025                                                                    #
# Version: 1.0.0                                                                        #
# OS: Ubuntu, Kubuntu, Linux Mint, Pop_OS!, Zorin OS, etc.                             #
#########################################################################################

echo "Fixing vaila desktop integration for different Linux desktop environments..."

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

# Create/update user desktop entry
echo "Creating/updating user desktop entry..."
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
EOF

# Create system-wide desktop entry for better compatibility
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
EOF

# Update desktop database
echo "Updating desktop database..."
if command -v update-desktop-database &> /dev/null; then
    update-desktop-database "$HOME/.local/share/applications"
    sudo update-desktop-database
fi

# Detect and handle different desktop environments
echo "Detecting desktop environment and applying specific fixes..."

# KDE Plasma
if command -v kbuildsycoca5 &> /dev/null; then
    echo "KDE Plasma 5 detected. Refreshing application menu..."
    kbuildsycoca5 --noincremental
elif command -v kbuildsycoca6 &> /dev/null; then
    echo "KDE Plasma 6 detected. Refreshing application menu..."
    kbuildsycoca6 --noincremental
elif command -v plasmashell &> /dev/null; then
    echo "KDE Plasma detected. Refreshing application menu..."
    # Try alternative methods for KDE
    if command -v kquitapp5 &> /dev/null; then
        kquitapp5 plasmashell && kstart5 plasmashell
    elif command -v kquitapp &> /dev/null; then
        kquitapp plasmashell && kstart plasmashell
    fi
fi

# XFCE
if command -v xfce4-appfinder &> /dev/null; then
    echo "XFCE detected. Refreshing application menu..."
    if command -v xfce4-panel &> /dev/null; then
        xfce4-panel --restart
    fi
    # Clear XFCE application cache
    rm -rf "$HOME/.cache/xfce4/desktop"
    rm -rf "$HOME/.cache/xfce4/panel"
fi

# GNOME
if command -v gnome-shell &> /dev/null; then
    echo "GNOME detected. Updating icon cache..."
    if command -v gtk-update-icon-cache &> /dev/null; then
        gtk-update-icon-cache -f -t "$HOME/.local/share/icons"
    fi
    # Refresh GNOME shell
    if command -v busctl &> /dev/null; then
        busctl --user call org.gnome.Shell /org/gnome/Shell org.gnome.Shell Eval s 'Meta.restart("Restarting…")'
    fi
fi

# Cinnamon
if command -v cinnamon-session &> /dev/null; then
    echo "Cinnamon detected. Refreshing application menu..."
    if command -v cinnamon --replace &> /dev/null; then
        cinnamon --replace &
    fi
fi

# MATE
if command -v mate-session &> /dev/null; then
    echo "MATE detected. Refreshing application menu..."
    if command -v mate-panel --replace &> /dev/null; then
        mate-panel --replace &
    fi
fi

# LXDE
if command -v lxsession &> /dev/null; then
    echo "LXDE detected. Refreshing application menu..."
    if command -v lxpanelctl &> /dev/null; then
        lxpanelctl restart
    fi
fi

# LXQt
if command -v lxqt-session &> /dev/null; then
    echo "LXQt detected. Refreshing application menu..."
    if command -v lxqt-panel &> /dev/null; then
        lxqt-panel --restart
    fi
fi

# Update icon cache for all environments
echo "Updating icon cache..."
if command -v gtk-update-icon-cache &> /dev/null; then
    gtk-update-icon-cache -f -t "$HOME/.local/share/icons" 2>/dev/null || true
    sudo gtk-update-icon-cache -f -t /usr/share/icons 2>/dev/null || true
fi

# Set proper permissions
echo "Setting proper permissions..."
chmod +x "$DESKTOP_ENTRY_PATH"
sudo chmod +x "$SYSTEM_DESKTOP_ENTRY_PATH"

echo ""
echo "=================================================================="
echo "Desktop integration fix completed!"
echo "Correção da integração com desktop concluída!"
echo "=================================================================="
echo ""
echo "The vaila application should now appear in your application menu."
echo "O aplicativo vaila deve agora aparecer no seu menu de aplicações."
echo ""
echo "If the application still doesn't appear, try:"
echo "Se o aplicativo ainda não aparecer, tente:"
echo "1. Log out and log back in"
echo "   1. Fazer logout e login novamente"
echo "2. Restart your desktop environment"
echo "   2. Reiniciar seu ambiente de desktop"
echo "3. Run this script again"
echo "   3. Executar este script novamente"
echo ""
echo "You can also run vaila directly from terminal:"
echo "Você também pode executar o vaila diretamente do terminal:"
echo "cd ~/vaila && ./run_vaila.sh"
echo ""
