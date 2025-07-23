#!/bin/bash  # (ou #!/bin/zsh para o mac)

# Author: Paulo Santiago
# Date: May 15, 2025
# Updated: July 23, 2025
# Version: 0.0.2
# OS: Ubuntu, Kubuntu, Linux Mint, Pop_OS!, Zorin OS, macOS, etc.

# Detect Conda installation (Anaconda or Miniconda)
if [ -d "$HOME/anaconda3" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -d "$HOME/miniconda3" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "Conda not found in anaconda3 or miniconda3."
    exit 1
fi

conda activate vaila
python3 ~/vaila/vaila.py
