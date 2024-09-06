#!/bin/bash

# Install vailá - Multimodal Toolbox on Linux
echo "Starting installation of vailá - Multimodal Toolbox on Linux..."

# Check if Conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda is not installed. Please install Conda first."
    exit 1
fi

# Check if the "vaila" environment already exists and remove it if it does
if conda info --envs | grep -q "^vaila"; then
    echo "Conda environment 'vaila' already exists. Removing it..."
    conda remove -n vaila --all -y
    if [ $? -eq 0 ]; then
        echo "Existing 'vaila' environment removed successfully."
    else
        echo "Failed to remove existing 'vaila' environment."
        exit 1
    fi
fi

# Navigate to the directory containing the YAML file
cd "$(dirname "$0")"

# Create the Conda environment
echo "Creating Conda environment from vaila_linux.yaml..."
conda env create -f yaml_for_conda_env/vaila_linux.yaml

# Check if the environment creation was successful
if [ $? -eq 0 ]; then
    echo "vailá environment created successfully on Linux."
else
    echo "Failed to create vailá environment."
    exit 1
fi

echo "Installation completed."

