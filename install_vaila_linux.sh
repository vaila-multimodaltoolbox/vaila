#!/bin/bash

# Install vailá - Multimodal Toolbox on Linux
echo "Starting installation of vailá - Multimodal Toolbox on Linux..."

# Check if Conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda is not installed. Please install Conda first."
    exit 1
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
