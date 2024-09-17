#!/bin/bash

# Install vailá - Multimodal Toolbox on Linux
echo "Starting installation of vailá - Multimodal Toolbox on Linux..."

# Check if Conda is installed
if ! command -v conda &> /dev/null
then
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
        echo "'vailá' environment created successfully on Linux."
    else
        echo "Failed to create 'vailá' environment."
        exit 1
    fi
fi

echo "Installation completed."

