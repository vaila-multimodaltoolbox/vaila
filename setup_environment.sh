#!/bin/bash

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda is not installed. Please install Conda first."
    exit
fi

# Create the environment based on the YAML file
conda env create -f vaila.yaml

# Activate the environment
if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "darwin"* ]]; then
    source activate vaila
elif [[ "$OSTYPE" == "msys" ]]; then
    conda activate vaila
else
    echo "Unsupported OS type: $OSTYPE"
    exit
fi

echo "Environment 'vailá' created and activated successfully."
