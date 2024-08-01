#!/bin/bash

# Navigate to the docs directory
cd docs

# Create directories
mkdir -p css js data tutorials images

# Create placeholder files
touch css/style.css
touch js/script.js
touch data/sample_data.csv
touch tutorials/tutorial1.html
touch tutorials/tutorial2.md
touch images/example1.png
touch images/example2.jpg
touch images/logo.png

# Inform the user that the structure has been created
echo "Directory structure created successfully."
