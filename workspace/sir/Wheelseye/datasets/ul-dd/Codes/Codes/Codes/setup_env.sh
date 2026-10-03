#!/bin/bash

# Define the environment name
ENV_NAME="myenv"

# Check if the user provided a custom name
if [ "$1" ]; then
  ENV_NAME=$1
fi

echo "Creating virtual environment: $ENV_NAME"

# Create virtual environment
python3 -m venv $ENV_NAME

# Activate the environment
source $ENV_NAME/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies from requirements.txt
pip install -r requirements.txt

echo "Environment '$ENV_NAME' is set up and ready."
echo "To activate it, run: source $ENV_NAME/bin/activate"