#!/bin/bash

# Set the required Python version
REQUIRED_PYTHON="python3.8"

# Check if the correct Python version is installed
if ! command -v $REQUIRED_PYTHON &> /dev/null
then
    echo "$REQUIRED_PYTHON not found. Please install it."
    exit 1
fi

# Check the Python version
PYTHON_VERSION=$($REQUIRED_PYTHON --version 2>&1 | awk '{print $2}')
if [[ "$PYTHON_VERSION" < "3.8" ]]
then
    echo "Python version is $PYTHON_VERSION, but 3.8 or higher is required."
    exit 1
fi

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    $REQUIRED_PYTHON -m pip install --upgrade pip
    $REQUIRED_PYTHON -m pip install -r requirements.txt
else
    echo "requirements.txt not found. Skipping dependency installation."
fi


# Run the Python script with parameters
SCRIPT_NAME="main.py"


if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Error: $SCRIPT_NAME not found!"
    exit 1
fi

echo "Running $SCRIPT_NAME"
$REQUIRED_PYTHON "$SCRIPT_NAME" 