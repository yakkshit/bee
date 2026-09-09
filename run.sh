#!/bin/bash

# Check if python3 is installed
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "Python is not installed. Please install Python 3.8+ to continue."
    exit 1
fi

echo "Using Python executable: $PYTHON_CMD"

VENV_DIR=".venv_run"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv $VENV_DIR
fi

echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Running Streamlit application..."
python -m streamlit run streamlit_app.py
