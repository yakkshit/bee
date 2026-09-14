#!/bin/bash

echo "Do you want to run with Docker or Python? (docker/python)"
read -r RUN_MODE

if [ "$RUN_MODE" = "docker" ]; then
    echo "Running Streamlit application with Docker..."
    if ! command -v docker &>/dev/null; then
        echo "Docker is not installed. Please install Docker to continue."
        exit 1
    fi
    # Assuming standard docker build and run for Streamlit
    docker build -t bee-visualization-app .
    docker run -p 8501:8501 bee-visualization-app
else
    echo "Running Streamlit application with Python..."
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
    $PYTHON_CMD -m streamlit run streamlit_app.py
fi
