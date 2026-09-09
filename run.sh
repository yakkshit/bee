#!/bin/bash

echo "Building Docker image..."
docker build -t bee-visualization-app .

echo "Running Docker container on port 8501..."
docker run -p 8501:8501 bee-visualization-app
