# Bee Observation Data Visualization

This is a Streamlit web application that visualizes bee observation data. It allows users to upload a CSV file or paste CSV data to automatically generate statistical models (logistic mortality, survival decay, Kaplan-Meier) and publication-grade plots.

## Features
- **Upload or Paste CSV Data**: Flexible data input methods.
- **Select specific plots**: Check which plots you want to view and generate.
- **Download High-Quality Plots**: PNG files generated at 300 DPI for publication.

## How to run locally (Python)
1. Install dependencies: `pip install -r requirements.txt`
2. Run Streamlit: `streamlit run streamlit_app.py`

## How to run with Docker
1. Make the script executable: `chmod +x run.sh`
2. Run the application: `./run.sh`
3. Open `http://localhost:8501` in your browser.
