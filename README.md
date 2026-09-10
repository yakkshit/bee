# Bee Observation Data Visualization

This is a Streamlit web application that visualizes bee observation data. It allows users to upload a CSV file or paste CSV data to automatically generate statistical models for generation of plots.

## Features
- **Single File Architecture**: All logic and data handling are bundled efficiently in `streamlit_app.py`.
- **Dynamic Observation Sheet Editor**: Effortlessly edit data in an interface modeled exactly after a standard bee observation sheet. You can dynamically add new observation days (rows) and new bees (columns).
- **Flexible Data Input**: Upload a pre-existing CSV or use the interactive grid.
- **PDF Export Tool**: Generate and download a perfectly formatted, printable digital replica of the observation sheet via HTML/CSS (Print to PDF).
- **Select Specific Plots**: Choose exactly which plots you want to view and generate.
- **Download High-Quality Plots**: Publication-grade PNG files (300 DPI) for your reports.

## How to run locally (Python)
1. Install dependencies: `pip install -r requirements.txt`
2. Run Streamlit: `streamlit run streamlit_app.py`

## How to run using the Setup Script (macOS/Linux)
You can use the provided bash script to automatically verify Python, create a virtual environment, install dependencies, and run the app.
1. Make the script executable: `chmod +x run.sh`
2. Run the script: `./run.sh`
3. The script will automatically open `http://localhost:8501` in your browser. you can edit the data or upload the data and also you can modify them in the browser it self. 
the hosted link is here : https://bee-any.streamlit.app/