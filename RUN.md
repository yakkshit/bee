# 🚀  Bee Analytics Studio — Installation & User Guide

Welcome to **Bee Analytics Studio**! This document provides complete, step-by-step instructions for installing Python, setting up dependencies, running the application, and utilizing each feature.

---

## 📋 Table of Contents
1. [System Prerequisites](#1-system-prerequisites)
2. [Step 1: Installing Python](#2-step-1-installing-python)
   - [macOS](#macos)
   - [Windows](#windows)
   - [Linux (Ubuntu / Debian / Fedora)](#linux-ubuntu--debian--fedora)
3. [Step 2: Downloading the Application](#3-step-2-downloading-the-application)
4. [Step 3: Setting Up a Virtual Environment](#4-step-3-setting-up-a-virtual-environment)
5. [Step 4: Installing Dependencies](#5-step-4-installing-dependencies)
6. [Step 5: Launching the Application](#6-step-5-launching-the-application)
7. [Step 6: User Guide — How to Use the Studio](#7-step-6-user-guide--how-to-use-the-studio)
   - [Blinded Experiment Mode & Legend Customization](#blinded-experiment-mode--legend-customization)
   - [Tab 1: Single CSV Analysis & Observation Sheet](#tab-1-single-csv-analysis--observation-sheet)
   - [Tab 2: Multi-CSV Probit Comparison](#tab-2-multi-csv-probit-comparison)
   - [Tab 3: Advanced Survival Statistics (Log-Rank & Cox)](#tab-3-advanced-survival-statistics-log-rank--cox)
   - [Tab 4: 🤖 AI Plot Studio (ChatGPT Code Runner)](#tab-4--ai-plot-studio-chatgpt-code-runner)
8. [Troubleshooting & Frequently Asked Questions](#8-troubleshooting--frequently-asked-questions)

---

## 1. System Prerequisites

- **Supported Operating Systems**: macOS (Apple Silicon or Intel), Windows 10/11, or Linux (Ubuntu 20.04+, Debian, Fedora, Arch).
- **Python Version**: Python **3.9, 3.10, or 3.11** (Python 3.11 is recommended).
- **Web Browser**: Any modern browser (Google Chrome, Firefox, Safari, or Microsoft Edge).

---

## 2. Step 1: Installing Python

If Python 3.9+ is not already installed on your machine, follow the instructions for your operating system:

### macOS

1. **Option A: Official Installer (Recommended for beginners)**:
   - Visit [python.org/downloads](https://www.python.org/downloads/).
   - Download the latest installer for macOS (e.g., Python 3.11).
   - Double-click the downloaded `.pkg` file and follow the standard installation wizard.

2. **Option B: Using Homebrew**:
   - Open **Terminal** and run:
     ```bash
     brew install python@3.11
     ```

3. **Verify installation**:
   - Open Terminal and run:
     ```bash
     python3 --version
     ```
   - You should see `Python 3.11.x` (or 3.9+/3.10+).

---

### Windows

1. **Download the Official Installer**:
   - Go to [python.org/downloads/windows](https://www.python.org/downloads/windows/).
   - Download the **Windows installer (64-bit)** for Python 3.11.

2. **Run Installer with PATH Enabled**:
   - Double-click the downloaded `.exe` installer.
   - ⚠️ **CRITICAL STEP**: On the very first installation window, check the box:
     **☑ Add python.exe to PATH**
   - Click **Install Now**.

3. **Verify installation**:
   - Open **Command Prompt** (press `Win + R`, type `cmd`, and press Enter).
   - Run:
     ```cmd
     python --version
     ```
   - You should see `Python 3.11.x`.

---

### Linux (Ubuntu / Debian / Fedora)

1. **Ubuntu / Debian**:
   ```bash
   sudo apt update
   sudo apt install -y python3 python3-pip python3-venv
   ```

2. **Fedora / RHEL**:
   ```bash
   sudo dnf install -y python3 python3-pip
   ```

3. **Verify installation**:
   ```bash
   python3 --version
   ```

---

## 3. Step 2: Downloading the Application

Clone the repository or download and extract the ZIP archive:

```bash
git clone https://github.com/your-username/bee-analytics-studio.git
cd bee-analytics-studio
```

---

## 4. Step 3: Setting Up a Virtual Environment

Creating an isolated Python virtual environment prevents library conflicts with your system.

### On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### On Windows:
- **Command Prompt (`cmd`)**:
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate.bat
  ```
- **PowerShell**:
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
  *(If PowerShell displays an execution policy error, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` and retry).*

Once activated, your terminal prompt will show `(.venv)`.

---

## 5. Step 4: Installing Dependencies

With your virtual environment active, install all required scientific libraries:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- `streamlit` (interactive web application engine)
- `pandas` & `numpy` (data manipulation and vectorized matrices)
- `scipy` (Finney Probit distribution and non-linear regression)
- `matplotlib` & `seaborn` (publication figure generation)
- `lifelines` (Kaplan-Meier, Mantel-Cox Log-Rank, and Cox Proportional Hazards)

---

## 6. Step 5: Launching the Application

### Method A: Direct Python Command (Universal across all platforms)
```bash
streamlit run streamlit_app.py
```

### Method B: Using Setup Script (macOS / Linux)
```bash
chmod +x run.sh
./run.sh
```

Your browser will automatically launch and open the studio at:
👉 **`http://localhost:8501`**

---

## 7. Step 6: User Guide — How to Use the Studio

### Blinded Experiment Mode & Legend Customization
Researchers often begin experiments blindly before the treatment identities are revealed.
1. When uploading CSV files without known treatments, the application sets the treatment to **`None / Blinded (Unknown)`**.
2. Under **"🎨 Customize Cohort Treatments, Colors, Markers & Legend Display"**, use the **Plot Legend Label Style** dropdown:
   - **`File Name Only (Blinded Study Mode)`**: Legends display only file names (e.g. `Batch_01.csv`).
   - **`Treatment Solution (Short Code)`**: Displays concise publication labels (e.g. `5% Acetone 2 µL`, `Sucrose 2 µL`).
   - **`Treatment Solution (Full Protocol Name)`**: Displays full names (e.g. `5 % acetone 2 microliter`).
   - **`File Name + Treatment Solution`**: Hybrid label (e.g. `Batch_01.csv (5% Acetone 2 µL)` or `Batch_01.csv (Blinded)`).
   - **`Custom Display Labels`**: Specify any custom label string per file.
3. Once the study is unblinded, simply select the actual treatment solution from the dropdown for each cohort card. All plot legends update immediately across all tabs!

---

### Tab 1: Single CSV Analysis & Observation Sheet
- **Interactive Grid**: Edit cell values in real time, add observation days (rows) or new bee columns.
- **Protocol Enforced**: Baseline Day 0 ($t=0$) is automatically excluded from lethal time estimation.
- **HTML Print-to-PDF**: Export a replica of the observation sheet with full study metadata.
- **Thematic Sub-Tabs**:
  1. *Mortality & Survival Models* (Logistic and Weibull fits)
  2. *Lethal Time (LTₓ) Laboratory* (Finney Probit $LT_{10}, LT_{50}, LT_{90}$)
  3. *Cohort & Individual Tracking* (Status matrices and daily distribution)
  4. *Environmental Dynamics* (Temperature and humidity correlation heatmap)
  5. *Biostatistical Models* (Probit residuals)
  6. *Publication Multi-Panel Figure* (Integrated 6-panel summary figure)

---

### Tab 2: Multi-CSV Probit Comparison
- Compare multiple cohorts simultaneously.
- Analyze Finney Probit linear regression lines, Abbott natural mortality corrections, and $LT_x$ comparative bar charts.
- **Multi-Cohort Sucrose Consumption**: Inspect daily intake ($\mu\text{L}$), cumulative feeding trajectories, and Kaplan-Meier time-to-anorexia curves.
- Download individual figures or package all plots into a unified **ZIP archive** at 300 DPI.

---

### Tab 3: Advanced Survival Statistics (Log-Rank & Cox)
- **Kaplan–Meier Curves**: Non-parametric step-down curves with Greenwood 95% confidence bands and censoring marks ($+$).
- **Mantel-Cox Log-Rank Test**: Calculates observed vs. expected deaths ($O$ vs $E$) under the null hypothesis with two-sided $\chi^2$ and $p$-values.
- **Cox Proportional Hazards Regression**: Quantifies hazard ratios ($HR$), standard errors, 95% confidence intervals, and generates publication Forest plots.

---

### Tab 4: 🤖 AI Plot Studio (ChatGPT Code Runner)
Designed specifically for researchers and professors who want custom figures without writing Python code manually:
1. **Step 1: Generate ChatGPT Prompt**:
   - Choose a plot idea (e.g. grouped boxplots, hazard rate curves, survival heatmaps) or type a request in plain English.
   - The studio automatically formats an optimized prompt containing the live dataset schema (`survival_df`, `df`, `summary_df`), column names, and bioassay protocol rules.
   - Copy the prompt and paste it into ChatGPT (GPT-4o / Claude).
2. **Step 2: Code Runner & Interactive Plot Canvas**:
   - Paste ChatGPT's Python code (or select one of the built-in quick-start templates).
   - Click **`▶️ Run Python Plot Code`**.
   - View your plot immediately in the browser and download it as a **300 DPI PNG** or vector **SVG**!

---

## 8. Troubleshooting & Frequently Asked Questions

### Port 8501 is already in use
If another Streamlit application is running on port 8501, run:
```bash
streamlit run streamlit_app.py --server.port 8502
```

### ModuleNotFoundError: No module named 'streamlit'
Your virtual environment is not activated. Run:
- **macOS / Linux**: `source .venv/bin/activate`
- **Windows**: `.venv\Scripts\activate.bat`
Then verify: `pip list`.

### Plot modifications feel slow
The studio uses on-screen rendering optimizations (140 DPI display) and data caching (`@st.cache_data`). If a large batch takes time, ensure you haven't opened multiple browser tabs of the same session.

### How do I update or reset the application?
```bash
git pull origin main
pip install -r requirements.txt
```

