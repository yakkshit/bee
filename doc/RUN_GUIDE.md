# 🚀 Run & Execution Guide

Step-by-step instructions for running the **Bee Analytics Studio** dashboard and CLI scripts.

---

## 1. Environment Setup

### Recommended Environment
- **Python**: Version 3.11 (configured in `.python-version` and Dockerfile)
- **Virtual Environment**: `.venv`

```bash
# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 2. Running the Web Application

```bash
streamlit run streamlit_app.py --server.port 8501
```

### URLs
- **Main Dashboard**: `http://localhost:8501`
- **Documentation Route**: `http://localhost:8501/doc` or `http://localhost:8501/?page=doc`

---

## 3. Running via CLI / Standalone Scripts

### Running Probit Regression Offline
```bash
python3 -c "
from modules.core.probit_engine import ProbitEngine
import numpy as np

days = np.arange(1, 16)
dead = np.array([0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])
engine = ProbitEngine(days, dead, total_bees=10)
print(engine.fit())
"
```

### Automated Batch CSV Auditor
```bash
python3 -c "
from modules.core.csv_auditor import audit_csv_files
results = audit_csv_files(['2026-30_export.csv'])
print(results)
"
```

