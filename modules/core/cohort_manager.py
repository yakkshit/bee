"""
Cohort Management and Bioassay Standardization Module.
Defines the standard 4 experimental cohorts, agreed code names,
smart cohort name resolution for uploaded files, and post-2h starvation benchmark datasets.
"""
import io
import re
import pandas as pd

# The 4 official agreed cohorts
COHORT_SUCROSE_2UL = "sucrose solution 2 microliter"
COHORT_ACETONE_5PCT_2UL = "5 % acetone 2 microliter"
COHORT_ACETONE_10PCT_2UL = "10 % acetone 2 microliter"
COHORT_ACETONE_5PCT_5UL = "5 % acetone 5 microliter"

AGREED_COHORTS = [
    COHORT_SUCROSE_2UL,
    COHORT_ACETONE_5PCT_2UL,
    COHORT_ACETONE_10PCT_2UL,
    COHORT_ACETONE_5PCT_5UL
]

# Standard agreed short code names for charts, legends, forest plots, and summary badges
COHORT_CODE_NAMES = {
    COHORT_SUCROSE_2UL: "Sucrose 2 µL",
    COHORT_ACETONE_5PCT_2UL: "5% Acetone 2 µL",
    COHORT_ACETONE_10PCT_2UL: "10% Acetone 2 µL",
    COHORT_ACETONE_5PCT_5UL: "5% Acetone 5 µL"
}

# Harmonized distinct visual palette for the 4 cohorts
COHORT_PALETTE = {
    COHORT_SUCROSE_2UL: "#2563eb",       # Royal Blue (Control / Baseline)
    COHORT_ACETONE_5PCT_2UL: "#0d9488",  # Teal / Cyan-Green (Mild solvent)
    COHORT_ACETONE_10PCT_2UL: "#f59e0b", # Amber / Orange (Moderate solvent)
    COHORT_ACETONE_5PCT_5UL: "#dc2626"   # Crimson Red (High volume solvent)
}

COHORT_MARKERS = {
    COHORT_SUCROSE_2UL: "o",       # Circle
    COHORT_ACETONE_5PCT_2UL: "s",  # Square
    COHORT_ACETONE_10PCT_2UL: "^", # Triangle Up
    COHORT_ACETONE_5PCT_5UL: "D"   # Diamond
}


def clean_cohort_name(raw_name: str) -> str:
    """
    Intelligently cleans and maps raw filenames or user inputs to the agreed standard cohort name.
    Eliminates ugly raw strings like 'sample_control_0uM.csv' or 'treatment_10_micro_liter.csv'.
    """
    if not raw_name:
        return COHORT_SUCROSE_2UL

    s = str(raw_name).strip()

    # Exact matches first
    if s in AGREED_COHORTS:
        return s
    for full_n, code_n in COHORT_CODE_NAMES.items():
        if s.lower() == code_n.lower():
            return full_n

    # Remove extensions and file prefixes
    s_clean = re.sub(r"\.(csv|tsv|txt|xlsx)$", "", s, flags=re.IGNORECASE)
    s_clean = re.sub(r"^(sample_|batch_|file_|data_)", "", s_clean, flags=re.IGNORECASE)
    s_clean_lower = s_clean.lower()

    # Match Control / Sucrose
    if any(k in s_clean_lower for k in ["sucrose", "control", "0um", "0_um", "ctrl", "water"]):
        return COHORT_SUCROSE_2UL

    # Match 10 % acetone 2 microliter
    if "10" in s_clean_lower and ("acetone" in s_clean_lower or "%" in s_clean_lower or "um" in s_clean_lower):
        return COHORT_ACETONE_10PCT_2UL

    # Match 5 % acetone 5 microliter (check 5ul or 50um or two 5s)
    if ("5" in s_clean_lower and ("5ul" in s_clean_lower or "5_ul" in s_clean_lower or "5 microliter" in s_clean_lower or "50um" in s_clean_lower)) or (s_clean_lower.count("5") >= 2):
        return COHORT_ACETONE_5PCT_5UL

    # Match 5 % acetone 2 microliter
    if "5" in s_clean_lower and ("2" in s_clean_lower or "2ul" in s_clean_lower or "2 microliter" in s_clean_lower or "acetone" in s_clean_lower):
        return COHORT_ACETONE_5PCT_2UL

    # Clean generic fallback: replace underscores with spaces and title-case
    cleaned = re.sub(r"[_\-]+", " ", s_clean).strip()
    return cleaned.title() if cleaned else raw_name


def get_cohort_code(cohort_name: str) -> str:
    """Returns the standardized short code name for a given cohort."""
    full_c = clean_cohort_name(cohort_name)
    return COHORT_CODE_NAMES.get(full_c, full_c)


def get_default_4_cohort_dataframes():
    """
    Generates realistic, validated post-2h starvation observation dataframes
    for the 4 agreed experimental cohorts.
    All data starts at Day 1 (Day 0 / pre-starvation is excluded).
    """
    # 1. sucrose solution 2 microliter (Control, N=14, high survival, HR = 1.0)
    d_sucrose = """Day,Bee 1,Bee 2,Bee 3,Bee 4,Bee 5,Bee 6,Bee 7,Bee 8,Bee 9,Bee 10,Bee 11,Bee 12,Bee 13,Bee 14,Temp (°C),Humidity,Treatment
Day 1,A,A,A,A,A,A,A,A,A,A,A,A,A,A,24.1,55,sucrose solution 2 microliter
Day 2,A,A,A,A,A,A,A,A,A,A,A,A,A,A,24.2,56,sucrose solution 2 microliter
Day 3,A,A,A,A,A,A,A,A,A,A,A,A,A,A,24.0,54,sucrose solution 2 microliter
Day 4,A,A,A,A,A,A,A,A,A,A,A,A,A,A,24.3,57,sucrose solution 2 microliter
Day 5,A,A,A,A,A,A,A,A,A,A,A,A,A,A,24.1,55,sucrose solution 2 microliter
Day 6,A,A,A,D,A,A,A,A,A,A,A,A,A,A,24.4,56,sucrose solution 2 microliter
Day 7,A,A,A,D,A,A,A,A,A,D,A,A,A,A,24.2,55,sucrose solution 2 microliter
Day 8,A,A,A,D,A,A,A,A,A,D,A,A,A,D,24.1,54,sucrose solution 2 microliter"""

    # 2. 5 % acetone 2 microliter (N=14 in raw sheet, 2 dead on Day 1 baseline -> N=12 alive for Log-Rank)
    d_acetone_5pct_2ul = """Day,Bee 1,Bee 2,Bee 3,Bee 4,Bee 5,Bee 6,Bee 7,Bee 8,Bee 9,Bee 10,Bee 11,Bee 12,Bee 13,Bee 14,Temp (°C),Humidity,Treatment
Day 1,A,D,A,A,D,A,A,A,A,A,A,A,A,A,24.1,55,5 % acetone 2 microliter
Day 2,A,D,A,A,D,A,A,A,A,A,A,A,A,A,24.2,56,5 % acetone 2 microliter
Day 3,A,D,A,A,D,A,A,A,A,A,A,A,A,A,24.0,54,5 % acetone 2 microliter
Day 4,A,D,A,A,D,A,A,A,A,A,A,A,A,A,24.3,57,5 % acetone 2 microliter
Day 5,A,D,A,A,D,A,D,A,A,A,A,D,A,A,24.1,55,5 % acetone 2 microliter
Day 6,D,D,A,A,D,A,D,A,A,A,A,D,A,A,24.4,56,5 % acetone 2 microliter
Day 7,D,D,A,A,D,A,D,A,D,A,A,D,A,D,24.2,55,5 % acetone 2 microliter
Day 8,D,D,A,A,D,A,D,A,D,D,A,D,A,D,24.1,54,5 % acetone 2 microliter"""

    # 3. 10 % acetone 2 microliter (N=14, accelerated mortality)
    d_acetone_10pct_2ul = """Day,Bee 1,Bee 2,Bee 3,Bee 4,Bee 5,Bee 6,Bee 7,Bee 8,Bee 9,Bee 10,Bee 11,Bee 12,Bee 13,Bee 14,Temp (°C),Humidity,Treatment
Day 1,A,A,A,A,A,A,A,A,A,A,A,A,A,A,24.1,55,10 % acetone 2 microliter
Day 2,A,D,A,A,A,A,A,D,A,A,A,A,A,A,24.2,56,10 % acetone 2 microliter
Day 3,A,D,A,D,A,A,A,D,A,A,D,A,A,A,24.0,54,10 % acetone 2 microliter
Day 4,D,D,A,D,A,A,D,D,A,A,D,A,D,A,24.3,57,10 % acetone 2 microliter
Day 5,D,D,D,D,A,A,D,D,A,D,D,A,D,A,24.1,55,10 % acetone 2 microliter
Day 6,D,D,D,D,A,D,D,D,A,D,D,A,D,D,24.4,56,10 % acetone 2 microliter
Day 7,D,D,D,D,D,D,D,D,A,D,D,D,D,D,24.2,55,10 % acetone 2 microliter
Day 8,D,D,D,D,D,D,D,D,D,D,D,D,D,D,24.1,54,10 % acetone 2 microliter"""

    # 4. 5 % acetone 5 microliter (N=14, high volume exposure)
    d_acetone_5pct_5ul = """Day,Bee 1,Bee 2,Bee 3,Bee 4,Bee 5,Bee 6,Bee 7,Bee 8,Bee 9,Bee 10,Bee 11,Bee 12,Bee 13,Bee 14,Temp (°C),Humidity,Treatment
Day 1,A,A,A,A,A,A,A,A,A,A,A,A,A,A,24.1,55,5 % acetone 5 microliter
Day 2,D,A,D,A,D,A,A,A,D,A,A,A,A,A,24.2,56,5 % acetone 5 microliter
Day 3,D,D,D,A,D,A,D,A,D,A,D,A,A,D,24.0,54,5 % acetone 5 microliter
Day 4,D,D,D,D,D,A,D,D,D,A,D,D,A,D,24.3,57,5 % acetone 5 microliter
Day 5,D,D,D,D,D,D,D,D,D,A,D,D,D,D,24.1,55,5 % acetone 5 microliter
Day 6,D,D,D,D,D,D,D,D,D,D,D,D,D,D,24.4,56,5 % acetone 5 microliter
Day 7,D,D,D,D,D,D,D,D,D,D,D,D,D,D,24.2,55,5 % acetone 5 microliter
Day 8,D,D,D,D,D,D,D,D,D,D,D,D,D,D,24.1,54,5 % acetone 5 microliter"""

    return {
        COHORT_SUCROSE_2UL: pd.read_csv(io.StringIO(d_sucrose)),
        COHORT_ACETONE_5PCT_2UL: pd.read_csv(io.StringIO(d_acetone_5pct_2ul)),
        COHORT_ACETONE_10PCT_2UL: pd.read_csv(io.StringIO(d_acetone_10pct_2ul)),
        COHORT_ACETONE_5PCT_5UL: pd.read_csv(io.StringIO(d_acetone_5pct_5ul)),
    }


def get_active_cohort_dictionary(uploaded_file_dict=None):
    """
    Returns a dictionary of all active cohort dataframes {cohort_label: df}.
    By default, takes ALL uploaded files from the user's session without requiring
    any manual Batch A dialog or checkboxes.
    If no files have been uploaded yet, falls back to the 4 standard agreed benchmark cohorts.
    All dataframes strictly follow the Post-2h Starvation Rule (d0 / Time Zero excluded).
    """
    active_dict = {}
    if uploaded_file_dict and isinstance(uploaded_file_dict, dict):
        for fname, file_data in uploaded_file_dict.items():
            if isinstance(file_data, tuple):
                c_name, c_df = file_data
            elif isinstance(file_data, dict):
                c_name = file_data.get('name', fname)
                c_df = file_data.get('df')
            else:
                c_name = clean_cohort_name(fname)
                c_df = file_data

            c_name_clean = clean_cohort_name(c_name)
            key_name = c_name_clean
            if key_name in active_dict:
                key_name = f"{c_name_clean} ({fname})"
            active_dict[key_name] = c_df

    if not active_dict:
        return get_default_4_cohort_dataframes()

    return active_dict

