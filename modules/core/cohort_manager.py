"""
Cohort Management and Bioassay Standardization Module.
Defines the standard 4 experimental cohorts, agreed code names,
smart cohort name resolution for uploaded files, and post-2h starvation benchmark datasets.
"""
import os
import io
import re
import numpy as np
import pandas as pd

try:
    import streamlit as st
    cache_data_fn = st.cache_data
except Exception:
    def cache_data_fn(func=None, **kwargs):
        if func is not None:
            return func
        return lambda f: f

# The 4 official agreed cohorts
COHORT_SUCROSE_2UL = "sucrose solution 2 microliter"
COHORT_ACETONE_5PCT_2UL = "5 % acetone 2 microliter"
COHORT_ACETONE_10PCT_2UL = "10 % acetone 2 microliter"
COHORT_ACETONE_5PCT_5UL = "5 % acetone 5 microliter"
COHORT_BLINDED = "None / Blinded (Unknown)"

AGREED_COHORTS = [
    COHORT_SUCROSE_2UL,
    COHORT_ACETONE_5PCT_2UL,
    COHORT_ACETONE_10PCT_2UL,
    COHORT_ACETONE_5PCT_5UL
]

# Extended options including blind experiment option
ALL_COHORT_OPTIONS = [
    COHORT_BLINDED,
    COHORT_SUCROSE_2UL,
    COHORT_ACETONE_5PCT_2UL,
    COHORT_ACETONE_10PCT_2UL,
    COHORT_ACETONE_5PCT_5UL,
    "Custom / Other"
]

# Standard agreed short code names for charts, legends, forest plots, and summary badges
COHORT_CODE_NAMES = {
    COHORT_SUCROSE_2UL: "Sucrose 2 µL",
    COHORT_ACETONE_5PCT_2UL: "5% Acetone 2 µL",
    COHORT_ACETONE_10PCT_2UL: "10% Acetone 2 µL",
    COHORT_ACETONE_5PCT_5UL: "5% Acetone 5 µL",
    COHORT_BLINDED: "Blinded"
}

# Harmonized distinct visual palette for the cohorts
COHORT_PALETTE = {
    COHORT_SUCROSE_2UL: "#2563eb",       # Royal Blue (Control / Baseline)
    COHORT_ACETONE_5PCT_2UL: "#0d9488",  # Teal / Cyan-Green (Mild solvent)
    COHORT_ACETONE_10PCT_2UL: "#f59e0b", # Amber / Orange (Moderate solvent)
    COHORT_ACETONE_5PCT_5UL: "#dc2626",  # Crimson Red (High volume solvent)
    COHORT_BLINDED: "#64748b"            # Slate Gray (Neutral Blinded)
}

COHORT_MARKERS = {
    COHORT_SUCROSE_2UL: "o",       # Circle
    COHORT_ACETONE_5PCT_2UL: "s",  # Square
    COHORT_ACETONE_10PCT_2UL: "^", # Triangle Up
    COHORT_ACETONE_5PCT_5UL: "D",  # Diamond
    COHORT_BLINDED: "o"            # Circle
}


def extract_key_from_df(df) -> str:
    """
    Extracts non-empty key value from DataFrame if a 'key' column exists.
    Strictly prioritizes Day 1 (first row) per bioassay standard, falling back
    to first non-empty entry only if Day 1 was blank.
    Returns empty string if column is absent or all values are empty / blank / NaN.
    """
    if df is not None and isinstance(df, pd.DataFrame):
        for col in df.columns:
            if str(col).strip().lower() == 'key':
                if len(df) > 0:
                    d1_val = str(df[col].iloc[0]).strip()
                    if d1_val and d1_val.lower() not in ['nan', 'none', 'null', '']:
                        return d1_val
                vals = [str(x).strip() for x in df[col].dropna() if str(x).strip() and str(x).strip().lower() not in ['nan', 'none', 'null', '']]
                if vals:
                    return vals[0]
    return ""


def get_dataset_key_or_filename(raw_name: str, df=None) -> str:
    """
    Returns the key value from the DataFrame if present and non-empty.
    If there is no key value, falls back to the file name.
    """
    key_val = extract_key_from_df(df)
    if key_val:
        return key_val
    return str(raw_name).strip() if raw_name else "Dataset"


def resolve_treatment_string(t_val: str) -> str:
    """
    Parses a single treatment string (e.g. '5% Acetone 2ml', '5ml, 5% Acetone', 'Sucrose 2ml')
    to an agreed standard cohort name, or returns COHORT_BLINDED if unknown/none/blinded.
    """
    s = str(t_val).strip().lower()
    if not s or any(k in s for k in ['none', 'blind', 'unknown', 'nan', 'null', 'temp']):
        return COHORT_BLINDED
    if any(k in s for k in ['sucrose', 'control', 'water', 'ctrl', '0um']):
        return COHORT_SUCROSE_2UL
    if '10' in s and ('acetone' in s or '%' in s or 'um' in s):
        return COHORT_ACETONE_10PCT_2UL
    if '5' in s and ('acetone' in s or '%' in s or 'um' in s):
        # Explicit volume 5 check
        if any(v in s for v in ['5ml', '5 ml', '5ul', '5 ul', '5 microliter', '5_ul', '50um']):
            return COHORT_ACETONE_5PCT_5UL
        # Explicit volume 2 check
        if any(v in s for v in ['2ml', '2 ml', '2ul', '2 ul', '2 microliter', '2_ul']):
            return COHORT_ACETONE_5PCT_2UL
        nums = re.findall(r'\b5\b|5%', s)
        if len(nums) >= 2 or s.count('5') >= 2:
            return COHORT_ACETONE_5PCT_5UL
        return COHORT_ACETONE_5PCT_2UL
    return ""


def clean_display_name(raw_name: str, df=None) -> str:
    """
    Returns a clean, title-cased display label based on key (if non-empty) or file name.
    Example: 'FM_261001_01A.csv' -> 'Fm 261001 01A'
    """
    key_val = extract_key_from_df(df)
    if key_val:
        s = key_val
    else:
        s = os.path.basename(str(raw_name).strip()) if raw_name else "Dataset"
    s_clean = re.sub(r"\.(csv|tsv|txt|xlsx)$", "", s, flags=re.IGNORECASE)
    s_clean = re.sub(r"^(sample_|batch_|file_|data_)", "", s_clean, flags=re.IGNORECASE)
    cleaned = re.sub(r"[_\-]+", " ", s_clean).strip()
    return cleaned.title() if cleaned else str(raw_name)


def clean_cohort_name(raw_name: str, df=None, fallback_to_blinded: bool = False) -> str:
    """
    Intelligently cleans and maps raw filenames or user inputs to the agreed standard cohort name.
    Inspects optional DataFrame's 'key' or 'Treatment' column, or checks local file on disk to accurately
    identify cohorts like '10 % Acetone' from '2026-30_export.csv'.
    If a non-empty 'key' is present in df, uses the key value for identification.
    If there is no key value, uses the file name.
    If fallback_to_blinded is True and no known treatment matches, returns COHORT_BLINDED.
    """
    # 1. Check DataFrame's 'Treatment' column first if available
    if df is not None and isinstance(df, pd.DataFrame):
        for col in df.columns:
            if str(col).strip().lower() in ['treatment', 'temp_treatment', 'temptreatment']:
                t_vals = [str(x).strip() for x in df[col].dropna() if str(x).strip() and str(x).strip().lower() not in ['nan', 'none', 'null', '']]
                for tv in t_vals[:5]:
                    resolved = resolve_treatment_string(tv)
                    if resolved and resolved != COHORT_BLINDED:
                        return resolved

    # Check if df contains a non-empty key value
    key_val = extract_key_from_df(df)
    effective_name = key_val if key_val else raw_name

    if not effective_name and df is None:
        return COHORT_BLINDED if fallback_to_blinded else COHORT_SUCROSE_2UL

    # Check for explicit blinded indicator
    if str(effective_name).strip() in [COHORT_BLINDED, "None", "None / Blinded", "Blinded (Unknown)", "Blinded"]:
        return COHORT_BLINDED

    s = str(effective_name).strip() if effective_name else ""

    # Exact matches first
    if s in AGREED_COHORTS or s == COHORT_BLINDED:
        return s
    for full_n, code_n in COHORT_CODE_NAMES.items():
        if s.lower() == code_n.lower():
            return full_n

    # 2. If raw_name points to an existing file on disk, peek at its Treatment column
    if s and (os.path.isfile(s) or os.path.isfile(os.path.join(".", s))):
        f_path = s if os.path.isfile(s) else os.path.join(".", s)
        try:
            peek_df = pd.read_csv(f_path, nrows=5)
            for c in peek_df.columns:
                if str(c).strip().lower() in ['treatment', 'temp_treatment', 'temptreatment']:
                    return clean_cohort_name(raw_name, df=peek_df, fallback_to_blinded=fallback_to_blinded)
        except Exception:
            pass

    # Remove extensions and file prefixes
    s_base = os.path.basename(s)
    s_clean = re.sub(r"\.(csv|tsv|txt|xlsx)$", "", s_base, flags=re.IGNORECASE)
    s_clean = re.sub(r"^(sample_|batch_|file_|data_)", "", s_clean, flags=re.IGNORECASE)
    s_clean_lower = s_clean.lower()

    if any(k in s_clean_lower for k in ["blind", "unknown"]):
        return COHORT_BLINDED

    # Match Control / Sucrose
    if any(k in s_clean_lower for k in ["sucrose", "control", "0um", "0_um", "ctrl", "water"]):
        return COHORT_SUCROSE_2UL

    # Match 10 % acetone 2 microliter
    if "10" in s_clean_lower and ("acetone" in s_clean_lower or "%" in s_clean_lower or "um" in s_clean_lower):
        return COHORT_ACETONE_10PCT_2UL

    # Match 5 % acetone 5 microliter
    if ("5" in s_clean_lower and ("5ul" in s_clean_lower or "5_ul" in s_clean_lower or "5 microliter" in s_clean_lower or "50um" in s_clean_lower)) or (s_clean_lower.count("5") >= 2 and "2" not in s_clean_lower):
        return COHORT_ACETONE_5PCT_5UL

    # Match 5 % acetone 2 microliter
    if "5" in s_clean_lower and ("2" in s_clean_lower or "2ul" in s_clean_lower or "2 microliter" in s_clean_lower or "acetone" in s_clean_lower):
        return COHORT_ACETONE_5PCT_2UL

    if fallback_to_blinded:
        return COHORT_BLINDED

    # Clean generic fallback: replace underscores with spaces and title-case
    cleaned = re.sub(r"[_\-]+", " ", s_clean).strip()
    return cleaned.title() if cleaned else raw_name


def get_cohort_code(cohort_name: str, df=None) -> str:
    """Returns the standardized short code name for a given cohort."""
    if not cohort_name or str(cohort_name).strip() in [COHORT_BLINDED, "None", "None / Blinded", "Blinded (Unknown)", "Blinded"]:
        return "Blinded"
    if cohort_name in COHORT_CODE_NAMES:
        return COHORT_CODE_NAMES[cohort_name]
    full_c = clean_cohort_name(cohort_name, df=df)
    return COHORT_CODE_NAMES.get(full_c, full_c)


def _attach_consumption_to_cohort(raw_csv_str: str, base_intake: float, intake_std: float = 2.0) -> pd.DataFrame:
    """Helper to interleave realistic sucrose consumption (µL) for each bee."""
    df = pd.read_csv(io.StringIO(raw_csv_str.strip()))
    # Seeded random generator for repeatable realistic numbers
    rng = np.random.RandomState(42)
    bee_cols = [c for c in df.columns if str(c).startswith('Bee ')]
    num_days = len(df)
    
    # Interleave Cons columns
    new_cols = ['Day']
    for b in bee_cols:
        b_idx = int(b.split()[-1])
        new_cols.append(b)
        cons_col_name = f"Cons {b_idx} (µL)"
        new_cols.append(cons_col_name)
        
        # Generate intake based on bee alive/dead state and day
        cons_vals = []
        # Give each bee a slight individual appetite factor (0.85 to 1.15)
        bee_factor = 0.85 + (b_idx % 7) * 0.05
        for d_idx in range(num_days):
            st_val = str(df.at[d_idx, b]).strip().upper()
            if st_val == 'D':
                cons_vals.append(0.0)
            else:
                # Mild natural day-to-day fluctuation
                day_decay = max(0.6, 1.0 - (d_idx * 0.03))
                val = max(1.0, round(base_intake * bee_factor * day_decay + rng.normal(0, intake_std), 1))
                cons_vals.append(val)
        df[cons_col_name] = cons_vals

    for rem in ['Temp (°C)', 'Humidity', 'Treatment']:
        if rem in df.columns:
            new_cols.append(rem)
            
    return df[[c for c in new_cols if c in df.columns]]


@cache_data_fn(show_spinner=False)
def get_default_4_cohort_dataframes():
    """
    Generates realistic, validated post-2h starvation observation dataframes
    for the 4 agreed experimental cohorts, including full per-bee sucrose consumption (µL).
    All data starts at Day 1 (Day 0 / pre-starvation is excluded).
    """
    import numpy as np

    # 1. sucrose solution 2 microliter (Control, N=14, high survival, high sucrose intake)
    d_sucrose = """Day,Bee 1,Bee 2,Bee 3,Bee 4,Bee 5,Bee 6,Bee 7,Bee 8,Bee 9,Bee 10,Bee 11,Bee 12,Bee 13,Bee 14,Temp (°C),Humidity,Treatment
Day 1,A,A,A,A,A,A,A,A,A,A,A,A,A,A,24.1,55,sucrose solution 2 microliter
Day 2,A,A,A,A,A,A,A,A,A,A,A,A,A,A,24.2,56,sucrose solution 2 microliter
Day 3,A,A,A,A,A,A,A,A,A,A,A,A,A,A,24.0,54,sucrose solution 2 microliter
Day 4,A,A,A,A,A,A,A,A,A,A,A,A,A,A,24.3,57,sucrose solution 2 microliter
Day 5,A,A,A,A,A,A,A,A,A,A,A,A,A,A,24.1,55,sucrose solution 2 microliter
Day 6,A,A,A,D,A,A,A,A,A,A,A,A,A,A,24.4,56,sucrose solution 2 microliter
Day 7,A,A,A,D,A,A,A,A,A,D,A,A,A,A,24.2,55,sucrose solution 2 microliter
Day 8,A,A,A,D,A,A,A,A,A,D,A,A,A,D,24.1,54,sucrose solution 2 microliter"""

    # 2. 5 % acetone 2 microliter (N=14 in raw sheet, moderate intake ~13 µL)
    d_acetone_5pct_2ul = """Day,Bee 1,Bee 2,Bee 3,Bee 4,Bee 5,Bee 6,Bee 7,Bee 8,Bee 9,Bee 10,Bee 11,Bee 12,Bee 13,Bee 14,Temp (°C),Humidity,Treatment
Day 1,A,D,A,A,D,A,A,A,A,A,A,A,A,A,24.1,55,5 % acetone 2 microliter
Day 2,A,D,A,A,D,A,A,A,A,A,A,A,A,A,24.2,56,5 % acetone 2 microliter
Day 3,A,D,A,A,D,A,A,A,A,A,A,A,A,A,24.0,54,5 % acetone 2 microliter
Day 4,A,D,A,A,D,A,A,A,A,A,A,A,A,A,24.3,57,5 % acetone 2 microliter
Day 5,A,D,A,A,D,A,D,A,A,A,A,D,A,A,24.1,55,5 % acetone 2 microliter
Day 6,D,D,A,A,D,A,D,A,A,A,A,D,A,A,24.4,56,5 % acetone 2 microliter
Day 7,D,D,A,A,D,A,D,A,D,A,A,D,A,D,24.2,55,5 % acetone 2 microliter
Day 8,D,D,A,A,D,A,D,A,D,D,A,D,A,D,24.1,54,5 % acetone 2 microliter"""

    # 3. 10 % acetone 2 microliter (N=14, accelerated mortality, depressed intake ~8 µL)
    d_acetone_10pct_2ul = """Day,Bee 1,Bee 2,Bee 3,Bee 4,Bee 5,Bee 6,Bee 7,Bee 8,Bee 9,Bee 10,Bee 11,Bee 12,Bee 13,Bee 14,Temp (°C),Humidity,Treatment
Day 1,A,A,A,A,A,A,A,A,A,A,A,A,A,A,24.1,55,10 % acetone 2 microliter
Day 2,A,D,A,A,A,A,A,D,A,A,A,A,A,A,24.2,56,10 % acetone 2 microliter
Day 3,A,D,A,D,A,A,A,D,A,A,D,A,A,A,24.0,54,10 % acetone 2 microliter
Day 4,D,D,A,D,A,A,D,D,A,A,D,A,D,A,24.3,57,10 % acetone 2 microliter
Day 5,D,D,D,D,A,A,D,D,A,D,D,A,D,A,24.1,55,10 % acetone 2 microliter
Day 6,D,D,D,D,A,D,D,D,A,D,D,A,D,D,24.4,56,10 % acetone 2 microliter
Day 7,D,D,D,D,D,D,D,D,A,D,D,D,D,D,24.2,55,10 % acetone 2 microliter
Day 8,D,D,D,D,D,D,D,D,D,D,D,D,D,D,24.1,54,10 % acetone 2 microliter"""

    # 4. 5 % acetone 5 microliter (N=14, acute toxicity, low intake ~4 µL)
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
        COHORT_SUCROSE_2UL: _attach_consumption_to_cohort(d_sucrose, base_intake=17.5, intake_std=2.2),
        COHORT_ACETONE_5PCT_2UL: _attach_consumption_to_cohort(d_acetone_5pct_2ul, base_intake=12.5, intake_std=2.0),
        COHORT_ACETONE_10PCT_2UL: _attach_consumption_to_cohort(d_acetone_10pct_2ul, base_intake=8.0, intake_std=1.8),
        COHORT_ACETONE_5PCT_5UL: _attach_consumption_to_cohort(d_acetone_5pct_5ul, base_intake=4.5, intake_std=1.5),
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


def compute_adaptive_marker_sizes(x_series_list, y_series_list, base_size=7.5, size_step=3.0, tol_x=0.03, tol_y=0.15):
    """
    Detects overlapping or colliding coordinates across multiple data series.
    When observation points from different batches share similar (x, y) coordinates,
    allocates tiered marker sizes (earlier/underlying series get stepped up in size)
    so concentric nested glyphs are formed and all cohort symbols remain fully visible.

    Args:
        x_series_list: list of array-like x-coordinates for each series
        y_series_list: list of array-like y-coordinates for each series
        base_size: default marker diameter size in points (e.g. 7.0)
        size_step: incremental size increase for each overlapping layer (e.g. 3.0)
        tol_x: coordinate tolerance in x dimension
        tol_y: coordinate tolerance in y dimension

    Returns:
        list of 1D numpy arrays containing marker diameter size (in points) for each point in each series.
    """
    n_series = len(x_series_list)
    result_sizes = [np.full(len(x_series_list[i]), float(base_size), dtype=float) for i in range(n_series)]

    # Collect valid coordinate points: (series_idx, point_idx, x, y)
    all_pts = []
    for s_idx in range(n_series):
        xs = np.array(x_series_list[s_idx], dtype=float)
        ys = np.array(y_series_list[s_idx], dtype=float)
        n_p = min(len(xs), len(ys))
        for p_idx in range(n_p):
            if np.isfinite(xs[p_idx]) and np.isfinite(ys[p_idx]):
                all_pts.append((s_idx, p_idx, float(xs[p_idx]), float(ys[p_idx])))

    # Cluster nearby points across different series
    visited = set()
    for i in range(len(all_pts)):
        if i in visited:
            continue
        cluster = [all_pts[i]]
        visited.add(i)
        s_i, p_i, x_i, y_i = all_pts[i]
        for j in range(i + 1, len(all_pts)):
            if j in visited:
                continue
            s_j, p_j, x_j, y_j = all_pts[j]
            if s_i == s_j:
                continue
            if abs(x_i - x_j) <= tol_x and abs(y_i - y_j) <= tol_y:
                cluster.append(all_pts[j])
                visited.add(j)

        if len(cluster) > 1:
            # Sort cluster by series index (earlier drawn series gets larger marker size)
            cluster.sort(key=lambda item: item[0])
            m = len(cluster)
            for rank, (s_idx, p_idx, _, _) in enumerate(cluster):
                alloc_size = float(base_size) + (m - 1 - rank) * float(size_step)
                result_sizes[s_idx][p_idx] = alloc_size

    return result_sizes

