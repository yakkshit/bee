import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

from scipy.stats import norm, linregress
from lifelines import KaplanMeierFitter
import io
import base64
import zipfile
import re

# Publication-grade typography and aesthetic parameters matching the paper
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#111111'
plt.rcParams['axes.linewidth'] = 1.6
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['xtick.major.size'] = 5.5
plt.rcParams['xtick.major.width'] = 1.6
plt.rcParams['ytick.major.size'] = 5.5
plt.rcParams['ytick.major.width'] = 1.6

st.set_page_config(page_title="Bee Analytics Studio", page_icon="🐝", layout="wide", initial_sidebar_state="expanded")

# Custom UI Design System Injection
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #0b0f19 0%, #0f172a 100%);
    color: #f3f4f6;
}

/* Hero Header Banner */
.hero-card {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.18) 0%, rgba(217, 119, 6, 0.08) 50%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 20px;
    padding: 30px 38px;
    margin-bottom: 24px;
    box-shadow: 0 14px 36px rgba(0, 0, 0, 0.4);
    position: relative;
    overflow: hidden;
}

.hero-card::after {
    content: "🐝";
    position: absolute;
    right: 28px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 90px;
    opacity: 0.16;
    pointer-events: none;
}

.hero-title {
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 50%, #fef3c7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}

.hero-subtitle {
    font-size: 14px;
    color: #9ca3af;
    font-weight: 500;
}

/* Glass Section Container */
.glass-card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 24px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}

/* Section Titles */
.section-title {
    font-size: 19px;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 10px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.06) 0%, rgba(255, 255, 255, 0.02) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 14px;
    padding: 16px 20px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-3px);
    border-color: rgba(245, 158, 11, 0.4);
}
.kpi-val {
    font-size: 26px;
    font-weight: 800;
    color: #fbbf24;
}
.kpi-label {
    font-size: 11px;
    font-weight: 700;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(245, 158, 11, 0.3) !important;
}
.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(245, 158, 11, 0.45) !important;
}

.stDownloadButton>button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3) !important;
}
.stDownloadButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.45) !important;
}

/* Sidebar Overrides */
[data-testid="stSidebar"] {
    background-color: #0b0f19 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
</style>
""", unsafe_allow_html=True)

# Sidebar Design
st.sidebar.markdown("<h2 style='color: #fbbf24; font-size: 20px; font-weight: 800; margin-bottom: 15px;'>🎨 Chart Studio Controls</h2>", unsafe_allow_html=True)
primary_color = st.sidebar.color_picker("Primary Color (Mortality)", "#ef4444")
secondary_color = st.sidebar.color_picker("Secondary Color (Survival)", "#f59e0b")
line_width = st.sidebar.slider("Line Width", min_value=1.0, max_value=5.0, value=2.4)
marker_size = st.sidebar.slider("Marker Size", min_value=3.0, max_value=12.0, value=7.0)
font_size = st.sidebar.slider("Font Size", min_value=8, max_value=18, value=11)

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='color: #fbbf24; font-size: 16px; font-weight: 700;'>🧪 Statistical Models</h3>", unsafe_allow_html=True)
abbott_c = st.sidebar.number_input("Abbott Control Mortality (C) %", min_value=0.0, max_value=100.0, value=0.0, step=1.0, help="Consider all as alive by default (C=0)")

line_style_map = {"Solid": "-", "Dashed": "--", "Dotted": ":", "Dash-Dot": "-."}
selected_line_style = st.sidebar.selectbox("Select Line Style", list(line_style_map.keys()))
line_style = line_style_map[selected_line_style]

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='color: #fbbf24; font-size: 16px; font-weight: 700;'>🔥 Heatmap Palette</h3>", unsafe_allow_html=True)
hm_alive_color = st.sidebar.color_picker("Alive Color", "#10b981")
hm_behavior_color = st.sidebar.color_picker("Behavior Change Color", "#f59e0b")
hm_dead_color = st.sidebar.color_picker("Dead Color", "#ef4444")

plt.rcParams['font.size'] = font_size

# Hero Banner
st.markdown("""
<div class="hero-card">
    <div class="hero-title">🐝 Bee Observation & Survival Analytics Studio</div>
    <div class="hero-subtitle">Publication-Grade Statistical Modeling, Environmental Correlation & Individual Bee Tracking</div>
</div>
""", unsafe_allow_html=True)

metadata_keys = ['Date', 'Start time', 'Weather', 'Observer', 'Treatment', 'Hive / colony ID', 'Bee type', 'Feeder / resource', 'Purpose / motivation', 'Outside temp(°C)', 'Outdoor humidity', 'Other conditions']
for k in metadata_keys:
    if f"meta_{k}" not in st.session_state:
        st.session_state[f"meta_{k}"] = ""

with st.expander("📝 Edit Observation Sheet Metadata", expanded=False):
    cols1 = st.columns(4)
    st.session_state['meta_Date'] = cols1[0].text_input("Date", value=st.session_state['meta_Date'] or "01.09.2026")
    st.session_state['meta_Start time'] = cols1[1].text_input("Start time", value=st.session_state['meta_Start time'] or "15:23")
    st.session_state['meta_Weather'] = cols1[2].text_input("Weather", value=st.session_state['meta_Weather'] or "sunny")
    st.session_state['meta_Observer'] = cols1[3].text_input("Observer", value=st.session_state['meta_Observer'] or "Fereshteh")
    
    cols2 = st.columns(4)
    st.session_state['meta_Treatment'] = cols2[0].text_input("Treatment", value=st.session_state['meta_Treatment'] or "Sucrose solution")
    st.session_state['meta_Hive / colony ID'] = cols2[1].text_input("Hive / colony ID", value=st.session_state['meta_Hive / colony ID'] or "Colony n.9")
    st.session_state['meta_Bee type'] = cols2[2].text_input("Bee type", value=st.session_state['meta_Bee type'] or "Forager")
    st.session_state['meta_Feeder / resource'] = cols2[3].text_input("Feeder / resource", value=st.session_state['meta_Feeder / resource'] or "Pipette: 2 µL")
    
    cols3 = st.columns(4)
    st.session_state['meta_Purpose / motivation'] = cols3[0].text_input("Purpose / motivation", value=st.session_state['meta_Purpose / motivation'] or "baseline")
    st.session_state['meta_Outside temp(°C)'] = cols3[1].text_input("Outside temp(°C)", value=st.session_state['meta_Outside temp(°C)'] or "24.6")
    st.session_state['meta_Outdoor humidity'] = cols3[2].text_input("Outdoor humidity", value=st.session_state['meta_Outdoor humidity'] or "44")
    st.session_state['meta_Other conditions'] = cols3[3].text_input("Other conditions", value=st.session_state['meta_Other conditions'] or "")


def parse_time_data(df):
    """Parses the 'Day' column into continuous elapsed days starting at Day 1 = 1.0."""
    time_col_name = 'Day' if 'Day' in df.columns else ('Date/time' if 'Date/time' in df.columns else None)
    
    if time_col_name:
        raw_times = df[time_col_name].astype(str).tolist()
    else:
        raw_times = [f"Day {i+1}" for i in range(len(df))]
        
    def parse_to_days(t_str):
        t_str = str(t_str).strip()
        # Match patterns like Day1, Day 1, Day1_2, Day1_2:10
        match = re.match(r"Day\s*(\d+)(?:_(\d+)(?::(\d+))?)?", t_str, re.IGNORECASE)
        if match:
            d = int(match.group(1))
            h = int(match.group(2)) if match.group(2) else 0
            m = int(match.group(3)) if match.group(3) else 0
            # Day 1 = 1.0 day, Day 1_2 = 1 + 2/24 = 1.0833 days, Day 2 = 2.0 days
            return d * 1.0 + (h / 24.0) + (m / 1440.0)
        try:
            val = float(t_str)
            return val if val >= 1.0 else val + 1.0
        except:
            return np.nan

    elapsed_days = np.array([parse_to_days(t) for t in raw_times])
    # Fallback for any missing values
    for i in range(len(elapsed_days)):
        if np.isnan(elapsed_days[i]):
            elapsed_days[i] = elapsed_days[i-1] + 1.0 if i > 0 else 1.0
            
    return raw_times, elapsed_days

def parse_batch_df(raw_df, abbott_c=0.0):
    df_copy = raw_df.copy()
    if 'bee_id' in df_copy.columns:
        df_copy = df_copy.set_index('bee_id').T
        df_copy.reset_index(inplace=True)
        df_copy.rename(columns={'index': 'Day'}, inplace=True)
        cols = list(df_copy.columns)
        for i in range(1, len(cols)):
            col_name = str(cols[i])
            if col_name.isdigit():
                cols[i] = f"Bee {col_name}"
            elif col_name == 'Temp':
                cols[i] = 'Temp (°C)'
        df_copy.columns = cols

    df_copy = df_copy.astype(str)
    df_copy.replace("nan", "", inplace=True)

    num_days = len(df_copy)
    raw_times, batch_elapsed_days = parse_time_data(df_copy)
    day_labels = raw_times

    cols_lower = [str(c).lower().strip() for c in df_copy.columns]
    has_bee_cols = any(str(c).lower().startswith('bee') for c in df_copy.columns)

    if not has_bee_cols and any(k in cols_lower for k in ['total', 'dead', 'mortality', '% mortality']):
        dead_col = next((c for c in df_copy.columns if 'dead' in str(c).lower() or 'mortality' in str(c).lower()), None)
        total_col = next((c for c in df_copy.columns if 'total' in str(c).lower() or 'n' in str(c).lower()), None)

        times_numeric = batch_elapsed_days

        if total_col and dead_col:
            totals = pd.to_numeric(df_copy[total_col], errors='coerce').fillna(10).values
            deads = pd.to_numeric(df_copy[dead_col], errors='coerce').fillna(0).values
            num_bees = int(np.max(totals)) if len(totals) > 0 else 10
            mortality_pct = np.clip((deads / totals) * 100.0, 0, 100)
        elif dead_col:
            mortality_pct = pd.to_numeric(df_copy[dead_col], errors='coerce').fillna(0).values
            mortality_pct = np.clip(mortality_pct, 0, 100)
            num_bees = 10
        else:
            mortality_pct = np.zeros(num_days)
            num_bees = 10
    else:
        bee_cols = [c for c in df_copy.columns if str(c).lower().startswith('bee')]
        num_bees = len(bee_cols) if len(bee_cols) > 0 else 10
        times_numeric = batch_elapsed_days

        if len(bee_cols) > 0:
            states_matrix = []
            for bc in bee_cols:
                states_matrix.append(df_copy[bc].fillna("A").tolist())
            dead_counts = []
            for d_idx in range(num_days):
                n_dead = sum(1 for states in states_matrix if str(states[d_idx]).strip() == 'D')
                dead_counts.append(n_dead)
            dead_counts = np.array(dead_counts)
            mortality_pct = (dead_counts / num_bees) * 100.0 if num_bees > 0 else np.zeros(num_days)
        else:
            mortality_pct = np.zeros(num_days)

    abbott_corrected_pct = np.copy(mortality_pct)
    if abbott_c < 100:
        abbott_corrected_pct = (mortality_pct - abbott_c) / (100.0 - abbott_c) * 100.0
        abbott_corrected_pct = np.clip(abbott_corrected_pct, 0, 100)

    survival_pct = 100.0 - mortality_pct

    clipped_mort = np.clip(abbott_corrected_pct, 0.01, 99.99)
    probit_vals = norm.ppf(clipped_mort / 100.0) + 5.0

    log_time = []
    probit_y = []
    for i, t in enumerate(batch_elapsed_days):
        if t > 0:
            log_time.append(np.log10(t))
            probit_y.append(probit_vals[i])

    log_time = np.array(log_time)
    probit_y = np.array(probit_y)


    if len(log_time) > 1:
        res = linregress(log_time, probit_y)
        a, b, r_val = res.intercept, res.slope, res.rvalue
        r_squared = r_val ** 2
        if b != 0:
            lt50 = 10 ** ((5.0 - a) / b)
            lt90 = 10 ** ((6.28155 - a) / b)
        else:
            lt50, lt90 = np.nan, np.nan
    else:
        a, b, r_val, r_squared = 0.0, 0.0, 0.0, 0.0
        lt50, lt90 = np.nan, np.nan

    return {
        'num_days': num_days,
        'num_bees': num_bees,
        'day_labels': day_labels,
        'times_numeric': times_numeric,
        'mortality_pct': mortality_pct,
        'abbott_corrected_pct': abbott_corrected_pct,
        'survival_pct': survival_pct,
        'probit_vals': probit_vals,
        'log_time': log_time,
        'probit_y': probit_y,
        'intercept_a': a,
        'slope_b': b,
        'r_squared': r_squared,
        'r_val': r_val,
        'lt50': lt50,
        'lt90': lt90
    }


DEFAULT_CSV = """Day,Date/time,Bee 1,Bee 2,Bee 3,Bee 4,Bee 5,Bee 6,Bee 7,Bee 8,Bee 9,Bee 10,Bee 11,Bee 12,Bee 13,Bee 14,Temp (°C),Humidity,Treatment,Observations/notes
Day 1,02.09.2026-16:00,A,D,A,A,A,A,A,A,D,A,A,A,A,D,23.3,55,Sucrose solution,
Day 2,03.09.2026-16:00,A,D,A,A,A,A,A,A,D,A,A,A,A,D,24.4,45,Sucrose solution,
Day 3,04.09.2026-16:00,A,D,A,A,B,A,A,A,D,A,A,A,A,D,24.7,58,Sucrose solution,
Day 4,05.09.2026-16:00,A,D,A,A,B,A,A,A,D,A,A,A,A,D,23.8,61,Sucrose solution,
Day 5,06.09.2026-16:00,A,D,D,A,B,A,A,A,D,A,A,D,A,D,23.7,51,Sucrose solution,
Day 6,07.09.2026-16:00,D,D,D,A,B,A,A,A,D,A,A,D,A,D,24.5,64,Sucrose solution,
Day 7,08.09.2026-16:00,D,D,D,B,B,A,A,A,D,A,A,D,A,D,24.8,62,Sucrose solution,
Day 8,09.09.2026-16:00,D,D,D,B,B,A,A,A,D,A,A,D,A,D,23.4,59,Sucrose solution,"""

tab1, tab2 = st.tabs(["🐝 Single CSV Analysis", "📈 Multi-CSV Probit Comparison"])


def render_single_batch_tab():





    import os
    CACHE_FILE = ".cached_df.csv"

    if 'df' not in st.session_state:
        if os.path.exists(CACHE_FILE):
            try:
                st.session_state.df = pd.read_csv(CACHE_FILE)
            except Exception:
                st.session_state.df = pd.read_csv(io.StringIO(DEFAULT_CSV))
        else:
            st.session_state.df = pd.read_csv(io.StringIO(DEFAULT_CSV))

        bee_cols_init = [c for c in st.session_state.df.columns if str(c).lower().startswith('bee')]
        for b in bee_cols_init:
            cons_col = b.replace("Bee ", "Cons ") + " (µL)"
            if cons_col not in st.session_state.df.columns:
                st.session_state.df.insert(st.session_state.df.columns.get_loc(b) + 1, cons_col, "0.0")

        # ensure string types
        st.session_state.df = st.session_state.df.astype(str)
        st.session_state.df.replace("nan", "", inplace=True)

    st.write("### Daily Individual Observations")

    uploaded_file = st.file_uploader("Upload CSV file (Optional)", type=["csv"])
    if uploaded_file is not None:
        if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
            try:
                new_df = pd.read_csv(uploaded_file, sep=None, engine='python')

                # Check if it's the transposed format
                if 'bee_id' in new_df.columns:
                    # Transpose the data
                    new_df = new_df.set_index('bee_id').T
                    new_df.reset_index(inplace=True)
                    new_df.rename(columns={'index': 'Day'}, inplace=True)

                    # Rename bee columns
                    cols = list(new_df.columns)
                    for i in range(1, len(cols)):
                        col_name = str(cols[i])
                        if col_name.isdigit():
                            cols[i] = f"Bee {col_name}"
                        elif col_name == 'Temp':
                            cols[i] = 'Temp (°C)'
                    new_df.columns = cols

                    # Add default columns if missing
                    if 'Date/time' not in new_df.columns:
                        new_df.insert(1, 'Date/time', '')

                    bee_cols_upload = [c for c in new_df.columns if str(c).lower().startswith('bee')]
                    for b in bee_cols_upload:
                        cons_col = b.replace("Bee ", "Cons ") + " (µL)"
                        if cons_col not in new_df.columns:
                            new_df.insert(new_df.columns.get_loc(b) + 1, cons_col, "0.0")

                    if 'Treatment' not in new_df.columns:
                        new_df['Treatment'] = ''
                    if 'Observations/notes' not in new_df.columns:
                        new_df['Observations/notes'] = ''

                new_df = new_df.astype(str)
                new_df.replace("nan", "", inplace=True)
                st.session_state.df = new_df
                st.session_state.last_uploaded = uploaded_file.name
                st.rerun()
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")


    col1, col2, col3 = st.columns([1.5, 1, 4])
    with col1:
        if st.button("➕ Add Bee Column"):
            bee_cols = [c for c in st.session_state.df.columns if str(c).lower().startswith('bee')]
            new_idx = len(bee_cols) + 1
            new_col_name = f"Bee {new_idx}"
            new_cons_name = f"Cons {new_idx} (µL)"
            try:
                temp_idx = list(st.session_state.df.columns).index("Temp (°C)")
            except ValueError:
                temp_idx = len(st.session_state.df.columns)
            st.session_state.df.insert(temp_idx, new_col_name, "A")
            st.session_state.df.insert(temp_idx + 1, new_cons_name, "0.0")
            st.rerun()
    with col2:
        if st.button("🔄 Reset Data"):
            if 'df' in st.session_state:
                del st.session_state['df']
            if os.path.exists(CACHE_FILE):
                os.remove(CACHE_FILE)
            st.rerun()
    with col3:
        st.markdown("You can edit the cells below. **To add a new Day (row)**, scroll to the bottom of the table and click the empty row.")

    edited_df = st.data_editor(st.session_state.df, num_rows="dynamic", use_container_width=True)
    st.session_state.df = edited_df
    df = edited_df.copy()

    try:
        df.to_csv(CACHE_FILE, index=False)
    except Exception:
        pass


    def generate_html_report(dataframe, metadata):
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            body {{ font-family: Arial, sans-serif; font-size: 11px; }}
            h1 {{ text-align: center; color: #1a2a5a; text-transform: uppercase; font-size: 18px; margin-bottom: 2px; }}
            .subtitle {{ text-align: center; font-style: italic; color: #555; margin-bottom: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #9fb4cc; padding: 6px; text-align: center; }}
            th {{ background-color: #e5eff8; color: #1a2a5a; font-weight: bold; }}
            .obs-table th {{ background-color: #1a3b5c; color: white; }}
            .meta-label {{ background-color: #e5eff8; font-weight: bold; color: #1a2a5a; text-align: left; width: 12%; }}
            .meta-val {{ text-align: left; width: 13%; font-style: italic; }}
            @media print {{
                @page {{ size: landscape; margin: 10mm; }}
                body {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
            }}
        </style>
        </head>
        <body>
            <h1>Honeybee Experiment Observation Sheet</h1>
            <div class="subtitle">Enter the study details, then record one observation per bee for each day.</div>

            <table>
                <tr>
                    <td class="meta-label">Date</td><td class="meta-val">{metadata['meta_Date']}</td>
                    <td class="meta-label">Start time</td><td class="meta-val">{metadata['meta_Start time']}</td>
                    <td class="meta-label">Weather</td><td class="meta-val">{metadata['meta_Weather']}</td>
                    <td class="meta-label">Observer</td><td class="meta-val">{metadata['meta_Observer']}</td>
                </tr>
                <tr>
                    <td class="meta-label">Treatment</td><td class="meta-val">{metadata['meta_Treatment']}</td>
                    <td class="meta-label">Hive / colony ID</td><td class="meta-val">{metadata['meta_Hive / colony ID']}</td>
                    <td class="meta-label">Bee type</td><td class="meta-val">{metadata['meta_Bee type']}</td>
                    <td class="meta-label">Feeder / resource</td><td class="meta-val">{metadata['meta_Feeder / resource']}</td>
                </tr>
                <tr>
                    <td class="meta-label">Purpose / motivation</td><td class="meta-val">{metadata['meta_Purpose / motivation']}</td>
                    <td class="meta-label">Outside temp(°C)</td><td class="meta-val">{metadata['meta_Outside temp(°C)']}</td>
                    <td class="meta-label">Outdoor humidity</td><td class="meta-val">{metadata['meta_Outdoor humidity']}</td>
                    <td class="meta-label">Other conditions</td><td class="meta-val">{metadata['meta_Other conditions']}</td>
                </tr>
            </table>

            <h3 style="color: #1a2a5a; margin-bottom: 5px; font-size: 14px;">DAILY INDIVIDUAL OBSERVATIONS</h3>
            <table class="obs-table">
                <tr>
        """
        for col in dataframe.columns:
            html += f"<th>{col}</th>\n"
        html += "</tr>\n"

        for _, row in dataframe.iterrows():
            html += "<tr>\n"
            for val in row:
                v = val if pd.notna(val) else ''
                html += f"<td>{v}</td>\n"
            html += "</tr>\n"

        html += """
            </table>

            <table style="border: none; margin-top: 10px;">
                <tr>
                    <td style="text-align: left; border: 1px solid #9fb4cc; background: #f5f8fc;"><b>Bee status:</b> A = alive, antennae moving &nbsp;|&nbsp; B = alive, antennae not moving &nbsp;|&nbsp; D = dead</td>
                    <td style="text-align: left; border: 1px solid #9fb4cc; background: #f5f8fc;"><b>Score key:</b> -, 0, +, ++ &nbsp;&nbsp;(define the meaning for your experimental protocol)</td>
                </tr>
                <tr>
                    <td style="text-align: left; border: 1px solid #9fb4cc; background: #f5f8fc;"><b>Additional note:</b> Record behaviour, feeding, movement and unusual events.</td>
                    <td style="text-align: left; border: 1px solid #9fb4cc; background: #f5f8fc;"><b>End time:</b> <i>Click or type here</i></td>
                </tr>
            </table>
        </body>
        </html>
        """
        return html

    st.markdown("---")
    st.header("🖨️ Observation Sheet Export")
    st.markdown("Download a printable version of your observation sheet (exactly matching the template). **Instructions:** Download the HTML file, open it in any web browser, and press **Ctrl+P (or Cmd+P)** to save it as a high-quality landscape PDF.")

    html_content = generate_html_report(df, st.session_state)
    st.download_button(
        label="📄 Download Observation Sheet (HTML format for Print-to-PDF)",
        data=html_content,
        file_name="Observation_Sheet.html",
        mime="text/html",
    )


    # ==========================================
    # EXTRACT DATA FOR PLOTS 
    # ==========================================
    st.markdown("---")
    st.header("📊 Plot Generation")

    day_labels, elapsed_days = parse_time_data(df)
    num_days = len(df)
    max_time = elapsed_days[-1] if len(elapsed_days) > 0 else 1.0

    bee_cols = [c for c in df.columns if str(c).lower().startswith('bee')]
    num_bees = len(bee_cols)

    temps_raw = df['Temp (°C)'].tolist() if 'Temp (°C)' in df.columns else [0]*num_days
    humidities_raw = df['Humidity'].tolist() if 'Humidity' in df.columns else [0]*num_days
    consumptions_raw = df['Consumption (µL)'].tolist() if 'Consumption (µL)' in df.columns else [0]*num_days
    temps = [float(x) if str(x).replace('.','',1).isdigit() else 0.0 for x in temps_raw]
    humidities = [float(x) if str(x).replace('.','',1).isdigit() else 0.0 for x in humidities_raw]

    cons_cols = [c for c in df.columns if str(c).startswith('Cons ') and '(µL)' in str(c)]
    cons_matrix = []
    for c in cons_cols:
        raw_vals = df[c].tolist()
        cons_matrix.append([float(x) if str(x).replace('.','',1).isdigit() else 0.0 for x in raw_vals])

    states_matrix = []
    for bc in bee_cols:
        states_matrix.append(df[bc].fillna("A").tolist())

    durations = []
    events = []

    for b_states in states_matrix:
        if 'D' in b_states:
            death_row_idx = b_states.index('D')
            durations.append(elapsed_days[death_row_idx])
            events.append(1)
        else:
            durations.append(elapsed_days[-1] if num_days > 0 else 0)
            events.append(0)

    durations = np.array(durations)
    events = np.array(events)

    alive_counts = []
    dead_counts = []

    for d_idx in range(num_days):
        n_dead = sum(1 for states in states_matrix if str(states[d_idx]).strip() == 'D')
        alive_counts.append(num_bees - n_dead)
        dead_counts.append(n_dead)

    alive_counts = np.array(alive_counts)
    dead_counts = np.array(dead_counts)
    survival_pct = alive_counts / num_bees * 100.0 if num_bees > 0 else np.zeros(num_days)
    mortality_pct = dead_counts / num_bees * 100.0 if num_bees > 0 else np.zeros(num_days)

    # Abbott Corrected Mortality
    abbott_corrected_pct = np.zeros_like(mortality_pct)
    if abbott_c < 100:
        abbott_corrected_pct = (mortality_pct - abbott_c) / (100.0 - abbott_c) * 100.0
        abbott_corrected_pct = np.clip(abbott_corrected_pct, 0, 100)

    # Probit Regression (Time-Mortality)
    clipped_mort = np.clip(abbott_corrected_pct, 0.01, 99.99)
    probit_vals = norm.ppf(clipped_mort / 100.0) + 5

    log_time = []
    probit_y = []
    for i, d in enumerate(elapsed_days):
        if d > 0:
            log_time.append(np.log10(d))
            probit_y.append(probit_vals[i])

    if len(log_time) > 1:
        res = linregress(log_time, probit_y)
        probit_a, probit_b, probit_r = res.intercept, res.slope, res.rvalue
    else:
        probit_a, probit_b, probit_r = 0, 0, 0

    # Binomial Standard Error for proportions
    mortality_se = np.sqrt((mortality_pct / 100.0) * (1.0 - mortality_pct / 100.0) / max(1, num_bees)) * 100.0
    survival_se = np.sqrt((survival_pct / 100.0) * (1.0 - survival_pct / 100.0) / max(1, num_bees)) * 100.0

    # FIT STATISTICAL MODELS
    def logistic_mort(t, L, k, t0):
        return L / (1.0 + np.exp(-k * (t - t0)))

    try:
        popt_mort, _ = curve_fit(
            logistic_mort, elapsed_days, mortality_pct,
            p0=[60.0, 0.5, 5.5],
            bounds=([20.0, 0.01, 0.0], [100.0, 5.0, 15.0]),
            maxfev=10000
        )
    except:
        popt_mort = [60.0, 0.5, 5.5]

    L_fit, k_fit, t0_fit = popt_mort
    residuals_mort = mortality_pct - logistic_mort(elapsed_days, *popt_mort)
    ss_res_mort = np.sum(residuals_mort**2)
    ss_tot_mort = np.sum((mortality_pct - np.mean(mortality_pct))**2)
    r2_mort = 1.0 - (ss_res_mort / ss_tot_mort) if ss_tot_mort != 0 else 1.0

    def survival_decay(t, S0, a, b):
        return S0 * np.exp(-a * (t**b))

    try:
        popt_surv, _ = curve_fit(
            survival_decay, elapsed_days, survival_pct,
            p0=[78.57, 0.001, 3.0],
            bounds=([50.0, 1e-6, 0.1], [100.0, 2.0, 10.0]),
            maxfev=10000
        )
    except:
        popt_surv = [78.57, 0.001, 3.0]

    S0_surv, a_surv, b_surv = popt_surv
    residuals_surv = survival_pct - survival_decay(elapsed_days, *popt_surv)
    ss_res_surv = np.sum(residuals_surv**2)
    ss_tot_surv = np.sum((survival_pct - np.mean(survival_pct))**2)
    r2_surv = 1.0 - (ss_res_surv / ss_tot_surv) if ss_tot_surv != 0 else 1.0

    kmf = KaplanMeierFitter()
    if len(durations) > 0:
        kmf.fit(durations, event_observed=events, timeline=np.linspace(0, max(elapsed_days)+1, max(int(max(elapsed_days)*20), 140)))

    t_dense = np.linspace(0, max_time * 1.05, 250) if num_days > 0 else np.array([])
    fit_mort_curve = logistic_mort(t_dense, *popt_mort) if num_days > 0 else np.array([])
    fit_surv_curve = survival_decay(t_dense, *popt_surv) if num_days > 0 else np.array([])

    # Executive KPI Summary Section
    st.markdown("### 📊 Experiment Overview & Key Metrics")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{num_bees}</div><div class="kpi-label">Observed Bees</div></div>', unsafe_allow_html=True)
    with kpi2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{num_days}</div><div class="kpi-label">Observation Days</div></div>', unsafe_allow_html=True)
    with kpi3:
        final_surv = f"{survival_pct[-1]:.1f}%" if num_days > 0 else "N/A"
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{final_surv}</div><div class="kpi-label">Final Survival Rate</div></div>', unsafe_allow_html=True)
    with kpi4:
        avg_temp = f"{np.mean(temps):.1f}°C" if any(temps) else "N/A"
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{avg_temp}</div><div class="kpi-label">Avg Temperature</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    plot_options = [
        "Mortality Curve", 
        "Survival Curve", 
        "Kaplan-Meier Survival Analysis", 
        "Individual Bee Status Heatmap", 
        "Environmental Conditions", 
        "Combined Multi-Panel Figure",
        "Interactive Daily Status (Bar Chart)",
        "Interactive Cumulative Mortality (Line Chart)",
        "Interactive Status Breakdown (Area Chart)",
        "Interactive Temp vs Humidity (Scatter Plot)",
        "Consumption Plot (Per Bee)",
        "Statistical Correlation Heatmap",
        "Abbott Corrected Mortality Curve",
        "Probit Regression (Time-Mortality)"
    ]
    col_sel, col_zip = st.columns([3, 1])
    with col_sel:
        selected_plots = st.multiselect("Choose which plots to view and download:", plot_options, default=plot_options)

    with col_zip:
        st.write("")
        zip_button_top = st.empty()

    plot_images_dict = {}


    def get_image_download_link(fig, filename, text):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        img_bytes = buf.getvalue()
        plot_images_dict[filename] = img_bytes
        st.download_button(label=text, data=img_bytes, file_name=filename, mime="image/png")
        plt.close(fig)

    cols = st.columns(2)
    col_idx = 0

    if "Mortality Curve" in selected_plots and num_days > 0:
        with cols[col_idx % 2]:
            st.subheader("Mortality Curve")
            fig1, ax = plt.subplots(figsize=(6.8, 5.4), dpi=300)
            fit_se = np.std(residuals_mort)
            ax.fill_between(t_dense, np.clip(fit_mort_curve - 1.96*fit_se, 0, 100), np.clip(fit_mort_curve + 1.96*fit_se, 0, 100), color=primary_color, alpha=0.12, label='95% Confidence Band')
            ax.plot(t_dense, fit_mort_curve, color=primary_color, linewidth=line_width, linestyle=line_style, label='Logistic Model Fit')
            ax.errorbar(elapsed_days, mortality_pct, yerr=mortality_se, fmt='o', color=primary_color, ecolor=primary_color, elinewidth=1.6, capsize=4, capthick=1.6, markersize=marker_size, markerfacecolor=primary_color, markeredgecolor='black', label='Observed (Mean ± SE)')
            ax.axhline(50, color='#333333', linestyle=':', linewidth=1.2)
            eq_text = f"Y = {L_fit:.1f} / (1 + exp(-{k_fit:.2f}(X - {t0_fit:.1f})))\nR² = {r2_mort:.4f}"
            ax.text(0.06, 0.90, eq_text, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#e0e0e0', alpha=0.85))
            ax.set_xlabel('Observation Time / Days', fontsize=13, fontweight='bold', labelpad=7)
            ax.set_ylabel('Mortality / %', fontsize=13, fontweight='bold', labelpad=7)
            ax.set_xlim(-0.1, max_time * 1.05)
            ax.set_ylim(-2, 102)
            ax.set_xticks(elapsed_days)
            ax.set_xticklabels(day_labels, rotation=45, ha="right")
            ax.yaxis.set_major_locator(MultipleLocator(25))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig1.tight_layout()
            st.pyplot(fig1)
            get_image_download_link(fig1, 'fig_mortality_curve.png', 'Download Mortality Curve')
        col_idx += 1

    if "Survival Curve" in selected_plots and num_days > 0:
        with cols[col_idx % 2]:
            st.subheader("Survival Curve")
            fig2, ax = plt.subplots(figsize=(6.8, 5.4), dpi=300)
            surv_se_fit = np.std(residuals_surv)
            ax.fill_between(t_dense, np.clip(fit_surv_curve - 1.96*surv_se_fit, 0, 100), np.clip(fit_surv_curve + 1.96*surv_se_fit, 0, 100), color=secondary_color, alpha=0.15, label='95% Confidence Band')
            ax.plot(t_dense, fit_surv_curve, color=secondary_color, linewidth=line_width, linestyle=line_style, label='Survival Model Fit')
            ax.errorbar(elapsed_days, survival_pct, yerr=survival_se, fmt='o', color=secondary_color, ecolor=secondary_color, elinewidth=1.6, capsize=4, capthick=1.6, markersize=marker_size, markerfacecolor=secondary_color, markeredgecolor='black', label='Observed (Mean ± SE)')
            ax.axhline(50, color='#333333', linestyle=':', linewidth=1.2)
            eq_surv_text = f"Y = {S0_surv:.1f} * exp(-{a_surv:.4f} * X^{b_surv:.2f})\nR² = {r2_surv:.4f}"
            ax.text(0.06, 0.32, eq_surv_text, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#e0e0e0', alpha=0.85))
            ax.set_xlabel('Observation Time / Days', fontsize=13, fontweight='bold', labelpad=7)
            ax.set_ylabel('Bee Survival / %', fontsize=13, fontweight='bold', labelpad=7)
            ax.set_xlim(-0.1, max_time * 1.05)
            ax.set_ylim(-2, 105)
            ax.set_xticks(elapsed_days)
            ax.set_xticklabels(day_labels, rotation=45, ha="right")
            ax.yaxis.set_major_locator(MultipleLocator(25))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig2.tight_layout()
            st.pyplot(fig2)
            get_image_download_link(fig2, 'fig_survival_curve.png', 'Download Survival Curve')
        col_idx += 1

    if "Kaplan-Meier Survival Analysis" in selected_plots and len(durations) > 0:
        with cols[col_idx % 2]:
            st.subheader("Kaplan-Meier Survival Analysis")
            fig3, ax = plt.subplots(figsize=(6.8, 5.4), dpi=300)
            km_timeline = kmf.survival_function_.index
            km_vals = kmf.survival_function_['KM_estimate'].values
            ci_df = kmf.confidence_interval_survival_function_
            ax.step(km_timeline, km_vals, where='post', color='#0066cc', linewidth=line_width, linestyle=line_style, label='Kaplan-Meier Estimate')
            ax.fill_between(km_timeline, ci_df.iloc[:, 0], ci_df.iloc[:, 1], step='post', color='#0066cc', alpha=0.15, label='95% Confidence Interval')
            censored_days = durations[events == 0]
            censored_surv = [kmf.survival_function_at_times(d).values[0] for d in censored_days] if len(censored_days) > 0 else []
            if len(censored_days) > 0:
                ax.plot(censored_days, censored_surv, '+', color='#002244', markersize=9, markeredgewidth=2, label=f'Censored (n={len(censored_days)})')
            ax.axhline(0.5, color='#333333', linestyle=':', linewidth=1.2)
            ax.set_xlabel('Observation Time / Days', fontsize=13, fontweight='bold', labelpad=7)
            ax.set_ylabel('Survival Probability', fontsize=13, fontweight='bold', labelpad=7)
            ax.set_xlim(-0.1, max_time * 1.05)
            ax.set_ylim(-0.02, 1.05)
            ax.set_xticks(elapsed_days)
            ax.set_xticklabels(day_labels, rotation=45, ha="right")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(loc='lower left', frameon=False, fontsize=10.5)
            fig3.tight_layout()
            st.pyplot(fig3)
            get_image_download_link(fig3, 'fig_kaplan_meier.png', 'Download KM Curve')
        col_idx += 1

    if "Individual Bee Status Heatmap" in selected_plots and num_days > 0 and num_bees > 0:
        with cols[col_idx % 2]:
            st.subheader("Individual Bee Status Matrix Heatmap")
            fig4, ax = plt.subplots(figsize=(8.5, 6.2), dpi=300)
            state_num_map = {'A': 1, 'B': 2, 'D': 3}
            color_map = {1: hm_alive_color, 2: hm_behavior_color, 3: hm_dead_color}
            mat = np.zeros((num_bees, num_days))
            for i, b_states in enumerate(states_matrix):
                for j, s in enumerate(b_states):
                    mat[i, j] = state_num_map.get(str(s).strip(), 1)
            for i in range(num_bees):
                for j in range(num_days):
                    val = mat[i, j]
                    c = color_map.get(val, '#cccccc')
                    ax.add_patch(plt.Rectangle((j, num_bees - 1 - i), 0.90, 0.85, color=c, ec='white', lw=1.5, zorder=2))
                    ax.text(j + 0.45, num_bees - 1 - i + 0.425, states_matrix[i][j], ha='center', va='center', color='white', fontweight='bold', fontsize=11, zorder=3)
            ax.set_xlim(-0.1, float(num_days))
            ax.set_ylim(-0.1, num_bees)
            ax.set_xticks(np.arange(num_days) + 0.45)
            ax.set_xticklabels(day_labels, fontsize=11.5, fontweight='bold', rotation=45, ha='right')
            ax.set_yticks(np.arange(num_bees) + 0.425)
            ax.set_yticklabels([f'Bee {num_bees - i}' for i in range(num_bees)], fontsize=10.5, fontweight='bold')
            ax.set_xlabel('Observation Day', fontsize=12.5, fontweight='bold', labelpad=7)
            ax.set_ylabel('Individual Bee ID', fontsize=12.5, fontweight='bold', labelpad=7)
            legend_elements = [
                Patch(facecolor=hm_alive_color, edgecolor='none', label='Alive (A)'),
                Patch(facecolor=hm_behavior_color, edgecolor='none', label='Behavior Change (B)'),
                Patch(facecolor=hm_dead_color, edgecolor='none', label='Dead (D)')
            ]
            ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=10.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(left=False, bottom=False)
            fig4.tight_layout()
            st.pyplot(fig4)
            get_image_download_link(fig4, 'fig_individual_bees_status.png', 'Download Status Heatmap')
        col_idx += 1

    if "Environmental Conditions" in selected_plots and any(temps):
        with cols[col_idx % 2]:
            st.subheader("Environmental Conditions")
            fig5, ax1 = plt.subplots(figsize=(7.0, 5.2), dpi=300)
            x_days = elapsed_days
            color_temp = '#d95f02'
            ax1.set_xlabel('Observation Time / Days', fontsize=12.5, fontweight='bold', labelpad=7)
            ax1.set_ylabel('Temperature / °C', color=color_temp, fontsize=12.5, fontweight='bold', labelpad=7)
            l1 = ax1.plot(x_days, temps, color=color_temp, marker='o', linewidth=2.2, markersize=7.5, label='Temperature (°C)')
            ax1.tick_params(axis='y', labelcolor=color_temp)
            if any(temps):
                ax1.set_ylim(min(temps)-1, max(temps)+1)
            ax2 = ax1.twinx()
            color_hum = '#1f78b4'
            ax2.set_ylabel('Relative Humidity / %', color=color_hum, fontsize=12.5, fontweight='bold', labelpad=7)
            l2 = ax2.plot(x_days, humidities, color=color_hum, marker='s', linewidth=2.2, markersize=7.5, linestyle='--', label='Relative Humidity (%)')
            ax2.tick_params(axis='y', labelcolor=color_hum)
            if any(humidities):
                ax2.set_ylim(min(humidities)-5, max(humidities)+5)
            ax1.set_xlim(-0.1, max_time * 1.05)
            ax1.set_xticks(elapsed_days)
            ax1.set_xticklabels(day_labels, rotation=45, ha="right")
            ax1.spines['top'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            lines = l1 + l2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left', frameon=True, framealpha=0.9, facecolor='white', edgecolor='none', fontsize=10.5)
            fig5.tight_layout()
            st.pyplot(fig5)
            get_image_download_link(fig5, 'fig_environmental_conditions.png', 'Download Environment Chart')
        col_idx += 1

    if "Combined Multi-Panel Figure" in selected_plots and num_days > 0:
        st.subheader("Combined Multi-Panel Figure")
        fig6 = plt.figure(figsize=(14, 11), dpi=300)
        gs = fig6.add_gridspec(2, 2, hspace=0.30, wspace=0.25)

        ax_a = fig6.add_subplot(gs[0, 0])
        ax_a.fill_between(t_dense, np.clip(fit_mort_curve - 1.96*fit_se, 0, 100), np.clip(fit_mort_curve + 1.96*fit_se, 0, 100), color=primary_color, alpha=0.12)
        ax_a.plot(t_dense, fit_mort_curve, color=primary_color, linewidth=line_width, linestyle=line_style)
        ax_a.errorbar(elapsed_days, mortality_pct, yerr=mortality_se, fmt='o', color=primary_color, ecolor=primary_color, elinewidth=1.6, capsize=4, capthick=1.6, markersize=marker_size, markerfacecolor=primary_color, markeredgecolor='black')
        ax_a.axhline(50, color='#333333', linestyle=':', linewidth=1.2)
        ax_a.text(0.06, 0.90, f"Y = {L_fit:.1f} / (1 + exp(-{k_fit:.2f}(X - {t0_fit:.1f})))\nR² = {r2_mort:.4f}", transform=ax_a.transAxes, fontsize=10.5, verticalalignment='top')
        ax_a.set_xlabel('Observation Time / Days', fontsize=11.5, fontweight='bold')
        ax_a.set_ylabel('Mortality / %', fontsize=11.5, fontweight='bold')
        ax_a.set_xlim(-0.1, max_time * 1.05)
        ax_a.set_ylim(-2, 102)
        ax_a.set_xticks(elapsed_days)
        ax_a.set_xticklabels(day_labels, rotation=45, ha="right")
        ax_a.yaxis.set_major_locator(MultipleLocator(25))
        ax_a.spines['top'].set_visible(False)
        ax_a.spines['right'].set_visible(False)
        ax_a.text(-0.14, 1.04, 'A', transform=ax_a.transAxes, fontsize=15, fontweight='bold')

        ax_b = fig6.add_subplot(gs[0, 1])
        ax_b.fill_between(t_dense, np.clip(fit_surv_curve - 1.96*surv_se_fit, 0, 100), np.clip(fit_surv_curve + 1.96*surv_se_fit, 0, 100), color=secondary_color, alpha=0.15)
        ax_b.plot(t_dense, fit_surv_curve, color=secondary_color, linewidth=line_width, linestyle=line_style)
        ax_b.errorbar(elapsed_days, survival_pct, yerr=survival_se, fmt='o', color=secondary_color, ecolor=secondary_color, elinewidth=1.6, capsize=4, capthick=1.6, markersize=marker_size, markerfacecolor=secondary_color, markeredgecolor='black')
        ax_b.axhline(50, color='#333333', linestyle=':', linewidth=1.2)
        ax_b.text(0.06, 0.32, f"Y = {S0_surv:.1f} * exp(-{a_surv:.4f} * X^{b_surv:.2f})\nR² = {r2_surv:.4f}", transform=ax_b.transAxes, fontsize=10.5, verticalalignment='top')
        ax_b.set_xlabel('Observation Time / Days', fontsize=11.5, fontweight='bold')
        ax_b.set_ylabel('Bee Survival / %', fontsize=11.5, fontweight='bold')
        ax_b.set_xlim(-0.1, max_time * 1.05)
        ax_b.set_ylim(-2, 105)
        ax_b.set_xticks(elapsed_days)
        ax_b.set_xticklabels(day_labels, rotation=45, ha="right")
        ax_b.yaxis.set_major_locator(MultipleLocator(25))
        ax_b.spines['top'].set_visible(False)
        ax_b.spines['right'].set_visible(False)
        ax_b.text(-0.14, 1.04, 'B', transform=ax_b.transAxes, fontsize=15, fontweight='bold')

        ax_c = fig6.add_subplot(gs[1, 0])
        if len(durations) > 0:
            ax_c.step(km_timeline, km_vals, where='post', color='#0066cc', linewidth=line_width, linestyle=line_style, label='KM Estimate')
            ax_c.fill_between(km_timeline, ci_df.iloc[:, 0], ci_df.iloc[:, 1], step='post', color='#0066cc', alpha=0.15, label='95% CI')
            if len(censored_days) > 0:
                ax_c.plot(censored_days, censored_surv, '+', color='#002244', markersize=8.5, markeredgewidth=2, label=f'Censored (n={len(censored_days)})')
        ax_c.axhline(0.5, color='#333333', linestyle=':', linewidth=1.2)
        ax_c.set_xlabel('Observation Time / Days', fontsize=11.5, fontweight='bold')
        ax_c.set_ylabel('Survival Probability', fontsize=11.5, fontweight='bold')
        ax_c.set_xlim(-0.1, max_time * 1.05)
        ax_c.set_ylim(-0.02, 1.05)
        ax_c.set_xticks(elapsed_days)
        ax_c.set_xticklabels(day_labels, rotation=45, ha="right")
        ax_c.spines['top'].set_visible(False)
        ax_c.spines['right'].set_visible(False)
        ax_c.legend(loc='lower left', frameon=False, fontsize=9.5)
        ax_c.text(-0.14, 1.04, 'C', transform=ax_c.transAxes, fontsize=15, fontweight='bold')

        if any(temps):
            ax_d1 = fig6.add_subplot(gs[1, 1])
            ax_d1.set_xlabel('Observation Time / Days', fontsize=11.5, fontweight='bold')
            ax_d1.set_ylabel('Temperature / °C', color=color_temp, fontsize=11.5, fontweight='bold')
            l1 = ax_d1.plot(elapsed_days, temps, color=color_temp, marker='o', linewidth=2.2, markersize=6.5, label='Temperature (°C)')
            ax_d1.tick_params(axis='y', labelcolor=color_temp)
            if any(temps):
                ax_d1.set_ylim(min(temps)-1, max(temps)+1)

            ax_d2 = ax_d1.twinx()
            ax_d2.set_ylabel('Relative Humidity / %', color=color_hum, fontsize=11.5, fontweight='bold')
            l2 = ax_d2.plot(elapsed_days, humidities, color=color_hum, marker='s', linewidth=2.2, markersize=6.5, linestyle='--', label='Humidity (%)')
            ax_d2.tick_params(axis='y', labelcolor=color_hum)
            if any(humidities):
                ax_d2.set_ylim(min(humidities)-5, max(humidities)+5)

            ax_d1.set_xlim(-0.1, max_time * 1.05)
            ax_d1.set_xticks(elapsed_days)
            ax_d1.set_xticklabels(day_labels, rotation=45, ha="right")
            ax_d1.spines['top'].set_visible(False)
            ax_d2.spines['top'].set_visible(False)
            lines_comb = l1 + l2
            ax_d1.legend(lines_comb, [l.get_label() for l in lines_comb], loc='upper left', frameon=True, framealpha=0.9, facecolor='white', edgecolor='none', fontsize=9.5)
            ax_d1.text(-0.14, 1.04, 'D', transform=ax_d1.transAxes, fontsize=15, fontweight='bold')

        st.pyplot(fig6)
        get_image_download_link(fig6, 'fig_combined_publication_summary.png', 'Download Combined Multi-Panel Figure')

    if "Interactive Daily Status (Bar Chart)" in selected_plots and num_days > 0:
        st.markdown("---")
        st.subheader("Interactive Daily Status")

        status_df = pd.DataFrame({
            'Alive': alive_counts,
            'Dead': dead_counts
        }, index=day_labels)

        st.bar_chart(status_df, color=[secondary_color, primary_color])

        fig_bar, ax_bar = plt.subplots(figsize=(6.8, 5.2), dpi=300)
        x_pos = np.arange(num_days)
        bar_w = 0.35
        ax_bar.bar(x_pos - bar_w/2, alive_counts, bar_w, label='Alive', color=secondary_color)
        ax_bar.bar(x_pos + bar_w/2, dead_counts, bar_w, label='Dead', color=primary_color)
        ax_bar.set_xlabel('Observation Day', fontsize=12, fontweight='bold', labelpad=7)
        ax_bar.set_ylabel('Bee Count', fontsize=12, fontweight='bold', labelpad=7)
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(day_labels, rotation=45, ha='right')
        ax_bar.legend(loc='upper right', frameon=False)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        fig_bar.tight_layout()
        get_image_download_link(fig_bar, 'fig_daily_status_bar.png', 'Download Daily Status Chart')

    if "Interactive Cumulative Mortality (Line Chart)" in selected_plots and num_days > 0:
        st.markdown("---")
        st.subheader("Interactive Cumulative Mortality")

        mortality_df = pd.DataFrame({
            'Cumulative Mortality': dead_counts
        }, index=day_labels)

        st.line_chart(mortality_df, color=[primary_color])

        fig_cum, ax_cum = plt.subplots(figsize=(6.8, 5.2), dpi=300)
        ax_cum.plot(elapsed_days, dead_counts, marker='o', color=primary_color, linewidth=line_width, markersize=marker_size, label='Cumulative Mortality')
        ax_cum.set_xlabel('Observation Time / Days', fontsize=12, fontweight='bold', labelpad=7)
        ax_cum.set_ylabel('Cumulative Dead Bees', fontsize=12, fontweight='bold', labelpad=7)
        ax_cum.set_xlim(-0.1, max_time * 1.05)
        ax_cum.set_xticks(elapsed_days)
        ax_cum.set_xticklabels(day_labels, rotation=45, ha='right')
        ax_cum.spines['top'].set_visible(False)
        ax_cum.spines['right'].set_visible(False)
        fig_cum.tight_layout()
        get_image_download_link(fig_cum, 'fig_cumulative_mortality.png', 'Download Cumulative Mortality Chart')

    if "Interactive Status Breakdown (Area Chart)" in selected_plots and num_days > 0:
        st.markdown("---")
        st.subheader("Interactive Status Breakdown")

        behavior_counts = [num_bees - a - d for a, d in zip(alive_counts, dead_counts)]

        area_df = pd.DataFrame({
            'Dead': dead_counts,
            'Behavior Change': behavior_counts,
            'Healthy / Alive': alive_counts
        }, index=day_labels)

        st.area_chart(area_df, color=[hm_dead_color, hm_behavior_color, hm_alive_color])

        fig_area, ax_area = plt.subplots(figsize=(6.8, 5.2), dpi=300)
        ax_area.stackplot(elapsed_days, dead_counts, behavior_counts, alive_counts,
                          labels=['Dead', 'Behavior Change', 'Healthy / Alive'],
                          colors=[hm_dead_color, hm_behavior_color, hm_alive_color], alpha=0.85)
        ax_area.set_xlabel('Observation Time / Days', fontsize=12, fontweight='bold', labelpad=7)
        ax_area.set_ylabel('Number of Bees', fontsize=12, fontweight='bold', labelpad=7)
        ax_area.set_xlim(-0.1, max_time * 1.05)
        ax_area.set_xticks(elapsed_days)
        ax_area.set_xticklabels(day_labels, rotation=45, ha='right')
        ax_area.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')
        ax_area.spines['top'].set_visible(False)
        ax_area.spines['right'].set_visible(False)
        fig_area.tight_layout()
        get_image_download_link(fig_area, 'fig_status_breakdown.png', 'Download Status Breakdown Chart')

    if "Interactive Temp vs Humidity (Scatter Plot)" in selected_plots and num_days > 0:
        st.markdown("---")
        st.subheader("Interactive Temp vs Humidity Scatter")
        if any(temps) and any(humidities):
            scatter_df = pd.DataFrame({
                'Temperature (°C)': temps,
                'Humidity (%)': humidities,
                'Day': day_labels
            })
            st.scatter_chart(scatter_df, x='Temperature (°C)', y='Humidity (%)', color=primary_color)

            fig_scat, ax_scat = plt.subplots(figsize=(6.8, 5.2), dpi=300)
            ax_scat.scatter(temps, humidities, color=primary_color, s=marker_size**2 * 4, edgecolor='black', alpha=0.85)
            for i, txt in enumerate(day_labels):
                ax_scat.annotate(txt, (temps[i], humidities[i]), fontsize=9, xytext=(5, 5), textcoords='offset points')
            ax_scat.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold', labelpad=7)
            ax_scat.set_ylabel('Humidity (%)', fontsize=12, fontweight='bold', labelpad=7)
            ax_scat.spines['top'].set_visible(False)
            ax_scat.spines['right'].set_visible(False)
            fig_scat.tight_layout()
            get_image_download_link(fig_scat, 'fig_temp_vs_humidity.png', 'Download Scatter Plot')
        else:
            st.info("No temperature or humidity data available for scatter plot.")

    if "Consumption Plot (Per Bee)" in selected_plots and num_days > 0:
        st.markdown("---")
        st.subheader("Consumption (µL) Per Bee")
        if len(cons_matrix) > 0 and any(any(v > 0 for v in bee_cons) for bee_cons in cons_matrix):
            cons_dict = {'Day': day_labels}
            for i, c in enumerate(cons_cols):
                cons_dict[c] = cons_matrix[i]
            cons_df = pd.DataFrame(cons_dict).set_index('Day')
            st.line_chart(cons_df)

            fig_cons, ax_cons = plt.subplots(figsize=(8.0, 5.0), dpi=300)
            x_days = elapsed_days
            colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(cons_cols))))
            for i, c in enumerate(cons_cols):
                bee_label = c.replace("Cons", "Bee").replace(" (µL)", "").strip()
                ax_cons.plot(x_days, cons_matrix[i], marker='o', linewidth=1.5, markersize=5, label=bee_label, color=colors[i])

            ax_cons.set_xlabel('Observation Time / Days', fontsize=12, fontweight='bold')
            ax_cons.set_ylabel('Consumption (µL)', fontsize=12, fontweight='bold')
            ax_cons.set_title('Daily Resource Consumption per Bee', fontsize=14, fontweight='bold', pad=15)
            ax_cons.set_xlim(-0.1, max_time * 1.05)
            ax_cons.set_xticks(elapsed_days)
            ax_cons.set_xticklabels(day_labels, rotation=45, ha='right')
            ax_cons.grid(True, linestyle='--', alpha=0.6)

            ax_cons.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()

            st.pyplot(fig_cons)
            get_image_download_link(fig_cons, 'fig_consumption_per_bee.png', 'Download Consumption Plot')
        else:
            st.info("No consumption data available for plotting.")

    if "Statistical Correlation Heatmap" in selected_plots and num_days > 0:
        st.markdown("---")
        st.subheader("Statistical Correlation Heatmap")
        st.markdown("Correlation between Mortality, Survival, Temperature, and Humidity.")

        corr_data = {
            'Mortality (%)': mortality_pct,
            'Survival (%)': survival_pct
        }
        if any(temps): corr_data['Temp'] = temps
        if any(humidities): corr_data['Humidity'] = humidities

        corr_df = pd.DataFrame(corr_data).corr()

        fig_corr, ax_corr = plt.subplots(figsize=(6, 5), dpi=300)
        cax = ax_corr.matshow(corr_df, cmap='coolwarm', vmin=-1, vmax=1)
        fig_corr.colorbar(cax)

        ax_corr.set_xticks(range(len(corr_df.columns)))
        ax_corr.set_yticks(range(len(corr_df.columns)))
        ax_corr.set_xticklabels(corr_df.columns, rotation=45, ha='left')
        ax_corr.set_yticklabels(corr_df.columns)

        for i in range(len(corr_df.columns)):
            for j in range(len(corr_df.columns)):
                ax_corr.text(j, i, f"{corr_df.iloc[i, j]:.2f}", ha='center', va='center', color='white' if abs(corr_df.iloc[i, j]) > 0.5 else 'black')

        st.pyplot(fig_corr)
        get_image_download_link(fig_corr, 'fig_correlation_heatmap.png', 'Download Correlation Heatmap')

    if "Abbott Corrected Mortality Curve" in selected_plots and num_days > 0:
        st.markdown("---")
        st.subheader("Abbott Corrected Mortality")
        fig_abbott, ax_abbott = plt.subplots(figsize=(6.8, 5.4), dpi=300)
        ax_abbott.plot(elapsed_days, abbott_corrected_pct, color=primary_color, marker='o', linewidth=line_width, linestyle=line_style, markersize=marker_size)
        ax_abbott.set_xlabel('Observation Time / Days', fontsize=13, fontweight='bold', labelpad=7)
        ax_abbott.set_ylabel('Corrected Mortality / %', fontsize=13, fontweight='bold', labelpad=7)
        ax_abbott.set_xlim(-0.1, max_time * 1.05)
        ax_abbott.set_ylim(-2, 102)
        ax_abbott.set_xticks(elapsed_days)
        ax_abbott.set_xticklabels(day_labels, rotation=45, ha="right")
        ax_abbott.spines['top'].set_visible(False)
        ax_abbott.spines['right'].set_visible(False)
        fig_abbott.tight_layout()
        st.pyplot(fig_abbott)
        get_image_download_link(fig_abbott, 'fig_abbott_corrected.png', 'Download Abbott Corrected Curve')

    if "Probit Regression (Time-Mortality)" in selected_plots and len(log_time) > 1:
        st.markdown("---")
        st.subheader("Probit Regression (Time-Mortality)")
        fig_probit, ax_probit = plt.subplots(figsize=(6.8, 5.4), dpi=300)
        ax_probit.scatter(log_time, probit_y, color=primary_color, s=marker_size**2)

        # Regression line
        x_line = np.linspace(min(log_time), max(log_time), 100)
        y_line = probit_a + probit_b * x_line
        ax_probit.plot(x_line, y_line, color='#333333', linewidth=line_width, linestyle=line_style)

        eq_text = f"PROBIT(P) = {probit_a:.2f} + {probit_b:.2f}x\nR² = {probit_r**2:.4f}"
        ax_probit.text(0.05, 0.95, eq_text, transform=ax_probit.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#e0e0e0', alpha=0.85))

        ax_probit.set_xlabel('Log10(Observation Time / Days)', fontsize=13, fontweight='bold', labelpad=7)
        ax_probit.set_ylabel('Probit(P)', fontsize=13, fontweight='bold', labelpad=7)
        ax_probit.spines['top'].set_visible(False)
        ax_probit.spines['right'].set_visible(False)
        fig_probit.tight_layout()
        st.pyplot(fig_probit)
        get_image_download_link(fig_probit, 'fig_probit_regression.png', 'Download Probit Regression Curve')

    if plot_images_dict:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for fname, data in plot_images_dict.items():
                zip_file.writestr(fname, data)
        zip_bytes = zip_buf.getvalue()

        zip_button_top.download_button(
            label="📦 Download All Plots (ZIP)",
            data=zip_bytes,
            file_name="all_plots.zip",
            mime="application/zip",
            key="zip_top_btn"
        )

        st.markdown("---")
        st.download_button(
            label="📦 Download All Plots in a Single ZIP File",
            data=zip_bytes,
            file_name="all_plots.zip",
            mime="application/zip",
            key="zip_bottom_btn"
        )



with tab1:
    render_single_batch_tab()

with tab2:
    st.header("📈 Multi-CSV File Probit & Survival Comparative Analysis")
    st.markdown("Upload multiple CSV files to perform comparative **Probit Analysis**, plot time-mortality & survival curves using distinct colors and shapes for each CSV file, and compare **LT₅₀** metrics.")

    uploaded_batches = st.file_uploader(
        "📂 Upload CSV Files for Comparison (Multiple files allowed)",
        type=["csv"],
        accept_multiple_files=True,
        key="multi_batch_uploader_key",
        help="Select multiple CSV files to compare (e.g., control_group.csv, treatment_10uM.csv, etc.)"
    )

    raw_batch_dict = {}

    if uploaded_batches:
        for f in uploaded_batches:
            try:
                b_df = pd.read_csv(f, sep=None, engine='python')
                b_name = f.name  # Use exact file name (e.g., experiment_group_A.csv)
                raw_batch_dict[b_name] = b_df
            except Exception as ex:
                st.error(f"Error reading file {f.name}: {ex}")
    else:
        st.info("💡 **Showing sample CSV files below.** Upload your custom CSV files above to compare your datasets.")

        d1 = """Day,Bee 1,Bee 2,Bee 3,Bee 4,Bee 5,Bee 6,Bee 7,Bee 8,Bee 9,Bee 10
Day 1,A,A,A,A,A,A,A,A,A,A
Day 2,A,A,A,A,A,A,A,A,A,A
Day 3,A,A,A,A,A,A,A,A,A,A
Day 4,A,A,A,A,A,D,A,A,A,A
Day 5,A,A,A,A,A,D,A,A,A,A
Day 6,A,D,A,A,A,D,A,A,A,A
Day 7,A,D,A,A,A,D,D,A,A,A
Day 8,A,D,A,A,A,D,D,A,A,D"""

        d2 = """Day,Bee 1,Bee 2,Bee 3,Bee 4,Bee 5,Bee 6,Bee 7,Bee 8,Bee 9,Bee 10
Day 1,A,A,A,A,A,A,A,A,A,A
Day 2,A,D,A,A,A,A,A,A,A,A
Day 3,A,D,A,D,A,A,A,A,A,A
Day 4,A,D,A,D,D,A,A,A,A,A
Day 5,A,D,D,D,D,D,A,A,A,A
Day 6,D,D,D,D,D,D,A,D,A,A
Day 7,D,D,D,D,D,D,A,D,D,A
Day 8,D,D,D,D,D,D,D,D,D,A"""

        d3 = """Day,Bee 1,Bee 2,Bee 3,Bee 4,Bee 5,Bee 6,Bee 7,Bee 8,Bee 9,Bee 10
Day 1,A,A,D,A,A,D,A,A,D,A
Day 2,D,A,D,A,D,D,A,A,D,A
Day 3,D,D,D,A,D,D,D,A,D,A
Day 4,D,D,D,D,D,D,D,A,D,D
Day 5,D,D,D,D,D,D,D,D,D,D
Day 6,D,D,D,D,D,D,D,D,D,D
Day 7,D,D,D,D,D,D,D,D,D,D
Day 8,D,D,D,D,D,D,D,D,D,D"""

        raw_batch_dict["sample_control_0uM.csv"] = pd.read_csv(io.StringIO(d1))
        raw_batch_dict["sample_treatment_10uM.csv"] = pd.read_csv(io.StringIO(d2))
        raw_batch_dict["sample_treatment_50uM.csv"] = pd.read_csv(io.StringIO(d3))

    if raw_batch_dict:
        MARKER_OPTIONS = {
            "Circle (●)": "o",
            "Square (■)": "s",
            "Triangle Up (▲)": "^",
            "Diamond (◆)": "D",
            "Triangle Down (▼)": "v",
            "Star (★)": "*",
            "Pentagon (⬟)": "p",
            "Hexagon (⬢)": "h",
            "Cross (✖)": "X",
            "Plus (✚)": "P",
            "Triangle Left (◄)": "<",
            "Triangle Right (►)": ">"
        }
        marker_names = list(MARKER_OPTIONS.keys())

        DEFAULT_COLORS = [
            "#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6",
            "#ec4899", "#06b6d4", "#f97316", "#84cc16", "#6366f1"
        ]

        LINE_STYLES = {"Solid": "-", "Dashed": "--", "Dotted": ":", "Dash-Dot": "-."}

        with st.expander("🎨 Customize File Colors, Shapes & Display Names", expanded=True):
            st.markdown("Set custom labels, distinct colors, marker shapes, line styles, and legend layout for each CSV file:")
            
            col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)
            with col_ctrl1:
                jitter_val = st.slider("Point Separation / Anti-Overlap Offset", min_value=0.0, max_value=0.05, value=0.015, step=0.003, help="Offsets data points horizontally side-by-side so markers from different CSV files don't overlap directly on top of each other.")
            with col_ctrl2:
                legend_detail = st.selectbox("Legend Detail Level", ["File Name + Equation + R²", "File Name + R² Only", "File Name Only"], index=0)
            with col_ctrl3:
                legend_loc_opt = st.selectbox("Legend Placement", ["Top (Horizontal)", "Bottom (Horizontal)", "Right (Outside)"], index=0)
            with col_ctrl4:
                show_grid = st.checkbox("Show Chart Gridlines", value=True)

            st.markdown("---")
            batch_config = []
            b_keys = list(raw_batch_dict.keys())

            c_cols = st.columns(min(3, max(1, len(b_keys))))
            for idx, orig_name in enumerate(b_keys):
                col_i = c_cols[idx % len(c_cols)]
                with col_i:
                    st.markdown(f"**File {idx+1}: `{orig_name}`**")
                    custom_label = st.text_input(f"Display Label", value=orig_name, key=f"bname_{idx}")
                    default_c = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
                    custom_color = st.color_picker(f"Color", default_c, key=f"bcol_{idx}")
                    
                    default_shape_idx = idx % len(marker_names)
                    custom_shape_name = st.selectbox(f"Shape (Marker)", marker_names, index=default_shape_idx, key=f"bshape_{idx}")
                    custom_shape = MARKER_OPTIONS[custom_shape_name]

                    custom_line_name = st.selectbox(f"Line Style", list(LINE_STYLES.keys()), index=0, key=f"bline_{idx}")
                    custom_linestyle = LINE_STYLES[custom_line_name]

                    batch_config.append({
                        'orig_name': orig_name,
                        'name': custom_label,
                        'color': custom_color,
                        'marker': custom_shape,
                        'marker_name': custom_shape_name,
                        'linestyle': custom_linestyle,
                        'raw_df': raw_batch_dict[orig_name]
                    })

        processed_batches = []
        for bcfg in batch_config:
            p_res = parse_batch_df(bcfg['raw_df'], abbott_c=abbott_c)
            p_res.update({
                'name': bcfg['name'],
                'color': bcfg['color'],
                'marker': bcfg['marker'],
                'linestyle': bcfg['linestyle']
            })
            processed_batches.append(p_res)

        st.markdown("### 📊 Multi-CSV Key Summary")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{len(processed_batches)}</div><div class="kpi-label">CSV Files</div></div>', unsafe_allow_html=True)
        with k2:
            valid_lt50 = [b['lt50'] for b in processed_batches if not np.isnan(b['lt50'])]
            min_lt50_str = f"{min(valid_lt50):.2f}d" if valid_lt50 else "N/A"
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{min_lt50_str}</div><div class="kpi-label">Fastest LT50</div></div>', unsafe_allow_html=True)
        with k3:
            max_lt50_str = f"{max(valid_lt50):.2f}d" if valid_lt50 else "N/A"
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{max_lt50_str}</div><div class="kpi-label">Slowest LT50</div></div>', unsafe_allow_html=True)
        with k4:
            r2_list = [b['r_squared'] for b in processed_batches]
            avg_r2 = f"{np.mean(r2_list):.4f}" if r2_list else "N/A"
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{avg_r2}</div><div class="kpi-label">Avg Probit R²</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        multi_plot_images = {}
        num_files = len(processed_batches)

        # 1. Probit Regression CSV File Comparison Plot
        st.subheader("1. Probit Regression Time-Mortality Comparison Across CSV Files")
        fig_p_multi, ax_p = plt.subplots(figsize=(10.5, 6.0), dpi=300)

        if show_grid:
            ax_p.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', zorder=1)

        all_log_times = []
        for b in processed_batches:
            if len(b['log_time']) > 0:
                all_log_times.extend(b['log_time'])

        x_min = max(-0.02, min(all_log_times) - 0.04) if all_log_times else 0.0
        x_max = max(all_log_times) + 0.08 if all_log_times else 1.0
        x_fit_line = np.linspace(x_min, x_max, 100)

        legend_handles_probit = []

        for idx, b in enumerate(processed_batches):
            # Calculate horizontal offset to prevent point stacking/overlap
            offset = (idx - (num_files - 1) / 2.0) * jitter_val
            x_jittered = b['log_time'] + offset

            # Plot scatter points with distinct shape & color
            ax_p.scatter(
                x_jittered, b['probit_y'],
                color=b['color'], marker=b['marker'],
                s=marker_size**2 * 1.5, edgecolor='black', linewidth=1.1,
                zorder=4
            )

            # Regression line
            if len(b['log_time']) > 1:
                y_fit_line = b['intercept_a'] + b['slope_b'] * x_fit_line
                ax_p.plot(
                    x_fit_line, y_fit_line,
                    color=b['color'], linestyle=b['linestyle'], linewidth=line_width,
                    zorder=3
                )

            # Single combined legend entry (Marker shape + Line style + Label)
            if legend_detail == "File Name + Equation + R²":
                lbl = f"{b['name']} (Y={b['intercept_a']:.2f}+{b['slope_b']:.2f}x, R²={b['r_squared']:.4f})"
            elif legend_detail == "File Name + R² Only":
                lbl = f"{b['name']} (R²={b['r_squared']:.4f})"
            else:
                lbl = b['name']

            legend_handles_probit.append(
                Line2D([0], [0], color=b['color'], marker=b['marker'], linestyle=b['linestyle'],
                       linewidth=line_width, markersize=marker_size, markeredgecolor='black',
                       label=lbl)
            )

        legend_handles_probit.append(
            Line2D([0], [0], color='#64748b', linestyle='--', linewidth=1.5, label='LT50 Reference (Probit = 5.0)')
        )

        ax_p.axhline(5.0, color='#64748b', linestyle='--', linewidth=1.5, zorder=2)
        ax_p.set_xlim(x_min, x_max)
        ax_p.set_ylim(2.5, 9.2)
        ax_p.yaxis.set_major_locator(MultipleLocator(1.0))

        ax_p.set_xlabel('Log10(Observation Time / Days)', fontsize=12.5, fontweight='bold', labelpad=8)
        ax_p.set_ylabel('Probit Value (P)', fontsize=12.5, fontweight='bold', labelpad=8)
        ax_p.set_title('Comparative Probit Regression Across CSV Files', fontsize=14, fontweight='bold', pad=14)
        ax_p.spines['top'].set_visible(False)
        ax_p.spines['right'].set_visible(False)

        # Place Legend based on user option
        if legend_loc_opt == "Top (Horizontal)":
            ax_p.legend(
                handles=legend_handles_probit, loc='lower center', bbox_to_anchor=(0.5, 1.02),
                ncol=min(3, max(1, num_files + 1)), frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=9.0
            )
            fig_p_multi.tight_layout()
        elif legend_loc_opt == "Bottom (Horizontal)":
            ax_p.legend(
                handles=legend_handles_probit, loc='upper center', bbox_to_anchor=(0.5, -0.16),
                ncol=min(3, max(1, num_files + 1)), frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=9.0
            )
            fig_p_multi.tight_layout()
        else: # Right (Outside)
            ax_p.legend(
                handles=legend_handles_probit, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=9.0
            )
            fig_p_multi.tight_layout(rect=[0, 0, 0.72, 1.0])

        st.pyplot(fig_p_multi)

        buf_p = io.BytesIO()
        fig_p_multi.savefig(buf_p, format="png", dpi=300, bbox_inches='tight')
        multi_plot_images['probit_multi_csv_comparison.png'] = buf_p.getvalue()
        st.download_button("📥 Download Probit Comparison Chart (PNG)", data=buf_p.getvalue(), file_name="probit_multi_csv_comparison.png", mime="image/png")
        plt.close(fig_p_multi)

        st.markdown("---")

        cols_m = st.columns(2)

        # Gather max observation time across all datasets for uniform linear scale
        all_times_flat = []
        for b in processed_batches:
            if len(b['times_numeric']) > 0:
                all_times_flat.extend(b['times_numeric'])
        max_t_flat = max(all_times_flat) if all_times_flat else 8.0

        # 2. Mortality % Comparison
        with cols_m[0]:
            st.subheader("2. Mortality Curve Comparison")
            fig_m_multi, ax_m = plt.subplots(figsize=(7.5, 5.2), dpi=300)
            if show_grid:
                ax_m.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', zorder=1)

            m_handles = []
            for idx, b in enumerate(processed_batches):
                offset = (idx - (num_files - 1) / 2.0) * (jitter_val * 6.0)
                x_days_jittered = b['times_numeric'] + offset
                ax_m.plot(
                    x_days_jittered, b['abbott_corrected_pct'],
                    color=b['color'], marker=b['marker'], linestyle=b['linestyle'],
                    linewidth=line_width, markersize=marker_size, markeredgecolor='black',
                    zorder=3
                )
                m_handles.append(
                    Line2D([0], [0], color=b['color'], marker=b['marker'], linestyle=b['linestyle'],
                           linewidth=line_width, markersize=marker_size, markeredgecolor='black',
                           label=b['name'])
                )
            m_handles.append(Line2D([0], [0], color='#64748b', linestyle=':', linewidth=1.4, label='50% Mortality'))
            ax_m.axhline(50.0, color='#64748b', linestyle=':', linewidth=1.4, zorder=2)
            ax_m.set_xlabel('Observation Time / Days', fontsize=12, fontweight='bold', labelpad=7)
            ax_m.set_ylabel('Abbott Corrected Mortality / %', fontsize=12, fontweight='bold', labelpad=7)
            ax_m.set_xlim(0.8, max_t_flat * 1.05)
            ax_m.set_ylim(-2, 105)
            ax_m.spines['top'].set_visible(False)
            ax_m.spines['right'].set_visible(False)

            if legend_loc_opt == "Right (Outside)":
                ax_m.legend(handles=m_handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=8.5)
                fig_m_multi.tight_layout(rect=[0, 0, 0.72, 1.0])
            else:
                ax_m.legend(handles=m_handles, loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=8.5)
                fig_m_multi.tight_layout()

            st.pyplot(fig_m_multi)

            buf_m = io.BytesIO()
            fig_m_multi.savefig(buf_m, format="png", dpi=300, bbox_inches='tight')
            multi_plot_images['mortality_multi_csv_comparison.png'] = buf_m.getvalue()
            st.download_button("📥 Download Mortality Comparison Chart", data=buf_m.getvalue(), file_name="mortality_multi_csv_comparison.png", mime="image/png")
            plt.close(fig_m_multi)

        # 3. Survival % Comparison
        with cols_m[1]:
            st.subheader("3. Survival Curve Comparison")
            fig_s_multi, ax_s = plt.subplots(figsize=(7.5, 5.2), dpi=300)
            if show_grid:
                ax_s.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', zorder=1)

            s_handles = []
            for idx, b in enumerate(processed_batches):
                offset = (idx - (num_files - 1) / 2.0) * (jitter_val * 6.0)
                x_days_jittered = b['times_numeric'] + offset
                ax_s.plot(
                    x_days_jittered, b['survival_pct'],
                    color=b['color'], marker=b['marker'], linestyle=b['linestyle'],
                    linewidth=line_width, markersize=marker_size, markeredgecolor='black',
                    zorder=3
                )
                s_handles.append(
                    Line2D([0], [0], color=b['color'], marker=b['marker'], linestyle=b['linestyle'],
                           linewidth=line_width, markersize=marker_size, markeredgecolor='black',
                           label=b['name'])
                )
            s_handles.append(Line2D([0], [0], color='#64748b', linestyle=':', linewidth=1.4, label='50% Survival'))
            ax_s.axhline(50.0, color='#64748b', linestyle=':', linewidth=1.4, zorder=2)
            ax_s.set_xlabel('Observation Time / Days', fontsize=12, fontweight='bold', labelpad=7)
            ax_s.set_ylabel('Survival Rate / %', fontsize=12, fontweight='bold', labelpad=7)
            ax_s.set_xlim(0.8, max_t_flat * 1.05)
            ax_s.set_ylim(-2, 105)
            ax_s.spines['top'].set_visible(False)
            ax_s.spines['right'].set_visible(False)

            if legend_loc_opt == "Right (Outside)":
                ax_s.legend(handles=s_handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=8.5)
                fig_s_multi.tight_layout(rect=[0, 0, 0.72, 1.0])
            else:
                ax_s.legend(handles=s_handles, loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=8.5)
                fig_s_multi.tight_layout()

            st.pyplot(fig_s_multi)

            buf_s = io.BytesIO()
            fig_s_multi.savefig(buf_s, format="png", dpi=300, bbox_inches='tight')
            multi_plot_images['survival_multi_csv_comparison.png'] = buf_s.getvalue()
            st.download_button("📥 Download Survival Comparison Chart", data=buf_s.getvalue(), file_name="survival_multi_csv_comparison.png", mime="image/png")
            plt.close(fig_s_multi)

        # 4. LT50 Bar Chart Comparison
        st.markdown("---")
        st.subheader("4. Lethal Time 50% (LT₅₀) Bar Chart Comparison")
        fig_lt, ax_lt = plt.subplots(figsize=(8.0, 4.8), dpi=300)
        if show_grid:
            ax_lt.grid(True, linestyle='--', alpha=0.35, color='#cccccc', axis='y', zorder=1)

        b_names = [b['name'] for b in processed_batches]
        lt50_vals = [b['lt50'] for b in processed_batches]
        b_colors = [b['color'] for b in processed_batches]

        bars = ax_lt.bar(b_names, lt50_vals, color=b_colors, edgecolor='black', width=0.45)
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h) and h > 0:
                ax_lt.text(bar.get_x() + bar.get_width()/2.0, h + 0.1, f"{h:.2f} Days", ha='center', va='bottom', fontweight='bold', fontsize=10)

        ax_lt.set_ylabel('LT50 (Days)', fontsize=12, fontweight='bold')
        ax_lt.set_title('Calculated LT50 across CSV Files', fontsize=14, fontweight='bold', pad=12)
        ax_lt.spines['top'].set_visible(False)
        ax_lt.spines['right'].set_visible(False)
        fig_lt.tight_layout()
        st.pyplot(fig_lt)

        buf_lt = io.BytesIO()
        fig_lt.savefig(buf_lt, format="png", dpi=300)
        multi_plot_images['lt50_bar_comparison.png'] = buf_lt.getvalue()
        plt.close(fig_lt)

        # Comparative Summary Metrics Table
        st.markdown("---")
        st.subheader("📋 Multi-CSV Probit Comparative Metrics Table")
        summary_rows = []
        for b in processed_batches:
            lt50_str = f"{b['lt50']:.2f}" if not np.isnan(b['lt50']) else "N/A"
            lt90_str = f"{b['lt90']:.2f}" if not np.isnan(b['lt90']) else "N/A"
            summary_rows.append({
                "CSV File Name": b['name'],
                "Sample Size (N)": b['num_bees'],
                "Days Observed": b['num_days'],
                "Probit Model": f"Y = {b['intercept_a']:.2f} + {b['slope_b']:.2f}x",
                "Slope (b)": round(b['slope_b'], 4),
                "Intercept (a)": round(b['intercept_a'], 4),
                "R² Score": round(b['r_squared'], 4),
                "LT50 (Days)": lt50_str,
                "LT90 (Days)": lt90_str,
                "Final Mortality (%)": f"{b['mortality_pct'][-1]:.1f}%" if len(b['mortality_pct']) > 0 else "N/A"
            })

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True)

        c_csv, c_zip = st.columns(2)
        with c_csv:
            csv_str = summary_df.to_csv(index=False)
            st.download_button("📊 Download Comparative Summary Table (CSV)", data=csv_str, file_name="multi_csv_probit_summary.csv", mime="text/csv")

        with c_zip:
            if multi_plot_images:
                zip_multi_buf = io.BytesIO()
                with zipfile.ZipFile(zip_multi_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for fn, data in multi_plot_images.items():
                        zf.writestr(fn, data)
                st.download_button("📦 Download All Multi-CSV Plots (ZIP)", data=zip_multi_buf.getvalue(), file_name="multi_csv_plots.zip", mime="application/zip")


