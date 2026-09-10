import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch
from scipy.optimize import curve_fit
from lifelines import KaplanMeierFitter
import io
import base64

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

st.set_page_config(page_title="Bee Observation Data Visualization", layout="wide")

st.sidebar.header("Graph Editor Settings")
primary_color = st.sidebar.color_picker("Primary Color (Mortality)", "#e60000")
secondary_color = st.sidebar.color_picker("Secondary Color (Survival)", "#ff9900")
line_width = st.sidebar.slider("Line Width", min_value=1.0, max_value=5.0, value=2.4)
marker_size = st.sidebar.slider("Marker Size", min_value=3.0, max_value=12.0, value=7.0)
font_size = st.sidebar.slider("Font Size", min_value=8, max_value=18, value=11)

st.sidebar.markdown("---")
st.sidebar.subheader("Line Style")
line_style_map = {"Solid": "-", "Dashed": "--", "Dotted": ":", "Dash-Dot": "-."}
selected_line_style = st.sidebar.selectbox("Select Line Style", list(line_style_map.keys()))
line_style = line_style_map[selected_line_style]

st.sidebar.markdown("---")
st.sidebar.subheader("Heatmap Colors")
hm_alive_color = st.sidebar.color_picker("Alive Color", "#2ca02c")
hm_behavior_color = st.sidebar.color_picker("Behavior Change Color", "#ff7f0e")
hm_dead_color = st.sidebar.color_picker("Dead Color", "#d62728")

plt.rcParams['font.size'] = font_size

st.title("Bee Observation Data Visualization")
st.markdown("Enter your metadata, fill the observation sheet below, and generate reports.")

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

DEFAULT_CSV = """Day,Date/time,Bee 1,Bee 2,Bee 3,Bee 4,Bee 5,Bee 6,Bee 7,Bee 8,Bee 9,Bee 10,Bee 11,Bee 12,Bee 13,Bee 14,Temp (°C),Humidity,Treatment,Observations/notes
Day 1,02.09.2026-16:00,A,D,A,A,A,A,A,A,D,A,A,A,A,D,23.3,55,Sucrose solution,
Day 2,03.09.2026-16:00,A,D,A,A,A,A,A,A,D,A,A,A,A,D,24.4,45,Sucrose solution,
Day 3,04.09.2026-16:00,A,D,A,A,B,A,A,A,D,A,A,A,A,D,24.7,58,Sucrose solution,
Day 4,05.09.2026-16:00,A,D,A,A,B,A,A,A,D,A,A,A,A,D,23.8,61,Sucrose solution,
Day 5,06.09.2026-16:00,A,D,D,A,B,A,A,A,D,A,A,D,A,D,23.7,51,Sucrose solution,
Day 6,07.09.2026-16:00,D,D,D,A,B,A,A,A,D,A,A,D,A,D,24.5,64,Sucrose solution,
Day 7,08.09.2026-16:00,D,D,D,B,B,A,A,A,D,A,A,D,A,D,24.8,62,Sucrose solution,
Day 8,09.09.2026-16:00,D,D,D,B,B,A,A,A,D,A,A,D,A,D,23.4,59,Sucrose solution,"""

if 'df' not in st.session_state:
    st.session_state.df = pd.read_csv(io.StringIO(DEFAULT_CSV))
    # ensure string types
    st.session_state.df = st.session_state.df.astype(str)
    st.session_state.df.replace("nan", "", inplace=True)

st.write("### Daily Individual Observations")

uploaded_file = st.file_uploader("Upload CSV file (Optional)", type=["csv"])
if uploaded_file is not None:
    if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        try:
            new_df = pd.read_csv(uploaded_file, sep=None, engine='python')
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
        try:
            temp_idx = list(st.session_state.df.columns).index("Temp (°C)")
        except ValueError:
            temp_idx = len(st.session_state.df.columns)
        st.session_state.df.insert(temp_idx, new_col_name, "A")
        st.rerun()
with col2:
    if st.button("🔄 Reset Data"):
        del st.session_state['df']
        st.rerun()
with col3:
    st.markdown("You can edit the cells below. **To add a new Day (row)**, scroll to the bottom of the table and click the empty row.")

edited_df = st.data_editor(st.session_state.df, num_rows="dynamic", use_container_width=True)
st.session_state.df = edited_df
df = edited_df.copy()


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

day_labels = df['Day'].tolist() if 'Day' in df.columns else [f"Day {i+1}" for i in range(len(df))]
num_days = len(df)
day_indices = np.arange(num_days)

bee_cols = [c for c in df.columns if str(c).lower().startswith('bee')]
num_bees = len(bee_cols)

temps_raw = df['Temp (°C)'].tolist() if 'Temp (°C)' in df.columns else [0]*num_days
humidities_raw = df['Humidity'].tolist() if 'Humidity' in df.columns else [0]*num_days
temps = [float(x) if str(x).replace('.','',1).isdigit() else 0.0 for x in temps_raw]
humidities = [float(x) if str(x).replace('.','',1).isdigit() else 0.0 for x in humidities_raw]

states_matrix = []
for bc in bee_cols:
    states_matrix.append(df[bc].fillna("A").tolist())

durations = []
events = []

for b_states in states_matrix:
    if 'D' in b_states:
        durations.append(b_states.index('D'))
        events.append(1)
    else:
        durations.append(num_days - 1 if num_days > 0 else 0)
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

# Binomial Standard Error for proportions
mortality_se = np.sqrt((mortality_pct / 100.0) * (1.0 - mortality_pct / 100.0) / max(1, num_bees)) * 100.0
survival_se = np.sqrt((survival_pct / 100.0) * (1.0 - survival_pct / 100.0) / max(1, num_bees)) * 100.0

# FIT STATISTICAL MODELS
def logistic_mort(t, L, k, t0):
    return L / (1.0 + np.exp(-k * (t - t0)))

try:
    popt_mort, _ = curve_fit(
        logistic_mort, day_indices, mortality_pct,
        p0=[60.0, 0.5, 5.5],
        bounds=([20.0, 0.01, 0.0], [100.0, 5.0, 15.0]),
        maxfev=10000
    )
except:
    popt_mort = [60.0, 0.5, 5.5]

L_fit, k_fit, t0_fit = popt_mort
residuals_mort = mortality_pct - logistic_mort(day_indices, *popt_mort)
ss_res_mort = np.sum(residuals_mort**2)
ss_tot_mort = np.sum((mortality_pct - np.mean(mortality_pct))**2)
r2_mort = 1.0 - (ss_res_mort / ss_tot_mort) if ss_tot_mort != 0 else 1.0

def survival_decay(t, S0, a, b):
    return S0 * np.exp(-a * (t**b))

try:
    popt_surv, _ = curve_fit(
        survival_decay, day_indices, survival_pct,
        p0=[78.57, 0.001, 3.0],
        bounds=([50.0, 1e-6, 0.1], [100.0, 2.0, 10.0]),
        maxfev=10000
    )
except:
    popt_surv = [78.57, 0.001, 3.0]

S0_surv, a_surv, b_surv = popt_surv
residuals_surv = survival_pct - survival_decay(day_indices, *popt_surv)
ss_res_surv = np.sum(residuals_surv**2)
ss_tot_surv = np.sum((survival_pct - np.mean(survival_pct))**2)
r2_surv = 1.0 - (ss_res_surv / ss_tot_surv) if ss_tot_surv != 0 else 1.0

kmf = KaplanMeierFitter()
if len(durations) > 0:
    kmf.fit(durations, event_observed=events, timeline=np.linspace(0, max(durations)+1, max(durations)*20 if max(durations)>0 else 140))

t_dense = np.linspace(0, num_days-0.5, 250) if num_days > 0 else np.array([])
fit_mort_curve = logistic_mort(t_dense, *popt_mort) if num_days > 0 else np.array([])
fit_surv_curve = survival_decay(t_dense, *popt_surv) if num_days > 0 else np.array([])

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
    "Statistical Correlation Heatmap"
]
selected_plots = st.multiselect("Choose which plots to view and download:", plot_options, default=plot_options)

def get_image_download_link(fig, filename, text):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    st.download_button(label=text, data=buf, file_name=filename, mime="image/png")

cols = st.columns(2)
col_idx = 0

if "Mortality Curve" in selected_plots and num_days > 0:
    with cols[col_idx % 2]:
        st.subheader("Mortality Curve")
        fig1, ax = plt.subplots(figsize=(6.8, 5.4), dpi=300)
        fit_se = np.std(residuals_mort)
        ax.fill_between(t_dense, np.clip(fit_mort_curve - 1.96*fit_se, 0, 100), np.clip(fit_mort_curve + 1.96*fit_se, 0, 100), color=primary_color, alpha=0.12, label='95% Confidence Band')
        ax.plot(t_dense, fit_mort_curve, color=primary_color, linewidth=line_width, linestyle=line_style, label='Logistic Model Fit')
        ax.errorbar(day_indices, mortality_pct, yerr=mortality_se, fmt='o', color=primary_color, ecolor=primary_color, elinewidth=1.6, capsize=4, capthick=1.6, markersize=marker_size, markerfacecolor=primary_color, markeredgecolor='black', label='Observed (Mean ± SE)')
        ax.axhline(50, color='#333333', linestyle=':', linewidth=1.2)
        eq_text = f"$Y = \\frac{{{L_fit:.1f}}}{{1 + e^{{-{k_fit:.2f}(X - {t0_fit:.1f})}}}}$\n$R^2 = {r2_mort:.4f}$"
        ax.text(0.06, 0.90, eq_text, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#e0e0e0', alpha=0.85))
        ax.set_xlabel('Observation Time / Days', fontsize=13, fontweight='bold', labelpad=7)
        ax.set_ylabel('Mortality / %', fontsize=13, fontweight='bold', labelpad=7)
        ax.set_xlim(-0.3, num_days-0.5)
        ax.set_ylim(-2, 102)
        ax.set_xticks(day_indices)
        ax.set_xticklabels(day_labels)
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
        ax.errorbar(day_indices, survival_pct, yerr=survival_se, fmt='o', color=secondary_color, ecolor=secondary_color, elinewidth=1.6, capsize=4, capthick=1.6, markersize=marker_size, markerfacecolor=secondary_color, markeredgecolor='black', label='Observed (Mean ± SE)')
        ax.axhline(50, color='#333333', linestyle=':', linewidth=1.2)
        eq_surv_text = f"$Y = {S0_surv:.1f} \\times e^{{-{a_surv:.4f} X^{{{b_surv:.2f}}}}}$\n$R^2 = {r2_surv:.4f}$"
        ax.text(0.06, 0.32, eq_surv_text, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#e0e0e0', alpha=0.85))
        ax.set_xlabel('Observation Time / Days', fontsize=13, fontweight='bold', labelpad=7)
        ax.set_ylabel('Bee Survival / %', fontsize=13, fontweight='bold', labelpad=7)
        ax.set_xlim(-0.3, num_days-0.5)
        ax.set_ylim(-2, 105)
        ax.set_xticks(day_indices)
        ax.set_xticklabels(day_labels)
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
        ax.set_xlim(-0.2, num_days-0.5)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xticks(day_indices)
        ax.set_xticklabels(day_labels)
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
        ax.set_xticklabels(day_labels, fontsize=11.5, fontweight='bold')
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
        x_days = np.arange(num_days)
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
        ax1.set_xlim(-0.3, num_days-0.7)
        ax1.set_xticks(day_indices)
        ax1.set_xticklabels(day_labels)
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
    ax_a.errorbar(day_indices, mortality_pct, yerr=mortality_se, fmt='o', color=primary_color, ecolor=primary_color, elinewidth=1.6, capsize=4, capthick=1.6, markersize=marker_size, markerfacecolor=primary_color, markeredgecolor='black')
    ax_a.axhline(50, color='#333333', linestyle=':', linewidth=1.2)
    ax_a.text(0.06, 0.90, f"$Y = \\frac{{{L_fit:.1f}}}{{1 + e^{{-{k_fit:.2f}(X - {t0_fit:.1f})}}}}$\n$R^2 = {r2_mort:.4f}$", transform=ax_a.transAxes, fontsize=10.5, verticalalignment='top')
    ax_a.set_xlabel('Observation Time / Days', fontsize=11.5, fontweight='bold')
    ax_a.set_ylabel('Mortality / %', fontsize=11.5, fontweight='bold')
    ax_a.set_xlim(-0.3, num_days-0.5)
    ax_a.set_ylim(-2, 102)
    ax_a.set_xticks(day_indices)
    ax_a.set_xticklabels(day_labels)
    ax_a.yaxis.set_major_locator(MultipleLocator(25))
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.text(-0.14, 1.04, 'A', transform=ax_a.transAxes, fontsize=15, fontweight='bold')
    
    ax_b = fig6.add_subplot(gs[0, 1])
    ax_b.fill_between(t_dense, np.clip(fit_surv_curve - 1.96*surv_se_fit, 0, 100), np.clip(fit_surv_curve + 1.96*surv_se_fit, 0, 100), color=secondary_color, alpha=0.15)
    ax_b.plot(t_dense, fit_surv_curve, color=secondary_color, linewidth=line_width, linestyle=line_style)
    ax_b.errorbar(day_indices, survival_pct, yerr=survival_se, fmt='o', color=secondary_color, ecolor=secondary_color, elinewidth=1.6, capsize=4, capthick=1.6, markersize=marker_size, markerfacecolor=secondary_color, markeredgecolor='black')
    ax_b.axhline(50, color='#333333', linestyle=':', linewidth=1.2)
    ax_b.text(0.06, 0.32, f"$Y = {S0_surv:.1f} \\times e^{{-{a_surv:.4f} X^{{{b_surv:.2f}}}}}$\n$R^2 = {r2_surv:.4f}$", transform=ax_b.transAxes, fontsize=10.5, verticalalignment='top')
    ax_b.set_xlabel('Observation Time / Days', fontsize=11.5, fontweight='bold')
    ax_b.set_ylabel('Bee Survival / %', fontsize=11.5, fontweight='bold')
    ax_b.set_xlim(-0.3, num_days-0.5)
    ax_b.set_ylim(-2, 105)
    ax_b.set_xticks(day_indices)
    ax_b.set_xticklabels(day_labels)
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
    ax_c.set_xlim(-0.2, num_days-0.5)
    ax_c.set_ylim(-0.02, 1.05)
    ax_c.set_xticks(day_indices)
    ax_c.set_xticklabels(day_labels)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.legend(loc='lower left', frameon=False, fontsize=9.5)
    ax_c.text(-0.14, 1.04, 'C', transform=ax_c.transAxes, fontsize=15, fontweight='bold')
    
    if any(temps):
        ax_d1 = fig6.add_subplot(gs[1, 1])
        ax_d1.set_xlabel('Observation Time / Days', fontsize=11.5, fontweight='bold')
        ax_d1.set_ylabel('Temperature / °C', color=color_temp, fontsize=11.5, fontweight='bold')
        l1 = ax_d1.plot(x_days, temps, color=color_temp, marker='o', linewidth=2.2, markersize=6.5, label='Temperature (°C)')
        ax_d1.tick_params(axis='y', labelcolor=color_temp)
        if any(temps):
            ax_d1.set_ylim(min(temps)-1, max(temps)+1)
        
        ax_d2 = ax_d1.twinx()
        ax_d2.set_ylabel('Relative Humidity / %', color=color_hum, fontsize=11.5, fontweight='bold')
        l2 = ax_d2.plot(x_days, humidities, color=color_hum, marker='s', linewidth=2.2, markersize=6.5, linestyle='--', label='Humidity (%)')
        ax_d2.tick_params(axis='y', labelcolor=color_hum)
        if any(humidities):
            ax_d2.set_ylim(min(humidities)-5, max(humidities)+5)
        
        ax_d1.set_xlim(-0.3, num_days-0.7)
        ax_d1.set_xticks(day_indices)
        ax_d1.set_xticklabels(day_labels)
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

if "Interactive Cumulative Mortality (Line Chart)" in selected_plots and num_days > 0:
    st.markdown("---")
    st.subheader("Interactive Cumulative Mortality")
    
    mortality_df = pd.DataFrame({
        'Cumulative Mortality': dead_counts
    }, index=day_labels)
    
    st.line_chart(mortality_df, color=[primary_color])

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
    else:
        st.info("No temperature or humidity data available for scatter plot.")

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
