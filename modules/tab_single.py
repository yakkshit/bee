import io
import os
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
from matplotlib.patches import Patch
from scipy.optimize import curve_fit
from scipy.stats import norm, linregress
from lifelines import KaplanMeierFitter

from modules.parser import DEFAULT_CSV, parse_time_data, extract_individual_bee_survival, calculate_lt_percentiles
from modules.survival_stats import (
    render_log_rank_analysis,
    render_cox_regression_analysis,
    render_logrank_math_card,
    render_cox_math_card
)


def render_single_batch_tab(settings=None):
    """Renders the comprehensive Single CSV Analysis tab."""
    if settings is None:
        settings = {}

    primary_color = settings.get('primary_color', '#ef4444')
    secondary_color = settings.get('secondary_color', '#f59e0b')
    line_width = settings.get('line_width', 2.4)
    marker_size = settings.get('marker_size', 7.0)
    font_size = settings.get('font_size', 11)
    abbott_c = settings.get('abbott_c', 0.0)
    line_style = settings.get('line_style', '-')
    hm_alive_color = settings.get('hm_alive_color', '#10b981')
    hm_behavior_color = settings.get('hm_behavior_color', '#f59e0b')
    hm_dead_color = settings.get('hm_dead_color', '#ef4444')

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

        st.session_state.df = st.session_state.df.astype(str)
        st.session_state.df.replace("nan", "", inplace=True)

    st.write("### Daily Individual Observations")

    uploaded_file = st.file_uploader("Upload CSV file (Optional)", type=["csv"])
    if uploaded_file is not None:
        if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
            try:
                new_df = pd.read_csv(uploaded_file, sep=None, engine='python')

                if 'bee_id' in new_df.columns:
                    new_df = new_df.set_index('bee_id').T
                    new_df.reset_index(inplace=True)
                    new_df.rename(columns={'index': 'Day'}, inplace=True)

                    cols = list(new_df.columns)
                    for i in range(1, len(cols)):
                        col_name = str(cols[i])
                        if col_name.isdigit():
                            cols[i] = f"Bee {col_name}"
                        elif col_name == 'Temp':
                            cols[i] = 'Temp (°C)'
                    new_df.columns = cols

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
    st.markdown("Download a printable version of your observation sheet (matching the template). Open in browser and press **Ctrl+P / Cmd+P** to save as landscape PDF.")

    html_content = generate_html_report(df, st.session_state)
    st.download_button(
        label="📄 Download Observation Sheet (HTML format for Print-to-PDF)",
        data=html_content,
        file_name="Observation_Sheet.html",
        mime="text/html",
    )

    st.markdown("---")
    st.header("📊 Plot Generation")

    times_numeric, raw_times, time_col = parse_time_data(df)
    elapsed_days = times_numeric
    day_labels = raw_times
    num_days = len(df)
    max_time = elapsed_days[-1] if len(elapsed_days) > 0 else 1.0

    bee_cols = [c for c in df.columns if str(c).lower().startswith('bee')]
    num_bees = len(bee_cols)

    temps_raw = df['Temp (°C)'].tolist() if 'Temp (°C)' in df.columns else [0]*num_days
    humidities_raw = df['Humidity'].tolist() if 'Humidity' in df.columns else [0]*num_days
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

    def get_latest_state_single(bee_states, up_to_day_idx):
        for k in range(up_to_day_idx, -1, -1):
            state = str(bee_states[k]).strip().upper()
            if state in ['A', 'B', 'D']:
                return state
        return 'A'

    for b_states in states_matrix:
        final_state = get_latest_state_single(b_states, num_days - 1)
        if final_state == 'D':
            last_d_idx = max(k for k in range(num_days) if str(b_states[k]).strip().upper() == 'D')
            durations.append(elapsed_days[last_d_idx])
            events.append(1)
        else:
            durations.append(elapsed_days[-1] if num_days > 0 else 0)
            events.append(0)

    durations = np.array(durations)
    events = np.array(events)

    alive_counts = []
    dead_counts = []

    for d_idx in range(num_days):
        n_dead = sum(1 for states in states_matrix if get_latest_state_single(states, d_idx) == 'D')
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

    # Finney empirical probit transformation
    p_lower = 0.5 / max(num_bees, 10)
    p_upper = 1.0 - p_lower
    probit_vals = np.array([norm.ppf(np.clip(m / 100.0, p_lower, p_upper)) + 5.0 for m in abbott_corrected_pct])

    first_nonzero_idx = next((i for i, m in enumerate(abbott_corrected_pct) if m > 0), len(abbott_corrected_pct) - 1)
    start_idx = max(0, first_nonzero_idx - 1)

    reg_log_time = []
    reg_probit_y = []
    for i in range(start_idx, len(elapsed_days)):
        d = elapsed_days[i]
        if d > 0:
            reg_log_time.append(np.log10(d))
            reg_probit_y.append(probit_vals[i])

    log_time = np.array([np.log10(d) for d in elapsed_days if d > 0])
    probit_y = probit_vals[:len(log_time)]

    if len(reg_log_time) > 1:
        res = linregress(reg_log_time, reg_probit_y)
        probit_a, probit_b, probit_r = res.intercept, res.slope, res.rvalue
    else:
        probit_a, probit_b, probit_r = 0, 0, 0

    mortality_se = np.sqrt((mortality_pct / 100.0) * (1.0 - mortality_pct / 100.0) / max(1, num_bees)) * 100.0
    survival_se = np.sqrt((survival_pct / 100.0) * (1.0 - survival_pct / 100.0) / max(1, num_bees)) * 100.0

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

    st.markdown("<br>", unsafe_allow_html=True)

    col_zip_top, col_info = st.columns([1.5, 3])
    with col_zip_top:
        zip_button_top = st.empty()
    with col_info:
        st.caption("💡 Figures are grouped into structured thematic categories below. Each figure can be downloaded individually at publication resolution (300 DPI) or packaged together as a ZIP archive.")

    plot_images_dict = {}

    def get_image_download_link(fig, filename, text, key_suffix=""):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        img_bytes = buf.getvalue()
        plot_images_dict[filename] = img_bytes
        btn_key = f"dl_{filename}_{key_suffix}" if key_suffix else f"dl_{filename}"
        st.download_button(label=f"📥 {text}", data=img_bytes, file_name=filename, mime="image/png", key=btn_key)
        plt.close(fig)

    # 6 Structured Thematic Categories
    tab_surv, tab_lt, tab_cohort, tab_env, tab_stats, tab_pub = st.tabs([
        "📈 Mortality & Survival Models",
        "⏱️ Lethal Time (LTₓ) Laboratory",
        "🐝 Cohort & Individual Tracking",
        "🌦️ Environmental Dynamics",
        "🔬 Biostatistical Models",
        "📑 Publication Multi-Panel Figure"
    ])

    # ==========================================
    # TAB 1: MORTALITY & SURVIVAL MODELS
    # ==========================================
    with tab_surv:
        st.markdown("#### Cohort Survival Dynamics & Kinetic Curves")
        st.caption("Mathematical characterization of cohort mortality progression, survival decay kinetics, and Kaplan-Meier product-limit estimators.")

        c1, c2 = st.columns(2)

        if num_days > 0:
            with c1:
                st.subheader("Logistic Mortality Curve")
                fig1, ax = plt.subplots(figsize=(7.0, 5.2), dpi=300)
                fit_se = np.std(residuals_mort)
                ax.fill_between(t_dense, np.clip(fit_mort_curve - 1.96*fit_se, 0, 100), np.clip(fit_mort_curve + 1.96*fit_se, 0, 100), color=primary_color, alpha=0.12, label='95% Confidence Band')
                ax.plot(t_dense, fit_mort_curve, color=primary_color, linewidth=line_width, linestyle=line_style, label='Logistic Model Fit')
                ax.errorbar(elapsed_days, mortality_pct, yerr=mortality_se, fmt='o', color=primary_color, ecolor=primary_color, elinewidth=1.6, capsize=4, capthick=1.6, markersize=marker_size, markerfacecolor=primary_color, markeredgecolor='black', label='Observed (Mean ± SE)')
                ax.axhline(50, color='#333333', linestyle=':', linewidth=1.2)
                eq_text = f"Y = {L_fit:.1f} / (1 + exp(-{k_fit:.2f}(X - {t0_fit:.1f})))\nR² = {r2_mort:.4f}"
                ax.text(0.06, 0.90, eq_text, transform=ax.transAxes, fontsize=10.5, verticalalignment='top', bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#e0e0e0', alpha=0.85))
                ax.set_xlabel('Observation Time / Days', fontsize=12, fontweight='bold', labelpad=7)
                ax.set_ylabel('Mortality / %', fontsize=12, fontweight='bold', labelpad=7)
                ax.set_xlim(-0.1, max_time * 1.05)
                ax.set_ylim(-2, 102)
                ax.set_xticks(elapsed_days)
                ax.set_xticklabels(day_labels, rotation=45, ha="right")
                ax.yaxis.set_major_locator(MultipleLocator(25))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                fig1.tight_layout()
                st.pyplot(fig1)
                get_image_download_link(fig1, 'fig_mortality_curve.png', 'Download Mortality Curve', 't1_mort')

            with c2:
                st.subheader("Survival Decay Curve")
                fig2, ax = plt.subplots(figsize=(7.0, 5.2), dpi=300)
                surv_se_fit = np.std(residuals_surv)
                ax.fill_between(t_dense, np.clip(fit_surv_curve - 1.96*surv_se_fit, 0, 100), np.clip(fit_surv_curve + 1.96*surv_se_fit, 0, 100), color=secondary_color, alpha=0.15, label='95% Confidence Band')
                ax.plot(t_dense, fit_surv_curve, color=secondary_color, linewidth=line_width, linestyle=line_style, label='Survival Model Fit')
                ax.errorbar(elapsed_days, survival_pct, yerr=survival_se, fmt='o', color=secondary_color, ecolor=secondary_color, elinewidth=1.6, capsize=4, capthick=1.6, markersize=marker_size, markerfacecolor=secondary_color, markeredgecolor='black', label='Observed (Mean ± SE)')
                ax.axhline(50, color='#333333', linestyle=':', linewidth=1.2)
                eq_surv_text = f"Y = {S0_surv:.1f} * exp(-{a_surv:.4f} * X^{b_surv:.2f})\nR² = {r2_surv:.4f}"
                ax.text(0.06, 0.32, eq_surv_text, transform=ax.transAxes, fontsize=10.5, verticalalignment='top', bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#e0e0e0', alpha=0.85))
                ax.set_xlabel('Observation Time / Days', fontsize=12, fontweight='bold', labelpad=7)
                ax.set_ylabel('Bee Survival / %', fontsize=12, fontweight='bold', labelpad=7)
                ax.set_xlim(-0.1, max_time * 1.05)
                ax.set_ylim(-2, 105)
                ax.set_xticks(elapsed_days)
                ax.set_xticklabels(day_labels, rotation=45, ha="right")
                ax.yaxis.set_major_locator(MultipleLocator(25))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                fig2.tight_layout()
                st.pyplot(fig2)
                get_image_download_link(fig2, 'fig_survival_curve.png', 'Download Survival Curve', 't1_surv')

        if len(durations) > 0:
            with c1:
                st.subheader("Kaplan-Meier Survival Analysis")
                fig3, ax = plt.subplots(figsize=(7.0, 5.2), dpi=300)
                km_timeline = kmf.survival_function_.index
                km_vals = kmf.survival_function_['KM_estimate'].values
                ci_df = kmf.confidence_interval_survival_function_
                ax.step(km_timeline, km_vals, where='post', color='#0284c7', linewidth=line_width, linestyle=line_style, label='Kaplan-Meier Estimate')
                ax.fill_between(km_timeline, ci_df.iloc[:, 0], ci_df.iloc[:, 1], step='post', color='#0284c7', alpha=0.15, label='95% Confidence Interval')
                censored_days = durations[events == 0]
                censored_surv = [kmf.survival_function_at_times(d).values[0] for d in censored_days] if len(censored_days) > 0 else []
                if len(censored_days) > 0:
                    ax.plot(censored_days, censored_surv, '+', color='#0f172a', markersize=9, markeredgewidth=2, label=f'Censored (n={len(censored_days)})')
                ax.axhline(0.5, color='#333333', linestyle=':', linewidth=1.2)
                ax.set_xlabel('Observation Time / Days', fontsize=12, fontweight='bold', labelpad=7)
                ax.set_ylabel('Survival Probability', fontsize=12, fontweight='bold', labelpad=7)
                ax.set_xlim(-0.1, max_time * 1.05)
                ax.set_ylim(-0.02, 1.05)
                ax.set_xticks(elapsed_days)
                ax.set_xticklabels(day_labels, rotation=45, ha="right")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.legend(loc='lower left', frameon=False, fontsize=10.0)
                fig3.tight_layout()
                st.pyplot(fig3)
                get_image_download_link(fig3, 'fig_kaplan_meier.png', 'Download Kaplan-Meier Curve', 't1_km')

            with c2:
                st.subheader("Abbott Corrected Mortality Curve")
                fig_abbott, ax_abbott = plt.subplots(figsize=(7.0, 5.2), dpi=300)
                ax_abbott.plot(elapsed_days, abbott_corrected_pct, color=primary_color, marker='o', linewidth=line_width, linestyle=line_style, markersize=marker_size)
                ax_abbott.set_xlabel('Observation Time / Days', fontsize=12, fontweight='bold', labelpad=7)
                ax_abbott.set_ylabel('Corrected Mortality / %', fontsize=12, fontweight='bold', labelpad=7)
                ax_abbott.set_xlim(-0.1, max_time * 1.05)
                ax_abbott.set_ylim(-2, 102)
                ax_abbott.set_xticks(elapsed_days)
                ax_abbott.set_xticklabels(day_labels, rotation=45, ha="right")
                ax_abbott.spines['top'].set_visible(False)
                ax_abbott.spines['right'].set_visible(False)
                fig_abbott.tight_layout()
                st.pyplot(fig_abbott)
                get_image_download_link(fig_abbott, 'fig_abbott_corrected.png', 'Download Abbott Corrected Curve', 't1_abbott')

    # ==========================================
    # TAB 2: LETHAL TIME (LTx) LABORATORY
    # ==========================================
    with tab_lt:
        st.markdown("#### Lethal Time (LTₓ) Spectrum & Sensitivity Analysis")
        st.caption("Calculate and visualize the time required to reach specific cohort mortality thresholds (LT₁₀, LT₂₀, LT₃₀, LT₅₀, LT₉₀, etc.) derived from Finney Probit regression and observed bioassay timestamps.")

        c_lt_preset, c_lt_custom = st.columns([1.5, 2.5])
        with c_lt_preset:
            lt_preset = st.radio(
                "Percentile Selection Presets:",
                [
                    "Standard (LT10, LT20, LT30, LT50, LT90)",
                    "Quartiles (LT25, LT50, LT75)",
                    "Full Decile Spectrum (10% to 90%)",
                    "Custom Percentile Choice"
                ],
                index=0,
                key="single_lt_preset"
            )

        if lt_preset == "Standard (LT10, LT20, LT30, LT50, LT90)":
            chosen_percentiles = [10, 20, 30, 50, 90]
        elif lt_preset == "Quartiles (LT25, LT50, LT75)":
            chosen_percentiles = [25, 50, 75]
        elif lt_preset == "Full Decile Spectrum (10% to 90%)":
            chosen_percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        else:
            with c_lt_custom:
                default_p = [10, 20, 50, 90]
                avail_options = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 75, 80, 85, 90, 95]
                mult_sel = st.multiselect("Choose Target Percentiles:", avail_options, default=default_p, key="single_lt_multiselect")
                c_add_val = st.number_input("Add Custom Percentile (1–99%):", min_value=1, max_value=99, value=50, step=1, key="single_lt_custom_num")
                combined_p = set(mult_sel)
                if c_add_val:
                    combined_p.add(int(c_add_val))
                chosen_percentiles = sorted(list(combined_p))

        # Compute LT_X records
        lt_records = calculate_lt_percentiles(
            intercept_a=probit_a,
            slope_b=probit_b,
            start_time=elapsed_days[0] if len(elapsed_days) > 0 else 1.0,
            times_numeric=elapsed_days,
            mortality_pct=abbott_corrected_pct,
            percentiles=chosen_percentiles
        )

        col_lt1, col_lt2 = st.columns(2)

        # Plot A: LT_X Threshold Spectrum Sigmoid Plot
        with col_lt1:
            st.subheader("Lethal Time (LTₓ) Threshold Spectrum Plot")
            fig_lt_spec, ax_lt_s = plt.subplots(figsize=(7.2, 5.4), dpi=300)

            # Background Sigmoid curve
            ax_lt_s.plot(t_dense, fit_mort_curve, color=primary_color, linewidth=line_width, label='Fitted Mortality Curve', zorder=3)
            ax_lt_s.scatter(elapsed_days, mortality_pct, color=primary_color, edgecolors='black', s=marker_size**2 * 1.4, zorder=4, label='Observed Points')

            cmap_spec = plt.cm.plasma
            n_recs = max(1, len(lt_records))

            for idx_r, rec in enumerate(lt_records):
                p_lvl = rec['percentile']
                lbl = rec['label']
                emp_val = rec.get('empirical_lt', np.nan)
                theo_val = rec.get('theoretical_lt', np.nan)
                t_val = emp_val if np.isfinite(emp_val) else theo_val
                elap_int = rec.get('effective_elapsed_int', np.nan)

                color_th = cmap_spec(idx_r / max(1, n_recs - 1) * 0.85 + 0.05) if n_recs > 1 else '#d97706'

                # Horizontal line for target mortality
                ax_lt_s.axhline(p_lvl, color=color_th, linestyle='--', linewidth=1.2, alpha=0.75, zorder=2)

                if np.isfinite(t_val) and t_val <= max_time * 1.25:
                    ax_lt_s.plot([t_val, t_val], [0, p_lvl], color=color_th, linestyle=':', linewidth=1.3, zorder=2)
                    ax_lt_s.plot([t_val], [p_lvl], marker='o', color=color_th, markersize=8, markeredgecolor='black', zorder=5)

                    txt_badge = f"{lbl} ({int(round(elap_int))}d)" if np.isfinite(elap_int) else f"{lbl} ({t_val:.1f}d)"
                    ax_lt_s.annotate(
                        txt_badge,
                        xy=(t_val, p_lvl),
                        xytext=(7, -10 if p_lvl > 70 else 5),
                        textcoords='offset points',
                        fontsize=9.0,
                        fontweight='bold',
                        color=color_th,
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=color_th, alpha=0.9, lw=1.2)
                    )

            ax_lt_s.set_xlabel('Observation Time / Days', fontsize=12, fontweight='bold', labelpad=7)
            ax_lt_s.set_ylabel('Mortality / %', fontsize=12, fontweight='bold', labelpad=7)
            ax_lt_s.set_xlim(-0.1, max_time * 1.08)
            ax_lt_s.set_ylim(-2, 105)
            ax_lt_s.set_xticks(elapsed_days)
            ax_lt_s.set_xticklabels(day_labels, rotation=45, ha="right")
            ax_lt_s.yaxis.set_major_locator(MultipleLocator(20))
            ax_lt_s.spines['top'].set_visible(False)
            ax_lt_s.spines['right'].set_visible(False)
            ax_lt_s.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=9.5)
            fig_lt_spec.tight_layout()
            st.pyplot(fig_lt_spec)
            get_image_download_link(fig_lt_spec, 'fig_lt_threshold_spectrum.png', 'Download LT Threshold Spectrum', 't2_spec')

        # Plot B: Horizontal Lollipop / Bar Chart
        with col_lt2:
            st.subheader("Time-to-Lethal Endpoint Comparison (LTₓ)")
            lbls = [r['label'] for r in lt_records]
            elap_vals = [r['effective_elapsed_int'] if np.isfinite(r.get('effective_elapsed_int', np.nan)) else 0.0 for r in lt_records]
            pcts = [r['percentile'] for r in lt_records]

            fig_lt_bar, ax_lt_b = plt.subplots(figsize=(7.2, max(4.8, len(lbls) * 0.52)), dpi=300)
            y_pos = np.arange(len(lbls))
            cmap_bar = plt.cm.viridis
            bar_colors = [cmap_bar(p / 100.0) for p in pcts]

            for i in range(len(lbls)):
                v = elap_vals[i]
                rec = lt_records[i]
                if np.isfinite(rec.get('effective_elapsed_int', np.nan)) and rec['effective_elapsed_int'] > 0:
                    ax_lt_b.plot([0, v], [i, i], color=bar_colors[i], linewidth=2.4, zorder=2)
                    ax_lt_b.plot(v, i, marker='o', markersize=9.5, color=bar_colors[i], markeredgecolor='black', zorder=3)
                    ax_lt_b.text(v + 0.25, i, f"{int(round(v))} Days ({rec['status']})", va='center', ha='left', fontsize=9.5, fontweight='bold', color='#1e293b')
                else:
                    ax_lt_b.plot([0, 0.4], [i, i], color='#94a3b8', linestyle=':', linewidth=1.5, zorder=2)
                    ax_lt_b.text(0.55, i, "Not Reached in Trial", va='center', ha='left', fontsize=9.0, fontstyle='italic', color='#64748b')

            ax_lt_b.set_yticks(y_pos)
            ax_lt_b.set_yticklabels(lbls, fontsize=11, fontweight='bold')
            ax_lt_b.set_xlabel('Elapsed Observation Time / Days', fontsize=12, fontweight='bold', labelpad=7)
            max_x = max([v for v in elap_vals if np.isfinite(v)], default=5.0)
            ax_lt_b.set_xlim(-0.2, max_x + 3.8)
            ax_lt_b.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax_lt_b.spines['top'].set_visible(False)
            ax_lt_b.spines['right'].set_visible(False)
            ax_lt_b.grid(True, linestyle='--', alpha=0.35, axis='x')
            fig_lt_bar.tight_layout()
            st.pyplot(fig_lt_bar)
            get_image_download_link(fig_lt_bar, 'fig_lt_horizon_comparison.png', 'Download LT Horizon Chart', 't2_bar')

        # LT Summary Metrics Table
        st.markdown("---")
        st.subheader("📋 Lethal Time (LTₓ) Toxicological Reference Table")
        table_rows = []
        for r in lt_records:
            t_elap = r.get('theoretical_elapsed', np.nan)
            e_elap = r.get('empirical_elapsed', np.nan)
            eff_int = r.get('effective_elapsed_int', np.nan)

            theo_str = f"{t_elap:.2f} d" if np.isfinite(t_elap) else "N/A"
            emp_str = f"{e_elap:.2f} d" if np.isfinite(e_elap) else "N/A"
            eff_int_str = f"{int(round(eff_int))} Days" if np.isfinite(eff_int) else "Not Reached"
            table_rows.append({
                "Lethal Level": r['label'],
                "Target Mortality (%)": f"{r['percentile']:.1f}%",
                "Finney Probit Score": round(r['probit_score'], 4),
                "Theoretical Probit LT (Elapsed)": theo_str,
                "Empirical Observed LT (Elapsed)": emp_str,
                "Effective Whole Days": eff_int_str,
                "Bioassay Status": r['status']
            })
        lt_table_df = pd.DataFrame(table_rows)
        st.dataframe(lt_table_df, use_container_width=True)

        c_lt_dl, _ = st.columns([2, 2])
        with c_lt_dl:
            st.download_button(
                label="📊 Download LT Table (CSV)",
                data=lt_table_df.to_csv(index=False),
                file_name="lethal_times_summary.csv",
                mime="text/csv",
                key="dl_lt_table_csv"
            )

    # ==========================================
    # TAB 3: BEE COHORT & TRACKING
    # ==========================================
    with tab_cohort:
        st.markdown("#### Individual Bee Observation Matrix & Population Distributions")
        st.caption("Granular tracking of individual bee health transitions (Alive, Behavioral change, Dead), cumulative mortality trajectory, and daily nutrient consumption.")

        c_b1, c_b2 = st.columns(2)

        if num_days > 0 and num_bees > 0:
            with c_b1:
                st.subheader("Individual Bee Status Matrix Heatmap")
                fig4, ax = plt.subplots(figsize=(8.0, 5.8), dpi=300)
                state_num_map = {'A': 1, 'B': 2, 'D': 3}
                color_map_hm = {1: hm_alive_color, 2: hm_behavior_color, 3: hm_dead_color}
                mat = np.zeros((num_bees, num_days))
                for i, b_states in enumerate(states_matrix):
                    for j, s in enumerate(b_states):
                        mat[i, j] = state_num_map.get(str(s).strip(), 1)
                for i in range(num_bees):
                    for j in range(num_days):
                        val = mat[i, j]
                        c = color_map_hm.get(val, '#cccccc')
                        ax.add_patch(plt.Rectangle((j, num_bees - 1 - i), 0.90, 0.85, color=c, ec='white', lw=1.5, zorder=2))
                        ax.text(j + 0.45, num_bees - 1 - i + 0.425, states_matrix[i][j], ha='center', va='center', color='white', fontweight='bold', fontsize=10.5, zorder=3)
                ax.set_xlim(-0.1, float(num_days))
                ax.set_ylim(-0.1, num_bees)
                ax.set_xticks(np.arange(num_days) + 0.45)
                ax.set_xticklabels(day_labels, fontsize=10.5, fontweight='bold', rotation=45, ha='right')
                ax.set_yticks(np.arange(num_bees) + 0.425)
                ax.set_yticklabels([f'Bee {num_bees - i}' for i in range(num_bees)], fontsize=10, fontweight='bold')
                ax.set_xlabel('Observation Day', fontsize=12, fontweight='bold', labelpad=7)
                ax.set_ylabel('Bee ID', fontsize=12, fontweight='bold', labelpad=7)

                legend_elements = [
                    Patch(facecolor=hm_alive_color, edgecolor='none', label='Alive (A)'),
                    Patch(facecolor=hm_behavior_color, edgecolor='none', label='Behavior Change (B)'),
                    Patch(facecolor=hm_dead_color, edgecolor='none', label='Dead (D)')
                ]
                ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, frameon=False, fontsize=9.5)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.tick_params(left=False, bottom=False)
                fig4.tight_layout()
                st.pyplot(fig4)
                get_image_download_link(fig4, 'fig_individual_bees_status.png', 'Download Status Heatmap', 't3_hm')

            with c_b2:
                st.subheader("Population Status Breakdown")
                behavior_counts = [num_bees - a - d for a, d in zip(alive_counts, dead_counts)]
                fig_area, ax_area = plt.subplots(figsize=(7.0, 5.2), dpi=300)
                ax_area.stackplot(elapsed_days, dead_counts, behavior_counts, alive_counts,
                                  labels=['Dead', 'Behavior Change', 'Healthy / Alive'],
                                  colors=[hm_dead_color, hm_behavior_color, hm_alive_color], alpha=0.85)
                ax_area.set_xlabel('Observation Time / Days', fontsize=12, fontweight='bold', labelpad=7)
                ax_area.set_ylabel('Number of Bees', fontsize=12, fontweight='bold', labelpad=7)
                ax_area.set_xlim(-0.1, max_time * 1.05)
                ax_area.set_xticks(elapsed_days)
                ax_area.set_xticklabels(day_labels, rotation=45, ha='right')
                ax_area.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none', fontsize=9.5)
                ax_area.spines['top'].set_visible(False)
                ax_area.spines['right'].set_visible(False)
                fig_area.tight_layout()
                st.pyplot(fig_area)
                get_image_download_link(fig_area, 'fig_status_breakdown.png', 'Download Status Breakdown', 't3_area')

            with c_b1:
                st.subheader("Daily Status Distribution")
                fig_bar, ax_bar = plt.subplots(figsize=(7.0, 5.0), dpi=300)
                x_pos = np.arange(num_days)
                bar_w = 0.35
                ax_bar.bar(x_pos - bar_w/2, alive_counts, bar_w, label='Alive', color=secondary_color)
                ax_bar.bar(x_pos + bar_w/2, dead_counts, bar_w, label='Dead', color=primary_color)
                ax_bar.set_xlabel('Observation Day', fontsize=12, fontweight='bold', labelpad=7)
                ax_bar.set_ylabel('Bee Count', fontsize=12, fontweight='bold', labelpad=7)
                ax_bar.set_xticks(x_pos)
                ax_bar.set_xticklabels(day_labels, rotation=45, ha='right')
                ax_bar.legend(loc='upper right', frameon=False, fontsize=9.5)
                ax_bar.spines['top'].set_visible(False)
                ax_bar.spines['right'].set_visible(False)
                fig_bar.tight_layout()
                st.pyplot(fig_bar)
                get_image_download_link(fig_bar, 'fig_daily_status_bar.png', 'Download Daily Status Chart', 't3_bar')

            with c_b2:
                st.subheader("Cumulative Mortality Progression")
                fig_cum, ax_cum = plt.subplots(figsize=(7.0, 5.0), dpi=300)
                ax_cum.plot(elapsed_days, dead_counts, marker='o', color=primary_color, linewidth=line_width, markersize=marker_size, label='Cumulative Mortality')
                ax_cum.set_xlabel('Observation Time / Days', fontsize=12, fontweight='bold', labelpad=7)
                ax_cum.set_ylabel('Cumulative Dead Bees', fontsize=12, fontweight='bold', labelpad=7)
                ax_cum.set_xlim(-0.1, max_time * 1.05)
                ax_cum.set_xticks(elapsed_days)
                ax_cum.set_xticklabels(day_labels, rotation=45, ha='right')
                ax_cum.spines['top'].set_visible(False)
                ax_cum.spines['right'].set_visible(False)
                fig_cum.tight_layout()
                st.pyplot(fig_cum)
                get_image_download_link(fig_cum, 'fig_cumulative_mortality.png', 'Download Cumulative Mortality Chart', 't3_cum')

        # Full-width Consumption Tracker
        if len(cons_matrix) > 0 and any(any(v > 0 for v in bee_cons) for bee_cons in cons_matrix):
            st.markdown("---")
            st.subheader("Daily Resource Consumption per Bee (µL)")
            fig_cons, ax_cons = plt.subplots(figsize=(10.0, 5.2), dpi=300)
            x_days = elapsed_days
            colors_c = plt.cm.tab20(np.linspace(0, 1, max(1, len(cons_cols))))
            for i, c in enumerate(cons_cols):
                bee_label = c.replace("Cons", "Bee").replace(" (µL)", "").strip()
                ax_cons.plot(x_days, cons_matrix[i], marker='o', linewidth=1.5, markersize=5, label=bee_label, color=colors_c[i])

            ax_cons.set_xlabel('Observation Time / Days', fontsize=12, fontweight='bold')
            ax_cons.set_ylabel('Consumption (µL)', fontsize=12, fontweight='bold')
            ax_cons.set_xlim(-0.1, max_time * 1.05)
            ax_cons.set_xticks(elapsed_days)
            ax_cons.set_xticklabels(day_labels, rotation=45, ha='right')
            ax_cons.grid(True, linestyle='--', alpha=0.45)
            ax_cons.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=8.5)
            fig_cons.tight_layout()
            st.pyplot(fig_cons)
            get_image_download_link(fig_cons, 'fig_consumption_per_bee.png', 'Download Consumption Plot', 't3_cons')

    # ==========================================
    # TAB 4: ENVIRONMENTAL DYNAMICS
    # ==========================================
    with tab_env:
        st.markdown("#### Ambient Temperature & Relative Humidity Interactions")
        st.caption("Temporal variation in rearing conditions, temp-humidity environmental space, and bivariate correlation matrix.")

        c_e1, c_e2 = st.columns(2)

        if any(temps):
            with c_e1:
                st.subheader("Environmental Conditions Timeline")
                fig5, ax1 = plt.subplots(figsize=(7.0, 5.2), dpi=300)
                x_days = elapsed_days
                color_temp = '#d95f02'
                ax1.set_xlabel('Observation Time / Days', fontsize=12, fontweight='bold', labelpad=7)
                ax1.set_ylabel('Temperature / °C', color=color_temp, fontsize=12, fontweight='bold', labelpad=7)
                l1 = ax1.plot(x_days, temps, color=color_temp, marker='o', linewidth=2.2, markersize=7.0, label='Temperature (°C)')
                ax1.tick_params(axis='y', labelcolor=color_temp)
                ax1.set_ylim(min(temps)-1, max(temps)+1)

                ax2 = ax1.twinx()
                color_hum = '#0284c7'
                ax2.set_ylabel('Relative Humidity / %', color=color_hum, fontsize=12, fontweight='bold', labelpad=7)
                l2 = ax2.plot(x_days, humidities, color=color_hum, marker='s', linewidth=2.2, markersize=7.0, linestyle='--', label='Relative Humidity (%)')
                ax2.tick_params(axis='y', labelcolor=color_hum)
                if any(humidities):
                    ax2.set_ylim(min(humidities)-5, max(humidities)+5)

                ax1.set_xlim(-0.1, max_time * 1.05)
                ax1.set_xticks(elapsed_days)
                ax1.set_xticklabels(day_labels, rotation=45, ha="right")
                ax1.spines['top'].set_visible(False)
                ax2.spines['top'].set_visible(False)
                lines_e = l1 + l2
                ax1.legend(lines_e, [l.get_label() for l in lines_e], loc='upper left', frameon=True, framealpha=0.9, facecolor='white', edgecolor='none', fontsize=9.5)
                fig5.tight_layout()
                st.pyplot(fig5)
                get_image_download_link(fig5, 'fig_environmental_conditions.png', 'Download Environmental Timeline', 't4_env')

            with c_e2:
                st.subheader("Temperature vs Humidity Scatter")
                fig_scat, ax_scat = plt.subplots(figsize=(7.0, 5.2), dpi=300)
                ax_scat.scatter(temps, humidities, color=primary_color, s=marker_size**2 * 4, edgecolor='black', alpha=0.85)
                for i, txt in enumerate(day_labels):
                    ax_scat.annotate(txt, (temps[i], humidities[i]), fontsize=9, xytext=(5, 5), textcoords='offset points')
                ax_scat.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold', labelpad=7)
                ax_scat.set_ylabel('Humidity (%)', fontsize=12, fontweight='bold', labelpad=7)
                ax_scat.spines['top'].set_visible(False)
                ax_scat.spines['right'].set_visible(False)
                fig_scat.tight_layout()
                st.pyplot(fig_scat)
                get_image_download_link(fig_scat, 'fig_temp_vs_humidity.png', 'Download Scatter Plot', 't4_scat')

            with c_e1:
                st.subheader("Statistical Pearson Correlation Heatmap")
                corr_data = {
                    'Mortality (%)': mortality_pct,
                    'Survival (%)': survival_pct,
                    'Temp': temps,
                    'Humidity': humidities
                }
                corr_df = pd.DataFrame(corr_data).corr()
                fig_corr, ax_corr = plt.subplots(figsize=(6.2, 5.0), dpi=300)
                cax = ax_corr.matshow(corr_df, cmap='coolwarm', vmin=-1, vmax=1)
                fig_corr.colorbar(cax)
                ax_corr.set_xticks(range(len(corr_df.columns)))
                ax_corr.set_yticks(range(len(corr_df.columns)))
                ax_corr.set_xticklabels(corr_df.columns, rotation=45, ha='left', fontsize=10)
                ax_corr.set_yticklabels(corr_df.columns, fontsize=10)
                for i in range(len(corr_df.columns)):
                    for j in range(len(corr_df.columns)):
                        ax_corr.text(j, i, f"{corr_df.iloc[i, j]:.2f}", ha='center', va='center', color='white' if abs(corr_df.iloc[i, j]) > 0.5 else 'black', fontweight='bold')
                fig_corr.tight_layout()
                st.pyplot(fig_corr)
                get_image_download_link(fig_corr, 'fig_correlation_heatmap.png', 'Download Correlation Heatmap', 't4_corr')
        else:
            st.info("No temperature or humidity sensor readings available in the uploaded dataset.")

    # ==========================================
    # TAB 5: BIOSTATISTICAL MODELS
    # ==========================================
    with tab_stats:
        st.markdown("#### Inferential Biostatistical Modeling")
        st.caption("Standard Finney Probit linear transformation, Log-Rank survival distributions test, and multivariable Cox Proportional Hazards regression.")

        subtab_prob, subtab_lr_s, subtab_cox_s = st.tabs([
            "📈 Finney Probit Regression",
            "📉 Log-Rank (Mantel-Cox) Test",
            "🌲 Cox Proportional Hazards Model"
        ])

        with subtab_prob:
            c_s1, c_s2 = st.columns(2)
            if len(log_time) > 1:
                with c_s1:
                    st.subheader("Probit Regression (Time-Mortality)")
                    fig_probit, ax_probit = plt.subplots(figsize=(7.0, 5.2), dpi=300)
                    ax_probit.scatter(log_time, probit_y, color=primary_color, s=marker_size**2 * 1.5, edgecolors='black')
                    x_line = np.linspace(min(log_time), max(log_time), 100)
                    y_line = probit_a + probit_b * x_line
                    ax_probit.plot(x_line, y_line, color='#1e293b', linewidth=line_width, linestyle=line_style)
                    eq_text = f"PROBIT(P) = {probit_a:.2f} + {probit_b:.2f}x\nR² = {probit_r**2:.4f}"
                    ax_probit.text(0.05, 0.95, eq_text, transform=ax_probit.transAxes, fontsize=10.5, verticalalignment='top', bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#e0e0e0', alpha=0.85))
                    ax_probit.set_xlabel('Log10(Observation Time / Days)', fontsize=12, fontweight='bold', labelpad=7)
                    ax_probit.set_ylabel('Probit(P)', fontsize=12, fontweight='bold', labelpad=7)
                    ax_probit.spines['top'].set_visible(False)
                    ax_probit.spines['right'].set_visible(False)
                    fig_probit.tight_layout()
                    st.pyplot(fig_probit)
                    get_image_download_link(fig_probit, 'fig_probit_regression.png', 'Download Probit Regression', 't5_probit')

                with c_s2:
                    st.subheader("Probit Model Parameters")
                    p_r2 = probit_r**2
                    st.markdown(f"""
                    <div class="glass-card">
                        <p><b>Model Formula:</b> <code>Probit(P) = a + b · log₁₀(t)</code></p>
                        <p><b>Slope (b):</b> <code>{probit_b:.4f}</code></p>
                        <p><b>Intercept (a):</b> <code>{probit_a:.4f}</code></p>
                        <p><b>Pearson Correlation (r):</b> <code>{probit_r:.4f}</code></p>
                        <p><b>Coefficient of Determination (R²):</b> <code>{p_r2:.4f}</code></p>
                        <p><b>Status:</b> {'✅ High Linear Fit' if p_r2 > 0.8 else ('⚠️ Moderate Linear Fit' if p_r2 > 0.5 else 'ℹ️ Low Linear Correlation')}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Insufficient observation days to fit Probit regression.")

        with subtab_lr_s:
            col_lrs_m, col_lrs_side = st.columns([2.2, 1.0])
            with col_lrs_m:
                st.subheader("Log-Rank Survival Test (By Treatment)")
                treatment_vals = [str(x).strip() for x in df['Treatment'].unique() if pd.notna(x) and str(x).strip() != ''] if 'Treatment' in df.columns else []
                if len(treatment_vals) >= 2:
                    single_bees_df = extract_individual_bee_survival(df, batch_name="Single Batch", elapsed_days=elapsed_days)
                    render_log_rank_analysis(single_bees_df, group_col='treatment', key_suffix="single")
                else:
                    t_curr = treatment_vals[0] if treatment_vals else "None"
                    st.info(f"💡 **Log-Rank Survival Test**: The uploaded file contains only 1 treatment cohort ('{t_curr}'). Log-Rank test requires at least 2 distinct treatment cohorts to compare. Upload multiple CSV files in the **'Multi-CSV Probit Comparison'** or **'Survival Statistics'** tab.")
            with col_lrs_side:
                render_logrank_math_card()

        with subtab_cox_s:
            col_cxs_m, col_cxs_side = st.columns([2.2, 1.0])
            with col_cxs_m:
                st.subheader("Cox Proportional Hazards Regression")
                single_bees_df = extract_individual_bee_survival(df, batch_name="Single Batch", elapsed_days=elapsed_days)
                treatment_vals = [str(x).strip() for x in df['Treatment'].unique() if pd.notna(x) and str(x).strip() != ''] if 'Treatment' in df.columns else []
                has_varying_cons = ('mean_consumption' in single_bees_df.columns and single_bees_df['mean_consumption'].nunique() > 1 and float(single_bees_df['mean_consumption'].std()) > 0.05)
                if len(treatment_vals) >= 2 or has_varying_cons:
                    render_cox_regression_analysis(single_bees_df, group_col='treatment' if len(treatment_vals) > 1 else 'cohort', key_suffix="single")
                else:
                    t_curr = treatment_vals[0] if treatment_vals else "None"
                    st.info(f"💡 **Cox Proportional Hazards Regression**: The uploaded file contains only 1 treatment cohort ('{t_curr}') and no varying individual bee feeding records. Cox regression evaluates relative hazard ratios between 2 or more exposure cohorts or across varying individual predictors. Please compare multiple cohorts in the **'Multi-CSV Probit Comparison'** or **'Survival Statistics'** tab.")
            with col_cxs_side:
                render_cox_math_card()

    # ==========================================
    # TAB 6: PUBLICATION MULTI-PANEL FIGURE
    # ==========================================
    with tab_pub:
        st.markdown("#### Publication-Grade Composite Summary Figure")
        st.caption("Composite 4-panel figure (Panels A–D) prepared to high-impact journal standards (Cell, Nature, ES&T, Journal of Insect Physiology) at 300 DPI.")

        if num_days > 0:
            fig6 = plt.figure(figsize=(14, 11), dpi=300)
            gs = fig6.add_gridspec(2, 2, hspace=0.30, wspace=0.25)

            # Panel A: Mortality
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

            # Panel B: Survival
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

            # Panel C: Kaplan-Meier
            ax_c = fig6.add_subplot(gs[1, 0])
            if len(durations) > 0:
                ax_c.step(km_timeline, km_vals, where='post', color='#0284c7', linewidth=line_width, linestyle=line_style, label='KM Estimate')
                ax_c.fill_between(km_timeline, ci_df.iloc[:, 0], ci_df.iloc[:, 1], step='post', color='#0284c7', alpha=0.15, label='95% CI')
                if len(censored_days) > 0:
                    ax_c.plot(censored_days, censored_surv, '+', color='#0f172a', markersize=8.5, markeredgewidth=2, label=f'Censored (n={len(censored_days)})')
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

            # Panel D: Environment
            if any(temps):
                ax_d1 = fig6.add_subplot(gs[1, 1])
                ax_d1.set_xlabel('Observation Time / Days', fontsize=11.5, fontweight='bold')
                ax_d1.set_ylabel('Temperature / °C', color='#d95f02', fontsize=11.5, fontweight='bold')
                l1 = ax_d1.plot(elapsed_days, temps, color='#d95f02', marker='o', linewidth=2.2, markersize=6.5, label='Temperature (°C)')
                ax_d1.tick_params(axis='y', labelcolor='#d95f02')
                ax_d1.set_ylim(min(temps)-1, max(temps)+1)

                ax_d2 = ax_d1.twinx()
                ax_d2.set_ylabel('Relative Humidity / %', color='#0284c7', fontsize=11.5, fontweight='bold')
                l2 = ax_d2.plot(elapsed_days, humidities, color='#0284c7', marker='s', linewidth=2.2, markersize=6.5, linestyle='--', label='Humidity (%)')
                ax_d2.tick_params(axis='y', labelcolor='#0284c7')
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
            get_image_download_link(fig6, 'fig_combined_publication_summary.png', 'Download Combined Multi-Panel Figure', 't6_pub')

    # ==========================================
    # ZIP PACKAGING (TOP AND BOTTOM BUTTONS)
    # ==========================================
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
            label="📦 Download All Generated Plots in a Single ZIP File",
            data=zip_bytes,
            file_name="all_plots.zip",
            mime="application/zip",
            key="zip_bottom_btn"
        )
