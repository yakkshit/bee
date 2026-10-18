import os
import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
from matplotlib.lines import Line2D

from modules.core.time_parser import filter_post_starvation_data
from modules.parser import (
    parse_batch_df,
    calculate_lt_percentiles,
    clean_cohort_name,
    get_cohort_code,
    get_default_4_cohort_dataframes,
    standardize_observation_df,
    extract_individual_bee_survival,
    AGREED_COHORTS,
    COHORT_CODE_NAMES,
    COHORT_PALETTE,
    COHORT_MARKERS,
    COHORT_SUCROSE_2UL,
    COHORT_ACETONE_5PCT_2UL
)
from modules.survival_stats import (
    render_kaplan_meier_analysis,
    render_log_rank_analysis,
    render_cox_regression_analysis,
    render_logrank_math_card,
    render_cox_math_card,
    render_kaplan_meier_math_card,
    render_methodology_comparison_card
)
from modules.multi_batch_consumption import render_multi_consumption_analysis




def render_multi_batch_tab(settings=None):
    """Renders the Multi-CSV Probit & Survival Comparative Analysis tab."""
    if settings is None:
        settings = {}

    line_width = settings.get('line_width', 2.4)
    marker_size = settings.get('marker_size', 7.0)
    abbott_c = settings.get('abbott_c', 0.0)

    st.header("📈 Multi-CSV File Probit & Survival Comparative Analysis")
    st.markdown("Upload multiple CSV files to perform comparative **Probit Analysis**, plot time-mortality & survival curves using distinct colors and shapes for each CSV file, and compare **LT₅₀** metrics.")

    st.info("🕒 **Bioassay Protocol Active**: Starting observations are strictly post-2h starvation period (Time Zero / baseline d0 is excluded from all kinetics, LT estimations, and hazard models).")

    if 'uploaded_file_dict' not in st.session_state:
        st.session_state['uploaded_file_dict'] = {}
    if 'user_uploaded_files' not in st.session_state:
        st.session_state['user_uploaded_files'] = set()

    for k in list(st.session_state['uploaded_file_dict'].keys()):
        if k not in st.session_state['user_uploaded_files']:
            del st.session_state['uploaded_file_dict'][k]

    col_up_main, col_up_clear = st.columns([3.8, 1.2])
    with col_up_main:
        uploaded_batches = st.file_uploader(
            "📂 Upload CSV Files for Comparison (Multiple files allowed)",
            type=["csv"],
            accept_multiple_files=True,
            key="multi_batch_uploader_key",
            help="Select experimental CSV files to compare. All uploaded files across Tab 1 and Tab 2 are automatically included and compared by default."
        )
    with col_up_clear:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear All Uploads", key="multi_clear_uploads_btn", help="Clears uploaded files and reverts to default 4 benchmark cohorts"):
            st.session_state['uploaded_file_dict'] = {}
            st.session_state['user_uploaded_files'] = set()
            if 'processed_batches' in st.session_state:
                del st.session_state['processed_batches']
            st.rerun()

    # Automatically register any newly uploaded batches into central uploaded_file_dict
    if uploaded_batches:
        multi_upload_sig = tuple((f.name, f.size) for f in uploaded_batches)
        if multi_upload_sig != st.session_state.get('tab2_upload_sig'):
            st.session_state['tab2_upload_sig'] = multi_upload_sig
            new_batch_added = False
            for f in uploaded_batches:
                try:
                    f.seek(0)
                    try:
                        b_raw = pd.read_csv(f, sep=None, engine='python')
                    except Exception:
                        f.seek(0)
                        b_raw = pd.read_csv(f)
                    b_clean = standardize_observation_df(b_raw)
                    b_clean = b_clean.astype(str)
                    b_clean.replace("nan", "", inplace=True)
                    norm_name = clean_cohort_name(f.name, b_clean)
                    st.session_state['user_uploaded_files'].add(f.name)
                    st.session_state['uploaded_file_dict'][f.name] = (norm_name, b_clean)
                    new_batch_added = True
                except Exception as ex:
                    st.error(f"Error reading file {f.name}: {ex}")
            if new_batch_added:
                st.rerun()
    elif 'tab2_upload_sig' in st.session_state:
        del st.session_state['tab2_upload_sig']

    # BY DEFAULT: TAKE ALL UPLOADED FILES!
    raw_batch_dict = {}
    uploaded_dict = st.session_state.get('uploaded_file_dict', {})
    if uploaded_dict:
        for fname, (cohort_name, c_df) in uploaded_dict.items():
            raw_batch_dict[fname] = filter_post_starvation_data(c_df.copy())

        st.success(f"✅ **All Uploaded Cohort Datasets Active ({len(raw_batch_dict)} files):** " + " • ".join([f"**{k}**" for k in raw_batch_dict.keys()]))
    else:
        st.info("💡 **Showing 4 Standard Experimental Cohorts below.** Upload your custom CSV files above to automatically compare all your datasets.")
        default_cohorts = get_default_4_cohort_dataframes()
        default_cohort_filenames = {
            "sucrose solution 2 microliter": "sucrose_2uL_control.csv",
            "5 % acetone 2 microliter": "5pct_acetone_2uL.csv",
            "10 % acetone 2 microliter": "10pct_acetone_2uL.csv",
            "5 % acetone 5 microliter": "5pct_acetone_5uL.csv",
        }
        for c_name, c_df in default_cohorts.items():
            def_fname = default_cohort_filenames.get(c_name, f"{c_name.replace(' ', '_')}.csv")
            raw_batch_dict[def_fname] = c_df

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
            "#2563eb", "#0d9488", "#f59e0b", "#dc2626", "#8b5cf6",
            "#ec4899", "#06b6d4", "#f97316", "#84cc16", "#6366f1"
        ]

        LINE_STYLES = {"Solid": "-", "Dashed": "--", "Dotted": ":", "Dash-Dot": "-."}

        with st.expander("🎨 Customize Cohort Colors, Shapes & Display Names", expanded=True):
            st.markdown("Assign cohorts to the 4 agreed experimental groups, adjust display labels, colors, and markers:")

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

            c_cols = st.columns(min(4, max(1, len(b_keys))))
            for idx, orig_name in enumerate(b_keys):
                col_i = c_cols[idx % len(c_cols)]
                with col_i:
                    st.markdown(f"**Cohort {idx+1}:**")
                    if orig_name in uploaded_dict:
                        resolved_cohort = uploaded_dict[orig_name][0]
                    else:
                        resolved_cohort = clean_cohort_name(orig_name)
                    code_name = get_cohort_code(resolved_cohort)

                    # Show agreed code name badge
                    st.markdown(f"<span style='background:#e0f2fe;color:#0369a1;padding:3px 9px;border-radius:12px;font-size:11.5px;font-weight:bold;'>🏷️ Code: {code_name}</span>", unsafe_allow_html=True)

                    cohort_options = AGREED_COHORTS + ["Custom / Other"]
                    sel_cohort_idx = AGREED_COHORTS.index(resolved_cohort) if resolved_cohort in AGREED_COHORTS else len(AGREED_COHORTS)
                    cohort_choice = st.selectbox("Agreed Protocol Name", cohort_options, index=sel_cohort_idx, key=f"bcohort_{idx}")

                    default_label = orig_name
                    custom_label = st.text_input("Display Label (File Name)", value=default_label, key=f"bname_{idx}")

                    default_c = COHORT_PALETTE.get(cohort_choice, DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])
                    custom_color = st.color_picker("Color", default_c, key=f"bcol_{idx}")

                    pref_marker = COHORT_MARKERS.get(cohort_choice, None)
                    pref_shape_name = next((k for k, v in MARKER_OPTIONS.items() if v == pref_marker), None) if pref_marker else None
                    default_shape_idx = marker_names.index(pref_shape_name) if pref_shape_name in marker_names else (idx % len(marker_names))
                    custom_shape_name = st.selectbox("Shape (Marker)", marker_names, index=default_shape_idx, key=f"bshape_{idx}")
                    custom_shape = MARKER_OPTIONS[custom_shape_name]

                    custom_line_name = st.selectbox("Line Style", list(LINE_STYLES.keys()), index=0, key=f"bline_{idx}")
                    custom_linestyle = LINE_STYLES[custom_line_name]

                    batch_config.append({
                        'orig_name': orig_name,
                        'filename': orig_name,
                        'name': custom_label,
                        'cohort_name': cohort_choice,
                        'code_name': get_cohort_code(cohort_choice),
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
                'orig_name': bcfg['orig_name'],
                'filename': bcfg['filename'],
                'name': bcfg['name'],
                'cohort_name': bcfg['cohort_name'],
                'code_name': bcfg['code_name'],
                'color': bcfg['color'],
                'marker': bcfg['marker'],
                'linestyle': bcfg['linestyle'],
                'raw_df': bcfg['raw_df']
            })
            if 'individual_bees' in p_res and isinstance(p_res['individual_bees'], pd.DataFrame):
                p_res['individual_bees']['cohort'] = bcfg['name']
                p_res['individual_bees']['filename'] = bcfg['filename']
                p_res['individual_bees']['cohort_code'] = bcfg['code_name']
                p_res['individual_bees']['color'] = bcfg['color']
            processed_batches.append(p_res)

        st.session_state['processed_batches'] = processed_batches


        st.markdown("### 📊 Multi-CSV Key Summary")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{len(processed_batches)}</div><div class="kpi-label">CSV Files</div></div>', unsafe_allow_html=True)
        with k2:
            valid_lt50 = [b['lt50'] for b in processed_batches if np.isfinite(b.get('lt50', np.nan)) and b['lt50'] > 0]
            min_lt50_str = f"{int(round(min(valid_lt50)))}d" if valid_lt50 else "N/A"
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{min_lt50_str}</div><div class="kpi-label">Fastest LT50</div></div>', unsafe_allow_html=True)
        with k3:
            max_lt50_str = f"{int(round(max(valid_lt50)))}d" if valid_lt50 else "N/A"
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
            offset = (idx - (num_files - 1) / 2.0) * jitter_val
            x_jittered = b['log_time'] + offset

            ax_p.scatter(
                x_jittered, b['probit_y'],
                color=b['color'], marker=b['marker'],
                s=marker_size**2 * 1.5, edgecolor='black', linewidth=1.1,
                zorder=4
            )

            if len(b['log_time']) > 1:
                y_fit_line = b['intercept_a'] + b['slope_b'] * x_fit_line
                ax_p.plot(
                    x_fit_line, y_fit_line,
                    color=b['color'], linestyle=b['linestyle'], linewidth=line_width,
                    zorder=3
                )

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
        ax_p.spines['top'].set_visible(False)
        ax_p.spines['right'].set_visible(False)

        if legend_loc_opt == "Top (Horizontal)":
            num_cols = min(3, max(1, num_files + 1))
            num_rows = (len(legend_handles_probit) + num_cols - 1) // num_cols
            dynamic_title_pad = 22 + (num_rows * 26)
            ax_p.set_title('Comparative Probit Regression Across CSV Files', fontsize=14, fontweight='bold', pad=dynamic_title_pad)
            ax_p.legend(
                handles=legend_handles_probit, loc='lower center', bbox_to_anchor=(0.5, 1.02),
                ncol=num_cols, frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=9.0
            )
            fig_p_multi.tight_layout()
        elif legend_loc_opt == "Bottom (Horizontal)":
            ax_p.set_title('Comparative Probit Regression Across CSV Files', fontsize=14, fontweight='bold', pad=14)
            ax_p.legend(
                handles=legend_handles_probit, loc='upper center', bbox_to_anchor=(0.5, -0.18),
                ncol=min(3, max(1, num_files + 1)), frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=9.0
            )
            fig_p_multi.tight_layout()
        else:
            ax_p.set_title('Comparative Probit Regression Across CSV Files', fontsize=14, fontweight='bold', pad=14)
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
            st.caption("Observed survival percentage across observation days. For formal non-parametric Kaplan–Meier step curves with censoring, Greenwood CI, Log-Rank tests, and Cox regression, see Section 5 below.")
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

        # 4. Lethal Time (LTx) Bar Chart Comparison (User-Selectable Metric)
        st.markdown("---")
        st.subheader("4. Lethal Time (LTₓ) Comparative Bar Chart")

        c_lt_choice, c_lt_slider = st.columns([2, 1])
        with c_lt_choice:
            lt_selection = st.selectbox(
                "Select Lethal Metric to Compare Across CSV Files:",
                [
                    "LT10 (10% Lethality / Early Onset)",
                    "LT20 (20% Lethality)",
                    "LT25 (25% Lethality - 1st Quartile)",
                    "LT30 (30% Lethality)",
                    "LT50 (50% Lethality - Standard Median)",
                    "LT75 (75% Lethality - 3rd Quartile)",
                    "LT90 (90% Lethality - Acute Lethality)",
                    "Custom Percentile..."
                ],
                index=4,
                key="multi_lt_selector"
            )

        if lt_selection == "Custom Percentile...":
            with c_lt_slider:
                chosen_target_pct = st.slider("Target Mortality (%)", min_value=1, max_value=99, value=50, step=1, key="multi_custom_pct")
                active_metric_label = f"LT{chosen_target_pct}"
        else:
            chosen_target_pct = int(lt_selection.split()[0].replace("LT", ""))
            active_metric_label = f"LT{chosen_target_pct}"

        if len(processed_batches) == 1:
            st.info(f"💡 **Note:** You have only uploaded 1 CSV file. The {active_metric_label} Bar Chart is designed for comparing **multiple** files. For detailed single-file metrics, please refer to the 'Single CSV Analysis' tab or the Summary Table below.")

        # Compute the selected LT_X for each batch
        b_names = []
        b_chosen_lt_vals = []
        b_colors = []

        for b in processed_batches:
            b_names.append(b['name'])
            b_colors.append(b['color'])

            # Look up or compute on the fly
            rec = calculate_lt_percentiles(
                intercept_a=b['intercept_a'],
                slope_b=b['slope_b'],
                start_time=b.get('start_time', 1.0),
                times_numeric=b.get('batch_elapsed_days', b.get('times_numeric', [])),
                mortality_pct=b.get('abbott_corrected_pct', b.get('mortality_pct', [])),
                percentiles=[chosen_target_pct]
            )
            if rec and len(rec) > 0:
                val = rec[0]['effective_elapsed_int']
            else:
                val = np.nan
            b_chosen_lt_vals.append(val)

        fig_lt, ax_lt = plt.subplots(figsize=(8.0, 4.8), dpi=300)
        if show_grid:
            ax_lt.grid(True, linestyle='--', alpha=0.35, color='#cccccc', axis='y', zorder=1)

        bar_width = 0.5 if len(processed_batches) <= 3 else 0.45
        bars = ax_lt.bar(b_names, b_chosen_lt_vals, color=b_colors, edgecolor='black', width=bar_width)
        max_val = max([v for v in b_chosen_lt_vals if np.isfinite(v)], default=1.0)

        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h) and h > 0:
                label_text = f"{int(round(h))} Days"
                ax_lt.text(bar.get_x() + bar.get_width()/2.0, h + (max_val * 0.03 if max_val > 0 else 0.15), label_text, ha='center', va='bottom', fontweight='bold', fontsize=10.5)

        ax_lt.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax_lt.set_ylabel(f'{active_metric_label} (Elapsed Days)', fontsize=12, fontweight='bold')
        ax_lt.set_title(f'Calculated {active_metric_label} Across CSV Files', fontsize=14, fontweight='bold', pad=12)
        ax_lt.spines['top'].set_visible(False)
        ax_lt.spines['right'].set_visible(False)

        plt.xticks(rotation=45, ha='right')
        fig_lt.tight_layout()
        st.pyplot(fig_lt)

        buf_lt = io.BytesIO()
        fig_lt.savefig(buf_lt, format="png", dpi=300, bbox_inches='tight')
        multi_plot_images[f'{active_metric_label.lower()}_bar_comparison.png'] = buf_lt.getvalue()
        plt.close(fig_lt)

        # Comparative Summary Metrics Table
        st.markdown("---")
        st.subheader("📋 Multi-CSV Probit Comparative Metrics Table")
        summary_rows = []
        for idx_b, b in enumerate(processed_batches):
            lt10_val = b.get('lt10', np.nan)
            lt20_val = b.get('lt20', np.nan)
            lt30_val = b.get('lt30', np.nan)
            lt50_val = b.get('lt50', np.nan)
            lt90_val = b.get('lt90', np.nan)

            lt10_str = f"{int(round(lt10_val))}d" if (np.isfinite(lt10_val) and lt10_val > 0) else "N/A"
            lt20_str = f"{int(round(lt20_val))}d" if (np.isfinite(lt20_val) and lt20_val > 0) else "N/A"
            lt30_str = f"{int(round(lt30_val))}d" if (np.isfinite(lt30_val) and lt30_val > 0) else "N/A"
            lt50_str = f"{int(round(lt50_val))}d" if (np.isfinite(lt50_val) and lt50_val > 0) else "N/A"
            lt90_str = f"{int(round(lt90_val))}d" if (np.isfinite(lt90_val) and lt90_val > 0) else "N/A"

            row_data = {
                "CSV File Name": b['name'],
                "Sample Size (N)": b['num_bees'],
                "Days Observed": b['num_days'],
                "Probit Model": f"Y = {b['intercept_a']:.2f} + {b['slope_b']:.2f}x",
                "Slope (b)": round(b['slope_b'], 4),
                "Intercept (a)": round(b['intercept_a'], 4),
                "R² Score": round(b['r_squared'], 4),
                "LT10": lt10_str,
                "LT20": lt20_str,
                "LT30": lt30_str,
                "LT50": lt50_str,
                "LT90": lt90_str,
            }

            if active_metric_label not in ["LT10", "LT20", "LT30", "LT50", "LT90"]:
                cur_chosen_val = b_chosen_lt_vals[idx_b]
                row_data[f"{active_metric_label}"] = f"{int(round(cur_chosen_val))}d" if (np.isfinite(cur_chosen_val) and cur_chosen_val > 0) else "N/A"

            row_data["Final Mortality (%)"] = f"{b['mortality_pct'][-1]:.1f}%" if len(b['mortality_pct']) > 0 else "N/A"
            summary_rows.append(row_data)

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True)

        c_csv, c_zip_top = st.columns(2)
        with c_csv:
            csv_str = summary_df.to_csv(index=False)
            st.download_button("📊 Download Comparative Summary Table (CSV)", data=csv_str, file_name="multi_csv_probit_summary.csv", mime="text/csv", key="dl_multi_probit_summary_csv")

        with c_zip_top:
            if multi_plot_images:
                zip_multi_buf = io.BytesIO()
                with zipfile.ZipFile(zip_multi_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for fn, data in multi_plot_images.items():
                        zf.writestr(fn, data)
                st.download_button("📦 Download Generated Plots (ZIP)", data=zip_multi_buf.getvalue(), file_name="multi_csv_plots.zip", mime="application/zip", key="dl_multi_zip_top")

        # 5. Dedicated Multi-Cohort Sucrose Consumption & Feeding Kinetics Comparative Analysis
        render_multi_consumption_analysis(processed_batches, settings=settings, multi_plot_images=multi_plot_images)

        # 6. Combined Kaplan-Meier Survival Analysis, Log-Rank Test & Cox Regression Across Batches
        st.markdown("---")
        st.subheader("6. Combined Kaplan-Meier Survival Analysis, Log-Rank Test & Cox Regression")
        st.caption("Comprehensive inferential survival framework: non-parametric Kaplan–Meier product-limit curves with step-by-step life tables, Mantel-Cox Log-Rank significance testing, and Cox Proportional Hazards regression.")
        all_cohort_bees = []
        c_map = {}
        for b in processed_batches:
            b_df = b.get('individual_bees')
            if b_df is not None and not b_df.empty:
                b_copy = b_df.copy()
                b_copy['cohort'] = b['name']
                all_cohort_bees.append(b_copy)
                c_map[b['name']] = b['color']

        if len(all_cohort_bees) >= 1:
            multi_survival_df = pd.concat(all_cohort_bees, ignore_index=True)

            subtab_multi_km, subtab_multi_lr, subtab_multi_cox = st.tabs([
                "📉 1. Kaplan–Meier Survival Curves Plot",
                "📊 2. Log-Rank (Mantel-Cox) Test & Observed vs. Expected Plot",
                "🌲 3. Cox Proportional Hazards Regression & Forest Plot"
            ])

            with subtab_multi_km:
                c_mkm_m, c_mkm_s = st.columns([2.2, 1.0])
                with c_mkm_m:
                    render_kaplan_meier_analysis(multi_survival_df, group_col='cohort', color_map=c_map, key_suffix="multi_km")
                with c_mkm_s:
                    card_tab_km, card_tab_comp_km = st.tabs(["📐 Kaplan–Meier", "📊 Method Comparison"])
                    with card_tab_km:
                        render_kaplan_meier_math_card()
                    with card_tab_comp_km:
                        render_methodology_comparison_card()

            with subtab_multi_lr:
                c_mlr_m, c_mlr_s = st.columns([2.2, 1.0])
                with c_mlr_m:
                    render_log_rank_analysis(multi_survival_df, group_col='cohort', color_map=c_map, key_suffix="multi_lr")
                with c_mlr_s:
                    card_tab_lr, card_tab_comp_lr = st.tabs(["📐 Log-Rank Math", "📊 Method Comparison"])
                    with card_tab_lr:
                        render_logrank_math_card()
                    with card_tab_comp_lr:
                        render_methodology_comparison_card()

            with subtab_multi_cox:
                c_mcox_m, c_mcox_s = st.columns([2.2, 1.0])
                with c_mcox_m:
                    render_cox_regression_analysis(multi_survival_df, group_col='cohort', key_suffix="multi_cox")
                with c_mcox_s:
                    card_tab_cox, card_tab_mc = st.tabs(["🌲 Cox Model", "📊 Method Comparison"])
                    with card_tab_cox:
                        render_cox_math_card()
                    with card_tab_mc:
                        render_methodology_comparison_card()
        else:
            st.info("No cohort survival data available to perform Log-Rank and Cox Regression.")

        # Bottom Unified ZIP Downloader packaging all plots across all sections
        if multi_plot_images:
            st.markdown("---")
            zip_multi_buf_bottom = io.BytesIO()
            with zipfile.ZipFile(zip_multi_buf_bottom, "w", zipfile.ZIP_DEFLATED) as zf:
                for fn, data in multi_plot_images.items():
                    zf.writestr(fn, data)
            st.download_button(
                "📦 Download All Multi-CSV Plots & Figures (ZIP Archive)",
                data=zip_multi_buf_bottom.getvalue(),
                file_name="all_multi_cohort_plots.zip",
                mime="application/zip",
                key="dl_multi_zip_bottom"
            )
