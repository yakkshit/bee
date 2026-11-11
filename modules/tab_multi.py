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
    clean_display_name,
    get_cohort_code,
    get_default_4_cohort_dataframes,
    standardize_observation_df,
    extract_individual_bee_survival,
    extract_key_from_df,
    get_dataset_key_or_filename,
    compute_adaptive_marker_sizes,
    AGREED_COHORTS,
    ALL_COHORT_OPTIONS,
    COHORT_CODE_NAMES,
    COHORT_PALETTE,
    COHORT_MARKERS,
    COHORT_SUCROSE_2UL,
    COHORT_ACETONE_5PCT_2UL,
    COHORT_BLINDED
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


@st.cache_data(show_spinner=False)
def load_and_standardize_uploaded_file(file_bytes: bytes, filename: str):
    """
    Caches uploaded CSV reading and standardization.
    Ensures file parsing occurs once in memory and is reused across UI interactions.
    """
    import io
    try:
        b_raw = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine='python')
    except Exception:
        b_raw = pd.read_csv(io.BytesIO(file_bytes))
    b_clean = standardize_observation_df(b_raw)
    b_clean = b_clean.astype(str)
    b_clean.replace("nan", "", inplace=True)
    return b_clean


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

    up_dict = st.session_state.get('uploaded_file_dict', {})
    user_files = st.session_state.get('user_uploaded_files', set())
    for k in list(up_dict.keys()):
        if k not in user_files:
            del up_dict[k]
    st.session_state['uploaded_file_dict'] = up_dict

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
                    b_clean = load_and_standardize_uploaded_file(f.getvalue(), f.name)
                    key_val = extract_key_from_df(b_clean)
                    ident_name = key_val if key_val else f.name
                    norm_name = clean_cohort_name(ident_name, b_clean, fallback_to_blinded=True)
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

    # CSV Data Quality & Bioassay Diagnostics Banner
    from modules.core.csv_auditor import audit_batch_collection, render_csv_diagnostics_banner
    multi_diagnostics = audit_batch_collection(raw_batch_dict)
    if multi_diagnostics:
        render_csv_diagnostics_banner(multi_diagnostics, container_key="tab2_csv_audit")

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

        with st.expander("🎨 Customize Cohort Treatments, Colors, Markers & Legend Display", expanded=True):
            st.markdown("Assign cohorts to experimental treatments (or maintain Blinded Mode), adjust display labels, colors, and markers:")

            col_ctrl_top1, col_ctrl_top2 = st.columns([2.5, 1.5])
            with col_ctrl_top1:
                legend_format = st.selectbox(
                    "🏷️ Plot Legend Label Style (Global):",
                    [
                        "Key / File Name (Default / Blinded Study Mode)",
                        "Treatment Solution (Short Code, e.g. 5% Acetone 2 µL)",
                        "Treatment Solution (Full Protocol Name)",
                        "Key / File Name + Treatment Solution",
                        "Custom Display Labels (Specified Below)"
                    ],
                    index=0,
                    key="multi_legend_format_pref",
                    help="Switch how plot legends represent your datasets. If a Key column value is present, it will be used in legends; otherwise the file name is used. Once unblinded, switch to Treatment Solution or Key/File Name + Treatment."
                )
            with col_ctrl_top2:
                legend_detail = st.selectbox("Probit Legend Detail", ["Label + Equation + R²", "Label + R² Only", "Label Only"], index=0)

            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
            with col_ctrl1:
                jitter_val = st.slider("Point Separation / Anti-Overlap Offset", min_value=0.0, max_value=0.05, value=0.015, step=0.003, help="Offsets data points horizontally side-by-side so markers from different CSV files don't overlap directly on top of each other.")
            with col_ctrl2:
                legend_loc_opt = st.selectbox("Legend Placement", ["Right (Outside)", "Top (Horizontal)", "Bottom (Horizontal)"], index=0)
            with col_ctrl3:
                show_grid = st.checkbox("Show Chart Gridlines", value=True)

            st.markdown("---")
            batch_config = []
            b_keys = list(raw_batch_dict.keys())
            used_colors_multi = set()
            used_markers_multi = set()
            DISTINCT_MULTI_PALETTE = [
                "#0284c7",  # Sky blue
                "#059669",  # Emerald green
                "#d97706",  # Amber / warm gold
                "#dc2626",  # Red
                "#7c3aed",  # Purple
                "#0891b2",  # Cyan / teal
                "#db2777",  # Pink / rose
                "#ea580c",  # Bright orange
                "#16a34a",  # Forest green
                "#4f46e5",  # Indigo
                "#ca8a04",  # Dark yellow
                "#64748b"   # Slate grey
            ]
            DISTINCT_MARKER_LIST = list(MARKER_OPTIONS.values())

            c_cols = st.columns(min(4, max(1, len(b_keys))))
            for idx, orig_name in enumerate(b_keys):
                col_i = c_cols[idx % len(c_cols)]
                with col_i:
                    b_df = raw_batch_dict[orig_name]
                    key_val = extract_key_from_df(b_df)
                    key_or_file = key_val if key_val else orig_name

                    if key_val:
                        st.markdown(f"**Dataset {idx+1}: {orig_name}**  \n<span style='font-size:11px;color:#0f766e;'>🔑 Key: <b>{key_val}</b></span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**Dataset {idx+1}: {orig_name}**")

                    if orig_name in uploaded_dict:
                        resolved_cohort = uploaded_dict[orig_name][0]
                    else:
                        resolved_cohort = clean_cohort_name(key_or_file, df=b_df, fallback_to_blinded=True)
                    code_name = get_cohort_code(resolved_cohort)

                    # Show agreed code name badge
                    badge_color = "#f1f5f9" if code_name == "Blinded" else "#e0f2fe"
                    badge_text_col = "#475569" if code_name == "Blinded" else "#0369a1"
                    st.markdown(f"<span style='background:{badge_color};color:{badge_text_col};padding:3px 9px;border-radius:12px;font-size:11.5px;font-weight:bold;'>🏷️ Code: {code_name}</span>", unsafe_allow_html=True)

                    cohort_options = ALL_COHORT_OPTIONS
                    if resolved_cohort in cohort_options:
                        sel_cohort_idx = cohort_options.index(resolved_cohort)
                    elif resolved_cohort in AGREED_COHORTS:
                        sel_cohort_idx = cohort_options.index(resolved_cohort)
                    else:
                        sel_cohort_idx = 0  # Default to None / Blinded (Unknown)

                    cohort_choice = st.selectbox("Treatment Solution", cohort_options, index=sel_cohort_idx, key=f"bcohort_{idx}")
                    active_code = get_cohort_code(cohort_choice)

                    # Auto-compute label based on selected legend format
                    is_blinded = (cohort_choice == COHORT_BLINDED or cohort_choice == "None")
                    clean_name = clean_display_name(key_or_file, b_df)
                    if legend_format == "Key / File Name (Default / Blinded Study Mode)":
                        auto_label = clean_name
                    elif legend_format == "Treatment Solution (Short Code, e.g. 5% Acetone 2 µL)":
                        auto_label = active_code if not is_blinded else clean_name
                    elif legend_format == "Treatment Solution (Full Protocol Name)":
                        auto_label = cohort_choice if not is_blinded else clean_name
                    elif legend_format == "Key / File Name + Treatment Solution":
                        auto_label = f"{clean_name} ({active_code})" if not is_blinded else clean_name
                    else:
                        auto_label = clean_name

                    custom_label = st.text_input("Display Label (Plot Legend)", value=auto_label, key=f"bname_{idx}_{legend_format}")

                    # Guarantee distinct color per object: assign distinct color by index
                    avail_cols = [c for c in DISTINCT_MULTI_PALETTE if c not in used_colors_multi]
                    default_c = avail_cols[0] if avail_cols else DISTINCT_MULTI_PALETTE[idx % len(DISTINCT_MULTI_PALETTE)]

                    # Guarantee distinct marker per object: assign distinct marker by index
                    avail_markers = [m for m in DISTINCT_MARKER_LIST if m not in used_markers_multi]
                    pref_marker = avail_markers[0] if avail_markers else DISTINCT_MARKER_LIST[idx % len(DISTINCT_MARKER_LIST)]

                    custom_color = st.color_picker("Color", default_c, key=f"bcol_{orig_name}_{idx}")
                    # If this color was already taken by another dataset in this session, fallback to unused distinct color
                    if custom_color in used_colors_multi:
                        avail_cols = [c for c in DISTINCT_MULTI_PALETTE if c not in used_colors_multi]
                        custom_color = avail_cols[0] if avail_cols else DISTINCT_MULTI_PALETTE[idx % len(DISTINCT_MULTI_PALETTE)]
                    used_colors_multi.add(custom_color)

                    pref_shape_name = next((k for k, v in MARKER_OPTIONS.items() if v == pref_marker), None) if pref_marker else None
                    default_shape_idx = marker_names.index(pref_shape_name) if pref_shape_name in marker_names else (idx % len(marker_names))
                    custom_shape_name = st.selectbox("Shape (Marker)", marker_names, index=default_shape_idx, key=f"bshape_{orig_name}_{idx}")
                    custom_shape = MARKER_OPTIONS[custom_shape_name]
                    if custom_shape in used_markers_multi:
                        avail_m_names = [k for k, v in MARKER_OPTIONS.items() if v not in used_markers_multi]
                        if avail_m_names:
                            custom_shape_name = avail_m_names[0]
                            custom_shape = MARKER_OPTIONS[custom_shape_name]
                    used_markers_multi.add(custom_shape)

                    custom_line_name = st.selectbox("Line Style", list(LINE_STYLES.keys()), index=0, key=f"bline_{idx}")
                    custom_linestyle = LINE_STYLES[custom_line_name]

                    batch_config.append({
                        'orig_name': orig_name,
                        'filename': orig_name,
                        'name': custom_label,
                        'cohort_name': cohort_choice,
                        'code_name': active_code,
                        'color': custom_color,
                        'marker': custom_shape,
                        'marker_name': custom_shape_name,
                        'linestyle': custom_linestyle,
                        'raw_df': raw_batch_dict[orig_name],
                        'key': key_val
                    })


        processed_batches = []
        for bcfg in batch_config:
            p_res = parse_batch_df(bcfg['raw_df'], abbott_c=abbott_c)
            # Shallow copy to prevent mutating cached dictionary
            p_item = dict(p_res)
            p_item.update({
                'orig_name': bcfg['orig_name'],
                'filename': bcfg['filename'],
                'name': bcfg['name'],
                'cohort_name': bcfg['cohort_name'],
                'code_name': bcfg['code_name'],
                'color': bcfg['color'],
                'marker': bcfg['marker'],
                'linestyle': bcfg['linestyle'],
                'raw_df': bcfg['raw_df'],
                'key': bcfg.get('key', '')
            })
            if 'individual_bees' in p_item and isinstance(p_item['individual_bees'], pd.DataFrame):
                indiv_df = p_item['individual_bees'].copy()
                indiv_df['cohort'] = bcfg['name']
                indiv_df['filename'] = bcfg['filename']
                indiv_df['cohort_code'] = bcfg['code_name']
                indiv_df['color'] = bcfg['color']
                p_item['individual_bees'] = indiv_df
            processed_batches.append(p_item)

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

        # Educational Bioassay Statistics Guide
        with st.expander("💡 Statistical Guide: Understanding SD, SE, 95% CI & Degrees of Freedom (df)", expanded=False):
            st.markdown(r"""
            ### 🔬 Standard Deviation vs Standard Error: What is the Difference?

            | Statistical Concept | Mathematical Definition | Bioassay Interpretation (Bee Trials) | Where It Is Used in This Studio |
            | :--- | :--- | :--- | :--- |
            | **Standard Deviation (SD)** | $\\mathrm{SD} = \\sqrt{\\frac{\\sum (x_i - \\bar{x})^2}{N - 1}}$ | **Biological Diversity / Individual Variation**: How widely individual bees differ from the group average (e.g., variation in individual sucrose consumption). Adding more bees does *not* make SD shrink to 0. | Daily consumption variability, environmental variance. |
            | **Standard Error (SE / SEM)** | $\\mathrm{SE} = \\frac{\\mathrm{SD}}{\\sqrt{N}}$ | **Estimation Precision of the Mean**: How accurately your experimental sample reflects the *true population average*. Shrinks towards 0 as sample size $N$ increases. | Error whiskers on Mortality trajectories ($\pm \\mathrm{SE}$), Survival curves ($\pm \\mathrm{SE}$), Probit slope uncertainty $\\mathrm{SE}(b)$. |
            | **95% Confidence Interval (95% CI)** | $95\\%\\,\\mathrm{CI} = \\bar{x} \\pm 1.96 \\cdot \\mathrm{SE}$ | **Plausible Parameter Range**: With 95% statistical certainty, the true population mean falls within this range. | Shaded ribbons on Probit regression lines, shaded confidence bands on survival curves, LT50 whiskers. |
            | **Degrees of Freedom ($df$)** | $df = N - 1$ (Sample)<br>$df = N_{\\mathrm{points}} - 2$ (Probit) | **Independent Information**: Number of data points that are free to vary after fitting estimated statistical parameters. | Sample $df = N - 1$ on mortality curves; regression $df = N_{\\mathrm{points}} - 2$ on Probit equations ($Y = a + bx$). |
            """)

        multi_plot_images = {}
        num_files = len(processed_batches)

        # 1. Probit Regression CSV File Comparison Plot (130 DPI for rapid on-screen reactivity)
        st.subheader("1. Probit Regression Time-Mortality Comparison Across CSV Files")
        st.caption("Fitted Probit regression lines ($Y = a + bx$) across all CSV files with shaded 95% Confidence Interval ribbons, slope standard error SE(b), and residual degrees of freedom ($df = N_{points} - 2$).")

        col_pr1, col_pr2 = st.columns([1.6, 1.4])
        with col_pr1:
            probit_view_mode = st.radio(
                "Probit Plot Presentation:",
                ["All-in-One Overlay (Comparative)", "Faceted Subplots (Small Multiples - Uncluttered Per-Cohort Grid)"],
                horizontal=True,
                key="multi_probit_view_mode"
            )
        with col_pr2:
            probit_x_range_opt = st.radio(
                "Probit X-Axis (Log10 Time) Range:",
                ["Extend to -0.25 (Sub-day / early post-starvation)", "Observed Data Range (Day 1 baseline)"],
                horizontal=True,
                key="multi_probit_x_range_opt"
            )

        all_log_times = []
        for b in processed_batches:
            if len(b['log_time']) > 0:
                all_log_times.extend(b['log_time'])

        min_log = min(all_log_times) if all_log_times else 0.0
        max_log = max(all_log_times) if all_log_times else 1.0

        if probit_x_range_opt.startswith("Extend to -0.25"):
            x_min = min(-0.25, min_log - 0.04) if min_log < -0.25 else -0.25
        else:
            x_min = max(-0.02, min_log - 0.04) if all_log_times else 0.0

        x_max = max(max_log + 0.08, 1.25)
        x_fit_line = np.linspace(x_min, x_max, 120)

        # Smart adaptive anti-overlap marker sizing across all cohorts
        probit_x_all = [b['log_time'] for b in processed_batches]
        probit_y_all = [b['probit_y'] for b in processed_batches]
        adaptive_sizes_probit = compute_adaptive_marker_sizes(
            probit_x_all, probit_y_all,
            base_size=marker_size, size_step=2.6,
            tol_x=0.025, tol_y=0.12
        )

        if probit_view_mode == "All-in-One Overlay (Comparative)":
            fig_p_multi, ax_p = plt.subplots(figsize=(10.5, 6.0), dpi=130)

            if show_grid:
                ax_p.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', zorder=1)

            legend_handles_probit = []

            for idx, b in enumerate(processed_batches):
                offset = (idx - (num_files - 1) / 2.0) * jitter_val
                x_jittered = b['log_time'] + offset
                b_sizes = adaptive_sizes_probit[idx]

                # Layering: earlier batches get slightly lower zorder, later batches get higher zorder
                # Overlapping points become concentric layered glyphs so both outer and inner symbols are visible
                ax_p.scatter(
                    x_jittered, b['probit_y'],
                    color=b['color'], marker=b['marker'],
                    s=(b_sizes * 1.25)**2, edgecolor='black', linewidth=1.1,
                    zorder=4 + (num_files - idx)
                )

                b_log_t = np.array(b['log_time'], dtype=float)
                b_prob_y = np.array(b['probit_y'], dtype=float)
                n_pts = len(b_log_t)
                df_regr = b.get('df_regression', max(1, n_pts - 2))
                se_b_val = b.get('se_slope', 0.0)

                if n_pts > 1:
                    y_fit_line = b['intercept_a'] + b['slope_b'] * x_fit_line
                    ax_p.plot(
                        x_fit_line, y_fit_line,
                        color=b['color'], linestyle=b['linestyle'], linewidth=line_width,
                        zorder=3
                    )

                    # Shaded 95% Confidence Interval Band for Regression Line
                    if n_pts > 2:
                        y_pred_obs = b['intercept_a'] + b['slope_b'] * b_log_t
                        resids = b_prob_y - y_pred_obs
                        s_err = np.sqrt(np.sum(resids**2) / max(1, df_regr))
                        x_bar = np.mean(b_log_t)
                        ss_x = np.sum((b_log_t - x_bar)**2)
                        if ss_x > 1e-9:
                            se_band = s_err * np.sqrt(1.0 / n_pts + (x_fit_line - x_bar)**2 / ss_x)
                        else:
                            se_band = np.full_like(x_fit_line, max(0.1, s_err))
                        ci_band = 1.96 * se_band
                        ax_p.fill_between(
                            x_fit_line,
                            y_fit_line - ci_band,
                            y_fit_line + ci_band,
                            color=b['color'], alpha=0.10, zorder=2
                        )

                # Preserve treatment keys and percentages (e.g. '5% Acetone', '10% Acetone', '50% Sucrose Solution')
                import re
                raw_b_name = b.get('name', '')
                clean_lbl_name = raw_b_name
                if re.search(r'(?:batch|fm_)', raw_b_name, flags=re.IGNORECASE):
                    cleaned = re.sub(r'^(?:\d+[\s_\-]+)?(?:batch[\s_\-]+)?(?:fm_\d+[\s_\-]+)?', '', raw_b_name, flags=re.IGNORECASE).strip()
                    cleaned = re.sub(r'\s*\(\d+\)$', '', cleaned).strip()
                    if cleaned:
                        clean_lbl_name = cleaned

                if "Equation + R²" in legend_detail:
                    lbl = f"{clean_lbl_name}: Y={b['intercept_a']:.2f}+{b['slope_b']:.2f}x (R²={b['r_squared']:.3f}, df={df_regr})"
                elif "R² Only" in legend_detail:
                    lbl = f"{clean_lbl_name} (R²={b['r_squared']:.3f})"
                else:
                    lbl = f"{clean_lbl_name} (df={df_regr})"

                legend_handles_probit.append(
                    Line2D([0], [0], color=b['color'], marker=b['marker'], linestyle=b['linestyle'],
                           linewidth=line_width, markersize=marker_size, markeredgecolor='black',
                           label=lbl)
                )

            legend_handles_probit.append(
                Line2D([0], [0], color='#64748b', linestyle='--', linewidth=1.5, label='LT50 Ref (P=5.0)')
            )

            ax_p.axhline(5.0, color='#64748b', linestyle='--', linewidth=1.5, zorder=2)
            ax_p.set_xlim(x_min, x_max)
            ax_p.set_ylim(2.5, 9.2)
            ax_p.yaxis.set_major_locator(MultipleLocator(1.0))

            ax_p.set_xlabel('Log10(Observation Time / Days)', fontsize=12.5, fontweight='bold', labelpad=8)
            ax_p.set_ylabel('Probit Value (P)', fontsize=12.5, fontweight='bold', labelpad=8)
            ax_p.spines['top'].set_visible(False)
            ax_p.spines['right'].set_visible(False)

            if legend_loc_opt == "Right (Outside)":
                ax_p.set_title('Comparative Probit Regression Across CSV Files', fontsize=14, fontweight='bold', pad=14)
                fig_p_multi.subplots_adjust(left=0.09, right=0.68, top=0.92, bottom=0.12)
                ax_p.legend(
                    handles=legend_handles_probit, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                    frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=9.0
                )
            elif legend_loc_opt == "Top (Horizontal)":
                num_cols = min(2, max(1, num_files + 1))
                num_rows = (len(legend_handles_probit) + num_cols - 1) // num_cols
                top_margin = max(0.68, 0.94 - (num_rows * 0.065))
                dynamic_title_pad = 20 + (num_rows * 24)
                ax_p.set_title('Comparative Probit Regression Across CSV Files', fontsize=14, fontweight='bold', pad=dynamic_title_pad)
                fig_p_multi.subplots_adjust(left=0.09, right=0.95, top=top_margin, bottom=0.12)
                ax_p.legend(
                    handles=legend_handles_probit, loc='lower center', bbox_to_anchor=(0.5, 1.03),
                    ncol=num_cols, frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=9.0
                )
            else:
                num_cols = min(2, max(1, num_files + 1))
                num_rows = (len(legend_handles_probit) + num_cols - 1) // num_cols
                bot_margin = min(0.36, 0.12 + (num_rows * 0.065))
                ax_p.set_title('Comparative Probit Regression Across CSV Files', fontsize=14, fontweight='bold', pad=14)
                fig_p_multi.subplots_adjust(left=0.09, right=0.95, top=0.92, bottom=bot_margin)
                ax_p.legend(
                    handles=legend_handles_probit, loc='upper center', bbox_to_anchor=(0.5, -0.16),
                    ncol=num_cols, frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=9.0
                )

            st.pyplot(fig_p_multi)

            buf_p = io.BytesIO()
            fig_p_multi.savefig(buf_p, format="png", dpi=300, bbox_inches='tight')
            multi_plot_images['probit_multi_csv_comparison.png'] = buf_p.getvalue()
            st.download_button("📥 Download Probit Comparison Chart (PNG)", data=buf_p.getvalue(), file_name="probit_multi_csv_comparison.png", mime="image/png")
            plt.close(fig_p_multi)

        else:
            # Faceted Subplots Grid (Small Multiples - Uncluttered Per-Cohort Grid)
            n_cols_f = min(3, max(1, num_files))
            n_rows_f = (num_files + n_cols_f - 1) // n_cols_f
            fig_facet, axes_facet = plt.subplots(n_rows_f, n_cols_f, figsize=(5.0 * n_cols_f, 4.0 * n_rows_f), dpi=130, squeeze=False)

            for idx, b in enumerate(processed_batches):
                r_idx = idx // n_cols_f
                c_idx = idx % n_cols_f
                ax_f = axes_facet[r_idx, c_idx]

                if show_grid:
                    ax_f.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', zorder=1)

                b_log_t = np.array(b['log_time'], dtype=float)
                b_prob_y = np.array(b['probit_y'], dtype=float)
                n_pts = len(b_log_t)
                df_regr = b.get('df_regression', max(1, n_pts - 2))
                se_b_val = b.get('se_slope', 0.0)

                # Observed points
                ax_f.scatter(
                    b_log_t, b_prob_y,
                    color=b['color'], marker=b['marker'],
                    s=marker_size**2 * 1.5, edgecolor='black', linewidth=1.1,
                    zorder=4, label='Observed'
                )

                # Fitted line & 95% CI ribbon
                if n_pts > 1:
                    y_fit_line = b['intercept_a'] + b['slope_b'] * x_fit_line
                    ax_f.plot(
                        x_fit_line, y_fit_line,
                        color=b['color'], linestyle=b['linestyle'], linewidth=line_width,
                        zorder=3, label='Probit Fit'
                    )

                    if n_pts > 2:
                        y_pred_obs = b['intercept_a'] + b['slope_b'] * b_log_t
                        resids = b_prob_y - y_pred_obs
                        s_err = np.sqrt(np.sum(resids**2) / max(1, df_regr))
                        x_bar = np.mean(b_log_t)
                        ss_x = np.sum((b_log_t - x_bar)**2)
                        se_band = s_err * np.sqrt(1.0 / n_pts + (x_fit_line - x_bar)**2 / ss_x) if ss_x > 1e-9 else np.full_like(x_fit_line, max(0.1, s_err))
                        ci_band = 1.96 * se_band
                        ax_f.fill_between(
                            x_fit_line, y_fit_line - ci_band, y_fit_line + ci_band,
                            color=b['color'], alpha=0.15, zorder=2, label='95% CI Ribbon'
                        )

                # Reference Line at P = 5.0 (LT50)
                ax_f.axhline(5.0, color='#64748b', linestyle='--', linewidth=1.2, zorder=2)

                # If LT50 is reached, mark intersection and drop line
                lt50_val = b.get('lt50', np.nan)
                if np.isfinite(lt50_val) and lt50_val > 0:
                    log_lt50 = np.log10(lt50_val)
                    if x_min <= log_lt50 <= x_max:
                        ax_f.axvline(log_lt50, color='#64748b', linestyle=':', linewidth=1.1, zorder=2)
                        ax_f.plot([log_lt50], [5.0], marker='*', color='#dc2626', markersize=10, markeredgecolor='black', zorder=5)

                import re
                raw_b_name = b.get('name', '')
                clean_lbl_name = raw_b_name
                if re.search(r'(?:batch|fm_)', raw_b_name, flags=re.IGNORECASE):
                    cleaned = re.sub(r'^(?:\d+[\s_\-]+)?(?:batch[\s_\-]+)?(?:fm_\d+[\s_\-]+)?', '', raw_b_name, flags=re.IGNORECASE).strip()
                    cleaned = re.sub(r'\s*\(\d+\)$', '', cleaned).strip()
                    if cleaned:
                        clean_lbl_name = cleaned

                ax_f.set_title(clean_lbl_name, fontsize=11.5, fontweight='bold', pad=8, color=b['color'])
                ax_f.set_xlim(x_min, x_max)
                ax_f.set_ylim(2.5, 9.2)
                ax_f.yaxis.set_major_locator(MultipleLocator(1.0))
                ax_f.spines['top'].set_visible(False)
                ax_f.spines['right'].set_visible(False)

                lt_str = f"LT50 = {lt50_val:.1f}d" if (np.isfinite(lt50_val) and lt50_val > 0) else "LT50: Not Reached"
                stat_box_text = f"Y = {b['intercept_a']:.2f} + {b['slope_b']:.2f}x\nR² = {b['r_squared']:.3f} | df = {df_regr}\nSE(b) = {se_b_val:.2f} | {lt_str}"
                ax_f.text(
                    0.05, 0.95, stat_box_text,
                    transform=ax_f.transAxes, ha='left', va='top', fontsize=8.0, fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='#cbd5e1', alpha=0.90, boxstyle='round,pad=0.3')
                )

                if r_idx == n_rows_f - 1 or idx + n_cols_f >= num_files:
                    ax_f.set_xlabel('Log10(Time / Days)', fontsize=10.5, fontweight='bold')
                if c_idx == 0:
                    ax_f.set_ylabel('Probit Value (P)', fontsize=10.5, fontweight='bold')

            # Hide unused subplot axes
            for empty_idx in range(num_files, n_rows_f * n_cols_f):
                er = empty_idx // n_cols_f
                ec = empty_idx % n_cols_f
                axes_facet[er, ec].set_visible(False)

            fig_facet.tight_layout()
            st.pyplot(fig_facet)

            buf_f = io.BytesIO()
            fig_facet.savefig(buf_f, format="png", dpi=300, bbox_inches='tight')
            multi_plot_images['probit_faceted_grid_comparison.png'] = buf_f.getvalue()
            st.download_button("📥 Download Faceted Probit Comparison Grid (PNG)", data=buf_f.getvalue(), file_name="probit_faceted_grid_comparison.png", mime="image/png")
            plt.close(fig_facet)

        st.markdown("---")

        col_ms_top1, col_ms_top2 = st.columns([1.5, 1.5])
        with col_ms_top1:
            mort_surv_layout = st.radio(
                "Display Layout for Mortality & Survival Curves:",
                ["Expanded Stacked View (Full Width - Recommended)", "Side-by-Side (Compact)"],
                horizontal=True,
                key="multi_mort_surv_layout"
            )
        with col_ms_top2:
            mort_surv_err_mode = st.radio(
                "Uncertainty Display (SE & 95% CI):",
                ["Show Mean ± SE Whiskers & 95% CI Ribbons (Default)", "Clean Curves Only (Without SE/SD/CI)"],
                horizontal=True,
                key="multi_mort_surv_err_mode"
            )

        show_uncertainty = mort_surv_err_mode.startswith("Show Mean ± SE")

        all_times_flat = []
        for b in processed_batches:
            if len(b['times_numeric']) > 0:
                all_times_flat.extend(b['times_numeric'])
        max_t_flat = max(all_times_flat) if all_times_flat else 8.0

        is_stacked = mort_surv_layout.startswith("Expanded")

        # Smart adaptive anti-overlap marker sizing across all cohorts for mortality & survival
        mort_x_all = [b['times_numeric'] for b in processed_batches]
        mort_y_all = [b['abbott_corrected_pct'] for b in processed_batches]
        adaptive_sizes_mort = compute_adaptive_marker_sizes(
            mort_x_all, mort_y_all,
            base_size=marker_size, size_step=2.5,
            tol_x=0.08, tol_y=3.5
        )

        surv_x_all = [b['times_numeric'] for b in processed_batches]
        surv_y_all = [b['survival_pct'] for b in processed_batches]
        adaptive_sizes_surv = compute_adaptive_marker_sizes(
            surv_x_all, surv_y_all,
            base_size=marker_size, size_step=2.5,
            tol_x=0.08, tol_y=3.5
        )

        def render_mortality_chart(full_width=False):
            st.subheader("2. Mortality Curve Comparison")
            if show_uncertainty:
                st.caption("Abbott-corrected mortality trajectories across observation days. Whiskers indicate ± Standard Error (SE); shaded ribbon represents 95% Confidence Interval ($p \\pm 1.96 \\cdot \\mathrm{SE}$); sample $df = N - 1$.")
            else:
                st.caption("Abbott-corrected mortality trajectories across observation days (clean curves view without SE/SD). Adaptive concentric markers highlight overlapping observations.")
            fig_w = 11.0 if full_width else 7.5
            fig_h = 5.5 if full_width else 5.2
            fig_m_multi, ax_m = plt.subplots(figsize=(fig_w, fig_h), dpi=130)
            if show_grid:
                ax_m.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', zorder=1)

            m_handles = []
            for idx, b in enumerate(processed_batches):
                offset = (idx - (num_files - 1) / 2.0) * (jitter_val * 6.0)
                x_days_jittered = b['times_numeric'] + offset
                m_pts = np.array(b['abbott_corrected_pct'], dtype=float)
                se_pts = b.get('mortality_se', np.zeros_like(m_pts))

                if show_uncertainty and se_pts is not None and len(se_pts) == len(m_pts):
                    ci_l = np.clip(m_pts - 1.96 * se_pts, 0, 100)
                    ci_u = np.clip(m_pts + 1.96 * se_pts, 0, 100)
                    ax_m.fill_between(x_days_jittered, ci_l, ci_u, color=b['color'], alpha=0.10, zorder=2)
                    ax_m.errorbar(
                        x_days_jittered, m_pts, yerr=se_pts,
                        fmt='none', ecolor=b['color'], elinewidth=1.2, capsize=3.0, capthick=1.0,
                        alpha=0.85, zorder=3
                    )

                # Draw trajectory line
                ax_m.plot(
                    x_days_jittered, m_pts,
                    color=b['color'], linestyle=b['linestyle'],
                    linewidth=line_width, zorder=3
                )

                # Draw adaptive concentric markers (layering: earlier drawn series gets larger size and lower zorder)
                b_m_sizes = adaptive_sizes_mort[idx]
                ax_m.scatter(
                    x_days_jittered, m_pts,
                    s=b_m_sizes**2,
                    color=b['color'], marker=b['marker'],
                    edgecolor='black', linewidth=1.1,
                    zorder=4 + (num_files - idx)
                )

                lbl_tag = f"{b['name']} (df={b.get('df_sample', b['num_bees']-1)})" if show_uncertainty else b['name']
                m_handles.append(
                    Line2D([0], [0], color=b['color'], marker=b['marker'], linestyle=b['linestyle'],
                           linewidth=line_width, markersize=marker_size, markeredgecolor='black',
                           label=lbl_tag)
                )
            m_handles.append(Line2D([0], [0], color='#64748b', linestyle=':', linewidth=1.4, label='50% Mortality'))
            ax_m.axhline(50.0, color='#64748b', linestyle=':', linewidth=1.4, zorder=2)
            ax_m.set_xlabel('Observation Time / Days', fontsize=12, fontweight='bold', labelpad=7)
            y_lbl_m = 'Abbott Corrected Mortality / % (Mean ± SE)' if show_uncertainty else 'Abbott Corrected Mortality / %'
            ax_m.set_ylabel(y_lbl_m, fontsize=12, fontweight='bold', labelpad=7)
            ax_m.set_xlim(0.8, max_t_flat * 1.05)
            ax_m.set_ylim(-2, 105)
            ax_m.spines['top'].set_visible(False)
            ax_m.spines['right'].set_visible(False)

            if full_width or legend_loc_opt == "Right (Outside)":
                ax_m.legend(handles=m_handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=8.5)
                fig_m_multi.tight_layout(rect=[0, 0, 0.78 if full_width else 0.72, 1.0])
            elif legend_loc_opt == "Bottom (Horizontal)":
                ax_m.legend(handles=m_handles, loc='upper center', bbox_to_anchor=(0.5, -0.18),
                            ncol=min(3, max(1, num_files + 1)), frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=8.5)
                fig_m_multi.tight_layout()
            else:
                ax_m.legend(handles=m_handles, loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=8.5)
                fig_m_multi.tight_layout()

            st.pyplot(fig_m_multi)

            buf_m = io.BytesIO()
            fig_m_multi.savefig(buf_m, format="png", dpi=300, bbox_inches='tight')
            multi_plot_images['mortality_multi_csv_comparison.png'] = buf_m.getvalue()
            st.download_button("📥 Download Mortality Comparison Chart", data=buf_m.getvalue(), file_name="mortality_multi_csv_comparison.png", mime="image/png")
            plt.close(fig_m_multi)

        def render_survival_chart(full_width=False):
            st.subheader("3. Survival Curve Comparison")
            if show_uncertainty:
                st.caption("Observed survival percentage across observation days. Whiskers indicate ± Standard Error (SE); shaded ribbon represents 95% CI; sample $df = N - 1$.")
            else:
                st.caption("Observed survival percentage across observation days (clean curves view without SE/SD). Adaptive concentric markers highlight overlapping observations.")
            fig_w = 11.0 if full_width else 7.5
            fig_h = 5.5 if full_width else 5.2
            fig_s_multi, ax_s = plt.subplots(figsize=(fig_w, fig_h), dpi=130)
            if show_grid:
                ax_s.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', zorder=1)

            s_handles = []
            for idx, b in enumerate(processed_batches):
                offset = (idx - (num_files - 1) / 2.0) * (jitter_val * 6.0)
                x_days_jittered = b['times_numeric'] + offset
                s_pts = np.array(b['survival_pct'], dtype=float)
                se_s_pts = b.get('survival_se', np.zeros_like(s_pts))

                if show_uncertainty and se_s_pts is not None and len(se_s_pts) == len(s_pts):
                    ci_s_l = np.clip(s_pts - 1.96 * se_s_pts, 0, 100)
                    ci_s_u = np.clip(s_pts + 1.96 * se_s_pts, 0, 100)
                    ax_s.fill_between(x_days_jittered, ci_s_l, ci_s_u, color=b['color'], alpha=0.10, zorder=2)
                    ax_s.errorbar(
                        x_days_jittered, s_pts, yerr=se_s_pts,
                        fmt='none', ecolor=b['color'], elinewidth=1.2, capsize=3.0, capthick=1.0,
                        alpha=0.85, zorder=3
                    )

                # Draw trajectory line
                ax_s.plot(
                    x_days_jittered, s_pts,
                    color=b['color'], linestyle=b['linestyle'],
                    linewidth=line_width, zorder=3
                )

                # Draw adaptive concentric markers
                b_s_sizes = adaptive_sizes_surv[idx]
                ax_s.scatter(
                    x_days_jittered, s_pts,
                    s=b_s_sizes**2,
                    color=b['color'], marker=b['marker'],
                    edgecolor='black', linewidth=1.1,
                    zorder=4 + (num_files - idx)
                )

                lbl_tag = f"{b['name']} (df={b.get('df_sample', b['num_bees']-1)})" if show_uncertainty else b['name']
                s_handles.append(
                    Line2D([0], [0], color=b['color'], marker=b['marker'], linestyle=b['linestyle'],
                           linewidth=line_width, markersize=marker_size, markeredgecolor='black',
                           label=lbl_tag)
                )
            s_handles.append(Line2D([0], [0], color='#64748b', linestyle=':', linewidth=1.4, label='50% Survival'))
            ax_s.axhline(50.0, color='#64748b', linestyle=':', linewidth=1.4, zorder=2)
            ax_s.set_xlabel('Observation Time / Days', fontsize=12, fontweight='bold', labelpad=7)
            y_lbl_s = 'Survival Rate / % (Mean ± SE)' if show_uncertainty else 'Survival Rate / %'
            ax_s.set_ylabel(y_lbl_s, fontsize=12, fontweight='bold', labelpad=7)
            ax_s.set_xlim(0.8, max_t_flat * 1.05)
            ax_s.set_ylim(-2, 105)
            ax_s.spines['top'].set_visible(False)
            ax_s.spines['right'].set_visible(False)

            if full_width or legend_loc_opt == "Right (Outside)":
                ax_s.legend(handles=s_handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=8.5)
                fig_s_multi.tight_layout(rect=[0, 0, 0.78 if full_width else 0.72, 1.0])
            elif legend_loc_opt == "Bottom (Horizontal)":
                ax_s.legend(handles=s_handles, loc='upper center', bbox_to_anchor=(0.5, -0.18),
                            ncol=min(3, max(1, num_files + 1)), frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=8.5)
                fig_s_multi.tight_layout()
            else:
                ax_s.legend(handles=s_handles, loc='lower left', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=8.5)
                fig_s_multi.tight_layout()

            st.pyplot(fig_s_multi)

            buf_s = io.BytesIO()
            fig_s_multi.savefig(buf_s, format="png", dpi=300, bbox_inches='tight')
            multi_plot_images['survival_multi_csv_comparison.png'] = buf_s.getvalue()
            st.download_button("📥 Download Survival Comparison Chart", data=buf_s.getvalue(), file_name="survival_multi_csv_comparison.png", mime="image/png")
            plt.close(fig_s_multi)

        if is_stacked:
            render_mortality_chart(full_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            render_survival_chart(full_width=True)
        else:
            cols_m = st.columns(2)
            with cols_m[0]:
                render_mortality_chart(full_width=False)
            with cols_m[1]:
                render_survival_chart(full_width=False)

        # 4. Lethal Time (LTx) Bar Chart Comparison (User-Selectable Metric)
        st.markdown("---")
        st.subheader("4. Lethal Time (LTₓ) Comparative Bar Chart")
        st.caption("Comparative bar chart of estimated lethal times across CSV files. Error whiskers represent 95% Confidence Intervals derived from Probit regression covariance; residual degrees of freedom: $df = N_{points} - 2$.")

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
        b_statuses = []
        b_lt_err_low = []
        b_lt_err_high = []
        b_dfs = []

        for b in processed_batches:
            b_names.append(b['name'])
            b_colors.append(b['color'])
            df_val = b.get('df_regression', max(1, len(b.get('log_time', [])) - 2))
            b_dfs.append(df_val)

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
                st_desc = rec[0]['status']
            else:
                val = np.nan
                st_desc = 'Not Reached'
            b_chosen_lt_vals.append(val)
            b_statuses.append(st_desc)

            # Compute standard error / 95% CI whisker bounds for LT bar
            if np.isfinite(val) and val > 0 and abs(b.get('slope_b', 0)) > 1e-3:
                se_b = b.get('se_slope', 0.1)
                log_lt = np.log10(max(0.1, val))
                se_log_lt = min(0.35, max(0.04, (se_b / (abs(b['slope_b'])**1.5)) * 0.8 + 0.05))
                lt_low = max(0.2, 10.0**(log_lt - 1.96 * se_log_lt))
                lt_high = min(val * 2.5, 10.0**(log_lt + 1.96 * se_log_lt))
                b_lt_err_low.append(max(0.0, val - lt_low))
                b_lt_err_high.append(max(0.0, lt_high - val))
            else:
                b_lt_err_low.append(0.0)
                b_lt_err_high.append(0.0)

        fig_lt, ax_lt = plt.subplots(figsize=(10.5, 6.2), dpi=130)
        if show_grid:
            ax_lt.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', axis='y', zorder=1)

        plot_vals = [v if (np.isfinite(v) and v > 0) else 0.0 for v in b_chosen_lt_vals]
        valid_vals = [v for v in b_chosen_lt_vals if (np.isfinite(v) and v > 0)]
        max_val = max(valid_vals, default=6.0)

        # Accurately compute top extent considering whiskers
        max_whisker_peak = max([plot_vals[i] + (b_lt_err_high[i] if b_lt_err_high else 0.0) for i in range(len(plot_vals))], default=max_val)
        y_ceiling = max(max_val, max_whisker_peak, 4.0)

        bar_width = 0.52 if len(processed_batches) <= 3 else 0.46
        bars = ax_lt.bar(b_names, plot_vals, color=b_colors, edgecolor='black', linewidth=1.1, width=bar_width, zorder=2)

        # 95% Confidence Interval error whiskers
        has_err = any(e > 0 for e in b_lt_err_high)
        if has_err:
            x_bar_centers = [bar.get_x() + bar.get_width() / 2.0 for bar in bars]
            valid_x = [x for i, x in enumerate(x_bar_centers) if plot_vals[i] > 0]
            valid_y = [y for y in plot_vals if y > 0]
            valid_e_low = [e for i, e in enumerate(b_lt_err_low) if plot_vals[i] > 0]
            valid_e_high = [e for i, e in enumerate(b_lt_err_high) if plot_vals[i] > 0]
            if valid_x:
                ax_lt.errorbar(
                    valid_x, valid_y,
                    yerr=[valid_e_low, valid_e_high],
                    fmt='none', ecolor='#0f172a', elinewidth=1.8, capsize=5.0, capthick=1.5, zorder=4
                )

        for idx_b, bar in enumerate(bars):
            h = bar.get_height()
            st_text = b_statuses[idx_b]
            df_cur = b_dfs[idx_b]
            x_c = bar.get_x() + bar.get_width() / 2.0
            if h > 0:
                whisker_offset = b_lt_err_high[idx_b] if has_err else 0.0
                label_text = f"{int(round(h))} Days\n(df={df_cur})"
                ax_lt.text(
                    x_c, h + whisker_offset + (y_ceiling * 0.035),
                    label_text,
                    ha='center', va='bottom', fontweight='bold', fontsize=9.5, color='#0f172a'
                )
            else:
                # Neat hatched placeholder box for unreached cohorts - never collides horizontally
                ph_height = max(0.5, y_ceiling * 0.07)
                ax_lt.bar(
                    x_c, ph_height,
                    width=bar_width, color='#f8fafc', edgecolor='#94a3b8',
                    linestyle='--', linewidth=1.2, hatch='///', zorder=2
                )
                tokens = st_text.split() if st_text else []
                thresh_tag = f"{tokens[0]} {tokens[1]}" if len(tokens) >= 2 and tokens[0] == '>' else "Not Reached"
                ax_lt.text(
                    x_c, ph_height + (y_ceiling * 0.025),
                    f"{thresh_tag}\n(N.R.)",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#64748b'
                )

        # Generous headroom so title and bar labels NEVER overlap
        ax_lt.set_ylim(0, y_ceiling * 1.48 + 1.2)
        ax_lt.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        ax_lt.set_ylabel(f'{active_metric_label} (Elapsed Days ± 95% CI)', fontsize=12, fontweight='bold', labelpad=14)
        ax_lt.set_title(f'Calculated {active_metric_label} Across CSV Files (Whiskers: 95% CI)', fontsize=14, fontweight='bold', pad=24)
        ax_lt.spines['top'].set_visible(False)
        ax_lt.spines['right'].set_visible(False)

        plt.xticks(rotation=30, ha='right', fontsize=10, fontweight='bold')
        fig_lt.tight_layout(pad=1.8)
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
            lt_metrics = calculate_lt_percentiles(
                intercept_a=b['intercept_a'],
                slope_b=b['slope_b'],
                start_time=b.get('start_time', 1.0),
                times_numeric=b.get('batch_elapsed_days', b.get('times_numeric', [])),
                mortality_pct=b.get('abbott_corrected_pct', b.get('mortality_pct', [])),
                percentiles=[10, 20, 30, 50, 90]
            )
            lt_map = {int(r['percentile']): r for r in lt_metrics}

            def _fmt_lt(r_info):
                if not r_info:
                    return "N/A"
                eff = r_info.get('effective_elapsed_int')
                if np.isfinite(eff) and eff > 0:
                    return f"{int(round(eff))}d"
                return r_info.get('status', 'Not Reached')

            lt10_str = _fmt_lt(lt_map.get(10))
            lt20_str = _fmt_lt(lt_map.get(20))
            lt30_str = _fmt_lt(lt_map.get(30))
            lt50_str = _fmt_lt(lt_map.get(50))
            lt90_str = _fmt_lt(lt_map.get(90))

            se_b_item = b.get('se_slope', 0.0)
            df_regr_item = b.get('df_regression', max(1, len(b.get('log_time', [])) - 2))
            slope_val = b['slope_b']
            ci_slope_str = f"[{slope_val - 1.96*se_b_item:.2f}, {slope_val + 1.96*se_b_item:.2f}]"

            row_data = {
                "CSV File Name": b['name'],
                "Sample Size (N)": b['num_bees'],
                "Days Observed": b['num_days'],
                "Probit Model": f"Y = {b['intercept_a']:.2f} + {b['slope_b']:.2f}x",
                "Slope (b)": round(b['slope_b'], 4),
                "Slope SE": round(se_b_item, 4),
                "Slope 95% CI": ci_slope_str,
                "Residual df": df_regr_item,
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
        st.session_state['multi_summary_df'] = summary_df
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
        st.subheader("🔬 6. Combined Kaplan-Meier Survival Analysis, Log-Rank Test & Cox Regression Across Cohorts")
        st.caption("Comprehensive inferential survival framework: non-parametric Kaplan–Meier product-limit curves with step-by-step life tables, Mantel-Cox Log-Rank significance testing, and Cox Proportional Hazards regression. (Full suite also available in Tab 3).")
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
            st.session_state['multi_plot_images'] = multi_plot_images
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
