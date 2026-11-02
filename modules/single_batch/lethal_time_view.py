"""
Tab 2: Lethal Time (LT_X) Laboratory & Sensitivity Analysis.
Computes and visualizes arbitrary lethal percentiles (LT10, LT20, LT30, LT50, LT90)
using Finney Probit theoretical models and empirical bioassay observation times.
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator

from modules.parser import calculate_lt_percentiles


def render_single_lethal_time_view(computed, settings=None, image_collector=None):
    """Renders the Lethal Time Spectrum analysis sub-tab."""
    if settings is None:
        settings = {}

    primary_color = settings.get('primary_color', '#ef4444')
    line_width = settings.get('line_width', 2.4)
    marker_size = settings.get('marker_size', 7.0)

    c = computed
    elapsed_days = c['elapsed_days']
    day_labels = c['day_labels']
    max_time = c['max_time']
    mortality_pct = c['mortality_pct']
    abbott_corrected_pct = c['abbott_corrected_pct']
    t_dense = c['t_dense']
    fit_mort_curve = c['fit_mort_curve']
    probit_a = c['probit_a']
    probit_b = c['probit_b']

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
        if image_collector:
            image_collector(fig_lt_spec, 'fig_lt_threshold_spectrum.png', 'Download LT Threshold Spectrum', 't2_spec')

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
        if image_collector:
            image_collector(fig_lt_bar, 'fig_lt_horizon_comparison.png', 'Download LT Horizon Chart', 't2_bar')

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
