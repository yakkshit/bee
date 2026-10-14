"""
Forest Plot Visualization for Cox Proportional Hazards Models.
Renders clinical-grade 95% Confidence Interval whiskers, square markers,
semi-transparent ribbons, and defensive bound guarantees.
"""
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def render_hazard_ratio_forest_plot(table_rows, group_col='cohort', key_suffix: str = "", res_df=None):
    """
    Renders an enterprise clinical Forest Plot of Hazard Ratios with 95% Confidence Intervals.
    Includes bold whiskers, square markers, top-centered legend, and PNG/CSV downloads.
    """
    if not table_rows:
        return

    st.markdown("#### 🌲 Hazard Ratio Forest Plot (95% CI Uncertainty)")
    fig_fp, ax_fp = plt.subplots(figsize=(8.8, max(3.6, len(table_rows) * 0.75 + 1.2)), dpi=300)
    ax_fp.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', axis='x', zorder=1)

    y_pos = np.arange(len(table_rows))[::-1]
    labels = [r.get("Covariate", "") for r in table_rows]

    # Robust p-value parsing
    p_vals = []
    for r in table_rows:
        p_raw = str(r.get("p-value", "1.0")).split()[0]
        try:
            p_vals.append(float(p_raw))
        except Exception:
            p_vals.append(1.0)

    # Clean and clip values to prevent any NaN / Inf / float overflow
    hr_list = []
    ci_l_list = []
    ci_u_list = []
    xerr_low = []
    xerr_high = []

    for r in table_rows:
        hr_clean = float(np.clip(r.get("Hazard Ratio (HR)", 1.0), 0.01, 50.0))
        ci_l_clean = float(np.clip(r.get("95% CI Lower", 0.01), 0.005, hr_clean))
        ci_u_clean = float(np.clip(r.get("95% CI Upper", hr_clean), hr_clean, 60.0))

        err_l = max(0.001, hr_clean - ci_l_clean)
        err_u = max(0.001, ci_u_clean - hr_clean)

        hr_list.append(hr_clean)
        ci_l_list.append(ci_l_clean)
        ci_u_list.append(ci_u_clean)
        xerr_low.append(err_l)
        xerr_high.append(err_u)

    max_ci = max(ci_u_list) if ci_u_list else 2.0
    min_ci = min(ci_l_list) if ci_l_list else 0.5

    pt_colors = []
    for i, r in enumerate(table_rows):
        if r.get("is_reference", False):
            pt_colors.append('#2563eb')
        elif p_vals[i] < 0.05:
            pt_colors.append('#dc2626' if hr_list[i] > 1.0 else '#16a34a')
        else:
            pt_colors.append('#64748b')

    for i in range(len(y_pos)):
        r = table_rows[i]
        if r.get("is_reference", False):
            # Baseline reference row plotted with diamond marker at null line
            ax_fp.plot([1.0], [y_pos[i]], marker='D', color='#2563eb', markersize=9.5, markeredgecolor='black', zorder=5)
            annot = "HR = 1.00 (Reference Baseline)"
            ax_fp.text(
                1.0 + (0.06 * (max_ci - min_ci + 1.0)), y_pos[i], annot,
                va='center', ha='left', fontsize=9.5, fontweight='bold', color='#1e3a8a',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#eff6ff', edgecolor='#93c5fd', alpha=0.95, lw=1.2)
            )
        else:
            # 1. Bold shaded CI ribbon
            ax_fp.plot([ci_l_list[i], ci_u_list[i]], [y_pos[i], y_pos[i]], color=pt_colors[i], linewidth=5.5, alpha=0.35, zorder=2)

            # 2. Prominent Errorbar with square marker and heavy end caps
            ax_fp.errorbar(
                hr_list[i], y_pos[i],
                xerr=[[xerr_low[i]], [xerr_high[i]]],
                fmt='s', color=pt_colors[i], ecolor=pt_colors[i],
                elinewidth=2.8, capsize=8.0, capthick=2.5, markersize=9.0, zorder=4
            )

            # 3. Formatted Badge with clear background box
            annot = f"HR = {hr_list[i]:.2f}  [95% CI: {ci_l_list[i]:.2f} – {ci_u_list[i]:.2f}]  (p = {p_vals[i]:.3g})"
            ax_fp.text(
                ci_u_list[i] + (0.06 * (max_ci - min_ci + 1.0)), y_pos[i], annot,
                va='center', ha='left', fontsize=9.5, fontweight='bold', color='#0f172a',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8fafc', edgecolor='#cbd5e1', alpha=0.95, lw=1.2)
            )

    # Reference neutral zone & null line
    ax_fp.axvline(1.0, color='#dc2626', linestyle='--', linewidth=1.6, zorder=3)
    ax_fp.axvspan(0.9, 1.1, color='#f1f5f9', alpha=0.75, zorder=1)
    ax_fp.set_yticks(y_pos)
    ax_fp.set_yticklabels(labels, fontsize=11, fontweight='bold')
    ax_fp.set_xlabel('Hazard Ratio (HR) with 95% Confidence Intervals', fontsize=12, fontweight='bold', labelpad=12)
    ax_fp.spines['top'].set_visible(False)
    ax_fp.spines['right'].set_visible(False)

    # Explicit Clinical Forest Plot Legend positioned cleanly above the plot
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#2563eb', markeredgecolor='black', markersize=9, label='Hazard Ratio (HR) Point Estimate'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#2563eb', markeredgecolor='black', markersize=9, label='Reference Baseline (HR = 1.0)'),
        Line2D([0], [0], color='#2563eb', lw=3, label='95% Confidence Interval (CI) Whisker'),
        Line2D([0], [0], color='#dc2626', linestyle='--', lw=1.6, label='Null Line (HR = 1.0)'),
        Patch(facecolor='#f1f5f9', edgecolor='#cbd5e1', label='Neutral Zone (HR 0.9 – 1.1)')
    ]

    ax_fp.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True, facecolor='white', framealpha=0.95, fontsize=8.8)

    left_bound = max(0.0, min_ci * 0.7 - 0.2)
    right_bound = max(max_ci * 1.65 + 1.2, 3.5)

    if not np.isfinite(left_bound) or left_bound < 0:
        left_bound = 0.0
    if not np.isfinite(right_bound) or right_bound <= left_bound:
        right_bound = max(left_bound + 2.0, 4.0)

    ax_fp.set_xlim(left=left_bound, right=right_bound)
    ax_fp.set_ylim(bottom=min(y_pos) - 0.9, top=max(y_pos) + 0.8)

    # Directional Guidance at bottom relative to HR = 1.0 null line
    ax_fp.text(0.85, min(y_pos) - 0.7, "◀ Favors Longer Survival (Protective)", fontsize=9.0, fontweight='bold', color='#16a34a', ha='right')
    ax_fp.text(1.15, min(y_pos) - 0.7, "Accelerates Mortality (Higher Risk) ▶", fontsize=9.0, fontweight='bold', color='#dc2626', ha='left')

    fig_fp.tight_layout()
    st.pyplot(fig_fp)

    buf_fp = io.BytesIO()
    fig_fp.savefig(buf_fp, format="png", dpi=300, bbox_inches='tight')

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        fp_k = f"dl_cox_forest_{group_col}_{key_suffix}" if key_suffix else f"dl_cox_forest_{group_col}"
        st.download_button("📥 Download Forest Plot (PNG)", data=buf_fp.getvalue(), file_name="cox_forest_plot.png", mime="image/png", key=fp_k)
    with col_dl2:
        if res_df is not None:
            sum_k = f"dl_cox_summary_{group_col}_{key_suffix}" if key_suffix else f"dl_cox_summary_{group_col}"
            st.download_button("📊 Download Cox Regression Summary (CSV)", data=res_df.to_csv(index=False), file_name="cox_regression_summary.csv", mime="text/csv", key=sum_k)
    plt.close(fig_fp)
