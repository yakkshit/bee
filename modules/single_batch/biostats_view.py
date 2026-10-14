"""
Tab 5: Biostatistical Models.
Orchestrates Finney Probit linear transformation, Log-Rank survival distributions tests,
and multivariable Cox Proportional Hazards regression with integrated educational math cards.
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from modules.parser import (
    extract_individual_bee_survival,
    get_cohort_code
)
from modules.survival.log_rank import render_log_rank_analysis
from modules.survival.cox_regression import render_cox_regression_analysis
from modules.survival.math_cards import render_logrank_math_card, render_cox_math_card


def render_single_biostats_view(computed, df, settings=None, image_collector=None):
    """Renders the Biostatistical Models sub-tab."""
    if settings is None:
        settings = {}

    primary_color = settings.get('primary_color', '#ef4444')
    line_width = settings.get('line_width', 2.4)
    marker_size = settings.get('marker_size', 7.0)
    line_style = settings.get('line_style', '-')

    c = computed
    log_time = c['log_time']
    probit_y = c['probit_y']
    probit_a = c['probit_a']
    probit_b = c['probit_b']
    probit_r = c['probit_r']
    elapsed_days = c['elapsed_days']

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
                if image_collector:
                    image_collector(fig_probit, 'fig_probit_regression.png', 'Download Probit Regression', 't5_probit')

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
            st.subheader("Log-Rank Survival Test")
            treatment_vals = [str(x).strip() for x in df['Treatment'].unique() if pd.notna(x) and str(x).strip() != ''] if 'Treatment' in df.columns else []
            single_bees_df = extract_individual_bee_survival(df, batch_name="Single Batch", elapsed_days=elapsed_days)

            if len(treatment_vals) >= 2:
                st.caption(f"📊 Multi-treatment sheet detected: comparing `{', '.join(treatment_vals)}`.")
                render_log_rank_analysis(single_bees_df, group_col='treatment', key_suffix="single_tr")
            else:
                t_curr = treatment_vals[0] if treatment_vals else "Single Batch"
                single_bees_df['cohort'] = t_curr
                single_bees_df['cohort_code'] = get_cohort_code(t_curr)
                color_map = {t_curr: primary_color}
                render_log_rank_analysis(single_bees_df, group_col='cohort', color_map=color_map, key_suffix="single_lr")

        with col_lrs_side:
            render_logrank_math_card()

    with subtab_cox_s:
        col_cxs_m, col_cxs_side = st.columns([2.2, 1.0])
        with col_cxs_m:
            st.subheader("Cox Proportional Hazards Regression")
            treatment_vals = [str(x).strip() for x in df['Treatment'].unique() if pd.notna(x) and str(x).strip() != ''] if 'Treatment' in df.columns else []
            single_bees_df = extract_individual_bee_survival(df, batch_name="Single Batch", elapsed_days=elapsed_days)

            if len(treatment_vals) >= 2:
                st.caption(f"📊 Multi-treatment sheet detected: quantifying hazard ratios across `{', '.join(treatment_vals)}`.")
                render_cox_regression_analysis(single_bees_df, group_col='treatment', key_suffix="single_cox_multi_tr")
            else:
                t_curr = treatment_vals[0] if treatment_vals else "Single Batch"
                single_bees_df['cohort'] = t_curr
                single_bees_df['cohort_code'] = get_cohort_code(t_curr)
                render_cox_regression_analysis(single_bees_df, group_col='cohort', key_suffix="single_cox")

        with col_cxs_side:
            render_cox_math_card()
