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
    get_cohort_code,
    extract_key_from_df
)
from modules.survival.kaplan_meier import render_kaplan_meier_analysis
from modules.survival.log_rank import render_log_rank_analysis
from modules.survival.cox_regression import render_cox_regression_analysis
from modules.survival.math_cards import (
    render_logrank_math_card,
    render_cox_math_card,
    render_kaplan_meier_math_card,
    render_methodology_comparison_card
)




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

    subtab_prob, subtab_km_s, subtab_lr_s, subtab_cox_s = st.tabs([
        "📈 Finney Probit Regression",
        "📉 Kaplan–Meier Survival Curves Plot",
        "📊 Log-Rank (Mantel-Cox) Test",
        "🌲 Cox Proportional Hazards Model"
    ])

    with subtab_prob:
        c_s1, c_s2 = st.columns(2)
        if len(log_time) > 1:
            df_probit = c.get('df_probit', max(1, len(log_time) - 2))
            probit_se_b = c.get('probit_se_b', 0.0)
            probit_se_a = c.get('probit_se_a', 0.0)
            probit_p = c.get('probit_p', 1.0)

            x_mean = np.mean(log_time)
            ss_x = np.sum((log_time - x_mean)**2)
            residuals_p = probit_y - (probit_a + probit_b * log_time)
            se_est = np.std(residuals_p)

            with c_s1:
                st.subheader("Probit Regression (Time-Mortality)")
                fig_probit, ax_probit = plt.subplots(figsize=(7.0, 5.2), dpi=140)
                ax_probit.scatter(log_time, probit_y, color=primary_color, s=marker_size**2 * 1.5, edgecolors='black', label='Observed Probit Points', zorder=4)
                x_line = np.linspace(min(log_time), max(log_time), 100)
                y_line = probit_a + probit_b * x_line

                se_line = se_est * np.sqrt(1.0 / len(log_time) + ((x_line - x_mean)**2) / max(1e-6, ss_x))
                t_crit = 1.96
                ci_low = y_line - t_crit * se_line
                ci_high = y_line + t_crit * se_line

                ax_probit.fill_between(x_line, ci_low, ci_high, color=primary_color, alpha=0.15, label='95% Confidence Interval')
                ax_probit.plot(x_line, y_line, color='#1e293b', linewidth=line_width, linestyle=line_style, label='Probit Model Fit Line', zorder=3)

                eq_text = (
                    f"Probit(P) = {probit_a:.2f} + {probit_b:.2f}x\n"
                    f"R² = {probit_r**2:.4f} | df = {df_probit}\n"
                    f"SE(slope) = {probit_se_b:.4f} | SE(est) = {se_est:.3f}\n"
                    f"95% CI Band (±1.96 SE)"
                )
                ax_probit.text(0.05, 0.95, eq_text, transform=ax_probit.transAxes, fontsize=9.5, verticalalignment='top', bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#e0e0e0', alpha=0.90))
                ax_probit.set_xlabel('Log10(Observation Time / Days)', fontsize=12, fontweight='bold', labelpad=7)
                ax_probit.set_ylabel('Probit(P)', fontsize=12, fontweight='bold', labelpad=7)
                ax_probit.spines['top'].set_visible(False)
                ax_probit.spines['right'].set_visible(False)
                ax_probit.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#e0e0e0', fontsize=8.5)
                fig_probit.tight_layout()
                st.pyplot(fig_probit)
                if image_collector:
                    image_collector(fig_probit, 'fig_probit_regression.png', 'Download Probit Regression', 't5_probit')

            with c_s2:
                st.subheader("Probit Model Parameters & Biostatistical Uncertainty")
                p_r2 = probit_r**2
                b_ci_l = probit_b - 1.96 * probit_se_b
                b_ci_u = probit_b + 1.96 * probit_se_b
                st.markdown(f"""
                <div class="glass-card">
                    <p><b>Model Formula:</b> <code>Probit(P) = a + b · log₁₀(t)</code></p>
                    <p><b>Degrees of Freedom (df):</b> <code>{df_probit}</code> (Residual df = N - 2)</p>
                    <p><b>Slope (b):</b> <code>{probit_b:.4f}</code> &nbsp;|&nbsp; <b>SE(b):</b> <code>±{probit_se_b:.4f}</code></p>
                    <p><b>Slope 95% Confidence Interval:</b> <code>[{b_ci_l:.4f} – {b_ci_u:.4f}]</code></p>
                    <p><b>Intercept (a):</b> <code>{probit_a:.4f}</code> &nbsp;|&nbsp; <b>SE(a):</b> <code>±{probit_se_a:.4f}</code></p>
                    <p><b>Standard Error of Estimate (SE):</b> <code>{se_est:.4f}</code></p>
                    <p><b>Standard Deviation (SD) of Probit Y:</b> <code>{np.std(probit_y):.4f}</code></p>
                    <p><b>Pearson Correlation (r):</b> <code>{probit_r:.4f}</code> (p = {probit_p:.4g})</p>
                    <p><b>Coefficient of Determination (R²):</b> <code>{p_r2:.4f}</code></p>
                    <p><b>Status:</b> {'✅ High Linear Fit' if p_r2 > 0.8 else ('⚠️ Moderate Linear Fit' if p_r2 > 0.5 else 'ℹ️ Low Linear Correlation')}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Insufficient observation days to fit Probit regression.")

    with subtab_km_s:
        col_kms_m, col_kms_side = st.columns([2.2, 1.0])
        with col_kms_m:
            st.subheader("Kaplan–Meier Survival Curves")
            treatment_vals = [str(x).strip() for x in df['Treatment'].unique() if pd.notna(x) and str(x).strip() != ''] if 'Treatment' in df.columns else []
            single_bees_df = extract_individual_bee_survival(df, batch_name="Single Batch", elapsed_days=elapsed_days)

            if len(treatment_vals) >= 2:
                st.caption(f"📊 Multi-treatment sheet detected: plotting `{', '.join(treatment_vals)}`.")
                render_kaplan_meier_analysis(single_bees_df, group_col='treatment', key_suffix="single_km_tr")
            else:
                key_or_name = extract_key_from_df(df) or st.session_state.get('active_sheet_key', 'Single Batch')
                t_curr = treatment_vals[0] if treatment_vals else key_or_name
                single_bees_df['cohort'] = t_curr
                single_bees_df['cohort_code'] = get_cohort_code(t_curr)
                color_map = {t_curr: primary_color}
                render_kaplan_meier_analysis(single_bees_df, group_col='cohort', color_map=color_map, key_suffix="single_km")

        with col_kms_side:
            card_tab_km_s, card_tab_comp_km_s = st.tabs(["📐 Kaplan–Meier", "📊 Method Comparison"])
            with card_tab_km_s:
                render_kaplan_meier_math_card()
            with card_tab_comp_km_s:
                render_methodology_comparison_card()

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
                key_or_name = extract_key_from_df(df) or st.session_state.get('active_sheet_key', 'Single Batch')
                t_curr = treatment_vals[0] if treatment_vals else key_or_name
                single_bees_df['cohort'] = t_curr
                single_bees_df['cohort_code'] = get_cohort_code(t_curr)
                color_map = {t_curr: primary_color}
                render_log_rank_analysis(single_bees_df, group_col='cohort', color_map=color_map, key_suffix="single_lr")

        with col_lrs_side:
            card_tab_lr_s, card_tab_comp_lr_s = st.tabs(["📐 Log-Rank Math", "📊 Method Comparison"])
            with card_tab_lr_s:
                render_logrank_math_card()
            with card_tab_comp_lr_s:
                render_methodology_comparison_card()

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
                key_or_name = extract_key_from_df(df) or st.session_state.get('active_sheet_key', 'Single Batch')
                t_curr = treatment_vals[0] if treatment_vals else key_or_name
                single_bees_df['cohort'] = t_curr
                single_bees_df['cohort_code'] = get_cohort_code(t_curr)
                render_cox_regression_analysis(single_bees_df, group_col='cohort', key_suffix="single_cox")

        with col_cxs_side:
            card_tab_cox_s, card_tab_mc_s = st.tabs(["🌲 Cox Model", "📊 Method Comparison"])
            with card_tab_cox_s:
                render_cox_math_card()
            with card_tab_mc_s:
                render_methodology_comparison_card()

