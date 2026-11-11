"""
Tab 1: Single Batch Mortality & Survival Kinetic Curves.
Renders Logistic mortality fit, exponential decay survival fit,
Kaplan-Meier product-limit curves with Greenwood CI, and Abbott-corrected curves.
"""
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def render_single_survival_curves(computed, settings=None, image_collector=None):
    """Renders the Mortality & Survival kinetic curves sub-tab."""
    if settings is None:
        settings = {}

    primary_color = settings.get('primary_color', '#ef4444')
    secondary_color = settings.get('secondary_color', '#f59e0b')
    line_width = settings.get('line_width', 2.4)
    marker_size = settings.get('marker_size', 7.0)
    line_style = settings.get('line_style', '-')

    c = computed
    num_days = c['num_days']
    elapsed_days = c['elapsed_days']
    day_labels = c['day_labels']
    max_time = c['max_time']
    mortality_pct = c['mortality_pct']
    mortality_se = c['mortality_se']
    mortality_sd = c.get('mortality_sd', mortality_se * np.sqrt(max(1, c.get('num_bees', 10))))
    survival_pct = c['survival_pct']
    survival_se = c['survival_se']
    survival_sd = c.get('survival_sd', survival_se * np.sqrt(max(1, c.get('num_bees', 10))))
    abbott_corrected_pct = c['abbott_corrected_pct']
    df_logistic = c.get('df_logistic', max(1, num_days - 3))
    df_decay = c.get('df_decay', max(1, num_days - 3))
    df_sample = c.get('df_sample', max(1, c.get('num_bees', 10) - 1))
    t_dense = c['t_dense']
    fit_mort_curve = c['fit_mort_curve']
    fit_surv_curve = c['fit_surv_curve']
    residuals_mort = c['residuals_mort']
    residuals_surv = c['residuals_surv']
    L_fit = c['L_fit']
    k_fit = c['k_fit']
    t0_fit = c['t0_fit']
    r2_mort = c['r2_mort']
    S0_surv = c['S0_surv']
    a_surv = c['a_surv']
    b_surv = c['b_surv']
    r2_surv = c['r2_surv']
    kmf = c['kmf']
    durations = c['durations']
    events = c['events']

    st.markdown("#### Cohort Survival Dynamics & Kinetic Curves")
    st.caption("Mathematical characterization of cohort mortality progression, survival decay kinetics, and Kaplan-Meier product-limit estimators.")

    c1, c2 = st.columns(2)

    if num_days > 0:
        with c1:
            st.subheader("Logistic Mortality Curve")
            fig1, ax = plt.subplots(figsize=(7.0, 5.2), dpi=130)
            fit_se = np.std(residuals_mort)
            mean_mort_sd = float(np.mean(mortality_sd)) if len(mortality_sd) > 0 else 0.0
            ax.fill_between(t_dense, np.clip(fit_mort_curve - 1.96 * fit_se, 0, 100), np.clip(fit_mort_curve + 1.96 * fit_se, 0, 100), color=primary_color, alpha=0.12, label='95% Confidence Interval')
            ax.plot(t_dense, fit_mort_curve, color=primary_color, linewidth=line_width, linestyle=line_style, label='Logistic Model Fit')
            ax.errorbar(elapsed_days, mortality_pct, yerr=mortality_se, fmt='o', color=primary_color, ecolor=primary_color, elinewidth=1.6, capsize=4, capthick=1.6, markersize=marker_size, markerfacecolor=primary_color, markeredgecolor='black', label='Observed (Mean ± SE)')
            ax.axhline(50, color='#333333', linestyle=':', linewidth=1.2)
            eq_text = f"Y = {L_fit:.1f} / (1 + exp(-{k_fit:.2f}(X - {t0_fit:.1f})))\nR² = {r2_mort:.4f} | df = {df_logistic}\nSE(fit) = {fit_se:.2f}% | Mean SD = {mean_mort_sd:.1f}%\n95% CI Band (±1.96 SE)"
            ax.text(0.06, 0.92, eq_text, transform=ax.transAxes, fontsize=9.5, verticalalignment='top', bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#e0e0e0', alpha=0.90))
            ax.set_xlabel('Observation Time / Days', fontsize=12, fontweight='bold', labelpad=7)
            ax.set_ylabel('Mortality / %', fontsize=12, fontweight='bold', labelpad=7)
            ax.set_xlim(-0.1, max_time * 1.05)
            ax.set_ylim(-2, 102)
            ax.set_xticks(elapsed_days)
            ax.set_xticklabels(day_labels, rotation=45, ha="right")
            ax.yaxis.set_major_locator(MultipleLocator(25))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#e0e0e0', fontsize=8.5)
            fig1.tight_layout()
            st.pyplot(fig1)
            if image_collector:
                image_collector(fig1, 'fig_mortality_curve.png', 'Download Mortality Curve', 't1_mort')

        with c2:
            st.subheader("Survival Decay Curve")
            fig2, ax = plt.subplots(figsize=(7.0, 5.2), dpi=130)
            surv_se_fit = np.std(residuals_surv)
            mean_surv_sd = float(np.mean(survival_sd)) if len(survival_sd) > 0 else 0.0
            ax.fill_between(t_dense, np.clip(fit_surv_curve - 1.96 * surv_se_fit, 0, 100), np.clip(fit_surv_curve + 1.96 * surv_se_fit, 0, 100), color=secondary_color, alpha=0.15, label='95% Confidence Interval')
            ax.plot(t_dense, fit_surv_curve, color=secondary_color, linewidth=line_width, linestyle=line_style, label='Survival Model Fit')
            ax.errorbar(elapsed_days, survival_pct, yerr=survival_se, fmt='o', color=secondary_color, ecolor=secondary_color, elinewidth=1.6, capsize=4, capthick=1.6, markersize=marker_size, markerfacecolor=secondary_color, markeredgecolor='black', label='Observed (Mean ± SE)')
            ax.axhline(50, color='#333333', linestyle=':', linewidth=1.2)
            eq_surv_text = f"Y = {S0_surv:.1f} * exp(-{a_surv:.4f} * X^{b_surv:.2f})\nR² = {r2_surv:.4f} | df = {df_decay}\nSE(fit) = {surv_se_fit:.2f}% | Mean SD = {mean_surv_sd:.1f}%\n95% CI Band (±1.96 SE)"
            ax.text(0.06, 0.35, eq_surv_text, transform=ax.transAxes, fontsize=9.5, verticalalignment='top', bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#e0e0e0', alpha=0.90))
            ax.set_xlabel('Observation Time / Days', fontsize=12, fontweight='bold', labelpad=7)
            ax.set_ylabel('Bee Survival / %', fontsize=12, fontweight='bold', labelpad=7)
            ax.set_xlim(-0.1, max_time * 1.05)
            ax.set_ylim(-2, 105)
            ax.set_xticks(elapsed_days)
            ax.set_xticklabels(day_labels, rotation=45, ha="right")
            ax.yaxis.set_major_locator(MultipleLocator(25))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#e0e0e0', fontsize=8.5)
            fig2.tight_layout()
            st.pyplot(fig2)
            if image_collector:
                image_collector(fig2, 'fig_survival_curve.png', 'Download Survival Curve', 't1_surv')

    if len(durations) > 0:
        with c1:
            st.subheader("Kaplan-Meier Survival Analysis")
            show_km_ci = st.checkbox("Show 95% Greenwood Confidence Interval Ribbon", value=True, key="single_km_show_ci")
            fig3, ax = plt.subplots(figsize=(7.0, 5.2), dpi=130)
            km_timeline = kmf.survival_function_.index
            km_vals = kmf.survival_function_['KM_estimate'].values
            ci_df = kmf.confidence_interval_survival_function_
            ax.step(km_timeline, km_vals, where='post', color='#0284c7', linewidth=line_width, linestyle=line_style, label='Kaplan-Meier Estimate')
            if show_km_ci:
                ax.fill_between(km_timeline, ci_df.iloc[:, 0], ci_df.iloc[:, 1], step='post', color='#0284c7', alpha=0.15, label='95% Greenwood Confidence Interval')
            censored_days = durations[events == 0]
            censored_surv = [kmf.survival_function_at_times(d).values[0] for d in censored_days] if len(censored_days) > 0 else []
            if len(censored_days) > 0:
                ax.plot(censored_days, censored_surv, '+', color='#0f172a', markersize=9, markeredgewidth=2, label=f'Censored (n={len(censored_days)})')
            ax.axhline(0.5, color='#333333', linestyle=':', linewidth=1.2)
            km_df_stat = max(1, len(durations) - 1)
            # Position info card cleanly in top right corner (ha='right', va='top') so it never collides with legend in lower left
            card_txt = f"N = {len(durations)} | df = {km_df_stat}\nSE: Greenwood formula\n95% CI: log-log bounds" if show_km_ci else f"N = {len(durations)} | df = {km_df_stat}\nKaplan-Meier Estimate"
            ax.text(0.96, 0.95, card_txt,
                    transform=ax.transAxes, ha='right', va='top', fontsize=8.5, fontweight='bold', bbox=dict(facecolor='white', edgecolor='#cbd5e1', alpha=0.90))
            ax.set_xlabel('Observation Time / Days', fontsize=12, fontweight='bold', labelpad=7)
            ax.set_ylabel('Survival Probability', fontsize=12, fontweight='bold', labelpad=7)
            ax.set_xlim(-0.1, max_time * 1.05)
            ax.set_ylim(-0.02, 1.05)
            ax.set_xticks(elapsed_days)
            ax.set_xticklabels(day_labels, rotation=45, ha="right")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=8.5)
            fig3.tight_layout()
            st.pyplot(fig3)
            if image_collector:
                image_collector(fig3, 'fig_kaplan_meier.png', 'Download Kaplan-Meier Curve', 't1_km')

        with c2:
            st.subheader("Abbott Corrected Mortality Curve")
            show_abbott_err = st.checkbox("Show SE Whiskers & 95% CI Band", value=True, key="single_mort_show_err")
            fig_abbott, ax_abbott = plt.subplots(figsize=(7.0, 5.2), dpi=130)
            if show_abbott_err:
                ci_abbott_low = np.clip(abbott_corrected_pct - 1.96 * mortality_se, 0, 100)
                ci_abbott_high = np.clip(abbott_corrected_pct + 1.96 * mortality_se, 0, 100)
                ax_abbott.fill_between(elapsed_days, ci_abbott_low, ci_abbott_high, color=primary_color, alpha=0.12, label='95% Confidence Interval')
                ax_abbott.errorbar(elapsed_days, abbott_corrected_pct, yerr=mortality_se, fmt='o', color=primary_color,
                                   ecolor=primary_color, elinewidth=1.6, capsize=4, capthick=1.6, markersize=marker_size,
                                   markerfacecolor=primary_color, markeredgecolor='black', label='Abbott Corrected (Mean ± SE)')
            else:
                ax_abbott.plot(elapsed_days, abbott_corrected_pct, 'o', color=primary_color, markersize=marker_size,
                               markerfacecolor=primary_color, markeredgecolor='black', label='Abbott Corrected')
            ax_abbott.plot(elapsed_days, abbott_corrected_pct, color=primary_color, linewidth=line_width, linestyle=line_style)
            ax_abbott.axhline(50, color='#333333', linestyle=':', linewidth=1.2)
            lbl_info = f"Baseline Control C = {settings.get('abbott_c', 0.0):.1f}%\nSE via Binomial Proportion\n95% CI = Mean ± 1.96 SE" if show_abbott_err else f"Baseline Control C = {settings.get('abbott_c', 0.0):.1f}%\nClean Trajectory"
            ax_abbott.text(0.06, 0.90, lbl_info,
                           transform=ax_abbott.transAxes, fontsize=8.5, verticalalignment='top', bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#e0e0e0', alpha=0.90))
            ax_abbott.set_xlabel('Observation Time / Days', fontsize=12, fontweight='bold', labelpad=7)
            y_lbl_sing = 'Corrected Mortality / % (Mean ± SE)' if show_abbott_err else 'Corrected Mortality / %'
            ax_abbott.set_ylabel(y_lbl_sing, fontsize=12, fontweight='bold', labelpad=7)
            ax_abbott.set_xlim(-0.1, max_time * 1.05)
            ax_abbott.set_ylim(-2, 102)
            ax_abbott.set_xticks(elapsed_days)
            ax_abbott.set_xticklabels(day_labels, rotation=45, ha="right")
            ax_abbott.spines['top'].set_visible(False)
            ax_abbott.spines['right'].set_visible(False)
            ax_abbott.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#e0e0e0', fontsize=8.5)
            fig_abbott.tight_layout()
            st.pyplot(fig_abbott)
            if image_collector:
                image_collector(fig_abbott, 'fig_abbott_corrected.png', 'Download Abbott Corrected Curve', 't1_abbott')
