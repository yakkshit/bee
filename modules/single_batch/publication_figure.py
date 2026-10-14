"""
Tab 6: Publication-Grade Composite Summary Figure.
Prepares 4-panel composite figures (Panels A–D) conforming to high-impact
journal standards (Nature, Cell, ES&T, Journal of Insect Physiology) at 300 DPI.
"""
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def render_single_publication_figure(computed, settings=None, image_collector=None):
    """Renders the Publication Multi-Panel Figure sub-tab."""
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
    survival_pct = c['survival_pct']
    survival_se = c['survival_se']
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
    temps = c['temps']
    humidities = c['humidities']

    st.markdown("#### Publication-Grade Composite Summary Figure")
    st.caption("Composite 4-panel figure (Panels A–D) prepared to high-impact journal standards (Cell, Nature, ES&T, Journal of Insect Physiology) at 300 DPI.")

    if num_days > 0:
        fig6 = plt.figure(figsize=(14, 11), dpi=300)
        gs = fig6.add_gridspec(2, 2, hspace=0.30, wspace=0.25)

        fit_se = np.std(residuals_mort)
        surv_se_fit = np.std(residuals_surv)

        # Panel A: Mortality
        ax_a = fig6.add_subplot(gs[0, 0])
        ax_a.fill_between(t_dense, np.clip(fit_mort_curve - 1.96 * fit_se, 0, 100), np.clip(fit_mort_curve + 1.96 * fit_se, 0, 100), color=primary_color, alpha=0.12)
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
        ax_b.fill_between(t_dense, np.clip(fit_surv_curve - 1.96 * surv_se_fit, 0, 100), np.clip(fit_surv_curve + 1.96 * surv_se_fit, 0, 100), color=secondary_color, alpha=0.15)
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
            km_timeline = kmf.survival_function_.index
            km_vals = kmf.survival_function_['KM_estimate'].values
            ci_df = kmf.confidence_interval_survival_function_
            ax_c.step(km_timeline, km_vals, where='post', color='#0284c7', linewidth=line_width, linestyle=line_style, label='KM Estimate')
            ax_c.fill_between(km_timeline, ci_df.iloc[:, 0], ci_df.iloc[:, 1], step='post', color='#0284c7', alpha=0.15, label='95% CI')
            censored_days = durations[events == 0]
            if len(censored_days) > 0:
                censored_surv = [kmf.survival_function_at_times(d).values[0] for d in censored_days]
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
            ax_d1.set_ylim(min(temps) - 1, max(temps) + 1)

            ax_d2 = ax_d1.twinx()
            ax_d2.set_ylabel('Relative Humidity / %', color='#0284c7', fontsize=11.5, fontweight='bold')
            l2 = ax_d2.plot(elapsed_days, humidities, color='#0284c7', marker='s', linewidth=2.2, markersize=6.5, linestyle='--', label='Humidity (%)')
            ax_d2.tick_params(axis='y', labelcolor='#0284c7')
            if any(humidities):
                ax_d2.set_ylim(min(humidities) - 5, max(humidities) + 5)

            ax_d1.set_xlim(-0.1, max_time * 1.05)
            ax_d1.set_xticks(elapsed_days)
            ax_d1.set_xticklabels(day_labels, rotation=45, ha="right")
            ax_d1.spines['top'].set_visible(False)
            ax_d2.spines['top'].set_visible(False)
            lines_comb = l1 + l2
            ax_d1.legend(lines_comb, [l.get_label() for l in lines_comb], loc='upper left', frameon=True, framealpha=0.9, facecolor='white', edgecolor='none', fontsize=9.5)
            ax_d1.text(-0.14, 1.04, 'D', transform=ax_d1.transAxes, fontsize=15, fontweight='bold')

        st.pyplot(fig6)
        if image_collector:
            image_collector(fig6, 'fig_combined_publication_summary.png', 'Download Combined Multi-Panel Figure', 't6_pub')
