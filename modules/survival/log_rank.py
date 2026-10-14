"""
Log-Rank (Mantel-Cox) Survival Analysis Module.
Performs non-parametric hypothesis testing between cohort survival curves,
pairwise post-hoc comparisons with Bonferroni correction, and Kaplan-Meier curves.
"""
import io
from itertools import combinations
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test


def render_log_rank_analysis(survival_df, group_col='cohort', color_map=None, key_suffix: str = ""):
    """
    Renders 1. Log-Rank (Mantel-Cox) Multi-Cohort Survival Test.
    Specifically considers alive bees on Day 1 (post-starvation baseline):
    e.g. if 2 dead bees out of 14 bees on Day 1, N total is 12 bees (not 14 bees).
    Computes chi2, p-value, df, pairwise comparisons with Bonferroni correction,
    and a multi-cohort Kaplan-Meier survival curves plot.
    """
    if survival_df is None or survival_df.empty:
        st.warning("No individual survival data available for Log-Rank test.")
        return

    # Alive Bee Cohort Entry Rule strictly for:
    # '1. Log-Rank (Mantel-Cox) Multi-Cohort Survival Test'
    # On Day 1, if there are 2 dead bees out of 14 bees, the N total is 12 bees (not 14 bees).
    # Bees recorded as Dead on Day 1 (duration <= 1.0 and event == 1) died at baseline post-starvation
    # and are excluded so N total strictly reflects the bees alive at the start of treatment.
    alive_mask = ~((survival_df['duration'] <= 1.0) & (survival_df['event'] == 1))
    valid_survival_df = survival_df[alive_mask].copy() if alive_mask.any() else survival_df.copy()

    groups = [g for g in valid_survival_df[group_col].unique() if pd.notna(g) and str(g).strip() != '']
    if len(groups) == 0:
        st.warning("No valid cohort data available for survival analysis.")
        return

    st.info("🕒 **Alive Bee Cohort Entry Protocol (Active for Log-Rank Model)**: Initial sample size ($N_{\\text{total}}$) strictly includes bees **alive on Day 1** post-starvation baseline. Any bees dead on Day 1 are excluded from the treated cohort at risk, ensuring $N_{\\text{total}}$ represents the true treated population (e.g., $N=12$ alive bees when 2 are dead on Day 1 out of 14).")

    T = valid_survival_df['duration'].values
    E = valid_survival_df['event'].values
    G = valid_survival_df[group_col].values

    total_events = int(np.sum(E))

    if len(groups) >= 2:
        if total_events == 0:
            chisq = 0.0
            p_val = 1.0
            df_val = len(groups) - 1
        else:
            try:
                if len(groups) == 2:
                    m1 = (G == groups[0])
                    m2 = (G == groups[1])
                    res_lr = logrank_test(T[m1], T[m2], event_observed_A=E[m1], event_observed_B=E[m2])
                    chisq = res_lr.test_statistic
                    p_val = res_lr.p_value
                    df_val = 1
                else:
                    res_lr = multivariate_logrank_test(T, G, event_observed=E)
                    chisq = res_lr.test_statistic
                    p_val = res_lr.p_value
                    df_val = getattr(res_lr, 'degrees_of_freedom', len(groups) - 1)
            except Exception as ex:
                st.error(f"Error computing Log-Rank test: {ex}")
                return

        is_significant = p_val < 0.05

        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        with col_k1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{chisq:.3f}</div><div class="kpi-label">Log-Rank χ² Statistic</div></div>', unsafe_allow_html=True)
        with col_k2:
            p_str = "< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}"
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{p_str}</div><div class="kpi-label">p-value (Mantel-Cox)</div></div>', unsafe_allow_html=True)
        with col_k3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{df_val}</div><div class="kpi-label">Degrees of Freedom (df)</div></div>', unsafe_allow_html=True)
        with col_k4:
            status_txt = "Significant (p < 0.05)" if is_significant else "Not Significant"
            status_col = "#10b981" if is_significant else "#f59e0b"
            st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color: {status_col}; font-size: 1.15rem;">{status_txt}</div><div class="kpi-label">Statistical Status</div></div>', unsafe_allow_html=True)
    else:
        # Single cohort metrics
        is_significant = False
        n_tot = len(T)
        n_ev = total_events
        n_cens = n_tot - n_ev
        km_single = KaplanMeierFitter()
        km_single.fit(T, event_observed=E)
        med_s = km_single.median_survival_time_
        med_str = f"{int(round(med_s))}d" if (np.isfinite(med_s) and med_s > 0) else "> Max Obs"

        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        with col_k1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{n_tot}</div><div class="kpi-label">Sample Size (N)</div></div>', unsafe_allow_html=True)
        with col_k2:
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{n_ev}</div><div class="kpi-label">Observed Deaths</div></div>', unsafe_allow_html=True)
        with col_k3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{n_cens}</div><div class="kpi-label">Censored / Alive</div></div>', unsafe_allow_html=True)
        with col_k4:
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{med_str}</div><div class="kpi-label">Median Survival (LT50)</div></div>', unsafe_allow_html=True)

        st.caption(f"💡 **Single Cohort Survival Profile (`{groups[0]}`)**: Displaying Kaplan-Meier survival curve with Greenwood 95% Confidence Interval. Multi-cohort Log-Rank χ² hypothesis testing evaluates differences between 2 or more cohorts; upload additional CSV files to compare between cohorts.")

    st.markdown("<br>", unsafe_allow_html=True)

    fig_km, ax_km = plt.subplots(figsize=(9.2, 5.5), dpi=300)
    ax_km.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', zorder=1)

    km_fitter = KaplanMeierFitter()
    cohort_summary = []
    default_palette = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4"]

    for idx_g, g in enumerate(groups):
        m = (G == g)
        sub_t = T[m]
        sub_e = E[m]
        col = color_map.get(g, default_palette[idx_g % len(default_palette)]) if color_map else default_palette[idx_g % len(default_palette)]

        n_total = len(sub_t)
        n_events = int(np.sum(sub_e))
        n_censored = n_total - n_events
        mort_rate = (n_events / n_total * 100.0) if n_total > 0 else 0.0

        km_fitter.fit(sub_t, event_observed=sub_e, label=f"{g} (n={n_total}, dead={n_events})")
        km_fitter.plot_survival_function(ax=ax_km, color=col, ci_show=True, ci_alpha=0.12, linewidth=2.2)

        cens_t = sub_t[sub_e == 0]
        if len(cens_t) > 0:
            try:
                cens_s = km_fitter.predict(cens_t).values
                ax_km.plot(cens_t, cens_s, '+', color=col, markersize=8, markeredgewidth=2)
            except Exception:
                pass

        median_surv = km_fitter.median_survival_time_
        median_str = f"{int(round(median_surv))}d" if (np.isfinite(median_surv) and median_surv > 0) else "> Max Obs"
        cohort_summary.append({
            "Cohort": g,
            "Sample Size (N)": n_total,
            "Deaths (Events)": n_events,
            "Censored": n_censored,
            "Mortality Rate (%)": f"{mort_rate:.1f}%",
            "Median Survival (LT50)": median_str
        })

    ax_km.axhline(0.5, color='#475569', linestyle=':', linewidth=1.3, label='50% Threshold (LT50)')
    ax_km.set_xlabel('Elapsed Observation Time (Days)', fontsize=12, fontweight='bold', labelpad=8)
    ax_km.set_ylabel('Survival Probability S(t)', fontsize=12, fontweight='bold', labelpad=8)
    ax_km.set_ylim(-0.02, 1.05)
    max_plot_t = max(T) if len(T) > 0 else 8.0
    ax_km.set_xlim(left=0.8, right=max(max_plot_t + 0.5, 8.0))
    ax_km.xaxis.set_major_locator(MultipleLocator(1))
    ax_km.spines['top'].set_visible(False)
    ax_km.spines['right'].set_visible(False)
    ax_km.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.92, fontsize=10)
    fig_km.tight_layout()

    st.pyplot(fig_km)

    buf_km = io.BytesIO()
    fig_km.savefig(buf_km, format="png", dpi=300, bbox_inches='tight')
    dl_km_key = f"dl_km_plot_{group_col}_{key_suffix}" if key_suffix else f"dl_km_plot_{group_col}"
    st.download_button("📥 Download Kaplan-Meier Multi-Cohort Plot (PNG)", data=buf_km.getvalue(), file_name="logrank_kaplan_meier_comparison.png", mime="image/png", key=dl_km_key)
    plt.close(fig_km)

    st.markdown("#### 👥 Cohort Survival Breakdown (Alive Post-Starvation Baseline)")
    st.dataframe(pd.DataFrame(cohort_summary), use_container_width=True)

    if len(groups) >= 3:
        st.markdown("#### 🔄 Post-Hoc Pairwise Log-Rank Comparisons (with Bonferroni Correction)")
        pairs = list(combinations(groups, 2))
        n_comp = max(1, len(pairs))
        pairwise_data = []
        for g1, g2 in pairs:
            m1 = (G == g1)
            m2 = (G == g2)
            e1 = E[m1]
            e2 = E[m2]
            if np.sum(e1) == 0 and np.sum(e2) == 0:
                pairwise_data.append({
                    "Comparison": f"{g1} vs. {g2}",
                    "Chi² (χ²)": 0.0,
                    "Raw p-value": "1.000",
                    "Bonferroni p-value": "1.000",
                    "Status (α = 0.05)": "⚪ Not Significant"
                })
                continue
            try:
                res_p = logrank_test(T[m1], T[m2], event_observed_A=e1, event_observed_B=e2)
                p_raw = res_p.p_value
                p_bonf = min(1.0, p_raw * n_comp)
                sig_label = "✅ Significant" if p_bonf < 0.05 else "⚪ Not Significant"
                pairwise_data.append({
                    "Comparison": f"{g1} vs. {g2}",
                    "Chi² (χ²)": round(res_p.test_statistic, 3),
                    "Raw p-value": f"{p_raw:.4g}",
                    "Bonferroni p-value": f"{p_bonf:.4g}",
                    "Status (α = 0.05)": sig_label
                })
            except Exception:
                pass
        if pairwise_data:
            st.dataframe(pd.DataFrame(pairwise_data), use_container_width=True)

    if len(groups) >= 2:
        st.markdown("#### 💡 Biological Interpretation")
        if is_significant:
            st.success(f"**Statistically Significant Difference Detected:** The overall survival distributions differ significantly across cohorts (Mantel-Cox Log-Rank $\\chi^2 = {chisq:.3f}, df = {df_val}, p = {p_str}$). Reject the null hypothesis of equal survival across treatment cohorts.")
        else:
            st.info(f"**No Statistically Significant Difference:** The survival distributions do not differ significantly at the $\\alpha = 0.05$ level (Mantel-Cox Log-Rank $\\chi^2 = {chisq:.3f}, df = {df_val}, p = {p_str}$).")
    else:
        st.markdown("#### 💡 Biological Summary")
        st.info(f"**Single Cohort Evaluated (`{groups[0]}`):** Median survival ($LT_{{50}}$) is estimated at **{med_str}** with a final mortality rate of **{mort_rate:.1f}%** ({n_events} deaths out of {n_total} bees alive post-starvation baseline). Upload additional cohort CSV files to run comparative Log-Rank hypothesis testing.")
