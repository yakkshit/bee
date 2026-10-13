import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from itertools import combinations
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test


def render_log_rank_analysis(survival_df, group_col='cohort', color_map=None):
    """
    Renders Log-Rank (Mantel-Cox) Survival Test across cohorts/treatments.
    Computes chi2, p-value, df, pairwise comparisons with Bonferroni correction,
    and a multi-cohort Kaplan-Meier survival curves plot.
    """
    if survival_df is None or survival_df.empty:
        st.warning("No individual survival data available for Log-Rank test.")
        return

    groups = [g for g in survival_df[group_col].unique() if pd.notna(g) and str(g).strip() != '']
    if len(groups) < 2:
        st.info(f"💡 Log-Rank test requires at least 2 distinct groups to compare. Found {len(groups)} group ({groups[0] if groups else 'None'}).")
        return

    T = survival_df['duration'].values
    E = survival_df['event'].values
    G = survival_df[group_col].values

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
            cens_s = [km_fitter.survival_function_at_times(ct).values[0] for ct in cens_t]
            ax_km.plot(cens_t, cens_s, '+', color=col, markersize=8, markeredgewidth=2)

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
    ax_km.spines['top'].set_visible(False)
    ax_km.spines['right'].set_visible(False)
    ax_km.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.92, fontsize=10)
    fig_km.tight_layout()

    st.pyplot(fig_km)

    buf_km = io.BytesIO()
    fig_km.savefig(buf_km, format="png", dpi=300, bbox_inches='tight')
    st.download_button("📥 Download Kaplan-Meier Multi-Cohort Plot (PNG)", data=buf_km.getvalue(), file_name="logrank_kaplan_meier_comparison.png", mime="image/png")
    plt.close(fig_km)

    st.markdown("#### 👥 Cohort Survival Breakdown")
    st.dataframe(pd.DataFrame(cohort_summary), use_container_width=True)

    if len(groups) >= 3:
        st.markdown("#### 🔄 Post-Hoc Pairwise Log-Rank Comparisons (with Bonferroni Correction)")
        pairs = list(combinations(groups, 2))
        n_comp = max(1, len(pairs))
        pairwise_data = []
        for g1, g2 in pairs:
            m1 = (G == g1)
            m2 = (G == g2)
            try:
                res_p = logrank_test(T[m1], T[m2], event_observed_A=E[m1], event_observed_B=E[m2])
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

    st.markdown("#### 💡 Biological Interpretation")
    if is_significant:
        st.success(f"**Statistically Significant Difference Detected:** The overall survival distributions differ significantly across cohorts (Mantel-Cox Log-Rank $\\chi^2 = {chisq:.3f}, df = {df_val}, p = {p_str}$). Reject the null hypothesis of equal survival across treatment cohorts.")
    else:
        st.info(f"**No Statistically Significant Difference:** The survival distributions do not differ significantly at the $\\alpha = 0.05$ level (Mantel-Cox Log-Rank $\\chi^2 = {chisq:.3f}, df = {df_val}, p = {p_str}$).")


def render_cox_regression_analysis(survival_df, group_col='cohort', reference_group=None):
    """
    Fits Cox Proportional Hazards Model, computes Hazard Ratios, C-index,
    and generates a Forest Plot with error whiskers.
    """
    if survival_df is None or survival_df.empty:
        st.warning("No individual survival data available for Cox Regression.")
        return

    groups = [g for g in survival_df[group_col].unique() if pd.notna(g) and str(g).strip() != ''] if (group_col and group_col in survival_df.columns) else []

    st.markdown("Configure covariates for the Cox Proportional Hazards Model:")
    c_cfg1, c_cfg2 = st.columns([1, 2])
    with c_cfg1:
        if len(groups) > 1:
            ref_selected = st.selectbox("Baseline Reference Cohort (HR = 1.0)", groups, index=0, key=f"cox_ref_{group_col}")
        else:
            ref_selected = groups[0] if groups else None

    with c_cfg2:
        cov_options = []
        for col_key, col_label in [('mean_consumption', "Mean Consumption (µL/day)"), ('temp', "Temperature (°C)"), ('humidity', "Humidity (%)")]:
            if col_key in survival_df.columns:
                s_clean = pd.to_numeric(survival_df[col_key], errors='coerce').dropna()
                if len(s_clean) > 1 and s_clean.nunique() > 1 and float(s_clean.std()) > 0.05:
                    cov_options.append(col_label)

        selected_covs = st.multiselect("Additional Covariates to Include:", cov_options, default=cov_options, key=f"cox_covs_{group_col}")

    cph_df = survival_df[['duration', 'event']].copy()

    if len(groups) > 1 and ref_selected is not None:
        dummies = pd.get_dummies(survival_df[group_col], prefix='Cohort', dtype=float)
        ref_col = f"Cohort_{ref_selected}"
        if ref_col in dummies.columns:
            dummies = dummies.drop(columns=[ref_col])
        valid_dummies = [c for c in dummies.columns if dummies[c].nunique() > 1 and dummies[c].std() > 0.01]
        if valid_dummies:
            cph_df = pd.concat([cph_df, dummies[valid_dummies]], axis=1)

    if "Mean Consumption (µL/day)" in selected_covs and 'mean_consumption' in survival_df.columns:
        s_c = pd.to_numeric(survival_df['mean_consumption'], errors='coerce').fillna(0.0)
        if s_c.nunique() > 1 and s_c.std() > 0.05:
            cph_df['Mean_Consumption_uL'] = s_c.astype(float)

    if "Temperature (°C)" in selected_covs and 'temp' in survival_df.columns:
        s_t = pd.to_numeric(survival_df['temp'], errors='coerce').fillna(25.0)
        if s_t.nunique() > 1 and s_t.std() > 0.05:
            cph_df['Temp_C'] = s_t.astype(float)

    if "Humidity (%)" in selected_covs and 'humidity' in survival_df.columns:
        s_h = pd.to_numeric(survival_df['humidity'], errors='coerce').fillna(60.0)
        if s_h.nunique() > 1 and s_h.std() > 0.05:
            cph_df['Humidity_pct'] = s_h.astype(float)

    covariate_cols = [c for c in cph_df.columns if c not in ['duration', 'event']]
    if not covariate_cols:
        st.info("💡 **Cox Proportional Hazards Model**: Requires at least 2 distinct cohorts (or individual covariates with variance such as feeding consumption) to quantify hazard ratios. Please select additional cohorts or covariates above.")
        return

    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(cph_df, duration_col='duration', event_col='event')
    except Exception as ex:
        st.error(f"Error fitting Cox Proportional Hazards Model: {ex}")
        return

    c_index = cph.concordance_index_
    lrt = cph.log_likelihood_ratio_test()
    aic_val = cph.AIC_partial_

    ck1, ck2, ck3, ck4 = st.columns(4)
    with ck1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{c_index:.3f}</div><div class="kpi-label">Concordance Index (C-index)</div></div>', unsafe_allow_html=True)
    with ck2:
        lrt_p_str = "< 0.0001" if lrt.p_value < 0.0001 else f"{lrt.p_value:.4f}"
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{lrt.test_statistic:.2f} (p={lrt_p_str})</div><div class="kpi-label">Likelihood Ratio Test (χ²)</div></div>', unsafe_allow_html=True)
    with ck3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{aic_val:.2f}</div><div class="kpi-label">Partial AIC</div></div>', unsafe_allow_html=True)
    with ck4:
        c_qual = "Strong (>0.80)" if c_index >= 0.8 else ("Good (>0.70)" if c_index >= 0.7 else "Moderate")
        st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color: #0284c7;">{c_qual}</div><div class="kpi-label">Discrimination Power</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    s = cph.summary
    table_rows = []
    for var_name, row in s.iterrows():
        b_coef = float(row['coef'])
        raw_hr = float(row['exp(coef)'])
        se_val = float(row['se(coef)']) if pd.notna(row['se(coef)']) else 0.0

        raw_ci_l = float(row['coef lower 95%']) if 'coef lower 95%' in row and pd.notna(row['coef lower 95%']) else (b_coef - 1.96 * se_val)
        raw_ci_u = float(row['coef upper 95%']) if 'coef upper 95%' in row and pd.notna(row['coef upper 95%']) else (b_coef + 1.96 * se_val)

        # Clip exponent arguments between -25 and 25 to prevent float overflow
        ci_l = float(np.exp(np.clip(raw_ci_l, -25.0, 25.0)))
        ci_u = float(np.exp(np.clip(raw_ci_u, -25.0, 25.0)))
        hr_val = float(np.exp(np.clip(b_coef, -25.0, 25.0)))

        if not np.isfinite(hr_val) or hr_val <= 0:
            hr_val = 1.0
        if not np.isfinite(ci_l) or ci_l <= 0:
            ci_l = 0.01
        if not np.isfinite(ci_u) or ci_u <= 0:
            ci_u = max(ci_l * 1.5, hr_val * 2.0, 2.0)
        if ci_u < ci_l:
            ci_u = ci_l * 1.5

        z_v = float(row['z']) if pd.notna(row['z']) else 0.0
        p_v = float(row['p']) if pd.notna(row['p']) else 1.0
        sig_str = "✅ p < 0.05" if p_v < 0.05 else "⚪ n.s."

        display_name = str(var_name).replace("Cohort_", "Cohort: ").replace("_", " ")
        table_rows.append({
            "Covariate": display_name,
            "Coef (β)": round(b_coef, 4),
            "Hazard Ratio (HR)": round(hr_val, 4),
            "95% CI Lower": round(ci_l, 4),
            "95% CI Upper": round(ci_u, 4),
            "Std Error (SE)": round(se_val, 4),
            "Wald z": round(z_v, 3),
            "p-value": f"{p_v:.4g}",
            "Significance": sig_str
        })
    res_df = pd.DataFrame(table_rows)
    st.markdown("#### 📋 Cox Proportional Hazards Model Coefficients & Hazard Ratios")
    st.dataframe(res_df, use_container_width=True)

    st.markdown("#### 🌲 Hazard Ratio Forest Plot (95% CI)")
    fig_fp, ax_fp = plt.subplots(figsize=(8.8, max(3.6, len(table_rows) * 0.75 + 1.2)), dpi=300)
    ax_fp.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', axis='x', zorder=1)

    y_pos = np.arange(len(table_rows))[::-1]
    labels = [r["Covariate"] for r in table_rows]
    p_vals = [float(r["p-value"]) for r in table_rows]

    # Ensure all values in Forest Plot are strictly positive, finite, and well-behaved
    hr_list = []
    ci_l_list = []
    ci_u_list = []
    xerr_low = []
    xerr_high = []

    for r in table_rows:
        hr_clean = float(np.clip(r["Hazard Ratio (HR)"], 0.01, 50.0))
        ci_l_clean = float(np.clip(r["95% CI Lower"], 0.005, hr_clean))
        ci_u_clean = float(np.clip(r["95% CI Upper"], hr_clean, 60.0))

        err_l = max(0.001, hr_clean - ci_l_clean)
        err_u = max(0.001, ci_u_clean - hr_clean)

        hr_list.append(hr_clean)
        ci_l_list.append(ci_l_clean)
        ci_u_list.append(ci_u_clean)
        xerr_low.append(err_l)
        xerr_high.append(err_u)

    pt_colors = []
    for hr_item, p_item in zip(hr_list, p_vals):
        if p_item < 0.05:
            pt_colors.append('#dc2626' if hr_item > 1.0 else '#16a34a')
        else:
            pt_colors.append('#64748b')

    for i in range(len(y_pos)):
        ax_fp.errorbar(
            hr_list[i], y_pos[i],
            xerr=[[xerr_low[i]], [xerr_high[i]]],
            fmt='o', color=pt_colors[i], ecolor=pt_colors[i],
            elinewidth=2.2, capsize=5, capthick=1.8, markersize=8.5, zorder=3
        )
        annot = f"HR = {hr_list[i]:.2f} [{ci_l_list[i]:.2f}, {ci_u_list[i]:.2f}] (p={p_vals[i]:.3g})"
        ax_fp.text(ci_u_list[i] * 1.04 + 0.05, y_pos[i], annot, va='center', ha='left', fontsize=9.5, fontweight='bold', color=pt_colors[i])

    ax_fp.axvline(1.0, color='#dc2626', linestyle='--', linewidth=1.5, label='Null Effect (HR = 1.0)', zorder=2)
    ax_fp.set_yticks(y_pos)
    ax_fp.set_yticklabels(labels, fontsize=11, fontweight='bold')
    ax_fp.set_xlabel('Hazard Ratio (HR) with 95% Confidence Intervals', fontsize=12, fontweight='bold', labelpad=8)
    ax_fp.spines['top'].set_visible(False)
    ax_fp.spines['right'].set_visible(False)
    ax_fp.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)

    max_ci = max(ci_u_list) if ci_u_list else 2.0
    min_ci = min(ci_l_list) if ci_l_list else 0.5

    left_bound = max(0.0, min_ci * 0.8 - 0.2)
    right_bound = min(max_ci * 1.45 + 0.5, 75.0)

    # Bulletproof check: Ensure bounds are strictly finite numbers
    if not np.isfinite(left_bound) or left_bound < 0:
        left_bound = 0.0
    if not np.isfinite(right_bound) or right_bound <= left_bound:
        right_bound = max(left_bound + 2.0, 4.0)

    ax_fp.set_xlim(left=left_bound, right=right_bound)

    fig_fp.tight_layout()
    st.pyplot(fig_fp)

    buf_fp = io.BytesIO()
    fig_fp.savefig(buf_fp, format="png", dpi=300, bbox_inches='tight')

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button("📥 Download Forest Plot (PNG)", data=buf_fp.getvalue(), file_name="cox_forest_plot.png", mime="image/png")
    with col_dl2:
        st.download_button("📊 Download Cox Regression Summary (CSV)", data=res_df.to_csv(index=False), file_name="cox_regression_summary.csv", mime="text/csv")
    plt.close(fig_fp)

    st.markdown("#### 💡 Biological Hazard Interpretation")
    for r in table_rows:
        hr_item = r["Hazard Ratio (HR)"]
        p_item = float(r["p-value"])
        cov_name = r["Covariate"]
        if p_item < 0.05:
            if hr_item > 1.0:
                pct_inc = (hr_item - 1.0) * 100.0
                st.write(f"- **{cov_name}**: Associated with a **{pct_inc:.1f}% increase** in instantaneous mortality risk relative to baseline ($HR = {hr_item:.2f}, p = {p_item:.4g}$). Accelerates mortality.")
            else:
                pct_dec = (1.0 - hr_item) * 100.0
                st.write(f"- **{cov_name}**: Associated with a **{pct_dec:.1f}% decrease** in instantaneous mortality risk relative to baseline ($HR = {hr_item:.2f}, p = {p_item:.4g}$). Conveys relative survival protection.")
        else:
            st.write(f"- **{cov_name}**: No statistically significant hazard effect detected ($HR = {hr_item:.2f}, p = {p_item:.4g}$).")
