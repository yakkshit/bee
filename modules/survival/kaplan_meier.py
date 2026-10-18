"""
Kaplan-Meier Survival Analysis Module.
Renders non-parametric step survival curves with Greenwood 95% confidence intervals,
median survival (LT50) estimations, censoring markers (+), and step-by-step product-limit life tables.
"""
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from lifelines import KaplanMeierFitter


def compute_kaplan_meier_life_table(durations, events):
    """
    Computes the exact step-by-step Kaplan-Meier product-limit life table:
        Ŝ(t) = ∏_{t_i ≤ t} (1 - d_i / n_i)
    Returns a pandas DataFrame with step-by-step arithmetic steps matching
    the exact formulation in the bioassay protocol (e.g. S(1) = 9/10 = 0.90, S(2) = 0.90 * 8/9 = 0.80).
    """
    if len(durations) == 0:
        return pd.DataFrame()

    df_raw = pd.DataFrame({'t': np.array(durations, dtype=float), 'e': np.array(events, dtype=int)}).sort_values('t')
    unique_times = sorted(df_raw['t'].unique())

    n_total = len(df_raw)
    current_at_risk = n_total
    cum_surv = 1.0
    greenwood_sum = 0.0

    rows = []

    for idx, t in enumerate(unique_times):
        sub = df_raw[df_raw['t'] == t]
        d_i = int(sub['e'].sum())
        c_i = int((sub['e'] == 0).sum())
        n_i = current_at_risk

        if n_i <= 0:
            break

        cond_p = (n_i - d_i) / n_i if n_i > 0 else 1.0

        # Build arithmetic formula string matching the bioassay guide
        t_label = f"{t:g}" if float(t).is_integer() else f"{t:.2f}"
        if d_i > 0:
            if idx == 0 or np.isclose(cum_surv, 1.0):
                formula_str = f"S({t_label}) = {n_i - d_i}/{n_i} = {cond_p:.3f}"
            else:
                formula_str = f"S({t_label}) = {cum_surv:.3f} × {n_i - d_i}/{n_i} = {cum_surv * cond_p:.3f}"
        else:
            formula_str = f"S({t_label}) = {cum_surv:.3f} (No deaths; {c_i} censored)"

        cum_surv = cum_surv * cond_p

        # Greenwood standard error formula: SE = S(t) * sqrt(sum(d_i / (n_i * (n_i - d_i))))
        if n_i > d_i and n_i > 0 and d_i > 0:
            greenwood_sum += d_i / (n_i * (n_i - d_i))
            se = cum_surv * np.sqrt(greenwood_sum)
        else:
            se = cum_surv * np.sqrt(greenwood_sum) if greenwood_sum > 0 else 0.0

        ci_lower = max(0.0, cum_surv - 1.96 * se)
        ci_upper = min(1.0, cum_surv + 1.96 * se)

        rows.append({
            "Time (Day)": f"Day {t_label}",
            "At Risk (nᵢ)": n_i,
            "Deaths (dᵢ)": d_i,
            "Censored (cᵢ)": c_i,
            "Conditional Survival (1 - dᵢ/nᵢ)": f"{cond_p:.4f}" if d_i > 0 else "1.0000",
            "Cumulative Survival Ŝ(t)": f"{cum_surv * 100.0:.1f}%",
            "Formula Calculation Step": formula_str,
            "Greenwood SE": f"{se:.4f}",
            "95% CI (Survival)": f"[{ci_lower * 100.0:.1f}%, {ci_upper * 100.0:.1f}%]"
        })

        current_at_risk -= (d_i + c_i)

    return pd.DataFrame(rows)


def render_kaplan_meier_analysis(survival_df, group_col='cohort', color_map=None, key_suffix: str = ""):
    """
    Renders the dedicated Plot 1: Kaplan–Meier Survival Curves Plot.
    Computes step-down survival kinetics, Greenwood 95% confidence intervals,
    censoring marks (+), median survival LT50, and step-by-step product-limit life tables.
    """
    if survival_df is None or survival_df.empty:
        st.warning("No individual survival data available for Kaplan-Meier analysis.")
        return

    # Cohort Baseline Entry Protocol
    entry_mode_key = f"km_entry_{group_col}_{key_suffix}" if key_suffix else f"km_entry_{group_col}"
    entry_mode = st.radio(
        "Cohort Baseline Entry Protocol:",
        [
            "Standard Kaplan-Meier (All bees alive at start of Day 1; Day 1 deaths included in calculation)",
            "Pre-treatment Starvation Exclusion (Exclude bees dead on Day 1 from starting N - Lab baseline rule)"
        ],
        index=0,
        horizontal=False,
        key=entry_mode_key,
        help="Standard Kaplan-Meier keeps all bees in starting N at Day 1. If 1 bee dies on Day 1 out of 10, S(1) = 9/10 = 0.90 (matching the ChatGPT formula). Pre-treatment exclusion assumes bees dead on Day 1 died during starvation setup and removes them from the cohort denominator."
    )

    if "Pre-treatment Starvation Exclusion" in entry_mode:
        alive_mask = ~((survival_df['duration'] <= 1.0) & (survival_df['event'] == 1))
        valid_survival_df = survival_df[alive_mask].copy() if alive_mask.any() else survival_df.copy()
        st.info("🕒 **Pre-Treatment Starvation Exclusion Active**: Initial sample size ($N_{\\text{total}}$) excludes bees dead on Day 1 baseline, ensuring $N_{\\text{total}}$ strictly represents bees surviving post-starvation.")
    else:
        valid_survival_df = survival_df.copy()
    # Biological Endpoint and Intake Stratification
    has_cons_col = ('mean_consumption' in survival_df.columns and (survival_df['mean_consumption'] > 0).any())
    endpoint_options = ["Overall Mortality & Survival (Standard Kaplan-Meier)"]
    if has_cons_col:
        endpoint_options.append("Feeding Persistence (Nutritional Anorexia Endpoint - Time to Cessation)")
        endpoint_options.append("Survival Stratified by Sucrose Intake (High Feeders vs Low Feeders)")

    endpoint_k = f"km_endpoint_{group_col}_{key_suffix}" if key_suffix else f"km_endpoint_{group_col}"
    sel_endpoint = st.radio(
        "Biological Endpoint & Stratification:",
        endpoint_options,
        index=0,
        horizontal=True,
        key=endpoint_k,
        help="Switch between classic mortality kinetics, nutritional feeding persistence (time-to-anorexia), or stratifying survival by daily sucrose consumption levels."
    )

    if "Feeding Persistence" in sel_endpoint and 'feeding_cessation_duration' in valid_survival_df.columns:
        T = valid_survival_df['feeding_cessation_duration'].values
        E = valid_survival_df['feeding_cessation_event'].values
        G = valid_survival_df[group_col].values
        groups = [g for g in valid_survival_df[group_col].unique() if pd.notna(g) and str(g).strip() != '']
        active_color_map = color_map or {}
        y_axis_label = 'Feeding Probability Ŝ_feed(t)'
        threshold_text = '50% Threshold (FT50)'
        metric_label_name = 'Median Feeding (FT50)'
        events_label_name = 'Ceased Feeding (Events)'
        cens_label_name = 'Still Feeding (Active)'
        st.info("🕒 **Nutritional Feeding Persistence Active**: Evaluating the probability of bees remaining active feeders over time. Event = permanent cessation of feeding ($C \\le 0$).")
    elif "Stratified by Sucrose Intake" in sel_endpoint and has_cons_col:
        med_intake = float(valid_survival_df['mean_consumption'].median())
        valid_survival_df['intake_stratum'] = valid_survival_df['mean_consumption'].apply(
            lambda v: f"High Feeders (≥ {med_intake:.1f} µL/d)" if v >= med_intake else f"Low Feeders (< {med_intake:.1f} µL/d)"
        )
        T = valid_survival_df['duration'].values
        E = valid_survival_df['event'].values
        G = valid_survival_df['intake_stratum'].values
        groups = sorted(list(valid_survival_df['intake_stratum'].unique()), reverse=True)
        active_color_map = {
            f"High Feeders (≥ {med_intake:.1f} µL/d)": "#10b981",
            f"Low Feeders (< {med_intake:.1f} µL/d)": "#ef4444"
        }
        y_axis_label = 'Survival Probability Ŝ(t)'
        threshold_text = '50% Threshold (LT50)'
        metric_label_name = 'Median Survival (LT50)'
        events_label_name = 'Observed Deaths'
        cens_label_name = 'Censored / Alive'
        st.info(f"🍯 **Sucrose Intake Stratification Active**: Bees partitioned into High vs. Low Feeders around median daily intake ({med_intake:.1f} µL/bee/day). Evaluates if nutrition significantly prolongs bee lifespan.")
    else:
        T = valid_survival_df['duration'].values
        E = valid_survival_df['event'].values
        G = valid_survival_df[group_col].values
        groups = [g for g in valid_survival_df[group_col].unique() if pd.notna(g) and str(g).strip() != '']
        active_color_map = color_map or {}
        y_axis_label = 'Survival Probability Ŝ(t)'
        threshold_text = '50% Threshold (LT50)'
        metric_label_name = 'Median Survival (LT50)'
        events_label_name = 'Observed Deaths'
        cens_label_name = 'Censored / Alive'

    if len(groups) == 0:
        st.warning("No valid cohort data available for Kaplan-Meier analysis.")
        return

    total_events = int(np.sum(E))

    # Top-Level Summary KPIs
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{len(T)}</div><div class="kpi-label">Total Bees (N)</div></div>', unsafe_allow_html=True)
    with col_k2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{total_events}</div><div class="kpi-label">{events_label_name}</div></div>', unsafe_allow_html=True)
    with col_k3:
        n_cens_all = len(T) - total_events
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{n_cens_all}</div><div class="kpi-label">{cens_label_name}</div></div>', unsafe_allow_html=True)
    with col_k4:
        km_all = KaplanMeierFitter()
        km_all.fit(T, event_observed=E)
        med_all = km_all.median_survival_time_
        med_all_str = f"{int(round(med_all))}d" if (np.isfinite(med_all) and med_all > 0) else "> Max Obs"
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{med_all_str}</div><div class="kpi-label">Overall {metric_label_name}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Render Plot 1: Kaplan-Meier Survival Curves
    fig_km, ax_km = plt.subplots(figsize=(9.2, 5.5), dpi=300)
    ax_km.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', zorder=1)

    km_fitter = KaplanMeierFitter()
    cohort_summary = []
    default_palette = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4"]

    for idx_g, g in enumerate(groups):
        m = (G == g)
        sub_t = T[m]
        sub_e = E[m]
        col = active_color_map.get(g, default_palette[idx_g % len(default_palette)]) if active_color_map else default_palette[idx_g % len(default_palette)]

        n_total = len(sub_t)
        n_events = int(np.sum(sub_e))
        n_censored = n_total - n_events
        event_rate = (n_events / n_total * 100.0) if n_total > 0 else 0.0

        km_fitter.fit(sub_t, event_observed=sub_e, label=f"{g} (n={n_total}, events={n_events})")
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
            "Cohort / Stratum": g,
            "Sample Size (N)": n_total,
            events_label_name: n_events,
            cens_label_name: n_censored,
            "Event Rate (%)": f"{event_rate:.1f}%",
            metric_label_name: median_str
        })

    # If stratified by intake or multiple groups with events, calculate Log-Rank test
    if len(groups) > 1 and total_events > 0:
        try:
            from lifelines.statistics import multivariate_logrank_test
            lr_res_km = multivariate_logrank_test(T, G, E)
            p_val_km = lr_res_km.p_value
            p_str_km = "p < 0.001" if p_val_km < 0.001 else f"p = {p_val_km:.4f}"
            ax_km.text(0.04, 0.12, f"Log-Rank: {p_str_km}", transform=ax_km.transAxes,
                       fontsize=10, fontweight='bold', bbox=dict(facecolor='white', edgecolor='#cbd5e1', alpha=0.92))
        except Exception:
            pass

    ax_km.axhline(0.5, color='#475569', linestyle=':', linewidth=1.3, label=threshold_text)
    ax_km.set_xlabel('Elapsed Observation Time (Days)', fontsize=12, fontweight='bold', labelpad=8)
    ax_km.set_ylabel(y_axis_label, fontsize=12, fontweight='bold', labelpad=8)
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
    dl_km_key = f"dl_km_plot_standalone_{group_col}_{key_suffix}" if key_suffix else f"dl_km_plot_standalone_{group_col}"
    st.download_button("📥 Download Kaplan-Meier Multi-Cohort Plot (PNG)", data=buf_km.getvalue(), file_name="kaplan_meier_survival_curves.png", mime="image/png", key=dl_km_key)
    plt.close(fig_km)

    st.markdown(f"#### 👥 Cohort / Stratum Breakdown & {metric_label_name} Summary")
    st.dataframe(pd.DataFrame(cohort_summary), use_container_width=True)

    with st.expander("📋 Step-by-Step Kaplan–Meier Life Tables & Calculation Breakdown (Formula: Ŝ(t) = ∏ (1 - dᵢ/nᵢ))", expanded=True):
        st.markdown("""
        Inspect the exact step-by-step product-limit conditional probabilities for each cohort, following the formula:
        $$\\hat{S}(t) = \\prod_{t_i \\le t} \\left(1 - \\frac{d_i}{n_i}\\right)$$
        """)

        if groups:
            lt_cohort_k = f"lt_cohort_sel_{group_col}_{key_suffix}" if key_suffix else f"lt_cohort_sel_{group_col}"
            sel_lt_cohort = st.selectbox("Select Cohort to Inspect Step-by-Step Product-Limit Math:", groups, key=lt_cohort_k)

            m_sel = np.array([str(g) == str(sel_lt_cohort) for g in G])
            t_sel = T[m_sel]
            e_sel = E[m_sel]

            life_tbl_df = compute_kaplan_meier_life_table(t_sel, e_sel)
        if not life_tbl_df.empty:
            st.dataframe(life_tbl_df, use_container_width=True)

            final_surv_str = life_tbl_df['Cumulative Survival Ŝ(t)'].iloc[-1] if 'Cumulative Survival Ŝ(t)' in life_tbl_df.columns else "N/A"
            total_deaths_c = int(np.sum(e_sel))
            n_init_c = len(t_sel)
            alive_c = n_init_c - total_deaths_c
            st.caption(f"💡 **Cohort Summary (`{sel_lt_cohort}`)**: {final_surv_str} final estimated survival. {alive_c} of the initial {n_init_c} bees remain alive / censored at the end of observation.")

            csv_lt = life_tbl_df.to_csv(index=False)
            safe_c_name = "".join(c if c.isalnum() else "_" for c in str(sel_lt_cohort)).lower()
            dl_lt_k = f"dl_lt_{safe_c_name}_{key_suffix}" if key_suffix else f"dl_lt_{safe_c_name}"
            st.download_button(
                f"📥 Download '{sel_lt_cohort}' Kaplan-Meier Life Table (CSV)",
                data=csv_lt,
                file_name=f"kaplan_meier_life_table_{safe_c_name}.csv",
                mime="text/csv",
                key=dl_lt_k
            )
