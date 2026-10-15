"""
Log-Rank (Mantel-Cox) Survival Analysis Module.
Performs formal non-parametric hypothesis testing between cohort survival curves,
computes Mantel-Haenszel Observed vs. Expected deaths under H₀, pairwise post-hoc tests with Bonferroni correction,
and renders the dedicated Log-Rank Observed vs. Expected Comparison Plot.
"""
import io
from itertools import combinations
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines import NelsonAalenFitter


def render_log_rank_analysis(survival_df, group_col='cohort', color_map=None, key_suffix: str = ""):
    """
    Renders Plot 2: Log-Rank (Mantel-Cox) Statistical Hypothesis Test.
    Evaluates whether survival distributions differ significantly between cohorts.
    Computes chi2, p-value, df, Observed vs. Expected deaths under H0,
    and renders the dedicated Log-Rank Comparison Plot and Pairwise Bonferroni Table.
    """
    if survival_df is None or survival_df.empty:
        st.warning("No individual survival data available for Log-Rank test.")
        return

    groups = [g for g in survival_df[group_col].unique() if pd.notna(g) and str(g).strip() != '']
    if len(groups) == 0:
        st.warning("No valid cohort data available for Log-Rank test.")
        return

    T = survival_df['duration'].values
    E = survival_df['event'].values
    G = survival_df[group_col].values

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

        st.markdown("<br>", unsafe_allow_html=True)

        # Compute Observed vs. Expected Deaths under H0 for each cohort
        death_times = sorted(survival_df[survival_df['event'] == 1]['duration'].unique())
        obs_dict = {g: 0 for g in groups}
        exp_dict = {g: 0.0 for g in groups}

        for t_death in death_times:
            at_risk_mask = (T >= t_death)
            total_at_risk = int(np.sum(at_risk_mask))
            total_deaths_at_t = int(np.sum((T == t_death) & (E == 1)))

            if total_at_risk == 0:
                continue

            for g in groups:
                g_at_risk = int(np.sum(at_risk_mask & (G == g)))
                g_deaths = int(np.sum((T == t_death) & (E == 1) & (G == g)))
                obs_dict[g] += g_deaths
                exp_dict[g] += total_deaths_at_t * (g_at_risk / total_at_risk)

        default_palette = ["#2563eb", "#0d9488", "#f59e0b", "#dc2626", "#8b5cf6", "#ec4899", "#06b6d4"]
        cohort_colors = [color_map.get(g, default_palette[i % len(default_palette)]) if color_map else default_palette[i % len(default_palette)] for i, g in enumerate(groups)]

        # Plot mode option: Observed vs Expected Bar Chart or Cumulative Hazard Curves
        plot_view_k = f"lr_plot_view_{group_col}_{key_suffix}" if key_suffix else f"lr_plot_view_{group_col}"
        col_pv1, col_pv2 = st.columns([1.5, 2])
        with col_pv1:
            plot_view = st.selectbox(
                "Log-Rank Visualization Mode:",
                [
                    "Observed vs. Expected Deaths (O vs. E) Bar Chart",
                    "Cumulative Hazard Function Curves: H(t) = -ln S(t)"
                ],
                index=0,
                key=plot_view_k
            )

        if "Observed vs. Expected" in plot_view:
            fig_lr, ax_lr = plt.subplots(figsize=(9.2, 5.2), dpi=300)
            ax_lr.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', axis='y', zorder=1)

            x_indices = np.arange(len(groups))
            bar_width = 0.35

            O_vals = [obs_dict[g] for g in groups]
            E_vals = [exp_dict[g] for g in groups]

            bars_obs = ax_lr.bar(x_indices - bar_width/2, O_vals, bar_width, label='Observed Deaths (O)', color=cohort_colors, edgecolor='black', linewidth=1.1, zorder=3)
            bars_exp = ax_lr.bar(x_indices + bar_width/2, E_vals, bar_width, label='Expected Deaths under H₀ (E)', color=cohort_colors, alpha=0.35, hatch='//', edgecolor='black', linewidth=1.1, zorder=3)

            max_y = max(max(O_vals, default=1), max(E_vals, default=1))

            for i, g in enumerate(groups):
                o_val = O_vals[i]
                e_val = E_vals[i]
                ax_lr.text(x_indices[i] - bar_width/2, o_val + (max_y * 0.02), str(o_val), ha='center', va='bottom', fontweight='bold', fontsize=9.5)
                ax_lr.text(x_indices[i] + bar_width/2, e_val + (max_y * 0.02), f"{e_val:.1f}", ha='center', va='bottom', fontsize=9.0)

                ratio = o_val / e_val if e_val > 0 else 1.0
                r_col = '#10b981' if ratio < 1.0 else '#ef4444'
                badge_text = f"O/E = {ratio:.2f}"
                ax_lr.text(x_indices[i], max(o_val, e_val) + (max_y * 0.12), badge_text, ha='center', va='bottom', fontweight='bold', color=r_col, fontsize=9.5, bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=r_col, alpha=0.92))

            ax_lr.set_ylabel('Bee Mortality Count (Deaths)', fontsize=12, fontweight='bold', labelpad=8)
            ax_lr.set_xticks(x_indices)
            ax_lr.set_xticklabels(groups, fontsize=10.5, fontweight='bold', rotation=15, ha='right')
            ax_lr.set_ylim(0, max_y * 1.25)
            ax_lr.set_title(f'Log-Rank Test: Observed vs. Expected Deaths Under H₀ (χ² = {chisq:.3f}, p = {p_str})', fontsize=13, fontweight='bold', pad=12)
            ax_lr.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.92, fontsize=10)
            ax_lr.spines['top'].set_visible(False)
            ax_lr.spines['right'].set_visible(False)
            fig_lr.tight_layout()

            st.pyplot(fig_lr)

            buf_lr = io.BytesIO()
            fig_lr.savefig(buf_lr, format="png", dpi=300, bbox_inches='tight')
            dl_lr_k = f"dl_lr_oe_plot_{group_col}_{key_suffix}" if key_suffix else f"dl_lr_oe_plot_{group_col}"
            st.download_button("📥 Download Log-Rank Observed vs. Expected Plot (PNG)", data=buf_lr.getvalue(), file_name="logrank_observed_vs_expected.png", mime="image/png", key=dl_lr_k)
            plt.close(fig_lr)

        else:
            # Cumulative Hazard Plot: H(t) via Nelson-Aalen Estimator
            fig_hz, ax_hz = plt.subplots(figsize=(9.2, 5.2), dpi=300)
            ax_hz.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', zorder=1)

            for idx_g, g in enumerate(groups):
                m = (G == g)
                sub_t = T[m]
                sub_e = E[m]
                col = cohort_colors[idx_g]
                naf_hz = NelsonAalenFitter()
                naf_hz.fit(sub_t, event_observed=sub_e, label=g)
                naf_hz.plot_cumulative_hazard(ax=ax_hz, color=col, linewidth=2.2, ci_show=False)

            ax_hz.set_xlabel('Elapsed Observation Time (Days)', fontsize=12, fontweight='bold', labelpad=8)
            ax_hz.set_ylabel('Cumulative Hazard H(t) = -ln Ŝ(t)', fontsize=12, fontweight='bold', labelpad=8)
            ax_hz.set_title(f'Log-Rank Cumulative Hazard Curves H(t) Across Cohorts (χ² = {chisq:.3f}, p = {p_str})', fontsize=13, fontweight='bold', pad=12)
            ax_hz.spines['top'].set_visible(False)
            ax_hz.spines['right'].set_visible(False)
            ax_hz.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.92, fontsize=10)
            fig_hz.tight_layout()

            st.pyplot(fig_hz)

            buf_hz = io.BytesIO()
            fig_hz.savefig(buf_hz, format="png", dpi=300, bbox_inches='tight')
            dl_hz_k = f"dl_lr_hz_plot_{group_col}_{key_suffix}" if key_suffix else f"dl_lr_hz_plot_{group_col}"
            st.download_button("📥 Download Cumulative Hazard Plot (PNG)", data=buf_hz.getvalue(), file_name="logrank_cumulative_hazard.png", mime="image/png", key=dl_hz_k)
            plt.close(fig_hz)

        # Observed vs. Expected Data Table
        st.markdown("#### 📊 Mantel-Cox Observed vs. Expected Breakdown")
        oe_table = []
        for g in groups:
            o_c = obs_dict[g]
            e_c = exp_dict[g]
            oe_ratio = o_c / e_c if e_c > 0 else 1.0
            diff = o_c - e_c
            contrib = (diff**2) / e_c if e_c > 0 else 0.0
            oe_table.append({
                "Cohort": g,
                "Observed Deaths (O)": o_c,
                "Expected Deaths (E)": f"{e_c:.2f}",
                "O - E Deviation": f"{diff:+.2f}",
                "O / E Ratio": f"{oe_ratio:.2f}",
                "Relative Risk Status": "🟢 Protected (Fewer Deaths)" if oe_ratio < 1.0 else "🔴 Elevated Hazard (Excess Deaths)",
                "Chi² Contribution ((O-E)²/E)": f"{contrib:.3f}"
            })
        st.dataframe(pd.DataFrame(oe_table), use_container_width=True)

        # Post-Hoc Pairwise Log-Rank Comparisons (with Bonferroni Correction)
        st.markdown("#### 🔄 Post-Hoc Pairwise Log-Rank Comparisons (with Bonferroni Correction)")
        col_lr_sel, col_lr_empty = st.columns([1, 1])
        with col_lr_sel:
            lr_ref_k = f"lr_ref_{group_col}_{key_suffix}" if key_suffix else f"lr_ref_{group_col}"
            default_lr_idx = 0
            for idx_g, g_val in enumerate(groups):
                if "sucrose" in str(g_val).lower() or "control" in str(g_val).lower():
                    default_lr_idx = idx_g
                    break
            ref_baseline_lr = st.selectbox("Baseline Reference Cohort for Pairwise Comparisons:", groups, index=default_lr_idx, key=lr_ref_k)

        pairs = []
        for g in groups:
            if g != ref_baseline_lr:
                pairs.append((ref_baseline_lr, g))
        if len(groups) >= 3:
            for g1, g2 in combinations(groups, 2):
                if (g1, g2) not in pairs and (g2, g1) not in pairs:
                    pairs.append((g1, g2))

        n_comp = max(1, len(pairs))
        pairwise_data = []
        for g1, g2 in pairs:
            m1 = (G == g1)
            m2 = (G == g2)
            e1 = E[m1]
            e2 = E[m2]
            if np.sum(e1) == 0 and np.sum(e2) == 0:
                pairwise_data.append({
                    "Comparison": f"{g2} vs. {g1} (Ref)" if g1 == ref_baseline_lr else f"{g1} vs. {g2}",
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
                    "Comparison": f"{g2} vs. {g1} (Ref)" if g1 == ref_baseline_lr else f"{g1} vs. {g2}",
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
            st.success(f"**Statistically Significant Difference Detected:** The overall survival distributions differ significantly across cohorts (Mantel-Cox Log-Rank $\\chi^2 = {chisq:.3f}, df = {df_val}, p = {p_str}$). Reject the null hypothesis of equal survival hazard across treatment cohorts.")
        else:
            st.info(f"**No Statistically Significant Difference:** The survival distributions do not differ significantly at the $\\alpha = 0.05$ level (Mantel-Cox Log-Rank $\\chi^2 = {chisq:.3f}, df = {df_val}, p = {p_str}$).")

    else:
        # Single cohort metrics
        st.info(f"💡 **Single Cohort Evaluated (`{groups[0]}`):** Log-Rank hypothesis testing evaluates differences between 2 or more cohorts under $H_0: S_1(t) = S_2(t)$. Upload additional CSV files to execute multi-cohort hypothesis testing.")
        col_k1, col_k2, col_k3 = st.columns(3)
        with col_k1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{len(T)}</div><div class="kpi-label">Sample Size (N)</div></div>', unsafe_allow_html=True)
        with col_k2:
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{total_events}</div><div class="kpi-label">Observed Deaths</div></div>', unsafe_allow_html=True)
        with col_k3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{len(T) - total_events}</div><div class="kpi-label">Censored / Alive</div></div>', unsafe_allow_html=True)
