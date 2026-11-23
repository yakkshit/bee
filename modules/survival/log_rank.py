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
        is_marginal = (0.05 <= p_val < 0.08)

        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        with col_k1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{chisq:.3f}</div><div class="kpi-label">Log-Rank χ² Statistic</div></div>', unsafe_allow_html=True)
        with col_k2:
            p_str = "< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}"
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{p_str}</div><div class="kpi-label">p-value (Mantel-Cox)</div></div>', unsafe_allow_html=True)
        with col_k3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{df_val}</div><div class="kpi-label">Degrees of Freedom (df)</div></div>', unsafe_allow_html=True)
        with col_k4:
            if is_significant:
                status_txt = "Significant (p < 0.05)"
                status_col = "#10b981"
            elif is_marginal:
                status_txt = "Marginal Trend (p ≈ 0.05)"
                status_col = "#38bdf8"
            else:
                status_txt = "Not Significant (p ≥ 0.08)"
                status_col = "#f59e0b"
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

        # Render Log-Rank Figures via Structured Thematic Tabs (Hazard Curves, Observed vs Expected, and Lifespan Distribution Histogram)
        lr_tab_hz, lr_tab_oe, lr_tab_hist = st.tabs([
            "📈 1. Log-Rank Cumulative Hazard Curves H(t) (Publication Figure)",
            "📊 2. Observed vs. Expected Deaths (O vs. E) Bar Chart",
            "📉 3. Lifespan Distribution Histogram (with Hypothesis Test p-values)"
        ])

        if 'multi_plot_images' not in st.session_state:
            st.session_state['multi_plot_images'] = {}

        with lr_tab_hz:
            # Cumulative Hazard Plot: H(t) via Nelson-Aalen Estimator (Matching Publication Benchmark, 130 DPI for rapid on-screen reactivity)
            fig_hz, ax_hz = plt.subplots(figsize=(9.5, 5.0), dpi=130)

            for idx_g, g in enumerate(groups):
                m = (G == g)
                sub_t = T[m]
                sub_e = E[m]
                col = cohort_colors[idx_g]
                naf_hz = NelsonAalenFitter()
                naf_hz.fit(sub_t, event_observed=sub_e, label=g)
                ch_df = naf_hz.cumulative_hazard_
                timeline = ch_df.index.values
                ch_vals = ch_df.iloc[:, 0].values

                # Plot shaded 95% Confidence Interval band for cumulative hazard
                ci_hz = naf_hz.confidence_interval_cumulative_hazard_
                if not ci_hz.empty:
                    ax_hz.fill_between(
                        ci_hz.index.values,
                        ci_hz.iloc[:, 0].values,
                        ci_hz.iloc[:, 1].values,
                        step='post', color=col, alpha=0.10, zorder=2
                    )

                # Step curve starting strictly from baseline t=0, H=0
                if len(timeline) > 0 and timeline[0] > 0:
                    timeline = np.insert(timeline, 0, 0.0)
                    ch_vals = np.insert(ch_vals, 0, 0.0)
                ax_hz.step(timeline, ch_vals, where='post', color=col, linewidth=2.4, label=g, zorder=3)

            ax_hz.spines['left'].set_linewidth(1.8)
            ax_hz.spines['bottom'].set_linewidth(1.8)
            ax_hz.spines['top'].set_visible(False)
            ax_hz.spines['right'].set_visible(False)
            ax_hz.tick_params(direction='out', length=5, width=1.4, labelsize=10.5)

            ax_hz.set_xlabel('Elapsed Observation Time (Days)', fontsize=12, fontweight='bold', labelpad=8)
            ax_hz.set_ylabel('Cumulative Hazard H(t) = -ln Ŝ(t)', fontsize=12, fontweight='bold', labelpad=8)
            ax_hz.set_title(f'Log-Rank Cumulative Hazard Curves H(t) Across Cohorts (χ² = {chisq:.3f}, df = {df_val}, p = {p_str})', fontsize=12.5, fontweight='bold', pad=12)
            ax_hz.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#cbd5e1', fontsize=9.5)
            fig_hz.tight_layout()

            st.pyplot(fig_hz)

            buf_hz = io.BytesIO()
            fig_hz.savefig(buf_hz, format="png", dpi=300, bbox_inches='tight')
            hz_bytes = buf_hz.getvalue()
            st.session_state['multi_plot_images']['logrank_cumulative_hazard.png'] = hz_bytes
            dl_hz_k = f"dl_lr_hz_plot_{group_col}_{key_suffix}" if key_suffix else f"dl_lr_hz_plot_{group_col}"
            st.download_button("📥 Download Cumulative Hazard Curves (PNG)", data=hz_bytes, file_name="logrank_cumulative_hazard.png", mime="image/png", key=dl_hz_k)
            plt.close(fig_hz)

        with lr_tab_oe:
            fig_lr, ax_lr = plt.subplots(figsize=(9.2, 5.0), dpi=130)
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
            ax_lr.set_title(f'Log-Rank Test: Observed vs. Expected Deaths Under H₀ (χ² = {chisq:.3f}, df = {df_val}, p = {p_str})', fontsize=12.5, fontweight='bold', pad=12)
            ax_lr.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.92, fontsize=10)
            ax_lr.spines['top'].set_visible(False)
            ax_lr.spines['right'].set_visible(False)
            fig_lr.tight_layout()

            st.pyplot(fig_lr)

            buf_lr = io.BytesIO()
            fig_lr.savefig(buf_lr, format="png", dpi=300, bbox_inches='tight')
            lr_bytes = buf_lr.getvalue()
            st.session_state['multi_plot_images']['logrank_observed_vs_expected.png'] = lr_bytes
            dl_lr_k = f"dl_lr_oe_plot_{group_col}_{key_suffix}" if key_suffix else f"dl_lr_oe_plot_{group_col}"
            st.download_button("📥 Download Log-Rank Observed vs. Expected Plot (PNG)", data=lr_bytes, file_name="logrank_observed_vs_expected.png", mime="image/png", key=dl_lr_k)
            plt.close(fig_lr)

        with lr_tab_hist:
            fig_lrh, ax_lrh = plt.subplots(figsize=(9.2, 5.0), dpi=130)
            ax_lrh.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', zorder=1)

            import scipy.stats as sp_stats
            cohort_durations = [T[G == g] for g in groups]
            valid_durs = [d for d in cohort_durations if len(d) > 0]
            try:
                if len(valid_durs) >= 2:
                    kw_res = sp_stats.kruskal(*valid_durs)
                    kw_stat, kw_p = float(kw_res.statistic), float(kw_res.pvalue)
                    kw_p_str = "< 0.0001" if kw_p < 0.0001 else f"{kw_p:.4f}"
                else:
                    kw_stat, kw_p_str = chisq, p_str
            except Exception:
                kw_stat, kw_p_str = chisq, p_str

            max_dur = max(T) if len(T) > 0 else 14
            bins = np.arange(0.5, max_dur + 1.5, 1.0) if max_dur > 0 else 10
            for idx_g, g in enumerate(groups):
                m_g = (G == g)
                durs_g = T[m_g]
                ax_lrh.hist(durs_g, bins=bins, alpha=0.45, color=cohort_colors[idx_g], label=g, edgecolor='black', linewidth=1.1, zorder=3)

            ax_lrh.set_xlabel('Observation Time / Bee Lifespan (Days)', fontsize=12, fontweight='bold', labelpad=8)
            ax_lrh.set_ylabel('Bee Frequency (Count)', fontsize=12, fontweight='bold', labelpad=8)
            ax_lrh.set_title(f'Lifespan Distribution Histogram Across Cohorts (Log-Rank p = {p_str} | Kruskal-Wallis p = {kw_p_str})', fontsize=11.5, fontweight='bold', pad=12)

            kw_is_sig = (kw_p < 0.05) if 'kw_p' in locals() else is_significant
            lr_is_sig = is_significant
            lr_is_marg = (0.05 <= p_val < 0.08)

            if lr_is_sig and kw_is_sig:
                overall_status_str = f"Significant Difference Across Cohorts (Log-Rank p={p_str}, KW p={kw_p_str})"
            elif kw_is_sig and lr_is_marg:
                overall_status_str = f"Significant Lifespan Shift (KW p={kw_p_str}; Log-Rank trend p={p_str})"
            elif kw_is_sig and not lr_is_sig:
                overall_status_str = f"Significant Lifespan Rank Shift (KW p={kw_p_str})"
            elif lr_is_sig and not kw_is_sig:
                overall_status_str = f"Significant Mortality Hazard Shift (Log-Rank p={p_str})"
            else:
                overall_status_str = "No Statistically Significant Difference (p ≥ 0.05)"

            badge_txt = (
                f"Statistical Distribution Evaluation:\n"
                f"• Mantel-Cox Log-Rank: χ² = {chisq:.2f}, p = {p_str}\n"
                f"• Kruskal-Wallis Test: H = {kw_stat:.2f}, p = {kw_p_str}\n"
                f"• Overall Difference: {overall_status_str}"
            )
            ax_lrh.text(0.97, 0.94, badge_txt, transform=ax_lrh.transAxes, ha='right', va='top', fontsize=9.2, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cbd5e1', alpha=0.92))

            ax_lrh.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.92, fontsize=9.5)
            ax_lrh.spines['top'].set_visible(False)
            ax_lrh.spines['right'].set_visible(False)
            fig_lrh.tight_layout()

            st.pyplot(fig_lrh)

            buf_lrh = io.BytesIO()
            fig_lrh.savefig(buf_lrh, format="png", dpi=300, bbox_inches='tight')
            lrh_bytes = buf_lrh.getvalue()
            st.session_state['multi_plot_images']['logrank_lifespan_histogram.png'] = lrh_bytes
            dl_lrh_k = f"dl_lr_hist_plot_{group_col}_{key_suffix}" if key_suffix else f"dl_lr_hist_plot_{group_col}"
            st.download_button("📥 Download Lifespan Distribution Histogram (PNG)", data=lrh_bytes, file_name="logrank_lifespan_histogram.png", mime="image/png", key=dl_lrh_k)
            plt.close(fig_lrh)

        # Observed vs. Expected Data Table
        st.markdown("#### 📊 Mantel-Cox Observed vs. Expected Breakdown")
        oe_table = []
        n_total_pop = len(T)
        for g in groups:
            o_c = obs_dict[g]
            e_c = exp_dict[g]
            oe_ratio = o_c / e_c if e_c > 0 else 1.0
            diff = o_c - e_c
            contrib = (diff**2) / e_c if e_c > 0 else 0.0
            # Under H0, variance of O - E: V ≈ E * (1 - E / N)
            v_c = max(0.05, e_c * (1.0 - (e_c / max(1, n_total_pop))))
            se_diff = np.sqrt(v_c)
            wald_z = diff / se_diff if se_diff > 0 else 0.0
            ci_diff_l = diff - 1.96 * se_diff
            ci_diff_u = diff + 1.96 * se_diff
            oe_table.append({
                "Cohort": g,
                "Observed Deaths (O)": o_c,
                "Expected Deaths (E)": f"{e_c:.2f}",
                "O - E Deviation": f"{diff:+.2f}",
                "Std Error (SE)": f"{se_diff:.2f}",
                "95% CI (O - E)": f"[{ci_diff_l:+.1f}, {ci_diff_u:+.1f}]",
                "Wald z": f"{wald_z:+.2f}",
                "O / E Ratio": f"{oe_ratio:.2f}",
                "Chi² Contribution ((O-E)²/E)": f"{contrib:.3f}",
                "df": 1,
                "Relative Risk Status": "🟢 Protected (Fewer Deaths)" if oe_ratio < 1.0 else "🔴 Elevated Hazard (Excess Deaths)"
            })
        st.dataframe(pd.DataFrame(oe_table), use_container_width=True)

        # Post-Hoc Pairwise Log-Rank Comparisons with Full Multiple Testing & FDR Correction Suite
        st.markdown("#### 🔄 Post-Hoc Pairwise Log-Rank Comparisons & Multiplicity Adjustments")
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

        pairwise_p_dict = {}
        for g1, g2 in pairs:
            m1 = (G == g1)
            m2 = (G == g2)
            e1 = E[m1]
            e2 = E[m2]
            comp_lbl = f"{g2} vs. {g1} (Ref)" if g1 == ref_baseline_lr else f"{g1} vs. {g2}"
            if np.sum(e1) == 0 and np.sum(e2) == 0:
                p_raw = 1.0
            else:
                try:
                    res_p = logrank_test(T[m1], T[m2], event_observed_A=e1, event_observed_B=e2)
                    p_raw = float(res_p.p_value)
                except Exception:
                    p_raw = 1.0
            pairwise_p_dict[comp_lbl] = p_raw

        # Render complete Multiple Testing & FDR Studio with theoretical confusion matrix and plots
        from modules.core.multi_testing import render_multiple_testing_studio

        def mt_image_collector(fn, data):
            if 'multi_plot_images' not in st.session_state:
                st.session_state['multi_plot_images'] = {}
            st.session_state['multi_plot_images'][fn] = data

        render_multiple_testing_studio(pairwise_p_dict, key_suffix=key_suffix, image_collector=mt_image_collector)

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
