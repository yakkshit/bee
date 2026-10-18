"""
Multi-Cohort Sucrose Consumption & Feeding Kinetics Comparative Module.
Compares daily sucrose intake, day-by-day cohort feeding dynamics,
cumulative nutritional intake, and Kaplan-Meier feeding cessation kinetics (time-to-anorexia).
"""
import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test


def render_multi_consumption_analysis(processed_batches, settings=None, multi_plot_images=None):
    """
    Renders the dedicated Multi-Cohort Sucrose Consumption Comparative Analysis section.
    Includes:
    1. Daily Mean Sucrose Consumption Across All Days (Line Plot with SEM ribbons)
    2. Day-by-Day Grouped Bar Chart of Sucrose Consumption (comparing all cohorts on each day)
    3. Multi-Cohort Kaplan-Meier Feeding Cessation (Anorexia Endpoint) Step Curves
    4. Cumulative Mean Sucrose Intake Over Time
    5. Cohort × Day Sucrose Intake Heatmap Matrix
    6. Individual Bee Total Intake vs Survival Time Joint Scatter
    7. Multi-Cohort Consumption Summary Table & CSV Export
    """
    if settings is None:
        settings = {}
    if multi_plot_images is None:
        multi_plot_images = {}

    line_width = settings.get('line_width', 2.4)
    marker_size = settings.get('marker_size', 7.0)
    jitter_val = settings.get('jitter_val', 0.015)

    # Filter batches that have valid consumption data
    batches_with_cons = [b for b in processed_batches if b.get('has_consumption', False)]

    if not batches_with_cons:
        st.info("ℹ️ **Sucrose Consumption Data Not Detected**: The currently active files do not contain consumption entries (e.g. `Cons 1 (µL)`). To unlock multi-cohort feeding kinetics, upload observation sheets with consumption columns or use the default benchmark cohorts.")
        return

    st.markdown("---")
    col_mc_head, col_mc_zip = st.columns([2.6, 1.4])
    with col_mc_head:
        st.header("🍯 5. Multi-Cohort Sucrose Consumption Comparative Analysis")
        st.caption("Comparative feeding kinetics across cohorts: evaluate whether chemical treatments induced **sublethal antifeedant deterrence (anorexia)**, track day-by-day feeding dynamics, and assess the correlation between nutrient consumption and bee longevity.")
    with col_mc_zip:
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        multi_cons_zip_placeholder = st.empty()

    multi_cons_plots = {}

    # Top KPI Metrics Cards
    cohort_avg_intakes = []
    total_cohort_intakes = []
    all_bees_count = sum(b['num_bees'] for b in batches_with_cons)

    for b in batches_with_cons:
        d_mean = b.get('daily_mean_cons', np.array([]))
        b_fname = b.get('filename') or b.get('orig_name') or b.get('name', 'Dataset')
        if len(d_mean) > 0:
            cohort_avg_intakes.append((b_fname, float(np.mean(d_mean))))
        all_c = b.get('daily_all_cons', [])
        if all_c:
            total_cohort_intakes.append(float(np.sum(all_c)))

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        if cohort_avg_intakes:
            highest_cohort = max(cohort_avg_intakes, key=lambda x: x[1])
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{highest_cohort[1]:.1f} µL</div><div class="kpi-label">Highest Intake ({highest_cohort[0]})</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="kpi-card"><div class="kpi-val">N/A</div><div class="kpi-label">Highest Intake</div></div>', unsafe_allow_html=True)
    with kpi2:
        if cohort_avg_intakes:
            lowest_cohort = min(cohort_avg_intakes, key=lambda x: x[1])
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{lowest_cohort[1]:.1f} µL</div><div class="kpi-label">Lowest Intake ({lowest_cohort[0]})</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="kpi-card"><div class="kpi-val">N/A</div><div class="kpi-label">Lowest Intake</div></div>', unsafe_allow_html=True)
    with kpi3:
        if len(cohort_avg_intakes) >= 2:
            base_val = cohort_avg_intakes[0][1]
            min_val = min(x[1] for x in cohort_avg_intakes)
            deficit_pct = ((min_val - base_val) / base_val * 100.0) if base_val > 0 else 0.0
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{deficit_pct:.1f}%</div><div class="kpi-label">Max Feeding Reduction vs Baseline</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="kpi-card"><div class="kpi-val">0.0%</div><div class="kpi-label">Feeding Deficit</div></div>', unsafe_allow_html=True)
    with kpi4:
        tot_all_str = f"{sum(total_cohort_intakes):.0f} µL" if total_cohort_intakes else "N/A"
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{tot_all_str}</div><div class="kpi-label">Total Sucrose Consumed (All Cohorts)</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive scope toggle
    col_scope1, col_scope2 = st.columns([2.5, 1.5])
    with col_scope1:
        calc_scope = st.radio(
            "Calculation Scope for Daily Consumption:",
            [
                "All Bees (Cohort Overall Intake - Total Consumption / Starting N)",
                "Surviving Bees Only (Physiological Intake per Live Bee on Day d)"
            ],
            index=0,
            horizontal=True,
            key="multi_cons_scope_radio",
            help="All Bees divides total consumption on day d by initial N (captures population-level nutritional depletion). Surviving Bees Only computes mean consumption exclusively among bees confirmed alive on that day."
        )
    with col_scope2:
        show_sem_ribbon = st.checkbox("Show ± SEM Error Ribbons / Bars", value=True, key="multi_cons_sem_check")

    use_alive_only = "Surviving" in calc_scope

    # Find global day range across batches
    max_days_count = max(b['num_days'] for b in batches_with_cons)
    ref_batch = max(batches_with_cons, key=lambda b: b['num_days'])
    day_labels_ref = ref_batch.get('day_labels', [f"Day {d+1}" for d in range(max_days_count)])

    # =========================================================================
    # PLOT 5.1: DAILY MEAN SUCROSE CONSUMPTION ACROSS ALL DAYS (LINE COMPARISON)
    # =========================================================================
    st.subheader("5.1 Daily Mean Sucrose Consumption Across All Observation Days")
    st.caption("Direct time-series comparison of daily sucrose consumption (µL/bee/day) across all cohorts, with shaded ± SEM uncertainty bands.")

    fig_cons_lines, ax_cl = plt.subplots(figsize=(10.5, 6.0), dpi=300)
    ax_cl.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', zorder=1)

    legend_handles_cons = []
    num_batches = len(batches_with_cons)

    for idx, b in enumerate(batches_with_cons):
        n_days_b = b['num_days']
        x_raw = np.arange(1, n_days_b + 1)
        offset = (idx - (num_batches - 1) / 2.0) * 0.08
        x_jittered = x_raw + offset

        mean_series = b.get('daily_alive_mean_cons', b.get('daily_mean_cons')) if use_alive_only else b.get('daily_mean_cons')
        sem_series = b.get('daily_alive_sem_cons', b.get('daily_sem_cons')) if use_alive_only else b.get('daily_sem_cons')

        if len(mean_series) == 0:
            continue

        # Plot shaded SEM ribbon
        if show_sem_ribbon and len(sem_series) == len(mean_series):
            ax_cl.fill_between(
                x_jittered,
                np.clip(mean_series - sem_series, 0, None),
                mean_series + sem_series,
                color=b['color'], alpha=0.15, zorder=2
            )

        # Plot line and markers
        ax_cl.plot(
            x_jittered, mean_series,
            color=b['color'], marker=b['marker'], linestyle=b['linestyle'],
            linewidth=line_width, markersize=marker_size, markeredgecolor='black',
            zorder=4
        )

        b_fname = b.get('filename') or b.get('orig_name') or b.get('name', f"Cohort {idx+1}")
        legend_handles_cons.append(
            Line2D([0], [0], color=b['color'], marker=b['marker'], linestyle=b['linestyle'],
                   linewidth=line_width, markersize=marker_size, markeredgecolor='black',
                   label=b_fname)
        )

    ax_cl.set_xlabel('Observation Day', fontsize=12.5, fontweight='bold', labelpad=8)
    ax_cl.set_ylabel('Sucrose Consumption (µL / bee / day)', fontsize=12.5, fontweight='bold', labelpad=8)
    ax_cl.set_xticks(np.arange(1, max_days_count + 1))
    ax_cl.set_xticklabels([day_labels_ref[i] if i < len(day_labels_ref) else f"Day {i+1}" for i in range(max_days_count)], rotation=45, ha='right', fontsize=10, fontweight='bold')
    ax_cl.set_xlim(0.5, max_days_count + 0.5)
    ax_cl.set_ylim(-0.5, None)
    ax_cl.spines['top'].set_visible(False)
    ax_cl.spines['right'].set_visible(False)

    num_cols_leg = min(3, max(1, num_batches))
    ax_cl.legend(handles=legend_handles_cons, loc='lower center', bbox_to_anchor=(0.5, 1.02),
                 ncol=num_cols_leg, frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=9.0)
    fig_cons_lines.tight_layout()
    st.pyplot(fig_cons_lines)

    buf_cl = io.BytesIO()
    fig_cons_lines.savefig(buf_cl, format="png", dpi=300, bbox_inches='tight')
    multi_plot_images['daily_sucrose_consumption_comparison.png'] = buf_cl.getvalue()
    multi_cons_plots['01_multi_daily_sucrose_consumption_lines.png'] = buf_cl.getvalue()
    st.download_button("📥 Download Daily Consumption Line Chart (PNG)", data=buf_cl.getvalue(),
                       file_name="daily_sucrose_consumption_comparison.png", mime="image/png", key="dl_multi_cons_lines")
    plt.close(fig_cons_lines)

    st.markdown("---")

    # =========================================================================
    # PLOT 5.2: DAY-BY-DAY GROUPED BAR CHART OF SUCROSE CONSUMPTION
    # =========================================================================
    st.subheader("5.2 Day-by-Day Grouped Bar Chart Comparison")
    st.caption("Granular day-by-day comparison of sucrose consumption: directly compares the average intake across all cohorts on each individual experimental day.")

    fig_grouped_bars, ax_gb = plt.subplots(figsize=(11.5, 6.0), dpi=300)
    ax_gb.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', axis='y', zorder=1)

    total_bar_width = 0.82
    single_bar_width = total_bar_width / max(1, num_batches)
    day_indices = np.arange(max_days_count)

    for idx, b in enumerate(batches_with_cons):
        mean_series = b.get('daily_alive_mean_cons', b.get('daily_mean_cons')) if use_alive_only else b.get('daily_mean_cons')
        sem_series = b.get('daily_alive_sem_cons', b.get('daily_sem_cons')) if use_alive_only else b.get('daily_sem_cons')

        # Pad to max_days_count with 0 if fewer days
        padded_means = [mean_series[d] if d < len(mean_series) else 0.0 for d in range(max_days_count)]
        padded_sems = [sem_series[d] if d < len(sem_series) else 0.0 for d in range(max_days_count)]

        bar_positions = day_indices - (total_bar_width / 2.0) + (idx + 0.5) * single_bar_width

        b_fname = b.get('filename') or b.get('orig_name') or b.get('name', f"Cohort {idx+1}")
        bars = ax_gb.bar(
            bar_positions, padded_means,
            width=single_bar_width * 0.90,
            yerr=padded_sems if show_sem_ribbon else None,
            capsize=3.0,
            color=b['color'], edgecolor='black', linewidth=1.0,
            label=b_fname,
            zorder=3
        )

        # Label numbers on top if reasonable number of days
        if max_days_count <= 10:
            for bar_i, bar in enumerate(bars):
                h = bar.get_height()
                if h > 0.5:
                    ax_gb.text(bar.get_x() + bar.get_width() / 2.0, h + 0.35, f"{h:.1f}",
                               ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#1e293b')

    ax_gb.set_xlabel('Observation Day', fontsize=12.5, fontweight='bold', labelpad=8)
    ax_gb.set_ylabel('Sucrose Consumption (µL / bee)', fontsize=12.5, fontweight='bold', labelpad=8)
    ax_gb.set_xticks(day_indices)
    ax_gb.set_xticklabels([day_labels_ref[i] if i < len(day_labels_ref) else f"Day {i+1}" for i in range(max_days_count)], rotation=45, ha='right', fontsize=10, fontweight='bold')
    ax_gb.set_xlim(-0.6, max_days_count - 0.4)
    ax_gb.set_ylim(0, None)
    ax_gb.spines['top'].set_visible(False)
    ax_gb.spines['right'].set_visible(False)
    ax_gb.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=9.5)
    fig_grouped_bars.tight_layout()
    st.pyplot(fig_grouped_bars)

    buf_gb = io.BytesIO()
    fig_grouped_bars.savefig(buf_gb, format="png", dpi=300, bbox_inches='tight')
    multi_plot_images['day_by_day_grouped_bar_consumption.png'] = buf_gb.getvalue()
    multi_cons_plots['02_multi_day_by_day_grouped_bar_consumption.png'] = buf_gb.getvalue()
    st.download_button("📥 Download Day-by-Day Grouped Bar Chart (PNG)", data=buf_gb.getvalue(),
                       file_name="day_by_day_grouped_bar_consumption.png", mime="image/png", key="dl_multi_cons_bars")
    plt.close(fig_grouped_bars)

    st.markdown("---")

    col_m1, col_m2 = st.columns(2)

    # =========================================================================
    # PLOT 5.3: MULTI-COHORT KAPLAN-MEIER FEEDING CESSATION STEP CURVES
    # =========================================================================
    with col_m1:
        st.subheader("5.3 Kaplan–Meier Feeding Persistence Curves")
        st.caption("Product-limit step curves for Feeding Cessation (Anorexia Endpoint). Event = Permanent cessation of sucrose feeding ($C \\le 0$).")

        fig_km_multi_feed, ax_km_mf = plt.subplots(figsize=(8.0, 5.5), dpi=300)
        ax_km_mf.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', zorder=1)

        kmf_feed_runner = KaplanMeierFitter()
        all_durations_feed = []
        all_groups_feed = []
        all_events_feed = []
        ft50_summary = []

        for b in batches_with_cons:
            b_bees = b.get('individual_bees')
            if b_bees is not None and not b_bees.empty and 'feeding_cessation_duration' in b_bees.columns:
                dur_feed = b_bees['feeding_cessation_duration'].values
                ev_feed = b_bees['feeding_cessation_event'].values
            else:
                dur_feed = np.array([float(b['times_numeric'][-1])] * b['num_bees'])
                ev_feed = np.zeros(b['num_bees'], dtype=int)

            b_fname = b.get('filename') or b.get('orig_name') or b.get('name', f"Cohort {idx+1}")
            all_durations_feed.extend(dur_feed)
            all_groups_feed.extend([b_fname] * len(dur_feed))
            all_events_feed.extend(ev_feed)

            kmf_feed_runner.fit(dur_feed, event_observed=ev_feed, label=b_fname)
            kmf_feed_runner.plot_survival_function(ax=ax_km_mf, color=b['color'], ci_show=True, ci_alpha=0.10, linewidth=2.2)

            # Mark censored bees (active feeders at end of bioassay)
            cens_f = dur_feed[ev_feed == 0]
            if len(cens_f) > 0:
                cens_s = kmf_feed_runner.survival_function_at_times(cens_f).values
                ax_km_mf.plot(cens_f, cens_s, '+', color=b['color'], markersize=8, markeredgewidth=2)

            med_ft = kmf_feed_runner.median_survival_time_
            med_ft_str = f"{int(round(med_ft))}d" if np.isfinite(med_ft) and med_ft > 0 else "> Max Obs"
            ft50_summary.append({
                "File Name": b_fname,
                "Cohort": b.get('cohort_name', b['name']),
                "Code": b['code_name'],
                "Median Feeding Time (FT50)": med_ft_str,
                "Ceased Feeding": int(np.sum(ev_feed)),
                "Still Feeding": int(np.sum(ev_feed == 0))
            })

        # Mantel-Cox Log-Rank Test for Feeding Cessation
        lr_feed_p = np.nan
        if len(set(all_groups_feed)) > 1 and sum(all_events_feed) > 0:
            try:
                lr_res = multivariate_logrank_test(all_durations_feed, all_groups_feed, all_events_feed)
                lr_feed_p = lr_res.p_value
            except Exception:
                pass

        ax_km_mf.axhline(0.5, color='#475569', linestyle=':', linewidth=1.3, label='50% Threshold (FT50)')
        ax_km_mf.set_xlabel('Elapsed Observation Time (Days)', fontsize=11.5, fontweight='bold', labelpad=7)
        ax_km_mf.set_ylabel('Feeding Probability Ŝ_feed(t)', fontsize=11.5, fontweight='bold', labelpad=7)
        ax_km_mf.set_ylim(-0.02, 1.05)
        ax_km_mf.set_xlim(0.8, max_days_count + 0.5)
        ax_km_mf.spines['top'].set_visible(False)
        ax_km_mf.spines['right'].set_visible(False)

        if np.isfinite(lr_feed_p):
            p_text = f"p < 0.001" if lr_feed_p < 0.001 else f"p = {lr_feed_p:.4f}"
            ax_km_mf.text(0.04, 0.12, f"Log-Rank (Feeding): {p_text}", transform=ax_km_mf.transAxes,
                          fontsize=9.5, fontweight='bold', bbox=dict(facecolor='white', edgecolor='#cbd5e1', alpha=0.9))

        ax_km_mf.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=8.5)
        fig_km_multi_feed.tight_layout()
        st.pyplot(fig_km_multi_feed)

        buf_km_mf = io.BytesIO()
        fig_km_multi_feed.savefig(buf_km_mf, format="png", dpi=300, bbox_inches='tight')
        multi_plot_images['multi_cohort_kaplan_meier_feeding_cessation.png'] = buf_km_mf.getvalue()
        multi_cons_plots['03_multi_kaplan_meier_feeding_cessation_curves.png'] = buf_km_mf.getvalue()
        st.download_button("📥 Download Feeding KM Curves (PNG)", data=buf_km_mf.getvalue(),
                           file_name="multi_cohort_kaplan_meier_feeding_cessation.png", mime="image/png", key="dl_multi_km_feed")
        plt.close(fig_km_multi_feed)

    # =========================================================================
    # PLOT 5.4: CUMULATIVE MEAN SUCROSE INTAKE OVER DAYS
    # =========================================================================
    with col_m2:
        st.subheader("5.4 Cumulative Mean Sucrose Intake Over Days")
        st.caption("Cumulative average sucrose dose ingested per bee (µL) over the observation window across cohorts.")

        fig_cum_intake, ax_ci = plt.subplots(figsize=(8.0, 5.5), dpi=300)
        ax_ci.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', zorder=1)

        for b in batches_with_cons:
            cum_series = b.get('cumulative_mean_cons', np.array([]))
            if len(cum_series) == 0:
                continue
            x_pts = np.arange(1, len(cum_series) + 1)
            b_fname = b.get('filename') or b.get('orig_name') or b.get('name', f"Cohort {idx+1}")
            ax_ci.plot(
                x_pts, cum_series,
                color=b['color'], marker=b['marker'], linestyle=b['linestyle'],
                linewidth=line_width, markersize=marker_size, markeredgecolor='black',
                label=f"{b_fname} ({cum_series[-1]:.1f} µL)", zorder=3
            )

        ax_ci.set_xlabel('Observation Day', fontsize=11.5, fontweight='bold', labelpad=7)
        ax_ci.set_ylabel('Cumulative Sucrose Intake (µL / bee)', fontsize=11.5, fontweight='bold', labelpad=7)
        ax_ci.set_xticks(np.arange(1, max_days_count + 1))
        ax_ci.set_xticklabels([day_labels_ref[i] if i < len(day_labels_ref) else f"Day {i+1}" for i in range(max_days_count)], rotation=45, ha='right', fontsize=10, fontweight='bold')
        ax_ci.set_xlim(0.8, max_days_count + 0.5)
        ax_ci.set_ylim(0, None)
        ax_ci.spines['top'].set_visible(False)
        ax_ci.spines['right'].set_visible(False)
        ax_ci.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=8.5)
        fig_cum_intake.tight_layout()
        st.pyplot(fig_cum_intake)

        buf_ci = io.BytesIO()
        fig_cum_intake.savefig(buf_ci, format="png", dpi=300, bbox_inches='tight')
        multi_plot_images['cumulative_mean_sucrose_intake.png'] = buf_ci.getvalue()
        multi_cons_plots['04_multi_cumulative_mean_sucrose_intake.png'] = buf_ci.getvalue()
        st.download_button("📥 Download Cumulative Intake Chart (PNG)", data=buf_ci.getvalue(),
                           file_name="cumulative_mean_sucrose_intake.png", mime="image/png", key="dl_multi_cum_intake")
        plt.close(fig_cum_intake)

    st.markdown("<br>", unsafe_allow_html=True)
    col_m3, col_m4 = st.columns(2)

    # =========================================================================
    # PLOT 5.5: COHORT × DAY SUCROSE INTAKE HEATMAP MATRIX
    # =========================================================================
    with col_m3:
        st.subheader("5.5 Cohort × Day Sucrose Intake Heatmap Matrix")
        st.caption("2D Heatmap matrix of average daily sucrose consumption (µL/bee) across all cohorts and days.")

        fig_heat_multi, ax_hm_m = plt.subplots(figsize=(8.2, 5.2), dpi=300)
        cohort_labels_rev = [b.get('filename') or b.get('orig_name') or b.get('name', f"Cohort {i+1}") for i, b in enumerate(reversed(batches_with_cons))]
        heat_matrix = []

        for b in reversed(batches_with_cons):
            m_s = b.get('daily_alive_mean_cons', b.get('daily_mean_cons')) if use_alive_only else b.get('daily_mean_cons')
            row = [m_s[d] if d < len(m_s) else 0.0 for d in range(max_days_count)]
            heat_matrix.append(row)

        heat_arr = np.array(heat_matrix)
        vmax_val = max(np.max(heat_arr) if heat_arr.size > 0 else 1.0, 1.0)
        norm_hm = plt.Normalize(vmin=0, vmax=vmax_val)
        cmap_hm = plt.cm.YlGnBu

        for i in range(len(cohort_labels_rev)):
            for j in range(max_days_count):
                val = heat_arr[i, j]
                bg_c = cmap_hm(norm_hm(val)) if val > 0 else '#f1f5f9'
                txt_c = 'white' if norm_hm(val) > 0.55 else '#0f172a'
                ax_hm_m.add_patch(plt.Rectangle((j, i), 0.90, 0.85, color=bg_c, ec='white', lw=1.5, zorder=2))
                ax_hm_m.text(j + 0.45, i + 0.425, f"{val:.1f}", ha='center', va='center', color=txt_c, fontweight='bold', fontsize=9.0, zorder=3)

        ax_hm_m.set_xlim(-0.1, float(max_days_count))
        ax_hm_m.set_ylim(-0.1, len(cohort_labels_rev))
        ax_hm_m.set_xticks(np.arange(max_days_count) + 0.45)
        ax_hm_m.set_xticklabels([day_labels_ref[i] if i < len(day_labels_ref) else f"Day {i+1}" for i in range(max_days_count)], rotation=45, ha='right', fontsize=9.5, fontweight='bold')
        ax_hm_m.set_yticks(np.arange(len(cohort_labels_rev)) + 0.425)
        ax_hm_m.set_yticklabels(cohort_labels_rev, fontsize=10, fontweight='bold')
        ax_hm_m.spines['top'].set_visible(False)
        ax_hm_m.spines['right'].set_visible(False)
        ax_hm_m.spines['left'].set_visible(False)
        ax_hm_m.spines['bottom'].set_visible(False)
        ax_hm_m.tick_params(left=False, bottom=False)

        sm_hm = plt.cm.ScalarMappable(cmap=cmap_hm, norm=norm_hm)
        sm_hm.set_array([])
        cbar_hm = fig_heat_multi.colorbar(sm_hm, ax=ax_hm_m, orientation='horizontal', fraction=0.046, pad=0.18)
        cbar_hm.set_label('Mean Daily Sucrose Consumption (µL / bee)', fontsize=9.5, fontweight='bold')
        fig_heat_multi.tight_layout()
        st.pyplot(fig_heat_multi)

        buf_hm_m = io.BytesIO()
        fig_heat_multi.savefig(buf_hm_m, format="png", dpi=300, bbox_inches='tight')
        multi_plot_images['cohort_day_consumption_heatmap.png'] = buf_hm_m.getvalue()
        multi_cons_plots['05_multi_cohort_day_consumption_heatmap.png'] = buf_hm_m.getvalue()
        st.download_button("📥 Download Heatmap Matrix (PNG)", data=buf_hm_m.getvalue(),
                           file_name="cohort_day_consumption_heatmap.png", mime="image/png", key="dl_multi_hm")
        plt.close(fig_heat_multi)

    # =========================================================================
    # PLOT 5.6: TOTAL INTAKE VS SURVIVAL TIME JOINT SCATTER BY COHORT
    # =========================================================================
    with col_m4:
        st.subheader("5.6 Total Sucrose Intake vs. Survival Longevity")
        st.caption("Subject-level bioassay scatter: evaluates whether individual bee mortality was mediated by antifeedant starvation or direct chemical toxicity.")

        fig_scatter_longev, ax_sl = plt.subplots(figsize=(8.2, 5.2), dpi=300)
        ax_sl.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', zorder=1)

        all_tot_cons = []
        all_durations_tot = []
        scatter_legend_handles = []

        for b in batches_with_cons:
            b_bees = b.get('individual_bees')
            if b_bees is not None and not b_bees.empty and 'total_consumption' in b_bees.columns:
                t_cons = b_bees['total_consumption'].values
                t_dur = b_bees['duration'].values
                all_tot_cons.extend(t_cons)
                all_durations_tot.extend(t_dur)

                ax_sl.scatter(
                    t_cons, t_dur,
                    color=b['color'], marker=b['marker'], s=55,
                    edgecolors='black', linewidth=1.1, alpha=0.85, zorder=3
                )

                b_fname = b.get('filename') or b.get('orig_name') or b.get('name', f"Cohort {idx+1}")
                scatter_legend_handles.append(
                    Line2D([0], [0], marker=b['marker'], color='w', markerfacecolor=b['color'],
                           markeredgecolor='black', markersize=7.5, label=b_fname)
                )

        # Global regression line
        if len(all_tot_cons) > 3 and np.std(all_tot_cons) > 1e-4:
            from scipy.stats import linregress
            res_tot = linregress(all_tot_cons, all_durations_tot)
            x_line = np.linspace(min(all_tot_cons), max(all_tot_cons), 50)
            y_line = res_tot.intercept + res_tot.slope * x_line
            ax_sl.plot(x_line, y_line, color='#0f172a', linestyle='--', linewidth=1.8, zorder=2)
            scatter_legend_handles.append(
                Line2D([0], [0], color='#0f172a', linestyle='--', linewidth=1.8,
                       label=f"Fit (r = {res_tot.rvalue:.3f}, p = {res_tot.pvalue:.3f})")
            )

        ax_sl.set_xlabel('Total Sucrose Consumed (µL)', fontsize=11.5, fontweight='bold', labelpad=7)
        ax_sl.set_ylabel('Survival Duration (Days)', fontsize=11.5, fontweight='bold', labelpad=7)
        ax_sl.spines['top'].set_visible(False)
        ax_sl.spines['right'].set_visible(False)
        ax_sl.legend(handles=scatter_legend_handles, loc='upper left', frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=8.5)
        fig_scatter_longev.tight_layout()
        st.pyplot(fig_scatter_longev)

        buf_sl = io.BytesIO()
        fig_scatter_longev.savefig(buf_sl, format="png", dpi=300, bbox_inches='tight')
        multi_plot_images['total_intake_vs_longevity_scatter.png'] = buf_sl.getvalue()
        multi_cons_plots['06_multi_total_intake_vs_longevity_scatter.png'] = buf_sl.getvalue()
        st.download_button("📥 Download Intake vs Longevity Scatter (PNG)", data=buf_sl.getvalue(),
                           file_name="total_intake_vs_longevity_scatter.png", mime="image/png", key="dl_multi_scatter")
        plt.close(fig_scatter_longev)

    # =========================================================================
    # MULTI-COHORT CONSUMPTION SUMMARY TABLE & CSV EXPORT
    # =========================================================================
    st.markdown("---")
    st.subheader("📋 Multi-Cohort Sucrose Consumption Summary Table")

    summary_cons_rows = []
    for b in batches_with_cons:
        m_daily = b.get('daily_mean_cons', np.array([]))
        cum_val = b.get('cumulative_mean_cons', np.array([]))
        b_bees = b.get('individual_bees')

        n_b = b['num_bees']
        mean_intake_str = f"{np.mean(m_daily):.2f} µL" if len(m_daily) > 0 else "N/A"
        cum_intake_str = f"{cum_val[-1]:.1f} µL" if len(cum_val) > 0 else "N/A"

        # Peak day
        if len(m_daily) > 0:
            p_idx = int(np.argmax(m_daily))
            p_day = day_labels_ref[p_idx] if p_idx < len(day_labels_ref) else f"Day {p_idx+1}"
            peak_str = f"{p_day} ({m_daily[p_idx]:.1f} µL)"
        else:
            peak_str = "N/A"

        # FT50
        b_fname = b.get('filename') or b.get('orig_name') or b.get('name', f"Cohort {idx_b+1}")
        ft_info = next((item for item in ft50_summary if item.get('File Name') == b_fname or item.get('Cohort') == b['name']), None)
        ft50_str = ft_info['Median Feeding Time (FT50)'] if ft_info else "N/A"
        cess_count = ft_info['Ceased Feeding'] if ft_info else 0
        cess_rate = f"{(cess_count / n_b * 100.0):.1f}%" if n_b > 0 else "0.0%"

        summary_cons_rows.append({
            "File Name": b_fname,
            "Cohort Name": b.get('cohort_name', b['name']),
            "Code Name": b['code_name'],
            "Sample Size (N)": n_b,
            "Observation Days": b['num_days'],
            "Mean Daily Intake (µL/bee/day)": mean_intake_str,
            "Cumulative Intake / Bee": cum_intake_str,
            "Peak Consumption Day": peak_str,
            "Median Feeding Time (FT50)": ft50_str,
            "Feeding Cessation Rate (%)": cess_rate
        })

    summary_cons_df = pd.DataFrame(summary_cons_rows)
    st.dataframe(summary_cons_df, use_container_width=True)

    csv_cons_str = summary_cons_df.to_csv(index=False)
    multi_cons_plots['multi_cohort_sucrose_consumption_summary.csv'] = csv_cons_str.encode('utf-8')

    # Generate Combined Multi-Cohort Consumption ZIP Archive
    multi_zip_buf = io.BytesIO()
    with zipfile.ZipFile(multi_zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn, data in multi_cons_plots.items():
            zf.writestr(fn, data)
    multi_cons_zip_bytes = multi_zip_buf.getvalue()

    # Populate top placeholder
    multi_cons_zip_placeholder.download_button(
        label="📦 Download All Combined Consumption Plots (ZIP)",
        data=multi_cons_zip_bytes,
        file_name="combined_multi_cohort_consumption_plots.zip",
        mime="application/zip",
        key="dl_multi_cons_zip_top",
        help="Download all 6 multi-cohort comparative consumption figures (300 DPI PNG) and summary data in a single ZIP file."
    )

    col_m_dl1, col_m_dl2 = st.columns([1.6, 2.4])
    with col_m_dl1:
        st.download_button(
            "📊 Download Multi-Cohort Sucrose Consumption Summary (CSV)",
            data=csv_cons_str,
            file_name="multi_cohort_sucrose_consumption_summary.csv",
            mime="text/csv",
            key="dl_multi_cons_summary_csv"
        )
    with col_m_dl2:
        st.download_button(
            "📦 Download All Combined Consumption Plots (ZIP Archive)",
            data=multi_cons_zip_bytes,
            file_name="combined_multi_cohort_consumption_plots.zip",
            mime="application/zip",
            key="dl_multi_cons_zip_bottom",
            help="Download all 6 multi-cohort consumption comparative figures and summary metrics as a ZIP archive."
        )
