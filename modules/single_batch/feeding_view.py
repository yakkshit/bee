"""
Tab 3: Cohort & Individual Tracking & Nutrient Consumption.
Renders individual bee health status matrix heatmaps, population composition stackplots,
daily status distributions, cumulative mortality, and per-bee feed consumption trajectories.
"""
import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def render_single_feeding_view(computed, df, settings=None, image_collector=None):
    """Renders the Cohort & Individual Tracking sub-tab."""
    if settings is None:
        settings = {}

    primary_color = settings.get('primary_color', '#ef4444')
    secondary_color = settings.get('secondary_color', '#f59e0b')
    line_width = settings.get('line_width', 2.4)
    marker_size = settings.get('marker_size', 7.0)
    hm_alive_color = settings.get('hm_alive_color', '#10b981')
    hm_behavior_color = settings.get('hm_behavior_color', '#f59e0b')
    hm_dead_color = settings.get('hm_dead_color', '#ef4444')

    c = computed
    num_days = c['num_days']
    num_bees = c['num_bees']
    elapsed_days = c['elapsed_days']
    day_labels = c['day_labels']
    max_time = c['max_time']
    states_matrix = c['states_matrix']
    alive_counts = c['alive_counts']
    dead_counts = c['dead_counts']
    cons_matrix = c['cons_matrix']
    cons_cols = c['cons_cols']

    st.markdown("#### Individual Bee Observation Matrix & Population Distributions")
    st.caption("Granular tracking of individual bee health transitions (Alive, Behavioral change, Dead), cumulative mortality trajectory, and daily nutrient consumption.")

    c_b1, c_b2 = st.columns(2)

    if num_days > 0 and num_bees > 0:
        with c_b1:
            st.subheader("Individual Bee Status Matrix Heatmap")
            fig4, ax = plt.subplots(figsize=(8.0, 5.8), dpi=300)
            state_num_map = {'A': 1, 'B': 2, 'D': 3}
            color_map_hm = {1: hm_alive_color, 2: hm_behavior_color, 3: hm_dead_color}
            mat = np.zeros((num_bees, num_days))
            for i, b_states in enumerate(states_matrix):
                for j, s in enumerate(b_states):
                    mat[i, j] = state_num_map.get(str(s).strip(), 1)
            for i in range(num_bees):
                for j in range(num_days):
                    val = mat[i, j]
                    col = color_map_hm.get(val, '#cccccc')
                    ax.add_patch(plt.Rectangle((j, num_bees - 1 - i), 0.90, 0.85, color=col, ec='white', lw=1.5, zorder=2))
                    ax.text(j + 0.45, num_bees - 1 - i + 0.425, states_matrix[i][j], ha='center', va='center', color='white', fontweight='bold', fontsize=10.5, zorder=3)
            ax.set_xlim(-0.1, float(num_days))
            ax.set_ylim(-0.1, num_bees)
            ax.set_xticks(np.arange(num_days) + 0.45)
            ax.set_xticklabels(day_labels, fontsize=10.5, fontweight='bold', rotation=45, ha='right')
            ax.set_yticks(np.arange(num_bees) + 0.425)
            ax.set_yticklabels([f'Bee {num_bees - i}' for i in range(num_bees)], fontsize=10, fontweight='bold')
            ax.set_xlabel('Observation Day', fontsize=12, fontweight='bold', labelpad=7)
            ax.set_ylabel('Bee ID', fontsize=12, fontweight='bold', labelpad=7)

            legend_elements = [
                Patch(facecolor=hm_alive_color, edgecolor='none', label='Alive (A)'),
                Patch(facecolor=hm_behavior_color, edgecolor='none', label='Behavior Change (B)'),
                Patch(facecolor=hm_dead_color, edgecolor='none', label='Dead (D)')
            ]
            ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, frameon=False, fontsize=9.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(left=False, bottom=False)
            fig4.tight_layout()
            st.pyplot(fig4)
            if image_collector:
                image_collector(fig4, 'fig_individual_bees_status.png', 'Download Status Heatmap', 't3_hm')

        with c_b2:
            st.subheader("Population Status Breakdown")
            behavior_counts = [num_bees - a - d for a, d in zip(alive_counts, dead_counts)]
            fig_area, ax_area = plt.subplots(figsize=(7.0, 5.2), dpi=300)
            ax_area.stackplot(elapsed_days, dead_counts, behavior_counts, alive_counts,
                              labels=['Dead', 'Behavior Change', 'Healthy / Alive'],
                              colors=[hm_dead_color, hm_behavior_color, hm_alive_color], alpha=0.85)
            ax_area.set_xlabel('Observation Time / Days', fontsize=12, fontweight='bold', labelpad=7)
            ax_area.set_ylabel('Number of Bees', fontsize=12, fontweight='bold', labelpad=7)
            ax_area.set_xlim(-0.1, max_time * 1.05)
            ax_area.set_xticks(elapsed_days)
            ax_area.set_xticklabels(day_labels, rotation=45, ha='right')
            ax_area.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none', fontsize=9.5)
            ax_area.spines['top'].set_visible(False)
            ax_area.spines['right'].set_visible(False)
            fig_area.tight_layout()
            st.pyplot(fig_area)
            if image_collector:
                image_collector(fig_area, 'fig_status_breakdown.png', 'Download Status Breakdown', 't3_area')

        with c_b1:
            st.subheader("Daily Status Distribution")
            fig_bar, ax_bar = plt.subplots(figsize=(7.0, 5.0), dpi=300)
            x_pos = np.arange(num_days)
            bar_w = 0.35
            ax_bar.bar(x_pos - bar_w / 2, alive_counts, bar_w, label='Alive', color=secondary_color)
            ax_bar.bar(x_pos + bar_w / 2, dead_counts, bar_w, label='Dead', color=primary_color)
            ax_bar.set_xlabel('Observation Day', fontsize=12, fontweight='bold', labelpad=7)
            ax_bar.set_ylabel('Bee Count', fontsize=12, fontweight='bold', labelpad=7)
            ax_bar.set_xticks(x_pos)
            ax_bar.set_xticklabels(day_labels, rotation=45, ha='right')
            ax_bar.legend(loc='upper right', frameon=False, fontsize=9.5)
            ax_bar.spines['top'].set_visible(False)
            ax_bar.spines['right'].set_visible(False)
            fig_bar.tight_layout()
            st.pyplot(fig_bar)
            if image_collector:
                image_collector(fig_bar, 'fig_daily_status_bar.png', 'Download Daily Status Chart', 't3_bar')

        with c_b2:
            st.subheader("Cumulative Mortality Progression")
            fig_cum, ax_cum = plt.subplots(figsize=(7.0, 5.0), dpi=300)
            ax_cum.plot(elapsed_days, dead_counts, marker='o', color=primary_color, linewidth=line_width, markersize=marker_size, label='Cumulative Mortality')
            ax_cum.set_xlabel('Observation Time / Days', fontsize=12, fontweight='bold', labelpad=7)
            ax_cum.set_ylabel('Cumulative Dead Bees', fontsize=12, fontweight='bold', labelpad=7)
            ax_cum.set_xlim(-0.1, max_time * 1.05)
            ax_cum.set_xticks(elapsed_days)
            ax_cum.set_xticklabels(day_labels, rotation=45, ha='right')
            ax_cum.spines['top'].set_visible(False)
            ax_cum.spines['right'].set_visible(False)
            fig_cum.tight_layout()
            st.pyplot(fig_cum)
            if image_collector:
                image_collector(fig_cum, 'fig_cumulative_mortality.png', 'Download Cumulative Mortality Chart', 't3_cum')

    # ----------------------------------------------------
    # Comprehensive Sucrose Consumption & Feeding Kinetics
    # ----------------------------------------------------
    has_cons = c.get('has_consumption', False)
    if has_cons and len(cons_matrix) > 0:
        st.markdown("---")
        col_c_head, col_c_zip = st.columns([2.6, 1.4])
        with col_c_head:
            st.header("🍯 Sucrose Consumption & Feeding Kinetics Laboratory")
            st.caption("Granular assessment of nutritional intake: track per-bee daily consumption, daily mean consumption across all bees, individual cumulative feeding trajectories, and feeding cessation kinetics (nutritional anorexia endpoint).")
        with col_c_zip:
            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
            single_cons_zip_placeholder = st.empty()

        # Collector specifically for Single-CSV consumption figures & summary table
        single_cons_zip_dict = {}

        def record_cons_plot(fig, filename_zip, filename_orig, btn_label, key_suffix):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            img_data = buf.getvalue()
            single_cons_zip_dict[filename_zip] = img_data
            if image_collector:
                image_collector(fig, filename_orig, btn_label, key_suffix)
            else:
                st.download_button(label=f"📥 {btn_label}", data=img_data, file_name=filename_orig, mime="image/png", key=f"dl_{filename_orig}_{key_suffix}")
                plt.close(fig)

        daily_mean_cons = c.get('daily_mean_cons', np.array([]))
        daily_sem_cons = c.get('daily_sem_cons', np.array([]))
        daily_alive_mean_cons = c.get('daily_alive_mean_cons', np.array([]))
        daily_alive_sem_cons = c.get('daily_alive_sem_cons', np.array([]))
        daily_all_cons = c.get('daily_all_cons', [])
        cumulative_mean_cons = c.get('cumulative_mean_cons', np.array([]))
        cumulative_cons_matrix = c.get('cumulative_cons_matrix', [])
        feeding_durations = c.get('feeding_durations', np.array([]))
        feeding_events = c.get('feeding_events', np.array([]))
        kmf_feeding = c.get('kmf_feeding', None)
        mean_cons_per_bee = c.get('mean_cons_per_bee', [])
        total_cons_per_bee = c.get('total_cons_per_bee', [])
        cons_corr_slope = c.get('cons_corr_slope', 0.0)
        cons_corr_intercept = c.get('cons_corr_intercept', 0.0)
        cons_corr_r = c.get('cons_corr_r', 0.0)
        cons_corr_p = c.get('cons_corr_p', 1.0)
        durations = c.get('durations', np.array([]))

        # KPI Summary Cards
        kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
        with kpi_c1:
            overall_avg = f"{np.mean(daily_mean_cons):.1f} µL" if len(daily_mean_cons) > 0 else "N/A"
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{overall_avg}</div><div class="kpi-label">Avg Daily Intake / Bee</div></div>', unsafe_allow_html=True)
        with kpi_c2:
            tot_intake = f"{np.sum(daily_all_cons):.0f} µL" if daily_all_cons else "N/A"
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{tot_intake}</div><div class="kpi-label">Total Cohort Intake</div></div>', unsafe_allow_html=True)
        with kpi_c3:
            peak_day_idx = int(np.argmax(daily_mean_cons)) if len(daily_mean_cons) > 0 else 0
            peak_day_str = day_labels[peak_day_idx] if peak_day_idx < len(day_labels) else f"Day {peak_day_idx+1}"
            peak_val = f"{daily_mean_cons[peak_day_idx]:.1f} µL" if len(daily_mean_cons) > 0 else "N/A"
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{peak_day_str}</div><div class="kpi-label">Peak Intake ({peak_val})</div></div>', unsafe_allow_html=True)
        with kpi_c4:
            if kmf_feeding and hasattr(kmf_feeding, 'median_survival_time_'):
                med_feed = kmf_feeding.median_survival_time_
                med_feed_str = f"{int(round(med_feed))}d" if np.isfinite(med_feed) and med_feed > 0 else "> Max Obs"
            else:
                med_feed_str = "N/A"
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{med_feed_str}</div><div class="kpi-label">Median Feeding (FT₅₀)</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_feed1, col_feed2 = st.columns(2)

        # ---------------------------------------------
        # Plot A: Individual Bee Sucrose Consumption Heatmap Matrix
        # ---------------------------------------------
        with col_feed1:
            st.subheader("A. Individual Bee Sucrose Consumption Heatmap (µL)")
            fig_hm_c, ax_hm_c = plt.subplots(figsize=(8.2, 5.8), dpi=300)
            mat_cons = np.array(cons_matrix)
            max_c_val = max(np.max(mat_cons) if mat_cons.size > 0 else 1.0, 1.0)

            cmap_c = plt.cm.YlGnBu
            norm_c = plt.Normalize(vmin=0, vmax=max_c_val)

            for i in range(num_bees):
                for j in range(num_days):
                    val = mat_cons[i, j]
                    if val > 0:
                        bg_col = cmap_c(norm_c(val))
                        txt_col = 'white' if norm_c(val) > 0.55 else '#0f172a'
                        txt_str = f"{val:g}"
                    else:
                        bg_col = '#f1f5f9'
                        txt_col = '#94a3b8'
                        txt_str = "0"
                    ax_hm_c.add_patch(plt.Rectangle((j, num_bees - 1 - i), 0.90, 0.85, color=bg_col, ec='white', lw=1.5, zorder=2))
                    ax_hm_c.text(j + 0.45, num_bees - 1 - i + 0.425, txt_str, ha='center', va='center', color=txt_col, fontweight='bold', fontsize=9.0, zorder=3)

            ax_hm_c.set_xlim(-0.1, float(num_days))
            ax_hm_c.set_ylim(-0.1, num_bees)
            ax_hm_c.set_xticks(np.arange(num_days) + 0.45)
            ax_hm_c.set_xticklabels(day_labels, fontsize=10, fontweight='bold', rotation=45, ha='right')
            ax_hm_c.set_yticks(np.arange(num_bees) + 0.425)
            ax_hm_c.set_yticklabels([f'Bee {num_bees - i}' for i in range(num_bees)], fontsize=9.5, fontweight='bold')
            ax_hm_c.set_xlabel('Observation Day', fontsize=11.5, fontweight='bold', labelpad=7)
            ax_hm_c.set_ylabel('Bee ID', fontsize=11.5, fontweight='bold', labelpad=7)
            ax_hm_c.spines['top'].set_visible(False)
            ax_hm_c.spines['right'].set_visible(False)
            ax_hm_c.spines['left'].set_visible(False)
            ax_hm_c.spines['bottom'].set_visible(False)
            ax_hm_c.tick_params(left=False, bottom=False)

            sm = plt.cm.ScalarMappable(cmap=cmap_c, norm=norm_c)
            sm.set_array([])
            cbar = fig_hm_c.colorbar(sm, ax=ax_hm_c, orientation='horizontal', fraction=0.046, pad=0.18)
            cbar.set_label('Sucrose Consumption (µL / day)', fontsize=10, fontweight='bold')
            fig_hm_c.tight_layout()
            st.pyplot(fig_hm_c)
            record_cons_plot(fig_hm_c, '01_single_bee_consumption_heatmap.png', 'fig_bee_consumption_heatmap.png', 'Download Consumption Heatmap', 't3_chm')

        # ---------------------------------------------
        # Plot B: Daily Average Sucrose Consumption Across All Bees
        # ---------------------------------------------
        with col_feed2:
            st.subheader("B. Daily Average Sucrose Consumption (µL/bee/day)")
            calc_mode = st.radio(
                "Calculation Population Scope:",
                ["All Bees (Cohort Overall Intake)", "Surviving Bees Only (Physiological Intake per Live Bee)"],
                index=0,
                key="single_cons_scope_radio",
                horizontal=True
            )
            use_alive = "Surviving" in calc_mode
            active_mean = daily_alive_mean_cons if use_alive else daily_mean_cons
            active_sem = daily_alive_sem_cons if use_alive else daily_sem_cons

            fig_avg, ax_avg = plt.subplots(figsize=(8.2, 5.8), dpi=300)
            ax_avg.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', zorder=1)

            x_idx = np.arange(num_days)
            rng = np.random.RandomState(42)
            for j in range(num_days):
                day_vals = [cons_matrix[i][j] for i in range(num_bees)]
                if use_alive:
                    alive_b = [i for i in range(num_bees) if str(states_matrix[i][j]).strip().upper() != 'D']
                    day_vals = [cons_matrix[i][j] for i in alive_b]
                if day_vals:
                    jitter = rng.normal(0, 0.08, size=len(day_vals))
                    ax_avg.scatter(x_idx[j] + jitter, day_vals, color='#94a3b8', alpha=0.55, s=30, zorder=2)

            ax_avg.fill_between(x_idx, np.clip(active_mean - active_sem, 0, None), active_mean + active_sem,
                                color='#0284c7', alpha=0.18, zorder=3, label='± SEM Band')
            ax_avg.plot(x_idx, active_mean, marker='o', color='#0284c7', linewidth=2.5, markersize=7.5,
                        markeredgecolor='black', zorder=4, label='Mean Daily Intake')

            for j, val in enumerate(active_mean):
                ax_avg.text(x_idx[j], val + (max(active_mean)*0.04 if max(active_mean)>0 else 0.5), f"{val:.1f}",
                            ha='center', va='bottom', fontsize=9, fontweight='bold', color='#0369a1')

            ax_avg.set_xticks(x_idx)
            ax_avg.set_xticklabels(day_labels, rotation=45, ha='right', fontsize=10, fontweight='bold')
            ax_avg.set_xlabel('Observation Day', fontsize=11.5, fontweight='bold', labelpad=7)
            ax_avg.set_ylabel('Sucrose Consumption (µL / bee)', fontsize=11.5, fontweight='bold', labelpad=7)
            ax_avg.set_ylim(-0.5, max(np.max(active_mean) * 1.25 if len(active_mean) > 0 else 5.0, 5.0))
            ax_avg.spines['top'].set_visible(False)
            ax_avg.spines['right'].set_visible(False)
            ax_avg.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=9.5)
            fig_avg.tight_layout()
            st.pyplot(fig_avg)
            record_cons_plot(fig_avg, '02_single_daily_average_consumption.png', 'fig_daily_average_consumption.png', 'Download Daily Average Intake Chart', 't3_avg')

        st.markdown("<br>", unsafe_allow_html=True)
        col_feed3, col_feed4 = st.columns(2)

        # ---------------------------------------------
        # Plot C: Individual Bee Cumulative Sucrose Consumption
        # ---------------------------------------------
        with col_feed3:
            st.subheader("C. Individual Bee Cumulative Intake (µL)")
            fig_cum_c, ax_cum_c = plt.subplots(figsize=(8.2, 5.6), dpi=300)
            ax_cum_c.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', zorder=1)

            colors_bees = plt.cm.tab20(np.linspace(0, 1, max(1, num_bees)))
            for i in range(num_bees):
                bee_label = f"Bee {i+1}"
                cum_trajectory = cumulative_cons_matrix[i] if i < len(cumulative_cons_matrix) else np.cumsum(cons_matrix[i])
                ax_cum_c.plot(x_idx, cum_trajectory, marker='o', linewidth=1.8, markersize=4.5,
                              label=bee_label, color=colors_bees[i], alpha=0.85)

            ax_cum_c.plot(x_idx, cumulative_mean_cons, color='#0f172a', linestyle='--', linewidth=3.0,
                          label='Cohort Mean Cumulative', zorder=5)

            ax_cum_c.set_xticks(x_idx)
            ax_cum_c.set_xticklabels(day_labels, rotation=45, ha='right', fontsize=10, fontweight='bold')
            ax_cum_c.set_xlabel('Observation Day', fontsize=11.5, fontweight='bold', labelpad=7)
            ax_cum_c.set_ylabel('Cumulative Sucrose Intake (µL)', fontsize=11.5, fontweight='bold', labelpad=7)
            ax_cum_c.spines['top'].set_visible(False)
            ax_cum_c.spines['right'].set_visible(False)
            ax_cum_c.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=8.5)
            fig_cum_c.tight_layout()
            st.pyplot(fig_cum_c)
            record_cons_plot(fig_cum_c, '03_single_cumulative_intake_per_bee.png', 'fig_cumulative_intake_per_bee.png', 'Download Cumulative Intake Chart', 't3_cum_bee')

        # ---------------------------------------------
        # Plot D: Kaplan-Meier Style Feeding Cessation Curve
        # ---------------------------------------------
        with col_feed4:
            st.subheader("D. Kaplan–Meier Feeding Persistence Curve")
            st.caption("Step-down non-parametric product-limit estimator for Feeding Cessation (Anorexia Endpoint). Event = Permanent cessation of sucrose feeding ($C \\le 0$).")
            fig_km_f, ax_km_f = plt.subplots(figsize=(8.2, 5.6), dpi=300)
            ax_km_f.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', zorder=1)

            if kmf_feeding and len(feeding_durations) > 0:
                timeline_f = kmf_feeding.survival_function_.index
                vals_f = kmf_feeding.survival_function_['KM_estimate'].values
                ci_f = kmf_feeding.confidence_interval_survival_function_

                ax_km_f.step(timeline_f, vals_f, where='post', color='#059669', linewidth=2.4, label='Feeding Persistence')
                ax_km_f.fill_between(timeline_f, ci_f.iloc[:, 0], ci_f.iloc[:, 1], step='post', color='#059669', alpha=0.15, label='95% Greenwood CI')

                cens_f = feeding_durations[feeding_events == 0]
                if len(cens_f) > 0:
                    cens_s = [kmf_feeding.survival_function_at_times(d).values[0] for d in cens_f]
                    ax_km_f.plot(cens_f, cens_s, '+', color='#0f172a', markersize=8.5, markeredgewidth=2, label=f'Active at End (n={len(cens_f)})')

                ax_km_f.axhline(0.5, color='#475569', linestyle=':', linewidth=1.3, label='50% Threshold (FT50)')

                med_f = kmf_feeding.median_survival_time_
                if np.isfinite(med_f) and med_f > 0:
                    ax_km_f.axvline(med_f, color='#059669', linestyle='--', linewidth=1.3)
                    ax_km_f.text(med_f + 0.15, 0.52, f"FT50 = {int(round(med_f))} Days", color='#059669', fontweight='bold', fontsize=9.5)

            ax_km_f.set_xlabel('Elapsed Observation Time (Days)', fontsize=11.5, fontweight='bold', labelpad=7)
            ax_km_f.set_ylabel('Feeding Probability Ŝ_feed(t)', fontsize=11.5, fontweight='bold', labelpad=7)
            ax_km_f.set_ylim(-0.02, 1.05)
            ax_km_f.set_xlim(0.8, max_time * 1.05)
            ax_km_f.spines['top'].set_visible(False)
            ax_km_f.spines['right'].set_visible(False)
            ax_km_f.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=9.0)
            fig_km_f.tight_layout()
            st.pyplot(fig_km_f)
            record_cons_plot(fig_km_f, '04_single_feeding_cessation_kaplan_meier.png', 'fig_feeding_cessation_kaplan_meier.png', 'Download Feeding Cessation KM Curve', 't3_km_feed')

        # ---------------------------------------------
        # Plot E: Sucrose Consumption vs Survival Duration Correlation
        # ---------------------------------------------
        st.markdown("---")
        st.subheader("E. Sucrose Consumption vs. Survival Duration Correlation")
        st.caption("Examines whether individual daily sucrose intake correlates with life expectancy. Points represent individual bees.")

        fig_corr, ax_corr = plt.subplots(figsize=(9.2, 5.0), dpi=300)
        ax_corr.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', zorder=1)

        valid_bees_idx = [i for i in range(len(mean_cons_per_bee)) if mean_cons_per_bee[i] > 0]
        x_pts = np.array([mean_cons_per_bee[i] for i in valid_bees_idx])
        y_pts = np.array([durations[i] for i in valid_bees_idx])
        ev_pts = np.array([c['events'][i] for i in valid_bees_idx])

        scatter_colors = [primary_color if ev == 1 else '#10b981' for ev in ev_pts]
        ax_corr.scatter(x_pts, y_pts, c=scatter_colors, s=70, edgecolors='black', linewidth=1.2, zorder=3)

        for i_pt, b_idx in enumerate(valid_bees_idx):
            ax_corr.annotate(f"Bee {b_idx+1}", (x_pts[i_pt], y_pts[i_pt]), textcoords="offset points",
                             xytext=(5, 5), fontsize=8.5, color='#475569')

        if len(x_pts) > 2 and np.std(x_pts) > 1e-5:
            x_trend = np.linspace(min(x_pts) * 0.9, max(x_pts) * 1.1, 50)
            y_trend = cons_corr_intercept + cons_corr_slope * x_trend
            ax_corr.plot(x_trend, y_trend, color='#0f172a', linestyle='-', linewidth=2.0, zorder=2,
                         label=f'Linear Fit: Y = {cons_corr_intercept:.1f} + {cons_corr_slope:.2f}X (r = {cons_corr_r:.3f}, p = {cons_corr_p:.3f})')

        ax_corr.set_xlabel('Mean Daily Sucrose Consumption (µL / day)', fontsize=11.5, fontweight='bold', labelpad=7)
        ax_corr.set_ylabel('Survival Duration (Days)', fontsize=11.5, fontweight='bold', labelpad=7)
        ax_corr.spines['top'].set_visible(False)
        ax_corr.spines['right'].set_visible(False)

        legend_elems_corr = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=primary_color, markeredgecolor='black', markersize=8, label='Deceased (Dead)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#10b981', markeredgecolor='black', markersize=8, label='Alive / Censored'),
            Line2D([0], [0], color='#0f172a', lw=2, label=f'Trendline (R² = {cons_corr_r**2:.3f}, p = {cons_corr_p:.3f})')
        ]
        ax_corr.legend(handles=legend_elems_corr, loc='upper left', frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=9.0)
        fig_corr.tight_layout()
        st.pyplot(fig_corr)
        record_cons_plot(fig_corr, '05_single_consumption_survival_correlation.png', 'fig_consumption_survival_correlation.png', 'Download Consumption vs Survival Correlation', 't3_corr')

        # Add per-bee consumption summary data to ZIP package
        summary_bee_cons_df = pd.DataFrame({
            'Bee_ID': [f"Bee {i+1}" for i in range(num_bees)],
            'Total_Sucrose_Intake_uL': [float(total_cons_per_bee[i]) if i < len(total_cons_per_bee) else 0.0 for i in range(num_bees)],
            'Mean_Daily_Intake_uL': [float(mean_cons_per_bee[i]) if i < len(mean_cons_per_bee) else 0.0 for i in range(num_bees)],
            'Feeding_Duration_Days': [float(feeding_durations[i]) if i < len(feeding_durations) else 0.0 for i in range(num_bees)],
            'Feeding_Cessation_Event': [int(feeding_events[i]) if i < len(feeding_events) else 0 for i in range(num_bees)],
            'Survival_Duration_Days': [float(durations[i]) if i < len(durations) else 0.0 for i in range(num_bees)],
            'Final_Status': [str(states_matrix[i][-1]) if states_matrix and i < len(states_matrix) else 'D' for i in range(num_bees)]
        })
        single_cons_zip_dict['single_bee_consumption_summary.csv'] = summary_bee_cons_df.to_csv(index=False).encode('utf-8')

        # Generate Single-CSV Consumption ZIP Archive
        single_zip_buf = io.BytesIO()
        with zipfile.ZipFile(single_zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fn, data in single_cons_zip_dict.items():
                zf.writestr(fn, data)
        single_cons_zip_bytes = single_zip_buf.getvalue()

        # Populate top and bottom download triggers
        single_cons_zip_placeholder.download_button(
            label="📦 Download All Single Consumption Plots (ZIP)",
            data=single_cons_zip_bytes,
            file_name="single_csv_consumption_plots.zip",
            mime="application/zip",
            key="dl_single_cons_zip_top",
            help="Download all 5 single-batch consumption plots (300 DPI PNG) and summary table in a single ZIP file."
        )

        st.markdown("---")
        col_s_dl1, col_s_dl2 = st.columns([1.6, 2.4])
        with col_s_dl1:
            st.download_button(
                label="📦 Download All Single-CSV Consumption Plots (ZIP Archive)",
                data=single_cons_zip_bytes,
                file_name="single_csv_consumption_plots.zip",
                mime="application/zip",
                key="dl_single_cons_zip_bottom",
                help="Download all 5 single-batch consumption plots and consumption metrics table as a ZIP archive."
            )
        with col_s_dl2:
            st.info("💡 **Single-Batch Consumption ZIP Archive**: Includes all 5 consumption figures (Heatmap, Daily Mean with SEM, Cumulative Intake, Feeding KM curve, Longevity correlation) at 300 DPI publication resolution plus the per-bee feeding metrics CSV.")
    else:
        st.info("ℹ️ No sucrose consumption columns (e.g. `Cons 1 (µL)`) detected. To unlock granular feeding tracking and feeding cessation models, ensure your observation sheet includes consumption entries.")
