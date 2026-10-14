"""
Tab 3: Cohort & Individual Tracking & Nutrient Consumption.
Renders individual bee health status matrix heatmaps, population composition stackplots,
daily status distributions, cumulative mortality, and per-bee feed consumption trajectories.
"""
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


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

    # Full-width Consumption Tracker
    if len(cons_matrix) > 0 and any(any(v > 0 for v in bee_cons) for bee_cons in cons_matrix):
        st.markdown("---")
        st.subheader("Daily Resource Consumption per Bee (µL)")
        fig_cons, ax_cons = plt.subplots(figsize=(10.0, 5.2), dpi=300)
        x_days = elapsed_days
        colors_c = plt.cm.tab20(np.linspace(0, 1, max(1, len(cons_cols))))
        for i, col_name in enumerate(cons_cols):
            bee_label = col_name.replace("Cons", "Bee").replace(" (µL)", "").strip()
            ax_cons.plot(x_days, cons_matrix[i], marker='o', linewidth=1.5, markersize=5, label=bee_label, color=colors_c[i])

        ax_cons.set_xlabel('Observation Time / Days', fontsize=12, fontweight='bold')
        ax_cons.set_ylabel('Consumption (µL)', fontsize=12, fontweight='bold')
        ax_cons.set_xlim(-0.1, max_time * 1.05)
        ax_cons.set_xticks(elapsed_days)
        ax_cons.set_xticklabels(day_labels, rotation=45, ha='right')
        ax_cons.grid(True, linestyle='--', alpha=0.45)
        ax_cons.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=8.5)
        fig_cons.tight_layout()
        st.pyplot(fig_cons)
        if image_collector:
            image_collector(fig_cons, 'fig_consumption_per_bee.png', 'Download Consumption Plot', 't3_cons')
        plt.close(fig_cons)
