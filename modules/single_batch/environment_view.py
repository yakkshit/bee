"""
Tab 4: Environmental Dynamics & Sensor Interactions.
Visualizes temporal fluctuations in rearing temperature and relative humidity,
bivariate temperature-humidity phase spaces, and Pearson correlation matrices.
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


def render_single_environment_view(computed, df, settings=None, image_collector=None):
    """Renders the Environmental Dynamics sub-tab."""
    if settings is None:
        settings = {}

    primary_color = settings.get('primary_color', '#ef4444')
    marker_size = settings.get('marker_size', 7.0)

    c = computed
    elapsed_days = c['elapsed_days']
    day_labels = c['day_labels']
    max_time = c['max_time']
    temps = c['temps']
    humidities = c['humidities']
    mortality_pct = c['mortality_pct']
    survival_pct = c['survival_pct']

    st.markdown("#### Ambient Temperature & Relative Humidity Interactions")
    st.caption("Temporal variation in rearing conditions, temp-humidity environmental space, and bivariate correlation matrix.")

    c_e1, c_e2 = st.columns(2)

    if any(temps):
        with c_e1:
            st.subheader("Environmental Conditions Timeline")
            fig5, ax1 = plt.subplots(figsize=(7.0, 5.2), dpi=300)
            x_days = elapsed_days
            color_temp = '#d95f02'
            ax1.set_xlabel('Observation Time / Days', fontsize=12, fontweight='bold', labelpad=7)
            ax1.set_ylabel('Temperature / °C', color=color_temp, fontsize=12, fontweight='bold', labelpad=7)
            l1 = ax1.plot(x_days, temps, color=color_temp, marker='o', linewidth=2.2, markersize=7.0, label='Temperature (°C)')
            ax1.tick_params(axis='y', labelcolor=color_temp)
            ax1.set_ylim(min(temps) - 1, max(temps) + 1)

            ax2 = ax1.twinx()
            color_hum = '#0284c7'
            ax2.set_ylabel('Relative Humidity / %', color=color_hum, fontsize=12, fontweight='bold', labelpad=7)
            l2 = ax2.plot(x_days, humidities, color=color_hum, marker='s', linewidth=2.2, markersize=7.0, linestyle='--', label='Relative Humidity (%)')
            ax2.tick_params(axis='y', labelcolor=color_hum)
            if any(humidities):
                ax2.set_ylim(min(humidities) - 5, max(humidities) + 5)

            ax1.set_xlim(-0.1, max_time * 1.05)
            ax1.set_xticks(elapsed_days)
            ax1.set_xticklabels(day_labels, rotation=45, ha="right")
            ax1.spines['top'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            lines_e = l1 + l2
            ax1.legend(lines_e, [l.get_label() for l in lines_e], loc='upper left', frameon=True, framealpha=0.9, facecolor='white', edgecolor='none', fontsize=9.5)
            fig5.tight_layout()
            st.pyplot(fig5)
            if image_collector:
                image_collector(fig5, 'fig_environmental_conditions.png', 'Download Environmental Timeline', 't4_env')

        with c_e2:
            st.subheader("Temperature vs Humidity Scatter")
            fig_scat, ax_scat = plt.subplots(figsize=(7.0, 5.2), dpi=300)
            ax_scat.scatter(temps, humidities, color=primary_color, s=marker_size**2 * 4, edgecolor='black', alpha=0.85)
            for i, txt in enumerate(day_labels):
                ax_scat.annotate(txt, (temps[i], humidities[i]), fontsize=9, xytext=(5, 5), textcoords='offset points')
            ax_scat.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold', labelpad=7)
            ax_scat.set_ylabel('Humidity (%)', fontsize=12, fontweight='bold', labelpad=7)
            ax_scat.spines['top'].set_visible(False)
            ax_scat.spines['right'].set_visible(False)
            fig_scat.tight_layout()
            st.pyplot(fig_scat)
            if image_collector:
                image_collector(fig_scat, 'fig_temp_vs_humidity.png', 'Download Scatter Plot', 't4_scat')

        with c_e1:
            st.subheader("Statistical Pearson Correlation Heatmap")
            corr_data = {
                'Mortality (%)': mortality_pct,
                'Survival (%)': survival_pct,
                'Temp': temps,
                'Humidity': humidities
            }
            corr_df = pd.DataFrame(corr_data).corr()
            fig_corr, ax_corr = plt.subplots(figsize=(6.2, 5.0), dpi=300)
            cax = ax_corr.matshow(corr_df, cmap='coolwarm', vmin=-1, vmax=1)
            fig_corr.colorbar(cax)
            ax_corr.set_xticks(range(len(corr_df.columns)))
            ax_corr.set_yticks(range(len(corr_df.columns)))
            ax_corr.set_xticklabels(corr_df.columns, rotation=45, ha='left', fontsize=10)
            ax_corr.set_yticklabels(corr_df.columns, fontsize=10)
            for i in range(len(corr_df.columns)):
                for j in range(len(corr_df.columns)):
                    ax_corr.text(j, i, f"{corr_df.iloc[i, j]:.2f}", ha='center', va='center', color='white' if abs(corr_df.iloc[i, j]) > 0.5 else 'black', fontweight='bold')
            fig_corr.tight_layout()
            st.pyplot(fig_corr)
            if image_collector:
                image_collector(fig_corr, 'fig_correlation_heatmap.png', 'Download Correlation Heatmap', 't4_corr')
    else:
        st.info("No temperature or humidity sensor readings available in the uploaded dataset.")
