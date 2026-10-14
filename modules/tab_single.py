"""
Single Batch Analysis Coordinator Tab.
Organizes data entry, mathematical computation, and the 6 thematic analysis tabs:
1. 📈 Mortality & Survival Models
2. ⏱️ Lethal Time (LTₓ) Laboratory
3. 🐝 Cohort & Individual Tracking
4. 🌦️ Environmental Dynamics
5. 🔬 Biostatistical Models
6. 📑 Publication Multi-Panel Figure
"""
import io
import zipfile
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from modules.single_batch import (
    render_observation_sheet_editor,
    compute_single_batch_data,
    render_single_survival_curves,
    render_single_lethal_time_view,
    render_single_feeding_view,
    render_single_environment_view,
    render_single_biostats_view,
    render_single_publication_figure,
)


def render_single_batch_tab(settings=None):
    """Renders the comprehensive Single CSV Analysis tab."""
    if settings is None:
        settings = {}

    # 1. Observation Sheet Data Entry & Printable HTML Export
    df = render_observation_sheet_editor()

    st.markdown("---")
    st.header("📊 Plot Generation & Analytics")

    # 2. Mathematical Modeling & Metric Extraction
    computed = compute_single_batch_data(df, settings)

    num_bees = computed['num_bees']
    num_days = computed['num_days']
    survival_pct = computed['survival_pct']
    temps = computed['temps']

    # 3. Top-Level Experiment KPI Cards
    st.markdown("### 📊 Experiment Overview & Key Metrics")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{num_bees}</div><div class="kpi-label">Observed Bees</div></div>', unsafe_allow_html=True)
    with kpi2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{num_days}</div><div class="kpi-label">Observation Days</div></div>', unsafe_allow_html=True)
    with kpi3:
        final_surv = f"{survival_pct[-1]:.1f}%" if num_days > 0 else "N/A"
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{final_surv}</div><div class="kpi-label">Final Survival Rate</div></div>', unsafe_allow_html=True)
    with kpi4:
        avg_temp = f"{np.mean(temps):.1f}°C" if any(temps) else "N/A"
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{avg_temp}</div><div class="kpi-label">Avg Temperature</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_zip_top, col_info = st.columns([1.5, 3])
    with col_zip_top:
        zip_button_top = st.empty()
    with col_info:
        st.caption("💡 Figures are grouped into structured thematic categories below. Each figure can be downloaded individually at publication resolution (300 DPI) or packaged together as a ZIP archive.")

    # Image Collector for individual downloads and unified ZIP packaging
    plot_images_dict = {}

    def image_collector(fig, filename, text, key_suffix=""):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        img_bytes = buf.getvalue()
        plot_images_dict[filename] = img_bytes
        btn_key = f"dl_{filename}_{key_suffix}" if key_suffix else f"dl_{filename}"
        st.download_button(label=f"📥 {text}", data=img_bytes, file_name=filename, mime="image/png", key=btn_key)
        plt.close(fig)

    # 4. The 6 Structured Thematic Tabs
    tab_surv, tab_lt, tab_cohort, tab_env, tab_stats, tab_pub = st.tabs([
        "📈 Mortality & Survival Models",
        "⏱️ Lethal Time (LTₓ) Laboratory",
        "🐝 Cohort & Individual Tracking",
        "🌦️ Environmental Dynamics",
        "🔬 Biostatistical Models",
        "📑 Publication Multi-Panel Figure"
    ])

    with tab_surv:
        render_single_survival_curves(computed, settings, image_collector)

    with tab_lt:
        render_single_lethal_time_view(computed, settings, image_collector)

    with tab_cohort:
        render_single_feeding_view(computed, df, settings, image_collector)

    with tab_env:
        render_single_environment_view(computed, df, settings, image_collector)

    with tab_stats:
        render_single_biostats_view(computed, df, settings, image_collector)

    with tab_pub:
        render_single_publication_figure(computed, settings, image_collector)

    # 5. ZIP Packaging (Top and Bottom Download Triggers)
    if plot_images_dict:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for fname, data in plot_images_dict.items():
                zip_file.writestr(fname, data)
        zip_bytes = zip_buf.getvalue()

        zip_button_top.download_button(
            label="📦 Download All Plots (ZIP)",
            data=zip_bytes,
            file_name="all_plots.zip",
            mime="application/zip",
            key="zip_top_btn"
        )

        st.markdown("---")
        st.download_button(
            label="📦 Download All Generated Plots in a Single ZIP File",
            data=zip_bytes,
            file_name="all_plots.zip",
            mime="application/zip",
            key="zip_bottom_btn"
        )
