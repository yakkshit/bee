"""
Global Exporter Module for Antigravity Bee Analytics Studio.
Provides unified, one-click global download capabilities across the entire application:
- All Single-CSV analysis plots in a ZIP package
- All Multi-CSV comparison plots in a ZIP package
- Complete Studio Master Bundle (Single + Multi + Statistical CSV summaries)
Renderable in both the header quick-actions bar and the sidebar.
"""
import io
import zipfile
import pandas as pd
import streamlit as st


def get_single_plots_zip() -> bytes:
    """Retrieves or builds the ZIP package containing all single-CSV analysis plots."""
    single_images = st.session_state.get('single_plot_images', {})
    if not single_images:
        return b""

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, data in single_images.items():
            zf.writestr(fname, data)
        # Also include active observation CSV if available
        active_df = st.session_state.get('df')
        if active_df is not None and isinstance(active_df, pd.DataFrame) and not active_df.empty:
            zf.writestr("observation_sheet_data.csv", active_df.to_csv(index=False).encode('utf-8'))
    return buf.getvalue()


def get_multi_plots_zip() -> bytes:
    """Retrieves or builds the ZIP package containing all multi-CSV comparison plots."""
    multi_images = st.session_state.get('multi_plot_images', {})
    if not multi_images:
        return b""

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, data in multi_images.items():
            zf.writestr(fname, data)
        # Include summary dataframe if available
        summary_df = st.session_state.get('multi_summary_df')
        if summary_df is not None and isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
            zf.writestr("multi_csv_probit_summary.csv", summary_df.to_csv(index=False).encode('utf-8'))
    return buf.getvalue()


def get_complete_master_zip() -> bytes:
    """Builds the comprehensive master ZIP archive bundling all Single and Multi plots + CSVs."""
    single_images = st.session_state.get('single_plot_images', {})
    multi_images = st.session_state.get('multi_plot_images', {})
    
    if not single_images and not multi_images:
        return b""

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Single CSV Analysis Folder
        for fname, data in single_images.items():
            zf.writestr(f"single_csv_analysis/{fname}", data)
        active_df = st.session_state.get('df')
        if active_df is not None and isinstance(active_df, pd.DataFrame) and not active_df.empty:
            zf.writestr("single_csv_analysis/observation_sheet_data.csv", active_df.to_csv(index=False).encode('utf-8'))

        # 2. Multi-CSV Comparison Folder
        for fname, data in multi_images.items():
            zf.writestr(f"multi_csv_comparison/{fname}", data)
        summary_df = st.session_state.get('multi_summary_df')
        if summary_df is not None and isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
            zf.writestr("multi_csv_comparison/multi_csv_probit_summary.csv", summary_df.to_csv(index=False).encode('utf-8'))

        # 3. Readme summary text inside the ZIP
        manifest_text = (
            "Antigravity Bee Analytics Studio — Complete Plot Export Bundle\n"
            "==============================================================\n\n"
            f"• Single-CSV Analysis Figures: {len(single_images)} plots\n"
            f"• Multi-CSV Comparison Figures: {len(multi_images)} plots\n"
            "• Resolution: Publication-grade 300 DPI\n"
            "• Protocol: Post-2h Starvation Rule Compliant (Day 0 Baseline Excluded)\n"
        )
        zf.writestr("BUNDLE_INFO.txt", manifest_text.encode('utf-8'))

    return buf.getvalue()


def render_sidebar_global_download():
    """Renders the persistent global download center inside the Streamlit sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h3 class='sidebar-h3'>📦 Global Download Center</h3>", unsafe_allow_html=True)
    st.sidebar.caption("Download all generated figures and summaries at once in 300 DPI publication resolution.")

    single_bytes = get_single_plots_zip()
    multi_bytes = get_multi_plots_zip()
    complete_bytes = get_complete_master_zip()

    n_single = len(st.session_state.get('single_plot_images', {}))
    n_multi = len(st.session_state.get('multi_plot_images', {}))

    if single_bytes:
        st.sidebar.download_button(
            label=f"📦 All Single-CSV Plots ({n_single} Figs)",
            data=single_bytes,
            file_name="all_single_csv_plots_300dpi.zip",
            mime="application/zip",
            use_container_width=True,
            key="sb_dl_single_all"
        )
    else:
        st.sidebar.info("💡 Single-CSV plots will be ready to download after viewing Tab 1.")

    if multi_bytes:
        st.sidebar.download_button(
            label=f"📊 All Multi-CSV Plots ({n_multi} Figs)",
            data=multi_bytes,
            file_name="all_multi_csv_plots_300dpi.zip",
            mime="application/zip",
            use_container_width=True,
            key="sb_dl_multi_all"
        )
    else:
        st.sidebar.info("💡 Multi-CSV plots will be ready to download after viewing Tab 2.")

    if complete_bytes:
        st.sidebar.download_button(
            label=f"🌟 Complete Studio Bundle ({n_single + n_multi} Figs + CSVs)",
            data=complete_bytes,
            file_name="complete_bee_analytics_studio_bundle.zip",
            mime="application/zip",
            use_container_width=True,
            key="sb_dl_complete_master"
        )


def render_header_global_download():
    """Renders an interactive quick-download bar directly in the top header."""
    single_bytes = get_single_plots_zip()
    multi_bytes = get_multi_plots_zip()
    complete_bytes = get_complete_master_zip()

    n_single = len(st.session_state.get('single_plot_images', {}))
    n_multi = len(st.session_state.get('multi_plot_images', {}))

    # Show only if plots are generated
    if not single_bytes and not multi_bytes:
        return

    st.markdown("""
<div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 12px; padding: 10px 18px; margin-bottom: 14px;">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
        <span style="font-weight: 700; color: #b45309; font-size: 13.5px;">⚡ Global One-Click Figure Exports (300 DPI Publication Ready):</span>
    </div>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        if single_bytes:
            st.download_button(
                label=f"📦 Download All Single Plots ({n_single} Figs)",
                data=single_bytes,
                file_name="all_single_csv_plots_300dpi.zip",
                mime="application/zip",
                use_container_width=True,
                key="hdr_dl_single_all"
            )
        else:
            st.button("📦 Single Plots (Open Tab 1)", disabled=True, use_container_width=True, key="hdr_dl_single_dis")

    with col2:
        if multi_bytes:
            st.download_button(
                label=f"📊 Download All Multi Plots ({n_multi} Figs)",
                data=multi_bytes,
                file_name="all_multi_csv_plots_300dpi.zip",
                mime="application/zip",
                use_container_width=True,
                key="hdr_dl_multi_all"
            )
        else:
            st.button("📊 Multi Plots (Open Tab 2)", disabled=True, use_container_width=True, key="hdr_dl_multi_dis")

    with col3:
        if complete_bytes:
            st.download_button(
                label=f"🌟 Complete Studio Bundle ({n_single + n_multi} Figs + CSVs)",
                data=complete_bytes,
                file_name="complete_bee_analytics_studio_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key="hdr_dl_master_all"
            )
        else:
            st.button("🌟 Complete Bundle (Awaiting Plots)", disabled=True, use_container_width=True, key="hdr_dl_master_dis")

