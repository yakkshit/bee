import streamlit as st
import matplotlib.pyplot as plt

def inject_custom_styles():
    """Injects publication-grade CSS design system with light/dark adaptive variables."""
    # Typography and aesthetic parameters matching publication specs
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#111111'
    plt.rcParams['axes.linewidth'] = 1.6
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.major.size'] = 5.5
    plt.rcParams['xtick.major.width'] = 1.6
    plt.rcParams['ytick.major.size'] = 5.5
    plt.rcParams['ytick.major.width'] = 1.6

    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg-main: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    --text-main: #0f172a;
    --text-muted: #475569;
    --card-bg: rgba(255, 255, 255, 0.9);
    --card-border: rgba(203, 213, 225, 0.8);
    --card-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
    --hero-bg: linear-gradient(135deg, rgba(254, 243, 199, 0.85) 0%, rgba(253, 230, 138, 0.45) 50%, rgba(255, 255, 255, 0.95) 100%);
    --hero-border: rgba(245, 158, 11, 0.4);
    --hero-title-gradient: linear-gradient(135deg, #d97706 0%, #b45309 60%, #78350f 100%);
    --kpi-bg: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(241, 245, 249, 0.85) 100%);
    --kpi-border: rgba(203, 213, 225, 0.85);
    --kpi-val-color: #d97706;
    --kpi-label-color: #475569;
    --sidebar-bg: #ffffff;
    --sidebar-border: rgba(226, 232, 240, 0.9);
    --sidebar-heading: #b45309;
}

@media (prefers-color-scheme: dark) {
    :root {
        --bg-main: linear-gradient(180deg, #0b0f19 0%, #0f172a 100%);
        --text-main: #f3f4f6;
        --text-muted: #9ca3af;
        --card-bg: rgba(30, 41, 59, 0.6);
        --card-border: rgba(255, 255, 255, 0.08);
        --card-shadow: 0 14px 36px rgba(0, 0, 0, 0.4);
        --hero-bg: linear-gradient(135deg, rgba(245, 158, 11, 0.18) 0%, rgba(217, 119, 6, 0.08) 50%, rgba(15, 23, 42, 0.9) 100%);
        --hero-border: rgba(245, 158, 11, 0.3);
        --hero-title-gradient: linear-gradient(135deg, #fbbf24 0%, #f59e0b 50%, #fef3c7 100%);
        --kpi-bg: linear-gradient(135deg, rgba(255, 255, 255, 0.06) 0%, rgba(255, 255, 255, 0.02) 100%);
        --kpi-border: rgba(255, 255, 255, 0.1);
        --kpi-val-color: #fbbf24;
        --kpi-label-color: #9ca3af;
        --sidebar-bg: #0b0f19;
        --sidebar-border: rgba(255, 255, 255, 0.08);
        --sidebar-heading: #fbbf24;
    }
}

[data-theme="dark"], html[data-theme="dark"], body[data-theme="dark"], .stApp[data-theme="dark"] {
    --bg-main: linear-gradient(180deg, #0b0f19 0%, #0f172a 100%);
    --text-main: #f3f4f6;
    --text-muted: #9ca3af;
    --card-bg: rgba(30, 41, 59, 0.6);
    --card-border: rgba(255, 255, 255, 0.08);
    --card-shadow: 0 14px 36px rgba(0, 0, 0, 0.4);
    --hero-bg: linear-gradient(135deg, rgba(245, 158, 11, 0.18) 0%, rgba(217, 119, 6, 0.08) 50%, rgba(15, 23, 42, 0.9) 100%);
    --hero-border: rgba(245, 158, 11, 0.3);
    --hero-title-gradient: linear-gradient(135deg, #fbbf24 0%, #f59e0b 50%, #fef3c7 100%);
    --kpi-bg: linear-gradient(135deg, rgba(255, 255, 255, 0.06) 0%, rgba(255, 255, 255, 0.02) 100%);
    --kpi-border: rgba(255, 255, 255, 0.1);
    --kpi-val-color: #fbbf24;
    --kpi-label-color: #9ca3af;
    --sidebar-bg: #0b0f19;
    --sidebar-border: rgba(255, 255, 255, 0.08);
    --sidebar-heading: #fbbf24;
}

[data-theme="light"], html[data-theme="light"], body[data-theme="light"], .stApp[data-theme="light"] {
    --bg-main: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    --text-main: #0f172a;
    --text-muted: #475569;
    --card-bg: rgba(255, 255, 255, 0.9);
    --card-border: rgba(203, 213, 225, 0.8);
    --card-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
    --hero-bg: linear-gradient(135deg, rgba(254, 243, 199, 0.85) 0%, rgba(253, 230, 138, 0.45) 50%, rgba(255, 255, 255, 0.95) 100%);
    --hero-border: rgba(245, 158, 11, 0.4);
    --hero-title-gradient: linear-gradient(135deg, #d97706 0%, #b45309 60%, #78350f 100%);
    --kpi-bg: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(241, 245, 249, 0.85) 100%);
    --kpi-border: rgba(203, 213, 225, 0.85);
    --kpi-val-color: #d97706;
    --kpi-label-color: #475569;
    --sidebar-bg: #ffffff;
    --sidebar-border: rgba(226, 232, 240, 0.9);
    --sidebar-heading: #b45309;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: var(--bg-main) !important;
    color: var(--text-main) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: rgba(241, 245, 249, 0.85) !important;
    padding: 6px !important;
    border-radius: 12px !important;
    border: 1px solid rgba(203, 213, 225, 0.7) !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    padding: 9px 20px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    transition: all 0.2s ease !important;
    color: #475569 !important;
}

.stTabs [aria-selected="true"] {
    background-color: #ffffff !important;
    color: #0f172a !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
}

.hero-card {
    background: var(--hero-bg) !important;
    border: 1px solid var(--hero-border) !important;
    border-radius: 18px;
    padding: 28px 34px;
    margin-bottom: 22px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
    position: relative;
    overflow: hidden;
}

.hero-badge-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
}

.saas-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid rgba(203, 213, 225, 0.8);
    border-radius: 9999px;
    padding: 4px 12px;
    font-size: 0.75rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: 0.3px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
}

.saas-pill-live {
    background: #ecfdf5;
    border-color: #a7f3d0;
    color: #047857;
}

.hero-title {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.6px;
    background: var(--hero-title-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}

.hero-subtitle {
    font-size: 14px;
    color: var(--text-muted) !important;
    font-weight: 500;
}

.glass-card {
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 22px;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
}

.section-title {
    font-size: 18px;
    font-weight: 700;
    color: var(--text-main) !important;
    margin-top: 10px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.kpi-card {
    background: var(--kpi-bg) !important;
    border: 1px solid var(--kpi-border) !important;
    border-top: 3px solid #f59e0b !important;
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.07);
}
.kpi-val {
    font-size: 26px;
    font-weight: 800;
    color: var(--kpi-val-color) !important;
}
.kpi-label {
    font-size: 11px;
    font-weight: 700;
    color: var(--kpi-label-color) !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}

.sidebar-h2 {
    color: var(--sidebar-heading) !important;
    font-size: 20px;
    font-weight: 800;
    margin-bottom: 15px;
}
.sidebar-h3 {
    color: var(--sidebar-heading) !important;
    font-size: 16px;
    font-weight: 700;
}

.stButton>button {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(245, 158, 11, 0.3) !important;
}
.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(245, 158, 11, 0.45) !important;
}

.stDownloadButton>button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3) !important;
}
.stDownloadButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.45) !important;
}

[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 1px solid var(--sidebar-border) !important;
}
</style>
""", unsafe_allow_html=True)


def render_sidebar_controls():
    """Renders the aesthetic, statistical, and chart controls in the sidebar."""
    st.sidebar.markdown("<h2 class='sidebar-h2'>🎨 Chart Studio & Color Studio</h2>", unsafe_allow_html=True)

    theme_presets = {
        "Default Bioassay (Crimson / Amber)": ("#ef4444", "#f59e0b", "#10b981"),
        "Nature Emerald & Forest Teal": ("#059669", "#0d9488", "#10b981"),
        "Clinical Sapphire & Sky Cyan": ("#2563eb", "#0284c7", "#10b981"),
        "High-Contrast Charcoal & Orange": ("#1e293b", "#ea580c", "#10b981"),
        "Toxic Twilight (Violet & Rose)": ("#7c3aed", "#e11d48", "#10b981"),
        "Custom Colors...": (None, None, None)
    }

    selected_theme = st.sidebar.selectbox("Scientific Palette Theme", list(theme_presets.keys()), index=0)
    preset_p, preset_s, preset_a = theme_presets[selected_theme]

    # Pre-populate session state if preset selected
    if preset_p is not None:
        if st.session_state.get('last_theme') != selected_theme:
            st.session_state['custom_primary'] = preset_p
            st.session_state['custom_secondary'] = preset_s
            st.session_state['last_theme'] = selected_theme

    default_prim = st.session_state.get('custom_primary', preset_p or "#ef4444")
    default_sec = st.session_state.get('custom_secondary', preset_s or "#f59e0b")

    c_cp1, c_cp2 = st.sidebar.columns(2)
    with c_cp1:
        primary_color = st.color_picker("Primary (Mortality)", default_prim, key="sb_prim_picker")
    with c_cp2:
        secondary_color = st.color_picker("Secondary (Survival)", default_sec, key="sb_sec_picker")

    line_width = st.sidebar.slider("Line Width", min_value=1.0, max_value=5.0, value=2.4)
    marker_size = st.sidebar.slider("Marker Size", min_value=3.0, max_value=12.0, value=7.0)
    font_size = st.sidebar.slider("Font Size", min_value=8, max_value=18, value=11)

    st.sidebar.markdown("---")
    st.sidebar.markdown("<h3 class='sidebar-h3'>🧪 Statistical Models</h3>", unsafe_allow_html=True)
    abbott_c = st.sidebar.number_input(
        "Abbott Control Mortality (C) %", min_value=0.0, max_value=100.0, value=0.0, step=1.0,
        help="Consider all as alive by default (C=0)"
    )

    line_style_map = {"Solid": "-", "Dashed": "--", "Dotted": ":", "Dash-Dot": "-."}
    selected_line_style = st.sidebar.selectbox("Select Line Style", list(line_style_map.keys()))
    line_style = line_style_map[selected_line_style]

    st.sidebar.markdown("---")
    st.sidebar.markdown("<h3 class='sidebar-h3'>🔥 Heatmap Palette</h3>", unsafe_allow_html=True)
    hm_alive_color = st.sidebar.color_picker("Alive Color", "#10b981")
    hm_behavior_color = st.sidebar.color_picker("Behavior Change Color", "#f59e0b")
    hm_dead_color = st.sidebar.color_picker("Dead Color", "#ef4444")

    plt.rcParams['font.size'] = font_size

    return {
        'primary_color': primary_color,
        'secondary_color': secondary_color,
        'line_width': line_width,
        'marker_size': marker_size,
        'font_size': font_size,
        'abbott_c': abbott_c,
        'line_style': line_style,
        'hm_alive_color': hm_alive_color,
        'hm_behavior_color': hm_behavior_color,
        'hm_dead_color': hm_dead_color
    }


def render_hero_banner():
    """Renders the top SaaS branding hero card."""
    st.markdown("""
<div class="hero-card">
    <div class="hero-badge-row">
        <span class="saas-pill saas-pill-live">● Bioassay Engine v2.4 Active</span>
        <span class="saas-pill">⚡ Probit & Cox PH Suite</span>
        <span class="saas-pill">🔬 EPA / OECD / Finney Standards</span>
    </div>
    <div class="hero-title">🐝 ApiaryBio™ Survival & Toxicological Analytics Studio</div>
    <div class="hero-subtitle">Enterprise Biostatistical Modeling • Probit LTₓ Spectrum • Mantel-Cox Log-Rank Tests • Cox Proportional Hazards Uncertainty</div>
</div>
""", unsafe_allow_html=True)


def render_metadata_expander():
    """Renders the observation metadata sheet editor."""
    metadata_keys = [
        'Date', 'Start time', 'Weather', 'Observer', 'Treatment',
        'Hive / colony ID', 'Bee type', 'Feeder / resource', 'Purpose / motivation',
        'Outside temp(°C)', 'Outdoor humidity', 'Other conditions'
    ]
    for k in metadata_keys:
        if f"meta_{k}" not in st.session_state:
            st.session_state[f"meta_{k}"] = ""

    with st.expander("📝 Edit Observation Sheet Metadata", expanded=False):
        cols1 = st.columns(4)
        st.session_state['meta_Date'] = cols1[0].text_input("Date", value=st.session_state['meta_Date'] or "01.09.2026")
        st.session_state['meta_Start time'] = cols1[1].text_input("Start time", value=st.session_state['meta_Start time'] or "15:23")
        st.session_state['meta_Weather'] = cols1[2].text_input("Weather", value=st.session_state['meta_Weather'] or "sunny")
        st.session_state['meta_Observer'] = cols1[3].text_input("Observer", value=st.session_state['meta_Observer'] or "Fereshteh")

        cols2 = st.columns(4)
        st.session_state['meta_Treatment'] = cols2[0].text_input("Treatment", value=st.session_state['meta_Treatment'] or "Sucrose solution")
        st.session_state['meta_Hive / colony ID'] = cols2[1].text_input("Hive / colony ID", value=st.session_state['meta_Hive / colony ID'] or "Colony n.9")
        st.session_state['meta_Bee type'] = cols2[2].text_input("Bee type", value=st.session_state['meta_Bee type'] or "Forager")
        st.session_state['meta_Feeder / resource'] = cols2[3].text_input("Feeder / resource", value=st.session_state['meta_Feeder / resource'] or "Pipette: 2 µL")

        cols3 = st.columns(4)
        st.session_state['meta_Purpose / motivation'] = cols3[0].text_input("Purpose / motivation", value=st.session_state['meta_Purpose / motivation'] or "baseline")
        st.session_state['meta_Outside temp(°C)'] = cols3[1].text_input("Outside temp(°C)", value=st.session_state['meta_Outside temp(°C)'] or "24.6")
        st.session_state['meta_Outdoor humidity'] = cols3[2].text_input("Outdoor humidity", value=st.session_state['meta_Outdoor humidity'] or "44")
        st.session_state['meta_Other conditions'] = cols3[3].text_input("Other conditions", value=st.session_state['meta_Other conditions'] or "")
