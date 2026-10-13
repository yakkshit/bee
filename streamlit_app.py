import streamlit as st

from modules.styles import (
    inject_custom_styles,
    render_sidebar_controls,
    render_hero_banner,
    render_metadata_expander
)
from modules.tab_single import render_single_batch_tab
from modules.tab_multi import render_multi_batch_tab
from modules.tab_survival import render_survival_tab
from modules.tab_spss import render_spss_tab

# Page configuration
st.set_page_config(
    page_title="Bee Analytics Studio",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply UI design system & theme styling
inject_custom_styles()

# Sidebar aesthetic & statistical controls
settings = render_sidebar_controls()

# Top Hero header banner & Metadata sheet editor
render_hero_banner()
render_metadata_expander()

# Main Application Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🐝 Single CSV Analysis",
    "📈 Multi-CSV Probit Comparison",
    "🔬 Survival Statistics (Log-Rank & Cox Regression)",
    "🔮 SPSS Suite (Coming Soon)"
])

with tab1:
    render_single_batch_tab(settings)

with tab2:
    render_multi_batch_tab(settings)

with tab3:
    render_survival_tab()

with tab4:
    render_spss_tab()
