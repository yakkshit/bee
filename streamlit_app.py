# Bee Analytics Studio Main Application
import sys
import logging
import streamlit as st

# Suppress stale browser media cache warning spam (MediaFileHandler)
logging.getLogger("streamlit.web.server.media_file_handler").setLevel(logging.ERROR)
logging.getLogger("tornado.access").setLevel(logging.ERROR)

# Automatically ensure newest module code loads without stale bytecode
for _m in list(sys.modules.keys()):
    if _m == 'modules' or _m.startswith('modules.'):
        del sys.modules[_m]
from modules.styles import (
    inject_custom_styles,
    render_sidebar_controls,
    render_hero_banner,
    render_metadata_expander
)
from modules.tab_single import render_single_batch_tab
from modules.tab_multi import render_multi_batch_tab
from modules.tab_survival import render_survival_tab
from modules.tab_ai_plots import render_ai_plots_tab
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

# Global Header Download Center (One-Click Bulk Downloads)
from modules.global_exporter import render_header_global_download
render_header_global_download()

# Main Application Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🐝 Single CSV Analysis",
    "📈 Multi-CSV Probit Comparison",
    "🔬 Survival Statistics (Log-Rank & Cox Regression)",
    "🤖 AI Plot Studio (ChatGPT Code Runner)",
    "🔮 SPSS Suite (Coming Soon)"
])

with tab1:
    render_single_batch_tab(settings)

with tab2:
    render_multi_batch_tab(settings)

with tab3:
    render_survival_tab()

with tab4:
    render_ai_plots_tab()

with tab5:
    render_spss_tab()
