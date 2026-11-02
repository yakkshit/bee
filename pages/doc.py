"""
Dedicated /doc route for  Bee Analytics Studio.
Serves full scientific documentation, bioassay protocol specifications, and mathematical foundations.
"""

import streamlit as st
from modules.tab_doc import render_doc_page
from modules.styles import inject_custom_styles

# Page configuration
st.set_page_config(
    page_title="Documentation & Protocol | Bee Analytics Studio",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply UI design system
inject_custom_styles()

# Render documentation
render_doc_page()

