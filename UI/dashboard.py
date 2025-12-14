"""
DocXtract Dashboard - Professional PDF Chart Analysis Tool
Main entry point with navigation
"""

import streamlit as st
import sys
import os

# Handle imports - support both relative and absolute

    # Try absolute imports (when PYTHONPATH is set to parent directory)
from UI.config import apply_page_config, apply_custom_css
from UI.page_modules import (
        render_home,
        render_charts_analysis,
        render_chart_data_tables,
        render_testing,
        render_about
)

# Apply configuration
apply_page_config()
apply_custom_css()

# Import DocXtract
try:
    from docxtract import read_pdf
except ImportError:
    st.error("❌ DocXtract library not found. Please install it first.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-header">📊 DocXtract Pro</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigation",
        ["🏠 Home", "📈 Charts Analysis", "📋 Chart Data Tables", "🧪 Testing", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Quick stats (if file processed)
    if 'result' in st.session_state:
        result = st.session_state.result
        st.markdown("### Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Charts", len(result.graphs))
        with col2:
            charts_with_data = sum(1 for g in result.graphs if hasattr(g, 'data') and g.data)
            st.metric("With Data", charts_with_data)

    st.markdown("---")
    st.markdown("**Built with:** Streamlit & DocXtract")

# Main content based on navigation
if page == "🏠 Home":
    render_home()
elif page == "📈 Charts Analysis":
    render_charts_analysis()
elif page == "📋 Chart Data Tables":
    render_chart_data_tables()
elif page == "🧪 Testing":
    render_testing()
elif page == "ℹ️ About":
    render_about()
