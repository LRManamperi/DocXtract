"""
DocXtract Dashboard - Main Entry Point
Streamlit dashboard for PDF chart and table extraction
"""

import streamlit as st
import sys
import os

# Handle imports - support both relative and absolute
try:
    # Try absolute imports (when PYTHONPATH is set to parent directory)
    from UI.config import apply_page_config, apply_custom_css
    from UI.page_modules import (
        render_home,
        render_charts_analysis,
        render_chart_data_tables,
        render_tables,
        render_testing,
        render_about
    )
except ImportError:
    try:
        # Try relative imports (when running from UI directory)
        from config import apply_page_config, apply_custom_css
        from page_modules import (
            render_home,
            render_charts_analysis,
            render_chart_data_tables,
            render_tables,
            render_testing,
            render_about
        )
    except ImportError as e:
        st.error(f"Failed to import modules: {e}")
        st.stop()

# Import AI analysis module
try:
    from UI.page_modules.ai_analysis import render_ai_analysis
except ImportError:
    try:
        from page_modules.ai_analysis import render_ai_analysis
    except ImportError:
        def render_ai_analysis():
            st.warning("AI Analysis module not available. Please check installation.")
            st.info("Required packages: langchain-core, langchain-openai, langchain-groq")

# Apply page configuration
apply_page_config()
apply_custom_css()

# Import DocXtract library
try:
    from docxtract import read_pdf
except ImportError:
    st.error("❌ DocXtract library not found. Please install it first.")
    st.stop()

def _get_page_count(result) -> int:
    """Get page count from result with fallback"""
    if hasattr(result, 'page_count'):
        return result.page_count
    # Fallback: count unique pages from tables and graphs
    pages = set()
    for t in result.tables:
        pages.add(t.page)
    for g in result.graphs:
        pages.add(g.page)
    return len(pages) if pages else 1


def main():
    """Main dashboard function"""
    
    # Sidebar
    with st.sidebar:
        st.markdown('<p class="sidebar-header">📊 DocXtract Pro</p>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["🏠 Home", "🤖 AI Analysis", "📈 Charts Analysis", "📋 Tables", "📄 Chart Data Tables", "🧪 Testing", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown("**Quick Stats**")
        if 'result' in st.session_state and st.session_state.result is not None:
            result = st.session_state.result
            page_count = _get_page_count(result)
            text_count = len(result.text_content) if hasattr(result, 'text_content') else 0
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Pages", page_count)
                st.metric("Charts", len(result.graphs))
            with col2:
                st.metric("Tables", len(result.tables))
                st.metric("Text Blocks", text_count)
        else:
            st.info("Upload a PDF to see stats")
    
    # Main content
    if page == "🏠 Home":
        render_home()
    elif page == "🤖 AI Analysis":
        render_ai_analysis()
    elif page == "📈 Charts Analysis":
        render_charts_analysis()
    elif page == "📋 Tables":
        render_tables()
    elif page == "📄 Chart Data Tables":
        render_chart_data_tables()
    elif page == "🧪 Testing":
        render_testing()
    elif page == "ℹ️ About":
        render_about()

if __name__ == "__main__":
    main()
