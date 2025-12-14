"""
Configuration and styling for DocXtract Dashboard
"""

import streamlit as st

# Page configuration
PAGE_CONFIG = {
    'page_title': "DocXtract Pro - PDF Chart Analysis",
    'page_icon': "📊",
    'layout': "wide",
    'initial_sidebar_state': "expanded"
}

# Custom CSS
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .upload-section {
        border: 2px dashed #d1d5db;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f9fafb;
        margin: 1rem 0;
    }
    .chart-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .status-success {
        color: #059669;
        font-weight: 600;
    }
    .status-info {
        color: #3b82f6;
        font-weight: 600;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 1rem;
    }
</style>
"""

def apply_page_config():
    """Apply page configuration"""
    st.set_page_config(**PAGE_CONFIG)

def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

