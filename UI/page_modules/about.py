"""
About page for DocXtract Dashboard
"""

import streamlit as st


def render_about():
    """Render the about page"""
    st.markdown('<h1 class="main-header">ℹ️ About DocXtract Pro</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📊 What is DocXtract Pro?
    
    DocXtract Pro is an advanced PDF document analysis tool that automatically extracts 
    charts, graphs, and tables from PDF documents using computer vision and machine learning.
    
    ---
    
    ### 🚀 Features
    
    | Feature | Description |
    |---------|-------------|
    | **Chart Detection** | Automatically detects bar charts, line charts, pie charts, scatter plots, and more |
    | **Table Extraction** | Extracts structured data from tables with support for complex layouts |
    | **Data Export** | Export extracted data to CSV or Excel formats |
    | **AI Analysis** | Analyze extracted data using LLMs (OpenAI, Groq) |
    | **OCR Support** | Text extraction from scanned documents using Tesseract |
    
    ---
    
    ### 🛠️ Technologies Used
    
    - **PyMuPDF (fitz)**: PDF parsing and rendering
    - **OpenCV**: Image processing and chart detection
    - **Tesseract OCR**: Optical character recognition
    - **Transformers**: Deep learning models for table detection
    - **LangChain**: LLM orchestration for AI analysis
    - **Streamlit**: Web dashboard framework
    
    ---
    
    ### 📖 How It Works
    
    1. **Upload**: Upload a PDF document
    2. **Detection**: AI models detect charts and tables in the document
    3. **Extraction**: Data is extracted from detected elements
    4. **Analysis**: Review extracted data and optionally run AI analysis
    5. **Export**: Download data in CSV or Excel format
    
    ---
    
    ### 🔧 Supported Chart Types
    
    - Bar Charts (horizontal and vertical)
    - Line Charts
    - Pie Charts
    - Scatter Plots
    - Area Charts
    - Combined/Mixed Charts
    
    ---
    
    ### 📋 Supported Table Types
    
    - Grid tables with clear borders
    - Borderless tables
    - Complex multi-row/column headers
    - Nested tables
    
    ---
    
    ### 🤖 AI Analysis Providers
    
    DocXtract Pro supports multiple LLM providers for intelligent data analysis:
    
    | Provider | Model | Notes |
    |----------|-------|-------|
    | **Groq** | llama-3.3-70b-versatile | Free tier available |
    | **OpenAI** | GPT-4o-mini | Requires API key |
    
    ---
    
    ### 📝 Version Information
    
    - **Version**: 1.0.0
    - **License**: MIT
    - **Author**: DocXtract Team
    
    ---
    
    ### 🐛 Troubleshooting
    
    **Common Issues:**
    
    1. **Tesseract not found**: Install Tesseract OCR and add to PATH
    2. **Low detection accuracy**: Use higher quality PDFs
    3. **Missing data**: Some charts may require manual extraction
    
    ---
    
    Built with ❤️ using Python and Streamlit
    """)


def _render_tech_card(icon, title, description):
    """Render a technology card"""
    st.markdown(f"""
    <div class="chart-card">
        <h4>{icon} {title}</h4>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)

