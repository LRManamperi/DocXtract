"""
About page for DocXtract Dashboard
"""

import streamlit as st


def render_about():
    """Render the about page"""
    st.markdown('<h2 class="sub-header">ℹ️ About DocXtract Pro</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📊 DocXtract Pro
    
    A professional PDF chart and table analysis tool built with:
    - **DocXtract**: Chart detection and data extraction library
    - **Streamlit**: Interactive web interface
    - **OpenCV & NumPy**: Image processing
    - **Pandas**: Data manipulation
    
    ### 🎯 Features
    
    ✅ **Chart Detection**
    - Bar charts (vertical & horizontal)
    - Line charts
    - Pie charts
    - Scatter plots
    
    ✅ **Data Extraction**
    - Automatic extraction of numerical values
    - Bar heights/widths
    - Line data points
    - Pie slice percentages
    - Scatter plot coordinates
    
    ✅ **Export Options**
    - CSV download for all extracted data
    - Chart metadata export
    - JSON format support
    
    ### 📖 Usage Guide
    
    1. **Upload**: Select a PDF file containing charts
    2. **Analyze**: Click "Analyze PDF" to process
    3. **Explore**: View charts in "Charts Analysis" tab
    4. **Extract**: Check "Chart Data Tables" for numerical data
    5. **Download**: Export data as CSV files
    
    ### 🔧 Technical Details
    
    **Chart Detection Method**: Axis-based detection
    - Detects perpendicular L-shaped axes
    - One chart per axis pair
    - Confidence scoring for reliability
    
    **Data Extraction**:
    - Image processing with OpenCV
    - Contour detection for shapes
    - Thresholding for data points
    - Normalized values (0-1 range)
    
    ### 💡 Tips
    
    - PDFs with clear, well-defined charts work best
    - Higher resolution PDFs yield better results
    - Charts with visible axes are easier to detect
    - Some hand-drawn or stylized charts may not be detected
    
    ### 📝 Limitations
    
    - Cannot extract data from purely image-based charts without structure
    - OCR for axis labels not yet implemented
    - Complex 3D charts may not be supported
    - Overlapping charts may cause detection issues
    
    ### 🚀 Future Enhancements
    
    - OCR for axis labels and legends
    - Support for 3D charts
    - Multi-chart PDFs with better separation
    - Real-time preview during upload
    - Batch processing improvements
    
    ---
    
    **Version**: 1.0.0  
    **Built with** ❤️ **using DocXtract & Streamlit**
    """)

    st.markdown('<h2 class="sub-header">🧪 Dataset Testing</h2>', unsafe_allow_html=True)

