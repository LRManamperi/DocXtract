"""
Home page for DocXtract Dashboard
"""

import streamlit as st
import pandas as pd
import tempfile
import os
import time
from PIL import Image

from docxtract import read_pdf


def render_home():
    """Render the home page"""
    # Hero section
    st.markdown('<h1 class="main-header">📊 DocXtract Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #6b7280;">Professional PDF Chart & Table Analysis Tool</p>', unsafe_allow_html=True)

    # Upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Your PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file to analyze",
        type=['pdf'],
        help="Upload a PDF document containing charts or tables for analysis",
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.markdown('<div class="status-info">File uploaded successfully! Click "Analyze PDF" to start processing.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Action button
    if uploaded_file:
        if st.button("🚀 Analyze PDF", type="primary", use_container_width=True):
            _process_pdf(uploaded_file)
    else:
        _render_welcome_message()


def _process_pdf(uploaded_file):
    """Process the uploaded PDF file"""
    with st.spinner("🔄 Processing PDF..."):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Save uploaded file temporarily
        status_text.text("📁 Saving file...")
        progress_bar.progress(10)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        status_text.text("🔍 Analyzing document structure...")
        progress_bar.progress(30)

        try:
            # Extract data
            status_text.text("📊 Detecting charts and extracting data...")
            progress_bar.progress(60)

            result = read_pdf(pdf_path, pages='all', flavor='lattice')
            st.session_state.result = result
            st.session_state.pdf_path = pdf_path
            st.session_state.filename = uploaded_file.name

            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)

            time.sleep(0.5)

            charts_with_data = sum(1 for g in result.graphs if hasattr(g, 'data') and g.data and g.data.get('values'))
            
            st.success(f"🎉 Successfully analyzed **{uploaded_file.name}**!")
            st.markdown(f"**Found:** {len(result.graphs)} charts ({charts_with_data} with data)")

            # Show quick results
            if result.graphs:
                _render_quick_results(result, charts_with_data)

        except Exception as e:
            st.error(f"❌ Error processing PDF: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            progress_bar.empty()
            status_text.empty()
        finally:
            # Clean up temporary file
            if os.path.exists(pdf_path):
                try:
                    os.unlink(pdf_path)
                except:
                    pass


def _render_quick_results(result, charts_with_data):
    """Render quick results preview"""
    st.markdown("### 📊 Detected Charts")
    chart_types = [g.graph_type.value.replace('_', ' ').title() for g in result.graphs]
    type_counts = pd.Series(chart_types).value_counts()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Charts", len(result.graphs))
    with col2:
        st.metric("With Data", charts_with_data)
    with col3:
        st.metric("Most Common", type_counts.index[0] if not type_counts.empty else "N/A")

    # Show chart previews
    st.markdown("#### Chart Previews")
    preview_cols = st.columns(min(3, len(result.graphs)))
    for i, graph in enumerate(result.graphs[:3]):
        with preview_cols[i]:
            if graph.image is not None:
                if len(graph.image.shape) == 3:
                    img = Image.fromarray(graph.image)
                else:
                    img = Image.fromarray(graph.image).convert('RGB')
                st.image(img, use_container_width=True, caption=f"{graph.graph_type.value.replace('_', ' ').title()}")

    if len(result.graphs) > 3:
        st.info(f"📈 View all {len(result.graphs)} charts in the Charts Analysis tab")


def _render_welcome_message():
    """Render welcome message and features"""
    st.markdown("---")
    st.markdown("### ✨ Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🎯 Chart Detection")
        st.write("Automatically detect and classify bar, line, pie charts, and scatter plots")
    with col2:
        st.markdown("#### 📊 Data Extraction")
        st.write("Extract actual values from charts as structured data")
    with col3:
        st.markdown("#### 💾 Export Options")
        st.write("Download chart data as CSV or JSON")

    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. **Upload** a PDF file using the uploader above
    2. **Click** "Analyze PDF" to start processing
    3. **Explore** extracted chart data in the Chart Data Tables tab
    4. **Download** data as CSV files
    """)

