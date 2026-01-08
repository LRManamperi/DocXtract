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
    """Render the home page with file upload and processing"""
    # Header
    st.markdown('<h1 class="main-header">📊 DocXtract Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6b7280;">Advanced PDF Chart & Table Extraction Tool</p>', unsafe_allow_html=True)
    
    # Check if we have existing results
    if 'result' in st.session_state and st.session_state.result is not None:
        _render_quick_results()
    else:
        _render_welcome_message()
    
    st.markdown("---")
    
    # Upload Section
    st.markdown('<p class="sub-header">📤 Upload PDF Document</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document containing charts, graphs, or tables for extraction"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(f"📄 **File:** {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
        with col2:
            process_btn = st.button("🚀 Extract Data", type="primary", use_container_width=True)
        
        if process_btn:
            _process_pdf(uploaded_file)
    
    # Show extraction results if available
    if 'result' in st.session_state and st.session_state.result is not None:
        _render_extraction_results()


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


def _render_welcome_message():
    """Render welcome message for new users"""
    st.markdown("""
    <div class="upload-section">
        <h3>👋 Welcome to DocXtract Pro!</h3>
        <p>Upload a PDF document to extract charts, graphs, and tables automatically.</p>
        <p><strong>Supported content:</strong> Bar charts, Line charts, Pie charts, Scatter plots, Tables, and more!</p>
    </div>
    """, unsafe_allow_html=True)


def _render_quick_results():
    """Render quick results summary"""
    result = st.session_state.result
    filename = st.session_state.get('filename', 'Unknown')
    page_count = _get_page_count(result)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">📄</div>
            <div class="metric-label">{filename[:20]}...</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(result.graphs)}</div>
            <div class="metric-label">Charts Found</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(result.tables)}</div>
            <div class="metric-label">Tables Found</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{page_count}</div>
            <div class="metric-label">Total Pages</div>
        </div>
        """, unsafe_allow_html=True)


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
        
        progress_bar.progress(20)
        status_text.text("📊 Extracting charts and tables...")
        
        try:
            # Process PDF
            result = read_pdf(pdf_path, pages='all', flavor='lattice')
            
            progress_bar.progress(80)
            status_text.text("✅ Processing complete!")
            
            # Store results in session state
            st.session_state.result = result
            st.session_state.pdf_path = pdf_path
            st.session_state.filename = uploaded_file.name
            
            progress_bar.progress(100)
            time.sleep(0.5)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Show success message
            st.success(f"✅ Successfully extracted {len(result.graphs)} charts and {len(result.tables)} tables!")
            st.rerun()
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error processing PDF: {str(e)}")
            
            # Cleanup
            try:
                os.unlink(pdf_path)
            except:
                pass


def _render_extraction_results():
    """Render extraction results summary"""
    result = st.session_state.result
    page_count = _get_page_count(result)
    text_count = len(result.text_content) if hasattr(result, 'text_content') else 0
    
    st.markdown("---")
    st.markdown('<p class="sub-header">📊 Extraction Results</p>', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Pages", page_count)
    with col2:
        st.metric("Charts Detected", len(result.graphs))
    with col3:
        st.metric("Tables Detected", len(result.tables))
    with col4:
        st.metric("Text Blocks", text_count)
    
    # Preview sections
    tab1, tab2 = st.tabs(["📈 Charts Preview", "📋 Tables Preview"])
    
    with tab1:
        if result.graphs:
            _render_charts_preview(result.graphs)
        else:
            st.info("No charts detected in this document.")
    
    with tab2:
        if result.tables:
            _render_tables_preview(result.tables)
        else:
            st.info("No tables detected in this document.")


def _render_charts_preview(graphs):
    """Render charts preview"""
    st.markdown(f"**Found {len(graphs)} chart(s)**")
    
    # Show first 3 charts
    cols = st.columns(min(3, len(graphs)))
    for i, graph in enumerate(graphs[:3]):
        with cols[i]:
            if graph.image is not None:
                try:
                    img = Image.fromarray(graph.image)
                    st.image(img, caption=f"Page {graph.page}: {graph.graph_type.name}", use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display chart {i+1}")
    
    if len(graphs) > 3:
        st.info(f"➕ {len(graphs) - 3} more chart(s) available in Charts Analysis page")


def _render_tables_preview(tables):
    """Render tables preview"""
    st.markdown(f"**Found {len(tables)} table(s)**")
    
    # Show first 2 tables
    for i, table in enumerate(tables[:2]):
        with st.expander(f"📋 Table {i+1} (Page {table.page})", expanded=i == 0):
            if hasattr(table, 'data') and table.data is not None:
                try:
                    df = pd.DataFrame(table.data)
                    st.dataframe(df.head(5), use_container_width=True)
                    if len(df) > 5:
                        st.caption(f"Showing 5 of {len(df)} rows")
                except Exception:
                    st.warning("Could not display table data")
            else:
                st.info("Table detected but data extraction pending")
    
    if len(tables) > 2:
        st.info(f"➕ {len(tables) - 2} more table(s) available in Tables page")

