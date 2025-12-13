"""
DocXtract Dashboard - Professional PDF Chart Analysis Tool
"""

import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import tempfile
import os
from pathlib import Path
import time
import json
import io

# Import DocXtract
try:
    from docxtract import read_pdf
except ImportError:
    st.error("❌ DocXtract library not found. Please install it first.")
    st.stop()

# Configure page
st.set_page_config(
    page_title="DocXtract Pro - PDF Chart Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
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
""", unsafe_allow_html=True)

# Helper function to safely extract DataFrame from table object
def extract_table_dataframe(table):
    """Extract DataFrame from table object with multiple fallback methods"""
    try:
        # Method 1: Direct df attribute
        if hasattr(table, 'df') and table.df is not None:
            if isinstance(table.df, pd.DataFrame) and not table.df.empty:
                return table.df
        
        # Method 2: Raw data attribute
        if hasattr(table, 'data') and table.data:
            if isinstance(table.data, list) and len(table.data) > 0:
                # Try to convert list to DataFrame
                try:
                    return pd.DataFrame(table.data[1:], columns=table.data[0] if len(table.data) > 1 else None)
                except:
                    return pd.DataFrame(table.data)
        
        # Method 3: Try to convert the table object itself
        if hasattr(table, 'to_dict'):
            return pd.DataFrame(table.to_dict())
            
    except Exception as e:
        st.warning(f"Error extracting table data: {str(e)}")
    
    return None

# Helper function to create downloadable CSV
def convert_df_to_csv(df):
    """Convert DataFrame to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-header">📊 DocXtract Pro</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigation",
        ["🏠 Home", "📈 Charts Analysis", "📋 Tables", "🧪 Testing", "ℹ️ About"],
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
            st.metric("Tables", len(result.tables))

    st.markdown("---")
    st.markdown("**Built with:** Streamlit & DocXtract")

# Main content based on navigation
if page == "🏠 Home":
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
            # Process the PDF
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
                    status_text.text("📊 Detecting charts and tables...")
                    progress_bar.progress(60)

                    result = read_pdf(pdf_path, pages='all', flavor='lattice')
                    st.session_state.result = result
                    st.session_state.pdf_path = pdf_path
                    st.session_state.filename = uploaded_file.name

                    status_text.text("✅ Analysis complete!")
                    progress_bar.progress(100)

                    time.sleep(0.5)

                    st.success(f"🎉 Successfully analyzed **{uploaded_file.name}**!")
                    st.markdown(f"**Found:** {len(result.graphs)} charts, {len(result.tables)} tables")

                    # Show quick results
                    if result.graphs:
                        st.markdown("### 📊 Detected Charts")
                        chart_types = [g.graph_type.value.replace('_', ' ').title() for g in result.graphs]
                        type_counts = pd.Series(chart_types).value_counts()

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Charts", len(result.graphs))
                        with col2:
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

                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()
                finally:
                    # Clean up temporary file
                    if os.path.exists(pdf_path):
                        try:
                            os.unlink(pdf_path)
                        except:
                            pass
    else:
        # Welcome message
        st.markdown("---")
        st.markdown("### ✨ Features")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 🎯 Chart Detection")
            st.write("Automatically detect and classify various chart types with confidence scores")
        with col2:
            st.markdown("#### 📋 Table Extraction")
            st.write("Extract tables as structured data with pandas DataFrames")
        with col3:
            st.markdown("#### 📈 Visual Analytics")
            st.write("Interactive charts and downloadable insights")

        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. **Upload** a PDF file using the uploader above
        2. **Click** "Analyze PDF" to start processing
        3. **Explore** the results in the Charts and Tables tabs
        4. **Download** extracted data as needed
        """)

elif page == "📈 Charts Analysis":
    st.markdown('<h2 class="sub-header">📈 Chart Analysis</h2>', unsafe_allow_html=True)

    if 'result' not in st.session_state:
        st.info("👆 Please upload and analyze a PDF first from the Home tab.")
    else:
        result = st.session_state.result

        if not result.graphs:
            st.warning("⚠️ No charts detected in this PDF.")
        else:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{len(result.graphs)}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Total Charts</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                chart_types = [g.graph_type.value.replace('_', ' ').title() for g in result.graphs]
                most_common = max(set(chart_types), key=chart_types.count) if chart_types else "N/A"
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{most_common}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Most Common Type</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                avg_confidence = sum(g.confidence for g in result.graphs) / len(result.graphs)
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{avg_confidence:.1%}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Avg Confidence</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col4:
                pages_with_charts = len(set(g.page for g in result.graphs))
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{pages_with_charts}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Pages with Charts</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Chart details table
            st.markdown("### 📋 Chart Metadata Table")
            chart_data = []
            for i, graph in enumerate(result.graphs):
                chart_data.append({
                    'Chart #': i + 1,
                    'Type': graph.graph_type.value.replace('_', ' ').title(),
                    'Page': graph.page,
                    'Confidence': f"{graph.confidence:.1%}",
                    'Width': f"{graph.bbox.width:.0f}px",
                    'Height': f"{graph.bbox.height:.0f}px",
                    'Area': f"{graph.bbox.area:.0f}px²",
                    'X': f"{graph.bbox.x:.0f}",
                    'Y': f"{graph.bbox.y:.0f}"
                })

            df_charts = pd.DataFrame(chart_data)
            st.dataframe(df_charts, use_container_width=True, hide_index=True)
            
            # Download button for chart metadata
            csv_charts = convert_df_to_csv(df_charts)
            st.download_button(
                label="📥 Download Chart Metadata as CSV",
                data=csv_charts,
                file_name="chart_metadata.csv",
                mime="text/csv",
            )

            # Chart type distribution
            st.markdown("### 📊 Chart Type Distribution")
            type_counts = pd.Series(chart_types).value_counts()
            st.bar_chart(type_counts)

            # Chart gallery with images
            st.markdown("### 🖼️ Chart Gallery")
            pages = sorted(set(graph.page for graph in result.graphs))

            for page_num in pages:
                st.markdown(f"#### Page {page_num}")
                page_graphs = [g for g in result.graphs if g.page == page_num]

                # Create responsive grid
                n_charts = len(page_graphs)
                if n_charts == 1:
                    cols = st.columns(1)
                elif n_charts == 2:
                    cols = st.columns(2)
                else:
                    cols = st.columns(3)

                for i, graph in enumerate(page_graphs):
                    with cols[i % len(cols)]:
                        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                        chart_type = graph.graph_type.value.replace('_', ' ').title()
                        st.markdown(f"**Chart {i+1}: {chart_type}**")

                        # Display image
                        try:
                            if graph.image is not None and graph.image.size > 0:
                                if len(graph.image.shape) == 3:
                                    img = Image.fromarray(graph.image)
                                else:
                                    img = Image.fromarray(graph.image).convert('RGB')
                                st.image(img, use_container_width=True, caption=f"Chart {i+1}")
                            else:
                                st.warning(f"Chart {i+1}: Image data not available")
                        except Exception as e:
                            st.error(f"Error displaying chart {i+1}: {str(e)}")

                        # Metadata
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.caption(f"Confidence: {graph.confidence:.1%}")
                        with col_b:
                            st.caption(f"Size: {graph.bbox.width:.0f}×{graph.bbox.height:.0f}")

                        st.markdown('</div>', unsafe_allow_html=True)

elif page == "📋 Tables":
    st.markdown('<h2 class="sub-header">📋 Chart Data Tables</h2>', unsafe_allow_html=True)
    st.info("💡 This section shows data extracted from detected charts as structured tables")

    if 'result' not in st.session_state:
        st.info("👆 Please upload and analyze a PDF first from the Home tab.")
    else:
        result = st.session_state.result

        # Check if we have charts with data
        if not result.graphs:
            st.warning("⚠️ No charts detected in this PDF. Upload a PDF with charts to see extracted data.")
        else:
            # Extract chart data into tables
            charts_with_data = []
            total_data_points = 0
            
            for i, graph in enumerate(result.graphs):
                chart_data = {
                    'index': i,
                    'graph': graph,
                    'data_table': None,
                    'has_data': False
                }
                
                # Try to extract data from the graph object
                try:
                    # Check for various data attributes
                    if hasattr(graph, 'data') and graph.data is not None:
                        # If data exists, convert to DataFrame
                        if isinstance(graph.data, dict):
                            df = pd.DataFrame(graph.data)
                            chart_data['data_table'] = df
                            chart_data['has_data'] = True
                            total_data_points += len(df)
                        elif isinstance(graph.data, list):
                            df = pd.DataFrame(graph.data)
                            chart_data['data_table'] = df
                            chart_data['has_data'] = True
                            total_data_points += len(df)
                    
                    # Alternative: extract data from image using OCR or other methods
                    # This is where you could add custom data extraction logic
                    
                except Exception as e:
                    st.warning(f"Could not extract data from chart {i+1}: {str(e)}")
                
                charts_with_data.append(chart_data)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{len(result.graphs)}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Total Charts</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                charts_with_extracted_data = sum(1 for c in charts_with_data if c['has_data'])
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{charts_with_extracted_data}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">With Data</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{total_data_points}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Data Points</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col4:
                pages_with_charts = len(set(g.page for g in result.graphs))
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{pages_with_charts}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Pages</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Chart data summary table
            st.markdown("### 📊 Chart Data Summary")
            summary_data = []
            for item in charts_with_data:
                i = item['index']
                graph = item['graph']
                df = item['data_table']
                
                row_count = len(df) if df is not None else 0
                col_count = len(df.columns) if df is not None else 0
                
                summary_data.append({
                    'Chart #': i + 1,
                    'Type': graph.graph_type.value.replace('_', ' ').title(),
                    'Page': graph.page,
                    'Data Points': row_count,
                    'Columns': col_count,
                    'Status': '✅ Data available' if row_count > 0 else '⚠️ No data extracted',
                    'Confidence': f"{graph.confidence:.1%}"
                })
            
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
            
            # Download summary
            csv_summary = convert_df_to_csv(df_summary)
            st.download_button(
                label="📥 Download Summary as CSV",
                data=csv_summary,
                file_name="chart_data_summary.csv",
                mime="text/csv"
            )

            # Display extracted chart data
            st.markdown("### 📊 Extracted Chart Data Tables")
            
            # Check if any charts have data
            if charts_with_extracted_data > 0:
                pages = sorted(set(item['graph'].page for item in charts_with_data))
                
                for page_num in pages:
                    st.markdown(f"#### Page {page_num}")
                    page_charts = [item for item in charts_with_data if item['graph'].page == page_num]
                    
                    for item in page_charts:
                        i = item['index']
                        graph = item['graph']
                        df = item['data_table']
                        
                        chart_type = graph.graph_type.value.replace('_', ' ').title()
                        
                        with st.expander(f"📊 Chart {i+1}: {chart_type} - Page {page_num}", expanded=True):
                            col_a, col_b = st.columns([2, 1])
                            with col_a:
                                st.markdown(f"**Type:** {chart_type}")
                                st.markdown(f"**Confidence:** {graph.confidence:.1%}")
                            with col_b:
                                st.markdown(f"**Page:** {page_num}")
                                st.markdown(f"**Size:** {graph.bbox.width:.0f}×{graph.bbox.height:.0f}px")
                            
                            # Show chart image
                            if graph.image is not None:
                                try:
                                    if len(graph.image.shape) == 3:
                                        img = Image.fromarray(graph.image)
                                    else:
                                        img = Image.fromarray(graph.image).convert('RGB')
                                    st.image(img, caption=f"Chart {i+1}", use_container_width=True)
                                except:
                                    pass
                            
                            st.markdown("---")
                            
                            if df is not None and not df.empty:
                                st.markdown("✅ **Extracted Data Table:**")
                                st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                                
                                # Display the data table
                                st.dataframe(df, use_container_width=True)
                                
                                # Download button
                                csv = convert_df_to_csv(df)
                                st.download_button(
                                    label=f"📥 Download Chart {i+1} Data as CSV",
                                    data=csv,
                                    file_name=f"chart_{i+1}_data_page_{page_num}.csv",
                                    mime="text/csv",
                                    key=f"download_chart_data_{i}"
                                )
                            else:
                                st.warning("⚠️ **Data extraction not available**")
                                st.markdown("""
                                **Why data might not be available:**
                                - DocXtract may not have extracted the underlying data values
                                - The chart might be image-based without embedded data
                                - OCR would be needed to read text values from the chart
                                - Manual data entry or specialized tools may be required
                                
                                **Alternative approaches:**
                                - Use specialized chart digitization tools
                                - Manually extract data points from the chart image
                                - Check if the source document has data tables alongside charts
                                """)
                                
                                # Show available attributes
                                st.markdown("**Available chart attributes:**")
                                attrs = []
                                for attr in dir(graph):
                                    if not attr.startswith('_'):
                                        try:
                                            value = getattr(graph, attr)
                                            if not callable(value):
                                                attrs.append(f"- `{attr}`: {type(value).__name__}")
                                        except:
                                            pass
                                
                                if attrs:
                                    for attr in attrs[:10]:  # Show first 10 attributes
                                        st.text(attr)
            else:
                st.info("📝 **No chart data was automatically extracted**")
                st.markdown("""
                ### 💡 What does this mean?
                
                DocXtract has detected and classified the charts in your PDF, but the underlying data points 
                (the actual numbers/values shown in the charts) are not automatically extracted by default.
                
                ### 🔧 Options to extract chart data:
                
                1. **Check the original source**: Look for data tables in the PDF that correspond to the charts
                2. **Manual extraction**: Read values from the chart images manually
                3. **OCR + Image Processing**: Use specialized tools like:
                   - WebPlotDigitizer (online tool)
                   - Engauge Digitizer
                   - Custom OCR solutions
                4. **Request source data**: Contact the document author for the raw data
                
                ### 📊 What DocXtract provides:
                - ✅ Chart detection and classification
                - ✅ Chart type identification (bar, line, pie, etc.)
                - ✅ Confidence scores
                - ✅ Bounding boxes and locations
                - ✅ Chart images extracted as arrays
                - ❌ Automatic data point extraction (not included)
                
                **Tip:** The Charts Analysis tab shows all detected chart metadata and images.
                """)
                
                # Show detected charts summary
                st.markdown("### 📋 Detected Charts (without data)")
                chart_summary = []
                for item in charts_with_data:
                    chart_summary.append({
                        'Chart #': item['index'] + 1,
                        'Type': item['graph'].graph_type.value.replace('_', ' ').title(),
                        'Page': item['graph'].page,
                        'Confidence': f"{item['graph'].confidence:.1%}"
                    })
                
                df_no_data = pd.DataFrame(chart_summary)
                st.dataframe(df_no_data, use_container_width=True, hide_index=True)

elif page == "🧪 Testing":
    st.markdown('<h2 class="sub-header">🧪 Dataset Testing</h2>', unsafe_allow_html=True)

    st.markdown("### Test Generated PDFs from DatasetGen")
    st.info("ℹ️ **Note:** Generated PDFs are designed for chart detection. Table detection may show 'structure only' results.")

    charts_dir = "chart_dataset/charts"
    metadata_file = "chart_dataset/metadata.json"

    if not os.path.exists(charts_dir):
        st.warning("⚠️ Chart dataset not found. Run DatasetGen notebook first.")
    else:
        metadata = None
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                st.success(f"📊 Found dataset with {metadata.get('total_charts', 0)} generated charts")
            except:
                pass

        pdf_files = [f for f in os.listdir(charts_dir) if f.endswith('.pdf')]
        st.info(f"📁 Found {len(pdf_files)} PDF files")

        if st.button("🚀 Run Tests on All PDFs", type="primary"):
            with st.spinner("🔄 Testing PDFs..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                test_results = []
                total_files = len(pdf_files)

                for i, pdf_file in enumerate(pdf_files):
                    status_text.text(f"Testing {pdf_file}... ({i+1}/{total_files})")
                    progress_bar.progress((i) / total_files)

                    pdf_path = os.path.join(charts_dir, pdf_file)

                    try:
                        result = read_pdf(pdf_path)

                        test_data = {
                            'filename': pdf_file,
                            'charts_found': len(result.graphs),
                            'tables_found': len(result.tables),
                            'charts': [],
                            'tables': [],
                            'table_dataframes': [],
                            'status': 'success'
                        }

                        # Chart details
                        for graph in result.graphs:
                            test_data['charts'].append({
                                'type': graph.graph_type.value.replace('_', ' ').title(),
                                'confidence': round(graph.confidence, 3),
                                'page': graph.page
                            })

                        # Table details
                        for table in result.tables:
                            df = extract_table_dataframe(table)
                            has_data = df is not None and not df.empty
                            shape = df.shape if df is not None else (0, 0)
                            
                            test_data['tables'].append({
                                'page': table.page,
                                'shape': shape,
                                'has_data': has_data
                            })
                            test_data['table_dataframes'].append(df)

                    except Exception as e:
                        test_data = {
                            'filename': pdf_file,
                            'status': 'error',
                            'error': str(e),
                            'charts_found': 0,
                            'tables_found': 0
                        }

                    test_results.append(test_data)

                progress_bar.progress(1.0)
                status_text.text("✅ Testing complete!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()

                st.session_state.test_results = test_results

        # Display results
        if 'test_results' in st.session_state:
            test_results = st.session_state.test_results

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{len(test_results)}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Total PDFs</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                successful = sum(1 for r in test_results if r.get('status') == 'success')
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{successful}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Successful</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                total_charts = sum(r.get('charts_found', 0) for r in test_results)
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{total_charts}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Charts Detected</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col4:
                total_tables = sum(r.get('tables_found', 0) for r in test_results)
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{total_tables}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Tables Detected</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Test results table
            st.markdown("### 📋 Test Results Summary")
            results_data = []
            for result in test_results:
                status_emoji = "✅" if result.get('status') == 'success' else "❌"
                results_data.append({
                    'File': result['filename'],
                    'Status': f"{status_emoji} {result.get('status', 'unknown').title()}",
                    'Charts': result.get('charts_found', 0),
                    'Tables': result.get('tables_found', 0),
                    'Error': result.get('error', '')[:50] if result.get('error') else ''
                })

            df_results = pd.DataFrame(results_data)
            st.dataframe(df_results, use_container_width=True, hide_index=True)
            
            # Download test results
            csv_results = convert_df_to_csv(df_results)
            st.download_button(
                label="📥 Download Test Results as CSV",
                data=csv_results,
                file_name="test_results.csv",
                mime="text/csv"
            )

            # Charts breakdown
            st.markdown("### 📊 Charts Breakdown")
            successful_results = [r for r in test_results if r.get('status') == 'success']

            if successful_results:
                chart_details = []
                for result in successful_results:
                    for chart in result.get('charts', []):
                        chart_details.append({
                            'File': result['filename'],
                            'Type': chart['type'],
                            'Confidence': f"{chart['confidence']:.1%}",
                            'Page': chart['page']
                        })

                if chart_details:
                    df_chart_details = pd.DataFrame(chart_details)
                    st.dataframe(df_chart_details, use_container_width=True, hide_index=True)
                    
                    csv_chart_details = convert_df_to_csv(df_chart_details)
                    st.download_button(
                        label="📥 Download Chart Details as CSV",
                        data=csv_chart_details,
                        file_name="chart_details.csv",
                        mime="text/csv"
                    )

            # Extracted tables section
            st.markdown("### 📊 Extracted Tables from Tests")
            tables_found = [r for r in test_results if r.get('status') == 'success' and r.get('tables', [])]

            if tables_found:
                for result in tables_found:
                    filename = result['filename']
                    tables = result.get('tables', [])
                    table_dfs = result.get('table_dataframes', [])

                    if tables:
                        with st.expander(f"📋 {filename} - {len(tables)} table(s)", expanded=False):
                            for i, (table_info, df) in enumerate(zip(tables, table_dfs)):
                                st.markdown(f"**Table {i+1}** (Page {table_info['page']})")
                                
                                if table_info['has_data'] and df is not None and not df.empty:
                                    st.markdown("✅ **Data extracted successfully**")
                                    st.caption(f"Shape: {table_info['shape'][0]} rows × {table_info['shape'][1]} columns")
                                    
                                    # Display table
                                    st.dataframe(df, use_container_width=True)
                                    
                                    # Download button
                                    csv = convert_df_to_csv(df)
                                    st.download_button(
                                        label=f"📥 Download as CSV",
                                        data=csv,
                                        file_name=f"{filename.replace('.pdf', '')}_table_{i+1}.csv",
                                        mime="text/csv",
                                        key=f"test_download_{filename}_{i}"
                                    )
                                else:
                                    st.warning("⚠️ Structure detected, no data extracted")
                                    st.caption(f"Detected: {table_info['shape'][0]} rows × {table_info['shape'][1]} columns")
                                
                                if i < len(tables) - 1:
                                    st.markdown("---")
            else:
                st.info("No tables extracted from test PDFs")

            # Export all results as JSON
            st.markdown("### 💾 Export All Results")
            if st.button("📥 Download Complete Results as JSON"):
                json_data = json.dumps(test_results, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="complete_test_results.json",
                    mime="application/json"
                )