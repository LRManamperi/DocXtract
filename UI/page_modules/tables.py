"""
Tables page for DocXtract Dashboard - Extract and download table data
"""

import streamlit as st
import pandas as pd
from PIL import Image
import io

# Handle imports
try:
    from UI.utils.helpers import extract_table_dataframe, convert_df_to_csv
except ImportError:
    try:
        from ..utils.helpers import extract_table_dataframe, convert_df_to_csv
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from UI.utils.helpers import extract_table_dataframe, convert_df_to_csv


def render_tables():
    """Render the tables extraction and data page"""
    st.markdown('<h1 class="main-header">📋 Extracted Tables</h1>', unsafe_allow_html=True)
    st.markdown("View and download extracted table data from your PDF")
    
    # Check if PDF has been processed
    if 'result' not in st.session_state or st.session_state.result is None:
        st.info("📤 Please upload and process a PDF file first from the Home page.")
        return
    
    result = st.session_state.result
    
    if not result.tables:
        _render_no_data_message()
        return
    
    # Render summary metrics
    _render_summary_metrics(result.tables)
    
    st.markdown("---")
    
    # Render extracted tables
    _render_extracted_tables(result.tables)


def _render_summary_metrics(tables):
    """Render summary metrics for tables"""
    col1, col2, col3, col4 = st.columns(4)
    
    # Count tables with extractable data
    tables_with_data = 0
    total_rows = 0
    total_cols = 0
    
    for table in tables:
        df = extract_table_dataframe(table)
        if df is not None and not df.empty:
            tables_with_data += 1
            total_rows += len(df)
            total_cols += len(df.columns)
    
    with col1:
        st.metric("Total Tables", len(tables))
    with col2:
        st.metric("Tables with Data", tables_with_data)
    with col3:
        st.metric("Total Rows", total_rows)
    with col4:
        avg_cols = total_cols // tables_with_data if tables_with_data > 0 else 0
        st.metric("Avg Columns", avg_cols)


def _render_extracted_tables(tables):
    """Render all extracted tables with download options"""
    st.markdown('<p class="sub-header">📊 Table Data</p>', unsafe_allow_html=True)
    
    # Table selection
    table_options = [f"Table {i+1} (Page {t.page})" for i, t in enumerate(tables)]
    
    # View mode selection
    view_mode = st.radio(
        "View Mode",
        ["Individual Tables", "All Tables"],
        horizontal=True
    )
    
    if view_mode == "Individual Tables":
        selected_table = st.selectbox("Select Table", table_options)
        table_idx = table_options.index(selected_table)
        _render_single_table(tables[table_idx], table_idx)
    else:
        # Show all tables
        for i, table in enumerate(tables):
            with st.expander(f"📋 Table {i+1} (Page {table.page})", expanded=i == 0):
                _render_single_table(table, i)


def _render_single_table(table, index):
    """Render a single table with its data and download options"""
    df = extract_table_dataframe(table)
    
    if df is not None and not df.empty:
        # Table info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"📄 Page: {table.page}")
        with col2:
            st.info(f"📏 Rows: {len(df)}")
        with col3:
            st.info(f"📐 Columns: {len(df.columns)}")
        
        # Display dataframe
        st.dataframe(df, use_container_width=True, height=300)
        
        # Download options
        col1, col2 = st.columns(2)
        with col1:
            csv = convert_df_to_csv(df)
            st.download_button(
                "📥 Download as CSV",
                csv,
                file_name=f"table_{index+1}_page_{table.page}.csv",
                mime="text/csv",
                key=f"csv_download_{index}"
            )
        with col2:
            # Excel download
            try:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name=f'Table_{index+1}')
                excel_data = output.getvalue()
                st.download_button(
                    "📥 Download as Excel",
                    excel_data,
                    file_name=f"table_{index+1}_page_{table.page}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"excel_download_{index}"
                )
            except Exception:
                st.caption("Excel export requires openpyxl package")
        
        # Show table image if available
        if hasattr(table, 'image') and table.image is not None:
            with st.expander("🖼️ View Table Image"):
                st.image(table.image, caption=f"Table {index+1} from Page {table.page}")
    else:
        st.warning("⚠️ Could not extract data from this table")
        # Show image if available even when data extraction fails
        if hasattr(table, 'image') and table.image is not None:
            st.image(table.image, caption=f"Table {index+1} from Page {table.page}")


def _render_no_data_message():
    """Render message when no tables are found"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h3>📋 No Tables Found</h3>
        <p>No tables were detected in the uploaded PDF.</p>
        <p>Tips for better table detection:</p>
        <ul style="text-align: left; display: inline-block;">
            <li>Ensure tables have clear borders or grid lines</li>
            <li>Use higher quality PDF scans</li>
            <li>Tables should be clearly separated from surrounding text</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
