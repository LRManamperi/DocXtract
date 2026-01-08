"""
Testing page for DocXtract Dashboard
"""

import streamlit as st
import pandas as pd
import os
import time

from docxtract import read_pdf

# Handle imports - support both relative and absolute
try:
    from UI.utils.helpers import extract_chart_data_as_df, convert_df_to_csv
except ImportError:
    try:
        from ..utils.helpers import extract_chart_data_as_df, convert_df_to_csv
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from UI.utils.helpers import extract_chart_data_as_df, convert_df_to_csv


def render_testing():
    """Render the testing page"""
    st.markdown('<h2 class="sub-header">🧪 Dataset Testing</h2>', unsafe_allow_html=True)

    st.markdown("### Test PDFs with Chart Detection")

    # Get the project root directory (parent of UI directory)
    import sys
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_file_dir))
    
    # Tab selection for different test sets
    tab1, tab2 = st.tabs(["📁 Chart Dataset", "🧪 Generated Test PDFs"])
    
    with tab1:
        charts_dir = os.path.join(project_root, "chart_dataset", "charts")
        metadata_file = os.path.join(project_root, "chart_dataset", "metadata.json")

        if not os.path.exists(charts_dir):
            st.warning(f"⚠️ Chart dataset not found at: {charts_dir}")
            st.info("💡 The chart dataset directory should be at: `chart_dataset/charts/` relative to the project root")
        else:
            pdf_files = [f for f in os.listdir(charts_dir) if f.endswith('.pdf')]
            st.info(f"📁 Found {len(pdf_files)} PDF files in {charts_dir}")

            if st.button("🚀 Run Tests on All PDFs", type="primary", key="test_dataset"):
                _run_tests(charts_dir, pdf_files)
    
    with tab2:
        st.markdown("#### Generated Test PDFs")
        st.info("These PDFs were generated using `generate_test_pdf.py` and contain various chart and table types for testing.")
        
        # Define test PDFs
        test_pdfs = [
            {
                'name': 'test_charts.pdf',
                'description': 'Bar, line, pie, and scatter charts',
                'pages': 4,
                'contains': ['Bar Chart', 'Line Chart', 'Pie Chart', 'Scatter Plot']
            },
            {
                'name': 'test_tables.pdf',
                'description': 'Structured and unstructured tables',
                'pages': 3,
                'contains': ['Grid Table', 'Space-separated Table', 'Mixed Table']
            },
            {
                'name': 'test_combined.pdf',
                'description': 'Mixed tables and charts',
                'pages': 5,
                'contains': ['Financial Table', 'Bar Chart', 'Line Chart', 'Inventory Table']
            }
        ]
        
        # Display available test PDFs
        for pdf_info in test_pdfs:
            pdf_path = os.path.join(project_root, pdf_info['name'])
            exists = os.path.exists(pdf_path)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                status_icon = "✅" if exists else "❌"
                st.markdown(f"**{status_icon} {pdf_info['name']}**")
                st.caption(f"{pdf_info['description']} • {pdf_info['pages']} pages")
                st.caption(f"Contains: {', '.join(pdf_info['contains'])}")
            with col2:
                if exists:
                    if st.button("Test", key=f"test_{pdf_info['name']}", type="secondary"):
                        _run_tests(project_root, [pdf_info['name']])
                else:
                    st.caption("Not found")
        
        st.markdown("---")
        
        # Generate test PDFs if missing
        missing_pdfs = [pdf['name'] for pdf in test_pdfs if not os.path.exists(os.path.join(project_root, pdf['name']))]
        
        if missing_pdfs:
            st.warning(f"⚠️ {len(missing_pdfs)} test PDF(s) not found: {', '.join(missing_pdfs)}")
            st.info("Run `python generate_test_pdf.py` to create test PDFs")
        
        # Test all generated PDFs
        all_test_pdfs = [pdf['name'] for pdf in test_pdfs if os.path.exists(os.path.join(project_root, pdf['name']))]
        if all_test_pdfs:
            if st.button("🚀 Test All Generated PDFs", type="primary", key="test_all_generated"):
                _run_tests(project_root, all_test_pdfs)

    # Display results
    if 'test_results' in st.session_state:
        _render_test_results()


def _run_tests(charts_dir, pdf_files):
    """Run tests on all PDF files"""
    with st.spinner("🔄 Testing PDFs..."):
        progress_bar = st.progress(0)
        status_text = st.empty()

        test_results = []
        total_files = len(pdf_files)

        for i, pdf_file in enumerate(pdf_files):
            status_text.text(f"Testing {pdf_file}... ({i+1}/{total_files})")
            progress_bar.progress(i / total_files)

            pdf_path = os.path.join(charts_dir, pdf_file)

            try:
                result = read_pdf(pdf_path)

                test_data = {
                    'filename': pdf_file,
                    'charts_found': len(result.graphs),
                    'charts': [],
                    'chart_dataframes': [],
                    'status': 'success'
                }

                # Chart details and data extraction
                for graph in result.graphs:
                    test_data['charts'].append({
                        'type': graph.graph_type.value.replace('_', ' ').title(),
                        'confidence': round(graph.confidence, 3),
                        'page': graph.page,
                        'has_data': bool(hasattr(graph, 'data') and graph.data and graph.data.get('values'))
                    })
                    
                    # Extract chart data
                    df = extract_chart_data_as_df(graph)
                    test_data['chart_dataframes'].append(df)

            except Exception as e:
                test_data = {
                    'filename': pdf_file,
                    'status': 'error',
                    'error': str(e),
                    'charts_found': 0
                }

            test_results.append(test_data)

        progress_bar.progress(1.0)
        status_text.text("✅ Testing complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()

        st.session_state.test_results = test_results


def _render_test_results():
    """Render test results"""
    test_results = st.session_state.test_results

    # Summary metrics
    _render_test_summary_metrics(test_results)
    
    # Test results table
    _render_test_results_table(test_results)
    
    # Extracted chart data
    _render_extracted_chart_data(test_results)


def _render_test_summary_metrics(test_results):
    """Render test summary metrics"""
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
        charts_with_data = sum(
            sum(1 for c in r.get('charts', []) if c.get('has_data'))
            for r in test_results if r.get('status') == 'success'
        )
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{charts_with_data}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">With Data</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def _render_test_results_table(test_results):
    """Render test results table"""
    st.markdown("### 📋 Test Results Summary")
    results_data = []
    for result in test_results:
        status_emoji = "✅" if result.get('status') == 'success' else "❌"
        charts_with_data = sum(1 for c in result.get('charts', []) if c.get('has_data'))
        
        results_data.append({
            'File': result['filename'],
            'Status': f"{status_emoji} {result.get('status', 'unknown').title()}",
            'Charts': result.get('charts_found', 0),
            'With Data': charts_with_data,
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


def _render_extracted_chart_data(test_results):
    """Render extracted chart data"""
    st.markdown("### 📊 Extracted Chart Data")
    successful_results = [r for r in test_results if r.get('status') == 'success']

    if successful_results:
        for result in successful_results:
            filename = result['filename']
            charts = result.get('charts', [])
            chart_dfs = result.get('chart_dataframes', [])
            
            charts_with_data_in_file = sum(1 for c in charts if c.get('has_data'))

            if charts:
                with st.expander(f"📊 {filename} - {len(charts)} chart(s), {charts_with_data_in_file} with data", expanded=False):
                    for i, (chart_info, df) in enumerate(zip(charts, chart_dfs)):
                        st.markdown(f"**Chart {i+1}** ({chart_info['type']}) - Page {chart_info['page']}")
                        st.caption(f"Confidence: {chart_info['confidence']:.1%}")
                        
                        if df is not None and not df.empty:
                            st.markdown("✅ **Data extracted**")
                            st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                            
                            # Display table
                            st.dataframe(df, use_container_width=True)
                            
                            # Download button
                            csv = convert_df_to_csv(df)
                            st.download_button(
                                label=f"📥 Download as CSV",
                                data=csv,
                                file_name=f"{filename.replace('.pdf', '')}_chart_{i+1}.csv",
                                mime="text/csv",
                                key=f"test_download_{filename}_{i}"
                            )
                        else:
                            st.warning("⚠️ No data extracted")
                        
                        if i < len(charts) - 1:
                            st.markdown("---")
    else:
        st.info("No successful extractions to display")

