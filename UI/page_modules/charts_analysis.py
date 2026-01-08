"""
Charts Analysis page for DocXtract Dashboard
"""

import streamlit as st
import pandas as pd
from PIL import Image
from collections import Counter


def render_charts_analysis():
    """Render the charts analysis page"""
    st.markdown('<h1 class="main-header">📈 Charts Analysis</h1>', unsafe_allow_html=True)
    st.markdown("Explore and analyze extracted charts from your PDF")
    
    # Check if PDF has been processed
    if 'result' not in st.session_state or st.session_state.result is None:
        st.info("📤 Please upload and process a PDF file first from the Home page.")
        return
    
    result = st.session_state.result
    
    if not result.graphs:
        _render_no_data_message()
        return
    
    # Render summary metrics
    _render_summary_metrics(result.graphs)
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Chart Gallery", "📋 Metadata Table", "📈 Chart Distribution"])
    
    with tab1:
        _render_chart_gallery(result)
    
    with tab2:
        _render_chart_metadata_table(result.graphs)
    
    with tab3:
        _render_chart_type_distribution(result.graphs)


def _render_summary_metrics(graphs):
    """Render summary metrics for charts"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Charts", len(graphs))
    
    with col2:
        # Count unique pages with charts
        unique_pages = len(set(g.page for g in graphs))
        st.metric("Pages with Charts", unique_pages)
    
    with col3:
        # Average confidence
        avg_confidence = sum(g.confidence for g in graphs) / len(graphs) if graphs else 0
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    with col4:
        # Count charts with extracted data
        charts_with_data = sum(1 for g in graphs if g.data and len(g.data) > 0)
        st.metric("Charts with Data", charts_with_data)


def _render_chart_gallery(result):
    """Render the chart gallery view"""
    st.markdown('<p class="sub-header">🖼️ Chart Gallery</p>', unsafe_allow_html=True)
    
    # Group charts by page
    pages = sorted(set(graph.page for graph in result.graphs))
    
    for page_num in pages:
        page_graphs = [g for g in result.graphs if g.page == page_num]
        
        with st.expander(f"📄 Page {page_num} ({len(page_graphs)} chart(s))", expanded=True):
            cols = st.columns(min(3, len(page_graphs)))
            
            for i, graph in enumerate(page_graphs):
                with cols[i % len(cols)]:
                    _render_chart_card(graph, i)


def _render_chart_card(graph, index):
    """Render a single chart card"""
    chart_type = graph.graph_type.name.replace('_', ' ').title()
    has_data = graph.data is not None and len(graph.data) > 0
    
    st.markdown(f"""
    <div class="chart-card">
        <strong>{chart_type}</strong><br>
        <small>Page {graph.page} • Confidence: {graph.confidence:.1%}</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Display chart image
    if graph.image is not None:
        try:
            img = Image.fromarray(graph.image)
            st.image(img, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not display image: {e}")
    else:
        st.info("No image available")
    
    # Display data info
    if has_data:
        data_points = len(graph.data.get('values', [])) if isinstance(graph.data, dict) else len(graph.data)
        st.caption(f"📊 {data_points} data points extracted")
        
        with st.expander("View Data"):
            if isinstance(graph.data, dict):
                st.json(graph.data)
            else:
                st.write(graph.data)
    else:
        st.caption("⚠️ No data extracted")


def _render_chart_metadata_table(graphs):
    """Render chart metadata as a table"""
    st.markdown('<p class="sub-header">📋 Chart Metadata</p>', unsafe_allow_html=True)
    
    metadata = []
    for i, graph in enumerate(graphs):
        has_data = graph.data is not None and len(graph.data) > 0
        data_points = 0
        if has_data:
            data_points = len(graph.data.get('values', [])) if isinstance(graph.data, dict) else len(graph.data)
        
        metadata.append({
            'Chart #': i + 1,
            'Type': graph.graph_type.name.replace('_', ' ').title(),
            'Page': graph.page,
            'Confidence': f"{graph.confidence:.1%}",
            'Has Data': '✅' if has_data else '❌',
            'Data Points': data_points
        })
    
    df = pd.DataFrame(metadata)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download Metadata CSV",
        csv,
        "chart_metadata.csv",
        "text/csv"
    )


def _render_chart_type_distribution(graphs):
    """Render chart type distribution"""
    st.markdown('<p class="sub-header">📈 Chart Type Distribution</p>', unsafe_allow_html=True)
    
    # Count chart types
    type_counts = Counter(g.graph_type.name for g in graphs)
    
    # Create dataframe for display
    df = pd.DataFrame([
        {'Chart Type': k.replace('_', ' ').title(), 'Count': v}
        for k, v in type_counts.items()
    ]).sort_values('Count', ascending=False)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    with col2:
        # Simple bar chart
        st.bar_chart(df.set_index('Chart Type'))


def _render_no_data_message():
    """Render message when no charts are found"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h3>📈 No Charts Found</h3>
        <p>No charts or graphs were detected in the uploaded PDF.</p>
        <p>Tips for better detection:</p>
        <ul style="text-align: left; display: inline-block;">
            <li>Ensure charts are clearly visible and not too small</li>
            <li>Use higher quality PDF scans</li>
            <li>Charts should have distinct colors and borders</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

