"""
Charts Analysis page for DocXtract Dashboard
"""

import streamlit as st
import pandas as pd
from PIL import Image


def render_charts_analysis():
    """Render the charts analysis page"""
    st.markdown('<h2 class="sub-header">📈 Chart Analysis</h2>', unsafe_allow_html=True)

    if 'result' not in st.session_state:
        st.info("👆 Please upload and analyze a PDF first from the Home tab.")
        return

    result = st.session_state.result

    if not result.graphs:
        st.warning("⚠️ No charts detected in this PDF.")
        return

    # Summary metrics
    charts_with_data = sum(1 for g in result.graphs if hasattr(g, 'data') and g.data and g.data.get('values'))
    
    _render_summary_metrics(result, charts_with_data)
    
    # Chart details table
    _render_chart_metadata_table(result, charts_with_data)
    
    # Chart type distribution
    _render_chart_type_distribution(result)
    
    # Chart gallery
    _render_chart_gallery(result)


def _render_summary_metrics(result, charts_with_data):
    """Render summary metrics"""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(result.graphs)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Charts</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{charts_with_data}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">With Data</div>', unsafe_allow_html=True)
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


def _render_chart_metadata_table(result, charts_with_data):
    """Render chart metadata table"""
    st.markdown("### 📋 Chart Metadata Table")
    chart_data = []
    for i, graph in enumerate(result.graphs):
        has_data = hasattr(graph, 'data') and graph.data and graph.data.get('values')
        data_count = len(graph.data.get('values', [])) if has_data else 0
        
        chart_data.append({
            'Chart #': i + 1,
            'Type': graph.graph_type.value.replace('_', ' ').title(),
            'Page': graph.page,
            'Confidence': f"{graph.confidence:.1%}",
            'Data Points': data_count,
            'Status': '✅ Data' if has_data else '⚠️ No Data',
            'Size': f"{graph.bbox.width:.0f}×{graph.bbox.height:.0f}px"
        })

    df_charts = pd.DataFrame(chart_data)
    st.dataframe(df_charts, use_container_width=True, hide_index=True)
    
    # Download button for chart metadata
    try:
        from UI.utils.helpers import convert_df_to_csv
    except ImportError:
        from ..utils.helpers import convert_df_to_csv
    csv_charts = convert_df_to_csv(df_charts)
    st.download_button(
        label="📥 Download Chart Metadata as CSV",
        data=csv_charts,
        file_name="chart_metadata.csv",
        mime="text/csv",
    )


def _render_chart_type_distribution(result):
    """Render chart type distribution chart"""
    st.markdown("### 📊 Chart Type Distribution")
    chart_types = [g.graph_type.value.replace('_', ' ').title() for g in result.graphs]
    type_counts = pd.Series(chart_types).value_counts()
    st.bar_chart(type_counts)


def _render_chart_gallery(result):
    """Render chart gallery"""
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
                has_data = hasattr(graph, 'data') and graph.data and graph.data.get('values')
                
                st.markdown(f"**Chart {i+1}: {chart_type}** {'✅' if has_data else '⚠️'}")

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
                    data_status = f"Data: {len(graph.data.get('values', []))} pts" if has_data else "No Data"
                    st.caption(data_status)

                st.markdown('</div>', unsafe_allow_html=True)

