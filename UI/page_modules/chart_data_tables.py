"""
Chart Data Tables page for DocXtract Dashboard
"""

import streamlit as st
from PIL import Image

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


def render_chart_data_tables():
    """Render the chart data tables page"""
    st.markdown('<h2 class="sub-header">� Extracted Chart Data</h2>', unsafe_allow_html=True)
    st.info("💡 This section shows numerical data extracted from detected charts with axis information")

    if 'result' not in st.session_state:
        st.info("👆 Please upload and analyze a PDF first from the Home tab.")
        return

    result = st.session_state.result

    if not result.graphs:
        st.warning("⚠️ No charts detected in this PDF.")
        return

    # Extract data from all charts
    charts_with_data = []
    total_data_points = 0
    
    for i, graph in enumerate(result.graphs):
        df = extract_chart_data_as_df(graph)
        has_data = df is not None and not df.empty
        
        if has_data:
            total_data_points += len(df)
        
        charts_with_data.append({
            'index': i,
            'graph': graph,
            'df': df,
            'has_data': has_data
        })
    
    # Summary metrics
    _render_summary_metrics(result, charts_with_data, total_data_points)
    
    # Display extracted data
    num_with_data = sum(1 for c in charts_with_data if c['has_data'])
    if num_with_data > 0:
        _render_extracted_data(charts_with_data)
    else:
        _render_no_data_message()


def _render_summary_metrics(result, charts_with_data, total_data_points):
    """Render summary metrics"""
    num_with_data = sum(1 for c in charts_with_data if c['has_data'])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(result.graphs)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Charts</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{num_with_data}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">With Data</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{total_data_points}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Data Points</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        extraction_rate = (num_with_data / len(result.graphs) * 100) if result.graphs else 0
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{extraction_rate:.0f}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Extraction Rate</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def _render_extracted_data(charts_with_data):
    """Render extracted data tables"""
    st.markdown("### 📊 Extracted Data Tables")
    
    pages = sorted(set(item['graph'].page for item in charts_with_data))
    
    for page_num in pages:
        st.markdown(f"#### Page {page_num}")
        page_items = [item for item in charts_with_data if item['graph'].page == page_num]
        
        for item in page_items:
            i = item['index']
            graph = item['graph']
            df = item['df']
            has_data = item['has_data']
            
            chart_type = graph.graph_type.value.replace('_', ' ').title()
            
            with st.expander(f"{'✅' if has_data else '⚠️'} Chart {i+1}: {chart_type} (Page {page_num})", expanded=has_data):
                # Chart metadata
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.markdown(f"**Type:** {chart_type}")
                    st.markdown(f"**Confidence:** {graph.confidence:.1%}")
                    # Show axis labels if available
                    if hasattr(graph, 'x_label') and graph.x_label:
                        st.markdown(f"**X-Axis:** {graph.x_label}")
                    elif hasattr(graph, 'data') and graph.data and 'x_axis_label' in graph.data:
                        st.markdown(f"**X-Axis:** {graph.data['x_axis_label']}")
                    
                    if hasattr(graph, 'y_label') and graph.y_label:
                        st.markdown(f"**Y-Axis:** {graph.y_label}")
                    elif hasattr(graph, 'data') and graph.data and 'y_axis_label' in graph.data:
                        st.markdown(f"**Y-Axis:** {graph.data['y_axis_label']}")
                with col_b:
                    st.markdown(f"**Page:** {page_num}")
                    st.markdown(f"**Size:** {graph.bbox.width:.0f}×{graph.bbox.height:.0f}px")
                    if has_data:
                        st.markdown(f"**Data Points:** {len(df)}")
                
                # Show chart image
                if graph.image is not None:
                    try:
                        if len(graph.image.shape) == 3:
                            img = Image.fromarray(graph.image)
                        else:
                            img = Image.fromarray(graph.image).convert('RGB')
                        st.image(img, caption=f"Chart {i+1}", width=400)
                    except:
                        pass
                
                st.markdown("---")
                
                # Show extracted data
                if has_data and df is not None:
                    st.markdown("✅ **Extracted Data:**")
                    st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                    
                    st.dataframe(df, use_container_width=True)
                    
                    # Download buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # CSV download
                        csv = convert_df_to_csv(df)
                        st.download_button(
                            label=f"📥 Download as CSV",
                            data=csv,
                            file_name=f"chart_{i+1}_data_page_{page_num}.csv",
                            mime="text/csv",
                            key=f"download_csv_{i}",
                            use_container_width=True
                        )
                    
                    with col2:
                        # JSON download
                        import json
                        json_data = {
                            'chart_type': chart_type,
                            'page': page_num,
                            'confidence': float(graph.confidence),
                            'data': df.to_dict('records')
                        }
                        if hasattr(graph, 'x_label') and graph.x_label:
                            json_data['x_label'] = graph.x_label
                        if hasattr(graph, 'y_label') and graph.y_label:
                            json_data['y_label'] = graph.y_label
                        
                        json_str = json.dumps(json_data, indent=2)
                        st.download_button(
                            label=f"📥 Download as JSON",
                            data=json_str,
                            file_name=f"chart_{i+1}_data_page_{page_num}.json",
                            mime="application/json",
                            key=f"download_json_{i}",
                            use_container_width=True
                        )
                else:
                    st.warning("⚠️ No data could be extracted from this chart")
                    
                    # Show what data is available
                    if hasattr(graph, 'data') and graph.data:
                        st.markdown("**Available data attributes:**")
                        st.json({k: str(v)[:100] + '...' if len(str(v)) > 100 else str(v) 
                               for k, v in graph.data.items()})


def _render_no_data_message():
    """Render message when no data is available"""
    st.info("📝 **No chart data was extracted**")
    st.markdown("""
    ### Possible reasons:
    - Charts may be images without embedded data
    - Chart detection may have identified visual elements but not numerical values
    - Some chart types are harder to extract data from automatically
    
    ### Tip:
    Check the "Charts Analysis" tab to see all detected charts with their metadata.
    """)

