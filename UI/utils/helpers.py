"""
Helper functions for DocXtract Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np


def extract_table_dataframe(table):
    """Extract DataFrame from table object with multiple fallback methods and proper error handling"""
    try:
        # Method 1: Use built-in to_dataframe method if available
        if hasattr(table, 'to_dataframe'):
            try:
                df = table.to_dataframe()
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df
            except Exception as e:
                print(f"to_dataframe() failed: {e}")
        
        # Method 2: Direct df property/attribute
        if hasattr(table, 'df'):
            try:
                df = table.df
                if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                    return df
            except Exception as e:
                print(f"df property failed: {e}")
        
        # Method 3: Raw data attribute (numpy array or list)
        if hasattr(table, 'data') and table.data is not None:
            # Handle numpy arrays
            if isinstance(table.data, np.ndarray):
                if table.data.size > 0:
                    try:
                        # Ensure the array is 2D
                        if table.data.ndim == 1:
                            # Convert 1D to 2D
                            table.data = table.data.reshape(-1, 1)
                        
                        # Check for consistent row lengths
                        if table.data.ndim == 2:
                            return pd.DataFrame(table.data)
                        else:
                            # Flatten irregular arrays
                            flat_data = []
                            for row in table.data:
                                if isinstance(row, (list, np.ndarray)):
                                    flat_data.append(list(row))
                                else:
                                    flat_data.append([row])
                            
                            # Pad rows to same length
                            if flat_data:
                                max_cols = max(len(row) for row in flat_data)
                                padded_data = []
                                for row in flat_data:
                                    padded_row = list(row) + [''] * (max_cols - len(row))
                                    padded_data.append(padded_row)
                                return pd.DataFrame(padded_data)
                    except Exception as e:
                        st.warning(f"Error converting numpy array to DataFrame: {str(e)}")
                        
            # Handle lists
            elif isinstance(table.data, list) and len(table.data) > 0:
                try:
                    # Check if it's a list of lists
                    if all(isinstance(row, (list, tuple)) for row in table.data):
                        # Find max row length
                        max_cols = max(len(row) for row in table.data)
                        
                        # Pad all rows to same length
                        padded_data = []
                        for row in table.data:
                            padded_row = list(row) + [''] * (max_cols - len(row))
                            padded_data.append(padded_row)
                        
                        # Try to use first row as header if it looks like header
                        if len(padded_data) > 1:
                            try:
                                return pd.DataFrame(padded_data[1:], columns=padded_data[0])
                            except:
                                return pd.DataFrame(padded_data)
                        else:
                            return pd.DataFrame(padded_data)
                    else:
                        # Single row or mixed types
                        return pd.DataFrame([table.data])
                except Exception as e:
                    st.warning(f"Error converting list to DataFrame: {str(e)}")
        
        # Method 3: Try to convert the table object itself
        if hasattr(table, 'to_dict'):
            try:
                table_dict = table.to_dict()
                if isinstance(table_dict, dict) and 'data' in table_dict:
                    return pd.DataFrame(table_dict['data'])
                return pd.DataFrame(table_dict)
            except:
                pass
            
    except Exception as e:
        st.warning(f"Error extracting table data: {str(e)}")
    
    return None


def convert_df_to_csv(df):
    """Convert DataFrame to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')


def extract_chart_data_as_df(graph):
    """Extract chart data and convert to DataFrame with axis information"""
    try:
        # Check if graph has data attribute
        if not hasattr(graph, 'data') or not graph.data:
            return None
        
        data = graph.data
        chart_type = graph.graph_type.name
        
        # Get axis labels if available
        x_label = getattr(graph, 'x_label', None) or data.get('x_axis_label', 'X')
        y_label = getattr(graph, 'y_label', None) or data.get('y_axis_label', 'Y')
        
        # Bar Chart
        if chart_type == "BAR_CHART":
            values = data.get('values', [])
            if not values:
                return None
            
            bars = data.get('bars', [])
            categories = data.get('categories', [f'Bar {i+1}' for i in range(len(values))])
            
            if bars:
                # Create detailed DataFrame with axis info
                df_data = []
                for i, (value, bar) in enumerate(zip(values, bars)):
                    category = categories[i] if i < len(categories) else f'Bar {i+1}'
                    df_data.append({
                        'Category': category,
                        y_label: f'{value:.3f}',
                        'Height (px)': bar.get('height', 'N/A'),
                        'Position': f"x={bar.get('x', 'N/A')}" if bar.get('orientation') == 'vertical' else f"y={bar.get('y', 'N/A')}"
                    })
                return pd.DataFrame(df_data)
            else:
                # Simple DataFrame with just values and axis labels
                return pd.DataFrame({
                    x_label: [categories[i] if i < len(categories) else f'Bar {i+1}' for i in range(len(values))],
                    y_label: [f'{v:.3f}' for v in values]
                })
        
        # Line Chart
        elif chart_type == "LINE_CHART":
            points = data.get('points', [])
            if not points:
                return None
            
            # Limit points for display (but mention total)
            original_count = len(points)
            if len(points) > 50:
                step = len(points) // 50
                points = points[::step]
            
            df = pd.DataFrame({
                'Point #': [i+1 for i in range(len(points))],
                x_label: [f'{x:.4f}' for x, y in points],
                y_label: [f'{y:.4f}' for x, y in points]
            })
            
            if original_count > 50:
                st.caption(f"Showing {len(points)} of {original_count} points (sampled for display)")
            
            return df
        
        # Pie Chart
        elif chart_type == "PIE_CHART":
            slice_count = data.get('slice_count', 0)
            slice_percentages = data.get('slice_percentages', [])
            slice_angles = data.get('slice_angles', [])
            values = data.get('values', [])
            categories = data.get('categories', [f'Slice {i+1}' for i in range(slice_count)])
            
            if slice_count <= 0:
                return None
            
            # If no percentages, calculate from values or distribute evenly
            if not slice_percentages:
                if values:
                    total = sum(values)
                    slice_percentages = [v/total if total > 0 else 1.0/slice_count for v in values]
                else:
                    slice_percentages = [1.0 / slice_count] * slice_count
            
            df_data = []
            for i in range(slice_count):
                category = categories[i] if i < len(categories) else f'Slice {i+1}'
                pct = slice_percentages[i] if i < len(slice_percentages) else (1.0 / slice_count)
                angle = slice_angles[i] if i < len(slice_angles) else None
                value = values[i] if i < len(values) else None
                
                row = {
                    'Category': category,
                    'Percentage': f'{pct:.1%}',
                }
                
                if value is not None:
                    row['Value'] = f'{value:.3f}'
                
                if angle is not None:
                    row['Angle (degrees)'] = f'{angle:.1f}'
                
                df_data.append(row)
            
            return pd.DataFrame(df_data)
        
        # Scatter Plot
        elif chart_type == "SCATTER_PLOT":
            points = data.get('points', [])
            if not points:
                return None
            
            # Limit points for display
            original_count = len(points)
            if len(points) > 100:
                step = len(points) // 100
                points = points[::step]
            
            df = pd.DataFrame({
                'Point #': [i+1 for i in range(len(points))],
                x_label: [f'{x:.4f}' for x, y in points],
                y_label: [f'{y:.4f}' for x, y in points]
            })
            
            if original_count > 100:
                st.caption(f"Showing {len(points)} of {original_count} points (sampled for display)")
            
            return df
        
    except Exception as e:
        st.warning(f"Error extracting chart data: {str(e)}")
    
    return None

