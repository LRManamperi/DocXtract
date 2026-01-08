"""
Comprehensive test for DocXtract data extraction capabilities
Tests chart data extraction, unstructured tables, and structured data output
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from docxtract.extractors import DocXtract
from docxtract.data_structures import ElementType
import json


def run_chart_data_extraction_test():
    """Test chart data extraction from various chart types (standalone script, not pytest)"""
    print("\n" + "="*70)
    print("TEST 1: Chart Data Extraction")
    print("="*70)
    
    # Create test PDF path (you'll need to provide actual test PDFs)
    test_pdf = "test_charts.pdf"  # Replace with actual test file
    
    if not os.path.exists(test_pdf):
        print(f"⚠️  Test file '{test_pdf}' not found. Skipping chart tests.")
        print("   Create a test PDF with bar charts, line charts, pie charts, and scatter plots.")
        return False
    
    try:
        # Initialize extractor with chart data extraction enabled
        extractor = DocXtract(use_ml=False, extract_chart_data=True)
        
        # Extract from PDF
        print(f"\n📄 Extracting from: {test_pdf}")
        result = extractor.extract(test_pdf, pages='all')
        
        print(f"\n📊 Found {result.n_graphs} charts")
        
        # Test each chart type
        for i, graph in enumerate(result.graphs, 1):
            print(f"\n--- Chart {i} ---")
            print(f"Type: {graph.graph_type.name}")
            print(f"Confidence: {graph.confidence:.2f}")
            print(f"Page: {graph.page}")
            print(f"Data Points: {graph.point_count}")
            
            if graph.extracted_values:
                print(f"Values: {graph.extracted_values[:5]}{'...' if len(graph.extracted_values) > 5 else ''}")
            
            if graph.data_labels:
                print(f"Labels: {graph.data_labels[:5]}{'...' if len(graph.data_labels) > 5 else ''}")
            
            if graph.legend:
                print(f"Legend: {graph.legend}")
            
            if graph.series:
                print(f"Series: {len(graph.series)}")
                for series in graph.series:
                    print(f"  - {series.name}: {len(series.values)} values")
            
            # Test DataFrame conversion
            try:
                df = graph.to_dataframe()
                print(f"DataFrame shape: {df.shape}")
                print(f"DataFrame preview:\n{df.head()}")
            except Exception as e:
                print(f"⚠️  DataFrame conversion failed: {e}")
        
        # Test filtering by type
        print(f"\n📊 Chart Type Summary:")
        print(f"  Bar Charts: {len(result.get_bar_charts())}")
        print(f"  Line Charts: {len(result.get_line_charts())}")
        print(f"  Pie Charts: {len(result.get_pie_charts())}")
        print(f"  Scatter Plots: {len(result.get_scatter_plots())}")
        
        # Export to JSON
        json_output = "test_chart_data.json"
        result.to_json(json_output)
        print(f"\n💾 Exported data to: {json_output}")
        
        print("\n✅ Chart data extraction test completed")
        return True
        
    except Exception as e:
        print(f"\n❌ Chart extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_unstructured_tables_test():
    """Test extraction of tables without clear grid lines (standalone script, not pytest)"""
    print("\n" + "="*70)
    print("TEST 2: Unstructured Table Extraction")
    print("="*70)
    
    test_pdf = "test_tables.pdf"  # Replace with actual test file
    
    if not os.path.exists(test_pdf):
        print(f"⚠️  Test file '{test_pdf}' not found. Skipping table tests.")
        print("   Create a test PDF with various table formats:")
        print("   - Tables with grid lines")
        print("   - Tables without grid lines (space-separated)")
        print("   - Tables with irregular spacing")
        return False
    
    try:
        # Initialize extractor with unstructured table handling enabled
        extractor = DocXtract(use_ml=False, handle_unstructured=True)
        
        print(f"\n📄 Extracting from: {test_pdf}")
        result = extractor.extract(test_pdf, pages='all')
        
        print(f"\n📋 Found {result.n_tables} tables")
        
        for i, table in enumerate(result.tables, 1):
            print(f"\n--- Table {i} ---")
            print(f"Page: {table.page}")
            print(f"Shape: {table.data.shape}")
            print(f"Confidence: {table.confidence:.2f}")
            print(f"Bounding Box: {table.bbox.to_dict()}")
            
            # Show table data
            print(f"\nTable Data:")
            print(table.data)
            
            # Test DataFrame conversion
            try:
                df = table.to_dataframe()
                print(f"\nDataFrame shape: {df.shape}")
                print(f"DataFrame preview:\n{df.head()}")
                
                # Export to CSV
                csv_file = f"test_table_{i}.csv"
                table.to_csv(csv_file)
                print(f"💾 Exported to: {csv_file}")
                
            except Exception as e:
                print(f"⚠️  DataFrame conversion failed: {e}")
        
        print("\n✅ Unstructured table extraction test completed")
        return True
        
    except Exception as e:
        print(f"\n❌ Table extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_combined_extraction_test():
    """Test extraction of both tables and charts from same document (standalone script, not pytest)"""
    print("\n" + "="*70)
    print("TEST 3: Combined Extraction (Tables + Charts)")
    print("="*70)
    
    test_pdf = "test_combined.pdf"  # Replace with actual test file
    
    if not os.path.exists(test_pdf):
        print(f"⚠️  Test file '{test_pdf}' not found. Skipping combined test.")
        print("   Create a test PDF with both tables and charts.")
        return False
    
    try:
        # Initialize with all features enabled
        extractor = DocXtract(
            use_ml=False,
            extract_chart_data=True,
            handle_unstructured=True
        )
        
        print(f"\n📄 Extracting from: {test_pdf}")
        result = extractor.extract(test_pdf, pages='all')
        
        # Print summary
        print(result.summary())
        
        # Export complete results
        json_output = "test_combined_results.json"
        result.to_json(json_output)
        print(f"\n💾 Complete results exported to: {json_output}")
        
        # Verify exported JSON
        with open(json_output, 'r') as f:
            data = json.load(f)
            print(f"\n📊 JSON Export Verification:")
            print(f"  Pages: {data['n_pages']}")
            print(f"  Tables: {data['n_tables']}")
            print(f"  Graphs: {data['n_graphs']}")
        
        print("\n✅ Combined extraction test completed")
        return True
        
    except Exception as e:
        print(f"\n❌ Combined extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_synthetic_data_test():
    """Test extraction on synthetic/generated data (standalone script, not pytest)"""
    print("\n" + "="*70)
    print("TEST 4: Synthetic Data Extraction")
    print("="*70)
    
    try:
        import cv2
        from docxtract.chart_extractors import ChartDataExtractor
        from docxtract.unstructured_table_parser import UnstructuredTableParser
        from docxtract.data_structures import BoundingBox
        
        print("\n🎨 Creating synthetic test images...")
        
        # Test 1: Simple bar chart
        print("\n--- Synthetic Bar Chart ---")
        bar_chart = np.ones((400, 500, 3), dtype=np.uint8) * 255
        
        # Draw bars
        bar_heights = [200, 150, 300, 250, 180]
        bar_width = 60
        base_y = 350
        
        for i, height in enumerate(bar_heights):
            x = 50 + i * 90
            cv2.rectangle(bar_chart, (x, base_y - height), (x + bar_width, base_y), 
                         (100, 100, 200), -1)
        
        # Draw axes
        cv2.line(bar_chart, (40, base_y), (450, base_y), (0, 0, 0), 2)  # X-axis
        cv2.line(bar_chart, (40, 50), (40, base_y), (0, 0, 0), 2)  # Y-axis
        
        # Extract data
        chart_extractor = ChartDataExtractor(use_ocr=False)
        chart_data = chart_extractor.extract_chart_data(bar_chart, ElementType.BAR_CHART)
        
        print(f"Detected bars: {len(chart_data.values)}")
        print(f"Values: {chart_data.values}")
        print(f"Confidence: {chart_data.confidence:.2f}")
        
        # Save for visual inspection
        cv2.imwrite("test_synthetic_bar_chart.png", bar_chart)
        print("💾 Saved to: test_synthetic_bar_chart.png")
        
        # Test 2: Simple table-like structure
        print("\n--- Synthetic Table ---")
        table_img = np.ones((300, 400, 3), dtype=np.uint8) * 255
        
        # Draw grid
        rows, cols = 5, 4
        cell_h, cell_w = 60, 100
        
        for i in range(rows + 1):
            y = i * cell_h
            cv2.line(table_img, (0, y), (cols * cell_w, y), (0, 0, 0), 1)
        
        for j in range(cols + 1):
            x = j * cell_w
            cv2.line(table_img, (x, 0), (x, rows * cell_h), (0, 0, 0), 1)
        
        # Save for visual inspection
        cv2.imwrite("test_synthetic_table.png", table_img)
        print("💾 Saved to: test_synthetic_table.png")
        
        print("\n✅ Synthetic data test completed")
        return True
        
    except Exception as e:
        print(f"\n❌ Synthetic data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_data_export_formats_test():
    """Test various data export formats (standalone script, not pytest)"""
    print("\n" + "="*70)
    print("TEST 5: Data Export Formats")
    print("="*70)
    
    try:
        import pandas as pd
        from docxtract.data_structures import Graph, Table, ChartSeries, BoundingBox, ElementType
        
        # Create sample graph data
        print("\n--- Testing Graph Export ---")
        sample_graph = Graph(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            bbox=BoundingBox(0, 0, 100, 100),
            page=1,
            graph_type=ElementType.BAR_CHART,
            confidence=0.85
        )
        
        # Add data
        sample_graph.extracted_values = [10.5, 20.3, 15.7, 25.1, 18.9]
        sample_graph.data_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        sample_graph.legend = ["Sales"]
        
        # Add series
        series = ChartSeries(
            name="Revenue",
            values=[10.5, 20.3, 15.7, 25.1, 18.9],
            labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
        )
        sample_graph.series.append(series)
        
        # Test dictionary export
        graph_dict = sample_graph.to_dict()
        print(f"Graph as dict: {json.dumps(graph_dict, indent=2)}")
        
        # Test DataFrame export
        graph_df = sample_graph.to_dataframe()
        print(f"\nGraph DataFrame:\n{graph_df}")
        
        # Test CSV export
        graph_csv = "test_graph_export.csv"
        sample_graph.to_csv(graph_csv)
        print(f"💾 Graph exported to CSV: {graph_csv}")
        
        # Create sample table data
        print("\n--- Testing Table Export ---")
        table_data = np.array([
            ["Product", "Q1", "Q2", "Q3", "Q4"],
            ["Widget A", "100", "150", "200", "180"],
            ["Widget B", "80", "90", "110", "120"],
            ["Widget C", "120", "130", "140", "150"]
        ])
        
        sample_table = Table(
            data=table_data,
            bbox=BoundingBox(0, 0, 300, 200),
            page=1,
            confidence=0.9
        )
        
        # Test DataFrame export
        table_df = sample_table.to_dataframe()
        print(f"\nTable DataFrame:\n{table_df}")
        
        # Test CSV export
        table_csv = "test_table_export.csv"
        sample_table.to_csv(table_csv)
        print(f"💾 Table exported to CSV: {table_csv}")
        
        # Test dictionary export
        table_dict = sample_table.to_dict()
        print(f"\nTable as dict keys: {list(table_dict.keys())}")
        
        print("\n✅ Data export formats test completed")
        return True
        
    except Exception as e:
        print(f"\n❌ Data export test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("DocXtract Data Extraction Test Suite")
    print("="*70)
    
    results = {
        "Chart Data Extraction": run_chart_data_extraction_test(),
        "Unstructured Tables": run_unstructured_tables_test(),
        "Combined Extraction": run_combined_extraction_test(),
        "Synthetic Data": run_synthetic_data_test(),
        "Data Export Formats": run_data_export_formats_test()
    }
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed or skipped")
        return 1


if __name__ == "__main__":
    exit(main())
