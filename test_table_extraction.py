#!/usr/bin/env python3
"""
Test table extraction functionality with CSV export
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from docxtract import DocXtract
import numpy as np

def test_table_extraction():
    """Test table extraction on a sample PDF with CSV export"""
    print("🔍 Testing table extraction and CSV export...")

    # Check if we have test PDFs
    test_dir = "chart_dataset/charts"
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.pdf')] if os.path.exists(test_dir) else []

    if not test_files:
        print("❌ No test PDFs found in chart_dataset/charts")
        print("⚠️  Please add PDF files to test directory")
        return

    # Test with first PDF
    pdf_path = os.path.join(test_dir, test_files[0])
    print(f"📄 Testing with: {pdf_path}")

    try:
        extractor = DocXtract()
        results = extractor.extract(pdf_path)

        print(f"📊 Found {len(results.tables)} tables and {len(results.graphs)} graphs")

        # Test tables
        if results.tables:
            for i, table in enumerate(results.tables):
                print(f"\n📋 Table {i+1}:")
                print(f"   Page: {table.page}")
                print(f"   Shape: {table.data.shape if hasattr(table.data, 'shape') else 'N/A'}")
                print(f"   Confidence: {table.confidence:.2f}")

                if table.data.size > 0:
                    print("   Sample data:")
                    for row_idx in range(min(3, table.data.shape[0])):
                        row_data = []
                        for col_idx in range(min(3, table.data.shape[1])):
                            cell = str(table.data[row_idx, col_idx]).strip()
                            if len(cell) > 20:
                                cell = cell[:17] + "..."
                            row_data.append(f"'{cell}'")
                        print(f"     Row {row_idx}: {row_data}")
                    if table.data.shape[0] > 3:
                        print("     ...")
                    
                    # Test CSV export
                    try:
                        csv_filename = f"test_table_{i+1}.csv"
                        table.to_csv(csv_filename)
                        print(f"   ✅ Exported to {csv_filename}")
                        
                        # Test DataFrame conversion
                        df = table.to_dataframe()
                        print(f"   ✅ DataFrame shape: {df.shape}")
                        
                        # Clean up test file
                        if os.path.exists(csv_filename):
                            os.remove(csv_filename)
                            print(f"   🧹 Cleaned up test file")
                    except Exception as e:
                        print(f"   ⚠️  CSV export failed: {e}")
                else:
                    print("   ❌ Empty table data")
        else:
            print("⚠️ No tables detected in this PDF")
        
        # Test charts
        if results.graphs:
            print(f"\n📈 Found {len(results.graphs)} chart(s):")
            for i, graph in enumerate(results.graphs):
                print(f"\n   Chart {i+1}:")
                print(f"   Type: {graph.graph_type.name}")
                print(f"   Page: {graph.page}")
                print(f"   Confidence: {graph.confidence:.2f}")
                
                if hasattr(graph, 'data') and graph.data:
                    print(f"   Data keys: {list(graph.data.keys())}")
                    if 'values' in graph.data and graph.data['values']:
                        values = graph.data['values'][:5]  # First 5 values
                        print(f"   Sample values: {[f'{v:.2f}' for v in values]}")
        
        print("\n✅ Test completed successfully!")

    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_table_extraction()