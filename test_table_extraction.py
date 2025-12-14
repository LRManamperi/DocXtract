#!/usr/bin/env python3
"""
Test table extraction functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from docxtract import DocXtract
import numpy as np

def test_table_extraction():
    """Test table extraction on a sample PDF"""
    print("🔍 Testing table extraction...")

    # Check if we have test PDFs
    test_dir = "chart_dataset/charts"
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.pdf')] if os.path.exists(test_dir) else []

    if not test_files:
        print("❌ No test PDFs found")
        return

    # Test with first PDF
    pdf_path = os.path.join(test_dir, test_files[0])
    print(f"📄 Testing with: {pdf_path}")

    try:
        extractor = DocXtract()
        results = extractor.extract(pdf_path)

        print(f"📊 Found {len(results.tables)} tables and {len(results.graphs)} graphs")

        for i, table in enumerate(results.tables):
            print(f"\n📋 Table {i+1}:")
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
            else:
                print("   ❌ Empty table data")

        if not results.tables:
            print("⚠️ No tables detected in this PDF")

    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_table_extraction()