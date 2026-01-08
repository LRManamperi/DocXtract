#!/usr/bin/env python3
"""
Test script to verify chart classification improvements
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docxtract.extractors import DocXtract
from docxtract.data_structures import ElementType
import cv2
import numpy as np

def run_chart_classification_test(pdf_path, expected_type):
    """Test chart classification for a PDF (standalone script, not pytest)"""
    print(f"\nTesting: {pdf_path}")
    print(f"Expected type: {expected_type}")

    # Initialize extractor
    extractor = DocXtract()

    # Process PDF
    results = extractor.extract(pdf_path)

    print(f"Total graphs found: {len(results.graphs)}")
    print(f"Total tables found: {len(results.tables)}")

    # Check results
    charts_found = []
    for graph in results.graphs:
        charts_found.append({
            'type': graph.graph_type,
            'confidence': graph.confidence,
            'data': graph.data
        })

    print(f"Charts detected: {len(charts_found)}")

    for i, chart in enumerate(charts_found):
        print(f"  Chart {i+1}: {chart['type'].name} (confidence: {chart['confidence']:.2f})")
        if chart['data']:
            print(f"    Data keys: {list(chart['data'].keys())}")
            # Show some data
            if 'bar_count' in chart['data']:
                print(f"    Bars detected: {chart['data']['bar_count']}")
            if 'point_count' in chart['data']:
                print(f"    Points detected: {chart['data']['point_count']}")

    # Check if classification matches expectation
    if charts_found:
        detected_type = charts_found[0]['type']
        confidence = charts_found[0]['confidence']

        if detected_type.name.lower() == expected_type.lower():
            print(f"✓ CORRECT: Detected {detected_type.name} with confidence {confidence:.2f}")
            return True
        else:
            print(f"✗ WRONG: Detected {detected_type.name} but expected {expected_type}")
            return False
    else:
        print("✗ No charts detected")
        return False

if __name__ == "__main__":
    # Test files
    test_files = [
        ("chart_dataset/charts/line_chart_0000.pdf", "line_chart"),
        ("chart_dataset/charts/line_chart_0004.pdf", "line_chart"),
        ("chart_dataset/charts/bar_chart_0001.pdf", "bar_chart"),
    ]

    results = []
    for pdf_path, expected in test_files:
        if os.path.exists(pdf_path):
            results.append(run_chart_classification_test(pdf_path, expected))
        else:
            print(f"File not found: {pdf_path}")

    print(f"\nSummary: {sum(results)}/{len(results)} tests passed")