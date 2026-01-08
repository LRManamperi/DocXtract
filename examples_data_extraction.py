"""
Quick start examples for DocXtract data extraction
Run this file to see the new capabilities in action
"""

from docxtract.extractors import DocXtract
from docxtract.data_structures import ElementType
import os

def example_1_basic_extraction():
    """Example 1: Basic extraction with all features enabled"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Extraction with All Features")
    print("="*70)
    
    # Initialize with all features
    extractor = DocXtract(
        use_ml=False,
        extract_chart_data=True,     # Extract data from charts
        handle_unstructured=True      # Handle tables without borders
    )
    
    # Note: Replace 'your_document.pdf' with an actual PDF file
    pdf_path = 'your_document.pdf'
    
    if not os.path.exists(pdf_path):
        print(f"\n⚠️  Please provide a PDF file at: {pdf_path}")
        print("   Or modify the pdf_path variable to point to your PDF")
        return
    
    # Extract data
    result = extractor.extract(pdf_path, pages='all')
    
    # Print summary
    print(result.summary())
    
    # Export results
    result.to_json('extraction_results.json')
    print("\n💾 Results saved to: extraction_results.json")


def example_2_chart_extraction():
    """Example 2: Extract and analyze chart data"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Chart Data Extraction")
    print("="*70)
    
    extractor = DocXtract(extract_chart_data=True, use_ml=False)
    
    pdf_path = 'charts_document.pdf'
    
    if not os.path.exists(pdf_path):
        print(f"\n⚠️  Please provide a PDF with charts at: {pdf_path}")
        return
    
    result = extractor.extract(pdf_path)
    
    # Filter by chart type
    bar_charts = result.get_bar_charts()
    line_charts = result.get_line_charts()
    pie_charts = result.get_pie_charts()
    
    print(f"\n📊 Found:")
    print(f"   Bar Charts: {len(bar_charts)}")
    print(f"   Line Charts: {len(line_charts)}")
    print(f"   Pie Charts: {len(pie_charts)}")
    
    # Analyze each bar chart
    for i, chart in enumerate(bar_charts, 1):
        print(f"\n--- Bar Chart {i} ---")
        print(f"Page: {chart.page}")
        print(f"Values: {chart.extracted_values}")
        print(f"Labels: {chart.data_labels}")
        
        # Export to CSV
        csv_file = f'bar_chart_{i}.csv'
        chart.to_csv(csv_file)
        print(f"💾 Exported to: {csv_file}")


def example_3_unstructured_tables():
    """Example 3: Handle tables without borders"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Unstructured Table Extraction")
    print("="*70)
    
    extractor = DocXtract(
        use_ml=False,
        handle_unstructured=True
    )
    
    pdf_path = 'tables_document.pdf'
    
    if not os.path.exists(pdf_path):
        print(f"\n⚠️  Please provide a PDF with tables at: {pdf_path}")
        return
    
    result = extractor.extract(pdf_path)
    
    print(f"\n📋 Found {result.n_tables} tables")
    
    for i, table in enumerate(result.tables, 1):
        print(f"\n--- Table {i} ---")
        print(f"Page: {table.page}")
        print(f"Shape: {table.data.shape}")
        print(f"Confidence: {table.confidence:.2f}")
        
        # View as DataFrame
        df = table.to_dataframe()
        print(f"\nFirst few rows:")
        print(df.head())
        
        # Export
        csv_file = f'table_{i}.csv'
        table.to_csv(csv_file)
        print(f"💾 Exported to: {csv_file}")


def example_4_filtering_and_export():
    """Example 4: Filter results and export in different formats"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Filtering and Export")
    print("="*70)
    
    extractor = DocXtract(
        extract_chart_data=True,
        handle_unstructured=True,
        use_ml=False
    )
    
    pdf_path = 'mixed_document.pdf'
    
    if not os.path.exists(pdf_path):
        print(f"\n⚠️  Please provide a PDF at: {pdf_path}")
        return
    
    result = extractor.extract(pdf_path)
    
    # Filter high-confidence tables
    high_conf_tables = [t for t in result.tables if t.confidence > 0.7]
    print(f"\n✅ High confidence tables: {len(high_conf_tables)}/{result.n_tables}")
    
    # Export all data as JSON
    result.to_json('all_data.json')
    print("💾 All data exported to: all_data.json")
    
    # Export individual charts
    for i, chart in enumerate(result.graphs, 1):
        print(f"\nChart {i}: {chart.graph_type.name}")
        
        # Export as CSV
        chart.to_csv(f'chart_{i}.csv')
        
        # Get as DataFrame
        df = chart.to_dataframe()
        if not df.empty:
            print(f"DataFrame shape: {df.shape}")
            print(df)


def example_5_custom_workflow():
    """Example 5: Custom extraction workflow"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Custom Workflow")
    print("="*70)
    
    from docxtract.detectors import LineBasedTableDetector
    
    # Custom detector configuration
    custom_detector = LineBasedTableDetector(min_area=3000)
    
    extractor = DocXtract(
        table_detector=custom_detector,
        use_ml=False,
        extract_chart_data=True,
        handle_unstructured=True
    )
    
    pdf_path = 'document.pdf'
    
    if not os.path.exists(pdf_path):
        print(f"\n⚠️  Please provide a PDF at: {pdf_path}")
        return
    
    # Extract with specific flavor
    result = extractor.extract(pdf_path, table_flavor='lattice')
    
    # Process results with custom logic
    for table in result.tables:
        if table.data.shape[0] > 5:  # Tables with more than 5 rows
            print(f"\nLarge table on page {table.page}: {table.data.shape}")
            table.to_csv(f'large_table_p{table.page}.csv')
    
    for chart in result.graphs:
        if len(chart.extracted_values) > 10:  # Charts with many data points
            print(f"\nDetailed chart on page {chart.page}: {len(chart.extracted_values)} points")
            chart.to_csv(f'detailed_chart_p{chart.page}.csv')


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("DocXtract Quick Start Examples")
    print("="*70)
    print("\n⚠️  Note: These examples require PDF files to run.")
    print("   Update the pdf_path variables in each example function.")
    print("   Or use the interactive dashboard: python run_dashboard.py")
    
    # Run examples (commented out by default)
    # Uncomment and provide PDF files to run
    
    # example_1_basic_extraction()
    # example_2_chart_extraction()
    # example_3_unstructured_tables()
    # example_4_filtering_and_export()
    # example_5_custom_workflow()
    
    print("\n📚 For more information, see:")
    print("   - README.md: Complete documentation")
    print("   - test_data_extraction.py: Comprehensive test suite")
    print("   - run_dashboard.py: Interactive web interface")


if __name__ == "__main__":
    main()
