"""
Detailed test of test_combined.pdf
"""
from docxtract import DocXtract

print('='*70)
print('TEST COMBINED PDF - DETAILED EXTRACTION')
print('='*70)

# Initialize extractor
extractor = DocXtract(
    use_ml=False, 
    extract_chart_data=True, 
    handle_unstructured=True
)

# Extract
print('\n📄 Extracting data...\n')
result = extractor.extract('test_combined.pdf')

# Summary
print('\n' + '='*70)
print('SUMMARY')
print('='*70)
print(f'Pages: {result.n_pages}')
print(f'Tables Found: {result.n_tables}')
print(f'Charts Found: {result.n_graphs}')

# Tables
if result.tables:
    print('\n' + '='*70)
    print('TABLES')
    print('='*70)
    for i, table in enumerate(result.tables, 1):
        print(f'\n📋 Table {i}')
        print(f'   Page: {table.page}')
        print(f'   Shape: {table.data.shape[0]} rows × {table.data.shape[1]} columns')
        print(f'   Confidence: {table.confidence:.2f}')
        print(f'   Bounding Box: ({table.bbox.x1:.0f}, {table.bbox.y1:.0f}) to ({table.bbox.x2:.0f}, {table.bbox.y2:.0f})')
        print(f'\n   Data Preview:')
        df = table.df
        print(f'   {df.to_string().replace(chr(10), chr(10) + "   ")}')
        
        # Export
        csv_file = f'table_{i}_page_{table.page}.csv'
        table.to_csv(csv_file)
        print(f'\n   ✅ Exported to: {csv_file}')

# Charts
if result.graphs:
    print('\n' + '='*70)
    print('CHARTS')
    print('='*70)
    for i, chart in enumerate(result.graphs, 1):
        print(f'\n📊 Chart {i}')
        print(f'   Type: {chart.graph_type.name}')
        print(f'   Page: {chart.page}')
        print(f'   Confidence: {chart.confidence:.2f}')
        print(f'   Data Points: {chart.point_count}')
        
        if chart.extracted_values:
            print(f'   Values: {chart.extracted_values[:10]}{"..." if len(chart.extracted_values) > 10 else ""}')
        
        if chart.data_labels:
            print(f'   Labels: {chart.data_labels[:10]}{"..." if len(chart.data_labels) > 10 else ""}')
        
        if chart.legend:
            print(f'   Legend: {chart.legend}')
        
        # Export
        csv_file = f'chart_{i}_page_{chart.page}.csv'
        chart.to_csv(csv_file)
        print(f'   ✅ Exported to: {csv_file}')

# Export JSON
print('\n' + '='*70)
print('EXPORT')
print('='*70)
json_file = 'test_combined_detailed.json'
result.to_json(json_file)
print(f'✅ Complete results exported to: {json_file}')

print('\n' + '='*70)
print('✅ TEST COMPLETE')
print('='*70)
