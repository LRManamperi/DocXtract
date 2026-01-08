"""
Test ML-based table detection vs traditional methods
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from docxtract import DocXtract

def run_ml_detection_test(pdf_path):
    """Compare ML vs traditional table detection (standalone script, not pytest)"""
    
    print("="*60)
    print("ML-BASED TABLE DETECTION TEST")
    print("="*60)
    
    # Test with ML-based detection
    print("\n🤖 Testing with ML-based detection (Table Transformer)...")
    print("-" * 60)
    
    try:
        extractor_ml = DocXtract(use_ml=True)
        result_ml = extractor_ml.extract(pdf_path, pages='all')
        
        print(f"\n✅ ML Detection Results:")
        print(f"   Tables detected: {len(result_ml.tables)}")
        print(f"   Charts detected: {len(result_ml.graphs)}")
        
        # Show data extraction results
        tables_with_data = sum(1 for t in result_ml.tables if t.data is not None and t.data.size > 0)
        print(f"   Tables with data: {tables_with_data}/{len(result_ml.tables)}")
        
        if tables_with_data > 0:
            print(f"\n   📊 Sample table data:")
            for i, table in enumerate(result_ml.tables[:3]):  # Show first 3 tables
                if table.data is not None and table.data.size > 0:
                    print(f"      Table {i+1}: {table.data.shape} - {table.data.size} cells")
                    df = table.df
                    if not df.empty:
                        print(f"         First row: {list(df.iloc[0].values)[:5]}")
        
    except Exception as e:
        print(f"❌ ML detection failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with traditional line-based detection for comparison
    print("\n\n📐 Testing with traditional line-based detection...")
    print("-" * 60)
    
    try:
        extractor_traditional = DocXtract(use_ml=False)
        result_traditional = extractor_traditional.extract(pdf_path, pages='all')
        
        print(f"\n✅ Traditional Detection Results:")
        print(f"   Tables detected: {len(result_traditional.tables)}")
        print(f"   Charts detected: {len(result_traditional.graphs)}")
        
        tables_with_data = sum(1 for t in result_traditional.tables if t.data is not None and t.data.size > 0)
        print(f"   Tables with data: {tables_with_data}/{len(result_traditional.tables)}")
        
    except Exception as e:
        print(f"❌ Traditional detection failed: {e}")
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        print("Usage: python test_ml_detection.py <pdf_path>")
        print("\nUsing default test (if PDF exists in current directory)")
        # Try to find a PDF in current directory
        import glob
        pdfs = glob.glob("*.pdf")
        if pdfs:
            pdf_path = pdfs[0]
            print(f"Found: {pdf_path}")
        else:
            print("No PDF found. Please provide a PDF path.")
            sys.exit(1)
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    run_ml_detection_test(pdf_path)
