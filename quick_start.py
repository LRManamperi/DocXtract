"""
Quick Start Guide for DocXtract with Table & Chart Extraction

Run this script to verify your installation and test the new features.
"""

import sys
import os

print("=" * 70)
print("DocXtract - Quick Start & Verification")
print("=" * 70)

# 1. Check Python version
print("\n1. Checking Python version...")
print(f"   Python: {sys.version}")
if sys.version_info < (3, 7):
    print("   ⚠️  Warning: Python 3.7+ recommended")
else:
    print("   ✅ Python version OK")

# 2. Check required packages
print("\n2. Checking required packages...")
required_packages = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'cv2': 'opencv-python',
    'PIL': 'Pillow',
    'fitz': 'PyMuPDF',
    'streamlit': 'streamlit'
}

missing_packages = []
for module, package in required_packages.items():
    try:
        __import__(module)
        print(f"   ✅ {package}")
    except ImportError:
        print(f"   ❌ {package} - MISSING")
        missing_packages.append(package)

# 3. Check optional packages
print("\n3. Checking optional packages...")
optional_packages = {
    'pytesseract': 'pytesseract (for table OCR)',
    'openpyxl': 'openpyxl (for Excel export)'
}

for module, description in optional_packages.items():
    try:
        __import__(module)
        print(f"   ✅ {description}")
    except ImportError:
        print(f"   ⚠️  {description} - Not installed (optional)")

# 4. Check Tesseract OCR
print("\n4. Checking Tesseract OCR...")
try:
    import pytesseract
    try:
        version = pytesseract.get_tesseract_version()
        print(f"   ✅ Tesseract installed: v{version}")
    except Exception as e:
        print(f"   ⚠️  Tesseract not found in PATH")
        print(f"      Install from: https://github.com/UB-Mannheim/tesseract/wiki")
except ImportError:
    print(f"   ⚠️  pytesseract not installed")

# 5. Check DocXtract module
print("\n5. Checking DocXtract module...")
try:
    from docxtract import DocXtract, read_pdf
    from docxtract.data_structures import Table, Graph
    print("   ✅ DocXtract core")
    print("   ✅ Data structures")
except ImportError as e:
    print(f"   ❌ DocXtract not found: {e}")
    missing_packages.append("docxtract")

# 6. Check UI modules
print("\n6. Checking UI modules...")
try:
    from UI.page_modules import (
        render_home,
        render_tables,
        render_charts_analysis,
        render_chart_data_tables,
        render_about
    )
    print("   ✅ All UI modules loaded")
except ImportError as e:
    print(f"   ⚠️  UI modules issue: {e}")

# 7. Test CSV export functionality
print("\n7. Testing CSV export functionality...")
try:
    import pandas as pd
    import numpy as np
    from docxtract.data_structures import Table, BoundingBox
    
    # Create a test table
    test_data = np.array([
        ['Header1', 'Header2', 'Header3'],
        ['Row1Col1', 'Row1Col2', 'Row1Col3'],
        ['Row2Col1', 'Row2Col2', 'Row2Col3']
    ])
    bbox = BoundingBox(0, 0, 100, 100)
    test_table = Table(test_data, bbox, page=1, confidence=0.95)
    
    # Test DataFrame conversion
    df = test_table.to_dataframe()
    print(f"   ✅ DataFrame conversion: {df.shape}")
    
    # Test CSV string
    csv_str = test_table.to_csv_string()
    print(f"   ✅ CSV string generation: {len(csv_str)} bytes")
    
except Exception as e:
    print(f"   ❌ CSV export test failed: {e}")

# 8. Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if missing_packages:
    print("\n❌ Missing required packages:")
    for pkg in missing_packages:
        print(f"   - {pkg}")
    print("\nInstall with:")
    print(f"   pip install {' '.join(missing_packages)}")
else:
    print("\n✅ All required packages installed!")

print("\n" + "=" * 70)
print("HOW TO RUN")
print("=" * 70)
print("\n1. Run the dashboard:")
print("   python run_dashboard.py")
print("   OR")
print("   streamlit run UI/dashboard.py")
print("\n2. Test table extraction:")
print("   python test_table_extraction.py")
print("\n3. Test individual components:")
print("   python test_ocr.py")
print("   python test_classification.py")

print("\n" + "=" * 70)
print("NEW FEATURES")
print("=" * 70)
print("""
✅ Table Extraction to CSV/Excel
   - Navigate to 'Tables' tab in dashboard
   - View and download extracted tables
   
✅ Chart Data Extraction with Axis Labels
   - Navigate to 'Chart Data Tables' tab
   - Download chart data as CSV or JSON
   
✅ Enhanced UI with Better Navigation
   - Dedicated tabs for tables and charts
   - Summary metrics in sidebar
   - Multiple export formats
""")

print("=" * 70)
print("For detailed documentation, see:")
print("   - FEATURE_UPDATE.md")
print("   - CODE_CORRECTIONS.md")
print("=" * 70)
