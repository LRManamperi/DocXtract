# DocXtract

A Python library for extracting tables and charts from PDF documents.

## Features

- Extract tables from PDFs using multiple detection strategies (lattice, stream, ML-based)
- Detect and classify various chart types (bar charts, line charts, pie charts, etc.)
- Camelot-compatible API for easy migration
- Extensible architecture with pluggable detectors and parsers

## Installation

```bash
pip install -r requirements.txt
```

For OCR functionality (optional):
```bash
# Install Tesseract OCR
# On Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# On macOS:
brew install tesseract

# On Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

## Usage

### Web Dashboard (Recommended)

Launch the professional web interface for an intuitive PDF analysis experience:

```bash
python run_dashboard.py
```

**Features:**
- 🎨 **Modern UI**: Clean, professional interface with sidebar navigation
- 📊 **Interactive Analytics**: Real-time progress bars and status updates
- 📈 **Visual Insights**: Gradient metric cards and chart galleries
- 📥 **One-Click Downloads**: Export tables as CSV with professional styling
- 🔄 **Session Management**: Persistent results across page navigation
- 📱 **Responsive Design**: Optimized for different screen sizes

**Dashboard Sections:**
- **🏠 Home**: Upload PDFs and start analysis with guided workflow
- **📈 Charts Analysis**: Detailed chart detection with confidence metrics
- **📋 Tables**: Extracted tables with preview and download options
- **ℹ️ About**: Documentation and feature overview

### Basic Usage

```python
from docxtract import read_pdf

# Extract tables and graphs from PDF
result = read_pdf('document.pdf', pages='all', flavor='lattice')

print(f"Found {len(result.tables)} tables and {len(result.graphs)} graphs")

# Access tables
for i, table in enumerate(result.tables):
    print(f"Table {i+1}: {table.shape}")
    df = table.df  # Get as pandas DataFrame
    table.to_csv(f'table_{i}.csv')

# Access graphs
for i, graph in enumerate(result.graphs):
    print(f"Graph {i+1}: {graph.graph_type.value}")
    graph.save_image(f'graph_{i}.png')
```

### Advanced Usage

```python
from docxtract import DocXtract
from docxtract.detectors import LineBasedTableDetector
from docxtract.parsers import GridBasedTableParser

# Custom configuration
extractor = DocXtract(
    table_detector=LineBasedTableDetector(),
    table_parser=GridBasedTableParser()
)

result = extractor.extract('document.pdf', pages=[1, 2, 3])
```

## Detection Flavors

- `lattice`: Line-based table detection (best for tables with borders)
- `stream`: Text clustering table detection (best for tables without borders)
- `ml`: Machine learning-based detection (requires custom model)

## Supported Chart Types

- Bar charts (vertical/horizontal)
- Line charts
- Pie charts
- Scatter plots
- Heatmaps
- Area charts
- Stacked bar charts

## Architecture

The library is organized into several modules:

- `data_structures.py`: Core data classes (Table, Graph, BoundingBox, etc.)
- `detectors.py`: Detection strategies for tables and graphs
- `parsers.py`: Parsing strategies for extracting structured data
- `extractors.py`: Main extraction orchestrator

## Extending DocXtract

### Custom Detectors

```python
from docxtract.detectors import BaseDetector
from docxtract.data_structures import BoundingBox

class MyCustomDetector(BaseDetector):
    def detect(self, page_image, page_num):
        # Your detection logic here
        return [BoundingBox(x1, y1, x2, y2), ...]
```

### Custom Parsers

```python
from docxtract.parsers import BaseParser
from docxtract.data_structures import Table

class MyCustomParser(BaseParser):
    def parse(self, region, bbox):
        # Your parsing logic here
        return Table(data, bbox, page, accuracy)
```

## Requirements

- Python 3.8+
- PyMuPDF (Fitz)
- OpenCV
- NumPy
- Pandas
- Pillow
- Matplotlib
- pytesseract (optional, for OCR)

## License

MIT License