"""
DocXtract - PDF Table and Chart Extraction Library
Enhanced with chart data extraction and unstructured table support
"""

from .data_structures import (
    ElementType, BoundingBox, Table, Graph, ExtractionResult,
    ChartSeries, AxisInfo, TableCell
)
from .detectors import (
    BaseDetector, LineBasedTableDetector, TextClusterTableDetector,
    MLTableDetector, GraphDetector
)
from .parsers import (
    BaseParser, GridBasedTableParser, TextBasedTableParser
)
from .extractors import DocXtract, read_pdf
from .chart_extractors import ChartDataExtractor
from .unstructured_table_parser import UnstructuredTableParser, StreamTableParser

# LangChain integration (optional)
try:
    from .langchain_pipeline import (
        LangChainExtractionPipeline,
        IntelligentDocumentAnalyzer,
        DocumentQueryAgent,
        DocumentAnalysis
    )
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

__version__ = "0.2.0"
__all__ = [
    # Data structures
    'ElementType', 'BoundingBox', 'Table', 'Graph', 'ExtractionResult',
    'ChartSeries', 'AxisInfo', 'TableCell',
    # Detectors
    'BaseDetector', 'LineBasedTableDetector', 'TextClusterTableDetector',
    'MLTableDetector', 'GraphDetector',
    # Parsers
    'BaseParser', 'GridBasedTableParser', 'TextBasedTableParser',
    'UnstructuredTableParser', 'StreamTableParser',
    # Extractors
    'DocXtract', 'read_pdf', 'ChartDataExtractor'
]

# Add LangChain components if available
if LANGCHAIN_AVAILABLE:
    __all__.extend([
        'LangChainExtractionPipeline',
        'IntelligentDocumentAnalyzer',
        'DocumentQueryAgent',
        'DocumentAnalysis',
        'LANGCHAIN_AVAILABLE'
    ])