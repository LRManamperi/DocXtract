"""
DocXtract - PDF Table and Chart Extraction Library
"""

from .data_structures import (
    ElementType, BoundingBox, Table, Graph, ExtractionResult
)
from .detectors import (
    BaseDetector, LineBasedTableDetector, TextClusterTableDetector,
    MLTableDetector, GraphDetector
)
from .parsers import (
    BaseParser, GridBasedTableParser, TextBasedTableParser
)
from .extractors import DocXtract, read_pdf

__version__ = "0.1.0"
__all__ = [
    # Data structures
    'ElementType', 'BoundingBox', 'Table', 'Graph', 'ExtractionResult',
    # Detectors
    'BaseDetector', 'LineBasedTableDetector', 'TextClusterTableDetector',
    'MLTableDetector', 'GraphDetector',
    # Parsers
    'BaseParser', 'GridBasedTableParser', 'TextBasedTableParser',
    # Main extractor
    'DocXtract', 'read_pdf'
]