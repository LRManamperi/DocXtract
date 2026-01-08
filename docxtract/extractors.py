"""
Main DocXtract extractor class - Enhanced with advanced data extraction
"""

from typing import List, Optional, Union
import os
import io
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import cv2

from .data_structures import ExtractionResult, BoundingBox, ElementType, Graph, ChartSeries, AxisInfo
from .detectors import (
    BaseDetector, LineBasedTableDetector, TextClusterTableDetector,
    MLTableDetector, GraphDetector
)
from .parsers import BaseParser, GridBasedTableParser, TextBasedTableParser
from .chart_extractors import ChartDataExtractor
from .unstructured_table_parser import UnstructuredTableParser, StreamTableParser

# Try to import ML detectors
try:
    from .ml_detectors import TableTransformerDetector, TableStructureExtractor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ ML detectors not available. Install: pip install torch torchvision transformers timm")


class DocXtract:
    """
    Main PDF extraction class
    Similar to Camelot's API but extended for graphs, charts, and unstructured tables
    """

    def __init__(self, table_detector: Optional[BaseDetector] = None,
                 table_parser: Optional[BaseParser] = None,
                 graph_detector: Optional[BaseDetector] = None,
                 use_ml: bool = True,
                 extract_chart_data: bool = True,
                 handle_unstructured: bool = True):
        """
        Initialize DocXtract
        
        Args:
            table_detector: Custom table detector (if None, will use ML or fallback to line-based)
            table_parser: Custom table parser
            graph_detector: Custom graph detector
            use_ml: Whether to use ML-based detection (requires torch, transformers)
            extract_chart_data: Whether to extract structured data from charts
            handle_unstructured: Whether to handle unstructured tables
        """
        if table_detector is not None:
            self.table_detector = table_detector
        elif use_ml and ML_AVAILABLE:
            self.table_detector = TableTransformerDetector(confidence_threshold=0.7)
            self.use_ml_extraction = True
        else:
            self.table_detector = LineBasedTableDetector(min_area=5000)
            self.use_ml_extraction = False
            
        self.table_parser = table_parser or GridBasedTableParser()
        self.graph_detector = graph_detector or GraphDetector()
        
        # Initialize ML structure extractor if available (will be set per-PDF)
        self.use_ml = use_ml
        self.structure_extractor = None
        
        # NEW: Initialize chart data extractor
        self.extract_chart_data = extract_chart_data
        self.chart_extractor = ChartDataExtractor(use_ocr=True) if extract_chart_data else None
        
        # NEW: Initialize unstructured table parsers
        self.handle_unstructured = handle_unstructured
        self.unstructured_parser = UnstructuredTableParser() if handle_unstructured else None
        self.stream_parser = StreamTableParser() if handle_unstructured else None

    def extract(self, pdf_path: str, pages: Union[str, List[int]] = 'all',
                table_flavor: str = 'lattice') -> ExtractionResult:
        """
        Extract tables and graphs from PDF

        Args:
            pdf_path: Path to PDF file
            pages: Pages to process ('all' or list of page numbers)
            table_flavor: Table detection method ('lattice', 'stream', 'ml')

        Returns:
            ExtractionResult containing all extracted elements
        """
        # Initialize ML structure extractor with PDF path
        if self.use_ml and ML_AVAILABLE:
            self.structure_extractor = TableStructureExtractor(
                confidence_threshold=0.3, 
                pdf_path=pdf_path
            )
        
        # Set table detector based on flavor
        if table_flavor == 'lattice':
            self.table_detector = LineBasedTableDetector()
        elif table_flavor == 'stream':
            self.table_detector = TextClusterTableDetector()
        elif table_flavor == 'ml':
            self.table_detector = MLTableDetector()
        else:
            raise ValueError(f"Unknown table flavor: {table_flavor}")

        # Open PDF
        doc = fitz.open(pdf_path)
        result = ExtractionResult(pdf_path)
        result.n_pages = len(doc)

        # Determine pages to process
        if pages == 'all':
            page_nums = list(range(len(doc)))
        else:
            page_nums = [p - 1 for p in pages]  # Convert to 0-based indexing

        # Process each page
        for page_num in page_nums:
            if page_num >= len(doc):
                continue

            page = doc[page_num]

            # Convert page to image with higher quality
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))  # Increased to 3x for better quality
            img_data = pix.tobytes("png")
            page_image = np.array(Image.open(io.BytesIO(img_data)))
            
            # Ensure correct color space
            if len(page_image.shape) == 3:
                if page_image.shape[2] == 4:  # RGBA
                    page_image = cv2.cvtColor(page_image, cv2.COLOR_RGBA2BGR)
                elif page_image.shape[2] == 3:  # RGB
                    page_image = cv2.cvtColor(page_image, cv2.COLOR_RGB2BGR)

            # Detect and extract graphs first (avoid double-detection with tables)
            graph_results = self.graph_detector.detect(page_image, page_num)
            graph_bboxes = set()  # Track detected graph areas
            
            # Handle both old format (3 items) and new format (4 items)
            for result_tuple in graph_results:
                if len(result_tuple) == 4:
                    # New format: (bbox, graph_type, confidence, chart_data)
                    bbox, graph_type, confidence, chart_data = result_tuple
                else:
                    # Old format: (bbox, graph_type, confidence)
                    bbox, graph_type, confidence = result_tuple
                    chart_data = {}
                
                # Extract region with padding
                x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
                
                # Add padding for better visualization
                padding = 5
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(page_image.shape[1], x2 + padding)
                y2 = min(page_image.shape[0], y2 + padding)
                
                region = page_image[y1:y2, x1:x2]

                # Create graph object with extracted data
                graph = Graph(region, bbox, page_num + 1, graph_type, confidence)
                
                # NEW: Extract structured data from charts
                if self.extract_chart_data and self.chart_extractor:
                    try:
                        chart_data_obj = self.chart_extractor.extract_chart_data(region, graph_type)
                        
                        # Populate graph object with extracted data
                        graph.extracted_values = chart_data_obj.values
                        graph.data_labels = chart_data_obj.labels
                        graph.legend = chart_data_obj.legend
                        
                        # Add series information
                        for series_dict in chart_data_obj.series:
                            series = ChartSeries(
                                name=series_dict.get('name', 'Series'),
                                values=series_dict.get('values', []),
                                labels=series_dict.get('labels')
                            )
                            graph.series.append(series)
                        
                        # Set axes information
                        if chart_data_obj.axes:
                            axes_dict = chart_data_obj.axes
                            graph.axes_info = AxisInfo(
                                orientation=axes_dict.get('orientation', 'vertical'),
                                x_label=axes_dict.get('x_label'),
                                y_label=axes_dict.get('y_label'),
                                x_range=axes_dict.get('x_range'),
                                y_range=axes_dict.get('y_range')
                            )
                        
                        # Update confidence based on extraction quality
                        if chart_data_obj.confidence > 0:
                            graph.confidence = (graph.confidence + chart_data_obj.confidence) / 2
                        
                        print(f"  📊 Extracted {len(chart_data_obj.values)} data points from {graph_type.name}")
                    except Exception as e:
                        print(f"  ⚠️ Chart data extraction failed: {e}")
                        # Fall back to basic data
                        graph.data = chart_data
                        graph.extracted_values = chart_data.get('values', [])
                else:
                    # Use basic chart data from detector
                    graph.data = chart_data
                    graph.extracted_values = chart_data.get('values', [])
                
                # Set point_count based on chart type
                if graph_type == ElementType.BAR_CHART:
                    graph.point_count = chart_data.get('bar_count', len(graph.extracted_values))
                elif graph_type == ElementType.LINE_CHART:
                    graph.point_count = chart_data.get('point_count', len(graph.extracted_values))
                elif graph_type == ElementType.SCATTER_PLOT:
                    graph.point_count = chart_data.get('point_count', len(graph.extracted_values))
                elif graph_type == ElementType.PIE_CHART:
                    graph.point_count = chart_data.get('slice_count', len(graph.extracted_values))
                else:
                    # Fallback for unknown types
                    graph.point_count = chart_data.get('point_count', chart_data.get('bar_count', len(graph.extracted_values)))
                
                result.graphs.append(graph)
                
                # Track this area to avoid detecting as table
                graph_bboxes.add((x1, y1, x2, y2))

            # Detect and extract tables (skip areas already identified as graphs)
            table_bboxes = self.table_detector.detect(page_image, page_num)
            
            print(f"📄 Page {page_num + 1}: Detected {len(table_bboxes)} potential tables")
            
            for bbox in table_bboxes:
                # Skip if overlaps with detected graph
                if self._overlaps_with_graphs(bbox, graph_bboxes):
                    print(f"  ⏭️  Skipping table (overlaps with graph)")
                    continue
                    
                # Extract region
                x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
                
                # Ensure bounds are valid
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(page_image.shape[1], x2)
                y2 = min(page_image.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    print(f"  ⏭️  Skipping table (invalid bounds)")
                    continue
                
                region = page_image[y1:y2, x1:x2]

                # Use ML extraction if available, with fallback to grid parser
                if self.structure_extractor is not None:
                    print(f"  🤖 Using ML extraction...")
                    table = self.structure_extractor.extract_structure(region, bbox, page, page_num)
                    
                    # If ML extraction fails, fallback to grid-based extraction
                    if table.data is None or table.data.size == 0:
                        print(f"  ⚠️  ML extraction failed, falling back to grid-based...")
                        table = self.table_parser.parse(region, bbox, page, pix.width, pix.height)
                else:
                    print(f"  📐 Using grid-based extraction...")
                    table = self.table_parser.parse(region, bbox, page, pix.width, pix.height)
                
                # NEW: If grid-based extraction fails, try unstructured parsers
                if self.handle_unstructured and (table.data is None or table.data.size == 0 or table.confidence < 0.5):
                    print(f"  🔄 Trying unstructured table extraction...")
                    
                    # Try text clustering approach
                    unstructured_table = self.unstructured_parser.parse(region, bbox, page, pix.width, pix.height)
                    
                    if unstructured_table.data is not None and unstructured_table.data.size > 0:
                        table = unstructured_table
                        print(f"  ✅ Unstructured extraction succeeded (shape: {table.data.shape})")
                    else:
                        # Try stream-based approach as last resort
                        print(f"  🔄 Trying stream-based extraction...")
                        stream_table = self.stream_parser.parse(region, bbox, page, pix.width, pix.height)
                        
                        if stream_table.data is not None and stream_table.data.size > 0:
                            table = stream_table
                            print(f"  ✅ Stream extraction succeeded (shape: {table.data.shape})")
                
                table.page = page_num + 1  # 1-based page numbering
                
                # Only add tables with data and reasonable size
                if table.data is not None and table.data.size > 0:
                    rows, cols = table.data.shape
                    # Filter out false positives:
                    # - 1-column tables are usually just text paragraphs
                    # - Very small tables (< 2x2) are usually false detections
                    # Accept: 2x2+, OR single row with 3+ cols, OR single col with 3+ rows
                    is_valid_table = (
                        (rows >= 2 and cols >= 2) or  # Standard table
                        (rows == 1 and cols >= 3) or  # Single row with multiple columns
                        (rows >= 3 and cols == 1)     # Single column with multiple rows (rare but valid)
                    )
                    if is_valid_table:
                        result.tables.append(table)
                        print(f"  ✅ Table added (shape: {table.data.shape})")
                    else:
                        print(f"  ⏭️  Table skipped (too small: {rows}x{cols})")
                else:
                    print(f"  ⚠️  Table skipped (no data extracted)")

        doc.close()
        return result

    def _overlaps_with_graphs(self, bbox: BoundingBox, graph_bboxes: set, threshold: float = 0.3) -> bool:
        """Check if bbox overlaps significantly with any detected graph"""
        bbox_area = bbox.area
        for gx1, gy1, gx2, gy2 in graph_bboxes:
            # Calculate intersection
            ix1 = max(bbox.x1, gx1)
            iy1 = max(bbox.y1, gy1)
            ix2 = min(bbox.x2, gx2)
            iy2 = min(bbox.y2, gy2)
            
            if ix2 > ix1 and iy2 > iy1:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                if intersection / bbox_area > threshold:
                    return True
        return False

    def extract_from_image(self, image_path: str) -> ExtractionResult:
        """
        Extract tables and graphs from a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            ExtractionResult containing all extracted elements
        """
        result = ExtractionResult(image_path)
        result.n_pages = 1
        
        # Load image
        page_image = cv2.imread(image_path)
        if page_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect and extract graphs
        graph_results = self.graph_detector.detect(page_image, 0)
        graph_bboxes = set()
        
        for result_tuple in graph_results:
            if len(result_tuple) == 4:
                bbox, graph_type, confidence, chart_data = result_tuple
            else:
                bbox, graph_type, confidence = result_tuple
                chart_data = {}
            
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
            region = page_image[y1:y2, x1:x2]
            
            graph = Graph(region, bbox, 1, graph_type, confidence)
            graph.data = chart_data
            
            # Extract values based on chart type (same logic as extract method)
            # All chart types now have 'values' as a list of floats
            graph.extracted_values = chart_data.get('values', [])
            
            # Set point_count based on chart type
            if graph_type == ElementType.BAR_CHART:
                graph.point_count = chart_data.get('bar_count', len(graph.extracted_values))
            elif graph_type == ElementType.LINE_CHART:
                graph.point_count = chart_data.get('point_count', len(graph.extracted_values))
            elif graph_type == ElementType.SCATTER_PLOT:
                graph.point_count = chart_data.get('point_count', len(graph.extracted_values))
            elif graph_type == ElementType.PIE_CHART:
                graph.point_count = chart_data.get('slice_count', len(graph.extracted_values))
            else:
                graph.point_count = chart_data.get('point_count', chart_data.get('bar_count', len(graph.extracted_values)))
            
            result.graphs.append(graph)
            graph_bboxes.add((x1, y1, x2, y2))
        
        # Detect and extract tables
        table_bboxes = self.table_detector.detect(page_image, 0)
        for bbox in table_bboxes:
            if self._overlaps_with_graphs(bbox, graph_bboxes):
                continue
            
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
            region = page_image[y1:y2, x1:x2]
            
            table = self.table_parser.parse(region, bbox)
            table.page = 1
            result.tables.append(table)
        
        return result

    def visualize_detections(self, pdf_path: str, output_path: str, 
                            pages: Union[str, List[int]] = 'all'):
        """
        Visualize detected elements on PDF pages
        
        Args:
            pdf_path: Path to input PDF
            output_path: Path to save visualization
            pages: Pages to visualize
        """
        doc = fitz.open(pdf_path)
        
        if pages == 'all':
            page_nums = list(range(len(doc)))
        else:
            page_nums = [p - 1 for p in pages]
        
        for page_num in page_nums:
            if page_num >= len(doc):
                continue
            
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
            img_data = pix.tobytes("png")
            page_image = np.array(Image.open(io.BytesIO(img_data)))
            
            if len(page_image.shape) == 3:
                if page_image.shape[2] == 4:
                    page_image = cv2.cvtColor(page_image, cv2.COLOR_RGBA2BGR)
                elif page_image.shape[2] == 3:
                    page_image = cv2.cvtColor(page_image, cv2.COLOR_RGB2BGR)
            
            # Detect graphs
            graph_results = self.graph_detector.detect(page_image, page_num)
            for result_tuple in graph_results:
                if len(result_tuple) == 4:
                    bbox, graph_type, confidence, chart_data = result_tuple
                else:
                    bbox, graph_type, confidence = result_tuple
                    chart_data = {}
                
                # Draw bounding box in green
                cv2.rectangle(page_image, 
                            (int(bbox.x1), int(bbox.y1)), 
                            (int(bbox.x2), int(bbox.y2)), 
                            (0, 255, 0), 3)
                
                # Add label
                label = f"{graph_type.name} ({confidence:.2f})"
                cv2.putText(page_image, label, 
                          (int(bbox.x1), int(bbox.y1) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add data info if available
                if chart_data:
                    data_label = f"Data: {chart_data.get('bar_count', chart_data.get('point_count', 'N/A'))}"
                    cv2.putText(page_image, data_label,
                              (int(bbox.x1), int(bbox.y2) + 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Detect tables
            table_bboxes = self.table_detector.detect(page_image, page_num)
            for bbox in table_bboxes:
                # Draw bounding box in blue
                cv2.rectangle(page_image,
                            (int(bbox.x1), int(bbox.y1)),
                            (int(bbox.x2), int(bbox.y2)),
                            (255, 0, 0), 3)
                
                cv2.putText(page_image, "TABLE",
                          (int(bbox.x1), int(bbox.y1) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Save visualization
            output_file = f"{output_path}_page_{page_num + 1}.png"
            cv2.imwrite(output_file, page_image)
            print(f"Saved visualization to {output_file}")
        
        doc.close()


# Convenience function for Camelot-like API
def read_pdf(pdf_path: str, pages: Union[str, List[int]] = 'all',
             flavor: str = 'ml', use_ml: bool = True) -> ExtractionResult:
    """
    Read PDF and extract tables and graphs

    Args:
        pdf_path: Path to PDF file
        pages: Pages to process ('all' or list of page numbers)
        flavor: Detection method ('lattice', 'stream', 'ml')
        use_ml: Use ML-based detection (recommended for better accuracy)

    Returns:
        ExtractionResult with extracted elements
    """
    extractor = DocXtract(use_ml=use_ml)
    return extractor.extract(pdf_path, pages, flavor)


def read_image(image_path: str, use_ml: bool = True) -> ExtractionResult:
    """
    Read image and extract tables and graphs
    
    Args:
        image_path: Path to image file
        use_ml: Use ML-based detection
        
    Returns:
        ExtractionResult with extracted elements
    """
    extractor = DocXtract(use_ml=use_ml)
    return extractor.extract_from_image(image_path)