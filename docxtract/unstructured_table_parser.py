"""
Unstructured table parser for DocXtract
Handles tables without clear grid lines using text clustering and layout analysis
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
import pytesseract
from PIL import Image
from collections import defaultdict
from sklearn.cluster import DBSCAN
import os

from .data_structures import Table, BoundingBox

# Configure Tesseract path if available
if os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class UnstructuredTableParser:
    """
    Parser for unstructured tables (tables without clear grid lines)
    Uses text detection, clustering, and layout analysis
    """
    
    def __init__(self, use_ocr: bool = True, min_rows: int = 2, min_cols: int = 2):
        """
        Initialize unstructured table parser
        
        Args:
            use_ocr: Whether to use OCR for text extraction
            min_rows: Minimum rows for valid table
            min_cols: Minimum columns for valid table
        """
        self.use_ocr = use_ocr
        self.min_rows = min_rows
        self.min_cols = min_cols
    
    def parse(self, region: np.ndarray, bbox: BoundingBox, 
              page_obj=None, img_width=None, img_height=None) -> Table:
        """
        Parse unstructured table region into structured data
        
        Args:
            region: Image region containing table
            bbox: Bounding box of table
            page_obj: PDF page object (for text extraction)
            img_width: Image width (for coordinate conversion)
            img_height: Image height (for coordinate conversion)
            
        Returns:
            Table object with extracted data
        """
        if region.size == 0:
            return Table(np.array([]), bbox, 0, 0.0)
        
        # Step 1: Detect text regions
        text_boxes = self._detect_text_regions(region)
        
        if len(text_boxes) < self.min_rows * self.min_cols:
            # Not enough text regions for a table
            return Table(np.array([]), bbox, 0, 0.5)
        
        # Step 2: Cluster text boxes into rows and columns
        grid_cells = self._cluster_into_grid(text_boxes, region.shape)
        
        if not grid_cells:
            return Table(np.array([]), bbox, 0, 0.5)
        
        # Step 3: Extract text from each cell
        cell_data = self._extract_cell_text(region, grid_cells)
        
        # Step 4: Convert to 2D array
        table_array = self._to_2d_array(cell_data)
        
        if table_array.size == 0:
            return Table(np.array([]), bbox, 0, 0.5)
        
        # Calculate confidence based on structure consistency
        confidence = self._calculate_confidence(grid_cells, table_array)
        
        return Table(
            data=table_array,
            bbox=bbox,
            page=1,
            confidence=confidence,
            image=region
        )
    
    def _detect_text_regions(self, image: np.ndarray) -> List[Dict]:
        """
        Detect text regions in image using multiple methods
        
        Returns:
            List of dictionaries with text box information
        """
        text_boxes = []
        
        # Method 1: Use Tesseract's text detection
        if self.use_ocr:
            ocr_boxes = self._detect_text_with_tesseract(image)
            text_boxes.extend(ocr_boxes)
        
        # Method 2: Use contour-based text detection as fallback
        if not text_boxes:
            contour_boxes = self._detect_text_with_contours(image)
            text_boxes.extend(contour_boxes)
        
        return text_boxes
    
    def _detect_text_with_tesseract(self, image: np.ndarray) -> List[Dict]:
        """Detect text regions using Tesseract OCR"""
        try:
            # Convert to PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # Get detailed OCR data
            ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            
            text_boxes = []
            n_boxes = len(ocr_data['text'])
            
            for i in range(n_boxes):
                # Only keep boxes with confidence and non-empty text
                if int(ocr_data['conf'][i]) > 30 and ocr_data['text'][i].strip():
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    text = ocr_data['text'][i].strip()
                    
                    text_boxes.append({
                        'x': x, 'y': y,
                        'width': w, 'height': h,
                        'text': text,
                        'confidence': ocr_data['conf'][i],
                        'center_x': x + w//2,
                        'center_y': y + h//2
                    })
            
            return text_boxes
        except Exception as e:
            print(f"OCR detection failed: {e}")
            return []
    
    def _detect_text_with_contours(self, image: np.ndarray) -> List[Dict]:
        """Detect text regions using contour analysis (fallback method)"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Dilate to connect text characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter based on size (text regions should be reasonably sized)
            if 5 < w < image.shape[1] * 0.8 and 5 < h < image.shape[0] * 0.3:
                # Aspect ratio check (text tends to be wider than tall)
                aspect_ratio = w / h if h > 0 else 0
                if 0.3 < aspect_ratio < 20:
                    text_boxes.append({
                        'x': x, 'y': y,
                        'width': w, 'height': h,
                        'text': '',  # No text extracted yet
                        'confidence': 50,
                        'center_x': x + w//2,
                        'center_y': y + h//2
                    })
        
        return text_boxes
    
    def _cluster_into_grid(self, text_boxes: List[Dict], image_shape: Tuple) -> List[Dict]:
        """
        Cluster text boxes into a grid structure (rows and columns)
        
        Returns:
            List of grid cells with row/column assignments
        """
        if not text_boxes:
            return []
        
        # Extract center coordinates
        centers = np.array([[box['center_x'], box['center_y']] for box in text_boxes])
        
        # Cluster by Y-coordinate to find rows
        y_coords = centers[:, 1].reshape(-1, 1)
        row_clustering = DBSCAN(eps=20, min_samples=1).fit(y_coords)
        row_labels = row_clustering.labels_
        
        # Cluster by X-coordinate to find columns
        x_coords = centers[:, 0].reshape(-1, 1)
        col_clustering = DBSCAN(eps=30, min_samples=1).fit(x_coords)
        col_labels = col_clustering.labels_
        
        # Assign row and column to each text box
        grid_cells = []
        for i, box in enumerate(text_boxes):
            cell = box.copy()
            cell['row'] = row_labels[i]
            cell['col'] = col_labels[i]
            grid_cells.append(cell)
        
        # Sort grid cells by row then column
        grid_cells = sorted(grid_cells, key=lambda c: (c['row'], c['col']))
        
        # Renumber rows and columns to be consecutive
        grid_cells = self._normalize_grid_indices(grid_cells)
        
        return grid_cells
    
    def _normalize_grid_indices(self, grid_cells: List[Dict]) -> List[Dict]:
        """Normalize row/column indices to be consecutive starting from 0"""
        if not grid_cells:
            return []
        
        # Get unique rows and columns
        unique_rows = sorted(set(cell['row'] for cell in grid_cells))
        unique_cols = sorted(set(cell['col'] for cell in grid_cells))
        
        # Create mapping
        row_map = {old: new for new, old in enumerate(unique_rows)}
        col_map = {old: new for new, old in enumerate(unique_cols)}
        
        # Apply mapping
        for cell in grid_cells:
            cell['row'] = row_map[cell['row']]
            cell['col'] = col_map[cell['col']]
        
        return grid_cells
    
    def _extract_cell_text(self, image: np.ndarray, grid_cells: List[Dict]) -> List[Dict]:
        """Extract text from each grid cell"""
        for cell in grid_cells:
            # If text already extracted from OCR, keep it
            if cell.get('text'):
                continue
            
            # Otherwise, extract from image region
            if self.use_ocr:
                x, y, w, h = cell['x'], cell['y'], cell['width'], cell['height']
                
                # Ensure bounds are valid
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(image.shape[1], x + w)
                y2 = min(image.shape[0], y + h)
                
                if x2 > x1 and y2 > y1:
                    cell_region = image[y1:y2, x1:x2]
                    cell['text'] = self._ocr_cell(cell_region)
                else:
                    cell['text'] = ''
            else:
                cell['text'] = ''
        
        return grid_cells
    
    def _ocr_cell(self, cell_image: np.ndarray) -> str:
        """Extract text from a single cell image"""
        if cell_image.size == 0:
            return ''
        
        try:
            # Preprocess for better OCR
            if len(cell_image.shape) == 3:
                gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = cell_image
            
            # Enhance contrast
            enhanced = cv2.equalizeHist(gray)
            
            # Convert to PIL
            pil_image = Image.fromarray(enhanced)
            
            # OCR with appropriate config
            text = pytesseract.image_to_string(pil_image, config='--psm 7').strip()
            
            return text
        except Exception as e:
            return ''
    
    def _to_2d_array(self, grid_cells: List[Dict]) -> np.ndarray:
        """Convert grid cells to 2D numpy array"""
        if not grid_cells:
            return np.array([])
        
        # Find grid dimensions
        max_row = max(cell['row'] for cell in grid_cells)
        max_col = max(cell['col'] for cell in grid_cells)
        
        rows = max_row + 1
        cols = max_col + 1
        
        # Check minimum requirements
        if rows < self.min_rows or cols < self.min_cols:
            return np.array([])
        
        # Create 2D array
        table_data = [['' for _ in range(cols)] for _ in range(rows)]
        
        # Fill in cell values
        for cell in grid_cells:
            row = cell['row']
            col = cell['col']
            text = cell.get('text', '').strip()
            table_data[row][col] = text
        
        # Convert to numpy array
        return np.array(table_data, dtype=object)
    
    def _calculate_confidence(self, grid_cells: List[Dict], table_array: np.ndarray) -> float:
        """Calculate confidence score for extracted table"""
        if table_array.size == 0:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Factor 1: Grid regularity (how well text boxes align)
        if len(grid_cells) > 0:
            rows = [cell['row'] for cell in grid_cells]
            cols = [cell['col'] for cell in grid_cells]
            
            row_regularity = len(set(rows)) / len(rows) if rows else 0
            col_regularity = len(set(cols)) / len(cols) if cols else 0
            
            confidence += 0.2 * (row_regularity + col_regularity)
        
        # Factor 2: Cell content (how many cells have text)
        non_empty = np.sum(table_array != '')
        total_cells = table_array.size
        fill_ratio = non_empty / total_cells if total_cells > 0 else 0
        
        confidence += 0.2 * fill_ratio
        
        # Factor 3: Table size (reasonable size indicates higher confidence)
        rows, cols = table_array.shape
        if rows >= self.min_rows and cols >= self.min_cols:
            confidence += 0.1
        
        return min(1.0, confidence)


class StreamTableParser:
    """
    Alternative parser for stream-based tables (space-separated values)
    Useful when tables have consistent spacing but no grid lines
    """
    
    def __init__(self, use_ocr: bool = True):
        self.use_ocr = use_ocr
    
    def parse(self, region: np.ndarray, bbox: BoundingBox,
              page_obj=None, img_width=None, img_height=None) -> Table:
        """Parse stream-based table using whitespace analysis"""
        if region.size == 0:
            return Table(np.array([]), bbox, 0, 0.0)
        
        # Extract all text with positions
        if not self.use_ocr:
            return Table(np.array([]), bbox, 0, 0.5)
        
        text_lines = self._extract_text_lines(region)
        
        if not text_lines:
            return Table(np.array([]), bbox, 0, 0.5)
        
        # Analyze spacing to determine columns
        columns = self._detect_columns(text_lines)
        
        # Parse each line into columns
        table_data = self._parse_lines_to_columns(text_lines, columns)
        
        if not table_data or len(table_data) < 2:
            return Table(np.array([]), bbox, 0, 0.5)
        
        # Convert to numpy array
        table_array = np.array(table_data, dtype=object)
        
        return Table(
            data=table_array,
            bbox=bbox,
            page=1,
            confidence=0.7,
            image=region
        )
    
    def _extract_text_lines(self, image: np.ndarray) -> List[Dict]:
        """Extract text lines with their positions"""
        try:
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # Get line-by-line OCR data
            ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            
            # Group by line
            lines = defaultdict(list)
            n_boxes = len(ocr_data['text'])
            
            for i in range(n_boxes):
                if ocr_data['text'][i].strip() and int(ocr_data['conf'][i]) > 30:
                    line_num = ocr_data['line_num'][i]
                    lines[line_num].append({
                        'text': ocr_data['text'][i].strip(),
                        'x': ocr_data['left'][i],
                        'y': ocr_data['top'][i],
                        'width': ocr_data['width'][i]
                    })
            
            # Convert to sorted list
            text_lines = []
            for line_num in sorted(lines.keys()):
                # Sort words in line by x position
                words = sorted(lines[line_num], key=lambda w: w['x'])
                text_lines.append({
                    'words': words,
                    'y': words[0]['y'] if words else 0
                })
            
            return text_lines
        except Exception as e:
            return []
    
    def _detect_columns(self, text_lines: List[Dict]) -> List[int]:
        """Detect column positions based on text alignment"""
        if not text_lines:
            return []
        
        # Collect all x positions
        all_x_positions = []
        for line in text_lines:
            for word in line['words']:
                all_x_positions.append(word['x'])
        
        if not all_x_positions:
            return []
        
        # Cluster x positions to find column boundaries
        x_array = np.array(all_x_positions).reshape(-1, 1)
        clustering = DBSCAN(eps=30, min_samples=2).fit(x_array)
        
        # Get cluster centers as column positions
        unique_labels = set(clustering.labels_)
        columns = []
        
        for label in unique_labels:
            if label != -1:  # Ignore noise
                cluster_points = x_array[clustering.labels_ == label]
                col_pos = int(np.mean(cluster_points))
                columns.append(col_pos)
        
        return sorted(columns)
    
    def _parse_lines_to_columns(self, text_lines: List[Dict], columns: List[int]) -> List[List[str]]:
        """Parse each line into columns based on column positions"""
        if not columns:
            return []
        
        table_data = []
        
        for line in text_lines:
            row = [''] * len(columns)
            
            for word in line['words']:
                # Find nearest column
                word_x = word['x']
                nearest_col = min(range(len(columns)), 
                                key=lambda i: abs(columns[i] - word_x))
                
                # Add word to that column
                if row[nearest_col]:
                    row[nearest_col] += ' ' + word['text']
                else:
                    row[nearest_col] = word['text']
            
            table_data.append(row)
        
        return table_data
