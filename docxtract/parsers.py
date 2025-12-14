"""
Parsing strategies for extracted elements - FIXED VERSION
"""

from abc import ABC, abstractmethod
from typing import List, Union, Tuple
import numpy as np
import cv2

from .data_structures import BoundingBox, Table


class BaseParser(ABC):
    """Base class for parsing detected elements"""

    @abstractmethod
    def parse(self, region: np.ndarray, bbox: BoundingBox) -> Table:
        """Parse a detected region into structured data"""
        pass


class GridBasedTableParser(BaseParser):
    """Parse tables by detecting grid structure - improved for accuracy"""

    def __init__(self, min_rows: int = 2, min_cols: int = 2):
        """
        Initialize parser
        
        Args:
            min_rows: Minimum number of rows for valid table
            min_cols: Minimum number of columns for valid table
        """
        self.min_rows = min_rows
        self.min_cols = min_cols

    def parse(self, region: np.ndarray, bbox: BoundingBox) -> Table:
        """Parse table region into structured data with better validation"""
        if region.size == 0:
            return Table(np.array([]), bbox, 0, 0.0)

        # Normalize region
        if len(region.shape) < 3:
            region = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Check if region has table-like structure before processing
        if not self._has_table_content(gray):
            return Table(np.array([]), bbox, 0, 0.0)

        # Detect lines
        edges = cv2.Canny(gray, 30, 100, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=80,
            minLineLength=max(30, region.shape[1]//8),
            maxLineGap=10
        )

        if lines is None or len(lines) < 4:  # Need minimum lines for a table
            return Table(np.array([]), bbox, 0, 0.0)

        # Separate horizontal and vertical lines with better accuracy
        h_lines = []
        v_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            
            # Calculate angle more precisely
            if dx == 0:
                angle = 90
            else:
                angle = np.abs(np.arctan2(dy, dx) * 180 / np.pi)

            if angle < 15 or angle > 165:  # Horizontal (with tolerance)
                h_lines.append((y1 + y2) / 2)
            elif 75 < angle < 105:  # Vertical (with tolerance)
                v_lines.append((x1 + x2) / 2)

        # Remove duplicates by clustering nearby lines
        h_lines = self._cluster_lines(sorted(h_lines))
        v_lines = self._cluster_lines(sorted(v_lines))

        # Need at least minimum rows and columns
        if len(h_lines) < self.min_rows + 1 or len(v_lines) < self.min_cols + 1:
            return Table(np.array([]), bbox, 0, 0.0)

        # Extract cells using OCR
        cells_data = self._extract_cells(region, h_lines, v_lines)

        # Convert to 2D numpy array
        data = self._cells_to_array(cells_data, len(h_lines) - 1, len(v_lines) - 1)

        # Validate table has actual content
        if not self._has_content(data):
            return Table(np.array([]), bbox, 0, 0.0)

        # Calculate accuracy based on grid consistency
        accuracy = self._calculate_accuracy(len(h_lines), len(v_lines), len(cells_data))

        return Table(data, bbox, 0, accuracy, region)

    def _has_table_content(self, gray: np.ndarray) -> bool:
        """Check if region looks like a table"""
        # Should have reasonable contrast
        mean = np.mean(gray)
        std = np.std(gray)
        
        if std < 8:  # Too uniform
            return False
        
        # Should not be too bright or too dark
        if mean < 30 or mean > 225:
            return False
            
        return True

    def _cluster_lines(self, lines: List[float], tolerance: int = 5) -> List[int]:
        """Cluster nearby lines to remove duplicates"""
        if not lines:
            return []
        
        clustered = []
        current_cluster = [lines[0]]
        
        for line in lines[1:]:
            if abs(line - current_cluster[-1]) < tolerance:
                current_cluster.append(line)
            else:
                # Use average of cluster
                clustered.append(int(np.mean(current_cluster)))
                current_cluster = [line]
        
        clustered.append(int(np.mean(current_cluster)))
        return clustered

    def _has_content(self, data: np.ndarray) -> bool:
        """Check if table has actual text content"""
        if data.size == 0:
            return False

        # Check if at least some cells have non-empty content
        total_cells = data.size
        
        if total_cells == 0:
            return False
            
        # Count empty/placeholder cells
        empty_cells = 0
        total_text_length = 0
        has_real_content = False
        
        for row in data:
            for cell in row:
                cell_str = str(cell).strip()
                if not cell_str or cell_str == '' or cell_str.startswith('[Cell '):
                    empty_cells += 1
                else:
                    total_text_length += len(cell_str)
                    has_real_content = True

        # If we have any real content, consider it valid
        if has_real_content:
            return True
            
        # If all cells are placeholders (OCR not available), still consider it a valid table structure
        # as long as we have the expected number of cells
        placeholder_ratio = empty_cells / total_cells if total_cells > 0 else 1.0
        
        return placeholder_ratio < 0.8  # Allow up to 80% placeholders for OCR-less operation

    def _extract_cells(self, image: np.ndarray, h_lines: List[int], 
                       v_lines: List[int]) -> List[Tuple[int, int, str]]:
        """Extract text from grid cells using OCR (robust to missing pytesseract)"""
        cells = []

        # Try to import pytesseract, fall back to empty text if unavailable
        try:
            import pytesseract
            has_ocr = True
        except ImportError:
            has_ocr = False
            print("Warning: pytesseract not available. Install it for OCR support.")

        # Check if Tesseract is actually installed
        tesseract_available = False
        if has_ocr:
            try:
                # Try to get tesseract version to check if it's installed
                pytesseract.get_tesseract_version()
                tesseract_available = True
            except Exception:
                print("Warning: Tesseract OCR not found. Table extraction will return empty cells.")
                print("Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")

        for i in range(len(h_lines) - 1):
            for j in range(len(v_lines) - 1):
                y1 = int(h_lines[i])
                y2 = int(h_lines[i + 1])
                x1 = int(v_lines[j])
                x2 = int(v_lines[j + 1])

                # Ensure valid coordinates
                if y2 <= y1 or x2 <= x1:
                    continue
                
                # Ensure within image bounds
                y1 = max(0, y1)
                x1 = max(0, x1)
                y2 = min(image.shape[0], y2)
                x2 = min(image.shape[1], x2)

                # Extract cell region
                cell_region = image[y1:y2, x1:x2]

                text = ''
                if tesseract_available and cell_region.size > 0:
                    try:
                        # Preprocess cell for better OCR
                        cell_gray = cv2.cvtColor(cell_region, cv2.COLOR_BGR2GRAY)
                        # Apply thresholding
                        _, cell_thresh = cv2.threshold(
                            cell_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                        )
                        
                        text = pytesseract.image_to_string(
                            cell_thresh,
                            config='--psm 6 --oem 3'
                        ).strip()
                    except Exception as e:
                        text = ''
                elif not tesseract_available:
                    # Provide placeholder text indicating OCR is not available
                    text = f"[Cell {i},{j}]"  # Placeholder for missing OCR

                cells.append((i, j, text))

        return cells

    def _cells_to_array(self, cells: List[Tuple[int, int, str]], 
                       n_rows: int, n_cols: int) -> np.ndarray:
        """Convert cells to 2D numpy array"""
        if not cells:
            return np.array([])

        # Initialize array with empty strings
        data = np.empty((n_rows, n_cols), dtype=object)
        data.fill('')

        # Fill in cell data
        for row, col, text in cells:
            if 0 <= row < n_rows and 0 <= col < n_cols:
                data[row, col] = text

        return data

    def _calculate_accuracy(self, n_h_lines: int, n_v_lines: int, n_cells: int) -> float:
        """Calculate parsing accuracy score"""
        expected_cells = max(0, (n_h_lines - 1) * (n_v_lines - 1))
        if expected_cells == 0:
            return 0.0
        
        actual_cells = n_cells
        accuracy = min(1.0, actual_cells / expected_cells)
        
        return accuracy


class TextBasedTableParser(BaseParser):
    """Parse tables by clustering text elements - improved for accuracy"""

    def __init__(self, row_tolerance: int = 20, col_tolerance: int = 50,
                 min_rows: int = 2, min_cols: int = 2):
        """
        Initialize parser
        
        Args:
            row_tolerance: Max vertical distance to consider text in same row
            col_tolerance: Max horizontal distance to consider text in same column
            min_rows: Minimum number of rows for valid table
            min_cols: Minimum number of columns for valid table
        """
        self.row_tolerance = row_tolerance
        self.col_tolerance = col_tolerance
        self.min_rows = min_rows
        self.min_cols = min_cols

    def parse(self, region: np.ndarray, bbox: BoundingBox) -> Table:
        """Parse table region into structured data with better validation"""
        if region.size == 0:
            return Table(np.array([]), bbox, 0, 0.0)

        # Try to import pytesseract
        try:
            import pytesseract
        except ImportError:
            print("Warning: pytesseract not available. Cannot parse text-based tables.")
            return Table(np.array([]), bbox, 0, 0.0)

        try:
            # Extract text using OCR
            text_data = pytesseract.image_to_data(
                region, 
                output_type=pytesseract.Output.DICT,
                config='--psm 6 --oem 3'
            )
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            return Table(np.array([]), bbox, 0, 0.0)

        # Filter out empty text and low confidence results
        valid_indices = [
            i for i, text in enumerate(text_data['text'])
            if text.strip() and int(text_data['conf'][i]) > 30
        ]

        if len(valid_indices) < 4:  # Need minimum text elements for a table
            return Table(np.array([]), bbox, 0, 0.0)

        # Extract positions and text
        positions = []
        texts = []

        for i in valid_indices:
            x = text_data['left'][i]
            y = text_data['top'][i]
            w = text_data['width'][i]
            h = text_data['height'][i]
            
            positions.append((x, y, w, h))
            texts.append(text_data['text'][i])

        # Cluster into rows and columns
        rows, cols = self._cluster_text(positions)

        if len(rows) < self.min_rows or len(cols) < self.min_cols:
            return Table(np.array([]), bbox, 0, 0.0)

        # Build table data
        data = self._build_table_data(texts, positions, rows, cols)

        # Validate table has actual content
        if not self._has_content(data):
            return Table(np.array([]), bbox, 0, 0.0)

        # Calculate accuracy based on text confidence
        avg_conf = np.mean([text_data['conf'][i] for i in valid_indices])
        accuracy = min(avg_conf / 100.0, 1.0)

        return Table(data, bbox, 0, accuracy, region)

    def _has_content(self, data: np.ndarray) -> bool:
        """Check if table has actual text content"""
        if data.size == 0:
            return False

        # Count non-empty cells
        non_empty = 0
        for row in data:
            for cell in row:
                if str(cell).strip():
                    non_empty += 1

        total_cells = data.size
        
        # If more than 80% cells are empty, consider it not a real table
        return (non_empty / total_cells) > 0.2 if total_cells > 0 else False

    def _cluster_text(self, positions: List[Tuple[int, int, int, int]]) -> Tuple[List[int], List[int]]:
        """Cluster text positions into rows and columns"""
        if not positions:
            return [], []

        # Extract y-coordinates for rows (use center of text)
        y_coords = [pos[1] + pos[3]//2 for pos in positions]
        y_coords.sort()

        # Cluster y-coordinates into rows
        rows = []
        current_row = [y_coords[0]]

        for y in y_coords[1:]:
            if y - current_row[-1] > self.row_tolerance:
                rows.append(int(np.mean(current_row)))
                current_row = [y]
            else:
                current_row.append(y)

        if current_row:
            rows.append(int(np.mean(current_row)))

        # Extract x-coordinates for columns (use left edge)
        x_coords = [pos[0] for pos in positions]
        x_coords.sort()

        # Cluster x-coordinates into columns
        cols = []
        current_col = [x_coords[0]]

        for x in x_coords[1:]:
            if x - current_col[-1] > self.col_tolerance:
                cols.append(int(np.mean(current_col)))
                current_col = [x]
            else:
                current_col.append(x)

        if current_col:
            cols.append(int(np.mean(current_col)))

        return rows, cols

    def _build_table_data(self, texts: List[str], 
                         positions: List[Tuple[int, int, int, int]],
                         rows: List[int], cols: List[int]) -> np.ndarray:
        """Build 2D table data from clustered text"""
        # Initialize array
        data = np.empty((len(rows), len(cols)), dtype=object)
        data.fill('')

        for text, (x, y, w, h) in zip(texts, positions):
            # Use center of text for row matching
            y_center = y + h // 2
            
            # Find closest row and column
            row_idx = min(range(len(rows)), key=lambda i: abs(y_center - rows[i]))
            col_idx = min(range(len(cols)), key=lambda i: abs(x - cols[i]))

            # Append text if cell already has content (handle merged cells)
            if data[row_idx, col_idx]:
                data[row_idx, col_idx] += ' ' + text
            else:
                data[row_idx, col_idx] = text

        return data


class HybridTableParser(BaseParser):
    """
    Hybrid parser that tries grid-based first, falls back to text-based
    """

    def __init__(self):
        self.grid_parser = GridBasedTableParser()
        self.text_parser = TextBasedTableParser()

    def parse(self, region: np.ndarray, bbox: BoundingBox) -> Table:
        """Parse using grid-based method, fall back to text-based if needed"""
        
        # Try grid-based first (faster and more accurate for bordered tables)
        table = self.grid_parser.parse(region, bbox)
        
        # If grid-based fails or has low confidence, try text-based
        if table.data.size == 0 or table.confidence < 0.3:
            table = self.text_parser.parse(region, bbox)
        
        return table


class SimpleTableParser(BaseParser):
    """
    Simple parser that just extracts the region without OCR
    Useful when you just want table locations, not content
    """

    def parse(self, region: np.ndarray, bbox: BoundingBox) -> Table:
        """Return table with just the image region, no text extraction"""
        # Create empty data array with estimated dimensions
        h, w = region.shape[:2]
        
        # Estimate rows and columns based on image size
        estimated_rows = max(2, h // 30)  # Assume ~30px per row
        estimated_cols = max(2, w // 100)  # Assume ~100px per column
        
        # Create placeholder data
        data = np.empty((estimated_rows, estimated_cols), dtype=object)
        data.fill('[Not extracted]')
        
        return Table(data, bbox, 0, 0.5, region)