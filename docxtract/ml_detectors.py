"""
ML-based detectors using Table Transformer and other models
"""

from typing import List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import torch

from .data_structures import BoundingBox, ElementType, Table
from .detectors import BaseDetector

# Try to import pdfplumber for hybrid extraction
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


class TableTransformerDetector(BaseDetector):
    """
    ML-based table detector using Microsoft's Table Transformer
    https://huggingface.co/microsoft/table-transformer-detection
    """
    
    def __init__(self, confidence_threshold: float = 0.7, use_gpu: bool = True):
        """
        Initialize Table Transformer detector
        
        Args:
            confidence_threshold: Minimum confidence score for detections
            use_gpu: Whether to use GPU acceleration if available
        """
        self.confidence_threshold = confidence_threshold
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the table transformer model"""
        try:
            from transformers import AutoImageProcessor, TableTransformerForObjectDetection
            
            model_name = "microsoft/table-transformer-detection"
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = TableTransformerForObjectDetection.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Table Transformer loaded on {self.device}")
        except Exception as e:
            print(f"⚠️ Failed to load Table Transformer: {e}")
            self.model = None
    
    def detect(self, page_image: np.ndarray, page_num: int) -> List[BoundingBox]:
        """
        Detect tables using ML model
        
        Args:
            page_image: Page image as numpy array (BGR format from OpenCV)
            page_num: Page number
            
        Returns:
            List of bounding boxes for detected tables
        """
        if self.model is None:
            print("⚠️ Model not loaded, returning empty list")
            return []
        
        # Convert BGR to RGB for model
        if len(page_image.shape) == 3 and page_image.shape[2] == 3:
            rgb_image = cv2.cvtColor(page_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = page_image
        
        pil_image = Image.fromarray(rgb_image)
        
        # Prepare image for model
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, 
            threshold=self.confidence_threshold,
            target_sizes=target_sizes
        )[0]
        
        # Extract bounding boxes
        tables = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = box.cpu().numpy()
            x1, y1, x2, y2 = box
            
            # Filter by confidence and size
            if score >= self.confidence_threshold:
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                # Filter out very small detections (likely noise)
                if area > 5000:
                    bbox = BoundingBox(
                        float(x1), float(y1), 
                        float(x2), float(y2)
                    )
                    tables.append(bbox)
        
        print(f"📊 Table Transformer detected {len(tables)} tables on page {page_num}")
        return tables


class TableStructureExtractor:
    """
    Extract table structure using Table Transformer Structure Recognition model
    """
    
    def __init__(self, confidence_threshold: float = 0.3, use_gpu: bool = True, pdf_path: str = None):
        """
        Initialize structure extractor
        
        Args:
            confidence_threshold: Minimum confidence for cell detection (lowered to 0.3 for better recall)
            use_gpu: Whether to use GPU if available
            pdf_path: Path to PDF file (for pdfplumber fallback)
        """
        self.confidence_threshold = confidence_threshold
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.pdf_path = pdf_path
        self._load_model()
    
    def _load_model(self):
        """Load the table structure recognition model"""
        try:
            from transformers import AutoImageProcessor, TableTransformerForObjectDetection
            
            model_name = "microsoft/table-transformer-structure-recognition"
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = TableTransformerForObjectDetection.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Table Structure model loaded on {self.device}")
        except Exception as e:
            print(f"⚠️ Failed to load structure model: {e}")
            self.model = None
    
    def extract_structure(self, table_image: np.ndarray, bbox: BoundingBox, 
                         page_obj=None, page_num: int = 0) -> Table:
        """
        Extract table structure and data using multiple methods
        
        Args:
            table_image: Cropped table region (BGR format)
            bbox: Bounding box of the table
            page_obj: PyMuPDF page object for text extraction
            page_num: Page number (0-indexed) for pdfplumber
            
        Returns:
            Table object with extracted data
        """
        if self.model is None:
            return Table(np.array([]), bbox, 0, 0.0)
        
        # Convert to RGB
        if len(table_image.shape) == 3 and table_image.shape[2] == 3:
            rgb_image = cv2.cvtColor(table_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = table_image
        
        pil_image = Image.fromarray(rgb_image)
        
        # Detect table structure (rows, columns, cells)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            threshold=self.confidence_threshold,
            target_sizes=target_sizes
        )[0]
        
        # Parse detected cells/rows/columns
        cells = []
        rows = []
        columns = []
        
        id2label = self.model.config.id2label
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = box.cpu().numpy()
            label_name = id2label[label.item()]
            
            if score >= self.confidence_threshold:
                if label_name == "table row":
                    rows.append((box[1], box[3]))  # y1, y2
                elif label_name == "table column":
                    columns.append((box[0], box[2]))  # x1, x2
                elif label_name == "table":
                    pass  # Skip table label itself
                else:  # Cells or other elements
                    cells.append({
                        'box': box,
                        'label': label_name,
                        'score': float(score)
                    })
        
        # Sort rows and columns
        rows = sorted(rows, key=lambda x: x[0])
        columns = sorted(columns, key=lambda x: x[0])
        
        print(f"     Structure found: {len(rows)} rows, {len(columns)} cols, {len(cells)} cells")
        
        # If structure detection found rows/columns, use them to create grid
        if rows and columns:
            print(f"     ✓ Using row/column grid")
            data = self._extract_cell_data(
                table_image, rows, columns, bbox, page_obj
            )
        elif cells:
            # Fallback: use detected cells directly
            print(f"     ✓ Using detected cells")
            data = self._extract_from_cells(
                table_image, cells, bbox, page_obj
            )
        else:
            # Fallback to grid-based parser
            print(f"     ⚠️  No structure found, trying grid-based fallback")
            from .parsers import GridBasedTableParser
            parser = GridBasedTableParser(min_rows=1, min_cols=1)
            
            # Get page dimensions if available
            img_width = table_image.shape[1] if page_obj else None
            img_height = table_image.shape[0] if page_obj else None
            
            table = parser.parse(table_image, bbox, page_obj, img_width, img_height)
            data = table.data
            
            # If grid parser also fails, try simple PDF text extraction
            if (data is None or data.size == 0) and page_obj is not None:
                print(f"     ℹ️  Grid parser failed, trying pdfplumber extraction")
                data = self._extract_with_pdfplumber(bbox, page_num)
                
                if data is None or data.size == 0:
                    print(f"     ℹ️  Pdfplumber failed, trying simple PDF text extraction")
                    data = self._extract_simple_pdf_text(table_image, bbox, page_obj)
        
        accuracy = len(rows) * len(columns) / max(1, len(cells)) if cells else 0.5
        
        return Table(data, bbox, 0, min(1.0, accuracy), table_image)
    
    def _extract_with_pdfplumber(self, bbox: BoundingBox, page_num: int) -> np.ndarray:
        """Extract table using pdfplumber - excellent for both bordered and borderless tables"""
        if not PDFPLUMBER_AVAILABLE:
            print(f"     ⚠️  pdfplumber not installed")
            return np.array([])
        
        if not self.pdf_path:
            print(f"     ⚠️  PDF path not provided")
            return np.array([])
        
        try:
            # Open PDF with pdfplumber
            pdf = pdfplumber.open(self.pdf_path)
            
            if page_num >= len(pdf.pages):
                pdf.close()
                return np.array([])
            
            page = pdf.pages[page_num]
            
            # Convert bbox from image coordinates (3x scale) to PDF coordinates
            scale = 1.0 / 3.0
            pdf_bbox = (
                bbox.x1 * scale,
                bbox.y1 * scale,
                bbox.x2 * scale,
                bbox.y2 * scale
            )
            
            # Crop to table region
            cropped = page.within_bbox(pdf_bbox)
            
            # Extract tables with better settings
            # Try with explicit table settings first
            table_settings = {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "explicit_vertical_lines": [],
                "explicit_horizontal_lines": [],
                "snap_tolerance": 3,
                "join_tolerance": 3,
                "edge_min_length": 3,
                "min_words_vertical": 1,
                "min_words_horizontal": 1,
                "text_tolerance": 3,
            }
            
            tables = cropped.extract_tables(table_settings)
            
            # If no tables found with lines, try text-based detection
            if not tables:
                table_settings["vertical_strategy"] = "text"
                table_settings["horizontal_strategy"] = "text"
                tables = cropped.extract_tables(table_settings)
            
            pdf.close()
            
            if not tables or len(tables) == 0:
                print(f"     ⚠️  pdfplumber found no tables in region")
                return np.array([])
            
            # Use the first table found
            table_data = tables[0]
            
            # Convert to numpy array
            if table_data:
                # Remove empty rows
                table_data = [row for row in table_data if any(cell for cell in row if cell)]
                
                if table_data:
                    # Convert None to empty string
                    data = np.array([[str(cell) if cell else '' for cell in row] for row in table_data], dtype=object)
                    print(f"     ✓ pdfplumber extracted {data.shape[0]}x{data.shape[1]} table")
                    return data
            
            return np.array([])
            
        except Exception as e:
            print(f"     ⚠️  pdfplumber extraction failed: {e}")
            return np.array([])
    
    def _extract_simple_pdf_text(self, table_image: np.ndarray, bbox: BoundingBox, page_obj) -> np.ndarray:
        """Simple fallback: extract all text from table region and split into cells"""
        try:
            import fitz
            
            # Scale from image coordinates (3x) back to PDF coordinates
            scale = 1.0 / 3.0
            rect = fitz.Rect(
                bbox.x1 * scale, bbox.y1 * scale,
                bbox.x2 * scale, bbox.y2 * scale
            )
            
            # Get all text in table region
            text = page_obj.get_text("text", clip=rect).strip()
            
            if not text:
                return np.array([])
            
            # Split by lines
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            if not lines:
                return np.array([])
            
            # Try to detect columns by splitting on multiple spaces or tabs
            rows = []
            for line in lines:
                # Split on 2+ spaces or tabs
                import re
                cells = re.split(r'\s{2,}|\t+', line)
                rows.append(cells)
            
            # Pad rows to same length
            if rows:
                max_cols = max(len(row) for row in rows)
                data = np.empty((len(rows), max_cols), dtype=object)
                for i, row in enumerate(rows):
                    for j, cell in enumerate(row):
                        data[i, j] = cell
                    # Fill remaining with empty strings
                    for j in range(len(row), max_cols):
                        data[i, j] = ''
                
                print(f"     ✓ Extracted {data.shape[0]}x{data.shape[1]} from PDF text")
                return data
            
            return np.array([])
        except Exception as e:
            print(f"     ⚠️  Simple PDF extraction failed: {e}")
            return np.array([])
    
    def _extract_cell_data(self, table_image: np.ndarray, rows: List[Tuple],
                          columns: List[Tuple], bbox: BoundingBox,
                          page_obj) -> np.ndarray:
        """Extract text from grid cells using PDF text first, then OCR"""
        n_rows = len(rows)
        n_cols = len(columns)
        
        if n_rows == 0 or n_cols == 0:
            return np.array([])
        
        # Initialize data array
        data = np.empty((n_rows, n_cols), dtype=object)
        data.fill('')
        
        # Method 1: Try PDF text extraction first (faster and more accurate)
        if page_obj is not None:
            try:
                import fitz
                text_extracted = False
                
                for i, (y1, y2) in enumerate(rows):
                    for j, (x1, x2) in enumerate(columns):
                        # Convert image coords to PDF coords (table_image is cropped from page)
                        # Note: bbox contains the table position in the full page image
                        pdf_x1 = bbox.x1 + x1
                        pdf_y1 = bbox.y1 + y1
                        pdf_x2 = bbox.x1 + x2
                        pdf_y2 = bbox.y1 + y2
                        
                        # Scale from image coordinates (3x) back to PDF coordinates
                        scale = 1.0 / 3.0
                        rect = fitz.Rect(
                            pdf_x1 * scale, pdf_y1 * scale,
                            pdf_x2 * scale, pdf_y2 * scale
                        )
                        
                        text = page_obj.get_text("text", clip=rect).strip()
                        if text:
                            data[i, j] = text
                            text_extracted = True
                
                # If we extracted any text from PDF, return it
                if text_extracted:
                    print(f"     ✓ Extracted text from PDF")
                    return data
                else:
                    print(f"     ℹ️  No text in PDF, trying OCR...")
            except Exception as e:
                print(f"     ⚠️  PDF text extraction failed: {e}")
        
        # Method 2: OCR fallback
        try:
            import pytesseract
            # Check if Tesseract is actually available
            try:
                pytesseract.get_tesseract_version()
            except Exception:
                print(f"     ⚠️  Tesseract OCR not installed - cannot extract text")
                return np.array([])
        except ImportError:
            print(f"     ⚠️  pytesseract not installed - cannot extract text")
            return np.array([])
        
        n_rows = len(rows)
        n_cols = len(columns)
        
        if n_rows == 0 or n_cols == 0:
            return np.array([])
        
        # Initialize data array
        data = np.empty((n_rows, n_cols), dtype=object)
        data.fill('')
        
        # Extract text from each cell
        for i, (y1, y2) in enumerate(rows):
            for j, (x1, x2) in enumerate(columns):
                # Crop cell region
                y1_int = int(max(0, y1))
                y2_int = int(min(table_image.shape[0], y2))
                x1_int = int(max(0, x1))
                x2_int = int(min(table_image.shape[1], x2))
                
                if y2_int <= y1_int or x2_int <= x1_int:
                    continue
                
                cell_img = table_image[y1_int:y2_int, x1_int:x2_int]
                
                if cell_img.size == 0:
                    continue
                
                # OCR on cell
                try:
                    # Preprocess cell for better OCR
                    if len(cell_img.shape) == 3:
                        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = cell_img
                    
                    # Multiple thresholding attempts for better text extraction
                    # Try Otsu's method first
                    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    text = pytesseract.image_to_string(
                        thresh1,
                        config='--psm 6 --oem 3'
                    ).strip()
                    
                    # If no text, try adaptive thresholding
                    if not text and gray.shape[0] > 10 and gray.shape[1] > 10:
                        thresh2 = cv2.adaptiveThreshold(
                            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 11, 2
                        )
                        text = pytesseract.image_to_string(
                            thresh2,
                            config='--psm 6 --oem 3'
                        ).strip()
                    
                    # If still no text, try without thresholding
                    if not text:
                        text = pytesseract.image_to_string(
                            gray,
                            config='--psm 6 --oem 3'
                        ).strip()
                    
                    data[i, j] = text
                except Exception as e:
                    data[i, j] = ''
        
        return data
    
    def _extract_from_cells(self, table_image: np.ndarray, cells: List[dict],
                           bbox: BoundingBox, page_obj) -> np.ndarray:
        """Extract data when individual cells are detected"""
        try:
            import pytesseract
        except ImportError:
            return np.array([])
        
        if not cells:
            return np.array([])
        
        # Determine grid dimensions by clustering cell positions
        y_positions = sorted(set([c['box'][1] for c in cells] + [c['box'][3] for c in cells]))
        x_positions = sorted(set([c['box'][0] for c in cells] + [c['box'][2] for c in cells]))
        
        # Simple clustering to find rows/columns
        rows = self._cluster_positions(y_positions, threshold=10)
        cols = self._cluster_positions(x_positions, threshold=10)
        
        if len(rows) < 2 or len(cols) < 2:
            return np.array([])
        
        n_rows = len(rows) - 1
        n_cols = len(cols) - 1
        
        data = np.empty((n_rows, n_cols), dtype=object)
        data.fill('')
        
        # Assign each cell to grid position
        for cell in cells:
            box = cell['box']
            center_y = (box[1] + box[3]) / 2
            center_x = (box[0] + box[2]) / 2
            
            # Find row and column
            row_idx = self._find_position(center_y, rows) - 1
            col_idx = self._find_position(center_x, cols) - 1
            
            if 0 <= row_idx < n_rows and 0 <= col_idx < n_cols:
                # Extract text from cell
                y1, y2 = int(box[1]), int(box[3])
                x1, x2 = int(box[0]), int(box[2])
                
                y1 = max(0, y1)
                y2 = min(table_image.shape[0], y2)
                x1 = max(0, x1)
                x2 = min(table_image.shape[1], x2)
                
                if y2 > y1 and x2 > x1:
                    cell_img = table_image[y1:y2, x1:x2]
                    
                    try:
                        if len(cell_img.shape) == 3:
                            gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
                        else:
                            gray = cell_img
                        
                        # Try multiple thresholding approaches
                        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        text = pytesseract.image_to_string(
                            thresh,
                            config='--psm 6 --oem 3'
                        ).strip()
                        
                        if not text and gray.shape[0] > 10 and gray.shape[1] > 10:
                            thresh2 = cv2.adaptiveThreshold(
                                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2
                            )
                            text = pytesseract.image_to_string(
                                thresh2,
                                config='--psm 6 --oem 3'
                            ).strip()
                        
                        data[row_idx, col_idx] = text
                    except Exception:
                        pass
        
        return data
    
    def _cluster_positions(self, positions: List[float], threshold: float = 10) -> List[float]:
        """Cluster nearby positions"""
        if not positions:
            return []
        
        clustered = [positions[0]]
        for pos in positions[1:]:
            if pos - clustered[-1] > threshold:
                clustered.append(pos)
        
        return clustered
    
    def _find_position(self, value: float, positions: List[float]) -> int:
        """Find which position bucket a value falls into"""
        for i, pos in enumerate(positions):
            if value < pos:
                return i
        return len(positions)
