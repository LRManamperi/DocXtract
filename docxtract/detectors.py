"""
Detection strategies for tables and graphs - FIXED VERSION with Axis-Based Detection
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2

from .data_structures import BoundingBox, ElementType


class BaseDetector(ABC):
    """Base class for element detection strategies"""

    @abstractmethod
    def detect(self, page_image: np.ndarray, page_num: int) -> List[BoundingBox]:
        """Detect elements and return bounding boxes"""
        pass


class LineBasedTableDetector(BaseDetector):
    """
    Detect tables based on line structures (borders)
    Similar to Camelot's lattice mode - improved for charts
    """

    def __init__(self, min_lines: int = 4, line_scale: int = 15, min_table_area: int = 5000):
        self.min_lines = min_lines
        self.line_scale = line_scale
        self.min_table_area = min_table_area

    def detect(self, page_image: np.ndarray, page_num: int) -> List[BoundingBox]:
        """Detect tables using line detection with better filtering"""
        gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(1, page_image.shape[1]//self.line_scale), 1)
        )
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(1, page_image.shape[0]//self.line_scale))
        )

        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)

        # Combine lines
        table_mask = cv2.add(h_lines, v_lines)

        # Find contours
        contours, _ = cv2.findContours(
            table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        bboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Stricter filtering for tables vs charts
            area = w * h
            if area < self.min_table_area:
                continue

            # Tables typically have more regular aspect ratios than charts
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                continue

            # Check if it looks like a table (has both horizontal and vertical lines)
            if self._has_table_structure(table_mask[y:y+h, x:x+w]):
                bboxes.append(BoundingBox(x, y, x+w, y+h))

        return bboxes

    def _has_table_structure(self, region_mask):
        """Check if region has table-like line structure"""
        # Look for both horizontal and vertical lines in the region
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))

        h_lines = cv2.morphologyEx(region_mask, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(region_mask, cv2.MORPH_OPEN, v_kernel)

        h_count = np.count_nonzero(h_lines)
        v_count = np.count_nonzero(v_lines)

        # Need both horizontal and vertical lines for a table
        return h_count > 100 and v_count > 100


class TextClusterTableDetector(BaseDetector):
    """
    Detect tables based on text alignment patterns
    Similar to Camelot's stream mode
    """

    def __init__(self, row_tol: float = 2, col_tol: float = 5):
        self.row_tol = row_tol
        self.col_tol = col_tol

    def detect(self, page_image: np.ndarray, page_num: int) -> List[BoundingBox]:
        """Detect tables using text clustering"""
        # This would use PyMuPDF text extraction
        # Simplified version here
        gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)

        # Detect text regions
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Find connected components (text blocks)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Cluster nearby text blocks
        text_blocks = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10 and h > 10:  # Minimum text size
                text_blocks.append((x, y, w, h))

        # Group into table regions (simplified)
        bboxes = self._cluster_text_blocks(text_blocks)
        return bboxes

    def _cluster_text_blocks(self, blocks: List[Tuple]) -> List[BoundingBox]:
        """Cluster text blocks into table regions"""
        if not blocks:
            return []

        # Sort by y-coordinate
        blocks = sorted(blocks, key=lambda b: b[1])

        # Group into rows
        rows = []
        current_row = [blocks[0]]

        for block in blocks[1:]:
            if abs(block[1] - current_row[-1][1]) < self.row_tol:
                current_row.append(block)
            else:
                rows.append(current_row)
                current_row = [block]
        rows.append(current_row)

        # Find rectangular table regions
        bboxes = []
        if len(rows) >= 2:
            x_min = min(b[0] for row in rows for b in row)
            y_min = min(b[1] for row in rows for b in row)
            x_max = max(b[0] + b[2] for row in rows for b in row)
            y_max = max(b[1] + b[3] for row in rows for b in row)

            bboxes.append(BoundingBox(x_min, y_min, x_max, y_max))

        return bboxes


class MLTableDetector(BaseDetector):
    """
    ML-based table detection
    Can be extended with custom models (YOLO, Faster R-CNN, etc.)
    """

    def __init__(self, model_path: Optional[str] = None, confidence: float = 0.5):
        self.model_path = model_path
        self.confidence = confidence
        self.model = None

        # Load model if provided
        if model_path:
            self._load_model()

    def _load_model(self):
        """Load custom ML model"""
        # Placeholder for custom model loading
        # Could use TensorFlow, PyTorch, or ONNX
        pass

    def detect(self, page_image: np.ndarray, page_num: int) -> List[BoundingBox]:
        """Detect tables using ML model"""
        if self.model is None:
            return []

        # Run inference
        # This is a placeholder - implement based on your model
        predictions = []
        return predictions


class GraphDetector(BaseDetector):
    """Detect and classify graphs/charts using AXIS-BASED detection"""

    def __init__(self, min_area: int = 10000, max_area_ratio: float = 0.8, 
                 min_confidence: float = 0.4):
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.min_confidence = min_confidence

    def detect(self, page_image: np.ndarray, page_num: int) -> List[Tuple[BoundingBox, ElementType, float, Dict]]:
        """
        Detect charts based on axes (one chart per axis pair)
        
        Returns:
            List of tuples: (bbox, chart_type, confidence, chart_data)
        """
        height, width = page_image.shape[:2]
        page_area = width * height

        gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Detect axes in the image
        axes = self._detect_axes(gray, page_image)
        
        if not axes:
            return []
        
        # Step 2: For each axis pair, detect the chart and extract data
        results = []
        for axis_info in axes:
            bbox = axis_info['bbox']
            
            # Check area constraints
            area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)
            if area < self.min_area or area > page_area * self.max_area_ratio:
                continue
            
            # Extract chart region
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
            chart_region = page_image[y1:y2, x1:x2]
            
            if chart_region.size == 0:
                continue
            
            # Classify chart type
            chart_type, confidence = self._classify_chart_from_axis(
                chart_region, axis_info
            )
            
            if confidence >= self.min_confidence:
                # Extract chart data
                chart_data = self._extract_chart_data(
                    chart_region, chart_type, axis_info
                )
                
                results.append((bbox, chart_type, confidence, chart_data))
        
        return results

    def _detect_axes(self, gray: np.ndarray, color_image: np.ndarray) -> List[Dict]:
        """Detect chart axes - each axis pair represents one chart"""
        h, w = gray.shape
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=50,
            minLineLength=max(30, min(h, w)//10),
            maxLineGap=10
        )
        
        if lines is None:
            return []
        
        # Separate into horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            if length < max(20, min(h, w)//15):  # Too short
                continue
            
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Horizontal line (close to 0° or 180°)
            if angle < 10 or angle > 170:
                horizontal_lines.append((x1, y1, x2, y2, length))
            # Vertical line (close to 90°)
            elif 80 < angle < 100:
                vertical_lines.append((x1, y1, x2, y2, length))
        
        # Find axis pairs (one horizontal + one vertical = one chart)
        axes = []
        matched_h = set()
        matched_v = set()
        
        for i, (hx1, hy1, hx2, hy2, h_len) in enumerate(horizontal_lines):
            if i in matched_h:
                continue
                
            for j, (vx1, vy1, vx2, vy2, v_len) in enumerate(vertical_lines):
                if j in matched_v:
                    continue
                
                # Check if they form an L-shape (axis)
                if self._forms_axis(hx1, hy1, hx2, hy2, vx1, vy1, vx2, vy2):
                    # Found an axis pair - define chart bounding box
                    intersection_x = vx1  # Vertical line x
                    intersection_y = hy1  # Horizontal line y
                    
                    # Chart extends from axis intersection
                    x_min = min(vx1, vx2) - 5
                    y_min = min(vy1, vy2) - 5
                    x_max = max(hx1, hx2) + 5
                    y_max = max(hy1, hy2) + 5
                    
                    # Clamp to image bounds
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(w, x_max)
                    y_max = min(h, y_max)
                    
                    bbox = BoundingBox(x_min, y_min, x_max, y_max)
                    
                    axes.append({
                        'bbox': bbox,
                        'h_axis': (hx1, hy1, hx2, hy2),
                        'v_axis': (vx1, vy1, vx2, vy2),
                        'intersection': (intersection_x, intersection_y)
                    })
                    
                    matched_h.add(i)
                    matched_v.add(j)
                    break  # Move to next horizontal line
        
        return axes

    def _forms_axis(self, hx1, hy1, hx2, hy2, vx1, vy1, vx2, vy2) -> bool:
        """Check if horizontal and vertical lines form an axis (L-shape)"""
        # Check if they intersect or are very close
        
        # Vertical line should be near the start/end of horizontal line
        h_left_x = min(hx1, hx2)
        h_right_x = max(hx1, hx2)
        v_x = vx1  # Vertical line x-coordinate
        
        # Horizontal line should be near the start/end of vertical line
        v_top_y = min(vy1, vy2)
        v_bottom_y = max(vy1, vy2)
        h_y = hy1  # Horizontal line y-coordinate
        
        # Check proximity
        x_intersects = h_left_x - 20 <= v_x <= h_left_x + 20
        y_intersects = v_bottom_y - 20 <= h_y <= v_bottom_y + 20
        
        return x_intersects and y_intersects

    def _classify_chart_from_axis(self, chart_region: np.ndarray, 
                                   axis_info: Dict) -> Tuple[ElementType, float]:
        """Classify chart type based on visual features within axis bounds"""
        if chart_region.size == 0:
            return ElementType.UNKNOWN, 0.0
        
        gray = cv2.cvtColor(chart_region, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Check for sufficient variation
        if np.std(gray) < 15:
            return ElementType.UNKNOWN, 0.0
        
        type_scores = {
            ElementType.PIE_CHART: 0.0,
            ElementType.BAR_CHART: 0.0,
            ElementType.LINE_CHART: 0.0,
            ElementType.SCATTER_PLOT: 0.0
        }
        
        # Pie charts: detect circles
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(h, w)//4,
            param1=50, param2=30, minRadius=max(5, min(h, w)//10),
            maxRadius=min(h, w)//2
        )
        
        if circles is not None and len(circles[0]) >= 1:
            type_scores[ElementType.PIE_CHART] = 0.85
        
        # Detect edges for other features
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Bar chart detection
        bars = self._detect_bars(contours, h, w)
        if bars['count'] >= 2:
            type_scores[ElementType.BAR_CHART] = min(0.90, 0.5 + bars['count'] * 0.08)
        
        # Line chart detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                                minLineLength=max(20, min(h, w)//8), maxLineGap=15)
        
        if lines is not None:
            angled_lines = sum(1 for line in lines 
                             if self._is_angled_line(line[0]))
            if angled_lines >= 2:
                type_scores[ElementType.LINE_CHART] = min(0.88, 0.45 + angled_lines * 0.08)
        
        # Scatter plot detection
        scatter_score = self._detect_scatter_points(chart_region)
        type_scores[ElementType.SCATTER_PLOT] = scatter_score
        
        # Return best match
        best_type = max(type_scores.items(), key=lambda x: x[1])
        return best_type[0], best_type[1]

    def _detect_bars(self, contours, h, w) -> Dict:
        """Detect rectangular bars in chart"""
        vertical_bars = []
        horizontal_bars = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            
            x, y, bw, bh = cv2.boundingRect(cnt)
            
            if bw > 5 and bh > 5:
                aspect_ratio = bw / bh if bh > 0 else 0
                
                # Vertical bars
                if 0.1 < aspect_ratio < 0.5 and bh > bw * 2:
                    vertical_bars.append({'x': x, 'y': y, 'w': bw, 'h': bh})
                # Horizontal bars
                elif aspect_ratio > 2.0 and bw > bh * 2:
                    horizontal_bars.append({'x': x, 'y': y, 'w': bw, 'h': bh})
        
        if len(vertical_bars) > len(horizontal_bars):
            return {'count': len(vertical_bars), 'orientation': 'vertical', 'bars': vertical_bars}
        else:
            return {'count': len(horizontal_bars), 'orientation': 'horizontal', 'bars': horizontal_bars}

    def _is_angled_line(self, line) -> bool:
        """Check if line is angled (not axis line)"""
        x1, y1, x2, y2 = line
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        return 15 < angle < 75 or 105 < angle < 165

    def _detect_scatter_points(self, image: np.ndarray) -> float:
        """Detect scatter plot points"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        points = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 30 < area < 300:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    points.append((cx, cy))
        
        if len(points) >= 8:
            coords = np.array(points)
            std_x = np.std(coords[:, 0])
            std_y = np.std(coords[:, 1])
            
            if std_x > 20 and std_y > 20:
                return min(0.85, 0.35 + len(points) * 0.03)
        
        return 0.0

    def _extract_chart_data(self, chart_region: np.ndarray, 
                           chart_type: ElementType, 
                           axis_info: Dict) -> Dict:
        """Extract data from chart based on type"""
        data = {
            'type': chart_type.name,
            'values': [],
            'labels': [],
            'axis_labels': {}
        }
        
        if chart_type == ElementType.BAR_CHART:
            data.update(self._extract_bar_data(chart_region))
        elif chart_type == ElementType.LINE_CHART:
            data.update(self._extract_line_data(chart_region))
        elif chart_type == ElementType.PIE_CHART:
            data.update(self._extract_pie_data(chart_region))
        elif chart_type == ElementType.SCATTER_PLOT:
            data.update(self._extract_scatter_data(chart_region))
        
        return data

    def _extract_bar_data(self, image: np.ndarray) -> Dict:
        """Extract bar values from bar chart"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = gray.shape
        bars = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = bw / bh if bh > 0 else 0
            
            # Detect vertical bars
            if 0.1 < aspect_ratio < 0.5 and bh > bw * 2:
                bars.append({
                    'x': x + bw//2,  # Center x
                    'height': bh,
                    'base_y': y + bh,  # Bottom of bar
                    'value': (h - (y + bh)) / h  # Normalized height
                })
            # Detect horizontal bars
            elif aspect_ratio > 2.0 and bw > bh * 2:
                bars.append({
                    'y': y + bh//2,  # Center y
                    'width': bw,
                    'base_x': x,  # Left of bar
                    'value': bw / w  # Normalized width
                })
        
        # Sort bars by position
        if bars:
            if 'x' in bars[0]:
                bars.sort(key=lambda b: b['x'])
            else:
                bars.sort(key=lambda b: b['y'])
        
        return {
            'values': [b['value'] for b in bars],
            'bar_count': len(bars),
            'bars': bars
        }

    def _extract_line_data(self, image: np.ndarray) -> Dict:
        """Extract line data points from line chart"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                                minLineLength=20, maxLineGap=15)
        
        points = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if self._is_angled_line(line[0]):
                    points.extend([(x1, y1), (x2, y2)])
        
        # Remove duplicates and sort by x
        points = list(set(points))
        points.sort(key=lambda p: p[0])
        
        h, w = gray.shape
        normalized_points = [(x/w, 1 - y/h) for x, y in points]
        
        return {
            'points': normalized_points,
            'point_count': len(points),
            'raw_points': points
        }

    def _extract_pie_data(self, image: np.ndarray) -> Dict:
        """Extract pie slice information"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find lines emanating from center (slice boundaries)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                                minLineLength=20, maxLineGap=10)
        
        h, w = gray.shape
        center = (w//2, h//2)
        
        slice_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line passes near center
                dist1 = np.sqrt((x1 - center[0])**2 + (y1 - center[1])**2)
                dist2 = np.sqrt((x2 - center[0])**2 + (y2 - center[1])**2)
                
                if min(dist1, dist2) < min(h, w) // 4:
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    slice_lines.append(angle)
        
        # Count slices
        slice_count = len(set([round(a/10)*10 for a in slice_lines]))  # Group similar angles
        
        return {
            'slice_count': max(slice_count, 2),
            'slices': slice_lines
        }

    def _extract_scatter_data(self, image: np.ndarray) -> Dict:
        """Extract scatter plot points"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        points = []
        h, w = gray.shape
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 30 < area < 300:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Normalize to 0-1 range
                    points.append((cx/w, 1 - cy/h))
        
        return {
            'points': points,
            'point_count': len(points)
        }