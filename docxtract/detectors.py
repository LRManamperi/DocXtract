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
        
        # Step 2: Deduplicate overlapping axis pairs (same chart detected multiple times)
        axes = self._deduplicate_axes(axes)
        
        # Step 3: For each axis pair, detect the chart and extract data
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
            
            # Calculate angle in degrees (0-180)
            dx = x2 - x1
            dy = y2 - y1
            angle = abs(np.arctan2(abs(dy), abs(dx)) * 180 / np.pi)
            
            # Horizontal line (close to 0°)
            if angle < 15 or angle > 165:
                horizontal_lines.append((x1, y1, x2, y2, length))
            # Vertical line (close to 90°)
            elif 75 < angle < 105:
                vertical_lines.append((x1, y1, x2, y2, length))
        
        # Merge collinear lines to form complete axes
        horizontal_lines = self._merge_collinear_lines(horizontal_lines, 'horizontal', tolerance=5)
        vertical_lines = self._merge_collinear_lines(vertical_lines, 'vertical', tolerance=5)
        
        # Find axis pairs (one horizontal + one vertical = one chart)
        # Group similar lines first to avoid duplicate detections
        axes = []
        
        # Group horizontal lines by approximate y-coordinate (within 30 pixels)
        h_groups = []
        for i, (hx1, hy1, hx2, hy2, h_len) in enumerate(horizontal_lines):
            avg_y = (hy1 + hy2) / 2
            found_group = False
            for group in h_groups:
                # Check if this line belongs to an existing group
                group_avg_y = group['avg_y']
                if abs(avg_y - group_avg_y) < 30:  # Within 30 pixels
                    # Use the longest line in the group
                    if h_len > group['max_len']:
                        group['line'] = (hx1, hy1, hx2, hy2, h_len)
                        group['max_len'] = h_len
                        group['avg_y'] = avg_y
                    found_group = True
                    break
            if not found_group:
                h_groups.append({'line': (hx1, hy1, hx2, hy2, h_len), 'avg_y': avg_y, 'max_len': h_len})
        
        # Group vertical lines by approximate x-coordinate (within 30 pixels)
        v_groups = []
        for j, (vx1, vy1, vx2, vy2, v_len) in enumerate(vertical_lines):
            avg_x = (vx1 + vx2) / 2
            found_group = False
            for group in v_groups:
                group_avg_x = group['avg_x']
                if abs(avg_x - group_avg_x) < 30:  # Within 30 pixels
                    if v_len > group['max_len']:
                        group['line'] = (vx1, vy1, vx2, vy2, v_len)
                        group['max_len'] = v_len
                        group['avg_x'] = avg_x
                    found_group = True
                    break
            if not found_group:
                v_groups.append({'line': (vx1, vy1, vx2, vy2, v_len), 'avg_x': avg_x, 'max_len': v_len})
        
        # Now find axis pairs using the grouped lines
        for h_group in h_groups:
            hx1, hy1, hx2, hy2, h_len = h_group['line']
            
            for v_group in v_groups:
                vx1, vy1, vx2, vy2, v_len = v_group['line']
                
                # Check if they form an L-shape (axis) - check intersection
                intersection_point = self._line_intersection(
                    hx1, hy1, hx2, hy2, vx1, vy1, vx2, vy2
                )
                
                if intersection_point is not None:
                    # Found an axis pair - define chart bounding box
                    ix, iy = intersection_point
                    
                    # Define bounding box: from intersection to the extent of the chart area
                    x_min = min(min(hx1, hx2), min(vx1, vx2)) - 10
                    y_min = min(min(vy1, vy2), min(hy1, hy2)) - 10
                    x_max = max(max(hx1, hx2), max(vx1, vx2)) + 10
                    y_max = max(max(vy1, vy2), max(hy1, hy2)) + 10
                    
                    # Clamp to image bounds
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(w, x_max)
                    y_max = min(h, y_max)
                    
                    # Ensure minimum size
                    if (x_max - x_min) < 50 or (y_max - y_min) < 50:
                        continue
                    
                    bbox = BoundingBox(x_min, y_min, x_max, y_max)
                    
                    axes.append({
                        'bbox': bbox,
                        'h_axis': (hx1, hy1, hx2, hy2),
                        'v_axis': (vx1, vy1, vx2, vy2),
                        'intersection': (int(ix), int(iy))
                    })
                    break  # Only one chart per horizontal line group
        
        return axes

    def _deduplicate_axes(self, axes: List[Dict], overlap_threshold: float = 0.3) -> List[Dict]:
        """
        Remove duplicate/overlapping axis pairs that represent the same chart
        
        Args:
            axes: List of axis info dictionaries
            overlap_threshold: Minimum IoU (Intersection over Union) to consider as duplicate (lower = more aggressive)
        
        Returns:
            Deduplicated list of axis pairs
        """
        if len(axes) <= 1:
            return axes
        
        # Sort by area (largest first) to keep the most complete detection
        axes_sorted = sorted(axes, key=lambda a: (a['bbox'].x2 - a['bbox'].x1) * (a['bbox'].y2 - a['bbox'].y1), reverse=True)
        
        deduplicated = []
        used_indices = set()
        
        for i, axis_info in enumerate(axes_sorted):
            if i in used_indices:
                continue
            
            bbox1 = axis_info['bbox']
            deduplicated.append(axis_info)
            
            # Check for overlaps with remaining axes
            for j in range(i + 1, len(axes_sorted)):
                if j in used_indices:
                    continue
                
                bbox2 = axes_sorted[j]['bbox']
                
                # Calculate intersection area
                x_overlap = max(0, min(bbox1.x2, bbox2.x2) - max(bbox1.x1, bbox2.x1))
                y_overlap = max(0, min(bbox1.y2, bbox2.y2) - max(bbox1.y1, bbox2.y1))
                intersection = x_overlap * y_overlap
                
                # Calculate areas
                area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1)
                area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1)
                
                if area1 == 0 or area2 == 0:
                    continue
                
                # Check for containment (one bbox mostly inside another)
                containment_ratio1 = intersection / area2  # How much of bbox2 is inside bbox1
                containment_ratio2 = intersection / area1  # How much of bbox1 is inside bbox2
                
                # If one bbox is mostly contained in another, it's a duplicate
                if containment_ratio1 > 0.7 or containment_ratio2 > 0.7:
                    used_indices.add(j)
                    continue
                
                # Calculate union area for IoU
                union = area1 + area2 - intersection
                
                if union > 0:
                    iou = intersection / union
                    
                    # If IoU exceeds threshold OR significant overlap, mark as duplicate
                    # Also check if centers are very close (within 20% of average dimension)
                    center1_x = (bbox1.x1 + bbox1.x2) / 2
                    center1_y = (bbox1.y1 + bbox1.y2) / 2
                    center2_x = (bbox2.x1 + bbox2.x2) / 2
                    center2_y = (bbox2.y1 + bbox2.y2) / 2
                    
                    center_distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
                    avg_dimension = (np.sqrt(area1) + np.sqrt(area2)) / 2
                    
                    # If IoU > threshold OR centers are very close (same chart, different bounding), remove duplicate
                    if iou > overlap_threshold or (center_distance < avg_dimension * 0.3 and iou > 0.1):
                        used_indices.add(j)
        
        return deduplicated

    def _merge_collinear_lines(self, lines: List[Tuple], orientation: str, tolerance: int = 5) -> List[Tuple]:
        """Merge collinear lines that are close together"""
        if not lines:
            return []
        
        merged = []
        used = set()
        
        for i, (x1, y1, x2, y2, len1) in enumerate(lines):
            if i in used:
                continue
            
            # Start with current line
            merged_x1, merged_y1 = x1, y1
            merged_x2, merged_y2 = x2, y2
            merged_length = len1
            used.add(i)
            
            # Try to merge with other lines
            for j, (ox1, oy1, ox2, oy2, len2) in enumerate(lines[i+1:], start=i+1):
                if j in used:
                    continue
                
                if orientation == 'horizontal':
                    # Check if y-coordinates are similar (within tolerance)
                    # For horizontal lines, y-coordinates should be nearly the same
                    avg_y1 = (y1 + y2) / 2
                    avg_oy1 = (oy1 + oy2) / 2
                    if abs(avg_y1 - avg_oy1) <= tolerance:
                        # Check if they overlap or are close in x-direction
                        h_min_x = min(x1, x2)
                        h_max_x = max(x1, x2)
                        o_min_x = min(ox1, ox2)
                        o_max_x = max(ox1, ox2)
                        
                        # If they overlap or are close, merge them
                        if (h_min_x <= o_max_x + tolerance and o_min_x <= h_max_x + tolerance):
                            merged_x1 = min(merged_x1, ox1, ox2)
                            merged_y1 = int((merged_y1 + oy1) / 2)  # Average y
                            merged_x2 = max(merged_x2, ox1, ox2)
                            merged_y2 = int((merged_y2 + oy2) / 2)
                            merged_length = np.sqrt((merged_x2-merged_x1)**2 + (merged_y2-merged_y1)**2)
                            used.add(j)
                
                elif orientation == 'vertical':
                    # Check if x-coordinates are similar (within tolerance)
                    # For vertical lines, x-coordinates should be nearly the same
                    avg_x1 = (x1 + x2) / 2
                    avg_ox1 = (ox1 + ox2) / 2
                    if abs(avg_x1 - avg_ox1) <= tolerance:
                        # Check if they overlap or are close in y-direction
                        v_min_y = min(y1, y2)
                        v_max_y = max(y1, y2)
                        o_min_y = min(oy1, oy2)
                        o_max_y = max(oy1, oy2)
                        
                        # If they overlap or are close, merge them
                        if (v_min_y <= o_max_y + tolerance and o_min_y <= v_max_y + tolerance):
                            merged_x1 = int((merged_x1 + ox1) / 2)  # Average x
                            merged_y1 = min(merged_y1, oy1, oy2)
                            merged_x2 = int((merged_x2 + ox2) / 2)
                            merged_y2 = max(merged_y2, oy1, oy2)
                            merged_length = np.sqrt((merged_x2-merged_x1)**2 + (merged_y2-merged_y1)**2)
                            used.add(j)
            
            merged.append((merged_x1, merged_y1, merged_x2, merged_y2, merged_length))
        
        return merged

    def _line_intersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """Calculate intersection point of two line segments, or check if they're close enough"""
        # Calculate intersection of infinite lines
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:  # Lines are parallel
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        # Check if intersection is within both line segments (or very close)
        if 0 <= t <= 1 and 0 <= u <= 1:
            # Intersection is within both segments
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            return (ix, iy)
        
        # Check if lines are close enough to form an axis (even if they don't exactly intersect)
        # For horizontal line (x1,y1)-(x2,y2) and vertical line (x3,y3)-(x4,y4)
        tolerance = 20  # pixels
        
        h_min_x = min(x1, x2)
        h_max_x = max(x1, x2)
        v_min_x = min(x3, x4)
        v_max_x = max(x3, x4)
        v_x = (x3 + x4) / 2  # Average x for vertical line
        
        v_min_y = min(y3, y4)
        v_max_y = max(y3, y4)
        h_min_y = min(y1, y2)
        h_max_y = max(y1, y2)
        h_y = (y1 + y2) / 2  # Average y for horizontal line
        
        # Check if vertical line's x is within or close to horizontal line's x range
        x_overlap = not (v_max_x < h_min_x - tolerance or v_min_x > h_max_x + tolerance)
        
        # Check if horizontal line's y is within or close to vertical line's y range
        y_overlap = not (h_max_y < v_min_y - tolerance or h_min_y > v_max_y + tolerance)
        
        # Also check if endpoints are very close (L-shape formation)
        endpoint_close = False
        # Check if vertical line endpoint is near horizontal line
        for vx, vy in [(x3, y3), (x4, y4)]:
            if h_min_x - tolerance <= vx <= h_max_x + tolerance:
                if abs(vy - h_y) <= tolerance:
                    endpoint_close = True
                    break
        # Check if horizontal line endpoint is near vertical line
        if not endpoint_close:
            for hx, hy in [(x1, y1), (x2, y2)]:
                if v_min_y - tolerance <= hy <= v_max_y + tolerance:
                    if abs(hx - v_x) <= tolerance:
                        endpoint_close = True
                        break
        
        if (x_overlap and y_overlap) or endpoint_close:
            # Return approximate intersection point
            # Prefer actual intersection if lines overlap, otherwise use midpoint
            if x_overlap and y_overlap:
                ix = v_x
                iy = h_y
            else:
                # Use the closer endpoint or midpoint
                ix = v_x
                iy = h_y
            return (ix, iy)
        
        return None

    def _forms_axis(self, hx1, hy1, hx2, hy2, vx1, vy1, vx2, vy2) -> bool:
        """Check if horizontal and vertical lines form an axis (L-shape)"""
        # Use the improved line intersection method
        return self._line_intersection(hx1, hy1, hx2, hy2, vx1, vy1, vx2, vy2) is not None

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
        
        # Line chart detection FIRST (before bar chart, as line charts may have markers that look like bars)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                                minLineLength=max(20, min(h, w)//8), maxLineGap=15)
        
        angled_lines_count = 0
        connected_line_segments = 0
        if lines is not None:
            # Count angled lines (not horizontal/vertical axis lines)
            angled_lines_count = sum(1 for line in lines if self._is_angled_line(line[0]))
            
            # Check for connected line segments (indicating a line chart trend)
            if len(lines) > 2:
                # Count how many lines are approximately connected
                line_points = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if self._is_angled_line(line[0]):
                        line_points.append((x1, y1))
                        line_points.append((x2, y2))
                
                # Check if points form a path (line chart characteristic)
                if len(line_points) >= 4:
                    # Sort by x-coordinate to check for trend
                    line_points.sort(key=lambda p: p[0])
                    # Count connected segments (points close to each other)
                    for i in range(len(line_points) - 1):
                        dist = np.sqrt((line_points[i+1][0] - line_points[i][0])**2 + 
                                      (line_points[i+1][1] - line_points[i][1])**2)
                        if dist < max(30, min(h, w)//5):  # Connected within reasonable distance
                            connected_line_segments += 1
                
            # Line charts should have multiple connected angled lines
            if angled_lines_count >= 3 or connected_line_segments >= 2:
                type_scores[ElementType.LINE_CHART] = min(0.92, 0.5 + (angled_lines_count + connected_line_segments) * 0.06)
        
        # Bar chart detection (check for distinct rectangular bars)
        bars = self._detect_bars(contours, h, w)
        if bars['count'] >= 2:
            # Bar charts should have distinct, separate bars
            # Check if bars are well-separated (not connected like line chart markers)
            bar_score = bars['count'] * 0.08
            
            # Penalize if we also detected many angled lines (more likely a line chart)
            if angled_lines_count > bars['count'] * 2:
                bar_score *= 0.5  # Reduce bar chart score if many angled lines present
            
            type_scores[ElementType.BAR_CHART] = min(0.90, 0.4 + bar_score)
        
        # Scatter plot detection
        scatter_score = self._detect_scatter_points(chart_region)
        type_scores[ElementType.SCATTER_PLOT] = scatter_score
        
        # Return best match
        best_type = max(type_scores.items(), key=lambda x: x[1])
        return best_type[0], best_type[1]

    def _detect_bars(self, contours, h, w) -> Dict:
        """Detect rectangular bars in chart - exclude small markers that might be line chart points"""
        vertical_bars = []
        horizontal_bars = []
        
        # Minimum bar size - line chart markers are typically smaller
        min_bar_area = max(200, (h * w) * 0.001)  # At least 0.1% of chart area
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_bar_area:  # Filter out small markers
                continue
            
            x, y, bw, bh = cv2.boundingRect(cnt)
            
            if bw > 5 and bh > 5:
                aspect_ratio = bw / bh if bh > 0 else 0
                
                # Vertical bars - should be distinct rectangles, not small squares/circles
                if 0.1 < aspect_ratio < 0.5 and bh > bw * 2 and bh > h * 0.15:
                    # Check if it's a solid rectangle (bar) vs a small marker
                    # Bars should have high fill ratio (area close to bounding box area)
                    fill_ratio = area / (bw * bh) if (bw * bh) > 0 else 0
                    if fill_ratio > 0.6:  # Bar should be mostly filled
                        vertical_bars.append({'x': x, 'y': y, 'w': bw, 'h': bh})
                
                # Horizontal bars
                elif aspect_ratio > 2.0 and bw > bh * 2 and bw > w * 0.15:
                    fill_ratio = area / (bw * bh) if (bw * bh) > 0 else 0
                    if fill_ratio > 0.6:
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
        """Extract bar values from bar chart with improved detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Use adaptive thresholding to find bars
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bars = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:  # Increased minimum area
                continue
            
            x, y, bw, bh = cv2.boundingRect(cnt)
            
            # Skip if too small
            if bw < 5 or bh < 5:
                continue
            
            aspect_ratio = bw / bh if bh > 0 else 0
            
            # Detect vertical bars (common in bar charts)
            if 0.1 < aspect_ratio < 0.6 and bh > bw * 1.5 and bh > h * 0.1:
                # Calculate value based on bar height relative to chart height
                bar_top = y
                bar_bottom = y + bh
                bar_height_pixels = bh
                
                # Normalize: taller bars have higher values
                # Assume bars start from bottom (y increases downward in images)
                value = bar_height_pixels / h
                
                bars.append({
                    'x': x + bw//2,  # Center x
                    'y': y,  # Top of bar
                    'width': bw,
                    'height': bh,
                    'base_y': bar_bottom,  # Bottom of bar
                    'value': value,  # Normalized height
                    'orientation': 'vertical'
                })
            
            # Detect horizontal bars
            elif aspect_ratio > 1.5 and bw > bh * 1.5 and bw > w * 0.1:
                bar_left = x
                bar_right = x + bw
                bar_width_pixels = bw
                
                # Normalize width
                value = bar_width_pixels / w
                
                bars.append({
                    'x': x,  # Left of bar
                    'y': y + bh//2,  # Center y
                    'width': bw,
                    'height': bh,
                    'base_x': bar_left,
                    'value': value,  # Normalized width
                    'orientation': 'horizontal'
                })
        
        # Sort bars by position
        if bars:
            if bars[0].get('orientation') == 'vertical':
                bars.sort(key=lambda b: b['x'])
            else:
                bars.sort(key=lambda b: b['y'])
        
        # Extract just the values
        values = [b['value'] for b in bars]
        
        return {
            'values': values,
            'bar_count': len(bars),
            'bars': bars,
            'orientation': bars[0].get('orientation') if bars else 'unknown'
        }

    def _extract_line_data(self, image: np.ndarray) -> Dict:
        """Extract line data points from line chart with improved detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Method 1: Detect data point markers (circles, squares)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        marker_points = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 20 < area < 300:  # Data point marker size
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Check if it's roughly circular (data point)
                    x, y, mw, mh = cv2.boundingRect(cnt)
                    aspect_ratio = mw / mh if mh > 0 else 0
                    if 0.6 < aspect_ratio < 1.4:  # Nearly square/circular
                        marker_points.append((cx, cy))
        
        # Method 2: Sample points along detected lines
        edges = cv2.Canny(gray, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25,
                                minLineLength=20, maxLineGap=20)
        
        line_points = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Only consider non-axis lines (angled lines)
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if 15 < angle < 75 or 105 < angle < 165:
                    # Sample points along the line
                    num_samples = max(2, int(np.sqrt((x2-x1)**2 + (y2-y1)**2) / 20))
                    for i in range(num_samples + 1):
                        t = i / num_samples
                        px = int(x1 + t * (x2 - x1))
                        py = int(y1 + t * (y2 - y1))
                        line_points.append((px, py))
        
        # Combine both methods, prefer marker points
        all_points = marker_points if marker_points else line_points
        
        # Remove duplicates (points close to each other)
        if all_points:
            filtered_points = []
            all_points_sorted = sorted(all_points, key=lambda p: p[0])  # Sort by x
            
            for point in all_points_sorted:
                # Check if this point is too close to any existing point
                is_duplicate = False
                for existing in filtered_points:
                    dist = np.sqrt((point[0] - existing[0])**2 + (point[1] - existing[1])**2)
                    if dist < 15:  # Within 15 pixels
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    filtered_points.append(point)
            
            all_points = filtered_points
        
        # Normalize to 0-1 range
        normalized_points = [(x/w, 1 - y/h) for x, y in all_points]
        
        return {
            'points': normalized_points,
            'point_count': len(all_points),
            'raw_points': all_points,
            'values': [1 - y/h for x, y in all_points]  # Y values for easy access
        }

    def _extract_pie_data(self, image: np.ndarray) -> Dict:
        """Extract pie slice information with improved detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find the center of the pie chart
        center = (w//2, h//2)
        
        # Find radial lines (slice boundaries)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25,
                                minLineLength=min(h, w)//10, maxLineGap=15)
        
        slice_angles = []
        radial_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Check if line passes near center
                dist1 = np.sqrt((x1 - center[0])**2 + (y1 - center[1])**2)
                dist2 = np.sqrt((x2 - center[0])**2 + (y2 - center[1])**2)
                
                min_dist = min(dist1, dist2)
                radius_threshold = min(h, w) // 3
                
                if min_dist < radius_threshold:
                    # Calculate angle of this radial line
                    # Use the point farther from center
                    if dist1 > dist2:
                        dx = x1 - center[0]
                        dy = y1 - center[1]
                    else:
                        dx = x2 - center[0]
                        dy = y2 - center[1]
                    
                    angle = np.arctan2(dy, dx) * 180 / np.pi
                    # Normalize to 0-360
                    if angle < 0:
                        angle += 360
                    
                    slice_angles.append(angle)
                    radial_lines.append((x1, y1, x2, y2))
        
        # Group similar angles (within 15 degrees)
        if slice_angles:
            slice_angles.sort()
            unique_angles = [slice_angles[0]]
            
            for angle in slice_angles[1:]:
                if angle - unique_angles[-1] > 15:  # More than 15 degrees apart
                    unique_angles.append(angle)
            
            slice_count = len(unique_angles)
        else:
            # Fall back to color-based detection
            slice_count = self._count_slices_by_color(image)
        
        # Calculate slice percentages if we have the angles
        slice_percentages = []
        if len(unique_angles) >= 2:
            # Calculate angular spans between consecutive angles
            for i in range(len(unique_angles)):
                next_i = (i + 1) % len(unique_angles)
                angle_diff = unique_angles[next_i] - unique_angles[i]
                if angle_diff < 0:
                    angle_diff += 360
                percentage = angle_diff / 360.0
                slice_percentages.append(percentage)
        
        return {
            'slice_count': max(slice_count, 2),
            'slices': unique_angles if slice_angles else [],
            'slice_angles': unique_angles if slice_angles else [],
            'slice_percentages': slice_percentages,
            'radial_lines': radial_lines,
            'values': slice_percentages if slice_percentages else [1.0/max(slice_count, 2)] * max(slice_count, 2)
        }
    
    def _count_slices_by_color(self, image: np.ndarray) -> int:
        """Count pie slices by detecting distinct colored regions"""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        
        # Sample colors in a circle around center
        center_x, center_y = w // 2, h // 2
        radius = min(w, h) // 3
        
        num_samples = 36  # Sample every 10 degrees
        sampled_colors = []
        
        for i in range(num_samples):
            angle = (i * 360 / num_samples) * np.pi / 180
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            
            # Ensure within bounds
            x = max(0, min(w-1, x))
            y = max(0, min(h-1, y))
            
            # Get hue value at this point
            hue = hsv[y, x, 0]
            sampled_colors.append(hue)
        
        # Count color transitions (changes in hue)
        transitions = 0
        threshold = 15  # Hue difference threshold
        
        for i in range(len(sampled_colors)):
            next_i = (i + 1) % len(sampled_colors)
            hue_diff = abs(int(sampled_colors[next_i]) - int(sampled_colors[i]))
            
            # Handle hue wraparound (0-180 in OpenCV)
            if hue_diff > 90:
                hue_diff = 180 - hue_diff
            
            if hue_diff > threshold:
                transitions += 1
        
        # Number of slices is approximately half the transitions
        # (each slice has 2 boundaries)
        slice_count = max(2, transitions // 2)
        
        return min(slice_count, 12)  # Cap at 12 slices

    def _extract_scatter_data(self, image: np.ndarray) -> Dict:
        """Extract scatter plot points with improved detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Use multiple thresholding methods to catch different point types
        _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Also try adaptive threshold for varying backgrounds
        thresh2 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Combine both thresholds
        combined_thresh = cv2.bitwise_or(thresh1, thresh2)
        
        contours, _ = cv2.findContours(combined_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        points = []
        point_sizes = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Scatter points are typically small, isolated shapes
            if 25 < area < 400:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Verify it's roughly circular/square (not a line segment)
                    x, y, pw, ph = cv2.boundingRect(cnt)
                    aspect_ratio = pw / ph if ph > 0 else 0
                    
                    if 0.5 < aspect_ratio < 2.0:  # Not too elongated
                        points.append((cx, cy))
                        point_sizes.append(area)
        
        # Normalize to 0-1 range
        normalized_points = [(x/w, 1 - y/h) for x, y in points]
        
        # Calculate dispersion statistics
        dispersion_x = 0
        dispersion_y = 0
        if points:
            coords = np.array(points)
            dispersion_x = np.std(coords[:, 0]) / w
            dispersion_y = np.std(coords[:, 1]) / h
        
        # Extract Y values for consistency with other chart types
        y_values = [point[1] for point in normalized_points] if normalized_points else []
        
        return {
            'points': normalized_points,
            'point_count': len(points),
            'raw_points': points,
            'dispersion_x': float(dispersion_x),
            'dispersion_y': float(dispersion_y),
            'avg_point_size': float(np.mean(point_sizes)) if point_sizes else 0,
            'values': y_values  # Y values as floats for consistency
        }