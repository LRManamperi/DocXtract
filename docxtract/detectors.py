# ============================================================
# docxtract/detectors.py  (FULLY CORRECTED & COMPATIBLE)
# ------------------------------------------------------------
# ✔ Keeps backward compatibility (TextClusterTableDetector)
# ✔ Fixes chart misclassification (pie vs line/scatter)
# ✔ Supports horizontal & vertical charts
# ✔ Strong axis pairing + deduplication
# ✔ Safe defaults so imports NEVER fail
# ============================================================

from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np
import cv2

from .data_structures import BoundingBox, ElementType

# ============================================================
# BASE DETECTOR
# ============================================================
class BaseDetector(ABC):
    @abstractmethod
    def detect(self, page_image: np.ndarray, page_num: int):
        pass


# ============================================================
# TABLE DETECTORS
# ============================================================
class LineBasedTableDetector(BaseDetector):
    """Stable line-based table detector"""

    def __init__(self, min_area: int = 2000):
        self.min_area = min_area

    def detect(self, page_image: np.ndarray, page_num: int) -> List[BoundingBox]:
        gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
        th = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            15, 3
        )

        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        h = cv2.morphologyEx(th, cv2.MORPH_OPEN, hk)
        v = cv2.morphologyEx(th, cv2.MORPH_OPEN, vk)
        mask = cv2.add(h, v)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tables = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < self.min_area:
                continue
            ar = w / h if h else 0
            if 0.3 < ar < 5:
                bbox = BoundingBox(x, y, x + w, y + h)
                # Split this region if it has significant gaps
                split_boxes = self._split_by_gaps(page_image, bbox)
                tables.extend(split_boxes)
        return tables
    
    def _split_by_gaps(self, image: np.ndarray, bbox: BoundingBox) -> List[BoundingBox]:
        """Split a region into multiple boxes if there are significant vertical gaps"""
        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
        region = image[y1:y2, x1:x2]
        
        if region.size == 0:
            return [bbox]
        
        # Convert to grayscale and check for horizontal gaps
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        
        # Calculate row densities (how much content in each row)
        row_densities = []
        for i in range(gray.shape[0]):
            # Count non-white pixels in row
            row = gray[i, :]
            density = np.sum(row < 240) / gray.shape[1]  # Percentage of non-white pixels
            row_densities.append(density)
        
        # Find gaps (rows with very low density)
        gap_threshold = 0.05  # Less than 5% content
        min_gap_height = int(gray.shape[0] * 0.05)  # At least 5% of region height
        
        gaps = []
        gap_start = None
        for i, density in enumerate(row_densities):
            if density < gap_threshold:
                if gap_start is None:
                    gap_start = i
            else:
                if gap_start is not None:
                    gap_height = i - gap_start
                    if gap_height >= min_gap_height:
                        gaps.append((gap_start, i))
                    gap_start = None
        
        # If no significant gaps found, return original bbox
        if not gaps:
            return [bbox]
        
        # Split region by gaps
        split_boxes = []
        prev_end = 0
        
        for gap_start, gap_end in gaps:
            if gap_start - prev_end > min_gap_height:  # Ensure region is tall enough
                split_boxes.append(BoundingBox(
                    x1, y1 + prev_end,
                    x2, y1 + gap_start
                ))
            prev_end = gap_end
        
        # Add final region
        if gray.shape[0] - prev_end > min_gap_height:
            split_boxes.append(BoundingBox(
                x1, y1 + prev_end,
                x2, y2
            ))
        
        # If splitting resulted in nothing useful, return original
        if not split_boxes:
            return [bbox]
        
        return split_boxes


class TextClusterTableDetector(LineBasedTableDetector):
    """
    BACKWARD-COMPATIBILITY DETECTOR
    --------------------------------
    Alias for LineBasedTableDetector.
    Keeps old imports working safely.
    """
    pass


class MLTableDetector(BaseDetector):
    """
    ML-based table detector (placeholder implementation)
    Falls back to line-based detection if ML model is unavailable
    """
    def __init__(self, min_area: int = 2000):
        self.min_area = min_area
        self.fallback_detector = LineBasedTableDetector(min_area=min_area)
    
    def detect(self, page_image: np.ndarray, page_num: int) -> List[BoundingBox]:
        """
        Detect tables using ML model (currently falls back to line-based detection)
        """
        # TODO: Implement ML-based detection when model is available
        # For now, use line-based detection as fallback
        return self.fallback_detector.detect(page_image, page_num)

# ============================================================
# GRAPH / CHART DETECTOR
# ============================================================
class GraphDetector(BaseDetector):
    def __init__(self, min_area: int = 10000):
        self.min_area = min_area

    # --------------------------------------------------------
    # PUBLIC
    # --------------------------------------------------------
    def detect(self, page_image: np.ndarray, page_num: int):
        h, w = page_image.shape[:2]
        gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)

        # Method 1: Traditional axis-based detection
        axes = self._detect_axes(gray)
        axes = self._deduplicate_axes(axes)
        
        # Method 2: Direct bar chart detection (without relying on axes)
        bar_chart_regions = self._detect_bar_chart_regions(page_image, gray)
        
        # Merge axis-based detections with bar chart regions
        for bar_bbox in bar_chart_regions:
            # Check if this region overlaps with any existing axis detection
            is_duplicate = False
            for ax in axes:
                if bar_bbox.iou(ax['bbox']) > 0.3:
                    is_duplicate = True
                    break
            if not is_duplicate:
                # Add as synthetic axis detection
                axes.append({
                    'bbox': bar_bbox,
                    'h': (0, 0, 0, 0),  # Placeholder
                    'v': (0, 0, 0, 0),  # Placeholder
                    'is_bar_chart': True  # Flag for direct bar chart detection
                })

        results = []
        for ax in axes:
            bbox = ax['bbox']
            area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)
            if area < self.min_area:
                continue
            
            # Reject overly large regions (likely false positives)
            if area > (h * w * 0.8):  # More than 80% of page
                continue
            
            # Check aspect ratio - charts shouldn't be extremely wide or tall
            aspect_ratio = bbox.width / bbox.height if bbox.height > 0 else 0
            if aspect_ratio > 10 or aspect_ratio < 0.1:  # Too extreme
                continue

            # Split region by gaps to detect multiple charts
            split_boxes = self._split_chart_region(page_image, bbox)
            
            for split_bbox in split_boxes:
                region = page_image[int(split_bbox.y1):int(split_bbox.y2), 
                                  int(split_bbox.x1):int(split_bbox.x2)]
                if region.size == 0:
                    continue
                
                # Check if region is mostly uniform (likely background)
                if self._is_uniform_region(region):
                    continue

                # If detected via bar chart method, give it priority
                if ax.get('is_bar_chart', False):
                    chart_type = ElementType.BAR_CHART
                    conf = 0.80  # High confidence for direct bar detection
                else:
                    chart_type, conf = self._classify(region)
                    if conf < 0.5:
                        continue

                results.append((split_bbox, chart_type, conf, {
                    'orientation': self._orientation(ax)
                }))

        return self._final_nms(results)
    
    def _is_uniform_region(self, region: np.ndarray) -> bool:
        """Check if region is mostly uniform color (likely background, not a chart)"""
        if region.size == 0:
            return True
        
        # Convert to LAB color space for better uniformity detection
        if len(region.shape) == 3:
            # Check color variance
            std_dev = np.std(region, axis=(0, 1))
            # If all channels have low variance, it's uniform
            if np.all(std_dev < 15):  # Very low variance
                return True
            
            # Also check if it's mostly one color
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region
        
        # Check grayscale variance
        if np.std(gray) < 10:  # Very uniform
            return True
        
        # Check if histogram is concentrated (mostly one color)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # If one bin has more than 80% of pixels, it's uniform
        if np.max(hist) > 0.8:
            return True
        
        return False
    
    def _detect_bar_chart_regions(self, page_image: np.ndarray, gray: np.ndarray) -> List[BoundingBox]:
        """
        Detect bar chart regions directly by finding groups of aligned rectangular bars.
        This is a fallback when axis detection fails.
        """
        h, w = gray.shape
        detected_regions = []
        
        # Binary threshold to find dark elements
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for bar-like shapes (rectangles with extreme aspect ratios)
        bar_shapes = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            
            # Skip very small shapes
            if area < 500:
                continue
                
            # Skip shapes that are too large (likely page borders or backgrounds)
            if area > (h * w * 0.3):
                continue
            
            aspect_ratio = cw / ch if ch > 0 else 0
            
            # Bar-like: either tall/narrow (vertical bar) or wide/short (horizontal bar)
            is_vertical_bar = aspect_ratio < 0.5 and ch > 30  # Tall and narrow
            is_horizontal_bar = aspect_ratio > 2.0 and cw > 30  # Wide and short
            
            if is_vertical_bar or is_horizontal_bar:
                bar_shapes.append({
                    'x': x, 'y': y, 'w': cw, 'h': ch,
                    'aspect': aspect_ratio,
                    'is_vertical': is_vertical_bar,
                    'center_x': x + cw // 2,
                    'center_y': y + ch // 2,
                    'bottom': y + ch,
                    'top': y
                })
        
        # Need at least 3 bar-like shapes to consider it a bar chart
        if len(bar_shapes) < 3:
            return detected_regions
        
        # Group bars by alignment (bars in a chart should be aligned)
        vertical_bars = [b for b in bar_shapes if b['is_vertical']]
        horizontal_bars = [b for b in bar_shapes if not b['is_vertical']]
        
        # Check for vertical bar chart (bars with aligned bottoms)
        if len(vertical_bars) >= 3:
            # Group by similar bottom positions (aligned bases)
            vertical_bars = sorted(vertical_bars, key=lambda b: b['bottom'])
            
            groups = []
            current_group = [vertical_bars[0]]
            
            for bar in vertical_bars[1:]:
                # Bars in same chart should have similar bottom positions
                if abs(bar['bottom'] - current_group[-1]['bottom']) < h * 0.05:
                    current_group.append(bar)
                else:
                    if len(current_group) >= 3:
                        groups.append(current_group)
                    current_group = [bar]
            
            if len(current_group) >= 3:
                groups.append(current_group)
            
            # Create bounding boxes for each group
            for group in groups:
                min_x = min(b['x'] for b in group) - 20
                max_x = max(b['x'] + b['w'] for b in group) + 20
                min_y = min(b['y'] for b in group) - 30
                max_y = max(b['y'] + b['h'] for b in group) + 50  # More padding at bottom for labels
                
                # Clamp to image bounds
                min_x = max(0, min_x)
                min_y = max(0, min_y)
                max_x = min(w, max_x)
                max_y = min(h, max_y)
                
                # Check minimum size
                if (max_x - min_x) > 100 and (max_y - min_y) > 100:
                    detected_regions.append(BoundingBox(min_x, min_y, max_x, max_y))
        
        # Check for horizontal bar chart (bars with aligned left edges)
        if len(horizontal_bars) >= 3:
            # Group by similar left positions
            horizontal_bars = sorted(horizontal_bars, key=lambda b: b['x'])
            
            groups = []
            current_group = [horizontal_bars[0]]
            
            for bar in horizontal_bars[1:]:
                # Bars in same chart should have similar left positions
                if abs(bar['x'] - current_group[-1]['x']) < w * 0.05:
                    current_group.append(bar)
                else:
                    if len(current_group) >= 3:
                        groups.append(current_group)
                    current_group = [bar]
            
            if len(current_group) >= 3:
                groups.append(current_group)
            
            # Create bounding boxes for each group
            for group in groups:
                min_x = min(b['x'] for b in group) - 50  # More padding on left for labels
                max_x = max(b['x'] + b['w'] for b in group) + 20
                min_y = min(b['y'] for b in group) - 20
                max_y = max(b['y'] + b['h'] for b in group) + 20
                
                # Clamp to image bounds
                min_x = max(0, min_x)
                min_y = max(0, min_y)
                max_x = min(w, max_x)
                max_y = min(h, max_y)
                
                # Check minimum size
                if (max_x - min_x) > 100 and (max_y - min_y) > 100:
                    detected_regions.append(BoundingBox(min_x, min_y, max_x, max_y))
        
        return detected_regions
    
    def _split_chart_region(self, image: np.ndarray, bbox: BoundingBox) -> List[BoundingBox]:
        """Split chart region if it contains multiple charts with gaps"""
        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
        region = image[y1:y2, x1:x2]
        
        if region.size == 0:
            return [bbox]
        
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        
        # Calculate row densities
        row_densities = []
        for i in range(gray.shape[0]):
            row = gray[i, :]
            density = np.sum(row < 240) / gray.shape[1]
            row_densities.append(density)
        
        # Find significant gaps (larger threshold for charts)
        gap_threshold = 0.03  # Less than 3% content
        min_gap_height = int(gray.shape[0] * 0.08)  # At least 8% of region height
        
        gaps = []
        gap_start = None
        for i, density in enumerate(row_densities):
            if density < gap_threshold:
                if gap_start is None:
                    gap_start = i
            else:
                if gap_start is not None:
                    gap_height = i - gap_start
                    if gap_height >= min_gap_height:
                        gaps.append((gap_start, i))
                    gap_start = None
        
        # If no significant gaps, return original
        if not gaps:
            return [bbox]
        
        # Split by gaps
        split_boxes = []
        prev_end = 0
        min_region_height = int(gray.shape[0] * 0.1)  # At least 10% of height
        
        for gap_start, gap_end in gaps:
            if gap_start - prev_end > min_region_height:
                split_boxes.append(BoundingBox(
                    x1, y1 + prev_end,
                    x2, y1 + gap_start
                ))
            prev_end = gap_end
        
        # Add final region
        if gray.shape[0] - prev_end > min_region_height:
            split_boxes.append(BoundingBox(
                x1, y1 + prev_end,
                x2, y2
            ))
        
        if not split_boxes:
            return [bbox]
        
        return split_boxes

    # --------------------------------------------------------
    # AXIS DETECTION
    # --------------------------------------------------------
    def _detect_axes(self, gray):
        h, w = gray.shape
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80,
                                minLineLength=min(h, w) // 8,
                                maxLineGap=15)
        if lines is None:
            return []

        H, V = [], []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            length = np.hypot(x2 - x1, y2 - y1)
            if length < min(h, w) // 10:
                continue
            if ang < 10 or ang > 170:
                H.append((x1, y1, x2, y2, length))
            elif 80 < ang < 100:
                V.append((x1, y1, x2, y2, length))

        # Only keep the longest lines (main axes, not grid lines)
        H = sorted(H, key=lambda x: x[4], reverse=True)[:3]  # Max 3 horizontal
        V = sorted(V, key=lambda x: x[4], reverse=True)[:3]  # Max 3 vertical

        axes = []
        used_h = set()
        used_v = set()
        
        for i, (hx1, hy1, hx2, hy2, _) in enumerate(H):
            if i in used_h:
                continue
                
            best_v = None
            best_v_idx = None
            best_d = w * 0.15  # Max 15% of width for pairing distance
            
            for j, (vx1, vy1, vx2, vy2, _) in enumerate(V):
                if j in used_v:
                    continue
                    
                # Check if lines are close to each other (potential axis pair)
                d = abs(vx1 - min(hx1, hx2))
                if d < best_d:
                    best_d = d
                    best_v = (vx1, vy1, vx2, vy2)
                    best_v_idx = j
            
            if best_v is None:
                continue

            # Mark as used
            used_h.add(i)
            used_v.add(best_v_idx)
            
            vx1, vy1, vx2, vy2 = best_v
            
            # Create bounding box with reasonable padding
            px, py = int(0.10 * w), int(0.10 * h)  # Reduced padding
            x1 = max(0, min(hx1, hx2, vx1, vx2) - px)
            y1 = max(0, min(hy1, hy2, vy1, vy2) - py)
            x2 = min(w, max(hx1, hx2, vx1, vx2) + px)
            y2 = min(h, max(hy1, hy2, vy1, vy2) + py)

            axes.append({
                'bbox': BoundingBox(x1, y1, x2, y2),
                'h': (hx1, hy1, hx2, hy2),
                'v': (vx1, vy1, vx2, vy2)
            })
        
        return axes

    def _orientation(self, ax):
        hx1, hy1, hx2, hy2 = ax['h']
        vx1, vy1, vx2, vy2 = ax['v']
        return 'vertical' if abs(vy2 - vy1) > abs(hx2 - hx1) else 'horizontal'

    # --------------------------------------------------------
    # DEDUPLICATION
    # --------------------------------------------------------
    def _deduplicate_axes(self, axes):
        """Deduplicate overlapping axis detections with improved merging"""
        if not axes:
            return []
        
        # Sort by area (largest first) to prioritize bigger detections
        axes = sorted(axes, key=lambda a: a['bbox'].area, reverse=True)
        
        out = []
        for a in axes:
            keep = True
            for b in out:
                iou = a['bbox'].iou(b['bbox'])
                # Higher threshold to avoid splitting single charts
                if iou > 0.3:  # Lower back down but still reasonable
                    keep = False
                    break
                # Also check if one bbox contains most of the other
                a_in_b = self._overlap_ratio(a['bbox'], b['bbox'])
                b_in_a = self._overlap_ratio(b['bbox'], a['bbox'])
                if a_in_b > 0.6 or b_in_a > 0.6:
                    keep = False
                    break
                # Check if bboxes are close spatially (likely same chart)
                if self._are_boxes_close(a['bbox'], b['bbox']):
                    keep = False
                    break
            if keep:
                out.append(a)
        return out
    
    def _are_boxes_close(self, box1: BoundingBox, box2: BoundingBox) -> bool:
        """Check if two boxes are spatially close (likely parts of same chart)"""
        # Get centers
        c1x, c1y = box1.center
        c2x, c2y = box2.center
        
        # Get average dimensions
        avg_w = (box1.width + box2.width) / 2
        avg_h = (box1.height + box2.height) / 2
        
        # If centers are within 50% of average dimension, consider them close
        dx = abs(c1x - c2x)
        dy = abs(c1y - c2y)
        
        return dx < avg_w * 0.5 and dy < avg_h * 0.5
    
    def _overlap_ratio(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate how much box1 overlaps with box2"""
        ix1 = max(box1.x1, box2.x1)
        iy1 = max(box1.y1, box2.y1)
        ix2 = min(box1.x2, box2.x2)
        iy2 = min(box1.y2, box2.y2)
        
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        
        intersection = (ix2 - ix1) * (iy2 - iy1)
        box1_area = box1.area
        
        return intersection / box1_area if box1_area > 0 else 0.0

    # --------------------------------------------------------
    # CLASSIFICATION (WITH BETTER ORDERING)
    # --------------------------------------------------------
    def _classify(self, img):
        """Classify chart type with improved accuracy"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Check if region has enough content
        non_white_pixels = np.sum(gray < 240)
        total_pixels = gray.shape[0] * gray.shape[1]
        content_ratio = non_white_pixels / total_pixels
        
        # Reject mostly blank regions
        if content_ratio < 0.05:
            return ElementType.UNKNOWN, 0.0

        # Try bar charts first (most common and distinctive)
        bars = self._bars(gray)
        if bars >= 3:  # Need at least 3 bar-like shapes
            return ElementType.BAR_CHART, 0.75

        # Check for line charts (angled lines)
        edges = cv2.Canny(gray, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50,
                                minLineLength=min(img.shape[:2]) // 6,
                                maxLineGap=10)
        angled = 0
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if 15 < ang < 75 or 105 < ang < 165:
                    angled += 1
        if angled >= 4:  # Need more angled lines for confidence
            return ElementType.LINE_CHART, 0.75

        # Check for pie charts (circles) - after other types to reduce false positives
        if self._is_pie(gray):
            return ElementType.PIE_CHART, 0.85

        # Check for scatter plots
        scatter = self._scatter(gray)
        if scatter > 0.5:  # Higher threshold
            return ElementType.SCATTER_PLOT, scatter

        return ElementType.UNKNOWN, 0.0

    def _is_pie(self, gray):
        """Detect pie charts with better validation to reduce false positives"""
        h, w = gray.shape
        
        # Must be roughly square
        if abs(h - w) > 0.3 * max(h, w):
            return False
        
        # Check if region has enough content (not mostly blank)
        non_white_pixels = np.sum(gray < 240)
        total_pixels = h * w
        content_ratio = non_white_pixels / total_pixels
        
        # Pie charts should have reasonable content (10-60%)
        if content_ratio < 0.1 or content_ratio > 0.6:
            return False
        
        # Detect circles
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1.2,
            min(h, w) // 3,
            param1=50, param2=30,  # More strict param2
            minRadius=min(h, w) // 5,  # Smaller min radius
            maxRadius=min(h, w) // 2
        )
        
        if circles is None:
            return False
        
        # Verify circle quality - should have at least one good circle
        circles = np.round(circles[0, :]).astype("int")
        
        # Check if the main circle is reasonably centered
        for (x, y, r) in circles:
            # Circle center should be roughly in middle of region
            center_offset_x = abs(x - w/2) / w
            center_offset_y = abs(y - h/2) / h
            
            # Allow some offset but not too much
            if center_offset_x < 0.3 and center_offset_y < 0.3:
                # Radius should be reasonable relative to region size
                if 0.2 < r/min(h,w) < 0.6:
                    return True
        
        return False

    def _bars(self, gray):
        """Detect bar charts with better validation"""
        # Check if region is mostly uniform (reject backgrounds)
        if np.std(gray) < 10:
            return 0
        
        # Check if region has enough content
        non_white_pixels = np.sum(gray < 240)
        total_pixels = gray.shape[0] * gray.shape[1]
        content_ratio = non_white_pixels / total_pixels
        
        # Bar charts should have some content but not too much
        if content_ratio < 0.05 or content_ratio > 0.7:
            return 0
        
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        count = 0
        bar_like = []
        
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            
            # Must have minimum area
            if area < 300:
                continue
            
            ar = w / h if h else 0
            
            # Vertical bars (tall and narrow) or horizontal bars (wide and short)
            if ar > 2.5 or ar < 0.4:
                bar_like.append((x, y, w, h, ar))
                count += 1
        
        # Need at least 2 bar-like shapes
        if count < 2:
            return 0
        
        # Check if bars are aligned (characteristic of bar charts)
        if len(bar_like) >= 2:
            # Check vertical alignment for vertical bars
            y_positions = [y for x, y, w, h, ar in bar_like if ar < 1]
            if len(y_positions) >= 2:
                y_var = np.var(y_positions) if len(y_positions) > 1 else 0
                # If bases are aligned, more likely a bar chart
                if y_var < gray.shape[0] * 0.01:  # Very aligned
                    count += 2  # Bonus for alignment
        
        return count

    def _scatter(self, gray):
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pts = [c for c in cnts if 20 < cv2.contourArea(c) < 250]
        if len(pts) >= 8:
            return min(0.8, 0.3 + len(pts) * 0.03)
        return 0.0

    # --------------------------------------------------------
    # FINAL NMS
    # --------------------------------------------------------
    def _final_nms(self, results):
        """Final Non-Maximum Suppression with improved merging"""
        if not results:
            return []
        
        # Sort by confidence (highest first)
        results = sorted(results, key=lambda x: x[2], reverse=True)
        
        out = []
        for r in results:
            keep = True
            for o in out:
                iou = r[0].iou(o[0])
                # Higher threshold to avoid splitting charts
                if iou > 0.4:  # Lower but still reasonable
                    keep = False
                    break
                # Check if boxes are very similar (likely same chart)
                overlap1 = self._overlap_ratio(r[0], o[0])
                overlap2 = self._overlap_ratio(o[0], r[0])
                if overlap1 > 0.7 or overlap2 > 0.7:
                    keep = False
                    break
                # Check spatial proximity
                if self._are_boxes_close(r[0], o[0]):
                    keep = False
                    break
            if keep:
                out.append(r)
        return out
