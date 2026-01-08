"""
Advanced chart data extraction module for DocXtract
Extracts structured data from bar charts, line charts, pie charts, and scatter plots
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from dataclasses import dataclass
import pytesseract
from PIL import Image
import os

from .data_structures import ElementType, BoundingBox

# Configure Tesseract path if available
if os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


@dataclass
class ChartData:
    """Structured data extracted from charts"""
    chart_type: ElementType
    values: List[float]
    labels: List[str]
    series: List[Dict]  # For multi-series charts
    axes: Dict[str, any]  # Axis information
    legend: List[str]  # Legend labels
    confidence: float


class ChartDataExtractor:
    """Extract structured data from different chart types"""
    
    def __init__(self, use_ocr: bool = True):
        """
        Initialize chart data extractor
        
        Args:
            use_ocr: Whether to use OCR for text extraction
        """
        self.use_ocr = use_ocr
    
    def extract_chart_data(self, image: np.ndarray, chart_type: ElementType) -> ChartData:
        """
        Extract data from chart based on its type
        
        Args:
            image: Chart image (numpy array)
            chart_type: Type of chart (BAR_CHART, LINE_CHART, PIE_CHART, SCATTER_PLOT)
            
        Returns:
            ChartData object with extracted information
        """
        if chart_type == ElementType.BAR_CHART:
            return self._extract_bar_chart(image)
        elif chart_type == ElementType.LINE_CHART:
            return self._extract_line_chart(image)
        elif chart_type == ElementType.PIE_CHART:
            return self._extract_pie_chart(image)
        elif chart_type == ElementType.SCATTER_PLOT:
            return self._extract_scatter_plot(image)
        else:
            return ChartData(
                chart_type=ElementType.UNKNOWN,
                values=[], labels=[], series=[], 
                axes={}, legend=[], confidence=0.0
            )
    
    # ============================================================
    # BAR CHART EXTRACTION
    # ============================================================
    
    def _extract_bar_chart(self, image: np.ndarray) -> ChartData:
        """Extract data from bar charts"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect bars
        bars = self._detect_bars(gray)
        
        if not bars:
            return ChartData(
                chart_type=ElementType.BAR_CHART,
                values=[], labels=[], series=[],
                axes={}, legend=[], confidence=0.0
            )
        
        # Determine orientation (vertical or horizontal bars)
        orientation = self._determine_bar_orientation(bars)
        
        # Extract axis information
        axes_info = self._extract_axes_info(image, bars, orientation)
        
        # Extract bar values based on axis scale
        values = self._extract_bar_values(bars, axes_info, orientation)
        
        # Extract labels using OCR
        labels = self._extract_bar_labels(image, bars, orientation)
        
        # Detect legend
        legend = self._extract_legend(image, bars)
        
        return ChartData(
            chart_type=ElementType.BAR_CHART,
            values=values,
            labels=labels,
            series=[{'name': 'Series 1', 'values': values, 'labels': labels}],
            axes=axes_info,
            legend=legend,
            confidence=0.8 if values else 0.3
        )
    
    def _detect_bars(self, gray: np.ndarray) -> List[Dict]:
        """Detect individual bars in bar chart"""
        # Threshold image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bars = []
        h, w = gray.shape
        
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = width * height
            
            # Filter based on area (must be significant)
            if area < 200 or area > h * w * 0.3:
                continue
            
            aspect_ratio = width / height if height > 0 else 0
            
            # Vertical bars (tall) or horizontal bars (wide)
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                bars.append({
                    'x': x, 'y': y,
                    'width': width, 'height': height,
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'centroid': (x + width//2, y + height//2)
                })
        
        # Sort bars by position (left to right for vertical, top to bottom for horizontal)
        if bars:
            # Determine if vertical or horizontal based on aspect ratios
            avg_ar = np.mean([b['aspect_ratio'] for b in bars])
            if avg_ar < 1:  # Vertical bars
                bars = sorted(bars, key=lambda b: b['x'])
            else:  # Horizontal bars
                bars = sorted(bars, key=lambda b: b['y'])
        
        return bars
    
    def _determine_bar_orientation(self, bars: List[Dict]) -> str:
        """Determine if bars are vertical or horizontal"""
        if not bars:
            return 'vertical'
        
        avg_aspect = np.mean([b['aspect_ratio'] for b in bars])
        return 'vertical' if avg_aspect < 1 else 'horizontal'
    
    def _extract_axes_info(self, image: np.ndarray, bars: List[Dict], orientation: str) -> Dict:
        """Extract axis scale and range information"""
        h, w = image.shape[:2]
        
        # Try to detect axis lines
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                                minLineLength=min(h, w)//4, maxLineGap=10)
        
        axes = {'orientation': orientation, 'x_range': (0, w), 'y_range': (0, h)}
        
        if lines is not None:
            # Find main axis lines
            h_lines = []
            v_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                
                if angle < 10 or angle > 170:  # Horizontal
                    h_lines.append(y1)
                elif 80 < angle < 100:  # Vertical
                    v_lines.append(x1)
            
            # Get axis positions (usually at bottom/left of chart)
            if h_lines:
                axes['x_axis_pos'] = max(h_lines)  # Bottom line
            if v_lines:
                axes['y_axis_pos'] = min(v_lines)  # Left line
        
        # Estimate scale from bar positions
        if bars:
            if orientation == 'vertical':
                bar_heights = [b['height'] for b in bars]
                axes['max_value_estimate'] = max(bar_heights) if bar_heights else h
            else:
                bar_widths = [b['width'] for b in bars]
                axes['max_value_estimate'] = max(bar_widths) if bar_widths else w
        
        return axes
    
    def _extract_bar_values(self, bars: List[Dict], axes_info: Dict, orientation: str) -> List[float]:
        """Extract numerical values from bars based on height/width"""
        values = []
        
        if not bars:
            return values
        
        max_estimate = axes_info.get('max_value_estimate', 100)
        
        for bar in bars:
            if orientation == 'vertical':
                # Normalize bar height
                normalized_value = bar['height'] / max_estimate
            else:
                # Normalize bar width
                normalized_value = bar['width'] / max_estimate
            
            # Scale to reasonable range (0-100 by default)
            value = normalized_value * 100
            values.append(round(value, 2))
        
        return values
    
    def _extract_bar_labels(self, image: np.ndarray, bars: List[Dict], orientation: str) -> List[str]:
        """Extract category labels from bar chart using OCR"""
        labels = []
        
        if not self.use_ocr or not bars:
            return [f"Bar {i+1}" for i in range(len(bars))]
        
        try:
            h, w = image.shape[:2]
            
            for i, bar in enumerate(bars):
                # Define region where label likely appears
                if orientation == 'vertical':
                    # Labels below bars
                    label_y1 = min(h - 1, bar['y'] + bar['height'])
                    label_y2 = h
                    label_x1 = max(0, bar['x'] - 10)
                    label_x2 = min(w, bar['x'] + bar['width'] + 10)
                else:
                    # Labels to the left of bars
                    label_x1 = 0
                    label_x2 = bar['x']
                    label_y1 = max(0, bar['y'] - 10)
                    label_y2 = min(h, bar['y'] + bar['height'] + 10)
                
                # Extract label region
                if label_y2 > label_y1 and label_x2 > label_x1:
                    label_region = image[label_y1:label_y2, label_x1:label_x2]
                    
                    # OCR to extract text
                    label_text = self._ocr_text(label_region)
                    labels.append(label_text if label_text else f"Bar {i+1}")
                else:
                    labels.append(f"Bar {i+1}")
        except Exception as e:
            # Fallback to generic labels
            labels = [f"Bar {i+1}" for i in range(len(bars))]
        
        return labels
    
    # ============================================================
    # LINE CHART EXTRACTION
    # ============================================================
    
    def _extract_line_chart(self, image: np.ndarray) -> ChartData:
        """Extract data from line charts"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect lines
        line_segments = self._detect_line_segments(gray)
        
        if not line_segments:
            return ChartData(
                chart_type=ElementType.LINE_CHART,
                values=[], labels=[], series=[],
                axes={}, legend=[], confidence=0.0
            )
        
        # Extract points from line segments
        points = self._extract_line_points(line_segments)
        
        # Extract axis information
        axes_info = self._extract_line_axes(image, points)
        
        # Convert pixel coordinates to values
        values = self._points_to_values(points, axes_info, image.shape)
        
        # Extract labels
        labels = self._extract_point_labels(image, points)
        
        # Detect legend
        legend = self._extract_legend(image, [])
        
        return ChartData(
            chart_type=ElementType.LINE_CHART,
            values=values,
            labels=labels,
            series=[{'name': 'Series 1', 'values': values, 'labels': labels}],
            axes=axes_info,
            legend=legend,
            confidence=0.75 if values else 0.3
        )
    
    def _detect_line_segments(self, gray: np.ndarray) -> List[Tuple]:
        """Detect line segments in line chart"""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                minLineLength=30, maxLineGap=15)
        
        if lines is None:
            return []
        
        # Filter for angled lines (not horizontal/vertical axes)
        line_segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            
            # Keep lines that are not horizontal or vertical
            if 10 < angle < 80 or 100 < angle < 170:
                line_segments.append((x1, y1, x2, y2))
        
        return line_segments
    
    def _extract_line_points(self, line_segments: List[Tuple]) -> List[Tuple]:
        """Extract data points from line segments"""
        if not line_segments:
            return []
        
        # Collect all endpoints
        points = set()
        for x1, y1, x2, y2 in line_segments:
            points.add((x1, y1))
            points.add((x2, y2))
        
        # Sort by x-coordinate
        points = sorted(list(points), key=lambda p: p[0])
        
        return points
    
    def _extract_line_axes(self, image: np.ndarray, points: List[Tuple]) -> Dict:
        """Extract axis information for line chart"""
        h, w = image.shape[:2]
        
        axes = {
            'x_min': 0, 'x_max': w,
            'y_min': 0, 'y_max': h,
            'orientation': 'standard'
        }
        
        if points:
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            axes['data_x_min'] = min(x_coords)
            axes['data_x_max'] = max(x_coords)
            axes['data_y_min'] = min(y_coords)
            axes['data_y_max'] = max(y_coords)
        
        return axes
    
    def _points_to_values(self, points: List[Tuple], axes_info: Dict, shape: Tuple) -> List[float]:
        """Convert pixel coordinates to normalized values"""
        if not points:
            return []
        
        h, w = shape[:2]
        values = []
        
        for x, y in points:
            # Normalize y-coordinate (inverted since image origin is top-left)
            # Higher pixel value = lower chart value
            normalized_y = (h - y) / h * 100
            values.append(round(normalized_y, 2))
        
        return values
    
    def _extract_point_labels(self, image: np.ndarray, points: List[Tuple]) -> List[str]:
        """Extract labels for data points"""
        if not self.use_ocr:
            return [f"Point {i+1}" for i in range(len(points))]
        
        # For now, return generic labels
        # Could be enhanced with OCR of x-axis labels
        return [f"Point {i+1}" for i in range(len(points))]
    
    # ============================================================
    # PIE CHART EXTRACTION
    # ============================================================
    
    def _extract_pie_chart(self, image: np.ndarray) -> ChartData:
        """Extract data from pie charts"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect pie slices using color segmentation
        slices = self._detect_pie_slices(image)
        
        if not slices:
            return ChartData(
                chart_type=ElementType.PIE_CHART,
                values=[], labels=[], series=[],
                axes={}, legend=[], confidence=0.0
            )
        
        # Calculate percentage for each slice
        values = [s['percentage'] for s in slices]
        
        # Extract labels
        labels = self._extract_pie_labels(image, slices)
        
        # Detect legend
        legend = self._extract_legend(image, [])
        
        return ChartData(
            chart_type=ElementType.PIE_CHART,
            values=values,
            labels=labels,
            series=[{'name': 'Slices', 'values': values, 'labels': labels}],
            axes={'total': sum(values)},
            legend=legend,
            confidence=0.8 if values else 0.3
        )
    
    def _detect_pie_slices(self, image: np.ndarray) -> List[Dict]:
        """Detect individual pie slices using contour analysis"""
        # Convert to different color space for better segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Get unique colors (slices should have different colors)
        # Flatten and cluster colors
        pixels = image.reshape(-1, 3)
        
        # Simple color clustering based on dominant colors
        from collections import Counter
        color_counts = Counter([tuple(p) for p in pixels])
        
        # Get most common colors (excluding white/background)
        common_colors = [c for c, count in color_counts.most_common(20) 
                        if count > 100 and not self._is_white(c)]
        
        slices = []
        total_area = 0
        
        for color in common_colors[:10]:  # Limit to 10 slices
            # Create mask for this color
            lower = np.array([max(0, color[0]-20), max(0, color[1]-20), max(0, color[2]-20)])
            upper = np.array([min(255, color[0]+20), min(255, color[1]+20), min(255, color[2]+20)])
            
            mask = cv2.inRange(image, lower, upper)
            
            # Find contours for this color
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum slice area
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        slices.append({
                            'area': area,
                            'color': color,
                            'centroid': (cx, cy),
                            'contour': contour
                        })
                        total_area += area
        
        # Calculate percentages
        if total_area > 0:
            for slice_data in slices:
                slice_data['percentage'] = round((slice_data['area'] / total_area) * 100, 2)
        
        return slices
    
    def _is_white(self, color: Tuple[int, int, int], threshold: int = 240) -> bool:
        """Check if color is white/background"""
        return all(c > threshold for c in color)
    
    def _extract_pie_labels(self, image: np.ndarray, slices: List[Dict]) -> List[str]:
        """Extract labels for pie slices"""
        if not self.use_ocr:
            return [f"Slice {i+1}" for i in range(len(slices))]
        
        labels = []
        for i, slice_data in enumerate(slices):
            # Try to find label near centroid
            cx, cy = slice_data['centroid']
            
            # Sample region around centroid
            x1 = max(0, cx - 30)
            x2 = min(image.shape[1], cx + 30)
            y1 = max(0, cy - 15)
            y2 = min(image.shape[0], cy + 15)
            
            label_region = image[y1:y2, x1:x2]
            label_text = self._ocr_text(label_region)
            
            labels.append(label_text if label_text else f"Slice {i+1}")
        
        return labels
    
    # ============================================================
    # SCATTER PLOT EXTRACTION
    # ============================================================
    
    def _extract_scatter_plot(self, image: np.ndarray) -> ChartData:
        """Extract data from scatter plots"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect scatter points
        points = self._detect_scatter_points(gray)
        
        if not points:
            return ChartData(
                chart_type=ElementType.SCATTER_PLOT,
                values=[], labels=[], series=[],
                axes={}, legend=[], confidence=0.0
            )
        
        # Extract axis information
        axes_info = self._extract_scatter_axes(image, points)
        
        # Convert to normalized values (using y-coordinates)
        values = [round((image.shape[0] - p[1]) / image.shape[0] * 100, 2) for p in points]
        
        # Generate labels
        labels = [f"Point {i+1}" for i in range(len(points))]
        
        # Detect legend
        legend = self._extract_legend(image, [])
        
        return ChartData(
            chart_type=ElementType.SCATTER_PLOT,
            values=values,
            labels=labels,
            series=[{'name': 'Series 1', 'values': values, 'labels': labels, 'points': points}],
            axes=axes_info,
            legend=legend,
            confidence=0.75 if values else 0.3
        )
    
    def _detect_scatter_points(self, gray: np.ndarray) -> List[Tuple]:
        """Detect individual points in scatter plot"""
        # Threshold to find dark points
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours (points)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        points = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Points should be small circles/dots
            if 10 < area < 300:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    points.append((cx, cy))
        
        return points
    
    def _extract_scatter_axes(self, image: np.ndarray, points: List[Tuple]) -> Dict:
        """Extract axis information for scatter plot"""
        h, w = image.shape[:2]
        
        axes = {
            'x_min': 0, 'x_max': w,
            'y_min': 0, 'y_max': h
        }
        
        if points:
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            axes['data_x_range'] = (min(x_coords), max(x_coords))
            axes['data_y_range'] = (min(y_coords), max(y_coords))
        
        return axes
    
    # ============================================================
    # HELPER METHODS
    # ============================================================
    
    def _extract_legend(self, image: np.ndarray, excluded_regions: List) -> List[str]:
        """Extract legend labels from chart"""
        if not self.use_ocr:
            return []
        
        # Legend is typically in corners or edges
        # This is a simplified implementation
        h, w = image.shape[:2]
        
        # Try top-right corner
        legend_region = image[0:h//4, 3*w//4:w]
        
        legend_text = self._ocr_text(legend_region)
        
        if legend_text:
            # Split by lines
            return [line.strip() for line in legend_text.split('\n') if line.strip()]
        
        return []
    
    def _ocr_text(self, image_region: np.ndarray) -> str:
        """Extract text from image region using OCR"""
        if not self.use_ocr or image_region.size == 0:
            return ""
        
        try:
            # Preprocess for better OCR
            if len(image_region.shape) == 3:
                gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_region
            
            # Enhance contrast
            enhanced = cv2.equalizeHist(gray)
            
            # OCR
            pil_image = Image.fromarray(enhanced)
            text = pytesseract.image_to_string(pil_image, config='--psm 6').strip()
            
            return text
        except Exception as e:
            return ""
