"""
Data structures for DocXtract - UPDATED with chart data support
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Tuple
from enum import Enum
import numpy as np


class ElementType(Enum):
    """Types of elements that can be detected"""
    TABLE = "table"
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    PIE_CHART = "pie_chart"
    SCATTER_PLOT = "scatter_plot"
    UNKNOWN = "unknown"


@dataclass
class ChartSeries:
    """Represents a data series in a chart"""
    name: str
    values: List[float]
    labels: Optional[List[str]] = None
    color: Optional[Tuple[int, int, int]] = None
    points: Optional[List[Tuple[float, float]]] = None  # For scatter/line charts
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        result = {
            'name': self.name,
            'values': self.values
        }
        if self.labels:
            result['labels'] = self.labels
        if self.color:
            result['color'] = self.color
        if self.points:
            result['points'] = self.points
        return result


@dataclass
class AxisInfo:
    """Information about chart axes"""
    orientation: str = 'vertical'  # 'vertical' or 'horizontal'
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    x_range: Optional[Tuple[float, float]] = None
    y_range: Optional[Tuple[float, float]] = None
    x_ticks: Optional[List[str]] = None
    y_ticks: Optional[List[str]] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'orientation': self.orientation,
            'x_label': self.x_label,
            'y_label': self.y_label,
            'x_range': self.x_range,
            'y_range': self.y_range,
            'x_ticks': self.x_ticks,
            'y_ticks': self.y_ticks
        }


@dataclass
class TableCell:
    """Represents a single table cell with metadata"""
    text: str
    row: int
    col: int
    bbox: Optional['BoundingBox'] = None
    confidence: float = 1.0
    is_header: bool = False
    rowspan: int = 1
    colspan: int = 1
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        result = {
            'text': self.text,
            'row': self.row,
            'col': self.col,
            'confidence': self.confidence,
            'is_header': self.is_header,
            'rowspan': self.rowspan,
            'colspan': self.colspan
        }
        if self.bbox:
            result['bbox'] = self.bbox.to_dict()
        return result


@dataclass
class BoundingBox:
    """Bounding box coordinates"""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def iou(self, other: 'BoundingBox') -> float:
        """
        Calculate Intersection over Union (IoU) with another bounding box
        
        Args:
            other: Another BoundingBox object
            
        Returns:
            IoU score between 0 and 1
        """
        # Calculate intersection coordinates
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        
        # Check if there's an intersection
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        
        # Calculate intersection area
        intersection = (ix2 - ix1) * (iy2 - iy1)
        
        # Calculate union area
        union = self.area + other.area - intersection
        
        # Return IoU
        return intersection / union if union > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2,
            'width': self.width,
            'height': self.height
        }


@dataclass
class Table:
    """Table data structure"""
    data: np.ndarray  # The actual table data as numpy array
    bbox: BoundingBox
    page: int = 1
    confidence: float = 1.0
    image: Optional[np.ndarray] = None
    _df_cache: Optional[Any] = field(default=None, repr=False, compare=False)

    @property
    def df(self):
        """Get DataFrame representation (cached)"""
        if self._df_cache is None:
            self._df_cache = self.to_dataframe()
        return self._df_cache

    def to_dataframe(self):
        """Convert to pandas DataFrame with proper handling"""
        try:
            import pandas as pd
            
            if self.data is None or (isinstance(self.data, np.ndarray) and self.data.size == 0):
                return pd.DataFrame()
            
            # Handle numpy array
            if isinstance(self.data, np.ndarray):
                # Ensure 2D array
                if self.data.ndim == 1:
                    data_2d = self.data.reshape(-1, 1)
                else:
                    data_2d = self.data
                    
                # Create DataFrame
                df = pd.DataFrame(data_2d)
                
                # Try to detect header row (first row with mostly text)
                if len(df) > 1:
                    first_row = df.iloc[0]
                    # If first row looks like headers (all strings or mostly strings)
                    non_numeric = sum(1 for val in first_row if not str(val).replace('.', '').replace('-', '').isdigit())
                    if non_numeric > len(first_row) * 0.5:
                        try:
                            # Create column names and ensure uniqueness
                            col_names = [str(val).strip() for val in first_row]
                            col_names = self._make_unique_columns(col_names)
                            df.columns = col_names
                            df = df.iloc[1:].reset_index(drop=True)
                        except:
                            pass  # Keep original structure if header assignment fails
                
                # Ensure columns are unique even with default names
                df.columns = self._make_unique_columns(list(df.columns))
                
                return df
            else:
                # Try to convert other types
                return pd.DataFrame(self.data)
                
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")
        except Exception as e:
            # Return empty DataFrame on error
            import pandas as pd
            return pd.DataFrame()

    def _make_unique_columns(self, col_names: List[str]) -> List[str]:
        """
        Ensure column names are unique by adding numeric suffixes to duplicates.
        Empty strings are replaced with 'Column'.
        
        Args:
            col_names: List of column names (may contain duplicates or empty strings)
            
        Returns:
            List of unique column names
        """
        seen = {}
        unique_names = []
        
        for name in col_names:
            # Replace empty strings with 'Column'
            if not name or name.strip() == '':
                name = 'Column'
            else:
                name = str(name).strip()
            
            # Handle duplicates
            if name in seen:
                seen[name] += 1
                unique_name = f"{name}_{seen[name]}"
            else:
                seen[name] = 0
                unique_name = name
            
            unique_names.append(unique_name)
        
        return unique_names

    def to_csv(self, path: str, include_header: bool = True, index: bool = False):
        """Save table to CSV file"""
        df = self.to_dataframe()
        if df.empty:
            raise ValueError("Cannot export empty table to CSV")
        df.to_csv(path, index=index, header=include_header)
        return path

    def to_csv_string(self, include_header: bool = True, index: bool = False) -> str:
        """Convert table to CSV string for download"""
        df = self.to_dataframe()
        if df.empty:
            return ""
        return df.to_csv(index=index, header=include_header)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'data': self.data.tolist() if isinstance(self.data, np.ndarray) else self.data,
            'bbox': self.bbox.to_dict(),
            'page': self.page,
            'confidence': self.confidence,
            'shape': self.data.shape if isinstance(self.data, np.ndarray) else None
        }


@dataclass
class Graph:
    """Graph/Chart data structure with extracted data"""
    image: np.ndarray
    bbox: BoundingBox
    page: int
    graph_type: ElementType
    confidence: float
    
    # NEW: Store extracted chart data with enhanced structure
    data: Dict = field(default_factory=dict)
    extracted_values: List[float] = field(default_factory=list)
    point_count: int = 0
    
    # Enhanced metadata
    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    series: List[ChartSeries] = field(default_factory=list)
    axes_info: Optional[AxisInfo] = None
    legend: List[str] = field(default_factory=list)
    
    # Additional analysis data
    data_labels: List[str] = field(default_factory=list)
    
    def add_series(self, name: str, values: List[float], labels: Optional[List[str]] = None):
        """Add a data series to the chart"""
        series = ChartSeries(name=name, values=values, labels=labels)
        self.series.append(series)
    
    def set_axes(self, orientation: str = 'vertical', x_label: str = None, y_label: str = None,
                 x_range: Tuple = None, y_range: Tuple = None):
        """Set axis information"""
        self.axes_info = AxisInfo(
            orientation=orientation,
            x_label=x_label,
            y_label=y_label,
            x_range=x_range,
            y_range=y_range
        )

    def save_image(self, path: str):
        """Save graph image to file"""
        import cv2
        cv2.imwrite(path, self.image)

    def to_dict(self) -> dict:
        """Convert to dictionary with all extracted data"""
        result = {
            'bbox': self.bbox.to_dict(),
            'page': self.page,
            'type': self.graph_type.name,
            'confidence': self.confidence,
            'point_count': self.point_count,
            'extracted_values': self.extracted_values,
            'data': self.data,
            'data_labels': self.data_labels,
            'legend': self.legend
        }
        
        if self.title:
            result['title'] = self.title
        if self.x_label:
            result['x_label'] = self.x_label
        if self.y_label:
            result['y_label'] = self.y_label
        if self.series:
            result['series'] = [s.to_dict() for s in self.series]
        if self.axes_info:
            result['axes_info'] = self.axes_info.to_dict()
            
        return result
    
    def to_dataframe(self):
        """Convert chart data to pandas DataFrame"""
        try:
            import pandas as pd
            
            if self.series:
                # Use series data
                data_dict = {}
                for i, s in enumerate(self.series):
                    labels = s.labels if s.labels else [f"Point {j+1}" for j in range(len(s.values))]
                    data_dict[s.name] = pd.Series(s.values, index=labels)
                return pd.DataFrame(data_dict)
            elif self.extracted_values:
                # Use extracted values
                labels = self.data_labels if self.data_labels else [f"Point {i+1}" for i in range(len(self.extracted_values))]
                return pd.DataFrame({
                    'Value': self.extracted_values
                }, index=labels)
            else:
                return pd.DataFrame()
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")
    
    def to_csv(self, path: str):
        """Export chart data to CSV"""
        df = self.to_dataframe()
        if not df.empty:
            df.to_csv(path)
            return path
        return None

    def get_bar_values(self) -> List[float]:
        """Get bar chart values"""
        if self.graph_type == ElementType.BAR_CHART:
            return self.extracted_values
        return []

    def get_line_points(self) -> List[tuple]:
        """Get line chart points (x, y coordinates)"""
        if self.graph_type == ElementType.LINE_CHART:
            return self.data.get('points', [])
        return []

    def get_scatter_points(self) -> List[tuple]:
        """Get scatter plot points (x, y coordinates)"""
        if self.graph_type == ElementType.SCATTER_PLOT:
            return self.data.get('points', [])
        return []

    def get_pie_slices(self) -> int:
        """Get number of pie chart slices"""
        if self.graph_type == ElementType.PIE_CHART:
            return self.data.get('slice_count', 0)
        return 0

    def summary(self) -> str:
        """Get a text summary of the chart"""
        summary = f"Chart Type: {self.graph_type.name}\n"
        summary += f"Confidence: {self.confidence:.2f}\n"
        summary += f"Page: {self.page}\n"
        summary += f"Bounding Box: ({self.bbox.x1:.0f}, {self.bbox.y1:.0f}) to ({self.bbox.x2:.0f}, {self.bbox.y2:.0f})\n"
        
        if self.graph_type == ElementType.BAR_CHART:
            summary += f"Number of Bars: {self.point_count}\n"
            if self.extracted_values:
                summary += f"Bar Values: {[f'{v:.2f}' for v in self.extracted_values[:5]]}"
                if len(self.extracted_values) > 5:
                    summary += "..."
                summary += "\n"
        
        elif self.graph_type == ElementType.LINE_CHART:
            summary += f"Number of Points: {self.point_count}\n"
            points = self.get_line_points()
            if points:
                summary += f"Sample Points: {points[:3]}"
                if len(points) > 3:
                    summary += "..."
                summary += "\n"
        
        elif self.graph_type == ElementType.SCATTER_PLOT:
            summary += f"Number of Points: {self.point_count}\n"
        
        elif self.graph_type == ElementType.PIE_CHART:
            summary += f"Number of Slices: {self.get_pie_slices()}\n"
        
        return summary


@dataclass
class ExtractionResult:
    """Result of extraction containing all detected elements"""
    source: str
    tables: List[Table] = field(default_factory=list)
    graphs: List[Graph] = field(default_factory=list)
    n_pages: int = 0

    @property
    def n_tables(self) -> int:
        return len(self.tables)

    @property
    def n_graphs(self) -> int:
        return len(self.graphs)

    def __repr__(self) -> str:
        return (f"ExtractionResult(source='{self.source}', "
                f"n_pages={self.n_pages}, "
                f"n_tables={self.n_tables}, "
                f"n_graphs={self.n_graphs})")

    def to_dict(self) -> dict:
        """Convert entire result to dictionary"""
        return {
            'source': self.source,
            'n_pages': self.n_pages,
            'n_tables': self.n_tables,
            'n_graphs': self.n_graphs,
            'tables': [table.to_dict() for table in self.tables],
            'graphs': [graph.to_dict() for graph in self.graphs]
        }

    def to_json(self, path: Optional[str] = None) -> str:
        """Convert to JSON string or save to file"""
        import json
        json_str = json.dumps(self.to_dict(), indent=2)
        
        if path:
            with open(path, 'w') as f:
                f.write(json_str)
        
        return json_str

    def summary(self) -> str:
        """Get a text summary of all extracted elements"""
        summary = f"=== Extraction Summary ===\n"
        summary += f"Source: {self.source}\n"
        summary += f"Pages: {self.n_pages}\n"
        summary += f"Tables: {self.n_tables}\n"
        summary += f"Graphs: {self.n_graphs}\n\n"
        
        if self.tables:
            summary += "--- Tables ---\n"
            for i, table in enumerate(self.tables, 1):
                summary += f"Table {i}: Page {table.page}, "
                summary += f"Shape: {table.data.shape if hasattr(table.data, 'shape') else 'N/A'}\n"
        
        if self.graphs:
            summary += "\n--- Graphs ---\n"
            for i, graph in enumerate(self.graphs, 1):
                summary += f"\nGraph {i}:\n"
                summary += graph.summary()
        
        return summary

    def filter_by_type(self, element_type: ElementType) -> List[Graph]:
        """Filter graphs by type"""
        return [g for g in self.graphs if g.graph_type == element_type]

    def get_bar_charts(self) -> List[Graph]:
        """Get all bar charts"""
        return self.filter_by_type(ElementType.BAR_CHART)

    def get_line_charts(self) -> List[Graph]:
        """Get all line charts"""
        return self.filter_by_type(ElementType.LINE_CHART)

    def get_pie_charts(self) -> List[Graph]:
        """Get all pie charts"""
        return self.filter_by_type(ElementType.PIE_CHART)

    def get_scatter_plots(self) -> List[Graph]:
        """Get all scatter plots"""
        return self.filter_by_type(ElementType.SCATTER_PLOT)