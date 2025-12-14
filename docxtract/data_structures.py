"""
Data structures for DocXtract - UPDATED with chart data support
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
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

    def to_dataframe(self):
        """Convert to pandas DataFrame"""
        try:
            import pandas as pd
            return pd.DataFrame(self.data)
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")

    def to_csv(self, path: str):
        """Save table to CSV"""
        df = self.to_dataframe()
        df.to_csv(path, index=False)

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
    
    # NEW: Store extracted chart data
    data: Dict = field(default_factory=dict)
    extracted_values: List[float] = field(default_factory=list)
    point_count: int = 0
    
    # Optional metadata
    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None

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
            'data': self.data
        }
        
        if self.title:
            result['title'] = self.title
        if self.x_label:
            result['x_label'] = self.x_label
        if self.y_label:
            result['y_label'] = self.y_label
            
        return result

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