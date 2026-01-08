"""Page modules for DocXtract Dashboard"""

from .home import render_home
from .tables import render_tables
from .charts_analysis import render_charts_analysis
from .chart_data_tables import render_chart_data_tables
from .testing import render_testing
from .about import render_about

__all__ = [
    'render_home',
    'render_tables',
    'render_charts_analysis',
    'render_chart_data_tables',
    'render_testing',
    'render_about'
]

