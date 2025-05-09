"""
EDA package for Tanzania Climate Analysis
Contains data analysis and visualization utilities
"""

from .data_analysis import (
    calculate_basic_stats,
    analyze_seasonal_patterns,
    analyze_yearly_trends
)

from .visualizations import (
    plot_temperature_distribution,
    plot_yearly_trend
)

__all__ = [
    'calculate_basic_stats',
    'analyze_seasonal_patterns',
    'analyze_yearly_trends',
    'plot_temperature_distribution',
    'plot_yearly_trend'
]