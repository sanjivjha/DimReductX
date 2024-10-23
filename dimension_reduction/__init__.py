"""
Dimension Reduction Analysis Package
==================================

A comprehensive package for automated dimension reduction analysis,
feature discovery, and optimization.

Main Components:
---------------
- DimensionReducer: Main class for dimension reduction
- FeatureDiscovery: Class for generating derived features
- DimensionReductionVisualizer: Class for visualization

Example Usage:
-------------
from dimension_reduction import DimensionReducer, run_analysis

# Quick usage
report, datasets = run_analysis(X, y, task_type='regression')

# Detailed usage
reducer = DimensionReducer(task_type='regression')
optimized_datasets = reducer.reduce_dimensions(X, y)
report = reducer.generate_report()
"""

from .dimension_reduction import DimensionReducer
from .feature_discovery import FeatureDiscovery
from .visualization import DimensionReductionVisualizer
from .main import run_analysis
from .utils import generate_synthetic_data, save_results, save_plot

__all__ = [
    'DimensionReducer',
    'FeatureDiscovery',
    'DimensionReductionVisualizer',
    'run_analysis',
    'generate_synthetic_data',
    'save_results',
    'save_plot'
]

__version__ = '1.0.0'