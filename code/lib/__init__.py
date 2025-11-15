"""Anomaly detection library for multivariate process monitoring."""

__version__ = '0.1.0'

# Import main modules
from . import models
from . import data
from . import evaluation
from . import visualization

__all__ = [
    'models',
    'data',
    'evaluation',
    'visualization',
]
