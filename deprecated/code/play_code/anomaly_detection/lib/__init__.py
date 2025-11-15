"""
Anomaly Detection Library

A comprehensive library for multivariate anomaly detection in industrial processes,
specifically designed for the Tennessee Eastman Process dataset.

This library provides:
- Multiple anomaly detection algorithms (MCUSUM, Autoencoder, EWMA variants)
- Data preprocessing and loading utilities
- Visualization and analysis tools
- Evaluation metrics and comparison functions
"""

__version__ = "1.0.0"
__author__ = "PhD Project"

# Import main components for easy access
from .config import (
    setup_experiment, 
    ExperimentConfig,
    print_configuration,
    get_all_detector_names,
    get_minimal_detector_names
)

from .data.preprocessing import (
    load_tep_data,
    prepare_experiment_data,
    prepare_batch_experiment_data,
    FAULT_INJECTION_POINT
)

from .evaluation.metrics import (
    run_batch_experiments,
    aggregate_experiment_results,
    compare_detector_performance
)

from .visualization.plots import (
    analyze_results,
    ModelComparisonAnalyzer
)

# Make key classes available at package level
from .detectors.mcusum import MCUSUMDetector
from .detectors.autoencoder import AutoencoderDetector, AutoencoderDetectorEnhanced
from .detectors.ewma import StandardMEWMA, MEWMS, MEWMV, REWMV, MaxMEWMV, MNSE

__all__ = [
    # Main functions
    'setup_experiment',
    'load_tep_data', 
    'prepare_experiment_data',
    'prepare_batch_experiment_data',
    'run_batch_experiments',
    'aggregate_experiment_results',
    'analyze_results',
    
    # Classes
    'ExperimentConfig',
    'ModelComparisonAnalyzer',
    'MCUSUMDetector',
    'AutoencoderDetector', 
    'AutoencoderDetectorEnhanced',
    'StandardMEWMA',
    'MEWMS',
    'MEWMV', 
    'REWMV',
    'MaxMEWMV',
    'MNSE',
    
    # Constants
    'FAULT_INJECTION_POINT',
    
    # Utility functions
    'print_configuration',
    'get_all_detector_names',
    'get_minimal_detector_names',
    'compare_detector_performance'
]