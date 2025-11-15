"""
Evaluation Package

Contains metrics and evaluation utilities for anomaly detection performance assessment.
"""

from .metrics import (
    get_first_false_alarm_index,
    get_first_detection_delay,
    calculate_arl0,
    calculate_arl1,
    calculate_detection_fraction,
    apply_detector_with_mcusum_diagnostics,
    optimize_mcusum_parameters,
    run_single_experiment,
    run_batch_experiments,
    aggregate_experiment_results,
    compare_detector_performance,
    calculate_performance_scores
)

__all__ = [
    'get_first_false_alarm_index',
    'get_first_detection_delay',
    'calculate_arl0',
    'calculate_arl1',
    'calculate_detection_fraction',
    'apply_detector_with_mcusum_diagnostics',
    'optimize_mcusum_parameters',
    'run_single_experiment',
    'run_batch_experiments',
    'aggregate_experiment_results',
    'compare_detector_performance',
    'calculate_performance_scores'
]