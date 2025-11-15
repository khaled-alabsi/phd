"""
Evaluation Metrics Module

This module contains functions for evaluating anomaly detection performance,
including ARL (Average Run Length) calculations and other metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Callable


def get_first_false_alarm_index(flags_normal: np.ndarray) -> Optional[int]:
    """
    Get the index of the first false alarm in normal data.
    
    Args:
        flags_normal: Binary array of predictions on normal data (1 = alarm)
        
    Returns:
        Index of first false alarm, or None if no false alarms
    """
    if np.any(flags_normal == 1):
        return int(np.argmax(flags_normal == 1))
    return None


def get_first_detection_delay(flags_anomaly: np.ndarray) -> Optional[int]:
    """
    Get the detection delay (index of first detection) in anomaly data.
    
    Args:
        flags_anomaly: Binary array of predictions on anomaly data (1 = detection)
        
    Returns:
        Index of first detection, or None if no detection
    """
    if np.any(flags_anomaly == 1):
        return int(np.argmax(flags_anomaly == 1))
    return None


def calculate_arl0(flags_normal: np.ndarray) -> Optional[float]:
    """
    Calculate ARL0 (Average Run Length for in-control process).
    ARL0 is the expected number of samples until a false alarm.
    
    Args:
        flags_normal: Binary array of predictions on normal data
        
    Returns:
        ARL0 value, or None if no false alarms detected
    """
    return get_first_false_alarm_index(flags_normal)


def calculate_arl1(flags_anomaly: np.ndarray) -> Optional[float]:
    """
    Calculate ARL1 (Average Run Length for out-of-control process).
    ARL1 is the expected number of samples until detection of an anomaly.
    
    Args:
        flags_anomaly: Binary array of predictions on anomaly data
        
    Returns:
        ARL1 value, or None if no detection
    """
    return get_first_detection_delay(flags_anomaly)


def calculate_detection_fraction(flags_anomaly: np.ndarray) -> float:
    """
    Calculate the fraction of anomaly samples correctly detected.
    
    Args:
        flags_anomaly: Binary array of predictions on anomaly data
        
    Returns:
        Detection fraction (0.0 to 1.0)
    """
    return np.mean(flags_anomaly)


def apply_detector_with_mcusum_diagnostics(detector, 
                                         X_incontrol_test: np.ndarray,
                                         X_outcontrol_test: np.ndarray,
                                         verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply MCUSUM detector and create diagnostic plots.
    
    Args:
        detector: Fitted MCUSUM detector instance
        X_incontrol_test: In-control test data
        X_outcontrol_test: Out-of-control test data
        verbose: Whether to create diagnostic plots
        
    Returns:
        Tuple of (flags_normal, flags_anomaly)
    """
    from ..visualization.plots import plot_mcusum_diagnostics
    
    stats_normal, flags_normal = detector.predict(X_incontrol_test)
    stats_anomaly, flags_anomaly = detector.predict(X_outcontrol_test)
    
    # Create diagnostic plots
    if verbose:
        plot_mcusum_diagnostics(stats_normal, stats_anomaly, detector.h, "Tennessee_Eastman")

    # Convert boolean flags to 0 (normal) and 1 (anomaly)
    return flags_normal.astype(int), flags_anomaly.astype(int)


def optimize_mcusum_parameters(X_incontrol_train_full: np.ndarray,
                              X_incontrol_test: np.ndarray,
                              X_outcontrol_test: np.ndarray,
                              k_range: np.ndarray,
                              h_value: float,
                              verbose: bool = False) -> Tuple[float, float, float]:
    """
    Optimize MCUSUM k parameter for best performance.
    
    Args:
        X_incontrol_train_full: Full in-control training data
        X_incontrol_test: In-control test data
        X_outcontrol_test: Out-of-control test data
        k_range: Range of k values to test
        h_value: Fixed h value to use
        verbose: Whether to print optimization progress
        
    Returns:
        Tuple of (best_k, best_arl0, best_arl1)
    """
    from ..detectors.mcusum import MCUSUMDetector
    
    best_k = None
    best_arl0 = -np.inf
    best_arl1 = np.inf

    for k_test in k_range:
        mcusum = MCUSUMDetector(k=k_test, h=h_value)
        mcusum.fit(X_incontrol_train_full)
        
        flags_normal, flags_anomaly = apply_detector_with_mcusum_diagnostics(
            mcusum, X_incontrol_test, X_outcontrol_test, verbose=False
        )
        
        arl0 = calculate_arl0(flags_normal) or len(flags_normal)
        arl1 = calculate_arl1(flags_anomaly) or len(flags_anomaly)
        
        # Prefer higher ARL0, then lower ARL1
        if (arl0 > best_arl0) or (arl0 == best_arl0 and arl1 < best_arl1):
            best_k = k_test
            best_arl0 = arl0
            best_arl1 = arl1
            
        if verbose:
            print(f"k={k_test:.2f}, ARL0={arl0}, ARL1={arl1}")

    return best_k, best_arl0, best_arl1


def run_single_experiment(detector_func: Callable,
                         X_incontrol_test: np.ndarray,
                         X_outcontrol_test: np.ndarray,
                         detector_name: str = "Unknown") -> Dict[str, Any]:
    """
    Run a single anomaly detection experiment and calculate metrics.
    
    Args:
        detector_func: Function that takes data and returns predictions
        X_incontrol_test: In-control test data
        X_outcontrol_test: Out-of-control test data
        detector_name: Name of the detector for logging
        
    Returns:
        Dictionary with experiment results
    """
    # Get predictions
    pred_normal = detector_func(X_incontrol_test)
    pred_anomaly = detector_func(X_outcontrol_test)
    
    # Calculate metrics
    arl0 = calculate_arl0(pred_normal)
    arl1 = calculate_arl1(pred_anomaly)
    detection_fraction = calculate_detection_fraction(pred_anomaly)
    
    return {
        'detector': detector_name,
        'ARL0': arl0,
        'ARL1': arl1,
        'detection_fraction': detection_fraction,
        'false_alarm_rate': np.mean(pred_normal),
        'true_positive_rate': detection_fraction
    }


def run_batch_experiments(detector_registry: Dict[str, Callable],
                         batch_data: Dict[str, Any],
                         simulation_runs: range,
                         fault_numbers: range,
                         verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Run batch experiments across multiple detectors, simulation runs, and fault types.
    
    Args:
        detector_registry: Dictionary mapping detector names to prediction functions
        batch_data: Batch data from preprocessing
        simulation_runs: Range of simulation runs to process
        fault_numbers: Range of fault numbers to process
        verbose: Whether to print progress
        
    Returns:
        List of experiment results
    """
    from ..data.preprocessing import get_experiment_data_for_run_fault
    
    results = []
    
    for simulation_run in simulation_runs:
        if verbose:
            print(f"Processing simulation run {simulation_run}...")
            
        # Prepare in-control data for this simulation run
        run_data = get_experiment_data_for_run_fault(batch_data, simulation_run, 1)  # Fault doesn't matter for in-control
        X_incontrol_test = run_data['X_incontrol_test']
        
        # Precompute in-control predictions for all detectors
        pred_normal_dict = {}
        for detector_name, detector_func in detector_registry.items():
            pred_normal_dict[detector_name] = detector_func(X_incontrol_test)

        for fault_number in fault_numbers:
            if verbose:
                print(f"  Processing fault {fault_number}...")
                
            # Get fault-specific data
            fault_data = get_experiment_data_for_run_fault(batch_data, simulation_run, fault_number)
            X_outcontrol_test = fault_data['X_faulty_test']

            # Run all detectors
            for detector_name, detector_func in detector_registry.items():
                pred_anomaly = detector_func(X_outcontrol_test)
                pred_normal = pred_normal_dict[detector_name]

                # Calculate metrics
                arl0 = calculate_arl0(pred_normal)
                arl1 = calculate_arl1(pred_anomaly)
                detection_fraction = calculate_detection_fraction(pred_anomaly)

                if verbose:
                    print(f"    {detector_name}: ARL0={arl0}, ARL1={arl1}")

                results.append({
                    "simulationRun": simulation_run,
                    "faultNumber": fault_number,
                    "model": detector_name,
                    "ARL0": arl0,
                    "ARL1": arl1,
                    "detection_fraction": detection_fraction
                })

    return results


def aggregate_experiment_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Aggregate experiment results into summary statistics.
    
    Args:
        results: List of experiment results from run_batch_experiments
        
    Returns:
        DataFrame with aggregated summary statistics
    """
    df_results = pd.DataFrame(results)
    
    # Group by model and fault number
    grouped = df_results.groupby(['model', 'faultNumber'])
    
    # Calculate conditional statistics (only for non-null values)
    conditional_arl0 = grouped['ARL0'].apply(lambda x: x.dropna().mean() if not x.dropna().empty else np.nan)
    sdrl0 = grouped['ARL0'].apply(lambda x: x.dropna().std() if len(x.dropna()) > 1 else np.nan)
    non_fa_fraction = grouped['ARL0'].apply(lambda x: x.isnull().mean())
    
    conditional_arl1 = grouped['ARL1'].apply(lambda x: x.dropna().mean() if not x.dropna().empty else np.nan)
    sdrl1 = grouped['ARL1'].apply(lambda x: x.dropna().std() if len(x.dropna()) > 1 else np.nan)
    non_detection_fraction = grouped['ARL1'].apply(lambda x: x.isnull().mean())
    
    avg_detection_fraction = grouped['detection_fraction'].mean()

    summary_df = pd.DataFrame({
        'conditional_ARL0': conditional_arl0,
        'SDRL0': sdrl0,
        'non_FA_fraction': non_fa_fraction,
        'conditional_ARL1': conditional_arl1,
        'SDRL1': sdrl1,
        'non_detection_fraction': non_detection_fraction,
        'avg_detection_fraction': avg_detection_fraction
    }).reset_index()
    
    return df_results, summary_df


def compare_detector_performance(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare detector performance across different metrics.
    
    Args:
        results_df: DataFrame with experiment results
        
    Returns:
        DataFrame with performance comparison
    """
    comparison_stats = []
    
    for detector in results_df['model'].unique():
        detector_data = results_df[results_df['model'] == detector]
        
        stats = {
            'Detector': detector,
            'Mean_ARL0': detector_data['ARL0'].dropna().mean(),
            'Std_ARL0': detector_data['ARL0'].dropna().std(),
            'Mean_ARL1': detector_data['ARL1'].dropna().mean(),
            'Std_ARL1': detector_data['ARL1'].dropna().std(),
            'Overall_Detection_Rate': detector_data['detection_fraction'].mean(),
            'False_Alarm_Rate': detector_data['ARL0'].notna().mean(),
            'Miss_Rate': detector_data['ARL1'].isna().mean()
        }
        
        comparison_stats.append(stats)
    
    return pd.DataFrame(comparison_stats).round(3)


def calculate_performance_scores(results_df: pd.DataFrame, 
                               arl0_weight: float = 0.3,
                               arl1_weight: float = 0.4,
                               detection_weight: float = 0.3) -> pd.DataFrame:
    """
    Calculate composite performance scores for detectors.
    
    Args:
        results_df: DataFrame with experiment results
        arl0_weight: Weight for ARL0 component (higher is better)
        arl1_weight: Weight for ARL1 component (lower is better)
        detection_weight: Weight for detection rate component (higher is better)
        
    Returns:
        DataFrame with performance scores
    """
    detector_scores = []
    
    for detector in results_df['model'].unique():
        detector_data = results_df[results_df['model'] == detector]
        
        # Normalize metrics (0-1 scale)
        mean_arl0 = detector_data['ARL0'].dropna().mean()
        mean_arl1 = detector_data['ARL1'].dropna().mean()
        detection_rate = detector_data['detection_fraction'].mean()
        
        # Calculate normalized scores (handle NaN)
        arl0_score = mean_arl0 / results_df['ARL0'].dropna().max() if not np.isnan(mean_arl0) else 0
        arl1_score = 1 - (mean_arl1 / results_df['ARL1'].dropna().max()) if not np.isnan(mean_arl1) else 0
        detection_score = detection_rate
        
        # Composite score
        composite_score = (arl0_weight * arl0_score + 
                          arl1_weight * arl1_score + 
                          detection_weight * detection_score)
        
        detector_scores.append({
            'Detector': detector,
            'ARL0_Score': arl0_score,
            'ARL1_Score': arl1_score,
            'Detection_Score': detection_score,
            'Composite_Score': composite_score
        })
    
    return pd.DataFrame(detector_scores).round(3).sort_values('Composite_Score', ascending=False)