"""
Model Evaluation Framework

This module provides a framework for evaluating multiple anomaly detection models
across different simulation runs and fault conditions. It computes comprehensive
performance metrics including ARL0, ARL1, and classification metrics.
"""

from typing import Any, Callable, Dict, List, Tuple
import numpy as np
import pandas as pd
from numpy.typing import NDArray


def get_first_false_alarm_index(flags_normal: NDArray) -> int | None:
    """
    Get the index of the first false alarm in in-control data.

    Args:
        flags_normal: Binary flags for in-control data (0=normal, 1=alarm)

    Returns:
        Index of first false alarm, or None if no false alarm occurred
    """
    if np.any(flags_normal == 1):
        return int(np.argmax(flags_normal == 1))
    return None


def get_first_detection_delay(flags_anomaly: NDArray) -> int | None:
    """
    Get the index of the first detection in out-of-control data.

    Args:
        flags_anomaly: Binary flags for out-of-control data (0=normal, 1=alarm)

    Returns:
        Index of first detection, or None if no detection occurred
    """
    if np.any(flags_anomaly == 1):
        return int(np.argmax(flags_anomaly == 1))
    return None


def compute_classification_metrics(pred_normal: NDArray, pred_anomaly: NDArray) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics from predictions.

    Args:
        pred_normal: Binary predictions for in-control data (negatives)
        pred_anomaly: Binary predictions for out-of-control data (positives)

    Returns:
        Dictionary containing confusion matrix components and derived metrics
    """
    # Confusion matrix components
    N_neg = int(len(pred_normal))
    N_pos = int(len(pred_anomaly))
    FP = int(np.sum(pred_normal))
    TN = int(N_neg - FP)
    TP = int(np.sum(pred_anomaly))
    FN = int(N_pos - TP)

    # Derived metrics
    recall = TP / N_pos if N_pos > 0 else np.nan
    specificity = TN / N_neg if N_neg > 0 else np.nan
    fpr = FP / N_neg if N_neg > 0 else np.nan
    fnr = FN / N_pos if N_pos > 0 else np.nan
    precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    accuracy = (TP + TN) / (N_pos + N_neg) if (N_pos + N_neg) > 0 else np.nan
    f1 = (2 * precision * recall) / (precision + recall) if (
        precision is not np.nan and recall is not np.nan and (precision + recall) > 0
    ) else np.nan

    detection_fraction = np.mean(pred_anomaly)

    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "accuracy": accuracy,
        "f1": f1,
        "detection_fraction": detection_fraction
    }


def evaluate_models(
    models: Dict[str, Callable[[NDArray], NDArray]],
    df_ff_test_raw: pd.DataFrame,
    df_f_test_raw: pd.DataFrame,
    scaler: Any,
    fault_injection_point: int,
    simulation_column: str = "simulationRun",
    fault_column: str = "faultNumber",
    columns_to_remove: List[str] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate multiple models across simulation runs and fault conditions.

    Args:
        models: Dictionary mapping model names to prediction functions
        df_ff_test_raw: Raw fault-free test data with metadata columns
        df_f_test_raw: Raw faulty test data with metadata columns
        scaler: Fitted scaler for data normalization
        fault_injection_point: Index where fault is injected
        simulation_column: Name of simulation run column
        fault_column: Name of fault number column
        columns_to_remove: List of metadata columns to remove before scaling
        verbose: Whether to print progress information

    Returns:
        Tuple of (df_results, summary_df):
            - df_results: Detailed results for each run/fault/model combination
            - summary_df: Aggregated summary statistics
    """
    if columns_to_remove is None:
        columns_to_remove = [simulation_column, fault_column, "sample"]

    # Extract unique simulation runs and fault numbers
    simulation_run_range = [int(x) for x in sorted(df_ff_test_raw[simulation_column].unique())]
    fault_number_range = [int(x) for x in sorted(df_f_test_raw[fault_column].unique())]

    results: List[Dict[str, Any]] = []

    total_iterations = len(simulation_run_range) * len(fault_number_range) * len(models)
    current_iteration = 0

    for simulation_run in simulation_run_range:
        # Prepare in-control data for this simulation run
        df_ff_test_sequenced = df_ff_test_raw.query(f"{simulation_column} == @simulation_run")
        X_incontrol_test_df = df_ff_test_sequenced.drop(columns=columns_to_remove, axis=1)
        X_incontrol_test_scaled = scaler.transform(X_incontrol_test_df)[fault_injection_point:]

        # Precompute in-control predictions once per simulation run (efficiency optimization)
        pred_normal_dict = {}
        for model_name, model_func in models.items():
            pred_normal_dict[model_name] = model_func(X_incontrol_test_scaled)

        for fault_number in fault_number_range:
            # Prepare out-of-control data for this fault
            df_f_test_sequenced = df_f_test_raw.query(
                f"{fault_column} == @fault_number and {simulation_column} == @simulation_run"
            )
            X_out_of_control_test_df = df_f_test_sequenced.drop(columns=columns_to_remove, axis=1)
            X_out_of_control_test_scaled = scaler.transform(X_out_of_control_test_df)[fault_injection_point:]

            # Evaluate each model
            for model_name, model_func in models.items():
                current_iteration += 1

                if verbose:
                    print(f"[{current_iteration}/{total_iterations}] "
                          f"Simulation: {simulation_run}, Fault: {fault_number}, Model: {model_name}")

                # Get predictions
                pred_anomaly = model_func(X_out_of_control_test_scaled)
                pred_normal = pred_normal_dict[model_name]

                # Compute ARL metrics
                arl1 = get_first_detection_delay(pred_anomaly)
                arl0 = get_first_false_alarm_index(pred_normal)

                if verbose:
                    print(f"  ARL0: {arl0}, ARL1: {arl1}")

                # Compute classification metrics
                metrics = compute_classification_metrics(pred_normal, pred_anomaly)

                # Store results
                results.append({
                    simulation_column: simulation_run,
                    fault_column: fault_number,
                    "model": model_name,
                    "ARL0": arl0,
                    "ARL1": arl1,
                    **metrics
                })

    if verbose:
        print("\nEvaluation complete. Aggregating results...")

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Aggregate summary statistics
    summary_df = create_summary_statistics(df_results)

    return df_results, summary_df


def create_summary_statistics(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregated summary statistics from detailed results.

    Args:
        df_results: Detailed results DataFrame

    Returns:
        Summary DataFrame with aggregated metrics per model and fault
    """
    grouped = df_results.groupby(['model', 'faultNumber'])

    # ARL0 statistics
    conditional_arl0 = grouped['ARL0'].apply(
        lambda x: x.dropna().mean() if not x.dropna().empty else np.nan
    )
    sdrl0 = grouped['ARL0'].apply(
        lambda x: x.dropna().std() if len(x.dropna()) > 1 else np.nan
    )
    non_fa_fraction = grouped['ARL0'].apply(lambda x: x.isnull().mean())

    # ARL1 statistics
    conditional_arl1 = grouped['ARL1'].apply(
        lambda x: x.dropna().mean() if not x.dropna().empty else np.nan
    )
    sdrl1 = grouped['ARL1'].apply(
        lambda x: x.dropna().std() if len(x.dropna()) > 1 else np.nan
    )
    non_detection_fraction = grouped['ARL1'].apply(lambda x: x.isnull().mean())

    # Other aggregated metrics
    avg_detection_fraction = grouped['detection_fraction'].mean()
    mean_precision = grouped['precision'].mean()
    mean_recall = grouped['recall'].mean()
    mean_specificity = grouped['specificity'].mean()
    mean_fpr = grouped['false_positive_rate'].mean()
    mean_fnr = grouped['false_negative_rate'].mean()
    mean_accuracy = grouped['accuracy'].mean()
    mean_f1 = grouped['f1'].mean()

    summary_df = pd.DataFrame({
        'conditional_ARL0': conditional_arl0,
        'SDRL0': sdrl0,
        'non_FA_fraction': non_fa_fraction,
        'conditional_ARL1': conditional_arl1,
        'SDRL1': sdrl1,
        'non_detection_fraction': non_detection_fraction,
        'avg_detection_fraction': avg_detection_fraction,
        'precision': mean_precision,
        'recall': mean_recall,
        'specificity': mean_specificity,
        'false_positive_rate': mean_fpr,
        'false_negative_rate': mean_fnr,
        'accuracy': mean_accuracy,
        'f1': mean_f1
    }).reset_index()

    return summary_df
