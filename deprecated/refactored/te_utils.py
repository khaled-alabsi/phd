"""
Tennessee Eastman Process (TEP) Utility Functions

This module contains utility functions for data processing, visualization,
and evaluation in the TEP anomaly detection project.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def check_python_version() -> None:
    """Check if Python version is compatible (3.11 or below)."""
    if sys.version_info >= (3, 12):
        print("Warning: This code was developed for Python 3.11 or below.")
        print(f"Current version: {sys.version}")
    else:
        print(f"Python version check passed: {sys.version}")


def compute_first_detection_delay(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window_size: int = 10
) -> Dict[int, float]:
    """
    Compute the first detection delay for each fault class.
    
    Args:
        y_true: True labels array
        y_pred: Predicted labels array
        window_size: Size of the sliding window for detection confirmation
        
    Returns:
        Dictionary mapping fault class to average detection delay
    """
    delays = {}
    
    for fault_class in np.unique(y_true):
        if fault_class == 0:  # Skip normal class
            continue
            
        fault_indices = np.where(y_true == fault_class)[0]
        class_delays = []
        
        for idx in fault_indices:
            # Find first detection within window
            start_idx = max(0, idx - window_size)
            end_idx = min(len(y_pred), idx + window_size)
            
            # Check if fault was detected in the window
            detected = False
            for i in range(start_idx, end_idx):
                if y_pred[i] == fault_class:
                    delay = abs(i - idx)
                    class_delays.append(delay)
                    detected = True
                    break
            
            if not detected:
                class_delays.append(np.inf)  # Not detected
        
        if class_delays:
            # Remove infinite delays and compute average
            valid_delays = [d for d in class_delays if d != np.inf]
            if valid_delays:
                delays[fault_class] = np.mean(valid_delays)
            else:
                delays[fault_class] = np.inf
    
    return delays


def save_plot(plot_name: str, suffix: str = "", plot_path: str = "default") -> None:
    """
    Save a plot with proper naming and path handling.
    
    Args:
        plot_name: Name of the plot
        suffix: Optional suffix (e.g., class ID or 'average')
        plot_path: Path where to save the plot
    """
    if plot_path == "default":
        plot_path = f"output/{VERSION}/"
    
    # Ensure directory exists
    os.makedirs(plot_path, exist_ok=True)
    
    # Create filename
    filename = f"{plot_name}_{VERSION}_{suffix}.png" if suffix else f"{plot_name}_{VERSION}.png"
    full_path = os.path.join(plot_path, filename)
    
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {full_path}")


def save_dataframe(df: pd.DataFrame, name: str, suffix: str = "") -> None:
    """
    Save a dataframe to CSV with proper naming.
    
    Args:
        df: DataFrame to save
        name: Name for the file
        suffix: Optional suffix (e.g. class id, 'summary')
    """
    output_dir = f"output/{VERSION}/"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{name}_{VERSION}_{suffix}.csv" if suffix else f"{name}_{VERSION}.csv"
    full_path = os.path.join(output_dir, filename)
    
    df.to_csv(full_path, index=True)
    print(f"DataFrame saved: {full_path}")


def check_balance_difference(df1: pd.DataFrame, df2: pd.DataFrame, threshold: int = 100) -> None:
    """
    Check the difference in class balance between two dataframes.
    
    Args:
        df1: First dataframe
        df2: Second dataframe
        threshold: Threshold for warning about imbalance
    """
    print("Checking class balance differences...")
    
    for df_name, df in [("df1", df1), ("df2", df2)]:
        if 'faultNumber' in df.columns:
            class_counts = df['faultNumber'].value_counts().sort_index()
            print(f"\nClass counts in {df_name}:")
            print(class_counts)
            
            # Check for severe imbalance
            max_count = class_counts.max()
            min_count = class_counts.min()
            if max_count - min_count > threshold:
                print(f"Warning: Severe class imbalance detected in {df_name}")
                print(f"Max count: {max_count}, Min count: {min_count}")


    def calculate_statistics_per_fault_run(
        df: pd.DataFrame,
        fault_col: str = 'faultNumber',
        time_col: str = 'time'
    ) -> pd.DataFrame:
        """
        Calculate statistics for each fault run.
        
        Args:
            df: Input dataframe
            fault_col: Column name for fault numbers
            time_col: Column name for time
            
        Returns:
            DataFrame with statistics per fault run
        """
        def compute_stats(group: pd.DataFrame) -> pd.DataFrame:
            stats = {
                'start_time': group[time_col].min(),
                'end_time': group[time_col].max(),
                'duration': group[time_col].max() - group[time_col].min(),
                'sample_count': len(group)
            }
            return pd.DataFrame([stats])
        
        # Group by fault number and time (assuming consecutive samples belong to same run)
        df_sorted = df.sort_values([fault_col, time_col])
        
        # Create fault run groups
        fault_runs = []
        current_run = []
        current_fault = None
        
        for _, row in df_sorted.iterrows():
            if row[fault_col] != current_fault:
                if current_run:
                    fault_runs.append(pd.DataFrame(current_run))
                current_run = [row]
                current_fault = row[fault_col]
            else:
                current_run.append(row)
        
        if current_run:
            fault_runs.append(pd.DataFrame(current_run))
        
        # Compute statistics for each run
        results = []
        for i, run_df in enumerate(fault_runs):
            stats = compute_stats(run_df)
            stats['run_id'] = i
            stats['fault_number'] = run_df[fault_col].iloc[0]
            results.append(stats)
        
        return pd.concat(results, ignore_index=True)


def compute_classification_scores_per_fault(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> pd.DataFrame:
    """
    Compute classification scores for each fault class.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        DataFrame with scores per fault class
    """
    # Compute the confusion matrix for multi-class classification
    cm = confusion_matrix(y_true, y_pred)
    
    # Get the number of classes from the confusion matrix shape
    n_classes: int = cm.shape[0]
    
    # List to store false positive rate (FPR) for each class
    fpr_list: List[float] = []
    
    # Iterate over each class index (one-vs-rest treatment)
    for i in range(n_classes):
        # False positives for class i: sum of predictions as class i
        fp = np.sum(cm[:, i]) - cm[i, i]
        
        # True negatives: all samples excluding the positives for class i
        # and the false positives for class i
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        
        # Compute FPR for this class
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Append this class's FPR to the list
        fpr_list.append(fpr)
    
    # Return the macro-average of FPRs across all classes
    return pd.DataFrame({
        'Class': range(n_classes),
        'False_Positive_Rate': fpr_list
    }).set_index('Class')


def false_alarm_rate_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> pd.DataFrame:
    """
    Compute false alarm rate for each class.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        DataFrame with false alarm rates per class
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Number of classes
    n_classes: int = cm.shape[0]
    
    # Dictionary to store FPR per class
    fpr_per_class: dict[int, float] = {}
    
    # Compute FPR for each class using one-vs-rest
    for i in range(n_classes):
        # False positives for class i
        fp = np.sum(cm[:, i]) - cm[i, i]
        
        # True negatives for class i
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        
        # Compute FPR
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fpr_per_class[i] = fpr
    
    return pd.DataFrame.from_dict(fpr_per_class,
                                orient='index',
                                columns=['False_Alarm_Rate'])


def positive_alarm_rate_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> pd.DataFrame:
    """
    Compute positive alarm rate for each class.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        DataFrame with positive alarm rates per class
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes: int = cm.shape[0]
    par_per_class: dict[int, float] = {}
    
    for i in range(n_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        par = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        par_per_class[i] = par
    
    return pd.DataFrame.from_dict(par_per_class,
                                orient='index',
                                columns=['Positive_Alarm_Rate'])


def compute_average_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> pd.DataFrame:
    """
    Compute average classification metrics across all classes.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        DataFrame with average metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes: int = cm.shape[0]
    
    # Lists to store metrics for each class
    acc_list: List[float] = []
    bal_acc_list: List[float] = []
    f1_list: List[float] = []
    precision_list: List[float] = []
    recall_list: List[float] = []
    
    # Compute metrics for each class
    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - tp - fp - fn
        
        # Accuracy for this class
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        acc_list.append(acc)
        
        # Balanced accuracy for this class
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bal_acc = (sens + spec) / 2
        bal_acc_list.append(bal_acc)
        
        # F1 score for this class
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)
    
    # Return average metrics
    return pd.DataFrame({
        "Macro accuracy": float(np.mean(acc_list)),  # "How often am I correct overall, regardless of class?"
        "Macro precision": float(np.mean(precision_list)),  # "How often am I correct when I predict a class?"
        "Macro recall": float(np.mean(recall_list)),  # "How often do I correctly identify a class?"
        "Macro F1": float(np.mean(f1_list)),  # "Harmonic mean of precision and recall"
        "Balanced Accuracy": float(np.mean(bal_acc_list)),  # "How well do I perform on both classes, accounting for imbalance?"
    }, index=[0])


# Global version variable
VERSION: str = "1.00"
