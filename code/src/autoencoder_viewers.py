"""
Separate viewer functions for autoencoder reconstructions.

Two distinct functions with clear purposes:

- display_autoencoder_with_fault_selection():
    Takes raw DataFrames
    Adds interactive dropdown widgets to switch fault/simulation on-the-fly
    Filters and scales data automatically when you change dropdowns
- display_autoencoder_reconstruction():
    Takes pre-processed numpy arrays
    Simple dropdown to switch between datasets
    No filtering - just display
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence
from datetime import datetime
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def display_autoencoder_reconstruction(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    datasets: Dict[str, np.ndarray],
    *,
    feature_names: Optional[Sequence[str]] = None,
    default_key: Optional[str] = None,
    default_feature_count: int = 1,
    error_threshold: Optional[float] = None,
    viewer_key: Optional[str] = None,
) -> None:
    """
    Display interactive viewer for comparing actual vs reconstructed signals.

    **Use this when:** You have pre-processed, scaled numpy arrays.

    Args:
        predict_fn: Function that takes (n_samples, n_features) and returns reconstructions.
        datasets: Dict mapping dataset name -> numpy array (n_samples, n_features).
                  Example: {"Train": X_train, "Test": X_test}
        feature_names: Optional feature names (defaults to "f0", "f1", ...).
        default_key: Which dataset to show initially.
        default_feature_count: Number of features to display initially.
        error_threshold: Optional threshold for highlighting high-error features.
        viewer_key: Unique key for this viewer instance (auto-generated if None).

    Example:
        >>> datasets = {
        ...     "Train": X_train_scaled,
        ...     "Test": X_test_scaled,
        ... }
        >>> display_autoencoder_reconstruction(
        ...     lambda X: model.predict(X, verbose=0),
        ...     datasets,
        ...     feature_names=["temp", "pressure", "flow"]
        ... )
    """
    from src.autoencoder_viewer import display_autoencoder_viewer

    if not datasets:
        raise ValueError("datasets cannot be empty. Provide at least one dataset.")

    if viewer_key is None:
        viewer_key = f"autoencoder_reconstruction_{uuid.uuid4().hex[:8]}"

    # Call the underlying function in simple mode
    display_autoencoder_viewer(
        predict_fn=predict_fn,
        datasets=datasets,
        feature_names=feature_names,
        default_key=default_key,
        default_feature_count=default_feature_count,
        error_threshold=error_threshold,
        fault_data=None,  # Not using fault mode
        viewer_key=viewer_key,
    )


def display_autoencoder_with_fault_selection(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    raw_dataframes: Dict[str, pd.DataFrame],
    scaler: Any,
    *,
    fault_column: str = "faultNumber",
    simulation_column: str = "simulationRun",
    metadata_columns: Optional[list[str]] = None,
    default_dataset: Optional[str] = None,
    default_fault: Optional[int] = None,
    default_simulation: Optional[int] = None,
    feature_names: Optional[Sequence[str]] = None,
    default_feature_count: int = 1,
    error_threshold: Optional[float] = None,
    viewer_key: Optional[str] = None,
) -> None:
    """
    Display interactive viewer with dynamic fault and simulation selection.

    **Use this when:** You have raw DataFrames with fault/simulation metadata
    and want users to interactively select different scenarios.

    Args:
        predict_fn: Function that takes (n_samples, n_features) and returns reconstructions.
        raw_dataframes: Dict mapping dataset name -> raw DataFrame with metadata.
                        Example: {"Fault-free": df_normal, "Faulty": df_faulty}
                        DataFrames MUST contain fault_column and simulation_column.
        scaler: Fitted scaler (e.g., StandardScaler) to transform filtered data.
        fault_column: Name of the column containing fault numbers.
        simulation_column: Name of the column containing simulation run numbers.
        metadata_columns: Columns to remove before scaling. If None, removes
                         [fault_column, simulation_column, "sample"].
        default_dataset: Which dataset to show initially (e.g., "Faulty").
        default_fault: Initial fault number to display.
        default_simulation: Initial simulation run to display.
        feature_names: Optional feature names (defaults to "f0", "f1", ...).
        default_feature_count: Number of features to display initially.
        error_threshold: Optional threshold for highlighting high-error features.
        viewer_key: Unique key for this viewer instance (auto-generated if None).

    Example:
        >>> raw_dfs = {
        ...     "Fault-free": df_normal_raw,  # Has faultNumber, simulationRun columns
        ...     "Faulty": df_faulty_raw,
        ... }
        >>> display_autoencoder_with_fault_selection(
        ...     lambda X: model.predict(X, verbose=0),
        ...     raw_dfs,
        ...     scaler=scaler_incontrol,
        ...     fault_column="faultNumber",
        ...     simulation_column="simulationRun",
        ...     default_fault=2,
        ...     default_simulation=1,
        ... )
    """
    from src.autoencoder_viewer import display_autoencoder_viewer

    if not raw_dataframes:
        raise ValueError("raw_dataframes cannot be empty.")

    if scaler is None:
        raise ValueError("scaler is required for fault selection mode.")

    # Validate that DataFrames have required columns
    for name, df in raw_dataframes.items():
        if fault_column not in df.columns:
            raise ValueError(f"DataFrame '{name}' missing column '{fault_column}'")
        if simulation_column not in df.columns:
            raise ValueError(f"DataFrame '{name}' missing column '{simulation_column}'")

    if metadata_columns is None:
        metadata_columns = [fault_column, simulation_column, "sample"]

    if viewer_key is None:
        viewer_key = f"autoencoder_fault_selection_{uuid.uuid4().hex[:8]}"

    # Call the underlying function in fault mode
    display_autoencoder_viewer(
        predict_fn=predict_fn,
        datasets={},  # Empty - using fault mode
        fault_data=raw_dataframes,
        scaler=scaler,
        fault_column_name=fault_column,
        simulation_column_name=simulation_column,
        columns_to_remove=metadata_columns,
        default_key=default_dataset,
        default_fault=default_fault,
        default_simulation=default_simulation,
        feature_names=feature_names,
        default_feature_count=default_feature_count,
        error_threshold=error_threshold,
        viewer_key=viewer_key,
    )


# Backward compatibility alias (deprecated)
def display_autoencoder_viewer(*args, **kwargs):
    """
    DEPRECATED: Use display_autoencoder_reconstruction() or
    display_autoencoder_with_fault_selection() instead.

    This function will be removed in a future version.
    """
    import warnings
    from src.autoencoder_viewer import display_autoencoder_viewer as _impl

    warnings.warn(
        "display_autoencoder_viewer() is deprecated. Use:\n"
        "  - display_autoencoder_reconstruction() for simple mode\n"
        "  - display_autoencoder_with_fault_selection() for fault selection mode",
        DeprecationWarning,
        stacklevel=2
    )

    return _impl(*args, **kwargs)
