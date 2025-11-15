"""
Data Loading and Preprocessing Module

This module contains functions for loading, preprocessing, and preparing 
Tennessee Eastman Process (TEP) data for anomaly detection experiments.
"""

import pyreadr
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any, Optional


# Configuration constants
TARGET_VARIABLE_COLUMN_NAME = "faultNumber"
SIMULATION_RUN_COLUMN_NAME = "simulationRun"
COLUMNS_TO_REMOVE = ["faultNumber", "simulationRun", "sample"]
FAULT_INJECTION_POINT = 160


def load_tep_data(data_dir: str = "../data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load Tennessee Eastman Process datasets from R data files.
    
    Args:
        data_dir: Directory containing the TEP data files (default: ../data)
        
    Returns:
        Tuple of (fault_free_training, fault_free_testing, faulty_training, faulty_testing)
    """
    # Load fault-free data
    fault_free_training_dict = pyreadr.read_r(f"{data_dir}/TEP_FaultFree_Training.RData")
    fault_free_testing_dict = pyreadr.read_r(f"{data_dir}/TEP_FaultFree_Testing.RData")
    
    # Load faulty data  
    faulty_training_dict = pyreadr.read_r(f"{data_dir}/TEP_Faulty_Training.RData")
    faulty_testing_dict = pyreadr.read_r(f"{data_dir}/TEP_Faulty_Testing.RData")
    
    # Extract DataFrames
    df_ff_training = fault_free_training_dict["fault_free_training"]
    df_ff_testing = fault_free_testing_dict["fault_free_testing"]
    df_f_training = faulty_training_dict["faulty_training"]
    df_f_testing = faulty_testing_dict["faulty_testing"]
    
    return df_ff_training, df_ff_testing, df_f_training, df_f_testing


def prepare_incontrol_data(df_ff_training_raw: pd.DataFrame, 
                          df_ff_testing_raw: pd.DataFrame,
                          simulation_run: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Prepare in-control (fault-free) training and testing data.
    
    Args:
        df_ff_training_raw: Raw fault-free training data
        df_ff_testing_raw: Raw fault-free testing data
        simulation_run: Specific simulation run to filter (if None, uses all data)
        
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, scaler)
    """
    # Filter by simulation run if specified
    if simulation_run is not None:
        df_ff_training = df_ff_training_raw.query("simulationRun == @simulation_run")
        df_ff_testing = df_ff_testing_raw.query("simulationRun == @simulation_run")
    else:
        df_ff_training = df_ff_training_raw.copy()
        df_ff_testing = df_ff_testing_raw.copy()
    
    # Remove metadata columns
    X_train_df = df_ff_training.drop(columns=COLUMNS_TO_REMOVE, axis=1)
    X_test_df = df_ff_testing.drop(columns=COLUMNS_TO_REMOVE, axis=1)
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)
    
    return X_train_scaled, X_test_scaled, scaler


def prepare_full_incontrol_data(df_ff_training_raw: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """
    Prepare the full in-control training dataset (all simulation runs).
    
    Args:
        df_ff_training_raw: Raw fault-free training data
        
    Returns:
        Tuple of (X_train_full_scaled, scaler)
    """
    # Remove metadata columns
    X_train_full_df = df_ff_training_raw.drop(columns=COLUMNS_TO_REMOVE, axis=1)
    
    # Scale the data
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full_df)
    
    return X_train_full_scaled, scaler


def prepare_faulty_data(df_f_training_raw: pd.DataFrame,
                       df_f_testing_raw: pd.DataFrame,
                       fault_number: int,
                       simulation_run: int,
                       scaler: StandardScaler,
                       apply_fault_injection_cutoff: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare out-of-control (faulty) training and testing data.
    
    Args:
        df_f_training_raw: Raw faulty training data
        df_f_testing_raw: Raw faulty testing data
        fault_number: Fault type to filter
        simulation_run: Simulation run to filter
        scaler: Pre-fitted scaler from in-control data
        apply_fault_injection_cutoff: Whether to apply fault injection point cutoff
        
    Returns:
        Tuple of (X_train_faulty_scaled, X_test_faulty_scaled)
    """
    # Filter by fault number and simulation run
    df_f_training = df_f_training_raw.query(
        "faultNumber == @fault_number and simulationRun == @simulation_run"
    )
    df_f_testing = df_f_testing_raw.query(
        "faultNumber == @fault_number and simulationRun == @simulation_run"
    )
    
    # Remove metadata columns
    X_train_faulty_df = df_f_training.drop(columns=COLUMNS_TO_REMOVE, axis=1)
    X_test_faulty_df = df_f_testing.drop(columns=COLUMNS_TO_REMOVE, axis=1)
    
    # Scale using the pre-fitted scaler
    X_train_faulty_scaled = scaler.transform(X_train_faulty_df)
    X_test_faulty_scaled = scaler.transform(X_test_faulty_df)
    
    # Apply fault injection cutoff if requested
    if apply_fault_injection_cutoff:
        X_train_faulty_scaled = X_train_faulty_scaled[FAULT_INJECTION_POINT:]
        X_test_faulty_scaled = X_test_faulty_scaled[FAULT_INJECTION_POINT:]
    
    return X_train_faulty_scaled, X_test_faulty_scaled


def prepare_experiment_data(df_ff_training_raw: pd.DataFrame,
                           df_ff_testing_raw: pd.DataFrame,
                           df_f_training_raw: pd.DataFrame,
                           df_f_testing_raw: pd.DataFrame,
                           simulation_run: int,
                           fault_number: int,
                           apply_fault_injection_cutoff: bool = True) -> Dict[str, np.ndarray]:
    """
    Prepare all data needed for a single anomaly detection experiment.
    
    Args:
        df_ff_training_raw: Raw fault-free training data
        df_ff_testing_raw: Raw fault-free testing data
        df_f_training_raw: Raw faulty training data
        df_f_testing_raw: Raw faulty testing data
        simulation_run: Simulation run to use
        fault_number: Fault type to use
        apply_fault_injection_cutoff: Whether to apply fault injection point cutoff
        
    Returns:
        Dictionary containing all prepared datasets:
        - 'X_incontrol_train': In-control training data
        - 'X_incontrol_test': In-control testing data
        - 'X_faulty_train': Out-of-control training data
        - 'X_faulty_test': Out-of-control testing data
        - 'scaler': Fitted scaler object
    """
    # Prepare in-control data
    X_incontrol_train, X_incontrol_test, scaler = prepare_incontrol_data(
        df_ff_training_raw, df_ff_testing_raw, simulation_run
    )
    
    # Prepare faulty data
    X_faulty_train, X_faulty_test = prepare_faulty_data(
        df_f_training_raw, df_f_testing_raw, fault_number, simulation_run, 
        scaler, apply_fault_injection_cutoff
    )
    
    # Apply fault injection cutoff to in-control data if requested
    if apply_fault_injection_cutoff:
        X_incontrol_train = X_incontrol_train[FAULT_INJECTION_POINT:]
        X_incontrol_test = X_incontrol_test[FAULT_INJECTION_POINT:]
    
    return {
        'X_incontrol_train': X_incontrol_train,
        'X_incontrol_test': X_incontrol_test,
        'X_faulty_train': X_faulty_train,
        'X_faulty_test': X_faulty_test,
        'scaler': scaler
    }


def prepare_batch_experiment_data(df_ff_training_raw: pd.DataFrame,
                                 df_ff_testing_raw: pd.DataFrame,
                                 df_f_training_raw: pd.DataFrame,
                                 df_f_testing_raw: pd.DataFrame,
                                 simulation_runs: range,
                                 fault_numbers: range,
                                 apply_fault_injection_cutoff: bool = True) -> Dict[str, Any]:
    """
    Prepare data for batch experiments across multiple simulation runs and fault types.
    
    Args:
        df_ff_training_raw: Raw fault-free training data
        df_ff_testing_raw: Raw fault-free testing data
        df_f_training_raw: Raw faulty training data
        df_f_testing_raw: Raw faulty testing data
        simulation_runs: Range of simulation runs to process
        fault_numbers: Range of fault numbers to process
        apply_fault_injection_cutoff: Whether to apply fault injection point cutoff
        
    Returns:
        Dictionary containing batch experiment data structure
    """
    # Prepare full in-control training data for model fitting
    X_incontrol_train_full, scaler_full = prepare_full_incontrol_data(df_ff_training_raw)
    
    # Store experiment configurations
    batch_data = {
        'X_incontrol_train_full': X_incontrol_train_full,
        'scaler_full': scaler_full,
        'simulation_runs': list(simulation_runs),
        'fault_numbers': list(fault_numbers),
        'apply_fault_injection_cutoff': apply_fault_injection_cutoff,
        'fault_injection_point': FAULT_INJECTION_POINT,
        'raw_data': {
            'df_ff_training_raw': df_ff_training_raw,
            'df_ff_testing_raw': df_ff_testing_raw,
            'df_f_training_raw': df_f_training_raw,
            'df_f_testing_raw': df_f_testing_raw
        }
    }
    
    return batch_data


def get_experiment_data_for_run_fault(batch_data: Dict[str, Any],
                                     simulation_run: int,
                                     fault_number: int) -> Dict[str, np.ndarray]:
    """
    Extract experiment data for a specific simulation run and fault number from batch data.
    
    Args:
        batch_data: Batch experiment data from prepare_batch_experiment_data
        simulation_run: Simulation run to extract
        fault_number: Fault number to extract
        
    Returns:
        Dictionary containing experiment data for the specific run/fault combination
    """
    raw_data = batch_data['raw_data']
    apply_cutoff = batch_data['apply_fault_injection_cutoff']
    
    return prepare_experiment_data(
        raw_data['df_ff_training_raw'],
        raw_data['df_ff_testing_raw'],
        raw_data['df_f_training_raw'],
        raw_data['df_f_testing_raw'],
        simulation_run,
        fault_number,
        apply_cutoff
    )


def compute_optimal_mcusum_k(X_incontrol_train_full: np.ndarray, 
                            delta_magnitude: float = 0.1) -> float:
    """
    Compute optimal reference value k for MCUSUM detector.
    
    Args:
        X_incontrol_train_full: Full in-control training data
        delta_magnitude: Magnitude of expected shift
        
    Returns:
        Optimal k value
    """
    from ..detectors.mcusum import MCUSUMDetector
    
    n_features = X_incontrol_train_full.shape[1]
    delta = np.ones(n_features) * delta_magnitude
    mu_0, sigma = MCUSUMDetector._estimate_incontrol_parameters(X_incontrol_train_full)
    k = MCUSUMDetector.compute_reference_value_k(delta, sigma)
    
    return k