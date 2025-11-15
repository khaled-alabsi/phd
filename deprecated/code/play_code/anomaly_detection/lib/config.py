"""
Configuration Module

This module contains configuration constants, model registry, and factory functions
for the anomaly detection library.
"""

import numpy as np
from typing import Dict, Callable, Any, Optional, Tuple
from .detectors.mcusum import MCUSUMDetector, mcusum_predict
from .detectors.autoencoder import AutoencoderDetector, AutoencoderDetectorEnhanced
from .detectors.ewma import (StandardMEWMA, MEWMS, MEWMV, REWMV, MaxMEWMV, MNSE,
                           AEWMA, SAMEWMA, AMFEWMA)

# Data preprocessing constants
TARGET_VARIABLE_COLUMN_NAME = "faultNumber"
SIMULATION_RUN_COLUMN_NAME = "simulationRun"
COLUMNS_TO_REMOVE = ["faultNumber", "simulationRun", "sample"]
FAULT_INJECTION_POINT = 160

# Experiment configuration
DEFAULT_SIMULATION_RUNS = range(1, 21)
DEFAULT_FAULT_NUMBERS = range(1, 21)

# Model hyperparameters
MCUSUM_CONFIG = {
    'default_k': 0.5,
    'default_h': None,
    'delta_magnitude': 0.1
}

AUTOENCODER_CONFIG = {
    'encoding_dim': 8,
    'epochs': 50,
    'batch_size': 32,
    'threshold_percentile': 95
}

AUTOENCODER_ENHANCED_CONFIG = {
    'encoding_dim': 8,
    'epochs': 50,
    'batch_size': 32,
    'threshold_percentile': 95
}

EWMA_CONFIG = {
    'lambda_': 0.2,
    'percentile': 99.5
}

# Global model instances (to be initialized)
_model_instances = {}


def initialize_models(X_incontrol_train_full: np.ndarray) -> Dict[str, Any]:
    """
    Initialize all anomaly detection models with training data.
    
    Args:
        X_incontrol_train_full: Full in-control training data
        
    Returns:
        Dictionary of initialized model instances
    """
    global _model_instances
    
    # Initialize MCUSUM
    from .data.preprocessing import compute_optimal_mcusum_k
    optimal_k = compute_optimal_mcusum_k(X_incontrol_train_full)
    mcusum = MCUSUMDetector(k=optimal_k)
    mcusum.fit(X_incontrol_train_full)
    
    # Initialize Autoencoders
    autoencoder = AutoencoderDetector(encoding_dim=AUTOENCODER_CONFIG['encoding_dim'])
    autoencoder.fit(X_incontrol_train_full, 
                   epochs=AUTOENCODER_CONFIG['epochs'],
                   batch_size=AUTOENCODER_CONFIG['batch_size'],
                   threshold_percentile=AUTOENCODER_CONFIG['threshold_percentile'])
    
    autoencoder_enhanced = AutoencoderDetectorEnhanced(encoding_dim=AUTOENCODER_ENHANCED_CONFIG['encoding_dim'])
    autoencoder_enhanced.fit(X_incontrol_train_full,
                           epochs=AUTOENCODER_ENHANCED_CONFIG['epochs'],
                           batch_size=AUTOENCODER_ENHANCED_CONFIG['batch_size'],
                           threshold_percentile=AUTOENCODER_ENHANCED_CONFIG['threshold_percentile'])
    
    # Initialize EWMA-based detectors
    mewma = StandardMEWMA(**EWMA_CONFIG)
    mewma.fit(X_incontrol_train_full)
    
    mewms = MEWMS(**EWMA_CONFIG)
    mewms.fit(X_incontrol_train_full)
    
    mewmv = MEWMV(**EWMA_CONFIG)
    mewmv.fit(X_incontrol_train_full)
    
    rewmv = REWMV(**EWMA_CONFIG)
    rewmv.fit(X_incontrol_train_full)
    
    max_mewmv = MaxMEWMV()
    max_mewmv.fit(X_incontrol_train_full)
    
    mnse = MNSE(**EWMA_CONFIG)
    mnse.fit(X_incontrol_train_full)
    
    _model_instances = {
        'mcusum': mcusum,
        'autoencoder': autoencoder,
        'autoencoder_enhanced': autoencoder_enhanced,
        'mewma': mewma,
        'mewms': mewms,
        'mewmv': mewmv,
        'rewmv': rewmv,
        'max_mewmv': max_mewmv,
        'mnse': mnse
    }
    
    return _model_instances


def get_model_instance(model_name: str) -> Any:
    """
    Get a model instance by name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model instance
        
    Raises:
        ValueError: If model not found or not initialized
    """
    if model_name not in _model_instances:
        raise ValueError(f"Model '{model_name}' not found or not initialized. "
                        "Call initialize_models() first.")
    return _model_instances[model_name]


def create_detector_registry() -> Dict[str, Callable]:
    """
    Create a registry of detector prediction functions.
    
    Returns:
        Dictionary mapping detector names to prediction functions
    """
    def mcusum_wrapper(X):
        return mcusum_predict(X, get_model_instance('mcusum'))
    
    def autoencoder_wrapper(X):
        return get_model_instance('autoencoder').predict(X)
    
    def autoencoder_enhanced_wrapper(X):
        return get_model_instance('autoencoder_enhanced').predict(X)
    
    def mewma_wrapper(X):
        return get_model_instance('mewma').predict(X)
    
    def mewms_wrapper(X):
        return get_model_instance('mewms').predict(X)
    
    def mewmv_wrapper(X):
        return get_model_instance('mewmv').predict(X)
    
    def rewmv_wrapper(X):
        return get_model_instance('rewmv').predict(X)
    
    def max_mewmv_wrapper(X):
        return get_model_instance('max_mewmv').predict(X)
    
    def mnse_wrapper(X):
        return get_model_instance('mnse').predict(X)
    
    return {
        "MCUSUM": mcusum_wrapper,
        "Autoencoder": autoencoder_wrapper,
        "AutoencoderEnhanced": autoencoder_enhanced_wrapper,
        "MEWMA": mewma_wrapper,
        "MEWMS": mewms_wrapper,
        "MEWMV": mewmv_wrapper,
        "REWMV": rewmv_wrapper,
        "MaxMEWMV": max_mewmv_wrapper,
        "MNSE": mnse_wrapper
    }


def create_minimal_detector_registry() -> Dict[str, Callable]:
    """
    Create a minimal registry with key detectors for faster experiments.
    
    Returns:
        Dictionary mapping detector names to prediction functions
    """
    full_registry = create_detector_registry()
    return {
        "MCUSUM": full_registry["MCUSUM"],
        "AutoencoderEnhanced": full_registry["AutoencoderEnhanced"],
        "MEWMA": full_registry["MEWMA"],
        "MEWMS": full_registry["MEWMS"]
    }


class ExperimentConfig:
    """Configuration class for anomaly detection experiments."""
    
    def __init__(self,
                 simulation_runs: range = DEFAULT_SIMULATION_RUNS,
                 fault_numbers: range = DEFAULT_FAULT_NUMBERS,
                 apply_fault_injection_cutoff: bool = True,
                 use_minimal_registry: bool = False,
                 verbose: bool = True):
        """
        Initialize experiment configuration.
        
        Args:
            simulation_runs: Range of simulation runs to process
            fault_numbers: Range of fault numbers to process
            apply_fault_injection_cutoff: Whether to apply fault injection cutoff
            use_minimal_registry: Whether to use minimal detector registry
            verbose: Whether to enable verbose output
        """
        self.simulation_runs = simulation_runs
        self.fault_numbers = fault_numbers
        self.apply_fault_injection_cutoff = apply_fault_injection_cutoff
        self.use_minimal_registry = use_minimal_registry
        self.verbose = verbose
    
    def get_detector_registry(self) -> Dict[str, Callable]:
        """Get the appropriate detector registry based on configuration."""
        if self.use_minimal_registry:
            return create_minimal_detector_registry()
        else:
            return create_detector_registry()


def setup_experiment(X_incontrol_train_full: np.ndarray,
                    config: Optional[ExperimentConfig] = None) -> Tuple[Dict[str, Any], Dict[str, Callable]]:
    """
    Set up everything needed for running experiments.
    
    Args:
        X_incontrol_train_full: Full in-control training data
        config: Experiment configuration (uses defaults if None)
        
    Returns:
        Tuple of (model_instances, detector_registry)
    """
    if config is None:
        config = ExperimentConfig()
    
    # Initialize models
    model_instances = initialize_models(X_incontrol_train_full)
    
    # Create detector registry
    detector_registry = config.get_detector_registry()
    
    if config.verbose:
        print(f"Initialized {len(model_instances)} models")
        print(f"Created detector registry with {len(detector_registry)} detectors:")
        for name in detector_registry.keys():
            print(f"  - {name}")
    
    return model_instances, detector_registry


# Convenience functions for quick access
def get_all_detector_names() -> list:
    """Get list of all available detector names."""
    return list(create_detector_registry().keys())


def get_minimal_detector_names() -> list:
    """Get list of minimal detector names."""
    return list(create_minimal_detector_registry().keys())


def print_configuration():
    """Print current configuration settings."""
    print("=" * 60)
    print("ANOMALY DETECTION LIBRARY CONFIGURATION")
    print("=" * 60)
    print(f"Fault injection point: {FAULT_INJECTION_POINT}")
    print(f"Default simulation runs: {list(DEFAULT_SIMULATION_RUNS)}")
    print(f"Default fault numbers: {list(DEFAULT_FAULT_NUMBERS)}")
    print()
    print("Model configurations:")
    print(f"  MCUSUM: {MCUSUM_CONFIG}")
    print(f"  Autoencoder: {AUTOENCODER_CONFIG}")
    print(f"  EWMA: {EWMA_CONFIG}")
    print()
    print(f"Available detectors: {get_all_detector_names()}")
    print(f"Minimal detectors: {get_minimal_detector_names()}")
    print("=" * 60)