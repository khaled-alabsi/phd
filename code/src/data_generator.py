"""Tennessee Eastman data generation and loading utilities."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd


def generate_synthetic_datasets(
    n_simulation_runs: int,
    n_faults: int,
    n_samples: int,
    n_features: int,
    random_state: int = 0,
    feature_prefix: str = "feature",
    fault_column_name: str = "faultNumber",
    simulation_column_name: str = "simulationRun",
    fault_injection_point: int = 160,
) -> Dict[str, pd.DataFrame]:
    """
    Create lightweight synthetic datasets with Tennessee Eastman-like structure.

    Args:
        n_simulation_runs: Number of simulation runs to generate
        n_faults: Number of different fault types
        n_samples: Number of samples per simulation
        n_features: Number of features (process variables)
        random_state: Random seed for reproducibility
        feature_prefix: Prefix for feature column names
        fault_column_name: Name of the fault number column
        simulation_column_name: Name of the simulation run column
        fault_injection_point: Sample index where fault is injected

    Returns:
        Dictionary with keys:
            - "fault_free_training": Fault-free training data
            - "fault_free_testing": Fault-free testing data
            - "faulty_training": Faulty training data
            - "faulty_testing": Faulty testing data
    """
    rng = np.random.default_rng(random_state)
    feature_cols = [f"{feature_prefix}_{i:02d}" for i in range(1, n_features + 1)]
    sample_index = np.arange(n_samples)

    def build_fault_free() -> pd.DataFrame:
        """Build fault-free dataset with multiple simulation runs."""
        frames = []
        for sim in range(1, n_simulation_runs + 1):
            data = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
            df = pd.DataFrame(data, columns=feature_cols)
            df[fault_column_name] = 0
            df[simulation_column_name] = sim
            df["sample"] = sample_index
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def build_faulty() -> pd.DataFrame:
        """Build faulty dataset with multiple fault types and simulation runs."""
        frames = []
        shift_start = min(fault_injection_point, n_samples)
        for fault in range(1, n_faults + 1):
            for sim in range(1, n_simulation_runs + 1):
                data = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
                if shift_start < n_samples:
                    # Introduce fault as a mean shift after fault_injection_point
                    shift = rng.normal(loc=fault * 0.3, scale=0.05, size=n_features)
                    data[shift_start:] += shift
                df = pd.DataFrame(data, columns=feature_cols)
                df[fault_column_name] = fault
                df[simulation_column_name] = sim
                df["sample"] = sample_index
                frames.append(df)
        return pd.concat(frames, ignore_index=True)

    return {
        "fault_free_training": build_fault_free(),
        "fault_free_testing": build_fault_free(),
        "faulty_training": build_faulty(),
        "faulty_testing": build_faulty(),
    }


def load_datasets(
    data_source: Literal["real", "synthetic"] = "synthetic",
    synthetic_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load Tennessee Eastman datasets from disk or generate synthetic stand-ins.

    Args:
        data_source: Either "real" to load from RData files or "synthetic" to generate
        synthetic_config: Configuration dict for synthetic data generation.
            Expected keys:
                - n_simulation_runs: int
                - n_faults: int
                - n_samples: int
                - n_features: int
                - random_state: int
                - feature_prefix: str

    Returns:
        Dictionary with keys:
            - "fault_free_training"
            - "fault_free_testing"
            - "faulty_training"
            - "faulty_testing"

    Raises:
        ImportError: If data_source is "real" but pyreadr is not installed
        ValueError: If data_source is not recognized
    """
    if data_source == "real":
        try:
            import pyreadr
        except ImportError as exc:
            raise ImportError(
                "pyreadr is required to load the Tennessee Eastman data from disk. "
                "Install it or switch data_source to 'synthetic'."
            ) from exc

        fault_free_training_dict = pyreadr.read_r("data/TEP_FaultFree_Training.RData")
        fault_free_testing_dict = pyreadr.read_r("data/TEP_FaultFree_Testing.RData")
        faulty_training_dict = pyreadr.read_r("data/TEP_Faulty_Training.RData")
        faulty_testing_dict = pyreadr.read_r("data/TEP_Faulty_Testing.RData")

        return {
            "fault_free_training": fault_free_training_dict["fault_free_training"],
            "fault_free_testing": fault_free_testing_dict["fault_free_testing"],
            "faulty_training": faulty_training_dict["faulty_training"],
            "faulty_testing": faulty_testing_dict["faulty_testing"],
        }

    if data_source == "synthetic":
        # Default configuration
        default_config: Dict[str, Any] = {
            "n_simulation_runs": 5,
            "n_faults": 5,
            "n_samples": 300,
            "n_features": 52,
            "random_state": 42,
            "feature_prefix": "feature",
        }
        if synthetic_config:
            default_config.update(synthetic_config)
        return generate_synthetic_datasets(**default_config)

    raise ValueError(f"Unknown data_source '{data_source}'. Expected 'real' or 'synthetic'.")
