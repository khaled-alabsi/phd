from typing import Dict, List, Tuple
import pandas as pd
import pyreadr
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


# =========================
# CONSTANTS
# =========================
DATA_PATHS: Dict[str, str] = {
    "fault_free_train": "data/TEP_FaultFree_Training.RData",
    "fault_free_test": "data/TEP_FaultFree_Testing.RData",
    "faulty_train": "data/TEP_Faulty_Training.RData",
    "faulty_test": "data/TEP_Faulty_Testing.RData",
}

TARGET_COLUMN: str = "faultNumber"
RUN_COLUMN: str = "simulationRun"
SAMPLE_COLUMN: str = "sample"
COLUMNS_TO_DROP: List[str] = [RUN_COLUMN, SAMPLE_COLUMN]

SKIPPED_FAULTS: List[int] = []
MERGED_FAULTS: List[int] = [3, 8, 9, 18, 15]
MERGED_FAULT_NUMBER: int = 3
FAULT_START_POINT: int = 25


# =========================
# DATA LOADING
# =========================
def load_rdata_to_df(path: str, key: str) -> pd.DataFrame:
    """Read an .RData file and extract the DataFrame by key."""
    return pyreadr.read_r(path)[key]


def load_all_data() -> Dict[str, pd.DataFrame]:
    """Load all TEP datasets into a dictionary."""
    return {
        "fault_free_train": load_rdata_to_df(DATA_PATHS["fault_free_train"], "fault_free_training"),
        "fault_free_test": load_rdata_to_df(DATA_PATHS["fault_free_test"], "fault_free_testing"),
        "faulty_train": load_rdata_to_df(DATA_PATHS["faulty_train"], "faulty_training"),
        "faulty_test": load_rdata_to_df(DATA_PATHS["faulty_test"], "faulty_testing"),
    }


# =========================
# DATA FILTERING
# =========================
def filter_skipped_faults(df: pd.DataFrame, skipped_faults: List[int]) -> pd.DataFrame:
    """Remove rows with fault numbers that should be skipped."""
    return df[~df[TARGET_COLUMN].isin(skipped_faults)].reset_index(drop=True)


def reduce_data(df: pd.DataFrame, run_range: Tuple[int, int], remove_columns: List[str], fault_only: bool = False) -> pd.DataFrame:
    """Reduce dataset for development, maintaining class balance."""
    condition = (df[RUN_COLUMN] > run_range[0]) & (df[RUN_COLUMN] < run_range[1])
    if fault_only:
        condition &= df[SAMPLE_COLUMN] > FAULT_START_POINT
    return df.loc[condition].drop(columns=remove_columns, axis=1)


# =========================
# ENCODING & SCALING
# =========================
def scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on training data and transform both train and test sets."""
    scaler = StandardScaler()
    scaler.fit(train_df)
    return scaler.transform(train_df), scaler.transform(test_df), scaler


def encode_labels(train_y: pd.Series, test_y: pd.Series) -> Tuple[List[int], List[int], LabelEncoder]:
    """Encode categorical fault labels into integers."""
    encoder = LabelEncoder()
    train_encoded = encoder.fit_transform(train_y)
    test_encoded = encoder.transform(test_y)
    return train_encoded, test_encoded, encoder


def one_hot_encode(train_y: pd.Series, test_y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """One-hot encode target variable for supervised models."""
    encoder = OneHotEncoder(sparse_output=False)
    train_encoded = encoder.fit_transform(train_y.to_numpy().reshape(-1, 1))
    test_encoded = encoder.transform(test_y.to_numpy().reshape(-1, 1))
    return train_encoded, test_encoded, encoder


# =========================
# MAIN PIPELINE
# =========================
data = load_all_data()

fault_free_train = data["fault_free_train"]
fault_free_test = data["fault_free_test"]
faulty_train = filter_skipped_faults(data["faulty_train"], SKIPPED_FAULTS)
faulty_test = filter_skipped_faults(data["faulty_test"], SKIPPED_FAULTS)

# Configuration for simulation runs
SIMULATION_RUN_CONFIG = {
    "incontrol_train": (0, 2),   # Run 1
    "incontrol_test": (1, 3),    # Run 2
    "faulty_train": (0, 2),      # Run 1
    "faulty_test": (1, 3),       # Run 2
}

# Fault to use for basic out-of-control test data
DEFAULT_TEST_FAULT = 1

# Reduced subsets for development
ff_train_reduced = reduce_data(fault_free_train, SIMULATION_RUN_CONFIG["incontrol_train"], COLUMNS_TO_DROP)
ff_test_reduced = reduce_data(fault_free_test, SIMULATION_RUN_CONFIG["incontrol_test"], COLUMNS_TO_DROP)

# For supervised learning - use fault_only=True to get only faulty samples
f_train_reduced = reduce_data(faulty_train, SIMULATION_RUN_CONFIG["faulty_train"], COLUMNS_TO_DROP, fault_only=True)
f_test_reduced_supervised = reduce_data(faulty_test, SIMULATION_RUN_CONFIG["faulty_test"], COLUMNS_TO_DROP, fault_only=True)

# For supervised combined dataset - single fault, fault samples only
f_test_reduced = f_test_reduced_supervised[f_test_reduced_supervised[TARGET_COLUMN] == DEFAULT_TEST_FAULT].reset_index(drop=True)

# Combine for supervised learning
train_combined = pd.concat([ff_train_reduced, f_train_reduced], ignore_index=True)
test_combined = pd.concat([ff_test_reduced, f_test_reduced], ignore_index=True)

# Standardization
X_train_scaled, X_test_scaled, scaler = scale_features(
    train_combined.drop(columns=[TARGET_COLUMN]),
    test_combined.drop(columns=[TARGET_COLUMN]),
)
y_train, y_test, label_encoder = encode_labels(
    train_combined[TARGET_COLUMN], test_combined[TARGET_COLUMN]
)
y_train_oh, y_test_oh, one_hot_encoder = one_hot_encode(
    train_combined[TARGET_COLUMN], test_combined[TARGET_COLUMN]
)

# =========================
# UNSUPERVISED VARIANTS
# =========================
def binary_fault_labels(df: pd.Series) -> pd.Series:
    """Convert fault numbers to binary labels: 0 = normal, 1 = faulty."""
    return df.apply(lambda x: 0 if x == 0 else 1)


# For MCUSUM and other unsupervised methods, align sample counts
# Both in-control and faulty data need to start from FAULT_START_POINT
ff_train_aligned = fault_free_train[
    (fault_free_train[RUN_COLUMN] > SIMULATION_RUN_CONFIG["incontrol_train"][0]) &
    (fault_free_train[RUN_COLUMN] < SIMULATION_RUN_CONFIG["incontrol_train"][1]) &
    (fault_free_train[SAMPLE_COLUMN] > FAULT_START_POINT)
].drop(columns=COLUMNS_TO_DROP)

ff_test_aligned = fault_free_test[
    (fault_free_test[RUN_COLUMN] > SIMULATION_RUN_CONFIG["incontrol_test"][0]) &
    (fault_free_test[RUN_COLUMN] < SIMULATION_RUN_CONFIG["incontrol_test"][1]) &
    (fault_free_test[SAMPLE_COLUMN] > FAULT_START_POINT)
].drop(columns=COLUMNS_TO_DROP)

# Get aligned faulty test data - filter for single fault and same sample range
f_test_aligned = faulty_test[
    (faulty_test[RUN_COLUMN] > SIMULATION_RUN_CONFIG["faulty_test"][0]) &
    (faulty_test[RUN_COLUMN] < SIMULATION_RUN_CONFIG["faulty_test"][1]) &
    (faulty_test[SAMPLE_COLUMN] > FAULT_START_POINT) &
    (faulty_test[TARGET_COLUMN] == DEFAULT_TEST_FAULT)
].drop(columns=COLUMNS_TO_DROP)

X_incontrol_train, X_incontrol_test, sc_incontrol = scale_features(
    ff_train_aligned.drop(columns=[TARGET_COLUMN]),
    ff_test_aligned.drop(columns=[TARGET_COLUMN]),
)
X_out_of_control_test = sc_incontrol.transform(
    f_test_aligned.drop(columns=[TARGET_COLUMN])
)

y_train_binary = binary_fault_labels(train_combined[TARGET_COLUMN])
y_test_binary = binary_fault_labels(test_combined[TARGET_COLUMN])
y_train_binary_oh, y_test_binary_oh, encoder_binary = one_hot_encode(y_train_binary, y_test_binary)


# =========================
# SIMULATION RUN HELPERS
# =========================

# Single run functions
def get_incontrol_train_run(run_number: int):
    """Get a specific in-control training run."""
    run_data = fault_free_train[fault_free_train[RUN_COLUMN] == run_number].drop(columns=COLUMNS_TO_DROP)
    X_raw = run_data.drop(columns=[TARGET_COLUMN]) if TARGET_COLUMN in run_data.columns else run_data
    X_scaled = sc_incontrol.transform(X_raw)
    return X_scaled, run_data


def get_incontrol_test_run(run_number: int):
    """Get a specific in-control test run."""
    run_data = fault_free_test[fault_free_test[RUN_COLUMN] == run_number].drop(columns=COLUMNS_TO_DROP)
    X_raw = run_data.drop(columns=[TARGET_COLUMN]) if TARGET_COLUMN in run_data.columns else run_data
    X_scaled = sc_incontrol.transform(X_raw)
    return X_scaled, run_data


def get_faulty_train_run(run_number: int):
    """Get a specific faulty training run (fault-only samples)."""
    base_df = faulty_train[faulty_train[RUN_COLUMN] == run_number]
    run_data = base_df[base_df[SAMPLE_COLUMN] > FAULT_START_POINT].drop(columns=COLUMNS_TO_DROP)
    X_raw = run_data.drop(columns=[TARGET_COLUMN]) if TARGET_COLUMN in run_data.columns else run_data
    X_scaled = sc_incontrol.transform(X_raw)
    return X_scaled, run_data


def get_faulty_test_run(run_number: int):
    """Get a specific faulty test run (fault-only samples)."""
    base_df = faulty_test[faulty_test[RUN_COLUMN] == run_number]
    run_data = base_df[base_df[SAMPLE_COLUMN] > FAULT_START_POINT].drop(columns=COLUMNS_TO_DROP)
    X_raw = run_data.drop(columns=[TARGET_COLUMN]) if TARGET_COLUMN in run_data.columns else run_data
    X_scaled = sc_incontrol.transform(X_raw)
    return X_scaled, run_data


# All runs functions
def get_incontrol_train_runs(max_runs=None):
    """
    Get all in-control training runs.

    Args:
        max_runs: Optional limit on number of runs to return

    Returns:
        List of tuples: (run_number, scaled_data, raw_data)
    """
    run_numbers = sorted(fault_free_train[RUN_COLUMN].unique())
    if max_runs:
        run_numbers = run_numbers[:max_runs]

    results = []
    for run in run_numbers:
        scaled, raw = get_incontrol_train_run(run)
        results.append((run, scaled, raw))
    return results


def get_incontrol_test_runs(max_runs=None):
    """
    Get all in-control test runs.

    Args:
        max_runs: Optional limit on number of runs to return

    Returns:
        List of tuples: (run_number, scaled_data, raw_data)
    """
    run_numbers = sorted(fault_free_test[RUN_COLUMN].unique())
    if max_runs:
        run_numbers = run_numbers[:max_runs]

    results = []
    for run in run_numbers:
        scaled, raw = get_incontrol_test_run(run)
        results.append((run, scaled, raw))
    return results


def get_faulty_train_runs(max_runs=None):
    """
    Get all faulty training runs.

    Args:
        max_runs: Optional limit on number of runs to return

    Returns:
        List of tuples: (run_number, scaled_data, raw_data)
    """
    run_numbers = sorted(faulty_train[RUN_COLUMN].unique())
    if max_runs:
        run_numbers = run_numbers[:max_runs]

    results = []
    for run in run_numbers:
        scaled, raw = get_faulty_train_run(run)
        results.append((run, scaled, raw))
    return results


def get_faulty_test_runs(max_runs=None, fault_number=None):
    """
    Get all faulty test runs, optionally filtered by fault number.

    Args:
        max_runs: Optional limit on number of runs to return
        fault_number: Optional specific fault number to filter (default: all faults)

    Returns:
        List of tuples: (run_number, scaled_data, raw_data)
    """
    # Filter by fault number if specified
    if fault_number is not None:
        base_df = faulty_test[faulty_test[TARGET_COLUMN] == fault_number]
    else:
        base_df = faulty_test

    run_numbers = sorted(base_df[RUN_COLUMN].unique())
    if max_runs:
        run_numbers = run_numbers[:max_runs]

    results = []
    for run in run_numbers:
        # Filter for specific run and fault
        run_data = base_df[base_df[RUN_COLUMN] == run]
        run_data = run_data[run_data[SAMPLE_COLUMN] > FAULT_START_POINT].drop(columns=COLUMNS_TO_DROP)

        X_raw = run_data.drop(columns=[TARGET_COLUMN]) if TARGET_COLUMN in run_data.columns else run_data
        X_scaled = sc_incontrol.transform(X_raw)

        results.append((run, X_scaled, run_data))
    return results


def get_available_fault_numbers():
    """Get list of available fault numbers in the dataset."""
    return sorted(faulty_test[TARGET_COLUMN].unique())


def set_simulation_run_config(
    incontrol_train_range = None,
    incontrol_test_range = None,
    faulty_train_range = None,
    faulty_test_range = None,
    default_test_fault = None
):
    """
    Update the simulation run configuration and regenerate datasets.

    Args:
        incontrol_train_range: Range (min, max) for in-control training runs
        incontrol_test_range: Range (min, max) for in-control test runs
        faulty_train_range: Range (min, max) for faulty training runs
        faulty_test_range: Range (min, max) for faulty test runs
        default_test_fault: Fault number to use for X_out_of_control_test
    """
    global SIMULATION_RUN_CONFIG, X_incontrol_train, X_incontrol_test, X_out_of_control_test
    global ff_train_reduced, ff_test_reduced, f_train_reduced, f_test_reduced
    global sc_incontrol, DEFAULT_TEST_FAULT

    if incontrol_train_range:
        SIMULATION_RUN_CONFIG["incontrol_train"] = incontrol_train_range
    if incontrol_test_range:
        SIMULATION_RUN_CONFIG["incontrol_test"] = incontrol_test_range
    if faulty_train_range:
        SIMULATION_RUN_CONFIG["faulty_train"] = faulty_train_range
    if faulty_test_range:
        SIMULATION_RUN_CONFIG["faulty_test"] = faulty_test_range
    if default_test_fault is not None:
        DEFAULT_TEST_FAULT = default_test_fault

    # Regenerate reduced datasets
    ff_train_reduced = reduce_data(fault_free_train, SIMULATION_RUN_CONFIG["incontrol_train"], COLUMNS_TO_DROP)
    ff_test_reduced = reduce_data(fault_free_test, SIMULATION_RUN_CONFIG["incontrol_test"], COLUMNS_TO_DROP)
    f_train_reduced = reduce_data(faulty_train, SIMULATION_RUN_CONFIG["faulty_train"], COLUMNS_TO_DROP, fault_only=True)
    f_test_reduced_supervised = reduce_data(faulty_test, SIMULATION_RUN_CONFIG["faulty_test"], COLUMNS_TO_DROP, fault_only=True)
    f_test_reduced = f_test_reduced_supervised[f_test_reduced_supervised[TARGET_COLUMN] == DEFAULT_TEST_FAULT].reset_index(drop=True)

    # For unsupervised methods, align sample counts by filtering from FAULT_START_POINT
    ff_train_aligned = fault_free_train[
        (fault_free_train[RUN_COLUMN] > SIMULATION_RUN_CONFIG["incontrol_train"][0]) &
        (fault_free_train[RUN_COLUMN] < SIMULATION_RUN_CONFIG["incontrol_train"][1]) &
        (fault_free_train[SAMPLE_COLUMN] > FAULT_START_POINT)
    ].drop(columns=COLUMNS_TO_DROP)

    ff_test_aligned = fault_free_test[
        (fault_free_test[RUN_COLUMN] > SIMULATION_RUN_CONFIG["incontrol_test"][0]) &
        (fault_free_test[RUN_COLUMN] < SIMULATION_RUN_CONFIG["incontrol_test"][1]) &
        (fault_free_test[SAMPLE_COLUMN] > FAULT_START_POINT)
    ].drop(columns=COLUMNS_TO_DROP)

    # Faulty test data filtered to same fault and same sample range
    f_test_aligned = faulty_test[
        (faulty_test[RUN_COLUMN] > SIMULATION_RUN_CONFIG["faulty_test"][0]) &
        (faulty_test[RUN_COLUMN] < SIMULATION_RUN_CONFIG["faulty_test"][1]) &
        (faulty_test[SAMPLE_COLUMN] > FAULT_START_POINT) &
        (faulty_test[TARGET_COLUMN] == DEFAULT_TEST_FAULT)
    ].drop(columns=COLUMNS_TO_DROP)

    # Regenerate scaled datasets with aligned data
    X_incontrol_train, X_incontrol_test, sc_incontrol = scale_features(
        ff_train_aligned.drop(columns=[TARGET_COLUMN]),
        ff_test_aligned.drop(columns=[TARGET_COLUMN]),
    )
    X_out_of_control_test = sc_incontrol.transform(
        f_test_aligned.drop(columns=[TARGET_COLUMN])
    )

    print(f"Updated simulation run config: {SIMULATION_RUN_CONFIG}")
    print(f"Using fault {DEFAULT_TEST_FAULT} for X_out_of_control_test")
    print(f"Shapes - X_incontrol_test: {X_incontrol_test.shape}, X_out_of_control_test: {X_out_of_control_test.shape}")


print("Training Fault Numbers:", train_combined[TARGET_COLUMN].unique())
print(f"\nData shapes (aligned from sample {FAULT_START_POINT+1}):")
print(f"  X_incontrol_train: {X_incontrol_train.shape}")
print(f"  X_incontrol_test: {X_incontrol_test.shape}")
print(f"  X_out_of_control_test: {X_out_of_control_test.shape}")
print(f"  Using fault {DEFAULT_TEST_FAULT} for X_out_of_control_test")
print(f"  Shapes should match: {X_incontrol_test.shape[0]} == {X_out_of_control_test.shape[0]}")
