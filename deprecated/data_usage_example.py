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

# Reduced subsets for development
ff_train_reduced = reduce_data(fault_free_train, (0, 2), COLUMNS_TO_DROP)
ff_test_reduced = reduce_data(fault_free_test, (2, 4), COLUMNS_TO_DROP)
f_train_reduced = reduce_data(faulty_train, (4, 6), COLUMNS_TO_DROP, fault_only=True)
f_test_reduced = reduce_data(faulty_test, (5, 7), COLUMNS_TO_DROP, fault_only=True)

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


X_incontrol_train, _, sc_in_train = scale_features(
    ff_train_reduced.drop(columns=[TARGET_COLUMN]),
    ff_test_reduced.drop(columns=[TARGET_COLUMN]),
)
X_out_of_control_test, _, sc_out_test = scale_features(
    f_test_reduced.drop(columns=[TARGET_COLUMN]),
    f_test_reduced.drop(columns=[TARGET_COLUMN]),
)

y_train_binary = binary_fault_labels(train_combined[TARGET_COLUMN])
y_test_binary = binary_fault_labels(test_combined[TARGET_COLUMN])
y_train_binary_oh, y_test_binary_oh, encoder_binary = one_hot_encode(y_train_binary, y_test_binary)

print("Training Fault Numbers:", train_combined[TARGET_COLUMN].unique())
