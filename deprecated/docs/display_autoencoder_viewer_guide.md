# display_autoencoder_viewer Parameter Guide

## The Two Modes

The `display_autoencoder_viewer` function has **TWO modes** for passing data:

---

## Mode 1: `datasets` (Pre-processed Mode)

**Use when:** You have already filtered and scaled your data

### Data Format
- **Type:** `Dict[str, np.ndarray]`
- **Contents:** Already scaled numpy arrays (no metadata columns)
- **Shape:** Each array is `(n_samples, n_features)`

### What You Provide
```python
datasets = {
    "Train": X_train_scaled,              # (1500, 52) - already scaled
    "Test": X_test_scaled,                # (300, 52) - already scaled
    "In-control": X_incontrol_scaled,     # (140, 52) - already scaled
}
```

### Example Usage
```python
display_autoencoder_viewer(
    predict_fn=lambda X: model.predict(X, verbose=0),
    datasets=AE_DATASETS,         # ✅ Pre-scaled numpy arrays
    feature_names=feature_labels,
)
```

### When to Use
- ✅ You've already filtered by fault/simulation
- ✅ Data is already scaled
- ✅ You want a simple viewer with no dynamic filtering
- ✅ You're comparing different pre-processed datasets

---

## Mode 2: `fault_data` (Raw Data with Dynamic Filtering)

**Use when:** You want to dynamically select faults and simulations using dropdown widgets

### Data Format
- **Type:** `Dict[str, pd.DataFrame]`
- **Contents:** Raw DataFrames with metadata columns intact
- **Columns:** Must include `faultNumber`, `simulationRun`, plus feature columns

### What You Provide
```python
fault_data = {
    "Fault-free test": DF_FF_TEST_RAW,    # DataFrame with all columns
    "Faulty test": DF_F_TEST_RAW,         # DataFrame with all columns
}
```

**DataFrame Structure:**
```
   feature_01  feature_02  ...  feature_52  faultNumber  simulationRun  sample
0    0.304717   -1.039984  ...    0.631288            0              1       0
1   -1.457156   -0.319671  ...    0.737516            0              1       1
2   -0.933618   -0.205438  ...    1.767930            2              3       2
...
```

### Required Additional Parameters
When using `fault_data`, you must also provide:

| Parameter | Type | Purpose | Example |
|-----------|------|---------|---------|
| `scaler` | StandardScaler | Scale data on-the-fly | `scaler_incontrol` |
| `fault_column_name` | str | Name of fault column | `"faultNumber"` |
| `simulation_column_name` | str | Name of simulation column | `"simulationRun"` |
| `columns_to_remove` | list[str] | Metadata columns to drop | `["faultNumber", "simulationRun", "sample"]` |
| `default_fault` | int | Initial fault to display | `2` |
| `default_simulation` | int | Initial simulation to display | `1` |

### Example Usage
```python
display_autoencoder_viewer(
    predict_fn=lambda X: model.predict(X, verbose=0),
    datasets={},                          # ✅ Empty! Using fault_data instead
    fault_data=FAULT_DATA_RAW,            # ✅ Raw DataFrames
    scaler=scaler_incontrol,              # ✅ Required for scaling
    fault_column_name="faultNumber",      # ✅ Metadata column names
    simulation_column_name="simulationRun",
    columns_to_remove=["faultNumber", "simulationRun", "sample"],
    default_fault=2,                      # ✅ Initial selections
    default_simulation=1,
    feature_names=feature_labels,
)
```

### What Happens Behind the Scenes
1. User selects fault `2` and simulation `1` from dropdowns
2. Function filters: `DF_F_TEST_RAW.query("faultNumber == 2 and simulationRun == 1")`
3. Function removes metadata columns: `.drop(columns=["faultNumber", "simulationRun", "sample"])`
4. Function scales data: `scaler.transform(filtered_df)`
5. Function passes scaled data to `predict_fn`

### When to Use
- ✅ You want interactive fault/simulation selection
- ✅ You have raw DataFrames with metadata
- ✅ You want users to explore different scenarios
- ✅ You haven't pre-filtered the data

---

## Visual Comparison

### Mode 1 (datasets)
```
┌─────────────────────────────────┐
│ Dataset: [In-control ▼]        │  ← Static dropdown (your pre-made datasets)
├─────────────────────────────────┤
│ Features: [f0, f1, f2 ...]     │
│ Plot: [actual vs reconstructed]│
└─────────────────────────────────┘
```

### Mode 2 (fault_data)
```
┌─────────────────────────────────┐
│ Dataset: [Faulty test ▼]       │  ← Choose dataset
│ Fault: [2 ▼]                   │  ← Dynamic! Choose fault number
│ Simulation: [1 ▼]              │  ← Dynamic! Choose simulation run
├─────────────────────────────────┤
│ Features: [f0, f1, f2 ...]     │
│ Plot: [actual vs reconstructed]│
└─────────────────────────────────┘
```

---

## Key Differences

| Aspect | `datasets` Mode | `fault_data` Mode |
|--------|-----------------|-------------------|
| **Data type** | `Dict[str, np.ndarray]` | `Dict[str, pd.DataFrame]` |
| **Pre-processing** | Already done by you | Done by viewer |
| **Scaling** | Already scaled | Uses `scaler` parameter |
| **Filtering** | Already filtered | Uses fault/simulation dropdowns |
| **Metadata columns** | None (removed) | Present in DataFrames |
| **Interactive filtering** | ❌ No | ✅ Yes |
| **Dropdown widgets** | 1 (dataset only) | 3 (dataset, fault, simulation) |

---

## Common Patterns

### Pattern 1: Simple Comparison (Use `datasets`)
```python
# Compare different processing stages
AE_DATASETS = {
    "Train (full)": X_INCONTROL_TRAIN_FULL_SCALED,
    "Validation (full)": X_INCONTROL_TEST_REDUCED_SCALED,
    "In-control cut": X_INCONTROL_TEST_REDUCED_SCALED_CUT,
    "Out-of-control cut": X_OUT_OF_CONTROL_TEST_REDUCED_SCALED_CUT,
}

display_autoencoder_viewer(
    predict_fn,
    datasets=AE_DATASETS,  # Simple mode
    feature_names=feature_labels,
)
```

### Pattern 2: Fault Exploration (Use `fault_data`)
```python
# Explore different faults and simulations interactively
FAULT_DATA_RAW = {
    "Fault-free test": DF_FF_TEST_RAW,
    "Faulty test": DF_F_TEST_RAW,
}

display_autoencoder_viewer(
    predict_fn,
    datasets={},           # Empty!
    fault_data=FAULT_DATA_RAW,  # Rich mode with filtering
    scaler=scaler_incontrol,
    fault_column_name="faultNumber",
    simulation_column_name="simulationRun",
    columns_to_remove=COLUMNS_TO_REMOVE,
    default_fault=2,
    default_simulation=1,
    feature_names=feature_labels,
)
```

---

## Rule of Thumb

**Use `datasets`** when you want:
- Simple comparison of pre-processed arrays
- No interactive filtering
- Faster setup (fewer parameters)

**Use `fault_data`** when you want:
- Interactive exploration
- Dynamic fault/simulation selection
- Users to experiment with different scenarios
- One viewer for multiple faults/simulations

---

## Can You Use Both?

**No!** The function detects which mode you're using:

```python
# Mode detection (inside the function)
use_fault_mode = fault_data is not None and scaler is not None
```

If `fault_data` and `scaler` are provided, it ignores `datasets` and uses fault mode.

---

## Your Current Usage

Looking at your notebook:

```python
display_autoencoder_viewer(
    detector_predict_fn,
    datasets={},                          # Empty = not using datasets mode
    fault_data=FAULT_DATA_RAW,            # Using fault_data mode!
    scaler=scaler_incontrol,              # Required for fault_data mode
    fault_column_name=TARGET_VARIABLE_COLUMN_NAME,
    simulation_column_name=SIMULATION_RUN_COLUMN_NAME,
    columns_to_remove=COLUMNS_TO_REMOVE,
    default_key="Faulty test",
    default_fault=2,
    default_simulation=1,
    feature_names=feature_labels,
)
```

**You're using Mode 2 (fault_data)** - the interactive mode with fault/simulation dropdowns!

That's why you need:
- `scaler` → to scale filtered data
- `fault_column_name` → to know which column has fault numbers
- `simulation_column_name` → to know which column has simulation runs
- `columns_to_remove` → to remove metadata before scaling
