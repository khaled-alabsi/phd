# Autoencoder Viewer Function Split - Summary

## Problem

The original `display_autoencoder_viewer()` function tried to handle two completely different use cases:
1. **Simple mode**: View pre-processed numpy arrays
2. **Fault selection mode**: Interactive fault/simulation filtering from raw DataFrames

This led to:
- ❌ Confusing parameter names (`fault_data` even for fault-free data)
- ❌ Unclear function purpose from name alone
- ❌ Too many parameters (12+)
- ❌ Complex internal logic switching between modes

## Solution

Split into **two dedicated functions** with clear names and purposes.

---

## New Functions

### 1. `display_autoencoder_reconstruction()`

**Purpose:** Simple viewer for pre-processed data

**When to use:** You have already scaled and filtered numpy arrays

**Parameters:**
```python
display_autoencoder_reconstruction(
    predict_fn,          # Reconstruction function
    datasets,            # Dict[str, np.ndarray] - pre-processed arrays
    feature_names=None,  # Optional feature labels
    default_key=None,    # Initial dataset selection
    viewer_key=None,     # Unique viewer ID
)
```

**Example:**
```python
datasets = {
    "Train": X_train_scaled,
    "Test": X_test_scaled,
}
display_autoencoder_reconstruction(
    lambda X: model.predict(X, verbose=0),
    datasets,
    feature_names=["f0", "f1", "f2"]
)
```

---

### 2. `display_autoencoder_with_fault_selection()`

**Purpose:** Interactive viewer with fault/simulation dropdowns

**When to use:** You want dynamic filtering of raw DataFrames

**Parameters:**
```python
display_autoencoder_with_fault_selection(
    predict_fn,              # Reconstruction function
    raw_dataframes,          # Dict[str, pd.DataFrame] - raw data
    scaler,                  # StandardScaler for on-the-fly scaling
    fault_column="faultNumber",
    simulation_column="simulationRun",
    metadata_columns=None,   # Columns to remove
    default_dataset=None,    # Initial dataset
    default_fault=None,      # Initial fault number
    default_simulation=None, # Initial simulation run
    feature_names=None,
    viewer_key=None,
)
```

**Example:**
```python
raw_dfs = {
    "Fault-free": df_normal_raw,
    "Faulty": df_faulty_raw,
}
display_autoencoder_with_fault_selection(
    lambda X: model.predict(X, verbose=0),
    raw_dfs,
    scaler=scaler_incontrol,
    fault_column="faultNumber",
    simulation_column="simulationRun",
    metadata_columns=["faultNumber", "simulationRun", "sample"],
    default_fault=2,
    default_simulation=1,
)
```

---

## Benefits

| Aspect | Before | After |
|--------|--------|-------|
| **Function name** | `display_autoencoder_viewer()` | Specific: `display_autoencoder_reconstruction()` or `display_autoencoder_with_fault_selection()` |
| **Purpose clarity** | Unclear from name | Clear from name |
| **Parameter confusion** | `fault_data` for all data | `raw_dataframes` (clear!) |
| **Mode switching** | Hidden internal logic | Explicit function choice |
| **Parameters** | 12+ mixed-purpose params | 6-7 focused params each |
| **Naming** | `datasets` vs `fault_data` confusion | `datasets` vs `raw_dataframes` (clear!) |

---

## Migration Guide

### Old Code (Confusing)
```python
# Simple mode - unclear purpose
display_autoencoder_viewer(
    predict_fn,
    datasets={"Train": X_train},  # Not obvious this is simple mode
    feature_names=labels,
)

# Fault mode - confusing "fault_data" name
display_autoencoder_viewer(
    predict_fn,
    datasets={},  # Why empty?
    fault_data={"Fault-free": df},  # Why "fault_data" for normal data?
    scaler=scaler,
    fault_column_name="faultNumber",
    ...12 more params...
)
```

### New Code (Clear)
```python
# Simple mode - obvious from name!
display_autoencoder_reconstruction(
    predict_fn,
    datasets={"Train": X_train},
    feature_names=labels,
)

# Fault mode - clear naming!
display_autoencoder_with_fault_selection(
    predict_fn,
    raw_dataframes={"Fault-free": df},  # Clear: raw DataFrames!
    scaler=scaler,
    fault_column="faultNumber",
    simulation_column="simulationRun",
    metadata_columns=["faultNumber", "simulationRun", "sample"],
)
```

---

## Backward Compatibility

The old `display_autoencoder_viewer()` still exists in `autoencoder_viewers.py` but shows a deprecation warning:

```python
DeprecationWarning: display_autoencoder_viewer() is deprecated. Use:
  - display_autoencoder_reconstruction() for simple mode
  - display_autoencoder_with_fault_selection() for fault selection mode
```

It will be removed in a future version.

---

## Files Modified

1. **NEW:** `src/autoencoder_viewers.py` - Two new dedicated functions
2. **UPDATED:** `anomaly_detection.ipynb` - All 5 usages updated with new function names
3. **UNCHANGED:** `src/autoencoder_viewer.py` - Original implementation kept for backward compatibility

---

## Notebook Changes

Updated 5 cells:
- **Cell 3d86e134**: Import statement
- **Cell 070fbef5**: autoencoder_detector (fault mode) → `display_autoencoder_with_fault_selection()`
- **Cell db6ffd3e**: autoencoder_detector_enhanced (fault mode) → `display_autoencoder_with_fault_selection()`
- **Cell kqmp2khf4hn**: Example fault mode → `display_autoencoder_with_fault_selection()`
- **Cell a676f53f**: Residual regressor (simple mode) → `display_autoencoder_reconstruction()`
- **Cell 9f1b7c2a**: Another simple mode → `display_autoencoder_reconstruction()`

---

## Key Improvements

✅ **Clear naming**: Function names explain what they do
✅ **Focused parameters**: Each function has only relevant parameters
✅ **Better parameter names**: `raw_dataframes` instead of `fault_data`
✅ **Explicit choice**: User chooses function, not hidden mode
✅ **Easier to use**: Fewer parameters to remember
✅ **Self-documenting**: Code reads clearly

---

## Summary

**Before:** One confusing function with 12+ parameters and hidden modes
**After:** Two focused functions with 6-7 clear parameters each

The purpose is now obvious from the function name!
