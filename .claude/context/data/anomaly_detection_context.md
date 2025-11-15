# Anomaly Detection Notebook Context

**Notebook**: `code/anomaly_detection.ipynb`
**Last Updated**: 2025-10-31

## Project Architecture

- **Frontend**: Jupyter notebook (`anomaly_detection.ipynb`) - experiments, visualization, and model orchestration
- **Backend**: Implementation code in `code/src/` directory
- **Pattern**: Notebook imports modules from `src/` and uses them as high-level API

## Data Structure

### Input Data Sources
- **Mode**: Toggle between `'real'` (RData files) or `'synthetic'` (generated data)
- **Datasets**:
  - `DF_FF_TRAINING_RAW`: Fault-free training data
  - `DF_FF_TEST_RAW`: Fault-free testing data
  - `DF_F_TRAINING_RAW`: Faulty training data (multiple faults)
  - `DF_F_TEST_RAW`: Faulty testing data

### Key Columns
- `TARGET_VARIABLE_COLUMN_NAME = "faultNumber"`: Fault identifier (0 = normal, 1-5 = fault types)
- `SIMULATION_RUN_COLUMN_NAME = "simulationRun"`: Simulation run identifier
- `COLUMNS_TO_REMOVE = ["faultNumber", "simulationRun", "sample"]`: Metadata columns removed before training

### Data Preprocessing
- **Scaler**: `StandardScaler` fitted on full fault-free training data (`X_INCONTROL_TRAIN_FULL_SCALED`)
- **Fault Injection Point**: `FAULT_INJECTION_POINT = 160` (samples before this are pre-fault)
- **Cut Datasets**: Data after fault injection point (suffix `_CUT`) used for ARL evaluation

## Implemented Models

### 1. MCUSUM (Multivariate CUSUM)
- **Location**: `src/mcusum.py`
- **Class**: `MCUSUMDetector`
- **Key Parameters**:
  - `k`: Reference value (tuned via grid search)
  - `h`: Control limit (default 2.8)
- **Training**: Fits on `X_INCONTROL_TRAIN_FULL_SCALED`
- **Predict Function**: `mcusum_predict(x_scaled)`

### 2. Basic Autoencoder
- **Location**: `src/autoencoder.py`
- **Class**: `AutoencoderDetector`
- **Features**: Grid search with caching
- **Key Parameters**:
  - `encoding_dim`: [8, 16]
  - `hidden_dim`: [16, 32]
  - `learning_rate`: [0.001, 0.0005]
  - `threshold_percentile`: 95
- **Cache**: `models/autoencoder_basic_*`
- **Threshold**: Computed on reconstruction error percentile

### 3. Enhanced Autoencoder
- **Location**: `src/autoencoder_enhanced.py`
- **Class**: `AutoencoderDetectorEnhanced`
- **Features**: Adds noise injection and dropout
- **Key Parameters**:
  - `noise_stddev`: 0.1
  - `dropout_rate`: 0.2
- **Cache**: `models/autoencoder_enhanced_*`

### 4. Flexible Autoencoder (Teacher Model)
- **Location**: `src/flexible_autoencoder.py`
- **Class**: `FlexibleAutoencoder`
- **Purpose**: Teacher model for residual regression
- **Features**:
  - Configurable encoder layers
  - Bottleneck grid search
  - Elbow curve analysis
- **Variants**:
  - `autoencoder`: Optimal compressed latent space
  - `autoencoder_overfitted`: Large latent dim (64) for overfitting experiments

### 5. Residual Regressor (Student Model)
- **Location**: `src/residual_regressor.py`
- **Class**: `ResidualRegressor`
- **Architecture**: Learns to predict autoencoder residuals directly
- **Input**: Raw features → **Output**: Predicted residual scores
- **Threshold**: `residual_threshold` at 99.5 percentile
- **Predict Function**: `autoencoder_residual_regressor_predict(X)`

### 6. Per-Feature Residual Regressor
- **Location**: `src/autoencoder_feature_residual_regressor.py`
- **Class**: `AutoencoderFeatureResidualRegressor`
- **Architecture**: Predicts residuals per feature (not aggregated)
- **Teacher**: Uses `AutoencoderDetectorEnhanced` as teacher
- **Aggregation**: `FEATURE_AGG = "mean"` (mean, max, sum, median)
- **Threshold**: `feature_residual_threshold` at 99.5 percentile
- **Predict Function**: `autoencoder_feature_residual_regressor_predict(X)`

## Key Global Variables

### Data Splits
- `X_INCONTROL_TRAIN_FULL_SCALED`: Full fault-free training (all simulations)
- `X_INCONTROL_TRAIN_REDUCED_SCALED`: Single simulation training
- `X_INCONTROL_TEST_REDUCED_SCALED`: Single simulation test
- `X_OUT_OF_CONTROL_TEST_REDUCED_SCALED`: Single fault test
- `X_INCONTROL_TEST_REDUCED_SCALED_CUT`: In-control after fault injection point
- `X_OUT_OF_CONTROL_TEST_REDUCED_SCALED_CUT`: Out-of-control after fault injection point

### Combined Chains (for temporal experiments)
- `COMBINED_NORMAL_FAULT_CHAIN`: Alternating normal → fault segments
- `COMBINED_FAULT_NORMAL_CHAIN`: Alternating fault → normal segments
- `COMBINED_SEGMENT_PERCENT = 0.2`: 20% segment length

### Scalers & Thresholds
- `scaler_incontrol`: StandardScaler fitted on fault-free data
- `residual_threshold`: Threshold for residual regressor
- `feature_residual_threshold`: Threshold for per-feature regressor
- `teacher_threshold`: Threshold for teacher AE model

## Evaluation Framework

### Model Dictionary
```python
MODELS = {
    "MCUSUM": mcusum_predict,
    "AE_Teacher": ae_teacher_predict,
    "AE_ResidualRegressor": autoencoder_residual_regressor_predict,
    "AE_FeatureResidualRegressor": autoencoder_feature_residual_regressor_predict,
    "Autoencoder": autoencoder_detector.predict,
    "AutoencoderEnhanced": autoencoder_detector_enhanced.predict,
}
```

### Evaluation Pipeline
- **Module**: `src/model_evaluator.py`
- **Function**: `evaluate_models()`
- **Output**:
  - `df_results`: Detailed per-simulation results
  - `summary_df`: Aggregated summary statistics

### Metrics
- **ARL0**: Average Run Length in-control (higher is better - fewer false alarms)
- **ARL1**: Average Run Length out-of-control (lower is better - faster detection)
- **Precision, Recall, F1, Accuracy, Specificity**
- **FPR, FNR**: False positive/negative rates
- **Confusion Matrix**: TP, TN, FP, FN

### Analysis & Visualization
- **Module**: `src/ModelComparisonAnalyzer.py`
- **Class**: `ModelComparisonAnalyzer`
- **Features**:
  - Summary tables
  - Model rankings (ratio, normalized, harmonic mean)
  - ARL analysis plots
  - Classification metric plots
  - Confusion matrix heatmaps
  - Per-fault analysis
  - Statistical distributions

## Interactive Viewers

### 1. Autoencoder Reconstruction Viewer
- **Function**: `display_autoencoder_reconstruction()`
- **Location**: `src/autoencoder_viewers.py`
- **Features**: Compare original vs reconstructed features

### 2. Autoencoder with Fault Selection
- **Function**: `display_autoencoder_with_fault_selection()`
- **Location**: `src/autoencoder_viewers.py`
- **Features**: Select fault type and simulation run dynamically

### 3. Residual Viewer
- **Function**: `display_residual_viewer()`
- **Location**: `src/results_visualizer.py`
- **Features**: Scatter plot and timeline of residuals

### 4. Feature Residual Timeline Viewer
- **Function**: `display_feature_residual_timeline_viewer()`
- **Location**: `src/__init__.py`
- **Features**: Per-feature residual comparison (AE vs regressor)

## Important Configuration Constants

```python
TARGET_VARIABLE_COLUMN_NAME = "faultNumber"
SIMULATION_RUN_COLUMN_NAME = "simulationRun"
FAULT_INJECTION_POINT = 160
DATA_SOURCE = "synthetic"  # or "real"
COMBINED_SEGMENT_PERCENT = 0.2
RESIDUAL_THRESHOLD_PERCENTILE = 99.5
FEATURE_RESIDUAL_THRESHOLD_PERCENTILE = 99.5
TEACHER_THRESHOLD_PERCENTILE = 99.5
FEATURE_AGG = "mean"  # aggregation method for per-feature residuals
```

## Model Caching System

All models use a caching system to avoid retraining:
- **Location**: `code/models/` directory
- **Files**:
  - `{cache_prefix}_config.json`: Hyperparameters
  - `{cache_prefix}_model.keras`: Trained model weights
  - `{cache_prefix}_metrics.json`: Training metrics
  - `{cache_prefix}_threshold.json`: Detection threshold
- **Reset**: Set `reset_experiment=True` or `reset=True` to force retrain

## Workflow Summary

1. **Load Data** → `load_datasets()` from `src/data_generator.py`
2. **Preprocess** → StandardScaler on fault-free training data
3. **Train Models** → Each model has `.fit()` method with grid search
4. **Evaluate** → `evaluate_models()` across all faults and simulations
5. **Analyze** → `ModelComparisonAnalyzer` for comprehensive comparison
6. **Visualize** → Interactive viewers for detailed inspection

## Dependencies

Key imports from `src/`:
- `mcusum`: MCUSUM detector
- `autoencoder`: Basic autoencoder
- `autoencoder_enhanced`: Enhanced autoencoder with noise/dropout
- `flexible_autoencoder`: Flexible teacher autoencoder
- `residual_regressor`: Residual regression student
- `autoencoder_feature_residual_regressor`: Per-feature residual regressor
- `data_generator`: Data loading utilities
- `model_evaluator`: Evaluation framework
- `ModelComparisonAnalyzer`: Analysis and visualization
- `results_visualizer`: Detection result viewers
- `autoencoder_viewers`: Autoencoder-specific viewers
