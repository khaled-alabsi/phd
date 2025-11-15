# DNN-CUSUM Q&A Documentation

## Overview

This document answers frequently asked questions about the DNN-CUSUM (Deep Neural Network-based Adaptive CUSUM) detector implementation.

---

## Table of Contents

1. [Architecture Questions](#architecture-questions)
2. [Feature Engineering](#feature-engineering)
3. [Training & Grid Search](#training--grid-search)
4. [System Wiring & Data Flow](#system-wiring--data-flow)
5. [Performance & Optimization](#performance--optimization)

---

## Architecture Questions

### Q1: What does the "6" mean in `n_features = self.n_dims * 6`?

**Answer:**

The **6** represents the **6 statistical features** extracted from each dimension of your data.

For **each dimension** in your data, DNN-CUSUM extracts these **6 statistical features** from the sliding window:

#### The 6 Features:

1. **Mean** - `np.mean(dim_data)`
   - Average value in the window
   - Captures the central tendency

2. **Standard Deviation** - `np.std(dim_data)`
   - Variability/spread of values
   - Detects increased volatility during faults

3. **Range** - `np.max(dim_data) - np.min(dim_data)`
   - Difference between max and min values
   - Shows how much the variable fluctuates

4. **Total Change** - `dim_data[-1] - dim_data[0]`
   - Change from first to last point in window
   - Captures trends (increasing/decreasing)

5. **Average Rate of Change** - `np.mean(np.diff(dim_data))`
   - Average step-by-step change
   - Detects how quickly values are changing

6. **Autocorrelation** - `np.corrcoef(dim_data[:-1], dim_data[1:])[0, 1]`
   - Correlation between consecutive points
   - Measures temporal dependency/smoothness

#### Example Calculation

For your **Tennessee Eastman Process data:**
- **52 dimensions** (variables like temperature, pressure, flow rates)
- **6 features per dimension**
- **Total features = 52 × 6 = 312**

**Example for 1 dimension (e.g., Temperature):**

Given a window of 50 temperature readings:
```
Window: [25.3, 25.4, 25.5, ..., 26.8, 27.2]
```

Extracted 6 features:
```python
[
  25.9,    # 1. Mean temperature
  0.45,    # 2. Std deviation
  1.9,     # 3. Range (27.2 - 25.3)
  1.9,     # 4. Total change (27.2 - 25.3)
  0.038,   # 5. Avg rate of change
  0.12     # 6. Autocorrelation
]
```

#### Why These 6 Features?

They capture different aspects of data behavior:
- **Static** (mean, std) - Current state
- **Dynamic** (change, rate) - Trends
- **Structural** (range, autocorr) - Pattern characteristics

This rich information helps the DNN decide optimal k and h values for CUSUM!

---

### Q2: How does the network architecture work?

**Answer:**

The DNN-CUSUM uses a multi-layer architecture:

```
Input Layer (312 features)
    ↓
Reshape Layer (52 dims, 6 features each)
    ↓
LSTM Layer(s) - Sequential processing
    ↓
Dense Layer(s) - Pattern recognition
    ↓
Dropout - Regularization
    ↓
Output Layer - Two heads:
    ├── k_output (softplus activation)
    └── h_output (softplus activation)
```

**Key Components:**

1. **LSTM Layers**: Process temporal sequences, learn patterns over the 52 dimensions
2. **Dense Layers**: Non-linear transformations for complex pattern recognition
3. **Dropout**: Prevents overfitting
4. **Softplus Activation**: Ensures k and h are always positive

**Example Configuration:**
```python
LSTM[64] → Dense[32] → Dropout(0.2) → [k_output, h_output]
```

---

## System Wiring & Data Flow

### Q3: How is the DNN connected to CUSUM? Show me the wiring.

**Answer:**

Here's the complete data flow from raw data to final anomaly detection:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DNN-CUSUM SYSTEM ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐
│   Raw Time Series   │
│   (52 dimensions)   │  e.g., [x₁, x₂, ..., x₅₂] at time t
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         SLIDING WINDOW EXTRACTION                        │
│  Extract last 50 time points for each dimension                         │
└──────────┬──────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FEATURE EXTRACTION LAYER                         │
│                                                                          │
│  For each of 52 dimensions, extract 6 features:                         │
│                                                                          │
│  Dimension 1 → [mean₁, std₁, range₁, change₁, rate₁, autocorr₁]        │
│  Dimension 2 → [mean₂, std₂, range₂, change₂, rate₂, autocorr₂]        │
│       ⋮                          ⋮                                       │
│  Dimension 52 → [mean₅₂, std₅₂, range₅₂, change₅₂, rate₅₂, autocorr₅₂] │
│                                                                          │
│  Output: Feature vector of size 312 (52 × 6)                            │
└──────────┬──────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FEATURE SCALING (StandardScaler)                 │
│  Normalize features to zero mean, unit variance                         │
└──────────┬──────────────────────────────────────────────────────────────┘
           │
           ▼
╔═════════════════════════════════════════════════════════════════════════╗
║                         DEEP NEURAL NETWORK (DNN)                        ║
║                                                                          ║
║  ┌────────────────────┐                                                 ║
║  │   Input Layer      │  Shape: (312,)                                  ║
║  │  312 features      │                                                 ║
║  └─────────┬──────────┘                                                 ║
║            │                                                             ║
║            ▼                                                             ║
║  ┌────────────────────┐                                                 ║
║  │   Reshape Layer    │  Shape: (52, 6)                                 ║
║  │  52 dims × 6 feat  │  Organize features by dimension                 ║
║  └─────────┬──────────┘                                                 ║
║            │                                                             ║
║            ▼                                                             ║
║  ┌────────────────────┐                                                 ║
║  │   LSTM Layer(s)    │  Units: 64 or 128                               ║
║  │  Temporal Learning │  Learns sequential patterns                     ║
║  └─────────┬──────────┘                                                 ║
║            │                                                             ║
║            ▼                                                             ║
║  ┌────────────────────┐                                                 ║
║  │   Dense Layer(s)   │  Units: 32 or 64                                ║
║  │  ReLU activation   │  Non-linear transformations                     ║
║  └─────────┬──────────┘                                                 ║
║            │                                                             ║
║            ▼                                                             ║
║  ┌────────────────────┐                                                 ║
║  │   Dropout Layer    │  Rate: 0.2 or 0.3                               ║
║  │  Regularization    │  Prevents overfitting                           ║
║  └─────────┬──────────┘                                                 ║
║            │                                                             ║
║            │  ┌──────────────────────────┐                              ║
║            ├──► k_output (Dense)         │  Softplus activation         ║
║            │  │ Adaptive reference value │  Output: k_t > 0             ║
║            │  └──────────┬───────────────┘                              ║
║            │             │                                               ║
║            │  ┌──────────▼───────────────┐                              ║
║            └──► h_output (Dense)         │  Softplus activation         ║
║               │ Adaptive threshold       │  Output: h_t > 0             ║
║               └──────────┬───────────────┘                              ║
║                          │                                               ║
╚══════════════════════════┼═══════════════════════════════════════════════╝
                           │
                           │  k_t and h_t are predicted for time t
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         PARAMETER ADAPTATION                             │
│                                                                          │
│  At each time point t:                                                  │
│    • DNN predicts: k_t (reference value)                                │
│    • DNN predicts: h_t (threshold)                                      │
│                                                                          │
│  These replace fixed k and h in traditional CUSUM                       │
└──────────┬──────────────────────────────────────────────────────────────┘
           │
           │  Adaptive parameters k_t and h_t
           │
           ▼
╔═════════════════════════════════════════════════════════════════════════╗
║                    MULTIVARIATE CUSUM COMPUTATION                        ║
║                      (from src/mcusum.py)                                ║
║                                                                          ║
║  Input:                                                                  ║
║    • Current observation: x_t (52 dimensions)                           ║
║    • Adaptive k_t (from DNN)                                            ║
║    • Adaptive h_t (from DNN)                                            ║
║    • Global statistics: μ₀, Σ (from training)                           ║
║                                                                          ║
║  CUSUM Recursion:                                                        ║
║    ┌──────────────────────────────────────┐                             ║
║    │ S_t = max(0, S_{t-1} + C_t - k_t)   │  Cumulative sum             ║
║    └──────────────────────────────────────┘                             ║
║                                                                          ║
║  where:                                                                  ║
║    C_t = (x_t - μ₀)ᵀ Σ⁻¹ (x_t - μ₀)     Mahalanobis distance           ║
║                                                                          ║
║  Decision Rule:                                                          ║
║    ┌──────────────────────────────────────┐                             ║
║    │ if S_t > h_t:                        │                             ║
║    │     Flag as ANOMALY (1)              │                             ║
║    │ else:                                 │                             ║
║    │     Flag as NORMAL (0)               │                             ║
║    └──────────────────────────────────────┘                             ║
║                                                                          ║
║  Output: (S_t, prediction_t)                                            ║
╚══════════════════════════┬══════════════════════════════════════════════╝
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FINAL OUTPUT                                     │
│                                                                          │
│  • Binary Prediction: 0 (normal) or 1 (anomaly)                         │
│  • CUSUM Statistic: S_t                                                  │
│  • Parameter History: {k_t, h_t, S_t} over time (optional)              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Q4: What's the difference between DNN-CUSUM and traditional CUSUM?

**Answer:**

#### Traditional CUSUM (Fixed Parameters)

```
┌──────────────┐
│  Raw Data    │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────┐
│  CUSUM with FIXED k and h    │  k = 0.5, h = 5.0 (constant)
│                              │
│  S_t = max(0, S_{t-1} + C_t - k)
│                              │
│  if S_t > h: ANOMALY         │
└──────┬───────────────────────┘
       │
       ▼
   [0 or 1]
```

**Limitations:**
- Same k and h for all time points
- Doesn't adapt to changing process conditions
- May be too sensitive (false alarms) or too conservative (missed detections)

---

#### DNN-CUSUM (Adaptive Parameters)

```
┌──────────────┐
│  Raw Data    │
└──────┬───────┘
       │
       ├─────────────────────────────────┐
       │                                 │
       │  Extract Features               │  Compute CUSUM
       │  (312 features)                 │
       ▼                                 │
┌──────────────────┐                     │
│      DNN         │                     │
│  Predict k_t, h_t│  ← Adapts to       │
└──────┬───────────┘    current state   │
       │                                 │
       │  k_t, h_t (dynamic)             │
       └─────────────┬───────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  CUSUM with          │
          │  ADAPTIVE k_t, h_t   │
          │                      │
          │  S_t = max(0, ...)   │
          │  if S_t > h_t: ...   │
          └──────┬───────────────┘
                 │
                 ▼
             [0 or 1]
```

**Advantages:**
- k_t and h_t adapt in real-time
- Lower k, lower h during faults → faster detection
- Higher k, higher h during normal → fewer false alarms
- DNN learns optimal parameters from training data

---

## Feature Engineering

### Q5: Why do we use a sliding window of 50 time points?

**Answer:**

The window size (default 50) is a balance between:

**Advantages of larger windows:**
- ✅ More historical context
- ✅ Better statistical estimates (mean, std)
- ✅ Captures longer-term trends

**Advantages of smaller windows:**
- ✅ Faster response to changes
- ✅ Less memory/computation
- ✅ More training samples

**Default 50** is chosen because:
- Tennessee Eastman faults typically evolve over 20-100 time steps
- 50 provides enough context without excessive delay
- Creates sufficient training samples from your dataset

You can adjust it:
```python
dnn_cusum = DNNCUSUMDetector(window_size=30)  # Faster response
dnn_cusum = DNNCUSUMDetector(window_size=70)  # More context
```

---

### Q6: How does feature scaling work?

**Answer:**

Features are standardized using `StandardScaler`:

```python
# Training
feature_scaler.fit(X_features_train)
X_scaled = feature_scaler.transform(X_features_train)

# Testing
X_scaled = feature_scaler.transform(X_features_test)  # Use same scaler!
```

**Why it's important:**
- Different features have different scales:
  - Mean might be ~25.0 (temperature)
  - Std might be ~0.5
  - Range might be ~5.0
- Without scaling, DNN would focus on large-magnitude features
- Scaling ensures all features contribute equally

**Formula:**
```
x_scaled = (x - μ) / σ
```

where μ and σ are computed from training data.

---

## Training & Grid Search

### Q7: What does grid search optimize?

**Answer:**

Grid search tests different combinations of hyperparameters to find the best configuration:

**Parameters searched:**

1. **Architecture** (network structure):
   - `{'units': [64], 'dense': [32]}` - Smaller network
   - `{'units': [128], 'dense': [64]}` - Larger network

2. **Learning Rate**:
   - `0.001` - Standard learning rate

3. **Batch Size**:
   - `32` - Number of samples per gradient update

4. **Dropout**:
   - `0.2` - Drop 20% of neurons (less regularization)
   - `0.3` - Drop 30% of neurons (more regularization)

**Total configurations:** 2 architectures × 1 lr × 1 bs × 2 dropout = **4 configs**

**Selection criteria:**
- Best configuration = lowest validation loss
- Validation loss measures how well the network predicts k and h on unseen data

---

### Q8: How long does training take?

**Answer:**

**First Run (with grid search):**
```
Grid search: 4 configs × ~10 min each = ~40 minutes
Final training: ~5 minutes
Total: ~45 minutes
```

**Subsequent Runs (loading saved model):**
```
Load model: < 1 second ⚡
```

**Optimization tips:**
- Reduce `window_size` (50 → 30): Faster, less context
- Skip grid search: Use default config (5 min total)
- Reduce training data: Use subset for quick testing

---

### Q9: What files are saved after training?

**Answer:**

After training completes, these files are created in `models/`:

```
models/
├── dnn_cusum_model.h5              # Trained neural network (~1 MB)
├── dnn_cusum_best_config.json      # Best hyperparameters (< 1 KB)
├── dnn_cusum_model_scaler.pkl      # Feature scaler (< 1 KB)
└── dnn_cusum_model_params.pkl      # Global μ₀, Σ, n_dims (< 1 KB)
```

**Purpose:**
- **model.h5**: Neural network weights (can predict k_t, h_t)
- **config.json**: Architecture details for rebuilding network
- **scaler.pkl**: Ensures consistent feature scaling
- **params.pkl**: Global statistics for CUSUM computation

**Reuse:**
- Set `force_retrain=False` to load these files instantly
- Delete files to force retraining

---

## Performance & Optimization

### Q10: How do I interpret the parameter evolution plot?

**Answer:**

After running DNN-CUSUM, you'll see a 4-panel plot:

```
┌─────────────────────────────────────────────────────────────┐
│ Panel 1: k(t) - Reference Value Over Time                   │
│                                                              │
│  k_t  │     ────────┐  ┌─────                               │
│       │             └──┘      ← Lower during fault          │
│       │  ─────────────────────                              │
│       └────────────────────────────────────── time          │
│                    ↑ Fault starts                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Panel 2: h(t) - Threshold Over Time                         │
│                                                              │
│  h_t  │     ────────┐  ┌─────                               │
│       │             └──┘      ← Lower during fault          │
│       │  ─────────────────────                              │
│       └────────────────────────────────────── time          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Panel 3: CUSUM Statistic S(t)                               │
│                                                              │
│  S_t  │                    ┌────────  ← Exceeds h_t         │
│       │                   ╱                                  │
│       │  ────────────────                                    │
│       └────────────────────────────────────── time          │
│                                                              │
│  Red line = h_t (adaptive threshold)                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Panel 4: Binary Predictions                                 │
│                                                              │
│       │  0  0  0  0  0  1  1  1  1  1  1  1                 │
│       │  ────────────────████████████████  ← Anomaly        │
│       │                 ↑                                    │
│       │            Fault detected                            │
│       └────────────────────────────────────── time          │
└─────────────────────────────────────────────────────────────┘
```

**Interpretation:**
- **Before fault**: k_t and h_t are higher → conservative (avoid false alarms)
- **During fault**: k_t and h_t drop → sensitive (quick detection)
- **CUSUM rises**: When S_t exceeds h_t → anomaly flagged
- **Detection delay**: Time from fault start to first "1" prediction

---

### Q11: How does DNN-CUSUM compare to other methods?

**Answer:**

Expected performance ranking (from best to worst for Tennessee Eastman):

```
Method              │ ARL1 (Detection Delay) │ ARL0 (False Alarms)
────────────────────┼────────────────────────┼─────────────────────
DNN-CUSUM          │ ★★★★★ (Fastest)       │ ★★★★★ (Fewest)
Fixed MCUSUM       │ ★★★★☆                 │ ★★★☆☆
AutoencoderEnhanced│ ★★★☆☆                 │ ★★★★☆
MEWMA              │ ★★☆☆☆                 │ ★★★★☆
Autoencoder        │ ★★☆☆☆                 │ ★★★☆☆
```

**Why DNN-CUSUM excels:**
- Adapts parameters in real-time
- Learns from both normal and fault data
- Combines DNN pattern recognition with CUSUM statistical rigor
- Handles multivariate dependencies (52 dimensions)

---

### Q12: Can I visualize what the DNN learned?

**Answer:**

Yes! The parameter statistics plot shows this:

```python
dnn_viz.plot_parameter_statistics(param_history, predictions)
```

**What it shows:**

```
┌─────────────────────────────────────────────────────────────┐
│ k Distribution during Normal vs Anomaly Periods              │
│                                                              │
│  During Normal:  k_t ~ 5.0-8.0  (higher, conservative)      │
│  During Anomaly: k_t ~ 0.5-2.0  (lower, sensitive)          │
│                                                              │
│  ┌────┐                                                      │
│  │    │                     ┌────┐                           │
│  │    │                     │    │                           │
│  └────┘                     └────┘                           │
│  Normal                     Anomaly                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ h Distribution during Normal vs Anomaly Periods              │
│                                                              │
│  During Normal:  h_t ~ 8.0-12.0 (higher threshold)          │
│  During Anomaly: h_t ~ 2.0-5.0  (lower threshold)           │
│                                                              │
│  ┌────┐                                                      │
│  │    │                     ┌────┐                           │
│  │    │                     │    │                           │
│  └────┘                     └────┘                           │
│  Normal                     Anomaly                          │
└─────────────────────────────────────────────────────────────┘
```

**Insights:**
- Clear separation → DNN learned to distinguish normal/fault
- Overlap → DNN struggles with ambiguous regions
- Extreme values → May need retraining or parameter clipping

---

## Troubleshooting

### Q13: Why am I getting "Model not fitted" error?

**Answer:**

You need to call `fit()` before `predict()`:

```python
# WRONG ORDER:
dnn_cusum = DNNCUSUMDetector()
predictions = dnn_cusum.predict(X_test)  # ❌ Error!

# CORRECT ORDER:
dnn_cusum = DNNCUSUMDetector()
dnn_cusum.fit(X_train_normal, X_train_fault)  # ✅ Train first
predictions = dnn_cusum.predict(X_test)        # ✅ Then predict
```

---

### Q14: How do I force retraining?

**Answer:**

Use `force_retrain=True`:

```python
dnn_cusum.fit(
    X_INCONTROL_TRAIN_FULL_SCALED,
    X_OUT_OF_CONTROL_TRAIN_PLAY_SCALED,
    force_retrain=True,  # ← Ignore saved model, retrain from scratch
    grid_search=True
)
```

This is useful when:
- You have new training data
- You want to try different configurations
- Saved model seems to perform poorly

---

### Q15: Can I use DNN-CUSUM without fault data?

**Answer:**

Yes, but performance may be reduced:

```python
# Train with only normal data
dnn_cusum.fit(
    X_INCONTROL_TRAIN_FULL_SCALED,
    X_outcontrol=None,  # ← No fault data
    force_retrain=True,
    grid_search=True
)
```

**What happens:**
- DNN only learns to set parameters for normal conditions
- May not adapt well when faults occur
- Still better than fixed CUSUM if process conditions vary

**Recommendation:**
- Use at least a small amount of fault data if available
- Even 1-2 fault types help the DNN learn adaptation patterns

---

## Mathematical Details

### Q16: What is the CUSUM formula used?

**Answer:**

**Multivariate CUSUM Recursion:**

```
S_t = max(0, S_{t-1} + C_t - k_t)
```

where:
- **S_t**: CUSUM statistic at time t (cumulative sum)
- **S_{t-1}**: Previous CUSUM value (starts at 0)
- **C_t**: Mahalanobis distance = (x_t - μ₀)ᵀ Σ⁻¹ (x_t - μ₀)
- **k_t**: Reference value (from DNN, replaces fixed k)

**Decision Rule:**
```
if S_t > h_t:
    prediction_t = 1  (ANOMALY)
else:
    prediction_t = 0  (NORMAL)
```

where:
- **h_t**: Threshold (from DNN, replaces fixed h)

**Mahalanobis Distance C_t:**
- Measures how far x_t is from normal mean μ₀
- Accounts for correlations via covariance matrix Σ
- Higher C_t → more abnormal observation

**Adaptive Parameters:**
- **k_t**: Controls sensitivity (lower k = more sensitive)
- **h_t**: Decision boundary (lower h = quicker alarms)

---

### Q17: How are optimal k and h computed during training?

**Answer:**

During training, "optimal" k and h are computed using heuristics:

**For Normal Windows (no fault):**
```python
k_optimal = base_k × 1.5 × (1.0 + 0.05 × volatility)
h_optimal = base_h × 1.2 × (1.0 + 0.05 × magnitude)
```
→ **Higher k, higher h** (conservative, avoid false alarms)

**For Fault Windows:**
```python
k_optimal = base_k × 0.3 × (1.0 + 0.1 × magnitude)
h_optimal = base_h × 0.6 × (1.0 + 0.1 × volatility)
```
→ **Lower k, lower h** (sensitive, quick detection)

**Where:**
- `base_k = 0.5`, `base_h = 5.0` (defaults)
- `magnitude = mean(|window - mean|)` - How abnormal the window is
- `volatility = mean(std(window))` - How much variation

**Clipping:**
```python
k_optimal = clip(k_optimal, min=0.1, max=10.0)
h_optimal = clip(h_optimal, min=1.0, max=15.0)
```

**DNN Training:**
- Input: 312 features from window
- Target: (k_optimal, h_optimal) computed above
- DNN learns to map features → optimal parameters

---

## Advanced Usage

### Q18: Can I customize the grid search space?

**Answer:**

Yes, modify the `param_grid` in `src/dnn_cusum.py`:

```python
# Current (fast):
param_grid = {
    'architecture': [
        {'units': [64], 'dense': [32]},
        {'units': [128], 'dense': [64]},
    ],
    'learning_rate': [0.001],
    'batch_size': [32],
    'dropout': [0.2, 0.3]
}
# Total: 4 configs

# More thorough (slow):
param_grid = {
    'architecture': [
        {'units': [64], 'dense': [32]},
        {'units': [128], 'dense': [64, 32]},
        {'units': [64, 64], 'dense': [32]},
        {'units': [128, 128], 'dense': [64]},
    ],
    'learning_rate': [0.001, 0.0001],
    'batch_size': [32, 64],
    'dropout': [0.1, 0.2, 0.3]
}
# Total: 4 × 2 × 2 × 3 = 48 configs (~8 hours)
```

---

### Q19: How do I save/load a trained model?

**Answer:**

Models are automatically saved and loaded:

**Save (automatic after training):**
```python
dnn_cusum.fit(X_train_normal, X_train_fault, force_retrain=True)
# Saves to: models/dnn_cusum_model.h5
#           models/dnn_cusum_best_config.json
#           models/dnn_cusum_model_scaler.pkl
#           models/dnn_cusum_model_params.pkl
```

**Load (automatic on next fit):**
```python
dnn_cusum.fit(X_train_normal, X_train_fault, force_retrain=False)
# If model exists: loads instantly (< 1 sec)
# If not: trains new model
```

**Manual save/load:**
```python
# Save
dnn_cusum.save_model('custom_path/my_model.h5', 'custom_path/my_config.json')

# Load
dnn_cusum.load_model('custom_path/my_model.h5', 'custom_path/my_config.json')
```

---

### Q20: Can I use this for other datasets (not Tennessee Eastman)?

**Answer:**

**Yes!** DNN-CUSUM is designed to be dataset-agnostic. Here's how:

```python
# Your custom dataset
X_normal_train = np.array(...)  # Shape: (n_samples, n_features)
X_fault_train = np.array(...)   # Shape: (n_samples, n_features)
X_test = np.array(...)          # Shape: (n_samples, n_features)

# Initialize (automatically adapts to your n_features)
dnn_cusum = DNNCUSUMDetector(window_size=50)

# Train
dnn_cusum.fit(X_normal_train, X_fault_train)

# Predict
predictions, param_history = dnn_cusum.predict(X_test, return_params=True)
```

**Requirements:**
- Data must be numerical (continuous variables)
- Must have in-control (normal) training data
- Fault data is optional but recommended
- All data should be scaled similarly (same units, ranges)

**Examples:**
- Manufacturing: Machine sensor data
- Healthcare: Patient vital signs
- Finance: Stock price indicators
- Energy: Power grid measurements
- IT: System performance metrics

---

## Summary

**Key Takeaways:**

1. **Architecture**: DNN extracts 312 features (52 dims × 6 features) → LSTM → predicts k_t, h_t → CUSUM detects anomalies

2. **Adaptive**: Unlike fixed CUSUM, parameters adjust in real-time based on current data patterns

3. **Training**: Grid search finds best architecture, then model is saved for instant reuse

4. **Performance**: Combines DNN learning with CUSUM statistical rigor for superior detection

5. **Flexible**: Works with any multivariate time series dataset

---

**Need more help?** Check these files:
- `DNN_CUSUM_README.md` - User guide
- `DNN_CUSUM_PAPER.md` - Technical paper
- `src/dnn_cusum.py` - Implementation code
- `INTEGRATION_COMPLETE.md` - Setup guide

**Happy anomaly detecting!** 🚀📊
