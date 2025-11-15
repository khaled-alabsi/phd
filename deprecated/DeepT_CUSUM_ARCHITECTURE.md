# Deep Threshold CUSUM (DeepT-CUSUM) Architecture

## Overview

**Deep Threshold CUSUM (DeepT-CUSUM)** is a hybrid anomaly detection method that combines:
- Statistical process control (CUSUM with fixed reference value)
- Deep learning (DNN for adaptive threshold prediction)
- Feedback mechanism (CUSUM state as input)

**Key Innovation:** Uses current CUSUM statistic as feedback to predict adaptive threshold.

---

## Method Comparison

| Method | k | h | Input | Output | Complexity |
|--------|---|---|-------|--------|------------|
| **Fixed CUSUM** | Fixed | Fixed | None | Detection | Low |
| **DNN-CUSUM** | DNN | DNN | 312 features | k_t, h_t | High |
| **DeepT-CUSUM** | Fixed | DNN | 53 features | h_t | Medium |

---

## Architecture

### Input Layer (53 features)

```
Input Vector = [x_t, S_{t-1}]

Where:
├── x_t: Current observation (52 dimensions)
│   ├── Tennessee Eastman Process variables:
│   ├── Temperatures, pressures, flow rates, etc.
│   └── Already scaled by StandardScaler
│
└── S_{t-1}: Previous CUSUM statistic (1 scalar)
    └── Provides feedback about current alarm state
```

**Total: 53 input features**

---

### Network Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT LAYER                          │
│                                                         │
│  [x₁, x₂, ..., x₅₂, S_{t-1}]                          │
│         52 dims    +   1                               │
│                                                         │
│  Shape: (53,)                                          │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                  DENSE LAYER 1                          │
│                                                         │
│  Units: 64 or 128                                      │
│  Activation: ReLU                                      │
│  Purpose: Extract patterns from features + state       │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                  DROPOUT LAYER                          │
│                                                         │
│  Rate: 0.2 - 0.3                                       │
│  Purpose: Regularization                               │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                  DENSE LAYER 2                          │
│                                                         │
│  Units: 32 or 64                                       │
│  Activation: ReLU                                      │
│  Purpose: Non-linear transformation                    │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                  DROPOUT LAYER                          │
│                                                         │
│  Rate: 0.2 - 0.3                                       │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                  OUTPUT LAYER                           │
│                                                         │
│  Units: 1                                              │
│  Activation: Softplus (ensures h_t > 0)                │
│  Output: h_t (adaptive threshold)                      │
└─────────────────────────────────────────────────────────┘
```

**Note:** Using Dense layers (not LSTM) because:
- Input is single time point (not sequence)
- Temporal information comes from S_{t-1}
- Simpler and faster than LSTM

---

## CUSUM Computation

### Step-by-Step Process

**At each time point t:**

```python
# 1. Get current observation (already scaled)
x_t = X_test[t, :]  # Shape: (52,)

# 2. Get previous CUSUM statistic
S_prev = S_{t-1}  # Scalar

# 3. Prepare DNN input
dnn_input = np.concatenate([x_t, [S_prev]])  # Shape: (53,)

# 4. Predict adaptive threshold
h_t = DNN(dnn_input)  # Scalar, h_t > 0

# 5. Compute Mahalanobis distance
C_t = (x_t - μ₀)ᵀ Σ⁻¹ (x_t - μ₀)  # Scalar

# 6. Update CUSUM statistic (fixed k)
S_t = max(0, S_prev + C_t - k_fixed)

# 7. Make decision
if S_t > h_t:
    prediction_t = 1  # Anomaly
else:
    prediction_t = 0  # Normal

# 8. Update state for next iteration
S_{t+1-1} = S_t  # Use as input for next time point
```

---

## Mathematical Formulation

### CUSUM Recursion
```
S_t = max(0, S_{t-1} + C_t - k)
```

Where:
- **S_t**: CUSUM statistic at time t
- **S_{t-1}**: Previous CUSUM statistic (feedback to DNN)
- **C_t**: Mahalanobis distance = (x_t - μ₀)ᵀ Σ⁻¹ (x_t - μ₀)
- **k**: Fixed reference value (from training or grid search)

### Adaptive Threshold
```
h_t = DNN(x_t, S_{t-1})
```

Where:
- **h_t**: Adaptive threshold (changes at each time point)
- **x_t**: Current 52-dimensional observation
- **S_{t-1}**: Feedback from CUSUM state

### Decision Rule
```
Alarm_t = {
    1  if S_t > h_t  (Anomaly detected)
    0  otherwise      (Normal)
}
```

---

## Training Strategy

### Training Data Generation

For each time point in training data:

```python
# Input features
x_t = observation at time t (52 dims)
S_{t-1} = CUSUM statistic at time t-1 (1 scalar)

# Target output (heuristic-based)
if has_fault(t):
    h_optimal = low_threshold  # e.g., 2.0-4.0
    # Want quick detection during faults
else:
    h_optimal = high_threshold  # e.g., 6.0-10.0
    # Want to avoid false alarms during normal

# Adjust based on S_{t-1}
if S_{t-1} > threshold:
    h_optimal *= 1.2  # Increase threshold if CUSUM rising
else:
    h_optimal *= 0.8  # Decrease if CUSUM stable
```

### Loss Function
```python
# Mean Squared Error for threshold prediction
loss = MSE(h_predicted, h_optimal)

# Or weighted loss to emphasize fault regions
loss = weighted_MSE(h_predicted, h_optimal, weights)
```

---

## Key Differences from DNN-CUSUM

| Aspect | DNN-CUSUM | DeepT-CUSUM |
|--------|-----------|-------------|
| **Input features** | 312 (6 stats × 52 dims) | 53 (52 raw + S_{t-1}) |
| **Window needed** | Yes (50 time points) | No (single time point) |
| **DNN outputs** | 2 (k_t, h_t) | 1 (h_t only) |
| **Network type** | LSTM (temporal) | Dense (feedforward) |
| **k parameter** | Adaptive | Fixed |
| **Feedback** | No | Yes (S_{t-1}) |
| **Complexity** | High | Medium |
| **Training time** | Longer | Shorter |

---

## Advantages

### 1. Simpler Architecture
- Only 53 input features (vs 312)
- Single output (vs 2)
- Dense network (vs LSTM)
- **Result:** Faster training, less overfitting

### 2. No Sliding Window
- Uses current observation only
- No need to buffer past 50 time points
- **Result:** Lower latency, less memory

### 3. Feedback Mechanism
- DNN sees current CUSUM state
- Can adapt threshold based on alarm urgency
- **Result:** Context-aware decisions

### 4. Statistical Foundation
- k is fixed (statistical guarantee)
- Only h adapts (decision boundary)
- **Result:** More interpretable

### 5. Focused Learning
- Network learns one task (predict h)
- Less hyperparameter space to search
- **Result:** Easier to tune

---

## Potential Challenges

### 1. Fixed k Selection
- Need to choose k before training
- Options:
  - Grid search (like best_k from notebook)
  - Compute from training data
  - Use standard value (e.g., 0.5)

### 2. Warm-up Period
- S_0 = 0 at start
- First predictions might be unstable
- Solution: Use default h for first few points

### 3. Training Heuristics
- Need to define "optimal h" for training
- Based on fault presence and S_{t-1}
- May need empirical tuning

### 4. Feedback Loop Stability
- S_t depends on h_t
- h_{t+1} depends on S_t
- Potential for oscillation
- Solution: Clip h_t to reasonable range

---

## Implementation Plan

### Files to Create

```
src/
├── deept_cusum.py           # Main detector class
├── deept_cusum_viz.py       # Visualization utilities
└── __init__.py              # Updated exports

models/
├── deept_cusum_model.h5     # Trained DNN
├── deept_cusum_config.json  # Network configuration
├── deept_cusum_scaler.pkl   # Feature scaler (for x_t)
└── deept_cusum_params.pkl   # Fixed k, μ₀, Σ
```

### Class Structure

```python
class DeepTCUSUMDetector:
    def __init__(self, k_fixed=None, model_dir='models/'):
        self.k_fixed = k_fixed  # Fixed reference value
        self.dnn_model = None
        self.feature_scaler = StandardScaler()
        self.global_mu_0 = None
        self.global_sigma = None
        self.is_fitted = False

    def build_network(self, config):
        # Build Dense network: 53 → 64 → 32 → 1
        pass

    def _compute_optimal_h(self, x_t, S_prev, has_fault):
        # Heuristic for optimal h during training
        pass

    def _generate_training_data(self, X_incontrol, X_outcontrol):
        # Generate (x_t, S_{t-1}) → h_optimal pairs
        pass

    def fit(self, X_incontrol, X_outcontrol, k_fixed=None):
        # Train DNN or load existing model
        pass

    def predict(self, X_test, return_thresholds=False):
        # Predict with adaptive h_t
        # Return predictions and optionally h_t history
        pass
```

---

## Expected Performance

### Comparison with Other Methods

**Hypothesis:**
```
Method              Detection Rate    ARL1    Training Time
─────────────────────────────────────────────────────────────
Fixed CUSUM         70-85%            10-15   None
DNN-CUSUM          80-95%             5-12    ~40 min
DeepT-CUSUM        75-90%             7-14    ~10-15 min
```

**Why DeepT-CUSUM might perform well:**
- Feedback from S_{t-1} provides context
- Simpler network → less overfitting
- Fixed k → statistical guarantee
- Adaptive h → flexibility

**Why it might underperform DNN-CUSUM:**
- Less information (no window history)
- Can't adapt k (less flexible)
- Dense network (vs LSTM temporal modeling)

---

## Visualization

### Plots to Generate

1. **Threshold Evolution**
   - h_t over time
   - Compare with fixed h
   - Show how h adapts to S_t

2. **CUSUM with Adaptive Threshold**
   - S_t (CUSUM statistic)
   - h_t (adaptive threshold)
   - Fixed h (baseline)
   - Detection points

3. **Feedback Analysis**
   - Scatter: S_{t-1} vs h_t
   - Shows relationship learned by DNN

4. **Comparison Plot**
   - DeepT-CUSUM vs Fixed CUSUM vs DNN-CUSUM
   - Side-by-side detection performance

---

## Research Value

### Novel Contributions

1. **Feedback-based threshold adaptation**
   - First to use CUSUM state as DNN input
   - Simple yet effective

2. **Hybrid approach**
   - Statistical k (theory-based)
   - Learning-based h (data-driven)
   - Best of both worlds

3. **Lightweight architecture**
   - No sliding window needed
   - Single time point input
   - Practical for real-time systems

### Potential Paper Sections

**Title:** "Deep Threshold CUSUM: Feedback-Driven Adaptive Anomaly Detection"

**Abstract highlights:**
- Combines CUSUM with DNN for adaptive thresholds
- Uses CUSUM state as feedback to network
- Simpler than window-based approaches
- Competitive performance with lower complexity

**Keywords:** Anomaly detection, CUSUM, Deep learning, Adaptive threshold, Process monitoring

---

## Next Steps

1. **Implementation**
   - Create `deept_cusum.py` with class structure
   - Implement training data generation
   - Build Dense network architecture

2. **Training**
   - Determine fixed k (use best_k from notebook)
   - Train on Tennessee Eastman data
   - Compare with DNN-CUSUM

3. **Evaluation**
   - Test on all 20 faults
   - Compare ARL0, ARL1 with other methods
   - Generate visualizations

4. **Paper Writing**
   - Document methodology
   - Present experimental results
   - Discuss advantages/limitations

---

## Summary

**DeepT-CUSUM** is a novel hybrid method that:
- **Input:** 52-dim observation + previous CUSUM statistic = 53 features
- **Network:** Simple Dense architecture (not LSTM)
- **Output:** Adaptive threshold h_t
- **CUSUM:** Fixed k, adaptive h_t, feedback loop
- **Advantages:** Simpler, faster, no window needed, feedback-aware
- **Trade-offs:** Less temporal context than DNN-CUSUM

**Ready to implement when you are!** 🚀
