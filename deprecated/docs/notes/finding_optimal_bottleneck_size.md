# Finding the Optimal Bottleneck Size for Autoencoders

## The Question

**How small can the bottleneck be while still preserving useful information?**

---

## Why Bottleneck Size Matters

The bottleneck is the **information constraint** of your autoencoder:

- **Too large:** Model memorizes everything (even noise) → Poor generalization
- **Too small:** Model loses important information → Can't reconstruct normal patterns
- **Just right:** Model learns compressed representation of essential patterns

---

## Best Practices for Finding Optimal Size

### 1. **The Theoretical Lower Bound**

**Minimum possible bottleneck:**
```
bottleneck_min = rank(X)  # Rank of your data matrix
```

For most real-world data with 52 features, the intrinsic dimensionality is usually **much lower** than 52.

**Practical minimum:** Start testing from ~**5-10% of input dimensions**

For 52 input features:
- Minimum to test: **3-5 dimensions**
- Maximum to test: **26 dimensions** (50%)
- Sweet spot often: **8-16 dimensions**

---

### 2. **Grid Search Strategy**

#### Method A: Coarse-to-Fine Search

**Step 1:** Wide range with large steps
```python
bottleneck_sizes = [4, 8, 16, 32, 64]  # Exponential spacing
```

**Step 2:** Narrow down based on results
```python
# If 8 was best, test around it:
bottleneck_sizes = [6, 7, 8, 9, 10, 12]
```

#### Method B: Logarithmic Search

```python
import numpy as np
bottleneck_sizes = [2**i for i in range(2, 7)]  # [4, 8, 16, 32, 64]
```

---

### 3. **Evaluation Metrics**

Test each bottleneck size using multiple metrics:

#### A. Reconstruction Quality (on validation set)

```python
metrics = {
    "mse": Mean Squared Error,
    "mae": Mean Absolute Error,
    "r2": R² score (goodness of fit)
}
```

**Plot:** Bottleneck size vs reconstruction error

```
Error
  |
  |     ╱
  |    ╱
  |   ╱
  |  ╱
  | ╱________
  |
  +-----------> Bottleneck Size
  4  8  16  32
  ↑ Elbow point = optimal size
```

#### B. Anomaly Detection Performance

```python
metrics = {
    "arl0": False alarm rate (should be high)
    "arl1": Detection speed (should be low)
    "f1_score": Overall detection quality
    "gap": abs(normal_error - anomaly_error)  # Should be large!
}
```

**Key metric:** Gap between normal and anomaly errors

```python
gap = mean(anomaly_errors) - mean(normal_errors)
# Maximize this!
```

#### C. Information Preservation

**Explained variance ratio:**
```python
# After training autoencoder
latent_features = encoder.predict(X_train)
variance_explained = np.var(latent_features) / np.var(X_train)
```

**Target:** At least 80-90% variance explained

---

### 4. **The Elbow Method**

Plot reconstruction error vs bottleneck size:

```
Reconstruction Error
     |
 0.9 |╲
     | ╲
 0.7 |  ╲
     |   ╲___________
 0.5 |
     +-------------------> Bottleneck Size
     2  4  8  16 32 64
           ↑ Elbow point
```

**Optimal size:** Where the curve "bends" (diminishing returns)

Beyond this point, increasing size gives minimal improvement.

---

### 5. **Practical Experimentation Workflow**

#### Step 1: Baseline Test (Wide Range)
```python
BOTTLENECK_SEARCH_GRID = {
    "latent_dim": [4, 8, 16, 32, 64],  # Wide range
    "encoder_layers": [(256, 128)],     # Fixed architecture
    "learning_rate": [1e-3],            # Fixed
    "epochs": [120],                    # Fixed
}
```

**Evaluate:** Record MSE, MAE, F1 for each size

#### Step 2: Analyze Results
```python
results = {
    4:  {"mse": 0.95, "f1": 0.60},
    8:  {"mse": 0.70, "f1": 0.75},
    16: {"mse": 0.55, "f1": 0.78},
    32: {"mse": 0.45, "f1": 0.76},
    64: {"mse": 0.40, "f1": 0.72},
}
# Best F1 at 16, but 8 is close and more constrained
```

#### Step 3: Fine-Tune Around Best
```python
REFINED_GRID = {
    "latent_dim": [6, 8, 10, 12, 16],  # Around best value
    "encoder_layers": [
        (256, 128),
        (512, 256, 128),  # Try different architectures too
    ],
}
```

---

### 6. **Rules of Thumb**

#### For Anomaly Detection (Your Use Case):

**Start with:** `bottleneck = input_dim / 6`
```python
input_dim = 52
bottleneck_start = 52 // 6 ≈ 8  # Good starting point!
```

**Test range:** `[input_dim / 10, input_dim / 4]`
```python
test_range = [5, 8, 10, 13, 16]  # For 52 features
```

#### For Feature Extraction (Residual Regressor):

**Start with:** `bottleneck = input_dim / 2` to `input_dim`
```python
# More preservation needed
test_range = [26, 32, 48, 64]
```

---

### 7. **Architecture Scaling Rules**

When testing bottleneck sizes, scale the architecture accordingly:

**Rule:** Hidden layers should smoothly interpolate between input and bottleneck

```python
def get_encoder_layers(input_dim, bottleneck_dim):
    # Geometric progression
    layer1 = int((input_dim * bottleneck_dim) ** 0.5)
    return (layer1,)

# Examples:
input_dim = 52
# bottleneck=4  → layers=(14,)  → 52→14→4
# bottleneck=8  → layers=(20,)  → 52→20→8
# bottleneck=16 → layers=(28,)  → 52→28→16
# bottleneck=64 → layers=(57,)  → 52→57→64 (needs expansion!)
```

**For very small bottlenecks:**
```python
# 52 → 32 → 16 → 8 → 4
encoder_layers = (32, 16, 8) if bottleneck < 8 else (128, 64)
```

---

### 8. **Residual Regressor Specific Considerations**

For `AutoencoderResidualRegressor`, you want **different** bottleneck than standalone:

#### Current (Large Bottleneck):
```python
latent_dim = 64
encoder_layers = (256, 128)
# Purpose: Good feature extraction + good reconstructions
```

#### Proposed (Tiny Bottleneck):
```python
latent_dim = 4-8
encoder_layers = (128, 64, 32) or (64, 32)
# Purpose: Force compression → larger reconstruction errors
```

**Why this might be better:**

1. **Larger base errors** → Easier for regressor to learn patterns
2. **More compression** → Anomalies create even larger errors
3. **Better discrimination** → Bigger gap between normal and anomaly errors

**Trade-off:**
- ✅ Better anomaly detection potential
- ⚠️ Regressor needs to work harder (but it should be able to)

---

### 9. **Validation Strategy**

**Critical:** Use held-out test set for final evaluation!

```python
# Split 1: Train autoencoder
X_ae_train, X_ae_val = train_test_split(X_train, test_size=0.2)

# Split 2: Train regressor
X_reg_train, X_reg_val = train_test_split(X_train, test_size=0.2)

# Split 3: Final evaluation (UNTOUCHED during tuning)
X_final_test = X_test  # Only use once!
```

**Evaluate on all three:**
1. Autoencoder validation loss
2. Regressor validation MAE
3. Final test set F1/ARL metrics

---

### 10. **Warning Signs**

#### Bottleneck Too Small:
- ❌ Reconstruction MSE doesn't decrease during training
- ❌ Validation loss explodes
- ❌ Can't reconstruct even normal patterns (F1 near 0)
- ❌ ARL0 < 5 (too many false alarms)

#### Bottleneck Too Large:
- ❌ Overfitting (train loss << val loss)
- ❌ Poor anomaly detection (small gap between normal/anomaly errors)
- ❌ F1 score peaks at smaller sizes then decreases
- ❌ ARL1 >> 10 (slow detection)

---

## Recommended Experiment for Residual Regressor

### Current State:
```python
latent_dim = 64  # Large, good reconstructions
```

### Proposed Search:
```python
BOTTLENECK_GRID = {
    "latent_dim": [4, 6, 8, 12, 16, 24, 32, 64],  # Include current
    "encoder_layers": [
        (128, 64),      # For small bottlenecks
        (256, 128),     # Current
        (512, 256),     # For large bottlenecks
    ],
}
```

### Expected Results:
- **4-8:** Higher reconstruction errors, potentially better detection
- **16-32:** Balanced trade-off
- **64:** Current performance (baseline)

**Hypothesis:** Smaller bottleneck (8-16) might improve overall detection by increasing the error signal for the regressor to learn from.

---

## Step-by-Step Action Plan (When Ready to Implement)

1. **Setup experiment:**
   - Grid search over bottleneck sizes [4, 6, 8, 12, 16, 24, 32, 64]
   - Fixed other hyperparameters initially

2. **Train autoencoders:**
   - For each bottleneck size
   - Record: train_loss, val_loss, reconstruction errors

3. **Train regressors:**
   - Use each autoencoder
   - Record: MAE, RMSE, R² on predicting reconstruction errors

4. **Evaluate anomaly detection:**
   - Test on fault data
   - Record: ARL0, ARL1, F1, Precision, Recall

5. **Analyze results:**
   - Plot elbow curve (size vs metrics)
   - Find optimal based on F1 or ARL1
   - Check if gap increases with smaller bottleneck

6. **Select final model:**
   - Balance between reconstruction quality and detection performance
   - Typically: smallest bottleneck with acceptable F1 (>0.7)

---

## Summary

**For Residual Regressor Autoencoder:**

| Goal | Recommended Bottleneck | Reasoning |
|------|----------------------|-----------|
| **Current** | 64 | Good reconstructions, feature extraction |
| **Tiny (aggressive)** | 4-6 | Maximum compression, largest errors |
| **Small (recommended)** | 8-12 | Good balance, proven effective |
| **Medium** | 16-24 | Safe middle ground |

**Best practice:** Start with 8, test [4, 6, 8, 12, 16], pick based on final F1 score.

**Expected winner:** Probably **8** (same as your basic/enhanced autoencoders) - proven to work well!
