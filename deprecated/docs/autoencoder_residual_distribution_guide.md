# Autoencoder Residual Distribution - Interpretation Guide

## What It Shows

The histogram displays the **reconstruction error distribution** (MSE - Mean Squared Error) for:
- **Training data**: How well the model reconstructs data it was trained on
- **Validation data**: How well the model generalizes to unseen data

## How to Interpret

### 1. Check for Overfitting

**Good (Healthy Model)**:
- Train and validation distributions **overlap significantly**
- This means the model generalizes well to unseen data
- Validation errors slightly higher than training is normal

**Slight Overfitting**:
- Validation distribution shifted slightly to the right
- Normal and acceptable behavior
- Model still generalizes reasonably well

**Severe Overfitting**:
- Large gap between distributions
- Training errors much lower than validation
- Model memorized training data instead of learning patterns
- **Action needed**: Reduce model complexity, add regularization, or get more data

### 2. Model Quality Assessment

**Lower Errors = Better Reconstruction**:
- Peak position indicates typical reconstruction error
- Lower peak = better reconstruction quality
- Compare across different model configurations

**Distribution Width**:
- **Tight distribution** (narrow peak): Consistent, reliable performance
- **Wide distribution** (flat, spread out): Inconsistent, unreliable reconstruction
- **Action if wide**: Increase model capacity or improve data quality

### 3. Anomaly Detection Threshold Selection

**Using the Distribution**:
- Most normal data falls within the main peak
- Set threshold at **95th or 99th percentile** of training errors
- Data with error > threshold = potential anomaly

**Example Threshold Selection**:
```python
train_errors = autoencoder.get_reconstruction_error(X_train)
threshold_95 = np.percentile(train_errors, 95)  # 95th percentile
threshold_99 = np.percentile(train_errors, 99)  # 99th percentile

# Flag anomalies
anomalies = reconstruction_error > threshold_95
```

### 4. Validation Check

**Healthy Model Indicators**:
- Train and validation overlap substantially
- Validation mean slightly higher than training (expected)
- Similar distribution shapes
- No severe separation

**Problem Indicators**:
- Large gap between train and validation
- Validation much wider than training
- Bimodal (two-peaked) distributions

### 5. Outliers and Long Tails

**Long Tail to the Right**:
- Represents samples that are hard to reconstruct
- Could be:
  - Edge cases in your data
  - Noise or corrupted samples
  - Actual anomalies in "normal" data
  - Underrepresented patterns

**Action**:
- Investigate high-error samples
- Consider removing corrupted data
- May need to increase model capacity

## Practical Examples

### Example 1: Normal Autoencoder (Small Latent Dimension)
```
Peak: 0.6-0.7 MSE
Width: Moderate
Train-Val Gap: Small
→ Good for anomaly detection (selective reconstruction)
```

### Example 2: Overfitted Autoencoder (Large Latent Dimension)
```
Peak: 0.2-0.3 MSE (lower = better reconstruction)
Width: Narrow
Train-Val Gap: Small
→ Poor for anomaly detection (reconstructs everything too well)
```

**The Reconstruction Paradox**:
- Better reconstruction ≠ better anomaly detection
- Overfitted models have lower reconstruction errors
- But they fail to distinguish normal from anomalous data
- Optimal latent dimension balances reconstruction and discrimination

## Model Comparison

When comparing multiple autoencoders:

1. **Compare peak positions**: Lower = better reconstruction
2. **Compare widths**: Narrower = more consistent
3. **Compare train-val overlap**: More overlap = better generalization
4. **Compare anomaly detection gap**:
   - Gap = mean(anomaly_errors) - mean(normal_errors)
   - Larger gap = better anomaly detector

## Recommended Actions Based on Plot

| Observation | Interpretation | Action |
|-------------|----------------|--------|
| Train-val overlap | Healthy generalization | Continue as-is |
| Large train-val gap | Overfitting | Reduce complexity, add regularization |
| Very low errors (<0.1) | Possible overfitting | Check if model is too large |
| Very high errors (>2.0) | Underfitting | Increase model capacity |
| Wide distribution | Inconsistent performance | Improve data quality or model |
| Bimodal (two peaks) | Mixed data types | Consider separate models |
| Long right tail | Outliers present | Investigate high-error samples |

## Code Examples

### Generate and Analyze Distribution
```python
# Train autoencoder
autoencoder.fit(X_train, X_val, param_grid=GRID)

# Plot residual distribution
autoencoder.plot_residual_distribution(X_train, X_val)

# Get statistics
train_errors = autoencoder.get_reconstruction_error(X_train)
val_errors = autoencoder.get_reconstruction_error(X_val)

print(f"Train mean: {np.mean(train_errors):.4f}")
print(f"Train std: {np.std(train_errors):.4f}")
print(f"Val mean: {np.mean(val_errors):.4f}")
print(f"Val std: {np.std(val_errors):.4f}")
```

### Compare Multiple Models
```python
# Train multiple autoencoders
ae_small = FlexibleAutoencoder(cache_prefix="ae_small")
ae_large = FlexibleAutoencoder(cache_prefix="ae_large")

ae_small.fit(X_train, X_val, param_grid=SMALL_GRID)
ae_large.fit(X_train, X_val, param_grid=LARGE_GRID)

# Compare distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Small autoencoder
errors_small = ae_small.get_reconstruction_error(X_test)
axes[0].hist(errors_small, bins=50)
axes[0].set_title(f"Small (latent={ae_small.config['latent_dim']})")

# Large autoencoder
errors_large = ae_large.get_reconstruction_error(X_test)
axes[1].hist(errors_large, bins=50)
axes[1].set_title(f"Large (latent={ae_large.config['latent_dim']})")

plt.tight_layout()
plt.show()
```

## Key Takeaways

1. **Overlap is good**: Train and validation should overlap
2. **Lower is better for reconstruction**: But not always better for anomaly detection
3. **Use for threshold selection**: 95th-99th percentile of training errors
4. **Check regularly**: Plot after every major model change
5. **Compare configurations**: Use plots to select optimal hyperparameters
