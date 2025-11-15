# Debug Checklist - 0% Detection Rate Issue

## Current Problem

**Symptom:** DNN-CUSUM reports 0.00% detection rate, ARL1 = None

**Expected:** Detection rate should be 50-95%, ARL1 should be < 50

---

## Quick Diagnosis Steps

### Step 1: Run Debug Cell ⚡

Add and run this cell after Cell 25 (testing):

```python
# Debug: Check what k and h values are being predicted
print("="*60)
print("DNN-CUSUM DEBUG ANALYSIS")
print("="*60)

# Check parameter ranges
k_values = dnn_param_history['k']
h_values = dnn_param_history['h']
cusum_stats = dnn_param_history['cusum_stat']

print(f"\nPredicted k values:")
print(f"  Min: {np.min(k_values):.4f}")
print(f"  Max: {np.max(k_values):.4f}")
print(f"  Mean: {np.mean(k_values):.4f}")
print(f"  Median: {np.median(k_values):.4f}")

print(f"\nPredicted h values:")
print(f"  Min: {np.min(h_values):.4f}")
print(f"  Max: {np.max(h_values):.4f}")
print(f"  Mean: {np.mean(h_values):.4f}")
print(f"  Median: {np.median(h_values):.4f}")

print(f"\nCUSUM statistic values:")
print(f"  Min: {np.min(cusum_stats):.4f}")
print(f"  Max: {np.max(cusum_stats):.4f}")
print(f"  Mean: {np.mean(cusum_stats):.4f}")
print(f"  Median: {np.median(cusum_stats):.4f}")

print(f"\nDetection analysis:")
print(f"  Times S_t exceeds h_t: {np.sum(np.array(cusum_stats) > np.array(h_values))}")
print(f"  Max ratio S_t/h_t: {np.max(np.array(cusum_stats) / np.array(h_values)):.4f}")

# Compare with fixed MCUSUM baseline
print(f"\n{'='*60}")
print("COMPARISON WITH FIXED MCUSUM")
print("="*60)
print(f"Best fixed k: {best_k:.4f}")
print(f"Best fixed h: {best_h:.4f}")
```

---

## Scenario Analysis

### Scenario 1: k and h Too High 🔴

**Indicators:**
- Mean k > 5.0 (vs best_k ~ 0.1-2.0)
- Mean h > 10.0 (vs best_h ~ 2.0-5.0)
- Max S_t << Mean h_t (CUSUM never reaches threshold)

**Root Cause:** Training heuristics predicting too conservative parameters

**Fix:**
1. Edit `src/dnn_cusum.py` line 177-178 (normal regions):
   ```python
   # BEFORE:
   k = base_k * 1.5 * (1.0 + 0.05 * volatility)
   h = base_h * 1.2 * (1.0 + 0.05 * magnitude)

   # AFTER (more sensitive):
   k = base_k * 0.8 * (1.0 + 0.05 * volatility)
   h = base_h * 0.6 * (1.0 + 0.05 * magnitude)
   ```

2. Retrain:
   ```python
   dnn_cusum.fit(
       X_INCONTROL_TRAIN_FULL_SCALED,
       X_OUT_OF_CONTROL_TRAIN_PLAY_SCALED,
       force_retrain=True,  # Force retrain
       grid_search=False    # Skip grid search (faster)
   )
   ```

3. Test again

---

### Scenario 2: Features Not Informative 🟡

**Indicators:**
- k and h are constant (no variation over time)
- Mean k ≈ median k (no adaptation)
- DNN not learning patterns

**Root Cause:** Features don't capture fault characteristics

**Fix:**
1. Check feature extraction:
   ```python
   # Add to debug cell:
   window = X_OUT_OF_CONTROL_TEST_PLAY_SCALED_CUT[100:150]
   features = dnn_cusum._extract_features(window)
   print(f"Sample features: {features[:12]}")  # First 2 dims
   ```

2. Try larger window:
   ```python
   dnn_cusum = DNNCUSUMDetector(window_size=70)  # Was 50
   ```

3. Retrain with larger window

---

### Scenario 3: CUSUM Logic Bug 🔴

**Indicators:**
- k and h look reasonable
- S_t (CUSUM statistic) is always 0 or very small
- S_t/h_t ratio << 1

**Root Cause:** Bug in predict() method

**Fix:**
1. Check CUSUM recursion in `src/dnn_cusum.py` lines 530-545
2. Verify:
   ```python
   # Should be:
   cusum_stat = max(0, prev_cusum + mahal_dist - k_t)

   # NOT:
   cusum_stat = max(0, prev_cusum - mahal_dist - k_t)  # Wrong sign!
   ```

3. Fix and reload module

---

### Scenario 4: Wrong Global Statistics 🟠

**Indicators:**
- Mahalanobis distance (C_t) is always very small
- CUSUM doesn't rise even during obvious faults

**Root Cause:** μ₀ or Σ computed incorrectly

**Fix:**
1. Verify global statistics:
   ```python
   print(f"Global mu_0 shape: {dnn_cusum.global_mu_0.shape}")
   print(f"Global Sigma shape: {dnn_cusum.global_sigma.shape}")
   print(f"Sample mu_0: {dnn_cusum.global_mu_0[:5]}")
   ```

2. Should be:
   - mu_0 shape: (52,)
   - Sigma shape: (52, 52)

3. If wrong, retrain from scratch

---

### Scenario 5: Softplus Saturation 🟡

**Indicators:**
- k and h are VERY large (> 20)
- DNN outputs are saturated

**Root Cause:** Softplus activation allowing unbounded outputs

**Fix:**
1. Add output clipping in `src/dnn_cusum.py` after DNN prediction:
   ```python
   # In predict() method, after DNN predicts k and h:
   k_t = np.clip(k_pred, 0.1, 5.0)  # Add reasonable bounds
   h_t = np.clip(h_pred, 1.0, 10.0)
   ```

2. Or change training targets to be in [0, 1] range, then scale

---

## Decision Tree

```
0% Detection Rate
    |
    ├─> k, h too high? (> 5.0)
    |   └─> FIX: Adjust training heuristics (Scenario 1)
    |
    ├─> S_t always small? (< 1.0)
    |   ├─> Check CUSUM logic (Scenario 3)
    |   └─> Check global statistics (Scenario 4)
    |
    ├─> k, h not adapting? (constant over time)
    |   └─> FIX: Improve features or window size (Scenario 2)
    |
    └─> k, h extremely high? (> 20)
        └─> FIX: Add output clipping (Scenario 5)
```

---

## Quick Tests

### Test 1: Sanity Check with Fixed Parameters

```python
# Use DNN-CUSUM with fixed k=0.5, h=3.0 to verify CUSUM logic works
# Manually override in predict() or test MCUSUMDetector separately
```

### Test 2: Check Training Data Quality

```python
# Verify training generated reasonable targets
X_feat, y_k, y_h = dnn_cusum._generate_training_data(
    X_INCONTROL_TRAIN_FULL_SCALED[:1000],
    X_OUT_OF_CONTROL_TRAIN_PLAY_SCALED
)

print(f"Training k range: [{y_k.min():.2f}, {y_k.max():.2f}]")
print(f"Training h range: [{y_h.min():.2f}, {y_h.max():.2f}]")
print(f"Training k mean: {y_k.mean():.2f}")
print(f"Training h mean: {y_h.mean():.2f}")
```

Expected:
- k range: [0.1, 5.0] or similar
- h range: [1.0, 10.0] or similar
- Not all the same value

### Test 3: Check DNN Is Learning

```python
# Plot training history (if saved)
# Should see loss decreasing over epochs
```

---

## Data to Collect

For new agent to debug, collect:

1. **Parameter Statistics:**
   - [ ] Mean, median, min, max of predicted k
   - [ ] Mean, median, min, max of predicted h
   - [ ] Mean, median, min, max of CUSUM statistic

2. **Comparison Data:**
   - [ ] Fixed MCUSUM best_k and best_h values
   - [ ] Fixed MCUSUM detection rate on same data

3. **Training Data:**
   - [ ] Training k range (from _generate_training_data)
   - [ ] Training h range
   - [ ] Number of training samples generated

4. **Visualization:**
   - [ ] Plot of k over time
   - [ ] Plot of h over time
   - [ ] Plot of S_t vs h_t

---

## Expected Values (Reference)

### Fixed MCUSUM (Baseline)

From notebook Cell 11-12:
- best_k: ~0.1 - 2.0 (low for sensitivity)
- best_h: ~2.0 - 5.0 (low for quick detection)

### DNN-CUSUM (Expected)

Should adapt around baseline:
- k during normal: 1.5 - 3.0 (higher)
- k during fault: 0.1 - 0.5 (lower)
- h during normal: 5.0 - 8.0 (higher)
- h during fault: 2.0 - 4.0 (lower)

### CUSUM Statistic

- S_t during normal: ~0 - 2.0
- S_t during fault: rises to > h_t (e.g., 5.0 - 15.0)

---

## Priority Actions

1. ⚡ **FIRST:** Run debug cell, collect parameter statistics
2. 🔍 **ANALYZE:** Compare with expected values above
3. 🛠️ **FIX:** Apply appropriate scenario fix
4. ✅ **VERIFY:** Retest, should see detection rate > 50%

---

## Success Criteria

After fix, should see:
- ✅ Detection rate: 50-95%
- ✅ ARL1: 5-20 (reasonable delay)
- ✅ Parameters adapt over time (k_t and h_t change)
- ✅ S_t exceeds h_t during fault period

---

## Files to Check

If modifying code:
- `src/dnn_cusum.py` lines 146-189 (training heuristics)
- `src/dnn_cusum.py` lines 479-560 (predict method)
- `src/dnn_cusum.py` lines 64-112 (network architecture)

If reloading:
- Run Cell 23 (import with reload)
- Rerun Cell 24 (training)
- Rerun Cell 25 (testing)

---

**Status:** Ready for debugging
**Next:** User runs debug cell and provides output
