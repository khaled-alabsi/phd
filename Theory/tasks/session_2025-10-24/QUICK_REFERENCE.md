# Quick Reference - Session Oct 24, 2025

## TL;DR

**What happened:** Continued DNN-CUSUM integration, fixed critical bug, but now getting 0% detection rate.

**Current issue:** DNN-CUSUM not detecting any anomalies (0% detection, ARL1=None)

**Next step:** Debug predicted k/h values to identify root cause

---

## Key Files

### Session Documentation
- **`SESSION_SUMMARY.md`** ⭐ - Complete history (20+ pages)
- **`DEBUG_CHECKLIST.md`** - Debugging guide for 0% detection
- **`QUICK_REFERENCE.md`** - This file

### Code Files
- **`code_v2/src/dnn_cusum.py`** - Main implementation (MODIFIED)
- **`code_v2/anomaly_detection.ipynb`** - Notebook (Cell 23, 30, 31 MODIFIED)

### Documentation
- **`code_v2/DNN_CUSUM_QA.md`** ⭐ - 20 Q&A with diagrams
- **`code_v2/INTEGRATION_COMPLETE.md`** - Integration status
- **`code_v2/BUG_FIX_SUMMARY.md`** - Recent bug fix

---

## What Was Fixed This Session

### Bug 1: Feature Dimension Mismatch ✅
**Problem:** `TypeError: 'NoneType' object is not subscriptable`
**Cause:** `window_size * 6 = 300` but actual features = `52 * 6 = 312`
**Fix:** Added `n_dims` attribute, fixed reshape to `(n_dims, 6)`

### Bug 2: Grid Search Too Slow ✅
**Problem:** 24 configs × 10 min = 4 hours
**Fix:** Reduced to 4 configs (~40 min)

### Bug 3: Module Reloading ✅
**Problem:** `importlib.reload(DNNCUSUMDetector)` failed
**Fix:** Reload module first: `importlib.reload(src.dnn_cusum)`

### Bug 4: Notebook Integration Incomplete ✅
**Problem:** DNN-CUSUM not in MODELS dictionary
**Fix:** Added Cell 30 (wrapper) and updated Cell 31 (MODELS dict)

---

## Current Issue: 0% Detection Rate ⚠️

### Symptoms
```python
DNN-CUSUM Results:
  Detection Delay (ARL1): None
  Detection Rate: 0.00%
```

### Likely Causes
1. **k and h too high** (DNN being too conservative)
2. **Training heuristics wrong** (targets in wrong range)
3. **CUSUM logic bug** (S_t computation error)
4. **Global statistics wrong** (μ₀ or Σ incorrect)

### Debug Process
1. Run debug cell (see `DEBUG_CHECKLIST.md`)
2. Check predicted k/h ranges
3. Compare with fixed MCUSUM baseline (best_k, best_h)
4. Apply appropriate fix

---

## Code Locations

### Training Heuristics (Most Likely Issue)
**File:** `src/dnn_cusum.py`
**Lines:** 146-189
**Method:** `_compute_optimal_params()`

Currently:
```python
# Normal regions:
k = base_k * 1.5 * (1.0 + 0.05 * volatility)  # Line 181
h = base_h * 1.2 * (1.0 + 0.05 * magnitude)    # Line 182

# Fault regions:
k = base_k * 0.3 * (1.0 + 0.1 * magnitude)     # Line 177
h = base_h * 0.6 * (1.0 + 0.1 * volatility)    # Line 178
```

**Potential Fix:** Lower the multipliers (1.5 → 0.8, 1.2 → 0.6)

### CUSUM Prediction Logic
**File:** `src/dnn_cusum.py`
**Lines:** 479-560
**Method:** `predict()`

Key recursion (around line 530):
```python
cusum_stat = max(0, prev_cusum + mahal_dist - k_t)
```

Verify this is correct (not `- mahal_dist`)

### Network Architecture
**File:** `src/dnn_cusum.py`
**Lines:** 64-112
**Method:** `build_network()`

Current fix:
```python
n_features = self.n_dims * 6  # Line 79 (FIXED)
x = layers.Reshape((self.n_dims, 6))(inputs)  # Line 85 (FIXED)
```

---

## Notebook Cells Modified

### Cell 23: Import with Reload
```python
import importlib
import src.dnn_cusum
import src.dnn_cusum_viz

importlib.reload(src.dnn_cusum)
importlib.reload(src.dnn_cusum_viz)

from src.dnn_cusum import DNNCUSUMDetector
from src.dnn_cusum_viz import DNNCUSUMVisualizer
```

### Cell 30: Wrapper Function (NEW)
```python
def dnn_cusum_predict(x_scaled):
    preds, _ = dnn_cusum.predict(x_scaled, return_params=False)
    return preds.astype(int)
```

### Cell 31: MODELS Dictionary (UPDATED)
```python
MODELS = {
    "MCUSUM": mcusum_predict,
    "Autoencoder": autoencoder_detector.predict,
    "AutoencoderEnhanced": autoencoder_detector_enhanced.predict,
    "MEWMA": mewma.predict,
    "DNN_CUSUM": dnn_cusum_predict  # ← ADDED
}
```

---

## User Context

### User Profile
- **Working on:** PhD research
- **Dataset:** Tennessee Eastman Process (52 dims, 20 faults)
- **Goal:** Compare DNN-CUSUM with other anomaly detectors
- **Deliverable:** IEEE conference paper

### User Preferences
- Small incremental steps
- Wants to understand "why"
- Values documentation and diagrams
- Avoids kernel restarts
- Doesn't want too many files

### Technical Level
- Strong Python knowledge
- Understands ML and anomaly detection
- Comfortable with debugging
- Asks good clarifying questions

---

## Expected Performance

### Fixed MCUSUM (Baseline)
From Cell 11-12:
- best_k: ~0.1 - 2.0
- best_h: ~2.0 - 5.0
- Detection rate: 70-90%
- ARL1: 5-15

### DNN-CUSUM (Target)
Should match or exceed baseline:
- Detection rate: 80-95%
- ARL1: 3-10 (faster than fixed)
- k adapts: 0.1-3.0 range
- h adapts: 2.0-8.0 range

---

## Debug Cell to Add

Add after Cell 25:

```python
# Debug: Check predicted parameters
k_values = dnn_param_history['k']
h_values = dnn_param_history['h']
cusum_stats = dnn_param_history['cusum_stat']

print(f"Predicted k: min={np.min(k_values):.2f}, max={np.max(k_values):.2f}, mean={np.mean(k_values):.2f}")
print(f"Predicted h: min={np.min(h_values):.2f}, max={np.max(h_values):.2f}, mean={np.mean(h_values):.2f}")
print(f"CUSUM stat: min={np.min(cusum_stats):.2f}, max={np.max(cusum_stats):.2f}")
print(f"Fixed k={best_k:.2f}, h={best_h:.2f}")
print(f"Detections: {np.sum(np.array(cusum_stats) > np.array(h_values))}")
```

---

## Next Actions (Priority)

1. **⚡ IMMEDIATE:** User runs debug cell
2. **🔍 ANALYZE:** Check if k/h too high
3. **🛠️ FIX:** Adjust training heuristics if needed
4. **🔄 RETRAIN:** With `force_retrain=True`
5. **✅ VERIFY:** Detection rate should be > 50%

---

## Quick Fixes

### If k/h too high (most likely):

**Edit:** `src/dnn_cusum.py` line 181-182

```python
# Change from:
k = base_k * 1.5 * (1.0 + 0.05 * volatility)
h = base_h * 1.2 * (1.0 + 0.05 * magnitude)

# To:
k = base_k * 0.8 * (1.0 + 0.05 * volatility)
h = base_h * 0.6 * (1.0 + 0.05 * magnitude)
```

**Then:**
```python
# Cell 23: Reload
importlib.reload(src.dnn_cusum)
from src.dnn_cusum import DNNCUSUMDetector

# Cell 24: Retrain
dnn_cusum = DNNCUSUMDetector(window_size=50, model_dir='models/')
dnn_cusum.fit(
    X_INCONTROL_TRAIN_FULL_SCALED,
    X_OUT_OF_CONTROL_TRAIN_PLAY_SCALED,
    force_retrain=True,   # Force retrain
    grid_search=False     # Skip grid search for speed
)

# Cell 25: Test again
```

---

## Git Changes This Session

### Modified
- `code_v2/src/dnn_cusum.py`
  - Added `n_dims` attribute
  - Fixed `build_network()` reshape
  - Optimized grid search (24→4 configs)
  - Reduced epochs (30→20)

- `code_v2/anomaly_detection.ipynb`
  - Cell 23: Fixed reload
  - Cell 30: Added wrapper (NEW)
  - Cell 31: Updated MODELS dict

### Created
- `code_v2/INTEGRATION_COMPLETE.md`
- `code_v2/BUG_FIX_SUMMARY.md`
- `code_v2/DNN_CUSUM_QA.md` ⭐
- `tasks/session_2025-10-24/SESSION_SUMMARY.md`
- `tasks/session_2025-10-24/DEBUG_CHECKLIST.md`
- `tasks/session_2025-10-24/QUICK_REFERENCE.md`

---

## Important Constants

```python
# Dataset
n_dims = 52                    # Tennessee Eastman dimensions
fault_injection_point = 160    # Where fault starts
window_size = 50               # Default window for DNN-CUSUM

# Training data sizes
X_INCONTROL_TRAIN_FULL_SCALED.shape    # (250000, 52)
X_OUT_OF_CONTROL_TRAIN_PLAY_SCALED.shape  # (500, 52)

# Testing data sizes
X_OUT_OF_CONTROL_TEST_PLAY_SCALED_CUT.shape  # (800, 52)

# Feature extraction
n_features_per_dim = 6         # [mean, std, range, change, rate, autocorr]
total_features = 52 * 6 = 312  # Input to DNN
```

---

## Contact Info

**User:** Khaled Alabsi
**Project:** PhD Research - Anomaly Detection
**Environment:** Jupyter Notebook, Python 3.11
**Dataset:** Tennessee Eastman Process

---

## Success Indicators

### Completed ✅
- Training runs without errors
- Model saves successfully
- Notebook integration complete

### Current Issue ⚠️
- 0% detection rate (need to debug)

### Target Metrics 🎯
- Detection rate: 70-95%
- ARL1: < 20
- Better than fixed MCUSUM

---

**Last Updated:** Oct 24, 2025
**Status:** Waiting for debug output
**Next Agent:** Start with debug cell, analyze k/h values
