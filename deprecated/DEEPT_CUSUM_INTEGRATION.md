# DeepT-CUSUM Integration Complete!

## What Was Added to the Notebook

Successfully integrated **DeepT-CUSUM (Deep Threshold CUSUM)** into `anomaly_detection.ipynb`.

### New Cells Added

**8 new cells** inserted after the DNN-CUSUM section:

1. **Markdown Header** - "## DeepT-CUSUM (Deep Threshold CUSUM)"

2. **Import Cell** - Loads DeepT-CUSUM modules with reload support
   ```python
   from src.deept_cusum import DeepTCUSUMDetector
   from src.deept_cusum_viz import DeepTCUSUMVisualizer
   ```

3. **Training Cell** - Initializes and trains DeepT-CUSUM
   - Uses `best_k` from MCUSUM grid search (fixed reference value)
   - Trains on full dataset for fair comparison
   - Grid search for optimal DNN architecture
   - Auto-saves model to `models/` directory

4. **Testing Cell** - Predicts on fault data with 4-subplot visualization
   - Plots threshold evolution (h_t over time)
   - Shows CUSUM statistic vs adaptive threshold
   - Displays binary predictions
   - Plots sample feature with anomalies

5. **Feedback Analysis Cell** - 4 subplots analyzing S_{t-1} → h_t relationship
   - Scatter plot of S_{t-1} vs h_t
   - Threshold distributions (normal vs anomaly)
   - CUSUM distributions
   - Threshold adaptation rate

6. **Comparison Cell** - Compares DeepT-CUSUM with Fixed MCUSUM (4 subplots)
   - Detection comparison
   - Agreement/disagreement analysis
   - Threshold comparison (adaptive vs fixed)
   - Performance metrics

7. **Statistics Cell** - 4 subplots of threshold statistics
   - Boxplot by prediction class
   - Ratio S_t/h_t distribution
   - Running average smoothness
   - Statistics summary

8. **Wrapper Function** - Adds `deept_cusum_predict()` for model comparison

### Updated Cells

**MODELS Dictionary** - Now includes DeepT-CUSUM:
```python
MODELS = {
    "MCUSUM": mcusum_predict,
    "DNN_CUSUM": dnn_cusum_predict,
    "DeepT_CUSUM": deept_cusum_predict  # NEW!
}
```

## How to Use

### Quick Start

1. **Run the new cells in order** (import → training → testing → plots)
   - First run will perform grid search (~15-20 minutes)
   - Subsequent runs load instantly (< 1 second)

2. **View the plots** - 16 total subplots across 4 figures:
   - Threshold evolution (how h_t changes over time)
   - Feedback analysis (relationship between S_{t-1} and h_t)
   - Comparison with Fixed CUSUM
   - Statistical analysis

3. **Run full comparison** - DeepT-CUSUM automatically included in:
   - 20 simulation runs × 20 fault types × 3 models = 1,200 experiments
   - Results appear in `df_results` and `summary_df`

### Expected Output

**First Run (with grid search):**
```
Initializing DeepT-CUSUM detector...
Using fixed k = 6.4000 (from MCUSUM grid search)

Starting Grid Search for Best Configuration
============================================================
[1/8] Testing config: Dense[64,32], lr=0.001, bs=32, dropout=0.2
  Validation Loss: X.XXXX
  *** New best configuration! ***
...
[8/8] Testing config: Dense[128,64], lr=0.0005, bs=64, dropout=0.3
  Validation Loss: X.XXXX

Best configuration found: Dense[64,32], lr=0.001, bs=32, dropout=0.2
Training final model...
Model saved to: models/deept_cusum_model.h5

Testing DeepT-CUSUM on fault data...
DeepT-CUSUM Results:
  Detection Delay (ARL1): X
  Detection Rate: XX.XX%
  Fixed k: 6.4000
  Adaptive h_t range: [X.XX, X.XX]
```

**Subsequent Runs:**
```
Found existing trained model. Loading...
Model loaded from: models/deept_cusum_model.h5
Testing DeepT-CUSUM on fault data...
```

## Architecture Details

### Key Differences from DNN-CUSUM

| Feature | DNN-CUSUM | DeepT-CUSUM |
|---------|-----------|-------------|
| **Input** | 312 features (52 dims × 6 stats) from sliding window | 53 features (52 raw + S_{t-1}) |
| **Network** | LSTM (temporal sequences) | Dense (single time point) |
| **k (reference)** | Adaptive (DNN predicts) | Fixed (from grid search) |
| **h (threshold)** | Adaptive (DNN predicts) | Adaptive (DNN predicts) |
| **Feedback** | No | Yes (S_{t-1} as input) |
| **Window size** | 50 samples | N/A (single point) |

### Feedback Mechanism

The key innovation:
```
Time t-1: Compute S_{t-1}
Time t:   DNN input = [x_t, S_{t-1}] → h_t
          C_t = Mahalanobis(x_t)
          S_t = max(0, S_{t-1} + C_t - k)
          if S_t > h_t: ALARM

Time t+1: DNN input = [x_{t+1}, S_t] → h_{t+1}
          ...
```

The DNN "sees" the alarm state and adjusts the threshold accordingly:
- When S is low (normal): h can be moderate
- When S is rising (potential fault): h can lower for quick detection
- When S is high but no fault: h can increase to avoid false alarm

## Files Created/Modified

### Created
- `code_v2/src/deept_cusum.py` (~500 lines) - Main detector
- `code_v2/src/deept_cusum_viz.py` (~400 lines) - 5 plotting functions
- `code_v2/DeepT_CUSUM_ARCHITECTURE.md` - Architecture specification

### Modified
- `code_v2/src/__init__.py` - Added DeepT-CUSUM exports
- `code_v2/anomaly_detection.ipynb` - Added 8 cells, updated MODELS dict

### Will Be Created (after first run)
- `models/deept_cusum_model.h5` (~500 KB) - Trained DNN
- `models/deept_cusum_best_config.json` (< 1 KB) - Best architecture
- `models/deept_cusum_model_scaler.pkl` (< 1 KB) - Feature scaler
- `models/deept_cusum_model_params.pkl` (< 1 KB) - Global statistics

## Visualization Summary

You'll get **16 beautiful subplots** across 4 figures:

### Figure 1: Threshold Evolution (4 subplots)
1. Adaptive h_t over time (with fixed h for comparison)
2. CUSUM S_t vs h_t (with shaded "alarm zone")
3. Binary predictions (scatter)
4. Sample feature with detected anomalies

### Figure 2: Feedback Analysis (4 subplots)
1. Scatter: S_{t-1} vs h_t (colored by prediction)
2. Threshold distribution (normal vs anomaly)
3. CUSUM distribution (normal vs anomaly)
4. Threshold adaptation rate |Δh|

### Figure 3: Comparison with Fixed CUSUM (4 subplots)
1. Detection comparison (scatter)
2. Agreement/disagreement analysis
3. Threshold comparison (adaptive vs fixed)
4. Performance metrics (text summary)

### Figure 4: Threshold Statistics (4 subplots)
1. Boxplot by prediction class
2. Ratio S_t/h_t distribution
3. Running average smoothness
4. Statistics summary (text)

## Performance Expectations

Based on the architecture:

**Strengths:**
- Fast inference (no sliding window overhead)
- State-aware adaptation (uses S_{t-1})
- Theoretically sound (k from CUSUM theory)
- Interpretable (can see when/why h adapts)

**Potential:**
- May outperform DNN-CUSUM on faults with gradual drift
- Should have lower false alarms (feedback prevents oscillation)
- Faster training (simpler architecture, no LSTM)

**Trade-offs:**
- Less temporal context (no window-based features)
- Relies on feedback loop (may lag initially)

## Next Steps

1. **Run the cells!** Execute cells in order to:
   - Train DeepT-CUSUM (first run ~15-20 min)
   - Generate all plots
   - See initial performance on Fault 2, Run 1

2. **Run full comparison** Execute the comparison cell to:
   - Test across all 20 faults and 20 runs
   - Generate comprehensive results
   - Compare with MCUSUM and DNN-CUSUM

3. **Analyze results** using `ModelComparisonAnalyzer`:
   - DeepT-CUSUM will appear in all summary tables
   - Compare ARL0 (false alarms) and ARL1 (detection delay)
   - Identify which faults benefit from adaptive thresholds

4. **Write up findings** for your research:
   - When does DeepT-CUSUM outperform fixed CUSUM?
   - How does feedback mechanism help?
   - Comparison with DNN-CUSUM (both k&h adaptive)

## Troubleshooting

### Training too slow
- Grid search is comprehensive (8 configs)
- Wait for first run, then it's instant
- Or set `grid_search=False` to skip

### Poor performance
- Check that `best_k` is reasonable (should be ~6-7)
- Ensure full dataset used for training
- Inspect plots to see how h adapts

### Plots not showing
- Make sure all cells run in order
- Import cell must run before others
- Training must complete before testing

## Success Indicators

✅ First run takes ~15-20 minutes (grid search)
✅ Subsequent runs take ~1 second (loading)
✅ 16 subplots appear across 4 figures
✅ h_t varies over time (not constant)
✅ DeepT-CUSUM appears in MODELS dictionary
✅ Full comparison includes DeepT-CUSUM results
✅ Models folder contains 4 new files

---

**You now have THREE CUSUM variants to compare:**

1. **MCUSUM** - Fixed k and h (classical approach)
2. **DNN-CUSUM** - Adaptive k and h from window features
3. **DeepT-CUSUM** - Fixed k, adaptive h with feedback

Time to run the experiments and see which performs best! 🚀
