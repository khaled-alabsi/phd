# DNN-CUSUM Notebook Integration Guide

This guide shows how to integrate DNN-CUSUM into your `anomaly_detection.ipynb` notebook.

## Step 1: Add Import Cell

Add this cell after your other imports (e.g., after the MEWMA imports):

```python
# Import DNN-CUSUM detector and visualizer
from src.dnn_cusum import DNNCUSUMDetector
from src.dnn_cusum_viz import DNNCUSUMVisualizer
```

## Step 2: Add Training Cell

Add this cell in a new section for DNN-CUSUM:

```python
## DNN-CUSUM (Deep Neural Network-based Adaptive CUSUM)

# Initialize detector
dnn_cusum = DNNCUSUMDetector(
    window_size=50,
    model_dir='models/'
)

# Train or load model
# This will:
# - Load existing model if available
# - Otherwise, perform grid search and train new model
# - Save trained model for future use
dnn_cusum.fit(
    X_INCONTROL_TRAIN_FULL_SCALED,
    X_OUT_OF_CONTROL_TRAIN_PLAY_SCALED,
    force_retrain=False,  # Set to True to retrain even if model exists
    grid_search=True      # Set to False to skip grid search
)
```

## Step 3: Test DNN-CUSUM

```python
# Test on cut data
predictions, param_history = dnn_cusum.predict(
    X_OUT_OF_CONTROL_TEST_PLAY_SCALED_CUT,
    return_params=True
)

# Calculate ARL metrics
arl0_test = None  # Would need to test on normal data
arl1_test = np.argmax(predictions == 1) if np.any(predictions == 1) else None

print(f"DNN-CUSUM Detection Delay (ARL1): {arl1_test}")
print(f"Detection Rate: {np.mean(predictions)*100:.2f}%")
```

## Step 4: Visualize Parameter Evolution

```python
# Create visualizer
viz = DNNCUSUMVisualizer()

# Plot parameter evolution
viz.plot_parameter_evolution(
    param_history,
    X_OUT_OF_CONTROL_TEST_PLAY_SCALED_CUT,
    predictions,
    fault_injection_point=0,  # Already cut
    title_suffix="- Fault 2"
)

# Plot parameter statistics
viz.plot_parameter_statistics(param_history, predictions)

# Plot sensitivity to a specific feature
viz.plot_parameter_sensitivity(
    param_history,
    X_OUT_OF_CONTROL_TEST_PLAY_SCALED_CUT,
    feature_idx=10  # Choose interesting feature
)
```

## Step 5: Compare with Fixed CUSUM

```python
# Get fixed CUSUM predictions for comparison
fixed_mcusum = MCUSUMDetector(k=best_k, h=best_h)
fixed_mcusum.fit(X_INCONTROL_TRAIN_FULL_SCALED)
fixed_predictions, _ = fixed_mcusum.predict(X_OUT_OF_CONTROL_TEST_PLAY_SCALED_CUT)

# Compare
viz.plot_comparison(
    fixed_predictions.astype(int),
    predictions,
    X_OUT_OF_CONTROL_TEST_PLAY_SCALED_CUT,
    param_history=param_history,
    fault_injection_point=0,
    title="- Fault 2"
)
```

## Step 6: Add to Model Comparison

Update your MODELS dictionary to include DNN-CUSUM:

```python
# Add prediction wrapper
def dnn_cusum_predict(x_scaled):
    preds, _ = dnn_cusum.predict(x_scaled, return_params=False)
    return preds.astype(int)

# Add to MODELS dictionary
MODELS = {
    "MCUSUM": mcusum_predict,
    "Autoencoder": autoencoder_detector.predict,
    "AutoencoderEnhanced": autoencoder_detector_enhanced.predict,
    "MEWMA": mewma.predict,
    "DNN_CUSUM": dnn_cusum_predict  # Add this line
}
```

## Step 7: Run Full Comparison

The existing comparison loop will now include DNN-CUSUM:

```python
for simulation_run in SIMULATION_RUN_RANGE:
    # ... existing loop code ...

    for fault_number in FAULT_NUMBER_RANGE:
        # ... existing loop code ...

        # Run all models (including DNN-CUSUM)
        for model_name, model_func in MODELS.items():
            # ... existing comparison code ...
```

## Expected Output

After running, you should see:

1. **Training/Loading Message:**
   ```
   ============================================================
   Found existing trained model. Loading...
   ============================================================
   Model loaded from: models/dnn_cusum_model.h5
   ```

   OR (if training for first time):
   ```
   ============================================================
   Starting Grid Search for Best Configuration
   ============================================================
   ...
   Training Complete!
   ============================================================
   ```

2. **Parameter Evolution Plot:** Shows k(t), h(t), CUSUM statistic, and predictions over time

3. **Comparison Results:** DNN-CUSUM performance compared with other methods

4. **Saved Files:**
   - `models/dnn_cusum_model.h5` - Trained model
   - `models/dnn_cusum_best_config.json` - Best configuration
   - `models/dnn_cusum_model_scaler.pkl` - Feature scaler
   - `models/dnn_cusum_model_params.pkl` - Global parameters

## Tips

- **First Run:** Will take longer due to grid search (10-30 minutes)
- **Subsequent Runs:** Will load saved model (< 1 second)
- **Retraining:** Set `force_retrain=True` if you want to retrain with new data
- **Grid Search:** Set `grid_search=False` to use default config (faster)

## Troubleshooting

### Issue: "Model not fitted" error
**Solution:** Make sure you run the `dnn_cusum.fit()` cell before `predict()`

### Issue: Grid search takes too long
**Solution:** Reduce the param_grid in `dnn_cusum.py` or set `grid_search=False`

### Issue: Out of memory during training
**Solution:** Reduce `window_size` or `batch_size` in the fit() call

### Issue: Poor performance
**Solution:** Try retraining with `force_retrain=True` after adjusting hyperparameters
