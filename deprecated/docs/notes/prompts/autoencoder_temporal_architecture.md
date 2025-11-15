# Autoencoder-Supervised Pointwise Residual Regressor

## Concept Summary
- Stage 1 trains an autoencoder on fault-free behaviour purely to extract the reconstruction error for each data point; the autoencoder itself is only a teacher.
- Stage 2 fits a deep neural network that **operates per time point** to approximate the autoencoder’s reconstruction error directly from the current sensor vector.
- Final anomaly decisions come from thresholding the DNN’s predicted reconstruction error; later work can feed these scores into a temporal model, but this step focuses solely on reproducing the autoencoder residual with a simpler feed-forward DNN.

## Component Breakdown
- **Data Ingestion & Preprocessing**
  - Standardise or normalise raw sensor streams (reuse existing `StandardScaler` pipeline).
  - Segment sequences by `simulationRun` so both models see chronologically ordered windows.
- Optionally downsample/aggregate very high-frequency channels to control memory footprint.
- **Autoencoder Module (Teacher)**
  - Encoder: stacked dense layers (e.g., `input_dim -> 256 -> 128 -> latent_dim`) with dropout or layer norm to prevent overfitting.
  - Latent space: 16–32 dimensions capturing joint sensor behaviour.
  - Decoder: mirror of encoder using tied/untied weights to reconstruct inputs.
  - Loss: MSE or smooth L1; train with early stopping on validation sequences that contain only normal behaviour.
  - Output is **only** the reconstruction residual (no thresholding); treat the AE as a self-supervised teacher.
- **Reconstruction Error Pipeline**
  - Computes point-wise error (per-sample MSE or per-feature absolute error aggregated by mean/max).
  - Stores the scalar residual (and optionally per-feature residuals) as supervision targets for the DNN.
- **Pointwise DNN Residual Estimator**
  - Inputs: current scaled sensor vector (optionally concatenated with simple statistics such as sensor-wise gradients if desired).
  - Architecture options: multilayer perceptron with residual connections, shallow convolution across feature dimension, or lightweight transformer encoder that attends across sensors at the same timestep.
  - Training objective: regression (e.g., MSE) to match the autoencoder’s reconstruction error value.
  - Output: predicted residual \(\hat{r}(t)\) that should approximate the autoencoder reconstruction error at time \(t\); this value is thresholded for anomaly decisions.
- **Decision Fusion & Thresholding**
  - Primary trigger: `predicted_residual > tau_dnn`.
  - Optional support rules: running mean/median smoothing or short persistence to dampen flicker.
  - Track predicted residuals over time for diagnostics and for future temporal modelling.
- **Training Signals & Labels**
  - Autoencoder: exclusively normal segments to learn baseline reconstruction.
  - Pointwise DNN: supervised targets equal to the teacher autoencoder’s reconstruction error; when available, anomaly labels can validate threshold selection.
- **Evaluation & Monitoring**
  - Metrics: regression fit between DNN and autoencoder residuals (R², MAE) plus downstream anomaly metrics after thresholding (precision/recall, ARL0/ARL1).
  - Visual diagnostics: overlay actual autoencoder residual, DNN prediction, and final anomaly decisions on process trends.
- **Refinement Checklist**
  1. Standardise or transform the reconstruction residual targets (z-score or `log(residual + ε)`) before training the regressor, and invert the transform at inference to stabilise heavy-tailed residual distributions.
  2. Expand the regressor hyperparameter grid (layers, dropout, learning rate, epochs) to ensure grid/random search has enough capacity to find well-calibrated models.

## Implementation Steps
1. **Data Preparation**
   - Split historical runs into train/validation/test with disjoint time segments.
   - Align each timestep with its raw sensor vector and (optionally) anomaly label.
2. **Autoencoder Prototype**
   - Build PyTorch/Keras module following the encoder/decoder design; add configurable latent size and regularisation.
   - Train on normal data; monitor validation loss and reconstruction error distribution.
3. **Teacher Signal Calibration**
   - Compute reconstruction errors on validation normal data.
   - Characterise typical residual distribution and derive pseudo-label heuristics (e.g., flag sequences where residual exceeds `mean + 3 * std` for N steps) to help supervise the temporal network when ground-truth labels are limited.
4. **Error Feature Generation**
   - For every timestep, compute and store the scalar reconstruction error from the autoencoder (optionally along with smoothed or z-scored variants).
   - Persist `(sensor_vector, residual_target)` pairs in arrays/DataFrames keyed by `simulationRun` for DNN training.
5. **Pointwise DNN Dataset Builder**
   - Inputs: current scaled sensor vector (optionally append simple handcrafted features if beneficial).
   - Target: autoencoder residual value at the same timestep.
   - Shuffle/batch samples while preserving validation partitions by simulation run.
6. **Pointwise Residual Model Implementation**
   - Define an MLP/Conv/Transformer architecture with configurable depth, hidden size, dropout, and normalisation.
   - Train using regression loss (MSE, smooth L1) to minimise the difference between predicted and teacher residuals.
   - Validate on held-out runs, tracking residual MAE/R² and downstream anomaly detection accuracy after applying the chosen threshold.
7. **Inference Pipeline & Thresholding**
   - Implement inference wrapper that:
     1. Scales incoming data and feeds autoencoder for reconstruction.
     2. Computes the DNN-predicted residual directly from the same scaled sensors.
     3. Compares the predicted residual against `tau_dnn`; optionally smooth predictions or add persistence.
   - Maintain both actual AE residuals (for audit) and DNN predictions to monitor teacher–student drift and to support future temporal modelling.
8. **Evaluation & Benchmarking**
   - Re-run existing MCUSUM/MEWMA workflows and compare ARL metrics against new pipeline.
   - Log confusion matrices and detection delays per fault type.
9. **Operationalisation**
   - Package both models with saved weights and preprocessing scalers.
   - Provide batch and streaming inference scripts; integrate with plotting utilities for diagnostics.
   - Document retraining schedule and threshold recalibration procedure.

## Concept Flow Diagram
```
+---------------------+        +--------------------+        +------------------------+
| Raw Sensor Streams  | -----> | Preprocess & Scale | -----> | Autoencoder (Teacher)  |
| (per simulation run)|        | (StandardScaler)   |        | Encoder/Decoder         |
+---------------------+        +--------------------+        +------------------------+
                                                                    |
                                                                    v
                                                         +-----------------------+
                                                        | Reconstruction Error  |
                                                        | Feature Builder       |
                                                         | - scalar residual     |
                                                         | - optional smoothing  |
                                                        +-----------------------+
                                                                    |
                                                                    v
                                              +------------------------------+
                                              | Pointwise DNN Residual Model |
                                              | Input: current scaled sensors|
                                              | Output: predicted residual   |
                                              +---------------+--------------+
                                                              |
                                                              v
                                   +----------------------------------------------+
                                   | Threshold & Optional Smoothing               |
                                   | if predicted residual > tau_dnn              |
                                   | (with optional persistence) -> anomaly       |
                                   +----------------------+-----------------------+
                                                          |
                                                          v
                                         +---------------------------+
                                         | Final Anomaly Output      |
                                         | flag + score + explanation|
                                         +---------------------------+
```
