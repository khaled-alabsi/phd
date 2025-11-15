# Task: Latent Anomaly Detector Implementation

## Objective
Implement a deep neural network (DNN) based anomaly detector that operates on latent representations from a trained autoencoder.

## Architecture
```
Raw Features → Encoder (frozen) → Latent Vector → DNN Classifier → Sigmoid → Anomaly Score (0.0-1.0)
```

## Key Requirements

### 1. Class: LatentAnomalyDetector
- **File:** `code/src/latent_anomaly_detector.py`
- **Pattern:** Follow DNN training pattern from `.claude/preference/dnn_training_pattern.md`

### 2. Training Mode
- **Semi-supervised:** Uses labeled data (0=normal, 1=anomaly)
- **Loss:** Binary cross-entropy
- **Encoder:** Frozen (passed as constructor argument)

### 3. Outputs
- **predict_proba(X):** Continuous anomaly scores (0.0-1.0)
- **predict(X, threshold):** Binary classification (0/1)

### 4. Grid Search
- Automatically find best DNN architecture
- Cache results to avoid retraining
- Support `reset=True` to force retrain

### 5. Hyperparameters (Grid Search)
- Hidden layers architecture
- Activation functions
- Dropout rates
- Learning rate
- Batch size
- Epochs & patience

## Implementation Status
- [x] Plan created
- [x] Class structure implemented
- [x] Model building implemented
- [x] Grid search implemented
- [x] Prediction methods implemented
- [x] Visualization methods implemented
- [ ] Testing complete (ready for user testing)

## Files Created
- `code/src/latent_anomaly_detector.py` - Main implementation
- `.claude/tasks/anomaly_detection/latent_anomaly_detector/usage_example.md` - Usage guide

## Related Files
- `code/src/flexible_autoencoder.py` - Encoder source
- `.claude/preference/dnn_training_pattern.md` - Implementation pattern
