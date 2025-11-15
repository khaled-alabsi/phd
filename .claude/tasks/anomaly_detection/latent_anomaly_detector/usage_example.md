# LatentAnomalyDetector Usage Example

## Basic Usage

```python
from code.src.flexible_autoencoder import FlexibleAutoencoder
from code.src.latent_anomaly_detector import LatentAnomalyDetector

# Step 1: Train autoencoder (or load existing)
autoencoder = FlexibleAutoencoder(verbose=1, cache_prefix="my_autoencoder")
autoencoder.fit(X_train, X_val)

# Step 2: Get encoder
encoder = autoencoder.get_encoder()

# Step 3: Create anomaly detector
detector = LatentAnomalyDetector(
    encoder=encoder,
    verbose=1,
    cache_prefix="my_anomaly_detector"
)

# Step 4: Train with labeled data (0=normal, 1=anomaly)
detector.fit(
    X_train, y_train,
    X_val, y_val,
    param_grid=None,  # Use default grid
    reset=False       # Load from cache if exists
)

# Step 5: Get predictions
# Continuous scores (0.0-1.0)
anomaly_scores = detector.predict_proba(X_test)

# Binary labels (0/1)
anomaly_labels = detector.predict(X_test, threshold=0.5)

# Step 6: Visualize results
detector.plot_summary(X_train, y_train, X_val, y_val, threshold=0.5)
```

## Custom Grid Search

```python
# Define custom parameter grid
CUSTOM_GRID = {
    "hidden_layers": [
        (128, 64),
        (256, 128, 64),
    ],
    "activation": ["relu"],
    "dropout_rate": [0.2, 0.3],
    "learning_rate": [1e-3, 5e-4],
    "batch_size": [256],
    "epochs": [100],
    "patience": [15],
}

# Train with custom grid
detector.fit(
    X_train, y_train,
    X_val, y_val,
    param_grid=CUSTOM_GRID,
    reset=True  # Force retrain
)
```

## Force Retrain

```python
# Force retrain even if cached model exists
detector.fit(
    X_train, y_train,
    X_val, y_val,
    reset=True  # This will retrain from scratch
)
```

## Access Latent Representations

```python
# Get latent vectors for analysis
latent_vectors = detector.get_latent(X_test)
```

## Using Different Cache Prefixes for Experiments

```python
# Experiment 1
detector_exp1 = LatentAnomalyDetector(
    encoder=encoder,
    cache_prefix="experiment_1_detector"
)
detector_exp1.fit(X_train, y_train, X_val, y_val)

# Experiment 2 (different prefix = different cached model)
detector_exp2 = LatentAnomalyDetector(
    encoder=encoder,
    cache_prefix="experiment_2_detector"
)
detector_exp2.fit(X_train, y_train, X_val, y_val)
```

## Threshold Tuning

```python
# Try different thresholds
for threshold in [0.3, 0.5, 0.7, 0.9]:
    predictions = detector.predict(X_test, threshold=threshold)
    print(f"Threshold {threshold}: {predictions.sum()} anomalies detected")
```

## Complete Workflow

```python
import numpy as np
from sklearn.model_selection import train_test_split
from code.src.flexible_autoencoder import FlexibleAutoencoder
from code.src.latent_anomaly_detector import LatentAnomalyDetector

# Prepare data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train autoencoder
autoencoder = FlexibleAutoencoder(
    verbose=1,
    cache_prefix="ae_for_anomaly"
)
autoencoder.fit(X_train, X_val, reset=False)

# Create and train detector
detector = LatentAnomalyDetector(
    encoder=autoencoder.get_encoder(),
    verbose=1,
    cache_prefix="latent_anomaly_detector"
)

detector.fit(
    X_train, y_train,
    X_val, y_val,
    reset=False
)

# Evaluate
test_scores = detector.predict_proba(X_test)
test_labels = detector.predict(X_test, threshold=0.5)

# Visualize
detector.plot_summary(X_train, y_train, X_val, y_val, threshold=0.5)

# Print results
print(f"\nTest Set Results:")
print(f"  Anomalies detected: {test_labels.sum()} / {len(test_labels)}")
print(f"  Mean anomaly score: {test_scores.mean():.4f}")
print(f"  Score range: [{test_scores.min():.4f}, {test_scores.max():.4f}]")
```
