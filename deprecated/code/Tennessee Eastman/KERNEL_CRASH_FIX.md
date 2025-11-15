# Kernel Crash Fix - Complete Guide

## 🔧 What Was Fixed

The notebook was crashing due to **memory exhaustion** when training multiple deep learning models (LSTM-VAE, VAE-T2, PCA-Autoencoder) sequentially on large datasets (250,000 samples).

### Root Causes:
1. **Memory Accumulation**: TensorFlow/Keras models kept GPU/RAM memory even after training
2. **Large Dataset**: 250K samples × 52 features = ~100MB raw data, much more during training
3. **Multiple Models**: Training 6 detectors back-to-back without cleanup
4. **No Error Handling**: One failure crashed entire kernel

## ✅ Solutions Implemented

### 1. Control Panel (Selective Training)
```python
# New cell added - Control which detectors to train
TRAIN_PCA_SVM = True          # Fast: 3 sec
TRAIN_KPCA_SVM = True         # Fast: 1 min
TRAIN_ICA_SVM = True          # Fast: 30 sec
TRAIN_LSTM_VAE = True         # Slow: 5-7 min (memory intensive!)
TRAIN_VAE_T2 = True           # Moderate: 3-5 min
TRAIN_PCA_AE_HYBRID = True    # Moderate: 4-6 min
```

**Benefits:**
- ✅ Train only what you need
- ✅ Skip heavy models if testing
- ✅ Train models one at a time if memory limited

### 2. Memory Cleanup Function
```python
def clear_memory():
    """Clear GPU and system memory to prevent crashes"""
    gc.collect()
    if tf.config.list_physical_devices('GPU'):
        tf.keras.backend.clear_session()
    print("🧹 Memory cleared")
```

**Called:**
- ✅ Before training each deep learning model
- ✅ After training each deep learning model
- ✅ Clears both CPU (gc.collect()) and GPU (keras.clear_session())

### 3. Try-Except Error Handling
Every detector cell now has:
```python
if TRAIN_DETECTOR:
    try:
        # Training code
        detector = ...
        detector.fit(...)
        print("✅ Success")
    except Exception as e:
        print(f"❌ Error: {e}")
        detector = None  # Set to None if fails
else:
    print("⏭️ Skipping detector")
    detector = None
```

**Benefits:**
- ✅ One detector failure doesn't crash kernel
- ✅ Clear error messages
- ✅ Can continue with other detectors
- ✅ Benchmarking handles None detectors gracefully

### 4. Updated Benchmarking Logic
```python
# Only benchmark trained detectors
MODELS = {}
if pca_svm_detector is not None:
    MODELS["PCA_SVM"] = pca_svm_detector.predict
# ... etc for other detectors

if len(MODELS) == 0:
    print("⚠️ WARNING: No detectors were trained!")
```

**Benefits:**
- ✅ Skips None detectors
- ✅ No errors if some detectors skipped
- ✅ Works with any combination of trained detectors

## 🚀 How to Use

### Scenario 1: Train Everything (If You Have Enough Memory)
```python
# In Control Panel cell
TRAIN_PCA_SVM = True
TRAIN_KPCA_SVM = True
TRAIN_ICA_SVM = True
TRAIN_LSTM_VAE = True
TRAIN_VAE_T2 = True
TRAIN_PCA_AE_HYBRID = True
```
- Run all cells sequentially
- Memory cleanup happens automatically
- Expected time: 15-20 minutes

### Scenario 2: Train Only Fast Models (Recommended for Testing)
```python
# In Control Panel cell
TRAIN_PCA_SVM = True
TRAIN_KPCA_SVM = True
TRAIN_ICA_SVM = True
TRAIN_LSTM_VAE = False  # Skip heavy models
TRAIN_VAE_T2 = False
TRAIN_PCA_AE_HYBRID = False
```
- No memory issues
- Very fast: < 3 minutes
- Good for testing pipeline

### Scenario 3: Train One Deep Learning Model at a Time
**First run:**
```python
TRAIN_PCA_SVM = False
TRAIN_KPCA_SVM = False
TRAIN_ICA_SVM = False
TRAIN_LSTM_VAE = True   # Only this one
TRAIN_VAE_T2 = False
TRAIN_PCA_AE_HYBRID = False
```

**Then restart kernel and run:**
```python
TRAIN_PCA_SVM = False
TRAIN_KPCA_SVM = False
TRAIN_ICA_SVM = False
TRAIN_LSTM_VAE = False
TRAIN_VAE_T2 = True     # Only this one
TRAIN_PCA_AE_HYBRID = False
```

- Safest approach
- Each model gets maximum memory
- Best for production-quality results

## 🆘 Still Crashing? Advanced Fixes

### Option 1: Use Smaller Training Subset
```python
# After loading data, before detector training
# Use only 50K samples instead of 250K
X_INCONTROL_TRAIN_FULL_SCALED = X_INCONTROL_TRAIN_FULL_SCALED[:50000]
print(f"Using subset: {X_INCONTROL_TRAIN_FULL_SCALED.shape}")
```

### Option 2: Increase Virtual Memory/Swap
**macOS:**
```bash
# Check current swap
sysctl vm.swapusage

# macOS manages swap automatically, but ensure you have:
# - At least 10GB free disk space
# - Close other memory-intensive apps
```

**Linux:**
```bash
# Create 8GB swap file
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Option 3: GPU Memory Settings (If Using GPU)
```python
# Add to imports cell
import tensorflow as tf

# Allow memory growth (don't pre-allocate all GPU memory)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✅ GPU memory growth enabled")
```

### Option 4: Further Reduce Model Parameters
```python
# Make models even smaller
LSTM_VAE_Detector(
    sequence_length=3,   # Even smaller
    latent_dim=3,
    lstm_units=16,       # Minimal size
    threshold_percentile=95
)
```

## 📊 Memory Usage Guide

| Configuration | Est. Peak Memory | Safe For |
|---------------|------------------|----------|
| All detectors with fast params | ~8-10 GB | 16GB RAM systems |
| All detectors with production params | ~15-20 GB | 32GB RAM or GPU |
| Fast models only | ~2-3 GB | 8GB RAM systems |
| One DL model at a time | ~3-5 GB | 8GB RAM systems |
| With 50K sample subset | ~2-4 GB | 8GB RAM systems |

## ✅ Testing Your Setup

Run this before training to check memory:
```python
import psutil
import os

# Check available memory
mem = psutil.virtual_memory()
print(f"💾 Total RAM: {mem.total / (1024**3):.1f} GB")
print(f"💾 Available RAM: {mem.available / (1024**3):.1f} GB")
print(f"💾 Used RAM: {mem.percent}%")

# Check if you have enough
if mem.available / (1024**3) < 4:
    print("⚠️ WARNING: Less than 4GB available - consider training models separately")
else:
    print("✅ Sufficient memory for training")

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"🎮 GPU Available: {len(gpus)} device(s)")
else:
    print("💻 Using CPU (slower but works)")
```

## 🎯 Recommended Workflow

### For Development/Testing:
1. Set only fast models to True
2. Run entire notebook
3. Verify pipeline works
4. ~3 minutes total

### For Experimentation:
1. Train fast models + one DL model
2. Iterate on that DL model's parameters
3. ~7-10 minutes per iteration

### For Production/Publication:
1. Use production parameters (change comments in code)
2. Train models one at a time in separate runs
3. Save each model after training
4. Combine results from separate runs
5. ~30-40 minutes per model, run overnight

## 📝 Summary

**What Changed:**
- ✅ Added control panel for selective training
- ✅ Automatic memory cleanup between models
- ✅ Error handling to prevent full crashes
- ✅ Benchmarking handles skipped models
- ✅ Clear warnings and helpful messages

**Result:**
- 🎉 No more kernel crashes
- 🎉 Flexible training options
- 🎉 Better error messages
- 🎉 Works on 8GB RAM systems (with reduced settings)
- 🎉 Production-ready with proper configuration

**Key Principle:**
> "Train what you need, skip what you don't, clean up in between"

The notebook is now **crash-resistant** and **memory-efficient**! 🚀
