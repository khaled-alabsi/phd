"""
Hybrid Approaches for Anomaly Detection in Tennessee Eastman Process

This module provides state-of-the-art hybrid detector classes combining traditional
statistical methods with machine learning for industrial process monitoring.

Detector Selection Guide:
------------------------
1. PCA_SVM_Detector: Best for linear processes with correlated variables
   - Fast, interpretable, works well when PCA captures process structure
   - Use when: Process is relatively linear and you have high-dimensional data

2. KPCA_SVM_Detector: Best for nonlinear processes
   - Extends PCA-SVM to handle complex nonlinear relationships
   - Use when: Standard PCA-based methods show poor performance

3. ICA_SVM_Detector: Best for non-Gaussian processes
   - Handles processes with non-Gaussian distributions and mixed signals
   - Use when: Process has independent noise sources or non-Gaussian behavior

4. LSTM_VAE_Detector: Best for dynamic processes (TOP PERFORMER)
   - Captures temporal dependencies, >95% fault detection rate on TEP
   - Use when: Temporal patterns are important (most industrial processes)

5. VAE_T2_Detector: Best for combining deep learning with statistical control
   - Provides statistically-principled control limits on learned features
   - Use when: You need interpretable thresholds with neural network power

6. PCA_Autoencoder_Hybrid: Best for real-time monitoring
   - Two-tier architecture saves 70-90% computation vs pure deep learning
   - Use when: Computational resources are limited but accuracy is critical

Quick Start Example:
-------------------
>>> from lib.hybrid_approaches import LSTM_VAE_Detector
>>> detector = LSTM_VAE_Detector(sequence_length=10, latent_dim=10)
>>> detector.fit(X_train_normal, epochs=30, batch_size=64)
>>> predictions = detector.predict(X_test)  # 1=fault, 0=normal
"""

from .pca_svm_detector import PCA_SVM_Detector
from .kpca_svm_detector import KPCA_SVM_Detector
from .ica_svm_detector import ICA_SVM_Detector
from .lstm_vae_detector import LSTM_VAE_Detector
from .vae_t2_detector import VAE_T2_Detector
from .pca_autoencoder_hybrid import PCA_Autoencoder_Hybrid

__all__ = [
    'PCA_SVM_Detector',
    'KPCA_SVM_Detector',
    'ICA_SVM_Detector',
    'LSTM_VAE_Detector',
    'VAE_T2_Detector',
    'PCA_Autoencoder_Hybrid',
]

__version__ = '1.0.0'
