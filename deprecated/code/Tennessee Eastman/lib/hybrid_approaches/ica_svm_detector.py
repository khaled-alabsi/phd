from sklearn.decomposition import FastICA
from sklearn.svm import OneClassSVM
import numpy as np


class ICA_SVM_Detector:
    """
    ICA + One-Class SVM Hybrid Detector for non-Gaussian process monitoring.
    Uses ICA for blind source separation to extract independent components, ideal for non-Gaussian distributions.
    Best for: Industrial processes with non-Gaussian noise/disturbances or mixed independent source signals.
    """
    def __init__(self, n_components=20, nu=0.05, max_iter=500):
        """
        Initialize ICA-SVM detector with specified parameters.

        Parameters:
        - n_components: Number of independent components to extract (10-30 typical). Start with 20.
                       Should be ≤ number of features. More components capture more sources but risk overfitting.
        - nu: Expected fraction of training outliers (0.01-0.1). Start with 0.05.
              Lower values = more sensitive detection but higher false alarm risk.
        - max_iter: Maximum ICA iterations for convergence (200-1000). Increase if convergence warnings appear.
                   500 is typically sufficient; increase to 1000 for difficult datasets.
        """
        self.n_components = n_components
        self.nu = nu
        self.max_iter = max_iter
        self.ica = None
        self.svm = None

    def fit(self, X_train):
        """
        Train detector on normal data by extracting independent components via ICA.
        X_train: Fault-free training data of shape (n_samples, n_features). Should contain only normal operation data.
        """
        # Independent Component Analysis
        self.ica = FastICA(
            n_components=self.n_components,
            max_iter=self.max_iter,
            random_state=42
        )
        X_independent = self.ica.fit_transform(X_train)

        # One-Class SVM training
        self.svm = OneClassSVM(nu=self.nu, kernel='rbf')
        self.svm.fit(X_independent)

        return self

    def predict(self, X_test):
        """
        Detect anomalies by analyzing independent components of test data.
        X_test: Test data of shape (n_samples, n_features).
        Returns: Binary array where 1 = anomaly (fault), 0 = normal (in-control).
        """
        X_independent = self.ica.transform(X_test)
        predictions = self.svm.predict(X_independent)
        return np.where(predictions == -1, 1, 0)
