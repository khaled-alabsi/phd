from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
import numpy as np


class PCA_SVM_Detector:
    """
    PCA + One-Class SVM Hybrid Detector for linear process monitoring.
    Uses PCA for dimensionality reduction, then One-Class SVM for anomaly detection in reduced space.
    Best for: Linear processes with high-dimensional correlated variables.
    """
    def __init__(self, n_components=0.95, nu=0.01, kernel='rbf', gamma='scale'):
        """
        Initialize PCA-SVM detector with specified parameters.

        Parameters:
        - n_components: PCA variance to retain (0.95 = 95% variance) or exact number of components (int).
                       Higher values (0.95-0.99) capture more process variation but increase computation.
        - nu: Expected fraction of outliers in training data (0.01-0.1). Lower = stricter detection.
              Start with 0.05 (5% outliers) and adjust based on false alarm rate.
        - kernel: SVM kernel type ('rbf', 'linear', 'poly'). 'rbf' works well for most cases.
        - gamma: Kernel coefficient ('scale' is auto, or float). Use 'scale' unless you have domain knowledge.
        """
        self.n_components = n_components
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.pca = None
        self.svm = None

    def fit(self, X_train):
        """
        Train the detector on normal (in-control) data only.
        X_train: Training data of shape (n_samples, n_features) - should be fault-free data.
        """
        # PCA dimensionality reduction
        self.pca = PCA(n_components=self.n_components)
        X_reduced = self.pca.fit_transform(X_train)

        # One-Class SVM training
        self.svm = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        self.svm.fit(X_reduced)

        return self

    def predict(self, X_test):
        """
        Detect anomalies in new data using the trained model.
        X_test: Test data of shape (n_samples, n_features).
        Returns: Array of predictions where 1 = anomaly (out-of-control), 0 = normal (in-control).
        """
        X_reduced = self.pca.transform(X_test)
        predictions = self.svm.predict(X_reduced)
        # Convert SVM output: -1 (outlier) to 1, 1 (inlier) to 0
        return np.where(predictions == -1, 1, 0)
