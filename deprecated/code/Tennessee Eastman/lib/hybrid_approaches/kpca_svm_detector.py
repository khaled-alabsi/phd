from sklearn.decomposition import KernelPCA
from sklearn.svm import OneClassSVM
import numpy as np


class KPCA_SVM_Detector:
    """
    Kernel PCA + One-Class SVM Hybrid Detector for nonlinear process monitoring.
    Extends PCA-SVM to handle nonlinear relationships using kernel trick for dimensionality reduction.
    Best for: Nonlinear processes where standard PCA fails to capture complex variable interactions.
    """
    def __init__(self, n_components=20, kernel='rbf', gamma=None, nu=0.05):
        """
        Initialize KPCA-SVM detector with specified parameters.

        Parameters:
        - n_components: Number of kernel components to retain (10-30 typical). Start with 20.
                       Too few components lose information; too many increase computation and overfitting risk.
        - kernel: Kernel function ('rbf' for smooth nonlinear, 'poly' for polynomial, 'sigmoid' for s-curves).
                 'rbf' is recommended as default for most industrial processes.
        - gamma: Kernel bandwidth (None for auto = 1/n_features, or float). None works well for most cases.
                Lower gamma = smoother decision boundary; higher = more complex/sensitive.
        - nu: Expected fraction of outliers (0.01-0.1). Start with 0.05 and tune based on false alarms.
        """
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.kpca = None
        self.svm = None

    def fit(self, X_train):
        """
        Train the detector on normal (fault-free) data using kernel PCA transformation.
        X_train: Training data of shape (n_samples, n_features) - should contain only in-control samples.
        """
        # Kernel PCA nonlinear dimensionality reduction
        self.kpca = KernelPCA(
            n_components=self.n_components,
            kernel=self.kernel,
            gamma=self.gamma,
            fit_inverse_transform=True
        )
        X_reduced = self.kpca.fit_transform(X_train)

        # One-Class SVM training
        self.svm = OneClassSVM(nu=self.nu, kernel='rbf')
        self.svm.fit(X_reduced)

        return self

    def predict(self, X_test):
        """
        Detect anomalies in test data using kernel-transformed feature space.
        X_test: Test data of shape (n_samples, n_features).
        Returns: Array where 1 = fault detected (out-of-control), 0 = normal (in-control).
        """
        X_reduced = self.kpca.transform(X_test)
        predictions = self.svm.predict(X_reduced)
        return np.where(predictions == -1, 1, 0)
