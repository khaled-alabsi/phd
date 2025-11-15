"""
MCUSUM (Multivariate Cumulative Sum) Anomaly Detection Module

This module implements the MCUSUM detector for multivariate anomaly detection
in industrial processes like the Tennessee Eastman dataset.
"""

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray


class MCUSUMDetector:
    """
    Multivariate Cumulative Sum (MCUSUM) Anomaly Detector.
    
    The MCUSUM is a statistical process control method that detects persistent 
    shifts in the mean of a multivariate process by accumulating deviations
    from the in-control state.
    """

    def __init__(self, k: float = 0.5, h: Optional[float] = None):
        """
        Initialize MCUSUM detector.
        
        Args:
            k: Reference value (sensitivity parameter)
            h: Control limit (threshold for detection). If None, will be estimated
        """
        self.k = k
        self.h = h
        self.mu_0 = None
        self.sigma = None
        self.is_fitted = False

    def fit(self, X_incontrol: NDArray[np.float64], verbose: bool = False) -> 'MCUSUMDetector':
        """
        Fit MCUSUM parameters using in-control training data.
        
        Args:
            X_incontrol: In-control (normal) training data
            verbose: Whether to print fitting information
            
        Returns:
            Self for method chaining
        """
        if verbose:
            print(f"🔧 **Fitting MCUSUM Parameters**")

        self.mu_0, self.sigma = self._estimate_incontrol_parameters(X_incontrol)
        
        if self.h is None:
            self.h = self._estimate_control_limit(X_incontrol)
        
        self.is_fitted = True

        if verbose:
            print(f"   Mean vector shape: {self.mu_0.shape}")
            print(f"   Covariance matrix shape: {self.sigma.shape}")
            print(f"   Reference value k: {self.k:.4f}")
            print(f"   Control limit h: {self.h:.4f}")

        return self
    
    def predict(self, X_test: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """
        Predict anomalies using the fitted MCUSUM detector.
        
        Args:
            X_test: Test data to evaluate
            
        Returns:
            Tuple of (statistics, flags) where:
            - statistics: MCUSUM statistic values for each sample
            - flags: Boolean array indicating anomalies (True = anomaly)
        """
        if not self.is_fitted:
            raise ValueError("MCUSUM detector must be fitted before prediction")

        statistics = self._compute_mcusum_scores(X_test, self.mu_0, self.sigma, self.k)
        flags = statistics > self.h
        
        return statistics, flags
    
    @staticmethod
    def _estimate_incontrol_parameters(X_incontrol: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Estimate mean vector and covariance matrix from in-control data.
        
        Args:
            X_incontrol: In-control training data
            
        Returns:
            Tuple of (mean vector, covariance matrix)
        """
        mu_0 = np.mean(X_incontrol, axis=0)
        sigma = np.cov(X_incontrol, rowvar=False, bias=False)
        
        # Ensure positive definite covariance matrix
        min_eigenval = np.min(np.linalg.eigvals(sigma))
        if min_eigenval <= 0:
            print(f"⚠️  Warning: Adding regularization to covariance matrix (min eigenvalue: {min_eigenval:.2e})")
            sigma += np.eye(sigma.shape[0]) * abs(min_eigenval) * 1.01
        
        return mu_0, sigma
    
    @staticmethod
    def _compute_mcusum_scores(X_test: NDArray[np.float64],
                              mu_0: NDArray[np.float64],
                              sigma: NDArray[np.float64],
                              k: float) -> NDArray[np.float64]:
        """
        Compute MCUSUM statistics for test data.
        
        Args:
            X_test: Test data
            mu_0: In-control mean vector
            sigma: In-control covariance matrix
            k: Reference value
            
        Returns:
            MCUSUM statistics for each sample
        """
        X_test = np.asarray(X_test)
        mu_0 = np.asarray(mu_0)
        sigma = np.asarray(sigma)
        
        n_samples, n_features = X_test.shape
        
        # Compute whitening transformation: Σ^{-1/2}
        try:
            eigvals, eigvecs = np.linalg.eigh(sigma)
            eigvals_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-12)))
            sigma_inv_sqrt = eigvecs @ eigvals_inv_sqrt @ eigvecs.T
        except np.linalg.LinAlgError:
            print("⚠️  Warning: Using pseudo-inverse for covariance matrix")
            sigma_inv_sqrt = np.linalg.pinv(sigma)
        
        # Whiten the data
        Z = (X_test - mu_0) @ sigma_inv_sqrt.T
        
        # MCUSUM recursion
        S_t = np.zeros(n_features)
        T = np.zeros(n_samples)
        
        for t in range(n_samples):
            V_t = S_t + Z[t]
            norm_V_t = np.linalg.norm(V_t)
            
            if norm_V_t <= k:
                S_t = np.zeros(n_features)
            else:
                shrinkage = 1.0 - k / norm_V_t
                S_t = V_t * shrinkage
            
            T[t] = np.linalg.norm(S_t)
        
        return T
    
    def _estimate_control_limit(self, X_incontrol: NDArray[np.float64],
                              n_simulations: int = 500,
                              percentile: float = 99.0,
                              verbose: bool = False) -> float:
        """
        Estimate control limit using Monte Carlo simulation.
        
        Args:
            X_incontrol: In-control training data
            n_simulations: Number of Monte Carlo simulations
            percentile: Percentile for control limit
            verbose: Whether to print estimation info
            
        Returns:
            Estimated control limit
        """
        max_T_values = []
        sample_size = min(300, X_incontrol.shape[0])
        
        for i in range(n_simulations):
            # Bootstrap sample from in-control data
            indices = np.random.choice(X_incontrol.shape[0], size=sample_size, replace=True)
            sample = X_incontrol[indices]
            
            # Compute MCUSUM statistics
            T = self._compute_mcusum_scores(sample, self.mu_0, self.sigma, self.k)
            max_T_values.append(np.max(T))
        
        h = np.percentile(max_T_values, percentile)
        if verbose:
            print(f"   Control limit (h) estimated at {percentile}th percentile: {h:.4f}")
        
        return h
    
    @staticmethod
    def compute_reference_value_k(delta: NDArray[np.float64],
                                 sigma: NDArray[np.float64]) -> float:
        """
        Compute optimal reference value k = 0.5 * ||Σ^{-1/2} δ||
        
        Args:
            delta: Expected shift vector
            sigma: In-control covariance matrix
            
        Returns:
            Optimal reference value
        """
        # Whitening matrix computation
        eigvals, eigvecs = np.linalg.eigh(sigma)
        eigvals_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-12)))
        sigma_inv_sqrt = eigvecs @ eigvals_inv_sqrt @ eigvecs.T
        
        # Transform delta and compute norm
        whitened_delta = sigma_inv_sqrt @ delta
        k = 0.5 * np.linalg.norm(whitened_delta)
        
        return k


def mcusum_predict(x_scaled: NDArray[np.float64], mcusum_instance: MCUSUMDetector) -> NDArray[np.int_]:
    """
    Convenience function to make predictions with a fitted MCUSUM detector.
    
    Args:
        x_scaled: Scaled input data
        mcusum_instance: Fitted MCUSUM detector instance
        
    Returns:
        Binary predictions (0 = normal, 1 = anomaly)
    """
    _, y_pred = mcusum_instance.predict(x_scaled)
    return y_pred.astype(int)