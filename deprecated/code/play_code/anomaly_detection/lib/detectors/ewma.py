"""
EWMA-based Anomaly Detection Module

This module implements various Exponentially Weighted Moving Average (EWMA) based
anomaly detectors for multivariate process monitoring, including:
- MEWMA (Multivariate EWMA)
- MEWMS (Multivariate EWMA for Variance)
- MEWMV (Multivariate EWMA for Mean and Variance)
- REWMV (Robust EWMA for Variance)
- MaxMEWMV (Maximum MEWMA for Variance)
- MNSE (Multivariate Normalized Squared Error)
"""

import pandas as pd
import numpy as np
from scipy.linalg import eigh
from typing import Dict, Any, List, Optional


class BaseEWMA:
    """
    Base class for EWMA-type multivariate control charts.
    Handles training mean and covariance estimation,
    and provides attributes used by specific MEWMA variants.
    """

    def __init__(self, lambda_: float = 0.2, percentile: float = 99.5):
        """
        Parameters
        ----------
        lambda_ : float
            Smoothing parameter for EWMA (0 < lambda_ <= 1).
            Higher values react faster to changes.
        percentile : float
            Percentile of in-control distribution used to set control limit.
        """
        self.lambda_ = lambda_
        self.percentile = percentile
        self.mu0 = None           # In-control mean vector
        self.p = None             # Number of variables
        self.Sigma = None         # In-control covariance matrix
        self.Sigma_inv = None     # Inverse covariance
        self.h = None             # Control limit threshold

    def fit(self, X_train: np.ndarray) -> None:
        """
        Estimate in-control parameters from training data.

        Parameters
        ----------
        X_train : np.ndarray of shape (n_samples, n_features)
            In-control training observations.
        """
        self.mu0 = np.mean(X_train, axis=0)
        self.p = X_train.shape[1]
        self.Sigma = np.cov(X_train.T)
        self.Sigma_inv = np.linalg.inv(self.Sigma)


class StandardMEWMA(BaseEWMA):
    """
    Standard Multivariate EWMA (MEWMA) Control Chart.

    Monitors the smoothed mean vector of a multivariate process.
    Detection statistic is a Hotelling-type T² on the EWMA vector.
    """

    def fit(self, X_train: np.ndarray) -> None:
        """
        Train the MEWMA chart on in-control data.
        Estimates the empirical distribution of the monitoring statistic
        and sets the control limit threshold.

        Parameters
        ----------
        X_train : np.ndarray of shape (n_samples, n_features)
            In-control training data.
        """
        # Step 1: Fit base parameters (mean, covariance, inverse covariance)
        super().fit(X_train)

        # Step 2: Initialize smoothed deviation vector Z
        n = X_train.shape[0]
        Z = np.zeros(self.p)
        Ts = []  # store monitoring statistics

        # Step 3: Iterate through training samples
        for i in range(n):
            # Deviation from in-control mean
            g = X_train[i] - self.mu0

            # Update EWMA vector
            Z = self.lambda_ * g + (1 - self.lambda_) * Z

            # Step 4: Monitoring statistic
            # Hotelling-type quadratic form, scaled by factor (2 - λ)/λ
            T = ((2 - self.lambda_) / self.lambda_) * np.dot(Z, self.Sigma_inv @ Z)
            Ts.append(T)

        # Step 5: Set control limit from empirical distribution
        self.h = np.percentile(Ts, self.percentile) if Ts else np.inf

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the trained MEWMA chart to new observations.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test sequence of observations.

        Returns
        -------
        pred : np.ndarray of shape (n_samples,)
            Boolean array: True if alarm (out-of-control), False otherwise.
        """
        n = X.shape[0]
        Z = np.zeros(self.p)
        pred = np.zeros(n, dtype=bool)

        for i in range(n):
            # Deviation from in-control mean
            g = X[i] - self.mu0

            # Update EWMA vector
            Z = self.lambda_ * g + (1 - self.lambda_) * Z

            # Compute monitoring statistic
            T = ((2 - self.lambda_) / self.lambda_) * np.dot(Z, self.Sigma_inv @ Z)

            # Compare against control limit
            pred[i] = T > self.h

        return pred.astype(int)


class MEWMS(BaseEWMA):
    """
    Multivariate Exponentially Weighted Moving Covariance (MEWMS) Chart.

    Monitors changes in the covariance structure by smoothing
    the outer product of deviations, standardized by the in-control covariance.
    """

    def fit(self, X_train: np.ndarray) -> None:
        """
        Train the MEWMS chart on in-control data.
        Estimates the distribution of the monitoring statistic
        and sets the control limit threshold.

        Parameters
        ----------
        X_train : np.ndarray
            In-control training data.
        """
        # Step 1: Fit base parameters (mean, covariance, inverse covariance)
        super().fit(X_train)

        # Step 2: Initialize smoothed covariance statistic Y
        # Y is a weighted moving average of outer products of deviations
        n = X_train.shape[0]
        Y = np.zeros((self.p, self.p))
        Ts = []  # store monitoring statistics for threshold estimation

        # Step 3: Iterate through training samples
        for i in range(n):
            # Deviation from in-control mean
            g = X_train[i] - self.mu0

            # Update smoothed covariance estimate (EWMA form)
            Y = self.lambda_ * np.outer(g, g) + (1 - self.lambda_) * Y

            # Step 4: Monitoring statistic (standardized quadratic form)
            # Using trace(Sigma_inv @ Y) normalizes by in-control variance/correlation
            T = np.trace(self.Sigma_inv @ Y)
            Ts.append(T)

        # Step 5: Set control limit from empirical distribution
        # Use high percentile of in-control statistic as threshold
        self.h = np.percentile(Ts, self.percentile) if Ts else np.inf

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the trained MEWMS chart to new observations.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test sequence of observations.

        Returns
        -------
        pred : np.ndarray of shape (n_samples,)
            Boolean array: True if alarm (out-of-control), False otherwise.
        """
        n = X.shape[0]
        Y = np.zeros((self.p, self.p))
        pred = np.zeros(n, dtype=bool)

        for i in range(n):
            # Deviation from in-control mean
            g = X[i] - self.mu0

            # Update smoothed covariance statistic
            Y = self.lambda_ * np.outer(g, g) + (1 - self.lambda_) * Y

            # Compute standardized monitoring statistic
            T = np.trace(self.Sigma_inv @ Y)

            # Compare against control limit
            pred[i] = T > self.h

        return pred.astype(int)


class MEWMV(BaseEWMA):
    """
    Multivariate EWMA for Mean and Variance (MEWMV) Control Chart.

    Monitors both:
    - Mean shifts (via smoothed EWMA vector Z)
    - Variance/covariance changes (via smoothed covariance matrix S)

    Detection statistic combines the standardized mean part (Hotelling-type)
    and the variance part (scaled covariance trace).
    """

    def fit(self, X_train: np.ndarray) -> None:
        """
        Train the MEWMV chart on in-control data.
        Estimates the empirical distribution of the monitoring statistic
        and sets the control limit threshold.

        Parameters
        ----------
        X_train : np.ndarray of shape (n_samples, n_features)
            In-control training data.
        """
        # Step 1: Fit base parameters (mean, covariance, inverse covariance)
        super().fit(X_train)

        # Step 2: Initialize smoothed mean and variance statistics
        n = X_train.shape[0]
        Z = np.zeros(self.p)               # EWMA mean shift vector
        S = np.zeros((self.p, self.p))     # EWMA covariance estimate
        Ts = []

        # Step 3: Iterate through training samples
        for i in range(n):
            # Mean deviation
            g_mean = X_train[i] - self.mu0

            # Update smoothed mean shift (EWMA form)
            Z = self.lambda_ * g_mean + (1 - self.lambda_) * Z

            # Update smoothed covariance contribution
            g_var = np.outer(g_mean, g_mean)
            S = self.lambda_ * g_var + (1 - self.lambda_) * S

            # Step 4: Monitoring statistic
            # Mean part: standardized quadratic form
            T_mean = ((2 - self.lambda_) / self.lambda_) * np.dot(Z, self.Sigma_inv @ Z)

            # Variance part: standardized covariance trace
            # Using trace(Sigma_inv @ S) instead of raw trace(S)
            T_var = np.trace(self.Sigma_inv @ S)

            # Combined statistic
            T = T_mean + T_var
            Ts.append(T)

        # Step 5: Set control limit from empirical in-control distribution
        self.h = np.percentile(Ts, self.percentile) if Ts else np.inf

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the trained MEWMV chart to new observations.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test sequence of observations.

        Returns
        -------
        pred : np.ndarray of shape (n_samples,)
            Boolean array: True if alarm (out-of-control), False otherwise.
        """
        n = X.shape[0]
        Z = np.zeros(self.p)
        S = np.zeros((self.p, self.p))
        pred = np.zeros(n, dtype=bool)

        for i in range(n):
            # Mean deviation
            g_mean = X[i] - self.mu0

            # Update smoothed mean
            Z = self.lambda_ * g_mean + (1 - self.lambda_) * Z

            # Update smoothed covariance
            g_var = np.outer(g_mean, g_mean)
            S = self.lambda_ * g_var + (1 - self.lambda_) * S

            # Monitoring statistic
            T_mean = ((2 - self.lambda_) / self.lambda_) * np.dot(Z, self.Sigma_inv @ Z)
            T_var = np.trace(self.Sigma_inv @ S)
            T = T_mean + T_var

            # Compare against control limit
            pred[i] = T > self.h

        return pred.astype(int)


class REWMV(BaseEWMA):
    """
    Robust EWMA for Monitoring Variance (REWMV).

    Focuses on detecting variance shifts in multivariate processes.
    Uses log-squared deviations to reduce the impact of outliers,
    then applies EWMA smoothing and sums across dimensions.
    """

    def fit(self, X_train: np.ndarray) -> None:
        """
        Train the REWMV chart on in-control data.
        Estimates the empirical distribution of the monitoring statistic
        and sets the control limit threshold.

        Parameters
        ----------
        X_train : np.ndarray of shape (n_samples, n_features)
            In-control training data.
        """
        # Step 1: Fit base parameters (mean, covariance, inverse covariance)
        super().fit(X_train)

        # Step 2: Initialize smoothed log-variance vector
        n = X_train.shape[0]
        Y = np.zeros(self.p)   # EWMA of log-squared deviations
        Ts = []

        # Step 3: Iterate through training samples
        for i in range(n):
            # Squared deviation from mean
            diffs = (X_train[i] - self.mu0) ** 2

            # Log-transform (robust variance signal)
            g = np.log(diffs + 1e-10)  # Add small constant to avoid log(0)

            # Update smoothed statistic
            Y = self.lambda_ * g + (1 - self.lambda_) * Y

            # Step 4: Monitoring statistic
            # Sum across dimensions → global variance change measure
            T = np.sum(Y)
            Ts.append(T)

        # Step 5: Control limit
        self.h = np.percentile(Ts, self.percentile) if Ts else np.inf

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the trained REWMV chart to new observations.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test sequence of observations.

        Returns
        -------
        pred : np.ndarray of shape (n_samples,)
            Boolean array: True if alarm (out-of-control), False otherwise.
        """
        n = X.shape[0]
        Y = np.zeros(self.p)
        pred = np.zeros(n, dtype=bool)

        for i in range(n):
            # Squared deviation
            diffs = (X[i] - self.mu0) ** 2

            # Log transform
            g = np.log(diffs + 1e-10)

            # Update smoothed statistic
            Y = self.lambda_ * g + (1 - self.lambda_) * Y

            # Monitoring statistic
            T = np.sum(Y)

            # Compare to control limit
            pred[i] = T > self.h

        return pred.astype(int)


class MaxMEWMV:
    """
    Maximum Multivariate Exponentially Weighted Moving Variance (MaxMEWMV) Control Chart.

    This method extends the concept of MEWMA (Multivariate EWMA) by monitoring
    the variance structure of the process over time. Instead of only tracking
    shifts in the mean vector, it is also sensitive to covariance changes.

    Output: Binary classification
        - 0 → In-control (normal)
        - 1 → Out-of-control (anomaly)
    """

    def __init__(self, lambda_: float = 0.2, threshold: float = 10.0) -> None:
        """
        Initialize the MaxMEWMV chart.

        Parameters
        ----------
        lambda_ : float, default=0.2
            Smoothing parameter (0 < lambda_ ≤ 1).
            Small values → long memory (slower adaptation).
            Large values → short memory (faster adaptation).
        threshold : float, default=10.0
            Control limit threshold. Values above this
            are classified as anomalies.
        """
        self.lambda_: float = lambda_
        self.threshold: float = threshold
        self.mu0: Optional[np.ndarray] = None
        self.cov0: Optional[np.ndarray] = None
        self.stat_history: List[float] = []

    def fit(self, X_train: np.ndarray) -> None:
        """
        Fit the MaxMEWMV chart using in-control training data.

        Parameters
        ----------
        X_train : np.ndarray
            2D array of shape (n_samples, n_features).
            Assumed to represent only in-control (normal) observations.
        """
        # Ensure input is 2D
        if X_train.ndim != 2:
            raise ValueError("X_train must be a 2D array.")

        # Store baseline mean and covariance (in-control reference)
        self.mu0 = np.mean(X_train, axis=0)
        self.cov0 = np.cov(X_train, rowvar=False)

        # Initialize monitoring statistic with zero
        n_features: int = X_train.shape[1]
        self.stat_history = [0.0] * n_features

        # Iterate over each sample in training to initialize control structure
        for i in range(X_train.shape[0]):
            deviation: np.ndarray = X_train[i] - self.mu0
            g: np.ndarray = np.outer(deviation, deviation)

            # Update exponentially weighted moving variance statistics
            self.stat_history = (
                self.lambda_ * np.diag(g) + (1 - self.lambda_) * np.array(self.stat_history)
            ).tolist()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies in new data.

        Parameters
        ----------
        X : np.ndarray
            2D array of shape (n_samples, n_features).
            Test or monitoring data (may include anomalies).

        Returns
        -------
        np.ndarray
            1D array of shape (n_samples,) with binary predictions:
            - 0 → Normal (in-control)
            - 1 → Anomaly (out-of-control)
        """
        if self.mu0 is None or self.cov0 is None:
            raise RuntimeError("Model must be fitted before prediction.")

        n_samples, n_features = X.shape
        predictions: List[int] = []

        # Initialize local monitoring statistic
        stat_local: List[float] = [0.0] * n_features

        for i in range(n_samples):
            deviation: np.ndarray = X[i] - self.mu0
            g: np.ndarray = np.outer(deviation, deviation)

            # Update variance statistics
            stat_local = (
                self.lambda_ * np.diag(g) + (1 - self.lambda_) * np.array(stat_local)
            ).tolist()

            # Use maximum variance component as monitoring statistic
            stat_value: float = float(np.max(stat_local))

            # Apply threshold to generate binary classification
            if stat_value > self.threshold:
                predictions.append(1)  # anomaly
            else:
                predictions.append(0)  # normal

        return np.array(predictions)


class MNSE(BaseEWMA):
    """
    Multivariate Normalized Squared Error (MNSE) Control Chart.
    
    Monitors normalized deviations using EWMA smoothing.
    """
    
    def fit(self, X_train: np.ndarray):
        """
        Train the MNSE chart on in-control data.
        
        Parameters
        ----------
        X_train : np.ndarray
            In-control training data.
        """
        super().fit(X_train)
        n = X_train.shape[0]
        Y = np.zeros(self.p)
        Ts = []
        
        for i in range(n):
            diff = X_train[i] - self.mu0
            norm = np.linalg.norm(diff)
            g = diff / norm if norm > 1e-10 else np.zeros(self.p)
            Y = self.lambda_ * g + (1 - self.lambda_) * Y
            T = np.dot(Y, Y)
            Ts.append(T)
            
        self.h = np.percentile(Ts, self.percentile) if Ts else np.inf

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the trained MNSE chart to new observations.
        
        Parameters
        ----------
        X : np.ndarray
            Test sequence of observations.
            
        Returns
        -------
        pred : np.ndarray
            Binary predictions (1 = anomaly, 0 = normal)
        """
        n = X.shape[0]
        Y = np.zeros(self.p)
        pred = np.zeros(n, dtype=bool)
        
        for i in range(n):
            diff = X[i] - self.mu0
            norm = np.linalg.norm(diff)
            g = diff / norm if norm > 1e-10 else np.zeros(self.p)
            Y = self.lambda_ * g + (1 - self.lambda_) * Y
            T = np.dot(Y, Y)
            pred[i] = T > self.h
            
        return pred.astype(int)


# Placeholder classes for adaptive variants
class AEWMA(StandardMEWMA):
    """
    Adaptive EWMA: lambda varies based on recent T_t.
    For simplicity, uses fixed lambda; extend with adaptive logic if needed.
    """
    pass


class SAMEWMA(StandardMEWMA):
    """
    Self-Adaptive MEWMA: integrate with external ML model.
    Example: def set_lambda(self, ml_model_prediction): ...
    """
    pass


class AMFEWMA(StandardMEWMA):
    """
    Adaptive Multivariate Functional EWMA: similar to standard for multivariate;
    extend for basis coefficients if data is functional.
    """
    pass