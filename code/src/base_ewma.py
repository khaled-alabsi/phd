import pandas as pd
import numpy as np
from scipy.linalg import eigh
from typing import Dict, Any, List

# Define classes for MEWMA variants

import numpy as np

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


