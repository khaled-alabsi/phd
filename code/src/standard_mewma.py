import numpy as np
from .base_ewma import BaseEWMA


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

        return pred
