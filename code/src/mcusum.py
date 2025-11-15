# MCUSUM (MULTIVARIATE CUMULATIVE SUM) IMPLEMENTATION

from typing import Optional, Tuple
from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray



class MCUSUMDetector:

    def __init__(self, k = 0.5, h = None):
        """
        Args:
            k: Reference value (sensitivity parameter)
            h: Control limit (threshold for detection)
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

    def predict(self, X_test: NDArray[np.float64]):
        if not self.is_fitted:
            raise ValueError("MCUSUM detector must be fitted before prediction")

        statistics = self._compute_mcusum_scores(X_test, self.mu_0, self.sigma, self.k)
        flags = statistics > self.h

        return statistics, flags

    @staticmethod
    def _estimate_incontrol_parameters(X_incontrol: NDArray[np.float64]) :
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
                              mu_0,
                              sigma,
                              k):
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
                              verbose: bool = False):
        """
        Estimate control limit using Monte Carlo simulation.

        Args:
            X_incontrol: In-control training data
            n_simulations: Number of Monte Carlo simulations
            percentile: Percentile for control limit

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
    def compute_reference_value_k(delta,
                                 sigma):
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


# ------


def plot_mcusum_diagnostics(
    statistics_normal: NDArray[np.float64],
    statistics_anomaly: Optional[NDArray[np.float64]] = None,
    h: float | None = 0.0,
    title_suffix: str = "",
    use_log: bool = True,
    diff_threshold: float = 1e-3
) -> None:
    """
    Create MCUSUM diagnostic plots with markers for differences and control limit crossings.
    """
    rows = 3 if statistics_anomaly is not None else 1
    fig, axs = plt.subplots(rows, 2, figsize=(15, 5 * rows))
    if rows == 1:
        axs = np.array([axs])

    # Helper to set y-axis log if requested
    def maybe_log(ax):
        if use_log:
            ax.set_yscale("log")

    # Normal histogram
    axs[0, 0].hist(statistics_normal, bins=50, alpha=0.7, color='blue', density=True)
    axs[0, 0].axvline(h, color='red', linestyle='--', linewidth=2, label=f'Control Limit (h={h:.2f})')
    maybe_log(axs[0, 0])
    axs[0, 0].set_xlabel("MCUSUM Statistic")
    axs[0, 0].set_ylabel("Density")
    axs[0, 0].set_title(f"Normal Data Histogram {title_suffix}")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # Normal time series
    axs[0, 1].plot(statistics_normal, color='blue', alpha=0.7, label="Normal Data")
    axs[0, 1].axhline(h, color='red', linestyle='--', linewidth=2, label=f'Control Limit (h={h:.2f})')
    maybe_log(axs[0, 1])
    axs[0, 1].set_xlabel("Sample Index")
    axs[0, 1].set_ylabel("MCUSUM Statistic")
    axs[0, 1].set_title(f"Normal Data Over Time {title_suffix}")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    if statistics_anomaly is not None:
        # Anomaly histogram
        axs[1, 0].hist(statistics_anomaly, bins=50, alpha=0.7, color='red', density=True)
        axs[1, 0].axvline(h, color='red', linestyle='--', linewidth=2, label=f'Control Limit (h={h:.2f})')
        maybe_log(axs[1, 0])
        axs[1, 0].set_xlabel("MCUSUM Statistic")
        axs[1, 0].set_ylabel("Density")
        axs[1, 0].set_title(f"Anomaly Data Histogram {title_suffix}")
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)

        # Anomaly time series
        axs[1, 1].plot(statistics_anomaly, color='red', alpha=0.7, label="Anomaly Data")
        axs[1, 1].axhline(h, color='red', linestyle='--', linewidth=2, label=f'Control Limit (h={h:.2f})')
        maybe_log(axs[1, 1])
        axs[1, 1].set_xlabel("Sample Index")
        axs[1, 1].set_ylabel("MCUSUM Statistic")
        axs[1, 1].set_title(f"Anomaly Data Over Time {title_suffix}")
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)

        # Overlay plot
        axs[2, 0].plot(statistics_normal, color='blue', alpha=0.7, label="Normal Data")
        axs[2, 0].plot(statistics_anomaly, color='red', alpha=0.7, label="Anomaly Data")
        axs[2, 0].axhline(h, color='red', linestyle='--', linewidth=2, label=f'Control Limit (h={h:.2f})')
        maybe_log(axs[2, 0])
        axs[2, 0].set_xlabel("Sample Index")
        axs[2, 0].set_ylabel("MCUSUM Statistic")
        axs[2, 0].set_title(f"Normal vs Anomaly Overlay {title_suffix}")
        axs[2, 0].grid(True, alpha=0.3)

        # First point above threshold difference
        diff = np.abs(statistics_normal - statistics_anomaly)
        if np.any(diff > diff_threshold):
            idx_diff = np.argmax(diff > diff_threshold)
            val_norm = statistics_normal[idx_diff]
            val_anom = statistics_anomaly[idx_diff]
            axs[2, 0].axvline(idx_diff, color='green', linestyle='--', linewidth=2,
                               label=f'First Diff (idx={idx_diff})')
            # axs[2, 0].annotate(f"{val_norm:.4f} / {val_anom:.4f}",
            #                     xy=(idx_diff, max(val_norm, val_anom)),
            #                     xytext=(idx_diff + 5, max(val_norm, val_anom) * 1.05),
            #                     arrowprops=dict(arrowstyle="->", color='green'),
            #                     color='green')

        # First point above control limit
        for series, color, label_prefix in zip([statistics_normal, statistics_anomaly],
                                               ['purple', 'orange'],
                                               ['Normal', 'Anomaly']):
            if np.any(series > h):
                idx_h = np.argmax(series > h)
                val_h = series[idx_h]
                axs[2, 0].axvline(idx_h, color=color, linestyle='--', linewidth=2,
                                   label=f'{label_prefix} first above h (idx={idx_h})')
                # axs[2, 0].annotate(f"{val_h:.4f}",
                #                     xy=(idx_h, val_h),
                #                     xytext=(idx_h + 5, val_h * 1.05),
                #                     arrowprops=dict(arrowstyle="->", color=color),
                #                     color=color)

        axs[2, 0].legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
        fig.delaxes(axs[2, 1])  # remove unused subplot

    plt.tight_layout()
    plt.show()
