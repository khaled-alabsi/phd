from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ResidualDiagnostics:
    """Utility plots for analysing autoencoder vs regressor residuals."""

    @staticmethod
    def scatter_true_vs_pred(
        true_residuals: np.ndarray,
        predicted_residuals: np.ndarray,
        *,
        title: str = "Residual comparison",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        ax = ax or plt.gca()
        ax.scatter(true_residuals, predicted_residuals, alpha=0.4, s=10, label="samples")
        lims = np.array([
            min(true_residuals.min(), predicted_residuals.min()),
            max(true_residuals.max(), predicted_residuals.max()),
        ])
        ax.plot(lims, lims, color="red", linestyle="--", linewidth=1.0, label="ideal")
        ax.set_xlabel("AE residual")
        ax.set_ylabel("Regressor residual")
        ax.set_title(title)
        ax.legend()
        if ax is plt.gca():
            plt.show()
        return ax

    @staticmethod
    def histogram(
        predicted_residuals: np.ndarray,
        *,
        threshold: Optional[float] = None,
        bins: int = 50,
        title: str = "Predicted residual distribution",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        ax = ax or plt.gca()
        ax.hist(predicted_residuals, bins=bins, alpha=0.7, color="steelblue")
        if threshold is not None:
            ax.axvline(threshold, color="darkorange", linestyle="--", label=f"threshold={threshold:.4f}")
            ax.legend()
        ax.set_xlabel("Predicted residual")
        ax.set_ylabel("Count")
        ax.set_title(title)
        if ax is plt.gca():
            plt.show()
        return ax

    @staticmethod
    def timeline(
        predicted_residuals: np.ndarray,
        *,
        true_residuals: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
        title: str = "Residual timeline",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        ax = ax or plt.gca()
        ax.plot(predicted_residuals, label="regressor residual", color="steelblue")
        if true_residuals is not None:
            ax.plot(true_residuals, label="AE residual", color="grey", alpha=0.6)
        if threshold is not None:
            ax.axhline(threshold, color="darkorange", linestyle="--", label=f"threshold={threshold:.4f}")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Residual value")
        ax.set_title(title)
        ax.legend()
        if ax is plt.gca():
            plt.show()
        return ax
