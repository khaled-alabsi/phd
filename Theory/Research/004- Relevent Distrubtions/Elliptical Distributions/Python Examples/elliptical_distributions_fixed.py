"""
Fixed Elliptical Distributions: Simulation and Analysis Tools

This module provides comprehensive tools for working with elliptical distributions,
including simulation, parameter estimation, and visualization capabilities.

Author: PhD Project
Date: September 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import gamma, kv
from scipy.optimize import minimize
from sklearn.covariance import EmpiricalCovariance, MinCovDet
import warnings
import os

warnings.filterwarnings('ignore')

class EllipticalDistribution:
    """
    Base class for elliptical distributions with common functionality.
    """
    
    def __init__(self, mu, Sigma, generator_params=None):
        """
        Initialize elliptical distribution.
        
        Parameters:
        -----------
        mu : array-like, shape (p,)
            Location parameter (mean vector)
        Sigma : array-like, shape (p, p)
            Scatter matrix (positive definite)
        generator_params : dict, optional
            Parameters specific to the generator function
        """
        self.mu = np.array(mu)
        self.Sigma = np.array(Sigma)
        self.p = len(mu)
        self.generator_params = generator_params or {}
        
        # Validate inputs
        self._validate_parameters()
        
        # Compute Cholesky decomposition for efficient sampling
        self.L = np.linalg.cholesky(self.Sigma)
    
    def _validate_parameters(self):
        """Validate input parameters."""
        if self.Sigma.shape != (self.p, self.p):
            raise ValueError("Sigma must be p x p matrix")
        
        # Check positive definiteness
        eigenvals = np.linalg.eigvals(self.Sigma)
        if np.any(eigenvals <= 0):
            raise ValueError("Sigma must be positive definite")
    
    def mahalanobis_distance(self, X):
        """
        Compute Mahalanobis distance for observations.
        
        Parameters:
        -----------
        X : array-like, shape (n, p)
            Observations
            
        Returns:
        --------
        distances : array, shape (n,)
            Squared Mahalanobis distances
        """
        X = np.atleast_2d(X)
        centered = X - self.mu
        Sigma_inv = np.linalg.inv(self.Sigma)
        distances = np.sum(centered @ Sigma_inv * centered, axis=1)
        return distances
    
    def generate_uniform_sphere(self, n_samples):
        """
        Generate uniform points on unit sphere.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        U : array, shape (n_samples, p)
            Uniform points on unit sphere
        """
        # Generate from standard normal and normalize
        Z = np.random.randn(n_samples, self.p)
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        U = Z / norms
        return U
    
    def plot_contours(self, ax=None, levels=[0.5, 0.9, 0.95, 0.99], 
                     colors=None, linestyles='-'):
        """
        Plot probability contours for 2D distributions.
        
        Parameters:
        -----------
        ax : matplotlib axis, optional
            Axis to plot on
        levels : list
            Probability levels for contours
        colors : list, optional
            Colors for contours
        linestyles : str or list
            Line styles for contours
        """
        if self.p != 2:
            raise ValueError("Contour plots only available for 2D distributions")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create grid
        x_range = 3 * np.sqrt(self.Sigma[0, 0])
        y_range = 3 * np.sqrt(self.Sigma[1, 1])
        
        x = np.linspace(self.mu[0] - x_range, self.mu[0] + x_range, 100)
        y = np.linspace(self.mu[1] - y_range, self.mu[1] + y_range, 100)
        X, Y = np.meshgrid(x, y)
        
        # Compute distances
        pos = np.stack([X.ravel(), Y.ravel()], axis=1)
        distances = self.mahalanobis_distance(pos)
        Z = distances.reshape(X.shape)
        
        # Convert probability levels to distance levels
        chi2_levels = [stats.chi2.ppf(level, df=2) for level in levels]
        
        # Plot contours
        if colors is None:
            cmap = plt.get_cmap('viridis')
            colors = [cmap(i / (len(levels) - 1)) for i in range(len(levels))]
        
        contours = ax.contour(X, Y, Z, levels=chi2_levels, 
                             colors=colors, linestyles=linestyles)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.0f%%')
        
        ax.set_xlabel('X₁')
        ax.set_ylabel('X₂')
        ax.set_title('Probability Contours')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        return ax


class MultivariateNormal(EllipticalDistribution):
    """Multivariate Normal Distribution"""
    
    def __init__(self, mu, Sigma):
        super().__init__(mu, Sigma)
        self.distribution_name = "Multivariate Normal"
    
    def sample(self, n_samples):
        """
        Generate samples from multivariate normal distribution.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        X : array, shape (n_samples, p)
            Generated samples
        """
        # Standard approach: mu + L @ Z where Z ~ N(0, I)
        Z = np.random.randn(n_samples, self.p)
        X = self.mu + Z @ self.L.T
        return X
    
    def log_pdf(self, X):
        """Compute log probability density."""
        X = np.atleast_2d(X)
        distances = self.mahalanobis_distance(X)
        log_det = 2 * np.sum(np.log(np.diag(self.L)))
        
        log_density = (-0.5 * self.p * np.log(2 * np.pi) 
                      - 0.5 * log_det 
                      - 0.5 * distances)
        return log_density
    
    def fit(self, X):
        """Fit parameters using MLE."""
        X = np.atleast_2d(X)
        n = X.shape[0]
        
        self.mu = np.mean(X, axis=0)
        centered = X - self.mu
        self.Sigma = (centered.T @ centered) / n
        self.L = np.linalg.cholesky(self.Sigma)
        
        return self


class MultivariateT(EllipticalDistribution):
    """Multivariate Student's t-Distribution"""
    
    def __init__(self, mu, Sigma, nu):
        super().__init__(mu, Sigma, {'nu': nu})
        self.nu = nu
        self.distribution_name = f"Multivariate t (ν={nu})"
    
    def sample(self, n_samples):
        """
        Generate samples using the scale mixture representation.
        
        X = μ + √(ν/W) * A * U
        where W ~ Gamma(ν/2, ν/2) and U ~ N(0, I)
        """
        # Generate gamma random variables
        W = np.random.gamma(self.nu / 2, 2 / self.nu, n_samples)
        
        # Generate standard normal
        Z = np.random.randn(n_samples, self.p)
        
        # Scale and transform
        X = self.mu + (Z @ self.L.T) / np.sqrt(W[:, np.newaxis])
        
        return X
    
    def log_pdf(self, X):
        """Compute log probability density."""
        X = np.atleast_2d(X)
        distances = self.mahalanobis_distance(X)
        log_det = 2 * np.sum(np.log(np.diag(self.L)))
        
        # Log normalization constant
        log_norm = (gamma((self.nu + self.p) / 2) - 
                   gamma(self.nu / 2) - 
                   0.5 * self.p * np.log(self.nu * np.pi) -
                   0.5 * log_det)
        
        # Log density
        log_density = (log_norm - 
                      0.5 * (self.nu + self.p) * np.log(1 + distances / self.nu))
        
        return log_density
    
    def fit_em(self, X, max_iter=100, tol=1e-6):
        """
        Fit parameters using EM algorithm.
        
        Parameters:
        -----------
        X : array, shape (n, p)
            Observations
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
        """
        X = np.atleast_2d(X)
        n = X.shape[0]
        
        # Initialize with sample moments
        self.mu = np.mean(X, axis=0)
        self.Sigma = np.cov(X.T)
        self.L = np.linalg.cholesky(self.Sigma)
        
        prev_loglik = -np.inf
        
        for iteration in range(max_iter):
            # E-step: compute weights
            distances = self.mahalanobis_distance(X)
            weights = (self.nu + self.p) / (self.nu + distances)
            
            # M-step: update parameters
            sum_weights = np.sum(weights)
            self.mu = np.sum(weights[:, np.newaxis] * X, axis=0) / sum_weights
            
            centered = X - self.mu
            self.Sigma = (centered.T * weights) @ centered / n
            self.L = np.linalg.cholesky(self.Sigma)
            
            # Check convergence
            loglik = np.sum(self.log_pdf(X))
            if abs(loglik - prev_loglik) < tol:
                break
            prev_loglik = loglik
        
        return self


class MultivariateLaplace(EllipticalDistribution):
    """Multivariate Laplace Distribution"""
    
    def __init__(self, mu, Sigma):
        super().__init__(mu, Sigma)
        self.distribution_name = "Multivariate Laplace"
    
    def sample(self, n_samples):
        """
        Generate samples using exponential scale mixture.
        
        X = μ + √W * A * U
        where W ~ Exponential(1) and U ~ Uniform(sphere)
        """
        # Generate exponential random variables
        W = np.random.exponential(1, n_samples)
        
        # Generate uniform on sphere
        U = self.generate_uniform_sphere(n_samples)
        
        # Transform
        X = self.mu + np.sqrt(W[:, np.newaxis]) * (U @ self.L.T)
        
        return X
    
    def log_pdf(self, X):
        """Compute log probability density (approximation)."""
        X = np.atleast_2d(X)
        distances = self.mahalanobis_distance(X)
        log_det = 2 * np.sum(np.log(np.diag(self.L)))
        
        # Approximation for multivariate Laplace
        log_density = (-0.5 * self.p * np.log(2 * np.pi) 
                      - 0.5 * log_det 
                      - np.sqrt(2 * distances))
        return log_density


class RobustEstimators:
    """Collection of robust estimators for elliptical distributions"""
    
    @staticmethod
    def tyler_estimator(X, max_iter=100, tol=1e-6):
        """
        Tyler's M-estimator for scatter matrix.
        
        Parameters:
        -----------
        X : array, shape (n, p)
            Observations (assumed centered)
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
            
        Returns:
        --------
        Sigma : array, shape (p, p)
            Tyler's scatter matrix estimate
        """
        X = np.atleast_2d(X)
        n, p = X.shape
        
        # Initialize with sample covariance
        Sigma = np.cov(X.T)
        
        for iteration in range(max_iter):
            Sigma_inv = np.linalg.inv(Sigma)
            
            # Compute weights
            distances = np.sum(X @ Sigma_inv * X, axis=1)
            weights = p / distances
            
            # Update scatter matrix
            Sigma_new = np.zeros((p, p))
            for i in range(n):
                Sigma_new += weights[i] * np.outer(X[i], X[i])
            Sigma_new = Sigma_new / n
            
            # Check convergence
            if np.linalg.norm(Sigma_new - Sigma, 'fro') < tol:
                break
            
            Sigma = Sigma_new
        
        return Sigma
    
    @staticmethod
    def mcd_estimator(X, h=None):
        """
        Minimum Covariance Determinant estimator.
        
        Parameters:
        -----------
        X : array, shape (n, p)
            Observations
        h : int, optional
            Subset size (default: (n+p+1)//2)
            
        Returns:
        --------
        mu : array, shape (p,)
            Robust location estimate
        Sigma : array, shape (p, p)
            Robust scatter estimate
        """
        mcd = MinCovDet(assume_centered=False, support_fraction=h)
        mcd.fit(X)
        
        return mcd.location_, mcd.covariance_


def set_output_directory():
    """Set output directory to current script location."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    return script_dir


def compare_distributions_demo():
    """
    Demonstration comparing different elliptical distributions.
    """
    # Set output directory
    output_dir = set_output_directory()
    
    # Set parameters
    mu = np.array([0, 0])
    Sigma = np.array([[2, 1], [1, 1.5]])
    n_samples = 1000
    
    # Create distributions
    normal = MultivariateNormal(mu, Sigma)
    t_heavy = MultivariateT(mu, Sigma, nu=3)
    t_light = MultivariateT(mu, Sigma, nu=10)
    laplace = MultivariateLaplace(mu, Sigma)
    
    distributions = [normal, t_heavy, t_light, laplace]
    
    # Generate samples
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, dist in enumerate(distributions):
        X = dist.sample(n_samples)
        
        ax = axes[i]
        ax.scatter(X[:, 0], X[:, 1], alpha=0.6, s=20)
        dist.plot_contours(ax)
        ax.set_title(dist.distribution_name)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'distribution_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return distributions


def parameter_estimation_demo():
    """
    Demonstration of parameter estimation methods.
    """
    # Set output directory
    output_dir = set_output_directory()
    
    # Generate contaminated data
    np.random.seed(42)
    
    # True parameters
    mu_true = np.array([2, -1])
    Sigma_true = np.array([[4, 1], [1, 2]])
    
    # Generate clean data
    normal = MultivariateNormal(mu_true, Sigma_true)
    X_clean = normal.sample(200)
    
    # Add outliers
    n_outliers = 20
    outliers = mu_true + 5 * np.random.randn(n_outliers, 2)
    X_contaminated = np.vstack([X_clean, outliers])
    
    # Estimate parameters
    estimators = {
        'Sample Mean/Cov': (np.mean(X_contaminated, axis=0), 
                           np.cov(X_contaminated.T)),
        'Tyler Estimator': (np.mean(X_contaminated, axis=0),
                           RobustEstimators.tyler_estimator(
                               X_contaminated - np.mean(X_contaminated, axis=0))),
        'MCD Estimator': RobustEstimators.mcd_estimator(X_contaminated)
    }
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (name, (mu_est, Sigma_est)) in enumerate(estimators.items()):
        ax = axes[i]
        
        # Plot data
        ax.scatter(X_clean[:, 0], X_clean[:, 1], 
                  alpha=0.6, c='blue', s=20, label='Clean data')
        ax.scatter(outliers[:, 0], outliers[:, 1], 
                  alpha=0.8, c='red', s=30, label='Outliers')
        
        # Plot estimated contours
        dist_est = MultivariateNormal(mu_est, Sigma_est)
        dist_est.plot_contours(ax, colors=['green'])
        
        # Plot true contours
        normal.plot_contours(ax, colors=['black'], linestyles='--')
        
        ax.set_title(f'{name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'parameter_estimation_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print numerical results
    print("Parameter Estimation Results:")
    print("="*50)
    print(f"True μ: {mu_true}")
    print(f"True Σ diagonal: {np.diag(Sigma_true)}")
    print()
    
    for name, (mu_est, Sigma_est) in estimators.items():
        print(f"{name}:")
        print(f"  Estimated μ: {mu_est}")
        print(f"  Estimated Σ diagonal: {np.diag(Sigma_est)}")
        print(f"  μ error: {np.linalg.norm(mu_est - mu_true):.3f}")
        print()


if __name__ == "__main__":
    print("Elliptical Distributions Demonstration")
    print("="*40)
    
    # Run demonstrations
    print("\n1. Comparing different elliptical distributions...")
    distributions = compare_distributions_demo()
    
    print("\n2. Parameter estimation with contaminated data...")
    parameter_estimation_demo()
    
    print("\nDemonstration complete! Check generated plots.")