"""
Advanced Visualization Tools for Elliptical Distributions

This module provides specialized plotting functions for elliptical distributions,
including contour plots, 3D visualizations, and diagnostic plots.

Author: PhD Project
Date: September 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy import stats
from elliptical_distributions import MultivariateNormal, MultivariateT, MultivariateLaplace

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_density_surface(distribution, xlim=(-4, 4), ylim=(-4, 4), resolution=50):
    """
    Plot 3D density surface for 2D elliptical distribution.
    
    Parameters:
    -----------
    distribution : EllipticalDistribution
        Distribution to plot
    xlim, ylim : tuple
        Plot limits
    resolution : int
        Grid resolution
    """
    if distribution.p != 2:
        raise ValueError("3D plots only available for 2D distributions")
    
    # Create grid
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute density
    pos = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = np.exp(distribution.log_pdf(pos)).reshape(X.shape)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    
    # Surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax1.set_xlabel('X₁')
    ax1.set_ylabel('X₂')
    ax1.set_zlabel('Density')
    ax1.set_title(f'3D Density Surface\n{distribution.distribution_name}')
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, levels=10, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.set_xlabel('X₁')
    ax2.set_ylabel('X₂')
    ax2.set_title(f'Density Contours\n{distribution.distribution_name}')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_qq_elliptical(X, distribution, ax=None):
    """
    Q-Q plot for elliptical distribution assumption.
    
    Parameters:
    -----------
    X : array, shape (n, p)
        Observations
    distribution : EllipticalDistribution
        Fitted distribution
    ax : matplotlib axis, optional
        Axis to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Compute Mahalanobis distances
    distances = distribution.mahalanobis_distance(X)
    n = len(distances)
    
    # Sort distances
    sorted_distances = np.sort(distances)
    
    # Theoretical quantiles (chi-squared with p degrees of freedom)
    p = distribution.p
    theoretical_quantiles = stats.chi2.ppf(np.arange(1, n+1) / (n+1), df=p)
    
    # Plot
    ax.scatter(theoretical_quantiles, sorted_distances, alpha=0.6)
    
    # Add reference line
    min_val = min(theoretical_quantiles.min(), sorted_distances.min())
    max_val = max(theoretical_quantiles.max(), sorted_distances.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    ax.set_xlabel(f'Theoretical Quantiles (χ²({p}))')
    ax.set_ylabel('Sample Quantiles (Mahalanobis Distance²)')
    ax.set_title('Q-Q Plot for Elliptical Distribution')
    ax.grid(True, alpha=0.3)
    
    # Compute correlation coefficient
    corr = np.corrcoef(theoretical_quantiles, sorted_distances)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
            transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='white'))
    
    return ax

def plot_tail_comparison(distributions, n_samples=1000, quantiles=[0.9, 0.95, 0.99]):
    """
    Compare tail behavior of different elliptical distributions.
    
    Parameters:
    -----------
    distributions : list
        List of fitted elliptical distributions
    n_samples : int
        Number of samples for empirical quantiles
    quantiles : list
        Quantiles to compare
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Generate samples and compute distances
    all_distances = []
    labels = []
    
    for dist in distributions:
        X = dist.sample(n_samples)
        distances = np.sqrt(dist.mahalanobis_distance(X))
        all_distances.append(distances)
        labels.append(dist.distribution_name)
    
    # Box plots
    ax1 = axes[0]
    ax1.boxplot(all_distances, labels=labels)
    ax1.set_ylabel('Mahalanobis Distance')
    ax1.set_title('Distribution of Mahalanobis Distances')
    ax1.grid(True, alpha=0.3)
    
    # Quantile-quantile plot between distributions
    ax2 = axes[1]
    base_dist = all_distances[0]  # Use first as reference
    colors = plt.cm.tab10(np.linspace(0, 1, len(distributions)))
    
    for i, (distances, label) in enumerate(zip(all_distances[1:], labels[1:]), 1):
        # Compute empirical quantiles
        q_base = np.percentile(base_dist, np.linspace(1, 99, 50))
        q_current = np.percentile(distances, np.linspace(1, 99, 50))
        
        ax2.scatter(q_base, q_current, alpha=0.7, 
                   color=colors[i], label=label, s=30)
    
    # Reference line
    min_val = min([min(d) for d in all_distances])
    max_val = max([max(d) for d in all_distances])
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
    
    ax2.set_xlabel(f'{labels[0]} Quantiles')
    ax2.set_ylabel('Other Distribution Quantiles')
    ax2.set_title('Quantile-Quantile Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_robustness_comparison(mu_true, Sigma_true, contamination_levels=[0, 0.05, 0.1, 0.2]):
    """
    Compare robustness of different estimators under contamination.
    
    Parameters:
    -----------
    mu_true : array
        True location parameter
    Sigma_true : array
        True scatter matrix
    contamination_levels : list
        Fraction of contaminated observations
    """
    n_clean = 200
    n_trials = 50
    
    estimator_names = ['Sample', 'Tyler', 'MCD']
    results = {name: {'mu_errors': [], 'sigma_errors': []} 
               for name in estimator_names}
    
    for cont_level in contamination_levels:
        mu_errors = {name: [] for name in estimator_names}
        sigma_errors = {name: [] for name in estimator_names}
        
        for trial in range(n_trials):
            # Generate clean data
            true_dist = MultivariateNormal(mu_true, Sigma_true)
            X_clean = true_dist.sample(n_clean)
            
            # Add contamination
            if cont_level > 0:
                n_cont = int(cont_level * n_clean)
                outliers = mu_true + 5 * np.random.randn(n_cont, len(mu_true))
                X = np.vstack([X_clean, outliers])
            else:
                X = X_clean
            
            # Estimate with different methods
            estimators = {
                'Sample': (np.mean(X, axis=0), np.cov(X.T)),
                'Tyler': (np.mean(X, axis=0), 
                         RobustEstimators.tyler_estimator(X - np.mean(X, axis=0))),
                'MCD': RobustEstimators.mcd_estimator(X)
            }
            
            for name, (mu_est, Sigma_est) in estimators.items():
                mu_error = np.linalg.norm(mu_est - mu_true)
                sigma_error = np.linalg.norm(Sigma_est - Sigma_true, 'fro')
                
                mu_errors[name].append(mu_error)
                sigma_errors[name].append(sigma_error)
        
        for name in estimator_names:
            results[name]['mu_errors'].append(mu_errors[name])
            results[name]['sigma_errors'].append(sigma_errors[name])
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Location parameter errors
    ax1 = axes[0]
    positions = np.arange(len(contamination_levels))
    width = 0.25
    
    for i, name in enumerate(estimator_names):
        means = [np.mean(errors) for errors in results[name]['mu_errors']]
        stds = [np.std(errors) for errors in results[name]['mu_errors']]
        
        ax1.bar(positions + i*width, means, width, 
               yerr=stds, label=name, alpha=0.7)
    
    ax1.set_xlabel('Contamination Level')
    ax1.set_ylabel('Location Parameter Error')
    ax1.set_title('Robustness: Location Parameter')
    ax1.set_xticks(positions + width)
    ax1.set_xticklabels([f'{100*c:.0f}%' for c in contamination_levels])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter parameter errors
    ax2 = axes[1]
    for i, name in enumerate(estimator_names):
        means = [np.mean(errors) for errors in results[name]['sigma_errors']]
        stds = [np.std(errors) for errors in results[name]['sigma_errors']]
        
        ax2.bar(positions + i*width, means, width, 
               yerr=stds, label=name, alpha=0.7)
    
    ax2.set_xlabel('Contamination Level')
    ax2.set_ylabel('Scatter Matrix Error (Frobenius)')
    ax2.set_title('Robustness: Scatter Parameter')
    ax2.set_xticks(positions + width)
    ax2.set_xticklabels([f'{100*c:.0f}%' for c in contamination_levels])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, results

def comprehensive_visualization_demo():
    """
    Comprehensive demonstration of visualization capabilities.
    """
    print("Running comprehensive visualization demo...")
    
    # Set parameters
    mu = np.array([1, -0.5])
    Sigma = np.array([[2, 0.8], [0.8, 1]])
    
    # Create distributions
    normal = MultivariateNormal(mu, Sigma)
    t_dist = MultivariateT(mu, Sigma, nu=4)
    laplace = MultivariateLaplace(mu, Sigma)
    
    distributions = [normal, t_dist, laplace]
    
    # 1. 3D density plots
    print("1. Creating 3D density plots...")
    for i, dist in enumerate(distributions):
        fig = plot_density_surface(dist)
        plt.savefig(f'density_3d_{i+1}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Generate sample data for diagnostics
    print("2. Generating sample data and diagnostic plots...")
    n_samples = 500
    X_normal = normal.sample(n_samples)
    
    # Fit different distributions to normal data
    t_fitted = MultivariateT(mu, Sigma, nu=4)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plot_qq_elliptical(X_normal, normal, axes[0])
    plot_qq_elliptical(X_normal, t_fitted, axes[1])
    plt.savefig('qq_plots_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Tail behavior comparison
    print("3. Comparing tail behaviors...")
    fig = plot_tail_comparison(distributions)
    plt.savefig('tail_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Robustness analysis
    print("4. Analyzing robustness properties...")
    from elliptical_distributions import RobustEstimators
    
    fig, results = plot_robustness_comparison(mu, Sigma)
    plt.savefig('robustness_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualization demo complete! Check generated plots.")
    
    return results

if __name__ == "__main__":
    print("Advanced Visualization Tools for Elliptical Distributions")
    print("="*60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run comprehensive demo
    results = comprehensive_visualization_demo()
    
    print("\nAll visualizations have been saved as PNG files.")
    print("Generated files:")
    print("- density_3d_1.png (Normal distribution)")
    print("- density_3d_2.png (t-distribution)")
    print("- density_3d_3.png (Laplace distribution)")
    print("- qq_plots_comparison.png")
    print("- tail_comparison.png")
    print("- robustness_analysis.png")