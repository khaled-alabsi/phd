"""
Gaussian Comparison Generator for Elliptical Distributions

This script generates detailed comparison plots showing how different elliptical 
distributions compare to Gaussian distributions using the same data samples.
Focuses on tail behavior and probability density differences.

Author: PhD Project
Date: September 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys

# Import our fixed elliptical distributions module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Python Examples'))
from elliptical_distributions_fixed import (
    MultivariateNormal, MultivariateT, MultivariateLaplace
)

# Set style for consistent plots
plt.style.use('default')
sns.set_palette("tab10")

def generate_gaussian_tail_comparison():
    """Generate detailed comparison of tail behaviors against Gaussian baseline."""
    
    print("Generating Gaussian vs Elliptical tail comparison...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Parameters
    mu = np.array([0, 0])
    Sigma = np.array([[1, 0.3], [0.3, 1]])
    n_samples = 1000
    
    # Define distributions to compare
    distributions = [
        (MultivariateNormal(mu, Sigma), "Multivariate Normal", "blue"),
        (MultivariateT(mu, Sigma, nu=3), "Student's t (ν=3)", "red"),
        (MultivariateT(mu, Sigma, nu=10), "Student's t (ν=10)", "orange"),
        (MultivariateLaplace(mu, Sigma), "Multivariate Laplace", "green")
    ]
    
    # Create Gaussian reference
    gaussian_ref = MultivariateNormal(mu, Sigma)
    
    # 1. Scatter plots comparison (2x2 grid)
    for i, (dist, name, color) in enumerate(distributions):
        ax = plt.subplot(3, 4, i + 1)
        
        # Generate samples
        X = dist.sample(n_samples)
        
        # Plot samples
        ax.scatter(X[:, 0], X[:, 1], alpha=0.6, s=15, color=color, label=name)
        
        # Plot contours
        x_range = np.linspace(-4, 4, 50)
        y_range = np.linspace(-4, 4, 50)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        pos = np.stack([X_grid.ravel(), Y_grid.ravel()], axis=1)
        
        # Plot distribution contours
        Z = np.exp(dist.log_pdf(pos)).reshape(X_grid.shape)
        ax.contour(X_grid, Y_grid, Z, levels=5, colors=color, alpha=0.8)
        
        # Plot Gaussian reference contours
        Z_gauss = np.exp(gaussian_ref.log_pdf(pos)).reshape(X_grid.shape)
        ax.contour(X_grid, Y_grid, Z_gauss, levels=5, colors='black', 
                  linestyles='--', alpha=0.5)
        
        ax.set_title(f'{name}\nvs Gaussian Reference (dashed)', fontweight='bold')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    # 2. Mahalanobis distance distributions
    ax_dist = plt.subplot(3, 2, 3)
    
    # Compute Mahalanobis distances for each distribution
    for dist, name, color in distributions:
        X = dist.sample(n_samples)
        distances = np.sqrt(gaussian_ref.mahalanobis_distance(X))
        
        # Plot histogram
        ax_dist.hist(distances, bins=30, alpha=0.6, density=True, 
                    label=name, color=color)
    
    # Add theoretical chi distribution (for Gaussian)
    x_theory = np.linspace(0, 5, 200)
    chi_pdf = stats.chi.pdf(x_theory, df=2)
    ax_dist.plot(x_theory, chi_pdf, 'k--', linewidth=2, 
                label='Theoretical χ (Gaussian)')
    
    ax_dist.set_xlabel('Mahalanobis Distance')
    ax_dist.set_ylabel('Density')
    ax_dist.set_title('Distance Distribution Comparison', fontweight='bold')
    ax_dist.legend()
    ax_dist.grid(True, alpha=0.3)
    
    # 3. Tail probability comparison
    ax_tail = plt.subplot(3, 2, 4)
    
    thresholds = np.linspace(1, 4, 20)
    
    for dist, name, color in distributions:
        X = dist.sample(n_samples)
        distances = np.sqrt(gaussian_ref.mahalanobis_distance(X))
        tail_probs = [np.mean(distances > t) for t in thresholds]
        
        ax_tail.semilogy(thresholds, tail_probs, 'o-', linewidth=2, 
                        label=name, color=color)
    
    # Theoretical tail probabilities for Gaussian
    theoretical_probs = [1 - stats.chi2.cdf(t**2, df=2) for t in thresholds]
    ax_tail.semilogy(thresholds, theoretical_probs, 'k--', linewidth=2, 
                    label='Theoretical (Gaussian)')
    
    ax_tail.set_xlabel('Distance Threshold')
    ax_tail.set_ylabel('P(Distance > Threshold)')
    ax_tail.set_title('Tail Probability Comparison', fontweight='bold')
    ax_tail.legend()
    ax_tail.grid(True, alpha=0.3)
    
    # 4. 1D marginal comparison
    ax_marg = plt.subplot(3, 1, 3)
    
    x_1d = np.linspace(-4, 4, 200)
    
    # Plot 1D marginals
    for dist, name, color in distributions:
        X = dist.sample(n_samples)
        ax_marg.hist(X[:, 0], bins=30, alpha=0.4, density=True, 
                    label=f'{name} (samples)', color=color)
    
    # Theoretical curves
    ax_marg.plot(x_1d, stats.norm.pdf(x_1d, 0, 1), 'b-', linewidth=2, 
                label='Normal PDF')
    ax_marg.plot(x_1d, stats.t.pdf(x_1d, 3), 'r-', linewidth=2, 
                label='t PDF (ν=3)')
    ax_marg.plot(x_1d, stats.laplace.pdf(x_1d, scale=1/np.sqrt(2)), 'g-', 
                linewidth=2, label='Laplace PDF')
    
    ax_marg.set_xlabel('Value')
    ax_marg.set_ylabel('Density')
    ax_marg.set_title('1D Marginal Distribution Comparison', fontweight='bold')
    ax_marg.legend(ncol=2)
    ax_marg.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gaussian_tail_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Gaussian tail comparison plot generated")

def generate_distribution_catalog():
    """Generate a comprehensive catalog of elliptical distributions."""
    
    print("Generating elliptical distributions catalog...")
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    axes = axes.ravel()
    
    # Parameters
    mu = np.array([0, 0])
    Sigma = np.array([[1, 0.5], [0.5, 1]])
    n_samples = 300
    
    # Comprehensive list of elliptical distributions
    distributions = [
        (MultivariateNormal(mu, Sigma), "Multivariate Normal\n(Gaussian)", "Most common,\nlight tails"),
        (MultivariateT(mu, Sigma, nu=1), "Student's t (ν=1)\n(Cauchy)", "Very heavy tails,\nno finite variance"),
        (MultivariateT(mu, Sigma, nu=3), "Student's t (ν=3)", "Heavy tails,\nfinite variance"),
        (MultivariateT(mu, Sigma, nu=10), "Student's t (ν=10)", "Moderate tails,\napproaches Normal"),
        (MultivariateLaplace(mu, Sigma), "Multivariate Laplace", "Exponential tails,\nsparsity-inducing"),
        (MultivariateNormal(mu, [[2, 0], [0, 0.5]]), "Normal (Anisotropic)", "Different variances\nin each direction"),
        (MultivariateT(mu, [[0.5, 0], [0, 2]], nu=4), "t-Distribution\n(Anisotropic)", "Heavy tails with\ndifferent scales"),
        (MultivariateNormal(mu, [[1, 0.9], [0.9, 1]]), "Normal (High Corr.)", "Strong positive\ncorrelation"),
        (MultivariateNormal(mu, [[1, -0.7], [-0.7, 1]]), "Normal (Neg. Corr.)", "Strong negative\ncorrelation")
    ]
    
    for i, (dist, name, description) in enumerate(distributions):
        ax = axes[i]
        
        # Generate samples
        X = dist.sample(n_samples)
        
        # Create scatter plot
        ax.scatter(X[:, 0], X[:, 1], alpha=0.6, s=12, c='steelblue')
        
        # Plot contours
        x_range = np.linspace(-4, 4, 50)
        y_range = np.linspace(-4, 4, 50)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        pos = np.stack([X_grid.ravel(), Y_grid.ravel()], axis=1)
        
        Z = np.exp(dist.log_pdf(pos)).reshape(X_grid.shape)
        contours = ax.contour(X_grid, Y_grid, Z, levels=4, colors='red', alpha=0.8)
        
        ax.set_title(name, fontweight='bold', fontsize=11)
        ax.text(0.02, 0.98, description, transform=ax.transAxes, 
                verticalalignment='top', fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.suptitle('Catalog of Elliptical Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('elliptical_distributions_catalog.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Elliptical distributions catalog generated")

def generate_detailed_comparison_table():
    """Generate a visual comparison table of distribution characteristics."""
    
    print("Generating detailed comparison table...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Tail heaviness comparison
    u = np.linspace(0, 10, 200)
    
    # Generator functions
    g_normal = np.exp(-u/2)
    g_t1 = (1 + u)**(-1.5)  # Cauchy
    g_t3 = (1 + u/3)**(-2.5)
    g_t10 = (1 + u/10)**(-6)
    g_laplace = np.exp(-np.sqrt(2*u))
    
    ax1.semilogy(u, g_normal, 'b-', linewidth=3, label='Normal')
    ax1.semilogy(u, g_t1, 'red', linewidth=3, label='t (ν=1, Cauchy)')
    ax1.semilogy(u, g_t3, 'orange', linewidth=3, label='t (ν=3)')
    ax1.semilogy(u, g_t10, 'brown', linewidth=3, label='t (ν=10)')
    ax1.semilogy(u, g_laplace, 'green', linewidth=3, label='Laplace')
    
    ax1.set_xlabel('Quadratic Form (u)')
    ax1.set_ylabel('Generator Function g(u)')
    ax1.set_title('Tail Behavior: Generator Functions', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Kurtosis comparison
    distributions = ['Normal', 't(ν=1)', 't(ν=3)', 't(ν=5)', 't(ν=10)', 'Laplace']
    kurtosis_values = [3, np.inf, 9, 5, 3.6, 6]  # Approximate values
    colors = ['blue', 'darkred', 'red', 'orange', 'brown', 'green']
    
    # Replace infinity with a large value for plotting
    kurtosis_plot = [k if k != np.inf else 50 for k in kurtosis_values]
    
    bars = ax2.bar(distributions, kurtosis_plot, color=colors, alpha=0.7)
    ax2.axhline(y=3, color='black', linestyle='--', alpha=0.5, label='Normal Reference')
    ax2.set_ylabel('Kurtosis')
    ax2.set_title('Excess Kurtosis Comparison', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add value labels on bars
    for bar, val in zip(bars, kurtosis_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val}' if val != np.inf else '∞',
                ha='center', va='bottom', fontweight='bold')
    
    # 3. Moment existence
    distributions_moments = ['Normal', 't(ν=1)', 't(ν=3)', 't(ν=5)', 't(ν=10)', 'Laplace']
    moment_orders = ['Mean', 'Variance', '3rd Moment', '4th Moment', 'All Moments']
    
    # Existence matrix (1 = exists, 0 = doesn't exist, 0.5 = conditional)
    existence_matrix = np.array([
        [1, 1, 1, 1, 1],  # Normal
        [0, 0, 0, 0, 0],  # t(ν=1) - Cauchy
        [1, 1, 0, 0, 0],  # t(ν=3)
        [1, 1, 1, 1, 0],  # t(ν=5)
        [1, 1, 1, 1, 1],  # t(ν=10)
        [1, 1, 1, 1, 1],  # Laplace
    ])
    
    im = ax3.imshow(existence_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax3.set_xticks(range(len(moment_orders)))
    ax3.set_xticklabels(moment_orders, rotation=45)
    ax3.set_yticks(range(len(distributions_moments)))
    ax3.set_yticklabels(distributions_moments)
    ax3.set_title('Moment Existence (Green=Exists, Red=Diverges)', fontweight='bold')
    
    # Add text annotations
    for i in range(len(distributions_moments)):
        for j in range(len(moment_orders)):
            text = '✓' if existence_matrix[i, j] == 1 else '✗'
            ax3.text(j, i, text, ha="center", va="center", 
                    color="white" if existence_matrix[i, j] == 0 else "black",
                    fontsize=12, fontweight='bold')
    
    # 4. Application domains
    ax4.axis('off')
    application_text = """
    ELLIPTICAL DISTRIBUTIONS: APPLICATION DOMAINS
    
    📊 MULTIVARIATE NORMAL (GAUSSIAN)
    • Financial portfolio analysis
    • Quality control (when outliers are rare)
    • Sensor fusion under normal conditions
    • Classical multivariate statistics
    
    📈 STUDENT'S T-DISTRIBUTION
    • Robust process monitoring
    • Financial risk management (heavy tails)
    • Small sample inference
    • Outlier-robust classification
    
    🔍 MULTIVARIATE LAPLACE
    • Sparse signal processing
    • Regularization in machine learning
    • Compressed sensing
    • Robust regression with sparsity
    
    ⚡ CAUCHY DISTRIBUTION (t, ν=1)
    • Extreme value modeling
    • Signal processing with impulsive noise
    • Robust statistics research
    • Theoretical probability studies
    """
    
    ax4.text(0.05, 0.95, application_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('detailed_comparison_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Detailed comparison table generated")

def main():
    """Generate all Gaussian comparison assets."""
    
    print("=" * 60)
    print("GENERATING GAUSSIAN COMPARISON ASSETS")
    print("=" * 60)
    
    # Ensure we're in the assets directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        # Generate all comparison visualizations
        generate_gaussian_tail_comparison()
        generate_distribution_catalog()
        generate_detailed_comparison_table()
        
        print()
        print("=" * 60)
        print("ALL GAUSSIAN COMPARISON ASSETS GENERATED!")
        print("=" * 60)
        
        # List new files
        new_files = [
            'gaussian_tail_comparison.png',
            'elliptical_distributions_catalog.png', 
            'detailed_comparison_table.png'
        ]
        
        print("Generated files:")
        for file in new_files:
            if os.path.exists(file):
                print(f"  ✓ {file}")
        
    except Exception as e:
        print(f"Error generating assets: {e}")
        raise

if __name__ == "__main__":
    main()