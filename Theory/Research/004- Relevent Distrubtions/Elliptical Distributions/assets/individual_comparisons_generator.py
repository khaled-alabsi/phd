"""
Individual Distribution Comparisons Generator

This script generates clear individual comparison plots for each elliptical distribution
against Gaussian reference, plus examples of non-elliptical distributions.

Author: PhD Project
Date: September 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import gamma
from matplotlib.patches import Ellipse
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

def generate_individual_comparisons():
    """Generate individual comparison plots for each distribution vs Gaussian."""
    
    print("Generating individual distribution comparisons...")
    
    # Set parameters
    np.random.seed(42)
    mu = np.array([0, 0])
    Sigma = np.array([[1, 0.3], [0.3, 1]])
    n_samples = 800
    
    # Gaussian reference
    gaussian = MultivariateNormal(mu, Sigma)
    
    # Distributions to compare
    distributions = [
        (MultivariateT(mu, Sigma, nu=3), "Student's t (ν=3)", "Heavy Tails"),
        (MultivariateT(mu, Sigma, nu=10), "Student's t (ν=10)", "Moderate Tails"), 
        (MultivariateLaplace(mu, Sigma), "Multivariate Laplace", "Exponential Tails")
    ]
    
    # Create individual comparison plots
    for i, (dist, name, description) in enumerate(distributions):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
        
        # Generate samples
        X_dist = dist.sample(n_samples)
        X_gauss = gaussian.sample(n_samples)
        
        # 1. Scatter plot comparison
        ax1.scatter(X_gauss[:, 0], X_gauss[:, 1], alpha=0.6, s=15, 
                   color='blue', label='Gaussian Reference')
        ax1.scatter(X_dist[:, 0], X_dist[:, 1], alpha=0.6, s=15, 
                   color='red', label=name)
        
        # Plot contours
        x_range = np.linspace(-4, 4, 50)
        y_range = np.linspace(-4, 4, 50)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        pos = np.stack([X_grid.ravel(), Y_grid.ravel()], axis=1)
        
        Z_gauss = np.exp(gaussian.log_pdf(pos)).reshape(X_grid.shape)
        Z_dist = np.exp(dist.log_pdf(pos)).reshape(X_grid.shape)
        
        ax1.contour(X_grid, Y_grid, Z_gauss, levels=4, colors='blue', alpha=0.8)
        ax1.contour(X_grid, Y_grid, Z_dist, levels=4, colors='red', alpha=0.8)
        
        ax1.set_title(f'Scatter Plot: {name} vs Gaussian', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-4, 4)
        ax1.set_ylim(-4, 4)
        ax1.axis('equal')
        
        # 2. Distance distributions
        dist_distances = np.sqrt(gaussian.mahalanobis_distance(X_dist))
        gauss_distances = np.sqrt(gaussian.mahalanobis_distance(X_gauss))
        
        ax2.hist(gauss_distances, bins=30, alpha=0.7, density=True, 
                color='blue', label='Gaussian')
        ax2.hist(dist_distances, bins=30, alpha=0.7, density=True, 
                color='red', label=name)
        
        # Theoretical chi distribution
        x_theory = np.linspace(0, 5, 200)
        chi_pdf = stats.chi.pdf(x_theory, df=2)
        ax2.plot(x_theory, chi_pdf, 'k--', linewidth=2, 
                label='Theoretical χ(2)')
        
        ax2.set_xlabel('Mahalanobis Distance')
        ax2.set_ylabel('Density')
        ax2.set_title('Distance Distributions', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q plot
        gauss_quantiles = np.sort(gauss_distances)
        dist_quantiles = np.sort(dist_distances)
        n = min(len(gauss_quantiles), len(dist_quantiles))
        
        ax3.plot(gauss_quantiles[:n], dist_quantiles[:n], 'ro', alpha=0.6)
        ax3.plot([0, 4], [0, 4], 'k--', alpha=0.5, label='Perfect Agreement')
        
        ax3.set_xlabel('Gaussian Quantiles')
        ax3.set_ylabel(f'{name} Quantiles')
        ax3.set_title('Q-Q Plot: Distance Comparison', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Tail probability comparison
        thresholds = np.linspace(1, 4, 20)
        gauss_tail_probs = [np.mean(gauss_distances > t) for t in thresholds]
        dist_tail_probs = [np.mean(dist_distances > t) for t in thresholds]
        theoretical_probs = [1 - stats.chi2.cdf(t**2, df=2) for t in thresholds]
        
        ax4.semilogy(thresholds, theoretical_probs, 'k--', linewidth=2, 
                    label='Theoretical (Gaussian)')
        ax4.semilogy(thresholds, gauss_tail_probs, 'bo-', linewidth=2, 
                    label='Gaussian (Empirical)')
        ax4.semilogy(thresholds, dist_tail_probs, 'ro-', linewidth=2, 
                    label=f'{name} (Empirical)')
        
        ax4.set_xlabel('Distance Threshold')
        ax4.set_ylabel('P(Distance > Threshold)')
        ax4.set_title('Tail Probability Comparison', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'{name}: {description}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = f'individual_comparison_{name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("ν", "nu").replace("=", "")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    print("✓ Individual comparison plots generated")

def generate_non_elliptical_examples():
    """Generate examples of non-elliptical distributions."""
    
    print("Generating non-elliptical distribution examples...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    np.random.seed(123)
    n_samples = 1000
    
    # 1. Skewed distribution (Log-normal)
    ax = axes[0]
    X1 = np.random.lognormal(0, 0.5, n_samples)
    X2 = np.random.lognormal(0, 0.3, n_samples) + 0.2 * X1
    
    ax.scatter(X1, X2, alpha=0.6, s=15, color='red')
    ax.set_title('Log-Normal Distribution\n(Skewed - Non-Elliptical)', fontweight='bold')
    ax.text(0.05, 0.95, 'Asymmetric contours\nPositive skewness', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    ax.grid(True, alpha=0.3)
    
    # 2. Mixture of Gaussians
    ax = axes[1]
    n1, n2 = n_samples//2, n_samples//2
    X1_mix = np.vstack([
        np.random.multivariate_normal([-2, -1], [[0.5, 0], [0, 0.5]], n1),
        np.random.multivariate_normal([2, 1], [[0.3, 0.1], [0.1, 0.7]], n2)
    ])
    
    ax.scatter(X1_mix[:n1, 0], X1_mix[:n1, 1], alpha=0.6, s=15, color='blue', label='Component 1')
    ax.scatter(X1_mix[n1:, 0], X1_mix[n1:, 1], alpha=0.6, s=15, color='green', label='Component 2')
    ax.set_title('Mixture of Gaussians\n(Multi-modal - Non-Elliptical)', fontweight='bold')
    ax.text(0.05, 0.95, 'Multiple modes\nNon-unimodal', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Banana-shaped distribution
    ax = axes[2]
    t = np.random.uniform(0, 2*np.pi, n_samples)
    r = np.random.exponential(1, n_samples)
    X_banana = np.column_stack([
        r * np.cos(t) + 0.3 * (r * np.cos(t))**2,
        r * np.sin(t)
    ])
    
    ax.scatter(X_banana[:, 0], X_banana[:, 1], alpha=0.6, s=15, color='purple')
    ax.set_title('Banana Distribution\n(Curved - Non-Elliptical)', fontweight='bold')
    ax.text(0.05, 0.95, 'Curved contours\nNon-linear dependence', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    ax.grid(True, alpha=0.3)
    
    # 4. Uniform on square
    ax = axes[3]
    X_uniform = np.random.uniform(-2, 2, (n_samples, 2))
    
    ax.scatter(X_uniform[:, 0], X_uniform[:, 1], alpha=0.6, s=15, color='orange')
    ax.set_title('Uniform on Square\n(Bounded - Non-Elliptical)', fontweight='bold')
    ax.text(0.05, 0.95, 'Sharp boundaries\nConstant density', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    ax.grid(True, alpha=0.3)
    
    # 5. Ring distribution
    ax = axes[4]
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    r = np.random.normal(2, 0.3, n_samples)
    X_ring = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    ax.scatter(X_ring[:, 0], X_ring[:, 1], alpha=0.6, s=15, color='brown')
    ax.set_title('Ring Distribution\n(Annular - Non-Elliptical)', fontweight='bold')
    ax.text(0.05, 0.95, 'Hollow center\nNon-convex support', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 6. Gaussian reference for comparison
    ax = axes[5]
    X_gauss = np.random.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], n_samples)
    
    ax.scatter(X_gauss[:, 0], X_gauss[:, 1], alpha=0.6, s=15, color='blue')
    
    # Add elliptical contours
    eigenvals, eigenvecs = np.linalg.eig([[1, 0.3], [0.3, 1]])
    for scale in [1, 2, 3]:
        ellipse = Ellipse((0, 0), 2*scale*np.sqrt(eigenvals[0]), 2*scale*np.sqrt(eigenvals[1]),
                         angle=np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])),
                         fill=False, color='red', linewidth=2, alpha=0.7)
        ax.add_patch(ellipse)
    
    ax.set_title('Gaussian Distribution\n(Elliptical - Reference)', fontweight='bold')
    ax.text(0.05, 0.95, 'Elliptical contours\nUnimodal, symmetric', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Set consistent axis limits
    for ax in axes:
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
    
    plt.suptitle('Examples: Elliptical vs Non-Elliptical Distributions', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('non_elliptical_examples.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Non-elliptical examples generated")

def generate_elliptical_conditions_plot():
    """Generate a plot showing when distributions are/aren't elliptical."""
    
    print("Generating elliptical conditions visualization...")
    
    # Create a large standalone plot for elliptical contours
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Multiple Elliptical contours from different generator functions - STANDALONE LARGE PLOT
    x = np.linspace(-4, 4, 150)
    y = np.linspace(-4, 4, 150)
    X, Y = np.meshgrid(x, y)
    
    # Shared covariance matrix
    Sigma = np.array([[1, 0.5], [0.5, 1]])
    Sigma_inv = np.linalg.inv(Sigma)
    
    # Quadratic form for all points
    Q = Sigma_inv[0,0]*X**2 + 2*Sigma_inv[0,1]*X*Y + Sigma_inv[1,1]*Y**2
    
    # Different generator functions create different elliptical densities
    
    # Normal generator: g(u) = exp(-u/2)
    Z_normal = np.exp(-Q/2)
    contour1 = ax1.contour(X, Y, Z_normal, levels=8, colors='blue', linewidths=2.5)
    ax1.clabel(contour1, inline=True, fontsize=10, fmt='Normal')
    
    # t-distribution generator: g(u) = (1 + u/ν)^(-(ν+p)/2), ν=3
    nu = 3
    Z_t = (1 + Q/nu)**(-2.5)  # -(ν+p)/2 = -(3+2)/2 = -2.5
    contour2 = ax1.contour(X, Y, Z_t, levels=8, colors='red', linewidths=2.5, linestyles='dashed')
    ax1.clabel(contour2, inline=True, fontsize=10, fmt='t(ν=3)')
    
    # Laplace generator: g(u) = exp(-√(2u))
    Z_laplace = np.exp(-np.sqrt(2*Q))
    contour3 = ax1.contour(X, Y, Z_laplace, levels=8, colors='green', linewidths=2.5, linestyles='dotted')
    ax1.clabel(contour3, inline=True, fontsize=10, fmt='Laplace')
    
    ax1.set_title('ELLIPTICAL CONTOURS FROM DIFFERENT GENERATORS\nf(x) = g((x-μ)ᵀΣ⁻¹(x-μ))', 
                  fontweight='bold', fontsize=14, pad=20)
    ax1.text(0.02, 0.98, 'Blue: Normal g(u)=exp(-u/2)\nRed: t-dist g(u)=(1+u/3)^-2.5\nGreen: Laplace g(u)=exp(-√2u)', 
            transform=ax1.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
            fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_xlabel('X₁', fontsize=12)
    ax1.set_ylabel('X₂', fontsize=12)
    
    # 2. Non-elliptical: Skewed
    Z_skewed = np.exp(-0.5 * (Q)) * (1 + 0.5*X)  # Skewness term
    contour2 = ax2.contour(X, Y, Z_skewed, levels=8, colors='red', linewidths=2)
    ax2.clabel(contour2, inline=True, fontsize=10)
    ax2.set_title('NON-ELLIPTICAL: Skewed Distribution\nAsymmetric around center', 
                  fontweight='bold', fontsize=14, pad=20)
    ax2.text(0.02, 0.98, 'Distorted ellipses\nAsymmetric contours', 
            transform=ax2.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
            fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.set_xlabel('X₁', fontsize=12)
    ax2.set_ylabel('X₂', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('elliptical_conditions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate separate plot for mathematical conditions (text only)
    fig_text, ax_text = plt.subplots(1, 1, figsize=(12, 10))
    ax_text.axis('off')
    conditions_text = """
    MATHEMATICAL CONDITIONS FOR ELLIPTICAL DISTRIBUTIONS
    
    ✅ ELLIPTICAL IF:
    
    1. Density Form: f(x) = |Σ|^(-1/2) g((x-μ)ᵀΣ⁻¹(x-μ))
       • g(·) is a non-negative, non-increasing generator function
       • Density depends ONLY on quadratic form
    
    2. Contour Property: {x : f(x) = c} forms ellipses for all c > 0
    
    3. Radial Symmetry: f(x) = f(y) if ||x-μ||²_Σ = ||y-μ||²_Σ
    
    4. Stochastic Representation: X =ᵈ μ + √R A Z
       • R ≥ 0 (radial component)
       • Z ~ Uniform on unit sphere
       • A: square root of scatter matrix
    
    ❌ NOT ELLIPTICAL IF:
    
    • Skewness: E[(X-μ)³] ≠ 0
    • Multi-modality: Multiple local maxima
    • Non-convex support: Holes, disconnected regions
    • Asymmetric contours: Non-elliptical level sets
    """
    
    ax_text.text(0.05, 0.95, conditions_text, transform=ax_text.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('elliptical_mathematical_conditions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate generator function examples plot
    fig_gen, ax_gen = plt.subplots(1, 1, figsize=(10, 8))
    u = np.linspace(0, 8, 200)
    
    # Valid generators
    g_normal = np.exp(-u/2)
    g_t = (1 + u/3)**(-2.5)
    g_laplace = np.exp(-np.sqrt(2*u))
    
    ax_gen.semilogy(u, g_normal, 'b-', linewidth=3, label='Normal: exp(-u/2)')
    ax_gen.semilogy(u, g_t, 'r-', linewidth=3, label='t-dist: (1+u/ν)^(-(ν+p)/2)')
    ax_gen.semilogy(u, g_laplace, 'g-', linewidth=3, label='Laplace: exp(-√(2u))')
    
    ax_gen.set_xlabel('Quadratic Form (u)', fontsize=12)
    ax_gen.set_ylabel('Generator Function g(u)', fontsize=12)
    ax_gen.set_title('Valid Generator Functions\n(Non-increasing, Non-negative)', fontweight='bold', fontsize=14)
    ax_gen.legend(fontsize=11)
    ax_gen.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('generator_functions_examples.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Elliptical conditions plots generated")

def main():
    """Generate all individual comparison and non-elliptical assets."""
    
    print("=" * 60)
    print("GENERATING INDIVIDUAL COMPARISONS AND NON-ELLIPTICAL EXAMPLES")
    print("=" * 60)
    
    # Ensure we're in the assets directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        # Generate all visualizations
        generate_individual_comparisons()
        generate_non_elliptical_examples()
        generate_elliptical_conditions_plot()
        
        print()
        print("=" * 60)
        print("ALL INDIVIDUAL COMPARISON ASSETS GENERATED!")
        print("=" * 60)
        
        # List new files
        expected_files = [
            'individual_comparison_students_t_nu3.png',
            'individual_comparison_students_t_nu10.png',
            'individual_comparison_multivariate_laplace.png',
            'non_elliptical_examples.png',
            'elliptical_conditions.png'
        ]
        
        print("Generated files:")
        for file in expected_files:
            if os.path.exists(file):
                print(f"  ✓ {file}")
        
    except Exception as e:
        print(f"Error generating assets: {e}")
        raise

if __name__ == "__main__":
    main()