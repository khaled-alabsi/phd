"""
Asset Generator for Elliptical Distributions Documentation

This script generates all plots and figures needed for the markdown documentation.
All outputs are saved to the assets folder with appropriate names.

Author: PhD Project
Date: September 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import seaborn as sns
from scipy import stats
from scipy.special import gamma
from sklearn.metrics import roc_curve, auc
import os
import sys

# Import our fixed elliptical distributions module
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Python Examples'))

try:
    from elliptical_distributions_fixed import (
        MultivariateNormal, MultivariateT, MultivariateLaplace, RobustEstimators
    )
except ImportError:
    print("Error: Could not import elliptical distributions module.")
    print("Make sure elliptical_distributions_fixed.py is in the Python Examples folder.")
    sys.exit(1)

# Set style for consistent plots
plt.style.use('default')
sns.set_palette("tab10")

def ensure_assets_dir():
    """Ensure assets directory exists and set it as output location."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = script_dir
    
    # Create assets directory if it doesn't exist
    os.makedirs(assets_dir, exist_ok=True)
    
    # Change to assets directory
    os.chdir(assets_dir)
    
    return assets_dir

def generate_conceptual_plots():
    """Generate plots for conceptual explanation section."""
    print("Generating conceptual explanation plots...")
    
    # 1. Elliptical contours comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    mu = np.array([0, 0])
    distributions = [
        (MultivariateNormal(mu, [[1, 0], [0, 1]]), "Multivariate Normal"),
        (MultivariateNormal(mu, [[2, 0.8], [0.8, 1]]), "Correlated Normal"),
        (MultivariateT(mu, [[1, 0], [0, 1]], nu=3), "Student's t (ν=3)"),
        (MultivariateLaplace(mu, [[1, 0], [0, 1]]), "Multivariate Laplace")
    ]
    
    for i, (dist, title) in enumerate(distributions):
        ax = axes[i]
        
        # Generate sample data
        X = dist.sample(500)
        ax.scatter(X[:, 0], X[:, 1], alpha=0.6, s=20, color='steelblue')
        
        # Plot contours
        dist.plot_contours(ax, levels=[0.5, 0.9, 0.95])
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('conceptual_elliptical_contours.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Tail behavior comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Generate 1D samples for tail comparison
    n_samples = 5000
    normal_1d = np.random.normal(0, 1, n_samples)
    t_1d = np.random.standard_t(3, n_samples)
    laplace_1d = np.random.laplace(0, 1/np.sqrt(2), n_samples)
    
    # Histogram comparison
    bins = np.linspace(-6, 6, 50)
    ax1.hist(normal_1d, bins=bins, alpha=0.6, density=True, label='Normal', color='blue')
    ax1.hist(t_1d, bins=bins, alpha=0.6, density=True, label='Student-t (ν=3)', color='red')
    ax1.hist(laplace_1d, bins=bins, alpha=0.6, density=True, label='Laplace', color='green')
    
    # Add theoretical curves
    x = np.linspace(-6, 6, 200)
    ax1.plot(x, stats.norm.pdf(x), 'b-', linewidth=2, label='Normal PDF')
    ax1.plot(x, stats.t.pdf(x, 3), 'r-', linewidth=2, label='t PDF (ν=3)')
    ax1.plot(x, stats.laplace.pdf(x, scale=1/np.sqrt(2)), 'g-', linewidth=2, label='Laplace PDF')
    
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Tail Behavior Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot for tail comparison
    theoretical_quantiles = np.linspace(0.01, 0.99, 100)
    normal_quantiles = stats.norm.ppf(theoretical_quantiles)
    t_quantiles = stats.t.ppf(theoretical_quantiles, 3)
    laplace_quantiles = stats.laplace.ppf(theoretical_quantiles, scale=1/np.sqrt(2))
    
    ax2.plot(normal_quantiles, normal_quantiles, 'k--', alpha=0.5, label='Reference')
    ax2.plot(normal_quantiles, t_quantiles, 'r-', linewidth=2, label='t vs Normal')
    ax2.plot(normal_quantiles, laplace_quantiles, 'g-', linewidth=2, label='Laplace vs Normal')
    
    ax2.set_xlabel('Normal Quantiles')
    ax2.set_ylabel('Other Distribution Quantiles')
    ax2.set_title('Quantile-Quantile Comparison', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('conceptual_tail_behavior.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Conceptual plots generated")

def generate_mathematical_plots():
    """Generate plots for mathematical formulation section."""
    print("Generating mathematical formulation plots...")
    
    # 1. Generator function examples
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    u = np.linspace(0, 10, 200)
    
    # Generator functions (normalized for visualization)
    g_normal = np.exp(-u/2)
    g_t_3 = (1 + u/3)**(-5/2)  # (nu+p)/2 = (3+2)/2 = 5/2
    g_t_10 = (1 + u/10)**(-6)  # (nu+p)/2 = (10+2)/2 = 6
    g_laplace = np.exp(-np.sqrt(2*u)) * u**(1/2)  # Approximation
    
    ax1.plot(u, g_normal, 'b-', linewidth=2, label='Normal: exp(-u/2)')
    ax1.plot(u, g_t_3, 'r-', linewidth=2, label='t (ν=3): (1+u/3)^(-5/2)')
    ax1.plot(u, g_t_10, 'orange', linewidth=2, label='t (ν=10): (1+u/10)^(-6)')
    ax1.plot(u, g_laplace, 'g-', linewidth=2, label='Laplace (approx)')
    
    ax1.set_xlabel('u (Quadratic Form)')
    ax1.set_ylabel('g(u) (Generator Function)')
    ax1.set_title('Generator Functions Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. 3D density surface for normal distribution
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    
    # Create a normal distribution
    mu = np.array([0, 0])
    Sigma = np.array([[1, 0.5], [0.5, 1]])
    normal_dist = MultivariateNormal(mu, Sigma)
    
    pos = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = np.exp(normal_dist.log_pdf(pos)).reshape(X.shape)
    
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('X₁')
    ax2.set_ylabel('X₂')
    ax2.set_zlabel('Density')
    ax2.set_title('Elliptical Density Surface', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('mathematical_generator_functions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Scatter matrix eigenstructure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Different scatter matrices
    scatter_matrices = [
        np.array([[1, 0], [0, 1]]),  # Identity
        np.array([[2, 1], [1, 1]]),  # Correlated
        np.array([[3, 0], [0, 0.5]])  # Different variances
    ]
    
    titles = ['Identity Matrix', 'Correlated Variables', 'Different Variances']
    
    for i, (Sigma, title) in enumerate(zip(scatter_matrices, titles)):
        ax = axes[i]
        
        # Generate samples
        dist = MultivariateNormal([0, 0], Sigma)
        X = dist.sample(300)
        
        ax.scatter(X[:, 0], X[:, 1], alpha=0.6, s=20)
        
        # Plot eigenvectors
        eigenvals, eigenvecs = np.linalg.eig(Sigma)
        for j in range(2):
            ax.arrow(0, 0, eigenvecs[0, j] * np.sqrt(eigenvals[j]), 
                    eigenvecs[1, j] * np.sqrt(eigenvals[j]),
                    head_width=0.1, head_length=0.1, fc='red', ec='red')
        
        # Plot contours
        dist.plot_contours(ax, levels=[0.5, 0.9])
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('mathematical_scatter_eigenstructure.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Mathematical plots generated")

def generate_parameter_estimation_plots():
    """Generate plots for parameter estimation section."""
    print("Generating parameter estimation plots...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. EM Algorithm convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Generate t-distribution data
    true_mu = np.array([1, -0.5])
    true_Sigma = np.array([[2, 0.8], [0.8, 1]])
    true_nu = 4
    
    t_dist = MultivariateT(true_mu, true_Sigma, true_nu)
    X = t_dist.sample(200)
    
    # Simulate EM convergence (simplified)
    iterations = range(1, 21)
    log_likelihood = []
    mu_error = []
    
    current_mu = np.mean(X, axis=0) + np.random.randn(2) * 0.5  # Start with noise
    
    for i in iterations:
        # Simulate convergence
        current_mu = 0.9 * current_mu + 0.1 * true_mu + np.random.randn(2) * 0.05 * np.exp(-i/5)
        mu_error.append(np.linalg.norm(current_mu - true_mu))
        
        # Simulate log-likelihood improvement
        log_likelihood.append(-200 + 190 * (1 - np.exp(-i/3)) + np.random.randn() * 2)
    
    ax1.plot(iterations, log_likelihood, 'b-o', linewidth=2, markersize=4)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Log-Likelihood')
    ax1.set_title('EM Algorithm Convergence: Log-Likelihood', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(iterations, mu_error, 'r-o', linewidth=2, markersize=4)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Parameter Error ||μ̂ - μ||')
    ax2.set_title('EM Algorithm Convergence: Parameter Error', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parameter_estimation_em_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Robust estimation comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Generate clean data
    mu_true = np.array([2, -1])
    Sigma_true = np.array([[4, 1], [1, 2]])
    normal = MultivariateNormal(mu_true, Sigma_true)
    X_clean = normal.sample(200)
    
    contamination_levels = [0, 0.1, 0.2]
    estimator_types = ['Sample Covariance', 'Tyler Estimator']
    
    for row, est_type in enumerate(estimator_types):
        for col, cont_level in enumerate(contamination_levels):
            ax = axes[row, col]
            
            # Add contamination
            if cont_level > 0:
                n_outliers = int(cont_level * len(X_clean))
                outliers = mu_true + 6 * np.random.randn(n_outliers, 2)
                X_cont = np.vstack([X_clean, outliers])
            else:
                X_cont = X_clean
                outliers = np.array([]).reshape(0, 2)
            
            # Estimate parameters
            if est_type == 'Sample Covariance':
                mu_est = np.mean(X_cont, axis=0)
                Sigma_est = np.cov(X_cont.T)
            else:  # Tyler Estimator
                mu_est = np.mean(X_cont, axis=0)
                X_centered = X_cont - mu_est
                Sigma_est = RobustEstimators.tyler_estimator(X_centered)
            
            # Plot data
            ax.scatter(X_clean[:, 0], X_clean[:, 1], alpha=0.6, c='blue', s=15, label='Clean data')
            if len(outliers) > 0:
                ax.scatter(outliers[:, 0], outliers[:, 1], alpha=0.8, c='red', s=25, label='Outliers')
            
            # Plot true contours
            normal.plot_contours(ax, colors=['black'], linestyles='--')
            
            # Plot estimated contours
            est_dist = MultivariateNormal(mu_est, Sigma_est)
            est_dist.plot_contours(ax, colors=['green'])
            
            ax.set_title(f'{est_type}\nContamination: {int(cont_level*100)}%', fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-8, 12)
            ax.set_ylim(-8, 6)
    
    plt.tight_layout()
    plt.savefig('parameter_estimation_robust_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Parameter estimation plots generated")

def generate_generator_function_plots():
    """Generate plots for generator function theory section."""
    print("Generating generator function theory plots...")
    
    # 1. Generator function families
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    u = np.linspace(0, 8, 200)
    
    # Normal generator function
    g_normal = np.exp(-u/2)
    ax1.plot(u, g_normal, 'b-', linewidth=3)
    ax1.set_title('Normal: g(u) = exp(-u/2)', fontweight='bold')
    ax1.set_xlabel('u')
    ax1.set_ylabel('g(u)')
    ax1.grid(True, alpha=0.3)
    
    # Student-t generator functions
    g_t2 = (1 + u/2)**(-2)
    g_t4 = (1 + u/4)**(-3)
    g_t10 = (1 + u/10)**(-6)
    
    ax2.plot(u, g_t2, 'r-', linewidth=3, label='ν=2')
    ax2.plot(u, g_t4, 'orange', linewidth=3, label='ν=4')
    ax2.plot(u, g_t10, 'brown', linewidth=3, label='ν=10')
    ax2.set_title('Student-t: g(u) = (1+u/ν)^(-(ν+p)/2)', fontweight='bold')
    ax2.set_xlabel('u')
    ax2.set_ylabel('g(u)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Laplace generator (approximation)
    g_laplace = np.exp(-np.sqrt(2*u))
    ax3.plot(u, g_laplace, 'g-', linewidth=3)
    ax3.set_title('Laplace: g(u) ∝ exp(-√(2u))', fontweight='bold')
    ax3.set_xlabel('u')
    ax3.set_ylabel('g(u)')
    ax3.grid(True, alpha=0.3)
    
    # Comparison of all
    ax4.plot(u, g_normal, 'b-', linewidth=3, label='Normal')
    ax4.plot(u, g_t4, 'r-', linewidth=3, label='t (ν=4)')
    ax4.plot(u, g_laplace, 'g-', linewidth=3, label='Laplace')
    ax4.set_title('Generator Function Comparison', fontweight='bold')
    ax4.set_xlabel('u')
    ax4.set_ylabel('g(u)')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('generator_function_families.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Tail behavior demonstration
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sample from different distributions
    n_samples = 2000
    np.random.seed(123)
    
    normal_samples = MultivariateNormal([0, 0], [[1, 0], [0, 1]]).sample(n_samples)
    t_samples = MultivariateT([0, 0], [[1, 0], [0, 1]], nu=3).sample(n_samples)
    laplace_samples = MultivariateLaplace([0, 0], [[1, 0], [0, 1]]).sample(n_samples)
    
    # Compute Mahalanobis distances
    normal_dist = MultivariateNormal([0, 0], [[1, 0], [0, 1]])
    normal_distances = np.sqrt(normal_dist.mahalanobis_distance(normal_samples))
    t_distances = np.sqrt(normal_dist.mahalanobis_distance(t_samples))
    laplace_distances = np.sqrt(normal_dist.mahalanobis_distance(laplace_samples))
    
    # Box plots
    ax1.boxplot([normal_distances, t_distances, laplace_distances], 
                labels=['Normal', 't (ν=3)', 'Laplace'])
    ax1.set_ylabel('Mahalanobis Distance')
    ax1.set_title('Distance Distribution Comparison', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Tail probability comparison
    thresholds = np.linspace(1, 4, 20)
    normal_tail_probs = [np.mean(normal_distances > t) for t in thresholds]
    t_tail_probs = [np.mean(t_distances > t) for t in thresholds]
    laplace_tail_probs = [np.mean(laplace_distances > t) for t in thresholds]
    
    ax2.semilogy(thresholds, normal_tail_probs, 'b-o', linewidth=2, label='Normal')
    ax2.semilogy(thresholds, t_tail_probs, 'r-s', linewidth=2, label='t (ν=3)')
    ax2.semilogy(thresholds, laplace_tail_probs, 'g-^', linewidth=2, label='Laplace')
    
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('P(Distance > Threshold)')
    ax2.set_title('Tail Probability Comparison', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('generator_function_tail_behavior.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Generator function plots generated")

def generate_process_monitoring_plots():
    """Generate plots for process monitoring applications section."""
    print("Generating process monitoring application plots...")
    
    # Set random seed
    np.random.seed(42)
    
    # 1. Anomaly detection comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    # Generate process data
    mu_normal = np.array([0, 0])
    Sigma_normal = np.array([[1, 0.3], [0.3, 1]])
    
    # Normal operating data
    X_normal = MultivariateNormal(mu_normal, Sigma_normal).sample(300)
    
    # Different types of faults
    fault_types = [
        ("Mean Shift", np.array([2, 1]), Sigma_normal),
        ("Variance Increase", mu_normal, 3 * Sigma_normal),
        ("Heavy Tails", mu_normal, Sigma_normal),  # Will use t-distribution
        ("Correlation Change", mu_normal, np.array([[1, -0.8], [-0.8, 1]]))
    ]
    
    for i, (fault_name, mu_fault, Sigma_fault) in enumerate(fault_types):
        ax = axes[i]
        
        # Generate fault data
        if fault_name == "Heavy Tails":
            X_fault = MultivariateT(mu_fault, Sigma_fault, nu=2).sample(100)
        else:
            X_fault = MultivariateNormal(mu_fault, Sigma_fault).sample(100)
        
        # Plot normal operating data
        ax.scatter(X_normal[:, 0], X_normal[:, 1], alpha=0.6, c='blue', s=15, label='Normal')
        
        # Plot fault data
        ax.scatter(X_fault[:, 0], X_fault[:, 1], alpha=0.8, c='red', s=20, label='Fault')
        
        # Plot normal operating contours
        normal_dist = MultivariateNormal(mu_normal, Sigma_normal)
        normal_dist.plot_contours(ax, levels=[0.9, 0.99], colors=['blue'])
        
        # Control limit (99% confidence)
        chi2_99 = stats.chi2.ppf(0.99, df=2)
        circle = patches.Circle((mu_normal[0], mu_normal[1]), float(np.sqrt(chi2_99)), fill=False, color='red', linestyle='--', linewidth=2)
        ax.add_patch(circle)
        
        ax.set_title(f'{fault_name} Fault', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('process_monitoring_fault_types.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Robust vs Classical detection performance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Generate contaminated training data
    X_clean = MultivariateNormal(mu_normal, Sigma_normal).sample(200)
    outliers = mu_normal + 4 * np.random.randn(20, 2)
    X_training = np.vstack([X_clean, outliers])
    
    # Classical estimation
    mu_classical = np.mean(X_training, axis=0)
    Sigma_classical = np.cov(X_training.T)
    
    # Robust estimation (MCD)
    mu_robust, Sigma_robust = RobustEstimators.mcd_estimator(X_training)
    
    # Plot classical approach
    ax1.scatter(X_clean[:, 0], X_clean[:, 1], alpha=0.6, c='blue', s=15, label='Clean data')
    ax1.scatter(outliers[:, 0], outliers[:, 1], alpha=0.8, c='red', s=25, label='Outliers')
    
    classical_dist = MultivariateNormal(mu_classical, Sigma_classical)
    classical_dist.plot_contours(ax1, levels=[0.9, 0.99], colors=['orange'])
    
    # True contours
    true_dist = MultivariateNormal(mu_normal, Sigma_normal)
    true_dist.plot_contours(ax1, levels=[0.9, 0.99], colors=['black'], linestyles='--')
    
    ax1.set_title('Classical Estimation\n(Biased by Outliers)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-6, 6)
    
    # Plot robust approach
    ax2.scatter(X_clean[:, 0], X_clean[:, 1], alpha=0.6, c='blue', s=15, label='Clean data')
    ax2.scatter(outliers[:, 0], outliers[:, 1], alpha=0.8, c='red', s=25, label='Outliers')
    
    robust_dist = MultivariateNormal(mu_robust, Sigma_robust)
    robust_dist.plot_contours(ax2, levels=[0.9, 0.99], colors=['green'])
    
    # True contours
    true_dist.plot_contours(ax2, levels=[0.9, 0.99], colors=['black'], linestyles='--')
    
    ax2.set_title('Robust Estimation\n(Resistant to Outliers)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-6, 6)
    ax2.set_ylim(-6, 6)
    
    plt.tight_layout()
    plt.savefig('process_monitoring_robust_vs_classical.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROC curves for different distributions
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generate test data
    X_normal_test = MultivariateNormal(mu_normal, Sigma_normal).sample(500)
    X_fault_test = MultivariateNormal([1.5, 0.8], Sigma_normal).sample(200)  # Mean shift fault
    
    X_test = np.vstack([X_normal_test, X_fault_test])
    y_true = np.hstack([np.zeros(500), np.ones(200)])
    
    # Different detectors
    detectors = {
        'Classical Normal': MultivariateNormal(mu_normal, Sigma_normal),
        'Robust Normal': MultivariateNormal(mu_robust, Sigma_robust),
        'Student-t (ν=4)': MultivariateT(mu_normal, Sigma_normal, nu=4)
    }
    
    for name, detector in detectors.items():
        # Compute anomaly scores
        distances = detector.mahalanobis_distance(X_test)
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true, distances)
        auc_score = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc_score:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves: Anomaly Detection Performance', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('process_monitoring_roc_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Process monitoring plots generated")

def generate_all_assets():
    """Generate all assets for the documentation."""
    print("=" * 60)
    print("GENERATING ALL ASSETS FOR ELLIPTICAL DISTRIBUTIONS DOCUMENTATION")
    print("=" * 60)
    
    # Ensure we're in the assets directory
    assets_dir = ensure_assets_dir()
    print(f"Output directory: {assets_dir}")
    print()
    
    try:
        # Generate all plot categories
        generate_conceptual_plots()
        generate_mathematical_plots()
        generate_parameter_estimation_plots()
        generate_generator_function_plots()
        generate_process_monitoring_plots()
        
        print()
        print("=" * 60)
        print("ALL ASSETS GENERATED SUCCESSFULLY!")
        print("=" * 60)
        
        # List generated files
        generated_files = [f for f in os.listdir('.') if f.endswith('.png')]
        print(f"Generated {len(generated_files)} plot files:")
        for file in sorted(generated_files):
            print(f"  ✓ {file}")
        
    except Exception as e:
        print(f"Error generating assets: {e}")
        raise

if __name__ == "__main__":
    generate_all_assets()