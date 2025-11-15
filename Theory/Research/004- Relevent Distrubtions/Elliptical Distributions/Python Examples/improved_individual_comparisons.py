"""
Improved Individual Distribution Comparisons with Better Outlier Handling
================================================================

This script generates individual comparison plots between elliptical distributions 
and Gaussian reference, with proper outlier handling and scaling for clear visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.covariance import EmpiricalCovariance
import os

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def robust_scatter_limits(data, percentile=95):
    """Calculate robust limits for scatter plots to avoid outlier compression."""
    p_low = (100 - percentile) / 2
    p_high = 100 - p_low
    
    x_low, x_high = np.percentile(data[:, 0], [p_low, p_high])
    y_low, y_high = np.percentile(data[:, 1], [p_low, p_high])
    
    # Add some padding
    x_range = x_high - x_low
    y_range = y_high - y_low
    padding = 0.1
    
    return {
        'xlim': (x_low - padding * x_range, x_high + padding * x_range),
        'ylim': (y_low - padding * y_range, y_high + padding * y_range)
    }

def multivariate_student_t(n_samples, nu, mu, Sigma):
    """Generate multivariate Student's t distribution samples."""
    p = len(mu)
    # Generate chi-squared random variable
    chi2_samples = np.random.chisquare(nu, n_samples)
    # Generate multivariate normal
    normal_samples = np.random.multivariate_normal(np.zeros(p), Sigma, n_samples)
    # Scale by chi-squared factor
    t_samples = mu + normal_samples * np.sqrt(nu / chi2_samples).reshape(-1, 1)
    return t_samples

def multivariate_laplace(n_samples, mu, Sigma):
    """Generate multivariate Laplace distribution samples."""
    p = len(mu)
    # Generate exponential random variable for scaling
    gamma_samples = np.random.exponential(1, n_samples)
    # Generate multivariate normal
    normal_samples = np.random.multivariate_normal(np.zeros(p), Sigma, n_samples)
    # Scale by gamma factor
    laplace_samples = mu + normal_samples * np.sqrt(gamma_samples).reshape(-1, 1)
    return laplace_samples

def calculate_mahalanobis_distances(data, mu, Sigma):
    """Calculate Mahalanobis distances."""
    cov_inv = np.linalg.inv(Sigma)
    diff = data - mu
    distances = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
    return distances

def create_individual_comparison(dist_data, gaussian_data, dist_name, mu, Sigma, output_dir):
    """Create individual comparison plot with improved scaling."""
    
    # Calculate robust limits for scatter plots
    combined_data = np.vstack([dist_data, gaussian_data])
    limits = robust_scatter_limits(combined_data, percentile=90)  # Use 90% to show some outliers
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{dist_name} vs Gaussian Distribution Comparison', fontsize=16, fontweight='bold')
    
    # 1. Scatter Plot Comparison
    ax1 = axes[0, 0]
    
    # Plot with limited outliers for better visualization
    gaussian_mask = (
        (gaussian_data[:, 0] >= limits['xlim'][0]) & 
        (gaussian_data[:, 0] <= limits['xlim'][1]) &
        (gaussian_data[:, 1] >= limits['ylim'][0]) & 
        (gaussian_data[:, 1] <= limits['ylim'][1])
    )
    dist_mask = (
        (dist_data[:, 0] >= limits['xlim'][0]) & 
        (dist_data[:, 0] <= limits['xlim'][1]) &
        (dist_data[:, 1] >= limits['ylim'][0]) & 
        (dist_data[:, 1] <= limits['ylim'][1])
    )
    
    ax1.scatter(gaussian_data[gaussian_mask, 0], gaussian_data[gaussian_mask, 1], 
               alpha=0.6, s=20, color='blue', label='Gaussian Reference')
    ax1.scatter(dist_data[dist_mask, 0], dist_data[dist_mask, 1], 
               alpha=0.6, s=20, color='red', label=dist_name)
    
    # Add elliptical contours
    theta = np.linspace(0, 2*np.pi, 100)
    eigenvals, eigenvecs = np.linalg.eigh(Sigma)
    
    for k in [1, 2, 3]:  # 1, 2, 3 sigma contours
        # Create ellipse points
        ellipse_points = np.array([np.cos(theta), np.sin(theta)]) * np.sqrt(k**2 * eigenvals[:, np.newaxis])
        ellipse_rotated = eigenvecs @ ellipse_points
        ellipse_final = ellipse_rotated.T + mu
        ax1.plot(ellipse_final[:, 0], ellipse_final[:, 1], 'k--', alpha=0.5, linewidth=1)
    
    ax1.set_xlim(limits['xlim'])
    ax1.set_ylim(limits['ylim'])
    ax1.set_xlabel('X₁')
    ax1.set_ylabel('X₂')
    ax1.set_title('Scatter Plot Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distance Distributions
    ax2 = axes[0, 1]
    
    # Calculate Mahalanobis distances
    gaussian_distances = calculate_mahalanobis_distances(gaussian_data, mu, Sigma)
    dist_distances = calculate_mahalanobis_distances(dist_data, mu, Sigma)
    
    # Plot histograms with better binning
    max_distance = min(np.percentile(np.concatenate([gaussian_distances, dist_distances]), 95), 8)
    bins = np.linspace(0, max_distance, 40)
    
    ax2.hist(gaussian_distances[gaussian_distances <= max_distance], bins=bins, alpha=0.7, 
             density=True, color='blue', label='Gaussian')
    ax2.hist(dist_distances[dist_distances <= max_distance], bins=bins, alpha=0.7, 
             density=True, color='red', label=dist_name)
    
    # Add theoretical chi-squared distribution for Gaussian
    x_theory = np.linspace(0, max_distance, 200)
    chi2_theory = stats.chi2.pdf(x_theory**2, df=2) * 2 * x_theory  # For 2D
    ax2.plot(x_theory, chi2_theory, 'k--', linewidth=2, label='Theoretical χ²')
    
    ax2.set_xlabel('Mahalanobis Distance')
    ax2.set_ylabel('Density')
    ax2.set_title('Distance Distributions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_distance)
    
    # 3. Q-Q Plot
    ax3 = axes[1, 0]
    
    # Use robust quantile range for Q-Q plot
    gaussian_sorted = np.sort(gaussian_distances)
    dist_sorted = np.sort(dist_distances)
    
    # Trim extreme outliers for better Q-Q plot visualization
    n_points = min(len(gaussian_sorted), len(dist_sorted))
    trim_pct = 0.02  # Trim top and bottom 2%
    start_idx = int(n_points * trim_pct)
    end_idx = int(n_points * (1 - trim_pct))
    
    # Use same quantiles for fair comparison
    quantiles = np.linspace(0, 1, end_idx - start_idx)
    gaussian_quantiles = np.quantile(gaussian_distances, quantiles)
    dist_quantiles = np.quantile(dist_distances, quantiles)
    
    ax3.scatter(gaussian_quantiles, dist_quantiles, alpha=0.6, s=15)
    
    # Add reference line
    min_val = min(gaussian_quantiles.min(), dist_quantiles.min())
    max_val = min(gaussian_quantiles.max(), dist_quantiles.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
    
    ax3.set_xlabel('Gaussian Quantiles')
    ax3.set_ylabel(f'{dist_name} Quantiles')
    ax3.set_title('Q-Q Plot (Distance Distributions)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Tail Probability Comparison
    ax4 = axes[1, 1]
    
    # Calculate tail probabilities
    distance_thresholds = np.linspace(1, 6, 50)
    gaussian_tail_probs = [np.mean(gaussian_distances > t) for t in distance_thresholds]
    dist_tail_probs = [np.mean(dist_distances > t) for t in distance_thresholds]
    
    ax4.semilogy(distance_thresholds, gaussian_tail_probs, 'b-', linewidth=2, label='Gaussian')
    ax4.semilogy(distance_thresholds, dist_tail_probs, 'r-', linewidth=2, label=dist_name)
    
    # Add theoretical tail probabilities for Gaussian (chi-squared with 2 DOF)
    theoretical_tail = 1 - stats.chi2.cdf(distance_thresholds**2, df=2)
    ax4.semilogy(distance_thresholds, theoretical_tail, 'k--', linewidth=2, label='Theoretical χ²')
    
    ax4.set_xlabel('Distance Threshold')
    ax4.set_ylabel('Tail Probability')
    ax4.set_title('Tail Probability Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save with improved filename
    filename = f"improved_individual_comparison_{dist_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved improved comparison: {filename}")

def main():
    """Generate improved individual comparison plots."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define parameters
    n_samples = 5000  # Increased sample size for better visualization
    mu = np.array([0, 0])
    Sigma = np.array([[2, 0.8], [0.8, 1]])
    
    # Create output directory
    output_dir = "../assets"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate Gaussian reference data
    print("Generating Gaussian reference data...")
    gaussian_data = np.random.multivariate_normal(mu, Sigma, n_samples)
    
    # Distribution configurations
    distributions = [
        {
            'name': "Student's t (ν=3)",
            'generator': lambda: multivariate_student_t(n_samples, nu=3, mu=mu, Sigma=Sigma)
        },
        {
            'name': "Student's t (ν=10)",
            'generator': lambda: multivariate_student_t(n_samples, nu=10, mu=mu, Sigma=Sigma)
        },
        {
            'name': "Multivariate Laplace",
            'generator': lambda: multivariate_laplace(n_samples, mu, Sigma)
        }
    ]
    
    # Generate improved individual comparisons
    for dist_config in distributions:
        print(f"Generating improved comparison for {dist_config['name']}...")
        dist_data = dist_config['generator']()
        create_individual_comparison(
            dist_data, gaussian_data, dist_config['name'], mu, Sigma, output_dir
        )
    
    print("\n✅ All improved individual comparison plots generated successfully!")
    print(f"📁 Plots saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()