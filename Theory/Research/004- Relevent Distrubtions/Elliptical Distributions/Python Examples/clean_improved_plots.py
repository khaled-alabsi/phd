"""
Clean Individual Distribution Comparisons with Better Visual Design
================================================================

This script generates clean individual comparison plots with:
1. No text overlays on plots (explanations go in markdown)
2. Fewer data points for cleaner appearance
3. Larger scatter plots to prevent point cutoff
4. Better color schemes and styling
5. 3D visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import os

# Set style for better plots
plt.style.use('default')
sns.set_context("notebook", font_scale=1.1)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

def multivariate_student_t(n_samples, nu, mu, Sigma):
    """Generate multivariate Student's t distribution samples."""
    p = len(mu)
    chi2_samples = np.random.chisquare(nu, n_samples)
    normal_samples = np.random.multivariate_normal(np.zeros(p), Sigma, n_samples)
    t_samples = mu + normal_samples * np.sqrt(nu / chi2_samples).reshape(-1, 1)
    return t_samples

def multivariate_laplace(n_samples, mu, Sigma):
    """Generate multivariate Laplace distribution samples."""
    p = len(mu)
    gamma_samples = np.random.exponential(1, n_samples)
    normal_samples = np.random.multivariate_normal(np.zeros(p), Sigma, n_samples)
    laplace_samples = mu + normal_samples * np.sqrt(gamma_samples).reshape(-1, 1)
    return laplace_samples

def calculate_mahalanobis_distances(data, mu, Sigma):
    """Calculate Mahalanobis distances."""
    cov_inv = np.linalg.inv(Sigma)
    diff = data - mu
    distances = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
    return distances

def create_clean_comparison(dist_data, gaussian_data, dist_name, mu, Sigma, output_dir):
    """Create clean comparison plot without text overlays."""
    
    # Use fewer points for cleaner visualization
    n_plot = 800  # Reduced from 5000
    idx_dist = np.random.choice(len(dist_data), min(n_plot, len(dist_data)), replace=False)
    idx_gauss = np.random.choice(len(gaussian_data), min(n_plot, len(gaussian_data)), replace=False)
    
    dist_plot = dist_data[idx_dist]
    gauss_plot = gaussian_data[idx_gauss]
    
    # Create figure with larger subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'{dist_name} vs Gaussian Distribution Comparison', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # 1. Scatter Plot Comparison - Upper Left (LARGER)
    ax1 = axes[0, 0]
    
    # Use better limits to prevent cutoff
    combined_data = np.vstack([dist_plot, gauss_plot])
    x_range = np.percentile(combined_data[:, 0], [5, 95])
    y_range = np.percentile(combined_data[:, 1], [5, 95])
    
    # Add padding to prevent cutoff
    x_padding = (x_range[1] - x_range[0]) * 0.15
    y_padding = (y_range[1] - y_range[0]) * 0.15
    
    x_limits = [x_range[0] - x_padding, x_range[1] + x_padding]
    y_limits = [y_range[0] - y_padding, y_range[1] + y_padding]
    
    # Plot with better colors and transparency
    ax1.scatter(gauss_plot[:, 0], gauss_plot[:, 1], 
               alpha=0.7, s=25, color='#1f77b4', label='Gaussian', edgecolors='none')
    ax1.scatter(dist_plot[:, 0], dist_plot[:, 1], 
               alpha=0.7, s=25, color='#d62728', label=dist_name, edgecolors='none')
    
    # Add elliptical contours (cleaner style)
    theta = np.linspace(0, 2*np.pi, 100)
    eigenvals, eigenvecs = np.linalg.eigh(Sigma)
    
    for k, alpha in zip([1, 2, 3], [0.8, 0.6, 0.4]):
        ellipse_points = np.array([np.cos(theta), np.sin(theta)]) * np.sqrt(k**2 * eigenvals[:, np.newaxis])
        ellipse_rotated = eigenvecs @ ellipse_points
        ellipse_final = ellipse_rotated.T + mu
        ax1.plot(ellipse_final[:, 0], ellipse_final[:, 1], 
                'k--', alpha=alpha, linewidth=1.5, label=f'{k}σ' if k == 1 else '')
    
    ax1.set_xlim(x_limits)
    ax1.set_ylim(y_limits)
    ax1.set_xlabel('X₁', fontsize=12)
    ax1.set_ylabel('X₂', fontsize=12)
    ax1.set_title('A. Scatter Plot Comparison', fontweight='bold', fontsize=14)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Distance Distributions - Upper Right
    ax2 = axes[0, 1]
    
    gaussian_distances = calculate_mahalanobis_distances(gaussian_data, mu, Sigma)
    dist_distances = calculate_mahalanobis_distances(dist_data, mu, Sigma)
    
    max_distance = 4.5
    bins = np.linspace(0, max_distance, 35)
    
    ax2.hist(gaussian_distances[gaussian_distances <= max_distance], bins=bins, alpha=0.7, 
             density=True, color='#1f77b4', label='Gaussian', edgecolor='white', linewidth=0.5)
    ax2.hist(dist_distances[dist_distances <= max_distance], bins=bins, alpha=0.7, 
             density=True, color='#d62728', label=dist_name, edgecolor='white', linewidth=0.5)
    
    # Theoretical chi-squared curve
    x_theory = np.linspace(0, max_distance, 200)
    chi2_theory = stats.chi2.pdf(x_theory**2, df=2) * 2 * x_theory
    ax2.plot(x_theory, chi2_theory, 'k-', linewidth=2.5, label='Theoretical χ²')
    
    ax2.set_xlabel('Mahalanobis Distance', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('B. Distance Distributions', fontweight='bold', fontsize=14)
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Q-Q Plot - Lower Left
    ax3 = axes[1, 0]
    
    quantiles = np.linspace(0.01, 0.99, 99)
    gaussian_quantiles = np.quantile(gaussian_distances, quantiles)
    dist_quantiles = np.quantile(dist_distances, quantiles)
    
    ax3.scatter(gaussian_quantiles, dist_quantiles, alpha=0.8, s=35, 
               color='#9467bd', edgecolors='white', linewidth=0.5)
    
    # Reference line
    max_val = 4.5
    ax3.plot([0, max_val], [0, max_val], 'r-', linewidth=2.5, alpha=0.8)
    
    ax3.set_xlabel('Gaussian Quantiles', fontsize=12)
    ax3.set_ylabel(f'{dist_name} Quantiles', fontsize=12)
    ax3.set_title('C. Q-Q Plot', fontweight='bold', fontsize=14)
    ax3.set_xlim(0, max_val)
    ax3.set_ylim(0, max_val)
    ax3.grid(True, alpha=0.3)
    
    # 4. Tail Probability Comparison - Lower Right
    ax4 = axes[1, 1]
    
    distance_thresholds = np.linspace(0, 4.5, 50)
    gaussian_tail_probs = [np.mean(gaussian_distances > t) for t in distance_thresholds]
    dist_tail_probs = [np.mean(dist_distances > t) for t in distance_thresholds]
    
    ax4.semilogy(distance_thresholds, gaussian_tail_probs, '#1f77b4', linewidth=2.5, label='Gaussian')
    ax4.semilogy(distance_thresholds, dist_tail_probs, '#d62728', linewidth=2.5, label=dist_name)
    
    # Theoretical tail probabilities
    theoretical_tail = 1 - stats.chi2.cdf(distance_thresholds**2, df=2)
    ax4.semilogy(distance_thresholds, theoretical_tail, 'k--', linewidth=2, label='Theoretical χ²')
    
    ax4.set_xlabel('Distance Threshold', fontsize=12)
    ax4.set_ylabel('Tail Probability (log scale)', fontsize=12)
    ax4.set_title('D. Tail Probability Comparison', fontweight='bold', fontsize=14)
    ax4.set_xlim(0, 4.5)
    ax4.set_ylim(1e-4, 1)
    ax4.legend(framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    dist_name_clean = dist_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace("'", '')
    filename = f"clean_{dist_name_clean}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved clean comparison: {filename}")

def create_3d_visualization(dist_data, gaussian_data, dist_name, mu, Sigma, output_dir):
    """Create 3D visualization of distributions."""
    
    # Use fewer points for 3D
    n_plot = 500
    idx_dist = np.random.choice(len(dist_data), min(n_plot, len(dist_data)), replace=False)
    idx_gauss = np.random.choice(len(gaussian_data), min(n_plot, len(gaussian_data)), replace=False)
    
    dist_plot = dist_data[idx_dist]
    gauss_plot = gaussian_data[idx_gauss]
    
    # Calculate distances for Z-axis
    dist_distances = calculate_mahalanobis_distances(dist_plot, mu, Sigma)
    gauss_distances = calculate_mahalanobis_distances(gauss_plot, mu, Sigma)
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 6))
    
    # First subplot: 3D scatter
    ax1 = fig.add_subplot(121, projection='3d')
    
    ax1.scatter(gauss_plot[:, 0], gauss_plot[:, 1], gauss_distances, 
               alpha=0.7, s=20, color='#1f77b4', label='Gaussian')
    ax1.scatter(dist_plot[:, 0], dist_plot[:, 1], dist_distances, 
               alpha=0.7, s=20, color='#d62728', label=dist_name)
    
    ax1.set_xlabel('X₁')
    ax1.set_ylabel('X₂')
    ax1.set_zlabel('Mahalanobis Distance')
    ax1.set_title('3D View: Position vs Distance')
    ax1.legend()
    
    # Second subplot: Surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Create grid for surface
    x_range = np.linspace(-3, 3, 30)
    y_range = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x_range, y_range)
    pos = np.dstack((X, Y))
    points = pos.reshape(-1, 2)
    
    # Calculate Gaussian PDF
    rv = stats.multivariate_normal(mean=mu, cov=Sigma)
    Z_gauss = rv.pdf(points).reshape(X.shape)
    
    # Plot surface
    surf = ax2.plot_surface(X, Y, Z_gauss, alpha=0.8, cmap='viridis')
    ax2.set_xlabel('X₁')
    ax2.set_ylabel('X₂')
    ax2.set_zlabel('Probability Density')
    ax2.set_title('3D Gaussian Density Surface')
    
    plt.tight_layout()
    
    # Save
    dist_name_clean = dist_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace("'", '')
    filename = f"3d_{dist_name_clean}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved 3D visualization: {filename}")

def create_generator_function_comparison(output_dir):
    """Create comprehensive generator function comparison plots."""
    
    # Define generator functions
    def gaussian_generator(u):
        return (2 * np.pi) ** (-1) * np.exp(-u / 2)
    
    def student_t_generator(u, nu):
        from scipy.special import gamma
        return (gamma((nu + 2) / 2) / (gamma(nu / 2) * np.sqrt(nu * np.pi))) * (1 + u / nu) ** (-(nu + 2) / 2)
    
    def laplace_generator(u):
        return 0.5 * np.exp(-np.sqrt(u))
    
    def cauchy_generator(u):
        return 1 / (np.pi * (1 + u))
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Generator Function Comparisons', fontsize=18, fontweight='bold')
    
    u = np.linspace(0, 8, 200)
    
    # Plot 1: Basic comparison
    ax1 = axes[0, 0]
    ax1.plot(u, gaussian_generator(u), 'b-', linewidth=2.5, label='Gaussian')
    ax1.plot(u, student_t_generator(u, 3), 'r-', linewidth=2.5, label="Student's t (ν=3)")
    ax1.plot(u, student_t_generator(u, 10), 'orange', linewidth=2.5, label="Student's t (ν=10)")
    ax1.plot(u, laplace_generator(u), 'g-', linewidth=2.5, label='Laplace')
    
    ax1.set_xlabel('u (Quadratic Form)', fontsize=12)
    ax1.set_ylabel('g(u)', fontsize=12)
    ax1.set_title('A. Generator Functions', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 0.4)
    
    # Plot 2: Log scale to show tails
    ax2 = axes[0, 1]
    ax2.semilogy(u, gaussian_generator(u), 'b-', linewidth=2.5, label='Gaussian')
    ax2.semilogy(u, student_t_generator(u, 3), 'r-', linewidth=2.5, label="Student's t (ν=3)")
    ax2.semilogy(u, student_t_generator(u, 10), 'orange', linewidth=2.5, label="Student's t (ν=10)")
    ax2.semilogy(u, laplace_generator(u), 'g-', linewidth=2.5, label='Laplace')
    
    ax2.set_xlabel('u (Quadratic Form)', fontsize=12)
    ax2.set_ylabel('log g(u)', fontsize=12)
    ax2.set_title('B. Generator Functions (Log Scale)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Tail behavior comparison
    ax3 = axes[1, 0]
    u_tail = np.linspace(2, 8, 100)
    ax3.plot(u_tail, gaussian_generator(u_tail), 'b-', linewidth=2.5, label='Gaussian')
    ax3.plot(u_tail, student_t_generator(u_tail, 3), 'r-', linewidth=2.5, label="Student's t (ν=3)")
    ax3.plot(u_tail, laplace_generator(u_tail), 'g-', linewidth=2.5, label='Laplace')
    ax3.plot(u_tail, cauchy_generator(u_tail), 'm-', linewidth=2.5, label='Cauchy (ν=1)')
    
    ax3.set_xlabel('u (Quadratic Form)', fontsize=12)
    ax3.set_ylabel('g(u)', fontsize=12)
    ax3.set_title('C. Tail Behavior (u > 2)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Normalized comparison at origin
    ax4 = axes[1, 1]
    u_center = np.linspace(0, 2, 100)
    
    # Normalize all functions to have same value at u=0
    norm_factor = gaussian_generator(0)
    ax4.plot(u_center, gaussian_generator(u_center) / norm_factor, 'b-', linewidth=2.5, label='Gaussian')
    ax4.plot(u_center, student_t_generator(u_center, 3) / student_t_generator(0, 3) * norm_factor / norm_factor, 'r-', linewidth=2.5, label="Student's t (ν=3)")
    ax4.plot(u_center, laplace_generator(u_center) / laplace_generator(0) * norm_factor / norm_factor, 'g-', linewidth=2.5, label='Laplace')
    
    ax4.set_xlabel('u (Quadratic Form)', fontsize=12)
    ax4.set_ylabel('Normalized g(u)', fontsize=12)
    ax4.set_title('D. Central Behavior (u < 2)', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    filename = "generator_function_comparison.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved generator function comparison: {filename}")

def main():
    """Generate all clean visualizations."""
    
    np.random.seed(42)
    
    # Parameters
    n_samples = 2000  # Reduced for cleaner plots
    mu = np.array([0, 0])
    Sigma = np.array([[2, 0.8], [0.8, 1]])
    
    output_dir = "../assets"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate reference data
    print("Generating Gaussian reference data...")
    gaussian_data = np.random.multivariate_normal(mu, Sigma, n_samples)
    
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
    
    # Generate clean comparisons
    for dist_config in distributions:
        print(f"Generating clean comparison for {dist_config['name']}...")
        dist_data = dist_config['generator']()
        create_clean_comparison(dist_data, gaussian_data, dist_config['name'], mu, Sigma, output_dir)
        create_3d_visualization(dist_data, gaussian_data, dist_config['name'], mu, Sigma, output_dir)
    
    # Generate generator function comparison
    print("Generating generator function comparisons...")
    create_generator_function_comparison(output_dir)
    
    print("\n✅ All clean visualizations generated successfully!")
    print(f"📁 Plots saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()