"""
Fixed Plotting Script - Addresses All Layout and Visualization Issues
==================================================================

This script creates clean, professional plots with:
1. Fixed title overlap issues
2. Proper legend-color matching  
3. Adequate scatter plot margins
4. Multiple generator function plots
5. Single representative 3D plot
6. Clean contour plot without text overlap
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import os

# Set style for professional plots
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

def create_fixed_comparison(dist_data, gaussian_data, dist_name, mu, Sigma, output_dir):
    """Create comparison plot with all issues fixed."""
    
    # Use fewer points for cleaner visualization
    n_plot = 800
    idx_dist = np.random.choice(len(dist_data), min(n_plot, len(dist_data)), replace=False)
    idx_gauss = np.random.choice(len(gaussian_data), min(n_plot, len(gaussian_data)), replace=False)
    
    dist_plot = dist_data[idx_dist]
    gauss_plot = gaussian_data[idx_gauss]
    
    # Create figure with FIXED spacing
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # FIXED: Title positioning to avoid overlap
    fig.suptitle(f'{dist_name} vs Gaussian Distribution Comparison', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # FIXED: Adequate subplot spacing
    plt.subplots_adjust(top=0.92, hspace=0.35, wspace=0.25)
    
    # 1. Scatter Plot - FIXED margins and ellipse visibility
    ax1 = axes[0, 0]
    
    # FIXED: Better margin calculation for ellipse visibility
    combined_data = np.vstack([dist_plot, gauss_plot])
    x_range = np.percentile(combined_data[:, 0], [2, 98])  # Wider percentile range
    y_range = np.percentile(combined_data[:, 1], [2, 98])
    
    # FIXED: Increased padding for ellipse visibility (25% instead of 15%)
    x_padding = (x_range[1] - x_range[0]) * 0.25
    y_padding = (y_range[1] - y_range[0]) * 0.25
    
    x_limits = [x_range[0] - x_padding, x_range[1] + x_padding]
    y_limits = [y_range[0] - y_padding, y_range[1] + y_padding]
    
    # Plot data with consistent colors
    ax1.scatter(gauss_plot[:, 0], gauss_plot[:, 1], 
               alpha=0.7, s=25, color='#1f77b4', label='Gaussian', edgecolors='none')
    ax1.scatter(dist_plot[:, 0], dist_plot[:, 1], 
               alpha=0.7, s=25, color='#d62728', label=dist_name, edgecolors='none')
    
    # Add elliptical contours with better visibility
    theta = np.linspace(0, 2*np.pi, 100)
    eigenvals, eigenvecs = np.linalg.eigh(Sigma)
    
    for k, alpha_val in zip([1, 2, 3], [0.8, 0.6, 0.4]):
        ellipse_points = np.array([np.cos(theta), np.sin(theta)]) * np.sqrt(k**2 * eigenvals[:, np.newaxis])
        ellipse_rotated = eigenvecs @ ellipse_points
        ellipse_final = ellipse_rotated.T + mu
        ax1.plot(ellipse_final[:, 0], ellipse_final[:, 1], 
                'k--', alpha=alpha_val, linewidth=1.5, 
                label=f'{k}σ contour' if k == 1 else '')
    
    # FIXED: Proper limits to show ellipses completely
    ax1.set_xlim(x_limits)
    ax1.set_ylim(y_limits)
    ax1.set_xlabel('X₁', fontsize=12)
    ax1.set_ylabel('X₂', fontsize=12)
    ax1.set_title('A. Scatter Plot Comparison', fontweight='bold', fontsize=14, pad=15)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Distance Distributions - FIXED legend matching
    ax2 = axes[0, 1]
    
    gaussian_distances = calculate_mahalanobis_distances(gaussian_data, mu, Sigma)
    dist_distances = calculate_mahalanobis_distances(dist_data, mu, Sigma)
    
    max_distance = 4.5
    bins = np.linspace(0, max_distance, 35)
    
    # FIXED: All plotted elements have legend entries
    ax2.hist(gaussian_distances[gaussian_distances <= max_distance], bins=bins, alpha=0.7, 
             density=True, color='#1f77b4', label='Gaussian', edgecolor='white', linewidth=0.5)
    ax2.hist(dist_distances[dist_distances <= max_distance], bins=bins, alpha=0.7, 
             density=True, color='#d62728', label=dist_name, edgecolor='white', linewidth=0.5)
    
    # Theoretical curve
    x_theory = np.linspace(0, max_distance, 200)
    chi2_theory = stats.chi2.pdf(x_theory**2, df=2) * 2 * x_theory
    ax2.plot(x_theory, chi2_theory, color='black', linewidth=2.5, label='Theoretical χ²')
    
    ax2.set_xlabel('Mahalanobis Distance', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('B. Distance Distributions', fontweight='bold', fontsize=14, pad=15)
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Q-Q Plot
    ax3 = axes[1, 0]
    
    quantiles = np.linspace(0.01, 0.99, 99)
    gaussian_quantiles = np.quantile(gaussian_distances, quantiles)
    dist_quantiles = np.quantile(dist_distances, quantiles)
    
    ax3.scatter(gaussian_quantiles, dist_quantiles, alpha=0.8, s=35, 
               color='#9467bd', edgecolors='white', linewidth=0.5)
    
    # Reference line
    max_val = 4.5
    ax3.plot([0, max_val], [0, max_val], 'r-', linewidth=2.5, alpha=0.8, label='y = x')
    
    ax3.set_xlabel('Gaussian Quantiles', fontsize=12)
    ax3.set_ylabel(f'{dist_name} Quantiles', fontsize=12)
    ax3.set_title('C. Q-Q Plot', fontweight='bold', fontsize=14, pad=15)
    ax3.set_xlim(0, max_val)
    ax3.set_ylim(0, max_val)
    ax3.legend(framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Tail Probability Comparison
    ax4 = axes[1, 1]
    
    distance_thresholds = np.linspace(0, 4.5, 50)
    gaussian_tail_probs = [np.mean(gaussian_distances > t) for t in distance_thresholds]
    dist_tail_probs = [np.mean(dist_distances > t) for t in distance_thresholds]
    
    ax4.semilogy(distance_thresholds, gaussian_tail_probs, color='#1f77b4', linewidth=2.5, label='Gaussian')
    ax4.semilogy(distance_thresholds, dist_tail_probs, color='#d62728', linewidth=2.5, label=dist_name)
    
    # Theoretical tail probabilities
    theoretical_tail = 1 - stats.chi2.cdf(distance_thresholds**2, df=2)
    ax4.semilogy(distance_thresholds, theoretical_tail, 'k--', linewidth=2, label='Theoretical χ²')
    
    ax4.set_xlabel('Distance Threshold', fontsize=12)
    ax4.set_ylabel('Tail Probability (log scale)', fontsize=12)
    ax4.set_title('D. Tail Probability Comparison', fontweight='bold', fontsize=14, pad=15)
    ax4.set_xlim(0, 4.5)
    ax4.set_ylim(1e-4, 1)
    ax4.legend(framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    
    # Save
    dist_name_clean = dist_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace("'", '')
    filename = f"fixed_{dist_name_clean}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved fixed comparison: {filename}")

def create_fixed_contour_comparison(output_dir):
    """Create contour comparison without text overlap."""
    
    mu = np.array([0, 0])
    Sigma = np.array([[2, 0.8], [0.8, 1]])
    
    x_range = np.linspace(-4, 4, 100)
    y_range = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    pos = np.dstack((X, Y))
    points = pos.reshape(-1, 2)
    
    # Calculate PDFs
    rv_gauss = stats.multivariate_normal(mean=mu, cov=Sigma)
    gaussian_pdf = rv_gauss.pdf(points).reshape(X.shape)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Gaussian contours
    cs_gauss = ax.contour(X, Y, gaussian_pdf, levels=6, colors='blue', alpha=0.8, linestyles='solid', linewidths=2)
    ax.clabel(cs_gauss, inline=1, fontsize=10, fmt='%.3f')
    
    # Calculate t-distribution PDF
    def multivariate_student_t_pdf(x, nu, mu, Sigma):
        p = len(mu)
        diff = x - mu
        Sigma_inv = np.linalg.inv(Sigma)
        quad_form = np.sum(diff @ Sigma_inv * diff, axis=1)
        
        from scipy.special import gamma
        norm_const = (gamma((nu + p) / 2) / 
                     (gamma(nu / 2) * (nu * np.pi)**(p/2) * np.sqrt(np.linalg.det(Sigma))))
        pdf = norm_const * (1 + quad_form / nu)**(-(nu + p) / 2)
        return pdf
    
    t3_pdf = multivariate_student_t_pdf(points, nu=3, mu=mu, Sigma=Sigma).reshape(X.shape)
    cs_t = ax.contour(X, Y, t3_pdf, levels=6, colors='red', alpha=0.8, linestyles='dashed', linewidths=2)
    ax.clabel(cs_t, inline=1, fontsize=10, fmt='%.3f')
    
    # FIXED: Title and legend positioning
    ax.set_title('Distribution Shape Comparison: Gaussian vs Student\'s t (ν=3)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # FIXED: Legend positioning to avoid overlap
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, linestyle='-', label='Gaussian'),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label='Student\'s t (ν=3)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98), framealpha=0.9)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X₁', fontsize=14)
    ax.set_ylabel('X₂', fontsize=14)
    ax.set_aspect('equal')
    
    # Save
    filename = "fixed_contour_comparison.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved fixed contour plot: {filename}")

def create_single_3d_visualization(output_dir):
    """Create one representative 3D visualization."""
    
    np.random.seed(42)
    n_samples = 1000
    mu = np.array([0, 0])
    Sigma = np.array([[2, 0.8], [0.8, 1]])
    
    # Generate data for most informative case (Student's t nu=3)
    gaussian_data = np.random.multivariate_normal(mu, Sigma, n_samples)
    t_data = multivariate_student_t(n_samples, nu=3, mu=mu, Sigma=Sigma)
    
    # Calculate distances
    gaussian_distances = calculate_mahalanobis_distances(gaussian_data, mu, Sigma)
    t_distances = calculate_mahalanobis_distances(t_data, mu, Sigma)
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 6))
    
    # 3D scatter
    ax1 = fig.add_subplot(121, projection='3d')
    
    ax1.scatter(gaussian_data[:, 0], gaussian_data[:, 1], gaussian_distances, 
               alpha=0.7, s=20, color='#1f77b4', label='Gaussian')
    ax1.scatter(t_data[:, 0], t_data[:, 1], t_distances, 
               alpha=0.7, s=20, color='#d62728', label='Student\'s t (ν=3)')
    
    ax1.set_xlabel('X₁')
    ax1.set_ylabel('X₂')
    ax1.set_zlabel('Mahalanobis Distance')
    ax1.set_title('3D Scatter: Position vs Distance')
    ax1.legend()
    
    # 3D surface
    ax2 = fig.add_subplot(122, projection='3d')
    
    x_range = np.linspace(-3, 3, 30)
    y_range = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x_range, y_range)
    pos = np.dstack((X, Y))
    points = pos.reshape(-1, 2)
    
    rv = stats.multivariate_normal(mean=mu, cov=Sigma)
    Z_gauss = rv.pdf(points).reshape(X.shape)
    
    surf = ax2.plot_surface(X, Y, Z_gauss, alpha=0.8, cmap='viridis')
    ax2.set_xlabel('X₁')
    ax2.set_ylabel('X₂')
    ax2.set_zlabel('Probability Density')
    ax2.set_title('3D Gaussian Density Surface')
    
    plt.suptitle('3D Visualization of Elliptical Distributions', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save
    filename = "single_3d_visualization.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved single 3D visualization: {filename}")

def create_multiple_generator_plots(output_dir):
    """Create multiple generator function plots as requested."""
    
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
    
    # Plot 1: Basic Comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    u = np.linspace(0, 8, 200)
    
    ax.plot(u, gaussian_generator(u), 'b-', linewidth=3, label='Gaussian')
    ax.plot(u, student_t_generator(u, 3), 'r-', linewidth=3, label="Student's t (ν=3)")
    ax.plot(u, student_t_generator(u, 10), 'orange', linewidth=3, label="Student's t (ν=10)")
    ax.plot(u, laplace_generator(u), 'g-', linewidth=3, label='Laplace')
    
    ax.set_xlabel('u (Quadratic Form)', fontsize=14)
    ax.set_ylabel('g(u)', fontsize=14)
    ax.set_title('Generator Functions: Basic Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.4)
    
    filename1 = "generator_basic_comparison.png"
    filepath1 = os.path.join(output_dir, filename1)
    plt.savefig(filepath1, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Log Scale Analysis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.semilogy(u, gaussian_generator(u), 'b-', linewidth=3, label='Gaussian')
    ax.semilogy(u, student_t_generator(u, 3), 'r-', linewidth=3, label="Student's t (ν=3)")
    ax.semilogy(u, student_t_generator(u, 10), 'orange', linewidth=3, label="Student's t (ν=10)")
    ax.semilogy(u, laplace_generator(u), 'g-', linewidth=3, label='Laplace')
    
    ax.set_xlabel('u (Quadratic Form)', fontsize=14)
    ax.set_ylabel('log g(u)', fontsize=14)
    ax.set_title('Generator Functions: Logarithmic Scale (Tail Emphasis)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    filename2 = "generator_log_scale.png"
    filepath2 = os.path.join(output_dir, filename2)
    plt.savefig(filepath2, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Tail Behavior Focus
    fig, ax = plt.subplots(figsize=(12, 8))
    u_tail = np.linspace(2, 8, 100)
    
    ax.plot(u_tail, gaussian_generator(u_tail), 'b-', linewidth=3, label='Gaussian')
    ax.plot(u_tail, student_t_generator(u_tail, 3), 'r-', linewidth=3, label="Student's t (ν=3)")
    ax.plot(u_tail, laplace_generator(u_tail), 'g-', linewidth=3, label='Laplace')
    ax.plot(u_tail, cauchy_generator(u_tail), 'm-', linewidth=3, label='Cauchy (ν=1)')
    
    ax.set_xlabel('u (Quadratic Form)', fontsize=14)
    ax.set_ylabel('g(u)', fontsize=14)
    ax.set_title('Generator Functions: Tail Behavior Focus (u > 2)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    filename3 = "generator_tail_behavior.png"
    filepath3 = os.path.join(output_dir, filename3)
    plt.savefig(filepath3, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Mathematical Properties
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Generator Function Mathematical Properties', fontsize=18, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
    
    # Derivative analysis
    ax1 = axes[0, 0]
    u_fine = np.linspace(0.1, 6, 200)
    gauss_vals = gaussian_generator(u_fine)
    gauss_deriv = np.gradient(gauss_vals, u_fine)
    t3_vals = student_t_generator(u_fine, 3)
    t3_deriv = np.gradient(t3_vals, u_fine)
    
    ax1.plot(u_fine, gauss_deriv, 'b-', linewidth=2, label='Gaussian')
    ax1.plot(u_fine, t3_deriv, 'r-', linewidth=2, label="Student's t (ν=3)")
    ax1.set_title('A. Derivatives (Monotonicity)', fontweight='bold')
    ax1.set_xlabel('u')
    ax1.set_ylabel("g'(u)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Curvature analysis
    ax2 = axes[0, 1]
    gauss_second_deriv = np.gradient(gauss_deriv, u_fine)
    t3_second_deriv = np.gradient(t3_deriv, u_fine)
    
    ax2.plot(u_fine, gauss_second_deriv, 'b-', linewidth=2, label='Gaussian')
    ax2.plot(u_fine, t3_second_deriv, 'r-', linewidth=2, label="Student's t (ν=3)")
    ax2.set_title('B. Second Derivatives (Curvature)', fontweight='bold')
    ax2.set_xlabel('u')
    ax2.set_ylabel('g"(u)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Ratio analysis
    ax3 = axes[1, 0]
    ratio_t3_gauss = t3_vals / gauss_vals
    ratio_laplace_gauss = laplace_generator(u_fine) / gauss_vals
    
    ax3.plot(u_fine, ratio_t3_gauss, 'r-', linewidth=2, label='t(ν=3) / Gaussian')
    ax3.plot(u_fine, ratio_laplace_gauss, 'g-', linewidth=2, label='Laplace / Gaussian')
    ax3.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Reference (ratio=1)')
    ax3.set_title('C. Ratios to Gaussian', fontweight='bold')
    ax3.set_xlabel('u')
    ax3.set_ylabel('Ratio')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Tail weight comparison
    ax4 = axes[1, 1]
    u_tail_extended = np.linspace(1, 10, 100)
    
    # Calculate relative tail weights
    gauss_tail = gaussian_generator(u_tail_extended)
    t3_tail = student_t_generator(u_tail_extended, 3)
    laplace_tail = laplace_generator(u_tail_extended)
    
    ax4.semilogy(u_tail_extended, gauss_tail / gauss_tail[0], 'b-', linewidth=2, label='Gaussian')
    ax4.semilogy(u_tail_extended, t3_tail / t3_tail[0], 'r-', linewidth=2, label="Student's t (ν=3)")
    ax4.semilogy(u_tail_extended, laplace_tail / laplace_tail[0], 'g-', linewidth=2, label='Laplace')
    ax4.set_title('D. Normalized Tail Decay', fontweight='bold')
    ax4.set_xlabel('u')
    ax4.set_ylabel('Normalized g(u) (log scale)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    filename4 = "generator_mathematical_properties.png"
    filepath4 = os.path.join(output_dir, filename4)
    plt.savefig(filepath4, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved generator function plots:")
    print(f"  - {filename1}")
    print(f"  - {filename2}")
    print(f"  - {filename3}")
    print(f"  - {filename4}")

def main():
    """Generate all fixed visualizations."""
    
    np.random.seed(42)
    
    # Parameters
    n_samples = 2000
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
    
    # Generate fixed comparisons
    for dist_config in distributions:
        print(f"Generating fixed comparison for {dist_config['name']}...")
        dist_data = dist_config['generator']()
        create_fixed_comparison(dist_data, gaussian_data, dist_config['name'], mu, Sigma, output_dir)
    
    # Generate fixed contour comparison
    print("Generating fixed contour comparison...")
    create_fixed_contour_comparison(output_dir)
    
    # Generate single 3D visualization
    print("Generating single 3D visualization...")
    create_single_3d_visualization(output_dir)
    
    # Generate multiple generator function plots
    print("Generating multiple generator function plots...")
    create_multiple_generator_plots(output_dir)
    
    print("\n✅ All fixed visualizations generated successfully!")
    print(f"📁 Plots saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()