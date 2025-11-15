"""
Enhanced Individual Distribution Comparisons with Better Scaling and Explanations
==============================================================================

This script generates enhanced individual comparison plots with:
1. Proper scaling for normal-looking distributions
2. Clear labels and explanations for each subplot
3. Better visual comparisons between distributions
4. Standardized scales across comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.covariance import EmpiricalCovariance
import os
from matplotlib.patches import Ellipse

# Set style for better plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

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

def create_enhanced_comparison(dist_data, gaussian_data, dist_name, mu, Sigma, output_dir):
    """Create enhanced comparison plot with better scaling and explanations."""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 13))
    fig.suptitle(f'{dist_name} vs Gaussian Distribution Comparison', fontsize=18, fontweight='bold', y=0.98)
    
    # Add a subtitle explaining the overall purpose
    plt.figtext(0.5, 0.93, 
                "Comparing key statistical properties of different elliptical distributions against a Gaussian reference", 
                ha="center", fontsize=12, style='italic', bbox={"facecolor":"lightgrey", "alpha":0.2, "pad":5})

    # Use gridspec for more control over layout
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Scatter Plot Comparison - Upper Left
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Use standardized scale for all distributions (-3 to 3)
    # Filter extreme outliers for better visualization
    def filter_standard_range(data, std_dev=3):
        mask = np.all(np.abs(data) < std_dev, axis=1)
        return data[mask]
    
    gaussian_filtered = filter_standard_range(gaussian_data, 3)
    dist_filtered = filter_standard_range(dist_data, 3)
    
    # Plot points with better aesthetics
    ax1.scatter(gaussian_filtered[:, 0], gaussian_filtered[:, 1], 
               alpha=0.6, s=20, color='blue', label='Gaussian Reference')
    ax1.scatter(dist_filtered[:, 0], dist_filtered[:, 1], 
               alpha=0.6, s=20, color='red', label=dist_name)
    
    # Add contour lines (1, 2, 3 sigma)
    theta = np.linspace(0, 2*np.pi, 100)
    eigenvals, eigenvecs = np.linalg.eigh(Sigma)
    
    for k in [1, 2, 3]:  # 1, 2, 3 sigma contours
        # Create ellipse points
        ellipse_points = np.array([np.cos(theta), np.sin(theta)]) * np.sqrt(k**2 * eigenvals[:, np.newaxis])
        ellipse_rotated = eigenvecs @ ellipse_points
        ellipse_final = ellipse_rotated.T + mu
        ax1.plot(ellipse_final[:, 0], ellipse_final[:, 1], 'k--', alpha=0.5, linewidth=1)
    
    # Set equal axis limits for all plots
    ax1.set_xlim(-3.2, 3.2)
    ax1.set_ylim(-3.2, 3.2)
    ax1.set_xlabel('X₁')
    ax1.set_ylabel('X₂')
    ax1.set_title('A. Scatter Plot Comparison', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add explanatory text box
    explanation_text = (
        "This plot shows random samples from both distributions.\n"
        "• The Gaussian (blue) samples cluster more tightly around the center\n"
        "• The dashed lines show the 1σ, 2σ, and 3σ contours\n"
    )
    if "Student's t" in dist_name:
        explanation_text += "• Note how the red points extend further from the center,\n  showing the heavier tails of the t-distribution"
    elif "Laplace" in dist_name:
        explanation_text += "• The Laplace distribution (red) shows more concentration\n  around the center with moderate outliers"
    
    ax1.text(0.5, -0.15, explanation_text, transform=ax1.transAxes,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
             ha='center', va='center', fontsize=10)
    
    # 2. Distance Distributions - Upper Right
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Calculate Mahalanobis distances
    gaussian_distances = calculate_mahalanobis_distances(gaussian_data, mu, Sigma)
    dist_distances = calculate_mahalanobis_distances(dist_data, mu, Sigma)
    
    # Set consistent scale for x-axis
    max_distance = 4.0
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
    ax2.set_title('B. Distance Distributions', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add explanatory text box
    explanation_text = (
        "This plot shows the distribution of Mahalanobis distances.\n"
        "• For Gaussian data, distances follow a chi-squared distribution\n"
        "• Theoretical χ² curve (dashed) matches blue Gaussian histogram\n"
    )
    if "Student's t" in dist_name:
        if "ν=3" in dist_name:
            explanation_text += "• The t-distribution with ν=3 shows significant deviations\n  with more mass at larger distances (heavy tails)"
        else:
            explanation_text += "• The t-distribution with higher ν shows distribution\n  closer to the Gaussian but still with heavier tails"
    elif "Laplace" in dist_name:
        explanation_text += "• The Laplace distribution shows moderate deviation\n  from the chi-squared pattern"
    
    ax2.text(0.5, -0.15, explanation_text, transform=ax2.transAxes,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
             ha='center', va='center', fontsize=10)
    
    # 3. Q-Q Plot - Lower Left
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Use percentile method for Q-Q plot to avoid extreme values
    quantiles = np.linspace(0.01, 0.99, 100)
    gaussian_quantiles = np.quantile(gaussian_distances, quantiles)
    dist_quantiles = np.quantile(dist_distances, quantiles)
    
    ax3.scatter(gaussian_quantiles, dist_quantiles, alpha=0.8, s=30, color='purple')
    
    # Add reference line
    max_val = 4.0
    ax3.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
    
    ax3.set_xlabel('Gaussian Quantiles')
    ax3.set_ylabel(f'{dist_name} Quantiles')
    ax3.set_title('C. Q-Q Plot (Distance Distributions)', fontweight='bold')
    ax3.set_xlim(0, max_val)
    ax3.set_ylim(0, max_val)
    ax3.grid(True, alpha=0.3)
    
    # Add explanatory text box
    explanation_text = (
        "This Q-Q plot compares quantiles of both distributions.\n"
        "• Points along the red dashed line indicate identical distributions\n"
        "• Points above the line show heavier tails than Gaussian\n"
        "• Points below the line would show lighter tails than Gaussian\n"
    )
    if "Student's t" in dist_name:
        explanation_text += "• The upward curve is characteristic of heavy-tailed distributions"
    elif "Laplace" in dist_name:
        explanation_text += "• The Laplace shows moderate deviation from the line"
    
    ax3.text(0.5, -0.15, explanation_text, transform=ax3.transAxes,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
             ha='center', va='center', fontsize=10)
    
    # 4. Tail Probability Comparison - Lower Right
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate tail probabilities
    distance_thresholds = np.linspace(0, 4, 50)
    gaussian_tail_probs = [np.mean(gaussian_distances > t) for t in distance_thresholds]
    dist_tail_probs = [np.mean(dist_distances > t) for t in distance_thresholds]
    
    ax4.semilogy(distance_thresholds, gaussian_tail_probs, 'b-', linewidth=2, label='Gaussian')
    ax4.semilogy(distance_thresholds, dist_tail_probs, 'r-', linewidth=2, label=dist_name)
    
    # Add theoretical tail probabilities for Gaussian (chi-squared with 2 DOF)
    theoretical_tail = 1 - stats.chi2.cdf(distance_thresholds**2, df=2)
    ax4.semilogy(distance_thresholds, theoretical_tail, 'k--', linewidth=2, label='Theoretical χ²')
    
    ax4.set_xlabel('Distance Threshold')
    ax4.set_ylabel('Probability (log scale)')
    ax4.set_title('D. Tail Probability Comparison', fontweight='bold')
    ax4.set_xlim(0, 4)
    ax4.set_ylim(1e-3, 1)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add explanatory text box
    explanation_text = (
        "This plot shows probability of exceeding distance thresholds.\n"
        "• Y-axis is logarithmic to emphasize tail behavior\n"
        "• Lower curve values mean fewer outliers at that distance\n"
    )
    if "Student's t" in dist_name:
        explanation_text += f"• The {dist_name} (red) shows much higher\n  probabilities in the tail region than Gaussian"
    elif "Laplace" in dist_name:
        explanation_text += "• The Laplace distribution shows moderately higher\n  tail probabilities than Gaussian"
    
    ax4.text(0.5, -0.15, explanation_text, transform=ax4.transAxes,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
             ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save with enhanced filename
    dist_name_clean = dist_name.lower().replace(' ', '_')
    dist_name_clean = dist_name_clean.replace('(', '').replace(')', '')
    dist_name_clean = dist_name_clean.replace('=', '').replace("'", '')
    filename = f"enhanced_{dist_name_clean}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved enhanced comparison: {filename}")

def create_all_contours_plot(output_dir):
    """Create single contour plot showing all distributions."""
    
    # Parameters
    mu = np.array([0, 0])
    Sigma = np.array([[2, 0.8], [0.8, 1]])
    
    # Create grid for contour plots
    x_range = np.linspace(-4, 4, 100)
    y_range = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    pos = np.dstack((X, Y))
    points = pos.reshape(-1, 2)
    
    # Calculate Gaussian PDF
    rv = stats.multivariate_normal(mean=mu, cov=Sigma)
    gaussian_pdf = rv.pdf(points).reshape(X.shape)
    
    # Create plot
    fig, ax = plt.figure(figsize=(10, 8)), plt.subplot(111)
    
    # Add contours for Gaussian
    cs0 = plt.contour(X, Y, gaussian_pdf, levels=6, colors='blue', alpha=0.8, linestyles='solid')
    plt.clabel(cs0, inline=1, fontsize=10, fmt='%.2f')
    
    # Function to calculate multivariate Student's t PDF
    def multivariate_student_t_pdf(x, nu, mu, Sigma):
        """Calculate multivariate Student's t PDF."""
        p = len(mu)
        diff = x - mu
        Sigma_inv = np.linalg.inv(Sigma)
        
        # Quadratic form
        quad_form = np.sum(diff @ Sigma_inv * diff, axis=1)
        
        # Normalization constant
        from scipy.special import gamma
        norm_const = (gamma((nu + p) / 2) / 
                     (gamma(nu / 2) * (nu * np.pi)**(p/2) * np.sqrt(np.linalg.det(Sigma))))
        
        # PDF
        pdf = norm_const * (1 + quad_form / nu)**(-(nu + p) / 2)
        return pdf
    
    # Add contours for t-distribution with nu=3
    t3_pdf = multivariate_student_t_pdf(points, nu=3, mu=mu, Sigma=Sigma).reshape(X.shape)
    cs1 = plt.contour(X, Y, t3_pdf, levels=6, colors='red', alpha=0.8, linestyles='dashed')
    plt.clabel(cs1, inline=1, fontsize=10, fmt='%.2f')
    
    # Add title and legend
    plt.title('Contour Comparison: Gaussian vs Student\'s t (ν=3)', fontsize=14, fontweight='bold')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, linestyle='-', label='Gaussian'),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label='Student\'s t (ν=3)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add annotations
    plt.annotate('Note how t-distribution\ncontours are more\nspread out', 
                xy=(2.5, 2), xytext=(3, 2.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8),
                fontsize=12)
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('X₁')
    plt.ylabel('X₂')
    
    # Save
    filename = "enhanced_contour_comparison.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved enhanced contour plot: {filename}")

def main():
    """Generate enhanced individual comparison plots with better scaling and explanations."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define parameters
    n_samples = 5000
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
    
    # Generate enhanced individual comparisons
    for dist_config in distributions:
        print(f"Generating enhanced comparison for {dist_config['name']}...")
        dist_data = dist_config['generator']()
        create_enhanced_comparison(
            dist_data, gaussian_data, dist_config['name'], mu, Sigma, output_dir
        )
    
    # Create enhanced contour plot
    create_all_contours_plot(output_dir)
    
    print("\n✅ All enhanced comparison plots generated successfully!")
    print(f"📁 Plots saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()