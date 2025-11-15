"""
Clean Contour Plots for Distribution Comparison
==============================================

This script generates clean contour plots that show the true shape of distributions
without outlier distortion.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Ellipse
import os

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

def multivariate_laplace_pdf(x, mu, Sigma):
    """Calculate multivariate Laplace PDF (approximation)."""
    p = len(mu)
    diff = x - mu
    Sigma_inv = np.linalg.inv(Sigma)
    
    # Mahalanobis distance
    distances = np.sqrt(np.sum(diff @ Sigma_inv * diff, axis=1))
    
    # Normalization constant (approximate)
    norm_const = 1 / (2 * np.pi * np.sqrt(np.linalg.det(Sigma)))
    
    # PDF (exponential decay)
    pdf = norm_const * np.exp(-distances)
    return pdf

def create_clean_contour_comparison():
    """Create clean contour plots showing distribution shapes."""
    
    # Parameters
    mu = np.array([0, 0])
    Sigma = np.array([[2, 0.8], [0.8, 1]])
    
    # Create grid for contour plots
    x_range = np.linspace(-4, 4, 100)
    y_range = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    pos = np.dstack((X, Y))
    points = pos.reshape(-1, 2)
    
    # Calculate PDFs
    gaussian_pdf = stats.multivariate_normal.pdf(points, mu, Sigma).reshape(X.shape)
    t3_pdf = multivariate_student_t_pdf(points, nu=3, mu=mu, Sigma=Sigma).reshape(X.shape)
    t10_pdf = multivariate_student_t_pdf(points, nu=10, mu=mu, Sigma=Sigma).reshape(X.shape)
    laplace_pdf = multivariate_laplace_pdf(points, mu, Sigma).reshape(X.shape)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Clean Distribution Shape Comparison', fontsize=16, fontweight='bold')
    
    # Color map
    cmap = 'viridis'
    
    # 1. Gaussian
    ax1 = axes[0, 0]
    cs1 = ax1.contour(X, Y, gaussian_pdf, levels=8, colors='blue', alpha=0.8)
    ax1.contourf(X, Y, gaussian_pdf, levels=20, cmap=cmap, alpha=0.6)
    ax1.set_title('Multivariate Gaussian', fontweight='bold')
    ax1.set_xlabel('X₁')
    ax1.set_ylabel('X₂')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 2. Student's t (ν=3)
    ax2 = axes[0, 1]
    cs2 = ax2.contour(X, Y, t3_pdf, levels=8, colors='red', alpha=0.8)
    ax2.contourf(X, Y, t3_pdf, levels=20, cmap=cmap, alpha=0.6)
    ax2.set_title("Student's t (ν=3) - Heavy Tails", fontweight='bold')
    ax2.set_xlabel('X₁')
    ax2.set_ylabel('X₂')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # 3. Student's t (ν=10)
    ax3 = axes[1, 0]
    cs3 = ax3.contour(X, Y, t10_pdf, levels=8, colors='orange', alpha=0.8)
    ax3.contourf(X, Y, t10_pdf, levels=20, cmap=cmap, alpha=0.6)
    ax3.set_title("Student's t (ν=10) - Moderate Tails", fontweight='bold')
    ax3.set_xlabel('X₁')
    ax3.set_ylabel('X₂')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # 4. Multivariate Laplace
    ax4 = axes[1, 1]
    cs4 = ax4.contour(X, Y, laplace_pdf, levels=8, colors='green', alpha=0.8)
    ax4.contourf(X, Y, laplace_pdf, levels=20, cmap=cmap, alpha=0.6)
    ax4.set_title('Multivariate Laplace', fontweight='bold')
    ax4.set_xlabel('X₁')
    ax4.set_ylabel('X₂')
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = "../assets"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "clean_distribution_contours.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved clean contour comparison: clean_distribution_contours.png")

def create_side_by_side_comparison():
    """Create side-by-side Gaussian vs other distribution comparisons."""
    
    # Parameters
    mu = np.array([0, 0])
    Sigma = np.array([[2, 0.8], [0.8, 1]])
    
    # Create grid
    x_range = np.linspace(-4, 4, 100)
    y_range = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    pos = np.dstack((X, Y))
    points = pos.reshape(-1, 2)
    
    # Calculate PDFs
    gaussian_pdf = stats.multivariate_normal.pdf(points, mu, Sigma).reshape(X.shape)
    
    distributions = [
        ("Student's t (ν=3)", multivariate_student_t_pdf(points, nu=3, mu=mu, Sigma=Sigma).reshape(X.shape), 'red'),
        ("Student's t (ν=10)", multivariate_student_t_pdf(points, nu=10, mu=mu, Sigma=Sigma).reshape(X.shape), 'orange'),
        ("Multivariate Laplace", multivariate_laplace_pdf(points, mu, Sigma).reshape(X.shape), 'green')
    ]
    
    output_dir = "../assets"
    os.makedirs(output_dir, exist_ok=True)
    
    for dist_name, dist_pdf, color in distributions:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{dist_name} vs Gaussian Comparison', fontsize=16, fontweight='bold')
        
        # Gaussian
        ax1.contour(X, Y, gaussian_pdf, levels=8, colors='blue', alpha=0.8)
        ax1.contourf(X, Y, gaussian_pdf, levels=20, cmap='Blues', alpha=0.6)
        ax1.set_title('Multivariate Gaussian', fontweight='bold')
        ax1.set_xlabel('X₁')
        ax1.set_ylabel('X₂')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Other distribution
        ax2.contour(X, Y, dist_pdf, levels=8, colors=color, alpha=0.8)
        ax2.contourf(X, Y, dist_pdf, levels=20, cmap='Reds' if color == 'red' else 'Oranges' if color == 'orange' else 'Greens', alpha=0.6)
        ax2.set_title(dist_name, fontweight='bold')
        ax2.set_xlabel('X₁')
        ax2.set_ylabel('X₂')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        plt.tight_layout()
        
        # Save
        filename = f"side_by_side_{dist_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved side-by-side comparison: {filename}")

def main():
    """Generate clean contour plots."""
    print("Generating clean contour plots...")
    create_clean_contour_comparison()
    create_side_by_side_comparison()
    print("\n✅ All clean contour plots generated successfully!")

if __name__ == "__main__":
    main()