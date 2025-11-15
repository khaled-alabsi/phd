import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal, norm, chi2
from scipy.linalg import sqrtm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def generate_symmetric_data(n_samples=1000, correlation=0.0):
    """
    Generate symmetric bivariate normal distribution data
    
    Parameters:
    n_samples: number of samples
    correlation: correlation coefficient between variables (-1 to 1)
    """
    # Mean vector (centered at origin for symmetry)
    mean = [0, 0]
    
    # Covariance matrix
    cov = [[1, correlation], 
           [correlation, 1]]
    
    # Generate samples
    data = np.random.multivariate_normal(mean, cov, n_samples)
    return data

def plot_2d_distributions():
    """Plot various 2D symmetric distributions"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Symmetric Bivariate Distributions', fontsize=16)
    
    correlations = [0.0, 0.5, 0.8, -0.5, -0.8, 0.9]
    titles = ['Independent (ρ=0)', 'Positive correlation (ρ=0.5)', 'Strong positive (ρ=0.8)',
              'Negative correlation (ρ=-0.5)', 'Strong negative (ρ=-0.8)', 'Very strong (ρ=0.9)']
    
    for i, (corr, title) in enumerate(zip(correlations, titles)):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Generate data
        data = generate_symmetric_data(1000, corr)
        x, y = data[:, 0], data[:, 1]
        
        # Scatter plot with density
        ax.hexbin(x, y, gridsize=20, cmap='Blues', alpha=0.7)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xlabel('Variable X')
        ax.set_ylabel('Variable Y')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add correlation coefficient text
        ax.text(0.05, 0.95, f'r = {corr}', transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def plot_contour_distributions():
    """Plot contour plots of symmetric distributions"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Contour Plots of Symmetric Bivariate Normal Distributions', fontsize=14)
    
    correlations = [0.0, 0.7, -0.7]
    titles = ['Independent (ρ=0)', 'Positive correlation (ρ=0.7)', 'Negative correlation (ρ=-0.7)']
    
    # Create grid for contour plots
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    
    for i, (corr, title) in enumerate(zip(correlations, titles)):
        ax = axes[i]
        
        # Define multivariate normal distribution
        mean = [0, 0]
        cov = [[1, corr], [corr, 1]]
        rv = multivariate_normal(mean, cov)
        
        # Calculate probability density
        pos = np.dstack((X, Y))
        Z = rv.pdf(pos)
        
        # Create contour plot
        contour = ax.contour(X, Y, Z, levels=10, colors='blue', alpha=0.6)
        ax.contourf(X, Y, Z, levels=20, cmap='Blues', alpha=0.3)
        ax.clabel(contour, inline=True, fontsize=8)
        
        ax.set_xlabel('Variable X')
        ax.set_ylabel('Variable Y')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def plot_3d_distribution():
    """Plot 3D surface of symmetric bivariate distribution"""
    fig = plt.figure(figsize=(12, 5))
    
    # Create grid
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    
    correlations = [0.0, 0.8]
    titles = ['Independent Variables (ρ=0)', 'Correlated Variables (ρ=0.8)']
    
    for i, (corr, title) in enumerate(zip(correlations, titles)):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        
        # Define distribution
        mean = [0, 0]
        cov = [[1, corr], [corr, 1]]
        rv = multivariate_normal(mean, cov)
        
        # Calculate density
        pos = np.dstack((X, Y))
        Z = rv.pdf(pos)
        
        # Create 3D surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        
        ax.set_xlabel('Variable X')
        ax.set_ylabel('Variable Y')
        ax.set_zlabel('Probability Density')
        ax.set_title(title)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.show()

def plot_marginal_distributions():
    """Plot joint distribution with marginal distributions"""
    # Generate correlated data
    data = generate_symmetric_data(2000, 0.6)
    x, y = data[:, 0], data[:, 1]
    
    # Create figure with GridSpec for custom layout
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main scatter plot
    ax_main = fig.add_subplot(gs[1:, :-1])
    ax_main.hexbin(x, y, gridsize=25, cmap='Blues')
    ax_main.set_xlabel('Variable X')
    ax_main.set_ylabel('Variable Y')
    ax_main.set_title('Joint Distribution with Marginals')
    
    # Top marginal (X distribution)
    ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_top.hist(x, bins=30, alpha=0.7, color='skyblue', density=True)
    ax_top.plot(np.linspace(-4, 4, 100), norm.pdf(np.linspace(-4, 4, 100)), 'r-', label='Normal PDF')
    ax_top.set_ylabel('Density')
    ax_top.set_title('Marginal Distribution of X')
    ax_top.legend()
    
    # Right marginal (Y distribution)
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)
    ax_right.hist(y, bins=30, alpha=0.7, color='lightcoral', orientation='horizontal', density=True)
    y_range = np.linspace(-4, 4, 100)
    ax_right.plot(norm.pdf(y_range), y_range, 'r-', label='Normal PDF')
    ax_right.set_xlabel('Density')
    ax_right.set_title('Marginal Distribution of Y', rotation=270, labelpad=20)
    ax_right.legend()
    
    plt.show()

def plot_custom_symmetric_distributions():
    """Plot various types of symmetric distributions"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Different Types of Symmetric Distributions', fontsize=16)
    
    n_samples = 1000
    
    # 1. Circular symmetric (independent)
    ax1 = axes[0, 0]
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    r = np.random.normal(0, 1, n_samples)
    x1 = r * np.cos(theta)
    y1 = r * np.sin(theta)
    ax1.scatter(x1, y1, alpha=0.6, s=20)
    ax1.set_title('Circular Symmetric Distribution')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 2. Elliptical symmetric
    ax2 = axes[0, 1]
    data2 = generate_symmetric_data(n_samples, 0.8)
    x2, y2 = data2[:, 0], data2[:, 1]
    ax2.scatter(x2, y2, alpha=0.6, s=20, c='orange')
    ax2.set_title('Elliptical Symmetric Distribution')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # 3. Student's t distribution (heavy tails)
    ax3 = axes[1, 0]
    # Generate t-distributed data
    df = 3  # degrees of freedom
    x3 = np.random.standard_t(df, n_samples)
    y3 = np.random.standard_t(df, n_samples)
    ax3.scatter(x3, y3, alpha=0.6, s=20, c='green')
    ax3.set_title('Heavy-tailed Symmetric Distribution')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-6, 6)
    ax3.set_ylim(-6, 6)
    ax3.set_aspect('equal')
    
    # 4. Uniform on disk (perfectly circular)
    ax4 = axes[1, 1]
    theta4 = np.random.uniform(0, 2*np.pi, n_samples)
    r4 = np.sqrt(np.random.uniform(0, 1, n_samples)) * 2  # uniform in disk
    x4 = r4 * np.cos(theta4)
    y4 = r4 * np.sin(theta4)
    ax4.scatter(x4, y4, alpha=0.6, s=20, c='purple')
    ax4.set_title('Uniform Distribution on Disk')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def main():
    """Run all plotting functions"""
    print("Generating symmetric bivariate distribution plots...")
    
    # 1. Basic 2D distributions with different correlations
    print("1. Basic 2D distributions with different correlations")
    plot_2d_distributions()
    
    # 2. Contour plots
    print("2. Contour plots")
    plot_contour_distributions()
    
    # 3. 3D surface plots
    print("3. 3D surface plots")
    plot_3d_distribution()
    
    # 4. Joint with marginal distributions
    print("4. Joint distribution with marginals")
    plot_marginal_distributions()
    
    # 5. Various symmetric distributions
    print("5. Different types of symmetric distributions")
    plot_custom_symmetric_distributions()
    
    print("\nAll plots generated successfully!")
    print("\nZusammenfassung der erzeugten Plots:")
    print("- 2D Scatter plots mit verschiedenen Korrelationen")
    print("- Kontur-Plots für Wahrscheinlichkeitsdichten")
    print("- 3D Oberflächen-Plots")
    print("- Gemeinsame Verteilung mit Randverteilungen")
    print("- Verschiedene Arten symmetrischer Verteilungen")

if __name__ == "__main__":
    main()