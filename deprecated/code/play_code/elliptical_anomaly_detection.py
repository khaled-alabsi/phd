import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal, chi2
from scipy.linalg import sqrtm
import pandas as pd
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EllipticalDistribution:
    """
    Class for generating and working with 2D elliptical distributions
    """
    
    def __init__(self, mean: np.ndarray = None, cov: np.ndarray = None, 
                 correlation: float = 0.0, std_x: float = 1.0, std_y: float = 1.0,
                 rotation_angle: float = 0.0):
        """
        Initialize elliptical distribution
        
        Parameters:
        -----------
        mean : array-like, shape (2,)
            Mean of the distribution
        cov : array-like, shape (2, 2)
            Covariance matrix
        correlation : float
            Correlation coefficient (-1 to 1)
        std_x, std_y : float
            Standard deviations for X and Y
        rotation_angle : float
            Rotation angle in radians
        """
        if mean is None:
            self.mean = np.array([0.0, 0.0])
        else:
            self.mean = np.array(mean)
            
        if cov is None:
            # Create covariance matrix from correlation and standard deviations
            self.cov = np.array([[std_x**2, correlation * std_x * std_y],
                                [correlation * std_x * std_y, std_y**2]])
        else:
            self.cov = np.array(cov)
            
        # Apply rotation if specified
        if rotation_angle != 0:
            self._apply_rotation(rotation_angle)
            
        self.distribution = multivariate_normal(self.mean, self.cov)
    
    def _apply_rotation(self, angle: float):
        """Apply rotation to the covariance matrix"""
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                   [np.sin(angle), np.cos(angle)]])
        # Transform covariance matrix
        self.cov = rotation_matrix @ self.cov @ rotation_matrix.T
    
    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples from the elliptical distribution"""
        return self.distribution.rvs(n_samples)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Calculate probability density function"""
        return self.distribution.pdf(x)
    
    def mahalanobis_distance(self, x: np.ndarray) -> np.ndarray:
        """Calculate Mahalanobis distance from center"""
        x = np.atleast_2d(x)
        diff = x - self.mean
        inv_cov = np.linalg.inv(self.cov)
        return np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))


class EllipticalEWMA:
    """
    Elliptical EWMA for multivariate anomaly detection
    """
    
    def __init__(self, lambda_ewma: float = 0.1, alpha: float = 0.01):
        """
        Initialize Elliptical EWMA
        
        Parameters:
        -----------
        lambda_ewma : float
            EWMA smoothing parameter (0 < lambda <= 1)
        alpha : float
            Significance level for control limits
        """
        self.lambda_ewma = lambda_ewma
        self.alpha = alpha
        self.mean_est = None
        self.cov_est = None
        self.ewma_values = []
        self.control_limit = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, initial_samples: int = 50):
        """
        Fit the EWMA model using initial data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, 2)
            Training data
        initial_samples : int
            Number of initial samples to estimate parameters
        """
        X = np.array(X)
        if X.shape[1] != 2:
            raise ValueError("Input data must be 2-dimensional")
        
        # Use initial samples to estimate mean and covariance
        initial_data = X[:initial_samples]
        self.mean_est = np.mean(initial_data, axis=0)
        self.cov_est = np.cov(initial_data.T)
        
        # Ensure covariance matrix is positive definite
        eigenvals = np.linalg.eigvals(self.cov_est)
        if np.any(eigenvals <= 0):
            self.cov_est += np.eye(2) * 1e-6
        
        # Calculate control limit based on chi-square distribution
        # For bivariate case with EWMA
        h = self.lambda_ewma / (2 - self.lambda_ewma)
        self.control_limit = h * chi2.ppf(1 - self.alpha, df=2)
        
        self.is_fitted = True
    
    def _mahalanobis_distance(self, x: np.ndarray) -> float:
        """Calculate Mahalanobis distance"""
        diff = x - self.mean_est
        try:
            inv_cov = np.linalg.inv(self.cov_est)
            return np.sqrt(diff @ inv_cov @ diff)
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudo-inverse
            inv_cov = np.linalg.pinv(self.cov_est)
            return np.sqrt(diff @ inv_cov @ diff)
    
    def update(self, x: np.ndarray) -> Tuple[float, bool]:
        """
        Update EWMA and detect anomaly
        
        Parameters:
        -----------
        x : array-like, shape (2,)
            New observation
        
        Returns:
        --------
        ewma_value : float
            Current EWMA value
        is_anomaly : bool
            Whether the observation is an anomaly
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before updating")
        
        x = np.array(x)
        
        # Calculate Mahalanobis distance for current observation
        mahal_dist = self._mahalanobis_distance(x)
        
        # Update EWMA
        if len(self.ewma_values) == 0:
            # First observation
            ewma_value = mahal_dist**2
        else:
            # EWMA update
            ewma_value = (self.lambda_ewma * mahal_dist**2 + 
                         (1 - self.lambda_ewma) * self.ewma_values[-1])
        
        self.ewma_values.append(ewma_value)
        
        # Check for anomaly
        is_anomaly = ewma_value > self.control_limit
        
        return ewma_value, is_anomaly
    
    def predict(self, X: np.ndarray) -> Tuple[List[float], List[bool]]:
        """
        Predict anomalies for a batch of observations
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, 2)
            Observations to evaluate
        
        Returns:
        --------
        ewma_values : list
            EWMA values for each observation
        anomalies : list
            Boolean indicators for anomalies
        """
        X = np.array(X)
        ewma_vals = []
        anomalies = []
        
        for x in X:
            ewma_val, is_anomaly = self.update(x)
            ewma_vals.append(ewma_val)
            anomalies.append(is_anomaly)
        
        return ewma_vals, anomalies


def generate_elliptical_data(n_samples: int = 1000, 
                           correlation: float = 0.0,
                           std_x: float = 1.0, 
                           std_y: float = 1.0,
                           rotation_angle: float = 0.0,
                           mean: np.ndarray = None) -> np.ndarray:
    """
    Generate 2D elliptical distribution data
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    correlation : float
        Correlation coefficient
    std_x, std_y : float
        Standard deviations
    rotation_angle : float
        Rotation angle in radians
    mean : array-like
        Mean of the distribution
    
    Returns:
    --------
    data : ndarray, shape (n_samples, 2)
        Generated samples
    """
    if mean is None:
        mean = np.array([0.0, 0.0])
    
    elliptical_dist = EllipticalDistribution(
        mean=mean, 
        correlation=correlation,
        std_x=std_x, 
        std_y=std_y,
        rotation_angle=rotation_angle
    )
    
    return elliptical_dist.sample(n_samples)


def inject_anomalies(data: np.ndarray, 
                    anomaly_fraction: float = 0.05,
                    anomaly_type: str = 'outlier') -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject anomalies into normal data
    
    Parameters:
    -----------
    data : ndarray
        Normal data
    anomaly_fraction : float
        Fraction of data to make anomalous
    anomaly_type : str
        Type of anomaly ('outlier', 'shift', 'clustered')
    
    Returns:
    --------
    contaminated_data : ndarray
        Data with anomalies
    anomaly_labels : ndarray
        Boolean array indicating anomalies
    """
    n_samples = len(data)
    n_anomalies = int(n_samples * anomaly_fraction)
    
    # Create copy of data
    contaminated_data = data.copy()
    anomaly_labels = np.zeros(n_samples, dtype=bool)
    
    # Select random indices for anomalies
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    anomaly_labels[anomaly_indices] = True
    
    if anomaly_type == 'outlier':
        # Generate outliers far from the center
        for idx in anomaly_indices:
            # Random direction
            angle = np.random.uniform(0, 2*np.pi)
            # Large distance
            distance = np.random.uniform(4, 8)
            contaminated_data[idx] = np.array([distance * np.cos(angle), 
                                             distance * np.sin(angle)])
    
    elif anomaly_type == 'shift':
        # Shift anomalies to a different region
        shift = np.array([5, 5])
        for idx in anomaly_indices:
            contaminated_data[idx] += shift + np.random.normal(0, 0.5, 2)
    
    elif anomaly_type == 'clustered':
        # Create a cluster of anomalies
        cluster_center = np.array([4, -3])
        for idx in anomaly_indices:
            contaminated_data[idx] = cluster_center + np.random.normal(0, 0.3, 2)
    
    return contaminated_data, anomaly_labels


def plot_elliptical_distributions():
    """Plot various elliptical distributions"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Elliptical Distributions with Different Parameters', fontsize=16)
    
    # Different parameter combinations
    params = [
        {'correlation': 0.0, 'std_x': 1.0, 'std_y': 1.0, 'rotation_angle': 0.0, 'title': 'Circular (ρ=0)'},
        {'correlation': 0.7, 'std_x': 1.0, 'std_y': 1.0, 'rotation_angle': 0.0, 'title': 'Elliptical (ρ=0.7)'},
        {'correlation': 0.0, 'std_x': 2.0, 'std_y': 0.5, 'rotation_angle': 0.0, 'title': 'Stretched (σx=2, σy=0.5)'},
        {'correlation': 0.5, 'std_x': 1.0, 'std_y': 1.0, 'rotation_angle': np.pi/4, 'title': 'Rotated 45°'},
        {'correlation': -0.8, 'std_x': 1.0, 'std_y': 1.0, 'rotation_angle': 0.0, 'title': 'Negative correlation'},
        {'correlation': 0.3, 'std_x': 1.5, 'std_y': 0.8, 'rotation_angle': np.pi/6, 'title': 'Complex ellipse'}
    ]
    
    for i, param in enumerate(params):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Generate data
        data = generate_elliptical_data(n_samples=1000, **{k: v for k, v in param.items() if k != 'title'})
        
        # Plot data
        ax.scatter(data[:, 0], data[:, 1], alpha=0.6, s=20)
        
        # Plot ellipse boundary (2-sigma)
        elliptical_dist = EllipticalDistribution(**{k: v for k, v in param.items() if k != 'title'})
        plot_confidence_ellipse(ax, elliptical_dist.mean, elliptical_dist.cov, n_std=2, 
                               color='red', alpha=0.3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(param['title'])
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
    
    plt.tight_layout()
    plt.show()


def plot_confidence_ellipse(ax, mean, cov, n_std=2, **kwargs):
    """Plot confidence ellipse"""
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    order = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
    
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigenvals)
    
    from matplotlib.patches import Ellipse
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)


def plot_control_chart(ewma_values: List[float], 
                      control_limit: float,
                      anomalies: List[bool],
                      title: str = "Elliptical EWMA Control Chart"):
    """
    Plot EWMA control chart
    
    Parameters:
    -----------
    ewma_values : list
        EWMA values over time
    control_limit : float
        Upper control limit
    anomalies : list
        Boolean indicators for anomalies
    title : str
        Chart title
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Time points
    time_points = range(len(ewma_values))
    
    # Plot EWMA values
    ax.plot(time_points, ewma_values, 'b-', linewidth=2, label='EWMA', alpha=0.8)
    
    # Plot control limit
    ax.axhline(y=control_limit, color='red', linestyle='--', linewidth=2, 
               label=f'UCL = {control_limit:.3f}')
    
    # Highlight anomalies
    anomaly_indices = [i for i, is_anom in enumerate(anomalies) if is_anom]
    if anomaly_indices:
        ax.scatter([time_points[i] for i in anomaly_indices],
                  [ewma_values[i] for i in anomaly_indices],
                  color='red', s=100, marker='o', alpha=0.8, 
                  label=f'Anomalies ({len(anomaly_indices)})', zorder=5)
    
    # Add shaded region above UCL
    ax.fill_between(time_points, control_limit, max(ewma_values)*1.1, 
                   alpha=0.2, color='red', label='Anomaly Region')
    
    ax.set_xlabel('Time Index')
    ax.set_ylabel('EWMA Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    n_anomalies = sum(anomalies)
    stats_text = f'Total Observations: {len(ewma_values)}\n'
    stats_text += f'Detected Anomalies: {n_anomalies}\n'
    stats_text += f'Anomaly Rate: {n_anomalies/len(ewma_values)*100:.2f}%'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def comprehensive_anomaly_detection_example():
    """
    Comprehensive example of elliptical EWMA anomaly detection
    """
    print("=" * 80)
    print("ELLIPTICAL EWMA ANOMALY DETECTION EXAMPLE")
    print("=" * 80)
    
    # Parameters
    n_normal = 500
    n_test = 200
    correlation = 0.6
    std_x, std_y = 1.2, 0.8
    rotation_angle = np.pi / 6  # 30 degrees
    
    print(f"Generating elliptical distribution:")
    print(f"- Correlation: {correlation}")
    print(f"- Standard deviations: σx={std_x}, σy={std_y}")
    print(f"- Rotation angle: {np.degrees(rotation_angle):.1f}°")
    print(f"- Normal samples: {n_normal}")
    print(f"- Test samples: {n_test}")
    
    # Generate normal training data
    normal_data = generate_elliptical_data(
        n_samples=n_normal,
        correlation=correlation,
        std_x=std_x,
        std_y=std_y,
        rotation_angle=rotation_angle
    )
    
    # Generate test data with anomalies
    test_data = generate_elliptical_data(
        n_samples=n_test,
        correlation=correlation,
        std_x=std_x,
        std_y=std_y,
        rotation_angle=rotation_angle
    )
    
    # Inject anomalies
    contaminated_data, true_anomalies = inject_anomalies(
        test_data, anomaly_fraction=0.1, anomaly_type='outlier'
    )
    
    # Initialize and fit EWMA
    ewma_detector = EllipticalEWMA(lambda_ewma=0.1, alpha=0.01)
    ewma_detector.fit(normal_data, initial_samples=50)
    
    print(f"\nEWMA Parameters:")
    print(f"- Lambda (smoothing): {ewma_detector.lambda_ewma}")
    print(f"- Alpha (significance): {ewma_detector.alpha}")
    print(f"- Control limit: {ewma_detector.control_limit:.3f}")
    
    # Detect anomalies
    ewma_values, detected_anomalies = ewma_detector.predict(contaminated_data)
    
    # Calculate performance metrics
    true_positives = sum(ta and da for ta, da in zip(true_anomalies, detected_anomalies))
    false_positives = sum(not ta and da for ta, da in zip(true_anomalies, detected_anomalies))
    false_negatives = sum(ta and not da for ta, da in zip(true_anomalies, detected_anomalies))
    true_negatives = sum(not ta and not da for ta, da in zip(true_anomalies, detected_anomalies))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nPerformance Metrics:")
    print(f"- True Positives: {true_positives}")
    print(f"- False Positives: {false_positives}")
    print(f"- False Negatives: {false_negatives}")
    print(f"- True Negatives: {true_negatives}")
    print(f"- Precision: {precision:.3f}")
    print(f"- Recall: {recall:.3f}")
    print(f"- F1-Score: {f1_score:.3f}")
    
    # Create visualizations
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Original elliptical distribution
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(normal_data[:, 0], normal_data[:, 1], alpha=0.6, s=20, label='Normal data')
    elliptical_dist = EllipticalDistribution(correlation=correlation, std_x=std_x, 
                                           std_y=std_y, rotation_angle=rotation_angle)
    plot_confidence_ellipse(ax1, elliptical_dist.mean, elliptical_dist.cov, 
                           n_std=2, color='blue', alpha=0.3, label='2σ ellipse')
    plot_confidence_ellipse(ax1, elliptical_dist.mean, elliptical_dist.cov, 
                           n_std=3, color='red', alpha=0.2, label='3σ ellipse')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Training Data Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Test data with anomalies
    ax2 = plt.subplot(2, 3, 2)
    normal_mask = ~np.array(true_anomalies)
    anomaly_mask = np.array(true_anomalies)
    
    ax2.scatter(contaminated_data[normal_mask, 0], contaminated_data[normal_mask, 1], 
               alpha=0.6, s=20, color='blue', label='Normal')
    ax2.scatter(contaminated_data[anomaly_mask, 0], contaminated_data[anomaly_mask, 1], 
               alpha=0.8, s=60, color='red', marker='x', label='True anomalies')
    plot_confidence_ellipse(ax2, elliptical_dist.mean, elliptical_dist.cov, 
                           n_std=2, color='blue', alpha=0.3)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Test Data with True Anomalies')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Plot 3: Detection results
    ax3 = plt.subplot(2, 3, 3)
    detected_mask = np.array(detected_anomalies)
    normal_detected = ~detected_mask
    
    ax3.scatter(contaminated_data[normal_detected, 0], contaminated_data[normal_detected, 1], 
               alpha=0.6, s=20, color='green', label='Normal (detected)')
    ax3.scatter(contaminated_data[detected_mask, 0], contaminated_data[detected_mask, 1], 
               alpha=0.8, s=60, color='red', marker='o', label='Detected anomalies')
    plot_confidence_ellipse(ax3, elliptical_dist.mean, elliptical_dist.cov, 
                           n_std=2, color='blue', alpha=0.3)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('EWMA Detection Results')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # Plot 4: Control Chart
    ax4 = plt.subplot(2, 1, 2)
    time_points = range(len(ewma_values))
    
    ax4.plot(time_points, ewma_values, 'b-', linewidth=2, label='EWMA', alpha=0.8)
    ax4.axhline(y=ewma_detector.control_limit, color='red', linestyle='--', 
               linewidth=2, label=f'UCL = {ewma_detector.control_limit:.3f}')
    
    # Highlight true and detected anomalies
    true_anomaly_indices = [i for i, is_anom in enumerate(true_anomalies) if is_anom]
    detected_anomaly_indices = [i for i, is_anom in enumerate(detected_anomalies) if is_anom]
    
    if true_anomaly_indices:
        ax4.scatter([time_points[i] for i in true_anomaly_indices],
                   [ewma_values[i] for i in true_anomaly_indices],
                   color='orange', s=100, marker='x', alpha=0.8, 
                   label=f'True anomalies ({len(true_anomaly_indices)})', zorder=5)
    
    if detected_anomaly_indices:
        ax4.scatter([time_points[i] for i in detected_anomaly_indices],
                   [ewma_values[i] for i in detected_anomaly_indices],
                   color='red', s=80, marker='o', alpha=0.6, 
                   label=f'Detected anomalies ({len(detected_anomaly_indices)})', zorder=4)
    
    ax4.fill_between(time_points, ewma_detector.control_limit, 
                    max(ewma_values)*1.1 if ewma_values else 1, 
                    alpha=0.2, color='red', label='Anomaly Region')
    
    ax4.set_xlabel('Time Index')
    ax4.set_ylabel('EWMA Value')
    ax4.set_title('Elliptical EWMA Control Chart')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'ewma_detector': ewma_detector,
        'ewma_values': ewma_values,
        'detected_anomalies': detected_anomalies,
        'true_anomalies': true_anomalies,
        'performance': {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives
        }
    }


def main():
    """Run all functions"""
    print("Generating elliptical distribution plots and anomaly detection examples...")
    
    # 1. Plot various elliptical distributions
    print("\n1. Plotting elliptical distributions with different parameters")
    plot_elliptical_distributions()
    
    # 2. Comprehensive anomaly detection example
    print("\n2. Running comprehensive anomaly detection example")
    results = comprehensive_anomaly_detection_example()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nKey Features Demonstrated:")
    print("- Elliptical distribution generation with various parameters")
    print("- EWMA-based anomaly detection for multivariate data")
    print("- Control chart visualization with anomaly highlighting")
    print("- Performance evaluation with precision, recall, and F1-score")
    print("- Confidence ellipse visualization")
    
    return results


if __name__ == "__main__":
    results = main()