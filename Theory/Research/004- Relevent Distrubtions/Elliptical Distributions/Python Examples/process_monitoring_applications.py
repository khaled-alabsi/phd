"""
Process Monitoring Applications with Elliptical Distributions

This module demonstrates the application of elliptical distributions in 
process monitoring, including anomaly detection and classification.

Author: PhD Project
Date: September 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
import seaborn as sns
from elliptical_distributions import (MultivariateNormal, MultivariateT, 
                                    MultivariateLaplace, RobustEstimators)

class EllipticalAnomalyDetector:
    """
    Anomaly detection using elliptical distributions.
    """
    
    def __init__(self, distribution_type='normal', **kwargs):
        """
        Initialize anomaly detector.
        
        Parameters:
        -----------
        distribution_type : str
            Type of distribution ('normal', 't', 'laplace')
        **kwargs : dict
            Distribution-specific parameters
        """
        self.distribution_type = distribution_type
        self.kwargs = kwargs
        self.distribution = None
        self.threshold = None
        
    def fit(self, X_normal, contamination=0.1):
        """
        Fit the distribution to normal operating data.
        
        Parameters:
        -----------
        X_normal : array, shape (n, p)
            Normal operating condition data
        contamination : float
            Expected fraction of anomalies (for threshold setting)
        """
        # Estimate parameters
        if self.distribution_type == 'normal':
            # Use robust estimation if specified
            if self.kwargs.get('robust', False):
                mu_est, Sigma_est = RobustEstimators.mcd_estimator(X_normal)
                self.distribution = MultivariateNormal(mu_est, Sigma_est)
            else:
                self.distribution = MultivariateNormal(
                    np.mean(X_normal, axis=0), np.cov(X_normal.T))
                
        elif self.distribution_type == 't':
            nu = self.kwargs.get('nu', 4)
            mu_est = np.mean(X_normal, axis=0)
            Sigma_est = np.cov(X_normal.T)
            self.distribution = MultivariateT(mu_est, Sigma_est, nu)
            # Fit using EM if requested
            if self.kwargs.get('fit_em', False):
                self.distribution.fit_em(X_normal)
                
        elif self.distribution_type == 'laplace':
            mu_est = np.mean(X_normal, axis=0)
            Sigma_est = np.cov(X_normal.T)
            self.distribution = MultivariateLaplace(mu_est, Sigma_est)
        
        # Set threshold based on contamination level
        distances = self.distribution.mahalanobis_distance(X_normal)
        self.threshold = np.percentile(distances, 100 * (1 - contamination))
        
        return self
    
    def predict(self, X):
        """
        Predict anomalies in new data.
        
        Parameters:
        -----------
        X : array, shape (n, p)
            Test data
            
        Returns:
        --------
        y_pred : array, shape (n,)
            Binary predictions (1 = anomaly, 0 = normal)
        scores : array, shape (n,)
            Anomaly scores (Mahalanobis distances)
        """
        scores = self.distribution.mahalanobis_distance(X)
        y_pred = (scores > self.threshold).astype(int)
        
        return y_pred, scores
    
    def decision_function(self, X):
        """Return anomaly scores."""
        return self.distribution.mahalanobis_distance(X)


class EllipticalClassifier:
    """
    Classification using elliptical distributions (generative model).
    """
    
    def __init__(self, distribution_type='normal', **kwargs):
        """
        Initialize classifier.
        
        Parameters:
        -----------
        distribution_type : str
            Type of distribution for each class
        **kwargs : dict
            Distribution-specific parameters
        """
        self.distribution_type = distribution_type
        self.kwargs = kwargs
        self.class_distributions = {}
        self.class_priors = {}
        self.classes_ = None
        
    def fit(self, X, y):
        """
        Fit class-conditional distributions.
        
        Parameters:
        -----------
        X : array, shape (n, p)
            Training features
        y : array, shape (n,)
            Training labels
        """
        self.classes_ = np.unique(y)
        
        for cls in self.classes_:
            X_cls = X[y == cls]
            n_cls = len(X_cls)
            
            # Estimate class prior
            self.class_priors[cls] = n_cls / len(X)
            
            # Fit distribution for this class
            if self.distribution_type == 'normal':
                if self.kwargs.get('robust', False):
                    mu_est, Sigma_est = RobustEstimators.mcd_estimator(X_cls)
                    self.class_distributions[cls] = MultivariateNormal(mu_est, Sigma_est)
                else:
                    self.class_distributions[cls] = MultivariateNormal(
                        np.mean(X_cls, axis=0), np.cov(X_cls.T))
                    
            elif self.distribution_type == 't':
                nu = self.kwargs.get('nu', 4)
                mu_est = np.mean(X_cls, axis=0)
                Sigma_est = np.cov(X_cls.T)
                dist = MultivariateT(mu_est, Sigma_est, nu)
                if self.kwargs.get('fit_em', False):
                    dist.fit_em(X_cls)
                self.class_distributions[cls] = dist
                
            elif self.distribution_type == 'laplace':
                mu_est = np.mean(X_cls, axis=0)
                Sigma_est = np.cov(X_cls.T)
                self.class_distributions[cls] = MultivariateLaplace(mu_est, Sigma_est)
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : array, shape (n, p)
            Test features
            
        Returns:
        --------
        probas : array, shape (n, n_classes)
            Class probabilities
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        probas = np.zeros((n_samples, n_classes))
        
        for i, cls in enumerate(self.classes_):
            log_likelihood = self.class_distributions[cls].log_pdf(X)
            log_prior = np.log(self.class_priors[cls])
            probas[:, i] = log_likelihood + log_prior
        
        # Convert to probabilities using softmax
        probas = probas - np.max(probas, axis=1, keepdims=True)  # For numerical stability
        probas = np.exp(probas)
        probas = probas / np.sum(probas, axis=1, keepdims=True)
        
        return probas
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : array, shape (n, p)
            Test features
            
        Returns:
        --------
        y_pred : array, shape (n,)
            Predicted class labels
        """
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


def generate_process_data(n_normal=500, n_fault=100, n_features=4, fault_types=2):
    """
    Generate synthetic process monitoring data.
    
    Parameters:
    -----------
    n_normal : int
        Number of normal operating samples
    n_fault : int
        Number of fault samples per fault type
    n_features : int
        Number of process variables
    fault_types : int
        Number of different fault types
        
    Returns:
    --------
    X_normal : array, shape (n_normal, n_features)
        Normal operating data
    X_fault : array, shape (n_fault * fault_types, n_features)
        Fault data
    y_fault : array, shape (n_fault * fault_types,)
        Fault type labels
    """
    np.random.seed(42)
    
    # Normal operating conditions (multivariate normal)
    mu_normal = np.zeros(n_features)
    Sigma_normal = np.eye(n_features)
    # Add some correlation
    for i in range(n_features-1):
        Sigma_normal[i, i+1] = Sigma_normal[i+1, i] = 0.3
    
    normal_dist = MultivariateNormal(mu_normal, Sigma_normal)
    X_normal = normal_dist.sample(n_normal)
    
    # Generate different fault types
    X_fault_list = []
    y_fault_list = []
    
    for fault_id in range(fault_types):
        if fault_id == 0:
            # Fault type 1: Mean shift in first variable
            mu_fault = mu_normal.copy()
            mu_fault[0] += 2.0
            fault_dist = MultivariateNormal(mu_fault, Sigma_normal)
            
        elif fault_id == 1:
            # Fault type 2: Variance increase (heavy tails)
            fault_dist = MultivariateT(mu_normal, 2 * Sigma_normal, nu=3)
            
        else:
            # Additional fault types: Random mean shifts
            mu_fault = mu_normal + np.random.randn(n_features) * 1.5
            fault_dist = MultivariateNormal(mu_fault, Sigma_normal)
        
        X_fault_cls = fault_dist.sample(n_fault)
        X_fault_list.append(X_fault_cls)
        y_fault_list.append(np.full(n_fault, fault_id))
    
    X_fault = np.vstack(X_fault_list)
    y_fault = np.hstack(y_fault_list)
    
    return X_normal, X_fault, y_fault


def anomaly_detection_demo():
    """
    Demonstrate anomaly detection with different elliptical distributions.
    """
    print("Anomaly Detection Demo")
    print("="*30)
    
    # Generate data
    X_normal, X_fault, y_fault = generate_process_data()
    
    # Combine for testing
    X_test = np.vstack([X_normal[:100], X_fault])  # Use some normal data for testing
    y_true = np.hstack([np.zeros(100), np.ones(len(X_fault))])
    
    # Test different detectors
    detectors = {
        'Normal (Classical)': EllipticalAnomalyDetector('normal'),
        'Normal (Robust)': EllipticalAnomalyDetector('normal', robust=True),
        'Student-t': EllipticalAnomalyDetector('t', nu=4, fit_em=True),
        'Laplace': EllipticalAnomalyDetector('laplace')
    }
    
    results = {}
    
    for name, detector in detectors.items():
        print(f"\nTraining {name} detector...")
        
        # Fit on normal data
        detector.fit(X_normal[100:], contamination=0.05)  # Use remaining normal data
        
        # Predict on test data
        y_pred, scores = detector.predict(X_test)
        
        # Compute metrics
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary')
        auc_score = roc_auc_score(y_true, scores)
        
        results[name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score,
            'y_pred': y_pred,
            'scores': scores
        }
        
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-score: {f1:.3f}")
        print(f"  AUC: {auc_score:.3f}")
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_true, result['scores'])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {result['auc']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Anomaly Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('anomaly_detection_roc.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


def classification_demo():
    """
    Demonstrate classification with elliptical distributions.
    """
    print("\nClassification Demo")
    print("="*30)
    
    # Generate data
    X_normal, X_fault, y_fault = generate_process_data(fault_types=3)
    
    # Create classification dataset
    X = np.vstack([X_normal, X_fault])
    y = np.hstack([np.full(len(X_normal), -1), y_fault])  # -1 for normal, 0,1,2 for faults
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Test different classifiers
    classifiers = {
        'Normal (Classical)': EllipticalClassifier('normal'),
        'Normal (Robust)': EllipticalClassifier('normal', robust=True),
        'Student-t': EllipticalClassifier('t', nu=4, fit_em=True),
        'Laplace': EllipticalClassifier('laplace')
    }
    
    results = {}
    
    for name, classifier in classifiers.items():
        print(f"\nTraining {name} classifier...")
        
        # Fit classifier
        classifier.fit(X_train, y_train)
        
        # Predict
        y_pred = classifier.predict(X_test)
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, classification_report
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'y_pred': y_pred,
            'classifier': classifier
        }
        
        print(f"  Accuracy: {accuracy:.3f}")
        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Normal', 'Fault 1', 'Fault 2', 'Fault 3']))
    
    # Plot confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
        axes[i].set_title(f'{name}\nAccuracy: {result["accuracy"]:.3f}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('classification_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


def process_monitoring_comparison():
    """
    Compare Gaussian vs. robust elliptical approaches in process monitoring.
    """
    print("\nProcess Monitoring Comparison: Gaussian vs. Robust")
    print("="*50)
    
    # Generate data with contamination
    np.random.seed(42)
    
    # Normal process data
    X_normal_clean = MultivariateNormal([0, 0], [[1, 0.3], [0.3, 1]]).sample(400)
    
    # Add contamination (sensor errors, outliers)
    n_contaminated = 50
    outliers = np.random.randn(n_contaminated, 2) * 3 + [2, -2]
    X_normal = np.vstack([X_normal_clean, outliers])
    
    # Fault data (mean shift)
    X_fault = MultivariateNormal([2, 1], [[1, 0.3], [0.3, 1]]).sample(200)
    
    # Test data
    X_test = np.vstack([X_normal_clean[:50], X_fault[:50]])
    y_true = np.hstack([np.zeros(50), np.ones(50)])
    
    # Compare detectors
    detectors = {
        'Gaussian (Standard)': EllipticalAnomalyDetector('normal'),
        'Gaussian (Robust MCD)': EllipticalAnomalyDetector('normal', robust=True),
        'Student-t (ν=4)': EllipticalAnomalyDetector('t', nu=4),
        'Student-t (ν=2, Heavy Tails)': EllipticalAnomalyDetector('t', nu=2)
    }
    
    # Fit and evaluate
    results = {}
    
    for name, detector in detectors.items():
        detector.fit(X_normal, contamination=0.1)
        y_pred, scores = detector.predict(X_test)
        
        from sklearn.metrics import f1_score, roc_auc_score
        f1 = f1_score(y_true, y_pred)
        auc_score = roc_auc_score(y_true, scores)
        
        results[name] = {'f1': f1, 'auc': auc_score}
        
        print(f"{name:25s} F1: {f1:.3f}, AUC: {auc_score:.3f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, (name, detector) in enumerate(detectors.items()):
        ax = axes[i]
        
        # Plot training data
        ax.scatter(X_normal_clean[:, 0], X_normal_clean[:, 1], 
                  alpha=0.6, c='blue', s=20, label='Normal (clean)')
        ax.scatter(outliers[:, 0], outliers[:, 1], 
                  alpha=0.8, c='orange', s=30, label='Contamination')
        
        # Plot test data
        ax.scatter(X_test[y_true==0, 0], X_test[y_true==0, 1], 
                  alpha=0.8, c='green', s=40, marker='s', label='Test normal')
        ax.scatter(X_test[y_true==1, 0], X_test[y_true==1, 1], 
                  alpha=0.8, c='red', s=40, marker='^', label='Test fault')
        
        # Plot decision boundary (contour)
        detector.fit(X_normal, contamination=0.1)
        
        x_range = np.linspace(-4, 5, 50)
        y_range = np.linspace(-4, 4, 50)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
        
        distances = detector.decision_function(grid_points)
        Z = distances.reshape(X_grid.shape)
        
        ax.contour(X_grid, Y_grid, Z, levels=[detector.threshold], 
                  colors=['black'], linestyles=['--'], linewidths=2)
        
        ax.set_title(f'{name}\nF1: {results[name]["f1"]:.3f}, AUC: {results[name]["auc"]:.3f}')
        ax.set_xlabel('Variable 1')
        ax.set_ylabel('Variable 2')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('process_monitoring_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


if __name__ == "__main__":
    print("Process Monitoring Applications with Elliptical Distributions")
    print("="*65)
    
    # Set random seed
    np.random.seed(42)
    
    # Run demonstrations
    print("1. Anomaly Detection Demo")
    anomaly_results = anomaly_detection_demo()
    
    print("\n" + "="*65)
    print("2. Classification Demo")
    classification_results = classification_demo()
    
    print("\n" + "="*65)
    print("3. Process Monitoring Comparison")
    comparison_results = process_monitoring_comparison()
    
    print("\n" + "="*65)
    print("Demo complete! Generated plots:")
    print("- anomaly_detection_roc.png")
    print("- classification_confusion_matrices.png") 
    print("- process_monitoring_comparison.png")
    
    print("\nKey Insights:")
    print("1. Robust estimators handle contamination better")
    print("2. Student-t distributions are more robust to outliers")
    print("3. Choice of distribution affects detection performance")
    print("4. Elliptical distributions provide flexible alternatives to Gaussian assumptions")