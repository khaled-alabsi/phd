"""
Tennessee Eastman Process (TEP) Data Loader

This module handles data loading, preprocessing, and basic data exploration
for the TEP anomaly detection project.
"""

import os
import numpy as np
import pandas as pd
import pyreadr
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, Dict, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class TEPDataLoader:
    """Class for loading and preprocessing TEP data."""
    
    def __init__(self, data_dir: str = "data/"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing TEP data files
        """
        self.data_dir = data_dir
        self.training_data = None
        self.testing_data = None
        self.scaler = StandardScaler()
        self.pca = None
        self.label_encoder = LabelEncoder()
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and testing data from RData files.
        
        Returns:
            Tuple of (training_data, testing_data)
        """
        print("Loading TEP data...")
        
        # Load training data
        train_file = os.path.join(self.data_dir, "TEP_FaultFree_Training.RData")
        if os.path.exists(train_file):
            train_data = pyreadr.read_r(train_file)
            self.training_data = train_data[list(train_data.keys())[0]]
            print(f"Training data loaded: {self.training_data.shape}")
        else:
            raise FileNotFoundError(f"Training data file not found: {train_file}")
        
        # Load testing data
        test_file = os.path.join(self.data_dir, "TEP_FaultFree_Testing.RData")
        if os.path.exists(test_file):
            test_data = pyreadr.read_r(test_file)
            self.testing_data = test_data[list(test_data.keys())[0]]
            print(f"Testing data loaded: {self.testing_data.shape}")
        else:
            raise FileNotFoundError(f"Testing data file not found: {test_file}")
        
        return self.training_data, self.testing_data
    
    def preprocess_data(self, 
                       normalize: bool = True,
                       apply_pca: bool = False,
                       n_components: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess the loaded data.
        
        Args:
            normalize: Whether to normalize the data
            apply_pca: Whether to apply PCA dimensionality reduction
            n_components: Number of PCA components (if None, keep 95% variance)
            
        Returns:
            Tuple of (processed_training_data, processed_testing_data)
        """
        if self.training_data is None or self.testing_data is None:
            raise ValueError("Data must be loaded before preprocessing")
        
        print("Preprocessing data...")
        
        # Separate features and target
        feature_cols = [col for col in self.training_data.columns if col != 'faultNumber']
        target_col = 'faultNumber'
        
        X_train = self.training_data[feature_cols].copy()
        y_train = self.training_data[target_col].copy()
        X_test = self.testing_data[feature_cols].copy()
        y_test = self.testing_data[target_col].copy()
        
        # Normalize features
        if normalize:
            print("Normalizing features...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            X_train = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
            X_test = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
        
        # Apply PCA if requested
        if apply_pca:
            if n_components is None:
                # Keep components that explain 95% of variance
                n_components = 0.95
            
            print(f"Applying PCA with {n_components} components...")
            self.pca = PCA(n_components=n_components)
            X_train_pca = self.pca.fit_transform(X_train)
            X_test_pca = self.pca.transform(X_test)
            
            # Create new column names for PCA components
            if isinstance(n_components, float):
                n_comp = X_train_pca.shape[1]
            else:
                n_comp = n_components
            
            pca_cols = [f'PC_{i+1}' for i in range(n_comp)]
            
            X_train = pd.DataFrame(X_train_pca, columns=pca_cols, index=X_train.index)
            X_test = pd.DataFrame(X_test_pca, columns=pca_cols, index=X_test.index)
            
            print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        # Recombine features and target
        processed_train = pd.concat([X_train, y_train], axis=1)
        processed_test = pd.concat([X_test, y_test], axis=1)
        
        print("Data preprocessing completed")
        return processed_train, processed_test
    
    def explore_data(self) -> Dict:
        """
        Perform basic data exploration and return summary statistics.
        
        Returns:
            Dictionary containing data exploration results
        """
        if self.training_data is None:
            raise ValueError("Data must be loaded before exploration")
        
        print("Exploring data...")
        
        exploration_results = {}
        
        # Basic info
        exploration_results['shape'] = self.training_data.shape
        exploration_results['columns'] = list(self.training_data.columns)
        exploration_results['dtypes'] = self.training_data.dtypes.to_dict()
        
        # Missing values
        missing_values = self.training_data.isnull().sum()
        exploration_results['missing_values'] = missing_values[missing_values > 0].to_dict()
        
        # Target distribution
        if 'faultNumber' in self.training_data.columns:
            target_dist = self.training_data['faultNumber'].value_counts().sort_index()
            exploration_results['target_distribution'] = target_dist.to_dict()
            exploration_results['n_classes'] = len(target_dist)
        
        # Feature statistics
        feature_cols = [col for col in self.training_data.columns if col != 'faultNumber']
        if feature_cols:
            feature_stats = self.training_data[feature_cols].describe()
            exploration_results['feature_statistics'] = feature_stats.to_dict()
        
        return exploration_results
    
    def visualize_data_distribution(self, save_plots: bool = True) -> None:
        """
        Create visualizations for data distribution and structure.
        
        Args:
            save_plots: Whether to save the plots
        """
        if self.training_data is None:
            raise ValueError("Data must be loaded before visualization")
        
        print("Creating data visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Target distribution
        if 'faultNumber' in self.training_data.columns:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            target_counts = self.training_data['faultNumber'].value_counts().sort_index()
            plt.bar(target_counts.index, target_counts.values, alpha=0.7)
            plt.title('Fault Distribution in Training Data')
            plt.xlabel('Fault Number')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
            plt.subplot(1, 2, 2)
            plt.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%')
            plt.title('Fault Distribution (Percentage)')
            
            plt.tight_layout()
            if save_plots:
                plt.savefig('output/1.00/data_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 2. Feature correlation heatmap (sample of features if too many)
        feature_cols = [col for col in self.training_data.columns if col != 'faultNumber']
        if feature_cols:
            # Sample features if there are too many
            if len(feature_cols) > 50:
                sampled_features = np.random.choice(feature_cols, 50, replace=False)
                corr_data = self.training_data[sampled_features].corr()
                title_suffix = " (50 sampled features)"
            else:
                corr_data = self.training_data[feature_cols].corr()
                title_suffix = ""
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_data, cmap='coolwarm', center=0, 
                       square=True, cbar_kws={"shrink": .8})
            plt.title(f'Feature Correlation Matrix{title_suffix}')
            plt.tight_layout()
            if save_plots:
                plt.savefig('output/1.00/feature_correlation.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 3. Feature distributions (sample)
        if feature_cols:
            n_features_to_plot = min(16, len(feature_cols))
            sampled_features = np.random.choice(feature_cols, n_features_to_plot, replace=False)
            
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            axes = axes.ravel()
            
            for i, feature in enumerate(sampled_features):
                if i < len(axes):
                    axes[i].hist(self.training_data[feature], bins=30, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'{feature}')
                    axes[i].set_xlabel('Value')
                    axes[i].set_ylabel('Frequency')
            
            # Hide unused subplots
            for i in range(n_features_to_plot, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle('Feature Distributions (Sample)', fontsize=16)
            plt.tight_layout()
            if save_plots:
                plt.savefig('output/1.00/feature_distributions.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def apply_tsne_visualization(self, 
                                n_samples: int = 5000,
                                perplexity: int = 30,
                                save_plot: bool = True) -> None:
        """
        Apply t-SNE for dimensionality reduction and visualization.
        
        Args:
            n_samples: Number of samples to use for t-SNE (for performance)
            perplexity: t-SNE perplexity parameter
            save_plot: Whether to save the plot
        """
        if self.training_data is None:
            raise ValueError("Data must be loaded before t-SNE visualization")
        
        print("Applying t-SNE visualization...")
        
        feature_cols = [col for col in self.training_data.columns if col != 'faultNumber']
        if not feature_cols:
            print("No features found for t-SNE")
            return
        
        # Sample data if too large
        if len(self.training_data) > n_samples:
            sampled_data = self.training_data.sample(n=n_samples, random_state=42)
            print(f"Sampled {n_samples} samples for t-SNE visualization")
        else:
            sampled_data = self.training_data
        
        X = sampled_data[feature_cols].values
        y = sampled_data['faultNumber'].values if 'faultNumber' in sampled_data.columns else None
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
        X_tsne = tsne.fit_transform(X)
        
        # Visualize
        plt.figure(figsize=(12, 8))
        
        if y is not None:
            # Color by fault number
            unique_faults = np.unique(y)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_faults)))
            
            for i, fault in enumerate(unique_faults):
                mask = y == fault
                plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                           c=[colors[i]], label=f'Fault {fault}', alpha=0.7, s=20)
            
            plt.legend(title='Fault Number', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7, s=20)
        
        plt.title('t-SNE Visualization of TEP Data')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('output/1.00/tsne_visualization.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_data_summary(self) -> str:
        """
        Get a comprehensive summary of the loaded data.
        
        Returns:
            String containing data summary
        """
        if self.training_data is None:
            return "No data loaded"
        
        summary = []
        summary.append("=" * 50)
        summary.append("TEP DATA SUMMARY")
        summary.append("=" * 50)
        
        # Basic info
        summary.append(f"Training data shape: {self.training_data.shape}")
        summary.append(f"Testing data shape: {self.testing_data.shape if self.testing_data is not None else 'N/A'}")
        summary.append(f"Number of features: {len([col for col in self.training_data.columns if col != 'faultNumber'])}")
        
        # Target info
        if 'faultNumber' in self.training_data.columns:
            target_dist = self.training_data['faultNumber'].value_counts().sort_index()
            summary.append(f"Number of classes: {len(target_dist)}")
            summary.append("Class distribution:")
            for fault, count in target_dist.items():
                summary.append(f"  Fault {fault}: {count} samples ({count/len(self.training_data)*100:.1f}%)")
        
        # Feature info
        feature_cols = [col for col in self.training_data.columns if col != 'faultNumber']
        if feature_cols:
            summary.append(f"\nFeature information:")
            summary.append(f"  Feature names: {', '.join(feature_cols[:5])}{'...' if len(feature_cols) > 5 else ''}")
            summary.append(f"  Data types: {self.training_data[feature_cols].dtypes.unique()}")
            
            # Check for missing values
            missing_values = self.training_data[feature_cols].isnull().sum().sum()
            if missing_values > 0:
                summary.append(f"  Missing values: {missing_values}")
            else:
                summary.append("  Missing values: None")
        
        summary.append("=" * 50)
        return "\n".join(summary)
