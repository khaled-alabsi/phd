"""
Tennessee Eastman Process (TEP) Model Training and Evaluation

This module handles model training, evaluation, and comparison for the TEP
anomaly detection project.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import joblib

from te_utils import (
    save_plot, save_dataframe, compute_classification_scores_per_fault,
    false_alarm_rate_per_class, positive_alarm_rate_per_class,
    compute_average_classification_metrics
)


class TEPModelTrainer:
    """Class for training and evaluating TEP anomaly detection models."""
    
    def __init__(self, output_dir: str = "output/1.00/"):
        """
        Initialize the model trainer.
        
        Args:
            output_dir: Directory for saving outputs
        """
        self.output_dir = output_dir
        self.models = {}
        self.model_results = {}
        self.best_model = None
        self.best_score = 0
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    def prepare_data(self, 
                    train_data: pd.DataFrame,
                    test_data: pd.DataFrame,
                    target_col: str = 'faultNumber',
                    binary_classification: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for model training.
        
        Args:
            train_data: Training data
            test_data: Testing data
            target_col: Target column name
            binary_classification: Whether to convert to binary classification
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("Preparing data for model training...")
        
        # Separate features and target
        feature_cols = [col for col in train_data.columns if col != target_col]
        
        X_train = train_data[feature_cols].values
        y_train = train_data[target_col].values
        X_test = test_data[feature_cols].values
        y_test = test_data[target_col].values
        
        # Convert to binary classification if requested
        if binary_classification:
            print("Converting to binary classification (0: normal, 1: faulty)")
            y_train_binary = (y_train > 0).astype(int)
            y_test_binary = (y_test > 0).astype(int)
            
            # Update target variables
            y_train = y_train_binary
            y_test = y_test_binary
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        print(f"Number of classes: {len(np.unique(y_train))}")
        
        return X_train, X_test, y_train, y_test
    
    def train_traditional_models(self, 
                                X_train: np.ndarray,
                                y_train: np.ndarray,
                                models_to_train: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train traditional machine learning models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            models_to_train: List of model names to train (if None, train all)
            
        Returns:
            Dictionary of trained models
        """
        print("Training traditional machine learning models...")
        
        # Define available models
        available_models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'lightgbm': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'svm': SVC(random_state=42, probability=True),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'naive_bayes': GaussianNB()
        }
        
        # Filter models if specified
        if models_to_train:
            available_models = {k: v for k, v in available_models.items() if k in models_to_train}
        
        # Train models
        for name, model in available_models.items():
            print(f"Training {name}...")
            try:
                model.fit(X_train, y_train)
                self.models[name] = model
                print(f"✓ {name} trained successfully")
            except Exception as e:
                print(f"✗ Error training {name}: {e}")
        
        return self.models
    
    def train_neural_network(self, 
                            X_train: np.ndarray,
                            y_train: np.ndarray,
                            n_classes: int = 2,
                            architecture: str = 'simple') -> Any:
        """
        Train a neural network model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_classes: Number of classes
            architecture: Architecture type ('simple', 'deep', 'wide')
            
        Returns:
            Trained neural network model
        """
        print("Training neural network...")
        
        # Convert labels to categorical if multi-class
        if n_classes > 2:
            from keras.utils import to_categorical
            y_train_cat = to_categorical(y_train, num_classes=n_classes)
        else:
            y_train_cat = y_train
        
        # Define architecture
        if architecture == 'simple':
            model = Sequential([
                Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(n_classes, activation='softmax' if n_classes > 2 else 'sigmoid')
            ])
        elif architecture == 'deep':
            model = Sequential([
                Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
                Dropout(0.4),
                Dense(128, activation='relu'),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.1),
                Dense(n_classes, activation='softmax' if n_classes > 2 else 'sigmoid')
            ])
        elif architecture == 'wide':
            model = Sequential([
                Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
                Dropout(0.5),
                Dense(256, activation='relu'),
                Dropout(0.3),
                Dense(n_classes, activation='softmax' if n_classes > 2 else 'sigmoid')
            ])
        
        # Compile model
        loss = 'categorical_crossentropy' if n_classes > 2 else 'binary_crossentropy'
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=loss,
            metrics=['accuracy']
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train_cat,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        self.models['neural_network'] = model
        print("✓ Neural network trained successfully")
        
        return model
    
    def evaluate_models(self, 
                       X_test: np.ndarray,
                       y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation results for each model
        """
        print("Evaluating models...")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            
            try:
                # Make predictions
                if name == 'neural_network':
                    # Handle neural network predictions
                    if len(np.unique(y_test)) > 2:
                        y_pred_proba = model.predict(X_test)
                        y_pred = np.argmax(y_pred_proba, axis=1)
                    else:
                        y_pred_proba = model.predict(X_test)
                        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                else:
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                results[name] = metrics
                
                # Store results
                self.model_results[name] = {
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'metrics': metrics
                }
                
                print(f"✓ {name} evaluated successfully")
                
            except Exception as e:
                print(f"✗ Error evaluating {name}: {e}")
                results[name] = {}
        
        return results
    
    def _calculate_metrics(self, 
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # ROC AUC if probabilities available
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            try:
                if y_pred_proba.ndim > 1:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                metrics['roc_auc'] = np.nan
        else:
            metrics['roc_auc'] = np.nan
        
        return metrics
    
    def create_confusion_matrices(self, save_plots: bool = True) -> None:
        """
        Create confusion matrices for all models.
        
        Args:
            save_plots: Whether to save the plots
        """
        print("Creating confusion matrices...")
        
        for name, results in self.model_results.items():
            if 'predictions' not in results:
                continue
                
            y_pred = results['predictions']
            y_true = results.get('true_labels', [])  # This should be set during evaluation
            
            if len(y_true) == 0:
                print(f"Skipping {name}: no true labels available")
                continue
            
            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=range(len(np.unique(y_true))),
                       yticklabels=range(len(np.unique(y_true))))
            plt.title(f'{name.replace("_", " ").title()} Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            
            if save_plots:
                save_plot(f"{name}_confusion_matrix", plot_path=f"{self.output_dir}confusion_matrix/")
            
            plt.show()
    
    def compare_models(self, save_results: bool = True) -> pd.DataFrame:
        """
        Compare all models and create a summary.
        
        Args:
            save_results: Whether to save the comparison results
            
        Returns:
            DataFrame with model comparison
        """
        print("Comparing models...")
        
        comparison_data = []
        
        for name, results in self.model_results.items():
            if 'metrics' in results:
                row = {'Model': name}
                row.update(results['metrics'])
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by F1 score (descending)
        if 'f1_score' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('f1_score', ascending=False)
        
        # Display results
        print("\nModel Comparison:")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        
        # Save results
        if save_results:
            save_dataframe(comparison_df, "model_comparison", "summary")
        
        return comparison_df
    
    def save_models(self, save_dir: str = "models/") -> None:
        """
        Save trained models to disk.
        
        Args:
            save_dir: Directory to save models
        """
        print("Saving models...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.models.items():
            try:
                if name == 'neural_network':
                    # Save Keras model
                    model_path = os.path.join(save_dir, f"{name}.h5")
                    model.save(model_path)
                else:
                    # Save scikit-learn models
                    model_path = os.path.join(save_dir, f"{name}.pkl")
                    joblib.dump(model, model_path)
                
                print(f"✓ {name} saved to {model_path}")
                
            except Exception as e:
                print(f"✗ Error saving {name}: {e}")
    
    def load_models(self, load_dir: str = "models/") -> Dict[str, Any]:
        """
        Load trained models from disk.
        
        Args:
            load_dir: Directory containing saved models
            
        Returns:
            Dictionary of loaded models
        """
        print("Loading models...")
        
        if not os.path.exists(load_dir):
            print(f"Models directory not found: {load_dir}")
            return {}
        
        for filename in os.listdir(load_dir):
            if filename.endswith('.pkl'):
                name = filename[:-4]
                model_path = os.path.join(load_dir, filename)
                try:
                    self.models[name] = joblib.load(model_path)
                    print(f"✓ {name} loaded successfully")
                except Exception as e:
                    print(f"✗ Error loading {name}: {e}")
            
            elif filename.endswith('.h5'):
                name = filename[:-3]
                model_path = os.path.join(load_dir, filename)
                try:
                    from keras.models import load_model
                    self.models[name] = load_model(model_path)
                    print(f"✓ {name} loaded successfully")
                except Exception as e:
                    print(f"✗ Error loading {name}: {e}")
        
        return self.models
    
    def get_best_model(self, metric: str = 'f1_score') -> Tuple[str, Any]:
        """
        Get the best performing model based on a specific metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model_name, model)
        """
        if not self.model_results:
            return None, None
        
        best_score = -np.inf
        best_model_name = None
        
        for name, results in self.model_results.items():
            if 'metrics' in results and metric in results['metrics']:
                score = results['metrics'][metric]
                if score > best_score:
                    best_score = score
                    best_model_name = name
        
        if best_model_name:
            self.best_model = self.models[best_model_name]
            self.best_score = best_score
            return best_model_name, self.models[best_model_name]
        
        return None, None
