"""
DNN-CUSUM: Deep Neural Network-based Adaptive CUSUM Detector

This module implements an adaptive CUSUM detector that uses a deep neural network
to dynamically predict optimal hyperparameters (k, h) for each time point.
"""

import os
import json
import pickle
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from numpy.typing import NDArray
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

from .mcusum import MCUSUMDetector

warnings.filterwarnings('ignore')


class DNNCUSUMDetector:
    """
    DNN-CUSUM: Adaptive CUSUM with neural network-based parameter prediction.

    Uses a deep neural network to predict optimal k and h parameters dynamically
    during monitoring, while leveraging the existing MCUSUMDetector for CUSUM computation.

    Attributes:
        window_size: Size of sliding window for feature extraction
        model_dir: Directory to save/load models and configs
        dnn_model: Trained neural network for parameter prediction
        feature_scaler: Scaler for input features
        global_mu_0: Global mean from training data
        global_sigma: Global covariance from training data
    """

    def __init__(self,
                 window_size: int = 50,
                 model_dir: str = 'models/'):
        """
        Initialize DNN-CUSUM detector.

        Args:
            window_size: Number of past observations to use for prediction
            model_dir: Directory for saving/loading models
        """
        self.window_size = window_size
        self.model_dir = model_dir
        self.dnn_model = None
        self.feature_scaler = StandardScaler()
        self.best_config = None
        self.global_mu_0 = None
        self.global_sigma = None
        self.is_fitted = False
        self.n_dims = None  # Number of dimensions in data (set during fit)

        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)

    def build_network(self, config: Dict[str, Any]) -> keras.Model:
        """
        Build DNN architecture from configuration.

        Args:
            config: Dictionary containing architecture parameters

        Returns:
            Compiled Keras model
        """
        arch = config['architecture']
        if self.n_dims is None:
            raise ValueError("n_dims must be set before building network. Call fit() first.")

        n_features = self.n_dims * 6  # 6 features per dimension

        # Input layer
        inputs = layers.Input(shape=(n_features,))

        # Reshape for LSTM: (n_dims, 6) where 6 features per dimension
        x = layers.Reshape((self.n_dims, 6))(inputs)

        # LSTM layers
        for i, units in enumerate(arch['units']):
            return_sequences = i < len(arch['units']) - 1
            x = layers.LSTM(units, return_sequences=return_sequences,
                          dropout=arch['dropout'])(x)

        # Dense layers
        for units in arch['dense']:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(arch['dropout'])(x)

        # Output heads for k and h (both positive values)
        k_output = layers.Dense(1, activation='softplus', name='k_output')(x)
        h_output = layers.Dense(1, activation='softplus', name='h_output')(x)

        # Create model
        model = models.Model(inputs=inputs, outputs=[k_output, h_output])

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
            loss={'k_output': 'mse', 'h_output': 'mse'},
            metrics={'k_output': 'mae', 'h_output': 'mae'}
        )

        return model

    def _extract_features(self, window: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Extract statistical features from a sliding window.

        Args:
            window: Data window of shape (window_size, n_dims)

        Returns:
            Feature vector combining multiple statistics (n_dims * 6)
        """
        if len(window) == 0:
            return np.zeros(self.n_dims * 6 if self.n_dims else 0)

        # Pad if window is too small
        if len(window) < self.window_size:
            padding = np.zeros((self.window_size - len(window), window.shape[1]))
            window = np.vstack([padding, window])

        # Take last window_size points
        window = window[-self.window_size:]

        # Compute features per dimension
        features = []
        for dim in range(window.shape[1]):
            dim_data = window[:, dim]

            # Compute autocorrelation, handle NaN for constant sequences
            if len(dim_data) > 1:
                autocorr = np.corrcoef(dim_data[:-1], dim_data[1:])[0, 1]
                autocorr = 0.0 if np.isnan(autocorr) else autocorr  # Replace NaN with 0
            else:
                autocorr = 0.0

            features.extend([
                np.mean(dim_data),           # Mean
                np.std(dim_data),            # Std deviation
                np.max(dim_data) - np.min(dim_data),  # Range
                dim_data[-1] - dim_data[0],  # Total change
                np.mean(np.diff(dim_data)),  # Average rate of change
                autocorr                      # Autocorrelation (NaN-safe)
            ])

        # Final safety: replace any remaining NaN with 0
        feature_array = np.array(features)
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
        return feature_array

    def _compute_optimal_params(self,
                               window: NDArray[np.float64],
                               has_fault: bool,
                               base_k: float = 0.5,
                               base_h: float = 5.0) -> Tuple[float, float]:
        """
        Compute optimal k and h for training using heuristic approach.

        Strategy:
        - For fault regions: use lower k (more sensitive) and lower h
        - For normal regions: use higher k (less sensitive) and higher h
        - Adjust based on window statistics

        Args:
            window: Data window
            has_fault: Whether window contains anomaly
            base_k: Base reference value
            base_h: Base threshold

        Returns:
            (optimal_k, optimal_h) tuple
        """
        if len(window) < 2:
            return base_k, base_h

        # Compute window statistics
        magnitude = np.mean(np.abs(window - np.mean(window, axis=0)))
        volatility = np.mean(np.std(window, axis=0))

        if has_fault:
            # For fault: lower k (sensitive), adjusted h
            k = base_k * 0.3 * (1.0 + 0.1 * magnitude)
            h = base_h * 0.6 * (1.0 + 0.1 * volatility)
        else:
            # For normal: higher k (conservative), higher h
            k = base_k * 1.5 * (1.0 + 0.05 * volatility)
            h = base_h * 1.2 * (1.0 + 0.05 * magnitude)

        # Ensure positive and bounded
        k = np.clip(k, 0.1, 10.0)
        h = np.clip(h, 1.0, 15.0)

        return float(k), float(h)

    def _generate_training_data(self,
                               X_incontrol: NDArray[np.float64],
                               X_outcontrol: Optional[NDArray[np.float64]] = None
                               ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Generate training data with sliding windows and optimal parameters.

        Args:
            X_incontrol: In-control (normal) data
            X_outcontrol: Out-of-control (fault) data

        Returns:
            (X_features, y_k, y_h) for training
        """
        X_features = []
        y_k = []
        y_h = []

        # Process in-control data
        print("Generating training data from in-control samples...")
        for i in range(self.window_size, len(X_incontrol)):
            window = X_incontrol[i-self.window_size:i]
            features = self._extract_features(window)
            k_opt, h_opt = self._compute_optimal_params(window, has_fault=False)

            X_features.append(features)
            y_k.append(k_opt)
            y_h.append(h_opt)

        # Process out-of-control data if provided
        if X_outcontrol is not None and len(X_outcontrol) > 0:
            print("Generating training data from out-of-control samples...")
            for i in range(self.window_size, len(X_outcontrol)):
                window = X_outcontrol[i-self.window_size:i]
                features = self._extract_features(window)
                k_opt, h_opt = self._compute_optimal_params(window, has_fault=True)

                X_features.append(features)
                y_k.append(k_opt)
                y_h.append(h_opt)

        return np.array(X_features), np.array(y_k), np.array(y_h)

    def grid_search(self,
                   X_train: NDArray,
                   y_k_train: NDArray,
                   y_h_train: NDArray,
                   param_grid: Dict[str, List]) -> Dict[str, Any]:
        """
        Perform grid search to find best DNN configuration.

        Args:
            X_train: Training features
            y_k_train: Target k values
            y_h_train: Target h values
            param_grid: Dictionary of parameters to search

        Returns:
            Best configuration dictionary
        """
        print("\n" + "="*60)
        print("Starting Grid Search for Best Configuration")
        print("="*60)

        best_score = float('inf')
        best_config = None

        # Generate all combinations
        architectures = param_grid['architecture']
        learning_rates = param_grid['learning_rate']
        batch_sizes = param_grid['batch_size']
        dropouts = param_grid['dropout']

        total_configs = len(architectures) * len(learning_rates) * len(batch_sizes) * len(dropouts)
        config_count = 0

        for arch in architectures:
            for lr in learning_rates:
                for bs in batch_sizes:
                    for dropout in dropouts:
                        config_count += 1

                        # Update architecture with dropout
                        arch_with_dropout = arch.copy()
                        arch_with_dropout['dropout'] = dropout

                        config = {
                            'architecture': arch_with_dropout,
                            'learning_rate': lr,
                            'batch_size': bs,
                            'window_size': self.window_size
                        }

                        print(f"\n[{config_count}/{total_configs}] Testing config: "
                              f"LSTM{arch['units']}, Dense{arch['dense']}, "
                              f"lr={lr}, bs={bs}, dropout={dropout}")

                        # Build and train model
                        try:
                            model = self.build_network(config)

                            # Split for validation
                            X_tr, X_val, y_k_tr, y_k_val, y_h_tr, y_h_val = train_test_split(
                                X_train, y_k_train, y_h_train,
                                test_size=0.2, random_state=42
                            )

                            # Train (reduced epochs for faster grid search)
                            history = model.fit(
                                X_tr,
                                {'k_output': y_k_tr, 'h_output': y_h_tr},
                                validation_data=(X_val, {'k_output': y_k_val, 'h_output': y_h_val}),
                                batch_size=bs,
                                epochs=20,  # Reduced from 30 to 20
                                verbose=0,
                                callbacks=[callbacks.EarlyStopping(patience=3, restore_best_weights=True)]  # Reduced patience
                            )

                            # Evaluate
                            val_loss = min(history.history['val_loss'])
                            print(f"  Validation Loss: {val_loss:.4f}")

                            # Track best
                            if val_loss < best_score:
                                best_score = val_loss
                                best_config = config.copy()
                                best_config['val_loss'] = val_loss
                                print(f"  *** New best configuration! ***")

                            # Clean up
                            del model
                            tf.keras.backend.clear_session()

                        except Exception as e:
                            print(f"  Error with config: {e}")
                            continue

        print("\n" + "="*60)
        print("Grid Search Complete!")
        print(f"Best Validation Loss: {best_score:.4f}")
        print(f"Best Config: {best_config}")
        print("="*60 + "\n")

        return best_config

    def save_model(self, model_path: str, config_path: str):
        """Save trained model and configuration."""
        self.dnn_model.save(model_path)
        with open(config_path, 'w') as f:
            json.dump(self.best_config, f, indent=2)

        # Save scalers and global params
        scaler_path = model_path.replace('.h5', '_scaler.pkl')
        params_path = model_path.replace('.h5', '_params.pkl')

        with open(scaler_path, 'wb') as f:
            pickle.dump(self.feature_scaler, f)

        with open(params_path, 'wb') as f:
            pickle.dump({
                'global_mu_0': self.global_mu_0,
                'global_sigma': self.global_sigma,
                'window_size': self.window_size,
                'n_dims': self.n_dims
            }, f)

        print(f"Model saved to: {model_path}")
        print(f"Config saved to: {config_path}")

    def load_model(self, model_path: str, config_path: str):
        """Load pre-trained model and configuration."""
        self.dnn_model = keras.models.load_model(model_path)

        with open(config_path, 'r') as f:
            self.best_config = json.load(f)

        # Load scalers and global params
        scaler_path = model_path.replace('.h5', '_scaler.pkl')
        params_path = model_path.replace('.h5', '_params.pkl')

        with open(scaler_path, 'rb') as f:
            self.feature_scaler = pickle.load(f)

        with open(params_path, 'rb') as f:
            params = pickle.load(f)
            self.global_mu_0 = params['global_mu_0']
            self.global_sigma = params['global_sigma']
            self.window_size = params['window_size']
            self.n_dims = params.get('n_dims', None)  # Backward compatibility

            # If n_dims not in saved params (old model), infer from global_mu_0
            if self.n_dims is None and self.global_mu_0 is not None:
                self.n_dims = len(self.global_mu_0)

        self.is_fitted = True
        print(f"Model loaded from: {model_path}")

    def fit(self,
            X_incontrol: NDArray[np.float64],
            X_outcontrol: Optional[NDArray[np.float64]] = None,
            force_retrain: bool = False,
            grid_search: bool = True):
        """
        Train DNN-CUSUM detector or load existing model.

        Args:
            X_incontrol: In-control training data
            X_outcontrol: Out-of-control training data (optional)
            force_retrain: Force retraining even if saved model exists
            grid_search: Perform grid search for best configuration

        Returns:
            Self for method chaining
        """
        model_path = os.path.join(self.model_dir, 'dnn_cusum_model.h5')
        config_path = os.path.join(self.model_dir, 'dnn_cusum_best_config.json')

        # Try to load existing model
        if not force_retrain and os.path.exists(model_path) and os.path.exists(config_path):
            print("\n" + "="*60)
            print("Found existing trained model. Loading...")
            print("="*60)
            self.load_model(model_path, config_path)
            return self

        print("\n" + "="*60)
        print("Training New DNN-CUSUM Model")
        print("="*60)

        # Set number of dimensions from data
        self.n_dims = X_incontrol.shape[1]
        print(f"Data dimensions: {self.n_dims}")

        # Compute global statistics
        self.global_mu_0 = np.mean(X_incontrol, axis=0)
        self.global_sigma = np.cov(X_incontrol, rowvar=False)

        # Generate training data
        X_features, y_k, y_h = self._generate_training_data(X_incontrol, X_outcontrol)

        print(f"\nGenerated {len(X_features)} training samples")
        print(f"Feature dimension: {X_features.shape[1]}")
        print(f"k range: [{y_k.min():.2f}, {y_k.max():.2f}]")
        print(f"h range: [{y_h.min():.2f}, {y_h.max():.2f}]")

        # Scale features
        X_features_scaled = self.feature_scaler.fit_transform(X_features)

        # Grid search or use default config
        if grid_search:
            param_grid = {
                'architecture': [
                    {'units': [64], 'dense': [32]},
                    {'units': [128], 'dense': [64]},  # Reduced from 2 to 1 arch
                ],
                'learning_rate': [0.001],  # Reduced from 2 to 1
                'batch_size': [32],        # Reduced from 2 to 1
                'dropout': [0.2, 0.3]      # Keep 2 for robustness
            }
            # Total: 2 architectures × 1 lr × 1 bs × 2 dropout = 4 configs (~40 min)

            self.best_config = self.grid_search(X_features_scaled, y_k, y_h, param_grid)
        else:
            # Default configuration
            self.best_config = {
                'architecture': {'units': [64], 'dense': [32], 'dropout': 0.2},
                'learning_rate': 0.001,
                'batch_size': 32,
                'window_size': self.window_size
            }

        # Train final model with best config
        print("\nTraining final model with best configuration...")
        self.dnn_model = self.build_network(self.best_config)

        history = self.dnn_model.fit(
            X_features_scaled,
            {'k_output': y_k, 'h_output': y_h},
            batch_size=self.best_config['batch_size'],
            epochs=50,
            validation_split=0.2,
            verbose=1,
            callbacks=[
                callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]
        )

        self.is_fitted = True

        # Save model
        self.save_model(model_path, config_path)

        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60 + "\n")

        return self

    def predict(self,
                X_test: NDArray[np.float64],
                return_params: bool = False
                ) -> Tuple[NDArray[np.int_], Optional[Dict[str, List]]]:
        """
        Predict anomalies with adaptive parameters.

        Args:
            X_test: Test data
            return_params: Whether to return parameter history

        Returns:
            predictions: Binary predictions (0=normal, 1=anomaly)
            param_history: Dictionary with k and h history (if return_params=True)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        n_samples = len(X_test)
        predictions = np.zeros(n_samples, dtype=int)
        param_history = {'k': [], 'h': [], 'cusum_stat': []} if return_params else None

        # Initialize CUSUM state
        cumsum_state = np.zeros(X_test.shape[1])

        for t in range(n_samples):
            # Extract window
            start_idx = max(0, t - self.window_size + 1)
            window = X_test[start_idx:t+1]

            # Extract and scale features
            features = self._extract_features(window).reshape(1, -1)
            features_scaled = self.feature_scaler.transform(features)

            # Predict parameters
            k_pred, h_pred = self.dnn_model.predict(features_scaled, verbose=0)
            k_t = float(k_pred[0, 0])
            h_t = float(h_pred[0, 0])

            # Use MCUSUM with predicted parameters
            current_obs = X_test[t:t+1]
            deviation = current_obs - self.global_mu_0

            # Compute whitening transformation
            eigvals, eigvecs = np.linalg.eigh(self.global_sigma)
            eigvals_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-12)))
            sigma_inv_sqrt = eigvecs @ eigvals_inv_sqrt @ eigvecs.T

            # Whiten the deviation
            Z = deviation @ sigma_inv_sqrt.T

            # CUSUM recursion (from MCUSUMDetector)
            V_t = cumsum_state + Z[0]
            norm_V_t = np.linalg.norm(V_t)

            if norm_V_t <= k_t:
                cumsum_state = np.zeros(X_test.shape[1])
            else:
                shrinkage = 1.0 - k_t / norm_V_t
                cumsum_state = V_t * shrinkage

            cusum_stat = np.linalg.norm(cumsum_state)

            # Detect anomaly
            predictions[t] = 1 if cusum_stat > h_t else 0

            # Record history
            if return_params:
                param_history['k'].append(k_t)
                param_history['h'].append(h_t)
                param_history['cusum_stat'].append(cusum_stat)

        if return_params:
            return predictions, param_history
        else:
            return predictions, None
