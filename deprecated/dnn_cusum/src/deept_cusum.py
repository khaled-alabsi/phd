"""
Deep Threshold CUSUM (DeepT-CUSUM) Detector

A hybrid anomaly detection method combining:
- Statistical CUSUM with fixed reference value k
- Deep Neural Network for adaptive threshold h prediction
- Feedback mechanism using previous CUSUM statistic

Input: [x_t (52 dims), S_{t-1} (1 scalar)] = 53 features
Output: h_t (adaptive threshold)
"""

import os
import json
import pickle
import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# TensorFlow/Keras imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks


class DeepTCUSUMDetector:
    """
    Deep Threshold CUSUM Detector with Adaptive Threshold.

    Uses a DNN to predict adaptive control limit h_t based on:
    - Current observation (52 dimensions)
    - Previous CUSUM statistic (feedback)

    CUSUM uses fixed reference value k and adaptive threshold h_t.
    """

    def __init__(self, k_fixed: Optional[float] = None, model_dir: str = 'models/'):
        """
        Initialize DeepT-CUSUM detector.

        Args:
            k_fixed: Fixed reference value for CUSUM (if None, will be computed)
            model_dir: Directory for saving/loading models
        """
        self.k_fixed = k_fixed
        self.model_dir = model_dir
        self.dnn_model = None
        self.feature_scaler = StandardScaler()
        self.best_config = None
        self.global_mu_0 = None
        self.global_sigma = None
        self.is_fitted = False
        self.n_dims = None

        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)

    def build_network(self, config: Dict[str, Any]) -> keras.Model:
        """
        Build Dense neural network for threshold prediction.

        Architecture: 53 → Dense → Dropout → Dense → Dropout → 1

        Args:
            config: Dictionary containing network parameters

        Returns:
            Compiled Keras model
        """
        n_inputs = self.n_dims + 1  # Features + S_{t-1}

        # Input layer
        inputs = layers.Input(shape=(n_inputs,), name='input')

        # First dense layer
        x = layers.Dense(config['dense1_units'], activation='relu', name='dense1')(inputs)
        x = layers.Dropout(config['dropout'], name='dropout1')(x)

        # Second dense layer
        x = layers.Dense(config['dense2_units'], activation='relu', name='dense2')(x)
        x = layers.Dropout(config['dropout'], name='dropout2')(x)

        # Output layer (single threshold value, must be positive)
        h_output = layers.Dense(1, activation='softplus', name='h_output')(x)

        # Create model
        model = models.Model(inputs=inputs, outputs=h_output)

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )

        return model

    def _compute_optimal_h(self,
                          x_t: NDArray[np.float64],
                          S_prev: float,
                          has_fault: bool,
                          base_h: float = 5.0) -> float:
        """
        Compute optimal threshold h for training.

        Strategy:
        - For fault regions: lower h (quick detection)
        - For normal regions: higher h (avoid false alarms)
        - Adjust based on S_prev (CUSUM feedback)

        Args:
            x_t: Current observation
            S_prev: Previous CUSUM statistic
            has_fault: Whether current point is in fault region
            base_h: Base threshold value

        Returns:
            Optimal h value
        """
        # Compute deviation magnitude
        if self.global_mu_0 is not None:
            deviation = np.mean(np.abs(x_t - self.global_mu_0))
        else:
            deviation = 0.0

        if has_fault:
            # Fault region: lower threshold for quick detection
            h = base_h * 0.5 * (1.0 + 0.1 * deviation)

            # If CUSUM is rising, keep threshold low
            if S_prev > 2.0:
                h *= 0.8
        else:
            # Normal region: higher threshold to avoid false alarms
            h = base_h * 1.5 * (1.0 + 0.05 * deviation)

            # If CUSUM is rising in normal region, increase threshold
            if S_prev > 2.0:
                h *= 1.3

        # Clip to reasonable range
        h = np.clip(h, 1.0, 15.0)

        return float(h)

    def _generate_training_data(self,
                               X_incontrol: NDArray[np.float64],
                               X_outcontrol: Optional[NDArray[np.float64]] = None
                               ) -> Tuple[NDArray, NDArray]:
        """
        Generate training data by simulating CUSUM process.

        For each observation, compute:
        - Input: [x_t, S_{t-1}]
        - Target: h_optimal

        Args:
            X_incontrol: In-control (normal) data
            X_outcontrol: Out-of-control (fault) data

        Returns:
            (X_features, y_h) for training
        """
        X_features = []
        y_h = []

        # Compute k if not set
        if self.k_fixed is None:
            # Use heuristic: k = 0.5 * average deviation
            deviations = np.mean(np.abs(X_incontrol - self.global_mu_0), axis=1)
            self.k_fixed = 0.5 * np.mean(deviations)
            print(f"Computed fixed k: {self.k_fixed:.4f}")

        # Process in-control data
        print("Generating training data from in-control samples...")
        S_t = 0.0  # Initialize CUSUM

        for i in range(len(X_incontrol)):
            x_t = X_incontrol[i]

            # Compute Mahalanobis distance
            diff = x_t - self.global_mu_0
            C_t = diff @ self.global_sigma @ diff

            # Feature: [x_t, S_{t-1}]
            features = np.concatenate([x_t, [S_t]])

            # Target: optimal h
            h_opt = self._compute_optimal_h(x_t, S_t, has_fault=False)

            X_features.append(features)
            y_h.append(h_opt)

            # Update CUSUM for next iteration
            S_t = max(0, S_t + C_t - self.k_fixed)

        # Process out-of-control data if provided
        if X_outcontrol is not None and len(X_outcontrol) > 0:
            print("Generating training data from out-of-control samples...")
            S_t = 0.0  # Reset CUSUM

            for i in range(len(X_outcontrol)):
                x_t = X_outcontrol[i]

                # Compute Mahalanobis distance
                diff = x_t - self.global_mu_0
                C_t = diff @ self.global_sigma @ diff

                # Feature: [x_t, S_{t-1}]
                features = np.concatenate([x_t, [S_t]])

                # Target: optimal h
                h_opt = self._compute_optimal_h(x_t, S_t, has_fault=True)

                X_features.append(features)
                y_h.append(h_opt)

                # Update CUSUM
                S_t = max(0, S_t + C_t - self.k_fixed)

        return np.array(X_features), np.array(y_h)

    def grid_search(self,
                   X_train: NDArray,
                   y_h_train: NDArray,
                   param_grid: Dict[str, List]) -> Dict[str, Any]:
        """
        Perform grid search to find best network configuration.

        Args:
            X_train: Training features
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
        dense1_units_list = param_grid['dense1_units']
        dense2_units_list = param_grid['dense2_units']
        learning_rates = param_grid['learning_rate']
        dropouts = param_grid['dropout']

        total_configs = len(dense1_units_list) * len(dense2_units_list) * len(learning_rates) * len(dropouts)
        config_count = 0

        for d1 in dense1_units_list:
            for d2 in dense2_units_list:
                for lr in learning_rates:
                    for dropout in dropouts:
                        config_count += 1

                        config = {
                            'dense1_units': d1,
                            'dense2_units': d2,
                            'learning_rate': lr,
                            'dropout': dropout
                        }

                        print(f"\n[{config_count}/{total_configs}] Testing config: "
                              f"Dense[{d1}, {d2}], lr={lr}, dropout={dropout}")

                        try:
                            # Build and train model
                            model = self.build_network(config)

                            # Split for validation
                            X_tr, X_val, y_h_tr, y_h_val = train_test_split(
                                X_train, y_h_train,
                                test_size=0.2, random_state=42
                            )

                            # Train
                            history = model.fit(
                                X_tr, y_h_tr,
                                validation_data=(X_val, y_h_val),
                                batch_size=32,
                                epochs=20,
                                verbose=0,
                                callbacks=[callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
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

    def fit(self,
            X_incontrol: NDArray[np.float64],
            X_outcontrol: Optional[NDArray[np.float64]] = None,
            k_fixed: Optional[float] = None,
            force_retrain: bool = False,
            grid_search: bool = True):
        """
        Train DeepT-CUSUM detector or load existing model.

        Args:
            X_incontrol: In-control training data
            X_outcontrol: Out-of-control training data (optional)
            k_fixed: Fixed reference value (if None, will be computed)
            force_retrain: Force retraining even if saved model exists
            grid_search: Perform grid search for best configuration

        Returns:
            Self for method chaining
        """
        model_path = os.path.join(self.model_dir, 'deept_cusum_model.h5')
        config_path = os.path.join(self.model_dir, 'deept_cusum_best_config.json')

        # Try to load existing model
        if not force_retrain and os.path.exists(model_path) and os.path.exists(config_path):
            print("\n" + "="*60)
            print("Found existing trained model. Loading...")
            print("="*60)
            self.load_model(model_path, config_path)
            return self

        print("\n" + "="*60)
        print("Training New DeepT-CUSUM Model")
        print("="*60)

        # Set k if provided
        if k_fixed is not None:
            self.k_fixed = k_fixed

        # Set number of dimensions from data
        self.n_dims = X_incontrol.shape[1]
        print(f"Data dimensions: {self.n_dims}")

        # Compute global statistics
        self.global_mu_0 = np.mean(X_incontrol, axis=0)
        self.global_sigma = np.linalg.inv(np.cov(X_incontrol, rowvar=False))

        # Generate training data
        X_features, y_h = self._generate_training_data(X_incontrol, X_outcontrol)

        print(f"\nGenerated {len(X_features)} training samples")
        print(f"Feature dimension: {X_features.shape[1]} (52 features + 1 CUSUM state)")
        print(f"h range: [{y_h.min():.2f}, {y_h.max():.2f}]")
        print(f"Fixed k: {self.k_fixed:.4f}")

        # Scale features
        X_features_scaled = self.feature_scaler.fit_transform(X_features)

        # Grid search or use default config
        if grid_search:
            param_grid = {
                'dense1_units': [64, 128],
                'dense2_units': [32, 64],
                'learning_rate': [0.001],
                'dropout': [0.2, 0.3]
            }
            # Total: 2 × 2 × 1 × 2 = 8 configs

            self.best_config = self.grid_search(X_features_scaled, y_h, param_grid)
        else:
            # Default configuration
            self.best_config = {
                'dense1_units': 64,
                'dense2_units': 32,
                'learning_rate': 0.001,
                'dropout': 0.2
            }

        # Train final model with best config
        print("\nTraining final model with best configuration...")
        self.dnn_model = self.build_network(self.best_config)

        history = self.dnn_model.fit(
            X_features_scaled, y_h,
            batch_size=32,
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
                return_thresholds: bool = False
                ) -> Tuple[NDArray[np.int_], Optional[Dict[str, List]]]:
        """
        Predict anomalies with adaptive threshold.

        Args:
            X_test: Test data
            return_thresholds: Whether to return threshold history

        Returns:
            predictions: Binary predictions (0=normal, 1=anomaly)
            threshold_history: Dictionary with h_t and S_t history (if return_thresholds=True)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        n_samples = len(X_test)
        predictions = np.zeros(n_samples, dtype=int)
        threshold_history = {'h': [], 'cusum_stat': []} if return_thresholds else None

        # Initialize CUSUM
        S_t = 0.0

        for i in range(n_samples):
            x_t = X_test[i]

            # Prepare DNN input: [x_t, S_{t-1}]
            dnn_input = np.concatenate([x_t, [S_t]])
            dnn_input_scaled = self.feature_scaler.transform(dnn_input.reshape(1, -1))

            # Predict adaptive threshold
            h_t = self.dnn_model.predict(dnn_input_scaled, verbose=0)[0, 0]

            # Compute Mahalanobis distance
            diff = x_t - self.global_mu_0
            C_t = diff @ self.global_sigma @ diff

            # Update CUSUM statistic
            S_t = max(0, S_t + C_t - self.k_fixed)

            # Make decision
            if S_t > h_t:
                predictions[i] = 1
            else:
                predictions[i] = 0

            # Store history
            if return_thresholds:
                threshold_history['h'].append(float(h_t))
                threshold_history['cusum_stat'].append(float(S_t))

        return predictions, threshold_history

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
                'k_fixed': self.k_fixed,
                'global_mu_0': self.global_mu_0,
                'global_sigma': self.global_sigma,
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
            self.k_fixed = params['k_fixed']
            self.global_mu_0 = params['global_mu_0']
            self.global_sigma = params['global_sigma']
            self.n_dims = params.get('n_dims', None)

            # Backward compatibility
            if self.n_dims is None and self.global_mu_0 is not None:
                self.n_dims = len(self.global_mu_0)

        self.is_fitted = True
        print(f"Model loaded from: {model_path}")
        print(f"Fixed k: {self.k_fixed:.4f}")
