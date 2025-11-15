import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
import numpy as np
from typing import Optional, Tuple

class AutoencoderDetectorEnhanced:
    def __init__(self, encoding_dim: int = 8):
        self.encoding_dim = encoding_dim
        self.model: Optional[models.Model] = None
        self.threshold: Optional[float] = None
        self.X_train: Optional[np.ndarray] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.history = None

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize data using z-score normalization."""
        if fit:
            self.scaler_mean = np.mean(X, axis=0)
            self.scaler_std = np.std(X, axis=0) + 1e-7  # Add small epsilon to avoid division by zero

        if self.scaler_mean is None or self.scaler_std is None:
            raise RuntimeError("Scaler not fitted. This should not happen.")

        return (X - self.scaler_mean) / self.scaler_std

    def _build_model(self, input_dim: int) -> models.Model:
        """Build an improved autoencoder architecture."""
        # Calculate layer dimensions dynamically
        layer1_dim = max(input_dim // 2, self.encoding_dim * 4)
        layer2_dim = max(input_dim // 4, self.encoding_dim * 2)

        # Encoder
        input_layer = layers.Input(shape=(input_dim,))

        # Add noise for denoising autoencoder capability (helps with robustness)
        noisy_input = layers.GaussianNoise(0.1)(input_layer)

        # Encoder layers with batch normalization and dropout for regularization
        encoded = layers.Dense(
            layer1_dim,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            kernel_initializer='he_normal'
        )(noisy_input)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(0.2)(encoded)

        encoded = layers.Dense(
            layer2_dim,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            kernel_initializer='he_normal'
        )(encoded)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(0.2)(encoded)

        # Bottleneck layer
        encoded = layers.Dense(
            self.encoding_dim,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            kernel_initializer='he_normal',
            name='bottleneck'
        )(encoded)

        # Decoder layers (symmetric to encoder)
        decoded = layers.Dense(
            layer2_dim,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            kernel_initializer='he_normal'
        )(encoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dropout(0.2)(decoded)

        decoded = layers.Dense(
            layer1_dim,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            kernel_initializer='he_normal'
        )(decoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dropout(0.2)(decoded)

        # Output layer with tanh activation (works better with normalized data)
        decoded = layers.Dense(
            input_dim,
            activation='tanh',
            kernel_initializer='glorot_uniform'
        )(decoded)

        model = models.Model(inputs=input_layer, outputs=decoded)
        return model

    def fit(self, X_train: np.ndarray, epochs: int = 50, batch_size: int = 32,
            threshold_percentile: float = 95) -> None:
        """
        Train the autoencoder on normal data.

        Args:
            X_train: Training data (assumed to be mostly normal samples)
            epochs: Number of training epochs
            batch_size: Batch size for training
            threshold_percentile: Percentile for threshold calculation
        """
        # Normalize the training data
        X_train_normalized = self._normalize(X_train, fit=True)
        self.X_train = X_train_normalized

        input_dim = X_train.shape[1]

        # Build the model
        self.model = self._build_model(input_dim)

        # Use a custom learning rate schedule
        initial_learning_rate = 0.001
        lr_schedule = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=0
        )

        # Early stopping to prevent overfitting
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )

        # Compile with Adam optimizer with custom learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']  # Add MAE as additional metric
        )

        # Train the model
        self.history = self.model.fit(
            X_train_normalized,
            X_train_normalized,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            validation_split=0.15,  # Slightly larger validation split
            callbacks=[lr_schedule, early_stopping],
            shuffle=True
        )

        # Compute reconstruction errors for threshold calculation
        X_pred = self.model.predict(X_train_normalized, verbose=0)

        # Calculate reconstruction error using multiple metrics
        mse_errors = np.mean((X_train_normalized - X_pred) ** 2, axis=1)
        mae_errors = np.mean(np.abs(X_train_normalized - X_pred), axis=1)

        # Combine errors (weighted average of MSE and MAE)
        combined_errors = 0.7 * mse_errors + 0.3 * mae_errors

        # Set threshold using the specified percentile
        self.threshold = np.percentile(combined_errors, threshold_percentile)

        # Store some statistics for potential debugging
        self.error_stats = {
            'mean': np.mean(combined_errors),
            'std': np.std(combined_errors),
            'min': np.min(combined_errors),
            'max': np.max(combined_errors),
            'threshold': self.threshold
        }

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict anomalies in test data.

        Args:
            X_test: Test data to check for anomalies

        Returns:
            Binary array where 1 indicates anomaly, 0 indicates normal
        """
        if self.model is None or self.threshold is None or self.X_train is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Normalize test data using training statistics
        X_test_normalized = self._normalize(X_test, fit=False)

        # Get predictions
        X_pred = self.model.predict(X_test_normalized, verbose=0)

        # Calculate reconstruction errors (same combination as in training)
        mse_errors = np.mean((X_test_normalized - X_pred) ** 2, axis=1)
        mae_errors = np.mean(np.abs(X_test_normalized - X_pred), axis=1)
        combined_errors = 0.7 * mse_errors + 0.3 * mae_errors

        # Return binary predictions
        return (combined_errors > self.threshold).astype(int)

    def get_anomaly_scores(self, X_test: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores (reconstruction errors) for test data.

        Args:
            X_test: Test data

        Returns:
            Array of anomaly scores (higher = more anomalous)
        """
        if self.model is None or self.X_train is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Normalize test data
        X_test_normalized = self._normalize(X_test, fit=False)

        # Get predictions
        X_pred = self.model.predict(X_test_normalized, verbose=0)

        # Calculate and return reconstruction errors
        mse_errors = np.mean((X_test_normalized - X_pred) ** 2, axis=1)
        mae_errors = np.mean(np.abs(X_test_normalized - X_pred), axis=1)
        combined_errors = 0.7 * mse_errors + 0.3 * mae_errors

        return combined_errors

    def get_encoder_output(self, X: np.ndarray) -> np.ndarray:
        """
        Get the encoded representation of input data.

        Args:
            X: Input data

        Returns:
            Encoded representation from the bottleneck layer
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Create encoder model
        encoder = models.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('bottleneck').output
        )

        # Normalize and encode
        X_normalized = self._normalize(X, fit=False)
        return encoder.predict(X_normalized, verbose=0)
