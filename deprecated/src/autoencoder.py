# AUTOENCODER ANOMALY DETECTOR IMPLEMENTATION

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try importing TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Please install: pip install tensorflow")


class AutoencoderDetector:
    """
    Deep Learning Autoencoder for anomaly detection.

    The autoencoder learns to reconstruct normal (in-control) data.
    Anomalies are detected when reconstruction error exceeds a threshold.
    """

    def __init__(
        self,
        encoding_dim: int = 10,
        hidden_layers: list = [32, 16],
        epochs: int = 50,
        batch_size: int = 32,
        threshold_percentile: float = 95.0,
        verbose: int = 0
    ):
        """
        Initialize the autoencoder detector.

        Args:
            encoding_dim: Dimension of the encoded representation
            hidden_layers: List of hidden layer sizes for encoder
            epochs: Number of training epochs
            batch_size: Batch size for training
            threshold_percentile: Percentile of training errors to use as threshold
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for AutoencoderDetector. Install with: pip install tensorflow")

        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_percentile = threshold_percentile
        self.verbose = verbose

        self.model = None
        self.encoder = None
        self.decoder = None
        self.threshold = None
        self.is_fitted = False
        self.input_dim = None
        self.history = None

    def _build_model(self, input_dim: int) -> None:
        """
        Build the autoencoder architecture.

        Args:
            input_dim: Number of input features
        """
        self.input_dim = input_dim

        # Input layer
        input_layer = layers.Input(shape=(input_dim,))

        # Encoder
        encoded = input_layer
        for units in self.hidden_layers:
            encoded = layers.Dense(units, activation='relu')(encoded)

        # Bottleneck (encoding)
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='encoding')(encoded)

        # Decoder (mirror of encoder)
        decoded = encoded
        for units in reversed(self.hidden_layers):
            decoded = layers.Dense(units, activation='relu')(decoded)

        # Output layer
        decoded = layers.Dense(input_dim, activation='linear')(decoded)

        # Build full autoencoder
        self.model = keras.Model(inputs=input_layer, outputs=decoded, name='autoencoder')

        # Build encoder model for getting encoded representations
        self.encoder = keras.Model(inputs=input_layer, outputs=encoded, name='encoder')

        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        if self.verbose > 0:
            print(f"Autoencoder Architecture:")
            print(f"  Input: {input_dim} features")
            print(f"  Encoder layers: {self.hidden_layers}")
            print(f"  Encoding dim: {self.encoding_dim}")
            print(f"  Total parameters: {self.model.count_params():,}")

    def fit(self, X_incontrol: NDArray[np.float64], validation_split: float = 0.2) -> 'AutoencoderDetector':
        """
        Train the autoencoder on in-control data.

        Args:
            X_incontrol: In-control (normal) training data
            validation_split: Fraction of training data to use for validation

        Returns:
            Self for method chaining
        """
        X_incontrol = np.asarray(X_incontrol)

        # Build model if not already built
        if self.model is None:
            self._build_model(X_incontrol.shape[1])

        if self.verbose > 0:
            print(f"\nTraining autoencoder on {len(X_incontrol)} samples...")

        # Train autoencoder
        self.history = self.model.fit(
            X_incontrol,
            X_incontrol,  # Target is to reconstruct input
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            verbose=self.verbose,
            shuffle=True
        )

        # Calculate reconstruction errors on training data
        train_reconstructions = self.model.predict(X_incontrol, verbose=0)
        train_errors = np.mean(np.square(X_incontrol - train_reconstructions), axis=1)

        # Set threshold based on training errors
        self.threshold = np.percentile(train_errors, self.threshold_percentile)

        self.is_fitted = True

        if self.verbose > 0:
            final_loss = self.history.history['loss'][-1]
            print(f"Training complete!")
            print(f"  Final training loss: {final_loss:.6f}")
            print(f"  Detection threshold (MSE): {self.threshold:.6f}")
            print(f"  Threshold percentile: {self.threshold_percentile}%")

        return self

    def predict(self, X_test: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """
        Predict anomalies in test data.

        Args:
            X_test: Test data to check for anomalies

        Returns:
            Tuple of (reconstruction_errors, anomaly_flags)
            - reconstruction_errors: MSE for each sample
            - anomaly_flags: Boolean array (True = anomaly)
        """
        if not self.is_fitted:
            raise ValueError("Autoencoder must be fitted before prediction")

        X_test = np.asarray(X_test)

        # Get reconstructions
        reconstructions = self.model.predict(X_test, verbose=0)

        # Calculate reconstruction errors (MSE per sample)
        reconstruction_errors = np.mean(np.square(X_test - reconstructions), axis=1)

        # Flag anomalies based on threshold
        anomaly_flags = reconstruction_errors > self.threshold

        return reconstruction_errors, anomaly_flags

    def get_reconstruction_error(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Get reconstruction error for each sample without flagging.

        Args:
            X: Data to reconstruct

        Returns:
            Array of reconstruction errors (MSE)
        """
        if not self.is_fitted:
            raise ValueError("Autoencoder must be fitted before getting reconstruction error")

        X = np.asarray(X)
        reconstructions = self.model.predict(X, verbose=0)
        return np.mean(np.square(X - reconstructions), axis=1)

    def get_encoding(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Get encoded representation of data.

        Args:
            X: Data to encode

        Returns:
            Encoded representations
        """
        if not self.is_fitted:
            raise ValueError("Autoencoder must be fitted before encoding")

        return self.encoder.predict(X, verbose=0)

    def set_threshold(self, threshold: float) -> None:
        """
        Manually set the detection threshold.

        Args:
            threshold: New threshold value (MSE)
        """
        self.threshold = threshold
        if self.verbose > 0:
            print(f"Threshold updated to: {threshold:.6f}")

    def get_summary(self) -> dict:
        """
        Get summary of the autoencoder configuration and performance.

        Returns:
            Dictionary with model information
        """
        if not self.is_fitted:
            return {"status": "not fitted"}

        return {
            "status": "fitted",
            "input_dim": self.input_dim,
            "encoding_dim": self.encoding_dim,
            "hidden_layers": self.hidden_layers,
            "threshold": self.threshold,
            "threshold_percentile": self.threshold_percentile,
            "total_parameters": self.model.count_params(),
            "final_train_loss": self.history.history['loss'][-1] if self.history else None,
            "final_val_loss": self.history.history['val_loss'][-1] if self.history and 'val_loss' in self.history.history else None
        }


class LightweightAutoencoder:
    """
    Lightweight autoencoder using only NumPy (no deep learning frameworks).
    Uses a simple linear autoencoder with PCA-like behavior.
    """

    def __init__(self, encoding_dim: int = 10, threshold_percentile: float = 95.0):
        """
        Initialize lightweight autoencoder.

        Args:
            encoding_dim: Dimension of encoded representation
            threshold_percentile: Percentile for threshold
        """
        self.encoding_dim = encoding_dim
        self.threshold_percentile = threshold_percentile
        self.encoder_weights = None
        self.decoder_weights = None
        self.threshold = None
        self.is_fitted = False
        self.mean = None
        self.std = None

    def fit(self, X_incontrol: NDArray[np.float64]) -> 'LightweightAutoencoder':
        """
        Fit using PCA-based linear autoencoder.

        Args:
            X_incontrol: In-control training data

        Returns:
            Self
        """
        X = np.asarray(X_incontrol)

        # Normalize
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        X_normalized = (X - self.mean) / self.std

        # Compute covariance and eigenvectors (PCA)
        cov_matrix = np.cov(X_normalized.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort by eigenvalues (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Keep top encoding_dim components
        self.encoder_weights = eigenvectors[:, :self.encoding_dim]
        self.decoder_weights = self.encoder_weights.T

        # Calculate threshold from training errors
        reconstructions = self._reconstruct(X_normalized)
        errors = np.mean(np.square(X_normalized - reconstructions), axis=1)
        self.threshold = np.percentile(errors, self.threshold_percentile)

        self.is_fitted = True
        return self

    def _reconstruct(self, X_normalized: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reconstruct data."""
        encoded = X_normalized @ self.encoder_weights
        decoded = encoded @ self.decoder_weights
        return decoded

    def predict(self, X_test: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """
        Predict anomalies.

        Args:
            X_test: Test data

        Returns:
            Tuple of (errors, flags)
        """
        if not self.is_fitted:
            raise ValueError("Must fit before predict")

        X = np.asarray(X_test)
        X_normalized = (X - self.mean) / self.std

        reconstructions = self._reconstruct(X_normalized)
        errors = np.mean(np.square(X_normalized - reconstructions), axis=1)
        flags = errors > self.threshold

        return errors, flags
