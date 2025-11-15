import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from scipy.stats import f
import warnings
warnings.filterwarnings('ignore')


class VAE_T2_Detector:
    """
    VAE + Hotelling's T² hybrid combining deep learning with statistical process control.
    Uses VAE to map data to latent space, then applies T² statistic for statistically-principled anomaly detection.
    Best for: Combining nonlinear feature learning with traditional SPC control limits and interpretable thresholds.
    """
    def __init__(self, latent_dim=10, hidden_units=64, threshold_alpha=0.01):
        """
        Initialize VAE-T² detector combining neural networks with statistical control charts.

        Parameters:
        - latent_dim: Latent space dimensions (5-20). Start with 10.
                     Lower = more compression/faster but loses information. Should be << n_features.
        - hidden_units: Neural network hidden layer size (32-128). Start with 64.
                       More units = better representation but slower training and overfitting risk.
        - threshold_alpha: T² control limit significance level (0.001-0.05). Start with 0.01 (99% confidence).
                          Lower alpha = stricter threshold = fewer false alarms but may miss faults.
                          0.01 is typical for industrial control (99% in-control samples pass).
        """
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        self.threshold_alpha = threshold_alpha
        self.model = None
        self.encoder = None
        self.decoder = None
        self.latent_mean = None
        self.latent_cov_inv = None
        self.control_limit = None

    def _build_model(self, input_dim):
        """Build VAE architecture"""
        # Encoder
        encoder_inputs = keras.Input(shape=(input_dim,))
        x = layers.Dense(self.hidden_units, activation='relu')(encoder_inputs)
        x = layers.Dense(self.hidden_units // 2, activation='relu')(x)

        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], self.latent_dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

        # Decoder
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(self.hidden_units // 2, activation='relu')(latent_inputs)
        x = layers.Dense(self.hidden_units, activation='relu')(x)
        decoder_outputs = layers.Dense(input_dim)(x)

        self.decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')

        # VAE model
        outputs = self.decoder(z)
        self.model = keras.Model(encoder_inputs, outputs, name='vae')

        # VAE loss
        def vae_loss(y_true, y_pred):
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred), axis=1))
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            return reconstruction_loss + kl_loss

        self.model.compile(optimizer='adam', loss=vae_loss)

    def fit(self, X_train, epochs=50, batch_size=32, verbose=0):
        """
        Train VAE and compute T² control limits from latent space statistics on normal data.
        X_train: Fault-free training data (n_samples, n_features). Should represent normal operation only.
        epochs: Training epochs (30-100). Start with 30-50. More epochs may overfit small datasets.
        batch_size: Samples per update (32-128). 64 is typical. Smaller for small datasets.
        """
        input_dim = X_train.shape[1]
        self._build_model(input_dim)

        # Train VAE
        self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=0.1
        )

        # Get latent representations for training data
        z_mean, _, _ = self.encoder.predict(X_train, verbose=0)

        # Calculate latent space statistics
        self.latent_mean = np.mean(z_mean, axis=0)
        latent_cov = np.cov(z_mean.T)

        # Add small regularization for numerical stability
        latent_cov += np.eye(self.latent_dim) * 1e-6
        self.latent_cov_inv = np.linalg.inv(latent_cov)

        # Calculate T² control limit using F-distribution
        n = len(X_train)
        p = self.latent_dim
        f_critical = f.ppf(1 - self.threshold_alpha, p, n - p)
        self.control_limit = (p * (n - 1) * (n + 1) * f_critical) / (n * (n - p))

        return self

    def _calculate_t2_statistic(self, z):
        """Calculate Hotelling's T² statistic"""
        centered = z - self.latent_mean
        t2 = np.sum(centered @ self.latent_cov_inv * centered, axis=1)
        return t2

    def predict(self, X_test):
        """
        Detect anomalies using Hotelling's T² statistic in learned latent space.
        X_test: Test data (n_samples, n_features).
        Returns: Binary array where 1 = out-of-control (T² exceeds limit), 0 = in-control (T² within limit).
        """
        z_mean, _, _ = self.encoder.predict(X_test, verbose=0)
        t2_values = self._calculate_t2_statistic(z_mean)
        return np.where(t2_values > self.control_limit, 1, 0)
