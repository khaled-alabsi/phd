import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class LSTM_VAE_Detector:
    """
    LSTM-based Variational Autoencoder for dynamic process monitoring with temporal dependencies.
    Captures time-series patterns using LSTM networks combined with probabilistic VAE framework.
    Best for: Dynamic processes where temporal correlations and sequence patterns are critical (>95% FDR on TEP).
    """
    def __init__(self, sequence_length=10, latent_dim=10, lstm_units=64, threshold_percentile=95):
        """
        Initialize LSTM-VAE detector for time-series anomaly detection.

        Parameters:
        - sequence_length: Number of consecutive time steps per sequence (5-20 typical). Start with 10.
                          Longer sequences capture more temporal context but increase computation.
                          Should match the characteristic time scale of faults in your process.
        - latent_dim: Dimension of compressed latent representation (5-20). Start with 10.
                     Higher = more expressive but slower training. Should be < sequence_length.
        - lstm_units: Number of LSTM hidden units (32-128). Start with 64.
                     More units = better pattern learning but higher overfitting risk and training time.
        - threshold_percentile: Percentile of training reconstruction errors for threshold (90-99). Start with 95.
                               Higher = fewer false alarms but may miss subtle faults. Tune based on validation set.
        """
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.threshold_percentile = threshold_percentile
        self.model = None
        self.encoder = None
        self.decoder = None
        self.threshold = None

    def _build_model(self, input_dim):
        """Build LSTM-VAE architecture"""
        # Encoder
        encoder_inputs = keras.Input(shape=(self.sequence_length, input_dim))
        lstm_out = layers.LSTM(self.lstm_units, return_sequences=False)(encoder_inputs)

        # Latent space parameters
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(lstm_out)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(lstm_out)

        # Sampling layer
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], self.latent_dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])

        # Build encoder model
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

        # Decoder
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        repeated_latent = layers.RepeatVector(self.sequence_length)(latent_inputs)
        lstm_decoded = layers.LSTM(self.lstm_units, return_sequences=True)(repeated_latent)
        decoder_outputs = layers.TimeDistributed(layers.Dense(input_dim))(lstm_decoded)

        # Build decoder model
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')

        # VAE model
        outputs = self.decoder(z)
        self.model = keras.Model(encoder_inputs, outputs, name='lstm_vae')

        # VAE loss
        def vae_loss(y_true, y_pred):
            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(y_true - y_pred), axis=[1, 2])
            )
            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            return reconstruction_loss + kl_loss

        self.model.compile(optimizer='adam', loss=vae_loss)

    def _create_sequences(self, X):
        """Convert data to sequences for LSTM"""
        sequences = []
        for i in range(len(X) - self.sequence_length + 1):
            sequences.append(X[i:i+self.sequence_length])
        return np.array(sequences)

    def fit(self, X_train, epochs=50, batch_size=32, verbose=0):
        """
        Train LSTM-VAE on normal operation sequences. Automatically creates sequences from data.
        X_train: Fault-free training data (n_samples, n_features). Should be continuous time-series from normal operation.
        epochs: Training epochs (30-100). More epochs improve fit but risk overfitting. Start with 30-50.
        batch_size: Samples per gradient update (32-128). Larger = faster but less stable. 64 is typical.
        """
        # Build model
        input_dim = X_train.shape[1]
        self._build_model(input_dim)

        # Create sequences
        X_train_seq = self._create_sequences(X_train)

        # Train the model
        self.model.fit(
            X_train_seq, X_train_seq,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=0.1
        )

        # Calculate threshold from training data
        train_errors = self._calculate_reconstruction_errors(X_train_seq)
        self.threshold = np.percentile(train_errors, self.threshold_percentile)

        return self

    def _calculate_reconstruction_errors(self, X_seq):
        """Calculate reconstruction errors for sequences"""
        reconstructed = self.model.predict(X_seq, verbose=0)
        errors = np.mean(np.square(X_seq - reconstructed), axis=(1, 2))
        return errors

    def predict(self, X_test):
        """
        Detect anomalies by comparing reconstruction errors to learned normal patterns.
        X_test: Test data (n_samples, n_features). Automatically creates overlapping sequences.
        Returns: Binary array where 1 = fault (high reconstruction error), 0 = normal (low reconstruction error).
        """
        # Create sequences
        X_test_seq = self._create_sequences(X_test)

        # Calculate reconstruction errors
        errors = self._calculate_reconstruction_errors(X_test_seq)

        # Classify based on threshold
        predictions = np.where(errors > self.threshold, 1, 0)

        # Map sequence predictions back to original samples
        # Use max pooling: if any sequence containing a sample is anomaly, mark as anomaly
        result = np.zeros(len(X_test), dtype=int)
        for i in range(len(predictions)):
            result[i:i+self.sequence_length] = np.maximum(
                result[i:i+self.sequence_length],
                predictions[i]
            )

        return result
