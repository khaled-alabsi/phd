from sklearn.decomposition import PCA
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class PCA_Autoencoder_Hybrid:
    """
    Two-tier hybrid detector with adaptive resource allocation for computational efficiency.
    Tier 1 (PCA) performs fast screening; Tier 2 (Autoencoder) does deep analysis only when triggered.
    Best for: Real-time monitoring where computational resources are limited but high accuracy is needed.
    """
    def __init__(self, pca_components=0.95, pca_threshold_percentile=90,
                 ae_hidden_dims=[32, 16], ae_threshold_percentile=95):
        """
        Initialize two-tier hybrid detector with fast PCA screening and deep autoencoder analysis.

        Parameters:
        - pca_components: PCA variance to retain (0.90-0.99). Start with 0.95 (95% variance).
                         Higher = more accurate Tier 1 screening but more computation.
        - pca_threshold_percentile: PCA SPE percentile to trigger Tier 2 (80-95). Start with 85-90.
                                   Lower = more samples go to deep analysis = slower but more sensitive.
                                   Tune to balance speed vs detection rate (aim for ~10-20% triggering).
        - ae_hidden_dims: Autoencoder architecture as list [layer1_size, layer2_size, ...]. Start with [32, 16].
                         Deeper/wider = better patterns but slower. Should decrease: [64, 32] or [32, 16, 8].
        - ae_threshold_percentile: Final anomaly threshold percentile (90-99). Start with 95.
                                  Higher = fewer false alarms. Only applies to Tier 2 triggered samples.
        """
        self.pca_components = pca_components
        self.pca_threshold_percentile = pca_threshold_percentile
        self.ae_hidden_dims = ae_hidden_dims
        self.ae_threshold_percentile = ae_threshold_percentile

        self.pca = None
        self.pca_threshold = None
        self.autoencoder = None
        self.ae_threshold = None

    def _build_autoencoder(self, input_dim):
        """Build autoencoder architecture"""
        # Encoder
        encoder_input = keras.Input(shape=(input_dim,))
        x = encoder_input
        for hidden_dim in self.ae_hidden_dims:
            x = layers.Dense(hidden_dim, activation='relu')(x)

        # Decoder
        for hidden_dim in reversed(self.ae_hidden_dims[:-1]):
            x = layers.Dense(hidden_dim, activation='relu')(x)
        decoder_output = layers.Dense(input_dim)(x)

        # Autoencoder model
        autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
        autoencoder.compile(optimizer='adam', loss='mse')

        return autoencoder

    def fit(self, X_train, ae_epochs=30, ae_batch_size=32, verbose=0):
        """
        Train both PCA (Tier 1 screener) and Autoencoder (Tier 2 analyzer) on normal data.
        X_train: Fault-free training data (n_samples, n_features). Should represent normal operation only.
        ae_epochs: Autoencoder training epochs (20-50). Start with 30. More for complex processes.
        ae_batch_size: Autoencoder batch size (32-128). 64 is typical. Adjust based on dataset size.
        """
        # Tier 1: Fit PCA
        self.pca = PCA(n_components=self.pca_components)
        X_pca = self.pca.fit_transform(X_train)

        # Calculate PCA reconstruction error (SPE)
        X_reconstructed_pca = self.pca.inverse_transform(X_pca)
        pca_spe = np.sum((X_train - X_reconstructed_pca)**2, axis=1)
        self.pca_threshold = np.percentile(pca_spe, self.pca_threshold_percentile)

        # Tier 2: Build and fit Autoencoder
        input_dim = X_train.shape[1]
        self.autoencoder = self._build_autoencoder(input_dim)
        self.autoencoder.fit(
            X_train, X_train,
            epochs=ae_epochs,
            batch_size=ae_batch_size,
            verbose=verbose,
            validation_split=0.1
        )

        # Calculate AE reconstruction error threshold
        X_reconstructed_ae = self.autoencoder.predict(X_train, verbose=0)
        ae_errors = np.mean((X_train - X_reconstructed_ae)**2, axis=1)
        self.ae_threshold = np.percentile(ae_errors, self.ae_threshold_percentile)

        return self

    def predict(self, X_test):
        """
        Two-tier adaptive detection: PCA screens all samples, autoencoder analyzes only suspicious ones.
        X_test: Test data (n_samples, n_features).
        Returns: Binary array where 1 = anomaly (confirmed by Tier 2 or obvious to Tier 1), 0 = normal.
        Note: Most normal samples only use fast PCA (Tier 1), saving ~70-90% computation time.
        """
        # Tier 1: PCA screening
        X_pca = self.pca.transform(X_test)
        X_reconstructed_pca = self.pca.inverse_transform(X_pca)
        pca_spe = np.sum((X_test - X_reconstructed_pca)**2, axis=1)

        # Initialize predictions
        predictions = np.zeros(len(X_test), dtype=int)

        # Identify samples that exceed PCA threshold (require deep analysis)
        trigger_mask = pca_spe > self.pca_threshold

        if np.any(trigger_mask):
            # Tier 2: Autoencoder analysis for triggered samples
            X_triggered = X_test[trigger_mask]
            X_reconstructed_ae = self.autoencoder.predict(X_triggered, verbose=0)
            ae_errors = np.mean((X_triggered - X_reconstructed_ae)**2, axis=1)

            # Mark as anomaly if AE error exceeds threshold
            predictions[trigger_mask] = np.where(ae_errors > self.ae_threshold, 1, 0)

        return predictions
