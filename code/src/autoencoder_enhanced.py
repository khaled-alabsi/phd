"""Enhanced autoencoder anomaly detector with stronger regularization and diagnostics.

Compared to the baseline `AutoencoderDetector`, this variant adds input normalization,
denoising noise injection, batch normalization, dropout, and L1/L2 regularization.
It also includes grid search, model caching, and automatic saving/loading.
"""

import json
import tensorflow as tf
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from tensorflow.keras import layers, models, callbacks, regularizers
import numpy as np
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

@dataclass
class AutoencoderArtifacts:
    """File paths for cached model artifacts."""
    config_path: Path
    model_path: Path
    metrics_path: Path
    scaler_path: Path


class AutoencoderDetectorEnhanced:
    """Autoencoder detector with normalization, regularization, grid search, and caching."""

    # Default configuration
    DEFAULT_CONFIG: Dict[str, Any] = {
        "encoding_dim": 8,
        "noise_stddev": 0.1,
        "dropout_rate": 0.2,
        "l1_reg": 1e-5,
        "l2_reg": 1e-4,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "patience": 10,
        "threshold_percentile": 95,
    }

    # Default grid search parameters
    DEFAULT_PARAM_GRID: Dict[str, Sequence[Any]] = {
        "encoding_dim": [8, 16, 24],
        "noise_stddev": [0.05, 0.1, 0.15],
        "dropout_rate": [0.1, 0.2, 0.3],
        "learning_rate": [0.001, 0.0005],
        "batch_size": [32, 64],
        "epochs": [50, 80],
        "patience": [10, 15],
        "threshold_percentile": [95],
    }

    def __init__(
        self,
        encoding_dim: int = 8,
        model_dir: Optional[Path] = None,
        cache_prefix: str = "autoencoder_enhanced",
        verbose: int = 0,
    ):
        """
        Initialize the enhanced detector.

        Parameters
        ----------
        encoding_dim
            Dimensionality of the latent bottleneck representation (used if no grid search).
        model_dir
            Directory for saving/loading models. Defaults to ../models relative to this file.
        cache_prefix
            Prefix for cache filenames.
        verbose
            Verbosity level: 0=silent, 1=progress messages.
        """
        self.encoding_dim = encoding_dim
        base_dir = Path(__file__).resolve().parent.parent
        self.model_dir: Path = model_dir or (base_dir / "models")
        self.cache_prefix = cache_prefix
        self.verbose = verbose

        self.model: Optional[models.Model] = None
        self.threshold: Optional[float] = None
        self.X_train: Optional[np.ndarray] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.history = None
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_metrics: Optional[Dict[str, Any]] = None

        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize data using z-score scaling, fitting statistics when requested."""
        if fit:
            self.scaler_mean = np.mean(X, axis=0)
            self.scaler_std = np.std(X, axis=0) + 1e-7  # Add small epsilon to avoid division by zero

        if self.scaler_mean is None or self.scaler_std is None:
            raise RuntimeError("Scaler not fitted. This should not happen.")

        return (X - self.scaler_mean) / self.scaler_std

    def _build_model(self, input_dim: int, config: Dict[str, Any]) -> models.Model:
        """Build an improved autoencoder architecture with regularization."""
        encoding_dim = int(config.get("encoding_dim", 8))
        noise_stddev = float(config.get("noise_stddev", 0.1))
        dropout_rate = float(config.get("dropout_rate", 0.2))
        l1_reg = float(config.get("l1_reg", 1e-5))
        l2_reg = float(config.get("l2_reg", 1e-4))
        learning_rate = float(config.get("learning_rate", 0.001))

        # Calculate layer dimensions dynamically
        layer1_dim = max(input_dim // 2, encoding_dim * 4)
        layer2_dim = max(input_dim // 4, encoding_dim * 2)

        # Encoder
        input_layer = layers.Input(shape=(input_dim,))

        # Add noise for denoising autoencoder capability (helps with robustness)
        noisy_input = layers.GaussianNoise(noise_stddev)(input_layer)

        # Encoder layers with batch normalization and dropout for regularization
        encoded = layers.Dense(
            layer1_dim,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
            kernel_initializer='he_normal'
        )(noisy_input)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(dropout_rate)(encoded)

        encoded = layers.Dense(
            layer2_dim,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
            kernel_initializer='he_normal'
        )(encoded)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(dropout_rate)(encoded)

        # Bottleneck layer
        encoded = layers.Dense(
            encoding_dim,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
            kernel_initializer='he_normal',
            name='bottleneck'
        )(encoded)

        # Decoder layers (symmetric to encoder)
        decoded = layers.Dense(
            layer2_dim,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
            kernel_initializer='he_normal'
        )(encoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dropout(dropout_rate)(decoded)

        decoded = layers.Dense(
            layer1_dim,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
            kernel_initializer='he_normal'
        )(decoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dropout(dropout_rate)(decoded)

        # Output layer with tanh activation (works better with normalized data)
        decoded = layers.Dense(
            input_dim,
            activation='tanh',
            kernel_initializer='glorot_uniform'
        )(decoded)

        model = models.Model(inputs=input_layer, outputs=decoded)

        # Compile with Adam optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']  # Add MAE as additional metric
        )

        return model

    def _train_single(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        config: Dict[str, Any],
    ) -> Tuple[models.Model, Dict[str, Any]]:
        """Train a single model with given configuration."""
        tf.keras.backend.clear_session()

        input_dim = X_train.shape[1]
        model = self._build_model(input_dim, config)

        # Training parameters
        epochs = int(config.get("epochs", 50))
        batch_size = int(config.get("batch_size", 32))
        patience = int(config.get("patience", 10))

        # Callbacks
        cb = []
        if patience > 0:
            cb.append(callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=0
            ))

        cb.append(callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=max(1, patience // 2),
            min_lr=1e-7,
            verbose=0
        ))

        # Train
        history = model.fit(
            X_train,
            X_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            validation_data=(X_val, X_val),
            callbacks=cb,
            shuffle=True
        )

        # Calculate metrics
        train_loss = float(np.min(history.history["loss"]))
        val_loss = float(np.min(history.history["val_loss"]))

        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        return model, metrics

    def _run_grid_search(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        param_grid: Dict[str, Sequence[Any]],
    ) -> Tuple[Dict[str, Any], models.Model, Dict[str, Any]]:
        """Run grid search to find best hyperparameters."""
        best_score = np.inf
        best_config: Optional[Dict[str, Any]] = None
        best_metrics: Optional[Dict[str, Any]] = None
        best_weights: Optional[list] = None

        configs = list(self._iter_param_grid(param_grid))
        total_configs = len(configs)

        if self.verbose:
            print(f"[GridSearch] Testing {total_configs} configurations...")

        for idx, cfg in enumerate(configs, 1):
            config = self._normalize_config(cfg)
            model, metrics = self._train_single(X_train, X_val, config)

            score = metrics["val_loss"]

            if self.verbose:
                print(f"[{idx}/{total_configs}] val_loss={score:.6f} | config={config}")

            if score < best_score:
                best_score = score
                best_config = config
                best_metrics = metrics
                best_weights = model.get_weights()

        if best_config is None or best_weights is None or best_metrics is None:
            raise RuntimeError("Grid search failed to find a valid model.")

        # Rebuild best model
        tf.keras.backend.clear_session()
        input_dim = X_train.shape[1]
        best_model = self._build_model(input_dim, best_config)
        best_model.set_weights(best_weights)

        if self.verbose:
            print(f"[GridSearch] Best config: {best_config}")
            print(f"[GridSearch] Best val_loss: {best_score:.6f}")

        return best_config, best_model, best_metrics

    def fit(
        self,
        X_train: np.ndarray,
        *,
        validation_split: float = 0.15,
        param_grid: Optional[Dict[str, Sequence[Any]]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        reset_experiment: bool = False,
    ) -> models.Model:
        """
        Train the autoencoder on normal data with optional grid search.

        Parameters
        ----------
        X_train
            Training matrix of predominantly normal samples, shape (n_samples, n_features).
        validation_split
            Fraction of training data to use for validation (only if grid search is used).
        param_grid
            Parameter grid for grid search. If None, uses DEFAULT_PARAM_GRID.
            Mutually exclusive with model_config.
        model_config
            Single configuration to train. Mutually exclusive with param_grid.
        reset_experiment
            If True, force retraining even if cached model exists.

        Returns
        -------
        models.Model
            The trained Keras model.
        """
        # Check cache first (unless resetting)
        artifacts = self._get_artifacts()
        if not reset_experiment and self._cache_exists(artifacts):
            self._load_model(artifacts)
            if self.verbose:
                print("[Model] Loaded cached model.")
            return self.model  # type: ignore[return-value]

        # Mutually exclusive parameters
        if model_config is not None and param_grid is not None:
            raise ValueError("Provide either model_config or param_grid, not both.")

        # Normalize the training data
        X_train_normalized = self._normalize(X_train, fit=True)
        self.X_train = X_train_normalized

        # Split validation data for grid search
        n_val = int(len(X_train_normalized) * validation_split)
        X_val = X_train_normalized[-n_val:]
        X_train_split = X_train_normalized[:-n_val]

        # Determine configuration
        if model_config is not None:
            # Single config training
            config = self._normalize_config(model_config)
            model, metrics = self._train_single(X_train_split, X_val, config)
            self.best_config = config
            self.best_metrics = metrics
            self.model = model
        else:
            # Grid search
            grid = param_grid if param_grid is not None else self.DEFAULT_PARAM_GRID
            config, model, metrics = self._run_grid_search(X_train_split, X_val, grid)
            self.best_config = config
            self.best_metrics = metrics
            self.model = model

        # Compute threshold
        X_pred = self.model.predict(X_train_normalized, verbose=0)

        # Calculate reconstruction errors (same as predict logic)
        mse_errors = np.mean((X_train_normalized - X_pred) ** 2, axis=1)
        mae_errors = np.mean(np.abs(X_train_normalized - X_pred), axis=1)
        combined_errors = 0.7 * mse_errors + 0.3 * mae_errors

        threshold_percentile = float(self.best_config.get("threshold_percentile", 95))
        self.threshold = np.percentile(combined_errors, threshold_percentile)

        # Store statistics
        self.error_stats = {
            'mean': float(np.mean(combined_errors)),
            'std': float(np.std(combined_errors)),
            'min': float(np.min(combined_errors)),
            'max': float(np.max(combined_errors)),
            'threshold': float(self.threshold)
        }

        # Save to cache
        self._save_model(artifacts)

        if self.verbose:
            print(f"[Model] Training complete. Threshold: {self.threshold:.6f}")

        return self.model

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict anomalies in test data.

        Parameters
        ----------
        X_test
            Test samples to classify, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Binary array where 1 indicates anomaly, 0 indicates normal behaviour.
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

        Parameters
        ----------
        X_test
            Samples to evaluate, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Combined reconstruction errors where higher values indicate more anomalous samples.
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

        Parameters
        ----------
        X
            Input samples to transform, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Bottleneck activations representing the latent embeddings of the inputs.
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

    # ------------------------------------------------------------------ #
    # Utility methods                                                   #
    # ------------------------------------------------------------------ #

    def _iter_param_grid(self, grid: Dict[str, Sequence[Any]]) -> Iterable[Dict[str, Any]]:
        """Generate all combinations of parameters."""
        keys = sorted(grid.keys())
        for values in product(*(grid[k] for k in keys)):
            yield {k: v for k, v in zip(keys, values)}

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and validate configuration."""
        cfg = dict(config)
        # Ensure numeric types
        for key in ["encoding_dim", "batch_size", "epochs", "patience"]:
            if key in cfg:
                cfg[key] = int(cfg[key])
        for key in ["noise_stddev", "dropout_rate", "l1_reg", "l2_reg", "learning_rate", "threshold_percentile"]:
            if key in cfg:
                cfg[key] = float(cfg[key])
        return cfg

    # ------------------------------------------------------------------ #
    # Persistence (Caching)                                             #
    # ------------------------------------------------------------------ #

    def _get_artifacts(self) -> AutoencoderArtifacts:
        """Get artifact file paths."""
        prefix = self.cache_prefix
        return AutoencoderArtifacts(
            config_path=self.model_dir / f"{prefix}_config.json",
            model_path=self.model_dir / f"{prefix}_model.keras",
            metrics_path=self.model_dir / f"{prefix}_metrics.json",
            scaler_path=self.model_dir / f"{prefix}_scaler.json",
        )

    def _cache_exists(self, artifacts: AutoencoderArtifacts) -> bool:
        """Check if all required cache files exist."""
        return (
            artifacts.config_path.exists()
            and artifacts.model_path.exists()
            and artifacts.scaler_path.exists()
        )

    def _load_model(self, artifacts: AutoencoderArtifacts) -> None:
        """Load model and configuration from cache."""
        # Load config
        with artifacts.config_path.open("r", encoding="utf-8") as f:
            self.best_config = json.load(f)

        # Load model
        self.model = tf.keras.models.load_model(artifacts.model_path)

        # Load metrics (optional)
        if artifacts.metrics_path.exists():
            with artifacts.metrics_path.open("r", encoding="utf-8") as f:
                self.best_metrics = json.load(f)

        # Load scaler
        with artifacts.scaler_path.open("r", encoding="utf-8") as f:
            scaler_data = json.load(f)
            self.scaler_mean = np.array(scaler_data["mean"])
            self.scaler_std = np.array(scaler_data["std"])
            self.threshold = float(scaler_data["threshold"])

            # Load error stats if available
            if "error_stats" in scaler_data:
                self.error_stats = scaler_data["error_stats"]

        # Reconstruct X_train (normalized) - we don't save this, so mark as loaded
        self.X_train = np.empty((0, len(self.scaler_mean)))  # Placeholder

    def _save_model(self, artifacts: AutoencoderArtifacts) -> None:
        """Save model and configuration to cache."""
        if self.model is None or self.best_config is None:
            raise RuntimeError("Model must be trained before saving.")

        # Save config
        with artifacts.config_path.open("w", encoding="utf-8") as f:
            json.dump(self.best_config, f, indent=2)

        # Save model
        self.model.save(artifacts.model_path)

        # Save metrics
        if self.best_metrics:
            with artifacts.metrics_path.open("w", encoding="utf-8") as f:
                json.dump(self.best_metrics, f, indent=2)

        # Save scaler and threshold
        scaler_data = {
            "mean": self.scaler_mean.tolist(),
            "std": self.scaler_std.tolist(),
            "threshold": float(self.threshold),
        }
        if hasattr(self, "error_stats"):
            scaler_data["error_stats"] = self.error_stats

        with artifacts.scaler_path.open("w", encoding="utf-8") as f:
            json.dump(scaler_data, f, indent=2)
