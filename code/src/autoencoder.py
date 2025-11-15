"""Utilities for training and running a dense autoencoder anomaly detector with grid search and caching."""

import json
import tensorflow as tf
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from tensorflow.keras import layers, models, callbacks
import numpy as np
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


@dataclass
class AutoencoderArtifacts:
    """File paths for cached model artifacts."""
    config_path: Path
    model_path: Path
    metrics_path: Path
    threshold_path: Path


class AutoencoderDetector:
    """Autoencoder-based anomaly detector with thresholding, grid search, and caching."""

    # Default configuration
    DEFAULT_CONFIG: Dict[str, Any] = {
        "encoding_dim": 8,
        "hidden_dim": 16,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "patience": 10,
        "threshold_percentile": 95,
    }

    # Default grid search parameters
    DEFAULT_PARAM_GRID: Dict[str, Sequence[Any]] = {
        "encoding_dim": [8, 16, 24],
        "hidden_dim": [16, 32],
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
        cache_prefix: str = "autoencoder",
        verbose: int = 0,
    ):
        """
        Create a detector with a configurable latent vector size.

        Parameters
        ----------
        encoding_dim
            Number of neurons in the latent (bottleneck) layer (used if no grid search).
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
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_metrics: Optional[Dict[str, Any]] = None

        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _build_model(self, input_dim: int, config: Dict[str, Any]) -> models.Model:
        """Build autoencoder architecture."""
        encoding_dim = int(config.get("encoding_dim", 8))
        hidden_dim = int(config.get("hidden_dim", 16))
        learning_rate = float(config.get("learning_rate", 0.001))

        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(hidden_dim, activation='relu')(input_layer)
        encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
        decoded = layers.Dense(hidden_dim, activation='relu')(encoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)

        model = models.Model(inputs=input_layer, outputs=decoded)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

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
        # Legacy parameters for backward compatibility
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        threshold_percentile: Optional[float] = None,
    ) -> models.Model:
        """
        Train the autoencoder and compute an anomaly threshold.

        Parameters
        ----------
        X_train
            Training data with shape (n_samples, n_features).
        validation_split
            Fraction of training data to use for validation (only if grid search is used).
        param_grid
            Parameter grid for grid search. If None and no legacy params, uses DEFAULT_PARAM_GRID.
            Mutually exclusive with model_config.
        model_config
            Single configuration to train. Mutually exclusive with param_grid.
        reset_experiment
            If True, force retraining even if cached model exists.
        epochs
            LEGACY: Number of training epochs (overrides config if provided).
        batch_size
            LEGACY: Mini-batch size (overrides config if provided).
        threshold_percentile
            LEGACY: Percentile for threshold (overrides config if provided).

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

        # Handle legacy API
        use_legacy = any(p is not None for p in [epochs, batch_size, threshold_percentile])
        if use_legacy:
            # Legacy mode: single config from parameters
            config = {
                "encoding_dim": self.encoding_dim,
                "hidden_dim": 16,
                "learning_rate": 0.001,
                "batch_size": batch_size or 32,
                "epochs": epochs or 50,
                "patience": 10,
                "threshold_percentile": threshold_percentile or 95,
            }
            model_config = config

        # Mutually exclusive parameters
        if model_config is not None and param_grid is not None:
            raise ValueError("Provide either model_config or param_grid, not both.")

        self.X_train = X_train

        # Split validation data for grid search
        n_val = int(len(X_train) * validation_split)
        X_val = X_train[-n_val:]
        X_train_split = X_train[:-n_val]

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
        X_pred = self.model.predict(X_train, verbose=0)
        errors = np.mean((X_train - X_pred) ** 2, axis=1)

        threshold_percentile = float(self.best_config.get("threshold_percentile", 95))
        self.threshold = np.percentile(errors, threshold_percentile)

        # Save to cache
        self._save_model(artifacts)

        if self.verbose:
            print(f"[Model] Training complete. Threshold: {self.threshold:.6f}")

        return self.model

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict anomalies for new samples using the trained autoencoder.

        Parameters
        ----------
        X_test
            Samples to evaluate with shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Binary array where 1 denotes an anomaly (error above threshold) and 0 denotes normal data.
        """
        if self.model is None or self.threshold is None or self.X_train is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        X_pred = self.model.predict(X_test, verbose=0)
        errors = np.mean((X_test - X_pred) ** 2, axis=1)
        return (errors > self.threshold).astype(int)

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
        for key in ["encoding_dim", "hidden_dim", "batch_size", "epochs", "patience"]:
            if key in cfg:
                cfg[key] = int(cfg[key])
        for key in ["learning_rate", "threshold_percentile"]:
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
            threshold_path=self.model_dir / f"{prefix}_threshold.json",
        )

    def _cache_exists(self, artifacts: AutoencoderArtifacts) -> bool:
        """Check if all required cache files exist."""
        return (
            artifacts.config_path.exists()
            and artifacts.model_path.exists()
            and artifacts.threshold_path.exists()
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

        # Load threshold
        with artifacts.threshold_path.open("r", encoding="utf-8") as f:
            threshold_data = json.load(f)
            self.threshold = float(threshold_data["threshold"])

        # Mark as trained
        self.X_train = np.empty((0, 0))  # Placeholder

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

        # Save threshold
        threshold_data = {
            "threshold": float(self.threshold),
        }
        with artifacts.threshold_path.open("w", encoding="utf-8") as f:
            json.dump(threshold_data, f, indent=2)
