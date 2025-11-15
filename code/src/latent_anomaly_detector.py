from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers


@dataclass
class AnomalyDetectorArtifacts:
    config_path: Path
    model_path: Path
    metrics_path: Path
    grid_results_path: Path


class LatentAnomalyDetector:
    """
    Deep neural network anomaly detector operating on latent representations.

    Architecture:
        Raw Features → Encoder (frozen) → Latent Vector → DNN → Anomaly Score (0.0-1.0)

    Features:
    - Train with grid search or single config
    - Semi-supervised learning (requires labels: 0=normal, 1=anomaly)
    - Automatic caching to avoid retraining
    - Continuous anomaly scores via predict_proba()
    - Binary classification via predict()

    Example:
        >>> # Get encoder from trained autoencoder
        >>> encoder = autoencoder.get_encoder()
        >>>
        >>> # Create anomaly detector
        >>> detector = LatentAnomalyDetector(encoder=encoder, verbose=1)
        >>>
        >>> # Train with grid search
        >>> detector.fit(X_train, y_train, X_val, y_val)
        >>>
        >>> # Get anomaly scores
        >>> scores = detector.predict_proba(X_test)
        >>>
        >>> # Get binary predictions
        >>> labels = detector.predict(X_test, threshold=0.5)
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "hidden_layers": (128, 64),
        "activation": "relu",
        "dropout_rate": 0.2,
        "learning_rate": 1e-3,
        "batch_size": 256,
        "epochs": 100,
        "patience": 15,
    }

    DEFAULT_PARAM_GRID: Dict[str, Sequence[Any]] = {
        "hidden_layers": [
            (64, 32),
            (128, 64),
            (256, 128, 64),
            (512, 256, 128, 64),
        ],
        "activation": ["relu", "selu"],
        "dropout_rate": [0.0, 0.2, 0.3],
        "learning_rate": [1e-3, 5e-4, 1e-4],
        "batch_size": [128, 256],
        "epochs": [100, 150],
        "patience": [15, 20],
    }

    def __init__(
        self,
        encoder: models.Model,
        model_dir: Optional[Path] = None,
        cache_prefix: str = "latent_anomaly_detector",
        verbose: int = 0,
    ) -> None:
        """
        Initialize LatentAnomalyDetector.

        Parameters
        ----------
        encoder
            Trained encoder model from FlexibleAutoencoder. Used to transform
            raw features to latent representations. Will be frozen during training.
        model_dir
            Directory to save/load models. Defaults to code/models/
        cache_prefix
            Prefix for cached model files. Use different prefixes for different experiments.
        verbose
            Verbosity level (0=silent, 1=progress messages).
        """
        base_dir = Path(__file__).resolve().parent.parent
        self.model_dir: Path = model_dir or (base_dir / "models")
        self.cache_prefix = cache_prefix
        self.verbose = verbose

        self.encoder = encoder
        self.encoder.trainable = False  # Freeze encoder

        self.model: Optional[models.Model] = None
        self.config: Optional[Dict[str, Any]] = None
        self.metrics: Optional[Dict[str, float]] = None
        self.grid_results: List[Dict[str, Any]] = []

        self.model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        *,
        param_grid: Optional[Dict[str, Sequence[Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
        reset: bool = False,
    ) -> models.Model:
        """
        Train (or load) the anomaly detector.

        Parameters
        ----------
        X_train
            Training features (raw, will be encoded internally).
        y_train
            Training labels (0=normal, 1=anomaly).
        X_val
            Validation features (raw, will be encoded internally).
        y_val
            Validation labels (0=normal, 1=anomaly).
        param_grid
            Grid of hyperparameters for grid search. If None, uses DEFAULT_PARAM_GRID.
        config
            Single config dict to skip grid search. Cannot be used with param_grid.
        reset
            If True, force retrain even if cached model exists.

        Returns
        -------
        models.Model
            Trained anomaly detector model.
        """
        artifacts = self._get_artifacts()
        if not reset and self._cache_exists(artifacts):
            self._load(artifacts)
            if self.verbose:
                print("[LatentAnomalyDetector] Loaded cached model.")
            return self.model  # type: ignore[return-value]

        if config is not None and param_grid is not None:
            raise ValueError("Provide either config or param_grid, not both.")

        # Encode features to latent representations
        if self.verbose:
            print("[LatentAnomalyDetector] Encoding features to latent space...")
        latent_train = self.encoder.predict(X_train, verbose=0)
        latent_val = self.encoder.predict(X_val, verbose=0)

        if config is not None:
            cfg = self._normalize_config(config)
            model, metrics = self._train_single(latent_train, y_train, latent_val, y_val, cfg)
        else:
            grid = param_grid or self.DEFAULT_PARAM_GRID
            cfg, model, metrics = self._run_grid_search(
                latent_train, y_train, latent_val, y_val, grid
            )

        self.model = model
        self.config = cfg
        self.metrics = metrics
        self._save(artifacts)
        return model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get continuous anomaly scores (0.0-1.0).

        Parameters
        ----------
        X : np.ndarray
            Input data (raw features, will be encoded internally).

        Returns
        -------
        np.ndarray
            Anomaly scores between 0.0 (normal) and 1.0 (anomaly).
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        latent = self.encoder.predict(X, verbose=0)
        scores = self.model.predict(latent, verbose=0)
        return scores.flatten()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Get binary anomaly predictions (0=normal, 1=anomaly).

        Parameters
        ----------
        X : np.ndarray
            Input data (raw features, will be encoded internally).
        threshold : float
            Decision threshold (default: 0.5). Scores >= threshold are classified as anomalies.

        Returns
        -------
        np.ndarray
            Binary labels (0=normal, 1=anomaly).
        """
        scores = self.predict_proba(X)
        return (scores >= threshold).astype(int)

    def get_latent(self, X: np.ndarray) -> np.ndarray:
        """
        Get latent representations for debugging/analysis.

        Parameters
        ----------
        X : np.ndarray
            Input data (raw features).

        Returns
        -------
        np.ndarray
            Latent representations.
        """
        return self.encoder.predict(X, verbose=0)

    def get_encoder(self) -> models.Model:
        """
        Get the encoder model.

        Returns
        -------
        models.Model
            Encoder model.
        """
        return self.encoder

    def plot_score_distribution(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        *,
        bins: int = 50,
        figsize: tuple = (12, 5),
        threshold: Optional[float] = None,
    ) -> None:
        """
        Plot anomaly score distributions for normal and anomalous samples.

        This plot is useful for:
        - Assessing model separation between normal and anomaly classes
        - Selecting optimal decision thresholds
        - Identifying model confidence and overlap regions
        - Comparing train vs validation performance

        Parameters
        ----------
        X_train : np.ndarray
            Training data (raw features).
        y_train : np.ndarray
            Training labels (0=normal, 1=anomaly).
        X_val : np.ndarray
            Validation data (raw features).
        y_val : np.ndarray
            Validation labels (0=normal, 1=anomaly).
        bins : int
            Number of histogram bins (default: 50).
        figsize : tuple
            Figure size (default: (12, 5)).
        threshold : float, optional
            Decision threshold to display as vertical line.

        Example
        -------
        >>> detector.fit(X_train, y_train, X_val, y_val)
        >>> detector.plot_score_distribution(X_train, y_train, X_val, y_val, threshold=0.5)
        """
        import matplotlib.pyplot as plt

        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Get scores
        train_scores = self.predict_proba(X_train)
        val_scores = self.predict_proba(X_val)

        # Separate by label
        train_normal = train_scores[y_train == 0]
        train_anomaly = train_scores[y_train == 1]
        val_normal = val_scores[y_val == 0]
        val_anomaly = val_scores[y_val == 1]

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Training set
        ax1.hist(train_normal, bins=bins, alpha=0.6, label="Normal", color="green")
        ax1.hist(train_anomaly, bins=bins, alpha=0.6, label="Anomaly", color="red")
        if threshold is not None:
            ax1.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold={threshold}")
        ax1.set_title("Training Set - Anomaly Score Distribution")
        ax1.set_xlabel("Anomaly Score")
        ax1.set_ylabel("Count")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Validation set
        ax2.hist(val_normal, bins=bins, alpha=0.6, label="Normal", color="green")
        ax2.hist(val_anomaly, bins=bins, alpha=0.6, label="Anomaly", color="red")
        if threshold is not None:
            ax2.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold={threshold}")
        ax2.set_title("Validation Set - Anomaly Score Distribution")
        ax2.set_xlabel("Anomaly Score")
        ax2.set_ylabel("Count")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print statistics
        print(f"\nAnomaly Score Statistics:")
        print(f"  Train Normal:  mean={np.mean(train_normal):.4f}, std={np.std(train_normal):.4f}")
        print(f"  Train Anomaly: mean={np.mean(train_anomaly):.4f}, std={np.std(train_anomaly):.4f}")
        print(f"  Val Normal:    mean={np.mean(val_normal):.4f}, std={np.std(val_normal):.4f}")
        print(f"  Val Anomaly:   mean={np.mean(val_anomaly):.4f}, std={np.std(val_anomaly):.4f}")

    def plot_summary(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        threshold: Optional[float] = 0.5,
    ) -> None:
        """
        Display comprehensive training summary with plots.

        Parameters
        ----------
        X_train : np.ndarray
            Training data (raw features).
        y_train : np.ndarray
            Training labels.
        X_val : np.ndarray
            Validation data (raw features).
        y_val : np.ndarray
            Validation labels.
        threshold : float, optional
            Decision threshold to display (default: 0.5).

        Example
        -------
        >>> detector.fit(X_train, y_train, X_val, y_val, param_grid=GRID)
        >>> detector.plot_summary(X_train, y_train, X_val, y_val)
        """
        if self.config is None or self.metrics is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Print config and metrics
        print(f"\n{'='*60}")
        print(f"Best Configuration:")
        print(f"{'='*60}")
        for key, value in self.config.items():
            print(f"  {key}: {value}")

        print(f"\n{'='*60}")
        print(f"Performance Metrics:")
        print(f"{'='*60}")
        for key, value in self.metrics.items():
            print(f"  {key}: {value:.4f}")
        print(f"{'='*60}\n")

        # Plot score distributions
        self.plot_score_distribution(
            X_train, y_train, X_val, y_val, threshold=threshold
        )

    # ------------------------------------------------------------------ #
    # Model Building                                                      #
    # ------------------------------------------------------------------ #

    def _build_model(self, latent_dim: int, cfg: Dict[str, Any]) -> models.Model:
        """
        Build DNN anomaly detector.

        Parameters
        ----------
        latent_dim : int
            Dimension of latent space.
        cfg : Dict[str, Any]
            Configuration with architecture parameters.

        Returns
        -------
        models.Model
            Compiled DNN model.
        """
        hidden_layers = cfg["hidden_layers"]
        activation = cfg["activation"]
        dropout_rate = cfg["dropout_rate"]

        # Build sequential model
        model_input = layers.Input(shape=(latent_dim,), name="latent_input")
        x = model_input

        # Hidden layers
        for i, units in enumerate(hidden_layers):
            x = layers.Dense(units, activation=activation, name=f"dense_{i}")(x)
            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate, name=f"dropout_{i}")(x)

        # Output layer (sigmoid for binary classification)
        output = layers.Dense(1, activation="sigmoid", name="output")(x)

        model = models.Model(model_input, output, name="latent_anomaly_detector")

        # Compile with binary cross-entropy
        optimizer = optimizers.Adam(learning_rate=cfg["learning_rate"])
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )

        return model

    def _train_single(
        self,
        latent_train: np.ndarray,
        y_train: np.ndarray,
        latent_val: np.ndarray,
        y_val: np.ndarray,
        cfg: Dict[str, Any],
    ) -> Tuple[models.Model, Dict[str, float]]:
        """Train a single model configuration."""
        latent_dim = latent_train.shape[1]
        model = self._build_model(latent_dim, cfg)

        cb = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg["patience"],
                restore_best_weights=True,
                verbose=0,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=cfg["patience"] // 2,
                verbose=0,
            ),
        ]

        history = model.fit(
            latent_train,
            y_train,
            validation_data=(latent_val, y_val),
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            callbacks=cb,
            verbose=0,
        )

        # Get best metrics
        best_epoch = np.argmin(history.history["val_loss"])
        metrics = {
            "train_loss": float(history.history["loss"][best_epoch]),
            "val_loss": float(history.history["val_loss"][best_epoch]),
            "train_accuracy": float(history.history["accuracy"][best_epoch]),
            "val_accuracy": float(history.history["val_accuracy"][best_epoch]),
            "train_auc": float(history.history["auc"][best_epoch]),
            "val_auc": float(history.history["val_auc"][best_epoch]),
        }

        return model, metrics

    def _run_grid_search(
        self,
        latent_train: np.ndarray,
        y_train: np.ndarray,
        latent_val: np.ndarray,
        y_val: np.ndarray,
        param_grid: Dict[str, Sequence[Any]],
    ) -> Tuple[Dict[str, Any], models.Model, Dict[str, float]]:
        """Run grid search over hyperparameters."""
        self.grid_results = []
        best_score = -float("inf")  # Maximize val_auc
        best_config = None
        best_model = None
        best_metrics = None

        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]

        total_combinations = np.prod([len(v) for v in values])
        if self.verbose:
            print(f"[Grid Search] Testing {total_combinations} configurations...")

        for i, combo in enumerate(product(*values), 1):
            config = dict(zip(keys, combo))
            config = self._normalize_config(config)

            if self.verbose:
                print(f"[Grid Search] [{i}/{total_combinations}] Testing: {config}")

            model, metrics = self._train_single(
                latent_train, y_train, latent_val, y_val, config
            )
            score = metrics["val_auc"]

            self.grid_results.append({
                "config": config.copy(),
                "metrics": metrics.copy(),
                "score": score,
            })

            if score > best_score:
                best_score = score
                best_config = config
                best_model = model
                best_metrics = metrics

            if self.verbose:
                print(f"[Grid Search] val_auc={score:.6f}")

            # Clear memory
            if model != best_model:
                del model
            tf.keras.backend.clear_session()

        if best_config is None or best_model is None or best_metrics is None:
            raise RuntimeError("Grid search failed to find a valid configuration.")

        if self.verbose:
            print(f"\n[Grid Search] Best config: {best_config}")
            print(f"[Grid Search] Best val_auc: {best_score:.6f}")

        return best_config, best_model, best_metrics

    # ------------------------------------------------------------------ #
    # Config Normalization                                                #
    # ------------------------------------------------------------------ #

    def _normalize_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize config to ensure correct types."""
        normalized = cfg.copy()
        if "hidden_layers" in normalized:
            layers_cfg = normalized["hidden_layers"]
            if isinstance(layers_cfg, (list, tuple)):
                normalized["hidden_layers"] = tuple(int(v) for v in layers_cfg)
        return normalized

    def _serialize_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize config for JSON storage."""
        serialized = cfg.copy()
        if "hidden_layers" in serialized:
            serialized["hidden_layers"] = list(serialized["hidden_layers"])
        return serialized

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #

    def _get_artifacts(self) -> AnomalyDetectorArtifacts:
        prefix = self.cache_prefix
        return AnomalyDetectorArtifacts(
            config_path=self.model_dir / f"{prefix}_config.json",
            model_path=self.model_dir / f"{prefix}_model.keras",
            metrics_path=self.model_dir / f"{prefix}_metrics.json",
            grid_results_path=self.model_dir / f"{prefix}_grid_results.json",
        )

    def _cache_exists(self, artifacts: AnomalyDetectorArtifacts) -> bool:
        return artifacts.config_path.exists() and artifacts.model_path.exists()

    def _load(self, artifacts: AnomalyDetectorArtifacts) -> None:
        with artifacts.config_path.open("r", encoding="utf-8") as fh:
            config = json.load(fh)
        self.config = self._normalize_config(config)
        self.model = tf.keras.models.load_model(artifacts.model_path)

        if artifacts.metrics_path.exists():
            with artifacts.metrics_path.open("r", encoding="utf-8") as fh:
                self.metrics = {k: float(v) for k, v in json.load(fh).items()}

        if artifacts.grid_results_path.exists():
            with artifacts.grid_results_path.open("r", encoding="utf-8") as fh:
                serialized_results = json.load(fh)
            self.grid_results = []
            for result in serialized_results:
                normalized_result = {
                    "config": self._normalize_config(result["config"]),
                    "metrics": result["metrics"],
                    "score": result["score"],
                }
                self.grid_results.append(normalized_result)

    def _save(self, artifacts: AnomalyDetectorArtifacts) -> None:
        if self.model is None or self.config is None:
            raise RuntimeError("Model must be trained before saving.")

        serialized_cfg = self._serialize_config(self.config)
        with artifacts.config_path.open("w", encoding="utf-8") as fh:
            json.dump(serialized_cfg, fh, indent=2)

        self.model.save(artifacts.model_path)

        if self.metrics:
            with artifacts.metrics_path.open("w", encoding="utf-8") as fh:
                json.dump(self.metrics, fh, indent=2)

        if self.grid_results:
            serialized_results = []
            for result in self.grid_results:
                serialized_result = {
                    "config": self._serialize_config(result["config"]),
                    "metrics": result["metrics"],
                    "score": result["score"],
                }
                serialized_results.append(serialized_result)
            with artifacts.grid_results_path.open("w", encoding="utf-8") as fh:
                json.dump(serialized_results, fh, indent=2)

        if self.verbose:
            print(f"[LatentAnomalyDetector] Model saved to {self.model_dir}")
