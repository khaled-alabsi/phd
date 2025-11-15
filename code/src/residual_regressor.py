from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers


@dataclass
class RegressorArtifacts:
    config_path: Path
    model_path: Path
    metrics_path: Path
    scaler_path: Path


class ResidualRegressor:
    """
    Regressor that predicts autoencoder reconstruction residuals.

    Workflow:
    1. Provide a trained autoencoder model
    2. fit() - Train regressor to predict residuals (with optional grid search)
    3. predict() - Get residual predictions for anomaly detection

    Features:
      * Automatic caching - models are saved and loaded to avoid retraining
      * Grid search support for regressor hyperparameters
      * Residual diagnostics and plotting
      * Target standardization for better training

    Example:
        >>> from src.flexible_autoencoder import FlexibleAutoencoder
        >>>
        >>> # Step 1: Train autoencoder separately
        >>> ae = FlexibleAutoencoder()
        >>> ae.fit(X_train, X_val, param_grid={...})
        >>>
        >>> # Step 2: Create regressor with trained autoencoder
        >>> regressor = ResidualRegressor(autoencoder=ae.autoencoder, verbose=1)
        >>>
        >>> # Step 3: Train regressor
        >>> regressor.fit(X_train, X_val, param_grid={...})
        >>>
        >>> # Step 4: Predict
        >>> residuals = regressor.predict(X_test)
    """

    DEFAULT_PARAM_GRID: Dict[str, Sequence[Any]] = {
        "hidden_layers": [
            (512, 256, 128),
            (512, 256, 128, 64),
            (256, 128, 64),
        ],
        "activation": ["relu", "selu"],
        "dropout": [0.0, 0.1, 0.3],
        "learning_rate": [1e-3, 5e-4, 1e-4],
        "batch_size": [128, 256],
        "epochs": [120, 180],
        "patience": [20],
    }

    TARGET_EPS = 1e-8

    def __init__(
        self,
        autoencoder: Optional[models.Model] = None,
        model_dir: Optional[Path] = None,
        cache_prefix: str = "residual_regressor",
        verbose: int = 0,
    ) -> None:
        """
        Initialize ResidualRegressor.

        Parameters
        ----------
        autoencoder
            Pre-trained autoencoder model. Can also be set later with set_autoencoder().
        model_dir
            Directory to save/load models. Defaults to code_v2/models/
        cache_prefix
            Prefix for cached model files. Use different prefixes for different experiments.
        verbose
            Verbosity level (0=silent, 1=progress messages).
        """
        base_dir = Path(__file__).resolve().parent.parent
        self.model_dir: Path = model_dir or (base_dir / "models")
        self.cache_prefix = cache_prefix
        self.verbose = verbose

        self.autoencoder: Optional[models.Model] = autoencoder

        self.model: Optional[models.Model] = None
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_metrics: Optional[Dict[str, Any]] = None

        self._target_scaler: Optional[Dict[str, float]] = None

        self.model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def set_autoencoder(self, autoencoder: models.Model) -> None:
        """
        Set the autoencoder model.

        Parameters
        ----------
        autoencoder
            Pre-trained autoencoder model.
        """
        self.autoencoder = autoencoder

    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        *,
        param_grid: Optional[Dict[str, Sequence[Any]]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        reset: bool = False,
    ) -> models.Model:
        """
        Train (or load) the residual regressor.

        The autoencoder MUST be set first (via __init__ or set_autoencoder()).

        Parameters
        ----------
        X_train
            Training features (same data used for autoencoder).
        X_val
            Validation features (same data used for autoencoder).
        param_grid
            Grid of hyperparameters for the regressor. If None, uses DEFAULT_PARAM_GRID.
        model_config
            Single config dict to skip grid search. Cannot be used with param_grid.
        reset
            If True, force retrain even if cached model exists.

        Returns
        -------
        models.Model
            Trained regressor model.

        Raises
        ------
        RuntimeError
            If autoencoder has not been set.
        """
        # Check that autoencoder is set
        if self.autoencoder is None:
            raise RuntimeError(
                "Autoencoder must be set first! "
                "Either pass it to __init__() or call set_autoencoder()."
            )

        artifacts = self._get_artifacts()
        if not reset and self._cache_exists(artifacts):
            self._load(artifacts)
            if self.verbose:
                print("[Regressor] Loaded cached model.")
            return self.model  # type: ignore[return-value]

        if model_config is not None and param_grid is not None:
            raise ValueError("Provide either model_config or param_grid, not both.")

        y_train_raw = self._compute_residuals(self.autoencoder, X_train)
        y_val_raw = self._compute_residuals(self.autoencoder, X_val)
        y_train, y_val = self._transform_targets(y_train_raw, y_val_raw)

        search_grid = param_grid or self.DEFAULT_PARAM_GRID

        if model_config is None:
            (
                self.best_config,
                self.model,
                self.best_metrics,
            ) = self._run_grid_search(
                X_train,
                y_train,
                y_train_raw,
                X_val,
                y_val,
                y_val_raw,
                search_grid,
            )
        else:
            cfg = self._normalize_config(model_config)
            model = self._build_regressor(X_train.shape[1], cfg)
            self._train_regressor(model, X_train, y_train, X_val, y_val, cfg)
            self.model = model
            self.best_config = cfg
            train_pred = self.predict(X_train)
            val_pred = self.predict(X_val)
            self.best_metrics = {
                "train": self._compute_metrics(y_train_raw, train_pred),
                "val": self._compute_metrics(y_val_raw, val_pred),
            }

        if self.model is None or self.best_config is None or self.best_metrics is None:
            raise RuntimeError("Regressor training failed to produce a model.")

        self._save(artifacts)
        return self.model

    def evaluate(self, X: np.ndarray, y_true: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate regressor performance.

        Parameters
        ----------
        X
            Input features.
        y_true
            True residuals. If None, computed from autoencoder.

        Returns
        -------
        Dict[str, float]
            Metrics (MAE, MSE, RMSE).
        """
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not set.")
        if self.model is None:
            raise RuntimeError("Regressor not trained.")

        target = (
            np.asarray(y_true).ravel()
            if y_true is not None
            else self._compute_residuals(self.autoencoder, X)
        )
        preds = self.predict(X)
        return self._compute_metrics(target, preds)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict residuals.

        Parameters
        ----------
        X
            Input features.

        Returns
        -------
        np.ndarray
            Predicted residuals.
        """
        if self.model is None:
            raise RuntimeError("Regressor not trained.")
        preds = self.model.predict(X, verbose=0).ravel()
        return self._inverse_transform_targets(preds)

    def autoencoder_residuals(self, X: np.ndarray) -> np.ndarray:
        """
        Get true autoencoder residuals.

        Parameters
        ----------
        X
            Input features.

        Returns
        -------
        np.ndarray
            True autoencoder residuals.
        """
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not set.")
        return self._compute_residuals(self.autoencoder, X)

    def plot_residual_fit(
        self,
        X: np.ndarray,
        *,
        y_true: Optional[np.ndarray] = None,
        sample_size: Optional[int] = None,
        random_state: Optional[int] = None,
        show: bool = True,
    ):
        """
        Plot scatter of true vs predicted residuals.

        Parameters
        ----------
        X
            Input features.
        y_true
            True residuals. If None, computed from autoencoder.
        sample_size
            Max number of samples to plot.
        random_state
            Random seed for sampling.
        show
            Whether to display the plot.
        """
        if self.autoencoder is None or self.model is None:
            raise RuntimeError("Autoencoder and regressor must be set/trained.")

        import matplotlib.pyplot as plt

        true_residuals = (
            np.asarray(y_true).ravel()
            if y_true is not None
            else self._compute_residuals(self.autoencoder, X)
        )
        pred_residuals = self.predict(X)

        if sample_size and len(true_residuals) > sample_size:
            rng = np.random.RandomState(random_state)
            idx = rng.choice(len(true_residuals), sample_size, replace=False)
            true_residuals = true_residuals[idx]
            pred_residuals = pred_residuals[idx]

        plt.figure(figsize=(8, 6))
        plt.scatter(true_residuals, pred_residuals, alpha=0.5, s=10)
        lim = [
            min(true_residuals.min(), pred_residuals.min()),
            max(true_residuals.max(), pred_residuals.max()),
        ]
        plt.plot(lim, lim, "r--", lw=2, label="Perfect fit")
        plt.xlabel("True Residual", fontsize=12)
        plt.ylabel("Predicted Residual", fontsize=12)
        plt.title("Residual Fit: True vs Predicted", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        if show:
            plt.show()

    def plot_residual_series(
        self,
        X: np.ndarray,
        *,
        agg: str = "mean",
        title: str = "Residual Timeline",
        show: bool = True,
    ):
        """
        Plot time series of residuals.

        Parameters
        ----------
        X
            Input features.
        agg
            Aggregation method for per-feature residuals (mean, max, median).
        title
            Plot title.
        show
            Whether to display the plot.
        """
        if self.autoencoder is None or self.model is None:
            raise RuntimeError("Autoencoder and regressor must be set/trained.")

        import matplotlib.pyplot as plt

        true_residuals = self._compute_residuals(self.autoencoder, X)
        pred_residuals = self.predict(X)

        plt.figure(figsize=(12, 5))
        plt.plot(true_residuals, label="True Residual", alpha=0.7)
        plt.plot(pred_residuals, label="Predicted Residual", alpha=0.7)
        plt.xlabel("Sample Index", fontsize=12)
        plt.ylabel("Residual Value", fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        if show:
            plt.show()

    # ------------------------------------------------------------------ #
    # Residual Computation                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_residuals(autoencoder: models.Model, X: np.ndarray) -> np.ndarray:
        """Compute mean reconstruction residuals."""
        recon = autoencoder.predict(X, verbose=0)
        squared_errors = np.square(X - recon)
        return np.mean(squared_errors, axis=1)

    def _transform_targets(
        self, y_train: np.ndarray, y_val: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Standardize targets for better training."""
        y_train = np.asarray(y_train).ravel()
        y_val = np.asarray(y_val).ravel()

        mean = float(np.mean(y_train))
        std = float(np.std(y_train))
        if std < self.TARGET_EPS:
            std = 1.0

        self._target_scaler = {"mean": mean, "std": std}

        y_train_scaled = (y_train - mean) / std
        y_val_scaled = (y_val - mean) / std
        return y_train_scaled, y_val_scaled

    def _inverse_transform_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse standardization."""
        if self._target_scaler is None:
            return y_scaled
        mean = self._target_scaler["mean"]
        std = self._target_scaler["std"]
        return y_scaled * std + mean

    # ------------------------------------------------------------------ #
    # Model Building                                                      #
    # ------------------------------------------------------------------ #

    def _build_regressor(self, input_dim: int, cfg: Dict[str, Any]) -> models.Model:
        """Build regressor model."""
        hidden_layers = cfg["hidden_layers"]
        activation = cfg["activation"]
        dropout = cfg.get("dropout", 0.0)
        learning_rate = cfg["learning_rate"]

        inp = layers.Input(shape=(input_dim,), name="input")
        x = inp
        for i, units in enumerate(hidden_layers):
            x = layers.Dense(units, activation=activation, name=f"dense_{i}")(x)
            if dropout > 0:
                x = layers.Dropout(dropout, name=f"dropout_{i}")(x)
        output = layers.Dense(1, activation="linear", name="output")(x)

        model = models.Model(inp, output, name="residual_regressor")
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        return model

    def _train_regressor(
        self,
        model: models.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        cfg: Dict[str, Any],
    ) -> None:
        """Train regressor."""
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

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            callbacks=cb,
            verbose=0,
        )

    def _run_grid_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        y_train_raw: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        y_val_raw: np.ndarray,
        param_grid: Dict[str, Sequence[Any]],
    ) -> Tuple[Dict[str, Any], models.Model, Dict[str, Any]]:
        """Run grid search over hyperparameters."""
        best_score = float("inf")
        best_config = None
        best_model = None
        best_metrics = None

        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]

        for combo in product(*values):
            config = dict(zip(keys, combo))
            config = self._normalize_config(config)

            if self.verbose:
                print(f"[Grid Search] Testing config: {config}")

            model = self._build_regressor(X_train.shape[1], config)
            self._train_regressor(model, X_train, y_train, X_val, y_val, config)

            train_pred_raw = self._inverse_transform_targets(
                model.predict(X_train, verbose=0).ravel()
            )
            val_pred_raw = self._inverse_transform_targets(
                model.predict(X_val, verbose=0).ravel()
            )

            train_metrics = self._compute_metrics(y_train_raw, train_pred_raw)
            val_metrics = self._compute_metrics(y_val_raw, val_pred_raw)
            score = val_metrics["mae"]

            if self.verbose:
                print(f"[Grid Search] val_mae={score:.6f}")

            if score < best_score:
                best_score = score
                best_config = config
                best_model = model
                best_metrics = {"train": train_metrics, "val": val_metrics}

            # Clear memory
            if model != best_model:
                del model
            tf.keras.backend.clear_session()

        if best_config is None or best_model is None or best_metrics is None:
            raise RuntimeError("Grid search failed to find a valid configuration.")

        if self.verbose:
            print(f"[Grid Search] Best config: {best_config}")
            print(f"[Grid Search] Best val_mae: {best_score:.6f}")

        return best_config, best_model, best_metrics

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute regression metrics."""
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        mae = float(np.mean(np.abs(y_true - y_pred)))
        mse = float(np.mean(np.square(y_true - y_pred)))
        rmse = float(np.sqrt(mse))
        return {"mae": mae, "mse": mse, "rmse": rmse}

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

    def _get_artifacts(self) -> RegressorArtifacts:
        prefix = self.cache_prefix
        return RegressorArtifacts(
            config_path=self.model_dir / f"{prefix}_config.json",
            model_path=self.model_dir / f"{prefix}_model.keras",
            metrics_path=self.model_dir / f"{prefix}_metrics.json",
            scaler_path=self.model_dir / f"{prefix}_scaler.json",
        )

    def _cache_exists(self, artifacts: RegressorArtifacts) -> bool:
        return (
            artifacts.config_path.exists()
            and artifacts.model_path.exists()
            and artifacts.scaler_path.exists()
        )

    def _load(self, artifacts: RegressorArtifacts) -> None:
        with artifacts.config_path.open("r", encoding="utf-8") as fh:
            config = json.load(fh)
        self.best_config = self._normalize_config(config)
        self.model = tf.keras.models.load_model(artifacts.model_path)

        with artifacts.scaler_path.open("r", encoding="utf-8") as fh:
            self._target_scaler = json.load(fh)

        if artifacts.metrics_path.exists():
            with artifacts.metrics_path.open("r", encoding="utf-8") as fh:
                self.best_metrics = json.load(fh)

    def _save(self, artifacts: RegressorArtifacts) -> None:
        if self.model is None or self.best_config is None:
            raise RuntimeError("Regressor must be trained before saving.")

        serialized_cfg = self._serialize_config(self.best_config)
        with artifacts.config_path.open("w", encoding="utf-8") as fh:
            json.dump(serialized_cfg, fh, indent=2)

        self.model.save(artifacts.model_path)

        if self._target_scaler:
            with artifacts.scaler_path.open("w", encoding="utf-8") as fh:
                json.dump(self._target_scaler, fh, indent=2)

        if self.best_metrics:
            with artifacts.metrics_path.open("w", encoding="utf-8") as fh:
                json.dump(self.best_metrics, fh, indent=2)
