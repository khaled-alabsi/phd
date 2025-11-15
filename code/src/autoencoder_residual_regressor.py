from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers


@dataclass
class AutoencoderArtifacts:
    config_path: Path
    model_path: Path
    metrics_path: Path
    grid_results_path: Path


@dataclass
class RegressorArtifacts:
    config_path: Path
    model_path: Path
    metrics_path: Path
    scaler_path: Path


class AutoencoderResidualRegressor:
    """
    Autoencoder teacher + regressor student that predicts reconstruction residuals.

    Workflow:
    1. train_autoencoder() - Train the teacher autoencoder (with optional grid search)
    2. fit() - Train the student regressor to predict autoencoder residuals (with optional grid search)
    3. predict() - Get residual predictions for anomaly detection

    Features:
      * Automatic caching - models are saved and loaded to avoid retraining
      * Grid search support for both teacher and student
      * Bottleneck optimization with elbow curve visualization
      * Residual diagnostics and plotting

    Example:
        >>> regressor = AutoencoderResidualRegressor(verbose=1)
        >>> # Step 1: Train teacher
        >>> regressor.train_autoencoder(X_train, X_val, autoencoder_param_grid={...})
        >>> # Step 2: Train student
        >>> regressor.fit(X_train, X_val, param_grid={...})
        >>> # Step 3: Predict
        >>> residuals = regressor.predict(X_test)
    """

    DEFAULT_AUTOENCODER_CONFIG: Dict[str, Any] = {
        "encoder_layers": (256, 128),
        "latent_dim": 64,
        "activation": "relu",
        "learning_rate": 1e-3,
        "batch_size": 256,
        "epochs": 120,
        "patience": 15,
    }

    DEFAULT_AUTOENCODER_PARAM_GRID: Dict[str, Sequence[Any]] = {
        "encoder_layers": [
            (256, 128),
            (512, 256, 128),
            (512, 256, 128, 64),
        ],
        "latent_dim": [32, 64, 96],  # Or use generate_bottleneck_grid() for dynamic sizing
        "activation": ["relu", "selu"],
        "learning_rate": [1e-3, 5e-4],
        "batch_size": [128, 256],
        "epochs": [120, 180],
        "patience": [15, 25],
    }

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
        model_dir: Optional[Path] = None,
        cache_prefix: str = "autoencoder_residual_regressor",
        verbose: int = 0,
    ) -> None:
        base_dir = Path(__file__).resolve().parent.parent
        self.model_dir: Path = model_dir or (base_dir / "models")
        self.cache_prefix = cache_prefix
        self.verbose = verbose

        self.autoencoder: Optional[models.Model] = None
        self.autoencoder_config: Optional[Dict[str, Any]] = None
        self.autoencoder_metrics: Optional[Dict[str, float]] = None

        self.model: Optional[models.Model] = None
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_metrics: Optional[Dict[str, Any]] = None

        self._target_scaler: Optional[Dict[str, float]] = None

        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Track grid search results for elbow analysis
        self.autoencoder_grid_results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Bottleneck Size Optimization                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def generate_bottleneck_range(input_dim: int, min_ratio: float = 8.0, max_ratio: float = 4.0) -> List[int]:
        """
        Generate bottleneck size range from input_dim/min_ratio to input_dim/max_ratio.

        Parameters
        ----------
        input_dim
            Number of input features.
        min_ratio
            Minimum ratio (larger value = smaller bottleneck). Default 8.0 → input_dim/8.
        max_ratio
            Maximum ratio (smaller value = larger bottleneck). Default 4.0 → input_dim/4.

        Returns
        -------
        List[int]
            List of bottleneck sizes to test.

        Example
        -------
        >>> generate_bottleneck_range(52)  # For 52 features
        [6, 7, 8, 9, 10, 11, 12, 13]  # From 52/8 to 52/4
        """
        min_size = max(4, int(input_dim / min_ratio))
        max_size = int(input_dim / max_ratio)

        # Generate range with reasonable step
        if max_size - min_size <= 8:
            # Small range: test every value
            return list(range(min_size, max_size + 1))
        else:
            # Large range: test every 2nd value
            return list(range(min_size, max_size + 1, 2))

    @staticmethod
    def find_elbow_point(x: np.ndarray, y: np.ndarray) -> int:
        """
        Find the elbow point in a curve using the maximum distance method.

        The elbow is the point with maximum perpendicular distance from
        the line connecting the first and last points.

        Parameters
        ----------
        x
            X-axis values (e.g., bottleneck sizes).
        y
            Y-axis values (e.g., validation losses).

        Returns
        -------
        int
            Index of the elbow point.
        """
        # Normalize to [0, 1] for both axes
        x_norm = (x - x[0]) / (x[-1] - x[0])
        y_norm = (y - y[0]) / (y[-1] - y[0])

        # Calculate perpendicular distance from each point to the line (0,0)-(1,1)
        distances = np.abs(x_norm - y_norm) / np.sqrt(2)

        # Return index of maximum distance
        return int(np.argmax(distances))

    def plot_elbow_curve(
        self,
        results: Optional[List[Dict[str, Any]]] = None,
        metric: str = "val_loss",
        show: bool = True,
        save_path: Optional[Path] = None,
    ) -> Tuple[int, float]:
        """
        Plot elbow curve for bottleneck size selection.

        Parameters
        ----------
        results
            List of grid search results. If None, uses self.autoencoder_grid_results.
        metric
            Metric to plot ('val_loss' or 'train_loss').
        show
            Whether to display the plot.
        save_path
            Optional path to save the figure.

        Returns
        -------
        optimal_size : int
            Optimal bottleneck size at elbow point.
        optimal_value : float
            Metric value at elbow point.
        """
        import matplotlib.pyplot as plt

        if results is None:
            results = self.autoencoder_grid_results

        if not results:
            raise ValueError("No grid search results available. Run train_autoencoder with grid search first.")

        # Extract bottleneck sizes and metric values
        bottleneck_sizes = np.array([r["config"]["latent_dim"] for r in results])
        metric_values = np.array([r["metrics"][metric] for r in results])

        # Sort by bottleneck size
        sort_idx = np.argsort(bottleneck_sizes)
        bottleneck_sizes = bottleneck_sizes[sort_idx]
        metric_values = metric_values[sort_idx]

        # Find elbow point
        elbow_idx = self.find_elbow_point(bottleneck_sizes, metric_values)
        optimal_size = int(bottleneck_sizes[elbow_idx])
        optimal_value = float(metric_values[elbow_idx])

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot curve
        ax.plot(bottleneck_sizes, metric_values, 'bo-', linewidth=2, markersize=8, label='Actual')

        # Mark elbow point
        ax.plot(optimal_size, optimal_value, 'r*', markersize=20,
                label=f'Elbow (size={optimal_size}, {metric}={optimal_value:.4f})', zorder=5)

        # Draw line from first to last point
        ax.plot([bottleneck_sizes[0], bottleneck_sizes[-1]],
                [metric_values[0], metric_values[-1]],
                'g--', alpha=0.5, label='First-Last line')

        # Formatting
        ax.set_xlabel('Bottleneck Size (latent_dim)', fontsize=12)
        ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12)
        ax.set_title('Elbow Method: Optimal Bottleneck Size Selection', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        # Add annotations
        for size, value in zip(bottleneck_sizes, metric_values):
            ax.annotate(f'{value:.3f}',
                       xy=(size, value),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       fontsize=8,
                       alpha=0.7)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"[Elbow] Plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return optimal_size, optimal_value

    def generate_bottleneck_grid(
        self,
        input_dim: int,
        base_grid: Optional[Dict[str, Sequence[Any]]] = None,
        min_ratio: float = 8.0,
        max_ratio: float = 4.0,
    ) -> Dict[str, Sequence[Any]]:
        """
        Generate parameter grid with dynamic bottleneck range.

        Parameters
        ----------
        input_dim
            Number of input features.
        base_grid
            Base parameter grid to modify. If None, uses DEFAULT_AUTOENCODER_PARAM_GRID.
        min_ratio
            Minimum ratio for bottleneck size (default 8.0 → input_dim/8).
        max_ratio
            Maximum ratio for bottleneck size (default 4.0 → input_dim/4).

        Returns
        -------
        Dict[str, Sequence[Any]]
            Parameter grid with updated latent_dim range.
        """
        if base_grid is None:
            base_grid = self.DEFAULT_AUTOENCODER_PARAM_GRID.copy()
        else:
            base_grid = dict(base_grid)

        # Generate bottleneck range
        bottleneck_range = self.generate_bottleneck_range(input_dim, min_ratio, max_ratio)
        base_grid["latent_dim"] = bottleneck_range

        if self.verbose:
            print(f"[Bottleneck] Generated range for input_dim={input_dim}: {bottleneck_range}")
            print(f"[Bottleneck] Range: {min(bottleneck_range)} to {max(bottleneck_range)}")

        return base_grid

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def train_autoencoder(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        *,
        autoencoder_param_grid: Optional[Dict[str, Sequence[Any]]] = None,
        autoencoder_config: Optional[Dict[str, Any]] = None,
        reset_autoencoder: bool = False,
    ) -> models.Model:
        """
        Train (or load) the autoencoder teacher.

        Call this BEFORE calling fit() to train the student regressor.

        Parameters
        ----------
        X_train
            Training features for the autoencoder.
        X_val
            Validation features for the autoencoder.
        autoencoder_param_grid
            Grid of hyperparameters for grid search. If None, uses DEFAULT_AUTOENCODER_PARAM_GRID.
        autoencoder_config
            Single config dict to skip grid search. Cannot be used with autoencoder_param_grid.
        reset_autoencoder
            If True, force retrain even if cached model exists.

        Returns
        -------
        models.Model
            Trained autoencoder model.
        """
        artifacts = self._ae_artifacts()
        if not reset_autoencoder and self._autoencoder_cache_exists(artifacts):
            self._load_autoencoder(artifacts)
            if self.verbose:
                print("[Autoencoder] Loaded cached model.")
            return self.autoencoder  # type: ignore[return-value]

        if autoencoder_config is not None and autoencoder_param_grid is not None:
            raise ValueError("Provide either autoencoder_config or autoencoder_param_grid, not both.")

        if autoencoder_config is not None:
            config = self._normalise_autoencoder_config(autoencoder_config)
            model, metrics = self._train_autoencoder_single(X_train, X_val, config)
        else:
            param_grid = autoencoder_param_grid or self.DEFAULT_AUTOENCODER_PARAM_GRID
            config, model, metrics = self._run_autoencoder_grid_search(X_train, X_val, param_grid)

        self.autoencoder = model
        self.autoencoder_config = config
        self.autoencoder_metrics = metrics
        self._save_autoencoder(artifacts)
        return model

    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        *,
        param_grid: Optional[Dict[str, Sequence[Any]]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        reset_experiment: bool = False,
    ) -> models.Model:
        """
        Train (or load) the residual regressor student.

        The autoencoder teacher MUST be trained first using train_autoencoder().

        Parameters
        ----------
        X_train
            Training features (same data used for autoencoder).
        X_val
            Validation features (same data used for autoencoder).
        param_grid
            Grid of hyperparameters for the student regressor. If None, uses DEFAULT_PARAM_GRID.
        model_config
            Single config dict to skip grid search. Cannot be used with param_grid.
        reset_experiment
            If True, force retrain even if cached model exists.

        Returns
        -------
        models.Model
            Trained regressor model.

        Raises
        ------
        RuntimeError
            If autoencoder has not been trained yet.
        """
        # Check that autoencoder is trained
        if self.autoencoder is None:
            raise RuntimeError(
                "Autoencoder teacher must be trained first! "
                "Call train_autoencoder() before calling fit()."
            )

        reg_artifacts = self._reg_artifacts()
        if not reset_experiment and self._regressor_cache_exists(reg_artifacts):
            self._load_regressor(reg_artifacts)
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
            ) = self._run_regressor_grid_search(
                X_train,
                y_train,
                y_train_raw,
                X_val,
                y_val,
                y_val_raw,
                search_grid,
            )
        else:
            cfg = self._normalise_config(model_config)
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
            raise RuntimeError("Residual regressor training failed to produce a model.")

        self._save_regressor(reg_artifacts)
        return self.model

    def evaluate(self, X: np.ndarray, y_true: Optional[np.ndarray] = None) -> Dict[str, float]:
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not trained.")
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
        if self.model is None:
            raise RuntimeError("Regressor not trained.")
        preds = self.model.predict(X, verbose=0).ravel()
        return self._inverse_transform_targets(preds)

    def autoencoder_residuals(self, X: np.ndarray) -> np.ndarray:
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not trained.")
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
        if self.autoencoder is None or self.model is None:
            raise RuntimeError("Train both autoencoder and regressor before plotting.")

        import matplotlib.pyplot as plt

        true_residuals = (
            np.asarray(y_true).ravel()
            if y_true is not None
            else self.autoencoder_residuals(X)
        )
        predicted = self.predict(X)

        if sample_size is not None and sample_size < len(true_residuals):
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(true_residuals), size=sample_size, replace=False)
            true_residuals = true_residuals[idx]
            predicted = predicted[idx]

        ax = plt.gca()
        ax.scatter(true_residuals, predicted, alpha=0.4, s=12, label="samples")
        lo = float(min(true_residuals.min(), predicted.min()))
        hi = float(max(true_residuals.max(), predicted.max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="red", label="ideal")
        ax.set_xlabel("Autoencoder residual")
        ax.set_ylabel("Regressor residual")
        ax.set_title("Residual fit comparison")
        ax.legend()
        if show:
            plt.show()
        return ax

    def plot_residual_series(
        self,
        X: np.ndarray,
        *,
        y_true: Optional[np.ndarray] = None,
        sample_slice: Optional[slice] = None,
        title: Optional[str] = None,
        show: bool = True,
    ):
        """
        Plot residual timeline comparing autoencoder (teacher) vs regressor (student).

        Args:
            X: feature matrix.
            y_true: optional precomputed AE residuals; otherwise computed on the fly.
            sample_slice: optional slice selecting which samples to plot (e.g., slice(0, 200)).
            title: optional custom title.
            show: display plot via plt.show().
        """
        if self.autoencoder is None or self.model is None:
            raise RuntimeError("Train both autoencoder and regressor before plotting.")

        import matplotlib.pyplot as plt

        teacher = (
            np.asarray(y_true).ravel()
            if y_true is not None
            else self.autoencoder_residuals(X)
        )
        student = self.predict(X)

        if sample_slice is None:
            sample_slice = slice(0, len(student))

        teacher_sel = teacher[sample_slice]
        student_sel = student[sample_slice]
        idx = np.arange(len(student_sel))

        plt.figure(figsize=(10, 4))
        plt.plot(idx, teacher_sel, label="AE residual", color="grey", alpha=0.7)
        plt.plot(idx, student_sel, label="Regressor residual", color="steelblue")
        plt.xlabel("Sample index")
        plt.ylabel("Residual value")
        plt.title(title or "Residual timeline comparison")
        plt.legend()
        plt.grid(alpha=0.2)
        if show:
            plt.show()
        return teacher_sel, student_sel

    # ------------------------------------------------------------------ #
    # Autoencoder helpers                                                #
    # ------------------------------------------------------------------ #

    def _train_autoencoder_single(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        config: Dict[str, Any],
    ) -> Tuple[models.Model, Dict[str, float]]:
        tf.keras.backend.clear_session()
        model = self._build_autoencoder(X_train.shape[1], config)
        history = self._fit_autoencoder(model, X_train, X_val, config)

        train_loss = float(np.min(history.history["loss"]))
        val_loss = float(np.min(history.history["val_loss"]))
        metrics = {"train_loss": train_loss, "val_loss": val_loss}
        return model, metrics

    def _run_autoencoder_grid_search(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        param_grid: Dict[str, Sequence[Any]],
    ) -> Tuple[Dict[str, Any], models.Model, Dict[str, float]]:
        best_score = np.inf
        best_config: Optional[Dict[str, Any]] = None
        best_metrics: Optional[Dict[str, float]] = None
        best_weights: Optional[List[np.ndarray]] = None

        # Clear previous results
        self.autoencoder_grid_results = []

        for cfg in self._iter_param_grid(param_grid):
            config = self._normalise_autoencoder_config(cfg)
            model, metrics = self._train_autoencoder_single(X_train, X_val, config)
            score = metrics["val_loss"]

            # Store result for elbow analysis
            self.autoencoder_grid_results.append({
                "config": config.copy(),
                "metrics": metrics.copy(),
                "score": score,
            })

            if self.verbose:
                print(f"[Autoencoder] Config={config} -> val_loss={score:.6f}")
            if score < best_score:
                best_score = score
                best_config = config
                best_metrics = metrics
                best_weights = model.get_weights()

        if best_config is None or best_weights is None or best_metrics is None:
            raise RuntimeError("Autoencoder grid search failed.")

        tf.keras.backend.clear_session()
        best_model = self._build_autoencoder(X_train.shape[1], best_config)
        best_model.set_weights(best_weights)
        return best_config, best_model, best_metrics

    def _fit_autoencoder(
        self,
        model: models.Model,
        X_train: np.ndarray,
        X_val: np.ndarray,
        config: Dict[str, Any],
    ):
        patience = int(config.get("patience", 15))
        cb: List[callbacks.Callback] = []
        if patience > 0:
            cb.append(
                callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    restore_best_weights=True,
                    verbose=self.verbose,
                )
            )

        history = model.fit(
            X_train,
            X_train,
            validation_data=(X_val, X_val),
            epochs=int(config.get("epochs", 120)),
            batch_size=int(config.get("batch_size", 256)),
            verbose=self.verbose,
            callbacks=cb,
        )
        return history

    # ------------------------------------------------------------------ #
    # Regressor helpers                                                  #
    # ------------------------------------------------------------------ #

    def _run_regressor_grid_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        y_train_raw: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        y_val_raw: np.ndarray,
        param_grid: Dict[str, Sequence[Any]],
    ) -> Tuple[Dict[str, Any], models.Model, Dict[str, Any]]:
        input_dim = X_train.shape[1]
        best_score = np.inf
        best_config: Optional[Dict[str, Any]] = None
        best_metrics: Optional[Dict[str, Any]] = None
        best_weights: Optional[List[np.ndarray]] = None

        for cfg in self._iter_param_grid(param_grid):
            config = self._normalise_config(cfg)
            tf.keras.backend.clear_session()
            model = self._build_regressor(input_dim, config)
            self._train_regressor(model, X_train, y_train, X_val, y_val, config)

            train_pred = self._inverse_transform_targets(model.predict(X_train, verbose=0).ravel())
            val_pred = self._inverse_transform_targets(model.predict(X_val, verbose=0).ravel())

            train_metrics = self._compute_metrics(y_train_raw, train_pred)
            val_metrics = self._compute_metrics(y_val_raw, val_pred)
            score = val_metrics["mae"]

            if self.verbose:
                print(
                    f"[Regressor] Config={config} -> val_mae={score:.6f}, "
                    f"val_rmse={val_metrics['rmse']:.6f}"
                )

            if score < best_score:
                best_score = score
                best_config = config
                best_metrics = {"train": train_metrics, "val": val_metrics}
                best_weights = model.get_weights()

        if best_config is None or best_weights is None or best_metrics is None:
            raise RuntimeError("Regressor grid search failed.")

        tf.keras.backend.clear_session()
        best_model = self._build_regressor(input_dim, best_config)
        best_model.set_weights(best_weights)
        return best_config, best_model, best_metrics

    def _train_regressor(
        self,
        model: models.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any],
    ) -> None:
        patience = int(config.get("patience", 20))
        cb: List[callbacks.Callback] = []
        if patience > 0:
            cb.append(
                callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    restore_best_weights=True,
                    verbose=self.verbose,
                )
            )

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=int(config.get("epochs", 120)),
            batch_size=int(config.get("batch_size", 256)),
            verbose=self.verbose,
            callbacks=cb,
        )

    # ------------------------------------------------------------------ #
    # Model builders                                                     #
    # ------------------------------------------------------------------ #

    def _build_autoencoder(self, input_dim: int, config: Dict[str, Any]) -> models.Model:
        encoder_layers = config.get("encoder_layers", (256, 128))
        latent_dim = int(config.get("latent_dim", 64))
        activation = config.get("activation", "relu")
        learning_rate = float(config.get("learning_rate", 1e-3))

        inputs = layers.Input(shape=(input_dim,))
        x = inputs
        for units in encoder_layers:
            x = layers.Dense(int(units), activation=activation)(x)
        latent = layers.Dense(latent_dim, activation=activation, name="latent")(x)

        x = latent
        for units in reversed(encoder_layers):
            x = layers.Dense(int(units), activation=activation)(x)
        outputs = layers.Dense(input_dim, activation="linear")(x)

        model = models.Model(inputs=inputs, outputs=outputs)
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="mse")
        return model

    def _build_regressor(self, input_dim: int, config: Dict[str, Any]) -> models.Model:
        hidden_layers = config.get("hidden_layers", (256, 128))
        activation = config.get("activation", "relu")
        dropout_rate = float(config.get("dropout", 0.0))
        learning_rate = float(config.get("learning_rate", 1e-3))

        inputs = layers.Input(shape=(input_dim,))
        x = inputs
        for units in hidden_layers:
            x = layers.Dense(int(units), activation=activation)(x)
            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1, activation="linear")(x)

        model = models.Model(inputs=inputs, outputs=outputs)
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError()])
        return model

    # ------------------------------------------------------------------ #
    # Utility methods                                                   #
    # ------------------------------------------------------------------ #

    def _compute_residuals(self, model: models.Model, X: np.ndarray) -> np.ndarray:
        recon = model.predict(X, verbose=0)
        return np.mean(np.square(X - recon), axis=1)

    def _transform_targets(
        self,
        y_train: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_train = np.asarray(y_train).ravel()
        y_val = np.asarray(y_val).ravel()
        y_all = np.concatenate([y_train, y_val])
        mean = float(np.mean(y_all))
        std = float(np.std(y_all) + self.TARGET_EPS)
        self._target_scaler = {"mean": mean, "std": std}
        return (y_train - mean) / std, (y_val - mean) / std

    def _inverse_transform_targets(self, y: np.ndarray) -> np.ndarray:
        if self._target_scaler is None:
            return y
        mean = self._target_scaler["mean"]
        std = self._target_scaler["std"]
        return y * std + mean

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean(np.square(y_true - y_pred))))
        ss_res = float(np.sum(np.square(y_true - y_pred)))
        ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
        r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
        return {"mae": mae, "rmse": rmse, "r2": r2}

    def _iter_param_grid(self, grid: Dict[str, Sequence[Any]]) -> Iterable[Dict[str, Any]]:
        keys = sorted(grid.keys())
        for values in product(*(grid[k] for k in keys)):
            yield {k: v for k, v in zip(keys, values)}

    @staticmethod
    def _normalise_config(config: Dict[str, Any]) -> Dict[str, Any]:
        cfg = dict(config)
        if "hidden_layers" in cfg:
            hl = cfg["hidden_layers"]
            if isinstance(hl, (list, tuple)):
                cfg["hidden_layers"] = tuple(int(v) for v in hl)
            else:
                raise ValueError("hidden_layers must be list or tuple.")
        return cfg

    @staticmethod
    def _normalise_autoencoder_config(config: Dict[str, Any]) -> Dict[str, Any]:
        cfg = dict(config)
        if "encoder_layers" in cfg:
            layers_cfg = cfg["encoder_layers"]
            if isinstance(layers_cfg, (list, tuple)):
                cfg["encoder_layers"] = tuple(int(v) for v in layers_cfg)
            else:
                raise ValueError("encoder_layers must be list or tuple.")
        return cfg

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #

    def _ae_artifacts(self) -> AutoencoderArtifacts:
        prefix = self.cache_prefix
        return AutoencoderArtifacts(
            config_path=self.model_dir / f"{prefix}_autoencoder_config.json",
            model_path=self.model_dir / f"{prefix}_autoencoder_model.keras",
            metrics_path=self.model_dir / f"{prefix}_autoencoder_metrics.json",
            grid_results_path=self.model_dir / f"{prefix}_autoencoder_grid_results.json",
        )

    def _reg_artifacts(self) -> RegressorArtifacts:
        prefix = self.cache_prefix
        return RegressorArtifacts(
            config_path=self.model_dir / f"{prefix}_regressor_config.json",
            model_path=self.model_dir / f"{prefix}_regressor_model.keras",
            metrics_path=self.model_dir / f"{prefix}_regressor_metrics.json",
            scaler_path=self.model_dir / f"{prefix}_target_scaler.json",
        )

    def _autoencoder_cache_exists(self, artifacts: AutoencoderArtifacts) -> bool:
        return artifacts.config_path.exists() and artifacts.model_path.exists()

    def _regressor_cache_exists(self, artifacts: RegressorArtifacts) -> bool:
        return (
            artifacts.config_path.exists()
            and artifacts.model_path.exists()
            and artifacts.scaler_path.exists()
        )

    def _load_autoencoder(self, artifacts: AutoencoderArtifacts) -> None:
        with artifacts.config_path.open("r", encoding="utf-8") as fh:
            config = json.load(fh)
        self.autoencoder_config = self._normalise_autoencoder_config(config)
        model = tf.keras.models.load_model(artifacts.model_path)
        self.autoencoder = model

        if artifacts.metrics_path.exists():
            with artifacts.metrics_path.open("r", encoding="utf-8") as fh:
                self.autoencoder_metrics = {
                    k: float(v) for k, v in json.load(fh).items()
                }

        # Load grid search results for elbow analysis
        if artifacts.grid_results_path.exists():
            with artifacts.grid_results_path.open("r", encoding="utf-8") as fh:
                serialisable_results = json.load(fh)
            self.autoencoder_grid_results = []
            for result in serialisable_results:
                normalised_result = {
                    "config": self._normalise_autoencoder_config(result["config"]),
                    "metrics": result["metrics"],
                    "score": result["score"],
                }
                self.autoencoder_grid_results.append(normalised_result)

    def _save_autoencoder(self, artifacts: AutoencoderArtifacts) -> None:
        if self.autoencoder is None or self.autoencoder_config is None:
            raise RuntimeError("Autoencoder must be trained before saving.")

        serialisable_cfg = self._serialise_autoencoder_config(self.autoencoder_config)
        with artifacts.config_path.open("w", encoding="utf-8") as fh:
            json.dump(serialisable_cfg, fh, indent=2)

        self.autoencoder.save(artifacts.model_path)

        if self.autoencoder_metrics:
            with artifacts.metrics_path.open("w", encoding="utf-8") as fh:
                json.dump(self.autoencoder_metrics, fh, indent=2)

        # Save grid search results for elbow analysis
        if self.autoencoder_grid_results:
            serialisable_results = []
            for result in self.autoencoder_grid_results:
                serialisable_result = {
                    "config": self._serialise_autoencoder_config(result["config"]),
                    "metrics": result["metrics"],
                    "score": result["score"],
                }
                serialisable_results.append(serialisable_result)
            with artifacts.grid_results_path.open("w", encoding="utf-8") as fh:
                json.dump(serialisable_results, fh, indent=2)

    def _load_regressor(self, artifacts: RegressorArtifacts) -> None:
        with artifacts.config_path.open("r", encoding="utf-8") as fh:
            config = json.load(fh)
        self.best_config = self._normalise_config(config)
        model = tf.keras.models.load_model(artifacts.model_path)
        self.model = model

        if artifacts.metrics_path.exists():
            with artifacts.metrics_path.open("r", encoding="utf-8") as fh:
                self.best_metrics = json.load(fh)

        with artifacts.scaler_path.open("r", encoding="utf-8") as fh:
            scaler = json.load(fh)
        self._target_scaler = {
            "mean": float(scaler.get("mean", 0.0)),
            "std": float(scaler.get("std", 1.0)),
        }

    def _save_regressor(self, artifacts: RegressorArtifacts) -> None:
        if (
            self.model is None
            or self.best_config is None
            or self._target_scaler is None
        ):
            raise RuntimeError("Regressor must be trained before saving.")

        serialisable_cfg = self._serialise_config(self.best_config)
        with artifacts.config_path.open("w", encoding="utf-8") as fh:
            json.dump(serialisable_cfg, fh, indent=2)

        self.model.save(artifacts.model_path)

        if self.best_metrics is not None:
            with artifacts.metrics_path.open("w", encoding="utf-8") as fh:
                json.dump(self.best_metrics, fh, indent=2)

        with artifacts.scaler_path.open("w", encoding="utf-8") as fh:
            json.dump(self._target_scaler, fh, indent=2)

    def _serialise_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        serialisable = dict(config)
        hidden = serialisable.get("hidden_layers")
        if isinstance(hidden, tuple):
            serialisable["hidden_layers"] = list(hidden)
        return serialisable

    def _serialise_autoencoder_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        serialisable = dict(config)
        layers_cfg = serialisable.get("encoder_layers")
        if isinstance(layers_cfg, tuple):
            serialisable["encoder_layers"] = list(layers_cfg)
        return serialisable
