from __future__ import annotations

"""
Autoencoder teacher + per-feature residual regressor (student).

Concept:
- Teacher: autoencoder reconstructs X; per-feature residual is squared error per feature.
- Student: DNN predicts the per-feature residual vector directly from X.

Notes:
- API mirrors AutoencoderResidualRegressor where practical (fit/predict/evaluate/plots).
- This module is standalone and does not depend on the scalar-residual implementation.
"""

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers


@dataclass
class AutoencoderArtifacts:
    config_path: Path
    model_path: Path
    metrics_path: Path


@dataclass
class RegressorArtifacts:
    config_path: Path
    model_path: Path
    metrics_path: Path
    scaler_path: Path


class AutoencoderFeatureResidualRegressor:
    """
    Autoencoder teacher + regressor student predicting per-feature residuals.

    Residual definition (teacher):
        r_i = (x_i - AE(x)_i)^2  for each feature i

    The pipeline supports:
      * Autoencoder grid search with caching.
      * Per-feature residual regressor grid search with target standardisation and caching.
      * Diagnostics (metrics + scatter/timeline plots).
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
        "latent_dim": [32, 64, 96],
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
        cache_prefix: str = "autoencoder_feature_residual_regressor",
        verbose: int = 0,
        autoencoder: Optional[models.Model] = None,
        teacher_preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        student_preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        base_dir = Path(__file__).resolve().parent.parent
        self.model_dir: Path = model_dir or (base_dir / "models")
        self.cache_prefix = cache_prefix
        self.verbose = verbose

        # Optional externally provided teacher
        self.autoencoder: Optional[models.Model] = autoencoder
        self.autoencoder_config: Optional[Dict[str, Any]] = None
        self.autoencoder_metrics: Optional[Dict[str, float]] = None
        # Optional preprocessing function to map X -> teacher domain (e.g., internal normalization)
        self._teacher_preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = teacher_preprocess
        # Optional preprocessing for student inputs (defaults to teacher_preprocess if not provided)
        self._student_preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = (
            student_preprocess if student_preprocess is not None else teacher_preprocess
        )

        self.model: Optional[models.Model] = None
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_metrics: Optional[Dict[str, Any]] = None

        # Per-feature scaler for target residuals
        self._target_scaler: Optional[Dict[str, np.ndarray]] = None

        self.model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
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
        """
        # If an external teacher is already attached and not resetting, reuse it.
        if self.autoencoder is not None and not reset_autoencoder and (
            autoencoder_param_grid is None and autoencoder_config is None
        ):
            return self.autoencoder

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

    def attach_teacher(
        self,
        model: models.Model,
        *,
        preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Attach an externally trained autoencoder teacher.

        Args:
            model: Keras model mapping X->reconstruction (on the same domain as preprocess(X)).
            preprocess: Optional function to map raw X to the teacher's domain
                (e.g., teacher._normalize(X, fit=False)). Residuals are computed
                w.r.t. preprocess(X) and its reconstruction.
            config: Optional metadata to store in-memory.
            metrics: Optional metadata to store in-memory.
        """
        self.autoencoder = model
        self._teacher_preprocess = preprocess
        self.autoencoder_config = config
        self.autoencoder_metrics = metrics

    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        *,
        param_grid: Optional[Dict[str, Sequence[Any]]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        autoencoder_param_grid: Optional[Dict[str, Sequence[Any]]] = None,
        autoencoder_config: Optional[Dict[str, Any]] = None,
        reset_experiment: bool = False,
    ) -> models.Model:
        """
        Train (or load) the per-feature residual regressor student.
        """
        # Ensure teacher is ready
        self.train_autoencoder(
            X_train,
            X_val,
            autoencoder_param_grid=autoencoder_param_grid,
            autoencoder_config=autoencoder_config,
            # Load from cache when available unless explicitly resetting
            reset_autoencoder=reset_experiment,
        )

        reg_artifacts = self._reg_artifacts()
        if not reset_experiment and self._regressor_cache_exists(reg_artifacts):
            self._load_regressor(reg_artifacts)
            if self.verbose:
                print("[Regressor] Loaded cached model.")
            return self.model  # type: ignore[return-value]

        if model_config is not None and param_grid is not None:
            raise ValueError("Provide either model_config or param_grid, not both.")

        # Targets: per-feature squared residuals
        y_train_raw = self._compute_feature_residuals(self.autoencoder, X_train)
        y_val_raw = self._compute_feature_residuals(self.autoencoder, X_val)
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
            X_train_in = self._student_preprocess(X_train) if self._student_preprocess is not None else X_train
            X_val_in = self._student_preprocess(X_val) if self._student_preprocess is not None else X_val
            model = self._build_regressor(X_train_in.shape[1], cfg)
            self._train_regressor(model, X_train_in, y_train, X_val_in, y_val, cfg)
            self.model = model
            self.best_config = cfg
            train_pred = self._inverse_transform_targets(model.predict(X_train_in, verbose=0))
            val_pred = self._inverse_transform_targets(model.predict(X_val_in, verbose=0))
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
            np.asarray(y_true)
            if y_true is not None
            else self._compute_feature_residuals(self.autoencoder, X)
        )
        preds = self.predict(X)
        return self._compute_metrics(target, preds)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Regressor not trained.")
        X_in = self._student_preprocess(X) if self._student_preprocess is not None else X
        preds = self.model.predict(X_in, verbose=0)
        return self._inverse_transform_targets(preds)

    def autoencoder_residuals(self, X: np.ndarray) -> np.ndarray:
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not trained.")
        return self._compute_feature_residuals(self.autoencoder, X)

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
        Scatter of AE per-feature residual vs regressor per-feature residual.
        Points are flattened across samples and features; optional subsampling.
        """
        if self.autoencoder is None or self.model is None:
            raise RuntimeError("Train both autoencoder and regressor before plotting.")

        import matplotlib.pyplot as plt

        true_residuals = (
            np.asarray(y_true)
            if y_true is not None
            else self.autoencoder_residuals(X)
        )
        predicted = self.predict(X)

        x = true_residuals.ravel()
        y = predicted.ravel()
        if sample_size is not None and sample_size < len(x):
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(x), size=sample_size, replace=False)
            x = x[idx]
            y = y[idx]

        ax = plt.gca()
        ax.scatter(x, y, alpha=0.35, s=10, color="steelblue", label="samples")
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="red", label="ideal")
        ax.set_xlabel("AE residual (per-feature)")
        ax.set_ylabel("Regressor residual (per-feature)")
        ax.set_title("Per-feature residual fit")
        ax.legend()
        if show:
            import matplotlib.pyplot as plt  # local import for safety
            plt.show()
        return ax

    def plot_residual_series(
        self,
        X: np.ndarray,
        *,
        y_true: Optional[np.ndarray] = None,
        feature_idx: Optional[int] = None,
        agg: str = "mean",
        sample_slice: Optional[slice] = None,
        title: Optional[str] = None,
        show: bool = True,
    ):
        """
        Plot residual timeline comparing autoencoder (teacher) vs regressor (student).

        Args:
            X: feature matrix.
            y_true: optional precomputed AE residual matrix; otherwise computed on the fly.
            feature_idx: if provided, plots that feature's residual timeline; otherwise aggregates.
            agg: aggregation over features when feature_idx is None (mean|max|sum|median).
            sample_slice: optional slice selecting which samples to plot (e.g., slice(0, 200)).
            title: optional custom title.
            show: display plot via plt.show().
        """
        if self.autoencoder is None or self.model is None:
            raise RuntimeError("Train both autoencoder and regressor before plotting.")

        import matplotlib.pyplot as plt

        teacher = (
            np.asarray(y_true)
            if y_true is not None
            else self.autoencoder_residuals(X)
        )
        student = self.predict(X)

        def reduce_fn(arr: np.ndarray) -> np.ndarray:
            if feature_idx is not None:
                return arr[:, int(feature_idx)]
            if agg == "mean":
                return np.mean(arr, axis=1)
            if agg == "max":
                return np.max(arr, axis=1)
            if agg == "sum":
                return np.sum(arr, axis=1)
            if agg == "median":
                return np.median(arr, axis=1)
            raise ValueError("Unknown agg; expected one of: mean|max|sum|median")

        teacher_1d = reduce_fn(teacher)
        student_1d = reduce_fn(student)

        if sample_slice is None:
            sample_slice = slice(0, len(student_1d))

        t_sel = teacher_1d[sample_slice]
        s_sel = student_1d[sample_slice]
        idx = np.arange(len(s_sel))

        plt.figure(figsize=(10, 4))
        plt.plot(idx, t_sel, label="AE residual", color="grey", alpha=0.7)
        plt.plot(idx, s_sel, label="Regressor residual", color="steelblue")
        plt.xlabel("Sample index")
        plt.ylabel("Residual value")
        ttl = title or (
            f"Residual timeline (feature={feature_idx})" if feature_idx is not None else f"Residual timeline (agg={agg})"
        )
        plt.title(ttl)
        plt.legend()
        plt.grid(alpha=0.2)
        if show:
            plt.show()
        return t_sel, s_sel

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

        for cfg in self._iter_param_grid(param_grid):
            config = self._normalise_autoencoder_config(cfg)
            tf.keras.backend.clear_session()
            model = self._build_autoencoder(X_train.shape[1], config)
            history = self._fit_autoencoder(model, X_train, X_val, config)
            val_loss = float(np.min(history.history["val_loss"]))
            if self.verbose:
                print(f"[Autoencoder] Config={config} -> val_loss={val_loss:.6f}")
            if val_loss < best_score:
                best_score = val_loss
                best_config = config
                best_metrics = {
                    "train_loss": float(np.min(history.history["loss"])) if "loss" in history.history else np.nan,
                    "val_loss": val_loss,
                }
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
    ) -> tf.keras.callbacks.History:
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
        # Apply student-side preprocessing if provided
        X_train_in = self._student_preprocess(X_train) if self._student_preprocess is not None else X_train
        X_val_in = self._student_preprocess(X_val) if self._student_preprocess is not None else X_val
        input_dim = X_train_in.shape[1]
        best_score = np.inf
        best_config: Optional[Dict[str, Any]] = None
        best_metrics: Optional[Dict[str, Any]] = None
        best_weights: Optional[List[np.ndarray]] = None

        for cfg in self._iter_param_grid(param_grid):
            config = self._normalise_config(cfg)
            tf.keras.backend.clear_session()
            model = self._build_regressor(input_dim, config)
            self._train_regressor(model, X_train_in, y_train, X_val_in, y_val, config)

            train_pred = self._inverse_transform_targets(model.predict(X_train_in, verbose=0))
            val_pred = self._inverse_transform_targets(model.predict(X_val_in, verbose=0))

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
        outputs = layers.Dense(input_dim, activation="linear")(x)

        model = models.Model(inputs=inputs, outputs=outputs)
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError()])
        return model

    # ------------------------------------------------------------------ #
    # Utility methods                                                    #
    # ------------------------------------------------------------------ #

    def _compute_feature_residuals(self, model: models.Model, X: np.ndarray) -> np.ndarray:
        X_in = self._teacher_preprocess(X) if self._teacher_preprocess is not None else X
        recon = model.predict(X_in, verbose=0)
        return np.square(X_in - recon)

    def _transform_targets(
        self,
        y_train: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_train = np.asarray(y_train)
        y_val = np.asarray(y_val)
        if y_train.ndim != 2 or y_val.ndim != 2:
            raise ValueError("Targets must be 2D arrays (samples x features).")
        y_all = np.vstack([y_train, y_val])
        mean = np.mean(y_all, axis=0)
        std = np.std(y_all, axis=0) + self.TARGET_EPS
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
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        # Flatten across samples and features
        yt = y_true.ravel()
        yp = y_pred.ravel()
        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean(np.square(yt - yp))))
        ss_res = float(np.sum(np.square(yt - yp)))
        ss_tot = float(np.sum(np.square(yt - np.mean(yt))))
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
                self.autoencoder_metrics = {k: float(v) for k, v in json.load(fh).items()}

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
        # Store as numpy arrays for computation
        self._target_scaler = {
            "mean": np.asarray(scaler.get("mean", []), dtype=float),
            "std": np.asarray(scaler.get("std", []), dtype=float),
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

        # Persist target scaler with lists for JSON compatibility
        mean = self._target_scaler["mean"].tolist()
        std = self._target_scaler["std"].tolist()
        with artifacts.scaler_path.open("w", encoding="utf-8") as fh:
            json.dump({"mean": mean, "std": std}, fh, indent=2)

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
