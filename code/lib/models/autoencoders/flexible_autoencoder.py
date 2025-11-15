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
class AutoencoderArtifacts:
    config_path: Path
    model_path: Path
    metrics_path: Path
    grid_results_path: Path


class FlexibleAutoencoder:
    """
    Flexible autoencoder with grid search, caching, and encoder extraction.

    Features:
    - Train with grid search or single config
    - Extract encoder-only model for latent representations
    - Extract decoder-only model for reconstruction from latent
    - Bottleneck optimization with elbow curve
    - Automatic caching to avoid retraining

    Example:
        >>> ae = FlexibleAutoencoder(verbose=1)
        >>> # Train full autoencoder
        >>> ae.fit(X_train, X_val, param_grid={...})
        >>> # Get reconstructions
        >>> reconstructions = ae.predict(X_test)
        >>> # Get latent representations
        >>> latent = ae.encode(X_test)
        >>> # Get encoder-only model
        >>> encoder = ae.get_encoder()
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "encoder_layers": (256, 128),
        "latent_dim": 64,
        "activation": "relu",
        "learning_rate": 1e-3,
        "batch_size": 256,
        "epochs": 120,
        "patience": 15,
    }

    DEFAULT_PARAM_GRID: Dict[str, Sequence[Any]] = {
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

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        cache_prefix: str = "flexible_autoencoder",
        verbose: int = 0,
        models_subdir: str = "trained_models",
    ) -> None:
        """
        Initialize FlexibleAutoencoder.

        Parameters
        ----------
        model_dir
            Directory to save/load models. If None, uses base_dir / models_subdir.
        cache_prefix
            Prefix for cached model files. Use different prefixes for different experiments.
        verbose
            Verbosity level (0=silent, 1=progress messages).
        models_subdir
            Subdirectory name under base_dir to use when model_dir is not provided.
        """
        base_dir = Path(__file__).resolve().parent.parent
        self.model_dir: Path = model_dir or (base_dir / models_subdir)
        self.cache_prefix = cache_prefix
        self.verbose = verbose

        self.autoencoder: Optional[models.Model] = None
        self.encoder: Optional[models.Model] = None
        self.decoder: Optional[models.Model] = None
        self.config: Optional[Dict[str, Any]] = None
        self.metrics: Optional[Dict[str, float]] = None
        self.grid_results: List[Dict[str, Any]] = []

        self.model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Bottleneck Size Optimization                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def generate_bottleneck_range(
        input_dim: int, min_ratio: float = 8.0, max_ratio: float = 4.0
    ) -> List[int]:
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
        >>> FlexibleAutoencoder.generate_bottleneck_range(52)
        [6, 7, 8, 9, 10, 11, 12, 13]  # From 52/8 to 52/4
        """
        min_size = max(4, int(input_dim / min_ratio))
        max_size = int(input_dim / max_ratio)

        if max_size - min_size <= 8:
            return list(range(min_size, max_size + 1))
        else:
            return list(range(min_size, max_size + 1, 2))

    @staticmethod
    def generate_bottleneck_grid(
        input_dim: int, min_ratio: float = 8.0, max_ratio: float = 4.0
    ) -> Dict[str, List[int]]:
        """
        Generate parameter grid with dynamic bottleneck range.

        Returns
        -------
        Dict with "latent_dim" key containing list of bottleneck sizes.
        """
        bottleneck_range = FlexibleAutoencoder.generate_bottleneck_range(
            input_dim, min_ratio, max_ratio
        )
        if len(bottleneck_range) > 0:
            print(
                f"[Bottleneck] Generated range for input_dim={input_dim}: {bottleneck_range}"
            )
            print(f"[Bottleneck] Range: {bottleneck_range[0]} to {bottleneck_range[-1]}")
        return {"latent_dim": bottleneck_range}

    @staticmethod
    def find_elbow_point(x: np.ndarray, y: np.ndarray) -> int:
        """
        Find the elbow point in a curve using the maximum distance method.

        The elbow is the point with maximum perpendicular distance from
        the line connecting first and last points.

        Parameters
        ----------
        x : np.ndarray
            X values (e.g., bottleneck sizes).
        y : np.ndarray
            Y values (e.g., validation losses).

        Returns
        -------
        int
            Index of the elbow point.
        """
        x_norm = (x - x[0]) / (x[-1] - x[0])
        y_norm = (y - y[0]) / (y[-1] - y[0])
        distances = np.abs(x_norm - y_norm) / np.sqrt(2)
        return int(np.argmax(distances))

    def plot_elbow_curve(
        self,
        results: Optional[List[Dict[str, Any]]] = None,
        metric: str = "val_loss",
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> Tuple[int, float]:
        """
        Plot elbow curve for bottleneck size selection.

        Parameters
        ----------
        results
            Grid search results. If None, uses self.grid_results.
        metric
            Metric to plot (default: "val_loss").
        show
            Whether to display the plot.
        save_path
            Optional path to save the plot.

        Returns
        -------
        Tuple[int, float]
            (optimal_bottleneck_size, optimal_metric_value)
        """
        import matplotlib.pyplot as plt

        if results is None:
            results = self.grid_results

        if not results:
            raise ValueError("No grid search results available. Run fit with grid search first.")

        # Extract bottleneck sizes and metric values
        bottleneck_sizes = np.array([r["config"]["latent_dim"] for r in results])
        metric_values = np.array([r["metrics"][metric] for r in results])

        # Sort by bottleneck size
        sort_idx = np.argsort(bottleneck_sizes)
        bottleneck_sizes = bottleneck_sizes[sort_idx]
        metric_values = metric_values[sort_idx]

        # Find elbow
        elbow_idx = self.find_elbow_point(bottleneck_sizes, metric_values)
        optimal_size = bottleneck_sizes[elbow_idx]
        optimal_value = metric_values[elbow_idx]

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(bottleneck_sizes, metric_values, "o-", linewidth=2, markersize=8)
        plt.plot(
            optimal_size,
            optimal_value,
            "r*",
            markersize=20,
            label=f"Elbow at {optimal_size}",
        )
        plt.xlabel("Bottleneck Size (latent_dim)", fontsize=12)
        plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
        plt.title("Elbow Method for Optimal Bottleneck Size", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[Plot] Saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return int(optimal_size), float(optimal_value)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        *,
        param_grid: Optional[Dict[str, Sequence[Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
        reset: bool = False,
    ) -> models.Model:
        """
        Train (or load) the autoencoder.

        Parameters
        ----------
        X_train
            Training features.
        X_val
            Validation features.
        param_grid
            Grid of hyperparameters for grid search. If None, uses DEFAULT_PARAM_GRID.
        config
            Single config dict to skip grid search. Cannot be used with param_grid.
        reset
            If True, force retrain even if cached model exists.

        Returns
        -------
        models.Model
            Trained autoencoder model.
        """
        artifacts = self._get_artifacts()
        if not reset and self._cache_exists(artifacts):
            self._load(artifacts)
            if self.verbose:
                print("[Autoencoder] Loaded cached model.")
            return self.autoencoder  # type: ignore[return-value]

        if config is not None and param_grid is not None:
            raise ValueError("Provide either config or param_grid, not both.")

        if config is not None:
            cfg = self._normalize_config(config)
            encoder, decoder, model, metrics = self._train_single(X_train, X_val, cfg)
            # Save encoder and decoder directly (they already have trained weights)
            self.encoder = encoder
            self.decoder = decoder
        else:
            grid = param_grid or self.DEFAULT_PARAM_GRID
            cfg, encoder, decoder, model, metrics = self._run_grid_search(X_train, X_val, grid)
            # Save encoder and decoder from grid search
            self.encoder = encoder
            self.decoder = decoder

        self.autoencoder = model
        self.config = cfg
        self.metrics = metrics
        # No need to call _extract_encoder_decoder() anymore - encoder/decoder already set
        self._save(artifacts)
        return model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get reconstructions from the autoencoder.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Reconstructed data.
        """
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not trained. Call fit() first.")
        return self.autoencoder.predict(X, verbose=0)

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Get latent representations (encoded values).

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Latent representations.
        """
        if self.encoder is None:
            raise RuntimeError("Encoder not available. Call fit() first.")
        return self.encoder.predict(X, verbose=0)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """
        Reconstruct from latent representations.

        Parameters
        ----------
        latent : np.ndarray
            Latent representations.

        Returns
        -------
        np.ndarray
            Reconstructed data.
        """
        if self.decoder is None:
            raise RuntimeError("Decoder not available. Call fit() first.")
        return self.decoder.predict(latent, verbose=0)

    def get_encoder(self) -> models.Model:
        """
        Get the encoder-only model.

        Returns
        -------
        models.Model
            Encoder model that outputs latent representations.
        """
        if self.encoder is None:
            raise RuntimeError("Encoder not available. Call fit() first.")
        return self.encoder

    def get_decoder(self) -> models.Model:
        """
        Get the decoder-only model.

        Returns
        -------
        models.Model
            Decoder model that reconstructs from latent representations.
        """
        if self.decoder is None:
            raise RuntimeError("Decoder not available. Call fit() first.")
        return self.decoder

    def get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Get reconstruction error (MSE) for each sample.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Reconstruction error per sample.
        """
        reconstructions = self.predict(X)
        return np.mean(np.square(X - reconstructions), axis=1)

    def plot_elbow_summary(self, metric: str = "val_loss") -> None:
        """
        Display training summary with elbow curve.

        Shows:
        - Model configuration
        - Training metrics
        - Elbow curve for bottleneck optimization (if grid search was performed)

        Parameters
        ----------
        metric : str
            Metric to use for elbow curve (default: "val_loss").

        Example
        -------
        >>> autoencoder.fit(X_train, X_val, param_grid=GRID)
        >>> autoencoder.plot_elbow_summary()
        """
        if self.config is None or self.metrics is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Print config and metrics
        print(f"\nConfig: {self.config}")
        print(f"Metrics: {self.metrics}")

        # Elbow curve (only if grid search was performed)
        if self.grid_results:
            optimal_size, _ = self.plot_elbow_curve(metric=metric, show=True)
            print(f"\nElbow-selected: {optimal_size}")
        else:
            print("\n[Note] Elbow curve not available (no grid search performed)")

    def plot_residual_distribution(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        *,
        bins: int = 60,
        figsize: tuple = (8, 4),
        title: str = "Autoencoder residual distribution",
    ) -> None:
        """
        Plot reconstruction error distribution for train and validation data.

        This plot is useful for:
        - Checking for overfitting (train-val overlap)
        - Assessing model quality (lower errors = better)
        - Selecting anomaly detection thresholds
        - Identifying outliers and data quality issues

        See docs/autoencoder_residual_distribution_guide.md for detailed interpretation.

        Parameters
        ----------
        X_train : np.ndarray
            Training data.
        X_val : np.ndarray
            Validation data.
        bins : int
            Number of histogram bins (default: 60).
        figsize : tuple
            Figure size (default: (8, 4)).
        title : str
            Plot title.

        Example
        -------
        >>> autoencoder.fit(X_train, X_val)
        >>> autoencoder.plot_residual_distribution(X_train, X_val)
        """
        import matplotlib.pyplot as plt

        if self.autoencoder is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        train_residuals = self.get_reconstruction_error(X_train)
        val_residuals = self.get_reconstruction_error(X_val)

        plt.figure(figsize=figsize)
        plt.hist(train_residuals, bins=bins, alpha=0.6, label="train")
        plt.hist(val_residuals, bins=bins, alpha=0.6, label="validation")
        plt.title(title)
        plt.xlabel("Reconstruction Error (MSE)")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # Print statistics
        print(f"\nReconstruction Error Statistics:")
        print(f"  Train: mean={np.mean(train_residuals):.4f}, std={np.std(train_residuals):.4f}")
        print(f"  Val:   mean={np.mean(val_residuals):.4f}, std={np.std(val_residuals):.4f}")

    def plot_summary(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        *,
        metric: str = "val_loss",
    ) -> None:
        """
        Display comprehensive training summary with all plots.

        Convenience method that calls both plot_elbow_summary() and
        plot_residual_distribution().

        Parameters
        ----------
        X_train : np.ndarray
            Training data.
        X_val : np.ndarray
            Validation data.
        metric : str
            Metric to use for elbow curve (default: "val_loss").

        Example
        -------
        >>> autoencoder.fit(X_train, X_val, param_grid=GRID)
        >>> autoencoder.plot_summary(X_train, X_val)
        """
        self.plot_elbow_summary(metric=metric)
        self.plot_residual_distribution(X_train, X_val)

    # ------------------------------------------------------------------ #
    # Model Building                                                      #
    # ------------------------------------------------------------------ #

    def _build_autoencoder(
        self, input_dim: int, cfg: Dict[str, Any]
    ) -> Tuple[models.Model, models.Model, models.Model]:
        """
        Build encoder, decoder, and full autoencoder models.

        Architecture:
            Input (input_dim) → Encoder Layers → Latent (latent_dim) → Decoder Layers → Output (input_dim)

        The key insight is that the autoencoder is built by nesting the encoder and decoder:
            autoencoder = Input → encoder(Input) → decoder(latent) → Output

        This means when we train the autoencoder, the encoder and decoder weights are
        automatically updated because they are part of the computational graph.

        Example with encoder_layers=(256, 128), latent_dim=64, input_dim=52:
            Encoder:  52 → 256 → 128 → 64 (latent)
            Decoder:  64 → 128 → 256 → 52 (reconstruction)

        Returns
        -------
        Tuple[models.Model, models.Model, models.Model]
            (encoder, decoder, autoencoder)
            - encoder: Standalone model that maps input → latent
            - decoder: Standalone model that maps latent → reconstruction
            - autoencoder: Full model that maps input → reconstruction (contains encoder + decoder)
        """
        encoder_layers = cfg["encoder_layers"]
        latent_dim = cfg["latent_dim"]
        activation = cfg["activation"]

        # ========== Build Encoder ==========
        # Maps raw input features to compressed latent representation
        encoder_input = layers.Input(shape=(input_dim,), name="encoder_input")
        x = encoder_input
        for i, units in enumerate(encoder_layers):
            x = layers.Dense(units, activation=activation, name=f"encoder_dense_{i}")(x)
        latent = layers.Dense(latent_dim, activation=activation, name="latent")(x)
        encoder = models.Model(encoder_input, latent, name="encoder")

        # ========== Build Decoder ==========
        # Maps latent representation back to original input space (mirror of encoder)
        decoder_input = layers.Input(shape=(latent_dim,), name="decoder_input")
        x = decoder_input
        for i, units in enumerate(reversed(encoder_layers)):
            x = layers.Dense(units, activation=activation, name=f"decoder_dense_{i}")(x)
        decoder_output = layers.Dense(input_dim, activation="linear", name="decoder_output")(x)
        decoder = models.Model(decoder_input, decoder_output, name="decoder")

        # ========== Build Full Autoencoder ==========
        # Combines encoder and decoder: Input → Latent → Reconstruction
        # IMPORTANT: encoder and decoder are used as LAYERS here, so they share weights
        autoencoder_input = layers.Input(shape=(input_dim,), name="input")
        encoded = encoder(autoencoder_input)  # Apply encoder as a layer
        decoded = decoder(encoded)             # Apply decoder as a layer
        autoencoder = models.Model(autoencoder_input, decoded, name="autoencoder")

        # Compile only the autoencoder (encoder and decoder don't need separate compilation)
        optimizer = optimizers.Adam(learning_rate=cfg["learning_rate"])
        autoencoder.compile(optimizer=optimizer, loss="mse")

        return encoder, decoder, autoencoder

    def _train_single(
        self, X_train: np.ndarray, X_val: np.ndarray, cfg: Dict[str, Any]
    ) -> Tuple[models.Model, models.Model, models.Model, Dict[str, float]]:
        """
        Train a single autoencoder configuration.

        Architecture:
            Input → Encoder → Latent → Decoder → Output

        The autoencoder is built as a composite model containing the encoder and decoder.
        When we train the autoencoder, the encoder and decoder weights are updated automatically
        because they are nested inside it.

        Returns
        -------
        Tuple[models.Model, models.Model, models.Model, Dict[str, float]]
            (encoder, decoder, autoencoder, metrics)
            - encoder: Trained encoder model (Input → Latent)
            - decoder: Trained decoder model (Latent → Output)
            - autoencoder: Trained full model (Input → Output)
            - metrics: Training and validation loss
        """
        input_dim = X_train.shape[1]

        # Build all three models: encoder, decoder, and autoencoder
        encoder, decoder, autoencoder = self._build_autoencoder(input_dim, cfg)

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

        # Train the autoencoder (this automatically trains encoder and decoder too)
        print(f"[Training] Config: {cfg}")
        history = autoencoder.fit(
            X_train,
            X_train,
            validation_data=(X_val, X_val),
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            callbacks=cb,
            verbose=0,
        )

        train_loss = float(np.min(history.history["loss"]))
        val_loss = float(np.min(history.history["val_loss"]))
        metrics = {"train_loss": train_loss, "val_loss": val_loss}

        # Return encoder and decoder along with autoencoder
        # They share the same trained weights since encoder/decoder are nested in autoencoder
        return encoder, decoder, autoencoder, metrics

    def _run_grid_search(
        self, X_train: np.ndarray, X_val: np.ndarray, param_grid: Dict[str, Sequence[Any]]
    ) -> Tuple[Dict[str, Any], models.Model, models.Model, models.Model, Dict[str, float]]:
        """
        Run grid search over hyperparameters.

        Tests all combinations of parameters and selects the best model based on validation loss.

        Returns
        -------
        Tuple[Dict[str, Any], models.Model, models.Model, models.Model, Dict[str, float]]
            (best_config, best_encoder, best_decoder, best_autoencoder, best_metrics)
        """
        self.grid_results = []
        best_score = float("inf")
        best_config = None
        best_encoder = None
        best_decoder = None
        best_model = None
        best_metrics = None

        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]

        for combo in product(*values):
            config = dict(zip(keys, combo))
            config = self._normalize_config(config)

            if self.verbose:
                print(f"[Grid Search] Testing config: {config}")

            # _train_single now returns encoder, decoder, autoencoder, metrics
            encoder, decoder, model, metrics = self._train_single(X_train, X_val, config)
            score = metrics["val_loss"]

            self.grid_results.append({
                "config": config.copy(),
                "metrics": metrics.copy(),
                "score": score,
            })

            if score < best_score:
                best_score = score
                best_config = config
                best_encoder = encoder
                best_decoder = decoder
                best_model = model
                best_metrics = metrics

            if self.verbose:
                print(f"[Grid Search] val_loss={score:.6f}")

            # Clear memory for non-best models
            if model != best_model:
                del encoder, decoder, model
            tf.keras.backend.clear_session()

        if best_config is None or best_model is None or best_metrics is None:
            raise RuntimeError("Grid search failed to find a valid configuration.")

        if self.verbose:
            print(f"[Grid Search] Best config: {best_config}")
            print(f"[Grid Search] Best val_loss: {best_score:.6f}")

        return best_config, best_encoder, best_decoder, best_model, best_metrics

    def _extract_encoder_decoder(self) -> None:
        """
        Extract encoder and decoder from loaded autoencoder.

        When loading from cache, the autoencoder contains the encoder and decoder
        as nested models (layers). This method extracts them directly.

        During training, encoder and decoder are already set, so this does nothing.
        During loading from cache, this extracts them from the autoencoder's layers.

        Architecture of loaded autoencoder:
            autoencoder.layers[0] = Input layer
            autoencoder.layers[1] = encoder (nested Model)
            autoencoder.layers[2] = decoder (nested Model)
        """
        # Already have encoder and decoder from training, nothing to do
        if self.encoder is not None and self.decoder is not None:
            return

        if self.autoencoder is None or self.config is None:
            return

        # Extract encoder and decoder directly from autoencoder's layers
        # They are nested models inside the autoencoder
        for layer in self.autoencoder.layers:
            if layer.name == "encoder":
                self.encoder = layer
                if self.verbose:
                    print("[Extract] Found encoder as nested model")
            elif layer.name == "decoder":
                self.decoder = layer
                if self.verbose:
                    print("[Extract] Found decoder as nested model")

        # If encoder/decoder not found as nested models (old cache format),
        # rebuild them and copy weights (fallback for backward compatibility)
        if self.encoder is None or self.decoder is None:
            if self.verbose:
                print("[Extract] Encoder/decoder not found as nested models, rebuilding...")

            input_dim = self.autoencoder.input_shape[1]
            self.encoder, self.decoder, _ = self._build_autoencoder(input_dim, self.config)

            # Copy weights from trained autoencoder (this is the old fragile method)
            encoder_layers = [l for l in self.autoencoder.layers if "encoder" in l.name or l.name == "latent"]
            decoder_layers = [l for l in self.autoencoder.layers if "decoder" in l.name]

            for ae_layer in encoder_layers:
                try:
                    enc_layer = self.encoder.get_layer(ae_layer.name)
                    enc_layer.set_weights(ae_layer.get_weights())
                except:
                    pass

            for ae_layer in decoder_layers:
                try:
                    dec_layer = self.decoder.get_layer(ae_layer.name)
                    dec_layer.set_weights(ae_layer.get_weights())
                except:
                    pass

    # ------------------------------------------------------------------ #
    # Config Normalization                                                #
    # ------------------------------------------------------------------ #

    def _normalize_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize config to ensure correct types."""
        normalized = cfg.copy()
        if "encoder_layers" in normalized:
            layers_cfg = normalized["encoder_layers"]
            if isinstance(layers_cfg, (list, tuple)):
                normalized["encoder_layers"] = tuple(int(v) for v in layers_cfg)
        return normalized

    def _serialize_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize config for JSON storage."""
        serialized = cfg.copy()
        if "encoder_layers" in serialized:
            serialized["encoder_layers"] = list(serialized["encoder_layers"])
        return serialized

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #

    def _get_artifacts(self) -> AutoencoderArtifacts:
        prefix = self.cache_prefix
        return AutoencoderArtifacts(
            config_path=self.model_dir / f"{prefix}_config.json",
            model_path=self.model_dir / f"{prefix}_model.keras",
            metrics_path=self.model_dir / f"{prefix}_metrics.json",
            grid_results_path=self.model_dir / f"{prefix}_grid_results.json",
        )

    def _cache_exists(self, artifacts: AutoencoderArtifacts) -> bool:
        return artifacts.config_path.exists() and artifacts.model_path.exists()

    def _load(self, artifacts: AutoencoderArtifacts) -> None:
        with artifacts.config_path.open("r", encoding="utf-8") as fh:
            config = json.load(fh)
        self.config = self._normalize_config(config)
        self.autoencoder = tf.keras.models.load_model(artifacts.model_path)

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

        self._extract_encoder_decoder()

    def _save(self, artifacts: AutoencoderArtifacts) -> None:
        if self.autoencoder is None or self.config is None:
            raise RuntimeError("Autoencoder must be trained before saving.")

        serialized_cfg = self._serialize_config(self.config)
        with artifacts.config_path.open("w", encoding="utf-8") as fh:
            json.dump(serialized_cfg, fh, indent=2)

        self.autoencoder.save(artifacts.model_path)

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
