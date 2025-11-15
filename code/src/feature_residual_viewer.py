from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np


def display_feature_residual_timeline_viewer(
    regressor: Any,
    dataset_map: Dict[str, np.ndarray],
    *,
    feature_names: Optional[Sequence[str]] = None,
    default_key: Optional[str] = None,
    default_feature: int = 0,
    sample_size: Optional[int] = None,
    random_state: Optional[int] = None,
    viewer_key: Optional[str] = "feature_residual_timeline_viewer",
) -> None:
    """
    Interactive viewer: per-feature residual timeline (AE teacher vs student).

    Args:
        regressor: Trained AutoencoderFeatureResidualRegressor-like object with
            predict(X) -> (n_samples, n_features) and autoencoder_residuals(X) -> same.
        dataset_map: label -> X (scaled) arrays.
        feature_names: optional list of feature names; defaults to f{i}.
        default_key: dataset shown initially (defaults to first key).
        default_feature: feature index to plot initially.
        sample_size: optional subsample of timepoints for speed; if set and len(X)>sample_size, take a random subset.
        random_state: RNG seed for subsampling.
        viewer_key: unique key to prevent duplicate widget instances.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output, DisplayHandle
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ipywidgets and matplotlib are required for the viewer.") from exc

    labels = list(dataset_map.keys())
    if not labels:
        raise ValueError("dataset_map must not be empty")
    default_key = default_key or labels[0]

    # Probe feature count
    first = dataset_map[default_key]
    n_features = int(first.shape[1])
    if feature_names is None:
        feature_names = tuple(f"f{i}" for i in range(n_features))

    # Precompute and cache per dataset
    cache: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rng = np.random.default_rng(random_state)

    def get_pair(label: str) -> tuple[np.ndarray, np.ndarray]:
        if label not in cache:
            X = dataset_map[label]
            yt = regressor.autoencoder_residuals(X)  # (n, d)
            yp = regressor.predict(X)                # (n, d)
            if sample_size is not None and len(yt) > sample_size:
                idx = rng.choice(len(yt), size=sample_size, replace=False)
                yt = yt[idx]
                yp = yp[idx]
            cache[label] = (yt, yp)
        return cache[label]

    # Controls
    ds_dropdown = widgets.Dropdown(
        options=labels,
        value=default_key,
        description="Dataset",
        layout=widgets.Layout(width="320px"),
        style={"description_width": "80px"},
    )
    feat_slider = widgets.IntSlider(
        value=int(default_feature),
        min=0,
        max=max(0, n_features - 1),
        step=1,
        description="Feature",
        layout=widgets.Layout(width="320px"),
        style={"description_width": "80px"},
    )
    feature_label = widgets.HTML()
    metrics_info = widgets.HTML()

    # Plot handle
    timeline_handle = DisplayHandle()
    initialized = False

    def fmt_metrics(y_true_1d: np.ndarray, y_pred_1d: np.ndarray) -> str:
        err = y_true_1d - y_pred_1d
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        return f"<small><b>MAE:</b> {mae:.5f} &nbsp; <b>RMSE:</b> {rmse:.5f} &nbsp; <b>N:</b> {len(err)}</small>"

    def update(_=None) -> None:
        nonlocal initialized
        label = ds_dropdown.value
        fidx = int(feat_slider.value)
        name = feature_names[fidx] if 0 <= fidx < len(feature_names) else f"f{fidx}"
        feature_label.value = f"<b>Selected feature:</b> {name} (index {fidx})"

        y_true, y_pred = get_pair(label)
        t = y_true[:, fidx]
        p = y_pred[:, fidx]
        metrics_info.value = fmt_metrics(t, p)

        fig = plt.figure(figsize=(9, 3.8))
        ax = fig.add_subplot(111)
        ax.plot(t, label="AE residual", color="grey", alpha=0.7)
        ax.plot(p, label="Regressor residual", color="steelblue")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Residual value")
        ax.set_title(f"Feature residual timeline: {label} — {name}")
        ax.legend()
        ax.grid(alpha=0.25)
        try:
            if not initialized:
                timeline_handle.display(fig)
                initialized = True
            else:
                timeline_handle.update(fig)
        finally:
            plt.close(fig)

    # Layout and display
    controls = widgets.VBox([
        widgets.HTML("<b>Per-feature residual timeline (AE vs regressor)</b>"),
        ds_dropdown,
        feat_slider,
        feature_label,
        metrics_info,
    ], layout=widgets.Layout(width="360px", border="1px solid #ddd", padding="10px", margin="0 12px 0 0"))

    clear_output(wait=True)
    display(controls)
    update()
    ds_dropdown.observe(update, names="value")
    feat_slider.observe(update, names="value")

