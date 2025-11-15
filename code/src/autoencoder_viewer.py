from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Sequence
from datetime import datetime
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def display_autoencoder_viewer(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    datasets: Dict[str, np.ndarray],
    *,
    feature_names: Optional[Sequence[str]] = None,
    default_key: Optional[str] = None,
    default_feature_count: int = 1,
    error_threshold: Optional[float] = None,
    fault_data: Optional[Dict[str, pd.DataFrame]] = None,
    scaler: Optional[Any] = None,
    fault_column_name: str = "faultNumber",
    simulation_column_name: str = "simulationRun",
    columns_to_remove: Optional[list[str]] = None,
    default_fault: Optional[int] = None,
    default_simulation: Optional[int] = None,
    show_debug_log: bool = False,
    viewer_key: Optional[str] = "autoencoder_viewer_default",
) -> None:
    """
    Render an interactive widget that overlays actual vs reconstructed signals.

    Args:
        predict_fn: callable that takes an array (samples x features) and returns
            reconstructed samples of identical shape.
        datasets: mapping of dataset label -> array (samples x features).
        feature_names: optional names for each feature (defaults to "f{index}").
        default_key: dataset shown initially.
        default_feature_count: number of features selected initially.
        error_threshold: optional threshold for highlighting high-error features.
        fault_data: optional mapping of dataset label -> raw DataFrame with fault info.
            When provided, enables dynamic fault/simulation selection.
        scaler: optional scaler to transform the data (e.g., StandardScaler).
        fault_column_name: name of the fault column in fault_data DataFrames.
        simulation_column_name: name of the simulation run column.
        columns_to_remove: list of column names to remove before scaling.
        default_fault: default fault number to display.
        default_simulation: default simulation run to display.
        show_debug_log: whether to show the debug log widget.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output, DisplayHandle
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ipywidgets is required for the interactive viewer.") from exc

    # Generate unique instance ID
    instance_id = str(uuid.uuid4())[:8]
    
    # Debug log widget
    log_output = widgets.Output(layout=widgets.Layout(height="100px", border="1px solid #ccc", overflow_y="auto"))
    
    def log(message: str) -> None:
        """Add a timestamped log message."""
        if show_debug_log:
            with log_output:
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[{timestamp}] [{instance_id}] {message}")

    # (Log moved post-guard to avoid duplicate lines when init races)

    # Ensure only one active viewer per key by closing previous widgets
    REGISTRY_NAME = "_AE_VIEWER_REGISTRY"
    if viewer_key is None:
        viewer_key = "autoencoder_viewer_default"
    if REGISTRY_NAME not in globals():
        globals()[REGISTRY_NAME] = {}
    registry = globals()[REGISTRY_NAME]

    # Initialization guard to avoid duplicate concurrent builds
    guard_key = f"{viewer_key}__init_in_progress"
    if registry.get(guard_key, False):
        log("Initialization already in progress for this key. Skipping.")
        return None
    registry[guard_key] = True
    log("=== NEW INSTANCE CREATED ===")
    try:
        prev = registry.get(viewer_key)
        if prev is not None:
            try:
                prev.close()
            except Exception as _e:
                pass
    finally:
        registry[viewer_key] = None  # reserve key early

    # Determine if we're using fault data mode
    use_fault_mode = fault_data is not None and scaler is not None

    # Setup for fault data mode
    available_faults: Dict[str, list[int]] = {}
    available_simulations: Dict[str, list[int]] = {}
    filtered_datasets: Dict[str, np.ndarray] = {}

    if use_fault_mode:
        if columns_to_remove is None:
            columns_to_remove = [fault_column_name, simulation_column_name, "sample"]

        # Extract available faults and simulations for each dataset
        for key, df in fault_data.items():
            if fault_column_name in df.columns:
                available_faults[key] = sorted(df[fault_column_name].unique().astype(int).tolist())
            if simulation_column_name in df.columns:
                available_simulations[key] = sorted(df[simulation_column_name].unique().astype(int).tolist())

        # Set defaults
        if default_fault is None and available_faults:
            first_key = next(iter(available_faults.keys()))
            default_fault = available_faults[first_key][0] if available_faults[first_key] else 0

        if default_simulation is None and available_simulations:
            first_key = next(iter(available_simulations.keys()))
            default_simulation = available_simulations[first_key][0] if available_simulations[first_key] else 1

    def get_filtered_data(dataset_key: str, fault: Optional[int], simulation: Optional[int]) -> np.ndarray:
        """Filter and scale data based on fault and simulation selection."""
        if not use_fault_mode:
            return datasets[dataset_key]

        df = fault_data[dataset_key]

        # Build query
        query_parts = []
        if fault is not None and fault_column_name in df.columns:
            query_parts.append(f"{fault_column_name} == @fault")
        if simulation is not None and simulation_column_name in df.columns:
            query_parts.append(f"{simulation_column_name} == @simulation")

        if query_parts:
            filtered_df = df.query(" and ".join(query_parts))
        else:
            filtered_df = df

        # Drop metadata columns and scale
        feature_df = filtered_df.drop(columns=[col for col in columns_to_remove if col in filtered_df.columns], axis=1)
        scaled_data = scaler.transform(feature_df)

        return scaled_data

    if not datasets and not use_fault_mode:
        raise ValueError("datasets mapping must not be empty.")

    # Determine dataset keys and feature count
    if use_fault_mode:
        dataset_keys = list(fault_data.keys())
        # Get feature count from first dataset
        first_key = dataset_keys[0]
        first_df = fault_data[first_key]
        feature_cols = [col for col in first_df.columns if col not in columns_to_remove]
        feature_count = len(feature_cols)
    else:
        dataset_keys = list(datasets.keys())
        feature_count = next(iter(datasets.values())).shape[1]
        for key, arr in datasets.items():
            if arr.shape[1] != feature_count:
                raise ValueError(f"Dataset '{key}' has {arr.shape[1]} features; expected {feature_count}.")

    if feature_names is None:
        feature_names = tuple(f"f{i}" for i in range(feature_count))

    default_key = default_key or dataset_keys[0]

    # Cache for reconstructions - now includes fault/simulation as cache key
    recon_cache: Dict[tuple, np.ndarray] = {}
    feature_errors_cache: Dict[tuple, Dict[int, float]] = {}
    current_fault = {"value": default_fault}
    current_simulation = {"value": default_simulation}

    def get_cache_key(label: str) -> tuple:
        """Generate cache key based on dataset, fault, and simulation."""
        if use_fault_mode:
            return (label, current_fault["value"], current_simulation["value"])
        return (label,)

    def get_data(label: str) -> tuple[np.ndarray, np.ndarray]:
        cache_key = get_cache_key(label)
        if cache_key not in recon_cache:
            X = get_filtered_data(label, current_fault["value"], current_simulation["value"])
            recon_cache[cache_key] = np.asarray(predict_fn(X))
        else:
            X = get_filtered_data(label, current_fault["value"], current_simulation["value"])
        return X, recon_cache[cache_key]

    def compute_feature_errors(label: str) -> Dict[int, float]:
        """Compute MAE for each feature."""
        cache_key = get_cache_key(label)
        if cache_key not in feature_errors_cache:
            X_source, recon_source = get_data(label)
            errors = {}
            for feat_idx in range(feature_count):
                mae = float(np.mean(np.abs(X_source[:, feat_idx] - recon_source[:, feat_idx])))
                errors[feat_idx] = mae
            feature_errors_cache[cache_key] = errors
        return feature_errors_cache[cache_key]

    def get_error_color(mae: float, threshold: Optional[float]) -> str:
        """Return HTML color based on error magnitude."""
        if threshold is None:
            return "black"
        if mae > threshold:
            return "red"
        elif mae > threshold * 0.5:
            return "orange"
        else:
            return "green"

    def format_feature_option(feat_idx: int, errors: Dict[int, float], threshold: Optional[float]) -> str:
        """Format feature option with error indicator."""
        name = feature_names[feat_idx]
        mae = errors.get(feat_idx, 0.0)
        color = get_error_color(mae, threshold)
        indicator = "⚠️" if threshold and mae > threshold else ""
        return f"{name} (MAE={mae:.4f}) {indicator}"

    def update_feature_options(label: str, sort_by: str) -> None:
        """Update feature selector options based on sorting."""
        errors = compute_feature_errors(label)

        # Create list of (index, name, error)
        feature_list = [(i, feature_names[i], errors[i]) for i in range(feature_count)]

        # Sort based on selection
        if sort_by == "Name (A-Z)":
            feature_list.sort(key=lambda x: x[1])
        elif sort_by == "Error (Low to High)":
            feature_list.sort(key=lambda x: x[2])
        elif sort_by == "Error (High to Low)":
            feature_list.sort(key=lambda x: x[2], reverse=True)

        # Update options with formatted labels
        new_options = [
            (format_feature_option(idx, errors, error_threshold), idx)
            for idx, name, mae in feature_list
        ]

        # Preserve current selection
        current_selection = feature_selector.value
        feature_selector.options = new_options
        # Restore selection if still valid
        feature_selector.value = tuple(v for v in current_selection if v < feature_count)

    # Widgets -----------------------------------------------------------------
    dataset_dropdown = widgets.Dropdown(
        options=dataset_keys,
        value=default_key if default_key in dataset_keys else dataset_keys[0],
        description="Dataset",
        layout=widgets.Layout(width="240px"),
        style={"description_width": "70px"},
    )

    # Fault and simulation selectors (only in fault mode)
    fault_dropdown = None
    simulation_dropdown = None

    if use_fault_mode:
        # Get faults available for the default dataset
        default_dataset_faults = available_faults.get(default_key, [0])
        default_dataset_simulations = available_simulations.get(default_key, [1])

        fault_dropdown = widgets.Dropdown(
            options=default_dataset_faults,
            value=default_fault if default_fault in default_dataset_faults else default_dataset_faults[0],
            description="Fault",
            layout=widgets.Layout(width="180px"),
            style={"description_width": "50px"},
        )

        simulation_dropdown = widgets.Dropdown(
            options=default_dataset_simulations,
            value=default_simulation if default_simulation in default_dataset_simulations else default_dataset_simulations[0],
            description="Simulation",
            layout=widgets.Layout(width="200px"),
            style={"description_width": "70px"},
        )

    sort_dropdown = widgets.Dropdown(
        options=["Name (A-Z)", "Error (Low to High)", "Error (High to Low)"],
        value="Name (A-Z)",
        description="Sort by",
        layout=widgets.Layout(width="240px"),
        style={"description_width": "60px"},
    )

    # Initialize with default sorting
    initial_errors = compute_feature_errors(default_key)
    default_selection = tuple(range(min(default_feature_count, feature_count)))

    feature_selector = widgets.SelectMultiple(
        options=[
            (format_feature_option(i, initial_errors, error_threshold), i)
            for i in range(feature_count)
        ],
        value=default_selection,
        description="Features",
        layout=widgets.Layout(width="100%", height="150px"),
        style={"description_width": "60px"},
    )

    help_select = widgets.HTML(
        value="<small><i>Hold Ctrl/Cmd to select multiple features</i></small>",
        layout=widgets.Layout(width="100%")
    )

    start_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=200,
        step=1,
        description="Start",
        continuous_update=False,
        layout=widgets.Layout(width="100%"),
        style={"description_width": "50px"},
    )

    window_slider = widgets.IntSlider(
        value=200,
        min=50,
        max=600,
        step=10,
        description="Window",
        continuous_update=False,
        layout=widgets.Layout(width="100%"),
        style={"description_width": "50px"},
    )

    metrics_display = widgets.HTML(layout=widgets.Layout(width="100%", margin="4px 0 0 0"))
    threshold_display = widgets.HTML(layout=widgets.Layout(width="100%", margin="0 0 6px 0"))
    # Plot will not be inside a widget Output; use a DisplayHandle instead.
    plot_handle = DisplayHandle()
    plot_initialized = {"done": False}

    def refresh_bounds(label: str) -> None:
        X_source, _ = get_data(label)
        n_samples = X_source.shape[0]
        start_slider.max = max(0, n_samples - 2)
        window_slider.max = max(50, n_samples)
        if start_slider.value > start_slider.max:
            start_slider.value = start_slider.max
        if window_slider.value > window_slider.max:
            window_slider.value = window_slider.max

    def format_metrics(errors: np.ndarray, threshold: Optional[float] = None) -> str:
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))
        max_abs = float(np.max(np.abs(errors)))

        html = f"<b>Overall Errors:</b><br/>MAE={mae:.4f}, RMSE={rmse:.4f}, Max={max_abs:.4f}"

        if threshold is not None:
            if mae > threshold:
                html += f"<br/><span style='color:red;'>⚠️ MAE above threshold ({threshold:.4f})</span>"
            else:
                html += f"<br/><span style='color:green;'>✓ MAE below threshold ({threshold:.4f})</span>"

        return html

    def format_threshold_info(threshold: Optional[float]) -> str:
        if threshold is None:
            return "<small><i>No threshold set</i></small>"
        return f"<small><b>Threshold:</b> {threshold:.4f}<br/>" \
               f"<span style='color:green;'>Green</span>: MAE &lt; {threshold*0.5:.4f}<br/>" \
               f"<span style='color:orange;'>Orange</span>: {threshold*0.5:.4f} ≤ MAE ≤ {threshold:.4f}<br/>" \
               f"<span style='color:red;'>Red ⚠️</span>: MAE &gt; {threshold:.4f}</small>"

    threshold_display.value = format_threshold_info(error_threshold)

    updating = {"active": False}
    queued = {"pending": False}

    def update_plot(*_args: object) -> None:
        log(f"update_plot() called - updating={updating['active']}")
        if updating["active"]:
            # Coalesce multiple triggers into one extra run
            queued["pending"] = True
            log("update_plot() SKIPPED (already updating)")
            return
        updating["active"] = True
        try:
            log("update_plot() EXECUTING")
            label = dataset_dropdown.value
            X_source, recon_source = get_data(label)
            n_samples = X_source.shape[0]
            start = max(0, min(start_slider.value, n_samples - 1))
            window = max(1, min(window_slider.value, n_samples - start))
            end = start + window

            idx_range = np.arange(start, end)
            actual = X_source[start:end]
            recon = recon_source[start:end]
            selected = [val for _, val in feature_selector.options if val in feature_selector.value]
            if not selected:
                selected = list(range(actual.shape[1]))

            errors = actual[:, selected] - recon[:, selected]
            metrics_display.value = format_metrics(errors, error_threshold)

            # Build the figure and update the display handle (not a widget Output)
            fig = plt.figure(figsize=(14, 5))
            ax = fig.add_subplot(111)
            colours = plt.cm.tab20(np.linspace(0.0, 1.0, len(selected)))

            for colour, feat_idx in zip(colours, selected):
                name = feature_names[feat_idx] if feature_names else f"f{feat_idx}"
                ax.plot(
                    idx_range,
                    actual[:, feat_idx],
                    color=colour,
                    label=f"{name} (actual)",
                    linewidth=1.5,
                )
                ax.plot(
                    idx_range,
                    recon[:, feat_idx],
                    linestyle="--",
                    color=colour,
                    label=f"{name} (recon)",
                    linewidth=1.5,
                )

            ax.set_title(f"Autoencoder reconstruction: {label} [{start}:{end}]")
            ax.set_xlabel("Sample index")
            ax.set_ylabel("Scaled value")
            ax.grid(alpha=0.25)

            # Create a single legend positioned outside the plot
            if len(selected) > 0:
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    fontsize="small",
                    frameon=True,
                    fancybox=True,
                )

            plt.tight_layout()
            # Update the single display handle to avoid duplicate plots
            try:
                if not plot_initialized["done"]:
                    plot_handle.display(fig)
                    plot_initialized["done"] = True
                else:
                    plot_handle.update(fig)
            finally:
                plt.close(fig)
            log("update_plot() COMPLETED")
        finally:
            updating["active"] = False
            if queued.get("pending", False):
                queued["pending"] = False
                log("update_plot() running queued refresh")
                update_plot()

    def on_dataset_change(change: dict) -> None:
        log(f"on_dataset_change() called: {change['old']} -> {change['new']}, updating={updating['active']}")
        if updating["active"]:
            log("on_dataset_change() SKIPPED (already updating)")
            return
        updating["active"] = True
        try:
            new_dataset = change["new"]

            # Update fault and simulation options based on new dataset
            if use_fault_mode and fault_dropdown is not None and simulation_dropdown is not None:
                dataset_faults = available_faults.get(new_dataset, [0])
                dataset_simulations = available_simulations.get(new_dataset, [1])

                # Temporarily unobserve to prevent cascading updates
                log("Unobserving fault and simulation dropdowns")
                fault_dropdown.unobserve(on_fault_change, names="value")
                simulation_dropdown.unobserve(on_simulation_change, names="value")

                try:
                    # Update fault dropdown options
                    old_fault = fault_dropdown.value
                    fault_dropdown.options = dataset_faults
                    if old_fault in dataset_faults:
                        fault_dropdown.value = old_fault
                        current_fault["value"] = old_fault
                        log(f"Kept fault={old_fault}")
                    else:
                        new_fault = dataset_faults[0] if dataset_faults else 0
                        fault_dropdown.value = new_fault
                        current_fault["value"] = new_fault
                        log(f"Changed fault to {new_fault}")

                    # Update simulation dropdown options
                    old_simulation = simulation_dropdown.value
                    simulation_dropdown.options = dataset_simulations
                    if old_simulation in dataset_simulations:
                        simulation_dropdown.value = old_simulation
                        current_simulation["value"] = old_simulation
                        log(f"Kept simulation={old_simulation}")
                    else:
                        new_simulation = dataset_simulations[0] if dataset_simulations else 1
                        simulation_dropdown.value = new_simulation
                        current_simulation["value"] = new_simulation
                        log(f"Changed simulation to {new_simulation}")

                    # Clear cache since fault/simulation may have changed
                    cache_key = get_cache_key(new_dataset)
                    if cache_key in recon_cache:
                        del recon_cache[cache_key]
                    if cache_key in feature_errors_cache:
                        del feature_errors_cache[cache_key]
                finally:
                    # Re-observe
                    log("Re-observing fault and simulation dropdowns")
                    fault_dropdown.observe(on_fault_change, names="value")
                    simulation_dropdown.observe(on_simulation_change, names="value")

            # Keep the lock while updating bounds and features to prevent cascading updates
            log("Updating bounds and features while locked")
            refresh_bounds(new_dataset)
            update_feature_options(new_dataset, sort_dropdown.value)
            log("on_dataset_change() calling update_plot()")
        finally:
            updating["active"] = False
            log("on_dataset_change() releasing lock")
        # Call update_plot AFTER releasing lock, but it will set the lock again
        update_plot()

    def on_fault_change(change: dict) -> None:
        """Handle fault selection change."""
        if change["type"] == "change" and change["name"] == "value":
            log(f"on_fault_change() called: {change['old']} -> {change['new']}, updating={updating['active']}")
            if updating["active"]:
                log("on_fault_change() SKIPPED (already updating)")
                return
            current_fault["value"] = change["new"]
            # Clear cache for this dataset to force recomputation
            label = dataset_dropdown.value
            cache_key = get_cache_key(label)
            if cache_key in recon_cache:
                del recon_cache[cache_key]
            if cache_key in feature_errors_cache:
                del feature_errors_cache[cache_key]
            refresh_bounds(label)
            update_feature_options(label, sort_dropdown.value)
            update_plot()

    def on_simulation_change(change: dict) -> None:
        """Handle simulation selection change."""
        if change["type"] == "change" and change["name"] == "value":
            log(f"on_simulation_change() called: {change['old']} -> {change['new']}, updating={updating['active']}")
            if updating["active"]:
                log("on_simulation_change() SKIPPED (already updating)")
                return
            current_simulation["value"] = change["new"]
            # Clear cache for this dataset to force recomputation
            label = dataset_dropdown.value
            cache_key = get_cache_key(label)
            if cache_key in recon_cache:
                del recon_cache[cache_key]
            if cache_key in feature_errors_cache:
                del feature_errors_cache[cache_key]
            refresh_bounds(label)
            update_feature_options(label, sort_dropdown.value)
            update_plot()

    def on_sort_change(change: dict) -> None:
        log(f"on_sort_change() called: {change['new']}")
        if change["type"] == "change" and change["name"] == "value":
            update_feature_options(dataset_dropdown.value, change["new"])
            update_plot()

    # (Observers wired after first render to avoid duplicate initial updates)

    help_text = widgets.HTML(
        "<b>Overlay actual vs reconstructed signals for selected features.</b>",
        layout=widgets.Layout(width="100%", margin="0 0 5px 0")
    )

    # Row 1: Dataset, Fault, Simulation, Sort
    row1_widgets = [dataset_dropdown]
    if use_fault_mode and fault_dropdown is not None and simulation_dropdown is not None:
        row1_widgets.extend([fault_dropdown, simulation_dropdown])
    row1_widgets.append(sort_dropdown)

    row1 = widgets.HBox(
        row1_widgets,
        layout=widgets.Layout(width="100%", justify_content="flex-start", margin="0 0 8px 0")
    )

    # Row 2: Group feature selection, window controls, and metrics into panels
    feature_panel = widgets.VBox(
        [
            widgets.HTML("<b>Feature selection</b>"),
            feature_selector,
            help_select,
        ],
        layout=widgets.Layout(width="30%", min_width="240px", padding="0 16px 0 0", flex="1 1 0%")
    )

    slider_panel = widgets.VBox(
        [
            widgets.HTML("<b>Sample window</b>"),
            start_slider,
            window_slider,
        ],
        layout=widgets.Layout(width="36%", min_width="260px", padding="0 16px", flex="1 1 0%")
    )

    metrics_panel = widgets.VBox(
        [
            widgets.HTML("<b>Error summary</b>"),
            metrics_display,
            threshold_display,
        ],
        layout=widgets.Layout(width="34%", min_width="220px", padding="0 0 0 16px", flex="1 1 0%")
    )

    row2 = widgets.HBox(
        [feature_panel, slider_panel, metrics_panel],
        layout=widgets.Layout(width="100%", justify_content="space-between", align_items="flex-start")
    )

    # Build controls list
    controls_list = [help_text, row1, row2]

    # Add debug log if enabled
    if show_debug_log:
        log_label = widgets.HTML(f"<b>Debug Log (Instance: {instance_id}):</b>")
        controls_list.append(log_label)
        controls_list.append(log_output)

    controls = widgets.VBox(
        controls_list,
        layout=widgets.Layout(width="95%", border="1px solid #ddd", padding="8px", margin="0 0 8px 0"),
    )
    # Display only the controls as a widget; the plot will be rendered
    # below using a non-widget display handle.
    container = controls
    # register the new container as the active viewer for this key
    registry[viewer_key] = container

    clear_output(wait=True)
    display(container)
    # Provide an initial empty content for plot area to reserve space
    # (optional; update_plot will overwrite it).
    # plot_handle.display(HTML("<small>Rendering...</small>"))

    # Release initialization guard
    registry[guard_key] = False
    log("Initializing widget")
    # Prepare sliders based on data before wiring observers
    refresh_bounds(dataset_dropdown.value)
    # Render once
    update_plot()

    # Now wire up observers (post-initialization)
    dataset_dropdown.observe(on_dataset_change, names="value")
    sort_dropdown.observe(on_sort_change, names="value")
    feature_selector.observe(lambda change: (log("feature_selector changed"), update_plot())[1], names="value")
    start_slider.observe(lambda change: (log("start_slider changed"), update_plot())[1], names="value")
    window_slider.observe(lambda change: (log("window_slider changed"), update_plot())[1], names="value")

    # Register fault mode event handlers
    if use_fault_mode and fault_dropdown is not None and simulation_dropdown is not None:
        fault_dropdown.observe(on_fault_change, names="value")
        simulation_dropdown.observe(on_simulation_change, names="value")
