from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


@dataclass
class DetectionResultsVisualizer:
    """
    Standalone plotting helper to explore anomaly detection benchmark results.

    Expects:
        - df_results: raw long-format metrics per (model, fault, simulation_run)
        - summary_df: aggregated metrics per (model, fault)
    """

    df_results: pd.DataFrame
    summary_df: pd.DataFrame

    def __post_init__(self) -> None:
        required_cols = {"model", "faultNumber"}
        missing_raw = required_cols - set(self.df_results.columns)
        missing_summary = required_cols - set(self.summary_df.columns)
        if missing_raw:
            raise ValueError(f"df_results missing columns: {missing_raw}")
        if missing_summary:
            raise ValueError(f"summary_df missing columns: {missing_summary}")

        self.models: Sequence[str] = tuple(sorted(self.df_results["model"].unique()))
        self.fault_numbers: Sequence[int] = tuple(
            sorted(self.summary_df["faultNumber"].unique())
        )

    # ------------------------------------------------------------------ #
    # Tabular utilities                                                  #
    # ------------------------------------------------------------------ #

    def summary_table(self) -> pd.DataFrame:
        """
        Compute comparative summary statistics per model.
        """
        stats_rows = []
        for model in self.models:
            model_rows = self.df_results[self.df_results["model"] == model]
            arl0 = model_rows["ARL0"].dropna()
            arl1 = model_rows["ARL1"].dropna()

            detection_fraction = model_rows["detection_fraction"]
            false_alarm_rate = (
                model_rows["ARL0"].notna().sum() / len(model_rows) * 100.0
                if len(model_rows)
                else np.nan
            )
            miss_rate = (
                model_rows["ARL1"].isna().sum() / len(model_rows) * 100.0
                if len(model_rows)
                else np.nan
            )

            stats_rows.append(
                {
                    "Model": model,
                    "Mean ARL0": arl0.mean(),
                    "Std ARL0": arl0.std(),
                    "Mean ARL1": arl1.mean(),
                    "Std ARL1": arl1.std(),
                    "Avg Detection Rate (%)": detection_fraction.mean() * 100,
                    "False Alarm Rate (%)": false_alarm_rate,
                    "Miss Rate (%)": miss_rate,
                    "Precision": model_rows["precision"].dropna().mean(),
                    "Recall": model_rows["recall"].dropna().mean(),
                    "Specificity": model_rows["specificity"].dropna().mean(),
                    "Accuracy": model_rows["accuracy"].dropna().mean(),
                    "F1": model_rows["f1"].dropna().mean(),
                    "FPR": model_rows["false_positive_rate"].dropna().mean(),
                    "FNR": model_rows["false_negative_rate"].dropna().mean(),
                }
            )

        summary = pd.DataFrame(stats_rows)
        numeric_cols = summary.select_dtypes(include=[np.number]).columns
        summary[numeric_cols] = summary[numeric_cols].round(3)
        return summary

    # ------------------------------------------------------------------ #
    # Visualization helpers                                              #
    # ------------------------------------------------------------------ #

    def plot_arl_profiles(
        self,
        *,
        models: Optional[Iterable[str]] = None,
        figsize: tuple[int, int] = (10, 5),
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Line plot of conditional ARL1 across faults for each model.
        """
        selected_models = tuple(models) if models else self.models
        pivot = (
            self.summary_df[self.summary_df["model"].isin(selected_models)]
            .pivot(index="faultNumber", columns="model", values="conditional_ARL1")
            .sort_index()
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        for model in pivot.columns:
            ax.plot(pivot.index, pivot[model], marker="o", label=model)
        ax.set_title("Detection Delay (ARL1) vs Fault Number")
        ax.set_xlabel("Fault number")
        ax.set_ylabel("Conditional ARL1")
        ax.grid(alpha=0.3)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        ax.set_xticks(pivot.index)
        ax.set_xticklabels([str(int(x)) for x in pivot.index], rotation=45, ha='right')
        return ax

    def plot_detection_fraction(
        self,
        *,
        models: Optional[Iterable[str]] = None,
        figsize: tuple[int, int] = (10, 5),
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Line plot of detection fraction across faults for each model.
        """
        selected_models = tuple(models) if models else self.models
        pivot = (
            self.summary_df[self.summary_df["model"].isin(selected_models)]
            .pivot(index="faultNumber", columns="model", values="avg_detection_fraction")
            .sort_index()
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        for model in pivot.columns:
            ax.plot(pivot.index, pivot[model], marker="o", label=model)
        ax.set_title("Detection Fraction vs Fault Number")
        ax.set_xlabel("Fault number")
        ax.set_ylabel("Average detection fraction")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        ax.set_xticks(pivot.index)
        ax.set_xticklabels([str(int(x)) for x in pivot.index], rotation=45, ha='right')
        return ax

    def plot_precision_recall(
        self,
        *,
        models: Optional[Iterable[str]] = None,
        figsize: tuple[int, int] = (10, 5),
    ) -> tuple[plt.Axes, plt.Axes]:
        """Side-by-side line plots for recall and precision across faults."""
        selected_models = tuple(models) if models else self.models
        rec = (
            self.summary_df[self.summary_df["model"].isin(selected_models)]
            .pivot(index="faultNumber", columns="model", values="recall")
            .sort_index()
        )
        prec = (
            self.summary_df[self.summary_df["model"].isin(selected_models)]
            .pivot(index="faultNumber", columns="model", values="precision")
            .sort_index()
        )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        for model in rec.columns:
            ax1.plot(rec.index, rec[model], marker="o", label=model)
        ax1.set_title("Recall (TPR) vs Fault Number")
        ax1.set_xlabel("Fault number")
        ax1.set_ylabel("Recall")
        ax1.set_ylim(0, 1)
        ax1.grid(alpha=0.3)
        ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        ax1.set_xticks(rec.index)
        ax1.set_xticklabels([str(int(x)) for x in rec.index], rotation=45, ha='right')
        for model in prec.columns:
            ax2.plot(prec.index, prec[model], marker="o", label=model)
        ax2.set_title("Precision vs Fault Number")
        ax2.set_xlabel("Fault number")
        ax2.set_ylabel("Precision")
        ax2.set_ylim(0, 1)
        ax2.grid(alpha=0.3)
        ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        ax2.set_xticks(prec.index)
        ax2.set_xticklabels([str(int(x)) for x in prec.index], rotation=45, ha='right')
        return ax1, ax2

    def plot_specificity_accuracy(
        self,
        *,
        models: Optional[Iterable[str]] = None,
        figsize: tuple[int, int] = (10, 5),
    ) -> tuple[plt.Axes, plt.Axes]:
        """Side-by-side line plots for specificity and accuracy across faults."""
        selected_models = tuple(models) if models else self.models
        spec = (
            self.summary_df[self.summary_df["model"].isin(selected_models)]
            .pivot(index="faultNumber", columns="model", values="specificity")
            .sort_index()
        )
        acc = (
            self.summary_df[self.summary_df["model"].isin(selected_models)]
            .pivot(index="faultNumber", columns="model", values="accuracy")
            .sort_index()
        )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        for model in spec.columns:
            ax1.plot(spec.index, spec[model], marker="o", label=model)
        ax1.set_title("Specificity (TNR) vs Fault Number")
        ax1.set_xlabel("Fault number")
        ax1.set_ylabel("Specificity")
        ax1.set_ylim(0, 1)
        ax1.grid(alpha=0.3)
        ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        ax1.set_xticks(spec.index)
        ax1.set_xticklabels([str(int(x)) for x in spec.index], rotation=45, ha='right')
        for model in acc.columns:
            ax2.plot(acc.index, acc[model], marker="o", label=model)
        ax2.set_title("Accuracy vs Fault Number")
        ax2.set_xlabel("Fault number")
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1)
        ax2.grid(alpha=0.3)
        ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        ax2.set_xticks(acc.index)
        ax2.set_xticklabels([str(int(x)) for x in acc.index], rotation=45, ha='right')
        return ax1, ax2

    def heatmap(
        self,
        metric: str,
        *,
        cmap: str = "viridis",
        figsize: tuple[int, int] = (8, 6),
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Heatmap of an aggregated metric across (fault, model).
        """
        if metric not in self.summary_df.columns:
            raise ValueError(f"Metric '{metric}' not found in summary_df columns.")

        pivot = (
            self.summary_df.pivot(index="faultNumber", columns="model", values=metric)
            .sort_index()
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            ax=ax,
            cbar_kws={"label": metric},
        )
        ax.set_title(f"{metric} heatmap")
        ax.set_xlabel("Model")
        ax.set_ylabel("Fault number")
        ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
        ax.set_yticklabels([str(int(y.get_text())) for y in ax.get_yticklabels()], rotation=0)
        return ax

    def bar_false_alarm_rate(
        self,
        *,
        figsize: tuple[int, int] = (8, 5),
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Bar chart of false alarm rate (occurrence of non-null ARL0) per model.
        """
        rates = []
        for model in self.models:
            rows = self.df_results[self.df_results["model"] == model]
            rate = rows["ARL0"].notna().mean() * 100 if len(rows) else np.nan
            rates.append({"model": model, "False Alarm Rate (%)": rate})

        rate_df = pd.DataFrame(rates)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(
            data=rate_df,
            x="model",
            y="False Alarm Rate (%)",
            palette="Set2",
            ax=ax,
        )
        ax.set_title("False Alarm Rate by Model")
        ax.set_xlabel("Model")
        ax.set_ylabel("False alarm rate (%)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticklabels(rate_df["model"], rotation=45, ha='right')
        return ax

    def plot_metrics_comparison_bars(
        self,
        metrics: Sequence[str],
        *,
        figsize: tuple[int, int] = (14, 8),
    ) -> plt.Figure:
        """
        Create simple bar charts showing overall average performance for each model.
        Shows mean ± std across all faults and simulation runs.
        """
        import numpy as np

        n_metrics = len(metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_metrics == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_metrics > 1 else axes

        for idx, metric in enumerate(metrics):
            if metric not in self.df_results.columns:
                continue

            ax = axes[idx]

            # Calculate mean and std for each model across all faults
            means = []
            stds = []
            model_names = []

            for model in self.models:
                data = self.df_results[self.df_results['model'] == model][metric].dropna()
                if not data.empty:
                    means.append(data.mean())
                    stds.append(data.std())
                    model_names.append(model)

            # Create bar chart with error bars
            x_pos = np.arange(len(model_names))
            colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))

            bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                         color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=11, fontweight='bold')
            ax.set_ylabel('Value', fontsize=10)
            ax.set_ylim(0, 1.0)
            ax.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=8)

        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Overall Model Performance (Mean ± Std across all faults)',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_metrics_per_fault_bars(
        self,
        metrics: Sequence[str],
        *,
        figsize: tuple[int, int] = (18, 10),
    ) -> plt.Figure:
        """
        Create grouped bar charts showing performance for each fault number.
        Similar to the combined ARL plot style - each fault shows all models side-by-side.
        """
        import numpy as np

        n_metrics = len(metrics)
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_metrics == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_metrics > 1 else axes

        fault_numbers = sorted(self.df_results['faultNumber'].unique())
        n_models = len(self.models)
        n_faults = len(fault_numbers)
        colors = plt.cm.Set3(np.linspace(0, 1, n_models))

        for idx, metric in enumerate(metrics):
            if metric not in self.summary_df.columns:
                continue

            ax = axes[idx]

            # Get pivot table for this metric
            metric_pivot = self.summary_df.pivot(
                index='faultNumber',
                columns='model',
                values=metric
            )

            # Set up x positions for grouped bars
            x = np.arange(n_faults)
            width = 0.8 / n_models  # Divide space among models

            # Create bars for each model
            for model_idx, model in enumerate(self.models):
                if model not in metric_pivot.columns:
                    continue

                values = metric_pivot[model].values
                offset = (model_idx - n_models/2 + 0.5) * width

                bars = ax.bar(x + offset, values, width,
                            label=model, color=colors[model_idx],
                            alpha=0.8, edgecolor='black', linewidth=0.5)

                # Add value labels on bars (only if not too crowded)
                if n_faults <= 10:
                    for bar in bars:
                        height = bar.get_height()
                        if not np.isnan(height):
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.2f}', ha='center', va='bottom', fontsize=7)

            ax.set_xlabel('Fault Number', fontsize=10)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
            ax.set_title(f'{metric.replace("_", " ").title()} by Fault',
                        fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([str(int(f)) for f in fault_numbers], fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8, loc='lower right')
            ax.grid(axis='y', alpha=0.3)

        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Model Performance Comparison by Fault Number',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig


# ---------------------------------------------------------------------- #
# Interactive residual diagnostics (in-control / out-of-control selector)
# ---------------------------------------------------------------------- #

def display_residual_viewer(
    regressor: Any,
    dataset_map: dict[str, np.ndarray],
    *,
    threshold: Optional[float] = None,
    threshold_percentile: float = 99.5,
    baseline_key: Optional[str] = None,
    sample_size: Optional[int] = 4000,
    random_state: Optional[int] = None,
    viewer_key: Optional[str] = "residual_viewer_default",
) -> None:
    """
    Show residual scatter + histogram with a dataset selector.

    - dataset_map: mapping label -> feature matrix (np.ndarray). Typical keys:
        {"In-control": X_ic, "Out-of-control": X_oc}
    - threshold: if None, computed as the given percentile over predicted residuals
        of the baseline_key dataset (defaults to the first dataset).
    - Renders plots via a single DisplayHandle each (no widgets.Output), so updates
      replace prior figures rather than stacking new ones.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output, DisplayHandle
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ipywidgets is required for the interactive viewer.") from exc

    labels = list(dataset_map.keys())
    if not labels:
        raise ValueError("dataset_map must not be empty")
    default_label = labels[0]
    baseline_key = baseline_key or default_label

    # Pre-compute residuals and cache them per dataset
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def get_residuals(label: str) -> tuple[np.ndarray, np.ndarray]:
        if label not in cache:
            X = dataset_map[label]
            true_res = regressor.autoencoder_residuals(X).ravel()
            pred_res = regressor.predict(X).ravel()
            cache[label] = (true_res, pred_res)
        return cache[label]

    # Compute threshold if not provided
    if threshold is None:
        _, base_pred = get_residuals(baseline_key)
        vals = base_pred
        if sample_size is not None and len(vals) > sample_size:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(vals), size=sample_size, replace=False)
            vals = vals[idx]
        threshold = float(np.percentile(vals, threshold_percentile))

    # Controls
    dataset_dropdown = widgets.Dropdown(
        options=labels,
        value=default_label,
        description="Dataset",
        layout=widgets.Layout(width="300px"),
        style={"description_width": "80px"},
    )
    metrics_info = widgets.HTML()
    threshold_info = widgets.HTML(
        value=f"<small><b>Threshold:</b> {threshold:.6f} (p{threshold_percentile} of '{baseline_key}')</small>"
    )

    # Plots via display handles
    scatter_handle = DisplayHandle()
    hist_handle = DisplayHandle()
    initialized = {"scatter": False, "hist": False}

    def fmt_metrics(true_res: np.ndarray, pred_res: np.ndarray) -> str:
        err = true_res - pred_res
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        return f"<b>MAE:</b> {mae:.5f} &nbsp; <b>RMSE:</b> {rmse:.5f} &nbsp; <b>N:</b> {len(err)}"

    def update_plots(_=None) -> None:
        label = dataset_dropdown.value
        true_res, pred_res = get_residuals(label)
        metrics_info.value = fmt_metrics(true_res, pred_res)

        # Scatter (optionally subsample)
        x = true_res
        y = pred_res
        if sample_size is not None and len(x) > sample_size:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(x), size=sample_size, replace=False)
            x = x[idx]
            y = y[idx]

        fig1 = plt.figure(figsize=(6.5, 5))
        ax1 = fig1.add_subplot(111)
        ax1.scatter(x, y, alpha=0.35, s=10, color="steelblue", label="samples")
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        ax1.plot([lo, hi], [lo, hi], linestyle="--", color="red", label="ideal")
        ax1.set_xlabel("AE residual (teacher)")
        ax1.set_ylabel("Regressor residual (student)")
        ax1.set_title(f"Residual fit: {label}")
        ax1.legend()
        plt.tight_layout()
        try:
            if not initialized["scatter"]:
                scatter_handle.display(fig1)
                initialized["scatter"] = True
            else:
                scatter_handle.update(fig1)
        finally:
            plt.close(fig1)

        # Histogram
        fig2 = plt.figure(figsize=(6.5, 5))
        ax2 = fig2.add_subplot(111)
        ax2.hist(pred_res, bins=60, alpha=0.75, color="tab:blue")
        if threshold is not None:
            ax2.axvline(threshold, color="darkorange", linestyle="--", label=f"threshold={threshold:.5f}")
            ax2.legend()
        ax2.set_xlabel("Predicted residual")
        ax2.set_ylabel("Count")
        ax2.set_title(f"Residual distribution: {label}")
        plt.tight_layout()
        try:
            if not initialized["hist"]:
                hist_handle.display(fig2)
                initialized["hist"] = True
            else:
                hist_handle.update(fig2)
        finally:
            plt.close(fig2)

    # Layout: controls as widget; plots render below via handles
    controls = widgets.VBox([
        widgets.HTML("<b>Residual diagnostics (AE vs regressor)</b>"),
        dataset_dropdown,
        metrics_info,
        threshold_info,
    ], layout=widgets.Layout(width="340px", border="1px solid #ddd", padding="10px", margin="0 12px 0 0"))

    clear_output(wait=True)
    display(controls)
    update_plots()
    dataset_dropdown.observe(update_plots, names="value")
