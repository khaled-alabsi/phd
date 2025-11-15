"""
Deep Threshold CUSUM (DeepT-CUSUM) Visualization Utilities

Provides comprehensive plotting functions for:
- Threshold evolution over time
- CUSUM statistic with adaptive threshold
- Feedback analysis
- Comparison with fixed CUSUM
- Performance metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from numpy.typing import NDArray


class DeepTCUSUMVisualizer:
    """Visualization utilities for DeepT-CUSUM detector."""

    @staticmethod
    def plot_threshold_evolution(threshold_history: Dict[str, List],
                                 X_test: NDArray,
                                 predictions: NDArray,
                                 fixed_h: Optional[float] = None,
                                 fault_injection_point: int = 0,
                                 title_suffix: str = "",
                                 figsize: tuple = (15, 10)):
        """
        Plot threshold evolution over time.

        Shows:
        1. Adaptive threshold h_t
        2. CUSUM statistic S_t
        3. Binary predictions
        4. Selected feature time series

        Args:
            threshold_history: Dict with 'h' and 'cusum_stat'
            X_test: Test data
            predictions: Binary predictions
            fixed_h: Fixed threshold for comparison (optional)
            fault_injection_point: Where fault starts
            title_suffix: Additional text for title
            figsize: Figure size
        """
        h_values = threshold_history['h']
        cusum_stats = threshold_history['cusum_stat']
        time_points = np.arange(len(h_values))

        fig, axes = plt.subplots(4, 1, figsize=figsize)
        fig.suptitle(f'DeepT-CUSUM: Threshold Evolution {title_suffix}', fontsize=14, fontweight='bold')

        # Plot 1: Adaptive Threshold h_t
        axes[0].plot(time_points, h_values, label='Adaptive h(t)', color='blue', linewidth=2)
        if fixed_h is not None:
            axes[0].axhline(y=fixed_h, color='red', linestyle='--', linewidth=2, label=f'Fixed h={fixed_h:.2f}')
        if fault_injection_point > 0:
            axes[0].axvline(x=fault_injection_point, color='orange', linestyle=':', linewidth=2, label='Fault starts')
        axes[0].set_ylabel('Threshold h(t)', fontsize=11)
        axes[0].set_title('Adaptive Threshold Over Time', fontsize=12)
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: CUSUM Statistic with Threshold
        axes[1].plot(time_points, cusum_stats, label='CUSUM S(t)', color='green', linewidth=2)
        axes[1].plot(time_points, h_values, label='Adaptive threshold h(t)', color='blue', linestyle='--', linewidth=2)
        if fixed_h is not None:
            axes[1].axhline(y=fixed_h, color='red', linestyle=':', linewidth=1.5, label=f'Fixed h={fixed_h:.2f}')
        if fault_injection_point > 0:
            axes[1].axvline(x=fault_injection_point, color='orange', linestyle=':', linewidth=2)
        axes[1].fill_between(time_points, 0, cusum_stats, where=(np.array(cusum_stats) > np.array(h_values)),
                            color='red', alpha=0.2, label='Exceeds threshold')
        axes[1].set_ylabel('CUSUM Statistic', fontsize=11)
        axes[1].set_title('CUSUM Statistic vs Adaptive Threshold', fontsize=12)
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Binary Predictions
        axes[2].scatter(time_points[predictions == 0], predictions[predictions == 0],
                       c='green', marker='|', s=100, label='Normal', alpha=0.6)
        axes[2].scatter(time_points[predictions == 1], predictions[predictions == 1],
                       c='red', marker='|', s=100, label='Anomaly', alpha=0.8)
        if fault_injection_point > 0:
            axes[2].axvline(x=fault_injection_point, color='orange', linestyle=':', linewidth=2, label='Fault starts')
        axes[2].set_ylabel('Prediction', fontsize=11)
        axes[2].set_ylim([-0.1, 1.1])
        axes[2].set_yticks([0, 1])
        axes[2].set_yticklabels(['Normal', 'Anomaly'])
        axes[2].set_title('Detection Results', fontsize=12)
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

        # Plot 4: Sample Feature Time Series (feature 0)
        axes[3].plot(time_points, X_test[:len(time_points), 0], color='purple', linewidth=1.5, label='Feature 0')
        if fault_injection_point > 0:
            axes[3].axvline(x=fault_injection_point, color='orange', linestyle=':', linewidth=2, label='Fault starts')
        # Highlight anomalies
        anomaly_indices = time_points[predictions == 1]
        axes[3].scatter(anomaly_indices, X_test[anomaly_indices, 0],
                       c='red', s=50, alpha=0.6, label='Detected anomalies', zorder=5)
        axes[3].set_ylabel('Feature Value', fontsize=11)
        axes[3].set_xlabel('Time', fontsize=11)
        axes[3].set_title('Sample Feature (Dimension 0)', fontsize=12)
        axes[3].legend(loc='upper right')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_feedback_analysis(threshold_history: Dict[str, List],
                              predictions: NDArray,
                              title_suffix: str = "",
                              figsize: tuple = (15, 8)):
        """
        Analyze relationship between CUSUM state and predicted threshold.

        Shows:
        1. Scatter: S_{t-1} vs h_t (feedback relationship)
        2. Threshold distribution (normal vs anomaly)
        3. CUSUM distribution (normal vs anomaly)

        Args:
            threshold_history: Dict with 'h' and 'cusum_stat'
            predictions: Binary predictions
            title_suffix: Additional text for title
            figsize: Figure size
        """
        h_values = np.array(threshold_history['h'])
        cusum_stats = np.array(threshold_history['cusum_stat'])

        # S_{t-1} is cusum_stats shifted by 1
        S_prev = np.concatenate([[0.0], cusum_stats[:-1]])

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'DeepT-CUSUM: Feedback Analysis {title_suffix}', fontsize=14, fontweight='bold')

        # Plot 1: Scatter S_{t-1} vs h_t (colored by prediction)
        normal_mask = predictions == 0
        anomaly_mask = predictions == 1

        axes[0, 0].scatter(S_prev[normal_mask], h_values[normal_mask],
                          c='green', alpha=0.5, s=30, label='Normal')
        axes[0, 0].scatter(S_prev[anomaly_mask], h_values[anomaly_mask],
                          c='red', alpha=0.7, s=30, label='Anomaly')
        axes[0, 0].set_xlabel('Previous CUSUM S(t-1)', fontsize=11)
        axes[0, 0].set_ylabel('Predicted Threshold h(t)', fontsize=11)
        axes[0, 0].set_title('Feedback Relationship: S(t-1) → h(t)', fontsize=12)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Threshold distribution
        axes[0, 1].hist(h_values[normal_mask], bins=30, alpha=0.6, color='green',
                       label=f'Normal (mean={h_values[normal_mask].mean():.2f})', density=True)
        axes[0, 1].hist(h_values[anomaly_mask], bins=30, alpha=0.6, color='red',
                       label=f'Anomaly (mean={h_values[anomaly_mask].mean():.2f})', density=True)
        axes[0, 1].set_xlabel('Threshold h(t)', fontsize=11)
        axes[0, 1].set_ylabel('Density', fontsize=11)
        axes[0, 1].set_title('Threshold Distribution', fontsize=12)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: CUSUM distribution
        axes[1, 0].hist(cusum_stats[normal_mask], bins=30, alpha=0.6, color='green',
                       label=f'Normal (mean={cusum_stats[normal_mask].mean():.2f})', density=True)
        axes[1, 0].hist(cusum_stats[anomaly_mask], bins=30, alpha=0.6, color='red',
                       label=f'Anomaly (mean={cusum_stats[anomaly_mask].mean():.2f})', density=True)
        axes[1, 0].set_xlabel('CUSUM Statistic S(t)', fontsize=11)
        axes[1, 0].set_ylabel('Density', fontsize=11)
        axes[1, 0].set_title('CUSUM Distribution', fontsize=12)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Threshold adaptation rate (change in h over time)
        h_changes = np.abs(np.diff(h_values))
        axes[1, 1].plot(h_changes, color='purple', linewidth=1.5, alpha=0.7)
        axes[1, 1].axhline(y=np.mean(h_changes), color='blue', linestyle='--',
                          label=f'Mean change={np.mean(h_changes):.3f}')
        axes[1, 1].set_xlabel('Time', fontsize=11)
        axes[1, 1].set_ylabel('|Δh(t)|', fontsize=11)
        axes[1, 1].set_title('Threshold Adaptation Rate', fontsize=12)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_comparison(fixed_predictions: NDArray,
                       deept_predictions: NDArray,
                       X_test: NDArray,
                       threshold_history: Optional[Dict[str, List]] = None,
                       fixed_h: Optional[float] = None,
                       fault_injection_point: int = 0,
                       title: str = ""):
        """
        Compare DeepT-CUSUM with Fixed CUSUM.

        Shows:
        1. Detection comparison
        2. CUSUM statistics comparison
        3. Confusion matrix
        4. Performance metrics

        Args:
            fixed_predictions: Predictions from fixed CUSUM
            deept_predictions: Predictions from DeepT-CUSUM
            X_test: Test data
            threshold_history: DeepT-CUSUM threshold history (optional)
            fixed_h: Fixed threshold value (optional)
            fault_injection_point: Where fault starts
            title: Plot title suffix
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'DeepT-CUSUM vs Fixed CUSUM {title}', fontsize=14, fontweight='bold')

        time_points = np.arange(len(fixed_predictions))

        # Plot 1: Detection Comparison
        axes[0, 0].scatter(time_points[fixed_predictions == 0], fixed_predictions[fixed_predictions == 0],
                          c='lightgreen', marker='s', s=50, label='Fixed: Normal', alpha=0.6)
        axes[0, 0].scatter(time_points[fixed_predictions == 1], fixed_predictions[fixed_predictions == 1] + 0.05,
                          c='salmon', marker='s', s=50, label='Fixed: Anomaly', alpha=0.6)
        axes[0, 0].scatter(time_points[deept_predictions == 0], deept_predictions[deept_predictions == 0] - 0.05,
                          c='green', marker='o', s=50, label='DeepT: Normal', alpha=0.6)
        axes[0, 0].scatter(time_points[deept_predictions == 1], deept_predictions[deept_predictions == 1],
                          c='red', marker='o', s=50, label='DeepT: Anomaly', alpha=0.6)
        if fault_injection_point > 0:
            axes[0, 0].axvline(x=fault_injection_point, color='orange', linestyle=':', linewidth=2,
                              label='Fault starts')
        axes[0, 0].set_ylabel('Detection', fontsize=11)
        axes[0, 0].set_ylim([-0.2, 1.2])
        axes[0, 0].set_title('Detection Results Comparison', fontsize=12)
        axes[0, 0].legend(loc='upper right', fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Agreement/Disagreement
        agreement = (fixed_predictions == deept_predictions).astype(int)
        both_detect = ((fixed_predictions == 1) & (deept_predictions == 1)).astype(int)
        only_fixed = ((fixed_predictions == 1) & (deept_predictions == 0)).astype(int)
        only_deept = ((fixed_predictions == 0) & (deept_predictions == 1)).astype(int)

        axes[0, 1].scatter(time_points[agreement == 1], agreement[agreement == 1],
                          c='blue', marker='|', s=100, label=f'Agreement ({agreement.sum()} pts)', alpha=0.6)
        axes[0, 1].scatter(time_points[only_fixed == 1], only_fixed[only_fixed == 1],
                          c='orange', marker='x', s=50, label=f'Only Fixed ({only_fixed.sum()} pts)')
        axes[0, 1].scatter(time_points[only_deept == 1], only_deept[only_deept == 1],
                          c='purple', marker='+', s=50, label=f'Only DeepT ({only_deept.sum()} pts)')
        if fault_injection_point > 0:
            axes[0, 1].axvline(x=fault_injection_point, color='orange', linestyle=':', linewidth=2)
        axes[0, 1].set_ylabel('Agreement', fontsize=11)
        axes[0, 1].set_ylim([-0.1, 1.1])
        axes[0, 1].set_title('Agreement Analysis', fontsize=12)
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Threshold Comparison (if available)
        if threshold_history is not None and fixed_h is not None:
            h_values = threshold_history['h']
            axes[1, 0].plot(time_points[:len(h_values)], h_values, label='DeepT: Adaptive h(t)',
                           color='blue', linewidth=2)
            axes[1, 0].axhline(y=fixed_h, color='red', linestyle='--', linewidth=2,
                              label=f'Fixed: h={fixed_h:.2f}')
            if fault_injection_point > 0:
                axes[1, 0].axvline(x=fault_injection_point, color='orange', linestyle=':', linewidth=2)
            axes[1, 0].set_ylabel('Threshold', fontsize=11)
            axes[1, 0].set_xlabel('Time', fontsize=11)
            axes[1, 0].set_title('Threshold Comparison', fontsize=12)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Threshold history not available',
                           ha='center', va='center', fontsize=12)
            axes[1, 0].set_title('Threshold Comparison', fontsize=12)

        # Plot 4: Performance Metrics
        metrics_text = []
        metrics_text.append("="*40)
        metrics_text.append("Detection Statistics Comparison")
        metrics_text.append("="*40)
        metrics_text.append("")

        # First detection
        fixed_first = np.argmax(fixed_predictions == 1) if np.any(fixed_predictions == 1) else None
        deept_first = np.argmax(deept_predictions == 1) if np.any(deept_predictions == 1) else None
        metrics_text.append("First Detection:")
        metrics_text.append(f"  Fixed CUSUM: {fixed_first if fixed_first is not None else 'No detection'}")
        metrics_text.append(f"  DeepT-CUSUM: {deept_first if deept_first is not None else 'No detection'}")
        if fixed_first is not None and deept_first is not None:
            improvement = fixed_first - deept_first
            metrics_text.append(f"  Improvement: {improvement} samples ({'earlier' if improvement > 0 else 'later'})")
        metrics_text.append("")

        # Detection rate
        fixed_rate = np.mean(fixed_predictions) * 100
        deept_rate = np.mean(deept_predictions) * 100
        metrics_text.append("Detection Rate:")
        metrics_text.append(f"  Fixed CUSUM: {fixed_rate:.2f}%")
        metrics_text.append(f"  DeepT-CUSUM: {deept_rate:.2f}%")
        metrics_text.append("")

        # Agreement
        agreement_pct = agreement.mean() * 100
        metrics_text.append(f"Agreement: {agreement_pct:.2f}%")
        metrics_text.append("="*40)

        axes[1, 1].axis('off')
        axes[1, 1].text(0.1, 0.9, '\n'.join(metrics_text), fontsize=10, family='monospace',
                       verticalalignment='top')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_training_history(history,
                             title: str = "DeepT-CUSUM Training History",
                             figsize: tuple = (12, 5)):
        """
        Plot training history (loss and metrics).

        Args:
            history: Keras history object
            title: Plot title
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # Loss
        axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Loss (MSE)', fontsize=11)
        axes[0].set_title('Training and Validation Loss', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Metrics (MAE)
        if 'mae' in history.history:
            axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
            axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=11)
            axes[1].set_ylabel('MAE', fontsize=11)
            axes[1].set_title('Mean Absolute Error', fontsize=12)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_threshold_statistics(threshold_history: Dict[str, List],
                                  predictions: NDArray,
                                  title_suffix: str = "",
                                  figsize: tuple = (12, 8)):
        """
        Plot comprehensive threshold statistics.

        Args:
            threshold_history: Dict with 'h' and 'cusum_stat'
            predictions: Binary predictions
            title_suffix: Additional text for title
            figsize: Figure size
        """
        h_values = np.array(threshold_history['h'])
        cusum_stats = np.array(threshold_history['cusum_stat'])

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'DeepT-CUSUM: Threshold Statistics {title_suffix}',
                    fontsize=14, fontweight='bold')

        # Plot 1: Boxplot by prediction class
        normal_h = h_values[predictions == 0]
        anomaly_h = h_values[predictions == 1]

        axes[0, 0].boxplot([normal_h, anomaly_h], labels=['Normal', 'Anomaly'])
        axes[0, 0].set_ylabel('Threshold h(t)', fontsize=11)
        axes[0, 0].set_title('Threshold by Prediction Class', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        # Plot 2: Ratio S_t / h_t
        ratio = cusum_stats / (h_values + 1e-10)
        axes[0, 1].hist(ratio[predictions == 0], bins=30, alpha=0.6, color='green',
                       label='Normal', density=True)
        axes[0, 1].hist(ratio[predictions == 1], bins=30, alpha=0.6, color='red',
                       label='Anomaly', density=True)
        axes[0, 1].axvline(x=1.0, color='black', linestyle='--', linewidth=2, label='Decision boundary')
        axes[0, 1].set_xlabel('S(t) / h(t)', fontsize=11)
        axes[0, 1].set_ylabel('Density', fontsize=11)
        axes[0, 1].set_title('Detection Ratio Distribution', fontsize=12)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Running average of threshold
        window = 50
        h_running_avg = np.convolve(h_values, np.ones(window)/window, mode='valid')
        axes[1, 0].plot(h_values, alpha=0.3, color='blue', label='Instantaneous')
        axes[1, 0].plot(np.arange(window-1, len(h_values)), h_running_avg,
                       color='blue', linewidth=2, label=f'Running avg (window={window})')
        axes[1, 0].set_xlabel('Time', fontsize=11)
        axes[1, 0].set_ylabel('Threshold h(t)', fontsize=11)
        axes[1, 0].set_title('Threshold Smoothness', fontsize=12)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Statistics summary
        stats_text = []
        stats_text.append("Threshold Statistics:")
        stats_text.append(f"  Mean: {h_values.mean():.3f}")
        stats_text.append(f"  Std:  {h_values.std():.3f}")
        stats_text.append(f"  Min:  {h_values.min():.3f}")
        stats_text.append(f"  Max:  {h_values.max():.3f}")
        stats_text.append("")
        stats_text.append("By Class:")
        stats_text.append(f"  Normal mean:  {normal_h.mean():.3f}")
        stats_text.append(f"  Anomaly mean: {anomaly_h.mean():.3f}")
        stats_text.append("")
        stats_text.append("CUSUM Statistics:")
        stats_text.append(f"  Mean: {cusum_stats.mean():.3f}")
        stats_text.append(f"  Std:  {cusum_stats.std():.3f}")
        stats_text.append(f"  Max:  {cusum_stats.max():.3f}")

        axes[1, 1].axis('off')
        axes[1, 1].text(0.1, 0.9, '\n'.join(stats_text), fontsize=11, family='monospace',
                       verticalalignment='top')

        plt.tight_layout()
        plt.show()
