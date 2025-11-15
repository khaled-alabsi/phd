"""
Visualization utilities for DNN-CUSUM detector.

Provides functions to visualize:
- Parameter evolution over time
- CUSUM statistics and predictions
- Comparison with fixed-parameter CUSUM
- Training history
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from numpy.typing import NDArray


class DNNCUSUMVisualizer:
    """Visualization utilities for DNN-CUSUM analysis."""

    @staticmethod
    def plot_parameter_evolution(param_history: Dict[str, List],
                                 X_test: NDArray[np.float64],
                                 predictions: NDArray[np.int_],
                                 fault_injection_point: Optional[int] = None,
                                 title_suffix: str = "",
                                 figsize: Tuple[int, int] = (15, 10)):
        """
        Plot evolution of k(t) and h(t) parameters over time with CUSUM statistics.

        Args:
            param_history: Dictionary with 'k', 'h', 'cusum_stat' lists
            X_test: Test data
            predictions: Binary predictions
            fault_injection_point: Index where fault starts (for visualization)
            title_suffix: Additional title text
            figsize: Figure size
        """
        k_values = np.array(param_history['k'])
        h_values = np.array(param_history['h'])
        cusum_stats = np.array(param_history['cusum_stat'])
        time_points = np.arange(len(k_values))

        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

        # Plot 1: k parameter evolution
        axes[0].plot(time_points, k_values, linewidth=2, color='blue', alpha=0.7)
        axes[0].fill_between(time_points, 0, k_values, alpha=0.3, color='blue')
        axes[0].set_ylabel('k (Reference Value)', fontsize=12, fontweight='bold')
        axes[0].set_title(f'DNN-CUSUM Adaptive Parameter Evolution {title_suffix}',
                         fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Highlight fault region if specified
        if fault_injection_point is not None:
            axes[0].axvline(fault_injection_point, color='red', linestyle='--',
                          linewidth=2, label='Fault Injection', alpha=0.7)
            axes[0].legend()

        # Plot 2: h parameter evolution
        axes[1].plot(time_points, h_values, linewidth=2, color='green', alpha=0.7)
        axes[1].fill_between(time_points, 0, h_values, alpha=0.3, color='green')
        axes[1].set_ylabel('h (Threshold)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        if fault_injection_point is not None:
            axes[1].axvline(fault_injection_point, color='red', linestyle='--',
                          linewidth=2, alpha=0.7)

        # Plot 3: CUSUM statistic with threshold
        axes[2].plot(time_points, cusum_stats, linewidth=2, color='purple', alpha=0.7,
                    label='CUSUM Statistic')
        axes[2].plot(time_points, h_values, linewidth=2, color='orange', linestyle='--',
                    alpha=0.7, label='Adaptive Threshold h(t)')
        axes[2].fill_between(time_points, 0, cusum_stats, alpha=0.2, color='purple')
        axes[2].set_ylabel('CUSUM Statistic', fontsize=12, fontweight='bold')
        axes[2].set_yscale('log')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

        if fault_injection_point is not None:
            axes[2].axvline(fault_injection_point, color='red', linestyle='--',
                          linewidth=2, alpha=0.7)

        # Plot 4: Predictions
        axes[3].fill_between(time_points, 0, predictions, alpha=0.5, color='red',
                            label='Detected Anomalies', step='mid')
        axes[3].set_ylabel('Anomaly Flag', fontsize=12, fontweight='bold')
        axes[3].set_xlabel('Time Point', fontsize=12, fontweight='bold')
        axes[3].set_ylim([-0.1, 1.1])
        axes[3].set_yticks([0, 1])
        axes[3].set_yticklabels(['Normal', 'Anomaly'])
        axes[3].legend(loc='upper right')
        axes[3].grid(True, alpha=0.3)

        if fault_injection_point is not None:
            axes[3].axvline(fault_injection_point, color='red', linestyle='--',
                          linewidth=2, alpha=0.7)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_parameter_statistics(param_history: Dict[str, List],
                                  predictions: NDArray[np.int_],
                                  figsize: Tuple[int, int] = (12, 8)):
        """
        Plot statistical distributions of k and h parameters.

        Args:
            param_history: Dictionary with 'k', 'h' lists
            predictions: Binary predictions
            figsize: Figure size
        """
        k_values = np.array(param_history['k'])
        h_values = np.array(param_history['h'])

        # Separate parameters by normal vs anomaly predictions
        k_normal = k_values[predictions == 0]
        k_anomaly = k_values[predictions == 1]
        h_normal = h_values[predictions == 0]
        h_anomaly = h_values[predictions == 1]

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # k distribution
        axes[0, 0].hist(k_normal, bins=30, alpha=0.7, color='blue',
                       label=f'Normal (n={len(k_normal)})', density=True)
        if len(k_anomaly) > 0:
            axes[0, 0].hist(k_anomaly, bins=30, alpha=0.7, color='red',
                          label=f'Anomaly (n={len(k_anomaly)})', density=True)
        axes[0, 0].set_xlabel('k (Reference Value)', fontweight='bold')
        axes[0, 0].set_ylabel('Density', fontweight='bold')
        axes[0, 0].set_title('Distribution of k Parameter', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # h distribution
        axes[0, 1].hist(h_normal, bins=30, alpha=0.7, color='blue',
                       label=f'Normal (n={len(h_normal)})', density=True)
        if len(h_anomaly) > 0:
            axes[0, 1].hist(h_anomaly, bins=30, alpha=0.7, color='red',
                          label=f'Anomaly (n={len(h_anomaly)})', density=True)
        axes[0, 1].set_xlabel('h (Threshold)', fontweight='bold')
        axes[0, 1].set_ylabel('Density', fontweight='bold')
        axes[0, 1].set_title('Distribution of h Parameter', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # k vs h scatter
        colors = ['blue' if p == 0 else 'red' for p in predictions]
        axes[1, 0].scatter(k_values, h_values, c=colors, alpha=0.5, s=10)
        axes[1, 0].set_xlabel('k (Reference Value)', fontweight='bold')
        axes[1, 0].set_ylabel('h (Threshold)', fontweight='bold')
        axes[1, 0].set_title('k vs h Parameter Space', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # Box plots
        data_k = [k_normal, k_anomaly] if len(k_anomaly) > 0 else [k_normal]
        data_h = [h_normal, h_anomaly] if len(h_anomaly) > 0 else [h_normal]
        labels = ['Normal', 'Anomaly'] if len(k_anomaly) > 0 else ['Normal']

        bp1 = axes[1, 1].boxplot(data_k, labels=labels, positions=[1, 2][:len(data_k)],
                                 patch_artist=True, widths=0.3)
        bp2 = axes[1, 1].boxplot(data_h, labels=labels, positions=[3, 4][:len(data_h)],
                                 patch_artist=True, widths=0.3)

        # Color box plots
        for patch in bp1['boxes']:
            patch.set_facecolor('lightblue')
        for patch in bp2['boxes']:
            patch.set_facecolor('lightgreen')

        axes[1, 1].set_xticks([1.5, 3.5])
        axes[1, 1].set_xticklabels(['k', 'h'], fontweight='bold')
        axes[1, 1].set_ylabel('Parameter Value', fontweight='bold')
        axes[1, 1].set_title('Parameter Statistics', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_comparison(fixed_predictions: NDArray[np.int_],
                       dnn_predictions: NDArray[np.int_],
                       X_test: NDArray[np.float64],
                       param_history: Optional[Dict[str, List]] = None,
                       fault_injection_point: Optional[int] = None,
                       title: str = "",
                       figsize: Tuple[int, int] = (15, 8)):
        """
        Compare fixed-parameter CUSUM with DNN-CUSUM.

        Args:
            fixed_predictions: Predictions from fixed CUSUM
            dnn_predictions: Predictions from DNN-CUSUM
            X_test: Test data
            param_history: Parameter history from DNN-CUSUM
            fault_injection_point: Index where fault starts
            title: Plot title
            figsize: Figure size
        """
        time_points = np.arange(len(fixed_predictions))

        n_plots = 3 if param_history is not None else 2
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)

        # Plot 1: Fixed CUSUM predictions
        axes[0].fill_between(time_points, 0, fixed_predictions, alpha=0.5,
                            color='orange', step='mid', label='Fixed CUSUM')
        axes[0].set_ylabel('Fixed CUSUM', fontweight='bold')
        axes[0].set_title(f'Comparison: Fixed CUSUM vs DNN-CUSUM {title}',
                         fontsize=14, fontweight='bold')
        axes[0].set_ylim([-0.1, 1.1])
        axes[0].set_yticks([0, 1])
        axes[0].set_yticklabels(['Normal', 'Anomaly'])
        axes[0].grid(True, alpha=0.3)

        if fault_injection_point is not None:
            axes[0].axvline(fault_injection_point, color='red', linestyle='--',
                          linewidth=2, label='Fault Injection', alpha=0.7)
        axes[0].legend()

        # Plot 2: DNN-CUSUM predictions
        axes[1].fill_between(time_points, 0, dnn_predictions, alpha=0.5,
                            color='purple', step='mid', label='DNN-CUSUM')
        axes[1].set_ylabel('DNN-CUSUM', fontweight='bold')
        axes[1].set_ylim([-0.1, 1.1])
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(['Normal', 'Anomaly'])
        axes[1].grid(True, alpha=0.3)

        if fault_injection_point is not None:
            axes[1].axvline(fault_injection_point, color='red', linestyle='--',
                          linewidth=2, alpha=0.7)
        axes[1].legend()

        # Plot 3: Adaptive parameters (if available)
        if param_history is not None:
            k_values = np.array(param_history['k'])
            h_values = np.array(param_history['h'])

            ax3_k = axes[2]
            ax3_h = ax3_k.twinx()

            line1 = ax3_k.plot(time_points, k_values, color='blue', linewidth=2,
                              alpha=0.7, label='k(t)')
            line2 = ax3_h.plot(time_points, h_values, color='green', linewidth=2,
                              alpha=0.7, label='h(t)')

            ax3_k.set_ylabel('k (Reference Value)', color='blue', fontweight='bold')
            ax3_h.set_ylabel('h (Threshold)', color='green', fontweight='bold')
            ax3_k.set_xlabel('Time Point', fontweight='bold')

            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax3_k.legend(lines, labels, loc='upper right')

            ax3_k.grid(True, alpha=0.3)

            if fault_injection_point is not None:
                ax3_k.axvline(fault_injection_point, color='red', linestyle='--',
                            linewidth=2, alpha=0.7)
        else:
            axes[1].set_xlabel('Time Point', fontweight='bold')

        plt.tight_layout()
        plt.show()

        # Print detection statistics
        print("\n" + "="*60)
        print("Detection Statistics Comparison")
        print("="*60)

        # Detection delays
        fixed_first_detect = np.argmax(fixed_predictions == 1) if np.any(fixed_predictions == 1) else None
        dnn_first_detect = np.argmax(dnn_predictions == 1) if np.any(dnn_predictions == 1) else None

        print(f"\nFirst Detection:")
        print(f"  Fixed CUSUM: {fixed_first_detect if fixed_first_detect is not None else 'No detection'}")
        print(f"  DNN-CUSUM:   {dnn_first_detect if dnn_first_detect is not None else 'No detection'}")

        if fixed_first_detect is not None and dnn_first_detect is not None:
            improvement = fixed_first_detect - dnn_first_detect
            print(f"  Improvement: {improvement} samples ({'earlier' if improvement > 0 else 'later'})")

        # Detection rates
        fixed_rate = np.mean(fixed_predictions)
        dnn_rate = np.mean(dnn_predictions)

        print(f"\nDetection Rate:")
        print(f"  Fixed CUSUM: {fixed_rate*100:.2f}%")
        print(f"  DNN-CUSUM:   {dnn_rate*100:.2f}%")

        # Agreement
        agreement = np.mean(fixed_predictions == dnn_predictions)
        print(f"\nAgreement: {agreement*100:.2f}%")
        print("="*60 + "\n")

    @staticmethod
    def plot_training_history(history,
                             figsize: Tuple[int, int] = (12, 5)):
        """
        Plot training history for DNN model.

        Args:
            history: Keras history object from model.fit()
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot loss
        axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontweight='bold')
        axes[0].set_ylabel('Loss', fontweight='bold')
        axes[0].set_title('Model Loss', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot metrics
        if 'k_output_mae' in history.history:
            axes[1].plot(history.history['k_output_mae'], label='k MAE (train)', linewidth=2)
            axes[1].plot(history.history['val_k_output_mae'], label='k MAE (val)', linewidth=2)
            axes[1].plot(history.history['h_output_mae'], label='h MAE (train)',
                        linewidth=2, linestyle='--')
            axes[1].plot(history.history['val_h_output_mae'], label='h MAE (val)',
                        linewidth=2, linestyle='--')
            axes[1].set_xlabel('Epoch', fontweight='bold')
            axes[1].set_ylabel('MAE', fontweight='bold')
            axes[1].set_title('Parameter Prediction Error', fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_parameter_sensitivity(param_history: Dict[str, List],
                                   X_test: NDArray[np.float64],
                                   feature_idx: int = 0,
                                   figsize: Tuple[int, int] = (12, 6)):
        """
        Plot how parameters respond to changes in a specific feature.

        Args:
            param_history: Parameter history
            X_test: Test data
            feature_idx: Index of feature to analyze
            figsize: Figure size
        """
        k_values = np.array(param_history['k'])
        h_values = np.array(param_history['h'])
        feature_values = X_test[:, feature_idx]

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Feature and k
        ax1 = axes[0]
        ax1_twin = ax1.twinx()

        line1 = ax1.plot(feature_values, color='black', linewidth=2, alpha=0.7,
                        label=f'Feature {feature_idx}')
        line2 = ax1_twin.plot(k_values, color='blue', linewidth=2, alpha=0.7,
                             label='k(t)')

        ax1.set_ylabel(f'Feature {feature_idx} Value', fontweight='bold')
        ax1_twin.set_ylabel('k (Reference Value)', color='blue', fontweight='bold')
        ax1.set_title(f'Parameter Sensitivity to Feature {feature_idx}',
                     fontsize=14, fontweight='bold')

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Feature and h
        ax2 = axes[1]
        ax2_twin = ax2.twinx()

        line3 = ax2.plot(feature_values, color='black', linewidth=2, alpha=0.7,
                        label=f'Feature {feature_idx}')
        line4 = ax2_twin.plot(h_values, color='green', linewidth=2, alpha=0.7,
                             label='h(t)')

        ax2.set_ylabel(f'Feature {feature_idx} Value', fontweight='bold')
        ax2_twin.set_ylabel('h (Threshold)', color='green', fontweight='bold')
        ax2.set_xlabel('Time Point', fontweight='bold')

        lines = line3 + line4
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
