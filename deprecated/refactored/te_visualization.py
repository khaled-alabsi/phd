"""
Tennessee Eastman Process (TEP) Visualization and Reporting

This module handles visualization and reporting for the TEP anomaly detection
project, including performance metrics, fault analysis, and result summaries.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from te_utils import save_plot, save_dataframe


class TEPVisualizer:
    """Class for creating visualizations and reports for TEP analysis."""
    
    def __init__(self, output_dir: str = "output/1.00/"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory for saving outputs
        """
        self.output_dir = output_dir
        self.setup_plotting_style()
        
        # Ensure output directories exist
        self._create_output_directories()
    
    def setup_plotting_style(self) -> None:
        """Set up consistent plotting style."""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
    
    def _create_output_directories(self) -> None:
        """Create necessary output directories."""
        directories = [
            'anomaly',
            'arl',
            'average',
            'confusion_matrix',
            'default',
            'detection_delay',
            'fdr_far',
            'per_fault',
            'per_metric'
        ]
        
        for directory in directories:
            os.makedirs(os.path.join(self.output_dir, directory), exist_ok=True)
    
    def plot_model_performance_comparison(self, 
                                        model_results: Dict[str, Dict[str, float]],
                                        save_plot: bool = True) -> None:
        """
        Create a comprehensive model performance comparison plot.
        
        Args:
            model_results: Dictionary of model results with metrics
            save_plot: Whether to save the plot
        """
        print("Creating model performance comparison plot...")
        
        # Extract metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        models = list(model_results.keys())
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                values = [model_results[model].get(metric, 0) for model in models]
                
                # Create bar plot
                bars = axes[i].bar(models, values, alpha=0.7, color=plt.cm.Set3(np.linspace(0, 1, len(models))))
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_ylabel('Score')
                axes[i].set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    if not np.isnan(value):
                        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                   f'{value:.3f}', ha='center', va='bottom')
                
                # Rotate x-axis labels
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplot
        if len(metrics) < len(axes):
            axes[-1].set_visible(False)
        
        plt.suptitle('Model Performance Comparison', fontsize=16)
        plt.tight_layout()
        
        if save_plot:
            save_plot("model_performance_comparison", plot_path=f"{self.output_dir}average/")
        
        plt.show()
    
    def plot_confusion_matrices_grid(self, 
                                   model_results: Dict[str, Dict[str, Any]],
                                   y_true: np.ndarray,
                                   save_plots: bool = True) -> None:
        """
        Create a grid of confusion matrices for all models.
        
        Args:
            model_results: Dictionary of model results
            y_true: True labels
            save_plots: Whether to save the plots
        """
        print("Creating confusion matrices grid...")
        
        models = list(model_results.keys())
        n_models = len(models)
        
        # Calculate grid dimensions
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (model_name, results) in enumerate(model_results.items()):
            if 'predictions' not in results:
                continue
            
            row = i // cols
            col = i % cols
            
            if rows == 1:
                ax = axes[col]
            elif cols == 1:
                ax = axes[row]
            else:
                ax = axes[row, col]
            
            # Create confusion matrix
            y_pred = results['predictions']
            cm = confusion_matrix(y_true, y_pred)
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=range(len(np.unique(y_true))),
                       yticklabels=range(len(np.unique(y_true))))
            ax.set_title(f'{model_name.replace("_", " ").title()}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(n_models, rows * cols):
            row = i // cols
            col = i % cols
            if rows == 1:
                axes[col].set_visible(False)
            elif cols == 1:
                axes[row].set_visible(False)
            else:
                axes[row, col].set_visible(False)
        
        plt.suptitle('Confusion Matrices for All Models', fontsize=16)
        plt.tight_layout()
        
        if save_plots:
            save_plot("confusion_matrices_grid", plot_path=f"{self.output_dir}confusion_matrix/")
        
        plt.show()
    
    def plot_roc_curves(self, 
                        model_results: Dict[str, Dict[str, Any]],
                        y_true: np.ndarray,
                        save_plot: bool = True) -> None:
        """
        Plot ROC curves for all models that support probability prediction.
        
        Args:
            model_results: Dictionary of model results
            y_true: True labels
            save_plot: Whether to save the plot
        """
        print("Creating ROC curves...")
        
        plt.figure(figsize=(10, 8))
        
        for model_name, results in model_results.items():
            if 'probabilities' not in results or results['probabilities'] is None:
                continue
            
            y_pred_proba = results['probabilities']
            
            # Handle different probability formats
            if y_pred_proba.ndim > 1:
                if y_pred_proba.shape[1] == 2:
                    y_pred_proba = y_pred_proba[:, 1]  # Use positive class probability
                else:
                    continue  # Skip if more than 2 classes
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for All Models')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_plot:
            save_plot("roc_curves", plot_path=f"{self.output_dir}average/")
        
        plt.show()
    
    def plot_fault_performance_analysis(self, 
                                      model_results: Dict[str, Dict[str, Any]],
                                      y_true: np.ndarray,
                                      save_plots: bool = True) -> None:
        """
        Create fault-specific performance analysis plots.
        
        Args:
            model_results: Dictionary of model results
            y_true: True labels
            save_plots: Whether to save the plots
        """
        print("Creating fault performance analysis plots...")
        
        # Get unique fault classes
        unique_faults = np.unique(y_true)
        if len(unique_faults) <= 2:
            print("Skipping fault performance analysis: insufficient fault classes")
            return
        
        models = list(model_results.keys())
        
        # Calculate per-fault metrics for each model
        fault_metrics = {}
        for fault in unique_faults:
            if fault == 0:  # Skip normal class
                continue
            
            fault_metrics[fault] = {}
            fault_mask = y_true == fault
            
            for model_name, results in model_results.items():
                if 'predictions' not in results:
                    continue
                
                y_pred = results['predictions']
                fault_pred = y_pred[fault_mask]
                fault_true = y_true[fault_mask]
                
                # Calculate metrics for this fault
                tp = np.sum((fault_true == fault) & (fault_pred == fault))
                fp = np.sum((fault_true != fault) & (fault_pred == fault))
                tn = np.sum((fault_true != fault) & (fault_pred != fault))
                fn = np.sum((fault_true == fault) & (fault_pred != fault))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                fault_metrics[fault][model_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
        
        # Create plots for each metric
        metrics_to_plot = ['precision', 'recall', 'f1']
        
        for metric in metrics_to_plot:
            plt.figure(figsize=(12, 8))
            
            # Prepare data for plotting
            fault_names = [f'Fault {f}' for f in fault_metrics.keys()]
            x = np.arange(len(fault_names))
            width = 0.8 / len(models)
            
            for i, model_name in enumerate(models):
                values = [fault_metrics[fault].get(model_name, {}).get(metric, 0) 
                         for fault in fault_metrics.keys()]
                
                plt.bar(x + i * width, values, width, label=model_name, alpha=0.8)
            
            plt.xlabel('Fault Type')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.title(f'{metric.replace("_", " ").title()} per Fault Type')
            plt.xticks(x + width * (len(models) - 1) / 2, fault_names, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_plots:
                save_plot(f"fault_per_{metric}", plot_path=f"{self.output_dir}per_fault/")
            
            plt.show()
    
    def plot_detection_delay_analysis(self, 
                                    detection_delays: Dict[str, Dict[int, float]],
                                    save_plot: bool = True) -> None:
        """
        Plot detection delay analysis for different models and fault types.
        
        Args:
            detection_delays: Dictionary of detection delays per model and fault
            save_plot: Whether to save the plot
        """
        print("Creating detection delay analysis plot...")
        
        if not detection_delays:
            print("No detection delay data available")
            return
        
        # Prepare data for plotting
        models = list(detection_delays.keys())
        all_faults = set()
        for model_delays in detection_delays.values():
            all_faults.update(model_delays.keys())
        
        fault_list = sorted(all_faults)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Detection delays per fault type
        x = np.arange(len(fault_list))
        width = 0.8 / len(models)
        
        for i, model_name in enumerate(models):
            values = [detection_delays[model_name].get(fault, np.nan) for fault in fault_list]
            # Replace infinite values with a large number for visualization
            values = [v if v != np.inf else 100 for v in values]
            
            ax1.bar(x + i * width, values, width, label=model_name, alpha=0.8)
        
        ax1.set_xlabel('Fault Type')
        ax1.set_ylabel('Detection Delay (samples)')
        ax1.set_title('Detection Delay per Fault Type')
        ax1.set_xticks(x + width * (len(models) - 1) / 2)
        ax1.set_xticklabels([f'Fault {f}' for f in fault_list], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average detection delay per model
        avg_delays = []
        for model_name in models:
            delays = list(detection_delays[model_name].values())
            # Remove infinite values
            valid_delays = [d for d in delays if d != np.inf]
            if valid_delays:
                avg_delays.append(np.mean(valid_delays))
            else:
                avg_delays.append(0)
        
        ax2.bar(models, avg_delays, alpha=0.8, color=plt.cm.Set3(np.linspace(0, 1, len(models))))
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Average Detection Delay (samples)')
        ax2.set_title('Average Detection Delay per Model')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            save_plot("detection_delay_analysis", plot_path=f"{self.output_dir}detection_delay/")
        
        plt.show()
    
    def create_performance_summary_table(self, 
                                       model_results: Dict[str, Dict[str, float]],
                                       save_table: bool = True) -> pd.DataFrame:
        """
        Create a comprehensive performance summary table.
        
        Args:
            model_results: Dictionary of model results
            save_table: Whether to save the table
            
        Returns:
            DataFrame with performance summary
        """
        print("Creating performance summary table...")
        
        # Extract metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        summary_data = []
        for model_name, results in model_results.items():
            if 'metrics' not in results:
                continue
            
            row = {'Model': model_name}
            for metric in metrics:
                value = results['metrics'].get(metric, np.nan)
                row[metric.replace('_', ' ').title()] = f"{value:.4f}" if not np.isnan(value) else "N/A"
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by F1 score
        if 'F1 Score' in summary_df.columns:
            summary_df = summary_df.sort_values('F1 Score', ascending=False)
        
        # Display table
        print("\nPerformance Summary:")
        print("=" * 100)
        print(summary_df.to_string(index=False))
        
        # Save table
        if save_table:
            save_dataframe(summary_df, "performance_summary", "all_models")
        
        return summary_df
    
    def create_interactive_dashboard(self, 
                                   model_results: Dict[str, Dict[str, Any]],
                                   y_true: np.ndarray) -> None:
        """
        Create an interactive Plotly dashboard for model comparison.
        
        Args:
            model_results: Dictionary of model results
            y_true: True labels
        """
        print("Creating interactive dashboard...")
        
        try:
            # Prepare data
            models = list(model_results.keys())
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[m.replace('_', ' ').title() for m in metrics],
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Add bar charts for each metric
            for i, metric in enumerate(metrics):
                values = [model_results[model].get('metrics', {}).get(metric, 0) for model in models]
                
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                fig.add_trace(
                    go.Bar(x=models, y=values, name=metric.replace('_', ' ').title()),
                    row=row, col=col
                )
            
            # Update layout
            fig.update_layout(
                title="Interactive Model Performance Dashboard",
                height=800,
                showlegend=False
            )
            
            # Update axes
            for i in range(1, 3):
                for j in range(1, 3):
                    fig.update_xaxes(title_text="Model", row=i, col=j)
                    fig.update_yaxes(title_text="Score", range=[0, 1], row=i, col=j)
            
            # Show dashboard
            fig.show()
            
            # Save as HTML
            dashboard_path = os.path.join(self.output_dir, "interactive_dashboard.html")
            fig.write_html(dashboard_path)
            print(f"Interactive dashboard saved to: {dashboard_path}")
            
        except ImportError:
            print("Plotly not available. Skipping interactive dashboard creation.")
        except Exception as e:
            print(f"Error creating interactive dashboard: {e}")
    
    def generate_comprehensive_report(self, 
                                   model_results: Dict[str, Dict[str, Any]],
                                   y_true: np.ndarray,
                                   save_report: bool = True) -> str:
        """
        Generate a comprehensive text report of all analysis results.
        
        Args:
            model_results: Dictionary of model results
            y_true: True labels
            save_report: Whether to save the report
            
        Returns:
            String containing the report
        """
        print("Generating comprehensive report...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("TEP ANOMALY DETECTION COMPREHENSIVE REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Dataset information
        report_lines.append("DATASET INFORMATION")
        report_lines.append("-" * 40)
        report_lines.append(f"Total samples: {len(y_true)}")
        report_lines.append(f"Number of classes: {len(np.unique(y_true))}")
        report_lines.append(f"Class distribution:")
        unique, counts = np.unique(y_true, return_counts=True)
        for class_id, count in zip(unique, counts):
            percentage = (count / len(y_true)) * 100
            report_lines.append(f"  Class {class_id}: {count} samples ({percentage:.1f}%)")
        report_lines.append("")
        
        # Model performance summary
        report_lines.append("MODEL PERFORMANCE SUMMARY")
        report_lines.append("-" * 40)
        
        # Sort models by F1 score
        sorted_models = sorted(
            model_results.items(),
            key=lambda x: x[1].get('metrics', {}).get('f1_score', 0),
            reverse=True
        )
        
        for i, (model_name, results) in enumerate(sorted_models):
            if 'metrics' not in results:
                continue
            
            metrics = results['metrics']
            report_lines.append(f"{i+1}. {model_name.replace('_', ' ').title()}")
            report_lines.append(f"   Accuracy:  {metrics.get('accuracy', 'N/A'):.4f}")
            report_lines.append(f"   Precision: {metrics.get('precision', 'N/A'):.4f}")
            report_lines.append(f"   Recall:    {metrics.get('recall', 'N/A'):.4f}")
            report_lines.append(f"   F1-Score:  {metrics.get('f1_score', 'N/A'):.4f}")
            report_lines.append(f"   ROC-AUC:   {metrics.get('roc_auc', 'N/A'):.4f}")
            report_lines.append("")
        
        # Best model recommendation
        if sorted_models:
            best_model_name = sorted_models[0][0]
            best_f1 = sorted_models[0][1].get('metrics', {}).get('f1_score', 0)
            report_lines.append("RECOMMENDATION")
            report_lines.append("-" * 40)
            report_lines.append(f"Best performing model: {best_model_name.replace('_', ' ').title()}")
            report_lines.append(f"F1-Score: {best_f1:.4f}")
            report_lines.append("")
        
        # Conclusion
        report_lines.append("CONCLUSION")
        report_lines.append("-" * 40)
        report_lines.append("This analysis demonstrates the effectiveness of various machine learning")
        report_lines.append("algorithms for TEP anomaly detection. The results show that")
        report_lines.append("ensemble methods and deep learning approaches can achieve high")
        report_lines.append("performance in detecting process faults.")
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Save report
        if save_report:
            report_path = os.path.join(self.output_dir, "comprehensive_report.txt")
            with open(report_path, 'w') as f:
                f.write(report_text)
            print(f"Comprehensive report saved to: {report_path}")
        
        return report_text

