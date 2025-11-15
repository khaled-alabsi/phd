import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class ModelComparisonAnalyzer:
    """Class to analyze and visualize model comparison results."""
    
    def __init__(self, df_results: pd.DataFrame, summary_df: pd.DataFrame):
        """
        Initialize the analyzer with results data.
        
        Args:
            df_results: DataFrame with raw results from all simulation runs
            summary_df: DataFrame with aggregated summary statistics
        """
        self.df_results = df_results
        self.summary_df = summary_df
        self.models = df_results['model'].unique()
        self.fault_numbers = sorted(df_results['faultNumber'].unique())
    
    def create_comparison_table(self, metric: str = 'all') -> pd.DataFrame:
        """
        Create a detailed comparison table for specified metrics.
        
        Args:
            metric: 'ARL0', 'ARL1', 'detection', or 'all'
        
        Returns:
            Formatted comparison DataFrame
        """
        if metric == 'all':
            # Create comprehensive comparison table
            pivot_tables = []
            
            # ARL1 comparison
            arl1_pivot = self.summary_df.pivot(
                index='faultNumber', 
                columns='model', 
                values='conditional_ARL1'
            ).round(2)
            arl1_pivot.columns = [f'{col}_ARL1' for col in arl1_pivot.columns]
            pivot_tables.append(arl1_pivot)
            
            # ARL0 comparison
            arl0_pivot = self.summary_df.pivot(
                index='faultNumber', 
                columns='model', 
                values='conditional_ARL0'
            ).round(2)
            arl0_pivot.columns = [f'{col}_ARL0' for col in arl0_pivot.columns]
            pivot_tables.append(arl0_pivot)
            
            # Detection fraction comparison
            det_pivot = self.summary_df.pivot(
                index='faultNumber', 
                columns='model', 
                values='avg_detection_fraction'
            ).round(3)
            det_pivot.columns = [f'{col}_DetFrac' for col in det_pivot.columns]
            pivot_tables.append(det_pivot)
            
            # Precision
            prec_pivot = self.summary_df.pivot(
                index='faultNumber',
                columns='model',
                values='precision'
            ).round(3)
            prec_pivot.columns = [f'{col}_Precision' for col in prec_pivot.columns]
            pivot_tables.append(prec_pivot)

            # Recall
            rec_pivot = self.summary_df.pivot(
                index='faultNumber',
                columns='model',
                values='recall'
            ).round(3)
            rec_pivot.columns = [f'{col}_Recall' for col in rec_pivot.columns]
            pivot_tables.append(rec_pivot)

            # Specificity
            spec_pivot = self.summary_df.pivot(
                index='faultNumber',
                columns='model',
                values='specificity'
            ).round(3)
            spec_pivot.columns = [f'{col}_Specificity' for col in spec_pivot.columns]
            pivot_tables.append(spec_pivot)

            # Accuracy
            acc_pivot = self.summary_df.pivot(
                index='faultNumber',
                columns='model',
                values='accuracy'
            ).round(3)
            acc_pivot.columns = [f'{col}_Accuracy' for col in acc_pivot.columns]
            pivot_tables.append(acc_pivot)

            # F1
            f1_pivot = self.summary_df.pivot(
                index='faultNumber',
                columns='model',
                values='f1'
            ).round(3)
            f1_pivot.columns = [f'{col}_F1' for col in f1_pivot.columns]
            pivot_tables.append(f1_pivot)

            # FPR / FNR
            fpr_pivot = self.summary_df.pivot(index='faultNumber', columns='model', values='false_positive_rate').round(3)
            fpr_pivot.columns = [f'{col}_FPR' for col in fpr_pivot.columns]
            pivot_tables.append(fpr_pivot)

            fnr_pivot = self.summary_df.pivot(index='faultNumber', columns='model', values='false_negative_rate').round(3)
            fnr_pivot.columns = [f'{col}_FNR' for col in fnr_pivot.columns]
            pivot_tables.append(fnr_pivot)

            # Combine all metrics
            comparison_table = pd.concat(pivot_tables, axis=1)
            comparison_table.index.name = 'Fault'
            
            return comparison_table
        
        elif metric == 'ARL1':
            return self._create_arl1_table()
        elif metric == 'ARL0':
            return self._create_arl0_table()
        elif metric == 'detection':
            return self._create_detection_table()
        else:
            raise ValueError("metric must be 'ARL0', 'ARL1', 'detection', or 'all'")
    
    def _create_arl1_table(self) -> pd.DataFrame:
        """Create ARL1 comparison table with mean and std."""
        arl1_mean = self.summary_df.pivot(
            index='faultNumber', 
            columns='model', 
            values='conditional_ARL1'
        )
        arl1_std = self.summary_df.pivot(
            index='faultNumber', 
            columns='model', 
            values='SDRL1'
        )
        
        # Combine mean ± std
        combined = pd.DataFrame(index=arl1_mean.index)
        for model in arl1_mean.columns:
            combined[model] = arl1_mean[model].apply(lambda x: f'{x:.1f}' if pd.notna(x) else 'N/A')
            combined[f'{model}_std'] = '±' + arl1_std[model].apply(lambda x: f'{x:.1f}' if pd.notna(x) else 'N/A')
        
        return combined
    
    def _create_arl0_table(self) -> pd.DataFrame:
        """Create ARL0 comparison table with mean and std."""
        arl0_mean = self.summary_df.pivot(
            index='faultNumber', 
            columns='model', 
            values='conditional_ARL0'
        )
        arl0_std = self.summary_df.pivot(
            index='faultNumber', 
            columns='model', 
            values='SDRL0'
        )
        
        # Combine mean ± std
        combined = pd.DataFrame(index=arl0_mean.index)
        for model in arl0_mean.columns:
            combined[model] = arl0_mean[model].apply(lambda x: f'{x:.1f}' if pd.notna(x) else 'N/A')
            combined[f'{model}_std'] = '±' + arl0_std[model].apply(lambda x: f'{x:.1f}' if pd.notna(x) else 'N/A')
        
        return combined
    
    def _create_detection_table(self) -> pd.DataFrame:
        """Create detection performance comparison table."""
        det_frac = self.summary_df.pivot(index='faultNumber', columns='model', values='avg_detection_fraction')
        non_det = self.summary_df.pivot(index='faultNumber', columns='model', values='non_detection_fraction')
        prec = self.summary_df.pivot(index='faultNumber', columns='model', values='precision')
        rec = self.summary_df.pivot(index='faultNumber', columns='model', values='recall')
        spec = self.summary_df.pivot(index='faultNumber', columns='model', values='specificity')
        acc = self.summary_df.pivot(index='faultNumber', columns='model', values='accuracy')
        f1 = self.summary_df.pivot(index='faultNumber', columns='model', values='f1')
        fpr = self.summary_df.pivot(index='faultNumber', columns='model', values='false_positive_rate')
        fnr = self.summary_df.pivot(index='faultNumber', columns='model', values='false_negative_rate')

        combined = pd.DataFrame(index=det_frac.index)
        for model in det_frac.columns:
            combined[f'{model}_DetRate'] = (det_frac[model] * 100).apply(lambda x: f'{x:.1f}%')
            combined[f'{model}_MissRate'] = (non_det[model] * 100).apply(lambda x: f'{x:.1f}%')
            combined[f'{model}_Precision'] = prec[model].apply(lambda x: f'{x:.3f}' if pd.notna(x) else 'N/A')
            combined[f'{model}_Recall'] = rec[model].apply(lambda x: f'{x:.3f}' if pd.notna(x) else 'N/A')
            combined[f'{model}_Specificity'] = spec[model].apply(lambda x: f'{x:.3f}' if pd.notna(x) else 'N/A')
            combined[f'{model}_Accuracy'] = acc[model].apply(lambda x: f'{x:.3f}' if pd.notna(x) else 'N/A')
            combined[f'{model}_F1'] = f1[model].apply(lambda x: f'{x:.3f}' if pd.notna(x) else 'N/A')
            combined[f'{model}_FPR'] = fpr[model].apply(lambda x: f'{x:.3f}' if pd.notna(x) else 'N/A')
            combined[f'{model}_FNR'] = fnr[model].apply(lambda x: f'{x:.3f}' if pd.notna(x) else 'N/A')

        return combined
    
    def create_summary_statistics(self) -> pd.DataFrame:
        """
        Create overall summary statistics for each model.
        
        Returns:
            DataFrame with overall performance metrics
        """
        stats = []
        
        for model in self.models:
            model_data = self.df_results[self.df_results['model'] == model]
            model_summary = self.summary_df[self.summary_df['model'] == model]
            
            stats.append({
                'Model': model,
                'Mean ARL0': model_data['ARL0'].dropna().mean(),
                'Std ARL0': model_data['ARL0'].dropna().std(),
                'Mean ARL1': model_data['ARL1'].dropna().mean(),
                'Std ARL1': model_data['ARL1'].dropna().std(),
                'Overall Detection Rate (%)': model_data['detection_fraction'].mean() * 100,
                'False Alarm Rate (%)': (model_data['ARL0'].notna().sum() / len(model_data)) * 100,
                'Miss Rate (%)': (model_data['ARL1'].isna().sum() / len(model_data)) * 100,
                'Precision': model_data['precision'].dropna().mean(),
                'Recall': model_data['recall'].dropna().mean(),
                'Specificity': model_data['specificity'].dropna().mean(),
                'Accuracy': model_data['accuracy'].dropna().mean(),
                'F1': model_data['f1'].dropna().mean(),
                'FPR': model_data['false_positive_rate'].dropna().mean(),
                'FNR': model_data['false_negative_rate'].dropna().mean(),
            })
        
        summary_stats_df = pd.DataFrame(stats)
        
        # Round numerical columns
        numeric_cols = ['Mean ARL0', 'Std ARL0', 'Mean ARL1', 'Std ARL1',
                        'Overall Detection Rate (%)', 'False Alarm Rate (%)', 'Miss Rate (%)',
                        'Precision', 'Recall', 'Specificity', 'Accuracy', 'F1', 'FPR', 'FNR']
        summary_stats_df[numeric_cols] = summary_stats_df[numeric_cols].round(3)
        
        return summary_stats_df
    
    def plot_arl_comparison(self, figsize: Tuple[int, int] = (15, 6)) -> plt.Figure:
        """
        Create bar plots comparing ARL0 and ARL1 across models and faults with shared legend.

        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # ARL1 comparison
        arl1_pivot = self.summary_df.pivot(
            index='faultNumber',
            columns='model',
            values='conditional_ARL1'
        )
        arl1_pivot.plot(kind='bar', ax=axes[0], width=0.8, legend=False)
        axes[0].set_title('ARL1 (Detection Delay)\nSmaller is better (faster detection)',
                         fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Fault Number', fontsize=12)
        axes[0].set_ylabel('ARL1 (samples)', fontsize=12)
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_xticks(range(len(arl1_pivot.index)))
        axes[0].set_xticklabels([str(int(x)) for x in arl1_pivot.index], rotation=45, ha='right')

        # ARL0 comparison
        arl0_pivot = self.summary_df.pivot(
            index='faultNumber',
            columns='model',
            values='conditional_ARL0'
        )
        arl0_pivot.plot(kind='bar', ax=axes[1], width=0.8, legend=False)
        axes[1].set_title('ARL0 (In-Control Run Length)\nBigger is better (fewer false alarms)',
                         fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Fault Number', fontsize=12)
        axes[1].set_ylabel('ARL0 (samples until false alarm)', fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_xticks(range(len(arl0_pivot.index)))
        axes[1].set_xticklabels([str(int(x)) for x in arl0_pivot.index], rotation=45, ha='right')

        # Single shared legend at the bottom
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title='Model', loc='lower center',
                  bbox_to_anchor=(0.5, -0.15), ncol=min(len(labels), 3),
                  fontsize=9, framealpha=0.9)

        plt.tight_layout()
        return fig
    
    def plot_detection_heatmap(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create heatmap of detection rates across models and faults.
        
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Detection fraction heatmap
        det_pivot = self.summary_df.pivot(
            index='faultNumber', 
            columns='model', 
            values='avg_detection_fraction'
        ) * 100  # Convert to percentage
        
        sns.heatmap(det_pivot.T, annot=True, fmt='.1f', cmap='RdYlGn', 
                   vmin=0, vmax=100, cbar_kws={'label': 'Detection Rate (%)'},
                   ax=axes[0])
        axes[0].set_title('Detection Rate Heatmap (Higher is Better)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Fault Number', fontsize=12)
        axes[0].set_ylabel('Model', fontsize=12)
        axes[0].set_xticklabels([str(int(x)) for x in det_pivot.index], rotation=45, ha='right')
        axes[0].set_yticklabels(det_pivot.columns, rotation=0)
        
        # ARL1 heatmap (lower is better)
        arl1_pivot = self.summary_df.pivot(
            index='faultNumber', 
            columns='model', 
            values='conditional_ARL1'
        )
        
        sns.heatmap(arl1_pivot.T, annot=True, fmt='.0f', cmap='RdYlGn_r',
                   cbar_kws={'label': 'ARL1 (samples)'}, ax=axes[1])
        axes[1].set_title('ARL1 Heatmap (Lower is Better)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Fault Number', fontsize=12)
        axes[1].set_ylabel('Model', fontsize=12)
        axes[1].set_xticklabels([str(int(x)) for x in arl1_pivot.index], rotation=45, ha='right')
        axes[1].set_yticklabels(arl1_pivot.columns, rotation=0)
        
        plt.tight_layout()
        return fig

    def plot_precision_recall(self, figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        """Line plots of precision and recall across faults for each model with shared legend."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Recall
        rec = self.summary_df.pivot(index='faultNumber', columns='model', values='recall').sort_index()
        for col in rec.columns:
            axes[0].plot(rec.index, rec[col], marker='o', label=col)
        axes[0].set_title('Recall (TPR) vs Fault', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Fault number', fontsize=10)
        axes[0].set_ylabel('Recall', fontsize=10)
        axes[0].set_ylim(0, 1.05)
        axes[0].grid(alpha=0.3)
        axes[0].set_xticks(rec.index)
        axes[0].set_xticklabels([str(int(x)) for x in rec.index], rotation=45, ha='right')

        # Precision
        prec = self.summary_df.pivot(index='faultNumber', columns='model', values='precision').sort_index()
        for col in prec.columns:
            axes[1].plot(prec.index, prec[col], marker='o', label=col)
        axes[1].set_title('Precision vs Fault', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Fault number', fontsize=10)
        axes[1].set_ylabel('Precision', fontsize=10)
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(alpha=0.3)
        axes[1].set_xticks(prec.index)
        axes[1].set_xticklabels([str(int(x)) for x in prec.index], rotation=45, ha='right')

        # Single shared legend at the bottom
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15),
                  ncol=min(len(labels), 3), fontsize=9, framealpha=0.9)

        plt.tight_layout()
        return fig

    def plot_specificity_accuracy(self, figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        """Line plots of specificity and accuracy across faults for each model with shared legend."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Specificity
        spec = self.summary_df.pivot(index='faultNumber', columns='model', values='specificity').sort_index()
        for col in spec.columns:
            axes[0].plot(spec.index, spec[col], marker='o', label=col)
        axes[0].set_title('Specificity (TNR) vs Fault', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Fault number', fontsize=10)
        axes[0].set_ylabel('Specificity', fontsize=10)
        axes[0].set_ylim(0, 1.05)
        axes[0].grid(alpha=0.3)
        axes[0].set_xticks(spec.index)
        axes[0].set_xticklabels([str(int(x)) for x in spec.index], rotation=45, ha='right')

        # Accuracy
        acc = self.summary_df.pivot(index='faultNumber', columns='model', values='accuracy').sort_index()
        for col in acc.columns:
            axes[1].plot(acc.index, acc[col], marker='o', label=col)
        axes[1].set_title('Accuracy vs Fault', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Fault number', fontsize=10)
        axes[1].set_ylabel('Accuracy', fontsize=10)
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(alpha=0.3)
        axes[1].set_xticks(acc.index)
        axes[1].set_xticklabels([str(int(x)) for x in acc.index], rotation=45, ha='right')

        # Single shared legend at the bottom
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15),
                  ncol=min(len(labels), 3), fontsize=9, framealpha=0.9)

        plt.tight_layout()
        return fig

    def plot_performance_boxplots(self, figsize: Tuple[int, int] = (15, 6)) -> plt.Figure:
        """
        Create boxplots showing distribution of ARL metrics across simulations.
        
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # ARL1 boxplot
        arl1_data = [self.df_results[self.df_results['model'] == model]['ARL1'].dropna() 
                     for model in self.models]
        bp1 = axes[0].boxplot(arl1_data, labels=self.models, patch_artist=True)
        axes[0].set_xticklabels(self.models, rotation=45, ha='right')
        axes[0].set_title('ARL1 Distribution', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('ARL1 (samples)', fontsize=12)
        axes[0].set_xlabel('Model', fontsize=12)
        axes[0].grid(axis='y', alpha=0.3)
        
        # ARL0 boxplot
        arl0_data = [self.df_results[self.df_results['model'] == model]['ARL0'].dropna() 
                     for model in self.models]
        bp2 = axes[1].boxplot(arl0_data, labels=self.models, patch_artist=True)
        axes[1].set_xticklabels(self.models, rotation=45, ha='right')
        axes[1].set_title('ARL0 Distribution', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('ARL0 (samples)', fontsize=12)
        axes[1].set_xlabel('Model', fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        
        # Detection fraction boxplot
        det_data = [self.df_results[self.df_results['model'] == model]['detection_fraction'] * 100
                    for model in self.models]
        bp3 = axes[2].boxplot(det_data, labels=self.models, patch_artist=True)
        axes[2].set_xticklabels(self.models, rotation=45, ha='right')
        axes[2].set_title('Detection Rate Distribution', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Detection Rate (%)', fontsize=12)
        axes[2].set_xlabel('Model', fontsize=12)
        axes[2].grid(axis='y', alpha=0.3)
        
        # Color the boxplots
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.models))) # type: ignore
        for bp, color in zip([bp1, bp2, bp3], [colors, colors, colors]):
            for patch, c in zip(bp['boxes'], color):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_fault_specific_comparison(self, fault_number: int, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create detailed comparison for a specific fault.
        
        Args:
            fault_number: The fault number to analyze
            
        Returns:
            Matplotlib figure object
        """
        fault_data = self.df_results[self.df_results['faultNumber'] == fault_number]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Fault {fault_number} - Detailed Comparison', fontsize=16, fontweight='bold')
        
        # ARL1 by simulation run
        for model in self.models:
            model_data = fault_data[fault_data['model'] == model]
            axes[0, 0].plot(model_data['simulationRun'], model_data['ARL1'], 
                          marker='o', label=model, alpha=0.7)
        axes[0, 0].set_title('ARL1 Across Simulation Runs')
        axes[0, 0].set_xlabel('Simulation Run')
        axes[0, 0].set_ylabel('ARL1')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # ARL0 by simulation run
        for model in self.models:
            model_data = fault_data[fault_data['model'] == model]
            axes[0, 1].plot(model_data['simulationRun'], model_data['ARL0'], 
                          marker='s', label=model, alpha=0.7)
        axes[0, 1].set_title('ARL0 Across Simulation Runs')
        axes[0, 1].set_xlabel('Simulation Run')
        axes[0, 1].set_ylabel('ARL0')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Detection fraction comparison
        det_by_model = fault_data.groupby('model')['detection_fraction'].mean() * 100
        bars = axes[1, 0].bar(range(len(self.models)), det_by_model.values)
        axes[1, 0].set_xticks(range(len(self.models)))
        axes[1, 0].set_xticklabels(self.models, rotation=45, ha='right')
        axes[1, 0].set_title('Average Detection Rate')
        axes[1, 0].set_ylabel('Detection Rate (%)')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, det_by_model.values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                          f'{value:.1f}%', ha='center', va='bottom')
        
        # Summary statistics table
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        
        summary_data = []
        for model in self.models:
            model_data = fault_data[fault_data['model'] == model]
            summary_data.append([
                model,
                f"{model_data['ARL1'].dropna().mean():.1f}",
                f"{model_data['ARL0'].dropna().mean():.1f}",
                f"{model_data['detection_fraction'].mean()*100:.1f}%"
            ])
        
        table = axes[1, 1].table(cellText=summary_data,
                                colLabels=['Model', 'Mean ARL1', 'Mean ARL0', 'Det. Rate'],
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.tight_layout()
        return fig

    def plot_metrics_comparison_bars(self, metrics: List[str], figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
        """
        Create bar charts comparing models across multiple metrics with error bars.
        Shows mean ± std for each metric.
        """
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

            # Calculate mean and std for each model
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
            ax.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=8)

        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Model Performance Comparison (Mean ± Std)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_metrics_radar(self, metrics: List[str] = None, figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Create radar/spider chart comparing models across multiple metrics.
        All metrics are normalized to 0-1 scale for comparison.
        """
        if metrics is None:
            metrics = ['precision', 'recall', 'specificity', 'accuracy', 'f1']

        # Filter available metrics
        metrics = [m for m in metrics if m in self.summary_df.columns]
        n_metrics = len(metrics)

        if n_metrics < 3:
            print("Need at least 3 metrics for radar chart")
            return None

        # Calculate mean values for each model
        model_values = {}
        for model in self.models:
            model_data = self.summary_df[self.summary_df['model'] == model]
            values = [model_data[metric].mean() for metric in metrics]
            model_values[model] = values

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

        colors = plt.cm.Set3(np.linspace(0, 1, len(self.models)))
        linestyles = ['-', '--', '-.', ':', '-', '--']

        for idx, (model, values) in enumerate(model_values.items()):
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2.5, label=model,
                   color=colors[idx], linestyle=linestyles[idx % len(linestyles)])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True)

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
        plt.title('Model Performance Comparison - Radar Chart',
                 fontsize=16, fontweight='bold', pad=20)

        return fig

    def plot_metrics_boxplots(self, metrics: List[str], figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
        """
        Create box plots comparing models across multiple metrics.
        Shows median, quartiles, and outliers.
        """
        n_metrics = len(metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_metrics == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_metrics > 1 else axes

        colors = plt.cm.Set3(np.linspace(0, 1, len(self.models)))

        for idx, metric in enumerate(metrics):
            if metric not in self.df_results.columns:
                continue

            ax = axes[idx]

            # Collect data for each model
            data_by_model = []
            model_names = []

            for model in self.models:
                data = self.df_results[self.df_results['model'] == model][metric].dropna()
                if not data.empty:
                    data_by_model.append(data)
                    model_names.append(model)

            # Create box plot
            bp = ax.boxplot(data_by_model, labels=model_names, patch_artist=True,
                           showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=6))

            # Color the boxes
            for patch, color in zip(bp['boxes'], colors[:len(model_names)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=11, fontweight='bold')
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(axis='y', alpha=0.3)

        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Model Performance Comparison - Box Plots', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_metrics_per_fault_bars(self, metrics: List[str], figsize: Tuple[int, int] = (18, 10)) -> plt.Figure:
        """
        Create grouped bar charts showing performance for each fault number.
        Similar to the combined ARL plot style - each fault shows all models side-by-side.

        Args:
            metrics: List of metric names to plot (e.g., ['precision', 'recall', 'specificity', 'accuracy', 'f1'])
            figsize: Figure size

        Returns:
            Matplotlib figure with subplots for each metric
        """
        n_metrics = len(metrics)
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_metrics == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_metrics > 1 else axes

        n_models = len(self.models)
        n_faults = len(self.fault_numbers)
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
            ax.set_xticklabels([str(int(f)) for f in self.fault_numbers], fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', alpha=0.3)

        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')

        # Single shared legend at the bottom
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                  ncol=min(len(labels), 3), fontsize=9, framealpha=0.9)

        plt.suptitle('Model Performance Comparison by Fault Number',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_metric_bars(self, metric: str, figsize: Tuple[int, int] = (8, 5)) -> plt.Figure:
        """Bar chart of an aggregate classification metric per model.

        Aggregates the metric across faults by mean.
        """
        if metric not in self.summary_df.columns:
            raise ValueError(f"Metric '{metric}' not found in summary_df.")
        agg = self.summary_df.groupby('model')[metric].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(range(len(agg)), agg.values)
        ax.set_xticks(range(len(agg)))
        ax.set_xticklabels(agg.index, rotation=45, ha='right')
        title = metric.replace('_', ' ').title()
        ax.set_title(f'{title} by Model')
        ax.set_ylabel(title)
        if metric not in {'conditional_ARL0', 'conditional_ARL1'}:
            ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        for bar, value in zip(bars, agg.values):
            if metric in {'conditional_ARL0', 'conditional_ARL1'}:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*bar.get_height(), f'{value:.2f}', ha='center', va='bottom')
            else:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{value:.2f}', ha='center', va='bottom')
        plt.tight_layout()
        return fig

    def plot_combined_arl_comparison(self, figsize: Tuple[int, int] = (18, 8)) -> plt.Figure:
        """
        Create a combined plot showing both ARL0 and ARL1 for each model and fault.
        This allows direct comparison of both metrics side-by-side.

        Returns:
            Matplotlib figure object with subplots for each fault number
        """
        # Get data
        arl1_pivot = self.summary_df.pivot(
            index='faultNumber',
            columns='model',
            values='conditional_ARL1'
        )
        arl0_pivot = self.summary_df.pivot(
            index='faultNumber',
            columns='model',
            values='conditional_ARL0'
        )

        n_faults = len(self.fault_numbers)
        n_models = len(self.models)

        # Create figure with subplots for each fault
        n_cols = min(3, n_faults)
        n_rows = (n_faults + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_faults == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_faults > 1 else axes

        # Color palette for models
        colors = plt.cm.Set3(np.linspace(0, 1, n_models))

        for idx, fault in enumerate(self.fault_numbers):
            ax = axes[idx]

            # Get data for this fault
            arl1_vals = arl1_pivot.loc[fault].values
            arl0_vals = arl0_pivot.loc[fault].values

            # Set up x positions for grouped bars
            x = np.arange(n_models)
            width = 0.35

            # Create grouped bars
            bars1 = ax.bar(x - width/2, arl1_vals, width, label='ARL1 (Detection Delay)',
                          color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            bars2 = ax.bar(x + width/2, arl0_vals, width, label='ARL0 (In-Control Run)',
                          color=colors, alpha=0.5, edgecolor='black', linewidth=0.5, hatch='//')

            # Customize subplot
            ax.set_xlabel('Model', fontsize=10)
            ax.set_ylabel('Samples', fontsize=10)
            ax.set_title(f'Fault {int(fault)}', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(self.models, rotation=45, ha='right', fontsize=8)
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.0f}', ha='center', va='bottom', fontsize=7)
            for bar in bars2:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.0f}', ha='center', va='bottom', fontsize=7)

        # Hide unused subplots
        for idx in range(n_faults, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Combined ARL0 and ARL1 Comparison by Fault and Model',
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        return fig

    def rank_models_by_combined_performance(self,
                                           arl0_weight: float = 0.5,
                                           arl1_weight: float = 0.5,
                                           method: str = 'normalized_score') -> pd.DataFrame:
        """
        Rank models based on combined ARL0 and ARL1 performance.

        Args:
            arl0_weight: Weight for ARL0 (higher is better). Default: 0.5
            arl1_weight: Weight for ARL1 (lower is better). Default: 0.5
            method: Ranking method. Options:
                - 'normalized_score': Normalize metrics and compute weighted score
                - 'ratio': Use ARL0/ARL1 ratio (higher is better)
                - 'harmonic_mean': Use harmonic mean of ARL0 and 1/ARL1

        Returns:
            DataFrame with models ranked by combined performance
        """
        results = []

        for model in self.models:
            model_summary = self.summary_df[self.summary_df['model'] == model]

            # Calculate average metrics across all faults
            avg_arl0 = model_summary['conditional_ARL0'].mean()
            avg_arl1 = model_summary['conditional_ARL1'].mean()

            # Calculate per-fault scores and ranks
            fault_scores = []
            for fault in self.fault_numbers:
                fault_data = model_summary[model_summary['faultNumber'] == fault]
                if len(fault_data) > 0:
                    arl0 = fault_data['conditional_ARL0'].values[0]
                    arl1 = fault_data['conditional_ARL1'].values[0]

                    if method == 'ratio' and not np.isnan(arl0) and not np.isnan(arl1) and arl1 > 0:
                        score = arl0 / arl1
                    elif method == 'harmonic_mean' and not np.isnan(arl0) and not np.isnan(arl1) and arl0 > 0 and arl1 > 0:
                        score = 2 / (1/arl0 + arl1)
                    else:
                        score = np.nan

                    fault_scores.append(score)

            avg_score = np.nanmean(fault_scores) if fault_scores else np.nan

            results.append({
                'Model': model,
                'Avg_ARL0': avg_arl0,
                'Avg_ARL1': avg_arl1,
                'Score': avg_score
            })

        ranking_df = pd.DataFrame(results)

        # Normalize and compute final score for 'normalized_score' method
        if method == 'normalized_score':
            # Normalize ARL0 (higher is better, so max = 1)
            arl0_max = ranking_df['Avg_ARL0'].max()
            arl0_min = ranking_df['Avg_ARL0'].min()
            if arl0_max > arl0_min:
                ranking_df['ARL0_normalized'] = (ranking_df['Avg_ARL0'] - arl0_min) / (arl0_max - arl0_min)
            else:
                ranking_df['ARL0_normalized'] = 1.0

            # Normalize ARL1 (lower is better, so min = 1)
            arl1_max = ranking_df['Avg_ARL1'].max()
            arl1_min = ranking_df['Avg_ARL1'].min()
            if arl1_max > arl1_min:
                ranking_df['ARL1_normalized'] = 1 - (ranking_df['Avg_ARL1'] - arl1_min) / (arl1_max - arl1_min)
            else:
                ranking_df['ARL1_normalized'] = 1.0

            # Combined score
            ranking_df['Score'] = (arl0_weight * ranking_df['ARL0_normalized'] +
                                  arl1_weight * ranking_df['ARL1_normalized'])

        # Sort by score (higher is better)
        ranking_df = ranking_df.sort_values('Score', ascending=False).reset_index(drop=True)
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)

        # Round numerical columns
        numeric_cols = ['Avg_ARL0', 'Avg_ARL1', 'Score']
        if 'ARL0_normalized' in ranking_df.columns:
            numeric_cols.extend(['ARL0_normalized', 'ARL1_normalized'])
        ranking_df[numeric_cols] = ranking_df[numeric_cols].round(3)

        # Reorder columns
        cols = ['Rank', 'Model', 'Avg_ARL0', 'Avg_ARL1', 'Score']
        if 'ARL0_normalized' in ranking_df.columns:
            cols.insert(-1, 'ARL0_normalized')
            cols.insert(-1, 'ARL1_normalized')

        return ranking_df[cols]
    
    def generate_full_report(self, save_path: str = None): # type: ignore
        """
        Generate a complete comparison report with all tables and plots.
        
        Args:
            save_path: Optional path to save the report figures
        """
        print("=" * 80)
        print("MODEL PERFORMANCE COMPARISON REPORT")
        print("=" * 80)
        print()
        
        # 1. Overall Summary Statistics
        print("1. OVERALL SUMMARY STATISTICS")
        print("-" * 40)
        summary_stats = self.create_summary_statistics()
        print(summary_stats.to_string(index=False))
        print()
        
        # 2. Comprehensive Comparison Table
        print("2. COMPREHENSIVE COMPARISON TABLE")
        print("-" * 40)
        comparison_table = self.create_comparison_table('all')
        print(comparison_table.to_string())
        print()
        
        # 3. ARL1 Detailed Table
        print("3. ARL1 COMPARISON (Mean ± Std)")
        print("-" * 40)
        arl1_table = self._create_arl1_table()
        print(arl1_table.to_string())
        print()
        
        # 4. Detection Performance Table
        print("4. DETECTION PERFORMANCE")
        print("-" * 40)
        det_table = self._create_detection_table()
        print(det_table.to_string())
        print()
        
        # Generate all plots
        fig1 = self.plot_arl_comparison()
        fig2 = self.plot_detection_heatmap()
        fig3 = self.plot_performance_boxplots()
        # Classification metric plots
        fig4 = self.plot_precision_recall()
        fig5 = self.plot_specificity_accuracy()
        fig6 = self.plot_metric_bars('precision')
        fig7 = self.plot_metric_bars('recall')
        fig8 = self.plot_metric_bars('specificity')
        fig9 = self.plot_metric_bars('accuracy')
        # New bar chart comparisons (replaced histograms)
        fig10 = self.plot_metrics_comparison_bars(['precision', 'recall', 'specificity', 'accuracy', 'f1'])
        fig11 = self.plot_metrics_per_fault_bars(['precision', 'recall', 'specificity', 'accuracy', 'f1'])

        if save_path:
            fig1.savefig(f"{save_path}_arl_comparison.png", dpi=300, bbox_inches='tight')
            fig2.savefig(f"{save_path}_detection_heatmap.png", dpi=300, bbox_inches='tight')
            fig3.savefig(f"{save_path}_performance_boxplots.png", dpi=300, bbox_inches='tight')
            fig4.savefig(f"{save_path}_precision_recall.png", dpi=300, bbox_inches='tight')
            fig5.savefig(f"{save_path}_specificity_accuracy.png", dpi=300, bbox_inches='tight')
            fig6.savefig(f"{save_path}_bars_precision.png", dpi=300, bbox_inches='tight')
            fig7.savefig(f"{save_path}_bars_recall.png", dpi=300, bbox_inches='tight')
            fig8.savefig(f"{save_path}_bars_specificity.png", dpi=300, bbox_inches='tight')
            fig9.savefig(f"{save_path}_bars_accuracy.png", dpi=300, bbox_inches='tight')
            fig10.savefig(f"{save_path}_metrics_overall_bars.png", dpi=300, bbox_inches='tight')
            fig11.savefig(f"{save_path}_metrics_per_fault_bars.png", dpi=300, bbox_inches='tight')
            print(f"Figures saved to {save_path}_*.png")

        plt.show()

        return summary_stats, comparison_table

    # ==================================================================
    # GROUPED METHODS FOR ORGANIZED NOTEBOOK PRESENTATION
    # ==================================================================

    def display_summary_tables(self) -> None:
        """
        Display all summary tables and statistics.
        Call this in a dedicated notebook cell for viewing tables.
        """
        print("=" * 80)
        print("MODEL PERFORMANCE SUMMARY TABLES")
        print("=" * 80)
        print()

        # 1. Overall Summary Statistics
        print("1. OVERALL SUMMARY STATISTICS")
        print("-" * 80)
        summary_stats = self.create_summary_statistics()
        from IPython.display import display
        display(summary_stats)
        print()

        # 2. Comprehensive Comparison Table
        print("\n2. COMPREHENSIVE COMPARISON TABLE")
        print("-" * 80)
        comparison_table = self.create_comparison_table('all')
        display(comparison_table)
        print()

    def display_model_rankings(self) -> None:
        """
        Display overall model ranking tables based on combined ARL0 and ARL1 performance.
        These rankings use averages across all faults.
        Call this in a dedicated notebook cell for viewing rankings.
        """
        from IPython.display import display

        print("=" * 80)
        print("OVERALL MODEL RANKINGS (Averaged Across All Faults)")
        print("=" * 80)
        print()

        # Method 1: Normalized score (weighted combination)
        print("1. Normalized Score Method (ARL0 weight=0.5, ARL1 weight=0.5)")
        print("-" * 80)
        ranking_normalized = self.rank_models_by_combined_performance(
            arl0_weight=0.5,
            arl1_weight=0.5,
            method='normalized_score'
        )
        display(ranking_normalized)
        print()

        # Method 2: Ratio method (ARL0/ARL1)
        print("2. Ratio Method (ARL0/ARL1 - higher is better)")
        print("-" * 80)
        ranking_ratio = self.rank_models_by_combined_performance(method='ratio')
        display(ranking_ratio)
        print()

        # Method 3: Harmonic mean
        print("3. Harmonic Mean Method")
        print("-" * 80)
        ranking_harmonic = self.rank_models_by_combined_performance(method='harmonic_mean')
        display(ranking_harmonic)
        print()

    def display_per_fault_rankings(self, method: str = 'ratio') -> None:
        """
        Display model rankings for each individual fault.
        Shows which model performs best for each specific fault condition.

        Args:
            method: Ranking method ('ratio', 'normalized_score', or 'harmonic_mean')
        """
        from IPython.display import display

        print("=" * 80)
        print("PER-FAULT MODEL RANKINGS")
        print("=" * 80)
        print()

        for fault in self.fault_numbers:
            print(f"\n{'='*80}")
            print(f"FAULT {int(fault)} - Best Models")
            print(f"{'='*80}\n")

            # Get data for this fault
            fault_data = self.summary_df[self.summary_df['faultNumber'] == fault].copy()

            if len(fault_data) == 0:
                print(f"No data available for Fault {int(fault)}")
                continue

            # Calculate scores
            if method == 'ratio':
                fault_data['Score'] = fault_data['conditional_ARL0'] / fault_data['conditional_ARL1'].replace(0, np.nan)
                score_label = 'ARL0/ARL1 Ratio'
            elif method == 'normalized_score':
                arl0_min = fault_data['conditional_ARL0'].min()
                arl0_max = fault_data['conditional_ARL0'].max()
                arl1_min = fault_data['conditional_ARL1'].min()
                arl1_max = fault_data['conditional_ARL1'].max()

                if arl0_max > arl0_min:
                    fault_data['ARL0_norm'] = (fault_data['conditional_ARL0'] - arl0_min) / (arl0_max - arl0_min)
                else:
                    fault_data['ARL0_norm'] = 1.0

                if arl1_max > arl1_min:
                    fault_data['ARL1_norm'] = 1 - (fault_data['conditional_ARL1'] - arl1_min) / (arl1_max - arl1_min)
                else:
                    fault_data['ARL1_norm'] = 1.0

                fault_data['Score'] = 0.5 * fault_data['ARL0_norm'] + 0.5 * fault_data['ARL1_norm']
                score_label = 'Normalized Score'
            else:  # harmonic_mean
                fault_data['Score'] = 2 / (1/fault_data['conditional_ARL0'] + fault_data['conditional_ARL1'])
                score_label = 'Harmonic Mean'

            # Sort and rank
            fault_data = fault_data.sort_values('Score', ascending=False).reset_index(drop=True)
            fault_data['Rank'] = range(1, len(fault_data) + 1)

            # Create display table
            display_cols = ['Rank', 'model', 'conditional_ARL0', 'conditional_ARL1', 'Score']
            display_df = fault_data[display_cols].copy()
            display_df.columns = ['Rank', 'Model', 'ARL0', 'ARL1', score_label]
            display_df = display_df.round(3)

            display(display_df)
            print()

    def plot_arl_averages(self, figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        """
        Plot average ARL0 and ARL1 values across all faults for each model.
        This matches the data shown in the ranking tables.
        Both plots use the same sort order (by ARL1) and consistent colors per model.

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure object
        """
        # Calculate average ARL0 and ARL1 per model
        avg_data = self.summary_df.groupby('model').agg({
            'conditional_ARL0': 'mean',
            'conditional_ARL1': 'mean'
        }).reset_index()

        # Sort by ARL1 (lower is better) - this order will be used for both plots
        avg_data_sorted = avg_data.sort_values('conditional_ARL1').reset_index(drop=True)

        # Assign consistent colors to each model
        colors = plt.cm.Set3(np.linspace(0, 1, len(avg_data_sorted)))

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # ARL1 averages (sorted by ARL1)
        axes[0].barh(avg_data_sorted['model'], avg_data_sorted['conditional_ARL1'], color=colors)
        axes[0].set_xlabel('Average ARL1 (samples)', fontsize=10)
        axes[0].set_title('Average Detection Delay\n(Lower is Better)', fontsize=12, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        for idx, (model, value) in enumerate(zip(avg_data_sorted['model'], avg_data_sorted['conditional_ARL1'])):
            axes[0].text(value, idx, f' {value:.2f}', va='center', fontsize=9)

        # ARL0 averages (same sort order as ARL1, same colors)
        axes[1].barh(avg_data_sorted['model'], avg_data_sorted['conditional_ARL0'], color=colors)
        axes[1].set_xlabel('Average ARL0 (samples)', fontsize=10)
        axes[1].set_title('Average In-Control Run Length\n(Higher is Better)', fontsize=12, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        for idx, (model, value) in enumerate(zip(avg_data_sorted['model'], avg_data_sorted['conditional_ARL0'])):
            axes[1].text(value, idx, f' {value:.2f}', va='center', fontsize=9)

        plt.suptitle('Average ARL Performance Across All Faults', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_confusion_matrix_components_per_fault(self, figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
        """
        Plot confusion matrix components (TP, FP, TN, FN) for each fault number.
        Shows how each model performs in terms of raw counts across different faults.

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure object
        """
        components = ['TP', 'FP', 'TN', 'FN']
        component_labels = {
            'TP': 'True Positives (Correct Alarms)',
            'FP': 'False Positives (False Alarms)',
            'TN': 'True Negatives (Correct Normal)',
            'FN': 'False Negatives (Missed Detections)'
        }

        # Calculate mean values per model and fault
        confusion_data = self.df_results.groupby(['model', 'faultNumber'])[components].mean().reset_index()

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        n_models = len(self.models)
        colors = plt.cm.Set3(np.linspace(0, 1, n_models))

        for idx, component in enumerate(components):
            ax = axes[idx]

            # Pivot data for this component
            pivot_data = confusion_data.pivot(
                index='faultNumber',
                columns='model',
                values=component
            )

            # Create grouped bar chart
            pivot_data.plot(kind='bar', ax=ax, width=0.8, color=colors, legend=False)

            ax.set_title(component_labels[component], fontsize=12, fontweight='bold')
            ax.set_xlabel('Fault Number', fontsize=10)
            ax.set_ylabel('Count (mean across runs)', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticks(range(len(pivot_data.index)))
            ax.set_xticklabels([str(int(x)) for x in pivot_data.index], rotation=45, ha='right')

        # Single shared legend at the bottom
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title='Model', loc='lower center',
                  bbox_to_anchor=(0.5, -0.08), ncol=min(len(labels), 3),
                  fontsize=9, framealpha=0.9)

        plt.suptitle('Confusion Matrix Components by Fault', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        return fig

    def plot_confusion_matrix_components_average(self, figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Plot average confusion matrix components (TP, FP, TN, FN) across all faults.
        Shows overall model performance in terms of raw counts.
        Layout: 2x2 grid with TP, FP in first row and TN, FN in second row.

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure object
        """
        components = ['TP', 'FP', 'TN', 'FN']
        component_labels = {
            'TP': 'True Positives\n(Correct Alarms)',
            'FP': 'False Positives\n(False Alarms)',
            'TN': 'True Negatives\n(Correct Normal)',
            'FN': 'False Negatives\n(Missed Detections)'
        }

        # Calculate mean values per model across all faults and runs
        avg_data = self.df_results.groupby('model')[components].mean().reset_index()

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        n_models = len(self.models)
        x = np.arange(n_models)
        colors = plt.cm.Set3(np.linspace(0, 1, n_models))

        for idx, component in enumerate(components):
            ax = axes[idx]

            # Get values for this component
            values = avg_data[component].values

            # Create bar chart
            bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

            ax.set_title(component_labels[component], fontsize=11, fontweight='bold')
            ax.set_ylabel('Average Count', fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(avg_data['model'], rotation=45, ha='right', fontsize=8)
            ax.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=8)

        plt.suptitle('Average Confusion Matrix Components Across All Faults',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_false_rates(self, figsize: Tuple[int, int] = (8, 5)) -> plt.Figure:
        """
        Plot False Positive Rate (FPR) and False Negative Rate (FNR) comparison.
        Shows mean ± std for each model across all faults.
        Lower values are better for both metrics.

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        metrics = ['false_positive_rate', 'false_negative_rate']
        titles = [
            'False Positive Rate (FPR)\n(Lower is Better)',
            'False Negative Rate (FNR)\n(Lower is Better)'
        ]

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx]

            # Calculate mean and std for each model
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
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('Rate', fontsize=10)
            ax.set_ylim(0, max(means) * 1.2 if means else 1)
            ax.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=8)

        plt.suptitle('False Positive Rate & False Negative Rate Comparison',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_confusion_matrix_average(self, figsize: Tuple[int, int] = (16, 5)) -> plt.Figure:
        """
        Plot confusion matrix heatmaps (average across all faults) for each model.
        Standard confusion matrix layout showing TN, FP, FN, TP in 2x2 format.

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure object
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        # Calculate average confusion matrix values per model
        avg_data = self.df_results.groupby('model')[['TN', 'FP', 'FN', 'TP']].mean().reset_index()

        n_models = len(self.models)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_models > 1 else axes

        # Heat colormap (yellow to orange to red)
        cmap = plt.cm.YlOrRd

        for idx, model in enumerate(self.models):
            ax = axes[idx]

            model_data = avg_data[avg_data['model'] == model]
            if model_data.empty:
                ax.axis('off')
                continue

            # Create confusion matrix in standard format:
            # [[TN, FP],
            #  [FN, TP]]
            cm = np.array([
                [model_data['TN'].values[0], model_data['FP'].values[0]],
                [model_data['FN'].values[0], model_data['TP'].values[0]]
            ])

            # Create heatmap
            im = ax.imshow(cm, cmap=cmap, aspect='auto', vmin=0, vmax=np.max(cm))

            # Set ticks and labels
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Predicted\nNegative', 'Predicted\nPositive'], fontsize=9)
            ax.set_yticklabels(['Actual\nNegative', 'Actual\nPositive'], fontsize=9)

            # Add text annotations
            for i in range(2):
                for j in range(2):
                    text = ax.text(j, i, f'{cm[i, j]:.1f}',
                                 ha="center", va="center", color="black" if cm[i, j] < np.max(cm) * 0.5 else "white",
                                 fontsize=11, fontweight='bold')

            # Add labels for each cell
            labels = [['TN', 'FP'], ['FN', 'TP']]
            for i in range(2):
                for j in range(2):
                    ax.text(j, i - 0.35, labels[i][j],
                           ha="center", va="center", color="gray",
                           fontsize=8, style='italic')

            ax.set_title(f'{model}', fontsize=10, fontweight='bold', pad=10)

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Confusion Matrices (Average Across All Faults)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_confusion_matrices_per_fault(self, figsize: Tuple[int, int] = (18, 12)) -> plt.Figure:
        """
        Plot confusion matrix heatmaps for each fault and model combination.
        Shows how confusion matrix varies across different faults.

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure object
        """
        import matplotlib.pyplot as plt

        # Calculate average confusion matrix values per model and fault
        fault_data = self.df_results.groupby(['model', 'faultNumber'])[['TN', 'FP', 'FN', 'TP']].mean().reset_index()

        n_models = len(self.models)
        n_faults = len(self.fault_numbers)

        fig, axes = plt.subplots(n_faults, n_models, figsize=figsize)
        if n_models == 1 and n_faults == 1:
            axes = np.array([[axes]])
        elif n_models == 1:
            axes = axes.reshape(-1, 1)
        elif n_faults == 1:
            axes = axes.reshape(1, -1)

        cmap = plt.cm.YlOrRd

        for fault_idx, fault in enumerate(self.fault_numbers):
            for model_idx, model in enumerate(self.models):
                ax = axes[fault_idx, model_idx]

                model_fault_data = fault_data[(fault_data['model'] == model) &
                                             (fault_data['faultNumber'] == fault)]

                if model_fault_data.empty:
                    ax.axis('off')
                    continue

                # Create confusion matrix
                cm = np.array([
                    [model_fault_data['TN'].values[0], model_fault_data['FP'].values[0]],
                    [model_fault_data['FN'].values[0], model_fault_data['TP'].values[0]]
                ])

                # Create heatmap
                im = ax.imshow(cm, cmap=cmap, aspect='auto', vmin=0, vmax=np.max(cm))

                # Set ticks and labels
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])

                # Only show labels on edges
                if fault_idx == n_faults - 1:
                    ax.set_xticklabels(['Pred\nNeg', 'Pred\nPos'], fontsize=8)
                else:
                    ax.set_xticklabels([])

                if model_idx == 0:
                    ax.set_yticklabels(['Actual\nNeg', 'Actual\nPos'], fontsize=8)
                else:
                    ax.set_yticklabels([])

                # Add text annotations
                for i in range(2):
                    for j in range(2):
                        text = ax.text(j, i, f'{cm[i, j]:.0f}',
                                     ha="center", va="center",
                                     color="black" if cm[i, j] < np.max(cm) * 0.5 else "white",
                                     fontsize=9, fontweight='bold')

                # Add labels for each cell (smaller)
                labels = [['TN', 'FP'], ['FN', 'TP']]
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i - 0.35, labels[i][j],
                               ha="center", va="center", color="gray",
                               fontsize=7, style='italic')

                # Title only on top row
                if fault_idx == 0:
                    ax.set_title(f'{model}', fontsize=9, fontweight='bold', pad=5)

                # Fault number label on left
                if model_idx == 0:
                    ax.text(-0.5, 0.5, f'Fault {int(fault)}',
                           transform=ax.transAxes, rotation=90,
                           va='center', ha='right', fontsize=9, fontweight='bold')

        plt.suptitle('Confusion Matrices Per Fault', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_arl_analysis(self, figsize_comparison: Tuple[int, int] = (15, 6),
                         figsize_combined: Tuple[int, int] = (20, 10)) -> None:
        """
        Plot all ARL-related analyses (ARL0 and ARL1).
        Call this in a dedicated notebook cell for ARL visualizations.

        Args:
            figsize_comparison: Figure size for side-by-side ARL comparison
            figsize_combined: Figure size for combined ARL by fault plot
        """
        print("Generating ARL Analysis Plots...")

        # 1. Side-by-side ARL comparison (per fault)
        fig1 = self.plot_arl_comparison(figsize=figsize_comparison)
        plt.show()

        # 2. Combined ARL by fault
        fig2 = self.plot_combined_arl_comparison(figsize=figsize_combined)
        plt.show()

        print("✓ ARL analysis plots complete")

    def plot_overall_metrics(self, metrics: List[str] = None,
                            figsize_bars: Tuple[int, int] = (15, 8),
                            figsize_radar: Tuple[int, int] = (12, 10),
                            include_radar: bool = True) -> None:
        """
        Plot overall classification metrics averaged across all faults.
        Call this in a dedicated notebook cell for overall metric visualizations.

        Args:
            metrics: List of metrics to plot. Defaults to ['precision', 'recall', 'specificity', 'accuracy', 'f1']
            figsize_bars: Figure size for bar charts
            figsize_radar: Figure size for radar chart
            include_radar: Whether to include radar chart
        """
        if metrics is None:
            metrics = ['precision', 'recall', 'specificity', 'accuracy', 'f1']

        print("Generating Overall Metrics Plots...")

        # 1. Bar charts with error bars
        fig1 = self.plot_metrics_comparison_bars(metrics, figsize=figsize_bars)
        plt.show()

        # 2. Radar chart (optional)
        if include_radar:
            fig2 = self.plot_metrics_radar(metrics, figsize=figsize_radar)
            if fig2:
                plt.show()

        print("✓ Overall metrics plots complete")

    def plot_per_fault_bars(self, metrics: List[str] = None,
                            figsize: Tuple[int, int] = (18, 10)) -> None:
        """
        Plot grouped bar charts showing classification metrics by fault number.
        Call this in a dedicated notebook cell for per-fault bar chart visualizations.

        Args:
            metrics: List of metrics to plot. Defaults to ['precision', 'recall', 'specificity', 'accuracy', 'f1']
            figsize: Figure size for grouped bar charts
        """
        if metrics is None:
            metrics = ['precision', 'recall', 'specificity', 'accuracy', 'f1']

        print("Generating Per-Fault Bar Charts...")

        fig = self.plot_metrics_per_fault_bars(metrics, figsize=figsize)
        plt.show()

        print("✓ Per-fault bar charts complete")

    def plot_per_fault_lines(self, figsize: Tuple[int, int] = (12, 5)) -> None:
        """
        Plot line charts showing classification metrics trends across faults.
        Call this in a dedicated notebook cell for per-fault line plot visualizations.

        Args:
            figsize: Figure size for line plots
        """
        print("Generating Per-Fault Line Plots...")

        # Precision/Recall line plots
        fig1 = self.plot_precision_recall(figsize=figsize)
        plt.show()

        # Specificity/Accuracy line plots
        fig2 = self.plot_specificity_accuracy(figsize=figsize)
        plt.show()

        print("✓ Per-fault line plots complete")

    def plot_per_fault_metrics(self, metrics: List[str] = None,
                               figsize_bars: Tuple[int, int] = (18, 10),
                               figsize_lines: Tuple[int, int] = (12, 5)) -> None:
        """
        Plot all per-fault metrics (both bar charts and line plots).
        This is a convenience method that calls both plot_per_fault_bars() and plot_per_fault_lines().

        Args:
            metrics: List of metrics to plot. Defaults to ['precision', 'recall', 'specificity', 'accuracy', 'f1']
            figsize_bars: Figure size for grouped bar charts
            figsize_lines: Figure size for line plots
        """
        print("Generating Per-Fault Metrics Plots...")

        # Bar charts
        self.plot_per_fault_bars(metrics, figsize=figsize_bars)

        # Line plots
        self.plot_per_fault_lines(figsize=figsize_lines)

        print("✓ All per-fault metrics plots complete")

    def plot_heatmaps(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot heatmap visualizations for detection rates and ARL1.
        Call this in a dedicated notebook cell for heatmap visualizations.

        Args:
            figsize: Figure size for heatmaps
        """
        print("Generating Heatmaps...")

        fig = self.plot_detection_heatmap(figsize=figsize)
        plt.show()

        print("✓ Heatmap plots complete")

    def plot_distributions(self, figsize: Tuple[int, int] = (15, 6)) -> None:
        """
        Plot distribution and variability analyses using boxplots.
        Call this in a dedicated notebook cell for distribution visualizations.

        Args:
            figsize: Figure size for boxplots
        """
        print("Generating Distribution Plots...")

        fig = self.plot_performance_boxplots(figsize=figsize)
        plt.show()

        print("✓ Distribution plots complete")
