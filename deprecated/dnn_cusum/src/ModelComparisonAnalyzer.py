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
        det_frac = self.summary_df.pivot(
            index='faultNumber', 
            columns='model', 
            values='avg_detection_fraction'
        )
        non_det = self.summary_df.pivot(
            index='faultNumber', 
            columns='model', 
            values='non_detection_fraction'
        )
        
        # Combine detection metrics
        combined = pd.DataFrame(index=det_frac.index)
        for model in det_frac.columns:
            combined[f'{model}_DetRate'] = (det_frac[model] * 100).apply(lambda x: f'{x:.1f}%')
            combined[f'{model}_MissRate'] = (non_det[model] * 100).apply(lambda x: f'{x:.1f}%')
        
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
                'Overall Detection Rate': model_data['detection_fraction'].mean() * 100,
                'False Alarm Rate': (model_data['ARL0'].notna().sum() / len(model_data)) * 100,
                'Miss Rate': (model_data['ARL1'].isna().sum() / len(model_data)) * 100,
                #'Best Fault Detection': model_summary['conditional_ARL1'].idxmin()[1] if not model_summary['conditional_ARL1'].isna().all() else 'N/A',
                #'Worst Fault Detection': model_summary['conditional_ARL1'].idxmax()[1] if not model_summary['conditional_ARL1'].isna().all() else 'N/A'
            })
        
        summary_stats_df = pd.DataFrame(stats)
        
        # Round numerical columns
        numeric_cols = ['Mean ARL0', 'Std ARL0', 'Mean ARL1', 'Std ARL1', 
                       'Overall Detection Rate', 'False Alarm Rate', 'Miss Rate']
        summary_stats_df[numeric_cols] = summary_stats_df[numeric_cols].round(2)
        
        return summary_stats_df
    
    def plot_arl_comparison(self, figsize: Tuple[int, int] = (15, 6)) -> plt.Figure:
        """
        Create bar plots comparing ARL0 and ARL1 across models and faults.
        
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
        arl1_pivot.plot(kind='bar', ax=axes[0], width=0.8)
        axes[0].set_title('ARL1 (Detection Delay) Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Fault Number', fontsize=12)
        axes[0].set_ylabel('ARL1 (samples)', fontsize=12)
        axes[0].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
        
        # ARL0 comparison
        arl0_pivot = self.summary_df.pivot(
            index='faultNumber', 
            columns='model', 
            values='conditional_ARL0'
        )
        arl0_pivot.plot(kind='bar', ax=axes[1], width=0.8)
        axes[1].set_title('ARL0 (False Alarm) Comparison', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Fault Number', fontsize=12)
        axes[1].set_ylabel('ARL0 (samples)', fontsize=12)
        axes[1].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
        
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
        axes[0].set_title('ARL1 Distribution', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('ARL1 (samples)', fontsize=12)
        axes[0].set_xlabel('Model', fontsize=12)
        axes[0].grid(axis='y', alpha=0.3)
        
        # ARL0 boxplot
        arl0_data = [self.df_results[self.df_results['model'] == model]['ARL0'].dropna() 
                     for model in self.models]
        bp2 = axes[1].boxplot(arl0_data, labels=self.models, patch_artist=True)
        axes[1].set_title('ARL0 Distribution', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('ARL0 (samples)', fontsize=12)
        axes[1].set_xlabel('Model', fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        
        # Detection fraction boxplot
        det_data = [self.df_results[self.df_results['model'] == model]['detection_fraction'] * 100 
                   for model in self.models]
        bp3 = axes[2].boxplot(det_data, labels=self.models, patch_artist=True)
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
        axes[1, 0].set_xticklabels(self.models)
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
        
        if save_path:
            fig1.savefig(f"{save_path}_arl_comparison.png", dpi=300, bbox_inches='tight')
            fig2.savefig(f"{save_path}_detection_heatmap.png", dpi=300, bbox_inches='tight')
            fig3.savefig(f"{save_path}_performance_boxplots.png", dpi=300, bbox_inches='tight')
            print(f"Figures saved to {save_path}_*.png")
        
        plt.show()
        
        return summary_stats, comparison_table

