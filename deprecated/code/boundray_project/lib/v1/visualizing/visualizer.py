import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from ipywidgets import interact, interactive, fixed, widgets
from IPython.display import display, clear_output
import warnings
warnings.filterwarnings('ignore')

class BoundaryVisualizer:
    """Interactive visualization for boundary verification and adjustment"""
    
    def __init__(self, boundary_calc, train_data: np.ndarray, test_data: np.ndarray = None):
        self.boundary_calc = boundary_calc
        self.config = boundary_calc.config
        self.train_data = train_data
        self.test_data = test_data
        
    def plot_distribution_with_boundaries(self, var_idx: int, level: str = 'All', 
                                         data_source: str = 'Both'):
        """
        Plot histogram with ALL N boundaries for a single variable
        
        Args:
            var_idx: Variable index
            level: 'Sensitive', 'Medium', 'Large', or 'All'
            data_source: 'Train', 'Test', or 'Both'
        """
        var_name = self.config.variable_names[var_idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot histogram of training data
        if data_source in ['Train', 'Both']:
            var_train = self.train_data[:, var_idx]
            ax.hist(var_train, bins=50, alpha=0.6, color='steelblue', 
                   edgecolor='black', linewidth=0.5, density=True,
                   label='Training Data Distribution')
        
        # Plot test data if available
        if data_source in ['Test', 'Both'] and self.test_data is not None:
            var_test = self.test_data[:, var_idx]
            ax.hist(var_test, bins=50, alpha=0.4, color='coral',
                   edgecolor='black', linewidth=0.5, density=True,
                   label='Test Data Distribution')
        
        # Plot boundaries
        colors = {'Sensitive': 'red', 'Medium': 'orange', 'Large': 'green'}
        line_styles = {'Sensitive': '--', 'Medium': '-.', 'Large': ':'}
        
        levels_to_plot = ['Sensitive', 'Medium', 'Large'] if level == 'All' else [level]
        
        for lev in levels_to_plot:
            boundaries_list = self.boundary_calc.get_boundaries(var_idx, lev)  # Now returns LIST
            color = colors[lev]
            linestyle = line_styles[lev]
            
            # Plot each boundary in the list
            for boundary_idx, bounds in enumerate(boundaries_list):
                # Plot vertical lines for boundaries
                ax.axvline(bounds['lower'], color=color, linestyle=linestyle,
                          linewidth=2, alpha=0.7, 
                          label=f"{lev} Boundary {boundary_idx+1} Lower" if boundary_idx == 0 else "")
                ax.axvline(bounds['upper'], color=color, linestyle=linestyle,
                          linewidth=2, alpha=0.7,
                          label=f"{lev} Boundary {boundary_idx+1} Upper" if boundary_idx == 0 else "")
                
                # Shade the acceptable region
                ax.axvspan(bounds['lower'], bounds['upper'], alpha=0.03, color=color)
                
                # Mark the center
                ax.axvline(bounds['center'], color=color, linestyle='-',
                          linewidth=1, alpha=0.4)
        
        # Labels and title
        ax.set_xlabel('Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        n_boundaries = len(boundaries_list) if level != 'All' else len(self.boundary_calc.get_boundaries(var_idx, 'Sensitive'))
        ax.set_title(f'Distribution with {n_boundaries} Boundaries per Level: {var_name} (Variable {var_idx})',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def plot_timeline_with_boundaries(self, var_idx: int, level: str = 'Sensitive',
                                     data_source: str = 'Both', max_samples: int = 500):
        """
        Plot variable timeline with ALL N boundary markers
        
        Args:
            var_idx: Variable index
            level: Sensitivity level to display
            data_source: 'Train', 'Test', or 'Both'
            max_samples: Maximum number of samples to plot
        """
        # Handle 'All' level - default to 'Sensitive' for timeline view
        if level == 'All':
            level = 'Sensitive'
            
        var_name = self.config.variable_names[var_idx]
        
        # Prepare data based on source selection
        if data_source == 'Train':
            combined_data = self.train_data[:max_samples, var_idx]
            time_indices = np.arange(len(combined_data))
            n_train = len(combined_data)
            n_test = 0
        elif data_source == 'Test' and self.test_data is not None:
            combined_data = self.test_data[:max_samples, var_idx]
            time_indices = np.arange(len(combined_data))
            n_train = 0
            n_test = len(combined_data)
        else:  # Both
            train_var = self.train_data[:max_samples, var_idx]
            n_train = len(train_var)
            
            if self.test_data is not None:
                test_var = self.test_data[:max_samples, var_idx]
                n_test = len(test_var)
                combined_data = np.concatenate([train_var, test_var])
            else:
                combined_data = train_var
                n_test = 0
            
            time_indices = np.arange(len(combined_data))
        
        # Get ALL boundaries for this level (now a list)
        boundaries_list = self.boundary_calc.get_boundaries(var_idx, level)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 5))
        
        # Plot timeline
        if data_source == 'Both' and n_test > 0:
            ax.plot(time_indices[:n_train], combined_data[:n_train],
                   'b-', linewidth=1, alpha=0.7, label='Training Data')
            ax.plot(time_indices[n_train:], combined_data[n_train:],
                   'r-', linewidth=1, alpha=0.7, label='Test Data (Out-of-Control)')
        elif data_source == 'Train':
            ax.plot(time_indices, combined_data, 'b-', linewidth=1, alpha=0.7,
                   label='Training Data')
        else:  # Test
            ax.plot(time_indices, combined_data, 'r-', linewidth=1, alpha=0.7,
                   label='Test Data')
        
        # Plot ALL N boundaries
        boundary_colors = plt.cm.Set3(np.linspace(0, 1, len(boundaries_list)))
        
        for boundary_idx, bounds in enumerate(boundaries_list):
            color = boundary_colors[boundary_idx]
            
            # Plot boundaries
            ax.axhline(bounds['upper'], color=color, linestyle='--',
                      linewidth=2, alpha=0.8, 
                      label=f'{level} Boundary {boundary_idx+1} Upper')
            ax.axhline(bounds['lower'], color=color, linestyle='--',
                      linewidth=2, alpha=0.8, 
                      label=f'{level} Boundary {boundary_idx+1} Lower')
            ax.axhline(bounds['center'], color=color, linestyle='-',
                      linewidth=1.5, alpha=0.5)
            
            # Shade the acceptable region
            ax.axhspan(bounds['lower'], bounds['upper'], alpha=0.08, color=color)
        
        # Mark transition point
        if data_source == 'Both' and n_test > 0:
            ax.axvline(n_train, color='black', linestyle=':', linewidth=2,
                      alpha=0.5, label='Train/Test Split')
        
        # Labels and title
        ax.set_xlabel('Time Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        title_suffix = f' ({data_source} Data)' if data_source != 'Both' else ''
        ax.set_title(f'Timeline with {len(boundaries_list)} {level} Boundaries: {var_name} (Variable {var_idx}){title_suffix}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_viewer(self):
        """Create interactive widget for viewing all variables"""
        
        def update_plot(var_idx, plot_type, level, data_source, max_samples):
            if plot_type == 'Distribution':
                self.plot_distribution_with_boundaries(var_idx, level, data_source)
            else:  # Timeline
                self.plot_timeline_with_boundaries(var_idx, level, data_source, max_samples)
        
        # Create widgets
        var_selector = widgets.Dropdown(
            options=[(f"{i}: {name}", i) for i, name in enumerate(self.config.variable_names)],
            value=0,
            description='Variable:',
            style={'description_width': 'initial'}
        )
        
        plot_type_selector = widgets.Dropdown(
            options=['Distribution', 'Timeline'],
            value='Distribution',
            description='Plot Type:',
            style={'description_width': 'initial'}
        )
        
        level_selector = widgets.Dropdown(
            options=['Sensitive', 'Medium', 'Large', 'All'],
            value='Sensitive',
            description='Level:',
            style={'description_width': 'initial'},
            tooltip='Note: Each level now shows N=3 boundaries'
        )
        
        data_source_selector = widgets.Dropdown(
            options=['Both', 'Train', 'Test'],
            value='Both',
            description='Data Source:',
            style={'description_width': 'initial'}
        )
        
        samples_slider = widgets.IntSlider(
            value=500,
            min=100,
            max=1000,
            step=50,
            description='Max Samples:',
            style={'description_width': 'initial'}
        )
        
        # Create interactive plot
        interact(update_plot, 
                var_idx=var_selector,
                plot_type=plot_type_selector,
                level=level_selector,
                data_source=data_source_selector,
                max_samples=samples_slider)

    def plot_cluster_scatter_view(self, var_idx: int, level: str = 'Sensitive',
                                    method: str = None, data_source: str = 'Train',
                                    max_samples: int = 500):
            """
            Plot scatter view showing clusters with convex hull boundaries.
            Similar to a 2D cluster visualization where we show:
            - Data points colored by cluster
            - Convex hull boundaries around each cluster
            - Violations shown outside all boundaries
            
            Args:
                var_idx: Variable index
                level: Sensitivity level ('Sensitive', 'Medium', 'Large')
                method: Clustering method (None uses fitted method)
                data_source: 'Train' or 'Test'
                max_samples: Maximum samples to plot
            """
            from scipy.spatial import ConvexHull
            
            var_name = self.config.variable_names[var_idx]
            
            # Get data
            if data_source == 'Train':
                plot_data = self.train_data[:max_samples, var_idx]
                data_label = 'Training'
            else:
                plot_data = self.test_data[:max_samples, var_idx]
                data_label = 'Test'
            
            # Get or calculate boundaries
            if method is None or method == 'Current (from fit)':
                boundaries = self.boundary_calc.get_boundaries(var_idx, level)
                method_used = self.boundary_calc.cluster_info[var_idx][level]['method']
            else:
                temp_method = self.config.clustering_method
                self.config.clustering_method = method
                
                if level == 'Sensitive':
                    boundaries = self.boundary_calc._calculate_sensitive_boundaries(plot_data, var_idx)
                elif level == 'Medium':
                    boundaries = self.boundary_calc._calculate_medium_boundaries(plot_data, var_idx)
                else:
                    boundaries = self.boundary_calc._calculate_large_boundaries(plot_data, var_idx)
                
                method_used = method
                self.config.clustering_method = temp_method
            
            # Create time indices for x-axis
            time_indices = np.arange(len(plot_data))
            
            # Assign each point to nearest cluster
            cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(boundaries)))
            cluster_assignments = np.full(len(plot_data), -1, dtype=int)
            
            for i, value in enumerate(plot_data):
                for cluster_idx, bound in enumerate(boundaries):
                    if bound['lower'] <= value <= bound['upper']:
                        cluster_assignments[i] = cluster_idx
                        break
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Plot each cluster with convex hull
            for cluster_idx in range(len(boundaries)):
                cluster_mask = cluster_assignments == cluster_idx
                if np.any(cluster_mask):
                    cluster_points = np.column_stack([
                        time_indices[cluster_mask],
                        plot_data[cluster_mask]
                    ])
                    
                    color = cluster_colors[cluster_idx]
                    
                    # Plot points
                    ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                            c=[color], alpha=0.6, s=60, edgecolors='black', linewidths=0.5,
                            label=f'Cluster {cluster_idx+1} ({np.sum(cluster_mask)} points)',
                            marker='o')
                    
                    # Draw convex hull if enough points
                    if len(cluster_points) >= 3:
                        try:
                            hull = ConvexHull(cluster_points)
                            # Plot the convex hull
                            for simplex in hull.simplices:
                                ax.plot(cluster_points[simplex, 0], cluster_points[simplex, 1],
                                    color=color, linewidth=3, alpha=0.8)
                            
                            # Fill the hull
                            hull_points = cluster_points[hull.vertices]
                            ax.fill(hull_points[:, 0], hull_points[:, 1],
                                color=color, alpha=0.15)
                        except:
                            # If convex hull fails, draw a rectangle
                            x_min, x_max = cluster_points[:, 0].min(), cluster_points[:, 0].max()
                            y_min, y_max = cluster_points[:, 1].min(), cluster_points[:, 1].max()
                            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                                fill=True, facecolor=color, alpha=0.15,
                                                edgecolor=color, linewidth=3)
                            ax.add_patch(rect)
                    
                    # Mark cluster center
                    bound = boundaries[cluster_idx]
                    center_time = np.mean(time_indices[cluster_mask])
                    ax.scatter([center_time], [bound['center']], 
                            c=[color], s=200, marker='*', edgecolors='black', 
                            linewidths=2, zorder=10)
            
            # Plot violations
            violation_mask = cluster_assignments == -1
            if np.any(violation_mask):
                ax.scatter(time_indices[violation_mask], plot_data[violation_mask],
                        c='red', marker='X', s=100, alpha=0.8,
                        label=f'Violations ({np.sum(violation_mask)} points)',
                        edgecolors='darkred', linewidths=1, zorder=5)
            
            # Labels and styling
            ax.set_xlabel('Time Step', fontsize=14, fontweight='bold')
            ax.set_ylabel(f'{var_name} Value', fontsize=14, fontweight='bold')
            ax.set_title(f'Cluster Scatter View: {var_name} (Variable {var_idx})\n' +
                        f'{level} Level | Method: {method_used} | {len(boundaries)} Clusters',
                        fontsize=15, fontweight='bold', pad=20)
            ax.legend(loc='best', fontsize=11, framealpha=0.9, edgecolor='black')
            ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
            
            plt.tight_layout()
            plt.show()
            
            # Print summary
            print(f"\n{'='*70}")
            print(f"CLUSTER ANALYSIS SUMMARY: {var_name}")
            print(f"{'='*70}")
            print(f"Sensitivity Level: {level}")
            print(f"Clustering Method: {method_used}")
            print(f"Total Data Points: {len(plot_data)}")
            print(f"\nCluster Breakdown:")
            for cluster_idx in range(len(boundaries)):
                cluster_mask = cluster_assignments == cluster_idx
                bound = boundaries[cluster_idx]
                print(f"\n  Cluster {cluster_idx+1}:")
                print(f"    Center: {bound['center']:.4f}")
                print(f"    Range: [{bound['lower']:.4f}, {bound['upper']:.4f}]")
                print(f"    Width: {bound['width_data_units']:.4f}")
                print(f"    Points: {np.sum(cluster_mask)} ({np.sum(cluster_mask)/len(plot_data)*100:.1f}%)")
            
            print(f"\n  Violations (outside all boundaries):")
            print(f"    Points: {np.sum(violation_mask)} ({np.sum(violation_mask)/len(plot_data)*100:.1f}%)")
            print(f"{'='*70}\n")



    def plot_cluster_1d_view(self, var_idx: int, level: str = 'Sensitive',
                         method: str = None, data_source: str = 'Train',
                         max_samples: int = 500):
        """
        Three alternative 1D visualizations of clustering:
        1. Flat line (y=0) with cluster colors
        2. Jittered for visibility
        3. Y-axis = cluster ID
        
        Args:
            var_idx: Variable index
            level: Sensitivity level ('Sensitive', 'Medium', 'Large')
            method: Clustering method (None uses fitted method)
            data_source: 'Train' or 'Test'
            max_samples: Maximum samples to plot
        """
        var_name = self.config.variable_names[var_idx]
        
        # Get data
        if data_source == 'Train':
            plot_data = self.train_data[:max_samples, var_idx]
            data_label = 'Training'
        else:
            plot_data = self.test_data[:max_samples, var_idx]
            data_label = 'Test'
        
        # Get or calculate boundaries
        if method is None or method == 'Current (from fit)':
            boundaries = self.boundary_calc.get_boundaries(var_idx, level)
            method_used = self.boundary_calc.cluster_info[var_idx][level]['method']
        else:
            temp_method = self.config.clustering_method
            self.config.clustering_method = method
            
            if level == 'Sensitive':
                boundaries = self.boundary_calc._calculate_sensitive_boundaries(plot_data, var_idx)
            elif level == 'Medium':
                boundaries = self.boundary_calc._calculate_medium_boundaries(plot_data, var_idx)
            else:
                boundaries = self.boundary_calc._calculate_large_boundaries(plot_data, var_idx)
            
            method_used = method
            self.config.clustering_method = temp_method
        
        # Assign each point to nearest cluster
        cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(boundaries)))
        cluster_assignments = np.full(len(plot_data), -1, dtype=int)
        
        for i, value in enumerate(plot_data):
            for cluster_idx, bound in enumerate(boundaries):
                if bound['lower'] <= value <= bound['upper']:
                    cluster_assignments[i] = cluster_idx
                    break
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        
        violation_mask = cluster_assignments == -1
        
        # ============================================================
        # METHOD 1: Flat line at y=0 with cluster colors
        # ============================================================
        ax1 = axes[0]
        
        # Plot each cluster
        for cluster_idx in range(len(boundaries)):
            cluster_mask = cluster_assignments == cluster_idx
            if np.any(cluster_mask):
                ax1.scatter(plot_data[cluster_mask], 
                           np.zeros(np.sum(cluster_mask)),
                           c=[cluster_colors[cluster_idx]], alpha=0.7, s=60,
                           edgecolors='black', linewidths=0.5,
                           label=f'Cluster {cluster_idx+1}')
        
        # Plot violations
        if np.any(violation_mask):
            ax1.scatter(plot_data[violation_mask],
                       np.zeros(np.sum(violation_mask)),
                       c='red', marker='X', s=100, alpha=0.9,
                       label='Violations', edgecolors='darkred', linewidths=1)
        
        # Mark cluster centers with vertical lines and X markers
        for cluster_idx, bound in enumerate(boundaries):
            color = cluster_colors[cluster_idx]
            ax1.axvline(bound['center'], color=color, linestyle='--', 
                       linewidth=2, alpha=0.6)
            ax1.scatter([bound['center']], [0], c=[color], marker='X', 
                       s=300, edgecolors='black', linewidths=2, zorder=10)
            
            # Draw boundary ranges as horizontal bars
            ax1.plot([bound['lower'], bound['upper']], [0.05, 0.05],
                    color=color, linewidth=4, alpha=0.7, solid_capstyle='round')
        
        ax1.set_ylim(-0.2, 0.3)
        ax1.set_ylabel('Flat View', fontsize=12, fontweight='bold')
        ax1.set_title(f'Method 1: All Points on Line (y=0) - Colored by Cluster\n' +
                     f'X markers show cluster centers, horizontal bars show boundary ranges',
                     fontsize=11, fontweight='bold', pad=10)
        ax1.legend(loc='upper right', fontsize=9, ncol=2)
        ax1.grid(alpha=0.3, linestyle=':', linewidth=0.5)
        ax1.set_yticks([])
        
        # ============================================================
        # METHOD 2: Jittered for visibility
        # ============================================================
        ax2 = axes[1]
        
        # Add random jitter to y-axis
        np.random.seed(42)  # For reproducibility
        jitter = np.random.normal(0, 0.08, len(plot_data))
        
        # Plot each cluster with jitter
        for cluster_idx in range(len(boundaries)):
            cluster_mask = cluster_assignments == cluster_idx
            if np.any(cluster_mask):
                ax2.scatter(plot_data[cluster_mask], 
                           jitter[cluster_mask],
                           c=[cluster_colors[cluster_idx]], alpha=0.6, s=50,
                           edgecolors='black', linewidths=0.4,
                           label=f'Cluster {cluster_idx+1}')
        
        # Plot violations with jitter
        if np.any(violation_mask):
            ax2.scatter(plot_data[violation_mask],
                       jitter[violation_mask],
                       c='red', marker='X', s=90, alpha=0.9,
                       label='Violations', edgecolors='darkred', linewidths=1)
        
        # Mark cluster centers
        for cluster_idx, bound in enumerate(boundaries):
            color = cluster_colors[cluster_idx]
            ax2.axvline(bound['center'], color=color, linestyle='--', 
                       linewidth=2, alpha=0.6)
            ax2.scatter([bound['center']], [0], c=[color], marker='X', 
                       s=300, edgecolors='black', linewidths=2, zorder=10)
            
            # Draw boundary ranges
            ax2.plot([bound['lower'], bound['upper']], [-0.3, -0.3],
                    color=color, linewidth=4, alpha=0.7, solid_capstyle='round')
        
        ax2.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.3)
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_ylabel('Jittered View', fontsize=12, fontweight='bold')
        ax2.set_title(f'Method 2: Random Vertical Jitter - Better Visibility of Overlapping Points',
                     fontsize=11, fontweight='bold', pad=10)
        ax2.legend(loc='upper right', fontsize=9, ncol=2)
        ax2.grid(alpha=0.3, linestyle=':', linewidth=0.5)
        ax2.set_yticks([])
        
        # ============================================================
        # METHOD 3: Y-axis shows cluster membership
        # ============================================================
        ax3 = axes[2]
        
        # Plot each cluster at its own y-level
        for cluster_idx in range(len(boundaries)):
            cluster_mask = cluster_assignments == cluster_idx
            if np.any(cluster_mask):
                # Add small random jitter within cluster level for visibility
                y_jitter = np.random.normal(cluster_idx, 0.05, np.sum(cluster_mask))
                ax3.scatter(plot_data[cluster_mask], 
                           y_jitter,
                           c=[cluster_colors[cluster_idx]], alpha=0.7, s=60,
                           edgecolors='black', linewidths=0.5,
                           label=f'Cluster {cluster_idx+1}')
        
        # Plot violations at y=-1
        if np.any(violation_mask):
            y_jitter_viol = np.random.normal(-1, 0.05, np.sum(violation_mask))
            ax3.scatter(plot_data[violation_mask],
                       y_jitter_viol,
                       c='red', marker='X', s=100, alpha=0.9,
                       label='Violations', edgecolors='darkred', linewidths=1)
        
        # Mark cluster centers with vertical lines
        for cluster_idx, bound in enumerate(boundaries):
            color = cluster_colors[cluster_idx]
            ax3.axvline(bound['center'], color=color, linestyle='--', 
                       linewidth=2, alpha=0.5)
            ax3.scatter([bound['center']], [cluster_idx], c=[color], marker='X', 
                       s=300, edgecolors='black', linewidths=2, zorder=10)
            
            # Draw boundary ranges as horizontal bars at cluster level
            ax3.plot([bound['lower'], bound['upper']], 
                    [cluster_idx + 0.3, cluster_idx + 0.3],
                    color=color, linewidth=5, alpha=0.7, solid_capstyle='round')
        
        # Draw horizontal lines separating clusters
        for cluster_idx in range(len(boundaries) - 1):
            ax3.axhline(cluster_idx + 0.5, color='gray', linestyle=':', 
                       linewidth=1, alpha=0.5)
        
        # Set y-ticks
        y_labels = [f'Cluster {i+1}' for i in range(len(boundaries))]
        if np.any(violation_mask):
            y_labels = ['Violations'] + y_labels
            ax3.set_ylim(-1.5, len(boundaries) - 0.5)
            ax3.set_yticks(list(range(-1, len(boundaries))))
        else:
            ax3.set_ylim(-0.5, len(boundaries) - 0.5)
            ax3.set_yticks(list(range(len(boundaries))))
        
        ax3.set_yticklabels(y_labels, fontsize=10, fontweight='bold')
        ax3.set_ylabel('Cluster Membership', fontsize=12, fontweight='bold')
        ax3.set_xlabel(f'{var_name} Value', fontsize=13, fontweight='bold')
        ax3.set_title(f'Method 3: Y-axis Shows Cluster ID - Clear Group Membership',
                     fontsize=11, fontweight='bold', pad=10)
        ax3.legend(loc='upper right', fontsize=9, ncol=2)
        ax3.grid(alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Overall title
        fig.suptitle(f'1D Cluster Visualizations: {var_name} (Variable {var_idx})\n' +
                    f'{level} Level | Method: {method_used} | {len(boundaries)} Clusters | ' +
                    f'Violations: {np.sum(violation_mask)} ({np.sum(violation_mask)/len(plot_data)*100:.1f}%)',
                    fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"1D CLUSTER VISUALIZATION SUMMARY")
        print(f"{'='*70}")
        print(f"Variable: {var_name} (Index {var_idx})")
        print(f"Level: {level} | Method: {method_used}")
        print(f"Data source: {data_source} | Samples: {len(plot_data)}")
        print(f"\nCluster Centers and Boundaries:")
        for cluster_idx, bound in enumerate(boundaries):
            cluster_mask = cluster_assignments == cluster_idx
            print(f"  Cluster {cluster_idx+1}:")
            print(f"    Center: {bound['center']:.4f}")
            print(f"    Range: [{bound['lower']:.4f}, {bound['upper']:.4f}]")
            print(f"    Points: {np.sum(cluster_mask)} ({np.sum(cluster_mask)/len(plot_data)*100:.1f}%)")
        print(f"\n  Violations: {np.sum(violation_mask)} ({np.sum(violation_mask)/len(plot_data)*100:.1f}%)")
        print(f"{'='*70}\n")
