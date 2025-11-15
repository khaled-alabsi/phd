import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from ipywidgets import interact, widgets
from matplotlib.colors import ListedColormap


class ExcursionVisualizer:
    """Interactive visualization for excursions/violations with multiple boundaries"""

    def __init__(self, violation_detector, data: np.ndarray):
        self.violation_detector = violation_detector
        self.config = violation_detector.config
        self.boundary_calc = violation_detector.boundary_calc
        self.data = data

    def plot_excursion_timeline(self, var_indices: List[int], level: str = 'Sensitive',
                                max_samples: int = 500):
        """
        Plot excursion timeline for selected variables with ALL N boundaries shown

        Args:
            var_indices: List of variable indices to plot
            level: Sensitivity level
            max_samples: Maximum samples to display
        """
        # Get violations for this level
        violations = self.violation_detector.violations[level][:max_samples, var_indices]
        n_samples, n_selected_vars = violations.shape

        # Create figure with subplots
        fig, axes_raw = plt.subplots(n_selected_vars, 1,
                                figsize=(16, 2.5 * n_selected_vars),
                                sharex=True)

        # Ensure axes is always a list
        axes = [axes_raw] if n_selected_vars == 1 else axes_raw

        time_indices = np.arange(n_samples)

        for idx, (ax, var_idx) in enumerate(zip(axes, var_indices)):
            var_name = self.config.variable_names[var_idx]
            var_violations = violations[:, idx]
            var_data = self.data[:max_samples, var_idx]

            # Plot the actual values
            ax.plot(time_indices, var_data, 'k-', linewidth=0.8, alpha=0.5,
                   label='Value')

            # Get ALL boundaries for this level (list of N boundaries)
            boundaries_list = self.boundary_calc.get_boundaries(var_idx, level)

            # Plot ALL N boundaries
            boundary_colors = plt.cm.Set3(np.linspace(0, 1, len(boundaries_list)))  # type: ignore

            for boundary_idx, bounds in enumerate(boundaries_list):
                color = boundary_colors[boundary_idx]

                # Plot boundaries
                ax.axhline(bounds['upper'], color=color, linestyle='--', linewidth=1.5,
                          alpha=0.7, label=f'Boundary {boundary_idx+1} Upper')
                ax.axhline(bounds['lower'], color=color, linestyle='--', linewidth=1.5,
                          alpha=0.7, label=f'Boundary {boundary_idx+1} Lower')

                # Shade acceptable region
                ax.axhspan(bounds['lower'], bounds['upper'], alpha=0.05, color=color)

            # Highlight violations (outside ALL boundaries)
            below_mask = var_violations == -1
            above_mask = var_violations == 1

            # Highlight violations
            if np.any(below_mask):
                ax.fill_between(time_indices, var_data.min(), var_data.max(),
                               where=below_mask, alpha=0.2, color='blue',
                               label='Below ALL Boundaries')

            if np.any(above_mask):
                ax.fill_between(time_indices, var_data.min(), var_data.max(),
                               where=above_mask, alpha=0.2, color='orange',
                               label='Above ALL Boundaries')

            # Title and labels
            n_violations = np.sum(var_violations != 0)
            violation_pct = (n_violations / n_samples) * 100

            ax.set_ylabel(f'{var_name}\n(violations: {n_violations})',
                         fontsize=10, fontweight='bold')
            ax.set_title(f'Variable {var_idx}: {var_name} - {level} Level ' +
                        f'({len(boundaries_list)} boundaries, {violation_pct:.1f}% violations)',
                        fontsize=11, fontweight='bold')
            ax.legend(loc='upper right', fontsize=7, ncol=3)
            ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)

        axes[-1].set_xlabel('Time Step', fontsize=12, fontweight='bold')
        fig.suptitle(f'Excursion Timeline - Violations occur when OUTSIDE ALL {len(boundaries_list)} boundaries',
                    fontsize=13, fontweight='bold', y=0.995)

        plt.tight_layout()
        plt.show()

    def plot_multi_variable_heatmap(self, var_indices: List[int] = None,
                                   level: str = 'Sensitive', max_samples: int = 500):
        """
        Plot heatmap of violations for multiple variables

        Args:
            var_indices: List of variable indices (None = all)
            level: Sensitivity level
            max_samples: Maximum samples to display
        """
        if var_indices is None:
            var_indices = list(range(len(self.config.variable_names)))

        # Get violations
        violations = self.violation_detector.violations[level][:max_samples, :][:, var_indices]

        # Create figure
        fig, ax = plt.subplots(figsize=(16, max(6, len(var_indices) * 0.3)))

        # Create custom colormap: blue (-1), white (0), orange (+1)
        cmap = ListedColormap(['blue', 'lightgreen', 'orange'])

        # Plot heatmap
        im = ax.imshow(violations.T, aspect='auto', cmap=cmap,
                      interpolation='nearest', vmin=-1, vmax=1)

        # Labels
        ax.set_xlabel('Time Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Variable Index', fontsize=12, fontweight='bold')
        n_boundaries = self.config.get_n_boundaries(level)
        ax.set_title(f'Violation Heatmap - {level} Level ({n_boundaries} boundaries per variable)\n' +
                    '(Blue = Below ALL, Green = Normal, Orange = Above ALL)',
                    fontsize=14, fontweight='bold')

        # Y-axis labels
        ax.set_yticks(range(len(var_indices)))
        ax.set_yticklabels([self.config.variable_names[i] for i in var_indices],
                          fontsize=8)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1])
        cbar.set_label('Violation Type', fontsize=10, fontweight='bold')
        cbar.ax.set_yticklabels(['Below ALL', 'Normal', 'Above ALL'])

        plt.tight_layout()
        plt.show()

    def create_interactive_excursion_viewer(self):
        """Create interactive widget for viewing excursions"""

        def update_plot(var_indices_str, level, plot_type, max_samples):
            # Parse variable indices
            try:
                if ',' in var_indices_str:
                    var_indices = [int(x.strip()) for x in var_indices_str.split(',')]
                elif '-' in var_indices_str:
                    start, end = var_indices_str.split('-')
                    var_indices = list(range(int(start.strip()), int(end.strip()) + 1))
                else:
                    var_indices = [int(var_indices_str.strip())]

                # Validate indices
                var_indices = [i for i in var_indices if 0 <= i < len(self.config.variable_names)]

                if not var_indices:
                    print("No valid variable indices provided.")
                    return

                if plot_type == 'Timeline':
                    self.plot_excursion_timeline(var_indices, level, max_samples)
                else:  # Heatmap
                    self.plot_multi_variable_heatmap(var_indices, level, max_samples)

            except Exception as e:
                print(f"Error: {e}")
                print("Please enter valid variable indices (e.g., '0,1,2' or '0-5')")

        # Create widgets
        var_input = widgets.Text(
            value='0,1,2,3,4',
            description='Variables:',
            placeholder='Enter indices: 0,1,2 or 0-5',
            style={'description_width': 'initial'}
        )

        level_selector = widgets.Dropdown(
            options=['Sensitive', 'Medium', 'Large'],
            value='Sensitive',
            description='Level:',
            style={'description_width': 'initial'}
        )

        plot_type_selector = widgets.Dropdown(
            options=['Timeline', 'Heatmap'],
            value='Timeline',
            description='Plot Type:',
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
                var_indices_str=var_input,
                level=level_selector,
                plot_type=plot_type_selector,
                max_samples=samples_slider)
