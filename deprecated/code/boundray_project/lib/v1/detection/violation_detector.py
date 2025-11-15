import numpy as np
from typing import List, Dict


class ViolationDetector:
    """
    Detect boundary violations with multiple boundaries per sensitivity level.

    Violation occurs when datapoint is OUTSIDE ALL N boundaries.
    Supports:
    - Binary mode: 0 = inside at least one boundary, 1 = outside all boundaries
    - Ternary mode: -1 = below all boundaries, 0 = inside at least one, +1 = above all boundaries
    """

    def __init__(self, boundary_calc):
        self.boundary_calc = boundary_calc
        self.config = boundary_calc.config
        self.violations = {}

    def _is_outside_all_boundaries(self, value: float, boundaries: List[Dict[str, float]]) -> bool:
        """
        Check if value is outside ALL N boundaries.

        Args:
            value: Data value to check
            boundaries: List of boundary dictionaries

        Returns:
            True if outside all boundaries, False if inside at least one
        """
        for bound in boundaries:
            # If inside any boundary, return False
            if bound['lower'] <= value <= bound['upper']:
                return False
        # Outside all boundaries
        return True

    def detect(self, data: np.ndarray, level: str, mode: str = 'ternary') -> np.ndarray:
        """
        Detect boundary violations with multiple boundaries.

        Args:
            data: Data to analyze (n_samples, n_variables)
            level: Sensitivity level ('Sensitive', 'Medium', 'Large')
            mode: 'binary' (0/1) or 'ternary' (-1/0/+1)

        Returns:
            Violation matrix (n_samples, n_variables)
        """
        n_samples, n_vars = data.shape
        violations = np.zeros((n_samples, n_vars), dtype=int)

        for var_idx in range(n_vars):
            var_data = data[:, var_idx]
            boundaries = self.boundary_calc.get_boundaries(var_idx, level)
            var_mean = self.boundary_calc.data_stats[var_idx]['mean']

            for sample_idx in range(n_samples):
                value = var_data[sample_idx]

                # Check if outside all boundaries
                if self._is_outside_all_boundaries(value, boundaries):
                    if mode == 'ternary':
                        # TODO: Research enhancement - Experiment with different violation classification logic:
                        # - Consider using median instead of mean for more robust classification
                        # - Try distance-based scoring (how far from nearest boundary)
                        # - Experiment with cluster-specific classification (which cluster is closest?)
                        # - Consider rate of change: sudden jumps vs gradual drift
                        # - Add confidence scoring based on distance from ALL boundaries
                        # - For co-occurrence analysis: consider directional relationships (both above/below)

                        # Current logic: Compare to mean
                        if value > var_mean:
                            violations[sample_idx, var_idx] = 1  # Above all boundaries
                        else:
                            violations[sample_idx, var_idx] = -1  # Below all boundaries
                    else:  # binary mode
                        violations[sample_idx, var_idx] = 1  # Outside all boundaries
                # else: stays 0 (inside at least one boundary)

        return violations

    def detect_all_levels(self, data: np.ndarray, mode: str = 'ternary') -> Dict[str, np.ndarray]:
        """
        Detect violations at all sensitivity levels.

        Returns:
            Dictionary mapping level name to violation matrix
        """
        self.violations = {}
        for level in self.config.sensitivity_levels.keys():
            self.violations[level] = self.detect(data, level, mode)

        return self.violations

    def summary(self, data: np.ndarray, mode: str = 'ternary'):
        """Print summary of violations detected"""
        n_samples, n_vars = data.shape

        print(f"\nViolation Summary ({mode} mode) for {n_samples} samples, {n_vars} variables:")
        print(f"Violation logic: Outside ALL N boundaries")
        print("=" * 80)

        for level, violations in self.violations.items():
            n_boundaries = self.config.get_n_boundaries(level)
            print(f"\n{level} Level ({n_boundaries} boundaries per variable):")

            if mode == 'ternary':
                below_violations = np.sum(violations == -1)
                above_violations = np.sum(violations == 1)
                total_violations = below_violations + above_violations

                violation_rate = total_violations / (n_samples * n_vars) * 100

                print(f"  Below all boundaries: {below_violations} ({below_violations/(n_samples*n_vars)*100:.2f}%)")
                print(f"  Above all boundaries: {above_violations} ({above_violations/(n_samples*n_vars)*100:.2f}%)")
                print(f"  Total violations: {total_violations} ({violation_rate:.2f}%)")
            else:  # binary
                total_violations = np.sum(violations)
                violation_rate = total_violations / (n_samples * n_vars) * 100

                print(f"  Total violations: {total_violations} ({violation_rate:.2f}%)")

            # Count variables with violations
            vars_with_violations = np.sum(np.any(violations != 0, axis=0))
            print(f"  Variables with violations: {vars_with_violations}/{n_vars}")
            print(f"  Avg violations per variable: {np.sum(np.abs(violations))/n_vars:.1f}")
