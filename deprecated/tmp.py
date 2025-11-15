
"""
Comprehensive EWMA-based Anomaly Detection for Tennessee Eastman Process Data
This script implements various types of EWMA charts for anomaly detection including:
1. Univariate EWMA
2. Multivariate EWMA (MEWMA) 
3. EWMA with variance control
4. Adaptive EWMA
5. Hybrid EWMA charts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2, multivariate_normal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance
import warnings
warnings.filterwarnings('ignore')

class TEP_EWMA_AnomalyDetector:
    """
    Comprehensive EWMA-based anomaly detection for Tennessee Eastman Process data
    """

    def __init__(self, lambda_param=0.2, alpha=0.05):
        """
        Initialize the EWMA anomaly detector

        Parameters:
        -----------
        lambda_param : float
            EWMA smoothing parameter (0 < lambda <= 1)
        alpha : float
            Significance level for control limits
        """
        self.lambda_param = lambda_param
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.ewma_history = []
        self.is_fitted = False

    def load_tep_data(self, file_path=None, fault_number=0):
        """
        Load Tennessee Eastman Process data from Kaggle

        Parameters:
        -----------
        file_path : str
            Path to the TEP dataset CSV file
        fault_number : int
            Fault number to filter (0 for normal operation)
        """
        if file_path is None:
            # Create sample TEP-like data for demonstration
            print("Creating sample TEP-like data for demonstration...")
            np.random.seed(42)
            n_samples = 1000
            n_features = 52  # TEP has 52 process variables

            # Generate normal operation data
            data = np.random.multivariate_normal(
                mean=np.zeros(n_features),
                cov=np.eye(n_features),
                size=n_samples
            )

            # Add some process-like trends and correlations
            for i in range(n_features):
                data[:, i] += np.sin(np.linspace(0, 4*np.pi, n_samples)) * 0.5
                if i > 0:
                    data[:, i] += 0.3 * data[:, i-1]  # Add some correlation

            # Create column names like TEP data
            columns = [f'xmeas_{i+1}' for i in range(41)] + [f'xmv_{i+1}' for i in range(11)]

            self.data = pd.DataFrame(data, columns=columns)

            # Add anomalies to last 100 samples
            anomaly_start = n_samples - 100
            self.data.iloc[anomaly_start:, :10] += np.random.normal(0, 3, (100, 10))

            print(f"Sample data created with shape: {self.data.shape}")

        else:
            # Load actual TEP data
            try:
                self.data = pd.read_csv(file_path)
                if 'faultNumber' in self.data.columns:
                    self.data = self.data[self.data['faultNumber'] == fault_number]

                # Select only process variables (xmeas and xmv)
                process_cols = [col for col in self.data.columns if col.startswith(('xmeas', 'xmv'))]
                self.data = self.data[process_cols]

                print(f"TEP data loaded with shape: {self.data.shape}")

            except Exception as e:
                print(f"Error loading data: {e}")
                print("Using sample data instead...")
                return self.load_tep_data()

        return self.data

    def preprocess_data(self, normalize=True):
        """
        Preprocess the TEP data for EWMA analysis

        Parameters:
        -----------
        normalize : bool
            Whether to normalize the data
        """
        # Handle missing values
        self.data = self.data.fillna(self.data.mean())

        if normalize:
            self.data_normalized = pd.DataFrame(
                self.scaler.fit_transform(self.data),
                columns=self.data.columns,
                index=self.data.index
            )
        else:
            self.data_normalized = self.data.copy()

        print(f"Data preprocessed. Shape: {self.data_normalized.shape}")
        return self.data_normalized

    def univariate_ewma(self, column_name, plot=True):
        """
        Implement univariate EWMA control chart

        Parameters:
        -----------
        column_name : str
            Name of the column to analyze
        plot : bool
            Whether to plot the results
        """
        if column_name not in self.data_normalized.columns:
            raise ValueError(f"Column {column_name} not found in data")

        series = self.data_normalized[column_name].values
        n = len(series)

        # Calculate EWMA statistics
        ewma = np.zeros(n)
        ewma[0] = series[0]

        for i in range(1, n):
            ewma[i] = self.lambda_param * series[i] + (1 - self.lambda_param) * ewma[i-1]

        # Calculate control limits
        sigma_hat = np.std(series)
        L = 3  # Control limit multiplier

        # Time-varying control limits
        ucl = np.zeros(n)
        lcl = np.zeros(n)

        for i in range(n):
            var_ewma = (self.lambda_param / (2 - self.lambda_param)) * \
                       (1 - (1 - self.lambda_param)**(2*(i+1))) * sigma_hat**2
            limit = L * np.sqrt(var_ewma)
            ucl[i] = limit
            lcl[i] = -limit

        # Detect anomalies
        anomalies = (ewma > ucl) | (ewma < lcl)

        results = {
            'ewma': ewma,
            'ucl': ucl,
            'lcl': lcl,
            'anomalies': anomalies,
            'anomaly_indices': np.where(anomalies)[0]
        }

        if plot:
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 1, 1)
            plt.plot(series, label='Original Data', alpha=0.7)
            plt.plot(ewma, label='EWMA', color='blue', linewidth=2)
            plt.fill_between(range(n), lcl, ucl, alpha=0.2, color='red', label='Control Limits')
            plt.plot(ucl, 'r--', label='UCL')
            plt.plot(lcl, 'r--', label='LCL')
            plt.scatter(np.where(anomalies)[0], ewma[anomalies], color='red', s=50, label='Anomalies')
            plt.title(f'Univariate EWMA Chart - {column_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 1, 2)
            plt.plot(series, label='Original Data')
            plt.scatter(np.where(anomalies)[0], series[anomalies], color='red', s=50, label='Detected Anomalies')
            plt.title('Original Data with Detected Anomalies')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        return results

    def multivariate_ewma(self, selected_columns=None, plot=True):
        """
        Implement Multivariate EWMA (MEWMA) control chart

        Parameters:
        -----------
        selected_columns : list
            List of columns to include in multivariate analysis
        plot : bool
            Whether to plot the results
        """
        if selected_columns is None:
            # Select first 10 columns for demonstration
            selected_columns = self.data_normalized.columns[:10]

        data_subset = self.data_normalized[selected_columns].values
        n, p = data_subset.shape

        # Calculate sample covariance matrix
        cov_matrix = np.cov(data_subset.T)
        mean_vector = np.mean(data_subset, axis=0)

        # Initialize MEWMA vectors
        mewma = np.zeros((n, p))
        mewma[0] = data_subset[0]

        # Calculate MEWMA vectors
        for i in range(1, n):
            mewma[i] = self.lambda_param * data_subset[i] + \
                       (1 - self.lambda_param) * mewma[i-1]

        # Calculate T² statistics
        t2_stats = np.zeros(n)

        for i in range(n):
            # Covariance matrix of MEWMA at time i
            H = (self.lambda_param / (2 - self.lambda_param)) * \
                (1 - (1 - self.lambda_param)**(2*(i+1)))
            cov_mewma = H * cov_matrix

            try:
                # T² statistic
                diff = mewma[i] - mean_vector
                t2_stats[i] = diff @ np.linalg.pinv(cov_mewma) @ diff
            except:
                t2_stats[i] = 0

        # Control limit for T² statistic
        h = 3  # Control limit parameter
        ucl_t2 = h**2

        # Detect anomalies
        anomalies = t2_stats > ucl_t2

        results = {
            'mewma': mewma,
            't2_stats': t2_stats,
            'ucl_t2': ucl_t2,
            'anomalies': anomalies,
            'anomaly_indices': np.where(anomalies)[0],
            'selected_columns': selected_columns
        }

        if plot:
            plt.figure(figsize=(15, 10))

            # Plot T² statistics
            plt.subplot(2, 2, 1)
            plt.plot(t2_stats, label='T² Statistics', color='blue')
            plt.axhline(y=ucl_t2, color='red', linestyle='--', label=f'UCL = {ucl_t2:.2f}')
            plt.scatter(np.where(anomalies)[0], t2_stats[anomalies], color='red', s=50, label='Anomalies')
            plt.title('MEWMA T² Control Chart')
            plt.xlabel('Sample Number')
            plt.ylabel('T² Statistic')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Plot first few MEWMA components
            plt.subplot(2, 2, 2)
            for i in range(min(3, len(selected_columns))):
                plt.plot(mewma[:, i], label=f'{selected_columns[i]}')
            plt.title('MEWMA Components (First 3 variables)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Plot original data with anomalies highlighted
            plt.subplot(2, 2, 3)
            for i in range(min(3, len(selected_columns))):
                plt.plot(data_subset[:, i], alpha=0.7, label=f'{selected_columns[i]}')

            # Highlight anomaly regions
            for idx in np.where(anomalies)[0]:
                plt.axvline(x=idx, color='red', alpha=0.5, linestyle=':')

            plt.title('Original Data with Anomaly Markers')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Anomaly distribution
            plt.subplot(2, 2, 4)
            anomaly_counts = np.bincount(np.where(anomalies)[0] // 50)  # Group by 50-sample windows
            plt.bar(range(len(anomaly_counts)), anomaly_counts)
            plt.title('Anomaly Distribution Over Time')
            plt.xlabel('Time Window (50 samples each)')
            plt.ylabel('Number of Anomalies')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        return results

    def ewma_variance_chart(self, column_name, plot=True):
        """
        Implement EWMA chart for monitoring variance

        Parameters:
        -----------
        column_name : str
            Name of the column to analyze
        plot : bool
            Whether to plot the results
        """
        series = self.data_normalized[column_name].values
        n = len(series)

        # Calculate moving range (for variance estimation)
        moving_range = np.abs(np.diff(series))

        # EWMA for variance monitoring
        ewma_var = np.zeros(n-1)
        ewma_var[0] = moving_range[0]

        for i in range(1, n-1):
            ewma_var[i] = self.lambda_param * moving_range[i] + \
                         (1 - self.lambda_param) * ewma_var[i-1]

        # Control limits for variance chart
        mr_bar = np.mean(moving_range)
        sigma_mr = mr_bar / 1.128  # d2 constant for n=2

        ucl_var = mr_bar + 3 * sigma_mr * np.sqrt(self.lambda_param / (2 - self.lambda_param))
        lcl_var = max(0, mr_bar - 3 * sigma_mr * np.sqrt(self.lambda_param / (2 - self.lambda_param)))

        # Detect variance anomalies
        var_anomalies = (ewma_var > ucl_var) | (ewma_var < lcl_var)

        results = {
            'ewma_var': ewma_var,
            'moving_range': moving_range,
            'ucl_var': ucl_var,
            'lcl_var': lcl_var,
            'var_anomalies': var_anomalies,
            'var_anomaly_indices': np.where(var_anomalies)[0]
        }

        if plot:
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 1, 1)
            plt.plot(moving_range, label='Moving Range', alpha=0.7)
            plt.plot(ewma_var, label='EWMA Variance', color='blue', linewidth=2)
            plt.axhline(y=ucl_var, color='red', linestyle='--', label=f'UCL = {ucl_var:.2f}')
            plt.axhline(y=lcl_var, color='red', linestyle='--', label=f'LCL = {lcl_var:.2f}')
            plt.scatter(np.where(var_anomalies)[0], ewma_var[var_anomalies], 
                       color='red', s=50, label='Variance Anomalies')
            plt.title(f'EWMA Variance Chart - {column_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 1, 2)
            plt.plot(series, label='Original Data')
            # Mark variance anomaly regions
            for idx in np.where(var_anomalies)[0]:
                plt.axvline(x=idx, color='orange', alpha=0.5, linestyle=':', label='Variance Change' if idx == np.where(var_anomalies)[0][0] else "")
            plt.title('Original Data with Variance Change Points')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        return results

    def adaptive_ewma(self, column_name, plot=True):
        """
        Implement Adaptive EWMA that changes lambda based on process conditions

        Parameters:
        -----------
        column_name : str
            Name of the column to analyze
        plot : bool
            Whether to plot the results
        """
        series = self.data_normalized[column_name].values
        n = len(series)

        # Adaptive parameters
        lambda_min = 0.05
        lambda_max = 0.3
        lambda_current = self.lambda_param

        ewma_adaptive = np.zeros(n)
        lambdas_used = np.zeros(n)
        ewma_adaptive[0] = series[0]
        lambdas_used[0] = lambda_current

        for i in range(1, n):
            # Adapt lambda based on recent variance
            if i > 10:  # Need some history to calculate variance
                recent_var = np.var(series[max(0, i-10):i])
                overall_var = np.var(series[:i])

                # If recent variance is much higher, increase lambda (be more reactive)
                if recent_var > 2 * overall_var:
                    lambda_current = min(lambda_max, lambda_current * 1.1)
                elif recent_var < 0.5 * overall_var:
                    lambda_current = max(lambda_min, lambda_current * 0.9)

            ewma_adaptive[i] = lambda_current * series[i] + (1 - lambda_current) * ewma_adaptive[i-1]
            lambdas_used[i] = lambda_current

        # Calculate control limits (using average lambda)
        avg_lambda = np.mean(lambdas_used)
        sigma_hat = np.std(series)
        L = 3

        ucl_adaptive = np.zeros(n)
        lcl_adaptive = np.zeros(n)

        for i in range(n):
            var_ewma = (avg_lambda / (2 - avg_lambda)) * \
                       (1 - (1 - avg_lambda)**(2*(i+1))) * sigma_hat**2
            limit = L * np.sqrt(var_ewma)
            ucl_adaptive[i] = limit
            lcl_adaptive[i] = -limit

        # Detect anomalies
        adaptive_anomalies = (ewma_adaptive > ucl_adaptive) | (ewma_adaptive < lcl_adaptive)

        results = {
            'ewma_adaptive': ewma_adaptive,
            'lambdas_used': lambdas_used,
            'ucl_adaptive': ucl_adaptive,
            'lcl_adaptive': lcl_adaptive,
            'adaptive_anomalies': adaptive_anomalies,
            'adaptive_anomaly_indices': np.where(adaptive_anomalies)[0]
        }

        if plot:
            plt.figure(figsize=(12, 10))

            plt.subplot(3, 1, 1)
            plt.plot(lambdas_used, label='Lambda Parameter', color='green', linewidth=2)
            plt.title('Adaptive Lambda Parameter Over Time')
            plt.ylabel('Lambda Value')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(3, 1, 2)
            plt.plot(series, label='Original Data', alpha=0.7)
            plt.plot(ewma_adaptive, label='Adaptive EWMA', color='blue', linewidth=2)
            plt.fill_between(range(n), lcl_adaptive, ucl_adaptive, alpha=0.2, color='red', label='Control Limits')
            plt.plot(ucl_adaptive, 'r--', label='UCL')
            plt.plot(lcl_adaptive, 'r--', label='LCL')
            plt.scatter(np.where(adaptive_anomalies)[0], ewma_adaptive[adaptive_anomalies], 
                       color='red', s=50, label='Anomalies')
            plt.title(f'Adaptive EWMA Chart - {column_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(3, 1, 3)
            plt.plot(series, label='Original Data')
            plt.scatter(np.where(adaptive_anomalies)[0], series[adaptive_anomalies], 
                       color='red', s=50, label='Detected Anomalies')
            plt.title('Original Data with Adaptive EWMA Detected Anomalies')
            plt.xlabel('Sample Number')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        return results

    def hybrid_ewma_cusum(self, column_name, plot=True):
        """
        Implement Hybrid EWMA-CUSUM chart for enhanced anomaly detection

        Parameters:
        -----------
        column_name : str
            Name of the column to analyze
        plot : bool
            Whether to plot the results
        """
        series = self.data_normalized[column_name].values
        n = len(series)

        # EWMA component
        ewma = np.zeros(n)
        ewma[0] = series[0]

        for i in range(1, n):
            ewma[i] = self.lambda_param * series[i] + (1 - self.lambda_param) * ewma[i-1]

        # CUSUM component
        k = 0.5  # Reference value (typically 0.5 * process shift to detect)
        h = 4    # Decision interval (typically 4-5)

        cusum_pos = np.zeros(n)  # Upper CUSUM
        cusum_neg = np.zeros(n)  # Lower CUSUM

        for i in range(1, n):
            cusum_pos[i] = max(0, cusum_pos[i-1] + ewma[i] - k)
            cusum_neg[i] = max(0, cusum_neg[i-1] - ewma[i] - k)

        # Hybrid detection: anomaly if either EWMA or CUSUM triggers

        # EWMA limits
        sigma_hat = np.std(series)
        L = 3

        ucl_ewma = np.zeros(n)
        lcl_ewma = np.zeros(n)

        for i in range(n):
            var_ewma = (self.lambda_param / (2 - self.lambda_param)) * \
                       (1 - (1 - self.lambda_param)**(2*(i+1))) * sigma_hat**2
            limit = L * np.sqrt(var_ewma)
            ucl_ewma[i] = limit
            lcl_ewma[i] = -limit

        # Combined anomaly detection
        ewma_anomalies = (ewma > ucl_ewma) | (ewma < lcl_ewma)
        cusum_anomalies = (cusum_pos > h) | (cusum_neg > h)
        hybrid_anomalies = ewma_anomalies | cusum_anomalies

        results = {
            'ewma': ewma,
            'cusum_pos': cusum_pos,
            'cusum_neg': cusum_neg,
            'ucl_ewma': ucl_ewma,
            'lcl_ewma': lcl_ewma,
            'ewma_anomalies': ewma_anomalies,
            'cusum_anomalies': cusum_anomalies,
            'hybrid_anomalies': hybrid_anomalies,
            'hybrid_anomaly_indices': np.where(hybrid_anomalies)[0]
        }

        if plot:
            plt.figure(figsize=(15, 12))

            # EWMA chart
            plt.subplot(3, 1, 1)
            plt.plot(series, label='Original Data', alpha=0.7)
            plt.plot(ewma, label='EWMA', color='blue', linewidth=2)
            plt.fill_between(range(n), lcl_ewma, ucl_ewma, alpha=0.2, color='red', label='EWMA Control Limits')
            plt.plot(ucl_ewma, 'r--', alpha=0.7)
            plt.plot(lcl_ewma, 'r--', alpha=0.7)
            plt.scatter(np.where(ewma_anomalies)[0], ewma[ewma_anomalies], 
                       color='red', s=30, label='EWMA Anomalies')
            plt.title('EWMA Component of Hybrid Chart')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # CUSUM chart
            plt.subplot(3, 1, 2)
            plt.plot(cusum_pos, label='Upper CUSUM', color='green')
            plt.plot(cusum_neg, label='Lower CUSUM', color='orange')
            plt.axhline(y=h, color='red', linestyle='--', label=f'CUSUM Limit = {h}')
            plt.axhline(y=-h, color='red', linestyle='--')
            plt.scatter(np.where(cusum_anomalies)[0], 
                       np.maximum(cusum_pos[cusum_anomalies], cusum_neg[cusum_anomalies]), 
                       color='red', s=30, label='CUSUM Anomalies')
            plt.title('CUSUM Component of Hybrid Chart')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Hybrid results
            plt.subplot(3, 1, 3)
            plt.plot(series, label='Original Data', alpha=0.8)
            plt.scatter(np.where(hybrid_anomalies)[0], series[hybrid_anomalies], 
                       color='red', s=50, label='Hybrid Detected Anomalies')
            plt.scatter(np.where(ewma_anomalies & ~cusum_anomalies)[0], 
                       series[ewma_anomalies & ~cusum_anomalies], 
                       color='blue', s=30, marker='^', label='EWMA Only')
            plt.scatter(np.where(cusum_anomalies & ~ewma_anomalies)[0], 
                       series[cusum_anomalies & ~ewma_anomalies], 
                       color='green', s=30, marker='s', label='CUSUM Only')
            plt.title(f'Hybrid EWMA-CUSUM Anomaly Detection - {column_name}')
            plt.xlabel('Sample Number')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        return results

    def comprehensive_analysis(self, selected_columns=None):
        """
        Run comprehensive EWMA-based anomaly detection analysis

        Parameters:
        -----------
        selected_columns : list
            Columns to analyze (if None, analyzes first few columns)
        """
        if selected_columns is None:
            selected_columns = self.data_normalized.columns[:5]

        print("="*60)
        print("COMPREHENSIVE EWMA ANOMALY DETECTION ANALYSIS")
        print("="*60)

        results = {}

        # 1. Univariate EWMA for each selected column
        print("\n1. UNIVARIATE EWMA ANALYSIS")
        print("-" * 40)
        for col in selected_columns:
            print(f"\nAnalyzing {col}...")
            results[f'univariate_{col}'] = self.univariate_ewma(col, plot=False)
            n_anomalies = len(results[f'univariate_{col}']['anomaly_indices'])
            print(f"  - Detected {n_anomalies} anomalies")

        # 2. Multivariate EWMA
        print("\n2. MULTIVARIATE EWMA (MEWMA) ANALYSIS")
        print("-" * 40)
        results['mewma'] = self.multivariate_ewma(selected_columns, plot=False)
        n_mewma_anomalies = len(results['mewma']['anomaly_indices'])
        print(f"  - Detected {n_mewma_anomalies} multivariate anomalies")

        # 3. Variance monitoring for first column
        print("\n3. EWMA VARIANCE MONITORING")
        print("-" * 40)
        first_col = selected_columns[0]
        results['variance'] = self.ewma_variance_chart(first_col, plot=False)
        n_var_anomalies = len(results['variance']['var_anomaly_indices'])
        print(f"  - Detected {n_var_anomalies} variance anomalies in {first_col}")

        # 4. Adaptive EWMA for first column
        print("\n4. ADAPTIVE EWMA ANALYSIS")
        print("-" * 40)
        results['adaptive'] = self.adaptive_ewma(first_col, plot=False)
        n_adaptive_anomalies = len(results['adaptive']['adaptive_anomaly_indices'])
        print(f"  - Detected {n_adaptive_anomalies} adaptive anomalies in {first_col}")

        # 5. Hybrid EWMA-CUSUM for first column
        print("\n5. HYBRID EWMA-CUSUM ANALYSIS")
        print("-" * 40)
        results['hybrid'] = self.hybrid_ewma_cusum(first_col, plot=False)
        n_hybrid_anomalies = len(results['hybrid']['hybrid_anomaly_indices'])
        print(f"  - Detected {n_hybrid_anomalies} hybrid anomalies in {first_col}")

        # Summary
        print("\n" + "="*60)
        print("SUMMARY OF DETECTED ANOMALIES")
        print("="*60)
        print(f"Total samples analyzed: {len(self.data_normalized)}")
        print(f"Multivariate anomalies (MEWMA): {n_mewma_anomalies}")
        print(f"Variance anomalies: {n_var_anomalies}")
        print(f"Adaptive EWMA anomalies: {n_adaptive_anomalies}")
        print(f"Hybrid EWMA-CUSUM anomalies: {n_hybrid_anomalies}")

        # Find consensus anomalies (detected by multiple methods)
        all_indices = set()
        if 'mewma' in results:
            all_indices.update(results['mewma']['anomaly_indices'])
        if 'variance' in results:
            all_indices.update(results['variance']['var_anomaly_indices'])
        if 'adaptive' in results:
            all_indices.update(results['adaptive']['adaptive_anomaly_indices'])
        if 'hybrid' in results:
            all_indices.update(results['hybrid']['hybrid_anomaly_indices'])

        print(f"\nTotal unique anomaly time points: {len(all_indices)}")
        print(f"Anomaly rate: {len(all_indices)/len(self.data_normalized)*100:.2f}%")

        return results

    def plot_comprehensive_summary(self, results, selected_columns=None):
        """
        Create a comprehensive summary plot of all EWMA methods
        """
        if selected_columns is None:
            selected_columns = self.data_normalized.columns[:5]

        first_col = selected_columns[0]
        n = len(self.data_normalized)

        plt.figure(figsize=(16, 12))

        # Original data
        plt.subplot(3, 2, 1)
        plt.plot(self.data_normalized[first_col], label=f'{first_col}', alpha=0.8)
        plt.title('Original TEP Data (First Variable)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Univariate EWMA
        plt.subplot(3, 2, 2)
        if f'univariate_{first_col}' in results:
            ewma = results[f'univariate_{first_col}']['ewma']
            anomalies = results[f'univariate_{first_col}']['anomalies']
            plt.plot(ewma, label='EWMA', color='blue')
            plt.scatter(np.where(anomalies)[0], ewma[anomalies], color='red', s=20)
            plt.title('Univariate EWMA')
            plt.legend()
        plt.grid(True, alpha=0.3)

        # MEWMA T² statistics
        plt.subplot(3, 2, 3)
        if 'mewma' in results:
            t2_stats = results['mewma']['t2_stats']
            ucl_t2 = results['mewma']['ucl_t2']
            anomalies = results['mewma']['anomalies']
            plt.plot(t2_stats, label='T² Statistics', color='blue')
            plt.axhline(y=ucl_t2, color='red', linestyle='--', label='UCL')
            plt.scatter(np.where(anomalies)[0], t2_stats[anomalies], color='red', s=20)
            plt.title('MEWMA T² Chart')
            plt.legend()
        plt.grid(True, alpha=0.3)

        # Variance chart
        plt.subplot(3, 2, 4)
        if 'variance' in results:
            ewma_var = results['variance']['ewma_var']
            ucl_var = results['variance']['ucl_var']
            var_anomalies = results['variance']['var_anomalies']
            plt.plot(ewma_var, label='EWMA Variance', color='orange')
            plt.axhline(y=ucl_var, color='red', linestyle='--', label='UCL')
            plt.scatter(np.where(var_anomalies)[0], ewma_var[var_anomalies], color='red', s=20)
            plt.title('EWMA Variance Chart')
            plt.legend()
        plt.grid(True, alpha=0.3)

        # Adaptive EWMA
        plt.subplot(3, 2, 5)
        if 'adaptive' in results:
            ewma_adaptive = results['adaptive']['ewma_adaptive']
            adaptive_anomalies = results['adaptive']['adaptive_anomalies']
            plt.plot(ewma_adaptive, label='Adaptive EWMA', color='green')
            plt.scatter(np.where(adaptive_anomalies)[0], ewma_adaptive[adaptive_anomalies], color='red', s=20)
            plt.title('Adaptive EWMA')
            plt.legend()
        plt.grid(True, alpha=0.3)

        # Anomaly comparison
        plt.subplot(3, 2, 6)
        methods = []
        anomaly_counts = []

        if f'univariate_{first_col}' in results:
            methods.append('Univariate')
            anomaly_counts.append(len(results[f'univariate_{first_col}']['anomaly_indices']))

        if 'mewma' in results:
            methods.append('MEWMA')
            anomaly_counts.append(len(results['mewma']['anomaly_indices']))

        if 'variance' in results:
            methods.append('Variance')
            anomaly_counts.append(len(results['variance']['var_anomaly_indices']))

        if 'adaptive' in results:
            methods.append('Adaptive')
            anomaly_counts.append(len(results['adaptive']['adaptive_anomaly_indices']))

        if 'hybrid' in results:
            methods.append('Hybrid')
            anomaly_counts.append(len(results['hybrid']['hybrid_anomaly_indices']))

        plt.bar(methods, anomaly_counts, color=['blue', 'red', 'orange', 'green', 'purple'][:len(methods)])
        plt.title('Anomaly Detection Comparison')
        plt.ylabel('Number of Anomalies Detected')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# Usage example function
def main_analysis_example():
    """
    Example usage of the TEP EWMA Anomaly Detection system
    """
    print("Tennessee Eastman Process EWMA Anomaly Detection")
    print("=" * 50)

    # Initialize detector
    detector = TEP_EWMA_AnomalyDetector(lambda_param=0.2, alpha=0.05)

    # Load data (replace with your actual TEP data path)
    # For Kaggle data, use: detector.load_tep_data('path/to/your/tep_data.csv')
    data = detector.load_tep_data()  # This creates sample data

    # Preprocess data
    processed_data = detector.preprocess_data(normalize=True)

    # Run comprehensive analysis
    results = detector.comprehensive_analysis()

    # Create summary plots
    detector.plot_comprehensive_summary(results)

    return detector, results

if __name__ == "__main__":
    detector, results = main_analysis_example()
