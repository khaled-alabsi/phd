Project Goal:
Create a DNN-CUSUM detector that uses a neural network to dynamically predict optimal CUSUM hyperparameters (k, h) during testing, while learning from in-control and out-of-control data during training.
Technical Requirements:
Architecture:
Use existing MCUSUM implementation from code_v2/src/mcusum.py as the CUSUM engine (DO NOT modify it)
Build a DNN wrapper that predicts k_t and h_t for each time point
DNN learns optimal parameters during training
Training Process:
Use grid search to find best DNN configuration (architecture, learning rate, etc.)
Save best configuration to a JSON/pickle file
Save trained model to a file (h5/pt format)
On subsequent runs: load saved config and model instead of retraining
File Structure (Minimal):
src/dnn_cusum.py - Single file containing:
DNNCUSUMDetector class
DNN architecture
Grid search logic
Model save/load functionality
src/dnn_cusum_viz.py - Visualization utilities:
Parameter evolution plots
Comparison plots
Diagnostic visualizations
Integration:
Add to notebook as a new model: MODELS["DNN_CUSUM"] = dnn_cusum_detector.predict
Compare with existing models (MCUSUM, Autoencoder, MEWMA)
Deliverables:
Working code integrated in notebook
Parameter evolution visualization
Research paper draft
Updated Implementation Plan
File Structure (Simplified)
code_v2/src/
├── dnn_cusum.py              # Single file for DNN-CUSUM (new)
├── dnn_cusum_viz.py          # Visualization functions (new)
├── mcusum.py                 # Existing CUSUM (DO NOT MODIFY)
└── ...                       # Other existing files

code_v2/models/               # New folder for saved models
├── dnn_cusum_best_config.json   # Best hyperparameters from grid search
└── dnn_cusum_model.h5           # Trained model weights

code_v2/anomaly_detection.ipynb  # Updated notebook
1. src/dnn_cusum.py - Complete Implementation
Class Structure:
class DNNCUSUMDetector:
    """
    DNN-CUSUM: Neural network-based adaptive CUSUM detector.
    Uses DNN to predict optimal k and h parameters dynamically.
    Wraps existing MCUSUMDetector for actual CUSUM computation.
    """
    
    # === Core Methods ===
    def __init__(self, window_size=50, ...):
        """Initialize with configuration"""
        
    def build_network(self, config):
        """Build DNN architecture from config"""
        
    def fit(self, X_incontrol, X_outcontrol=None, ...):
        """
        Train DNN to predict optimal parameters.
        
        Process:
        1. Check if saved model exists
           - If yes: load model and config, skip training
           - If no: proceed with training
        
        2. Generate training data:
           - Sliding windows from X_incontrol and X_outcontrol
           - Compute "optimal" k, h for each window using classical methods
        
        3. Grid search for best DNN config (if not loaded)
        
        4. Train DNN to predict k, h
        
        5. Save model and config to files
        """
        
    def predict(self, X_test, return_params=False):
        """
        Predict anomalies with adaptive parameters.
        
        Process:
        1. For each time point t:
           - Extract window: X_test[t-window_size:t]
           - DNN predicts: k_t, h_t
           - Use MCUSUMDetector with k_t, h_t
           - Compute CUSUM statistic
        
        2. Return predictions + parameter history (if requested)
        """
        
    # === Grid Search ===
    def grid_search(self, X_train, y_train, param_grid):
        """
        Grid search for best DNN configuration.
        
        param_grid example:
        {
            'hidden_dims': [[64], [64, 32], [128, 64]],
            'learning_rate': [0.001, 0.0001],
            'batch_size': [32, 64],
            'dropout': [0.2, 0.3]
        }
        """
        
    # === Save/Load ===
    def save_model(self, model_path, config_path):
        """Save trained model and config"""
        
    def load_model(self, model_path, config_path):
        """Load pre-trained model and config"""
        
    # === Utilities ===
    def _extract_features(self, window):
        """Extract features from sliding window"""
        
    def _compute_optimal_params(self, window, has_fault):
        """Compute optimal k, h for training (ground truth)"""
Key Implementation Details:
Feature Extraction:
def _extract_features(self, window):
    """
    Extract statistical features from window.
    
    Features:
    - Mean, std, min, max per dimension
    - Trend (linear regression slope)
    - Autocorrelation
    - Recent change magnitude
    """
DNN Architecture (LSTM-based):
def build_network(self, config):
    """
    Input: features from window
    
    Architecture:
    - LSTM layer(s) for temporal dependencies
    - Dense layers for feature processing
    - Two output heads:
      * k_head: predicts k (using softplus activation)
      * h_head: predicts h (using softplus activation)
    """
Training with Optimal Parameter Ground Truth:
def _compute_optimal_params(self, window, has_fault):
    """
    Use classical method to find optimal k, h for this window.
    
    Method:
    - If window has fault: use low k (sensitive)
    - If window is normal: use high k (conservative)
    - Compute using ARL optimization or empirical search
    """
Model Persistence:
def fit(self, X_incontrol, X_outcontrol, ...):
    model_path = "models/dnn_cusum_model.h5"
    config_path = "models/dnn_cusum_best_config.json"
    
    # Try to load existing model
    if os.path.exists(model_path) and os.path.exists(config_path):
        print("Loading saved model...")
        self.load_model(model_path, config_path)
        return self
    
    # Otherwise, train from scratch
    print("Training new model...")
    # ... grid search and training ...
    
    # Save results
    self.save_model(model_path, config_path)
2. src/dnn_cusum_viz.py - Visualization Functions
class DNNCUSUMVisualizer:
    """Visualization utilities for DNN-CUSUM"""
    
    def plot_parameter_evolution(self, param_history, X_test, predictions, 
                                  fault_injection_point=None):
        """
        Plot k(t) and h(t) over time with CUSUM statistic.
        
        3 subplots:
        - Top: k(t) evolution
        - Middle: h(t) evolution  
        - Bottom: CUSUM statistic + threshold + predictions
        """
        
    def plot_parameter_heatmap(self, param_history, feature_names):
        """
        Heatmap showing parameter variation across features.
        """
        
    def plot_comparison(self, fixed_cusum_results, dnn_cusum_results, 
                        X_test, title=""):
        """
        Side-by-side comparison of fixed vs adaptive CUSUM.
        """
        
    def plot_training_history(self, history):
        """Plot training loss, validation metrics"""
3. Integration in Notebook
New Cells to Add:
# Cell: Import DNN-CUSUM
from src.dnn_cusum import DNNCUSUMDetector
from src.dnn_cusum_viz import DNNCUSUMVisualizer

# Cell: Train DNN-CUSUM (only runs once, then loads saved model)
dnn_cusum = DNNCUSUMDetector(
    window_size=50,
    model_dir='models/'  # Will save/load from here
)

# This will either train or load existing model
dnn_cusum.fit(
    X_INCONTROL_TRAIN_FULL_SCALED,
    X_OUT_OF_CONTROL_TRAIN_PLAY_SCALED,
    grid_search=True  # Set to False to skip grid search if config exists
)

# Cell: Test and visualize
predictions, param_history = dnn_cusum.predict(
    X_OUT_OF_CONTROL_TEST_PLAY_SCALED_CUT,
    return_params=True
)

# Visualize parameter evolution
viz = DNNCUSUMVisualizer()
viz.plot_parameter_evolution(
    param_history, 
    X_OUT_OF_CONTROL_TEST_PLAY_SCALED_CUT,
    predictions,
    fault_injection_point=0  # Already cut
)

# Cell: Add to model comparison
def dnn_cusum_predict(x_scaled):
    preds, _ = dnn_cusum.predict(x_scaled)
    return preds.astype(int)

MODELS["DNN_CUSUM"] = dnn_cusum_predict

# Then run the full comparison loop...
4. Grid Search Configuration
Search Space:
param_grid = {
    'architecture': [
        {'type': 'lstm', 'units': [64], 'dense': [32]},
        {'type': 'lstm', 'units': [128], 'dense': [64, 32]},
        {'type': 'lstm', 'units': [64, 64], 'dense': [32]},
    ],
    'learning_rate': [0.001, 0.0001],
    'batch_size': [32, 64],
    'dropout': [0.2, 0.3],
    'window_size': [30, 50, 70]
}

# Grid search will try all combinations
# Best config saved to: models/dnn_cusum_best_config.json
Evaluation Metric for Grid Search:
# Combined metric:
score = alpha * ARL0_penalty + beta * ARL1_reward + gamma * stability

where:
- ARL0_penalty: penalize false alarms on normal data
- ARL1_reward: reward fast detection on fault data
- stability: penalize parameter oscillation
5. Saved Files Format
models/dnn_cusum_best_config.json:
{
  "window_size": 50,
  "architecture": {
    "type": "lstm",
    "units": [64],
    "dense": [32],
    "dropout": 0.2
  },
  "learning_rate": 0.001,
  "batch_size": 32,
  "training_params": {
    "epochs": 50,
    "validation_split": 0.2
  },
  "grid_search_score": 0.923,
  "timestamp": "2024-10-24T00:30:00"
}
models/dnn_cusum_model.h5:
TensorFlow/Keras model file with trained weights
6. How MCUSUM is Used (Without Modification)
# Inside DNNCUSUMDetector.predict():

from .mcusum import MCUSUMDetector

for t in range(len(X_test)):
    # 1. Extract window
    window = X_test[max(0, t-self.window_size):t+1]
    
    # 2. DNN predicts parameters
    k_t, h_t = self.dnn_model.predict(features_from_window)
    
    # 3. Use existing MCUSUM with predicted params
    cusum_temp = MCUSUMDetector(k=k_t, h=h_t)
    cusum_temp.mu_0 = self.global_mu_0  # Use global stats
    cusum_temp.sigma = self.global_sigma
    cusum_temp.is_fitted = True
    
    # 4. Compute statistic
    stats, flags = cusum_temp.predict(window[-1:])
    
    # 5. Store results
    predictions[t] = flags[0]
    param_history['k'].append(k_t)
    param_history['h'].append(h_t)
7. Research Paper Outline (Updated)
Title: "DNN-CUSUM: Deep Learning-Based Adaptive Hyperparameter Selection for Multivariate CUSUM Control Charts" Key Sections:
Introduction
Problem: Fixed CUSUM parameters suboptimal across different fault types
Solution: DNN learns to adapt parameters based on recent observations
Using existing MCUSUM as computational engine
Methodology
MCUSUM baseline (reference to existing implementation)
DNN architecture for parameter prediction
Grid search for optimal configuration
Training with optimal parameter ground truth
Results
Comparison: Fixed MCUSUM vs DNN-CUSUM
Parameter evolution plots showing adaptivity
Performance across 20 fault types
Discussion
When adaptive parameters help (certain fault types)
Computational cost vs performance trade-off
Model persistence benefits
Summary of Changes from Original Plan
Aspect	Original Plan	Updated Plan
Files	4 files	2 files (dnn_cusum.py, dnn_cusum_viz.py)
CUSUM	Implement new CUSUM	Use existing mcusum.py
Training	Manual tuning	Grid search + save results
Persistence	No	Save model + config, load on rerun
Complexity	High	Simplified, practical
Ready to Proceed?
This plan: ✅ Uses only 2 new files
✅ Leverages existing MCUSUM (no modification)
✅ Includes grid search with saved results
✅ Saves trained model for reuse
✅ Integrates seamlessly with notebook
✅ Provides visualization of adaptive parameters Should I start implementing this plan?