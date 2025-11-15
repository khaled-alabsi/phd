# Tennessee Eastman Process (TEP) Anomaly Detection - Refactored Version

This directory contains a refactored, modular version of the TEP anomaly detection analysis. The original monolithic Jupyter notebook has been broken down into maintainable, reusable Python modules.

## 🏗️ Architecture Overview

The refactored code follows a clean, modular architecture with clear separation of concerns:

```
te_utils.py              # Utility functions and common operations
te_data_loader.py        # Data loading, preprocessing, and exploration
te_models.py            # Model training, evaluation, and comparison
te_visualization.py     # Visualization and reporting functionality
main_refactored.py      # Main execution script and workflow orchestration
```

## 📁 File Structure

- **`te_utils.py`**: Core utility functions for data handling, metrics calculation, and file operations
- **`te_data_loader.py`**: `TEPDataLoader` class for data management
- **`te_models.py`**: `TEPModelTrainer` class for machine learning pipeline
- **`te_visualization.py`**: `TEPVisualizer` class for plots and reports
- **`main_refactored.py`**: Main script that orchestrates the entire workflow
- **`README_REFACTORED.md`**: This documentation file

## 🚀 Quick Start

### Prerequisites

Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

### Basic Usage

Run the complete analysis pipeline:

```bash
python main_refactored.py
```

### Advanced Usage

```bash
# Skip model training (load existing models)
python main_refactored.py --skip-training

# Skip visualization and reporting
python main_refactored.py --skip-visualization

# Use custom logging level
python main_refactored.py --log-level DEBUG

# Get help
python main_refactored.py --help
```

## 🔧 Configuration

The script uses a default configuration that can be customized. Key configuration options:

- **Data Directory**: `data/` (contains TEP RData files)
- **Output Directory**: `output/1.00/` (version-specific output)
- **Models to Train**: Random Forest, XGBoost, LightGBM, Neural Network
- **Classification Type**: Binary (normal vs. faulty) or multi-class
- **Feature Preprocessing**: Normalization, PCA dimensionality reduction
- **Neural Network Architecture**: Simple, deep, or wide

## 📊 Features

### Data Management
- **Automatic Data Loading**: Handles RData files from TEP dataset
- **Data Exploration**: Statistical summaries, distribution analysis
- **Preprocessing Pipeline**: Normalization, PCA, feature engineering
- **t-SNE Visualization**: Dimensionality reduction for data exploration

### Model Training
- **Traditional ML Models**: Random Forest, XGBoost, LightGBM, SVM, etc.
- **Neural Networks**: Configurable architectures with Keras
- **Automatic Evaluation**: Comprehensive metrics calculation
- **Model Persistence**: Save/load trained models

### Analysis & Visualization
- **Performance Comparison**: Multi-metric model comparison
- **Confusion Matrices**: Per-model classification results
- **ROC Curves**: Model discrimination analysis
- **Fault-Specific Analysis**: Performance breakdown by fault type
- **Detection Delay Analysis**: Temporal performance metrics
- **Interactive Dashboard**: Plotly-based interactive visualizations

### Reporting
- **Comprehensive Reports**: Text-based analysis summaries
- **Performance Tables**: Structured metric comparisons
- **Automated Plot Saving**: Organized output directory structure
- **Logging**: Detailed execution logging

## 🎯 Key Improvements Over Original Code

### 1. **Modularity**
- **Before**: Single 2MB+ Jupyter notebook with mixed concerns
- **After**: Clean separation into focused, reusable modules

### 2. **Maintainability**
- **Before**: Hard to modify specific functionality
- **After**: Easy to update individual components without affecting others

### 3. **Reusability**
- **Before**: Code tightly coupled to notebook execution
- **After**: Functions and classes can be imported and used independently

### 4. **Error Handling**
- **Before**: Limited error handling and debugging
- **After**: Comprehensive logging, exception handling, and graceful failures

### 5. **Configuration**
- **Before**: Hard-coded parameters throughout the code
- **After**: Centralized configuration with command-line options

### 6. **Testing**
- **Before**: Difficult to test individual components
- **After**: Modular structure enables unit testing of each component

### 7. **Documentation**
- **Before**: Limited inline documentation
- **After**: Comprehensive docstrings, type hints, and README

## 🔍 Usage Examples

### Basic Data Analysis

```python
from te_data_loader import TEPDataLoader

# Load and explore data
loader = TEPDataLoader()
train_data, test_data = loader.load_data()
exploration_results = loader.explore_data()

# Create visualizations
loader.visualize_data_distribution()
loader.apply_tsne_visualization()
```

### Model Training

```python
from te_models import TEPModelTrainer

# Initialize trainer
trainer = TEPModelTrainer()

# Train models
models = trainer.train_traditional_models(X_train, y_train)
neural_net = trainer.train_neural_network(X_train, y_train)

# Evaluate
results = trainer.evaluate_models(X_test, y_test)
```

### Visualization

```python
from te_visualization import TEPVisualizer

# Create visualizer
visualizer = TEPVisualizer()

# Generate plots
visualizer.plot_model_performance_comparison(results)
visualizer.plot_confusion_matrices_grid(model_results, y_true)
visualizer.create_interactive_dashboard(model_results, y_true)
```

## 📈 Output Structure

The refactored code maintains the same output structure as the original:

```
output/1.00/
├── anomaly/           # Anomaly detection metrics
├── arl/              # Average run length analysis
├── average/           # Average performance metrics
├── confusion_matrix/  # Confusion matrices
├── default/           # Default analysis results
├── detection_delay/   # Detection delay analysis
├── fdr_far/          # False detection rate analysis
├── per_fault/        # Per-fault performance
├── per_metric/       # Per-metric analysis
├── tep_analysis.log  # Execution log
└── comprehensive_report.txt  # Summary report
```

## 🧪 Testing

To test individual components:

```python
# Test data loading
python -c "from te_data_loader import TEPDataLoader; print('Data loader imported successfully')"

# Test utility functions
python -c "from te_utils import check_python_version; check_python_version()"

# Test model trainer
python -c "from te_models import TEPModelTrainer; print('Model trainer imported successfully')"
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the correct directory
2. **Missing Dependencies**: Install all requirements from `requirements.txt`
3. **Data File Issues**: Verify TEP data files are in the `data/` directory
4. **Memory Issues**: For large datasets, consider using PCA or sampling

### Debug Mode

Run with debug logging for detailed information:

```bash
python main_refactored.py --log-level DEBUG
```

## 🔮 Future Enhancements

- **Configuration Files**: YAML/JSON configuration support
- **Parallel Processing**: Multi-core model training
- **Hyperparameter Tuning**: Automated model optimization
- **Additional Models**: Support for more ML algorithms
- **Web Interface**: Streamlit-based web application
- **API Endpoints**: RESTful API for model serving

## 📚 References

- **Original TEP Dataset**: [Tennessee Eastman Process](https://doi.org/10.1016/0009-2509(93)E1011-J)
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM, Keras
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Data Processing**: Pandas, NumPy, PyReadR

## 👥 Contributing

To contribute to the refactored code:

1. Follow the existing code structure and style
2. Add comprehensive docstrings and type hints
3. Include error handling and logging
4. Update this README for new features
5. Test your changes thoroughly

## 📄 License

This refactored code maintains the same license as the original project.

---

**Note**: This refactored version maintains full compatibility with the original analysis while providing a much more maintainable and extensible codebase.

